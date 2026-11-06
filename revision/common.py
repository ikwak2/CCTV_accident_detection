"""Shared data/model contract for the revision; legacy scripts are left intact."""
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import random
import time
import types
import uuid
from pathlib import Path

import av
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve
from torch.utils.data import Dataset
from transformers import AutoImageProcessor, TimesformerForVideoClassification
from transformers import VideoMAEForVideoClassification, VivitForVideoClassification, VivitImageProcessor

cv2.setNumThreads(0)
MODELS = {
    'vivit': ('google/vivit-b-16x2-kinetics400', 32, VivitForVideoClassification),
    'timesformer': ('facebook/timesformer-base-finetuned-k400', 32, TimesformerForVideoClassification),
    'videomae': ('MCG-NJU/videomae-base-finetuned-kinetics', 16, VideoMAEForVideoClassification),
}


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))
    os.replace(temp, path)


def stable_seed(*values):
    return int.from_bytes(hashlib.sha256('|'.join(map(str, values)).encode()).digest()[:8], 'big') % (2**32)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with open(path, encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def write_csv(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text('')
        return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def environment():
    packages = {}
    for name in ['torch', 'torchvision', 'transformers', 'peft', 'av', 'numpy', 'scikit-learn']:
        packages[name] = importlib.metadata.version(name)
    code = {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')}
    return {'packages': packages, 'code_sha256': code,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'gpu_uuid': os.environ.get('NVIDIA_VISIBLE_DEVICES'),
            'source_commit': os.environ.get('ORV_SOURCE_COMMIT', 'uncommitted'),
            'created_unix': time.time()}


def patch_videomae_projection_calls(model):
    """Preserve HF Q/V biases while actually calling PEFT-wrapped Q/K/V modules.

    HF 4.46's functional linear(..., module.weight) bypasses a LoRA forward.
    The preflight checks output equivalence before LoRA and nonzero LoRA gradients.
    """
    def forward(self, hidden_states, head_mask=None, output_attentions=False):
        queries = self.query(hidden_states)
        keys = self.key(hidden_states)
        values = self.value(hidden_states)
        if self.q_bias is not None:
            queries = queries + self.q_bias
            values = values + self.v_bias
        query = self.transpose_for_scores(queries)
        key = self.transpose_for_scores(keys)
        value = self.transpose_for_scores(values)
        if output_attentions or head_mask is not None:
            weights = (query @ key.transpose(-1, -2) / math.sqrt(self.attention_head_size)).softmax(-1)
            weights = self.dropout(weights)
            if head_mask is not None:
                weights = weights * head_mask
            context = weights @ value
        else:
            context = F.scaled_dot_product_attention(query, key, value,
                dropout_p=self.dropout.p if self.training else 0.)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(context.shape[0], context.shape[1], self.all_head_size)
        return (context, weights) if output_attentions else (context,)

    for module in model.modules():
        if module.__class__.__name__ in ['VideoMAESelfAttention', 'VideoMAESdpaSelfAttention']:
            module.forward = types.MethodType(forward, module)


def build_model(name, lora, root, seed=42, legacy=None):
    seed_all(seed)
    snapshots = json.loads((Path(root)/'pretrained/snapshots.json').read_text())
    snapshot = snapshots[name]['path']
    kwargs = {'torch_dtype': torch.float32}
    if name != 'timesformer':
        kwargs['attn_implementation'] = 'sdpa'
    model = MODELS[name][2].from_pretrained(snapshot, **kwargs)
    model.config.num_labels = 2
    dim = model.classifier.in_features
    model.classifier = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(), nn.Dropout(.3), nn.Linear(256, 2))
    for module in model.classifier.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    if name == 'videomae':
        patch_videomae_projection_calls(model)
    if name == 'timesformer':
        targets = ['attention.output.dense', 'temporal_attention.output.dense']
        if lora == 'qkvo':
            targets += ['attention.attention.qkv', 'temporal_attention.attention.qkv']
    else:
        targets = ['attention.output.dense']
        if lora == 'qkvo':
            targets += ['attention.attention.query', 'attention.attention.key', 'attention.attention.value']
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=.1,
                                           target_modules=targets, modules_to_save=['classifier']))
    # PEFT freezes the unused original_module; do not unfreeze it by substring.
    assert not any(p.requires_grad for n,p in model.named_parameters() if '.original_module.' in n)
    if legacy:
        state = torch.load(legacy, map_location='cpu', weights_only=True)
        state = state.get('state_dict', state)
        state = {k.removeprefix('module.'): v for k,v in state.items()}
        model.load_state_dict(state, strict=True)
    processor_cls = VivitImageProcessor if name == 'vivit' else AutoImageProcessor
    processor = processor_cls.from_pretrained(snapshot)
    return model, processor, snapshots[name]


def read_window(path, start, span, num_frames, fps):
    """Decode one bounded interval; all backbones share the same time endpoints."""
    if fps <= 0 or span <= 0:
        raise ValueError(f'Invalid timing: {path}')
    targets = np.linspace(start, start + max(0., span - 1/fps), num_frames)
    frames, previous = [], None
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'SLICE'
        stream.codec_context.thread_count = 1
        base = float((stream.start_time or 0) * stream.time_base)
        container.seek(max(0, int((start+base)/float(stream.time_base))), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            stamp = float(frame.pts * stream.time_base) - base
            # Use the last available frame at or before a requested instant.
            while len(frames) < len(targets) and stamp > targets[len(frames)] + 1e-8:
                if previous is None:
                    previous = frame.to_ndarray(format='rgb24')
                frames.append(previous)
            if len(frames) >= len(targets):
                break
            previous = frame.to_ndarray(format='rgb24')
        if previous is None:
            raise RuntimeError(f'No decoded frames: {path}')
        while len(frames) < len(targets):
            frames.append(previous)
    return frames, targets.tolist()


def pixels_from_frames(frames, processor, flip=False):
    # Common aspect-preserving resize/center crop; checkpoint-specific normalization.
    cropped = []
    for frame in frames:
        h, w = frame.shape[:2]
        scale = 256/min(h, w)
        resized = cv2.resize(frame, (round(w*scale), round(h*scale)), interpolation=cv2.INTER_LINEAR)
        top, left = (resized.shape[0]-224)//2, (resized.shape[1]-224)//2
        crop = resized[top:top+224, left:left+224]
        if flip:
            crop = crop[:, ::-1].copy()
        cropped.append(crop)
    return processor(cropped, do_resize=False, do_center_crop=False, return_tensors='pt')['pixel_values'][0]


class ClipDataset(Dataset):
    def __init__(self, rows, processor, num_frames, window=4., training=False, seed=42):
        self.rows, self.processor, self.num_frames = rows, processor, num_frames
        self.window, self.training, self.seed, self.epoch = window, training, seed, 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        duration, fps = float(row['duration']), float(row['fps'])
        span = min(self.window, duration)
        rng = np.random.default_rng(stable_seed(self.seed, self.epoch, row['id']))
        start = float(rng.uniform(0, max(0, duration-span))) if self.training else max(0, duration-span)/2
        frames, _ = read_window(row['path'], start, span, self.num_frames, fps)
        tensor = pixels_from_frames(frames, self.processor, flip=self.training and rng.random()<.5)
        return tensor, int(row['label']), row['id']


def metrics(y, score, threshold=.5):
    y, score = np.asarray(y, dtype=int), np.asarray(score, dtype=float)
    tn, fp, fn, tp = confusion_matrix(y, score >= threshold, labels=[0,1]).ravel()
    both = len(np.unique(y)) == 2
    return {'n': len(y), 'positive': int(y.sum()), 'negative': int((1-y).sum()),
            'threshold': float(threshold), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
            'accuracy': float((tn+tp)/len(y)) if len(y) else None,
            'auroc': float(roc_auc_score(y,score)) if both else None,
            'average_precision': float(average_precision_score(y,score)) if both else None,
            'recall': float(tp/(tp+fn)) if tp+fn else None,
            'missed_detection_rate': float(fn/(tp+fn)) if tp+fn else None,
            'fpr': float(fp/(fp+tn)) if fp+tn else None}


def select_threshold(y, scores, budget=.1):
    if len(np.unique(y)) < 2:
        return .5
    fpr, tpr, thresholds = roc_curve(y, scores, drop_intermediate=False)
    allowed = np.flatnonzero((fpr <= budget) & np.isfinite(thresholds))
    if not len(allowed):
        return float(np.nextafter(np.max(scores), np.inf))
    i = allowed[np.argmax(tpr[allowed])]
    return float(thresholds[i])
