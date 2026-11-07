"""Prepare and run the requested 10 accident + 10 non-accident CAM comparison.

All frames/identifiers/arrays remain in the private server results directory.
Model code and checkpoints are the existing grouped revision experiments.
"""
import argparse
import json
from pathlib import Path
import sys
import time

if Path('/app/revision').is_dir():
    sys.path.append('/app/revision')

import cv2
import numpy as np
import torch

from common import (MODELS, atomic_json, build_model, environment, pixels_from_frames,
                    read_csv, read_window, sha256)
from cam_methods import activation_volume, divided_rollout, gradient_cams, joint_rollout, restore_divided_grid


LORA = {'vivit': 'qkvo', 'timesformer': 'qkvo', 'videomae': 'out'}


def prepare(root, output, selection_file=None):
    manifest = {r['id']: r for r in read_csv(root / 'manifests/grouped.csv')}
    source = root / 'annotations/common_explanation_cases_private.csv'
    candidates = read_csv(source)
    selected, groups, counts = [], set(), {0: 0, 1: 0}
    for index, case in enumerate(candidates):
        row = manifest[case['id']]
        label = int(row['label'])
        if index < 3 or int(row['quarantined']) or row['group'] in groups or counts[label] >= 10:
            continue
        counts[label] += 1
        groups.add(row['group'])
        selected.append({'case': f'case_{index:03d}', 'display_id': ('A' if label else 'N') + f'{counts[label]:02d}',
                         'id': row['id'], 'label': label, 'group': row['group']})
    if counts != {0: 10, 1: 10}:
        raise ValueError(f'Insufficient eligible fixed cases: {counts}')
    selection = {'selection': 'First 10 per dataset label in the existing fixed 60-case list, excluding prior cases 0..2, '
                 'quarantined candidates and repeated groups. No selection on heatmap appearance.',
                 'source_manifest_sha256': sha256(source), 'counts': counts,
                 'label_scope': 'Dataset labels, not human-verified road-only normal or collision-onset annotations',
                 'cases': selected}
    if selection_file is not None:
        selection = json.loads(selection_file.read_text())
        selected = selection['cases']
        if len(selected) != 20 or sum(int(r['label']) for r in selected) != 10 or len({r['group'] for r in selected}) != 20:
            raise ValueError('Need 10 positive and 10 negative independent groups')
        for item in selected:
            row = manifest[item['id']]
            if int(row['label']) != int(item['label']) or row['group'] != item['group'] or int(row['quarantined']):
                raise ValueError('Selection does not match eligible source manifest')
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'selection_private.json'
    if target.exists() and json.loads(target.read_text()) != json.loads(json.dumps(selection)):
        raise ValueError('Existing selection differs; use a new output directory')
    atomic_json(target, selection)
    frames_dir = output / 'frames'
    frames_dir.mkdir(exist_ok=True)
    for item in selected:
        row = manifest[item['id']]
        span = min(4., float(row['duration']))
        start = max(0., float(row['duration']) - span) / 2
        frames, times = read_window(row['path'], start, span, 4, float(row['fps']))
        crops = []
        for frame in frames:
            h, w = frame.shape[:2]
            scale = 256 / min(h, w)
            resized = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_LINEAR)
            y, x = (resized.shape[0] - 224) // 2, (resized.shape[1] - 224) // 2
            crops.append(resized[y:y + 224, x:x + 224])
        np.savez_compressed(frames_dir / f"{item['case']}.npz", frames=np.stack(crops), timestamps=np.asarray(times))
    print('PREPARED', counts, flush=True)


def capture_rollout(model, pixels, name, backbone):
    if name != 'timesformer':
        with torch.inference_mode():
            outputs = model(pixels, output_attentions=True)
            maps = joint_rollout(outputs.attentions, (pixels.shape[1] // 2, 14, 14), name == 'vivit')
            logits = outputs.logits.detach()
        return maps.repeat_interleave(2, 1), logits
    temporal = []
    handles = []

    def force_attention(module, args, kwargs):
        kwargs = dict(kwargs)
        if len(args) > 1:
            kwargs.pop('output_attentions', None)
            return (args[0], True, *args[2:]), kwargs
        return args, dict(kwargs, output_attentions=True)

    def save_attention(module, args, result):
        temporal.append(result[1].detach().float().mean(1))

    try:
        for block in backbone.encoder.layer:
            module = block.temporal_attention.attention
            handles.append(module.register_forward_pre_hook(force_attention, with_kwargs=True))
            handles.append(module.register_forward_hook(save_attention))
        with torch.inference_mode():
            outputs = model(pixels, output_attentions=True)
            spatial = [a.float().mean(1) for a in outputs.attentions]
            internal_frames = backbone.config.num_frames
            patch_width = backbone.config.image_size // backbone.config.patch_size
            internal_patches = (14 * 14 * pixels.shape[1]) // internal_frames
            maps = divided_rollout(spatial, temporal, internal_frames, internal_patches // patch_width, patch_width)
            maps = restore_divided_grid(maps, pixels.shape[1])
            logits = outputs.logits.detach()
        return maps, logits
    finally:
        for handle in handles:
            handle.remove()


def run(root, output, name):
    selection_path = output / 'selection_private.json'
    selection = json.loads(selection_path.read_text())
    tag = f'{name}_{LORA[name]}_mixed_s42'
    checkpoint_path = root / 'runs' / tag / 'best.pt'
    destination = output / 'maps' / name
    destination.mkdir(parents=True, exist_ok=True)
    model, processor, _ = build_model(name, LORA[name], root, 42)
    configured_frames = model.config.num_frames
    contract = {'checkpoint_sha256': sha256(checkpoint_path), 'selection_sha256': sha256(selection_path),
                'model': name, 'tag': tag, 'target_class': 1, 'target_quantity': 'accident logit for both labels',
                'target_layer': f'{name}.encoder.layer[-1].layernorm_before',
                'input_frames': MODELS[name][1], 'configured_frames': configured_frames,
                'configuration_warning': 'Input/configured frame mismatch: preserve checkpoint behavior; inverse token grouping for display'
                    if name == 'timesformer' and configured_frames != MODELS[name][1] else None,
                'code_sha256': {p.name: sha256(p) for p in [Path(__file__), Path(__file__).with_name('cam_methods.py')]},
                'precision': 'FP32; no autocast', 'gradcampp_axes': 'joint T,H,W; first-gradient reference approximation',
                'rollout': 'head mean, residual identity mixing, all layers; explicit spatial+temporal adaptation for TimeSformer'}
    contract_path = destination / 'contract.json'
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError('Existing map contract differs; refusing to overwrite')
    atomic_json(contract_path, contract)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    model.cuda().eval()
    base = model.get_base_model()
    backbone = getattr(base, name)
    target = backbone.encoder.layer[-1].layernorm_before
    manifest = {r['id']: r for r in read_csv(root / 'manifests/grouped.csv')}
    diagnostics = []
    for index, item in enumerate(selection['cases']):
        array_path = destination / f"{item['case']}.npz"
        record_path = destination / f"{item['case']}.json"
        if array_path.exists() and record_path.exists():
            diagnostics.append(json.loads(record_path.read_text()))
            print('REUSED', name, item['display_id'], flush=True)
            continue
        started = time.monotonic()
        row = manifest[item['id']]
        span = min(4., float(row['duration']))
        start = max(0., float(row['duration']) - span) / 2
        frames, times = read_window(row['path'], start, span, MODELS[name][1], float(row['fps']))
        pixels = pixels_from_frames(frames, processor).unsqueeze(0).cuda().requires_grad_(True)
        captured = []
        handle = target.register_forward_hook(lambda module, args, result: captured.append(result))
        try:
            with torch.enable_grad():
                logits = model(pixels).logits
                if len(captured) != 1:
                    raise ValueError('Expected one target activation per forward')
                gradients = torch.autograd.grad(logits[:, 1].sum(), captured[0])[0]
                target_shape = list(captured[0].shape)
                activations = activation_volume(captured[0].detach(), name, pixels.shape[1], attention_frames=configured_frames)
                gradient_volume = activation_volume(gradients.detach(), name, pixels.shape[1], attention_frames=configured_frames)
                maps = gradient_cams(activations, gradient_volume)
                probability = float(logits.detach().softmax(-1)[0, 1])
        finally:
            handle.remove()
        gradient_max = float(gradient_volume.abs().max())
        del logits, gradients, activations, gradient_volume
        captured.clear()
        pixels = pixels.detach()
        for method in maps:
            if name != 'timesformer':
                maps[method] = maps[method].repeat_interleave(2, 1)
        rollout, rollout_logits = capture_rollout(model, pixels, name, backbone)
        maps['rollout'] = rollout
        rollout_probability = float(rollout_logits.softmax(-1)[0, 1])
        if abs(probability - rollout_probability) > 2e-4:
            raise ValueError('Attention extraction changed the forward prediction')
        arrays = {method: values[0].detach().cpu().numpy() for method, values in maps.items()}
        for method, values in arrays.items():
            if values.shape != (MODELS[name][1], 14, 14) or not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f'Invalid raw map: {method}')
        np.savez_compressed(array_path, timestamps=np.asarray(times), **arrays)
        record = {'case': item['case'], 'display_id': item['display_id'], 'label': item['label'],
                  'probability': probability, 'rollout_probability': rollout_probability,
                  'gradient_abs_max': gradient_max, 'elapsed_seconds': time.monotonic() - started,
                  'target_activation_shape': target_shape,
                  'methods': {method: {'status': 'nonzero' if values.max() > 0 else 'zero_signal',
                                       'max': float(values.max()), 'mean': float(values.mean())}
                              for method, values in arrays.items()}}
        atomic_json(record_path, record)
        diagnostics.append(record)
        atomic_json(destination / 'progress.json', {'completed': len(diagnostics), 'total': len(selection['cases']),
                                                   'last': item['display_id']})
        print('CAM', name, index + 1, '/', len(selection['cases']), item['display_id'],
              'seconds', round(record['elapsed_seconds'], 2), 'signal', record['methods'], flush=True)
        del pixels, maps, rollout, rollout_logits
    atomic_json(destination / 'summary.json', {'contract': contract, 'environment': environment(),
        'cases': len(diagnostics), 'records': diagnostics,
        'limitations': ['Dataset labels; no onset/box annotations', 'Center window can miss the accident',
                        'Raw rollout is not class-specific', 'CAM color normalization does not compare absolute evidence',
                        'TimeSformer rollout is an explicit structural adaptation, not validated accident localization']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run'])
    parser.add_argument('--root', type=Path, default=Path('/results'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', choices=list(LORA))
    parser.add_argument('--selection-file', type=Path, help='Optional input-only scene-screened 20-case selection')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.root, args.out, args.selection_file)
    elif args.model is None:
        parser.error('--model is required for run')
    else:
        run(args.root, args.out, args.model)
