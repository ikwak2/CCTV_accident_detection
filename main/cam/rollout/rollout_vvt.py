import os
import cv2
import numpy as np
import torch
import av
import json
from transformers import VivitImageProcessor, VivitForVideoClassification, VivitConfig
import torch.nn as nn
import traceback
import matplotlib.pyplot as plt
import imageio

import sys
sys.path.append("/Data3/hj24/CCTV/main")
from vivit import sample_frame_indices, read_video_pyav

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "4,5"


def load_checkpoint_auto(path):
    import os, torch
    ext = os.path.splitext(path)[1].lower()
    if ext == ".safetensors":
        from safetensors.torch import load_file
        sd = load_file(path)
    else:
        # 여기로 오면 .pth/.bin 등 pickle 경로 → 현 torch<2.6 에선 대부분 차단
        raise RuntimeError(
            f"Blocked loading pickle checkpoint: {path}\n"
            f"→ safetensors 로 변환해 사용하세요."
        )
    # 'module.' prefix 정리
    fixed = {}
    for k, v in sd.items():
        nk = k[7:] if k.startswith("module.") else k
        fixed[nk] = v
    return fixed


class VivitAttentionRollout:
    def __init__(self, model, discard_ratio=0.0):
        self.model = model.module if isinstance(model, torch.nn.DataParallel) else model
        self.discard_ratio = discard_ratio
        self.attentions = []
        self._register_hooks()

    def _register_hooks(self):
        self.hooks = []
        for i, blk in enumerate(self.model.vivit.encoder.layer):
            # attention 모듈 위치가 버전에 따라 다르므로 안전하게 탐색
            target = getattr(blk.attention, "attention", None) or getattr(blk.attention, "attn", None) or blk.attention
            hook = target.register_forward_hook(self._get_attention_hook())
            self.hooks.append(hook)

    def _get_attention_hook(self):
        def hook(module, input, output):
            attn = None
            if isinstance(output, (tuple, list)) and len(output) >= 2:
                attn = output[1]
            elif isinstance(output, dict):
                attn = output.get("attentions", None)
            if attn is None:
                return
            # 바로 CPU로 옮겨서 메모리 부담 줄이기
            self.attentions.append(attn.detach().cpu())
        return hook

    def compute_rollout(self):
        if not self.attentions:
            raise RuntimeError("No attentions captured. Did the hooks run?")

        device = self.attentions[0].device  # cpu일 것
        B, _, S, _ = self.attentions[0].shape
        result = torch.eye(S, device=device).unsqueeze(0).expand(B, S, S)  # (B, S, S)

        for attn in self.attentions:
            # 평균 헤드 + residual
            A = attn.mean(dim=1)  # (B, S, S)
            A = A + torch.eye(S, device=device).unsqueeze(0)
            A = A / A.sum(dim=-1, keepdim=True)
            result = torch.matmul(A, result)

        return result

    def clear_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

def save_cam_video(frames, cams, save_path, fps=8):
    H0, W0 = frames[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(save_path, fourcc, fps, (W0, H0))
    if not writer.isOpened():
        raise RuntimeError(f"[ERROR] cv2.VideoWriter open failed for {save_path}")

    T = min(cams.shape[1], len(frames))
    for t in range(T):
        frame = frames[t]
        cam   = cams[0, t]  # (h, w)

        # 1) 프레임 전처리
        if frame.ndim == 2:
            frame = np.repeat(frame[..., None], 3, axis=2)
        if frame.shape[2] == 4:
            frame = frame[..., :3]
        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        if frame.shape[:2] != (H0, W0):
            frame = cv2.resize(frame, (W0, H0), interpolation=cv2.INTER_LINEAR)

        # frame 은 RGB → BGR 로 변환
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 2) CAM 리사이즈 및 정규화
        cam_resized = cv2.resize(cam, (W0, H0), interpolation=cv2.INTER_CUBIC)
        cam_resized = np.clip(cam_resized, 0, 1)

        cam_uint8 = np.uint8(cam_resized * 255)
        cam_color = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)  # 이미 BGR

        # 3) Heatmap + 원본(BGR) overlay
        overlay_bgr = cv2.addWeighted(cam_color, 0.4, frame_bgr, 0.6, 0)

        writer.write(overlay_bgr)

    writer.release()

def _get_temporal_tubelet_size(config):
    ts = getattr(config, "tubelet_size", 1)
    if isinstance(ts, (list, tuple)):
        ts_t = int(ts[0]) if len(ts) > 0 else 1
    elif isinstance(ts, dict):
        ts_t = int(ts.get("t", 1))
    else:
        ts_t = int(ts)
    return max(1, ts_t)


def _get_num_frames(config):
    for key in ["num_frames", "num_video_frames", "num_image_with_fps"]:
        v = getattr(config, key, None)
        if isinstance(v, int) and v > 0:
            return v
    return 32


def compute_rollout_from_attentions(attentions):
    """
    attentions: tuple/list (num_layers)
      각 원소: (B, num_heads, S, S)
    """
    if not attentions:
        raise RuntimeError("No attentions provided.")

    device = attentions[0].device
    B, _, S, _ = attentions[0].shape

    result = torch.eye(S, device=device).unsqueeze(0).expand(B, S, S)  # (B, S, S)

    for attn in attentions:
        A = attn.mean(dim=1)  # (B, S, S)
        A = A + torch.eye(S, device=device).unsqueeze(0)
        A = A / A.sum(dim=-1, keepdim=True)
        result = torch.matmul(A, result)

    return result  # (B, S, S)


def rollout_to_cam(rollout, config):
    """
    rollout: (B, seq, seq)
    CLS가 모든 patch 토큰(시간×공간)으로 보내는 주목을 CAM으로 복원.
    """
    cls_attn = rollout[:, 0, 1:]  # (B, num_patches_total)
    B, num_patches_total = cls_attn.shape

    T_input      = _get_num_frames(config)
    tubelet_size = _get_temporal_tubelet_size(config)
    num_time_tokens = T_input // tubelet_size

    if num_patches_total % num_time_tokens != 0:
        raise ValueError(
            f"Token mismatch: total={num_patches_total}, "
            f"num_time_tokens={num_time_tokens} (T={T_input}, tubelet={tubelet_size})"
        )

    patches_per_time = num_patches_total // num_time_tokens

    # (h,w) factorization
    h = int(np.sqrt(patches_per_time))
    if h * h == patches_per_time:
        w = h
    else:
        best = None
        for d in range(1, int(np.sqrt(patches_per_time)) + 1):
            if patches_per_time % d == 0:
                hw = (d, patches_per_time // d)
                if best is None or abs(hw[0] - hw[1]) < abs(best[0] - best[1]):
                    best = hw
        if best is None:
            raise ValueError(f"Cannot factor {patches_per_time} into (h, w)")
        h, w = best

    cams = cls_attn.reshape(B, num_time_tokens, h, w)
    if tubelet_size > 1:
        cams = cams.repeat_interleave(tubelet_size, dim=1)  # (B, T_input, h, w)

    cams = cams.detach().cpu().numpy()
    cams = (cams - cams.min()) / (cams.max() - cams.min() + 1e-8)

    print("[DEBUG] num_patches_total:", cls_attn.shape[1])
    print("[DEBUG] T_input:", _get_num_frames(config))
    print("[DEBUG] tubelet_size:", _get_temporal_tubelet_size(config))

    return cams


def process_video(video_path, model, processor, output_dir, label_str):
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    indices = sample_frame_indices(clip_len=32, frame_sample_rate=4, seg_len=total_frames)
    frames = read_video_pyav(container, indices)

    processed_frames = []
    original_frames = []
    for f in frames:
        f = np.squeeze(f)
        f_uint8 = (f * 255).astype(np.uint8) if f.dtype != np.uint8 else f
        processed_frames.append(cv2.resize(f_uint8, (224, 224)))
        original_frames.append(f_uint8)

    inputs = processor(list(processed_frames), return_tensors="pt")
    input_tensor = inputs["pixel_values"].cuda()

    with torch.no_grad():
        outputs = model(pixel_values=input_tensor, output_attentions=True)
        attentions = outputs.attentions

    rollout = compute_rollout_from_attentions(list(attentions))

    base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    base_model.config.num_frames = 32

    cams = rollout_to_cam(rollout, base_model.config)  # (1, T, h, w)

    video_name = f"{label_str}_{os.path.basename(video_path).replace('.mp4', '')}_rollout.mp4"
    video_save_path = os.path.join(output_dir, video_name)

    save_cam_video(original_frames, cams, video_save_path)
    print(f"[DONE] Saved: {video_save_path}")


if __name__ == '__main__':
    model_path = '/Data3/hj24/CCTV/main/vivit/vivit_accident_classifier.pth'
    predictions_file = "/Data3/hj24/CCTV/main/gradcam/vivit_pred/vivit_predictions_origin.json"
    output_dir = '/Data3/hj24/CCTV/main/gradcam/vivit_pred/vvt_rollout_output'
    os.makedirs(output_dir, exist_ok=True)

    model = VivitForVideoClassification.from_pretrained(
        "google/vivit-b-16x2-kinetics400",
        use_safetensors=True,
        attn_implementation="eager",
        dtype=torch.float32
    )
    config = VivitConfig.from_pretrained("google/vivit-b-16x2-kinetics400")
    config.num_labels = 2

    model.classifier = torch.nn.Linear(model.classifier.in_features, 2)

    ckpt_sft = "/Data3/hj24/CCTV/main/vivit/vivit_accident_classifier.safetensors"
    ckpt_pth = "/Data3/hj24/CCTV/main/vivit/vivit_accident_classifier.pth"

    state = None
    try:
        state = load_checkpoint_auto(ckpt_sft)
        print(f"[INFO] loaded safetensors: {ckpt_sft}")
    except Exception as e:
        print(f"[WARN] safetensors not usable: {e}")
        print(f"[INFO] trying fallback .pth: {ckpt_pth}")
        raw = torch.load(ckpt_pth, map_location="cpu")
        if isinstance(raw, dict) and "state_dict" in raw:
            raw = raw["state_dict"]
        state = { (k[7:] if k.startswith("module.") else k): v for k, v in raw.items() }

    (model.module if hasattr(model, "module") else model).load_state_dict(state, strict=False)

    model = torch.nn.DataParallel(model)
    model = model.cuda().eval()

    processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")

    with open(predictions_file, 'r') as f:
        predictions = json.load(f)

    non_accident_videos = [v for v, label in predictions.items() if label == 0]
    accident_videos = [v for v, label in predictions.items() if label == 1]
    selected_videos = non_accident_videos + accident_videos

    for video_path in selected_videos:
        label_str = 'accident' if predictions[video_path] == 1 else 'non-accident'
        print(f"[DEBUG] Starting: {video_path} ({label_str})")
        try:
            process_video(video_path, model, processor, output_dir, label_str)
        except Exception as e:
            print(f"[ERROR] Failed on {video_path}: {type(e).__name__} - {e}")
            traceback.print_exc()
