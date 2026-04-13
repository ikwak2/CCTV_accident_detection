#!/usr/bin/env python3
# Timesformer - Attention Rollout CAM (robust, no hooks, safe T reconstruction)

import os, json, cv2, av, traceback, numpy as np, torch, torch.nn as nn
from typing import List, Sequence, Literal
from transformers import AutoImageProcessor, TimesformerForVideoClassification

import sys
sys.path.append("/Data3/hj24/CCTV/main")
from vivit import sample_frame_indices, read_video_pyav  # provided by you

# =========================
# Defaults (so `python rollout.py` works)
# =========================
DEFAULTS = {
    "device": "cuda",
    "gpu_id": 5,
    "model_path": "/Data3/hj24/CCTV/main/vivit/timesformer_accident_classifier.pth",
    "predictions_file": "/Data3/hj24/CCTV/main/gradcam/tformer_pred/tformer_predictions_origin.json",
    "output_dir": "/Data3/hj24/CCTV/main/gradcam/tformer_pred/tformer_rollout_output",
    "num_frames": 32,
    "frame_sample_rate": 4,
    "fps": 30,
    "head_fusion": "max",      # or "mean"
    "discard_ratio": 0.2,      # 0~0.9
    "last_n_layers": 3,        # only last N layers in rollout
    "p_low": 70.0,
    "p_high": 99.0,
    "gamma": 0.8,
    "cutoff": 0.15,
    "config_path": "rollout_config.json",  # optional local JSON to override
}

# =========================
# Args (single definition)
# =========================
import argparse
def get_args():
    p = argparse.ArgumentParser("Timesformer Attention Rollout CAM (no-grad)", add_help=True)

    p.add_argument("--device", default=DEFAULTS["device"])
    p.add_argument("--gpu-id", type=int, default=DEFAULTS["gpu_id"])
    p.add_argument("--model-path", default=DEFAULTS["model_path"])
    p.add_argument("--predictions-file", default=DEFAULTS["predictions_file"])
    p.add_argument("--output-dir", default=DEFAULTS["output_dir"])

    p.add_argument("--num-frames", type=int, default=DEFAULTS["num_frames"])
    p.add_argument("--frame-sample-rate", type=int, default=DEFAULTS["frame_sample_rate"])
    p.add_argument("--fps", type=int, default=DEFAULTS["fps"])

    p.add_argument("--head-fusion", choices=["mean","max"], default=DEFAULTS["head_fusion"])
    p.add_argument("--discard-ratio", type=float, default=DEFAULTS["discard_ratio"])
    p.add_argument("--last-n-layers", type=int, default=DEFAULTS["last_n_layers"])

    p.add_argument("--p-low", type=float, default=DEFAULTS["p_low"])
    p.add_argument("--p-high", type=float, default=DEFAULTS["p_high"])
    p.add_argument("--gamma", type=float, default=DEFAULTS["gamma"])
    p.add_argument("--cutoff", type=float, default=DEFAULTS["cutoff"])

    p.add_argument("--config", default=DEFAULTS["config_path"],
                   help="Optional JSON config to override defaults. Keys = CLI flags without dashes.")

    args = p.parse_args()

    # Optional config file override
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg = json.load(f)
        for k, v in cfg.items():
            key = k.replace("-", "_")
            if hasattr(args, key):
                setattr(args, key, v)

    return args

# =========================
# Utils
# =========================
def norm_percentile(x: np.ndarray, low=5.0, high=99.0) -> np.ndarray:
    lo, hi = np.percentile(x, low), np.percentile(x, high)
    if hi > lo:
        return np.clip((x - lo) / (hi - lo), 0, 1)
    return np.zeros_like(x)

def normalize_adaptive(cam: np.ndarray, low=70, high=99, gamma=0.8, cutoff=0.15) -> np.ndarray:
    cam = norm_percentile(cam, low, high)
    if gamma != 1.0:
        cam = np.power(cam, gamma)
    cam[cam < cutoff] = 0.0
    return cam

def infer_hw_from_config(cfg):
    H = getattr(cfg, "image_size", 224)
    ps = getattr(cfg, "patch_size", 16)
    h = int(np.ceil(H / ps))
    w = int(np.ceil(H / ps))
    return h, w

def get_tubelet_size(model) -> int:
    # HF TimeSformer patch 3D conv: (kT, kH, kW)
    proj = model.timesformer.embeddings.patch_embeddings.projection
    kT = proj.kernel_size[0]
    sT = proj.stride[0]
    # 대부분 kT==sT==tubelet_size
    return int(kT)

def rollout_one_phase(
    video_path, model, processor, fps,
    num_frames, frame_sample_rate,
    head_fusion, discard_ratio, last_n_layers,
    p_low, p_high, gamma, cutoff,
    phase, tube):
    """
    phase: [0 .. tube-1] 중 선택. 시작 오프셋을 달리해 tubelet 구간을 평행 이동.
    return: cams_per_frame: list[np.ndarray (h,w)], len = T_frames
    """
    container = av.open(video_path)
    total = container.streams.video[0].frames
    # 원래 인덱스에서 phase만큼 밀어서 샘플
    base_idx = sample_frame_indices(clip_len=num_frames, frame_sample_rate=frame_sample_rate, seg_len=total)
    idx = [min(i+phase, total-1) for i in base_idx]
    frames = read_video_pyav(container, idx); container.close()

    # 기본 전처리 (네 코드 그대로)
    fixed = []
    for f in frames:
        f = np.squeeze(f)
        if f.ndim != 3 or f.shape[2] != 3: raise ValueError(f"Invalid frame {f.shape}")
        if f.dtype != np.uint8: f = (np.clip(f,0,1)*255).astype(np.uint8)
        fixed.append(f)
    frames = fixed
    if len(frames) < num_frames:
        while len(frames) < num_frames: frames.append(frames[-1].copy())
    else:
        frames = frames[:num_frames]

    inputs = processor(list(frames), return_tensors="pt")["pixel_values"].to(model.device)

    with torch.no_grad():
        outputs = model(pixel_values=inputs, output_attentions=True)
        atts = [a.detach() for a in outputs.attentions]
    if not atts: raise RuntimeError("no attentions")

    R = fuse_attentions(atts, head_fusion=head_fusion, discard_ratio=discard_ratio, last_n_layers=last_n_layers)
    cams_tokens = rollout_to_cam_safe(R, model.config)  # (1, T_tokens, h, w)
    _, Ttok, h, w = cams_tokens.shape
    T = len(frames)

    # tubelet으로 인해 Ttok ~= ceil(T / tube)
    # 각 토큰을 담당 프레임 구간에 매핑하여 프레임별 CAM을 구성
    cams_per_frame = [None]*T
    for ttok in range(Ttok):
        f0 = ttok * tube
        f1 = min((ttok+1)*tube, T)  # 반열림 [f0,f1)
        cam = cams_tokens[0, ttok]
        # 퍼센타일/감마/컷오프
        cam = normalize_adaptive(cam, low=p_low, high=p_high, gamma=gamma, cutoff=cutoff)
        for f in range(f0, f1):
            cams_per_frame[f] = cam if cams_per_frame[f] is None else np.maximum(cams_per_frame[f], cam)

    # 혹시 None 남으면 가장 가까운 CAM으로 채움
    last = None
    for i in range(T):
        if cams_per_frame[i] is None:
            cams_per_frame[i] = last if last is not None else np.zeros((h,w), np.float32)
        last = cams_per_frame[i]
    return frames, cams_per_frame

def rollout_to_cam_safe(rollout: torch.Tensor, cfg) -> np.ndarray:
    """
    rollout: (B, S, S), S = 1 + T_tokens * (h*w)
    return: (B, T_tokens, h, w) in [0,1]
    """
    assert rollout.dim() == 3, f"rollout shape expected (B,S,S), got {tuple(rollout.shape)}"
    B, S, _ = rollout.shape
    cls_to_all = rollout[:, 0, 1:]  # (B, S-1)

    h, w = infer_hw_from_config(cfg)
    patches_per_time = h * w
    if (S - 1) % patches_per_time == 0:
        T_tokens = (S - 1) // patches_per_time
    else:
        best = None
        for d in range(1, int(np.sqrt(S-1)) + 1):
            if (S - 1) % d == 0:
                other = (S - 1) // d
                hh = int(np.sqrt(d))
                if hh * hh == d:
                    cand = (hh, hh, other)
                    if best is None or abs(cand[0]-cand[1]) < abs(best[0]-best[1]):
                        best = cand
        if best is None:
            raise ValueError(f"Cannot factor S-1={S-1} into (h*w)*T.")
        h, w, T_tokens = best

    cams = cls_to_all.reshape(B, T_tokens, h, w)
    cams = cams.detach().cpu().numpy()
    cams = (cams - cams.min()) / (cams.max() - cams.min() + 1e-8)
    return cams

def fuse_attentions(att_list: Sequence[torch.Tensor],
                    head_fusion: Literal["mean","max"]="max",
                    discard_ratio: float=0.2,
                    last_n_layers: int=9999) -> torch.Tensor:
    """
    att_list: list of (B, H, S, S) from HF outputs.attentions
    returns rollout R: (B, S, S)
    """
    if last_n_layers < len(att_list):
        att_list = att_list[-last_n_layers:]

    A_list: List[torch.Tensor] = []
    for att in att_list:
        if head_fusion == "max":
            A = att.max(dim=1).values  # (B,S,S)
        else:
            A = att.mean(dim=1)        # (B,S,S)

        if discard_ratio > 0:
            B, S, _ = A.shape
            k = int((1.0 - discard_ratio) * S)
            k = max(1, min(k, S))
            topk = torch.topk(A, k=k, dim=-1).values[..., -1].unsqueeze(-1)
            A = torch.where(A >= topk, A, torch.zeros_like(A))

        I = torch.eye(A.size(-1), device=A.device).unsqueeze(0)
        A = A + I
        A = A / A.sum(dim=-1, keepdim=True)
        A_list.append(A)

    R = torch.eye(A_list[0].size(-1), device=A_list[0].device).unsqueeze(0)  # (B,S,S)
    for A in A_list:
        R = A @ R
    return R

def save_cam_video(frames: Sequence[np.ndarray], cams: np.ndarray, out_path: str, fps: int=30,
                   p_low=70, p_high=99, gamma=0.8, cutoff=0.15):
    """
    frames: list of HxWx3 uint8 (RGB or BGR ok)
    cams:   (1, T, h, w) in [0,1]
    """
    H0, W0 = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(out_path, fourcc, fps, (W0, H0))
    if not vw.isOpened():
        raise RuntimeError(f"VideoWriter open failed: {out_path}")

    T = min(cams.shape[1], len(frames))
    for t in range(T):
        frame = frames[t]
        if frame.ndim != 3:
            if frame.ndim == 2:
                frame = np.repeat(frame[...,None], 3, axis=2)
            else:
                raise ValueError(f"Bad frame ndim {frame.ndim}")
        if frame.shape[2] == 4:
            frame = frame[..., :3]
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (np.clip(frame,0,1)*255).astype(np.uint8)
            else:
                frame = np.clip(frame,0,255).astype(np.uint8)
        if frame.shape[:2] != (H0, W0):
            frame = cv2.resize(frame, (W0, H0), interpolation=cv2.INTER_LINEAR)

        cam = cams[0, t]
        cam_up = cv2.resize(cam, (W0, H0), interpolation=cv2.INTER_CUBIC)
        cam_up = normalize_adaptive(cam_up, low=p_low, high=p_high, gamma=gamma, cutoff=cutoff)

        heat = cv2.applyColorMap(np.uint8(cam_up * 255), cv2.COLORMAP_JET)
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # assume frames are RGB
        over = cv2.addWeighted(heat, 0.6, bgr, 0.4, 0)
        vw.write(over)

    vw.release()

# =========================
# One video pipeline
# =========================

def process_video(video_path: str,
                model: TimesformerForVideoClassification,
                processor: AutoImageProcessor,
                out_dir: str,
                fps: int,
                num_frames: int,
                frame_sample_rate: int,
                head_fusion: str,
                discard_ratio: float,
                last_n_layers: int,
                p_low: float, p_high: float, gamma: float, cutoff: float, phases:int, merge:str):
    tube = get_tubelet_size(model)  # 보통 8
    phases = max(1, min(phases, tube))  # 과한 중복 방지

    # phase별 CAM 계산
    all_cams = []
    for ph in range(phases):
        frames, cams_ph = rollout_one_phase(
            video_path, model, processor, fps,
            num_frames, frame_sample_rate,
            head_fusion, discard_ratio, last_n_layers,
            p_low, p_high, gamma, cutoff,
            phase=ph, tube=tube
        )
        all_cams.append(cams_ph)

    # 병합(프레임축 동일 길이)
    T = len(all_cams[0])
    merged = []
    for f in range(T):
        stack = np.stack([all_cams[p][f].astype(np.float32) for p in range(phases)], axis=0)
        if merge == "max":
            merged.append(np.max(stack, axis=0))
        else:
            merged.append(np.mean(stack, axis=0))

    # 저장 (frames는 phase=0 기준 사용)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir,
        f"rollout_{head_fusion}_dr{discard_ratio}_L{last_n_layers}_P{phases}_{merge}_{os.path.basename(video_path)}")
    save_cam_video(frames, np.expand_dims(np.stack(merged,0),0), out_path,
                   fps=fps, p_low=p_low, p_high=p_high, gamma=gamma, cutoff=cutoff)
    print(f"  -> saved {out_path}")

    
    
    

# =========================
# Main
# =========================
def main():
    args = get_args()
    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    backbone = "facebook/timesformer-base-finetuned-k400"
    model = TimesformerForVideoClassification.from_pretrained(backbone)
    model.classifier = nn.Linear(model.classifier.in_features, 2)

    sd = torch.load(args.model_path, map_location="cpu")
    sd = {k.replace("module.",""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()

    processor = AutoImageProcessor.from_pretrained(backbone)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.predictions_file, "r") as f:
        pred_map = json.load(f)
    pred_map = {k: int(v) for k, v in pred_map.items()}

    for video_path, label in pred_map.items():
        print(f"\n[Video] {video_path}")
        try:
            process_video(
                video_path=video_path,
                model=model,
                processor=processor,
                out_dir=args.output_dir,
                fps=args.fps,
                num_frames=args.num_frames,
                frame_sample_rate=args.frame_sample_rate,
                head_fusion=args.head_fusion,
                discard_ratio=args.discard_ratio,
                last_n_layers=args.last_n_layers,
                p_low=args.p_low, p_high=args.p_high, gamma=args.gamma, cutoff=args.cutoff
            )
        except Exception as e:
            print(f"[ERROR] Failed on {video_path}: {type(e).__name__} - {e}")
            traceback.print_exc()

    torch.cuda.empty_cache()
    print("\nDone.")

if __name__ == "__main__":
    main()
