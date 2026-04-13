#!/usr/bin/env python3
# VideoMAE + LoRA (PEFT) - Attention Rollout CAM

import os, json, cv2, av, traceback
import numpy as np
import torch
import torch.nn as nn
from typing import List, Sequence, Literal
from transformers import (
    VideoMAEConfig,
    VideoMAEForVideoClassification,
    AutoImageProcessor,
)
from peft import LoraConfig, get_peft_model

import sys
sys.path.append("/Data3/hj24/CCTV/main")
from vivit import sample_frame_indices, read_video_pyav   # 네 util 그대로 사용

# =========================
# 고정 설정값
# =========================
CONF = {
    "device": "cuda",
    "gpu_id": 5,

    # ---- 경로들 ----
    "model_path": "/Data3/hj24/CCTV/main/vivit/videomae_accident_classifier.pth",
    "predictions_file": "/Data3/hj24/CCTV/main/gradcam/mae_pred/mae_predictions_origin.json",
    "output_dir": "/Data3/hj24/CCTV/main/gradcam/mae_pred/mae_rollout_output",

    # ---- 비디오 샘플링 ----
    "num_frames": 16,
    "frame_sample_rate": 4,
    "fps": 30,

    # ---- rollout 하이퍼 (Grid artifact 완화) ----
    "head_fusion": "mean",   # "max" 보다 grid 적음
    "discard_ratio": 0.15,   # 0.1~0.2 추천
    "last_n_layers": 4,      # 1~4 사이에서 조절

    # ---- heatmap visibility ----
    "p_low": 70.0,
    "p_high": 99.0,
    "gamma": 1.25,           # 1.2~1.3 정도
    "cutoff": 0.15,
}

# =========================
# Utils
# =========================
def norm_percentile(x: np.ndarray, low=5.0, high=99.0) -> np.ndarray:
    lo, hi = np.percentile(x, low), np.percentile(x, high)
    if hi > lo:
        return np.clip((x - lo) / (hi - lo), 0, 1)
    return np.zeros_like(x)

def normalize_adaptive(cam: np.ndarray, low=70, high=99, gamma=1.25, cutoff=0.15) -> np.ndarray:
    cam = norm_percentile(cam, low, high)
    if gamma != 1.0:
        cam = np.power(cam, gamma)
    cam[cam < cutoff] = 0.0
    return cam

def infer_hw_from_config_like(image_size: int = 224, patch_size=16):
    """
    VideoMAE 계열: patch_size가 (tubelet, H, W) 튜플일 수 있음.
    여기서는 H/W 방향만 사용.
    """
    if isinstance(patch_size, (list, tuple)):
        ps_spatial = patch_size[-1]
    else:
        ps_spatial = patch_size
    h = int(np.ceil(image_size / ps_spatial))
    w = int(np.ceil(image_size / ps_spatial))
    return h, w

def rollout_to_cam_safe(rollout: torch.Tensor, image_size: int, patch_size=16) -> np.ndarray:
    """
    rollout: (B, S, S), S = 1 + T_tokens * (h*w)
    return:  (B, T_tokens, h, w) in [0,1]
    """
    assert rollout.dim() == 3, f"rollout shape expected (B,S,S), got {tuple(rollout.shape)}"
    B, S, _ = rollout.shape
    cls_to_all = rollout[:, 0, 1:]  # (B, S-1) CLS→patch

    h, w = infer_hw_from_config_like(image_size, patch_size)
    patches_per_time = h * w
    if (S - 1) % patches_per_time == 0:
        T_tokens = (S - 1) // patches_per_time
    else:
        # 비정상인 경우에도 최대한 factorization 시도
        best = None
        for d in range(1, int(np.sqrt(S - 1)) + 1):
            if (S - 1) % d == 0:
                other = (S - 1) // d
                hh = int(np.sqrt(d))
                if hh * hh == d:
                    cand = (hh, hh, other)
                    if best is None or abs(cand[0] - cand[1]) < abs(best[0] - best[1]):
                        best = cand
        if best is None:
            raise ValueError(f"Cannot factor S-1={S-1} into (h*w)*T.")
        h, w, T_tokens = best

    cams = cls_to_all.reshape(B, T_tokens, h, w)
    cams = cams.detach().cpu().numpy()
    cams = (cams - cams.min()) / (cams.max() - cams.min() + 1e-8)
    return cams

def fuse_attentions(att_list: Sequence[torch.Tensor],
                    head_fusion: Literal["mean","max"] = "mean",
                    discard_ratio: float = 0.15,
                    last_n_layers: int = 9999) -> torch.Tensor:
    """
    att_list: list of (B, H, S, S)
    returns rollout R: (B, S, S)
    """
    if last_n_layers < len(att_list):
        att_list = att_list[-last_n_layers:]

    A_list: List[torch.Tensor] = []
    for att in att_list:
        # head fusion
        if head_fusion == "max":
            A = att.max(dim=1).values   # (B,S,S)
        else:
            A = att.mean(dim=1)         # (B,S,S)

        # token sparsification
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

    # rollout
    R = torch.eye(A_list[0].size(-1), device=A_list[0].device).unsqueeze(0)  # (B,S,S)
    for A in A_list:
        R = A @ R
    return R

def save_cam_video(frames, cams: np.ndarray, out_path: str, fps: int = 30,
                   p_low=70, p_high=99, gamma=1.25, cutoff=0.15):
    """
    frames: list of HxWx3 (RGB uint8)
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

        if frame.ndim == 2:
            frame = np.repeat(frame[..., None], 3, axis=2)
        if frame.shape[2] == 4:
            frame = frame[..., :3]
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
            else:
                frame = np.clip(frame, 0, 255).astype(np.uint8)

        if frame.shape[:2] != (H0, W0):
            frame = cv2.resize(frame, (W0, H0), interpolation=cv2.INTER_LINEAR)

        cam = cams[0, t]
        cam_up = cv2.resize(cam, (W0, H0), interpolation=cv2.INTER_LINEAR)
        cam_up = normalize_adaptive(cam_up, low=p_low, high=p_high, gamma=gamma, cutoff=cutoff)

        # grid 완화용 블러 (필요 시 (21,21) 같은 큰 커널도 시도 가능)
        cam_up = cv2.GaussianBlur(cam_up, (9, 9), 0)

        heat = cv2.applyColorMap(np.uint8(cam_up * 255), cv2.COLORMAP_JET)
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        over = cv2.addWeighted(heat, 0.6, bgr, 0.4, 0)
        vw.write(over)

    vw.release()


# =========================
# 학습 시 사용한 VideoMAE + LoRA 구성 재현
# =========================
def build_trained_videomae(checkpoint_path: str, device: torch.device):
    backbone = "MCG-NJU/videomae-base-finetuned-kinetics"

    # config에 attention 출력 설정
    config = VideoMAEConfig.from_pretrained(backbone)
    config.output_attentions = True
    config.attn_implementation = "eager"

    # 1) base VideoMAE 로드
    base_model = VideoMAEForVideoClassification.from_pretrained(
        backbone,
        config=config,
        use_safetensors=True,
        attn_implementation="eager",
        torch_dtype=torch.float32,
    )

    # 2) 분류기 구조/num_labels 훈련 코드와 동일하게
    #    (지금 코드 기준: MODE="binary" → NUM_CLASSES = 2)
    NUM_CLASSES = 2   # 3클래스로 학습했다면 여기를 3으로 바꾸고 classifier도 맞춰야 함
    base_model.config.num_labels = NUM_CLASSES

    base_model.classifier = nn.Sequential(
        nn.Linear(base_model.classifier.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, NUM_CLASSES),
    )

    # 3) LoRA 설정도 학습 코드와 동일하게
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["attention.output.dense"],
        modules_to_save=["classifier"],   # classifier는 항상 학습/저장
    )

    model = get_peft_model(base_model, lora_config)

    # 4) 체크포인트 로드 (훈련 때 model.module.state_dict() 저장했으므로 그대로 들어감)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print("[INFO] missing keys:", missing)
    print("[INFO] unexpected keys:", unexpected)

    # 여기 missing/unexpected는 이론상 거의 없어야 정상 (LoRA/분류기까지 다 맞췄기 때문)
    # 남더라도 encoder쪽이 없어서는 안 됨

    # 5) attention 구현 강제: sdpa → eager
    #    base_model.model: 실제 VideoMAE backbone (HF 구조)
    model.base_model.model.set_attn_implementation("eager")

    model.to(device).eval()
    return model


# =========================
# 한 개 비디오 처리
# =========================
@torch.no_grad()
def process_video(
    video_path: str,
    model,
    processor: AutoImageProcessor,
    image_size: int,
    patch_size,
    cfg: dict,
):
    print(f"[Video] {video_path}")
    try:
        container = av.open(video_path)
    except Exception as e:
        print(f"  -> open fail: {e}")
        return

    # 총 프레임 확인
    try:
        total = int(container.streams.video[0].frames or 0)
    except Exception:
        total = 0
    if total <= 0:
        print("  -> no frames, skip.")
        container.close()
        return

    # 프레임 인덱스 샘플링 (네 vivit.util 함수)
    idx = sample_frame_indices(
        clip_len=cfg["num_frames"],
        frame_sample_rate=cfg["frame_sample_rate"],
        seg_len=total
    )

    frames = read_video_pyav(container, idx)
    container.close()

    # numpy array/list → list[H,W,3] uint8
    fixed = []
    for f in frames:
        f = np.array(f)
        if f.ndim == 3 and f.shape[2] == 3:
            pass
        elif f.ndim == 2:
            f = np.repeat(f[..., None], 3, axis=2)
        elif f.ndim == 3 and f.shape[0] == 3:
            # CHW → HWC
            f = np.transpose(f, (1, 2, 0))
        else:
            raise ValueError(f"Invalid frame shape: {f.shape}")

        if f.dtype != np.uint8:
            if f.max() <= 1.0:
                f = (np.clip(f, 0, 1) * 255).astype(np.uint8)
            else:
                f = np.clip(f, 0, 255).astype(np.uint8)

        fixed.append(f)

    frames = fixed
    # 길이 보정
    if len(frames) < cfg["num_frames"]:
        while len(frames) < cfg["num_frames"]:
            frames.append(frames[-1].copy())
    else:
        frames = frames[:cfg["num_frames"]]

    frames_resized = [
        cv2.resize(fr, (image_size, image_size), interpolation=cv2.INTER_AREA)
        for fr in frames
    ]

    # HF processor로 전처리 (훈련 때와 동일)
    inputs = processor(frames_resized, return_tensors="pt")["pixel_values"].to(next(model.parameters()).device)

    # PeftModel → 내부 VideoMAE로 forward, attentions 추출
    outputs = model(pixel_values=inputs, output_attentions=True)
    logits = outputs.logits
    atts   = outputs.attentions      # tuple of (B,H,S,S)

    pred_cls = int(torch.argmax(logits, dim=-1).item())
    atts = [a.detach() for a in atts]

    # 디버그: 시퀀스 길이/토큰 수
    S = atts[-1].shape[-1]
    h, w = infer_hw_from_config_like(image_size, patch_size)
    patches_per_time = h * w
    T_tokens_est = (S - 1) // patches_per_time if (S - 1) % patches_per_time == 0 else -1
    print(f"  [DBG-pre] S={S}, (S-1)={S-1}, h*w={patches_per_time} -> T_tokens_est={T_tokens_est}")
    print(f"  [DBG-pre] frames_in={len(frames)}, num_frames_cfg={cfg['num_frames']}, sample_rate={cfg['frame_sample_rate']}")

    # rollout
    R = fuse_attentions(
        atts,
        head_fusion=cfg["head_fusion"],
        discard_ratio=cfg["discard_ratio"],
        last_n_layers=cfg["last_n_layers"],
    )
    cams_tokens = rollout_to_cam_safe(R, image_size, patch_size)  # (B, T_tokens, h, w)
    B, T_tokens, h, w = cams_tokens.shape
    T_frames = len(frames)
    print(f"  [DBG-post] cams_tokens.shape={cams_tokens.shape} (T_tokens={T_tokens}, frames={T_frames})")

    # 시간 축 맞추기 (토큰 → 프레임)
    if T_tokens == 0:
        raise RuntimeError("No temporal tokens in CAM.")

    if T_tokens == T_frames:
        cams = cams_tokens
    else:
        reps = int(np.ceil(T_frames / T_tokens))
        cams = np.repeat(cams_tokens, repeats=reps, axis=1)
        cams = cams[:, :T_frames]

    print(f"  [DBG-post] cams_upsampled.shape={cams.shape}")

    os.makedirs(CONF["output_dir"], exist_ok=True)
    out_path = os.path.join(
        CONF["output_dir"],
        f"rollout_{cfg['head_fusion']}_dr{cfg['discard_ratio']}_L{cfg['last_n_layers']}_pred{pred_cls}_{os.path.basename(video_path)}"
    )
    save_cam_video(
        frames_resized,
        cams,
        out_path,
        fps=cfg["fps"],
        p_low=cfg["p_low"],
        p_high=cfg["p_high"],
        gamma=cfg["gamma"],
        cutoff=cfg["cutoff"],
    )
    print(f"  -> saved {out_path}")


# =========================
# Main
# =========================
def main():
    if CONF["device"] == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(CONF["gpu_id"])
    device = torch.device(CONF["device"] if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    # ---- 학습 때 쓰던 VideoMAE + LoRA 구조 재현 + 가중치 로드 ----
    model = build_trained_videomae(CONF["model_path"], device)

    # VideoMAE config에서 image_size / patch_size 가져오기
    # (PeftModel.base_model.model.config 가 실제 VideoMAE backbone config)
    vcfg = model.base_model.model.config
    image_size = vcfg.image_size
    patch_size = vcfg.patch_size   # (tubelet, 16, 16) 같은 튜플일 수 있음

    processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")

    os.makedirs(CONF["output_dir"], exist_ok=True)
    with open(CONF["predictions_file"], "r") as f:
        pred_map = json.load(f)
    pred_map = {k: int(v) for k, v in pred_map.items()}

    for video_path, _label in pred_map.items():
        try:
            process_video(
                video_path=video_path,
                model=model,
                processor=processor,
                image_size=image_size,
                patch_size=patch_size,
                cfg=CONF,
            )
        except Exception as e:
            print(f"[ERROR] Failed on {video_path}: {type(e).__name__} - {e}")
            traceback.print_exc()

    torch.cuda.empty_cache()
    print("\nDone.")

if __name__ == "__main__":
    main()
