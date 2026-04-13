#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Timesformer Grad-CAM 시각화 스크립트 (설정/경로 모두 코드 내부 고정 버전)

- 모델: facebook/timesformer-base-finetuned-k400 를 base로 하고
        timesformer_accident_classifier.pth 로 fine-tuned 가중치 로드
- 입력: tformer_predictions_origin.json (test set 예측 결과)
- 출력: method 별 overlay 비디오 + 스냅샷 이미지

※ 아래 CONFIG 섹션에서 method, 경로, 하이퍼파라미터만 바꿔서 사용
"""

import os, json, cv2, av, numpy as np, torch, torch.nn as nn
from transformers import AutoImageProcessor, TimesformerForVideoClassification

from pytorch_grad_cam import (
    GradCAM, XGradCAM, GradCAMPlusPlus, EigenCAM, EigenGradCAM,
    LayerCAM, AblationCAM, FullGrad
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import sys
sys.path.append("/Data3/hj24/CCTV/main")
from vivit import sample_frame_indices, read_video_pyav

from typing import Sequence

# ============================================================
# 1) CONFIG: 여기만 수정해서 사용
# ============================================================

# 디바이스 설정
DEVICE      = "cuda"   # 또는 "cpu"
GPU_ID      = 1        # 사용할 GPU id

# 모델 / JSON / 출력 경로
MODEL_PATH  = "/Data3/hj24/CCTV/main/vivit/timesformer_accident_classifier.pth"
PREDICTIONS = "/Data3/hj24/CCTV/main/gradcam/tformer_pred/tformer_predictions_origin.json"

# CAM 방법: "layercam" 또는 "gradcam++"
CAM_METHOD  = "gradcam++"  # ← 여기 바꿔가면서 2번 돌리면 됨

# 출력 루트 디렉토리 (method 이름을 서브폴더에 붙임)
OUTPUT_ROOT = "/Data3/hj24/CCTV/main/gradcam/tformer_pred"

# 비디오 관련 설정
NUM_FRAMES        = 32
FRAME_SAMPLE_RATE = 4
FPS               = 30
TOKEN_ORDER       = "t_hw"   # "t_hw" 또는 "hw_t"

# Post-processing 하이퍼파라미터
#   - LayerCAM: 국소 + 안정 강조용
#   - GradCAM++: 약간 더 넓게/샤프하게
if CAM_METHOD.lower() == "layercam":
    EMA_ALPHA  = 0.75
    BLUR_SIGMA = 0.8
    PCT_LOW    = 75.0
    PCT_HIGH   = 99.0
    CUTOFF     = 0.25
elif CAM_METHOD.lower() == "gradcam++":
    EMA_ALPHA  = 0.6
    BLUR_SIGMA = 1.0
    PCT_LOW    = 60.0
    PCT_HIGH   = 99.5
    CUTOFF     = 0.20
else:
    # 기본값(다른 메소드 쓰고 싶을 때)
    EMA_ALPHA  = 0.7
    BLUR_SIGMA = 0.9
    PCT_LOW    = 70.0
    PCT_HIGH   = 99.0
    CUTOFF     = 0.2

# ============================================================
# 2) Grad-CAM helper 설정
# ============================================================

class VideoTargetSizeMixin:
    """비디오 입력 [B, T, 3, H, W] 기준으로 target size를 (W,H)로 고정."""
    def get_target_width_height(self, input_tensor):
        return (int(input_tensor.size(-1)), int(input_tensor.size(-2)))

class VideoGradCAM(VideoTargetSizeMixin, GradCAM): pass
class VideoXGradCAM(VideoTargetSizeMixin, XGradCAM): pass
class VideoGradCAMPlusPlus(VideoTargetSizeMixin, GradCAMPlusPlus): pass
class VideoEigenCAM(VideoTargetSizeMixin, EigenCAM): pass
class VideoEigenGradCAM(VideoTargetSizeMixin, EigenGradCAM): pass
class VideoLayerCAM(VideoTargetSizeMixin, LayerCAM): pass
class VideoAblationCAM(VideoTargetSizeMixin, AblationCAM): pass
class VideoFullGrad(VideoTargetSizeMixin, FullGrad): pass

VIDEO_CAM_METHODS = {
    "gradcam":    VideoGradCAM,
    "xgradcam":   VideoXGradCAM,
    "gradcam++":  VideoGradCAMPlusPlus,
    "eigencam":   VideoEigenCAM,
    "eigengradcam": VideoEigenGradCAM,
    "layercam":   VideoLayerCAM,
    "ablationcam":VideoAblationCAM,
    "fullgrad":   VideoFullGrad,
}

# ============================================================
# 3) reshape / overlay / 후처리 함수
# ============================================================

def make_reshape_transform(image_size, patch_size, token_order):
    H = W = image_size // patch_size
    def reshape_transform(tokens_BNC):
        # tokens_BNC: [B, N, C], N = 1 + T*H*W
        import torch
        B, N, C = tokens_BNC.shape
        patch = tokens_BNC[:, 1:, :]
        HW = H * W
        assert patch.shape[1] % HW == 0, f"Token count mismatch: {patch.shape[1]} not multiple of {HW}"
        T = patch.shape[1] // HW

        if token_order == "t_hw":
            x = patch.view(B, T, H, W, C).permute(0,1,4,2,3).contiguous()   # [B,T,C,H,W]
        else:
            x = patch.view(B, H, W, T, C).permute(0,3,4,1,2).contiguous()   # [B,T,C,H,W]
        return x.view(B*T, C, H, W)  # pytorch-grad-cam이 기대하는 [B',C,H,W]
    return reshape_transform

def reshape_patch_embed_flexible(act):
    """patch_embed 출력이 5D(B,C,T,H,W) 또는 4D(B,T,H,W)로 들어올 때 대응."""
    import torch
    if act.dim() == 5:
        B, C, T, H, W = act.shape
        return act.permute(0, 2, 1, 3, 4).contiguous().view(B*T, C, H, W)
    elif act.dim() == 4:
        return act
    else:
        raise RuntimeError(f"patch_embed expected 4D/5D, got {tuple(act.shape)}")

def heatmap_overlay(cam01, frame_rgb):
    """
    cam01: [H,W] 0~1
    frame_rgb: HxWx3 uint8 (RGB라고 가정)
    """
    heat = cv2.applyColorMap(np.uint8(cam01 * 255), cv2.COLORMAP_JET)
    bgr  = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(heat, 0.6, bgr, 0.4, 0)

def temporal_ema(cams_3d: np.ndarray, alpha=0.7):
    """cams_3d: (T,H,W) in [0,1]"""
    out, prev = [], None
    for t in range(cams_3d.shape[0]):
        cur = cams_3d[t]
        sm = cur if prev is None else (alpha * prev + (1 - alpha) * cur)
        out.append(sm); prev = sm
    return np.stack(out, axis=0)

def spatial_refine(cm: np.ndarray, sigma=0.8, pct_low=70.0, pct_high=99.0, cutoff=0.2):
    cm = cv2.GaussianBlur(cm, ksize=(0,0), sigmaX=sigma)
    lo, hi = np.percentile(cm, pct_low), np.percentile(cm, pct_high)
    if hi > lo:
        cm = np.clip((cm - lo) / (hi - lo), 0, 1)
    cm[cm < cutoff] = 0.0
    return cm

def minmax01(x: np.ndarray):
    x = x - x.min()
    m = x.max()
    return x / m if m > 0 else x

# ============================================================
# 4) 메인 로직
# ============================================================

def main():
    # ---------- 디바이스 ----------
    if DEVICE == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(GPU_ID)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[INFO] Using device: {device}")

    # ---------- 모델 로드 ----------
    backbone_name = "facebook/timesformer-base-finetuned-k400"
    model = TimesformerForVideoClassification.from_pretrained(backbone_name)
    model.classifier = nn.Linear(model.classifier.in_features, 2)

    sd = torch.load(MODEL_PATH, map_location="cpu")
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("[INFO] missing keys:", missing)
    print("[INFO] unexpected keys:", unexpected)

    model.to(device).eval()

    # logits만 출력하는 wrapper
    class LogitsWrapper(torch.nn.Module):
        def __init__(self, model): 
            super().__init__()
            self.model = model
        def forward(self, x): 
            return self.model(x).logits

    model_logits = LogitsWrapper(model).to(device).eval()

    # ---------- processor & reshape ----------
    proc      = AutoImageProcessor.from_pretrained(backbone_name)
    img_size  = model.config.image_size
    patch_size= model.config.patch_size
    reshape_transform = make_reshape_transform(img_size, patch_size, TOKEN_ORDER)

    # ---------- IO 준비 ----------
    method_lower = CAM_METHOD.lower()
    if method_lower not in VIDEO_CAM_METHODS:
        raise ValueError(f"Unsupported CAM_METHOD: {CAM_METHOD}")

    out_dir = os.path.join(OUTPUT_ROOT, f"tformer_{method_lower}_output")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] CAM_METHOD = {CAM_METHOD}, OUTPUT_DIR = {out_dir}")

    with open(PREDICTIONS, "r") as f:
        pred_map = json.load(f)
    pred_map = {k: int(v) for k, v in pred_map.items()}
    print(f"[INFO] 대상 비디오 수: {len(pred_map)}")

    # ---------- 각 비디오별 CAM ----------
    for video_path, pred_cls in pred_map.items():
        print(f"\n[Video] {video_path}")
        try:
            container = av.open(video_path)
        except Exception as e:
            print(f"  -> open fail: {e}")
            continue

        total = container.streams.video[0].frames
        if total == 0:
            print("  -> no frames")
            container.close()
            continue

        idx = sample_frame_indices(NUM_FRAMES, FRAME_SAMPLE_RATE, total)
        frames = read_video_pyav(container, idx)
        container.close()

        # frame 정리
        fixed = []
        for fr in frames:
            fr = np.squeeze(fr)
            if fr.ndim != 3 or fr.shape[2] != 3:
                raise ValueError(f"Invalid frame shape: {fr.shape}")
            if fr.dtype != np.uint8:
                fr = (np.clip(fr,0,1)*255).astype(np.uint8)
            fixed.append(fr)
        frames = fixed[:NUM_FRAMES]
        while len(frames) < NUM_FRAMES:
            frames.append(frames[-1].copy())

        # 입력 텐서
        inputs = proc(list(frames), return_tensors="pt")["pixel_values"].to(device)  # [1,T,3,224,224]

        # 예측 클래스 (pred_json과 실제 모델 출력이 다를 수도 있지만, 여기선 모델 예측 확인만)
        with torch.no_grad():
            logits = model_logits(inputs)
        model_pred = int(torch.argmax(logits, dim=-1).item())
        print(f"  -> pred_json cls={pred_cls}, model cls={model_pred}")

        # ---------- target layer 후보들 (우선순위) ----------
        L = model.timesformer.encoder.layer
        target_schemes = [
            dict(name="patch_embed",
                 layer=[model.timesformer.embeddings.patch_embeddings.projection],
                 reshape=reshape_patch_embed_flexible),
            dict(name="mlp_intermediate_m3",
                 layer=[L[-3].intermediate.dense],
                 reshape=reshape_transform),
            dict(name="mlp_intermediate_m1",
                 layer=[L[-1].intermediate.dense],
                 reshape=reshape_transform),
        ]

        # ---------- CAM 계산 (스킴 fallback) ----------
        CAMClass = VIDEO_CAM_METHODS[method_lower]
        cam_bt_hw = None
        found = False

        for scheme in target_schemes:
            try:
                cam_obj = CAMClass(
                    model=model_logits,
                    target_layers=scheme["layer"],
                    reshape_transform=scheme["reshape"],
                )
                cam_obj.batch_size = min(8, NUM_FRAMES)

                cam_bt_hw = cam_obj(
                    input_tensor=inputs,
                    targets=[ClassifierOutputTarget(pred_cls)],
                )  # numpy [B*T, Hc, Wc]

                cmin, cmax = float(cam_bt_hw.min()), float(cam_bt_hw.max())
                print(f"[TRY:{scheme['name']}] CAM stats min={cmin:.6f} max={cmax:.6f} shape={cam_bt_hw.shape}")
                if cmax > 1e-7:
                    print(f"[OK] use target='{scheme['name']}'")
                    found = True
                    break
                else:
                    print(f"[WARN] target='{scheme['name']}' produced ~0. Trying next...")
            except Exception as e:
                print(f"[ERR] target='{scheme['name']}' failed: {type(e).__name__} - {e}")

        if not found or cam_bt_hw is None:
            print("[FATAL] All target schemes produced zero CAM. Skip.")
            continue

        # ---------- (B*T,Hc,Wc) -> (T,Hc,Wc) ----------
        B, T = inputs.shape[0], inputs.shape[1]
        Hc, Wc = cam_bt_hw.shape[-2], cam_bt_hw.shape[-1]
        cam_bthw = cam_bt_hw.reshape(B, T, Hc, Wc)    # [1,T,Hc,Wc]

        # per-frame 통계
        per_frame_max  = [float(cam_bthw[0,t].max())  for t in range(T)]
        per_frame_mean = [float(cam_bthw[0,t].mean()) for t in range(T)]
        print("[FRAME MAX][:8]  ", [round(x,4) for x in per_frame_max[:8]], "...")
        print("[FRAME MEAN][:8] ", [round(x,4) for x in per_frame_mean[:8]], "...")

        # ---------- 후처리: EMA + blur + percentile cut ----------
        cams_3d = cam_bthw[0].copy()                   # (T,Hc,Wc)
        for t in range(T):
            cams_3d[t] = minmax01(cams_3d[t])
        cams_3d = temporal_ema(cams_3d, alpha=EMA_ALPHA)
        for t in range(T):
            cams_3d[t] = spatial_refine(
                cams_3d[t],
                sigma=BLUR_SIGMA,
                pct_low=PCT_LOW,
                pct_high=PCT_HIGH,
                cutoff=CUTOFF,
            )

        cams_list = [cams_3d[t] for t in range(T)]     # list of (Hc,Wc), [0,1]

        # ---------- 비디오 저장 ----------
        h, w = frames[0].shape[:2]
        out_vid = os.path.join(
            out_dir,
            f"{CAM_METHOD}_pred{pred_cls}_{os.path.basename(video_path)}"
        )
        vw = cv2.VideoWriter(out_vid, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (w, h))
        for fr, cm in zip(frames, cams_list):
            cm_up = cv2.resize(cm.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
            out_bgr = heatmap_overlay(cm_up, fr)
            vw.write(out_bgr)
        vw.release()
        print(f"  -> saved video: {out_vid}")

        # ---------- 스냅샷 저장 (가장 뜨거운 frame) ----------
        max_idx = int(np.argmax([c.max() for c in cams_list]))
        cm_up = cv2.resize(cams_list[max_idx], (w, h), interpolation=cv2.INTER_CUBIC)
        snap = heatmap_overlay(cm_up, frames[max_idx])
        snap_path = os.path.join(out_dir, f"{os.path.basename(video_path)}_snap.jpg")
        cv2.imwrite(snap_path, snap)
        print(f"  -> saved snapshot: {snap_path}")

    torch.cuda.empty_cache()
    print("\n[Done]")

# ============================================================
# 5) 엔트리 포인트
# ============================================================

if __name__ == "__main__":
    main()
