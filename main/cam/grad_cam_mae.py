#!/usr/bin/env python3
# VideoMAE Grad-CAM for all videos in a prediction JSON
# 사용법: python mae_gradcam_full.py

import os, json, cv2, av, numpy as np, torch, torch.nn as nn
from typing import List, Dict, Any

from transformers import VideoMAEForVideoClassification

from pytorch_grad_cam import GradCAMPlusPlus, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ==============================================================================
# 1. 설정 (여기만 수정해서 쓰면 됨)
# ==============================================================================
CONFIG = {
    # 학습된 VideoMAE 분류기 가중치
    "MODEL_PATH": "/Data3/hj24/CCTV/main/vivit/videomae_accident_classifier.pth",

    # prediction JSON
    # (키: 비디오 경로 or 파일명, 값: int 또는 {"prediction": int, ...})
    "PREDICTIONS_FILE": "/Data3/hj24/CCTV/main/gradcam/mae_pred/mae_predictions_origin.json",

    # 비디오 데이터 루트
    "DATA_DIR": "/Data3/hj24/CCTV/DATA/cctv_data/cctv_seg",

    # Grad-CAM 결과(mp4, jpg) 출력 폴더
    "OUTPUT_DIR": "/Data3/hj24/CCTV/main/gradcam/mae_pred/mae_grad2cam_output",

    # 사용할 CAM 방법: "gradcam++" 또는 "layercam"
    "CAM_METHOD": "gradcam++",

    # GPU ID
    "GPU_ID": 5,

    # Video 샘플링 설정 (학습 시 사용한 설정과 맞추는게 좋음)
    "NUM_FRAMES": 16,
    "FRAME_SAMPLE_RATE": 4,
    "FPS": 30,

    # Grid 완화용 하이퍼파라미터
    "EMA_ALPHA": 0.7,    # temporal EMA (0.6~0.8 권장)
    "BLUR_SIGMA": 0.8,   # Gaussian blur sigma (0.6~1.2)
    "PCT_LOW": 60.0,     # contrast stretch low percentile
    "PCT_HIGH": 99.5,    # contrast stretch high percentile
    "CUTOFF": 0.2,       # 이 값 아래는 0으로 날려서 약한 grid 제거
}
# ==============================================================================


# -----------------------------
# 2. Video util (PyAV + preprocessing)
# -----------------------------
def preprocess_frames_manual(frames: List[np.ndarray],
                             img_size: int,
                             device: torch.device) -> torch.Tensor:
    """
    frames: list of (H, W, 3) RGB uint8
    return: torch.Tensor of shape (1, T, 3, H, W)
    => 학습 때 image_processor가 만들어주던 shape와 동일 (B, T, C, H, W)
    """
    proc = []
    for fr in frames:
        fr_resized = cv2.resize(fr, (img_size, img_size),
                                interpolation=cv2.INTER_LINEAR)  # (H,W,3)
        fr_f = fr_resized.astype(np.float32) / 255.0
        proc.append(fr_f)

    arr = np.stack(proc, axis=0)        # (T, H, W, 3)
    arr = arr.transpose(0, 3, 1, 2)     # (T, 3, H, W)

    # VideoMAE ImageNet mean/std
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[None, :, None, None]
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)[None, :, None, None]
    arr = (arr - mean) / std            # (T, 3, H, W)

    arr = np.expand_dims(arr, axis=0)   # (1, T, 3, H, W)  <-- 여기까지
    inputs = torch.from_numpy(arr).to(device)
    return inputs

def sample_frame_indices(num_frames: int, sample_rate: int, total: int) -> List[int]:
    """총 total 프레임에서 sample_rate 간격으로 최대 num_frames 샘플."""
    if total <= 0:
        return []
    indices = list(range(0, total, sample_rate))
    if len(indices) < num_frames:
        indices = list(range(total))
    if len(indices) > num_frames:
        indices = indices[:num_frames]
    return indices

def read_video_pyav(path: str, indices: List[int]) -> List[np.ndarray]:
    """지정된 인덱스 프레임만 RGB uint8로 읽기."""
    container = av.open(path)
    stream = container.streams.video[0]
    frames = []
    idx_set = set(indices)
    for i, frame in enumerate(container.decode(stream)):
        if i in idx_set:
            frames.append(frame.to_ndarray(format="rgb24"))
        if i > max(indices):
            break
    container.close()
    return frames

# -----------------------------
# 3. reshape_transform for VideoMAE (patch embedding)
# -----------------------------
def reshape_patch_embed_flexible(act: torch.Tensor):
    """
    patch_embeddings.projection (Conv3D) 출력에 대응:
      - 입력: [B, C, T, H, W]
      - 출력: [B*T, C, H, W]
    """
    if act.dim() == 5:
        B, C, T, H, W = act.shape
        x = act.permute(0, 2, 1, 3, 4).contiguous()  # [B, T, C, H, W]
        return x.view(B * T, C, H, W)
    elif act.dim() == 4:
        # 이미 [B, C, H, W] 인 경우
        return act
    else:
        raise RuntimeError(f"Unexpected activation shape: {act.shape}")

# -----------------------------
# 4. CAM용 wrapper (logits만 뽑기)
# -----------------------------
class LogitsWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
    def forward(self, x):
        # x: (B, T, 3, H, W)  ← 학습 때와 동일
        out = self.model(x)
        return out.logits

# -----------------------------
# 5. CAM method mapping
# -----------------------------
class VideoTargetSizeMixin:
    """입력이 [B, T, 3, H, W]일 때 target size를 (W, H)로 고정."""
    def get_target_width_height(self, input_tensor: torch.Tensor):
        return int(input_tensor.size(-1)), int(input_tensor.size(-2))

class VideoGradCAMPlusPlus(VideoTargetSizeMixin, GradCAMPlusPlus): pass
class VideoLayerCAM(VideoTargetSizeMixin, LayerCAM): pass

VIDEO_CAM_METHODS = {
    "gradcam++": VideoGradCAMPlusPlus,
    "layercam": VideoLayerCAM,
}

# -----------------------------
# 6. Grid 완화용 후처리 함수들
# -----------------------------
def minmax01(x: np.ndarray):
    x = x - x.min()
    m = x.max()
    return x / m if m > 0 else x

def temporal_ema(cams_3d: np.ndarray, alpha: float = 0.7) -> np.ndarray:
    """
    cams_3d: (T, H, W) [0,1] 기준
    시간 방향으로 EMA 해서 프레임 간 튀는 거 줄이기
    """
    out = []
    prev = None
    for t in range(cams_3d.shape[0]):
        cur = cams_3d[t]
        if prev is None:
            sm = cur
        else:
            sm = alpha * prev + (1 - alpha) * cur
        out.append(sm)
        prev = sm
    return np.stack(out, axis=0)

def spatial_refine(cm: np.ndarray,
                   sigma: float = 0.8,
                   pct_low: float = 60.0,
                   pct_high: float = 99.5,
                   cutoff: float = 0.2) -> np.ndarray:
    """
    - GaussianBlur로 patch 경계 부드럽게
    - percentile stretch로 대비 조정
    - cutoff 아래는 0으로 날려서 weak grid 제거
    """
    cm = cv2.GaussianBlur(cm, ksize=(0, 0), sigmaX=sigma)

    lo, hi = np.percentile(cm, pct_low), np.percentile(cm, pct_high)
    if hi > lo:
        cm = np.clip((cm - lo) / (hi - lo), 0, 1)

    cm[cm < cutoff] = 0.0
    return cm

# -----------------------------
# 7. Overlay helper
# -----------------------------
def overlay_heatmap(cam01: np.ndarray, frame_rgb: np.ndarray) -> np.ndarray:
    """
    cam01: (H,W) [0,1]
    frame_rgb: HxWx3 RGB uint8
    -> BGR uint8 with heatmap overlay
    """
    heat = cv2.applyColorMap(
        np.uint8(cam01 * 255), cv2.COLORMAP_JET
    )
    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    out = cv2.addWeighted(heat, 0.6, bgr, 0.4, 0)
    return out

# -----------------------------
# 8. 메인
# -----------------------------
def main():
    # 디바이스 설정
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(CONFIG["GPU_ID"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}, GPU_ID={CONFIG['GPU_ID']}")

    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)

    # -------- 8.1 모델 로드 --------
    print(f"[INFO] Loading VideoMAE backbone + checkpoint: {CONFIG['MODEL_PATH']}")
    model = VideoMAEForVideoClassification.from_pretrained(
        "MCG-NJU/videomae-base-finetuned-kinetics",
        use_safetensors=True,
        attn_implementation="sdpa",
        torch_dtype=torch.float32,
    )

    # 이진 classifier head 교체 후 checkpoint 로드
    model.classifier = nn.Linear(model.classifier.in_features, 2)
    sd = torch.load(CONFIG["MODEL_PATH"], map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)

    model.to(device).eval()
    logits_model = LogitsWrapper(model).to(device).eval()

    img_size = model.config.image_size

    # target layer = patch embedding (Conv3D)
    target_layer = model.videomae.embeddings.patch_embeddings.projection

    cam_method = CONFIG["CAM_METHOD"]
    CAMClass = VIDEO_CAM_METHODS[cam_method]

    cam = CAMClass(
        model=logits_model,
        target_layers=[target_layer],
        reshape_transform=reshape_patch_embed_flexible,
    )
    cam.batch_size = 1  # 영상 단위 처리

    # -------- 8.2 prediction JSON 로드 --------
    with open(CONFIG["PREDICTIONS_FILE"], "r", encoding="utf-8") as f:
        pred_map_raw: Dict[str, Any] = json.load(f)

    # pred_map: {video_key -> pred_class(int)}
    pred_map: Dict[str, int] = {}
    for k, v in pred_map_raw.items():
        if isinstance(v, dict) and "prediction" in v:
            pred_map[k] = int(v["prediction"])
        elif isinstance(v, int):
            pred_map[k] = int(v)
        else:
            print(f"[WARN] Unknown JSON format for key={k}, value={v}. Skip.")
    print(f"[INFO] Loaded {len(pred_map)} prediction entries from JSON")

    data_dir = CONFIG["DATA_DIR"]

    # -------- 8.3 각 비디오에 대해 CAM 생성 --------
    for vid_key, pred_cls in pred_map.items():
        # vid_key가 full-path 일 수도, 파일명만 일 수도 있음
        cand_paths = []
        if os.path.exists(vid_key):
            cand_paths.append(vid_key)
        if data_dir:
            candidate = os.path.join(data_dir, vid_key)
            if os.path.exists(candidate):
                cand_paths.append(candidate)

        if not cand_paths:
            print(f"[WARN] Cannot find video for key '{vid_key}'. Skip.")
            continue

        video_path = cand_paths[0]
        print(f"\n[Video] {video_path} | pred_cls={pred_cls}")

        # 1) 프레임 메타
        try:
            container = av.open(video_path)
            total_frames = container.streams.video[0].frames
            container.close()
        except Exception as e:
            print(f"  -> [ERR] open fail: {e}")
            continue

        if total_frames == 0:
            print("  -> [WARN] no frames. Skip.")
            continue

        # 2) 프레임 샘플링 & 디코딩
        indices = sample_frame_indices(
            CONFIG["NUM_FRAMES"],
            CONFIG["FRAME_SAMPLE_RATE"],
            total_frames
        )
        try:
            frames = read_video_pyav(video_path, indices)
        except Exception as e:
            print(f"  -> [ERR] decode fail: {e}")
            continue

        if len(frames) == 0:
            print("  -> [WARN] No decoded frames. Skip.")
            continue

        # 부족하면 반복 패딩
        if len(frames) < CONFIG["NUM_FRAMES"]:
            while len(frames) < CONFIG["NUM_FRAMES"]:
                frames.append(frames[-1].copy())
        else:
            frames = frames[:CONFIG["NUM_FRAMES"]]

        T_frames = len(frames)
        H0, W0 = frames[0].shape[:2]

        # 3) 입력 텐서 생성: (1, T, 3, H, W)
        inputs = preprocess_frames_manual(
            frames,
            img_size=img_size,
            device=device
        )
        print(f"  -> inputs.shape = {tuple(inputs.shape)}")

        # 4) CAM 계산
        try:
            cams_bt_hw = cam(
                input_tensor=inputs,
                targets=[ClassifierOutputTarget(pred_cls)],
            )
        except Exception as e:
            print(f"  -> [ERR] CAM computation failed: {e}")
            continue

        BT, Hc, Wc = cams_bt_hw.shape
        B = inputs.shape[0]
        T_tokens = BT // B
        cams_b_t_h_w = cams_bt_hw.reshape(B, T_tokens, Hc, Wc)
        cams_tokens = cams_b_t_h_w[0]  # (T_tokens, Hc, Wc)

        # VideoMAE: tubelet_size=2라서 T_tokens ~= T_frames/2 인 경우가 많음
        if T_tokens == T_frames:
            cams_3d = cams_tokens.copy()
        else:
            reps = int(np.ceil(T_frames / T_tokens))
            cams_3d = np.repeat(cams_tokens, repeats=reps, axis=0)[:T_frames]

        # ---- grid 완화 후처리 ----
        cams_3d = cams_3d.astype(np.float32)

        # (1) per-frame min-max
        for t in range(T_frames):
            cams_3d[t] = minmax01(cams_3d[t])

        # (2) temporal EMA
        cams_3d = temporal_ema(cams_3d, alpha=CONFIG["EMA_ALPHA"])

        # (3) spatial refine (Gaussian + percentile + cutoff)
        for t in range(T_frames):
            cams_3d[t] = spatial_refine(
                cams_3d[t],
                sigma=CONFIG["BLUR_SIGMA"],
                pct_low=CONFIG["PCT_LOW"],
                pct_high=CONFIG["PCT_HIGH"],
                cutoff=CONFIG["CUTOFF"],
            )

        # 전체 클립 기준 한 번 더 0~1 정규화
        cams_3d -= cams_3d.min()
        if cams_3d.max() > 0:
            cams_3d /= cams_3d.max()

        print(f"  -> cams_3d.shape = {cams_3d.shape} (frames={T_frames}, Hc={Hc}, Wc={Wc})")

        # 5) 비디오 저장
        out_video = os.path.join(
            CONFIG["OUTPUT_DIR"],
            f"{cam_method}_pred{pred_cls}_" + os.path.basename(video_path),
        )
        vw = cv2.VideoWriter(
            out_video,
            cv2.VideoWriter_fourcc(*"mp4v"),
            CONFIG["FPS"],
            (W0, H0),
        )
        if not vw.isOpened():
            print(f"  -> [ERR] VideoWriter open failed: {out_video}")
            continue

        out_frames = []
        for fr, cm in zip(frames, cams_3d):
            cm_up = cv2.resize(cm, (W0, H0), interpolation=cv2.INTER_CUBIC)
            out_bgr = overlay_heatmap(cm_up, fr)
            vw.write(out_bgr)
            out_frames.append(out_bgr)
        vw.release()
        print(f"  -> saved video: {out_video}")

        # 6) 스냅샷 저장 (최대 CAM 프레임)
        idx_max = int(np.argmax([c.max() for c in cams_3d]))
        snap = out_frames[idx_max]
        snap_path = os.path.join(
            CONFIG["OUTPUT_DIR"],
            os.path.basename(video_path) + "_snap.jpg",
        )
        cv2.imwrite(snap_path, snap)
        print(f"  -> saved snapshot: {snap_path}")

    torch.cuda.empty_cache()
    print("\n[INFO] All done.")


if __name__ == "__main__":
    main()
