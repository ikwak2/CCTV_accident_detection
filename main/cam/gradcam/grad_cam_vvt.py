#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ViViT + LoRA용 Grad-CAM++ / LayerCAM 스크립트 (test JSON 기반)

- 학습에 사용한 vivit_accident_classifier.pth 로드
- vivit_predictions_origin.json (evaluate_test()에서 저장한) 기준으로
  test set 영상만 대상으로 CAM 생성
- CAM_METHOD = "gradcampp" 또는 "layercam" 선택
- 각 비디오마다:
    - center frame jpg
    - CAM npy
    - overlay mp4 저장
"""

import os
import cv2
import av
import json
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VivitForVideoClassification, VivitImageProcessor
from peft import LoraConfig, get_peft_model

# =========================
# 기본 환경 세팅
# =========================

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"   # CAM은 1개 GPU로도 충분
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 설정 =====
# "gradcampp" 또는 "layercam" (원하면 "gradcam"도 추가 가능)
CAM_METHOD = "layercam"

MODEL_PATH = "/Data3/hj24/CCTV/main/vivit/vivit_accident_classifier.pth"
PRED_JSON  = "/Data3/hj24/CCTV/main/gradcam/vivit_pred/vivit_predictions_origin.json"

OUT_ROOT    = f"/Data3/hj24/CCTV/main/gradcam/vivit_pred/vvt_{CAM_METHOD}"
OUT_FRAME_DIR = os.path.join(OUT_ROOT, "frames")
OUT_CAM_DIR   = os.path.join(OUT_ROOT, "npy")
OUT_VIDEO_DIR = os.path.join(OUT_ROOT, "vis")

NUM_CLASSES = 2
CLIP_LEN = 32
FRAME_SAMPLE_RATE = 4

# =========================
# 비디오 유틸 (학습 때 쓰던 것 그대로)
# =========================

def safe_total_frames(video_path, container=None):
    """PyAV frame count가 0/None이면 OpenCV로 폴백."""
    try:
        if container is not None and container.streams.video and container.streams.video[0].frames:
            tf = int(container.streams.video[0].frames)
            if tf > 0:
                return tf
    except Exception:
        pass

    cap = cv2.VideoCapture(video_path)
    tf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return tf


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    """
    학습 코드에서 쓰던 버전.
    clip_len: 모델에 넣을 프레임 개수(예: 32)
    frame_sample_rate: 샘플링 간격(예: 4)
    seg_len: 비디오 전체 프레임 수
    """
    converted_len = int(clip_len * frame_sample_rate)

    if seg_len == 0:
        raise ValueError("ERROR: 비디오의 총 프레임 수가 0입니다. 해당 비디오는 처리 불가")

    if seg_len < converted_len:
        converted_len = seg_len

    end_idx = np.random.randint(converted_len, seg_len) if seg_len > converted_len else seg_len
    start_idx = max(0, end_idx - converted_len)

    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)

    return indices


def read_video_pyav(container, indices, max_try=2):
    """
    robust PyAV + OpenCV 폴백 버전.
    반환: (T, H, W, 3) RGB np.uint8
    """
    idx_set = set(int(i) for i in indices.tolist() if i >= 0)
    frames = []

    # 1차: PyAV
    try:
        container.seek(0)
        for i, frame in enumerate(container.decode(video=0)):
            if i > max(idx_set):
                break
            if i in idx_set:
                frames.append(frame.to_ndarray(format="rgb24"))
    except Exception:
        frames = []

    if len(frames) > 0:
        return np.stack(frames, axis=0)

    # 2차: OpenCV 폴백
    video_path = getattr(container, "name", None)
    if not video_path:
        raise ValueError("OpenCV 폴백에 필요한 경로가 없어 디코딩 실패")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"OpenCV로 열 수 없음: {video_path}")

    frames_cv = []
    for idx in sorted(idx_set):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, bgr = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames_cv.append(rgb)
        if len(frames_cv) >= len(idx_set):
            break
    cap.release()

    if len(frames_cv) == 0 and max_try > 0:
        cap = cv2.VideoCapture(video_path)
        ok = True
        while ok and len(frames_cv) < len(idx_set):
            ok, bgr = cap.read()
            if ok:
                frames_cv.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cap.release()

    if len(frames_cv) == 0:
        raise ValueError("need at least one array to stack (모든 디코딩 경로 실패)")

    T = len(indices)
    if len(frames_cv) < T:
        last = frames_cv[-1]
        frames_cv += [last] * (T - len(frames_cv))

    return np.stack(frames_cv[:T], axis=0)


# =========================
# ViViT + LoRA 모델 복원
# =========================

def build_vivit_model(model_path, num_classes=2):
    """
    학습 때 썼던 구조 그대로 복원:
    - VivitForVideoClassification (google/vivit-b-16x2-kinetics400)
    - classifier: Linear -> 256 -> ReLU -> Dropout -> Linear(num_classes)
    - LoRA 설정 동일
    """
    base_model = VivitForVideoClassification.from_pretrained(
        "google/vivit-b-16x2-kinetics400",
        use_safetensors=True,
        attn_implementation="sdpa",
        dtype=torch.float32,
    )

    processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")

    base_model.config.num_labels = num_classes
    in_feats = base_model.classifier.in_features
    base_model.classifier = nn.Sequential(
        nn.Linear(in_feats, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=[
            "attention.attention.query",
            "attention.attention.key",
            "attention.attention.value",
            "attention.output.dense",
        ],
        modules_to_save=["classifier"],
    )
    model = get_peft_model(base_model, lora_config)

    state = torch.load(model_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("[INFO] missing keys:", missing)
    print("[INFO] unexpected keys:", unexpected)

    model.to(device)
    model.eval()
    return model, processor


# =========================
# ViViT 전용 reshape_transform
# =========================

def vivit_reshape_transform(tensor, config):
    """
    tensor: [B, N, C]
      - ViViT 인코더 레이어 출력 (CLS 포함 or 미포함일 수 있음)
    config: backbone.config (VivitConfig)

    동작:
      1) image_size, tubelet_size로 프레임당 패치 개수 계산
      2) N, 패치 개수로 프레임 수 F 추정
      3) [B, F, H, W, C]로 reshape 후 시간(F) 방향 평균
      4) [B, C, H, W] 반환
    """
    B, N, C = tensor.shape

    img_size = config.image_size

    # ViViT는 tubelet_size = [t, h, w]
    if hasattr(config, "tubelet_size"):
        patch_h = config.tubelet_size[1]
        patch_w = config.tubelet_size[2]
    elif hasattr(config, "patch_size"):
        patch_h = patch_w = config.patch_size
    else:
        raise AttributeError("VivitConfig에 tubelet_size/patch_size 둘 다 없습니다.")

    h_patches = img_size // patch_h
    w_patches = img_size // patch_w
    num_patches_per_frame = h_patches * w_patches  # 예: 14*14=196

    # CLS 토큰 존재 여부에 따라 분기
    if N % num_patches_per_frame == 0:
        F = N // num_patches_per_frame
        patch_tokens = tensor
    elif (N - 1) % num_patches_per_frame == 0:
        F = (N - 1) // num_patches_per_frame
        patch_tokens = tensor[:, 1:, :]  # CLS 제외
    else:
        raise ValueError(
            f"Cannot infer (frames, patches) from sequence length N={N} "
            f"and patches/frame={num_patches_per_frame}"
        )

    patch_tokens = patch_tokens.reshape(B, F, h_patches, w_patches, C)
    patch_tokens = patch_tokens.mean(dim=1)          # [B, H, W, C]
    patch_tokens = patch_tokens.permute(0, 3, 1, 2)  # [B, C, H, W]
    return patch_tokens


# =========================
# CAM 계산 함수 (Grad-CAM / Grad-CAM++ / LayerCAM)
# =========================

def compute_cam(A_map, G_map, method="gradcampp"):
    """
    A_map: [1, C, H, W] activation
    G_map: [1, C, H, W] gradient
    method: "gradcam", "gradcampp", "layercam"
    return: [H, W] (numpy, 0~1)
    """
    # 양수 activation만 사용 (음수 쪽은 억제)
    A_pos = torch.relu(A_map)

    if method.lower() == "gradcam":
        # 기본 Grad-CAM
        weights = G_map.mean(dim=(2, 3), keepdim=True)          # [1,C,1,1]
        cam = (weights * A_pos).sum(dim=1)                      # [1,H,W]

    elif method.lower() == "gradcampp":
        # Grad-CAM++
        eps = 1e-6
        G2 = G_map ** 2
        G3 = G_map ** 3

        # Σ_{i,j} A * G^3
        sum_A_G3 = (A_pos * G3).sum(dim=(2, 3), keepdim=True)   # [1,C,1,1]

        alpha = G2 / (2 * G2 + sum_A_G3 + eps)                  # [1,C,H,W]
        reluG = torch.relu(G_map)                               # [1,C,H,W]

        # weights_k = Σ_{i,j} alpha_{ij}^k * ReLU(grad_{ij}^k)
        weights = (alpha * reluG).sum(dim=(2, 3), keepdim=True) # [1,C,1,1]

        cam = (weights * A_pos).sum(dim=1)                      # [1,H,W]

    elif method.lower() == "layercam":
        # LayerCAM: 위치별로 ReLU(grad) * ReLU(activation) → 채널 합
        reluG = torch.relu(G_map)                               # [1,C,H,W]
        cam = (reluG * A_pos).sum(dim=1)                        # [1,H,W]
    else:
        raise ValueError(f"Unknown CAM method: {method}")

    cam = torch.relu(cam)                                      # 음수 제거
    cam = cam[0].detach().cpu().numpy()                        # [H,W]

    # 정규화
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    else:
        cam = np.zeros_like(cam)

    return cam


# =========================
# 단일 비디오에 대해 CAM 생성
# =========================

def cam_for_video(
    model,
    backbone,
    processor,
    video_path,
    out_frame_dir,
    out_cam_dir,
    out_video_dir,
    cam_method="gradcampp",
    target_class=None,
    clip_len=32,
    frame_sample_rate=4,
):
    print(f"\n[INFO] ==== Processing video ====")
    print(f"[INFO] video_path: {video_path}")

    # 1) 비디오 디코딩
    try:
        container = av.open(video_path, options={"err_detect": "ignore_err"})
    except Exception as e:
        print(f"[WARN] av.open 실패: {video_path} -> {e}")
        return

    total_frames = safe_total_frames(video_path, container)
    print(f"[DEBUG] total_frames: {total_frames}")
    if total_frames is None or total_frames <= 0:
        print(f"[WARN] total_frames=0: {video_path}")
        container.close()
        return

    if total_frames < 5:
        print(f"[WARN] 너무 짧은 비디오: {video_path}")
        container.close()
        return

    indices = sample_frame_indices(clip_len=clip_len,
                                   frame_sample_rate=frame_sample_rate,
                                   seg_len=total_frames)
    print(f"[DEBUG] sampled indices (len={len(indices)}): {indices[:5]} ... {indices[-5:]}")

    try:
        frames = read_video_pyav(container, indices)
    finally:
        try:
            container.close()
        except:
            pass

    print(f"[DEBUG] frames type: {type(frames)}, shape: {getattr(frames, 'shape', None)}")
    if not isinstance(frames, np.ndarray):
        print("[ERROR] frames is not a numpy array!")
        return

    T, H, W, _ = frames.shape
    print(f"[DEBUG] frames (T,H,W): {T}, {H}, {W}")

    # 2) Processor
    inputs = processor(list(frames), return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)   # [1,T,3,224,224]
    print(f"[DEBUG] pixel_values shape: {tuple(pixel_values.shape)}")

    # 3) target layer 선택 (MLP output.dense -> attention.output.dense -> layernorm 순)
    enc = backbone.encoder
    if hasattr(enc, "layer"):
        last_block = enc.layer[-1]
    elif hasattr(enc, "layers"):
        last_block = enc.layers[-1]
    else:
        print("[ERROR] encoder에 layer(s)가 없습니다.")
        return

    if hasattr(last_block, "output") and hasattr(last_block.output, "dense"):
        target_layer = last_block.output.dense
        print("[DEBUG] target_layer: last_block.output.dense")
    elif hasattr(last_block, "attention") and hasattr(last_block.attention, "output") \
            and hasattr(last_block.attention.output, "dense"):
        target_layer = last_block.attention.output.dense
        print("[DEBUG] target_layer: last_block.attention.output.dense")
    elif hasattr(last_block, "layernorm"):
        target_layer = last_block.layernorm
        print("[DEBUG] target_layer: last_block.layernorm (fallback)")
    else:
        target_layer = last_block
        print("[DEBUG] target_layer: last_block (fallback)")

    activations = {}
    gradients = {}

    def fwd_hook(module, inp, out):
        activations["value"] = out

    def bwd_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0]

    handle_fwd = target_layer.register_forward_hook(fwd_hook)
    handle_bwd = target_layer.register_full_backward_hook(bwd_hook)

    # 4) Forward & Backward
    try:
        model.zero_grad()
        outputs = model(pixel_values)
        logits = outputs.logits   # [1, num_classes]
        print(f"[DEBUG] logits shape: {tuple(logits.shape)}")

        if target_class is None:
            cls = int(torch.argmax(logits, dim=1).item())
        else:
            cls = int(target_class)
        print(f"[DEBUG] CAM target_class: {cls}")

        score = logits[0, cls]
        score.backward()

        A = activations["value"]   # [1, N, C] or [1, H', W', C] (ViViT는 [1,N,C])
        G = gradients["value"]     # same shape as A
        print(f"[DEBUG] A shape: {A.shape}, G shape: {G.shape}")

    except Exception as e:
        print(f"[ERROR] Grad-CAM forward/backward 단계에서 에러: {e}")
        handle_fwd.remove()
        handle_bwd.remove()
        return

    handle_fwd.remove()
    handle_bwd.remove()

    # 5) 토큰 -> [B,C,H,W] 변환
    config = backbone.config
    # ViViT encoder는 [B,N,C] 구조일 가능성이 높아서 reshape_transform 적용
    if A.dim() == 3:
        A_map = vivit_reshape_transform(A, config=config)  # [1,C,H,W]
        G_map = vivit_reshape_transform(G, config=config)  # [1,C,H,W]
    elif A.dim() == 4:
        # 혹시 이미 [B,H,W,C] 형태일 경우 대비 (거의 없을 것)
        B_, H_, W_, C_ = A.shape
        A_map = A.permute(0, 3, 1, 2)                      # [B,C,H,W]
        G_map = G.permute(0, 3, 1, 2)
    else:
        print(f"[ERROR] Unsupported activation dim: {A.dim()}")
        return

    print(f"[DEBUG] A_map shape: {A_map.shape}, G_map shape: {G_map.shape}")

    # 6) CAM 계산
    cam = compute_cam(A_map, G_map, method=cam_method)
    print(f"[DEBUG] cam stats: min={cam.min():.6f}, max={cam.max():.6f}, mean={cam.mean():.6f}")

    # 7) 결과 저장
    os.makedirs(out_cam_dir, exist_ok=True)
    os.makedirs(out_frame_dir, exist_ok=True)
    os.makedirs(out_video_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    cam_npy_path = os.path.join(out_cam_dir, base_name + ".npy")
    np.save(cam_npy_path, cam.astype(np.float32))

    center_idx = len(frames) // 2
    center_frame = frames[center_idx]  # RGB
    frame_save_path = os.path.join(out_frame_dir, base_name + ".jpg")
    cv2.imwrite(frame_save_path, cv2.cvtColor(center_frame, cv2.COLOR_RGB2BGR))

    # overlay 비디오 생성
    cam_resized = cv2.resize(cam, (W, H), interpolation=cv2.INTER_CUBIC)
    cam_resized = np.clip(cam_resized, 0, 1)

    out_vid_path = os.path.join(out_video_dir, base_name + f"_{cam_method}_overlay.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_vid_path, fourcc, 25, (W, H))

    for f in frames:
        frame_bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)  # uint8 0~255

        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam_resized),
            cv2.COLORMAP_JET
        )  # BGR, uint8

        # 히트맵 비중 크게 (0.7) 잡아서 눈에 잘 띄게
        overlay_bgr = cv2.addWeighted(frame_bgr, 0.3, heatmap, 0.7, 0)
        writer.write(overlay_bgr)

    writer.release()

    print(f"[✔] {os.path.basename(video_path)}")
    print(f"     frame : {frame_save_path}")
    print(f"     cam   : {cam_npy_path}")
    print(f"     video : {out_vid_path}")


# =========================
# 메인: .pth + .json 기반 test 영상 CAM
# =========================

def main():
    print(f"[INFO] CAM_METHOD = {CAM_METHOD}")
    os.makedirs(OUT_FRAME_DIR, exist_ok=True)
    os.makedirs(OUT_CAM_DIR, exist_ok=True)
    os.makedirs(OUT_VIDEO_DIR, exist_ok=True)

    model, processor = build_vivit_model(MODEL_PATH, num_classes=NUM_CLASSES)

    # backbone: ViViT 본체
    if hasattr(model, "base_model") and hasattr(model.base_model, "vivit"):
        backbone = model.base_model.vivit
    elif hasattr(model, "vivit"):
        backbone = model.vivit
    else:
        raise AttributeError("ViViT backbone(vivit)을 찾을 수 없습니다.")

    # test prediction json 로드
    with open(PRED_JSON, "r") as f:
        pred_dict = json.load(f)

    items = list(pred_dict.items())
    print(f"[INFO] CAM 대상 (test set) 비디오 수: {len(items)}")

    for video_path, pred_class in tqdm(items, desc=f"{CAM_METHOD} for test videos"):
        try:
            cam_for_video(
                model=model,
                backbone=backbone,
                processor=processor,
                video_path=video_path,
                out_frame_dir=OUT_FRAME_DIR,
                out_cam_dir=OUT_CAM_DIR,
                out_video_dir=OUT_VIDEO_DIR,
                cam_method=CAM_METHOD,
                target_class=int(pred_class),     # 예측 클래스 기준 CAM
                clip_len=CLIP_LEN,
                frame_sample_rate=FRAME_SAMPLE_RATE,
            )
        except Exception as e:
            print(f"[✘] Failed on {video_path}: {e}")


if __name__ == "__main__":
    main()
