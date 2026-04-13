# =========================
# ViViT Rollout CAM + YOLO (max 2 boxes, heat>=0.5 only, persistent tracking)
# Paste-and-run version
# =========================

import os
import cv2
import numpy as np
import torch
import av
import json
from ultralytics import YOLO
from transformers import VivitImageProcessor, VivitForVideoClassification, VivitConfig
import traceback

import sys
sys.path.append("/Data3/hj24/CCTV/main")
from vivit import sample_frame_indices, read_video_pyav

# -------- GPU 설정 --------
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"

# -------- YOLO 모델 --------
yolo_model = YOLO('yolov8n.pt')

# -------- 체크포인트 로더 --------
def load_checkpoint_auto(path):
    import os
    ext = os.path.splitext(path)[1].lower()
    if ext == ".safetensors":
        from safetensors.torch import load_file
        sd = load_file(path)
    else:
        raise RuntimeError(
            f"Blocked loading pickle checkpoint: {path}\n"
            f"→ safetensors 로 변환해 사용하세요."
        )
    fixed = {}
    for k, v in sd.items():
        nk = k[7:] if k.startswith("module.") else k
        fixed[nk] = v
    return fixed

# -------- IoU 유틸 --------
def compute_iou(box1, box2):
    """
    box: [x1,y1,x2,y2]
    """
    x1 = max(int(box1[0]), int(box2[0]))
    y1 = max(int(box1[1]), int(box2[1]))
    x2 = min(int(box1[2]), int(box2[2]))
    y2 = min(int(box1[3]), int(box2[3]))

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0

    area1 = max(0, int(box1[2]) - int(box1[0])) * max(0, int(box1[3]) - int(box1[1]))
    area2 = max(0, int(box2[2]) - int(box2[0])) * max(0, int(box2[3]) - int(box2[1]))
    union = area1 + area2 - inter
    return float(inter) / float(union + 1e-8)

# -------- Rollout 계산 --------
def compute_rollout_from_attentions(attentions):
    """
    attentions: tuple/list (num_layers)
      each: (B, num_heads, S, S)
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

def rollout_to_cam(rollout, config):
    """
    rollout: (B, seq, seq)
    CLS -> patch tokens attention to CAM (time x spatial)
    """
    cls_attn = rollout[:, 0, 1:]  # (B, num_patches_total)
    B, num_patches_total = cls_attn.shape

    T_input = _get_num_frames(config)
    tubelet_size = _get_temporal_tubelet_size(config)
    num_time_tokens = T_input // tubelet_size

    if num_patches_total % num_time_tokens != 0:
        raise ValueError(
            f"Token mismatch: total={num_patches_total}, "
            f"num_time_tokens={num_time_tokens} (T={T_input}, tubelet={tubelet_size})"
        )

    patches_per_time = num_patches_total // num_time_tokens

    # factorization for (h,w)
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

# -------- YOLO + CAM 결합 + "유지" + "최대 2개" --------
def apply_yolo_fusion_tracked(
    frame_rgb,
    heatmap,
    track_state,
    conf_thresh=0.25,
    heat_thresh=0.4,
    max_tracks=2,
    iou_thresh=0.2,      # 낮출수록 빠르게 따라가게
    max_miss=3,          
    ema_alpha=0.6,       # 박스 좌표 EMA smoothing (0.5~0.7)
    invert_heatmap=False # True면 색상 반전(1-score)
):
    """
    Tracking-aware YOLO + ViViT rollout fusion
    - heat_thresh: 사고 관련성(roi mean) 임계값
    - iou_thresh: track association 문턱
    - max_miss: YOLO miss 허용 프레임 수
    - ema_alpha: detection으로 갱신될 때 좌표를 부드럽게 보정
    - invert_heatmap: True면 heatmap을 1-heatmap으로 뒤집어 컬러 반전
    """
    H, W = frame_rgb.shape[:2]

    # --- (1) heatmap invert option ---
    heatmap_vis = (1.0 - heatmap) if invert_heatmap else heatmap
    heatmap_uint8 = np.uint8(np.clip(heatmap_vis, 0, 1) * 255)

    # Heatmap overlay (JET: 0=blue, 1=red)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    fused = cv2.addWeighted(heatmap_rgb, 0.4, frame_rgb, 0.6, 0)

    prev_tracks = track_state.get("tracked", [])
    new_tracks = []

    # 1) YOLO inference
    results = yolo_model(frame_rgb, classes=[2, 3, 5, 7], verbose=False, conf=conf_thresh)

    # 2) candidates: ROI mean score >= heat_thresh (여기서 score는 원래 heatmap 기준)
    candidates = []
    if results and len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        for box in boxes:
            x1, y1, x2, y2 = box
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            roi = heatmap[y1:y2, x1:x2]   # threshold 판단은 원래 heatmap 기준 유지
            if roi.size == 0:
                continue
            score = float(np.mean(roi))
            if score >= heat_thresh:
                candidates.append({"box": [x1, y1, x2, y2], "score": score})

    candidates.sort(key=lambda x: x["score"], reverse=True)
    used = [False] * len(candidates)

    # 3) IoU association + EMA smoothing
    for tr in prev_tracks:
        best_iou, best_idx = 0.0, -1
        for i, cand in enumerate(candidates):
            if used[i]:
                continue
            iou = compute_iou(tr["box"], cand["box"])
            if iou > best_iou:
                best_iou, best_idx = iou, i
        if best_idx >= 0 and best_iou >= iou_thresh:
            used[best_idx] = True
            cand = candidates[best_idx]

            # 여기서 EMA smoothing 적용 (완화 작용)
            old_box = tr["box"]
            det_box = cand["box"]

            smoothed_box = [
                ema_alpha * det_box[k] + (1.0 - ema_alpha) * old_box[k]
                for k in range(4)
            ]

            cand["box"] = smoothed_box
            cand["miss"] = 0

            new_tracks.append(cand)
        
        else:
            # YOLO miss → persistence (이 부분이 "이전 box 유지" 원인)
            tr["miss"] = tr.get("miss", 0) + 1
            if tr["miss"] <= max_miss and tr.get("score", 0.0) >= heat_thresh:
                new_tracks.append(tr)

    # 4) add new objects
    for i, cand in enumerate(candidates):
        if used[i]:
            continue
        if len(new_tracks) >= max_tracks:
            break
        cand["miss"] = 0
        new_tracks.append(cand)

    # 5) keep top-k
    new_tracks.sort(key=lambda x: x["score"], reverse=True)
    new_tracks = new_tracks[:max_tracks]
    track_state["tracked"] = new_tracks

    # 6) draw boxes
    for i, tr in enumerate(new_tracks):
        x1, y1, x2, y2 = [int(round(v)) for v in tr["box"]]
        is_best = (i == 0)

        # 색상은 박스 강조용 (원하면 여기에도 invert 논리 넣을 수 있음)
        color = (255, 0, 0) if is_best else (180, 180, 180)  # RGB
        thickness = 3 if is_best else 1

        cv2.rectangle(fused, (x1, y1), (x2, y2), color, thickness)
        if is_best:
            cv2.putText(
                fused,
                f"Accident prob: {tr['score']:.2f}",
                (x1, max(15, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    return fused, track_state

# -------- CAM 비디오 저장 --------
def save_cam_video(frames, cams, save_path, fps=8,
                   conf_thresh=0.25, heat_thresh=0.5,
                   max_tracks=2, iou_thresh=0.2, max_miss=3, ema_alpha=0.6, invert_heatmap=False):
    """
    frames: list of uint8 RGB frames
    cams: (1, T, h, w) float in [0,1]
    """
    H0, W0 = frames[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(save_path, fourcc, fps, (W0, H0))
    if not writer.isOpened():
        raise RuntimeError(f"[ERROR] cv2.VideoWriter open failed for {save_path}")

    T = min(cams.shape[1], len(frames))

    # persistent tracking state across frames
    track_state = {"tracked": []}

    for t in range(T):
        frame = frames[t]
        cam = cams[0, t]

        # frame sanitize
        if frame.ndim == 2:
            frame = np.repeat(frame[..., None], 3, axis=2)
        if frame.shape[2] == 4:
            frame = frame[..., :3]
        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        if frame.shape[:2] != (H0, W0):
            frame = cv2.resize(frame, (W0, H0), interpolation=cv2.INTER_LINEAR)

        # cam resize
        cam_resized = cv2.resize(cam, (W0, H0), interpolation=cv2.INTER_CUBIC)
        cam_resized = np.clip(cam_resized, 0, 1)

        fused_rgb, track_state = apply_yolo_fusion_tracked(
            frame_rgb=frame,
            heatmap=cam_resized,
            track_state=track_state,
            conf_thresh=conf_thresh,
            heat_thresh=heat_thresh,
            max_tracks=max_tracks,
            iou_thresh=iou_thresh,
            max_miss=max_miss,
            ema_alpha=ema_alpha,
            invert_heatmap=invert_heatmap
        )

        # write BGR for OpenCV
        writer.write(cv2.cvtColor(fused_rgb, cv2.COLOR_RGB2BGR))

    writer.release()

# -------- 비디오 처리 --------
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
        attentions = outputs.attentions  # tuple of layers

    rollout = compute_rollout_from_attentions(list(attentions))

    base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    base_model.config.num_frames = 32

    cams = rollout_to_cam(rollout, base_model.config)  # (1, T, h, w)

    video_name = f"{label_str}_{os.path.basename(video_path).replace('.mp4','')}_rollout_yolo_trk2_thr05.mp4"
    video_save_path = os.path.join(output_dir, video_name)

    save_cam_video(
        original_frames, cams, video_save_path,
        fps=8,
        conf_thresh=0.25,
        heat_thresh=0.5,   # 50% 이상만 박스
        max_tracks=2,      # 동시 2개
        iou_thresh=0.2,
        max_miss=3,
        ema_alpha=0.6,
        invert_heatmap=True # 색상 반전
    )

    print(f"[DONE] Saved: {video_save_path}")

# -------- main --------
if __name__ == '__main__':
    predictions_file = "/Data3/hj24/CCTV/main/gradcam/vivit_pred/vivit_predictions_origin.json"
    output_dir = "/Data3/hj24/CCTV/main/gradcam/vivit_pred/vvt_rollout_yolo_output_trk2_thr05"
    os.makedirs(output_dir, exist_ok=True)

    # model load
    model = VivitForVideoClassification.from_pretrained(
        "google/vivit-b-16x2-kinetics400",
        use_safetensors=True,
        attn_implementation="eager",
        dtype=torch.float32
    )
    model.classifier = torch.nn.Linear(model.classifier.in_features, 2)

    ckpt_sft = "/Data3/hj24/CCTV/main/vivit/vivit_accident_classifier.safetensors"
    ckpt_pth = "/Data3/hj24/CCTV/main/vivit/vivit_accident_classifier.pth"

    try:
        state = load_checkpoint_auto(ckpt_sft)
        print(f"[INFO] loaded safetensors: {ckpt_sft}")
    except Exception as e:
        print(f"[WARN] safetensors not usable: {e}")
        print(f"[INFO] trying fallback .pth: {ckpt_pth}")
        raw = torch.load(ckpt_pth, map_location="cpu")
        if isinstance(raw, dict) and "state_dict" in raw:
            raw = raw["state_dict"]
        state = {(k[7:] if k.startswith("module.") else k): v for k, v in raw.items()}

    (model.module if hasattr(model, "module") else model).load_state_dict(state, strict=False)

    model = torch.nn.DataParallel(model).cuda().eval()
    processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")

    # load predictions
    with open(predictions_file, "r") as f:
        predictions = json.load(f)

    # run (sample a few)
    target_videos = [v for v, _ in predictions.items()]

    for video_path in target_videos:
        label = predictions.get(video_path, 0)
        label_str = "accident" if label == 1 else "non-accident"
        print(f"[DEBUG] Starting: {video_path} ({label_str})")
        try:
            process_video(video_path, model, processor, output_dir, label_str)
        except Exception as e:
            print(f"[ERROR] Failed on {video_path}: {type(e).__name__} - {e}")
            traceback.print_exc()
