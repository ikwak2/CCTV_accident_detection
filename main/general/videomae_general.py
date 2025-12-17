#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UCF-Crime 일반화 평가 (VideoMAE 버전, robust I/O + 수동 전처리)
- 사전학습 VideoMAE(+LoRA) 이진 분류 체크포인트(.pth)로
  UCF-Crime을 세 가지 매핑 중 하나로 평가
- 결과물:
    preds.csv
    vmae_metrics.json
    vmae_roc.png
    vmae_pr.png
    vmae_confusion_matrix.png
    skip_summary.json
"""
SEED = 42

import os, glob, json, time, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import av
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_auc_score,
    average_precision_score, roc_curve, precision_recall_curve
)

from transformers import VideoMAEForVideoClassification, AutoImageProcessor
from peft import LoraConfig, get_peft_model

# =========================
# 하드코딩 설정
# =========================
GPU_ID = 4
DATA_ROOT = "/Data3/hj24/CCTV/DATA/ucf_data/UCF_Crimes/Videos"   # UCF-Crime의 Videos 폴더
CKPT_PATH = "/Data3/hj24/CCTV/main/vivit/videomae_accident_classifier.pth"  # VideoMAE .pth
MAPPING   = "roadacc_only_positive"   # "all_normal_negative" | "roadacc_only_positive" | "all_anomaly_positive"

CLIP_LEN = 32               # ★ build_model_and_load에서 model.config.num_frames로 덮어씀
FRAME_SAMPLE_RATE = 4
USE_CPU = False
BASE_OUTDIR = "/Data3/hj24/CCTV/main/vivit/general/vmae_out_ucfcrime_generalization"

# GPU 고정
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

# =========================
# 튜닝 옵션 (멀티클립/임계값)
# =========================
N_CLIPS_PER_VIDEO = 5        # 한 영상에서 추출할 클립 수
AGGREGATION = "mean"         # "mean" | "median" | "max" | "vote"
USE_DETERMINISTIC_SAMPLING = True   # True면 영상별 고정 프레임
TARGET_FPR = 0.10            # 임계값 탐색 시 맞추고 싶은 FPR
ENABLE_TTA_FLIP = False

# =========================
# UCF-Crime 클래스/폴더 정의
# =========================
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

# 정상(negative) 폴더명들 — *데이터셋 실제 구조 반영*
NORMAL_DIRS = {
    "z_Normal_Videos_event",
    "Testing_Normal_Videos_Anomaly",
    "Training_Normal_Videos_Anomaly",
}

# anomaly 폴더들
ANOMALY_ALL = {
    "Abuse","Arrest","Arson","Assault","Burglary","Explosion",
    "Fighting","Robbery","Shooting","Shoplifting","Stealing","Vandalism",
    "RoadAccidents"
}
ANOMALY_ROADACC_ONLY = {"RoadAccidents"}

def get_anomaly_set(mode: str):
    if mode == "roadacc_only_positive":
        return ANOMALY_ROADACC_ONLY
    else:
        return ANOMALY_ALL

# 전체 후보
def get_ucf_classes(mode: str):
    return sorted(get_anomaly_set(mode) | NORMAL_DIRS)

# ---------- 라벨 매핑 ----------
def make_mapping(mode: str):
    """
    반환:
      mapping: dict[str -> {0,1}]  # 폴더명 -> 라벨
      allowed: set[str]             # 스캔 허용 폴더 집합
    """
    UCF_CLASSES = get_ucf_classes(mode)

    if mode == "all_normal_negative":
        mapping = {c: 0 for c in NORMAL_DIRS}
        allowed = set(NORMAL_DIRS)
        return mapping, allowed

    elif mode == "roadacc_only_positive":
        mapping = {c: (1 if c in ANOMALY_ROADACC_ONLY else 0) for c in UCF_CLASSES}
        allowed = set(UCF_CLASSES)
        return mapping, allowed

    elif mode == "all_anomaly_positive":
        mapping = {c: (1 if c in ANOMALY_ALL else 0) for c in UCF_CLASSES}
        allowed = set(UCF_CLASSES)
        return mapping, allowed

    else:
        raise ValueError("MAPPING must be one of: all_normal_negative | roadacc_only_positive | all_anomaly_positive")

# ---------- 비디오 경로 스캔 ----------
def iter_video_paths(root_dir: str, allowed=None):
    if allowed is None:
        allowed = set(get_ucf_classes(MAPPING))

    root_dir = os.path.abspath(root_dir)
    for cls in allowed:
        cdir = os.path.join(root_dir, cls)
        if not os.path.isdir(cdir):
            continue
        # 재귀적으로 모든 파일 탐색
        for p in glob.iglob(os.path.join(cdir, "**", "*"), recursive=True):
            ext = os.path.splitext(p)[1].lower()
            if ext in VIDEO_EXTS and os.path.isfile(p):
                yield cls, p

# =========================
# Robust Video I/O
# =========================

class _SkipSample(Exception):
    """이 샘플은 건너뛰어야 함을 명시하는 내부 예외"""
    pass

# PyAV/FFmpeg 로그 억제
try:
    av.logging.set_level(av.logging.ERROR)
except Exception:
    pass

# OpenCV 로그 억제
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
except Exception:
    pass

# OpenCV 스레드 수 0으로 (워커 다중일 때 안전)
try:
    cv2.setNumThreads(0)
except Exception:
    pass

def _open_container_safely(path):
    # 손상 무시 + 지연 최소화
    return av.open(path, options={'err_detect': 'ignore_err', 'max_delay': '0'})

def read_frames_by_indices_cv2(path, indices, clip_len):
    """OpenCV로 인덱스 기반 프레임 읽기 + 해상도 통일"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        raise _SkipSample("OpenCV cannot open video")

    frames = []
    target_size = None  # (W, H)

    for idx in sorted(set(int(i) for i in indices)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if target_size is None:
            target_size = (rgb.shape[1], rgb.shape[0])  # (W,H)
        else:
            if (rgb.shape[1], rgb.shape[0]) != target_size:
                rgb = cv2.resize(rgb, target_size)

        frames.append(rgb)
    cap.release()

    if len(frames) == 0:
        # 마지막 폴백: 앞에서부터 연속으로 clip_len장 읽어본다
        cap = cv2.VideoCapture(path)
        ok = True
        seq = []
        target_size = None
        while ok and len(seq) < clip_len:
            ok, bgr = cap.read()
            if ok:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if target_size is None:
                    target_size = (rgb.shape[1], rgb.shape[0])
                else:
                    if (rgb.shape[1], rgb.shape[0]) != target_size:
                        rgb = cv2.resize(rgb, target_size)
                seq.append(rgb)
        cap.release()
        if len(seq) == 0:
            raise _SkipSample("No readable frames by OpenCV")
        frames = seq

    # 길이 보정
    if len(frames) < clip_len:
        frames += [frames[-1]] * (clip_len - len(frames))

    return np.stack(frames[:clip_len], axis=0)   # (T,H,W,3)

def read_video_pyav(container, indices, path_for_cv2, clip_len):
    """
    1) PyAV로 필요한 인덱스 범위까지만 순회
    2) 실패/부족/해상도 불일치 시 OpenCV 폴백
    """
    idx = np.asarray(indices, dtype=np.int64)
    need = set(int(x) for x in idx)
    last_need = int(idx.max())

    frames = []
    target_size = None  # (W,H)

    try:
        container.seek(0)
        for i, frame in enumerate(container.decode(video=0)):
            if i > last_need:
                break
            if i in need:
                arr = frame.to_ndarray(format="rgb24")  # (H,W,3)
                if target_size is None:
                    target_size = (arr.shape[1], arr.shape[0])
                else:
                    if (arr.shape[1], arr.shape[0]) != target_size:
                        arr = cv2.resize(arr, target_size)
                frames.append(arr)
    except av.AVError:
        frames = []

    if len(frames) >= clip_len:
        return np.stack(frames[:clip_len], axis=0)

    # PyAV로 부족/실패 → OpenCV 폴백
    return read_frames_by_indices_cv2(path_for_cv2, indices, clip_len)

# ---------- 프레임 샘플링 ----------
def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    """
    seg_len이 작아도 최소 clip_len은 확보하도록 보정.
    """
    if seg_len <= 0:
        raise ValueError("ERROR: 비디오의 총 프레임 수가 0입니다. 해당 비디오는 처리 불가")

    converted_len = max(int(clip_len * frame_sample_rate), clip_len)
    converted_len = min(converted_len, max(clip_len, seg_len))

    if seg_len > converted_len:
        end_idx = np.random.randint(converted_len, seg_len)
        start_idx = end_idx - converted_len
    else:
        end_idx = seg_len
        start_idx = 0

    if end_idx <= start_idx:
        end_idx = min(seg_len, start_idx + converted_len)

    indices = np.linspace(start_idx, end_idx - 1, num=clip_len)
    indices = np.clip(indices, start_idx, max(start_idx, end_idx - 1)).astype(np.int64)
    return indices

def sample_frame_indices_deterministic(clip_len, frame_sample_rate, seg_len, key=None):
    if seg_len <= 0:
        raise ValueError("seg_len==0")
    rs = np.random.RandomState(abs(hash(key)) % (2**32)) if key is not None else np.random
    converted_len = int(clip_len * frame_sample_rate)
    if seg_len < converted_len:
        converted_len = seg_len
    end_idx = rs.randint(converted_len, seg_len) if seg_len > converted_len else seg_len
    start_idx = max(0, end_idx - converted_len)
    idx = np.linspace(start_idx, end_idx - 1, num=clip_len)
    return np.clip(idx, start_idx, max(start_idx, end_idx - 1)).astype(np.int64)

def _indices_for_clip(total_frames, clip_len, frame_sample_rate, key=None):
    if USE_DETERMINISTIC_SAMPLING:
        return sample_frame_indices_deterministic(clip_len, frame_sample_rate, total_frames, key=key)
    else:
        return sample_frame_indices(clip_len, frame_sample_rate, total_frames)

# ---------- Video → pixel_values (수동 전처리) ----------
def frames_to_pixel_values(frames, image_processor, device):
    """
    frames: (T, H, W, 3) uint8 or float32 [0-255]
    반환: (1, T, 3, H, W) float32  (VideoMAE 입력 형식)
    """
    frames = np.asarray(frames)
    assert frames.ndim == 4 and frames.shape[-1] == 3, f"frames shape 이상: {frames.shape}"

    # 1) 크기 결정 (image_processor.size 사용)
    size = image_processor.size
    if isinstance(size, dict):
        target_h = size.get("height", 224)
        target_w = size.get("width", 224)
    else:
        # size가 int라면 짧은 변 기준이지만, 여기서는 정사각 가정
        target_h = target_w = int(size)

    resized = []
    for f in frames:
        f_resized = cv2.resize(f, (target_w, target_h))  # (W,H) 순서로 넣음
        resized.append(f_resized)
    video = np.stack(resized, axis=0)  # (T, H, W, 3)

    # 2) [0,1] 스케일링
    video = torch.from_numpy(video).float() / 255.0      # (T,H,W,3)

    # 3) (T,3,H,W) 로 변경
    video = video.permute(0, 3, 1, 2)                    # (T,3,H,W)

    # 4) ImageProcessor의 mean/std 그대로 사용
    mean = torch.tensor(image_processor.image_mean).view(1, 3, 1, 1)
    std  = torch.tensor(image_processor.image_std).view(1, 3, 1, 1)
    mean = mean.to(video.device)
    std  = std.to(video.device)
    video = (video - mean) / std                         # (T,3,H,W)

    # 5) 배치 차원 추가 (B=1, T, 3, H, W)
    pixel_values = video.unsqueeze(0).to(device)         # (1,T,3,H,W)

    return pixel_values

def load_clip_at_indices(video_path, container, indices, image_processor, hflip, device):
    # robust 프레임 로딩
    video = read_video_pyav(container, indices, video_path, CLIP_LEN)  # (T,H,W,3)

    if hflip:
        video = video[:, :, ::-1, :]  # horizontal flip

    # 길이 보정(추가 안전)
    if len(video) < CLIP_LEN:
        pad = CLIP_LEN - len(video)
        video = np.concatenate(
            [video, np.repeat(video[-1][None, ...], pad, axis=0)],
            axis=0
        )

    pixel_values = frames_to_pixel_values(video, image_processor, device)  # (1,T,3,H,W)
    return pixel_values

def multi_clip_score(video_path, model, image_processor, device):
    """
    한 영상에서 N_CLIPS_PER_VIDEO개 샘플링해 확률을 집계.
    PyAV + OpenCV 폴백 + _SkipSample 사용.
    """
    try:
        container = _open_container_safely(video_path)
    except Exception:
        container = None

    total_frames = 0
    if container is not None:
        try:
            total_frames = int(container.streams.video[0].frames or 0)
        except Exception:
            total_frames = 0
    seg_len = max(total_frames, CLIP_LEN)

    probs = []
    try:
        for t in range(N_CLIPS_PER_VIDEO):
            idxs = _indices_for_clip(seg_len, CLIP_LEN, FRAME_SAMPLE_RATE, key=f"{video_path}#{t}")

            if container is None:
                # PyAV 못 열었으면 OpenCV만으로 처리
                video = read_frames_by_indices_cv2(video_path, idxs, CLIP_LEN)
                pixel_values = frames_to_pixel_values(video, image_processor, device) # (1,T,3,H,W)
                logits = model(pixel_values).logits
                p = torch.softmax(logits, dim=1)[0, 1].item()
                probs.append(p)

                if ENABLE_TTA_FLIP:
                    video_f = video[:, :, ::-1, :]
                    pv_f = frames_to_pixel_values(video_f, image_processor, device)
                    p_f = torch.softmax(model(pv_f).logits, dim=1)[0,1].item()
                    probs.append(p_f)

            else:
                clip = load_clip_at_indices(video_path, container, idxs, image_processor,
                                            hflip=False, device=device)
                logits = model(clip).logits
                p = torch.softmax(logits, dim=1)[0, 1].item()
                probs.append(p)

                if ENABLE_TTA_FLIP:
                    clip_f = load_clip_at_indices(video_path, container, idxs, image_processor,
                                                  hflip=True, device=device)
                    p_f = torch.softmax(model(clip_f).logits, dim=1)[0, 1].item()
                    probs.append(p_f)

    except _SkipSample as e:
        if container is not None:
            try:
                container.close()
            except Exception:
                pass
        raise e
    except Exception:
        if container is not None:
            try:
                container.close()
            except Exception:
                pass
        raise
    finally:
        if container is not None:
            try:
                container.close()
            except Exception:
                pass

    if not probs:
        # 이 영상은 유효 클립이 없음 → 상위에서 SKIP 처리
        raise _SkipSample("No valid clip prob for this video")

    arr = np.array(probs, dtype=np.float32)
    if AGGREGATION == "mean":
        return float(arr.mean())
    elif AGGREGATION == "median":
        return float(np.median(arr))
    elif AGGREGATION == "max":
        return float(arr.max())
    elif AGGREGATION == "vote":
        return float((arr >= 0.5).mean())
    else:
        return float(arr.mean())

# ---------- 모델 로드 (VideoMAE) ----------
def build_model_and_load(ckpt: str, device: str):
    model = VideoMAEForVideoClassification.from_pretrained(
        "MCG-NJU/videomae-base-finetuned-kinetics",
        attn_implementation="sdpa",
        torch_dtype=torch.float32
    )
    image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")

    model.config.num_labels = 2
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, 2)

    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["attention.output.dense"],
        modules_to_save=["classifier"],
    )
    model = get_peft_model(model, lora_cfg)

    wrap = torch.cuda.device_count() > 1 and device.startswith("cuda")
    if wrap:
        model = nn.DataParallel(model)

    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    (model.module if wrap else model).load_state_dict(state, strict=False)

    # ★ 모델 설정의 num_frames로 전역 CLIP_LEN 동기화
    global CLIP_LEN
    CLIP_LEN = int(getattr(model.config, "num_frames", 16))

    return model.to(device).eval(), image_processor

def ensure_outdir(base, mapping):
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_tag = f"{mapping}_T{CLIP_LEN}_SR{FRAME_SAMPLE_RATE}_gpu{GPU_ID}_{Path(CKPT_PATH).stem}_{ts}"
    outdir = Path(base) / run_tag
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir/"vmae_config.json", "w") as f:
        json.dump({
            "gpu_id": GPU_ID,
            "data_root": DATA_ROOT,
            "ckpt": CKPT_PATH,
            "mapping": MAPPING,
            "clip_len": CLIP_LEN,
            "frame_sample_rate": FRAME_SAMPLE_RATE,
            "seed": SEED,
            "n_clips": N_CLIPS_PER_VIDEO,
            "agg": AGGREGATION,
            "det_sampling": USE_DETERMINISTIC_SAMPLING,
            "tta_flip": ENABLE_TTA_FLIP,
            "target_fpr": TARGET_FPR
        }, f, indent=2)
    return outdir

def pick_threshold(y_true, y_score, target_fpr=None):
    y_true = np.asarray(y_true); y_score = np.asarray(y_score)
    fpr, tpr, thr = roc_curve(y_true, y_score)
    if target_fpr is not None:
        mask = fpr <= target_fpr + 1e-12
        if mask.any():
            best_idx = np.argmax(tpr[mask])
            return float(thr[mask][best_idx]), {"fpr": float(fpr[mask][best_idx]), "tpr": float(tpr[mask][best_idx])}
    j = tpr - fpr
    best_idx = int(np.argmax(j))
    return float(thr[best_idx]), {
        "fpr": float(fpr[best_idx]),
        "tpr": float(tpr[best_idx]),
        "youdenJ": float(j[best_idx])
    }

@torch.inference_mode()
def evaluate():
    # 디바이스
    device = "cpu" if USE_CPU else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if device == "cuda":
        torch.cuda.manual_seed_all(SEED)

    # 모델/프로세서 (먼저 로드해서 CLIP_LEN 동기화)
    model, image_processor = build_model_and_load(CKPT_PATH, device)

    # 출력 디렉터리 (동기화된 CLIP_LEN 사용)
    out_dir = ensure_outdir(BASE_OUTDIR, MAPPING)
    print(f"[INFO] outdir: {str(out_dir)}")

    # 라벨 매핑
    mapping, allowed = make_mapping(MAPPING)

    # 평가 목록
    vids = list(iter_video_paths(DATA_ROOT, allowed=allowed))
    if not vids:
        raise RuntimeError("No videos found. DATA_ROOT 또는 클래스 폴더 구성을 확인하세요.")
    print(f"[INFO] #videos to eval: {len(vids)}")

    records, y_true, y_score = [], [], []

    # 동일 에러 첫 1회만 출력 + 전체 집계
    skip_counts = {}           # {error_message: count}
    skip_first_shown = set()   # 이미 한 번 보여준 에러 메시지 집합

    for cls, vpath in tqdm(vids, desc=f"Eval[{MAPPING}]"):
        try:
            p_acc = multi_clip_score(vpath, model, image_processor, device)
        except _SkipSample as e:
            msg = str(e)
            skip_counts[msg] = skip_counts.get(msg, 0) + 1
            if msg not in skip_first_shown:
                print(f"[SKIP - showing once] {vpath}: {msg}")
                skip_first_shown.add(msg)
            continue
        except Exception as e:
            msg = f"UNKNOWN: {e}"
            skip_counts[msg] = skip_counts.get(msg, 0) + 1
            if msg not in skip_first_shown:
                print(f"[SKIP - showing once] {vpath}: {msg}")
                skip_first_shown.add(msg)
            continue

        gt = mapping[cls]
        y_true.append(gt); y_score.append(p_acc)
        records.append({"video": vpath, "class": cls, "gt": gt, "prob_accident": p_acc})

    # === 루프 끝난 뒤: SKIP 요약(한 번만) ===
    if skip_counts:
        print("\n[SKIP SUMMARY]")
        for msg, cnt in skip_counts.items():
            print(f"  - {msg}  -> {cnt} files")
        with open(out_dir / "skip_summary.json", "w") as f:
            json.dump(skip_counts, f, indent=2, ensure_ascii=False)

    print(f"#valid videos used: {len(y_true)}")
    if len(y_true) > 0:
        unique, counts = np.unique(y_true, return_counts=True)
        label_dist = dict(zip(unique.tolist(), counts.tolist()))
        print("label distribution:", label_dist)
    else:
        print("label distribution: EMPTY")

    # ----- 메트릭/저장 -----
    y_true = np.array(y_true); y_score = np.array(y_score)

    def compute_metrics_at_threshold(thr):
        if y_true.size == 0:
            return (
                float("nan"), float("nan"), float("nan"),
                (0, 0, 0, 0),
                float("nan"), float("nan"),
                np.zeros((2, 2), dtype=int)
            )

        y_pred = (y_score >= thr).astype(int)
        acc = float(accuracy_score(y_true, y_pred))
        try:
            auroc = float(roc_auc_score(y_true, y_score))
        except Exception:
            auroc = float("nan")
        try:
            auprc = float(average_precision_score(y_true, y_score))
        except Exception:
            auprc = float("nan")

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0
        fpr = float(fp / (fp + tn + 1e-9)) if (fp + tn) > 0 else float("nan")
        tpr = float(tp / (tp + fn + 1e-9)) if (tp + fn) > 0 else float("nan")
        return acc, auroc, auprc, (int(tn), int(fp), int(fn), int(tp)), fpr, tpr, cm

    # (A) 0.5 임계값
    acc, auroc, auprc, (tn, fp, fn, tp), fpr, tpr, cm = compute_metrics_at_threshold(0.5)

    # (B) best threshold
    def safe_pick_threshold(y_true, y_score, target_fpr):
        uniques = np.unique(y_true)
        if y_true.size == 0 or uniques.size < 2:
            return 0.5, {"fpr": float("nan"), "tpr": float("nan")}
        return pick_threshold(y_true, y_score, target_fpr=target_fpr)

    best_thr, best_stats = safe_pick_threshold(y_true, y_score, TARGET_FPR)
    acc_b, auroc_b, auprc_b, (tn_b, fp_b, fn_b, tp_b), fpr_b, tpr_b, cm_b = compute_metrics_at_threshold(best_thr)

    # 예측 저장
    pd.DataFrame.from_records(records).to_csv(out_dir/"preds.csv", index=False)

    metrics = {
        "mapping": MAPPING, "data_root": DATA_ROOT, "ckpt": CKPT_PATH,
        "agg": AGGREGATION, "n_clips": N_CLIPS_PER_VIDEO, "det_sampling": USE_DETERMINISTIC_SAMPLING,
        "tta_flip": ENABLE_TTA_FLIP,
        "thr@0.5": 0.5, "acc@0.5": acc, "auroc": auroc, "auprc": auprc,
        "tn@0.5": tn, "fp@0.5": fp, "fn@0.5": fn, "tp@0.5": tp, "fpr@0.5": fpr, "tpr@0.5": tpr,
        "thr@best": best_thr, "acc@best": acc_b, "auroc@best": auroc_b, "auprc@best": auprc_b,
        "tn@best": tn_b, "fp@best": fp_b, "fn@best": fn_b, "tp@best": tp_b, "fpr@best": fpr_b, "tpr@best": tpr_b,
        "best_stats": best_stats
    }
    with open(out_dir/"vmae_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    # ROC / PR
    can_plot_curves = (y_true.size > 0) and (np.unique(y_true).size == 2)
    if can_plot_curves:
        fpr_arr, tpr_arr, _ = roc_curve(y_true, y_score)
        plt.figure()
        plt.plot(fpr_arr, tpr_arr)
        plt.plot([0, 1], [0, 1], "--")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"ROC ({MAPPING})")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir/"vmae_roc.png", dpi=160)
        plt.close()

        prec, rec, _ = precision_recall_curve(y_true, y_score)
        plt.figure()
        plt.plot(rec, prec)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR ({MAPPING})")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir/"vmae_pr.png", dpi=160)
        plt.close()
    else:
        print("[WARN] ROC/PR skipped: need both positive and negative samples.")

    # 혼동행렬(0.5 기준)
    if y_true.size > 0:
        plt.figure()
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["pred0", "pred1"],
            yticklabels=["true0", "true1"]
        )
        plt.title(f"Confusion Matrix (thr=0.5, {MAPPING})")
        plt.tight_layout()
        plt.savefig(out_dir/"vmae_confusion_matrix.png", dpi=160)
        plt.close()
    else:
        print("[WARN] Confusion matrix skipped: no valid samples.")

if __name__ == "__main__":
    evaluate()
