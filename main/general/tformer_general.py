#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UCF-Crime 일반화 평가 (경로/옵션 하드코딩 버전)
- Timesformer(+LoRA) 이진 분류 체크포인트로
  UCF-Crime을 세 가지 매핑 중 하나로 평가
- 결과물: preds.csv, tf_metrics.json, tf_roc.png, tf_pr.png, tf_confusion_matrix.png
"""
SEED = 42

import os, glob, json, time
from pathlib import Path

import numpy as np
import torch, torch.nn as nn
from tqdm import tqdm
import av, cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_auc_score,
    average_precision_score, roc_curve, precision_recall_curve
)

# from transformers import TimesformerForVideoClassification, AutoImageProcessor
from transformers import TimesformerConfig, TimesformerForVideoClassification, AutoImageProcessor
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file

# =========================
# 하드코딩 설정 (여기만 바꿔서 사용)
# =========================
GPU_ID = 4
DATA_ROOT = "/Data3/hj24/CCTV/DATA/ucf_data/UCF_Crimes/Videos"  # UCF-Crime 루트
CKPT_PATH = "/Data3/hj24/CCTV/main/vivit/timesformer_accident_classifier.safetensors"
# CKPT_PATH = "/Data3/hj24/CCTV/main/vivit/timesformer_accident_classifier.pth"
MAPPING = "roadacc_only_positive"   # "all_normal_negative" | "roadacc_only_positive" | "all_anomaly_positive"
CLIP_LEN = 32
FRAME_SAMPLE_RATE = 4
USE_CPU = False

# 결과 출력 디렉터리
BASE_OUTDIR = "/Data3/hj24/CCTV/main/vivit/general/tf_out_ucfcrime_generalization"

# GPU 고정
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

# =========================
# 튜닝 옵션 (Vivit general과 동일)
# =========================
N_CLIPS_PER_VIDEO = 5        # 한 영상에서 뽑아볼 클립 수
AGGREGATION = "mean"         # "mean" | "median" | "max" | "vote"
USE_DETERMINISTIC_SAMPLING = True   # True면 영상별 고정 프레임
TARGET_FPR = 0.10            # 임계값 탐색 시 맞추고 싶은 FPR
ENABLE_TTA_FLIP = False      # 좌우 flip TTA 사용 여부

# =========================
# UCF-Crime 클래스 정의
# =========================
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

# 정상(negative) 폴더명들 — 데이터셋 사양 반영
NORMAL_DIRS = {
    "z_Normal_Videos_event",
    "Testing_Normal_Videos_Anomaly",
    "Training_Normal_Videos_Anomaly",
}

# 이상(positive 후보)
ANOMALY_DIRS = {
    "RoadAccidents"
}

# 전체 후보
UCF_CLASSES = sorted(ANOMALY_DIRS | NORMAL_DIRS)

# ---------- 라벨 매핑 ----------
def make_mapping(mode: str):
    if mode == "all_normal_negative":
        mapping = {c: 0 for c in NORMAL_DIRS}
        allowed = set(NORMAL_DIRS)
        return mapping, allowed

    elif mode == "roadacc_only_positive":
        pos = {"RoadAccidents"}
        mapping = {c: (1 if c in pos else 0) for c in UCF_CLASSES}
        return mapping, set(UCF_CLASSES)

    elif mode == "all_anomaly_positive":
        pos = set(UCF_CLASSES) - NORMAL_DIRS
        mapping = {c: (1 if c in pos else 0) for c in UCF_CLASSES}
        return mapping, set(UCF_CLASSES)

    else:
        raise ValueError("MAPPING must be one of: all_normal_negative | roadacc_only_positive | all_anomaly_positive")

# ---------- 비디오 경로 스캔 ----------
def iter_video_paths(root_dir, allowed=None):
    if allowed is None:
        allowed = set(UCF_CLASSES)

    root_dir = os.path.abspath(root_dir)
    for cls in UCF_CLASSES:
        if cls not in allowed:
            continue
        cdir = os.path.join(root_dir, cls)
        if not os.path.isdir(cdir):
            continue
        # 재귀적으로 모든 파일 탐색
        for p in glob.iglob(os.path.join(cdir, "**", "*"), recursive=True):
            ext = os.path.splitext(p)[1].lower()
            if ext in VIDEO_EXTS and os.path.isfile(p):
                yield cls, p

# ---------- 프레임 샘플링 ----------
def read_video_pyav(container, indices):
    frames = []
    idx_set = set(indices.tolist())  # membership O(1)
    container.seek(0)
    start_idx, end_idx = indices[0], indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_idx:
            break
        if i >= start_idx and i in idx_set:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    if seg_len <= 0:
        raise ValueError("seg_len==0")
    converted_len = int(clip_len * frame_sample_rate)
    if seg_len < converted_len:
        converted_len = seg_len
    end_idx = np.random.randint(converted_len, seg_len) if seg_len > converted_len else seg_len
    start_idx = max(0, end_idx - converted_len)
    idx = np.linspace(start_idx, end_idx, num=clip_len)
    return np.clip(idx, start_idx, max(start_idx, end_idx - 1)).astype(np.int64)

def sample_frame_indices_deterministic(clip_len, frame_sample_rate, seg_len, key=None):
    if seg_len <= 0:
        raise ValueError("seg_len==0")
    rs = np.random.RandomState(abs(hash(key)) % (2**32)) if key is not None else np.random
    converted_len = int(clip_len * frame_sample_rate)
    if seg_len < converted_len:
        converted_len = seg_len
    end_idx = rs.randint(converted_len, seg_len) if seg_len > converted_len else seg_len
    start_idx = max(0, end_idx - converted_len)
    idx = np.linspace(start_idx, end_idx, num=clip_len)
    return np.clip(idx, start_idx, max(start_idx, end_idx - 1)).astype(np.int64)

def _indices_for_clip(total_frames, clip_len, frame_sample_rate, key=None):
    if USE_DETERMINISTIC_SAMPLING:
        return sample_frame_indices_deterministic(clip_len, frame_sample_rate, total_frames, key=key)
    else:
        return sample_frame_indices(clip_len, frame_sample_rate, total_frames)

def load_clip_at_indices(container, indices, image_processor, hflip=False):
    video = read_video_pyav(container, indices)  # (T,H,W,3) RGB
    if hflip:
        video = video[:, :, ::-1, :]  # horizontal flip
    if len(video) < CLIP_LEN:
        pad = CLIP_LEN - len(video)
        video = np.concatenate([video, np.repeat(video[-1][None, ...], pad, axis=0)], axis=0)
    inputs = image_processor(list(video), return_tensors="pt")
    return inputs["pixel_values"]  # (1,C,T,H,W)

def multi_clip_score(video_path, model, image_processor, device):
    """
    Vivit general 코드와 동일한 multi-clip aggregation 버전.
    """
    container = av.open(video_path)
    total = container.streams.video[0].frames
    if total <= 0:
        container.close()
        raise ValueError(f"Zero-frame: {video_path}")

    probs = []
    for t in range(N_CLIPS_PER_VIDEO):
        idxs = _indices_for_clip(total, CLIP_LEN, FRAME_SAMPLE_RATE, key=f"{video_path}#{t}")
        clip = load_clip_at_indices(container, idxs, image_processor, hflip=False).to(device)
        logits = model(clip).logits
        p = torch.softmax(logits, dim=1)[0, 1].item()
        probs.append(p)

        if ENABLE_TTA_FLIP:
            clip_f = load_clip_at_indices(container, idxs, image_processor, hflip=True).to(device)
            logits_f = model(clip_f).logits
            p_f = torch.softmax(logits_f, dim=1)[0, 1].item()
            probs.append(p_f)

    container.close()

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

# ---------- Timesformer + LoRA ----------
# def build_model_and_load(ckpt: str, device: str):
#     model = TimesformerForVideoClassification.from_pretrained(
#         "facebook/timesformer-base-finetuned-k400"
#     )
#     image_processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")

#     model.config.num_labels = 2
#     in_features = model.classifier.in_features
#     model.classifier = nn.Sequential(
#         nn.Linear(in_features, 256),
#         nn.ReLU(),
#         nn.Dropout(p=0.3),
#         nn.Linear(256, 2)
#     )

#     lora_cfg = LoraConfig(
#         r=8, lora_alpha=16, lora_dropout=0.1,
#         target_modules=[
#             "attention.attention.qkv",
#             "attention.output.dense",
#             "temporal_attention.attention.qkv",
#             "temporal_attention.output.dense",
#         ],
#         modules_to_save=["classifier"],
#     )
#     model = get_peft_model(model, lora_cfg)

#     wrap = torch.cuda.device_count() > 1 and device.startswith("cuda")
#     if wrap:
#         model = nn.DataParallel(model)

#     def load_checkpoint_auto(path):
#         ext = os.path.splitext(path)[1].lower()
#         if ext == ".safetensors":
#             sd = load_file(path, device="cpu")
#             return sd
#         else:
#             raise RuntimeError(
#                 f"현재 환경에서는 .pth 같은 pickle 기반 체크포인트를 torch.load로 읽을 수 없습니다.\n"
#                 f".safetensors로 변환한 뒤 그 경로를 CKPT_PATH에 지정하세요.\n"
#                 f"지금 전달된 경로: {path}"
#             )

#     state = load_checkpoint_auto(ckpt)
#     if "state_dict" in state:
#         state = state["state_dict"]

#     (model.module if wrap else model).load_state_dict(state, strict=False)

#     return model.to(device).eval(), image_processor

def build_model_and_load(ckpt: str, device: str):
    """
    torch.load 취약성 회피 버전:
    - from_pretrained()로 .bin을 안 읽고
    - Config만 받아서 랜덤 초기화 모델 생성
    - safetensors(state_dict)를 그대로 load_state_dict로 얹는다.
    """
    # 1) Config만 가져오기 (여긴 torch.load 안 씀)
    config = TimesformerConfig.from_pretrained("facebook/timesformer-base-finetuned-k400")
    config.num_labels = 2

    # 2) 모델 생성 (랜덤 초기화)
    model = TimesformerForVideoClassification(config)

    # 3) 이미지 프로세서 (이건 JSON/토큰만 읽어서 torch.load 안 씀)
    image_processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")

    # 4) classifier 헤드 재정의 (네가 학습할 때 썼던 구조 그대로)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(256, 2)
    )

    # 5) LoRA 래핑 (구조는 기존과 동일)
    lora_cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.1,
        target_modules=[
            "attention.attention.qkv",
            "attention.output.dense",
            "temporal_attention.attention.qkv",
            "temporal_attention.output.dense",
        ],
        modules_to_save=["classifier"],
    )
    model = get_peft_model(model, lora_cfg)

    wrap = torch.cuda.device_count() > 1 and device.startswith("cuda")
    if wrap:
        model = nn.DataParallel(model)

    # 6) safetensors 로드 (여긴 원래 잘 돌아가던 부분)
    def load_checkpoint_auto(path):
        ext = os.path.splitext(path)[1].lower()
        if ext == ".safetensors":
            sd = load_file(path, device="cpu")
            return sd
        else:
            raise RuntimeError(
                f"현재 환경에서는 .pth 같은 pickle 기반 체크포인트를 torch.load로 읽을 수 없습니다.\n"
                f"먼저 .safetensors로 변환한 뒤 그 경로를 CKPT_PATH에 지정해서 사용하세요.\n"
                f"지금 전달된 경로: {path}"
            )

    state = load_checkpoint_auto(ckpt)
    if "state_dict" in state:  # 혹시 모를 경우 대비
        state = state["state_dict"]

    (model.module if wrap else model).load_state_dict(state, strict=False)

    return model.to(device).eval(), image_processor

def ensure_outdir(base, mapping):
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_tag = f"{mapping}_T{CLIP_LEN}_SR{FRAME_SAMPLE_RATE}_gpu{GPU_ID}_{Path(CKPT_PATH).stem}_{ts}"
    outdir = Path(base) / run_tag
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir/"tf_config.json", "w") as f:
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
    """
    Vivit general 버전과 동일:
    - target_fpr이 있으면 FPR<=target인 구간에서 TPR 최대 임계값
    - 없으면 Youden's J (TPR-FPR) 최대 임계값
    """
    y_true = np.asarray(y_true); y_score = np.asarray(y_score)
    fpr, tpr, thr = roc_curve(y_true, y_score)
    if target_fpr is not None:
        mask = fpr <= target_fpr + 1e-12
        if mask.any():
            best_idx = np.argmax(tpr[mask])
            return float(thr[mask][best_idx]), {
                "fpr": float(fpr[mask][best_idx]),
                "tpr": float(tpr[mask][best_idx])
            }
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

    import random
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if device == "cuda":
        torch.cuda.manual_seed_all(SEED)

    # 출력 디렉터리
    out_dir = ensure_outdir(BASE_OUTDIR, MAPPING)
    print(f"[INFO] outdir: {str(out_dir)}")

    # 모델 로드
    model, image_processor = build_model_and_load(CKPT_PATH, device)

    # 라벨 매핑
    mapping, allowed = make_mapping(MAPPING)

    # 평가 목록
    vids = list(iter_video_paths(DATA_ROOT, allowed=allowed))
    if not vids:
        raise RuntimeError("No videos found. DATA_ROOT 또는 클래스 폴더 구성을 확인하세요.")
    print(f"[INFO] #videos to eval: {len(vids)}")

    records, y_true, y_score = [], [], []

    for cls, vpath in tqdm(vids, desc=f"Eval[{MAPPING}]"):
        if cls not in mapping:
            continue
        try:
            p_acc = multi_clip_score(vpath, model, image_processor, device)
        except Exception as e:
            print(f"[SKIP] {vpath}: {e}")
            continue

        gt = mapping[cls]
        y_true.append(gt); y_score.append(p_acc)
        records.append({"video": vpath, "class": cls, "gt": gt, "prob_accident": p_acc})

    # ----- 메트릭/저장 -----
    import pandas as pd
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

    # (A) thr = 0.5
    acc, auroc, auprc, (tn, fp, fn, tp), fpr, tpr, cm = compute_metrics_at_threshold(0.5)

    # (B) best threshold (TARGET_FPR 우선)
    def safe_pick_threshold(y_true_, y_score_, target_fpr):
        uniques = np.unique(y_true_)
        if y_true_.size == 0 or uniques.size < 2:
            return 0.5, {"fpr": float("nan"), "tpr": float("nan")}
        return pick_threshold(y_true_, y_score_, target_fpr)

    best_thr, best_stats = safe_pick_threshold(y_true, y_score, TARGET_FPR)
    acc_b, auroc_b, auprc_b, (tn_b, fp_b, fn_b, tp_b), fpr_b, tpr_b, cm_b = compute_metrics_at_threshold(best_thr)

    # 예측/메트릭 저장
    pd.DataFrame.from_records(records).to_csv(out_dir/"preds.csv", index=False)
    metrics = {
        "mapping": MAPPING, "data_root": DATA_ROOT, "ckpt": CKPT_PATH,
        "agg": AGGREGATION, "n_clips": N_CLIPS_PER_VIDEO, "det_sampling": USE_DETERMINISTIC_SAMPLING,
        "tta_flip": ENABLE_TTA_FLIP,
        "thr@0.5": 0.5, "acc@0.5": acc, "auroc": auroc, "auprc": auprc,
        "tn@0.5": tn, "fp@0.5": fp, "fn@0.5": fn, "tp@0.5": tp, "fpr@0.5": fpr, "tpr@0.5": tpr,
        "thr@best": best_thr, "acc@best": acc_b, "auroc@best": auroc_b, "auprc@best": auprc_b,
        "tn@best": tn_b, "fp@best": fp_b, "fn@best": fn_b, "tp@best": tp_b,
        "fpr@best": fpr_b, "tpr@best": tpr_b,
        "best_stats": best_stats
    }
    with open(out_dir/"tf_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    # ROC / PR 곡선
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
        plt.savefig(out_dir/"tf_roc.png", dpi=160)
        plt.close()

        prec, rec, _ = precision_recall_curve(y_true, y_score)
        plt.figure()
        plt.plot(rec, prec)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR ({MAPPING})")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out_dir/"tf_pr.png", dpi=160)
        plt.close()
    else:
        print("[WARN] ROC/PR skipped: need both positive and negative samples.")

    # 혼동행렬 (thr=0.5 기준)
    plt.figure()
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["pred0", "pred1"],
        yticklabels=["true0", "true1"]
    )
    plt.title(f"Confusion Matrix (thr=0.5, {MAPPING})")
    plt.tight_layout()
    plt.savefig(out_dir/"tf_confusion_matrix.png", dpi=160)
    plt.close()

if __name__ == "__main__":
    evaluate()
