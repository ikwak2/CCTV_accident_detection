# torch 환경 세팅 버전(기록용)
# pip install torch==2.2.2 torchvision==0.17.2+cu121 --index-url https://download.pytorch.org/whl/cu121

import cv2
import os
import torch
import av
import random
import time
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from transformers import AutoImageProcessor, TimesformerModel, TimesformerForVideoClassification, TimesformerConfig
from torch.utils.data import Dataset, DataLoader, random_split
from torchmetrics.classification import Accuracy
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from tqdm import tqdm
import json
from glob import iglob
from PIL import Image
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix
matplotlib.use('Agg')

# 시드 고정
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# CUDA 연산에서 deterministic 설정
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0,1,2,3,4,5"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400", attn_implementation="sdpa", torch_dtype=torch.float32)
model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400",
                                                          use_safetensors=True, attn_implementation="eager", dtype=torch.float32)
image_processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")

configuration = TimesformerConfig()
configuration = model.config

#####################################################################################################################
########################################### Def function ############################################################

def read_video_pyav(container, indices, max_try=2):
    """
    우선 PyAV로 선택 프레임을 모으고, 실패 시 OpenCV로 폴백.
    반환: (T,H,W,3) RGB np.uint8
    """
    idx_set = set(int(i) for i in indices.tolist() if i >= 0)
    frames = []

    # --- 1차: PyAV ---
    try:
        container.seek(0)
        for i, frame in enumerate(container.decode(video=0)):
            if i > max(idx_set):
                break
            if i in idx_set:
                frames.append(frame.to_ndarray(format="rgb24"))
    except Exception:
        frames = []

    # --- 성공했으면 반환 ---
    if len(frames) > 0:
        return np.stack(frames, axis=0)

    # --- 2차: OpenCV 폴백 (무조건 RGB) ---
    video_path = getattr(container, "name", None)
    if not video_path:
        # 일부 av.open 핸들러는 name이 없어서 다시 열기
        # 호출부에서 path를 알고 있으니, 호출부에서 cv2로 열도록 변경할 수도 있음
        raise ValueError("OpenCV 폴백에 필요한 경로가 없어 디코딩 실패")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"OpenCV로 열 수 없음: {video_path}")

    frames_cv = []
    # OpenCV는 랜덤 접근이 느릴 수 있으니 오름차순으로 순회
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
        # 마지막 시도: 앞에서부터 clip_len만큼 연속 프레임 읽기
        cap = cv2.VideoCapture(video_path)
        ok = True
        while ok and len(frames_cv) < len(idx_set):
            ok, bgr = cap.read()
            if ok:
                frames_cv.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cap.release()

    if len(frames_cv) == 0:
        raise ValueError("need at least one array to stack (모든 디코딩 경로 실패)")

    # 부족하면 마지막 프레임 반복 패딩(모델 입력 맞추기)
    T = len(indices)
    if len(frames_cv) < T:
        last = frames_cv[-1]
        frames_cv += [last] * (T - len(frames_cv))

    return np.stack(frames_cv[:T], axis=0)

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)
    
    if seg_len == 0:
        raise ValueError("ERROR: 비디오의 총 프레임 수가 0입니다. 해당 비디오는 처리 불가")
    
    # seg_len이 너무 작을 경우 방지
    if seg_len < converted_len:
        # print(f"ERROR: `seg_len`({seg_len})이 `converted_len`({converted_len})보다 작아서 샘플링 불가")
        converted_len = seg_len
    
    # 샘플링할 프레임 범위 조정
    end_idx = np.random.randint(converted_len, seg_len) if seg_len > converted_len else seg_len
    start_idx = max(0, end_idx - converted_len)  # 음수가 되지 않도록 조정

    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    
    return indices

def process_video(video_file):
    container = av.open(video_file)
    
    # 비디오의 총 프레임 개수 가져오기
    total_frames = container.streams.video[0].frames
    if total_frames == 0:
        raise ValueError(f"ERROR: 비디오 {video_file}의 총 프레임 수가 0이므로 처리 불가")

    indices = sample_frame_indices(clip_len=32, frame_sample_rate=1, seg_len=total_frames)
    video = read_video_pyav(container=container, indices=indices)
    frames = np.stack(video)  # 또는 np.array(frames)
    inputs = image_processor(frames, return_tensors="pt")
    
    # inputs = image_processor(list(video), return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    
    return last_hidden_states.shape

# LoRA 있/없 비교용
def _bytes_to_mb(x): 
    return round(x / (1024**2), 3)

def _bytes_to_gb(x): 
    return round(x / (1024**3), 4)

def get_model_size_report(model):
    """
    - parameters: 학습 가능한/전체 파라미터 개수, 바이트
    - buffers: running stats 등 버퍼 크기
    - state_dict_bytes: state_dict 기준 총 바이트(대략적인 ckpt 크기 상한)
    """
    # DataParallel/Distributed 대응
    m = model.module if hasattr(model, "module") else model

    # LoRA 적용 여부 힌트(PEFT 모델이면 보통 이 속성/타입이 있음)
    is_peft = hasattr(m, "peft_config") or (m.__class__.__name__.lower().find("peft") >= 0)

    total_params = 0
    trainable_params = 0
    param_bytes_total = 0
    param_bytes_trainable = 0

    for p in m.parameters():
        n = p.numel()
        b = n * p.element_size()
        total_params += n
        param_bytes_total += b
        if p.requires_grad:
            trainable_params += n
            param_bytes_trainable += b

    buffer_bytes_total = 0
    for b in m.buffers():
        buffer_bytes_total += b.numel() * b.element_size()

    # state_dict 텐서 바이트 합(저장 ckpt 크기와 거의 비슷한 감)
    sd = m.state_dict()
    state_dict_bytes = 0
    for k, v in sd.items():
        if torch.is_tensor(v):
            state_dict_bytes += v.numel() * v.element_size()

    return {
        "is_peft_like": bool(is_peft),
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "param_bytes_total_MB": _bytes_to_mb(param_bytes_total),
        "param_bytes_trainable_MB": _bytes_to_mb(param_bytes_trainable),
        "buffer_bytes_total_MB": _bytes_to_mb(buffer_bytes_total),
        "state_dict_tensor_bytes_MB": _bytes_to_mb(state_dict_bytes),
    }

def get_optimizer_state_size_report(optimizer):
    """
    Adam류는 state가 커서 LoRA on/off 차이가 크게 남.
    - optimizer.state 내부 텐서 바이트 합산
    """
    total_bytes = 0
    n_tensors = 0
    for _param, st in optimizer.state.items():
        for _k, v in st.items():
            if torch.is_tensor(v):
                total_bytes += v.numel() * v.element_size()
                n_tensors += 1
    return {
        "optimizer_state_tensors": int(n_tensors),
        "optimizer_state_bytes_MB": _bytes_to_mb(total_bytes),
    }

def get_file_size_report(path):
    if not os.path.exists(path):
        return {"exists": False, "path": path}
    sz = os.path.getsize(path)
    return {"exists": True, "path": path, "bytes": int(sz), "MB": _bytes_to_mb(sz), "GB": _bytes_to_gb(sz)}

###################################################################################################################
########################################### Data input ############################################################

# 데이터셋 클래스 정의
class VideoDataset(Dataset):
    def __init__(self, video_files, labels, image_processor):
        self.video_files = video_files
        self.labels = labels
        self.image_processor = image_processor
        self.bad_files = set()  # 실패 파일 기록

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        # 실패 파일은 즉시 다음 샘플로
        if self.video_files[idx] in self.bad_files:
            return self.__getitem__((idx + 1) % len(self.video_files))

        path = self.video_files[idx]
        try:
            # container = av.open(path)
            container = av.open(path, options={'err_detect': 'ignore_err'})
        except Exception:
            self.bad_files.add(path)
            return self.__getitem__((idx + 1) % len(self.video_files))

        try:
            total_frames = safe_total_frames(path, container)
            if total_frames is None or total_frames <= 0:
                raise ValueError("총 프레임=0/None")

            if total_frames < 5:
                # 너무 짧으면 다음 샘플
                container.close()
                return self.__getitem__((idx + 1) % len(self.video_files))

            indices = sample_frame_indices(clip_len=32, frame_sample_rate=4, seg_len=total_frames)

            # 견고 디코더 사용
            try:
                video = read_video_pyav(container, indices)
            finally:
                container.close()

            # 부족하면 패딩(이중 안전장치)
            if len(video) < 32:
                pad = 32 - len(video)
                video = np.concatenate([video, np.repeat(video[-1][None, ...], pad, axis=0)], axis=0)

            inputs = self.image_processor(list(video), return_tensors="pt")
            return inputs["pixel_values"].squeeze(0), torch.tensor(self.labels[idx], dtype=torch.long)

        except Exception as e:
            # 문제 파일은 기록 후 스킵
            try:
                container.close()
            except:
                pass
            self.bad_files.add(path)
            # (선택) 간헐 에러 대비해 1~2회 재시도하고 싶다면 여기서 재귀 횟수 제한 추가 가능
            return self.__getitem__((idx + 1) % len(self.video_files))

def safe_total_frames(video_path, container=None):
    """PyAV의 frame count가 0/None이면 OpenCV로 폴백."""
    try:
        if container is not None and container.streams.video and container.streams.video[0].frames:
            tf = int(container.streams.video[0].frames)
            if tf > 0:
                return tf
    except Exception:
        pass

    # OpenCV fallback
    cap = cv2.VideoCapture(video_path)
    tf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return tf

# ======= 스위치 / 경로 =======
MODE = "binary"            # "3cls" | "binary"
INCLUDE_AFTER = True       # after-accident은 사고(1)
INCLUDE_UTIC = True

# UTIC 소스들을 (디렉토리, 패턴) 리스트로 관리
UTIC_SOURCES = [
    ("/Data3/hj24/CCTV/DATA/utic/utic_data/record_final", "*.mp4"),  # 예: AA_*, CC_* 등 전부
    ("/Data3/hj24/CCTV/DATA/utic/utic_data/record",       "*.mp4"),  # 예: cctv_* 등 전부
]
UTIC_MAX_SAMPLES = None  # 필요 시 정수로 다운샘플

def _iter_mp4s(root_dir, pattern="*.mp4"):
    if not root_dir or not os.path.exists(root_dir):
        return
    for p in iglob(os.path.join(root_dir, "**", pattern), recursive=True):
        yield p

def build_trainval_filelist(root):
    """
    라벨 정책(바이너리):
      accident/*.mp4        -> 1
      after-accident/*.mp4  -> 1 (INCLUDE_AFTER=True)
      non-accident/*.mp4    -> 0
    + UTIC_SOURCES의 *.mp4   -> 0
    """
    video_files, labels = [], []

    # accident -> 1
    acc_dir = os.path.join(root, "accident")
    acc_list = list(_iter_mp4s(acc_dir, "*.mp4")) or []
    video_files += acc_list
    labels     += [1] * len(acc_list)

    # non-accident -> 0
    na_dir = os.path.join(root, "non-accident")
    na_list = list(_iter_mp4s(na_dir, "*.mp4")) or []
    video_files += na_list
    labels     += [0] * len(na_list)

    # after-accident -> 1
    aa_list = []
    if INCLUDE_AFTER:
        aa_dir = os.path.join(root, "after-accident")
        aa_list = list(_iter_mp4s(aa_dir, "*.mp4")) or []
        video_files += aa_list
        labels     += [1] * len(aa_list)

    # UTIC(두 경로 모두) -> 0
    utic_total = 0
    utic_details = []
    if INCLUDE_UTIC:
        for utic_dir, utic_pattern in UTIC_SOURCES:
            cur = list(_iter_mp4s(utic_dir, utic_pattern)) or []
            if UTIC_MAX_SAMPLES is not None and len(cur) > UTIC_MAX_SAMPLES:
                rng = np.random.default_rng(42)
                cur = list(rng.choice(cur, size=UTIC_MAX_SAMPLES, replace=False))
            utic_total += len(cur)
            utic_details.append((utic_dir, len(cur)))
            video_files += cur
            labels     += [0] * len(cur)

    # 중복 제거
    if len(video_files) != len(set(video_files)):
        uniq_files, uniq_labels, seen = [], [], set()
        for p, y in zip(video_files, labels):
            if p in seen:
                continue
            seen.add(p); uniq_files.append(p); uniq_labels.append(y)
        video_files, labels = uniq_files, uniq_labels

    # 확인용 로그
    print(f"[SRC] accident(1): {len(acc_list)} | after-accident(1): {len(aa_list)} | non-accident(0): {len(na_list)}")
    if INCLUDE_UTIC:
        for d, n in utic_details:
            print(f"[UTIC] 추가: {n}개 (dir={d})")
        print(f"[UTIC] 총 추가: {utic_total}개")
    print(f"[DATA] 총 {len(video_files)}개 | 비사고(0)={labels.count(0)} | 사고(1)={labels.count(1)}")

    # 안전 체크: after-accident 샘플이 있으면 반드시 1이어야 함
    if aa_list:
        assert all(lbl == 1 for lbl in labels[-len(aa_list)-utic_total: -utic_total or None]), "after-accident 라벨 오류"

    return video_files, labels

# class VideoClassifier(nn.Module):
#     def __init__(self, base_model):
#         super(VideoClassifier, self).__init__()
#         self.tformer = base_model
#         # self.classifier = nn.Linear(self.tformer.config.hidden_size, 2)  # classificaiton model 사용 시 생략
        
#     def forward(self, x):
#         outputs = self.tformer(x)
#         # classificaiton model 사용 시 생략
#         # features = outputs.last_hidden_state[:, 0, :] 
#         # return self.classifier(features)
#         return outputs.logits
        
# def initialize_classifier_weights(model):
#     """분류 부분 가중치 초기화 코드"""
#     for m in model.modules():
#         if isinstance(m, nn.Linear):
#             init.xavier_uniform_(m.weight)
#             if m.bias is not None:
#                 init.zeros_(m.bias)

# 초기화 모델 # 모든 linear에 적용되지 않게끔
def init_head_only(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight); 
        if m.bias is not None: init.zeros_(m.bias)
        
# LoRA
# print(model)  # 서브모듈 이름 확인 
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=[
        # spatial attention
        "attention.attention.qkv",
        "attention.output.dense",
        # temporal attention
        "temporal_attention.attention.qkv",
        "temporal_attention.output.dense",
    ],
    modules_to_save=["classifier"],   # classifier는 항상 학습/저장되게
)

# 분류기 출력 차원 수정
NUM_CLASSES = 3 if MODE == "3cls" else 2
model.config.num_labels = NUM_CLASSES

# model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
model.classifier = nn.Sequential(
    nn.Linear(model.classifier.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),  # 선택적으로 추가
    nn.Linear(256, NUM_CLASSES)
)

# model = get_peft_model(model, lora_config)
# model.print_trainable_parameters() # 확인용
# model = VideoClassifier(base_model)
# model.apply(initialize_classifier_weights)

# --- LoRA ON ---
for p in model.parameters():
    p.requires_grad = False

for n, p in model.named_parameters():
    if "classifier" in n or "lora_" in n.lower():
        p.requires_grad = True
        
# --- LoRA OFF ---
# for p in model.parameters():
#     p.requires_grad = True

        
head = model.module.classifier if hasattr(model, "module") else model.classifier
head.apply(init_head_only)

# LoRA 적용 여부 / 모델 용량 1차 기록 
model_size_before_dp = get_model_size_report(model)
print("[MODEL SIZE BEFORE DP]", model_size_before_dp)

# multi gpu 할당 적용
model = nn.DataParallel(model).to(device)
# model.module.print_trainable_parameters() # 확인용

# 데이터로더 생성
video_folder_path = "/Data3/hj24/CCTV/DATA/cctv_data/cctv_seg"
video_files, labels = build_trainval_filelist(video_folder_path)
dataset = VideoDataset(video_files, labels, image_processor)
generator = torch.Generator().manual_seed(42)

# 분리
total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.2 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator)
print(f"Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")

# DataLoader 설정 (매 에폭마다 데이터 순서를 랜덤하게)
train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
val_loader   = DataLoader(val_dataset,   batch_size=20, shuffle=False,
                          num_workers=4, pin_memory=True, persistent_workers=True)
test_loader  = DataLoader(test_dataset,  batch_size=20, shuffle=False,
                          num_workers=4, pin_memory=True, persistent_workers=True)
###################################################################################################################
########################################### Classification ########################################################

# 학습 설정
criterion = nn.CrossEntropyLoss()
# trainable_params = list(model.module.tformer.parameters()) + list(model.module.classifier.parameters())
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(trainable_params, lr=5e-5)
# ====== (추가) optimizer state 용량(초기에는 보통 0에 가까움) ======
opt_size_init = get_optimizer_state_size_report(optimizer)
print("[OPT SIZE INIT]", opt_size_init)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose = True)
accuracy_metric = Accuracy(task="multiclass", num_classes=NUM_CLASSES).to(device)

train_acc, val_acc, train_loss, val_loss, epoch_times, epoch_size_log = [], [], [], [], [], []
patience = 5
best_val_loss = float('inf')
counter = 0

if torch.cuda.is_available():
    # GPU peak memory 통계를 초기화
    torch.cuda.reset_peak_memory_stats()

# 모델 학습 루프
for epoch in tqdm(range(50), desc="Total Epoch Progress"):
    epoch_start_time = time.time()  # 에폭 시작 시각
    
    model.train()
    total_train_loss = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        logits = outputs.logits
        
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        accuracy_metric.update(preds, labels)
        
    model_size = get_model_size_report(model)
    opt_size   = get_optimizer_state_size_report(optimizer)

    
    epoch_size_log.append({
        "epoch": int(epoch + 1),
        "model": model_size,
        "optimizer": opt_size,
        "total_train_footprint_MB": round(
            model_size["param_bytes_trainable_MB"] + opt_size["optimizer_state_bytes_MB"], 3
        ),
    })
    
    epoch_accuracy = accuracy_metric.compute().item()
    accuracy_metric.reset()
    
    train_acc.append(epoch_accuracy)
    train_loss.append(total_train_loss/len(train_loader))
    # train_loss.append(loss.item())

    # 평가 모드
    model.eval()  
    total_val_loss = 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            logits = outputs.logits
            
            loss = criterion(logits, labels)
            total_val_loss += loss.item()
            
            probs = torch.softmax(logits, dim=1) 
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy()) 
            
            accuracy_metric.update(preds, labels)

    val_accuracy = accuracy_metric.compute().item()
    accuracy_metric.reset()
    
    val_acc.append(val_accuracy)
    val_loss.append(total_val_loss / len(val_loader) )
    # val_loss.append(loss.item())
    
    to_save = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    
    if val_loss[-1] < best_val_loss:
        best_val_loss = val_loss[-1]
        counter = 0
        torch.save(to_save, "tf_best_model.pth")
    else:
        counter += 1

    if counter >= patience:
        print("Early stopping triggered.")
        break
    
    tqdm.write(f"[Epoch {epoch+1}/50] Train Loss: {train_loss[-1]:.4f} | Train Acc: {epoch_accuracy:.4f} || "
           f"Val Loss: {val_loss[-1]:.4f} | Val Acc: {val_accuracy:.4f}")
    
    scheduler.step(val_loss[-1])
    
    # === 에폭당 시간(분) 기록 ===
    epoch_minutes = (time.time() - epoch_start_time) / 60.0
    epoch_times.append(epoch_minutes)

torch.save(to_save, "timesformer_accident_classifier.pth")

# === 학습 시간 및 GPU 메모리 사용량 기록 ===
training_total_minutes = (time.time() - training_start_time) / 60.0

best_ckpt_report = get_file_size_report("tf_best_model.pth")
final_ckpt_report = get_file_size_report("timesformer_accident_classifier.pth")
print("[CKPT SIZE BEST]", best_ckpt_report)
print("[CKPT SIZE FINAL]", final_ckpt_report)

gpu_memory_gb = {}
if torch.cuda.is_available():
    # DataParallel에서 사용 중인 모든 GPU에 대해 peak memory 확인
    num_devices = torch.cuda.device_count()
    for i in range(num_devices):
        peak_bytes = torch.cuda.max_memory_allocated(i)  # bytes
        gpu_memory_gb[f"cuda:{i}"] = round(peak_bytes / (1024 ** 3), 4)  # GB 단위로 반올림

# Early stopping 여부 (학습 루프가 patience로 끝났는지)
early_stopped = counter >= patience

final_model_size = get_model_size_report(model)
final_opt_size   = get_optimizer_state_size_report(optimizer)

train_profile = {
    "total_training_minutes": training_total_minutes,
    "epoch_times_minutes": epoch_times,
    "num_epochs_trained": len(epoch_times),
    "early_stopped": early_stopped,
    "max_gpu_memory_GB": gpu_memory_gb,

    "model_size_before_dp": model_size_before_dp,
    "optimizer_state_init": opt_size_init,

    # 매 epoch 기록
    "epoch_size_log": epoch_size_log,

    # 최종 총산(원하는 “총 메모리 footprint”)
    "final_size": {
        "model": final_model_size,
        "optimizer": final_opt_size,
        "total_train_footprint_MB": round(
            final_model_size["param_bytes_trainable_MB"] + final_opt_size["optimizer_state_bytes_MB"], 3
        ),
    },

    # ckpt 용량
    "ckpt_best": best_ckpt_report,
    "ckpt_final": final_ckpt_report,
    "ckpt_total_MB": round(
        (best_ckpt_report["MB"] if best_ckpt_report.get("exists") else 0.0) +
        (final_ckpt_report["MB"] if final_ckpt_report.get("exists") else 0.0),
        3
    ),
}

train_profile_path = "/Data3/hj24/CCTV/main/vivit/output/tformer_train_profile.json"
os.makedirs(os.path.dirname(train_profile_path), exist_ok=True)
with open(train_profile_path, "w") as f:
    json.dump(train_profile, f, ensure_ascii=False, indent=2)

print(f"[INFO] 학습 프로파일 JSON 저장 완료: {train_profile_path}")

###################################################################################################################
########################################### 학습 결과 그래프 #########################################################

stats_path = "/Data3/hj24/CCTV/main/vivit/output/tf_train_val_stats.npz"
np.savez(stats_path, train_acc=train_acc, val_acc=val_acc, train_loss=train_loss, val_loss=val_loss,
         all_probs=np.array(all_probs), all_labels=np.array(all_labels))
val_stats = np.load(stats_path, allow_pickle=True)

val_probs = val_stats["all_probs"]
val_labels = val_stats["all_labels"]

# 🔹 Accuracy 그래프
plt.figure(figsize=(10, 5))
plot_dir = "/Data3/hj24/CCTV/main/vivit/output/plots"
os.makedirs(plot_dir, exist_ok=True)

plt.subplot(1, 2, 1)  # 1행 2열 중 첫 번째 그래프
plt.plot(range(1, len(train_acc)+1), train_acc, label="Train Accuracy", marker="o")
plt.plot(range(1, len(val_acc)+1), val_acc, label="Validation Accuracy", marker="s")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Train vs Validation Accuracy")
plt.legend()
plt.grid(True)

# 🔹 Loss 그래프
plt.subplot(1, 2, 2)  # 1행 2열 중 두 번째 그래프
plt.plot(range(1, len(train_loss)+1), train_loss, label="Train Loss", marker="o", color="red")
plt.plot(range(1, len(val_loss)+1), val_loss, label="Validation Loss", marker="s", color="orange")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Validation Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "tformer_train_val_metrics.png"))

# === Validation / Test 공통: 확률 히스토그램 ===
def plot_prob_hist_by_true_class(y_true, y_probs, title_prefix, out_png, output_dir="..", labels_3cls=None):
    """
    y_true: (N,) 정답 라벨
    y_probs: (N, NUM_CLASSES) softmax 확률
    """
    import numpy as np
    import os
    import matplotlib.pyplot as plt

    # 기본 맵(3클래스일 때만 사용)
    if labels_3cls is None:
        labels_3cls = {"non-accident": 0, "after-accident": 1, "accident": 2}

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    plt.figure(figsize=(10,5))
    if NUM_CLASSES == 3:
        mask_non   = (y_true == labels_3cls["non-accident"])
        mask_after = (y_true == labels_3cls["after-accident"])
        mask_acc   = (y_true == labels_3cls["accident"])

        plt.hist(y_probs[mask_non,   labels_3cls["non-accident"]], bins=20, alpha=0.6, label="Non-Accident")
        plt.hist(y_probs[mask_after, labels_3cls["after-accident"]], bins=20, alpha=0.6, label="After-Accident")
        plt.hist(y_probs[mask_acc,   labels_3cls["accident"]],      bins=20, alpha=0.6, label="Accident")
        plt.xlabel("Predicted Probability of TRUE Class")
        plt.title(f"{title_prefix}: Probability Distribution by TRUE Class (3-class)")
    else:
        mask_acc = (y_true == 1)
        mask_non = (y_true == 0)
        plt.hist(y_probs[mask_acc, 1], bins=20, alpha=0.6, label="Accident")
        plt.hist(y_probs[mask_non, 1], bins=20, alpha=0.6, label="Non-Accident")
        plt.xlabel("Predicted Probability for Class 1 (Accident)")
        plt.title(f"{title_prefix}: Probability Distribution by Class (Binary)")

    plt.ylabel("Sample Count")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, os.path.basename(out_png))
    plt.savefig(save_path)
    plt.close()
    print(f"그래프 저장 완료: {save_path}")
    return save_path


plot_prob_hist_by_true_class(all_labels, all_probs, "Validation", out_png="tformer_val_prob_hist_split.png", output_dir=plot_dir)

###################################################################################################################
################################################# Test ############################################################

# LABELS_3CLS = {"non-accident":0, "after-accident":1, "accident":2}
# NUM_CLASSES = 3 if MODE == "3cls" else 2
CLASS_NAMES = {0: "Non-Accident", 1: ("After-Accident" if NUM_CLASSES == 3 else "Accident?"), 2: "Accident"} if NUM_CLASSES == 3 else {0: "Non-Accident", 1: "Accident"}

def _get_video_path_from_dataset(ds, global_idx):
    """
    Subset이든 원본 Dataset이든 안전하게 비디오 파일 경로를 가져온다.
    - ds: test_loader.dataset
    """
    # Subset?
    if hasattr(ds, "dataset") and hasattr(ds, "indices"):
        base = ds.dataset
        idx  = ds.indices[global_idx]
        if hasattr(base, "video_files"):  # 사용자 정의 Dataset에 video_files가 있을 때
            return base.video_files[idx]
        # 필요 시 base 내부에서 경로 재구성 로직 추가
        raise AttributeError("기저 Dataset에서 video_files를 찾을 수 없습니다.")
    else:
        # Subset이 아닌 경우
        if hasattr(ds, "video_files"):
            return ds.video_files[global_idx]
        raise AttributeError("Dataset에 video_files 속성이 없습니다.")

def evaluate_test(test_loader, device, model, output_dir=".."):
    model.eval()
    accuracy_metric.reset()
    total_test_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    predictions = {}

    os.makedirs(output_dir, exist_ok=True)
    pred_json_path = "/Data3/hj24/CCTV/main/gradcam/tformer_pred/tformer_predictions_origin.json"
    os.makedirs(os.path.dirname(pred_json_path), exist_ok=True)

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            logits = outputs.logits

            # CrossEntropyLoss는 logits에 바로 넣는 게 정석
            loss = criterion(logits, labels)
            total_test_loss += loss.item()

            probs = torch.softmax(logits, dim=1)        # (B, NUM_CLASSES)
            preds = torch.argmax(probs, dim=1)          # (B,)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            accuracy_metric.update(preds, labels)

            # === 비디오 저장(배치 단위) ===
            # 글로벌 인덱스(Subset 대응) 계산
            start_gi = batch_idx * test_loader.batch_size
            for i, (pred, label) in enumerate(zip(preds, labels)):
                gi = start_gi + i  # global index in Subset wrapper
                try:
                    video_file = _get_video_path_from_dataset(test_loader.dataset, gi)
                except Exception as e:
                    print(f"[WARN] 영상 경로 추출 실패 (gi={gi}): {e}")
                    continue

                predictions[video_file] = int(pred.item())

                # 원본 프레임 일부 재디코딩 (Grad-CAM용 썸네일/미리보기)
                try:
                    container = av.open(video_file)
                    total_frames = container.streams.video[0].frames
                    indices = sample_frame_indices(clip_len=32, frame_sample_rate=4, seg_len=total_frames)
                    frames = read_video_pyav(container, indices)  # (T, H, W, 3) RGB
                    container.close()
                except Exception as e:
                    print(f"[WARN] 디코딩 실패: {video_file} -> {e}")
                    continue

                # 비디오 저장
                out_path = os.path.join(output_dir, f"pred_{os.path.basename(video_file)}")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(out_path, fourcc, 30, (224, 224))
                if not out.isOpened():
                    print(f"[ERROR] VideoWriter open 실패: {out_path}")
                    continue

                # 텍스트 준비
                pred_cls = int(pred.item())
                label_cls = int(label.item())
                text = f"Pred: {CLASS_NAMES.get(pred_cls, str(pred_cls))}"
                color = (0, 255, 0) if pred_cls == label_cls else (0, 0, 255)

                for frame in frames:
                    frame = cv2.resize(frame, (224, 224))
                    # RGB→BGR
                    if frame.shape[2] == 3:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    out.write(frame)
                out.release()

    acc = accuracy_metric.compute().item()
    avg_loss = total_test_loss / len(test_loader)

    with open(pred_json_path, "w") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    return acc, avg_loss, all_preds, all_labels, all_probs

acc, test_loss, all_preds, all_labels, all_probs = evaluate_test(
    test_loader=test_loader,
    device=device,
    model=model,
    output_dir="/Data3/hj24/CCTV/main/gradcam/tformer_pred/tformer_output"
)
plot_prob_hist_by_true_class(all_labels, np.array(all_probs), "Test", out_png="tformer_test_prob_hist_split.png", output_dir=plot_dir)

# ===== F1 & ROC-AUC plots =====
def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def plot_roc_auc(y_true, y_probs, class_names, out_png="roc_auc.png", output_dir=".."):
    """
    ROC 곡선과 AUC 저장.
    - y_true: (N,) 정답 라벨 (int)
    - y_probs: (N, C) softmax 확률
    - class_names: {idx:name} dict 또는 name list
    """
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    C = y_probs.shape[1]
    if isinstance(class_names, dict):
        class_names = [class_names[i] if i in class_names else str(i) for i in range(C)]
    else:
        assert len(class_names) == C, "class_names 길이가 NUM_CLASSES와 일치해야 합니다."

    _ensure_dir(output_dir)
    save_path = os.path.join(output_dir, os.path.basename(out_png))

    plt.figure(figsize=(7,6))
    roc_info = {}

    if C == 2:
        # binary: 양성 클래스(1) 점수 사용
        fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
        roc_auc = auc(fpr, tpr)
        roc_info["auc"] = float(roc_auc)

        plt.plot(fpr, tpr, label=f"ROC (AUC={roc_auc:.3f})")
        plt.plot([0,1],[0,1],"--", label="Chance")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (Binary)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path); plt.close()
        print(f"[ROC] saved: {save_path}")
        return roc_info

    # multiclass: OvR
    y_true_bin = label_binarize(y_true, classes=np.arange(C))
    fpr, tpr, roc_auc = {}, {}, {}

    present_classes = []
    for i in range(C):
        # 해당 클래스가 전혀 없으면 ROC 정의 불가 -> skip
        pos = y_true_bin[:, i].sum()
        neg = len(y_true) - pos
        if pos == 0 or neg == 0:
            continue
        present_classes.append(i)
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f"{class_names[i]} (AUC={roc_auc[i]:.3f})")

    # micro-average
    if len(present_classes) > 0:
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin[:, present_classes].ravel(),
                                                  y_probs[:, present_classes].ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        plt.plot(fpr["micro"], tpr["micro"], linestyle="--", linewidth=2,
                 label=f"micro-average (AUC={roc_auc['micro']:.3f})")

        # macro-average
        all_fpr = np.unique(np.concatenate([fpr[i] for i in present_classes]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in present_classes:
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= len(present_classes)
        roc_auc["macro"] = auc(all_fpr, mean_tpr)
        plt.plot(all_fpr, mean_tpr, linestyle=":", linewidth=2,
                 label=f"macro-average (AUC={roc_auc['macro']:.3f})")

    plt.plot([0,1],[0,1],"k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Multiclass, OvR)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path); plt.close()
    print(f"[ROC] saved: {save_path}")
    return {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in roc_auc.items()}

def plot_f1_binary_vs_threshold(y_true, y_probs, out_png="tformer_f1_vs_threshold.png", output_dir=".."):
    """
    바이너리 전용: 임계값별 F1 곡선 + 최적 threshold 표시
    - y_probs: (N,2) 또는 (N,) 점수. (N,2)면 양성클래스=열1 사용
    """
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    if y_probs.ndim == 2:
        y_score = y_probs[:, 1]
    else:
        y_score = y_probs

    thresholds = np.linspace(0.0, 1.0, 101)
    f1s = []
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        f1s.append(f1_score(y_true, y_pred, zero_division=0))
    f1s = np.asarray(f1s)
    best_idx = int(np.argmax(f1s))
    best_t, best_f1 = thresholds[best_idx], float(f1s[best_idx])

    _ensure_dir(output_dir)
    save_path = os.path.join(output_dir, os.path.basename(out_png))

    plt.figure(figsize=(7,5))
    plt.plot(thresholds, f1s, label="F1")
    plt.axvline(best_t, linestyle="--", label=f"best t={best_t:.2f}, F1={best_f1:.3f}")
    plt.xlabel("Threshold")
    plt.ylabel("F1 score")
    plt.title("F1 vs Threshold (Binary)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path); plt.close()
    print(f"[F1-curve] saved: {save_path}")
    return {"best_threshold": float(best_t), "best_f1": best_f1}

def plot_f1_multiclass_bars(y_true, y_probs, class_names, out_png="tformer_f1_per_class.png", output_dir=".."):
    """
    멀티클래스 전용: 클래스별 F1 막대 + micro/macro/weighted F1 출력
    """
    from sklearn.metrics import f1_score
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    C = y_probs.shape[1]
    y_pred = np.argmax(y_probs, axis=1)

    if isinstance(class_names, dict):
        class_names = [class_names[i] if i in class_names else str(i) for i in range(C)]
    else:
        assert len(class_names) == C

    f1_per_class = f1_score(y_true, y_pred, average=None, labels=np.arange(C), zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    _ensure_dir(output_dir)
    save_path = os.path.join(output_dir, os.path.basename(out_png))

    plt.figure(figsize=(8,5))
    x = np.arange(C)
    plt.bar(x, f1_per_class)
    plt.xticks(x, class_names, rotation=15)
    plt.ylim(0, 1.0)
    plt.ylabel("F1 score")
    plt.title(f"Per-class F1 | micro={f1_micro:.3f}, macro={f1_macro:.3f}, weighted={f1_weighted:.3f}")
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(save_path); plt.close()
    print(f"[F1-bars] saved: {save_path}")

    return {
        "per_class": {cn: float(v) for cn, v in zip(class_names, f1_per_class)},
        "micro": float(f1_micro),
        "macro": float(f1_macro),
        "weighted": float(f1_weighted),
    }
    
CLASS_NAMES = {0:"Non-Accident", 1:"After-Accident", 2:"Accident"} if NUM_CLASSES==3 else {0:"Non-Accident", 1:"Accident"}
roc_info = plot_roc_auc(all_labels, np.array(all_probs), CLASS_NAMES, out_png="tformer_roc_auc_val.png", output_dir=plot_dir)
print("ROC-AUC info:", roc_info)

if NUM_CLASSES == 2:
    f1_curve = plot_f1_binary_vs_threshold(all_labels, np.array(all_probs),out_png="tformer_f1_vs_threshold_val.png", output_dir=plot_dir)
    print("F1(best):", f1_curve)
else:
    f1_info = plot_f1_multiclass_bars(all_labels, np.array(all_probs), CLASS_NAMES, out_png="tformer_f1_per_class_val.png", output_dir=plot_dir)
    print("F1 summary:", f1_info)