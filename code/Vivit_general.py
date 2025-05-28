# torch 환경 세팅 버전(기록용)
# !pip install -U torch==2.3.1
# !pip install torchvision==0.17.2
# batch size 16까지는 가능

import av
import os
import torch
import random
import numpy as np
from transformers import VivitImageProcessor, VivitForVideoClassification, VivitConfig
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchmetrics.classification import Accuracy
from peft import LoraConfig, get_peft_model
from tqdm import tqdm

# 시드 고정
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# CUDA 연산에서 deterministic 설정
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# print(torch.cuda. is_available())

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Device:', device)  # 출력결과: cuda 
print('Count of using GPUs:', torch.cuda.device_count())  
print('Current cuda device:', torch.cuda.current_device()) 

model = VivitForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400")
image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")

# Initializing a ViViT google/vivit-b-16x2-kinetics400 style configuration
configuration = VivitConfig()
# Accessing the model configuration
configuration = model.config

#####################################################################################################################
########################################### Def function ############################################################

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


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
    print(f"Processing: {video_file}") 
    container = av.open(video_file)
    
    # 비디오의 총 프레임 개수 가져오기
    total_frames = container.streams.video[0].frames
    if total_frames == 0:
        raise ValueError(f"ERROR: 비디오 {video_file}의 총 프레임 수가 0이므로 처리 불가")

    indices = sample_frame_indices(clip_len=32, frame_sample_rate=1, seg_len=total_frames)
    video = read_video_pyav(container=container, indices=indices)
    inputs = image_processor(list(video), return_tensors="pt")
    print(f"Processed input shape: {inputs['pixel_values'].shape}")
    
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    
    return last_hidden_states.shape

# 데이터 경로
# video_folder_path = "/Data3/hj24/CCTV/DATA/crashdata"
video_folder_path = "/Data/data"
video_files = [os.path.join(video_folder_path, filename) for filename in os.listdir(video_folder_path) if filename.endswith(".mp4")]

# map()을 사용하여 모든 비디오 파일을 처리
results = map(process_video, video_files)

###################################################################################################################
########################################### Data input ############################################################

# 데이터셋 클래스 정의
class VideoDataset(Dataset):
    def __init__(self, video_files, labels, image_processor):
        self.video_files = video_files
        self.labels = labels
        self.image_processor = image_processor

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        if idx >= len(self.video_files):
            print(f"⚠️ Warning: Index {idx} out of range for dataset size {len(self.video_files)}")
            idx = idx % len(self.video_files)
            
        container = av.open(self.video_files[idx])
        total_frames = container.streams.video[0].frames
        
        if total_frames == 0:
            raise ValueError(f"ERROR: 비디오 {self.video_files[idx]}의 프레임 수가 0")

        indices = sample_frame_indices(clip_len=32, frame_sample_rate=4, seg_len=total_frames)
        video = read_video_pyav(container, indices)
        inputs = self.image_processor(list(video), return_tensors="pt")
        return inputs["pixel_values"].squeeze(0), torch.tensor(self.labels[idx], dtype=torch.long)

# 데이터 분류
# accident, non-accident label 생성을 위해 데이터 폴더 분리 + 이걸 이용해 레이블 생성
video_folder_path = "/Data/data"
video_files = []
labels = []

accident_folder = os.path.join(video_folder_path, "accident")
if os.path.exists(accident_folder):
    for filename in os.listdir(accident_folder):
        if filename.endswith(".mp4"):
            video_files.append(os.path.join(accident_folder, filename))
            labels.append(1) 

non_accident_folder = os.path.join(video_folder_path, "non-accident")
if os.path.exists(non_accident_folder): 
    for filename in os.listdir(non_accident_folder):
        if filename.endswith(".mp4"):
            video_files.append(os.path.join(non_accident_folder, filename))
            labels.append(0) 

class VideoClassifier(nn.Module):
    def __init__(self, base_model):
        super(VideoClassifier, self).__init__()
        self.vivit = base_model
        self.classifier = nn.Linear(self.vivit.config.hidden_size, 2)  # 2개의 클래스 (accident, non-accident)

    def forward(self, x):
        outputs = self.vivit(x)
        features = outputs.last_hidden_state[:, 0, :]
        return self.classifier(features)

def initialize_classifier_weights(model):
    """분류 부분 가중치 초기화 코드"""
    for m in model.modules():
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)

# LoRA
lora_config = LoraConfig(
    r=8,  # LoRA 랭크
    lora_alpha=16,  # 스케일링 파라미터 # r*2
    target_modules=["attention.output.dense"],  # 타겟 모듈 # 늘릴 수 있음
    lora_dropout=0.1
)

# model = VivitForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400")
model.classifier = nn.Linear(model.classifier.in_features, 2)
model = get_peft_model(model, lora_config)
# model = VideoClassifier(base_model)
model.apply(initialize_classifier_weights)

# multi gpu 할당 적용
model = nn.DataParallel(model).to(device)

# 데이터로더 생성
image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
dataset = VideoDataset(video_files, labels, image_processor)
generator = torch.Generator().manual_seed(torch.initial_seed()) # 랜덤하게 데이터 섞기

# 분리
total_size = len(dataset)
train_size = int(0.7 * total_size)
# val_size = int(0.2 * total_size)
val_size = total_size - train_size
# test_size = total_size - train_size - val_size

# train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator)
# print(f"Train: {len(train_dataset)}, Validation: {len(val_dataset)}, Test: {len(test_dataset)}")
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator)
print(f"Train: {len(train_dataset)}, Validation: {len(val_dataset)}")

# DataLoader 설정 (매 에폭마다 데이터 순서를 랜덤하게)
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 새 test data import

def load_test_dataset(test_data_path):
    video_files, labels = [], []

    image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")

    # 1. RoadAccidents (label 1)
    roadaccident_dir = os.path.join(test_data_path, "RoadAccidents")
    accident_videos = [os.path.join(roadaccident_dir, f) for f in os.listdir(roadaccident_dir) if f.endswith(".mp4")]
    accident_videos = accident_videos[:150]  # 150개만 사용
    video_files.extend(accident_videos)
    labels.extend([1] * len(accident_videos))

    # 2. Non-accident (다른 폴더들에서 랜덤하게 150개)
    non_accident_dirs = [
        d for d in os.listdir(test_data_path)
        if os.path.isdir(os.path.join(test_data_path, d)) and d != "RoadAccidents"
    ]

    non_accident_videos = []
    for dir_name in non_accident_dirs:
        full_dir = os.path.join(test_data_path, dir_name)
        non_accident_videos.extend([
            os.path.join(full_dir, f) for f in os.listdir(full_dir) if f.endswith(".mp4")
        ])

    random.shuffle(non_accident_videos)
    non_accident_videos = non_accident_videos[:150]

    video_files.extend(non_accident_videos)
    labels.extend([0] * len(non_accident_videos))

    return VideoDataset(video_files, labels, image_processor)

# 새 test dataset/dataloader 정의
test_data_path = "/Data/data/crimedata/UCF_Crimes/Videos"
test_dataset = load_test_dataset(test_data_path)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

###################################################################################################################
########################################### Classification ########################################################

# 학습 설정
for name, param in model.named_parameters():
    if "encoder" in name:
        param.requires_grad = True

criterion = nn.CrossEntropyLoss()
# trainable_params = list(model.module.vivit.parameters()) + list(model.module.classifier.parameters())
trainable_params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.Adam(trainable_params, lr=5e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose = True)
accuracy_metric = Accuracy(task="multiclass", num_classes=2).to(device)

train_acc, val_acc, train_loss, val_loss = [], [], [], []

# 모델 학습 루프
for epoch in tqdm(range(50), desc="Total Epoch Progress"):
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

            probs = torch.softmax(logits, dim=1)              # 확률로 변환
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())        # 사고(class 1)의 확률만 저장

            accuracy_metric.update(preds, labels)

    val_accuracy = accuracy_metric.compute().item()
    accuracy_metric.reset()
    
    val_acc.append(val_accuracy)
    val_loss.append(total_val_loss / len(val_loader) )
    # val_loss.append(loss.item())
    
    print(f"Epoch {epoch+1}, Loss: {train_loss[-1]:.4f}, Accuracy: {epoch_accuracy:.4f}, Validation Loss:{val_loss[-1]:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    scheduler.step(val_loss[-1])

torch.save(model.state_dict(), "vivit_accident_classifier.pth")
np.savez("train_val_stats.npz", 
         train_acc=train_acc,
         val_acc=val_acc,
         train_loss=train_loss,
         val_loss=val_loss,
         all_probs=all_probs,
         all_labels=all_labels)

val_stats = np.load("/Data/cctv/train_val_stats.npz", allow_pickle=True)
val_probs = val_stats["all_probs"]
val_labels = val_stats["all_labels"]

# 사고/비사고 분리
accident_probs = val_probs[val_labels == 1]
non_accident_probs = val_probs[val_labels == 0]

# 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.hist(accident_probs, bins=20, alpha=0.6, color="tomato", label="Accident")
plt.hist(non_accident_probs, bins=20, alpha=0.6, color="skyblue", label="Non-Accident")
plt.xlabel("Predicted Probability for Class 1 (Accident)")
plt.ylabel("Sample Count")
plt.title("Validation Set Probability Distribution by Class")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("vivit_val_prob_hist_split.png")
print("그래프 저장 완료: vivit_val_prob_hist_split.png")

###################################################################################################################
################################################# Test ############################################################
def evaluate_on_test(model, test_loader):
    model.eval()
    accuracy_metric.reset()
    total_test_loss = 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            logits = outputs.logits

            loss = criterion(logits, labels)

            total_test_loss += loss.item()
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

            accuracy_metric.update(preds, labels)

    acc = accuracy_metric.compute().item()

    print(f"[TEST] Accuracy: {acc:.4f}, Loss: {total_test_loss/len(test_loader):.4f}")

    return acc, total_test_loss / len(test_loader), all_preds, all_labels, all_probs

import collections
import matplotlib.pyplot as plt

# 예측 결과 분포 계산
acc, test_loss, all_preds, all_labels, all_probs = evaluate_on_test(model, test_loader)
np.savez("test_stats.npz",
         acc=acc,
         test_loss=test_loss,
         all_preds=all_preds,
         all_labels=all_labels,
         all_probs=all_probs)

# 재사용 시에만 사용하는 코드
# test_stats = np.load("/Data/cctv/test_stats.npz")
# all_probs = test_stats["all_probs"]
# all_preds = test_stats["all_preds"]
# all_labels = test_stats["all_labels"]
# counter = collections.Counter(all_preds)

# 결과 출력
# print("Prediction Distribution:", dict(counter))

# 시각화
# 사고/비사고 분리
all_probs = np.array(all_probs)
all_labels = np.array(all_labels)
assert all_preds.shape == all_labels.shape == all_probs.shape, (
    f"Shape mismatch: preds={all_preds.shape}, labels={all_labels.shape}, probs={all_probs.shape}"
)

accident_probs = all_probs[all_labels == 1]
non_accident_probs = all_probs[all_labels == 0]

# 시각화
plt.figure(figsize=(10, 5))
plt.hist(accident_probs, bins=20, alpha=0.6, color="tomato", label="Accident")
plt.hist(non_accident_probs, bins=20, alpha=0.6, color="skyblue", label="Non-Accident")
plt.xlabel("Predicted Probability for Class 1 (Accident)")
plt.ylabel("Sample Count")
plt.title("Probability Distribution by Class (Test Set)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("vivit_test_prob_hist_split.png")
print("그래프 저장 완료: vivit_test_prob_hist_split.png")


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 테스트 결과 불러오기
test_stats = np.load("/Data/cctv/test_stats.npz")
y_true = test_stats["all_labels"]
y_scores = test_stats["all_probs"]

# ROC curve 계산
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# 그래프 그리기
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("vivit_test_roc_curve.png")

"ROC curve 저장 완료: vivit_test_roc_curve.png"