# torch 환경 세팅 버전(기록용)
# !pip install -U torch==2.3.1
# !pip install torchvision==0.17.2

import torch.multiprocessing as mp
import torch.distributed as dist

import av
import os
import torch
import random
import numpy as np
from transformers import VivitModel, VivitImageProcessor, VivitForVideoClassification, VivitConfig
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchmetrics.classification import Accuracy
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp
import torch.distributed as dist

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# CUDA 연산에서 deterministic 설정
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5"
    
    world_size = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'  # 비어 있는 포트 설정
    
    mp.spawn(working,
             args=(world_size,),
             nprocs=world_size,
             join=True)

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

def device_setting(rank):
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    if rank == 0:
        print('Device:', device)  # 출력결과: cuda 
        print('Count of using GPUs:', torch.cuda.device_count())  
        print('Current cuda device:', torch.cuda.current_device()) 
    
    return device

def build_model(rank, device):
        
    # model = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400", attn_implementation="sdpa", torch_dtype=torch.float32)
    model = VivitForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400")
    
    # Initializing a ViViT google/vivit-b-16x2-kinetics400 style configuration
    configuration = VivitConfig()
    # Accessing the model configuration
    configuration = model.config
    
         # LoRA
    lora_config = LoraConfig(
        r=8,  # LoRA 랭크
        lora_alpha=16,  # 스케일링 파라미터 # r*2
        target_modules=["attention.output.dense"],  # 타겟 모듈 # 늘릴 수 있음
        lora_dropout=0.1
    )

    model.classifier = nn.Linear(model.classifier.in_features, 2)
    model = get_peft_model(model, lora_config)
    model.apply(initialize_classifier_weights)

    # multi gpu 할당 적용
    # model = nn.DataParallel(model).to(device)
    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids = [rank])
    
    return model

def process_video(video_file):
    print(f"Processing: {video_file}") 
    container = av.open(video_file)
    
    # 비디오의 총 프레임 개수 가져오기
    total_frames = container.streams.video[0].frames
    if total_frames == 0:
        raise ValueError(f"ERROR: 비디오 {video_file}의 총 프레임 수가 0이므로 처리 불가")
    
    image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
    model = VivitForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400")
    
    indices = sample_frame_indices(clip_len=32, frame_sample_rate=1, seg_len=total_frames)
    video = read_video_pyav(container=container, indices=indices)
    inputs = image_processor(list(video), return_tensors="pt")
    print(f"Processed input shape: {inputs['pixel_values'].shape}")
    
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    
    return last_hidden_states.shape

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
    
@staticmethod
def data_load():
    video_folder_path = "/Data3/hj24/CCTV/DATA/crashdata"
    # video_folder_path = "/Data/data"
    video_files = [os.path.join(video_folder_path, filename) for filename in os.listdir(video_folder_path) if filename.endswith(".mp4")]

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
                
    image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
    dataset = VideoDataset(video_files, labels, image_processor)
    generator = torch.Generator().manual_seed(torch.initial_seed()) # 랜덤하게 데이터 섞기
    
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator)
    
    return train_dataset, val_dataset, test_dataset

def prepare_dataloader(model, train_dataset, val_dataset, test_dataset, rank, world_size):    
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=8, sampler = train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=8, sampler=val_sampler)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    return train_loader, val_loader, test_loader, train_sampler, val_sampler

def initialize_classifier_weights(model):
    """분류 부분 가중치 초기화 코드"""
    for m in model.modules():
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)

def working(rank, world_size):
    dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
    
    device = device_setting(rank) 
    
    model = build_model(rank)
    train_dataset, val_dataset, test_dataset = VideoDataset.data_load()
    train_loader, val_loader, test_loader, train_sampler, val_sampler = prepare_dataloader(
        train_dataset, val_dataset, test_dataset, rank, world_size
    )
    
    # 학습 설정
    criterion = nn.CrossEntropyLoss()
    # trainable_params = list(model.module.vivit.parameters()) + list(model.module.classifier.parameters())
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose = True)
    accuracy_metric = Accuracy(task="multiclass", num_classes=2).to(device)

    train_acc, val_acc, train_loss, val_loss = [], [], [], []
    patience = 5
    best_val_loss = float('inf')
    counter = 0

    # 모델 학습 루프
    for epoch in tqdm(range(50), desc="Total Epoch Progress", position=0):
        train_sampler.set_epoch(epoch)
        
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

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                logits = outputs.logits
                
                loss = criterion(logits, labels)
                total_val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                accuracy_metric.update(preds, labels)

        val_accuracy = accuracy_metric.compute().item()
        accuracy_metric.reset()
        
        val_acc.append(val_accuracy)
        val_loss.append(total_val_loss / len(val_loader) )
        # val_loss.append(loss.item())
        
        if val_loss[-1] < best_val_loss:
            best_val_loss = val_loss[-1]
            counter = 0
            torch.save(model.state_dict(), "best_model.pth")  # 가장 좋은 모델 저장
        else:
            counter += 1

        if counter >= patience:
            print("Early stopping triggered.")
            break
        
        if rank == 0:
            print(f"[Epoch {epoch+1}/{50}] Train Loss: {train_loss[-1]:.4f} | Train Acc: {epoch_accuracy:.4f} || "f"Val Loss: {val_loss[-1]:.4f} | Val Acc: {val_accuracy:.4f}")
        
        scheduler.step(val_loss[-1])
    
    if rank == 0:    
        torch.save(model.state_dict(), "vivit_accident_classifier.pth")
    
    dist.destroy_process_group()
    
    return train_acc, val_acc, train_loss, val_loss

    ###################################################################################################################
    ########################################### 학습 결과 그래프 #########################################################

def output(train_acc, val_acc, train_loss, val_loss, rank):
    
    plt.figure(figsize=(20, 5))

    # 🔹 Accuracy 그래프
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

    if rank == 0:
        plt.show()
        plt.savefig("vivit.png")  # 저장만!
        print("✅ 그래프 저장 완료")

    ###################################################################################################################
    ################################################# Test ############################################################

def evaluate_train(model, test_loader, device):
    
    model.eval()
    criterion = nn.CrossEntropyLoss()
    accuracy_metric = Accuracy(task="multiclass", num_classes=2).to(device)
    total_test_loss = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_test_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            accuracy_metric.update(preds, labels)
            
    acc = accuracy_metric.compute().item()
    print(f"[TEST] Accuracy: {acc:.4f}, Loss: {total_test_loss/len(test_loader):.4f}")
    
def evaluate_test(test_loader, device, model, rank):
    
    # 테스트 예측값 저장
    all_preds = []
    all_labels = []

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

    with torch.no_grad():
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)

    all_preds.extend(preds.cpu().numpy())
    all_labels.extend(labels.cpu().numpy())

    if rank == 0:
        evaluate_train(model, test_loader)

    # Confusion Matrix 계산
    cm = confusion_matrix(all_labels, all_preds)

    # Confusion Matrix 시각화
    if rank == 0:
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Accident", "Accident"], yticklabels=["Non-Accident", "Accident"])
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.show()


if __name__ == '__main__':
    main()