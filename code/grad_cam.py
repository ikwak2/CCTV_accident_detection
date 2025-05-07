import av
import os
import argparse
import cv2
import numpy as np
import torch
import json
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from peft import LoraConfig, get_peft_model
from transformers import VivitImageProcessor, VivitForVideoClassification

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam.ablation_layer import AblationLayerVit

import sys
sys.path.append("/Data3/hj24/CCTV/main/vivit")
from vivit import sample_frame_indices, read_video_pyav

def get_args():
    parser = argparse.ArgumentParser(description='Apply Grad-CAM to ViViT model for visualizing attention on CCTV crash detection videos.')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--model-path', type=str, default='/Data3/hj24/CCTV/main/vivit/vivit_accident_classifier.pth')
    parser.add_argument('--predictions-file', type=str, default='/Data3/hj24/CCTV/gradcam/vivit_pred/vivit_predictions.json')
    parser.add_argument('--output-dir', type=str, default='/Data3/hj24/CCTV/gradcam/vivit_pred/vivit_gradcam_output')
    parser.add_argument('--method', type=str, default='gradcam', 
                       help='Grad-CAM variant to use (options: gradcam, scorecam, gradcam++, ablationcam, xgradcam, eigencam, eigengradcam, layercam, fullgrad).')
    parser.add_argument('--aug_smooth', action='store_true', help='Apply test time augmentation to smooth the CAM.')
    parser.add_argument('--eigen_smooth', action='store_true', help='Reduce noise by taking the first principle component.')
    args = parser.parse_args()
    if args.device:
        print(f'Using device "{args.device}" for acceleration')
    else:
        print('Using CPU for computation')
    return args

# clip_len과 동일하게
time = 32

def reshape_transform(tensor, height=14, width=14, time=time):
    batch_size, total_patches, feature_dim = tensor.size()
    num_patches = height * width
    patch_tensor = tensor[:, 1:, :]  # 클래스 토큰 제외
    result = patch_tensor.reshape(batch_size, time, height, width, feature_dim)
    result = result.permute(0, 4, 1, 2, 3)  # (batch_size, feature_dim, time, height, width)
    return result    
    
def main():
    args = get_args()
    device = torch.device(args.device)

    # 모델 로드 (LoRA 적용)
    model = VivitForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400").to(device)
    
    lora_config = LoraConfig(
        r=8, lora_alpha=16, target_modules=["attention.output.dense"], lora_dropout=0.1
    )
    model = get_peft_model(model, lora_config)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # 타겟 레이어 설정
    target_layers = [model.vivit.encoder.layers[-1].norm]

    # 예측 결과 읽기
    with open(args.predictions_file, "r") as f:
        predictions = json.load(f)

    # Grad-CAM 초기화
    methods = {
        "gradcam": GradCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad
    }
    if args.method not in methods:
        raise Exception(f"method should be one of {list(methods.keys())}")
    
    if args.method == "ablationcam":
        cam = methods[args.method](model=model,
                                  target_layers=target_layers,
                                  reshape_transform=lambda x: reshape_transform(x, time=time),
                                  ablation_layer=AblationLayerVit())
    else:
        cam = methods[args.method](model=model,
                                  target_layers=target_layers,
                                  reshape_transform=lambda x: reshape_transform(x, time=time))
    cam.batch_size = 8

    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)

    # 비디오별 Grad-CAM 계산
    image_processor = VivitImageProcessor.from_pretrained("google/vivit-b-16x2-kinetics400")
    for video_path, target_class in predictions.items():
        print(f"Processing {video_path} with target class {target_class}")
        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        if total_frames == 0:
            print(f"Skipping {video_path}: No frames found")
            continue

        indices = sample_frame_indices(clip_len=32, frame_sample_rate=4, seg_len=total_frames)
        frames = read_video_pyav(container, indices)
        container.close()

        # 입력 텐서 준비
        inputs = image_processor(list(frames), return_tensors="pt")
        input_tensor = inputs["pixel_values"].to(device)  # (1, 32, 3, 224, 224)

        # Grad-CAM 계산
        grayscale_cams = cam(input_tensor=input_tensor, 
                            targets=[torch.tensor([target_class]).to(device)],
                            eigen_smooth=args.eigen_smooth,
                            aug_smooth=args.aug_smooth)  # (1, 32, 224, 224)

        # Grad-CAM 비디오 저장
        output_path = os.path.join(args.output_dir, f'{args.method}_class_{target_class}_{os.path.basename(video_path)}')
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (224, 224))
        for frame, cam_mask in zip(frames, grayscale_cams[0]):
            frame = np.float32(frame) / 255
            cam_image = show_cam_on_image(frame, cam_mask, use_rgb=True)
            # 예측 클래스 텍스트 추가
            text = f"Pred: {'Accident' if target_class == 1 else 'Non-Accident'}"
            cv2.putText(cam_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            out.write(cam_image[:, :, ::-1])
        out.release()

if __name__ == '__main__':
    main()