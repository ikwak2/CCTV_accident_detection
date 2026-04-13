# Real-Time CCTV-Based Traffic Accident Detection for Transportation Safety Monitoring Using Parameter-Efficient Video Transformers

## Overview
This repository implements a real-time CCTV-based accident detection system using parameter-efficient video transformers (Binary ViViT + LoRA).

--------------------------------------------

## Dataset

CADP data: https://ankitshah009.github.io/accident_forecasting_traffic_camera

Generalization: UCF Crime 101 https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/
 
UTIC: https://www.utic.go.kr/guide/utisRefCctv.do

![UTIC](https://github.com/user-attachments/assets/9386a57e-2811-4896-afc8-a774dd25883d)

--------------------------------------------

## Repository Structure

```text
CCTV_accident_detection/
│
├── assets/        # visualization results (gif)
├── cam/           # Grad-CAM and Attention-Rollout
├── models/        # video transformer models (ViViT, TimeSformer, VideoMAE)
├── utils/         # utility functions for model handling
│
├── LICENSE
└── README.md
```

--------------------------------------------

## Installation

```text
pip install -r requirements.txt
```

--------------------------------------------

## Training

```text
python train.py
```

--------------------------------------------

## Visualization
We provide attention rollout visualization results.
![Attention_rollout_1](https://github.com/ikwak2/CCTV_accident_detection/blob/dfecce75efd3148149cb0f1832958db653589fa5/main/assets/Attention_rollout_1.gif)
