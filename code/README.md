### 작업 중인 vivit, VideoMAE, Timesformer code file upload

- 사용 데이터(블랙박스 영상) 링크: https://github.com/Cogito2012/CarCrashDataset?tab=readme-ov-file#overview

- 기존에 사용하던 car accident cctv data: https://drive.google.com/drive/folders/1ozfZbz8wRoNqKknC59IPXP_twKgmCOaA

#### 250429 개선 사항

- 메모리 부담을 줄이기 위해 DistributedDataParallel를 사용해 진행, vivit model에 먼저 적용

- grad cam code: 수정 중에 있음, classification과 파일 분리 체제로 진행 예정
