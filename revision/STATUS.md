# ORV revision 실행 상태

확인 시각: **2026-11-06T23:58:37+09:00** (Asia/Seoul). 이 문서는 해당 시각의 GitHub 스냅샷이다. 최신 큐와 진행률은 server1의 `/disk1/orv_revision_20261106/results/STATUS.md`, `queue_status.json`, 각 run의 `progress.json`에 자동 기록된다.

## 현재 실행

| server1 GPU | 실행 내용 | 진행 |
|---|---|---|
| GPU 0 | 기존 VideoMAE checkpoint의 MLP 헤드를 교정한 FP32 외부 평가 | 741/1150영상 |
| GPU 1 | ViViT Q/K/V/output LoRA, CADP+UTIC, seed 42 | epoch 2, step 700/1246 (첫 epoch 완료) |

후속 작업은 공통 우선순위 큐에 등록돼 있다. **8설정 × 3seed = 24회 학습**, 각 best checkpoint의 dev/test/UCF 평가, 세 모델 자원 benchmark, 통계 집계, 주석 템플릿, 공통 설명·perturbation 진단을 순서대로 실행한다.

| 큐 상태 | 작업 수 |
|---|---:|
| pending | 56 |
| running | 1 |
| external_running | 1 |
| completed | 3 |
| failed | 0 |

완료 3개는 모델 다운로드·데이터 자동 감사·GPU preflight 선행 작업이다. 학습 24회가 완료됐다는 의미가 아니다.

## 데이터 자동 감사 완료

- 검사 파일: 6,147개; 유효 6,137개, 디코딩/타이밍 검사 제외 10개.
- 원본·카메라 ID 및 식별된 콘텐츠 후보를 반영한 그룹 간 train/dev/test 중복: **0개**.
- 콘텐츠 매칭 후보 기록: 1,158개. 이는 확정 중복 사건 수가 아니다.
- UCF 중복 가능 후보 12개(양성 7, 정상 5)를 보수적으로 격리한 결과와 전체 UCF 결과를 함께 보존한다.
- 전체 UCF는 양성 150/정상 1,000, 후보 격리 후는 양성 143/정상 995다. AP 비교에는 표본 수·class prior 변화를 함께 보고해야 한다.
- 자동 유사도 검사가 찾지 못한 사건·카메라 별칭은 남을 수 있으므로 최종 원고를 확정하기 전에 후보를 수동 검토한다.

| Split | CADP 양성 | CADP 정상 | UTIC 정상 | 합계 | 그룹 수 |
|---|---:|---:|---:|---:|---:|
| train | 1583 | 98 | 810 | 2491 | 1574 |
| dev | 486 | 27 | 224 | 737 | 450 |
| test | 220 | 14 | 109 | 343 | 225 |

분할 manifest SHA256: `74fd4acfac1da2d85e98a888fef5c36b17d1083b9771b78634a416ec574ce9db`. 이전 파일명 기반 후보에서 자동 디코딩·콘텐츠 검사를 추가한 실행용 분할이다. 개인 영상·카메라 이름·전체 경로 명세는 서버에 보관한다.

## 모델·큐 검증 완료

- VideoMAE projection forward 보정 전후 FP32 logit 최대 차이: `2.46e-07`.
- 아래 모든 설정에서 학습될 LoRA B의 실제 gradient를 확인했고, 사용되지 않는 `original_module`은 고정했다.
- 실제 기존 VideoMAE checkpoint의 MLP 헤드가 `strict=True`로 로드됨을 확인했다.
- 새 ViViT 첫 epoch를 완료했고 best/last checkpoint 저장 및 best의 strict 재로딩·manifest 해시 일치를 확인했다. 첫 epoch 약 282.5초, peak allocated VRAM 약 5.18 GiB다.
- 두 worker의 중복 claim 방지·선행 작업 실패 차단·자신의 중단 작업 복구 테스트 통과.
- 인코딩된 영상에서 요청 시간창 이후 프레임을 입력하지 않는 검사와 토큰 grid 검증 통과.
- paired group bootstrap 및 source-calibration 재선택을 포함하는 MDR 분석 smoke check 통과. 실제 CI는 평가 완료 후 산출된다.

| Backbone | LoRA | 실제 활성 trainable parameters |
|---|---|---:|
| videomae | out | 344,834 |
| videomae | qkvo | 787,202 |
| vivit | qkvo | 787,202 |
| vivit | out | 344,834 |
| timesformer | qkvo | 1,082,114 |

## 실행 재현성·복구

- 실행 코드 commit: [`71a25a6`](https://github.com/ikwak2/CCTV_accident_detection/commit/71a25a64c3c17645ee372edf3e650ebbfb97811b). worker는 해당 commit의 읽기 전용 release를 사용한다.
- Docker image ID: `sha256:d250164d203b7a16e74362c6e4b4b48eff5f4348ab101192d4505a5ca408a61e`.
- GPU UUID로 server1의 물리 GPU 0·1을 각각 고정하고 다른 GPU는 사용하지 않는다.
- 원본 데이터는 읽기 전용 mount, 결과·cache는 별도 서버 디렉터리에 저장한다.
- worker 컨테이너는 `unless-stopped`로 재시작하며, 학습은 마지막 완료 epoch checkpoint부터 이어간다.
- 실패는 로그와 큐에 기록하고 성공으로 바꾸지 않는다. 선행 작업이 실패한 후속 작업은 대기한다.

## 사람의 주석·추가 검증이 필요한 항목

- 도로 교통 정상 영상 구분, 사고 onset, 사고 차량 box·persistent ID.
- 콘텐츠/카메라 후보 검토, 현장 최소 recall·허용 오경보·delay 요구.
- 위 정보 없이 road-only recall, event delay, localization hit rate, IDF1, operator-grade/현장 배치 적합성을 보고하지 않는다.
- 공통 설명·perturbation 자동 진단은 주석 전 탐색 결과로 저장하며 ORV pass/fail에 곧바로 사용하지 않는다.

운영 명령과 산출물 위치는 [README.md](README.md), 리뷰어별 전략은 [PLAN.md](PLAN.md)에 정리돼 있다.
