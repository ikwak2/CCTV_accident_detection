# ORV 리비전 계획 및 추가 실험

작성일: 2026-11-06 (Asia/Seoul). 이 문서는 리뷰어 1의 7개, 리뷰어 2의 9개 의견에 대한 실행 계획이다. 실험 결과와 완료 여부는 별도의 `STATUS.md`에 기록한다. 계획을 이미 수행한 결과로 서술하지 않는다.

2026-11-07 확인 결과와 추후 코드·원고 수정 위치는 [FOLLOW_UP_NOTES.md](FOLLOW_UP_NOTES.md)에 기록했다. 사용자 요청으로 이번 추가 업데이트는 문서만 작성하며, 새로 발견한 평가 디코딩 오류에 대한 실행 코드 수정·재평가는 수행하지 않았다.

## 핵심 방향

ORV의 범위를 **현장 평가 전 비교·선별 절차**로 한정한다. 현재의 “ViViT만 deployment-ready”와 “VideoMAE의 masked pretraining 때문에 성능이 붕괴했다”는 결론은 구현 감사와 재실험이 끝나기 전까지 유보한다. 상대적 성능 유지, 절대적 탐지 요구, 자원 요구, 설명의 재현성을 구분한다.

사용 목적은 우선 **교통관제원의 검토를 돕는 사고 후보 영상 선별**로 정의한다. 자동 경보·적시 탐지·운영자 확인 시간 단축·안정적 차량 추적은 각각 별도 검증이 있어야 주장한다. 모든 후보가 운영 요구를 만족하지 못하는 결과도 허용한다.

## P0 — 결과를 바꾸기 전에 평가 구현과 데이터를 검증

### P0.1 VideoMAE 평가 헤드 교정

- 학습 코드와 저장 checkpoint의 분류 헤드는 `768 → 256 → 2` MLP다.
- 기존 `main/utils/videomae_general.py`는 `768 → 2` 선형 헤드를 만들고 `strict=False`로 checkpoint 불일치를 무시한다.
- 학습과 같은 MLP, LoRA, 전처리를 사용하고 missing/unexpected key를 검사한다. 핵심 헤드·LoRA 키가 맞지 않으면 평가를 중단한다.
- 기존 checkpoint를 정확히 불러온 진단 평가를 먼저 수행한다. 기존 분류 코드와 교정 코드의 차이를 기록하되, 기존 가중치 평가는 새 그룹 split의 성능 검증을 대체하지 않는다.
- `VideoMAE`의 Q/K/V projection이 함수형 `linear(weight=module.weight)`를 통해 계산되는 구현에서는 PEFT를 삽입하는 것만으로 LoRA가 작동하지 않을 수 있다. 실제 projection 호출과 gradient를 검증하고, 필요하면 가중치와 bias를 보존하는 동등한 forward를 사용한다.

### P0.2 설명 구현 검증

- VideoMAE는 CLS 없이 `16/2 × 14 × 14 = 1,568`개 토큰을 mean pooling한다.
- 기존 `rollout[:, 0, 1:]`와 임의 factorization을 제거하고 토큰 grid를 명시적으로 검증한다. 기존 함수의 합성 입력 출력은 `(1, 1567, 1, 1)`로 재현됐다.
- ViViT의 CLS readout, VideoMAE의 mean-pooling readout, TimeSformer의 divided attention을 구분한다.
- 시각화 도구의 비호환성을 곧바로 모델의 Criterion 3 실패로 처리하지 않는다. 기존 stripe/grid를 masking 탓으로 설명한 문구를 수정한다.

### P0.3 데이터 명세·그룹 분할·중복 감사

- CADP의 현재 파일 수: accident 1,494, after-accident 804, non-accident 139, 합계 2,437. 원본 ID는 1,416개다.
- UTIC 파일 수는 1,144개다. 총 학습 후보는 3,581개다. 원고 Table 1의 CADP 정상 1,283은 CADP 139 + UTIC 1,144와 같아 전사 또는 중복 집계 여부를 바로잡아야 한다.
- 현재 코드의 clip-level split을 seed 42로 복원한 test 목록 359개는 세 모델의 저장 예측 목록과 모두 일치했다. 복원한 CADP test 239클립 중 132개가 train과 원본 ID를 공유한다. 당시 train/dev manifest는 별도로 확인되지 않았다.
- `000045_seg003.mp4`의 원본 ID `000045`로 묶고, 세 라벨 폴더의 같은 원본을 모두 같은 split에 넣는다.
- 같은 사고의 재업로드·중복 원본은 하나의 그룹으로 합친다. UTIC 반복 녹화는 카메라 ID로 묶고, 숫자 ID와 카메라 이름의 대응을 확인한다.
- 정확한 파일 해시와 여러 시점의 영상 지문으로 중복 후보를 찾는다. CADP–UCF 간 중복·유사 영상도 확인한다. 자동 후보는 확정 중복과 구분하고 수동 검토가 필요한 경우 그대로 기록한다.
- decoding 실패를 다른 split의 다음 샘플로 대체하지 않는다. 공통 유효 영상 명세와 제외 사유를 저장한다.
- 파일명 그룹 분할은 원본 영상 독립성을 보장하며, 완전한 사고 이벤트·카메라 독립성을 자동으로 입증하지는 않는다.
- 원본 파일은 이동·덮어쓰기하지 않는다. 비공개 UTIC 영상·카메라 이름·절대 경로 명세는 GitHub에 올리지 않는다.

### P0.4 기존 표 감사

저장된 세 외부 평가 CSV는 동일한 1,150영상(사고 150, 정상 1,000)을 포함한다. 아래는 기존 출력을 재계산한 값이며 교정된 새 성능이 아니다.

| 모델 | TN | FP | FN | TP | Accuracy | FPR | AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| ViViT | 903 | 97 | 64 | 86 | 0.860000 | 0.097 | 0.832453 |
| TimeSformer | 902 | 98 | 65 | 85 | 0.858261 | 0.098 | 0.829740 |
| VideoMAE | 901 | 99 | 142 | 8 | 0.790435 | 0.099 | 0.388047 |

현재 저장값에서 TimeSformer/VideoMAE의 정확도는 위 혼동행렬과 일치한다. 원고 FPR 및 ViViT AUROC 표기와 대조해야 한다. threshold는 test labels로 선택됐으므로 retrospective diagnostic이다.

GPU별 메모리 profile 항목은 ViViT 4개, TimeSformer 5개, VideoMAE 1개다. 원고의 동일 4 GPU 설명과 대조한다. LoRA·활성 분류 헤드·PEFT의 사용되지 않는 원본 헤드 복제본을 구분해 파라미터와 optimizer state를 재계산한다.

## P1 — 원본 그룹 분할로 주 실험과 UTIC ablation

세 모델을 원래 사전학습 checkpoint에서 다시 학습한다. 새 test로 옮긴 영상을 이미 학습한 기존 사고 적응 가중치를 초기값으로 사용하지 않는다.

| 설정 | Backbone | LoRA | 학습 자료 | Seed |
|---|---|---|---|---|
| 주 실험 | ViViT | Q/K/V/output | CADP+UTIC | 42, 43, 44 |
| 주 실험 | TimeSformer | spatial/temporal QKV/output | CADP+UTIC | 42, 43, 44 |
| 주 실험 | VideoMAE | output-only | CADP+UTIC | 42, 43, 44 |
| UTIC ablation | 위 3모델 | 위와 동일 | CADP-only | 42, 43, 44 |

6설정 × 3seed = 18회 학습. 우선 seed 42의 주 실험 3개를 실행해 새 조건에서의 동작과 비용을 확인한다. 이후 같은 split에서 나머지 seed와 ablation을 순서대로 수행한다.

- 분할 manifest와 평가 영상·시간 구간은 모든 설정이 공유한다.
- CADP 사고, CADP 정상, UTIC 정상, 전체 UCF, 도로 교통 정상에 제한한 UCF 결과를 각각 보고한다.
- CADP-only와 CADP+UTIC의 test는 동일하게 유지한다. 학습 정상 수와 class prior가 달라지는 영향도 구분한다.
- CADP-only는 정상 샘플이 적다. 모델 성능으로 split을 고르지 않으며 클래스 균형 조절 여부를 명시한다.
- UTIC는 정상만 제공하므로 국내 사고 recall 향상을 직접 입증하지 못한다.
- 모든 모델의 실제 입력 시간 길이를 초 단위로 맞춘다. 16/32프레임을 같은 시작/끝 시간에서 샘플링한다. 같은 frame stride만 쓰는 것은 같은 시간 범위가 아니다.
- 학습 microbatch와 effective batch를 구분하고 gradient accumulation으로 공통 effective batch를 유지한다. 새 precision 설정을 모든 모델에 동일하게 적용한다.
- validation으로 best checkpoint를 선택하고 test로 early stopping이나 hyperparameter를 조정하지 않는다.
- 시간 샘플링은 Python `hash(path)` 대신 안정적인 ID와 해시·명시적 시점을 사용한다. dev/test 시점 목록을 저장한다.

## P2 — LoRA 교차 통제

| 추가 설정 | 학습 자료 | 시간 범위 | Seed |
|---|---|---|---|
| ViViT output-only | CADP+UTIC | 주 실험과 동일 | 42, 43, 44 |
| VideoMAE Q/K/V/output | CADP+UTIC | 주 실험과 동일 | 42, 43, 44 |

추가 6회, 총 8설정·24회 학습. 이 통제를 해도 아키텍처와 사전학습 이력의 차이는 남으므로 pretraining objective의 독립적인 인과 효과로 단정하지 않는다. VideoMAE는 masked pretraining 후 supervised Kinetics-400 fine-tuning을 거친 checkpoint다.

## P3 — 통계·운영 기준·추론 자원

- 원본/사고 그룹 단위 paired bootstrap으로 AUROC, AP, MDR와 모델 차이의 CI를 구한다. 모델마다 같은 재표집 그룹을 사용한다.
- training seed 변동과 test-sampling CI를 구분한다. AP와 trapezoidal PR-AUC를 혼용하지 않는다.
- Drop Ratio 기준 10/15/20/25%와 파라미터 예산 기준을 바꾸어 개별·종합 판정의 안정 범위를 보고한다.
- 20% drop과 2% trainable ratio를 현장에서 검증된 보편적 기준으로 표현하지 않는다.
- source dev에서 정한 threshold를 외부 자료에 그대로 적용하는 결과를 우선 보고한다. target calibration을 허용하면 별도의 사고 그룹을 분리하고 이미 사용한 benchmark임을 명시한다.
- MDR CI에서 threshold 고정 조건부 분석과 calibration 재선택까지 포함한 nested 분석을 구분한다.
- 사용 목적에 따른 최소 recall, 허용 미탐, false alerts/camera-hour, delay 요구를 독립적으로 둔다. 이해관계자 근거가 없으면 illustrative 조건으로 표시하고 운영 적합성은 미확립으로 둔다.
- server1의 같은 단일 GPU에서 decode·전처리·분류, 설명+YOLO 포함 end-to-end p50/p95 latency, peak VRAM, throughput, 동시 스트림 수를 측정한다.
- batch 24로 정상 학습되는 모델을 batch 28이 불가능하다는 이유만으로 운영 부적합 판정하지 않는다.

## P4 — 설명·적시 탐지·객체 추적

자동 학습 큐와 별도로 사람의 주석·운영 요구 확인이 필요한 단계다. 주석이 없는 지표를 만들어 보고하지 않는다.

1. **공통 설명 표본**: TP/FN/FP와 조명·가림 조건을 포함한 고정 공통 영상·시점에서 세 모델을 비교한다. 사고 관련 차량 box를 주석해 localization hit rate, area-aware attention mass를 평가한다. 상위 영역 occlusion은 같은 면적의 무작위 대조군과 비교한다. 평가자 평정을 사용하면 공통 척도와 평가자 간 일치도를 보고한다.
2. **인과적 스트림 평가**: 입력 끝 시각 t에서 `[t-W,t]`만 사용한다. onset, 경보 병합, 누적 조건, refractory period, event matching을 사전 정의한다. event recall, 시간 안 recall, alarm delay, false alerts/camera-hour를 보고한다. 미탐을 제외한 delay만 보고하지 않는다.
3. **객체 출력 구분**: window classification probability, YOLO confidence, attention-based relevance를 구분한다. 박스 내 평균 attention을 accident probability라고 부르지 않는다.
4. **추적 평가**: 연속 시퀀스의 timestamp와 persistent ID를 제시하고 localization P/R와 IDF1을 별도 평가한다. YOLO version, detection/relevance threshold, IoU, max_miss, max_tracks를 명시한다. 추적 검증을 생략하면 object-association prototype으로 주장을 제한한다.

## 리뷰 의견과 수정 위치

| 의견 | 대응 | 주요 수정 위치 |
|---|---|---|
| R1-1 | screen 범위, inference·operator workload 보완 | §2, §5.3, §6 |
| R1-2 | 사용 목적·절대 acceptance와 deployment 표현 | 초록, §2.2, §5.5, 결론 |
| R1-3 | drop threshold 민감도·불확실성·재보정 | §2.2, 새 표 |
| R1-4 | head 오류 교정, 시간·LoRA 통제, 사전학습 해석 제한 | §3.4, §5.2, 초록, 결론 |
| R1-5 | 공통 설명 프로토콜·정량 지표/평가자 일치도 | §2.4, §5.4, Table 6 |
| R1-6 | group CI·paired comparison·threshold 선택 불확실성 | §4.3, Tables 3–4 |
| R1-7 | reference system / screen-passing / deployment-ready 구분 | 원고 전체 |
| R2-1 | 원본/사고 그룹 split·재학습·중복 감사 | §4.1, §5.1, Tables 1,3 |
| R2-2 | UTIC ablation·데이터셋별 결과·road-only 정상 | §4.1, §6.1 |
| R2-3 | LoRA 교차 통제·동일 실제 시간 범위 | Table 2, §3.4, §5.2 |
| R2-4 | VideoMAE token handling 교정·기존 artifacts 해석 철회 | §3.3.1, §5.4.1, Table 6 |
| R2-5 | 상대 drop과 절대 요구·자원 예산 구분 | §2, §5.5, Fig.1, 결론 |
| R2-6 | offline classification / causal stream·window score 구분 | §3.2, §3.3.2, §4.1, §6.2 |
| R2-7 | 공통 설명 표본·localization·perturbation·효용 표현 축소 | Fig.4, Table 6, §5.4 |
| R2-8 | localization / tracking / relevance 분리 | §3.3.2, Eq.6, Fig.5 |
| R2-9 | 동일 유효 영상·threshold·confusion matrix·표 오류 | Table 4, 부록 |

## 본문 표현 원칙

예시 문안은 결과와 무관하게 범위를 설명하는 문장이다.

> We present ORV as a comparative pre-deployment screening protocol for video-level CCTV accident classification. Its outcomes do not establish that a candidate meets the detection, alerting, or human-factors requirements of a target deployment.

> Differences among the evaluated configurations cannot be attributed independently to pretraining objectives because architecture, training history, temporal sampling, and adaptation settings also differ.

새 결과에서 VideoMAE가 회복되면 붕괴 서사를 수정하고, ViViT와 TimeSformer의 차이가 불확실하면 우열을 확정하지 않는다. Fig.1과 Table 7의 판정은 새 결과를 확인한 뒤 작성한다. 설명 방법이 아직 검증되지 않은 경우 `not assessed`를 모델 `fail`과 구분한다.

## 실행 환경과 큐

- GPU는 **server1의 물리 GPU 0·1**만 사용한다. 다른 GPU는 사용하지 않는다.
- worker당 GPU UUID 하나만 Docker에 노출하고, 원본 데이터는 read-only mount한다.
- 두 worker가 우선순위와 선행 조건이 있는 공통 큐에서 작업을 하나씩 가져간다.
- 각 작업은 설정·코드 식별자·분할 해시·로그·checkpoint·표본별 예측·실패 원인을 저장한다.
- 실패 작업을 성공으로 표시하지 않는다. 중단 시 재시작 가능한 checkpoint를 보존하고 후속 작업은 선행 조건을 확인한다.
- GPU 자원을 독점하는 다른 작업이 발견되면 해당 GPU의 새 작업 시작을 기다린다.
- 공개 GitHub에는 코드, 계획, 집계 결과만 게시한다. 원본 영상과 비공개 manifest·cache·가중치는 서버에 보관한다.

## 참고 구현

- [공식 VideoMAE 구현](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/videomae/modeling_videomae.py)
- [사용된 VideoMAE checkpoint의 학습 이력](https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics)

패키지 버전, 모델 revision, 실제 실행 코드는 각 실험 산출물에 고정·기록한다.
