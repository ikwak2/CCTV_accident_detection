# 주 실험 중간 결과

확인 시각: **2026-11-07 21:34 KST**. 학습 24회는 모두 완료했고 평가 24건 중 20건을 마쳤다. 아래는 완료된 CADP+UTIC 주 실험의 seed 42/43/44 지표를 직접 집계한 값이다. 전체 통계 작업 `stats_all`과 LoRA 교차 통제 4개 평가는 아직 남아 있다.

## 공통 조건과 평가 집합

- 원본/식별된 콘텐츠 그룹 분할을 사용해 사전학습 checkpoint부터 새로 학습했다. 자동 그룹화가 완전한 사건·카메라 독립성을 입증하지는 않는다.
- test는 공통 유효 342영상(양성 220/정상 122), 외부 평가는 UCF 1,150영상(양성 150/정상 1,000)이다.
- 모든 모델에 5개 4초 시간창과 동일 평가 manifest를 적용했다. 손상 UTIC test 영상 1개를 공통 제외한 이유는 [DECODING_RECOVERY.md](DECODING_RECOVERY.md)에 있다.
- 표의 SD는 **3개 학습 seed 사이의 표본 표준편차**이며 test-sampling CI가 아니다. 전체 group bootstrap·민감도 집계 완료와 구분한다.

## CADP+UTIC 주 실험

| 모델 | LoRA | test AUROC 평균 | 외부 AUROC 평균 ± SD | 외부 AP 평균 |
|---|---|---:|---:|---:|
| ViViT | Q/K/V/output | 0.93044 | 0.74056 ± 0.02369 | 0.34914 |
| TimeSformer | spatial/temporal QKV/output | 0.94919 | 0.81182 ± 0.02333 | 0.43756 |
| VideoMAE | output-only | 0.95212 | 0.89478 ± 0.00653 | 0.66965 |

자동 중복 가능 후보 12개를 보수적으로 격리한 외부 1,138영상(양성 143/정상 995)의 AUROC 평균은 ViViT 0.73959, TimeSformer 0.80705, VideoMAE 0.89070이다. 격리 후보는 확정 중복 사건이 아니며 수동 검토가 필요하다.

## source-dev threshold의 외부 전달

각 run의 source dev에서 FPR 예산 0.1로 threshold를 정해 UCF에 그대로 적용했다. 아래 값은 3-seed 평균이다. UCF labels로 threshold를 다시 고르지 않았다.

| 모델 | 외부 recall 평균 | 외부 FPR 평균 |
|---|---:|---:|
| ViViT | 0.99111 | 0.87267 |
| TimeSformer | 0.95778 | 0.75333 |
| VideoMAE | 0.94222 | 0.49967 |

AUROC 순위와 운영 threshold의 적합성을 구분해야 한다. source-dev의 10% FPR 조건이 외부에서도 유지된 것은 아니며, 높은 recall만으로 현장 적합성을 주장할 수 없다. 이 FPR은 영상 단위로, camera-hour당 오경보율이 아니다.

## 원고 수정에 주는 의미

- 새 그룹 분할 주 실험에서는 VideoMAE의 외부 AUROC가 높게 나왔다. 기존의 VideoMAE 성능 붕괴를 masked pretraining의 본질적 한계로 설명하는 서술을 유지하면 안 된다.
- 아키텍처, 사전학습 이력, LoRA 적용 범위 차이가 남아 있으므로 위 순위를 pretraining objective의 인과 효과로 해석하지 않는다. 진행 중인 LoRA 교차 통제를 함께 본다.
- 세 모델 모두 source-dev threshold의 외부 FPR이 높다. relative drop과 별도로 threshold 전달·절대 오경보 요구를 논의한다. 자동 결과만으로 deployment-ready를 선언하지 않는다.
- 설명 진단 파일 생성은 완료됐지만, 사고 onset·차량 box/ID·운영자 평정 없이 localization, tracking, operator utility가 검증됐다고 보고하지 않는다.

서버 근거는 `/disk1/orv_revision_20261106/results/runs/{model}_{lora}_mixed_s{seed}/metrics.json`이다. 각 파일에 checkpoint/manifest 해시, dev threshold, 출처별 지표를 보존했다. 첫 seed baseline의 paired bootstrap 9개 비교는 `analysis/baseline/summary.json`, 최종 전체 집계는 완료 후 `analysis/all/`에 저장된다.
