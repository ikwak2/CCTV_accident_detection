# 나중에 코드·원고 수정 시 확인할 기록

작성 시각: **2026-11-07 10:35 KST**. 사용자 요청에 따라 확인 결과와 추후 작업만 기록한다. 이번 문서 업데이트에서 실행 코드를 수정하거나 큐를 재시작하지 않았다. 원고 수정은 사용자가 추후 진행한다. 전체 리뷰 대응 전략은 [PLAN.md](PLAN.md), 같은 시각의 실행 현황은 [STATUS.md](STATUS.md)를 참고한다.

## 1. 현재 우선 확인할 평가 디코딩 오류

### 확인된 사실

- 새 학습 24회 중 16회 완료, 2회 실행, 6회 대기. 학습 실패는 없다.
- seed 42의 `vivit_qkvo`, `timesformer_qkvo`, `videomae_out` × `mixed`, `cadp` 평가 6건이 각각 2회 시도 후 실패했다.
- 각 run의 `dev/predictions_private.jsonl`은 737건, `test/predictions_private.jsonl`은 274건이다. manifest 순서상 다음 입력은 여섯 run 모두 같은 test 275번째 UTIC 정상 영상이다.
- 파일 SHA256: `e0b3d55169a72d78481f2ace037d12ffdd290b16ddec49651860c7e1fc4ed2ca`. 해당 입력은 서버의 `results/manifests/grouped.csv`에서 `file_sha256`으로 찾을 수 있다. 비공개 UTIC 파일명·카메라 이름은 이 문서에 올리지 않는다.
- 로그 끝 오류는 `av.error.InvalidDataError: [Errno 1094995529] Invalid data found when processing input; last error log: [h264] no frame!`다. 직전에 `error while decoding MB 44 29, bytestream -6`, `missing picture in access unit with size 56`가 기록됐다.
- 예외는 test 점수 계산 중 발생한다. 학습 best checkpoint와 앞서 저장된 dev/test 점수는 서버에 남아 있다. 완성된 새 grouped 평가 `metrics.json`은 아직 없다.
- 영상 파일 자체의 손상과 seek/decoder 동작을 구분하는 독립 재현 검사는 수행하지 않았다. 현재 증거는 같은 입력 지점과 같은 디코딩 예외가 반복된다는 것이다.

### 볼 코드 위치

줄 번호와 링크는 실행 코드 [71a25a6](https://github.com/ikwak2/CCTV_accident_detection/commit/71a25a64c3c17645ee372edf3e650ebbfb97811b) 기준이다.

| 코드 | 현재 동작과 확인할 부분 |
|---|---|
| [prepare.py:82–120](https://github.com/ikwak2/CCTV_accident_detection/blob/71a25a6/revision/prepare.py#L82-L120) | 초기 검사는 CADP 3시점, UTIC/UCF 9시점의 프레임을 디코딩한다. 평가에서 실제 읽는 5개 시간창을 모두 검사하지는 않는다. |
| [common.py:165](https://github.com/ikwak2/CCTV_accident_detection/blob/71a25a6/revision/common.py#L165) | `read_window`의 seek와 `container.decode(stream)`에서 예외가 난다. |
| [run.py:75–117](https://github.com/ikwak2/CCTV_accident_detection/blob/71a25a6/revision/run.py#L75-L117) | `score_rows`: 공통 4초 범위의 5개 시간창. 한 영상의 모든 시간창 계산 후 저널에 저장하므로 실패 영상은 완료 기록이 없다. 다른 영상으로 대체하지 않는다. |
| [run.py:132–186](https://github.com/ikwak2/CCTV_accident_detection/blob/71a25a6/revision/run.py#L132-L186) | `evaluate`: dev → test → external 순서다. test에서 중단돼 external 새 평가로 넘어가지 못했다. |
| [experiment_queue.py:193](https://github.com/ikwak2/CCTV_accident_detection/blob/71a25a6/revision/experiment_queue.py#L193) | 2회 실패 후 `failed`. 같은 worker를 재시작하는 것만으로 실패 작업을 자동 재등록하지 않는다. 선행 평가가 실패한 통계·주석 작업은 대기한다. |

로그 위치는 `/disk1/orv_revision_20261106/results/logs/eval_RUN_ID.log`, 점수 저널은 `results/runs/RUN_ID/{dev,test}/predictions_private.jsonl`이다. `RUN_ID` 예: `vivit_qkvo_mixed_s42`.

### 추후 수정·재개 순서 — 미실행

1. 동일 이미지·PyAV 버전으로 해당 입력의 평가 시간창을 재현한다. 16/32프레임, 요청 시작·끝 시각, 실패한 seek/구간을 남긴다. 전체 프레임이 정상인 경우와 임의 패킷을 버려 복구한 경우를 구분한다.
2. dev/test/external에 대해 실제 평가와 같은 시간창의 디코딩 유효성을 확인한다. 부분 프레임 검사를 통과했다는 이유만으로 전체 평가 가능이라고 가정하지 않는다.
3. 처리 규칙을 모델별 점수와 무관하게 고정한다. 입력을 제외한다면 **모든 모델·학습 조건·seed가 동일한 평가 가능 영상 목록**을 사용하고, 제외 사유·개수·클래스·그룹·해시를 보존한다. 예외를 삼키거나 다른 영상을 대신 넣지 않는다.
4. 학습용 manifest와 이미 학습한 checkpoint의 해시를 보존한다. 평가 가능 목록은 별도 버전/해시로 관리하는 방안을 우선 검토한다. test 입력 한 건의 디코딩 문제만으로 기존 모든 학습을 다시 해야 한다고 단정하지 않는다. 학습/dev 입력이나 split을 바꾸면 재학습·재보정 필요성을 별도로 판단한다.
5. 저널 재사용 전에 checkpoint·입력·전처리·샘플 시각이 같은지 확인한다. 디코딩/전처리 규칙이 바뀐 영상의 기존 점수는 재계산한다. dev 유효 목록이 바뀌면 source-dev threshold도 다시 정한다.
6. 수정 코드를 새 commit과 읽기 전용 release로 배포한 뒤 실패 평가를 명시적으로 재등록한다. 현재 실행 release를 덮어쓰지 않는다. 학습 checkpoint와 실패 로그를 보존한다.
7. 동일 평가 분모를 확인한 뒤 외부 평가, seed 집계·group bootstrap, 주석 템플릿, 설명 진단을 재개한다.

이 절차는 나중에 구현할 권고다. 이번 기록 작업에서는 데이터 제외·원본 재인코딩·failed 상태 변경·작업 중단을 하지 않았다.

## 2. 원고에서 먼저 수정할 VideoMAE 결과 해석

### 헤드 불일치와 완료된 기존 가중치 진단

- 원 학습 [main/models/VideoMAE.py:428](https://github.com/ikwak2/CCTV_accident_detection/blob/71a25a6/main/models/VideoMAE.py#L428)은 `768 → 256 → 2` MLP를 저장한다.
- 기존 외부 평가 [main/utils/videomae_general.py:458](https://github.com/ikwak2/CCTV_accident_detection/blob/71a25a6/main/utils/videomae_general.py#L458)은 `768 → 2` 선형 헤드를 만들고 [476행](https://github.com/ikwak2/CCTV_accident_detection/blob/71a25a6/main/utils/videomae_general.py#L476)에서 `strict=False`로 불일치를 무시한다.
- 새 runner [common.py:126–159](https://github.com/ikwak2/CCTV_accident_detection/blob/71a25a6/revision/common.py#L126-L159)는 학습과 같은 헤드와 LoRA를 구성하고 `strict=True`로 기존 checkpoint를 불러왔다. 이 경로의 진단은 이미 완료됐다.

서버 결과: `/disk1/orv_revision_20261106/results/runs/legacy_videomae_corrected/metrics.json`.

| 결과 | 값 | 원고에 쓸 때 의미 |
|---|---|---|
| 전체 UCF | 사고 150, 정상 1,000 | 기존 가중치 FP32 진단의 평가 집합 |
| AUROC / AP | 0.887733 / 0.565066 | 새 그룹 분할 재학습 결과로 쓰지 않는다. |
| threshold 0.5 | recall 0.94, FPR 0.365 | 높은 recall만으로 운영 적합성을 주장하지 않는다. |
| 같은 UCF labels로 FPR ≤ 0.1 선택 | threshold 0.944450, recall 0.666667, MDR 0.333333 | target-test labels를 사용한 사후 진단이다. source-dev 고정 threshold 결과와 구분한다. |

**수정 방향:** 기존 AUROC 0.388을 VideoMAE의 masked pretraining 탓으로 설명한 문구와 모델 순위 결론을 유보한다. 헤드 호환성 문제를 공개하고, 같은 평가 계약으로 재평가한 결과로 표·서술을 맞춘다. 이번 진단은 시간 샘플링·전처리도 달라졌으므로 0.388→0.888 변화 전체를 헤드 교정만의 인과 효과라고 표현하지 않는다. 초록, §3.4, §5.2, 결론과 R1-4/R2-3 대응에 반영한다.

추가로 이 legacy 진단은 콘텐츠 감사 완료 전에 시작해 전체 UCF를 사용했다. 해당 JSON의 `UCF_candidate_quarantined` 항목도 1,150건으로, 자동 후보 12개를 제외한 결과가 아니다. 새 그룹 평가의 전체 UCF 1,150건/후보 격리 후 1,138건 결과와 혼동하지 않는다. legacy 결과의 환경 기록은 `source_commit=uncommitted`이므로 실제 기록된 파일별 코드 해시와 모델 revision을 함께 보존한다.

### 설명 heatmap 해석

- 기존 [main/cam/rollout/rollout_mae.py:78](https://github.com/ikwak2/CCTV_accident_detection/blob/71a25a6/main/cam/rollout/rollout_mae.py#L78)은 `rollout[:, 0, 1:]`로 CLS를 가정하고 남은 토큰을 임의 factorization한다.
- 현재 VideoMAE 설정은 CLS 없이 `8 × 14 × 14 = 1,568`토큰을 mean pooling한다. 기존 합성 입력 검사는 `(1,1567,1,1)` 출력으로 잘못된 grid를 재현했다.
- 새 구현은 [explanations.py:6](https://github.com/ikwak2/CCTV_accident_detection/blob/71a25a6/revision/explanations.py#L6)의 명시적 토큰 grid와 readout을 사용한다. 자동 설명 진단은 평가 선행 조건이 충족될 때까지 대기한다.
- stripe/grid 현상을 masking의 본질적 한계 또는 모델의 Criterion 3 실패로 단정한 서술을 수정한다. 보정 rollout도 사고 localization·faithfulness·운영자 효용의 정량 검증을 대체하지 않는다. §3.3.1, §5.4.1, Table 6 및 R2-4에 반영한다.

### Q/K/V LoRA 동작 검증

[common.py:92](https://github.com/ikwak2/CCTV_accident_detection/blob/71a25a6/revision/common.py#L92)는 Transformers 4.46 계열의 함수형 `linear(..., module.weight)`가 PEFT forward를 우회하는 경로를 보정한다. preflight에서 보정 전후 FP32 logit 차이 `2.46e-07`, 다섯 모델/LoRA 설정의 실제 LoRA B gradient를 확인했다. 이 검증은 학습 경로가 작동한다는 근거이며 최종 성능의 우월성을 입증하지 않는다.

## 3. 분할·표·운영 주장 수정 시 함께 볼 항목

| 항목 | 확인 근거와 추후 반영 |
|---|---|
| Reviewer 2의 split 지적 | 기존 clip-level split 복원 test 359개가 세 저장 예측 목록과 일치했다. CADP test 239클립 중 132개가 train과 원본 ID를 공유한다. 원본/식별된 콘텐츠 그룹으로 다시 학습한 결과가 필요하다. |
| 새 split | train 2,491 / dev 737 / test 343. manifest SHA256 `74fd4acfac1da2d85e98a888fef5c36b17d1083b9771b78634a416ec574ce9db`. 식별된 그룹 간 중복 0과 완전한 사건·카메라 독립성을 구분한다. 추가 디코딩 처리가 확정되기 전 test 평가 분모를 변경했다고 쓰지 않는다. |
| Table 1 | CADP 정상 파일은 139개, UTIC 1,144개. 1,283은 두 수의 합이다. 출처별 class 수를 구분한다. |
| Table 4 | 기존 저장 출력 재계산: ViViT TN/FP/FN/TP=903/97/64/86, AUROC .832453; TimeSformer 902/98/65/85, .829740; VideoMAE 901/99/142/8, .388047. 원고 전사 오류를 바로잡되 이 값은 교정된 새 결과가 아니다. 자세한 표는 PLAN P0.4 참조. |
| 중복 후보 | 자동 콘텐츠 후보 1,158기록은 확정 중복 사건 수가 아니다. UCF 격리 후보 12개도 수동 검토가 필요하다. 전체/후보 격리 결과는 클래스 수와 함께 보고한다. |
| threshold | 최종 새 결과는 source-dev에서 정한 threshold를 외부 평가에 그대로 적용한다. target labels로 고른 10% FPR 지점은 별도 사후 진단으로 표시한다. |
| UTIC ablation | CADP-only와 CADP+UTIC의 평가 집합을 같게 유지하고, 정상 수/class prior 차이를 함께 설명한다. 국내 정상만으로 국내 사고 recall 향상을 입증하지 않는다. |
| 자원 | 완료된 benchmark는 분류 그래프와 단일 시간창의 반복 측정이다. [run.py:285](https://github.com/ikwak2/CCTV_accident_detection/blob/71a25a6/revision/run.py#L285)는 pretrained+LoRA 모델을 만들고 best 가중치를 읽지 않는다. YOLO/설명/동시 스트림 포함 전체 지연으로 쓰지 않는다. |
| ORV 결론 | 상대 drop·절대 recall/오경보·자원 예산을 구분한다. 현장 요구·onset·box/ID·운영자 평가가 없으면 deployment-ready, 조기 탐지, 검토 시간 단축, 추적 정확도 주장을 확정하지 않는다. |

## 4. 서버 결과를 읽을 때 주의할 점

- 실행 코드 release: `/disk1/orv_revision_20261106/releases/71a25a6`; 결과: `/disk1/orv_revision_20261106/results`. 문서 commit을 올려도 실행 중인 컨테이너의 코드가 자동으로 바뀌지 않는다.
- `training_complete.json`과 큐의 train 상태는 학습 완료 근거다. `history.jsonl` 마지막 epoch와 `progress.json`의 중간 step이 다를 수 있다.
- 예측 완료 수는 저널의 고유 ID로 확인한다. 이번 평가에서는 `progress.json`이 261/343이어도 실제 저장 완료는 274건이다.
- 학습 loss/dev loss만으로 세 모델의 test/UCF 성능 순위를 정하지 않는다. 공통 평가가 완성되기 전 부분 예측 결과를 최종 표로 사용하지 않는다.
- 이전 안내의 첫 비교 시각은 디코딩 오류로 지연됐다. 최종 완료 ETA는 오류 처리·재평가 일정이 정해진 후 다시 계산한다.
