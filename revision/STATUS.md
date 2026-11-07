# ORV revision 실행 상태

확인 시각: **2026-11-07T16:01:17+09:00** (Asia/Seoul). 이 문서는 해당 시각의 GitHub 스냅샷이다. 최신 큐와 진행률은 server1의 `/disk1/orv_revision_20261106/results/STATUS.md`, `queue_status.json`, 각 run의 `progress.json`에 자동 기록된다.

사용자 요청에 따라 이번 업데이트는 **기록만 작성**한다. 실행 코드 수정, worker 교체, 데이터 제외, 실패 작업 재등록은 수행하지 않았다. 기존 worker와 큐는 계속 실행 중이다. 나중에 코드와 원고를 수정할 때 확인할 근거·위치·주의점은 [FOLLOW_UP_NOTES.md](FOLLOW_UP_NOTES.md)에 모았다.

## 현재 실행

| server1 GPU | 실행 내용 | 진행 |
|---|---|---|
| GPU 0 | ViViT output-only LoRA, CADP+UTIC, seed 43 | 3 epoch 완료, 다음 epoch 진행 |
| GPU 1 | ViViT output-only LoRA, CADP+UTIC, seed 42 | 13 epoch 완료, 다음 epoch 진행 |

후속 작업은 공통 우선순위 큐에 등록돼 있다. **8설정 × 3seed = 24회 학습**, 각 best checkpoint의 dev/test/UCF 평가, 세 모델 자원 benchmark, 통계 집계, 주석 템플릿, 공통 설명·perturbation 진단을 순서대로 실행한다.

| 큐 상태 | 작업 수 |
|---|---:|
| pending | 13 |
| running | 2 |
| external_running | 0 |
| completed | 28 |
| failed | 18 |

완료 28개는 **학습 21회 + 선행 작업 3개 + 기존 VideoMAE 진단 1개 + 분류 benchmark 3개**다. 학습은 총 24회 중 21회 완료(87.5%), 2회 실행, 1회 대기이며 학습 실패는 0회다. 주 실험·UTIC ablation 18회와 VideoMAE Q/K/V/output 교차 통제 3회가 모두 학습을 마쳤다. 새 grouped 평가 24건은 18건 실패, 6건 대기로 **완료된 최종 성능 비교는 아직 없다**.

| 학습 설정 | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| ViViT Q/K/V/output, CADP+UTIC | 완료 | 완료 | 완료 |
| TimeSformer QKV/output, CADP+UTIC | 완료 | 완료 | 완료 |
| VideoMAE output-only, CADP+UTIC | 완료 | 완료 | 완료 |
| ViViT Q/K/V/output, CADP-only | 완료 | 완료 | 완료 |
| TimeSformer QKV/output, CADP-only | 완료 | 완료 | 완료 |
| VideoMAE output-only, CADP-only | 완료 | 완료 | 완료 |
| ViViT output-only, CADP+UTIC | 실행 중 | 실행 중 | 대기 |
| VideoMAE Q/K/V/output, CADP+UTIC | 완료 | 완료 | 완료 |

### 평가 중단 원인

세 모델 × 두 학습 조건 × 세 seed 평가 18건이 각각 2회 시도 후 `av.error.InvalidDataError: [h264] no frame!`로 중단됐다. 18건 모두 저널을 확인했으며 dev 737영상과 test 274영상의 점수가 저장돼 있고, 저널상 다음 입력은 동일한 **test 275번째 UTIC 정상 영상**이다. `progress.json`은 20개 간격으로 기록되므로 마지막 표시 261/343과 실제 완료 저널 274/343을 구분해야 한다. 10:35 확인 당시 6건에서 이후 12건이 추가됐으며, 같은 예외의 반복이다.

예외 경로는 `run.py:evaluate → score_rows → common.py:read_window → container.decode`다. 이 입력의 파일 손상/seek 동작 원인을 분리하는 재현 검사는 아직 하지 않았다. 초기 데이터 검사는 일부 시점의 프레임을 확인하므로 실제 평가의 5개 4초 시간창 전체가 정상 디코딩된다는 보장이 없었다. 현재 학습 checkpoint가 실패했다는 근거는 없다.

실패한 평가에 의존하는 baseline 통계·주석 템플릿과 그 후속 설명 진단은 대기한다. 기존 큐는 실행 가능한 다른 학습·평가를 계속 가져간다. 처리 규칙을 수정하지 않으면 후속 평가도 같은 입력에서 중단될 가능성이 있다. 향후 수정 절차는 [FOLLOW_UP_NOTES.md](FOLLOW_UP_NOTES.md)에 기록했다.

### 관측 속도와 완료 전망

아래는 완료된 run별 epoch 평균의 범위다. 초기화·평가 시간은 포함하지 않으며, 디스크/CPU 경합이 포함될 수 있다.

| 모델 | CADP+UTIC: 분/epoch | CADP-only: 분/epoch |
|---|---:|---:|
| ViViT Q/K/V/output | 4.00–5.55 | 2.57–3.41 |
| TimeSformer QKV/output | 10.43–10.94 | 7.32–7.56 |
| VideoMAE output-only | 3.01 (seed 42) | 1.62–1.74 |

추가 완료된 VideoMAE Q/K/V/output 세 run은 12/18/13 epoch, run별 평균 약 5.68–8.31분/epoch였다. 이 평균에는 일부 긴 epoch가 포함돼 시간 변동이 크다. 실행 중인 ViViT output-only 두 run은 평균 약 3.82–4.29분/epoch다. 대기 중인 학습은 같은 설정 seed 44 한 건이다.

현재 속도와 지금까지 완료된 run의 종료 epoch(7–18)를 기준으로 **학습 24회 완료는 11월 7일 17–19시 전후로 추정**한다. early stopping 시점, 자원 경합, 중간 평가 작업에 따라 달라지며 보장 시각은 아니다. **전체 평가·모델 비교표·통계의 완료 시각은 별개다.** 디코딩 오류 처리와 실패 평가 재등록이 아직 수행되지 않았으므로 현재 큐가 학습을 마쳐도 최종 비교가 자동 완성되지는 않는다. 앞서 안내한 1차 비교 예상 시각은 이 오류로 지연됐고, 전체 ETA는 수정·재평가 일정이 정해진 뒤 다시 계산해야 한다.

## 완료된 진단 결과

기존 VideoMAE checkpoint를 학습과 같은 MLP 헤드로 strict 로드한 FP32 외부 진단은 1,150영상 모두 완료됐다.

| 조건 | AUROC | AP | Recall | FPR | TN / FP / FN / TP |
|---|---:|---:|---:|---:|---|
| 고정 threshold 0.5 | 0.887733 | 0.565066 | 0.940000 | 0.365000 | 635 / 365 / 9 / 141 |
| 같은 외부 labels로 FPR ≤ 0.1 선택: 사후 진단 | 0.887733 | 0.565066 | 0.666667 | 0.100000 | 900 / 100 / 50 / 100 |

이는 기존 가중치 진단이며 새 그룹 분할 재학습 결과가 아니다. 시간 샘플링·전처리도 기존 평가와 달라졌으므로 개선 폭 전체를 헤드 수정의 독립 효과로 해석하지 않는다. 두 번째 행의 threshold는 source-dev에서 정한 값이 아니므로 운영 성능으로 사용하지 않는다. 기존 진단은 콘텐츠 감사 전 전체 UCF를 사용했고, 그 결과의 `UCF_candidate_quarantined` 항목도 실제 후보 제거 결과가 아니다. 자세한 원고 반영 메모는 [FOLLOW_UP_NOTES.md](FOLLOW_UP_NOTES.md)에 있다.

분류 benchmark 3개도 완료됐다. RTX A6000, BF16, batch 1에서 측정한 참고값이다.

| 모델 | forward p50/p95 (ms) | decode+전처리+forward p50/p95 (ms) |
|---|---:|---:|
| ViViT | 26.7 / 27.2 | 286.4 / 401.7 |
| TimeSformer | 85.3 / 85.4 | 353.1 / 397.9 |
| VideoMAE | 10.6 / 10.8 | 235.2 / 268.1 |

단일 CADP dev 영상의 한 시간창을 반복 측정한 분류 그래프 결과다. benchmark 함수는 pretrained+LoRA 모델을 만들며 학습 best checkpoint를 로드하지 않는다. YOLO·설명·동시 스트림은 포함되지 않아 full-pipeline/현장 처리량으로 보고하면 안 된다.

## 데이터 자동 감사 완료

- 검사 파일: 6,147개; 초기 부분 디코딩 검사를 통과한 파일 6,137개, 검사 제외 10개. 이후 평가에서 추가 디코딩 오류가 발견됐으므로 6,137개 전체의 모든 시간창이 유효하다는 뜻은 아니다.
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
