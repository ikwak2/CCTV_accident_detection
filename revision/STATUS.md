# ORV revision 실행 상태

확인 시각: **2026-11-07T21:34:16+09:00** (Asia/Seoul). 이 문서는 해당 시각의 GitHub 스냅샷이다. 서버의 `/disk1/orv_revision_20261106/results/STATUS.md`, `queue_status.json`, 각 run의 `progress.json`이 최신 상태를 기록한다.

사용자가 디코딩 오류 수정 후 재실행을 요청해 **수정·검증·배포와 평가 재개를 완료**했다. 수정 근거와 재현 방법은 [DECODING_RECOVERY.md](DECODING_RECOVERY.md), 기존 원고 수정 메모는 [FOLLOW_UP_NOTES.md](FOLLOW_UP_NOTES.md)를 참고한다. 전체 실험이 완료됐다는 뜻은 아니다.

## 현재 실행

| server1 물리 GPU | 실행 내용 | 확인된 진행 |
|---|---|---|
| GPU 0 | ViViT output-only, mixed, seed 42 평가 | dev 737/737 완료; test 진행 표시 1/342 |
| GPU 1 | VideoMAE Q/K/V/output, mixed, seed 44 평가 | dev 737/737, test 342/342 완료; UCF 진행 표시 661/1150 |

디코딩 수정 후 실패는 계속 **0건**이다. 위 수는 `progress.json`의 표시이며 20개 간격으로 갱신돼 점수 저널의 실제 건수보다 적을 수 있다. 남은 평가는 현재 실행 2건과 ViViT output-only seed 43/44 대기 2건이다.

| 큐 상태 | 작업 수 |
|---|---:|
| pending | 3 |
| running | 2 |
| external_running | 0 |
| completed | 57 |
| failed | 0 |

총 62작업 중 **57개 완료, 2개 실행, 3개 대기**다. 학습 24회는 전부 완료했고, dev/test/UCF 평가 24건은 20건 완료·2건 실행·2건 대기다. 분류 benchmark 3개, 첫 baseline 통계, 주석 템플릿 생성, 자동 설명 진단 3개도 완료했다. 마지막 평가는 ViViT output-only seed 43/44이며, 모든 평가 뒤 `stats_all`이 전체 집계를 수행한다. 복구 전 디코딩 실패 18건의 상태·오류와 기존 큐 DB는 보존했다.

| 학습 설정 | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| ViViT Q/K/V/output, CADP+UTIC | 완료 | 완료 | 완료 |
| TimeSformer QKV/output, CADP+UTIC | 완료 | 완료 | 완료 |
| VideoMAE output-only, CADP+UTIC | 완료 | 완료 | 완료 |
| ViViT Q/K/V/output, CADP-only | 완료 | 완료 | 완료 |
| TimeSformer QKV/output, CADP-only | 완료 | 완료 | 완료 |
| VideoMAE output-only, CADP-only | 완료 | 완료 | 완료 |
| ViViT output-only, CADP+UTIC | 완료 | 완료 | 완료 |
| VideoMAE Q/K/V/output, CADP+UTIC | 완료 | 완료 | 완료 |

학습은 **24/24회 완료**했으며 학습 실패는 0회다. 새 grouped 주 실험과 CADP-only ablation의 18개 평가를 모두 마쳤다. LoRA 교차 통제 6개 평가 중 VideoMAE seed 42/43은 완료했고 나머지 4개 평가를 진행한다. 완료된 학습을 다시 학습하지 않는다.

## 디코딩 오류 처리와 공통 평가 집합

- 동일 UTIC 정상 영상이 마지막 4초 시간창에서 16/32프레임 모두 실패했고, 연속 디코딩에서도 손상 H.264 패킷 예외를 재현했다.
- dev/test/UCF **2,230영상 전체**에 실제 평가와 같은 5개 4초 시간창을 16/32프레임으로 검사했다. 실패는 해당 UTIC 영상 1개뿐이었다.
- 모든 모델·조건·seed에 동일한 공통 평가 목록을 적용한다. 모델 점수로 제외 영상을 선택하지 않았으며, 다른 영상이나 임의 프레임으로 대체하지 않았다.
- 학습용 `grouped.csv`와 원본 영상은 유지한다. 별도 `evaluation.csv`의 해시, 평가 전후 출처별 클래스 수와 제외 사유를 저장한다.

| 평가 split | 검사 전 | 검사 후 | 양성 / 정상 | 검사 후 그룹 수 |
|---|---:|---:|---:|---:|
| dev | 737 | 737 | 486 / 251 | 450 |
| test | 343 | 342 | 220 / 122 | 224 |
| UCF external | 1150 | 1150 | 150 / 1000 | 926 |

평가 manifest SHA256: `f16ba14732608dcd43641d21c730833b5c4227962d1f68e86f5264565229c5a9`. test에서 UTIC 정상만 109→108개로 바뀌었고 CADP 양성 220/정상 14는 같다. UCF 후보 격리 후의 양성 143/정상 995도 유지한다. 초기 부분 프레임 검사 통과와 이번 실제 평가 시간창 검사를 구분한다.

기존 18개 평가의 **18,198개 영상 점수**는 같은 decoder/전처리 해시, precision, 출처·라벨·그룹과 샘플 시각 검증을 통과했다. 이 점수는 재사용하고 남은 입력부터 계산한다. run별 checkpoint·학습/평가 manifest·decoder 계약을 저장하며, 집계 시 서로 다른 평가 집합을 섞지 않는다.

스레드 병렬 검사에서 처리량 저하를 관측해 독립 spawn 프로세스 4개로 전환했다. 이미 검사한 1,637개를 재사용하고 남은 593개를 약 222.5초에 검사했다. 검사 함수와 영상 샘플링·전처리 수식은 변경하지 않았다. 실제 인코딩 영상의 spawn 검사와 큐·공통 제외·기존 점수 검증을 포함한 **9개 테스트가 통과**했다.

## 속도와 완료 전망

현재 실행 중인 평가의 진행과 ViViT output-only dev 737개 처리에 약 16분이 걸린 점을 기준으로, 남은 평가·전체 통계는 **11월 7일 23시 전후(대략 22:30–23:30) 완료를 추정**한다. 영상별 비용과 후속 집계 시간에 따라 달라지며 보장 시각은 아니다. 사람의 도로 정상·onset·객체 box/ID 주석은 이 자동 큐 완료에 포함되지 않는다.

worker는 `unless-stopped`로 유지되고 재시작 시 점수 저널에서 이어간다. 현재 GPU 0·1은 남은 LoRA 교차 통제 평가를 처리한다. 완료된 주 실험의 3-seed 중간 집계는 [INTERIM_RESULTS.md](INTERIM_RESULTS.md)에 기록했다. 전체 통계 집계 전 자료임을 구분한다.

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

- 기존 학습 코드 commit은 [`71a25a6`](https://github.com/ikwak2/CCTV_accident_detection/commit/71a25a64c3c17645ee372edf3e650ebbfb97811b), 수정 평가·재개 학습의 실행 코드는 [`e534642`](https://github.com/ikwak2/CCTV_accident_detection/commit/e5346421158fbfbc8cf3ff9e84f9a7f77eed33ec)다. 새 worker는 `/disk1/orv_revision_20261106/releases/e534642`를 읽기 전용으로 사용한다. sampler·전처리·모델 구성의 `common.py` 해시는 같다.
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
