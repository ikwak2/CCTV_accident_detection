# 새 학습 모델의 시각화 비교 기록

2026-11-07 사용자 요청: 기존 원고에서는 ViViT이 좋았지만 새 분류 결과는 다른 모델이 더 높으므로, 다른 모델의 시각화도 다시 확인한다. 원고 자체는 수정하지 않았다.

**후속 확인:** 원고 §5.4 / Table 6의 비교 방법은 attention rollout, LayerCAM, Grad-CAM++다. 아래 gradient × input은 추가 진단이며 원고의 CAM 방법 재현과 다르다. 원고의 세 방법으로 확장한 20사례 비교와 TimeSformer의 입력/내부 프레임 설정 불일치는 [CAM_COMPARISON_20.md](CAM_COMPARISON_20.md)에 기록한다. 아래 TimeSformer 관련 해석은 이 새 확인 사항과 함께 읽어야 한다.

## 확인한 결과

**VideoMAE의 분류 성능이 높다는 결과와 설명 맵의 품질은 구분해야 한다.** 완료된 CADP+UTIC 주 실험의 3-seed 평균 UCF AUROC는 ViViT 0.74056, TimeSformer 0.81182, VideoMAE 0.89478이다. 이 수치는 공간적 사고 위치 설명의 우열을 측정하지 않는다. 분류 수치의 조건·외부 FPR 한계는 [INTERIM_RESULTS.md](INTERIM_RESULTS.md)에 있다.

새 grouped 학습의 seed 42 best checkpoint로 이미 계산한 공통 60사례 진단을 확인했다. 세 모델 모두 60/60사례에서 유한한 양의 gradient × input 맵을 생성한다. 이번에는 같은 RGB 프레임을 별도로 추출해 첫 세 사례의 비교 그림을 다시 렌더링했다. 추가 학습이나 평가 worker 중단은 하지 않았다.

| 모델 | 시각화 checkpoint 설정 | 공통 gradient × input | architecture별 attention rollout |
|---|---|---|---|
| ViViT | Q/K/V/output LoRA, mixed, seed 42 | 60사례 완료 | 첫 3사례, CLS readout |
| TimeSformer | QKV/output LoRA, mixed, seed 42 | 60사례 완료 | 미평가: divided attention용 전파 규칙 검증 필요 |
| VideoMAE | output-only LoRA, mixed, seed 42 | 60사례 완료 | 첫 3사례, CLS 없는 mean readout |

세 backbone의 LoRA 설정과 입력 프레임 수가 다르다. 이것은 각 주 실험 설정의 결과 비교이며, backbone만 바꾼 완전 통제 실험은 아니다. 남은 LoRA 교차 통제 평가는 기존 큐에서 계속 진행한다.

## 기존에 다른 모델의 그림이 잘 나오지 않았던 이유

- 기존 `main/cam/rollout/rollout_mae.py`는 `rollout[:, 0, 1:]`를 CLS→patch attention으로 해석했다. 현재 사용하는 VideoMAE는 CLS 없이 mean pooling한다. 수정 구현은 16프레임 / tubelet 2 / 공간 14×14의 1,568토큰을 명시하고 mean readout을 사용한다. 예전 token 수를 임의로 인수분해해 얻은 그림을 유효한 공간 맵으로 해석하면 안 된다.
- 기존 VideoMAE의 외부 분류 loader와 학습 MLP head가 달랐던 문제는 별도의 분류 오류다. 그 수정이 attention의 타당성을 자동으로 보장하지 않는다. 자세한 근거는 [FOLLOW_UP_NOTES.md](FOLLOW_UP_NOTES.md)에 있다.
- TimeSformer의 spatial/temporal divided attention은 ViViT의 전체 token attention과 구조가 다르다. 공통 pixel-space 진단은 생성했지만, 아직 검증하지 않은 rollout을 성공한 그림으로 표시하지 않았다.
- 이번 공통 비교 그림은 **positive gradient × normalized input**이다. Grad-CAM/Grad-CAM++ 결과라고 부르면 안 된다. 두 방법의 target layer·채널 가중·baseline이 다르다.

## 그림에서 관찰한 내용과 한계

고정 사례 목록의 **첫 3개**를 사용했다. 모두 양성 UCF 사례이며, 예쁜 맵이 나온 순서로 고르지 않았다. 기존 60사례 목록 자체는 모델 오류 유형 등을 반영한 진단용 선택이므로 모집단의 무작위 표본도 아니다.

- Case C의 공통 gradient × input에서는 ViViT이 차량 바퀴 부근에 반응하는 프레임이 있다. TimeSformer는 노면의 일부에, VideoMAE는 차량 위쪽·노면 등에도 반응한다. 이를 사고 당사 차량의 정답 위치라고 확인한 것은 아니다.
- Case B에서는 VideoMAE가 화면 하단 자막·가장자리 부근에 반응하는 모습이 보인다. 수정한 rollout에서도 배경 반응이 함께 보인다. 데이터 출처·자막 단서에 의존할 가능성을 후속 검증할 근거이지, 이 사례만으로 shortcut을 확정하지 않는다.
- 세 모델 모두 높은 사고 확률을 내는 중앙 구간에 실제 충돌 시작이 들어 있는지는 확인되지 않았다. 일부는 충돌 이후 장면으로 보인다. 중앙 4초의 설명과 전체 영상의 5-window 평가 점수를 혼동하지 않는다.
- 색은 각 모델 내부의 상대 강도다. 색이 더 진하다고 모델 간 절대 기여가 더 크거나 설명이 더 정확하다는 뜻은 아니다. 사고 onset·객체 box가 없으므로 localization hit rate와 설명 품질 순위는 계산하지 않는다.

공통 60사례의 `mean(random-masked probability - top-attribution-masked probability)`는 ViViT 0.13165, TimeSformer 0.02235, VideoMAE 0.02191이다. 상위 20% spatiotemporal patch를 정규화 공간의 0으로 가린 진단이다. 모델별 확률 포화·calibration, 가림에 따른 분포 변화, 미주석 시간창, 사례 선택의 영향을 받는다. CI 없는 이 값으로 설명 우열을 확정하지 않는다. 세부 집계는 로컬 `diagnostic_summary.json`에 저장했다.

## 표시 방법과 재현

관련 코드:

- `explanations.py`, `explain.py`: 새 checkpoint의 원래 설명 추론과 측정값.
- `visualization_frames.py`: 데이터에 읽기 전용 접근, CPU로 공통 RGB 프레임 추출.
- `render_visualizations.py`: 저장된 원래 맵을 사용해 사례당 공통 gradient × input 비교와 rollout 비교를 PNG/PDF로 렌더링하고 HTML 갤러리를 만든다.

입력과 같은 short-side 256 resize / center crop 224를 사용한다. 네 표시 시각의 RGB는 공통으로 추출한다. ViViT·TimeSformer는 32프레임, VideoMAE는 16프레임이므로 **맵의 시간축만 선형 보간**한다. 이는 표시용 처리이며 분류·perturbation 지표에 사용하지 않는다. 시각은 요청 시각이고, decoder는 해당 시각 이전의 마지막 이용 가능한 프레임을 선택한다.

Gradient × input은 모델별 전체 4초 맵의 최대값으로 나눠 0–1로 표시한다. Rollout은 각 프레임의 공간 평균으로 나눠 공통 0–3 범위로 표시한다. 1은 공간적으로 균일한 attention, 3 초과는 표시 상한이다. 시간·공간 임의 인수분해나 프레임마다 min/max를 늘리는 처리를 하지 않는다. Rollout은 class-specific attribution과 다르다.

서버 image의 Python에서 `/app/revision`을 `PYTHONPATH`에 포함하고 다음 CPU 추출을 실행한다. 원본 영상 경로는 서버의 기존 private manifest에서만 읽는다.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=/app/revision python visualization_frames.py \
  --root /results --output /results/visual_comparison_20261107/frames --count 3

# explanations/와 frames/가 들어 있는 private 디렉터리를 로컬로 복사한 다음
MPLCONFIGDIR=/tmp/orv-mplconfig python3 revision/render_visualizations.py \
  --source /path/to/private --output /path/to/figures
```

렌더러는 입력 이미지 shape/dtype, 맵의 유한성·비음수, 시간 범위, 모델별 사례 ID 일치를 검사한다. `render_manifest.json`에 입력 파일·렌더러 SHA256과 numpy/matplotlib 버전, 정규화·보간 규칙을 기록한다. 렌더링한 6장과 각 PDF, HTML 링크를 확인했다.

서버 RGB 산출물은 `/disk1/orv_revision_20261106/results/visual_comparison_20261107/frames`, 원래 설명 결과는 `results/explanations/`에 있다. 로컬 그림은 이 작업의 `outputs/ORV_visualization_comparison_20261107/`에 저장했다. 영상 프레임·전체 경로·식별 명세는 공개 GitHub에 추가하지 않고 재현 코드와 이 기록만 추가한다.

## 원고에 반영할 전략

현재는 “분류 순위가 바뀌었으므로 설명도 VideoMAE가 우수하다”는 서술을 피한다. ViViT만 잘 보였다는 기존 설명은 코드 오류 가능성과 target/method 차이를 함께 다시 검토한다. 새 결과는 구현을 수정한 뒤의 비교 진단으로 제시할 수 있다.

설명 성능을 원고의 정량 주장으로 삼으려면, 모델 맵을 보지 않고 사고 onset·차량 box/ID와 road-only 정상·FP 사례를 먼저 주석한다. 모든 모델에 같은 사건 전/중/후 시간창과 accident-vs-normal target을 적용하고 localization, perturbation/random control, 설명 계산 시간과 실패율을 함께 보고한다. Grad-CAM 계열을 추가할 때도 backbone별 token grid와 실제 gradient가 흐르는 target layer를 검증한 다음 비교한다. 검증하지 않은 TimeSformer rollout의 빈칸을 임의 맵으로 채우지 않는다.
