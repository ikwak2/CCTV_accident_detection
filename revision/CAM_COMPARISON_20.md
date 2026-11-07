# 원고의 세 설명 방법: 사고 10개 · 도로 비사고 10개

2026-11-07 사용자 요청으로 **attention rollout, LayerCAM, Grad-CAM++**를 새 grouped 학습 seed 42의 세 모델에 적용했다. 사용자 후속 요청에 따라 파생 비교 이미지와 묶음 PDF를 [viz/](viz/README.md)에 공개 Git 산출물로 추가한다. 원본 영상·전체 경로·영상 ID 목록·NPZ 배열은 서버와 로컬 private 결과 폴더에 보존한다. 원고 자체는 수정하지 않았다.

## 원고와 방법의 관계

첨부 원고 §2.4, §5.4와 Table 6의 비교 방법은 **attention rollout / LayerCAM / Grad-CAM++**다. 앞선 첫 세 사례의 gradient × input은 이번 revision에서 추가한 공통 진단이며 원고에서 사용한 방법이 아니다.

이번 묶음은 원고의 세 방법을 **새 checkpoint와 명시적인 토큰 배치·target layer**에 적용한 후속 비교다. 기존 원고의 checkpoint·잘못된 loader·임의 token factorization을 그대로 재실행한 결과라고 부르지 않는다.

## 공개 산출물

- `viz/A01_comparison.png`–`A10_comparison.png`: 추가 사고 10사례.
- `viz/N01_comparison.png`–`N10_comparison.png`: 도로 비사고 10사례.
- `viz/all_20_comparisons.pdf`: 전체 비교표 20쪽.
- `viz/accident_10_details.pdf`, `viz/nonaccident_10_details.pdf`: 클래스별 40쪽. 사례당 세 모델 × 세 방법 비교표 1쪽과 방법별 네 시점 비교 3쪽이다.
- `viz/index.html`: 폴더를 내려받아 여는 오프라인 갤러리. 상세 PDF의 사례 시작 페이지를 연결했다.
- `viz/diagnostics.csv`, `summary.json`, `artifact_manifest.json`: generic case ID별 원래 신호 크기·생성 상태·구간 확률과 파일 해시. 원본 영상 ID·경로는 포함하지 않는다.

PDF를 묶을 때 동일 객체를 중복 제거했으며 페이지를 다시 rasterize하거나 색·모델 맵을 수정하지 않았다. 개별 PNG도 저장한 비교 그림을 그대로 복사했다.

## 표본 선정

사고는 기존 고정 60사례 목록에서 이전에 보여준 0–2번, 자동 격리 후보, 반복 그룹을 제외한 추가 양성 10개다. 모두 UCF이며 중앙 4초를 사용한다. 충돌 시작을 사람이 주석한 표본이 아니므로 해당 구간이 충돌 전·후 장면일 수 있다.

처음에는 같은 목록의 음성 10개도 계산했지만 원본 화면에서 실내 장면이 많음을 확인했다. 이 진단 결과도 `cam_comparison_20261107`에 별도 보존했다. 교통 CCTV 비교를 위해 UCF 정상 영상에서 ID 순서로 첫 80개 적격 독립 그룹의 중앙 **원본 화면만** 추가 검사했다. 차량이 있는 도로·교차로가 분명한 10개를 선택했다. 선별 후보 번호는 2, 12, 16, 17, 26, 29, 31, 32, 52, 53이다. 실내, 주유소·주차장, 도로 여부가 불분명한 장면을 제외했으며 모델 점수·맵 모양을 선택 기준으로 사용하지 않았다.

이 도로 선별은 **Codex의 중앙 한 시점 예비 시각 검사**다. 사람이 전체 영상을 검증한 road-only 정상 평가집합이나 사고 부재의 새 ground truth로 취급하지 않는다. 비사고 라벨은 UCF 라벨을 사용한다. 선택 근거·후보 CSV·원본 화면 sheet·선택 manifest 해시는 private 결과에 보존했다. 새 20사례는 자동 그룹 기준으로 서로 다른 20그룹이다.

## 계산과 표시

| 모델 | checkpoint 설정 | 입력 프레임 | CAM 대상 activation shape |
|---|---|---:|---|
| ViViT | Q/K/V/output LoRA, mixed, seed 42 | 32 | `[1,3137,768]`, CLS 제외 후 16×14×14 tubelets |
| TimeSformer | QKV/output LoRA, mixed, seed 42 | 32 | `[8,785,768]`, 아래 설정 불일치 참고 |
| VideoMAE | output-only LoRA, mixed, seed 42 | 16 | `[1,1568,768]`, CLS 없이 8×14×14 tubelets |

모두 best checkpoint를 strict 로드하고 FP32/eval mode로 계산했다. CAM 대상 층은 마지막 encoder block의 `layernorm_before`다. ViViT·VideoMAE는 joint attention 전, TimeSformer는 spatial attention 전의 정규화 출력이다. 마지막 CLS 분류 출력 이후의 patch token을 CAM 대상으로 삼으면 patch gradient가 0일 수 있으므로 그 위치와 구분했다.

LayerCAM·Grad-CAM++의 target은 **양성·음성 모두 사고 logit (class 1)**으로 고정했다. 비사고 맵이 비사고 판정의 근거라는 뜻은 아니다. 예측 클래스가 바뀔 때 시각화 target까지 달라지는 비교를 피하고 사고 출력에 대한 근거를 본다. 원래 signed activation을 유지하고 계산 마지막에 ReLU를 적용한다.

- LayerCAM: 위치별 positive gradient와 activation을 곱해 채널 합산한다.
- Grad-CAM++: pytorch-grad-cam의 first-gradient power approximation과 `eps=1e-6`를 사용한다. 채널 가중 계산의 축은 시간·공간 전체 `T,H,W`다. 진짜 2·3차 autograd 미분을 직접 계산한 구현은 아니다.
- Rollout: 모든 층의 head mean에 residual identity를 더해 행 정규화한다. ViViT은 CLS, VideoMAE는 mean readout이다. 읽을 행만 역순으로 전파하는 구현은 작은 full rollout 행렬과 수치적으로 일치한다. Raw attention은 class-specific attribution이 아니다.
- TimeSformer rollout: spatial attention만으로 joint grid를 꾸미지 않는다. temporal attention도 hook으로 수집하고 temporal → spatial 연산, frame-mean CLS, 각 residual 경로를 명시적으로 반영한 구조적 attention-flow 확장이다. value/output projection, MLP와 채널 효과는 다른 raw rollout과 마찬가지로 포함하지 않는다.

공통 RGB는 입력과 같은 resize-short-side-256 / center-crop-224로 추출했다. 같은 네 요청 시각을 사용하고 16/32프레임 맵의 시간축만 표시용 선형 보간한다. PNG 비교표는 미리 정한 세 번째 표시 시각을 사용한다. 사례를 보고 가장 잘 나온 프레임을 고르지 않았다.

CAM은 각 모델·방법의 전체 4초 최대값으로 나눠 0–1로 표시한다. Rollout은 각 프레임 공간 평균으로 나누고 0–3 범위를 공통 사용한다. 1은 공간적으로 균일한 attention이다. 각 방법의 색은 서로 같은 양을 측정하지 않는다. percentile stretch, 시간 smoothing, 임의 fallback layer, zero-map 대체를 하지 않았다.

계산 수식의 기준은 [pytorch-grad-cam LayerCAM 구현](https://github.com/jacobgil/pytorch-grad-cam/blob/master/pytorch_grad_cam/layer_cam.py), [Grad-CAM++ 구현](https://github.com/jacobgil/pytorch-grad-cam/blob/master/pytorch_grad_cam/grad_cam_plusplus.py)이다. TimeSformer 실제 동작은 서버에 설치된 `transformers 4.46.3` 소스를 직접 확인했다.

## 확인 결과와 해석 한계

| 모델 | Attention rollout | LayerCAM | Grad-CAM++ |
|---|---:|---:|---:|
| ViViT | 20/20 양의 신호 | 20/20 양의 신호 | 20/20 양의 신호 |
| TimeSformer | 20/20 양의 신호 | 20/20 양의 신호 | 20/20 양의 신호 |
| VideoMAE | 20/20 양의 신호 | 20/20 양의 신호 | 20/20 양의 신호 |

**180개 맵 모두 유한·비음수·양의 신호를 생성**했다. 실제 대상 층 gradient의 사례별 최대 절댓값은 세 모델 모두 0보다 크다. attention 출력 수집 전·후의 사고 확률 차이가 `2e-4` 이하인지도 각 추론에서 확인했다. 5개 수치 검증은 명시적 token 배치, zero-gradient 처리, joint rollout 동등성, divided flow와 작은 dense transition의 동등성, 내부/입력 시간 grouping 역변환을 검사한다.

맵이 계산됐다는 것은 맵이 사고 객체를 잘 설명하거나 운영자가 사용할 수 있다는 뜻이 아니다. LayerCAM은 일부 장면에서 국소 신호만 보이고 Grad-CAM++는 배경에도 반응한다. 원고의 “Grad-CAM++ 구조적 비호환”은 model 전체에 대한 결론으로 유지하기 어렵다. target 층·구현·실패율을 지정한 경험적 관찰로 다시 써야 한다. 사고 onset·차량 box/ID, perturbation/random control, 사람의 평정 없이 localization 정확도나 설명 품질 순위를 선언하지 않는다.

## 새로 발견한 TimeSformer 프레임 설정 문제 — 후속 우선순위

입력은 32프레임이지만 실제 checkpoint의 `config.num_frames`는 **8**이었다. HF embedding은 실제 입력을 196개 공간 patch × 32프레임 순서로 만든다. encoder는 내부 설정 8을 사용해 784개 가상 공간 위치 × 8프레임으로 재해석한다. 실제 마지막 spatial LayerNorm shape가 `[8,785,768]`인 것이 확인 근거다. 올바른 32프레임 구성이라면 `[32,197,768]`이 기대된다.

기존 가중치의 출력을 보존하기 위해 이번 그림은 **내부 설정을 바꾸지 않았다.** 8×784의 내부 grouping을 원래 입력의 32×196 순서로 역변환한 뒤 표시한다. 단순히 785토큰을 임의로 14×14에 맞추지 않았다. Rollout도 내부 8프레임의 spatial/temporal 전파를 계산한 뒤 동일한 역변환을 사용한다. 첫 두 실패의 오류·코드·contract를 private attempt 폴더에 보존하고 매핑과 hook 호출을 수정해 완료했다.

`revision/common.py::build_model`과 원래 `main/models/TimeSformer.py`, `main/utils/tformer_general.py`에는 입력 길이에 맞춰 내부 frame 설정을 명시적으로 동기화하는 처리가 없다. 원래 학습 환경의 모델 config/version도 재확인해야 한다. 현재 분류 점수는 현재 가중치의 실측값으로 보존하지만, 구조 자체의 성능·시각화 한계로 단정하지 않는다. [INTERIM_RESULTS.md](INTERIM_RESULTS.md)에도 이 해석 보류를 표시했다.

후속 실험의 우선순위는 다음과 같다. 이 단계는 **계획 기록**이며 기존 checkpoint나 진행 중인 분류 평가를 임의로 바꾸지 않았다.

1. TimeSformer 입력/encoder 설정을 32로 동기화하는 별도 구현에서 token shape, pretrained temporal-position 처리, strict 저장/복원과 LoRA gradient를 검증한다. 기존 checkpoint의 config만 바꾼 사후 추론을 수정 재학습 결과라고 부르지 않는다.
2. 새 run ID로 mixed/CADP-only × seed 42/43/44를 동일 manifest·평가 계약에서 재학습·재평가한다. 현재 결과를 덮어쓰지 않는다.
3. 수정 설정의 checkpoint로 같은 설명 사례와 세 방법을 다시 계산한다. 모델 간 사전학습·LoRA 차이도 함께 통제한다.
4. 도로 정상·사고 onset·객체 box를 사람이 검토한 뒤 설명의 정량 지표와 ORV Criterion 3 판정을 다시 구성한다.

## 실행 위치와 재현 코드

- `cam_methods.py`: 모델별 token volume, LayerCAM·Grad-CAM++, joint/divided rollout과 grouping 복원.
- `cam_comparison.py`: private 선택·RGB 추출, strict checkpoint 추론, 원래 값과 신호 진단.
- `cam_queue_runner.py`: 등록된 CAM 작업만 원자적으로 실행, GPU 0·1 UUID 허용 목록과 프로세스별 GPU 메모리 40% 제한.
- `screen_road_normals.py`: 모델 점수·맵을 사용하지 않는 원본 화면 후보 sheet.
- `test_cam_methods.py`: 위 5개 수치 검증.
- `render_cam_comparison.py`: 개별 20 PNG, 4쪽 사례별 PDF와 갤러리.
- `package_cam_viz.py`: 요청한 공개 파생 그림과 20/40/40쪽 PDF를 `viz/`에 묶고 원본 ID·경로·배열은 제외.

서버 주 산출물은 `/disk1/orv_revision_20261106/results/cam_road20_20261107`, 입력 후보 sheet는 `cam_road_screen_20261107`, 처음 실내 음성을 포함한 진단은 `cam_comparison_20261107`이다. 모델 코드는 기존 읽기 전용 `e534642` release, 추가 CAM 코드는 결과 디렉터리의 별도 snapshot을 사용했다. Docker는 네트워크 없이 원본 데이터에 읽기 전용 접근하며 server1의 물리 GPU 0·1만 사용했다. 기존 분류 평가를 중단하지 않았다. 일회성 CAM 컨테이너는 `restart=no`로 종료한다.

로컬 주 산출물은 이 작업의 `outputs/ORV_CAM_ROAD20_20261107/`이다. 공개 파일은 사용자의 후속 게시 요청에 따라 `revision/viz/`에 복사했다. private 선택·모델 계약·원래 맵·렌더러·최종 파일 해시를 각각 보존한다.
