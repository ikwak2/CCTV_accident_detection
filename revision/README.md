# ORV revision experiments

논문 수정 전략과 리뷰 의견별 대응은 [PLAN.md](PLAN.md), 실제 실행 상태는 [STATUS.md](STATUS.md)를 참고한다. 나중에 코드·원고를 수정할 때 볼 확인 근거와 위치, 현재 평가 디코딩 오류의 후속 절차는 [FOLLOW_UP_NOTES.md](FOLLOW_UP_NOTES.md)에 기록한다.

2026-11-07 사용자 요청으로 디코딩 오류 수정·재평가를 진행한다. 수정 방식과 복구 절차는 [DECODING_RECOVERY.md](DECODING_RECOVERY.md)에 기록한다.

## 실행 범위

- **server1의 물리 GPU 0·1**에 worker 하나씩 배정한다. 컨테이너 안에서는 배정된 GPU만 `cuda:0`으로 보인다.
- 기존 `main/` 학습·평가 파일은 보존하고, 이 폴더의 공통 모델·데이터 계약으로 새 실험을 실행한다.
- 전체 자동 큐는 8설정 × 3seed = 24회 학습, 각 best checkpoint의 dev/test/external 평가, 세 모델 benchmark, 두 차례 집계와 주석 템플릿 생성을 포함한다.
- 중복 영상의 자동 후보는 보수적으로 묶거나 외부 평가에서 격리하고, 전체 UCF 결과도 함께 보존한다. 자동 검사가 완전한 사건·카메라 독립성을 입증하지는 않는다.
- road-only normal, onset, 사고 차량 box/ID는 사람의 주석이 필요하다. 미주석 지표는 계산했다고 표시하지 않는다.

## 현재 서버 경로

실행 루트는 `/disk1/orv_revision_20261106`이다. 원본 데이터는 `/disk1/data/hj24/CCTV`에서 읽기 전용으로 연결한다. `results/`와 Hugging Face cache는 서버에 보관하고 Git에는 넣지 않는다.

```bash
ssh server1
docker ps --filter name=orv-revision
cat /disk1/orv_revision_20261106/results/STATUS.md
cat /disk1/orv_revision_20261106/results/bootstrap_status.json
docker logs --tail 40 orv-revision-gpu0
docker logs --tail 40 orv-revision-gpu1
tail -n 40 /disk1/orv_revision_20261106/results/logs/JOB_ID.log
```

자동 데이터 감사와 교정된 기존 가중치 평가는 초기 별도 컨테이너로 시작된다. bootstrap은 필요한 결과 파일을 기다렸다가 큐를 등록한다. 두 worker는 이후 우선순위와 선행 조건에 따라 작업을 가져간다.

새 작업 시작을 일시 중지하려면 서버에서 아래 파일을 만든다. 실행 중인 학습은 현재 작업을 계속한다.

```bash
touch /disk1/orv_revision_20261106/results/PAUSE
# 새 작업 재개
rm /disk1/orv_revision_20261106/results/PAUSE
```

컨테이너를 중단하려면 `docker stop orv-revision-gpu0 orv-revision-gpu1`을 사용한다. 재시작 시 자신의 중단된 작업을 회수하고 마지막 완료 epoch에서 이어간다. 다른 worker의 실행 중 작업은 lease 시간이 오래됐다는 이유로 가져오지 않는다.

## 산출물

- `manifests/grouped.csv`: 서버 전용 그룹·split·유효 영상 명세
- `manifests/evaluation.csv`, `evaluation_audit.json`: 실제 5개 평가 시간창을 16/32프레임으로 검사한 공통 평가 목록과 원본/평가 manifest·decoder 해시. 학습용 grouped manifest는 유지한다.
- `manifests/evaluation_exclusions_private.json`: 시간창 디코딩 실패 영상과 제외 근거. 모든 모델·조건·seed에 같은 제외를 적용한다.
- `manifests/audit.json`: 집계 수와 자동 검사 한계
- `manifests/content_candidates_private.json`: 수동 검토할 중복/카메라 후보
- `preflight/preflight.json`: 모델·LoRA backward, legacy checkpoint, 전처리 검증
- `runs/RUN_ID/config.json`: seed, split SHA256, 실제 패키지·코드·모델 revision
- `runs/RUN_ID/best.pt`, `last.pt`, `history.jsonl`: best 및 재시작 checkpoint
- `runs/RUN_ID/{dev,test,external}/`: 영상 점수와 실제 샘플 시각
- `runs/RUN_ID/metrics.json`: source-dev threshold 및 기본 threshold에서의 결과
- `runs/RUN_ID/evaluation_contract.json`: checkpoint·학습/평가 manifest·decoder·precision 계약. 기존 점수를 재사용할 때 영상별 샘플 시각도 검사한다.
- `analysis/{baseline,all}/`: 집계·민감도·paired group bootstrap
- `annotations/`: 공통 설명 사례·도로 정상·onset·객체 box 주석 양식
- `queue_status.json`, `STATUS.md`, `logs/`: 재시작 가능한 큐와 작업별 로그

기존 가중치 진단은 FP32로 실행한다. 새 실험은 두 RTX A6000에서 BF16 autocast, 공통 4초 범위, architecture별 16/32프레임, effective batch 24를 사용한다. microbatch 기본값은 2이며 OOM이면 1로 낮추고 effective batch는 유지한다. 새 학습은 전체 후보에 동일한 비가중 cross entropy를 사용한다. CADP-only의 class prior 차이는 UTIC ablation 해석에서 함께 보고해야 한다.

## 검증

`test_queue.py`는 두 worker의 중복 claim 방지, 선행 작업 실패 시 차단, 자신의 중단 작업 복구를 검사한다. `test_evaluation_audit.py`는 마지막 시간창 실패의 공통 제외, 평가 목록 변경 감지, 기존 점수의 샘플 시각 검증, 실패 로그·가중치를 보존하는 큐 복구를 검사한다. GPU preflight는 학습될 LoRA B의 실제 gradient와 VideoMAE forward 보정의 수치 동등성을 확인한다.

```bash
python revision/test_queue.py
python -m unittest discover -s revision -p 'test_*.py' -v
python revision/run.py preflight --root /results --out /results/preflight
```

결과가 나오기 전에 모델 순위나 ORV 통과 여부를 정하지 않는다. 운영자 효용, 연속 경보 지연, full-pipeline 처리량, 추적 정확도는 아직 별도 검증 항목이다.
