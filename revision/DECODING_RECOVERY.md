# 평가 디코딩 오류 수정과 재실행

2026-11-07 사용자 요청: 디코딩 오류를 수정한 뒤 실패 평가를 다시 실행한다. 이전의 기록만 작성하라는 요청 이후 새로 승인된 실행 작업이다. 최신 완료 수와 실행 상태는 [STATUS.md](STATUS.md)를 본다.

## 원인 재현

- 기존 Docker image/PyAV 13.1.0 환경에서 문제 입력의 5개 평가 시간창을 16/32프레임으로 각각 재현했다.
- 길이 약 79.909초의 UTIC 정상 영상으로, 마지막 시간창 시작은 약 75.909초다. 두 프레임 수 모두 이 시간창에서 `av.error.InvalidDataError: [h264] no frame!`가 발생한다.
- 처음부터 연속 디코딩해도 같은 예외가 난다. packet별 진단에서 1,200프레임 이후 크기 56 bytes인 손상 패킷의 예외를 확인했다. 마지막 정상 프레임 PTS는 약 79.842초다.
- 파일 SHA256은 `e0b3d55169a72d78481f2ace037d12ffdd290b16ddec49651860c7e1fc4ed2ca`다. 비공개 파일명·카메라명은 공개하지 않는다.
- 초기에 사용한 부분 시점 프레임 검사가 실제 평가의 모든 시간창을 검증하지 못한 것이 반복 실패의 배경이다.

## 수정한 평가 계약

1. [audit_evaluation.py](audit_evaluation.py)는 dev/test/external 전체를 실제 평가와 같은 5개 4초 시간창, 16/32프레임으로 검사한다. 하나라도 예외가 나면 영상 전체를 모든 모델에 공통 제외한다. 모델 점수·성능으로 제외 대상을 선택하지 않는다.
2. 원본 `grouped.csv`와 학습 split은 유지한다. 별도의 `evaluation.csv`, 그 해시와 원본 manifest/decoder 해시를 가진 `evaluation_audit.json`을 만든다. 제외 명세와 검사 저널은 서버 전용이다. 이 정책은 엄격한 디코딩 예외 검사이며 모든 화질 결함을 판별하는 수동 영상 검토를 대체하지 않는다.
3. [run.py](run.py)는 완성된 공통 검사 결과가 없거나 manifest/decoder 해시가 다르면 평가를 중단한다. best checkpoint의 학습 manifest와 모델/LoRA/seed도 확인한다.
4. 기존 점수의 재사용은 동일 checkpoint·전처리/decoder 해시·precision을 전제로 한다. 영상별 라벨/그룹/출처와 5개 시간창의 샘플 시각을 검증하고, 최종 CSV와 지표에는 공통 유효 목록의 영상만 넣는다. 모델 입력·시간 샘플링·전처리 수식은 바꾸지 않았다.
5. metrics에 학습 및 평가 manifest 해시, 평가 전후 출처별 클래스 수를 함께 남긴다. dev가 변경되면 해당 유효 dev의 점수로 threshold를 다시 계산한다.
6. [analyze.py](analyze.py)는 서로 다른 평가 manifest를 섞거나 모델별 영상 집합의 교집합만으로 비교하지 않도록 검사한다.

## 배포·복구

- 기존 worker의 새 작업 claim을 `results/PAUSE`로 막고, 진행 중 학습은 유지한 채 CPU 전용 컨테이너에서 평가 시간창을 검사한다.
- 수정 코드는 새 commit의 읽기 전용 release로 배포한다. 원래 release, 원본 영상, 완료된 학습 가중치는 보존한다.
- old worker 두 개를 중단한 뒤 [recover_evaluations.py](recover_evaluations.py)를 실행한다. 이 도구는 queue DB와 이전 작업 상태/오류를 `results/recovery/`에 백업한 후 실패·대기 평가를 재등록한다. 가중치·예측 저널은 변경하지 않는다.
- 학습 도중 worker 교체가 필요한 경우 마지막 완료 epoch에서 재개한다. 완료된 학습은 재학습하지 않는다. 첫 seed baseline 평가 3건 뒤에 남은 학습을 우선 배치한다.
- 물리 GPU UUID로 고정한 server1 GPU 0·1의 worker를 새 release로 시작하고 PAUSE를 해제한다. 새 평가 결과를 확인한 뒤 상태 문서를 갱신한다.

Docker 내부에서 사용하는 명령:

```bash
python revision/audit_evaluation.py --root /results --workers 4 --executor process
# 위 검사 완료 및 기존 worker 중단 후
python revision/recover_evaluations.py --root /results --workers-stopped
```

## 검증

같은 Docker image의 CPU 환경에서 데이터/토큰 계약, 큐 동시 claim·복구, 공통 평가 제외·해시·기존 점수 검증, 실제 인코딩 영상의 spawn 프로세스 검사를 포함한 **9개 테스트가 통과**했다. 원래 temporal sampler와 전처리는 변경하지 않았다. 실제 서버 평가의 디코딩 오류 지점 통과와 최종 평가 완료 여부는 STATUS에 별도로 기록한다.

처음 8개 디코딩 스레드를 사용했으나 다중 스레드의 처리량이 단독 검사보다 크게 낮아지는 병목을 관측했다. 같은 입력 한 건의 단독 16/32프레임·5시간창 검사는 약 1.53초였다. 검사 함수를 바꾸지 않고 독립된 spawn 프로세스 4개로 격리하도록 변경했다. 이미 완료된 검사 저널의 동일 decoder/manifest 계약을 확인하고 이어간다. 실행 방식과 재사용 검사 수는 audit 결과에 저장한다.
