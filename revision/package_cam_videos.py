"""Publish derived MP4 comparisons and timing notes under revision/viz/videos/."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(source, repository):
    viz = repository / 'revision/viz'
    destination = viz / 'videos'
    manifest = json.loads((source / 'video_manifest.json').read_text())
    records = manifest['cases']
    if len(records) != 20 or sum(r['label'] for r in records) != 10:
        raise ValueError('Expected ten accident and ten non-accident cases')
    names = []
    for record in records:
        name = record['case'] + '.mp4'
        if digest(source / name) != record['sha256'] or record['codec'] != 'h264' or record['fps'] != 24:
            raise ValueError('Video does not match validated encoding record')
        names.append(name)
    for key in ['accident_10', 'nonaccident_10']:
        name = key + '.mp4'
        if digest(source / name) != manifest['compilations'][key]['sha256']:
            raise ValueError('Compilation checksum mismatch')
        names.append(name)
    destination.mkdir(parents=True, exist_ok=True)
    for name in names + ['video_manifest.json']:
        shutil.copy2(source / name, destination / name)
    # revision/.gitignore excludes raw videos. Whitelist only these explicitly
    # requested derived comparison assets in this one output directory.
    (destination / '.gitignore').write_text('# Requested derived MP4 comparisons only.\n' +
                                            ''.join('!' + name + '\n' for name in names))
    old_times = {}
    with (viz / 'diagnostics.csv').open() as stream:
        for row in csv.DictReader(stream):
            old_times[row['case']] = float(row['display_time'])
    with (destination / 'timing.csv').open('w', newline='') as stream:
        columns = ['case', 'label', 'existing_image_time', 'window_start', 'last_video_time', 'frames', 'fps', 'playback_seconds']
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for r in records:
            writer.writerow({'case': r['case'], 'label': r['label'], 'existing_image_time': old_times[r['case']],
                             'window_start': r['input_window_start'], 'last_video_time': r['last_requested_video_time'],
                             'frames': r['frames'], 'fps': r['fps'], 'playback_seconds': r['duration']})
    readme = '''# MP4 비교 영상 — 사고 10개 · 도로 비사고 10개

- [사고 10개 이어 보기](accident_10.mp4)
- [도로 비사고 10개 이어 보기](nonaccident_10.mp4)
- [브라우저용 갤러리](index.html): 폴더를 내려받아 연다.
- [시점 CSV](timing.csv), [파일 해시·프레임 수·입력 계약](video_manifest.json)

사례별 약 4초 MP4 20개와 클래스별 약 40초 묶음 2개다. H.264 / yuv420p / 24 FPS / 1024×1168 / 무음이며 faststart를 적용했다. 묶음은 개별 MP4의 packet을 재인코딩 없이 연결했다.

**어느 시점인가?** 기존 정지 그림은 영상 중앙 약 4초 구간의 네 표시 시점 중 세 번째, 즉 구간 약 2/3 시점이었다. 시작 프레임이 아니다. 예를 들어 A01의 기존 그림은 8.44초이고 모델 입력 구간은 5.80–9.77초다. 이 MP4는 해당 입력 구간 시작부터 재생한다. 추가 정중앙 정지 이미지는 게시하지 않았다.

열은 ViViT / TimeSformer / VideoMAE, 행은 원본 / attention rollout / LayerCAM / Grad-CAM++다. 상단에 원본 영상의 요청 시각과 구간 내 경과 시간을 표시한다. 원본 RGB는 각 요청 시각 이전의 마지막 이용 가능한 프레임을 사용한다.

**새 실시간 추론 결과는 아니다.** 기존 동일 checkpoint·동일 중앙 구간에서 측정한 16/32개 시점의 맵을 영상 시각에 선형 보간했다. 상단 사고 확률은 그 4초 구간의 값으로 고정되어 있다. CAM target은 사고·비사고 모두 사고 logit이다. Rollout은 class-specific하지 않다. 맵 범위 밖 시각에는 외삽하지 않는다.

기존 그림과 같은 crop, 상대 강도 규칙을 사용했다. 색은 turbo, RGB 62% + heatmap 38%다. 원래 영상 소리나 전체 길이 영상을 포함하지 않는다. 사고 시작 시각·객체 box를 주석한 데이터는 아니므로 충돌 위치나 설명 정확도를 검증한 영상으로 해석하지 않는다.

TimeSformer 입력 32 / 내부 설정 8프레임 불일치는 기존 checkpoint 동작을 보존했다. 자세한 한계는 [CAM_COMPARISON_20.md](../../CAM_COMPARISON_20.md)에 있다. 원본 영상 ID·경로·NPZ는 공개하지 않고 파생 비교 MP4와 generic case별 시점만 저장했다.

| 사고 | 도로 비사고 |
|---|---|
'''
    for i in range(1, 11):
        readme += f'| [A{i:02d}.mp4](A{i:02d}.mp4) | [N{i:02d}.mp4](N{i:02d}.mp4) |\n'
    shorter = [f"{r['case']} {r['duration']:.2f}초" for r in records if r['duration'] < 3.9]
    if shorter:
        readme += '\n원본 구간이 짧아 재생 길이가 더 짧은 사례: ' + ', '.join(shorter) + '.\n'
    (destination / 'README.md').write_text(readme)
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>ORV MP4 비교</title><style>body{max-width:1100px;margin:30px auto;padding:0 18px;font:16px/1.7 sans-serif;color:#17212b}video{display:block;max-width:100%;max-height:85vh;background:#111;margin:15px auto}article{border-top:1px solid #ccd5df;margin-top:35px;padding-top:15px}a{color:#175fad}</style>
<h1>사고 10개 · 도로 비사고 10개 MP4 비교</h1>
<p>모델 입력에 사용한 중앙 약 4초 구간을 원본·세 모델·세 방법으로 비교한다. 기존 맵의 시간축 보간이며 새 실시간 추론이 아니다.
영상 상단 확률은 구간 값으로 고정되어 있고 원본 영상 시각은 계속 바뀐다. TimeSformer 입력 32 / 내부 설정 8프레임 불일치를 보존했다.
<a href="README.md">전체 해석·한계</a> · <a href="timing.csv">시점 CSV</a></p>
<h2>사고 10개 이어 보기</h2><video controls playsinline preload="metadata" src="accident_10.mp4"></video>
<h2>도로 비사고 10개 이어 보기</h2><video controls playsinline preload="metadata" src="nonaccident_10.mp4"></video>'''
    for r in records:
        name = r['case']
        page += f'<article><h2>{name} · {"사고" if r["label"] else "도로 비사고"}</h2><video controls playsinline preload="none" src="{name}.mp4"></video><a href="{name}.mp4">MP4 다운로드</a></article>'
    (destination / 'index.html').write_text(page + '</html>')
    top_readme = viz / 'README.md'
    content = top_readme.read_text()
    marker = '## MP4 비교 영상'
    if marker not in content:
        content += '\n' + marker + '\n\n[사례별 20개 MP4와 사고·비사고 묶음 영상](videos/README.md)을 추가했다. 기존 PNG는 입력 구간의 약 2/3 시점이며, MP4는 같은 중앙 약 4초 구간 시작부터 재생한다. 맵은 기존 16/32개 측정 시점 사이를 보간했다.\n'
    top_readme.write_text(content)
    top_html = viz / 'index.html'
    content = top_html.read_text()
    if 'videos/index.html' not in content:
        content = content.replace('</h1>', '</h1><p><a href="videos/index.html">새 MP4 비교 영상 보기</a></p>', 1)
    top_html.write_text(content)
    top_manifest = viz / 'artifact_manifest.json'
    data = json.loads(top_manifest.read_text())
    for path in [top_readme, top_html, *sorted(destination.iterdir())]:
        if path.is_file():
            data['files'][str(path.relative_to(viz))] = {'bytes': path.stat().st_size, 'sha256': digest(path)}
    top_manifest.write_text(json.dumps(data, indent=2) + '\n')
    root_readme = repository / 'revision/README.md'
    content = root_readme.read_text()
    if 'viz/videos/README.md' not in content:
        content += '\n같은 20사례의 중앙 약 4초 구간을 [MP4 비교 영상](viz/videos/README.md)으로 추가했다. 정중앙 정지 이미지는 추가하지 않았으며, 기존 맵을 시간축 보간해 재생한다.\n'
    root_readme.write_text(content)
    notes = repository / 'revision/CAM_COMPARISON_20.md'
    content = notes.read_text()
    if '## MP4 비교 영상' not in content:
        content += '\n## MP4 비교 영상\n\n사용자 후속 요청으로 같은 20사례의 MP4와 사고·비사고별 이어 보기 2개를 `viz/videos/`에 추가했다. 기존 PNG 시점은 중앙 입력 구간 약 2/3 지점이며 시작 프레임이 아니다. MP4는 해당 구간 시작부터 약 4초를 재생한다. 추가 정중앙 이미지는 요청 취소에 따라 만들지 않았다.\n\nCPU에서 공통 decoder/crop으로 RGB를 추출하고 기존 16/32시점 맵만 시간 선형 보간했다. 새 모델 추론·checkpoint 변경·GPU 작업은 하지 않았다. 24 FPS H.264/yuv420p/faststart, 무음이며 모든 프레임의 디코딩·단조 증가 timestamp·개수·크기를 검사했다. 두 묶음은 재인코딩 없이 packet remux했다. 재현 코드는 `render_cam_videos.py`, 게시 코드는 `package_cam_videos.py`다. 시각은 원본 영상의 요청 시각이며 RGB는 그 이전 마지막 프레임이다. 확률은 고정 구간 값이고 raw attention·CAM의 해석 한계와 TimeSformer 설정 문제는 위와 같다.\n'
    notes.write_text(content)
    for filename in ['render_cam_videos.py', 'package_cam_videos.py']:
        local = Path(__file__).with_name(filename)
        target = repository / 'revision' / filename
        if local.resolve() != target.resolve():
            shutil.copy2(local, target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--repository', type=Path, required=True)
    args = parser.parse_args()
    publish(args.source, args.repository)
