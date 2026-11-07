"""Render fixed-case scientific comparisons from private, measured artifacts.

Needs numpy, matplotlib, and Pillow. This script does not run a model or change
its maps. Use visualization_frames.py for the common RGB display frames.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np


MODELS = [('ViViT', 'vivit_qkvo_mixed_s42'),
          ('TimeSformer', 'timesformer_qkvo_mixed_s42'),
          ('VideoMAE', 'videomae_out_mixed_s42')]


def align_maps(values, source_times, target_times):
    """Linear display interpolation only; never used for inference or metrics."""
    if len(values) != len(source_times) or np.any(np.diff(source_times) <= 0):
        raise ValueError('Invalid time axis')
    if target_times[0] < source_times[0] - 1e-6 or target_times[-1] > source_times[-1] + 1e-6:
        raise ValueError('Display instants must lie inside the measured interval')
    flat = values.reshape(len(values), -1)
    aligned = np.stack([np.interp(target_times, source_times, flat[:, j])
                        for j in range(flat.shape[1])], axis=-1)
    return aligned.reshape(len(target_times), *values.shape[1:])


def load_maps(source, case, tag, method, times):
    with np.load(source / 'explanations' / tag / f'{case}_{method}.npz') as data:
        raw, source_times = data['maps'], data['timestamps']
    if not np.isfinite(raw).all() or np.any(raw < 0):
        raise ValueError('Expected finite nonnegative explanation maps')
    if method == 'gradient_input':
        values = raw / max(float(raw.max()), 1e-12)
    else:
        # A uniform attention distribution is 1 on this common display scale.
        values = raw / np.maximum(raw.mean(axis=(-2, -1), keepdims=True), 1e-12)
    return align_maps(values, source_times, times)


def figure(source, output, case, method, diagnostic_rows):
    with np.load(source / 'frames' / f'{case}_frames.npz') as data:
        frames, times = data['frames'], data['timestamps']
    if frames.shape != (4, 224, 224, 3) or frames.dtype != np.uint8 or len(times) != 4:
        raise ValueError('Expected four RGB input crops with matching instants')
    selected = MODELS if method == 'gradient_input' else [MODELS[0], MODELS[2]]
    fig, axes = plt.subplots(len(selected) + 1, 4, figsize=(12, 3 * (len(selected) + 1)))
    fig.subplots_adjust(left=.14, right=.93, top=.88, bottom=.10, wspace=.025, hspace=.04)
    case_label = f"Case {chr(65 + int(case.split('_')[1]))}"
    title = 'Class-specific gradient x input' if method == 'gradient_input' else 'Architecture-compatible attention rollout'
    fig.suptitle(f'{case_label} | {title}', fontsize=17, y=.965, fontweight='bold')
    fig.text(.14, .925, 'Grouped retraining, seed 42 | fixed center window | identical displayed RGB frames', fontsize=10)
    for column, (frame, stamp) in enumerate(zip(frames, times)):
        axes[0, column].imshow(frame)
        axes[0, column].set_title(f'{stamp:.2f} s', fontsize=11)
    axes[0, 0].set_ylabel('Input', fontsize=12, rotation=0, ha='right', va='center', labelpad=14)
    vmax = 1 if method == 'gradient_input' else 3
    for row_index, (name, tag) in enumerate(selected, start=1):
        values = load_maps(source, case, tag, method, times)
        for column, frame in enumerate(frames):
            axes[row_index, column].imshow(frame)
            axes[row_index, column].imshow(values[column], extent=(-.5, 223.5, 223.5, -.5),
                                           cmap='turbo', vmin=0, vmax=vmax,
                                           interpolation='bilinear', alpha=.38)
        p = diagnostic_rows[tag][int(case.split('_')[1])]['center_window_probability']
        label = f'{name}\np(accident)={float(p):.3f}'
        axes[row_index, 0].set_ylabel(label, fontsize=10, rotation=0, ha='right', va='center', labelpad=14)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    color_axis = fig.add_axes([.95, .22, .012, .48])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, vmax), cmap='turbo'), cax=color_axis)
    cb.set_label('Relative positive attribution' if method == 'gradient_input' else 'Attention / frame spatial mean', fontsize=9)
    note = ('Accident-minus-normal logit target. Scale: each model / full-clip maximum; color is not comparable absolute evidence.\n'
            '16/32-frame maps interpolated to common display instants. No onset or object-box annotations.')
    if method == 'rollout':
        note = ('ViViT: CLS readout; VideoMAE: mean readout, no CLS. Shared scale: 1 = spatially uniform; values >3 are clipped.\n'
                'TimeSformer rollout not assessed (divided attention). Raw attention is not class-specific or proof of localization.')
    fig.text(.14, .042, note, fontsize=8, linespacing=1.5)
    stem = output / f'{case}_{method}_comparison'
    fig.savefig(stem.with_suffix('.png'), dpi=160, facecolor='white')
    fig.savefig(stem.with_suffix('.pdf'), facecolor='white')
    plt.close(fig)
    return stem.with_suffix('.png').name


def render(source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((source / 'frames/frames_private.json').read_text())
    diagnostics = {}
    summary = []
    for name, tag in MODELS:
        with (source / 'explanations' / tag / 'diagnostics.csv').open() as stream:
            diagnostics[tag] = list(csv.DictReader(stream))
        rows = diagnostics[tag]
        summary.append({'model': name, 'cases': len(rows),
                        'positive_attribution_cases': sum(r['positive_attribution'].lower() == 'true' for r in rows),
                        'mean_top_minus_random_drop': float(np.mean([float(r['top_minus_random_drop']) for r in rows]))})
    images = []
    for item in metadata['cases']:
        case_index = int(item['case'].split('_')[1])
        if any(diagnostics[tag][case_index]['id'] != item['id'] for _, tag in MODELS):
            raise ValueError('Frame and model diagnostic identifiers do not match')
        images.append(figure(source, output, item['case'], 'gradient_input', diagnostics))
        images.append(figure(source, output, item['case'], 'rollout', diagnostics))
    (output / 'diagnostic_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    inputs = [source / 'frames/frames_private.json']
    for item in metadata['cases']:
        inputs.append(source / 'frames' / f"{item['case']}_frames.npz")
        for name, tag in MODELS:
            inputs.append(source / 'explanations' / tag / f"{item['case']}_gradient_input.npz")
            if name != 'TimeSformer':
                inputs.append(source / 'explanations' / tag / f"{item['case']}_rollout.npz")
    for _, tag in MODELS:
        inputs.extend(source / 'explanations' / tag / filename for filename in ['diagnostics.csv', 'scope.json'])
    manifest = {'selection': metadata['selection'],
                'input_sha256': {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in inputs},
                'renderer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'numpy': np.__version__, 'matplotlib': matplotlib.__version__,
                'display_interpolation': 'linear time interpolation of maps only',
                'gradient_scale': 'per-model whole-clip maximum',
                'rollout_scale': 'per-frame spatial mean, shared display range 0..3',
                'outputs': images}
    (output / 'render_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    html = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>ORV model explanations</title>
<style>body{max-width:1200px;margin:32px auto;font:16px/1.7 sans-serif;color:#17212b;padding:0 20px}img{width:100%;height:auto}article{margin:50px 0}a{color:#175fad}</style>
<h1>ORV: 모델별 설명 맵 비교</h1>
<p>새 그룹 분할로 학습한 seed 42 모델의 기존 고정 60사례 중 첫 3개를 표시했다. 그림의 모양으로 사례를 고르지 않았다.
세 모델의 설명 입력 구간은 동일하지만 ViViT·TimeSformer는 32프레임, VideoMAE는 16프레임을 사용한다.
표시 시각의 RGB는 공통으로 추출했고, 설명 맵만 시간축 선형 보간했다. 표시된 확률은 중앙 4초 구간의 값이며 전체 영상 평가 점수가 아니다.</p>
<p><b>해석:</b> 분류 AUROC와 설명의 공간적 타당성은 별도다. 밝은 부분은 사고 객체의 정답 표시가 아니다.
현재 사례에는 사람이 검증한 사고 시작 시각·객체 box가 없고, 중앙 구간이 충돌 이후 장면일 수 있다.
Gradient × input과 attention rollout은 다른 양을 측정하므로 두 방법의 색을 직접 비교하지 않는다.
TimeSformer의 divided attention rollout은 검증되지 않아 표시하지 않았다.</p>'''
    for name in images:
        html += f'<article><a href="{name}"><img src="{name}" alt="{name}"></a><p><a href="{Path(name).with_suffix(".pdf")}">PDF 다운로드</a></p></article>'
    html += '<p>원본 프레임과 식별 정보는 비공개 결과 폴더에서 관리하며 공개 Git 저장소에는 포함하지 않는다.</p></html>'
    (output / 'index.html').write_text(html)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='private/ directory containing explanations/ and frames/')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    render(args.source, args.output)
