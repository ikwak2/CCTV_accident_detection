"""Offline 20-case gallery: original-paper CAM methods, all three models.

Outputs: 20 PNG comparison sheets, per-case four-page PDFs (main comparison and
four-timepoint sheets for each method), combined/class PDFs, and an audit table.
"""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
import numpy as np

from render_visualizations import align_maps


MODELS = [('vivit', 'ViViT'), ('timesformer', 'TimeSformer'), ('videomae', 'VideoMAE')]
METHODS = [('rollout', 'Attention rollout'), ('layercam', 'LayerCAM'), ('gradcampp', 'Grad-CAM++')]


def prepare_case(source, item):
    with np.load(source / 'frames' / f"{item['case']}.npz") as data:
        frames, times = data['frames'], data['timestamps']
    if frames.shape != (4, 224, 224, 3) or frames.dtype != np.uint8:
        raise ValueError('Invalid common display crops')
    maps, records, raw_stats = {}, {}, []
    for key, name in MODELS:
        folder = source / 'maps' / key
        record = json.loads((folder / f"{item['case']}.json").read_text())
        contract = json.loads((folder / 'contract.json').read_text())
        if record['display_id'] != item['display_id'] or record['label'] != item['label'] or contract['target_class'] != 1:
            raise ValueError('Case identity, label, or target mismatch')
        records[key] = record
        with np.load(folder / f"{item['case']}.npz") as data:
            for method, _ in METHODS:
                raw = data[method]
                if raw.shape[-2:] != (14, 14) or not np.isfinite(raw).all() or np.any(raw < 0):
                    raise ValueError('Invalid measured map')
                maximum = float(raw.max())
                if method == 'rollout':
                    values = raw / np.maximum(raw.mean(axis=(-2, -1), keepdims=True), 1e-30)
                else:
                    values = raw / maximum if maximum else raw
                maps[key, method] = align_maps(values, data['timestamps'], times)
                raw_stats.append({'case': item['display_id'], 'label': item['label'], 'model': name,
                                  'method': method, 'status': 'nonzero' if maximum else 'zero_signal',
                                  'raw_max': maximum, 'raw_mean': float(raw.mean()),
                                  'gradient_abs_max': record['gradient_abs_max'],
                                  'accident_probability': record['probability'], 'display_time': float(times[2])})
    return frames, times, maps, records, raw_stats


def image_axis(ax, frame, values=None, vmax=1, zero=False):
    ax.imshow(frame)
    if values is not None and not zero:
        ax.imshow(values, extent=(-.5, 223.5, 223.5, -.5), cmap='turbo', vmin=0, vmax=vmax,
                  interpolation='bilinear', alpha=.38)
    if zero:
        ax.text(.5, .05, 'ZERO SIGNAL', color='white', fontsize=10, ha='center', transform=ax.transAxes,
                bbox={'facecolor': '#333333', 'alpha': .8, 'pad': 3})
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main_figure(item, frames, times, maps, records):
    fig, axes = plt.subplots(4, 3, figsize=(10, 12))
    fig.subplots_adjust(left=.18, right=.88, top=.87, bottom=.115, wspace=.025, hspace=.11)
    label = 'Accident' if item['label'] else 'Non-accident'
    fig.suptitle(f"{item['display_id']} | {label} (dataset label) | t = {times[2]:.2f} s", fontsize=15, fontweight='bold', y=.965)
    fig.text(.18, .922, 'Grouped retraining, seed 42 | same displayed RGB frame | fixed center 4-second input', fontsize=9)
    for col, (key, name) in enumerate(MODELS):
        image_axis(axes[0, col], frames[2])
        axes[0, col].set_title(f"{name}\np(accident) = {records[key]['probability']:.3f}", fontsize=11)
    axes[0, 0].set_ylabel('Input', rotation=0, labelpad=15, ha='right', va='center', fontsize=11)
    for row, (method, name) in enumerate(METHODS, 1):
        vmax = 3 if method == 'rollout' else 1
        for col, (key, _) in enumerate(MODELS):
            values = maps[key, method]
            image_axis(axes[row, col], frames[2], values[2], vmax, zero=not values.any())
        axes[row, 0].set_ylabel(name.replace('Attention ', 'Attention\n'), rotation=0, labelpad=15,
                              ha='right', va='center', fontsize=10)
        pos = axes[row, -1].get_position()
        color_ax = fig.add_axes([.90, pos.y0, .014, pos.height])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, vmax), cmap='turbo'), cax=color_ax)
        cb.set_label('Attention / mean' if method == 'rollout' else 'CAM / clip max', fontsize=8)
        cb.ax.tick_params(labelsize=8)
    fig.text(.18, .05,
             'LayerCAM / Grad-CAM++ target: accident logit, for both labels. Rollout is not class-specific.\n'
             'Maps are interpolated in time for display. Each CAM uses its own full-clip maximum.\n'
             'TimeSformer rollout: spatial + temporal flow; checkpoint config 8 / input 32 frames preserved.\n'
             'No onset / object-box annotations. TimeSformer requires a corrected-config retraining control.',
             fontsize=8, linespacing=1.6)
    return fig


def temporal_figure(item, method, method_name, frames, times, maps, records):
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.subplots_adjust(left=.14, right=.93, top=.88, bottom=.09, wspace=.025, hspace=.035)
    fig.suptitle(f"{item['display_id']} | {method_name} | {'Accident' if item['label'] else 'Non-accident'} label", fontsize=16, y=.963)
    fig.text(.14, .925, 'Same four RGB display instants | model inputs: 32 / 32 / 16 frames | seed 42', fontsize=10)
    for col, (frame, stamp) in enumerate(zip(frames, times)):
        image_axis(axes[0, col], frame)
        axes[0, col].set_title(f'{stamp:.2f} s', fontsize=11)
    axes[0, 0].set_ylabel('Input', rotation=0, ha='right', va='center', labelpad=14)
    vmax = 3 if method == 'rollout' else 1
    for row, (key, name) in enumerate(MODELS, 1):
        for col, frame in enumerate(frames):
            values = maps[key, method]
            image_axis(axes[row, col], frame, values[col], vmax, zero=not values.any())
        axes[row, 0].set_ylabel(f"{name}\np(accident)={records[key]['probability']:.3f}", rotation=0,
                              ha='right', va='center', labelpad=14, fontsize=9)
    color_ax = fig.add_axes([.95, .22, .012, .48])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, vmax), cmap='turbo'), cax=color_ax)
    cb.set_label('Attention / frame spatial mean' if method == 'rollout' else 'CAM / model full-clip maximum', fontsize=9)
    fig.text(.14, .04, 'Accident-logit target for CAMs, also on non-accident clips. Raw rollout is not class-specific.\n'
             'Display-only interpolation. TimeSformer preserves checkpoint config 8 / input 32 frames; inverse grouping for display.\n'
             'Relative colors do not establish localization accuracy or compare absolute evidence.', fontsize=8)
    return fig


def render(source, output):
    output.mkdir(parents=True, exist_ok=True)
    selection = json.loads((source / 'selection_private.json').read_text())
    cases = selection['cases']
    assert len(cases) == 20 and sum(r['label'] == 1 for r in cases) == 10
    assert len({r['group'] for r in cases}) == 20
    rows, figures = [], []
    with PdfPages(output / 'all_20_comparisons.pdf') as combined, \
         PdfPages(output / 'accident_10_comparisons.pdf') as accidents, \
         PdfPages(output / 'nonaccident_10_comparisons.pdf') as normals:
        for item in cases:
            frames, times, maps, records, statistics = prepare_case(source, item)
            rows.extend(statistics)
            stem = item['display_id']
            fig = main_figure(item, frames, times, maps, records)
            fig.savefig(output / f'{stem}_comparison.png', dpi=150, facecolor='white')
            combined.savefig(fig)
            (accidents if item['label'] else normals).savefig(fig)
            with PdfPages(output / f'{stem}_details.pdf') as details:
                details.savefig(fig)
                plt.close(fig)
                for method, name in METHODS:
                    temporal = temporal_figure(item, method, name, frames, times, maps, records)
                    details.savefig(temporal)
                    plt.close(temporal)
            figures.append(f'{stem}_comparison.png')
            print('RENDERED', stem, flush=True)
    with (output / 'diagnostics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for label in [1, 0]:
        for _, model in MODELS:
            for method, _ in METHODS:
                subset = [r for r in rows if r['label'] == label and r['model'] == model and r['method'] == method]
                summary.append({'label': label, 'model': model, 'method': method, 'cases': len(subset),
                                'nonzero': sum(r['status'] == 'nonzero' for r in subset),
                                'zero_signal': sum(r['status'] == 'zero_signal' for r in subset)})
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    inputs = [source / 'selection_private.json', *sorted((source / 'frames').glob('*.npz')),
              *sorted((source / 'maps').glob('*/*.npz')), *sorted((source / 'maps').glob('*/*.json'))]
    provenance = {'selection': selection['selection'], 'display_sample_index': 2,
                  'cam_target': 'accident logit for both dataset labels',
                  'input_sha256': {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                  'renderer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'alignment_code_sha256': hashlib.sha256(Path(__file__).with_name('render_visualizations.py').read_bytes()).hexdigest(),
                  'versions': {'numpy': np.__version__, 'matplotlib': matplotlib.__version__}, 'figures': figures}
    (output / 'render_manifest.json').write_text(json.dumps(provenance, indent=2) + '\n')
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>ORV: 사고 10 · 비사고 10 CAM 비교</title><style>
body{max-width:1100px;margin:30px auto;padding:0 18px;font:16px/1.7 sans-serif;color:#17212b}img{width:100%;height:auto}
article{margin:45px 0;border-top:1px solid #ccd5df;padding-top:20px}nav{position:sticky;top:0;background:#fff;padding:12px 0;z-index:1}
button{padding:9px 15px;margin-right:8px;border:1px solid #789;border-radius:6px;background:#f4f7fb;cursor:pointer}a{color:#175fad}
</style><h1>사고 10개 · 비사고 10개: 세 모델 × 세 설명 방법</h1>
<p>원고에서 비교한 <b>attention rollout, LayerCAM, Grad-CAM++</b>를 새 grouped 학습 seed 42 모델에 적용했다.
사고는 이전 세 사례·격리 후보·중복 그룹을 제외한 기존 고정 목록에서 10개를 선택했다.
비사고 도로 장면의 추가 선별 여부와 범위는 아래 선택 기록에 명시했다. 맵의 모양으로 고르지 않았다.</p>
<p>CAM은 사고·비사고 모두 <b>사고 logit</b>을 대상으로 한다. 비사고 그림은 정상 판정의 근거가 아니라 사고 출력에 기여하는 부분을 보여준다.
라벨은 데이터셋 기준이며, 비사고가 모두 도로 장면이라는 뜻은 아니다. 중앙 4초가 실제 충돌을 포함하는지도 별도 주석이 필요하다.</p>
<p><b>TimeSformer 주의:</b> 시각화 과정에서 입력 32프레임과 내부 설정 8프레임의 불일치를 발견했다.
기존 학습 가중치의 동작을 유지하고 실제 토큰 순서를 역변환해 표시했다. 올바른 32프레임 설정의 재학습 비교가 필요하므로,
이 그림으로 TimeSformer 구조 자체의 시각화 품질을 단정하지 않는다.</p>
<p>큰 그림은 공통 네 표시 시각 중 세 번째의 동일 RGB 프레임이다. 개별 PDF는 비교표와 세 방법의 네 시점 연속 비교를 포함한 4쪽이다.
색은 상대 강도다. Rollout은 class-specific하지 않고, TimeSformer는 공간·시간 attention을 모두 사용한 구조적 확장이다.
0 신호는 ZERO SIGNAL로 표시하며, smoothing·percentile stretch·임의 fallback을 적용하지 않았다.</p>
<p><a href="all_20_comparisons.pdf">20개 전체 비교 PDF</a> · <a href="accident_10_comparisons.pdf">사고 10개 PDF</a> ·
<a href="nonaccident_10_comparisons.pdf">비사고 10개 PDF</a> · <a href="diagnostics.csv">측정값 CSV</a></p>
<nav><button onclick="filterCases('all')">전체 20개</button><button onclick="filterCases('1')">사고 10개</button><button onclick="filterCases('0')">비사고 10개</button></nav>'''
    page += '<p><b>선택 기록:</b> ' + html.escape(selection.get('selection_ko', selection['selection'])) + '</p>'
    for item in cases:
        stem = html.escape(item['display_id'])
        label = '사고' if item['label'] else '비사고'
        page += f'<article data-label="{item["label"]}"><h2>{stem} · {label}</h2><a href="{stem}_comparison.png"><img loading="lazy" src="{stem}_comparison.png" alt="{stem} 세 방법·세 모델 비교"></a><p><a href="{stem}_details.pdf">4쪽 상세 PDF: 비교표 + 방법별 네 시점</a></p></article>'
    page += '''<script>function filterCases(label){document.querySelectorAll('article').forEach(el=>{el.hidden=label!=='all'&&el.dataset.label!==label})}</script></html>'''
    (output / 'index.html').write_text(page)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    render(args.source, args.output)
