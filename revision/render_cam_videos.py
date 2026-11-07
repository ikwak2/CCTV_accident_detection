"""CPU-only MP4 rendering of the existing 20-case CAM comparison.

Uses the existing fixed-window maps; no model inference or center still images.
Video frames are decoded at the display clock and maps linearly interpolated.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import sys

sys.path.append('/app/revision')
import av
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from common import read_csv, read_window


MODELS = [('vivit', 'ViViT'), ('timesformer', 'TimeSformer'), ('videomae', 'VideoMAE')]
METHODS = [('rollout', 'Attention rollout'), ('layercam', 'LayerCAM'), ('gradcampp', 'Grad-CAM++')]
WIDTH, HEIGHT, TILE = 1024, 1168, 224
LEFT, TOP, STEP = 160, 126, 236


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def align(values, source_times, targets):
    source_times, targets = np.asarray(source_times), np.asarray(targets)
    if len(values) != len(source_times) or np.any(np.diff(source_times) <= 0):
        raise ValueError('Invalid measured time axis')
    if targets[0] < source_times[0] - 1e-7 or targets[-1] > source_times[-1] + 1e-7:
        raise ValueError('Display times exceed measured map support')
    left = np.searchsorted(source_times, targets, side='right') - 1
    left = np.clip(left, 0, len(source_times) - 2)
    weight = ((targets - source_times[left]) / (source_times[left + 1] - source_times[left]))[:, None, None]
    return values[left] * (1 - weight) + values[left + 1] * weight


def inspect_video(path, expected_frames=None, qa_directory=None):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.codec_context.name != 'h264' or stream.width != WIDTH or stream.height != HEIGHT:
            raise ValueError('Unexpected video codec or dimensions')
        if container.streams.audio:
            raise ValueError('Comparison videos must contain no audio stream')
        count, previous = 0, -1.
        for frame in container.decode(stream):
            timestamp = float(frame.pts * frame.time_base)
            if timestamp <= previous:
                raise ValueError('Non-monotonic playback timestamps')
            previous = timestamp
            if qa_directory is not None and count in [0, 48, 95]:
                qa_directory.mkdir(parents=True, exist_ok=True)
                frame.to_image().save(qa_directory / f'{Path(path).stem}_{count:03d}.jpg')
            count += 1
        if expected_frames is not None and count != expected_frames:
            raise ValueError(f'Expected {expected_frames} decoded frames, received {count}')
        return {'frames': count, 'fps': float(stream.average_rate), 'width': stream.width, 'height': stream.height,
                'duration': float(stream.duration * stream.time_base), 'codec': 'h264', 'audio': False}


def static_canvas(records, font_path):
    canvas = Image.new('RGB', (WIDTH, HEIGHT), 'white')
    draw = ImageDraw.Draw(canvas)
    fonts = {size: ImageFont.truetype(str(font_path), size) for size in [13, 14, 17, 19, 21, 24]}
    for column, (key, name) in enumerate(MODELS):
        center = LEFT + column * STEP + TILE // 2
        draw.text((center, 58), name, font=fonts[24], fill='#17212b', anchor='mt')
        draw.text((center, 90), f"p(accident) = {records[key]['probability']:.3f}",
                  font=fonts[17], fill='#17212b', anchor='mt')
    labels = ['Input crop', 'Attention\nrollout', 'LayerCAM', 'Grad-CAM++']
    for row, label in enumerate(labels):
        draw.multiline_text((18, TOP + row * STEP + 92), label, font=fonts[19], fill='#17212b', spacing=5)
    draw.text((18, 1081), 'CAM target: accident logit for both labels. Window probabilities stay fixed.', font=fonts[14], fill='#17212b')
    draw.text((18, 1104), 'Maps: measured 16/32 samples, linearly interpolated for display; no new inference.', font=fonts[14], fill='#17212b')
    draw.text((18, 1127), 'TimeSformer checkpoint: input 32 / config 8 preserved. No onset or box annotations.', font=fonts[14], fill='#17212b')
    result = np.asarray(canvas).copy()
    for row, (method, _) in enumerate(METHODS, start=1):
        y = TOP + row * STEP
        scale = np.linspace(255, 0, TILE).astype(np.uint8)[:, None]
        bar = cv2.cvtColor(cv2.applyColorMap(np.repeat(scale, 16, 1), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
        result[y:y + TILE, 882:898] = bar
    canvas = Image.fromarray(result)
    draw = ImageDraw.Draw(canvas)
    for row, (method, _) in enumerate(METHODS, start=1):
        y = TOP + row * STEP
        draw.text((905, y), '3' if method == 'rollout' else '1', font=fonts[14], fill='#17212b')
        draw.text((905, y + TILE - 18), '0', font=fonts[14], fill='#17212b')
        draw.multiline_text((905, y + 88), 'attention\n/ mean' if method == 'rollout' else 'CAM\n/ clip max',
                            font=fonts[13], fill='#17212b', spacing=4)
    return np.asarray(canvas).copy(), fonts[21]


def render_case(job):
    root, source, output, font_path, fps, item = job
    root, source, output, font_path = map(Path, [root, source, output, font_path])
    cv2.setNumThreads(1)
    filename = output / f"{item['display_id']}.mp4"
    metadata_path = output / f"{item['display_id']}.json"
    inputs = {'renderer': digest(__file__), 'font': digest(font_path), 'fps': fps,
              'selection': digest(source / 'selection_private.json'),
              'maps': {key: digest(source / 'maps' / key / f"{item['case']}.npz") for key, _ in MODELS}}
    if filename.exists() and metadata_path.exists():
        old = json.loads(metadata_path.read_text())
        if old['input_contract'] != inputs:
            raise ValueError('Existing video input contract differs')
        inspect_video(filename, old['frames'])
        return old
    manifest = {r['id']: r for r in read_csv(root / 'manifests/grouped.csv')}
    row = manifest[item['id']]
    raw, times, records = {}, {}, {}
    for key, _ in MODELS:
        folder = source / 'maps' / key
        with np.load(folder / f"{item['case']}.npz") as arrays:
            times[key] = arrays['timestamps'].copy()
            for method, _ in METHODS:
                value = arrays[method].copy()
                if not np.isfinite(value).all() or np.any(value < 0):
                    raise ValueError('Invalid measured map')
                if method == 'rollout':
                    value = value / np.maximum(value.mean(axis=(-2, -1), keepdims=True), 1e-30)
                elif value.max() > 0:
                    value = value / value.max()
                raw[key, method] = value
        records[key] = json.loads((folder / f"{item['case']}.json").read_text())
        if records[key]['display_id'] != item['display_id'] or records[key]['label'] != item['label']:
            raise ValueError('Case identifiers do not match')
    start = max(t[0] for t in times.values())
    end = min(t[-1] for t in times.values())
    count = math.floor((end - start) * fps + 1e-7) + 1
    targets = start + np.arange(count) / fps
    # The common reader subtracts one source-frame period from span. This span
    # therefore requests exactly the regular MP4 display clock, without extrapolation.
    decode_span = (count - 1) / fps + 1 / float(row['fps'])
    frames, requested = read_window(row['path'], start, decode_span, count, float(row['fps']))
    if not np.allclose(requested, targets, atol=1e-7):
        raise ValueError('RGB clock differs from map display clock')
    crops = []
    for frame in frames:
        h, w = frame.shape[:2]
        scale = 256 / min(h, w)
        resized = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_LINEAR)
        y, x = (resized.shape[0] - TILE) // 2, (resized.shape[1] - TILE) // 2
        crops.append(resized[y:y + TILE, x:x + TILE])
    del frames
    maps = {(key, method): align(values, times[key], targets) for (key, method), values in raw.items()}
    template, header_font = static_canvas(records, font_path)
    temporary = filename.with_suffix('.partial.mp4')
    with av.open(str(temporary), 'w', format='mp4', options={'movflags': '+faststart'}) as container:
        stream = container.add_stream('libx264', rate=fps)
        stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, 'yuv420p'
        stream.codec_context.thread_count = 2
        stream.codec_context.gop_size = fps * 2
        stream.codec_context.max_b_frames = 0
        stream.options = {'preset': 'veryfast', 'crf': '20'}
        for index, (rgb, stamp) in enumerate(zip(crops, targets)):
            canvas = template.copy()
            for column, (key, _) in enumerate(MODELS):
                x = LEFT + column * STEP
                canvas[TOP:TOP + TILE, x:x + TILE] = rgb
                for method_row, (method, _) in enumerate(METHODS, start=1):
                    y = TOP + method_row * STEP
                    value = maps[key, method][index]
                    maximum = 3 if method == 'rollout' else 1
                    heat = cv2.resize(value.astype(np.float32), (TILE, TILE), interpolation=cv2.INTER_LINEAR)
                    heat = np.round(np.clip(heat / maximum, 0, 1) * 255).astype(np.uint8)
                    heat = cv2.cvtColor(cv2.applyColorMap(heat, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
                    overlay = cv2.addWeighted(rgb, .62, heat, .38, 0) if raw[key, method].any() else rgb
                    canvas[y:y + TILE, x:x + TILE] = overlay
            image = Image.fromarray(canvas)
            draw = ImageDraw.Draw(image)
            label = 'ACCIDENT LABEL' if item['label'] else 'NON-ACCIDENT LABEL'
            draw.text((18, 16), f"{item['display_id']} | {label} | video t={stamp:.2f}s | window +{stamp - start:.2f}s",
                      font=header_font, fill='#17212b')
            draw.rectangle((18, 1157, 1006, 1160), fill='#dce3eb')
            draw.rectangle((18, 1157, 18 + round(988 * (index + 1) / count), 1160), fill='#2864ad')
            video_frame = av.VideoFrame.from_ndarray(np.asarray(image), format='rgb24')
            video_frame.pts, video_frame.time_base = index, Fraction(1, fps)
            for packet in stream.encode(video_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    temporary.replace(filename)
    check = inspect_video(filename, count, output / 'private_qa' if item['display_id'] in ['A01', 'N01'] else None)
    record = {'case': item['display_id'], 'label': item['label'], **check,
              'source_video_duration': float(row['duration']), 'source_fps': float(row['fps']),
              'input_window_start': float(start), 'measured_window_end': float(end),
              'first_requested_video_time': float(targets[0]), 'last_requested_video_time': float(targets[-1]),
              'display_times': targets.tolist(), 'input_contract': inputs, 'sha256': digest(filename),
              'scope': 'Existing center-window maps only; linear time interpolation for display; fixed window probabilities',
              'timesformer_warning': 'Input 32 / config 8 checkpoint preserved; token grouping restored for display'}
    metadata_path.write_text(json.dumps(record, indent=2) + '\n')
    print('VIDEO', item['display_id'], count, 'frames', 'seconds', round(check['duration'], 3), flush=True)
    return record


def concatenate(paths, destination):
    """Lossless packet remux of compatible CFR H.264 clips; no re-encoding."""
    offset = 0
    with av.open(str(paths[0])) as first:
        initial = first.streams.video[0]
        extra, base = initial.codec_context.extradata, initial.time_base
        with av.open(str(destination), 'w', format='mp4', options={'movflags': '+faststart'}) as output:
            stream = output.add_stream(template=initial)
            for path in paths:
                with av.open(str(path)) as source:
                    original = source.streams.video[0]
                    if original.codec_context.extradata != extra or original.time_base != base:
                        raise ValueError('Compilation clips must share H.264 parameters and clock')
                    next_offset = offset
                    for packet in source.demux(original):
                        if packet.dts is None:
                            continue
                        packet.pts += offset
                        packet.dts += offset
                        next_offset = max(next_offset, packet.pts + packet.duration)
                        packet.stream = stream
                        output.mux(packet)
                    offset = next_offset


def run(args):
    source, output = args.source, args.output
    output.mkdir(parents=True, exist_ok=True)
    selection = json.loads((source / 'selection_private.json').read_text())
    cases = selection['cases'][:args.limit] if args.limit else selection['cases']
    jobs = [(str(args.root), str(source), str(output), str(args.font), args.fps, item) for item in cases]
    with ProcessPoolExecutor(args.jobs, mp_context=multiprocessing.get_context('spawn')) as pool:
        records = list(pool.map(render_case, jobs))
    compilations = {}
    if not args.limit:
        for label, name in [(1, 'accident_10'), (0, 'nonaccident_10')]:
            subset = [r for r in records if r['label'] == label]
            if len(subset) != 10:
                raise ValueError('Expected ten cases per class')
            filename = output / f'{name}.mp4'
            concatenate([output / f"{r['case']}.mp4" for r in subset], filename)
            compilations[name] = {**inspect_video(filename, sum(r['frames'] for r in subset)),
                                  'sha256': digest(filename), 'cases': [r['case'] for r in subset]}
    manifest = {'renderer_sha256': digest(__file__), 'fps': args.fps, 'cases': records,
                'compilations': compilations, 'no_center_still_images': True,
                'timing': 'Regular display clock within measured center-window support; last available RGB frame at/before each requested instant',
                'normalization': 'CAM / whole-clip maximum; rollout / per-frame spatial mean; alpha 0.38, turbo color map',
                'codec': 'H.264 yuv420p, faststart MP4, no audio; compilation packet remux without re-encoding'}
    (output / 'video_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/results'))
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--font', type=Path, required=True)
    parser.add_argument('--fps', type=int, default=24)
    parser.add_argument('--jobs', type=int, default=2)
    parser.add_argument('--limit', type=int)
    run(parser.parse_args())
