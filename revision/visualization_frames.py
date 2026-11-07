"""Export private RGB frames for the fixed explanation cases (CPU only).

Run inside the existing revision image with the data mounted read-only. Outputs
contain source frames/identifiers and must stay outside the public repository.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from common import atomic_json, read_csv, read_window


def export_frames(root, output, count=3):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {row['id']: row for row in read_csv(root / 'manifests/grouped.csv')}
    cases = read_csv(root / 'annotations/common_explanation_cases_private.csv')[:count]
    records = []
    for index, case in enumerate(cases):
        row = manifest[case['id']]
        span = min(4., float(row['duration']))
        start = max(0., float(row['duration']) - span) / 2
        frames, times = read_window(row['path'], start, span, 4, float(row['fps']))
        crops = []
        for frame in frames:
            h, w = frame.shape[:2]
            scale = 256 / min(h, w)
            resized = cv2.resize(frame, (round(w * scale), round(h * scale)),
                                 interpolation=cv2.INTER_LINEAR)
            top, left = (resized.shape[0] - 224) // 2, (resized.shape[1] - 224) // 2
            crops.append(resized[top:top + 224, left:left + 224])
        case_name = f'case_{index:03d}'
        np.savez_compressed(output / f'{case_name}_frames.npz',
                            frames=np.stack(crops), timestamps=np.asarray(times))
        records.append({'case': case_name, 'id': row['id'], 'label': row['label'],
                        'start': start, 'span': span, 'requested_times': times})
    atomic_json(output / 'frames_private.json', {
        'selection': 'First cases of the existing fixed 60-case explanation manifest; no heatmap selection',
        'crop': 'Same resize-short-side-256 / center-crop-224 as model inputs',
        'timing': 'Requested sample instants; decoder uses last available frame at or before each target',
        'cases': records,
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='/results')
    parser.add_argument('--output', required=True)
    parser.add_argument('--count', type=int, default=3)
    args = parser.parse_args()
    export_frames(args.root, args.output, args.count)
