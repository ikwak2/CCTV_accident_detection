"""Input-only contact sheet for preliminary road-scene screening (no scores)."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import sys

if Path('/app/revision').is_dir():
    sys.path.append('/app/revision')

import cv2
from PIL import Image, ImageDraw
from common import read_csv, read_window, write_csv


def thumbnail(row):
    duration = float(row['duration'])
    frames, _ = read_window(row['path'], max(0., duration / 2), min(.2, duration), 1, float(row['fps']))
    image = Image.fromarray(frames[0])
    image.thumbnail((200, 130))
    return row, image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/results'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--count', type=int, default=80)
    args = parser.parse_args()
    rows, groups = [], set()
    for row in sorted(read_csv(args.root / 'manifests/grouped.csv'), key=lambda r: r['id']):
        if not row['id'].startswith('ucf/') or int(row['label']) or int(row['quarantined']) or row['group'] in groups:
            continue
        groups.add(row['group'])
        rows.append(row)
        if len(rows) == args.count:
            break
    args.out.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError('No eligible UCF negative candidates')
    sheet = Image.new('RGB', (8 * 200, ((len(rows) + 7) // 8) * 154), 'white')
    draw = ImageDraw.Draw(sheet)
    with ProcessPoolExecutor(4, mp_context=multiprocessing.get_context('spawn')) as pool:
        records = []
        for index, (row, image) in enumerate(pool.map(thumbnail, rows)):
            x, y = (index % 8) * 200, (index // 8) * 154
            sheet.paste(image, (x, y + 24))
            draw.text((x + 6, y + 5), f'{index + 1:02d}', fill='black')
            image.save(args.out / f'candidate_{index + 1:03d}.jpg')
            records.append({'index': index + 1, 'id': row['id'], 'group': row['group'], 'label': row['label']})
            if index % 10 == 0:
                print('SCREEN', index + 1, '/', len(rows), flush=True)
    write_csv(args.out / 'candidates_private.csv', records)
    sheet.save(args.out / 'input_only_sheet.jpg')
