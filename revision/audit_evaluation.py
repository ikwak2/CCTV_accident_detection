"""Strict, shared evaluation eligibility using the exact inference windows.

Training data are never edited. Video paths and failure records stay on server.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np

from common import atomic_json, read_csv, read_window, sha256, write_csv


PROTOCOL = {'window_seconds': 4., 'windows': 5, 'frame_counts': [16, 32],
            'failure_policy': 'exclude entire video for every model on any decode exception'}


def check_video(row):
    duration, fps = float(row['duration']), float(row['fps'])
    span = min(PROTOCOL['window_seconds'], duration)
    start = None
    count = None
    try:
        for count in PROTOCOL['frame_counts']:
            for start in np.linspace(0, max(0., duration-span), PROTOCOL['windows']):
                frames, timestamps = read_window(row['path'], float(start), span, count, fps)
                if len(frames) != count or len(timestamps) != count:
                    raise ValueError('Incomplete sampled window')
        return {'id': row['id'], 'file_sha256': row['file_sha256'], 'eligible': True}
    except Exception as exc:
        return {'id': row['id'], 'file_sha256': row['file_sha256'], 'eligible': False,
                'frame_count': count, 'start_seconds': float(start) if start is not None else None,
                'error': type(exc).__name__+': '+str(exc)}


def counts(rows):
    return dict(sorted(Counter(f"{r['split']}:{r['dataset']}:{r['label']}" for r in rows).items()))


def audit(root, workers=4, executor='process'):
    root = Path(root)
    source = root/'manifests/grouped.csv'
    rows = [r for r in read_csv(source) if r['split'] in ['dev', 'test', 'external']]
    out = root/'manifests'
    contract = {'source_manifest_sha256': sha256(source), 'protocol': PROTOCOL,
                'decoder_code_sha256': sha256(Path(__file__).with_name('common.py'))}
    contract_file = out/'evaluation_audit_contract.json'
    journal = out/'evaluation_checks_private.jsonl'
    previous = {}
    if journal.exists():
        if not contract_file.exists() or json.loads(contract_file.read_text()) != contract:
            raise RuntimeError('Existing decode checks have a different contract; archive before rerun')
        for line in journal.read_text().splitlines():
            item = json.loads(line)
            previous[item['id']] = item
    atomic_json(contract_file, contract)
    started = time.time()
    reused = len(previous)
    pool_type = (partial(ProcessPoolExecutor, mp_context=multiprocessing.get_context('spawn'))
                 if executor == 'process' else ThreadPoolExecutor)
    with pool_type(max_workers=workers) as pool, open(journal, 'a') as stream:
        futures = {pool.submit(check_video, r): r for r in rows if r['id'] not in previous}
        for future in as_completed(futures):
            item = future.result()
            previous[item['id']] = item
            stream.write(json.dumps(item, ensure_ascii=False)+'\n')
            stream.flush()
            if len(previous) % 50 == 0 or not item['eligible']:
                print('EVALUATION_AUDIT', len(previous), '/', len(rows),
                      'excluded', sum(not r['eligible'] for r in previous.values()), flush=True)
                atomic_json(out/'evaluation_audit_progress.json',
                            {'checked': len(previous), 'total': len(rows), 'seconds': time.time()-started})
    assert len(previous) == len(rows)
    assert all(previous[r['id']]['file_sha256'] == r['file_sha256'] for r in rows)
    kept = [r for r in rows if previous[r['id']]['eligible']]
    excluded = [r for r in rows if not previous[r['id']]['eligible']]
    destination = out/'evaluation.csv'
    write_csv(destination, kept)
    atomic_json(out/'evaluation_exclusions_private.json', [dict(previous[r['id']],
                split=r['split'], dataset=r['dataset'], label=int(r['label'])) for r in excluded])
    report = dict(contract, evaluation_manifest_sha256=sha256(destination),
                  total_before=len(rows), total_eligible=len(kept), total_excluded=len(excluded),
                  counts_before=counts(rows), counts_eligible=counts(kept), counts_excluded=counts(excluded),
                  seconds=time.time()-started, training_manifest_unchanged=True,
                  executor=executor, workers=workers, reused_checks=reused)
    atomic_json(out/'evaluation_audit.json', report)
    print('EVALUATION_AUDIT_COMPLETE', json.dumps(report), flush=True)
    return report


def evaluation_rows(root):
    """Refuse evaluation until a complete, matching shared audit exists."""
    root = Path(root)
    report = json.loads((root/'manifests/evaluation_audit.json').read_text())
    if report['source_manifest_sha256'] != sha256(root/'manifests/grouped.csv'):
        raise RuntimeError('Training/source manifest changed after evaluation audit')
    if report['evaluation_manifest_sha256'] != sha256(root/'manifests/evaluation.csv'):
        raise RuntimeError('Shared evaluation manifest changed after audit')
    if report['decoder_code_sha256'] != sha256(Path(__file__).with_name('common.py')) or report['protocol'] != PROTOCOL:
        raise RuntimeError('Decoder or evaluation windows changed after audit')
    rows = read_csv(root/'manifests/evaluation.csv')
    if len(rows) != report['total_eligible'] or len({r['id'] for r in rows}) != len(rows):
        raise RuntimeError('Incomplete or duplicate evaluation eligibility')
    return rows, report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/results')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--executor', choices=['process', 'thread'], default='process')
    args = parser.parse_args()
    audit(args.root, args.workers, args.executor)
