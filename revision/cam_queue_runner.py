"""Run ONLY the registered CAM jobs in a bounded Docker process on GPU 0/1.

Existing evaluation jobs may continue on the same physical GPU. Require less
than 2 GiB already used and reserve at most 40% of GPU memory for this process.
Use a separate container with CPU/RAM limits; never steal another worker's job.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

sys.path.append('/app/revision')
import torch
from experiment_queue import connect, now, status
from cam_comparison import LORA, run


ALLOWED = {'GPU-9564a23b-7611-050b-d550-1af42c186be4', 'GPU-965e8bf9-fd59-6338-4527-7ddbde441845'}


def execute(root, output, models, worker, job_prefix='cam_comparison_20_'):
    visible = os.environ.get('NVIDIA_VISIBLE_DEVICES', '')
    if visible not in ALLOWED:
        raise RuntimeError('Only the explicitly authorized physical GPU 0/1 may be used')
    used = int(subprocess.check_output(['nvidia-smi', '--id=' + visible, '--query-gpu=memory.used',
                                        '--format=csv,noheader,nounits'], text=True).strip())
    if used >= 2048:
        raise RuntimeError(f'GPU already uses {used} MiB; leave jobs to the normal queue')
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.4, 0)
    db = connect(root)
    for name in models:
        job_id = job_prefix + name
        db.execute('BEGIN IMMEDIATE')
        job = db.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()
        if job is None:
            db.execute('ROLLBACK')
            raise RuntimeError('CAM job was not registered')
        # A Docker restart killed this worker's entire previous process tree.
        eligible = job['state'] == 'pending' or (job['state'] == 'running' and job['worker'] == worker)
        dependencies = json.loads(job['dependencies'])
        ready = all(db.execute('SELECT state FROM jobs WHERE id=?', (dep,)).fetchone()['state'] == 'completed'
                    for dep in dependencies)
        if not eligible or not ready:
            db.execute('COMMIT')
            print('SKIP', job_id, job['state'], flush=True)
            continue
        db.execute("UPDATE jobs SET state='running',worker=?,attempts=attempts+1,started=?,heartbeat=?,error=NULL WHERE id=?",
                   (worker, now(), time.time(), job_id))
        db.execute('COMMIT')
        status(root)
        stop = threading.Event()

        def heartbeat():
            connection = connect(root)
            while not stop.wait(10):
                connection.execute('UPDATE jobs SET heartbeat=? WHERE id=? AND worker=?', (time.time(), job_id, worker))
            connection.close()

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        error = None
        print(now(), 'START', job_id, 'gpu_fraction=0.4', flush=True)
        try:
            run(Path(root), Path(output), name)
        except Exception:
            error = traceback.format_exc()
            print(error, flush=True)
        finally:
            stop.set()
            thread.join()
            db.execute('UPDATE jobs SET state=?,finished=?,heartbeat=?,exit_code=?,error=? WHERE id=? AND worker=?',
                       ('failed' if error else 'completed', now(), time.time(), 1 if error else 0, error, job_id, worker))
            status(root)
        print(now(), 'FAILED' if error else 'COMPLETED', job_id, flush=True)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='/results')
    parser.add_argument('--out', required=True)
    parser.add_argument('--models', nargs='+', choices=list(LORA), required=True)
    parser.add_argument('--worker', required=True)
    parser.add_argument('--job-prefix', default='cam_comparison_20_')
    args = parser.parse_args()
    execute(args.root, args.out, args.models, args.worker, args.job_prefix)
