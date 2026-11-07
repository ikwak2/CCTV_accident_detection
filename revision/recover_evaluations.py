"""Back up and requeue decode-failed evaluations after workers are stopped."""
import argparse
import json
import os
from pathlib import Path
import sqlite3
import sys

from audit_evaluation import evaluation_rows
from common import atomic_json
from experiment_queue import add_job, connect, now, status


def recover(root, workers_stopped=False):
    if not workers_stopped:
        raise RuntimeError('Stop both old worker containers before migrating the queue')
    root=Path(root)
    _,audit=evaluation_rows(root)
    db=connect(root)
    before=[dict(r) for r in db.execute('SELECT * FROM jobs ORDER BY id')]
    if any(r['state']=='running' and not r['id'].startswith('train_') for r in before):
        raise RuntimeError('Unexpected running non-training job; inspect before migration')
    backup=root/'recovery'/now().replace(':','').replace('+','_')
    backup.mkdir(parents=True,exist_ok=False)
    destination=sqlite3.connect(backup/'queue.sqlite')
    db.backup(destination)
    destination.close()
    atomic_json(backup/'jobs_before.json',before)
    resumed=[]
    db.execute('BEGIN IMMEDIATE')
    try:
        add_job(db,'evaluation_audit',6,[sys.executable,'-u','/app/revision/audit_evaluation.py','--root','/results'],
                ['prepare'],state='completed')
        for row in before:
            if row['id'].startswith('eval_') and row['state']!='completed':
                deps=json.loads(row['dependencies'])
                if 'evaluation_audit' not in deps:deps.append('evaluation_audit')
                db.execute("UPDATE jobs SET state='pending',worker=NULL,attempts=0,heartbeat=NULL,started=NULL,finished=NULL,exit_code=NULL,error=NULL,dependencies=? WHERE id=?",
                           (json.dumps(deps),row['id']))
                resumed.append(row['id'])
            elif row['id'].startswith('train_') and row['state'] in ['running','pending']:
                # Finish remaining training after the three first-seed baseline evals.
                db.execute("UPDATE jobs SET state='pending',worker=NULL,heartbeat=NULL,priority=30,error='resume after decoder eligibility deployment' WHERE id=?",(row['id'],))
        db.execute('COMMIT')
    except BaseException:
        db.execute('ROLLBACK')
        raise
    report={'created':now(),'source_commit':os.environ.get('ORV_SOURCE_COMMIT'),
            'backup_directory':str(backup),'requeued_evaluations':resumed,
            'evaluation_manifest_sha256':audit['evaluation_manifest_sha256'],
            'source_manifest_sha256':audit['source_manifest_sha256'],
            'weights_and_prediction_journals_unchanged':True}
    atomic_json(root/'recovery/latest.json',report)
    status(root)
    print('RECOVERY_READY',json.dumps(report),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',default='/results')
    parser.add_argument('--workers-stopped',action='store_true')
    args=parser.parse_args()
    recover(args.root,args.workers_stopped)
