"""Two single-GPU workers sharing a durable, dependency-aware SQLite queue."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time
import uuid


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(timespec='seconds')


def write_json(path,data):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2))
    os.replace(temp,path)


def connect(root):
    root=Path(root)
    root.mkdir(parents=True,exist_ok=True)
    connection=sqlite3.connect(root/'queue.sqlite',timeout=30,isolation_level=None)
    connection.row_factory=sqlite3.Row
    connection.execute('PRAGMA journal_mode=WAL')
    connection.execute('PRAGMA synchronous=FULL')
    connection.execute('''CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY, priority INTEGER NOT NULL, command TEXT NOT NULL,
        dependencies TEXT NOT NULL, state TEXT NOT NULL DEFAULT 'pending',
        worker TEXT, attempts INTEGER NOT NULL DEFAULT 0, heartbeat REAL,
        started TEXT, finished TEXT, exit_code INTEGER, error TEXT)''')
    return connection


def add_job(connection,job_id,priority,command,deps=(),state='pending'):
    connection.execute('INSERT OR IGNORE INTO jobs (id,priority,command,dependencies,state) VALUES (?,?,?,?,?)',
                       (job_id,priority,json.dumps(command),json.dumps(list(deps)),state))


def initialize(root):
    root=Path(root)
    db=connect(root)
    for marker,relative in [('prefetch','pretrained/snapshots.json'),('prepare','manifests/audit.json'),('preflight','preflight/preflight.json')]:
        if not (root/relative).exists():
            raise RuntimeError('External prerequisite not complete: '+marker)
        if marker=='prefetch' and len(json.loads((root/relative).read_text()))!=3:
            raise RuntimeError('Not all pretrained models have been downloaded')
        add_job(db,marker,0,[],state='completed')
    base=[sys.executable,'-u','/app/revision/run.py']
    legacy_state='completed' if (root/'runs/legacy_videomae_corrected/metrics.json').exists() else 'external_running' if (root/'legacy_external.json').exists() else 'pending'
    add_job(db,'legacy_videomae_corrected',5,base+['legacy','--out','/results/runs/legacy_videomae_corrected','--precision','fp32'],['prepare','preflight'],legacy_state)
    configurations=[('vivit','qkvo'),('timesformer','qkvo'),('videomae','out')]
    evaluations=[]
    baseline_evaluations=[]
    for condition in ['mixed','cadp']:
        for seed in [42,43,44]:
            for order,(model,lora) in enumerate(configurations):
                tag=f'{model}_{lora}_{condition}_s{seed}'
                priority=20+order if condition=='mixed' and seed==42 else 40+order if seed==42 else 60+order
                path='/results/runs/'+tag
                common=['--out',path,'--model',model,'--lora',lora,'--condition',condition,'--seed',str(seed)]
                add_job(db,'train_'+tag,priority,base+['train']+common,['prepare','preflight'])
                add_job(db,'eval_'+tag,priority+5,base+['evaluate']+common,['train_'+tag])
                evaluations.append('eval_'+tag)
                if condition=='mixed' and seed==42:
                    baseline_evaluations.append('eval_'+tag)
                    add_job(db,'bench_'+model,35,base+['benchmark','--out','/results/benchmarks/'+model,
                        '--model',model,'--lora',lora],['train_'+tag])
    for model,lora in [('vivit','out'),('videomae','qkvo')]:
        for seed in [42,43,44]:
            tag=f'{model}_{lora}_mixed_s{seed}'
            common=['--out','/results/runs/'+tag,'--model',model,'--lora',lora,'--condition','mixed','--seed',str(seed)]
            add_job(db,'train_'+tag,80,base+['train']+common,['prepare','preflight'])
            add_job(db,'eval_'+tag,85,base+['evaluate']+common,['train_'+tag])
            evaluations.append('eval_'+tag)
    add_job(db,'stats_baselines',36,[sys.executable,'-u','/app/revision/analyze.py','--root','/results','--stage','baseline'],baseline_evaluations)
    add_job(db,'annotation_templates',37,[sys.executable,'-u','/app/revision/annotation_pack.py','--root','/results'],baseline_evaluations)
    for model,lora in configurations:
        add_job(db,'explanation_diagnostics_'+model,50,[sys.executable,'-u','/app/revision/explain.py','--root','/results',
            '--model',model,'--lora',lora],['annotation_templates'])
    add_job(db,'stats_all',100,[sys.executable,'-u','/app/revision/analyze.py','--root','/results','--stage','all'],evaluations)
    write_json(root/'queue_protocol.json',{'created':now(),'worker_gpus':'server1 physical GPU 0 and GPU 1 only',
        'training_configurations':8,'training_runs':24,'source_commit':os.environ.get('ORV_SOURCE_COMMIT'),
        'manifest_sha256':json.loads((root/'manifests/audit.json').read_text())['manifest_sha256'],
        'manual_pending':['road-only normal annotation','accident onset annotation','object boxes and persistent IDs',
                          'review of conservative near-duplicate/camera candidate merges']})
    status(root)


def claim(db,worker):
    """Atomically claim one runnable job, never a failed dependency or active lease."""
    db.execute('BEGIN IMMEDIATE')
    try:
        # Other workers' leases are never stolen: their CUDA child could still be alive.
        # A worker reclaims ONLY its own jobs when its Docker process tree restarts.
        states={r['id']:r['state'] for r in db.execute('SELECT id,state FROM jobs')}
        candidates=db.execute("SELECT * FROM jobs WHERE state='pending' ORDER BY priority,id").fetchall()
        selected=None
        for row in candidates:
            if all(states.get(dep)=='completed' for dep in json.loads(row['dependencies'])):
                selected=row
                break
        if selected is not None:
            db.execute("UPDATE jobs SET state='running',worker=?,attempts=attempts+1,heartbeat=?,started=?,error=NULL WHERE id=?",
                       (worker,time.time(),now(),selected['id']))
        db.execute('COMMIT')
        return dict(selected) if selected else None
    except BaseException:
        db.execute('ROLLBACK')
        raise


def status(root):
    root=Path(root)
    db=connect(root)
    jobs=[dict(r) for r in db.execute('SELECT * FROM jobs ORDER BY priority,id')]
    states={r['id']:r['state'] for r in jobs}
    for row in jobs:
        row['command']=json.loads(row['command'])
        row['dependencies']=json.loads(row['dependencies'])
        row['blocked_by']=[dep for dep in row['dependencies'] if states.get(dep)!='completed']
    counts={state:sum(r['state']==state for r in jobs) for state in ['pending','running','external_running','completed','failed']}
    report={'updated':now(),'counts':counts,'jobs':jobs}
    write_json(root/'queue_status.json',report)
    lines=['# ORV server1 experiment queue','',f'Updated: {report["updated"]}','',
           '| State | Count |','|---|---:|']+[f'| {k} | {v} |' for k,v in counts.items()]
    lines+=['','| Job | State | Worker | Attempts | Blocked by |','|---|---|---|---:|---|']
    lines += [f'| {r["id"]} | {r["state"]} | {r["worker"] or ""} | {r["attempts"]} | {", ".join(r["blocked_by"])} |' for r in jobs]
    temp=root/('STATUS.'+uuid.uuid4().hex+'.tmp')
    temp.write_text('\n'.join(lines)+'\n')
    os.replace(temp,root/'STATUS.md')
    return report


def gpu_available():
    visible=os.environ.get('NVIDIA_VISIBLE_DEVICES','')
    if not visible.startswith('GPU-') or ',' in visible:
        raise RuntimeError('Worker requires exactly one explicit GPU UUID')
    result=subprocess.run(['nvidia-smi','--id='+visible,'--query-gpu=memory.used','--format=csv,noheader,nounits'],
                          capture_output=True,text=True,check=True)
    return int(result.stdout.strip())<768


def worker(root,worker_name):
    root=Path(root)
    db=connect(root)
    # This worker's previous process tree was killed on container restart.
    recover_worker(db,worker_name)
    active=None
    stopping=False

    def stop(signum,frame):
        nonlocal stopping
        stopping=True
        if active is not None:
            active.terminate()
    signal.signal(signal.SIGTERM,stop)
    signal.signal(signal.SIGINT,stop)
    while not stopping:
        refresh_external(db,root)
        if (root/'PAUSE').exists() or not gpu_available():
            write_json(root/f'{worker_name}.json',{'state':'waiting_for_gpu_or_pause','updated':now()})
            time.sleep(15)
            continue
        job=claim(db,worker_name)
        if job is None:
            status(root)
            write_json(root/f'{worker_name}.json',{'state':'idle_or_waiting_dependencies','updated':now()})
            time.sleep(15)
            continue
        job_id=job['id']
        command=json.loads(job['command'])
        logfile=root/'logs'/(job_id+'.log')
        logfile.parent.mkdir(parents=True,exist_ok=True)
        print(now(),'START',job_id,flush=True)
        write_json(root/f'{worker_name}.json',{'state':'running','job':job_id,'updated':now()})
        with open(logfile,'a',buffering=1) as log:
            log.write('\n'+now()+' ATTEMPT '+str(job['attempts']+1)+'\n')
            active=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,cwd='/app')
            while active.poll() is None:
                db.execute('UPDATE jobs SET heartbeat=? WHERE id=? AND worker=?',(time.time(),job_id,worker_name))
                status(root)
                time.sleep(5)
            code=active.returncode
        active=None
        attempts=db.execute('SELECT attempts FROM jobs WHERE id=?',(job_id,)).fetchone()[0]
        state='completed' if code==0 else 'pending' if attempts<2 or stopping else 'failed'
        tail=''
        if code:
            tail=logfile.read_text(errors='replace')[-4000:]
        # OOM fallback preserves effective batch and is recorded in the next config.
        if code and 'out of memory' in tail.lower() and 'train' in command and '--microbatch' not in command:
            command+=['--microbatch','1']
            state='pending'
            db.execute('UPDATE jobs SET command=? WHERE id=?',(json.dumps(command),job_id))
        db.execute('UPDATE jobs SET state=?,finished=?,exit_code=?,error=?,heartbeat=? WHERE id=?',
                   (state,now(),code,tail or None,time.time(),job_id))
        status(root)
        print(now(),state.upper(),job_id,'exit',code,flush=True)


def recover_worker(db,worker_name):
    """Only call after this worker's previous process tree is known to have died."""
    db.execute("UPDATE jobs SET state='pending',worker=NULL,error='worker restarted' WHERE state='running' AND worker=?",(worker_name,))


def refresh_external(db,root):
    root=Path(root)
    if (root/'runs/legacy_videomae_corrected/metrics.json').exists():
        db.execute("UPDATE jobs SET state='completed',finished=? WHERE id='legacy_videomae_corrected' AND state='external_running'",(now(),))
    elif (root/'legacy_exit.json').exists():
        result=json.loads((root/'legacy_exit.json').read_text())
        db.execute("UPDATE jobs SET state='pending',error=? WHERE id='legacy_videomae_corrected' AND state='external_running'",
                   ('external legacy run failed or did not produce metrics: '+str(result),))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['init','worker','status'])
    parser.add_argument('--root',default='/results')
    parser.add_argument('--worker',default=os.environ.get('ORV_WORKER','worker'))
    args=parser.parse_args()
    if args.command=='init':initialize(args.root)
    elif args.command=='worker':worker(args.root,args.worker)
    else:print(json.dumps(status(args.root),ensure_ascii=False,indent=2))
