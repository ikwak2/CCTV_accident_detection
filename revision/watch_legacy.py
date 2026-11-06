"""Host-side read-only Docker wait; persist the outcome of the pre-existing run."""
import argparse
import json
from pathlib import Path
import subprocess


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--container',default='orv-revision-legacy')
    args=parser.parse_args()
    result=subprocess.run(['docker','wait',args.container],capture_output=True,text=True)
    code=int(result.stdout.strip()) if result.returncode==0 else -1
    destination=Path(args.root)/'legacy_exit.json'
    temporary=destination.with_suffix('.tmp')
    temporary.write_text(json.dumps({'exit_code':code,'docker_wait_error':result.stderr}))
    temporary.replace(destination)
