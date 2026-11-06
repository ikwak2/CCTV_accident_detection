"""Initialize the durable queue when externally running preflight/audit finish."""
import argparse
import json
from pathlib import Path
import time

from experiment_queue import initialize,write_json,now


def main(root):
    root=Path(root)
    prerequisites=['pretrained/snapshots.json','preflight/preflight.json','manifests/audit.json']
    while True:
        missing=[p for p in prerequisites if not (root/p).exists()]
        if not missing:
            initialize(root)
            write_json(root/'bootstrap_status.json',{'state':'queue_initialized','updated':now()})
            return
        write_json(root/'bootstrap_status.json',{'state':'waiting_prerequisites','missing':missing,'updated':now()})
        print(now(),'WAIT',missing,flush=True)
        time.sleep(15)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',default='/results')
    main(parser.parse_args().root)
