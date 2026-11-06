"""Prepare private templates; never invent road/onset/object annotations."""
import argparse
import json
from pathlib import Path

import numpy as np

from common import atomic_json,read_csv,write_csv


def make_pack(root):
    root=Path(root)
    manifest={r['id']:r for r in read_csv(root/'manifests/grouped.csv')}
    runs=['vivit_qkvo_mixed_s42','timesformer_qkvo_mixed_s42','videomae_out_mixed_s42']
    predictions={run:{r['id']:r for r in read_csv(root/'runs'/run/'external/predictions.csv')} for run in runs}
    keys=sorted(set.intersection(*(set(p) for p in predictions.values())))
    rng=np.random.default_rng(20261106)
    buckets={'positive':[],'negative':[]}
    for key in keys:
        buckets['positive' if int(manifest[key]['label']) else 'negative'].append(key)
    selected=set()
    # Procedural error coverage for EVERY model; sampling is independent of explanation quality.
    for run in runs:
        threshold=json.loads((root/'runs'/run/'metrics.json').read_text())['threshold_from_source_dev']
        for category in ['TP','FN','FP']:
            candidates=[]
            for key in keys:
                truth=int(manifest[key]['label'])
                predicted=float(predictions[run][key]['score'])>=threshold
                actual='TP' if truth and predicted else 'FN' if truth else 'FP' if predicted else 'TN'
                if actual==category:candidates.append(key)
            if candidates:selected.update(rng.choice(candidates,size=min(3,len(candidates)),replace=False))
    for bucket,quota in [('positive',40),('negative',20)]:
        members=[k for k in buckets[bucket] if k not in selected]
        already=sum(k in selected for k in buckets[bucket])
        need=max(0,min(quota-already,len(members)))
        if need:selected.update(rng.choice(members,size=need,replace=False))
    records=[]
    for key in sorted(selected):
        row=manifest[key]
        records.append({'id':key,'path':row['path'],'label':row['label'],'group':row['group'],
                        'road_traffic':'','lighting':'','occlusion':'','accident_onset_seconds':'',
                        'involved_vehicle_ids':'','annotation_reviewer':'','notes':''})
    out=root/'annotations'
    write_csv(out/'common_explanation_cases_private.csv',records)
    write_csv(out/'road_normal_review_private.csv',[
        {'id':k,'path':manifest[k]['path'],'road_traffic':'','reviewer':'','notes':''}
        for k in keys if not int(manifest[k]['label'])])
    write_csv(out/'onset_review_private.csv',[
        {'id':k,'path':manifest[k]['path'],'event_id':'','onset_seconds':'','end_seconds':'','reviewer':'','notes':''}
        for k in keys if int(manifest[k]['label'])])
    write_csv(out/'object_box_template.csv',[{'id':'','timestamp_seconds':'','object_id':'','x1':'','y1':'','x2':'','y2':'','accident_involved':'','reviewer':''}])
    atomic_json(out/'README.json',{'status':'manual_annotation_required','common_explanation_cases':len(records),
        'selection':'seed 20261106; up to three TP/FN/FP per model, then fill to 40 positive and 20 negative; no selection on explanation quality',
        'instructions':['Mark road traffic using scene content, independently of model scores',
                        'Mark collision onset and continuous vehicle boxes/IDs before computing event/tracking metrics',
                        'Keep camera names, paths and video assets private']})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',default='/results')
    make_pack(parser.parse_args().root)
