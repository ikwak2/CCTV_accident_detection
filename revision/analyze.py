"""Aggregate completed runs; paired original-group bootstrap on common videos."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from common import atomic_json, read_csv, select_threshold, write_csv


def bootstrap_pair(a,b,dev_a=None,dev_b=None,repetitions=2000):
    a,b={r['id']:r for r in a},{r['id']:r for r in b}
    if set(a)!=set(b):
        raise ValueError('Paired comparison requires identical evaluated video sets')
    keys=sorted(a)
    if not keys:return None
    assert all(a[k]['label']==b[k]['label'] and a[k]['group']==b[k]['group'] for k in keys)
    grouped=defaultdict(list)
    for i,k in enumerate(keys):grouped[a[k]['group']].append(i)
    groups=sorted(grouped)
    y=np.array([int(a[k]['label']) for k in keys])
    p=np.array([float(a[k]['score']) for k in keys])
    q=np.array([float(b[k]['score']) for k in keys])
    rng=np.random.default_rng(20261106)
    calibration=None
    if dev_a is not None and dev_b is not None:
        da,db={r['id']:r for r in dev_a},{r['id']:r for r in dev_b}
        dk=sorted(set(da)&set(db))
        dy=np.array([int(da[k]['label']) for k in dk])
        dp=np.array([float(da[k]['score']) for k in dk])
        dq=np.array([float(db[k]['score']) for k in dk])
        dg=defaultdict(list)
        for i,k in enumerate(dk):dg[da[k]['group']].append(i)
        calibration=(dy,dp,dq,list(dg.values()))
        fixed_a,fixed_b=select_threshold(dy,dp),select_threshold(dy,dq)
    samples=[]
    for _ in range(repetitions):
        indices=np.concatenate([grouped[groups[g]] for g in rng.integers(len(groups),size=len(groups))])
        yy=y[indices]
        if len(np.unique(yy))<2:continue
        pa,pb=p[indices],q[indices]
        au_a,au_b=roc_auc_score(yy,pa),roc_auc_score(yy,pb)
        ap_a,ap_b=average_precision_score(yy,pa),average_precision_score(yy,pb)
        entry=[au_a,au_b,au_a-au_b,ap_a,ap_b,ap_a-ap_b]
        if calibration is not None:
            dy,dp,dq,dg=calibration
            di=np.concatenate([dg[i] for i in rng.integers(len(dg),size=len(dg))])
            ta,tb=select_threshold(dy[di],dp[di]),select_threshold(dy[di],dq[di])
            positive=yy==1
            conditional_a=float((pa[positive]<fixed_a).mean())
            conditional_b=float((pb[positive]<fixed_b).mean())
            nested_a=float((pa[positive]<ta).mean())
            nested_b=float((pb[positive]<tb).mean())
            entry += [conditional_a,conditional_b,conditional_a-conditional_b,nested_a,nested_b,nested_a-nested_b]
        samples.append(entry)
    values=np.array(samples)
    labels=['a_auroc','b_auroc','auroc_difference_a_minus_b','a_average_precision','b_average_precision','ap_difference_a_minus_b']
    if calibration is not None:
        labels += ['a_mdr_fixed_dev_threshold','b_mdr_fixed_dev_threshold','mdr_difference_fixed',
                   'a_mdr_nested_source_calibration','b_mdr_nested_source_calibration','mdr_difference_nested']
    return {'common_videos':len(keys),'original_or_candidate_groups':len(groups),'bootstrap_replicates':len(samples),
            'method':'paired cluster bootstrap, percentile 95% CI; conditional on trained models',
            'ci95':{label:np.percentile(values[:,i],[2.5,97.5]).tolist() for i,label in enumerate(labels)}}


def analyze(root,stage):
    root=Path(root)
    out=root/'analysis'/stage
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for file in sorted((root/'runs').glob('*/metrics.json')):
        if file.parent.name.startswith('legacy_'):continue
        result=json.loads(file.read_text())
        if 'test' not in result or 'external' not in result:continue
        config=json.loads((file.parent/'config.json').read_text())
        test=result['test']['at_dev_threshold']['all']
        external=result['external']['at_dev_threshold']['all']
        drop=(test['auroc']-external['auroc'])/test['auroc'] if test['auroc'] else None
        entry={k:config[k] for k in ['model','lora','condition','seed','manifest_sha256']}
        entry['evaluation_manifest_sha256']=result['evaluation_manifest_sha256']
        entry.update(run=file.parent.name,test_auroc=test['auroc'],external_auroc=external['auroc'],
                     external_ap=external['average_precision'],external_mdr=external['missed_detection_rate'],
                     external_fpr=external['fpr'],drop_ratio=drop)
        screened=result['external']['at_dev_threshold'].get('UCF_candidate_quarantined',{})
        entry.update(external_candidate_screened_auroc=screened.get('auroc'),
                     external_candidate_screened_ap=screened.get('average_precision'),
                     external_candidate_screened_mdr=screened.get('missed_detection_rate'),
                     external_candidate_screened_n=screened.get('n'))
        for boundary in [.1,.15,.2,.25]:entry[f'passes_drop_{boundary}']=int(drop<=boundary) if drop is not None else None
        rows.append(entry)
    if len({r['evaluation_manifest_sha256'] for r in rows})>1:
        raise ValueError('Cannot aggregate runs with different evaluation eligibility')
    write_csv(out/'run_metrics.csv',rows)
    comparisons={}
    configurations=['vivit_qkvo_mixed_s42','timesformer_qkvo_mixed_s42','videomae_out_mixed_s42']
    for i,a in enumerate(configurations):
        for b in configurations[i+1:]:
            for split in ['test','external']:
                fa=root/'runs'/a/split/'predictions.csv'
                fb=root/'runs'/b/split/'predictions.csv'
                if fa.exists() and fb.exists():
                    da=read_csv(root/'runs'/a/'dev/predictions.csv')
                    db=read_csv(root/'runs'/b/'dev/predictions.csv')
                    ra,rb=read_csv(fa),read_csv(fb)
                    comparisons[a+'__'+b+'__'+split]=bootstrap_pair(ra,rb,da,db)
                    if split=='external':
                        comparisons[a+'__'+b+'__external_candidate_screened']=bootstrap_pair(
                            [r for r in ra if not int(r['quarantined'])],
                            [r for r in rb if not int(r['quarantined'])],da,db)
    grouped=defaultdict(list)
    for row in rows:grouped[(row['model'],row['lora'],row['condition'])].append(row)
    seed_variation=[]
    for key,group in grouped.items():
        entry={'model':key[0],'lora':key[1],'condition':key[2],'training_seeds':[r['seed'] for r in group]}
        for measure in ['test_auroc','external_auroc','external_ap','external_mdr']:
            values=[r[measure] for r in group if r[measure] is not None]
            entry[measure+'_mean']=float(np.mean(values)) if values else None
            entry[measure+'_sd']=float(np.std(values,ddof=1)) if len(values)>1 else None
        seed_variation.append(entry)
    atomic_json(out/'summary.json',{'status':'automated_results_manual_data_annotations_and_field_requirements_pending',
         'paired_comparisons':comparisons,'training_seed_variation':seed_variation,
         'limitations':['Nested MDR uses source-dev group recalibration; no target-site calibration is claimed',
                       'Video-level FPR is not alerts per camera-hour',
                       'No road-only/onset/localization metrics without manual annotations',
                       'No deployment-ready certification is assigned']})
    print('ANALYSIS_COMPLETE',stage,'runs',len(rows),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',default='/results')
    parser.add_argument('--stage',choices=['baseline','all'],default='baseline')
    args=parser.parse_args()
    analyze(args.root,args.stage)
