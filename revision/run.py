"""Preflight, strict legacy evaluation, grouped training, evaluation and benchmarks."""
import argparse
import contextlib
import json
import os
from pathlib import Path
import time

import av
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from common import (MODELS, ClipDataset, atomic_json, build_model, environment, metrics,
                    patch_videomae_projection_calls, pixels_from_frames, read_csv, read_window,
                    seed_all, select_threshold, sha256, write_csv)
from audit_evaluation import PROTOCOL, evaluation_rows


def autocast(precision):
    return torch.autocast('cuda',dtype=torch.bfloat16) if precision=='bf16' else contextlib.nullcontext()


def rows_for(root):
    return read_csv(Path(root)/'manifests/grouped.csv')


def preflight(root, out):
    from transformers import VideoMAEForVideoClassification
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    snapshots=json.loads((Path(root)/'pretrained/snapshots.json').read_text())
    base=VideoMAEForVideoClassification.from_pretrained(snapshots['videomae']['path'],attn_implementation='sdpa').cuda().eval()
    x=torch.randn(1,16,3,224,224,device='cuda')
    with torch.no_grad():
        before=base(x).logits
        patch_videomae_projection_calls(base)
        after=base(x).logits
    difference=float((before-after).abs().max())
    assert torch.allclose(before,after,atol=2e-5,rtol=2e-5), difference
    del base,x,before,after
    torch.cuda.empty_cache()
    reports=[]
    for name,lora in [('videomae','out'),('videomae','qkvo'),('vivit','qkvo'),('vivit','out'),('timesformer','qkvo')]:
        model,processor,snapshot=build_model(name,lora,root)
        model.cuda().train()
        # Exercise the actual preprocessing contract as well as the backward path.
        frames=[np.full((240,320,3),50+i*3,dtype=np.uint8) for i in range(MODELS[name][1])]
        pixels=pixels_from_frames(frames,processor).unsqueeze(0).cuda()
        assert tuple(pixels.shape)==(1,MODELS[name][1],3,224,224)
        with autocast('bf16'):
            loss=nn.functional.cross_entropy(model(pixels).logits,torch.ones(1,dtype=torch.long,device='cuda'))
        loss.backward()
        gradients={n:float(p.grad.float().norm()) if p.grad is not None else None
                   for n,p in model.named_parameters() if 'lora_B' in n}
        assert gradients and all(v is not None and v>0 for v in gradients.values()), gradients
        assert all(p.grad is None for n,p in model.named_parameters() if '.original_module.' in n)
        report={'model':name,'lora':lora,'loss':float(loss),'trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),
                'total':sum(p.numel() for p in model.parameters()),'lora_gradient_norms':gradients,
                'snapshot':snapshot,'pixel_shape':list(pixels.shape)}
        print('PREFLIGHT_PASS',name,lora,'trainable',report['trainable'],flush=True)
        reports.append(report)
        del model,pixels,loss
        torch.cuda.empty_cache()
    # Verify checkpoint head compatibility on the real legacy model.
    model,_,_=build_model('videomae','out',root,legacy='/data/main/vivit/videomae_accident_classifier.pth')
    legacy_head=[list(p.shape) for n,p in model.named_parameters() if 'modules_to_save.default' in n]
    del model
    atomic_json(out/'preflight.json',{'environment':environment(),'videomae_forward_max_abs_difference':difference,
                                    'legacy_head_shapes':legacy_head,'configurations':reports})


@torch.inference_mode()
def score_rows(model,processor,name,rows,out,precision='bf16',window=4.,n_clips=5,resume=True):
    """Incremental per-video scores and exact sampling timestamps; no label-based sampling."""
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    journal=out/'predictions_private.jsonl'
    records={}
    if resume and journal.exists():
        for line in journal.read_text().splitlines():
            try:
                row=json.loads(line)
                records[row['id']]=row
            except json.JSONDecodeError:
                pass
    for row in rows:
        if row['id'] not in records:
            continue
        cached=records[row['id']]
        duration,fps=float(row['duration']),float(row['fps'])
        span=min(window,duration)
        expected=[np.linspace(start,start+max(0.,span-1/fps),MODELS[name][1])
                  for start in np.linspace(0,max(0,duration-span),n_clips)]
        if (any(str(cached[k])!=str(row[k]) for k in ['dataset','category','label','group','split','quarantined'])
            or np.shape(cached.get('sample_timestamps'))!=np.shape(expected)
            or not np.allclose(cached['sample_timestamps'],expected,rtol=0,atol=1e-8)
            or not np.isfinite(cached['score']) or not 0<=cached['score']<=1):
            raise RuntimeError('Cached prediction does not match evaluation contract: '+row['id'])
    model.eval()
    with open(journal,'a') as f:
        for i,row in enumerate(rows):
            if row['id'] in records:
                continue
            duration,fps=float(row['duration']),float(row['fps'])
            span=min(window,duration)
            starts=np.linspace(0,max(0,duration-span),n_clips)
            probabilities,timestamps=[],[]
            for start in starts:
                try:
                    frames,times=read_window(row['path'],float(start),span,MODELS[name][1],fps)
                except Exception as exc:
                    raise RuntimeError(f"Decode failed for {row['id']} at start={start}, frames={MODELS[name][1]}") from exc
                pixels=pixels_from_frames(frames,processor).unsqueeze(0).cuda()
                with autocast(precision):
                    logits=model(pixels).logits
                probabilities.append(float(logits.float().softmax(-1)[0,1]))
                timestamps.append(times)
            result={k:row[k] for k in ['id','dataset','category','label','group','split','quarantined']}
            result.update(label=int(row['label']),quarantined=int(row['quarantined']),
                          score=float(np.mean(probabilities)),clip_scores=probabilities,sample_timestamps=timestamps)
            records[row['id']]=result
            f.write(json.dumps(result,ensure_ascii=False)+'\n')
            f.flush()
            if i%20==0:
                print('SCORED',name,i+1,'/',len(rows),flush=True)
                atomic_json(out/'progress.json',{'scored':len(records),'total':len(rows),'time':time.time()})
    ordered=[records[r['id']] for r in rows]
    write_csv(out/'predictions.csv',[{k:r[k] for k in ['id','dataset','category','label','group','split','quarantined','score']} for r in ordered])
    atomic_json(out/'progress.json',{'state':'complete','scored':len(ordered),'total':len(rows),'time':time.time()})
    return ordered


def summarize_predictions(records,threshold):
    groups={'all':records}
    for dataset in ['CADP','UTIC','UCF']:
        selected=[r for r in records if r['dataset']==dataset]
        if selected:
            groups[dataset]=selected
            for label in [0,1]:
                subset=[r for r in selected if int(r['label'])==label]
                if subset:groups[dataset+':'+str(label)]=subset
    if any(r['dataset']=='UCF' for r in records):
        groups['UCF_candidate_quarantined']=[r for r in records if r['dataset']=='UCF' and not int(r['quarantined'])]
    return {k:metrics([r['label'] for r in v],[r['score'] for r in v],threshold) for k,v in groups.items() if v}


def evaluate(root,run_dir,name,lora,seed,precision='bf16',legacy=False):
    run_dir=Path(run_dir)
    if legacy and not (Path(root)/'manifests/grouped.csv').exists():
        # This diagnostic can run while the independent, full content audit continues.
        from prepare import list_data
        rows=[]
        failures=[]
        for row in list_data('/data'):
            if row['dataset']!='UCF':continue
            try:
                with av.open(row['path']) as container:
                    stream=container.streams.video[0]
                    row['fps']=float(stream.average_rate or stream.guessed_rate or 0)
                    row['duration']=float(stream.duration*stream.time_base) if stream.duration else float(container.duration or 0)/av.time_base
                assert row['fps']>0 and row['duration']>0
                row.update(group=row['source_group'],split='external',quarantined=0)
                rows.append(row)
            except Exception as exc:
                failures.append({'id':row['id'],'error':str(exc)})
        atomic_json(run_dir/'metadata_failures_private.json',failures)
        atomic_json(run_dir/'diagnostic_scope.json',{'curation':'full content audit still pending',
                                                  'videos':len(rows),'metadata_failures':len(failures)})
    else:
        rows=rows_for(root)
    if legacy:
        ckpt='/data/main/vivit/videomae_accident_classifier.pth'
        model,processor,snapshot=build_model(name,lora,root,seed,legacy=ckpt)
        selected=[r for r in rows if r['dataset']=='UCF']
        model.cuda()
        records=score_rows(model,processor,name,selected,run_dir/'external',precision)
        diag_threshold=select_threshold([r['label'] for r in records],[r['score'] for r in records])
        report={'status':'legacy_checkpoint_diagnostic_not_clean_split_training',
                'checkpoint_sha256':sha256(ckpt),'model_snapshot':snapshot,'environment':environment(),
                'sampling':'five equally-spaced 4-second windows, shared seconds across models',
                'metrics_at_0.5':summarize_predictions(records,.5),
                'metrics_test_selected_diagnostic':summarize_predictions(records,diag_threshold)}
        atomic_json(run_dir/'metrics.json',report)
        return
    rows,audit=evaluation_rows(root)
    checkpoint=run_dir/'best.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    config=saved['config']
    if config['manifest_sha256']!=audit['source_manifest_sha256']:
        raise RuntimeError('Checkpoint training manifest does not match audited source')
    if (config['model'],config['lora'],config['seed'])!=(name,lora,seed):
        raise RuntimeError('Checkpoint model/LoRA/seed does not match requested evaluation')
    checkpoint_hash=sha256(checkpoint)
    contract={'checkpoint_sha256':checkpoint_hash,'source_manifest_sha256':audit['source_manifest_sha256'],
              'evaluation_manifest_sha256':audit['evaluation_manifest_sha256'],
              'decoder_code_sha256':audit['decoder_code_sha256'],'sampling':PROTOCOL,'precision':precision}
    contract_path=run_dir/'evaluation_contract.json'
    if contract_path.exists():
        if json.loads(contract_path.read_text())!=contract:
            raise RuntimeError('Evaluation contract changed; archive old scores before rerunning')
    else:
        # The original sampler/preprocessor is unchanged. Only eligibility changes.
        has_cached_scores=any(run_dir.glob('*/predictions_private.jsonl'))
        if has_cached_scores and (config['environment']['code_sha256']['common.py']!=audit['decoder_code_sha256']
                                  or config['precision']!=precision):
            raise RuntimeError('Cannot adopt cached scores from a different decoder/preprocessor or precision')
        atomic_json(contract_path,contract)
        atomic_json(run_dir/'evaluation_resume.json',{'adopted_existing_scores':has_cached_scores,
                    'reason':'same checkpoint, sampler/preprocessor hash, precision and verified per-video timestamps',
                    'evaluation_manifest_sha256':audit['evaluation_manifest_sha256']})
    model,processor,snapshot=build_model(name,lora,root,seed)
    model.load_state_dict(saved['model'],strict=True)
    model.cuda()
    dev=score_rows(model,processor,name,[r for r in rows if r['split']=='dev'],run_dir/'dev',precision)
    threshold=select_threshold([r['label'] for r in dev],[r['score'] for r in dev])
    result={'status':'grouped_split_with_automated_content_screen_manual_review_pending',
            'checkpoint_sha256':checkpoint_hash,'manifest_sha256':audit['source_manifest_sha256'],
            'evaluation_manifest_sha256':audit['evaluation_manifest_sha256'],'evaluation_audit':audit,
            'model_snapshot':snapshot,'environment':environment(),'calibration':'source dev; five windows; FPR budget 0.1',
            'threshold_from_source_dev':threshold,'dev':summarize_predictions(dev,threshold)}
    for split in ['test','external']:
        predictions=score_rows(model,processor,name,[r for r in rows if r['split']==split],run_dir/split,precision)
        result[split]={'at_dev_threshold':summarize_predictions(predictions,threshold),
                       'at_0.5':summarize_predictions(predictions,.5)}
        atomic_json(run_dir/'metrics.json',result)


def train(root,out,name,lora,condition,seed,max_epochs=50,microbatch=2,effective_batch=24,workers=4,precision='bf16'):
    seed_all(seed)
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    manifest=Path(root)/'manifests/grouped.csv'
    rows=rows_for(root)
    train_rows=[r for r in rows if r['split']=='train' and (condition=='mixed' or r['dataset']=='CADP')]
    dev_rows=[r for r in rows if r['split']=='dev']
    # A fixed CADP dev loss selects checkpoints for BOTH conditions. UTIC dev calibrates later.
    dev_loss_rows=[r for r in dev_rows if r['dataset']=='CADP']
    assert {int(r['label']) for r in train_rows}=={0,1}
    assert effective_batch%microbatch==0
    model,processor,snapshot=build_model(name,lora,root,seed)
    model.cuda()
    train_data=ClipDataset(train_rows,processor,MODELS[name][1],training=True,seed=seed)
    dev_data=ClipDataset(dev_loss_rows,processor,MODELS[name][1],training=False,seed=seed)
    generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(train_data,batch_size=microbatch,shuffle=True,num_workers=workers,pin_memory=True,
                      persistent_workers=False,generator=generator)
    dev_loader=DataLoader(dev_data,batch_size=microbatch,shuffle=False,num_workers=workers,pin_memory=True)
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.Adam(parameters,lr=5e-5)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=3)
    config={'model':name,'lora':lora,'condition':condition,'seed':seed,'max_epochs':max_epochs,
            'microbatch':microbatch,'effective_batch':effective_batch,'precision':precision,
            'window_seconds':4.,'train_n':len(train_rows),'dev_selection_n':len(dev_loss_rows),
            'class_weighting':'unweighted cross entropy; class/source count confounding reported',
            'dev_selection':'fixed CADP-only source-dev, one center window; source-dev calibration uses five windows',
            'manifest_sha256':sha256(manifest),'snapshot':snapshot,'environment':environment()}
    atomic_json(out/'config.json',config)
    start,best,stale=0,float('inf'),0
    last=out/'last.pt'
    if last.exists():
        checkpoint=torch.load(last,map_location='cpu',weights_only=False)
        if checkpoint['config']['manifest_sha256']!=config['manifest_sha256']:
            raise RuntimeError('Refusing to resume a different split')
        model.load_state_dict(checkpoint['model'],strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        generator.set_state(checkpoint['loader_rng'])
        torch.set_rng_state(checkpoint['torch_rng'])
        torch.cuda.set_rng_state(checkpoint['cuda_rng'])
        start,best,stale=checkpoint['epoch']+1,checkpoint['best'],checkpoint['stale']
    accum=effective_batch//microbatch
    history=[]
    for epoch in range(start,max_epochs):
        if stale>=5:break
        since=time.monotonic()
        train_data.epoch=epoch
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss,count=0.,0
        for step,(pixels,labels,_) in enumerate(loader):
            pixels,labels=pixels.cuda(non_blocking=True),labels.cuda(non_blocking=True)
            # Last accumulation group is normalized by its actual sample count.
            group_start=(step//accum)*effective_batch
            group_n=min(effective_batch,len(train_data)-group_start)
            with autocast(precision):
                loss=nn.functional.cross_entropy(model(pixels).logits,labels)
            (loss*len(labels)/group_n).backward()
            total_loss+=float(loss.detach())*len(labels)
            count+=len(labels)
            if (step+1)%accum==0 or step+1==len(loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if step%100==0:
                print('TRAIN',name,condition,seed,'epoch',epoch+1,'step',step,'/',len(loader),'loss',float(loss.detach()),flush=True)
                atomic_json(out/'progress.json',{'state':'training','epoch':epoch+1,'step':step,'steps':len(loader),'time':time.time()})
        model.eval()
        dev_loss,dev_n=0.,0
        with torch.inference_mode():
            for pixels,labels,_ in dev_loader:
                pixels,labels=pixels.cuda(non_blocking=True),labels.cuda(non_blocking=True)
                with autocast(precision):
                    loss=nn.functional.cross_entropy(model(pixels).logits,labels)
                dev_loss+=float(loss)*len(labels)
                dev_n+=len(labels)
        value=dev_loss/dev_n
        scheduler.step(value)
        if value<best:
            best,stale=value,0
            torch.save({'model':model.state_dict(),'config':config,'epoch':epoch},out/'best.pt.tmp')
            os.replace(out/'best.pt.tmp',out/'best.pt')
        else:stale+=1
        entry={'epoch':epoch+1,'train_loss':total_loss/count,'dev_loss':value,'best_dev_loss':best,
               'stale':stale,'seconds':time.monotonic()-since,'peak_vram_gb':torch.cuda.max_memory_allocated()/2**30}
        history.append(entry)
        with open(out/'history.jsonl','a') as f:f.write(json.dumps(entry)+'\n')
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                    'config':config,'epoch':epoch,'best':best,'stale':stale,
                    'loader_rng':generator.get_state(),'torch_rng':torch.get_rng_state(),
                    'cuda_rng':torch.cuda.get_rng_state()},out/'last.pt.tmp')
        os.replace(out/'last.pt.tmp',last)
        print('EPOCH_DONE',json.dumps(entry),flush=True)
    atomic_json(out/'training_complete.json',{'best_dev_loss':best,'epochs':start+len(history),'stale':stale})


def benchmark(root,out,name,lora):
    model,processor,snapshot=build_model(name,lora,root)
    model.cuda().eval()
    sample=next(r for r in rows_for(root) if r['dataset']=='CADP' and r['split']=='dev')
    span=min(4.,float(sample['duration']))
    frames,_=read_window(sample['path'],0,span,MODELS[name][1],float(sample['fps']))
    pixels=pixels_from_frames(frames,processor).unsqueeze(0).cuda()
    samples=[]
    with torch.inference_mode(),autocast('bf16'):
        for i in range(30):
            torch.cuda.synchronize()
            begin=time.perf_counter()
            model(pixels)
            torch.cuda.synchronize()
            if i>=5:samples.append(time.perf_counter()-begin)
    end_to_end=[]
    with torch.inference_mode():
        for _ in range(10):
            begin=time.perf_counter()
            frames,_=read_window(sample['path'],0,span,MODELS[name][1],float(sample['fps']))
            tensor=pixels_from_frames(frames,processor).unsqueeze(0).cuda()
            with autocast('bf16'):model(tensor)
            torch.cuda.synchronize()
            end_to_end.append(time.perf_counter()-begin)
    atomic_json(Path(out)/'benchmark.json',{'model':name,'snapshot':snapshot,'environment':environment(),
        'scope':'classification only; explanation/YOLO/concurrent streams not included',
        'precision':'bf16','batch':1,'forward_seconds_p50_p95':np.percentile(samples,[50,95]).tolist(),
        'decode_preprocess_forward_seconds_p50_p95':np.percentile(end_to_end,[50,95]).tolist(),
        'max_allocated_vram_gb':torch.cuda.max_memory_allocated()/2**30})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['preflight','train','evaluate','legacy','benchmark'])
    parser.add_argument('--root',default='/results')
    parser.add_argument('--out',required=True)
    parser.add_argument('--model',choices=list(MODELS),default='videomae')
    parser.add_argument('--lora',choices=['out','qkvo'],default='out')
    parser.add_argument('--condition',choices=['mixed','cadp'],default='mixed')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--max-epochs',type=int,default=50)
    parser.add_argument('--microbatch',type=int,default=2)
    parser.add_argument('--effective-batch',type=int,default=24)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--precision',choices=['bf16','fp32'],default='bf16')
    args=parser.parse_args()
    if args.command=='preflight':preflight(args.root,args.out)
    elif args.command=='train':train(args.root,args.out,args.model,args.lora,args.condition,args.seed,
                                    args.max_epochs,args.microbatch,args.effective_batch,args.workers,args.precision)
    elif args.command in ['evaluate','legacy']:evaluate(args.root,args.out,args.model,args.lora,args.seed,args.precision,args.command=='legacy')
    else:benchmark(args.root,args.out,args.model,args.lora)


if __name__=='__main__':
    main()
