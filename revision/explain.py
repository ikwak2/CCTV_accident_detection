"""Common unannotated explanation/perturbation diagnostics, with private artifacts."""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from common import (MODELS,atomic_json,build_model,environment,pixels_from_frames,
                    read_csv,read_window,stable_seed,write_csv)
from explanations import gradient_times_input,token_rollout


def contact_sheet(frames,maps,path):
    indices=np.linspace(0,len(frames)-1,8).astype(int)
    norm=maps/(maps.max()+1e-12)
    sheet=np.zeros((2*224,4*224,3),np.uint8)
    for i,t in enumerate(indices):
        image=frames[t]
        h,w=image.shape[:2]
        resized=cv2.resize(image,(round(w*256/min(h,w)),round(h*256/min(h,w))))
        top,left=(resized.shape[0]-224)//2,(resized.shape[1]-224)//2
        rgb=resized[top:top+224,left:left+224]
        heat=cv2.applyColorMap((cv2.resize(norm[t],(224,224))*255).clip(0,255).astype(np.uint8),cv2.COLORMAP_JET)
        overlay=cv2.addWeighted(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),.65,heat,.35,0)
        sheet[i//4*224:(i//4+1)*224,i%4*224:(i%4+1)*224]=overlay
    cv2.imwrite(str(path),sheet)


def diagnostics(root,name,lora):
    root=Path(root)
    tag=f'{name}_{lora}_mixed_s42'
    out=root/'explanations'/tag
    out.mkdir(parents=True,exist_ok=True)
    model,processor,_=build_model(name,lora,root,42)
    checkpoint=torch.load(root/'runs'/tag/'best.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(checkpoint['model'],strict=True)
    model.cuda().eval()
    manifest={r['id']:r for r in read_csv(root/'manifests/grouped.csv')}
    cases=read_csv(root/'annotations/common_explanation_cases_private.csv')
    results=[]
    for i,case in enumerate(cases):
        row=manifest[case['id']]
        span=min(4.,float(row['duration']))
        start=max(0,float(row['duration'])-span)/2
        frames,times=read_window(row['path'],start,span,MODELS[name][1],float(row['fps']))
        pixels=pixels_from_frames(frames,processor).unsqueeze(0).cuda()
        attribution,logits=gradient_times_input(model,pixels)
        positive_signal=bool(attribution.sum()>0)
        flat=attribution.flatten()
        k=max(1,int(.2*flat.numel()))
        top_indices=flat.topk(k).indices
        rng=np.random.default_rng(stable_seed('perturbation',row['id']))
        random_indices=torch.tensor(rng.choice(flat.numel(),k,replace=False),device=pixels.device)
        probabilities=[]
        with torch.inference_mode():
            for indices in [top_indices,random_indices]:
                mask=torch.zeros_like(flat)
                mask[indices]=1
                mask=F.interpolate(mask.reshape(pixels.shape[1],1,14,14),size=(224,224),mode='nearest')
                masked=pixels*(1-mask[None])
                probabilities.append(float(model(masked).logits.softmax(-1)[0,1]))
        base_probability=float(logits.softmax(-1)[0,1])
        maps=attribution[0].cpu().numpy()
        identifier=f'case_{i:03d}'
        np.savez_compressed(out/(identifier+'_gradient_input.npz'),maps=maps,timestamps=np.array(times))
        contact_sheet(frames,maps,out/(identifier+'_gradient_input.jpg'))
        entry={'id':row['id'],'label':int(row['label']),'center_window_probability':base_probability,
               'positive_attribution':positive_signal,'top_20pct_probability':probabilities[0],
               'random_20pct_probability':probabilities[1],
               'top_minus_random_drop':probabilities[1]-probabilities[0] if positive_signal else None,
               'rollout':'not_requested_for_this_case'}
        if i<3 and name in ['vivit','videomae']:
            with torch.inference_mode():
                outputs=model(pixels,output_attentions=True)
                grid=(MODELS[name][1]//2,14,14)
                try:
                    rollout=token_rollout(outputs.attentions,grid,'cls' if name=='vivit' else 'mean')
                    upsampled=rollout.repeat_interleave(2,dim=1)[0].cpu().numpy()
                    np.savez_compressed(out/(identifier+'_rollout.npz'),maps=upsampled,timestamps=np.array(times))
                    contact_sheet(frames,upsampled,out/(identifier+'_rollout.jpg'))
                    entry['rollout']='model_layout_validated'
                except ValueError as exc:
                    entry['rollout']='not_assessed: '+str(exc)
                del outputs
        elif name=='timesformer':
            entry['rollout']='not_assessed: divided attention needs a separate validated propagation rule'
        results.append(entry)
        write_csv(out/'diagnostics.csv',results)
        if i%5==0:print('EXPLANATION',name,i+1,'/',len(cases),flush=True)
    atomic_json(out/'scope.json',{'status':'unannotated_center_window_diagnostics',
        'environment':environment(),'cases':len(cases),'method':'positive gradient times normalized input; accident-vs-normal logit margin',
        'perturbation':'20% spatiotemporal patches replaced by zero in normalized space; equal-size seeded random control',
        'limitations':['Center window may not contain the collision onset; inspect annotated times before localization claims',
                      'No localization hit rate or operator-grade pass/fail without annotations',
                      'Raw attention maps are not class probabilities or proof of faithfulness',
                      'Perturbation can create distribution shift; effect is diagnostic']})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',default='/results')
    parser.add_argument('--model',choices=list(MODELS),required=True)
    parser.add_argument('--lora',choices=['out','qkvo'],required=True)
    args=parser.parse_args()
    diagnostics(args.root,args.model,args.lora)
