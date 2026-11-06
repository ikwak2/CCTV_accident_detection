"""Private on-server decoding, content screening, and grouped data manifests."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import time

import av
import cv2
import numpy as np
import torch
from huggingface_hub import model_info, snapshot_download

from common import MODELS, atomic_json, sha256, write_csv


def prefetch(root):
    root = Path(root)
    snapshots = {}
    for name, (model_id, _, _) in MODELS.items():
        info = model_info(model_id)
        path = snapshot_download(model_id, revision=info.sha,
            allow_patterns=['config.json', 'preprocessor_config.json', 'model.safetensors', 'pytorch_model.bin'])
        snapshots[name] = {'model_id': model_id, 'revision': info.sha, 'path': path}
        print('PREFETCH', name, info.sha, flush=True)
        atomic_json(root/'pretrained/snapshots.json', snapshots)
    return snapshots


class UnionFind:
    def __init__(self, keys):
        self.parents = {k:k for k in keys}

    def find(self, key):
        parent = self.parents[key]
        if parent != key:
            self.parents[key] = self.find(parent)
        return self.parents[key]

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parents[max(a,b)] = min(a,b)


def list_data(data):
    data = Path(data)
    rows = []
    for category, label in [('accident',1), ('after-accident',1), ('non-accident',0)]:
        for path in sorted((data/'DATA/cctv_data/cctv_seg'/category).glob('*.mp4')):
            source = path.stem.split('_seg')[0]
            rows.append({'id':'cadp/'+category+'/'+path.name, 'dataset':'CADP', 'category':category,
                         'path':str(path), 'label':label, 'source_group':'cadp:'+source, 'role':'train'})
    for folder in ['record', 'record_final']:
        for path in sorted((data/'DATA/utic/utic_data'/folder).glob('*.mp4')):
            match = re.match(r'cctv_(\d+)_', path.name)
            group = 'utic-number:'+match.group(1) if match else 'utic-name:'+path.stem
            rows.append({'id':'utic/'+folder+'/'+path.name, 'dataset':'UTIC', 'category':'utic-'+folder,
                         'path':str(path), 'label':0, 'source_group':group, 'role':'train'})
    ucf = data/'DATA/ucf_data/UCF_Crimes/Videos'
    for directory in sorted(ucf.iterdir()):
        if not directory.is_dir() or (directory.name != 'RoadAccidents' and 'normal' not in directory.name.lower()):
            continue
        for path in sorted(directory.rglob('*')):
            if path.suffix.lower() not in ['.mp4','.avi','.mkv','.mov']:
                continue
            stem = re.sub(r'(_x264)?$', '', path.stem)
            rows.append({'id':'ucf/'+str(path.relative_to(ucf)), 'dataset':'UCF', 'category':directory.name,
                         'path':str(path), 'label':int(directory.name=='RoadAccidents'),
                         'source_group':'ucf:'+stem, 'role':'external'})
    # Original CADP videos provide whole-scene fingerprints for cross-dataset screening.
    for path in sorted((data/'DATA/cctv_data/cctv_videos').glob('*.mp4')):
        rows.append({'id':'cadp-original/'+path.name, 'dataset':'CADP-original', 'category':'original',
                     'path':str(path), 'label':-1, 'source_group':'cadp:'+path.stem, 'role':'fingerprint'})
    assert sum(r['dataset']=='CADP' for r in rows) > 2000
    assert sum(r['dataset']=='UCF' and r['label']==1 for r in rows) > 100
    return rows


def scan_one(row):
    row = dict(row)
    try:
        row['file_sha256'] = sha256(row['path'])
        signatures = []
        with av.open(row['path']) as container:
            stream = container.streams.video[0]
            stream.thread_type = 'SLICE'
            stream.codec_context.thread_count = 1
            fps = float(stream.average_rate or stream.guessed_rate or 0)
            duration = float(stream.duration*stream.time_base) if stream.duration else float(container.duration or 0)/av.time_base
            if fps <= 0 or duration <= 0:
                raise ValueError('Missing positive fps/duration')
            row.update(fps=fps, duration=duration, width=stream.width, height=stream.height)
            # Metadata checks for all clips; rich fingerprints for original/external/UTIC.
            fractions = [.05,.5,.95] if row['dataset']=='CADP' else [.02,.1,.2,.35,.5,.65,.8,.9,.98]
            base = float((stream.start_time or 0)*stream.time_base)
            for fraction in fractions:
                target = min(duration-1/fps, max(0, duration*fraction))
                container.seek(max(0,int((target+base)/float(stream.time_base))), stream=stream, backward=True)
                image = None
                for frame in container.decode(stream):
                    if frame.pts is not None and float(frame.pts*stream.time_base)-base >= target:
                        image = frame.to_ndarray(format='gray')
                        break
                if image is None:
                    raise ValueError(f'Cannot decode requested location {fraction}')
                small = cv2.resize(image, (32,32), interpolation=cv2.INTER_AREA).astype(np.float32)
                if small.std() < 8:
                    continue
                # Exclude DC; retain all 64 slots for an exact integer Hamming metric.
                dct = cv2.dct(small)[:8,:8].reshape(-1)
                signature = (dct > np.median(dct[1:])).astype(np.uint8)
                signature[0] = 0
                signatures.append({'fraction':fraction, 'bits':signature.tolist()})
        row.update(valid=True, error='', signatures=signatures)
    except Exception as exc:
        row.update(valid=False, error=f'{type(exc).__name__}: {exc}', signatures=[])
    return row


def group_split(rows):
    groups = sorted({r['group'] for r in rows})
    categories = sorted({r['category'] for r in rows})
    gi, ci = {k:i for i,k in enumerate(groups)}, {k:i for i,k in enumerate(categories)}
    counts = np.zeros((len(groups),len(categories)), dtype=int)
    for row in rows:
        counts[gi[row['group']], ci[row['category']]] += 1
    if len(groups) < 30:
        raise RuntimeError('Too few independent groups after content screening')
    cut1, cut2 = int(.7*len(groups)), int(.9*len(groups))
    target = np.array([.7,.2,.1])[:,None]*counts.sum(0)[None,:]
    rng, best = np.random.default_rng(42), None
    for _ in range(20000):
        order = rng.permutation(len(groups))
        chunks = [order[:cut1], order[cut1:cut2], order[cut2:]]
        actual = np.stack([counts[c].sum(0) for c in chunks])
        loss = float(np.square((actual-target)/np.maximum(target,1)).sum())
        if best is None or loss < best[0]:
            best = (loss,[c.copy() for c in chunks])
    assignment = {groups[i]:s for s,indices in zip(['train','dev','test'],best[1]) for i in indices}
    for row in rows:
        row['split'] = assignment[row['group']]
    for split in ['train','dev','test']:
        assert {int(r['label']) for r in rows if r['split']==split} == {0,1}
    return rows


def prepare(root, data, workers=6):
    root = Path(root)
    directory = root/'manifests'
    directory.mkdir(parents=True, exist_ok=True)
    listed = list_data(data)
    cache_path = directory/'scanned_private.jsonl'
    cache = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            try:
                item = json.loads(line)
                cache[item['id']] = item
            except json.JSONDecodeError:
                pass
    pending = [r for r in listed if r['id'] not in cache]
    with open(cache_path,'a') as log, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(scan_one,r):r['id'] for r in pending}
        for i, future in enumerate(as_completed(futures)):
            item = future.result()
            cache[item['id']] = item
            log.write(json.dumps(item,ensure_ascii=False)+'\n')
            log.flush()
            if i%50 == 0:
                print('SCAN',len(cache),'/',len(listed),'last_valid',item['valid'],flush=True)
                atomic_json(directory/'prepare_progress.json',{'scanned':len(cache),'total':len(listed),'time':time.time()})
    scanned = [cache[r['id']] for r in listed]
    valid = [r for r in scanned if r['valid']]
    uf = UnionFind({r['source_group'] for r in valid})
    hashes = defaultdict(list)
    for r in valid:
        hashes[r['file_sha256']].append(r)
    candidates = []
    quarantine = set()
    for digest, same in hashes.items():
        for a in same[1:]:
            uf.union(same[0]['source_group'],a['source_group'])
        train_labels = {r['label'] for r in same if r['role']=='train'}
        if len(train_labels)>1:
            quarantine.update(r['id'] for r in same if r['role']=='train')
        if any(r['dataset'].startswith('CADP') or r['dataset']=='UTIC' for r in same):
            quarantine.update(r['id'] for r in same if r['dataset']=='UCF')
        if len(same)>1:
            candidates.append({'kind':'exact','ids':[r['id'] for r in same],'sha256':digest})
    # Near-duplicate screening on scene representatives; clip IDs inherit original grouping.
    representatives = [r for r in valid if r['dataset']!='CADP']
    bits, owners = [], []
    for i,row in enumerate(representatives):
        for sig in row['signatures']:
            bits.append(sig['bits'])
            owners.append(i)
    print('CONTENT_SCREEN',len(bits),'fingerprints',flush=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    matrix = torch.tensor(np.array(bits,dtype=np.float32)*2-1,device=device)
    owner = np.array(owners)
    pair_counts = Counter()
    pair_min = {}
    # FP32 is exact here: sums of sixty-four signed ones.
    for start in range(0,len(bits),256):
        distance = ((64-matrix[start:start+256] @ matrix.T)/2).round()
        ii,jj = torch.where(distance <= 8)
        distances = distance[ii,jj].cpu().numpy()
        ii,jj = ii.cpu().numpy()+start,jj.cpu().numpy()
        for a,b,d in zip(ii,jj,distances):
            if a>=b or owner[a]==owner[b]:
                continue
            x,y = sorted([int(owner[a]),int(owner[b])])
            pair_counts[(x,y)] += 1
            pair_min[(x,y)] = min(float(d),pair_min.get((x,y),64))
    for (i,j),count in pair_counts.items():
        a,b = representatives[i],representatives[j]
        if a['source_group']==b['source_group']:
            continue
        if count<2 and pair_min[(i,j)]>3:
            continue
        # Conservative grouping/quarantine of ALL identified candidates avoids selection by scores.
        uf.union(a['source_group'],b['source_group'])
        candidates.append({'kind':'perceptual_candidate','ids':[a['id'],b['id']],
                           'matching_frame_pairs':count,'min_phash_hamming':pair_min[(i,j)]})
    training_groups = {uf.find(r['source_group']) for r in valid if r['role']=='train'}
    for row in valid:
        row['group'] = uf.find(row['source_group'])
        if row['dataset']=='UCF' and row['group'] in training_groups:
            quarantine.add(row['id'])
    train = group_split([r for r in valid if r['role']=='train' and r['id'] not in quarantine])
    groups = {s:{r['group'] for r in train if r['split']==s} for s in ['train','dev','test']}
    assert not (groups['train']&groups['dev'] or groups['train']&groups['test'] or groups['dev']&groups['test'])
    for row in valid:
        if row['dataset']=='UCF':
            row['split']='external'
    clean_columns = ['id','dataset','category','label','group','source_group','split','path','fps','duration','width','height','file_sha256','quarantined']
    exported = []
    for row in train+[r for r in valid if r['dataset']=='UCF']:
        row['quarantined'] = int(row['id'] in quarantine)
        exported.append({k:row[k] for k in clean_columns})
    write_csv(directory/'grouped.csv',exported)
    atomic_json(directory/'content_candidates_private.json',candidates)
    atomic_json(directory/'excluded_private.json',[{k:v for k,v in r.items() if k!='signatures'} for r in scanned if not r['valid']])
    report = {'status':'automated_screen_complete_manual_event_camera_and_road_annotations_pending',
              'group_overlap':0,'valid_files':len(valid),'invalid_files':len(scanned)-len(valid),
              'quarantined_ids':len(quarantine),'content_candidates':len(candidates),
              'split_counts':{s:dict(Counter(r['dataset']+':'+str(r['label']) for r in train if r['split']==s)) for s in groups},
              'split_groups':{s:len(g) for s,g in groups.items()},
              'manifest_sha256':sha256(directory/'grouped.csv'),
              'limitations':['Perceptual matches are conservative candidates, not confirmed duplicates',
                            'Unidentified event/camera aliases can remain; manual review required for final claims',
                            'Full UCF and candidate-quarantined UCF metrics must both be retained',
                            'Road-only normal/onset/object annotations are not yet available']}
    atomic_json(directory/'audit.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prefetch','prepare'])
    parser.add_argument('--root',default='/results')
    parser.add_argument('--data',default='/data')
    args=parser.parse_args()
    prefetch(args.root) if args.command=='prefetch' else prepare(args.root,args.data)
