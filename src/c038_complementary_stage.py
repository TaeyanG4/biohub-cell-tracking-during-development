"""C038 local replay adapter: appearance proposals checked by independent motion.

Runs on all eligible predicted nodes. No GT, diagnostic source whitelist or
label-derived decision enters this stage. Uses existing C035/C036 implementations.
"""
from __future__ import annotations
import collections
import copy
import os
from pathlib import Path
import numpy as np
import torch

from reid_probe_local import Recorder, install_recorder
from reid_augmented_local import AppearanceEncoder, RECIPE, patches
from local_registration_probe import local_motion
from frame_motion_audit import phase_shift, VOX


class AllNodeRecorder(Recorder):
    def __init__(self, ns):
        super().__init__(ns, {}, [])
        self.enabled = False

    def begin(self, nodes):
        # Deliberately no GT matching or annotation-based eligibility.
        self.groups = {}; self.sources = set(nodes) if self.enabled else set()
        self.nodes = copy.deepcopy(nodes) if self.enabled else {}


@torch.no_grad()
def appearance_vectors(ns, dataset, nodes, needed, model):
    plain={i:(int(nodes[i]['t']),*(float(nodes[i][k]) for k in ['z','y','x'])) for i in needed}
    by_t=collections.defaultdict(list)
    for i,n in plain.items():by_t[n[0]].append(i)
    vectors={}
    for t,ids in sorted(by_t.items()):
        ids=np.array(sorted(ids),np.int64)
        crops,_=patches(ns,{},dataset,ids,plain)
        for start in range(0,len(ids),64):
            batch=torch.from_numpy(crops[start:start+64].astype(np.float32))[:,None]
            values=model(batch).numpy()
            vectors.update({int(i):v for i,v in zip(ids[start:start+64],values)})
    return vectors


def install_complementary(ns, checkpoint):
    torch.set_num_threads(4)
    rec=AllNodeRecorder(ns);install_recorder(ns,rec)
    original=ns['filter_output_graph']
    ns['C038_MODE']='off'
    cache={}

    def wrapper(nodes_by_id,raw_edges,dataset=None,deepcenter_bundle=None):
        mode=str(ns['C038_MODE'])
        if mode not in ['off','appearance','agreement']:raise ValueError('invalid C038 mode')
        rec.enabled=mode!='off'
        nodes,edges,stats=original(nodes_by_id,raw_edges,dataset=dataset,deepcenter_bundle=deepcenter_bundle)
        if mode=='off' or not edges:return nodes,edges,stats
        if not dataset:raise ValueError('C038 requires dataset image access')
        if 'model' not in cache:
            state=torch.load(checkpoint,map_location='cpu',weights_only=True)
            assert state['recipe']==RECIPE and state['arm']=='weak_aug'
            model=AppearanceEncoder().cpu().eval();model.load_state_dict(state['state_dict'])
            cache['model']=model
        incoming=collections.defaultdict(list);outgoing=collections.defaultdict(list)
        for j,e in enumerate(edges):
            a,b=int(e['source_id']),int(e['target_id'])
            outgoing[a].append(j);incoming[b].append(j)
        protected=set()
        for e in edges:
            a,b=int(e['source_id']),int(e['target_id'])
            if len(outgoing[a])!=1 or len(incoming[b])!=1 or int(nodes[b]['t'])!=int(nodes[a]['t'])+1:
                protected.update([a,b])
        eligible={};needed=set()
        for sid,(tids,geometry,chosen,phase) in rec.groups.items():
            if sid not in nodes or sid in protected or len(outgoing[sid])!=1 or len(tids)<2:continue
            current=int(edges[outgoing[sid][0]]['target_id'])
            if current not in tids or current in protected:continue
            if any(int(t) not in nodes for t in tids):continue
            if any(int(nodes[int(t)]['t'])!=int(nodes[sid]['t'])+1 for t in tids):continue
            eligible[int(sid)]=(tids,geometry,current)
            needed.add(int(sid));needed.update(map(int,tids))
        if not eligible:
            stats['c038_proposals']=0;stats['c038_changed_edges']=0
            return nodes,edges,stats
        # Cache embeddings only when original positions and needed ids are identical.
        signature=(dataset,tuple((i,rec.nodes[i]['t'],rec.nodes[i]['z'],rec.nodes[i]['y'],rec.nodes[i]['x']) for i in sorted(needed)))
        if cache.get('signature')!=signature:
            cache['vectors']=appearance_vectors(ns,dataset,rec.nodes,needed,cache['model'])
            cache['signature']=signature;cache['registration']={};cache['prior']={}
        vectors=cache['vectors'];proposals={};candidate_prob={}
        for sid,(tids,geo,current) in eligible.items():
            similarity=np.array([float(np.dot(vectors[sid],vectors[int(t)])) for t in tids])
            order=np.argsort(-similarity,kind='stable');top,second=order[:2]
            old=int(np.flatnonzero(tids==current)[0]);target=int(tids[top])
            if target==current or target in protected:continue
            if similarity[top]-similarity[old]<RECIPE['cosine_gain'] or similarity[top]-similarity[second]<RECIPE['cosine_margin']:continue
            if geo[top,3]>geo[old,3]+RECIPE['max_cost_increase']:continue
            if mode=='agreement':
                if sid not in cache['registration']:
                    n=rec.nodes[sid];t=int(n['t'])
                    a=ns['read_test_frame'](dataset,t,{})
                    b=ns['read_test_frame'](dataset,t+1,{})
                    # Scale/clip exactly as C036 before registration, using supplied movie quantiles.
                    import zarr
                    group=zarr.open_group(str(Path(ns['TEST_DIR'])/(dataset+'.zarr')),mode='r')
                    q=group.attrs['image_statistics']['quantiles'];lo,hi=float(q['0.001']),float(q['0.999'])
                    a=np.clip((a.astype(np.float32)-lo)/(hi-lo+1e-6),0.,3.)
                    b=np.clip((b.astype(np.float32)-lo)/(hi-lo+1e-6),0.,3.)
                    if t not in cache['prior']:
                        aa,bb=a[:,::2,::2],b[:,::2,::2]
                        cache['prior'][t]=phase_shift(aa-aa.mean(),bb-bb.mean())*np.array([1.,2.,2.])
                    point=np.array([n[k] for k in ['z','y','x']],float)
                    cache['registration'][sid]=local_motion(a,b,point,cache['prior'][t])
                result=cache['registration'][sid]
                if not result['valid']:continue
                points=np.array([[rec.nodes[int(i)][k] for k in ['z','y','x']] for i in tids])
                nearest=int(tids[np.argmin(np.linalg.norm((points-result['point'])*VOX,axis=1))])
                if nearest!=target:continue
            proposals[sid]=target;candidate_prob[sid]=float(geo[top,2])
        # Reconnect to unused targets or execute reciprocal two-source swaps.
        # No unilateral stealing, forks, gaps or GT-dependent source selection.
        accepted={};used=set()
        for sid,target in sorted(proposals.items()):
            if sid in accepted or target in used:continue
            old=int(edges[outgoing[sid][0]]['target_id'])
            holders=incoming[target]
            if not holders:
                accepted[sid]=target;used.add(target)
            elif len(holders)==1:
                other=int(edges[holders[0]]['source_id'])
                if other!=sid and other not in accepted and other in proposals and proposals[other]==old and old not in used:
                    accepted[sid]=target;accepted[other]=old;used.update([target,old])
        result=copy.deepcopy(edges)
        for sid,target in accepted.items():
            j=outgoing[sid][0];result[j]['target_id']=target
            result[j]['edge_prob']=candidate_prob[sid]
            result[j]['distance_um']=ns['edge_distance_um'](nodes[sid],nodes[target])
            result[j]['c038_reassigned']=1
        before_out=collections.Counter(int(e['source_id']) for e in edges)
        after_out=collections.Counter(int(e['source_id']) for e in result)
        after_in=collections.Counter(int(e['target_id']) for e in result)
        assert len(result)==len(edges) and before_out==after_out
        assert all(count<=max(1,len(incoming[target])) for target,count in after_in.items())
        assert len({(int(e['source_id']),int(e['target_id'])) for e in result})==len(result)
        stats['c038_proposals']=len(proposals);stats['c038_changed_edges']=len(accepted)
        print(f'[{dataset}] C038 {mode}: eligible={len(eligible)} proposed={len(proposals)} changed={len(accepted)}',flush=True)
        return nodes,result,stats
    ns['filter_output_graph']=wrapper
