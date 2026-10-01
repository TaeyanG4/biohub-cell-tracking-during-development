#!/usr/bin/env python3
"""Read-only C052 annotated daughter-link survival; no scorer or graph edits."""
from __future__ import annotations
import argparse
import collections
import copy
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import c052_division_transformer as pilot
import c037_transformer_study as training
import c038_followup_local as audit
from c047_hard_example_study import read
from reid_probe_local import ROOT,save_json,sha,stamp
from reid_augmented_local import evaluation_plan


def mapping(ns,nodes,gt):
    found,_=ns['match_nodes_bipartite'](nodes,gt,max_dist=7.)
    return {int(g):int(p) for p,g in found.items()}


def plain(coords):
    return {i:(int(a[0]),*[max(0,int(round(v))) for v in a[1:]]) for i,a in enumerate(coords)}


def run(extension,pilot_only=False):
    study=pilot.DEST;ns=pilot.ns_for();engine=training.engine(study)
    full,_,_=engine.load_model(training.PRIMARY_WEIGHTS,torch.device('cuda'))
    teacher=full.transformer;student=copy.deepcopy(teacher);teacher.eval();student.eval()
    teacher_hash=training.state_digest(teacher.state_dict())
    pilot_split={s:sp for sp,s in evaluation_plan()};events=read(study/'event_audit.json')['events']
    rows=[];verified_packets=0
    for group in pilot.GROUPS:
        opposite='6bba' if group=='44b6' else '44b6'
        checkpoint=study/'models'/f'{opposite}_step600.pt';ckpt=torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert ckpt['train_embryo']==opposite and ckpt['base_transformer_sha256']==teacher_hash
        student.load_state_dict(ckpt['state_dict']);student.eval()
        for stem in sorted({e['stem'] for e in events if e['embryo']==group and (not pilot_only or e['stem'] in pilot_split)}):
            is_pilot=stem in pilot_split;folder=study if is_pilot else extension
            on_name=('division600_' if is_pilot else 'extension_')+group
            off_name='off_'+pilot_split[stem] if is_pilot else 'off_'+group
            gt,ge=ns['graph_to_plain'](ns['graph_from_geff'](ROOT/'data/train'/(stem+'.geff')))
            with np.load(folder/'e2e'/on_name/'edge_cache'/(stem+'.npz')) as z:
                cache={k:z[k].copy() for k in z.files}
            with np.load(study/'e2e/capture_division87/edge_cache'/(stem+'.npz')) as z:
                base={k:z[k].copy() for k in z.files}
            assert np.array_equal(cache['coords'],base['coords']),('frozen detector coordinates',stem)
            fused={kind:{(int(a),int(b)):float(p) for a,b,p in zip(v['edge_src'],v['edge_tgt'],v['edge_prob'])} for kind,v in [('off',base),('on',cache)]}
            admitted={kind:{(int(a),int(b)) for a,b,*_ in v['admitted']} for kind,v in [('off',base),('on',cache)]}
            stages={}
            for kind,name in [('off',off_name),('on',on_name)]:
                for stage in ['ilp','motion','pre_restore','final']:
                    nodes,edges=audit.graph(folder/'graphs'/name,stem,stage)
                    stages[(kind,stage)]=(mapping(ns,nodes,gt),edges)
            movie_events=[e for e in events if e['stem']==stem]
            for t in sorted({e['t'] for e in movie_events}):
                path=folder/'evaluation_features'/stem/f'{t:03d}_{t+1:03d}.npz'
                assert path.exists(),('division packet missing',path)
                with np.load(path) as z:packet={k:z[k].copy() for k in z.files}
                ia=np.flatnonzero(cache['coords'][:,0]==t);ib=np.flatnonzero(cache['coords'][:,0]==t+1)
                assert np.array_equal(cache['coords'][ia,1:],packet['coords_src']) and np.array_equal(cache['coords'][ib,1:],packet['coords_tgt'])
                ids=np.r_[ia,ib];local=mapping(ns,plain(cache['coords'][ids]),gt)
                with torch.no_grad():
                    before=training.forward(teacher,packet);after=training.forward(student,packet)
                    error=float(np.max(np.abs(after.cpu().numpy()[np.ix_(packet['probe_rows'],packet['probe_cols'])]-packet['probe_logits'])))
                    assert error<2e-5,('actual learned packet parity',stem,t,error)
                    before=torch.softmax(before,dim=0).cpu().numpy();after=torch.softmax(after,dim=0).cpu().numpy()
                verified_packets+=1
                for event in [e for e in movie_events if e['t']==t]:
                    for child in event['children']:
                        pidx=local.get(event['parent']);cidx=local.get(child)
                        available=pidx is not None and cidx is not None and pidx<len(ia) and cidx>=len(ia)
                        row=dict(stem=stem,embryo=group,parent=event['parent'],child=child,t=t,detected_pair=available)
                        if available:
                            j=cidx-len(ia);pair=(int(ids[pidx]),int(ids[cidx]))
                            row.update(primary_off=float(before[pidx,j]),primary_on=float(after[pidx,j]),
                                primary_rank_off=int((before[:,j]>before[pidx,j]).sum()+1),primary_rank_on=int((after[:,j]>after[pidx,j]).sum()+1))
                            for kind in ['off','on']:
                                row['fused_'+kind]=fused[kind].get(pair)
                                row['fused_'+kind+'_below_cache_cutoff']=pair not in fused[kind]
                                row['admitted_'+kind]=pair in admitted[kind]
                        for (kind,stage),(m,edges) in stages.items():
                            a,b=m.get(event['parent']),m.get(child)
                            row[stage+'_'+kind+'_matched']=a is not None and b is not None
                            row[stage+'_'+kind]=a is not None and b is not None and (a,b) in edges
                        rows.append(row)
    assert training.state_digest(teacher.state_dict())==teacher_hash
    destination=study/'pilot_division_survival' if pilot_only else extension
    destination.mkdir(exist_ok=True)
    data=pd.DataFrame(rows);data.to_csv(destination/'division_survival.csv',index=False)
    summary=[]
    for group in pilot.GROUPS:
        frame=data[data.embryo==group]
        for stage in ['admitted','ilp','motion','pre_restore','final']:
            a=frame[stage+'_off'].fillna(False).astype(bool);b=frame[stage+'_on'].fillna(False).astype(bool)
            summary.append(dict(embryo=group,stage=stage,daughter_annotations=len(frame),off_links=int(a.sum()),on_links=int(b.sum()),gained=int((~a&b).sum()),lost=int((a&~b).sum())))
    save_json(destination/'division_survival.json',dict(status='passed',created=stamp(),rows=summary,verified_actual_learned_packets=verified_packets,
        daughter_rows=len(rows),cache_cutoff=.02,teacher_unchanged=True,
        interpretation='Diagnostic annotated daughter links matched per stage with original official7um matcher;NOT official fork TP/FP nor independent biological events. Blank fused probability means <=.02,not zero. All detection inputs exact. No graph edits or threshold selection.'))
    print('Division survival diagnostic completed',len(rows),'daughter annotations;',verified_packets,'actual packets',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--extension',type=Path,default=pilot.DEST/'extension75');p.add_argument('--pilot-only',action='store_true')
    a=p.parse_args();run(a.extension.resolve(),a.pilot_only)
