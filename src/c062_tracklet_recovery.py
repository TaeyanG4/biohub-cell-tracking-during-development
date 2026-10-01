"""C062 preflight provenance correction; scientific scoring/source unchanged.

The original adapter incorrectly required pre-linefit image centers to equal
post-linefit submitted positions. Observe actual object continuity across that
existing transform instead; never infer correspondence from GT or scores.
"""
from __future__ import annotations
import argparse
import ast
import copy
import inspect
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import c062_tracklet_probe as original
from c055_guarded_readmit import save_json,sha,read

ROOT=original.ROOT
STEMS=original.STEMS
DEST=ROOT/'experiments/candidates/c062_tracklet_appearance/recovery'
POLICY=dict(original.POLICY,original_center_provenance='actual same object before/after original linefit; recorder equals observed pre-linefit coordinates; padding support uses original float32 centers')


def dependencies():
    return sorted(set(original.dependencies())|{Path(__file__),
        ROOT/'src/c062_tracklet_study.py',
        ROOT/'state/c062_preflight/launch_recovery.ps1',
        ROOT/'state/c062_preflight/recovery_controls/recovery_controls.json',
        ROOT/'experiments/candidates/c062_tracklet_appearance/preflight/plan.json',
        ROOT/'experiments/candidates/c062_tracklet_appearance/preflight/status.json',
        ROOT/'state/c062_preflight/empty_diagnosis/coverage.json'})


def corrected_selector():
    source=inspect.getsource(original.select_groups)
    anchor='if not np.array_equal(center,txyz[j,1:]):'
    assert source.count(anchor)==1
    source=source.replace(anchor,"if i not in rec.verified_linefit_origins:")
    # Only the erroneous equality guard is replaced; all topology/support and
    # full-candidate requirements use the original implementation verbatim.
    env=dict(original.__dict__)
    exec(compile(source,'C062:verified_linefit_origin','exec'),env)
    return env['select_groups']


def capture(out,stem):
    old_build=original.harness.build_namespace
    old_select=original.select_groups
    corrected=corrected_selector()
    observations={}
    def build(*args,**kwargs):
        ns=old_build(*args,**kwargs)
        linefit=ns['linefit_smooth_output_graph']
        def watch_linefit(nodes,edges,stats):
            before={int(i):tuple(float(n[k]) for k in ['t','z','y','x']) for i,n in nodes.items()}
            refs=dict(nodes)
            result=linefit(nodes,edges,stats)
            assert set(result)==set(refs),'Unexpected node admission/removal in linefit'
            assert all(result[i] is refs[i] for i in refs),'Linefit changed object identity'
            observations.update(before=before,refs=refs)
            return result
        ns['linefit_smooth_output_graph']=watch_linefit
        filtering=ns['filter_output_graph']
        def watch_final(*args,**kwargs):
            result=filtering(*args,**kwargs)
            observations['final_refs']=dict(result[0])
            return result
        ns['filter_output_graph']=watch_final
        return ns
    def select(rec,final,shape):
        assert observations and 'final_refs' in observations
        verified=set()
        for i,n in rec.nodes.items():
            if i not in observations['before'] or i not in observations['final_refs']:continue
            if observations['final_refs'][i] is not observations['refs'][i]:continue
            position=tuple(float(n[k]) for k in ['t','z','y','x'])
            if position==observations['before'][i]:verified.add(int(i))
        rec.verified_linefit_origins=verified
        groups,needed,counts=corrected(rec,final,shape)
        assert set(needed)<=verified
        counts.update(verified_prelinefit_original_nodes=len(verified),
                      linefit_input_nodes=len(observations['before']),
                      final_object_nodes=len(observations['final_refs']))
        folder=Path(out)/stem
        save_json(folder/'origin_proof.json',dict(status='passed',counts=counts,
            actual_original_linefit_observed=True,same_node_objects=True,
            recorder_positions_equal_prelinefit=True,selected_nodes=len(needed),
            selector_change='replace incorrect final-position equality with actual original transform/object proof',
            original_source_sha256=sha(Path(original.__file__))))
        return groups,needed,counts
    original.harness.build_namespace=build
    original.select_groups=select
    original.POLICY=POLICY
    try:return original.capture(out,stem)
    finally:
        original.harness.build_namespace=old_build
        original.select_groups=old_select


def controls(out):
    original.controls(out)
    # Replicate the original function's real-support topology fixture.
    ids=np.arange(1,10,dtype=np.int64)
    points=np.array([[t,12,64,64] for t in [0,1,2,3,4,5,3,4,5]],np.int64)
    nodes={int(i):dict(zip(['t','z','y','x'],row)) for i,row in zip(ids,points)}
    rec=SimpleNamespace(nodes=nodes,groups={3:(np.array([4,7]),np.zeros((2,7)),4,'test')},verified_linefit_origins=set(ids))
    final=dict(ids=ids,txyz=points.copy(),edges=np.array([[1,2],[2,3],[3,4],[4,5],[5,6],[7,8],[8,9]]),gap_synthetic=np.zeros(9,bool))
    selector=corrected_selector()
    final['txyz'][:,1:]+=1 # legitimate output-coordinate change, separately proven
    groups,needed,counts=selector(rec,final,(8,32,128,128))
    assert len(groups)==1 and len(needed)==9
    rec.verified_linefit_origins.remove(7)
    assert not selector(rec,final,(8,32,128,128))[0]
    save_json(Path(out)/'recovery_controls.json',dict(status='passed',
        legitimate_observed_smoothing_keeps_groups=True,unproven_replacement_rejected=True,
        no_graph_or_score_change=True,original_sha256=sha(Path(original.__file__))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('verb',choices=['controls','capture','benchmark','encode','analyse','prepare','run'])
    parser.add_argument('--out',type=Path,default=DEST)
    parser.add_argument('--phase',choices=['preflight','signal'])
    parser.add_argument('--stem',choices=STEMS)
    a=parser.parse_args()
    original.POLICY=POLICY
    if a.verb in ['prepare','run']:
        import c062_tracklet_study as study
        study.probe=sys.modules[__name__]
        getattr(study,a.verb)(a.out,a.phase)
    elif a.verb in ['controls','capture']:
        globals()[a.verb](a.out,*([a.stem] if a.verb=='capture' else []))
    else:
        getattr(original,a.verb)(a.out,*([a.stem] if a.verb in ['benchmark','encode'] else []))


import sys
if __name__=='__main__':main()
