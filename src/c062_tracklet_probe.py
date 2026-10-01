"""Two-movie frozen C048 trajectory appearance information probe, no fitting.

Reuses C038 candidate recording/appearance encoding and C058 exact C023 capture.
Selection is entirely prediction/image-support based. GT annotates saved groups
only after selection; unknown identities never become negatives. No graph edits.
"""
from __future__ import annotations
import argparse
import collections
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

import c055_guarded_readmit as pp
import c058_localizer_baseline as frozen
import c058_raw_localizer_study as images
import c038_complementary_stage as appearance
import eval_pp_variants_local as harness
from frame_motion_audit import read_frame
from reid_augmented_local import AppearanceEncoder, RECIPE as APPEARANCE_RECIPE
from reid_probe_local import install_recorder, sha, save_json

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'state/c062_preflight/association/probe'
C048 = ROOT / 'experiments/candidates/c048_embryo_holdout_appearance'
C058 = ROOT / 'experiments/candidates/c058_raw_localizer'
STEMS = ('44b6_12dfb391', '6bba_05db0fb1')
POLICY = dict(version=1, stems=list(STEMS), crop=[8,32,32], history=3, future=3,
    score='cosine(normalized mean of 3 original C048 unit embeddings)',
    control='same checkpoint single source/child snapshot on identical full candidate groups',
    eligibility='whole candidate group requires all original real centers and unbranched consecutive 3-node contexts',
    synthetic_exclusion='C058 actual gap_synthetic OR passively observed gap2 insertion',
    checkpoint='frozen opposite-whole-embryo C048 weak_aug',
    proposal_thresholds={k:APPEARANCE_RECIPE[k] for k in ['cosine_gain','cosine_margin','max_cost_increase']},
    gate='each embryo: actual admission-rule known rescue and positive net versus single; no increase in known-distinct choices versus single or baseline; report baseline rescue separately',
    no_fit=True, no_graph_edit=True, no_official_gain_claim=True)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def model_path(stem):
    source = '6bba' if stem.startswith('44b6_') else '44b6'
    return C048 / 'models' / f'{source}_weak_aug.pt'


def dependencies():
    """Finite input paths for a parent-owned queue to pin before any run."""
    paths = {Path(__file__), pp.BASE, C048/'plan.json', C048/'fold_proof.json',
             C048/'artifact_hashes.json', C058/'output_hashes.json'}
    for name in ['c038_complementary_stage.py','reid_probe_local.py','reid_augmented_local.py',
                 'division_crops_extract.py','division_cnn_train.py','c055_guarded_readmit.py',
                 'c055_guarded_readmit_stage.py','run_last_days_local.py','local_registration_probe.py',
                 'c058_localizer_baseline.py','c058_raw_localizer_study.py',
                 'c058_raw_localizer_model.py','eval_pp_variants_local.py','frame_motion_audit.py',
                 'evaluate_local.py']:
        paths.add(ROOT/'src'/name)
    # Existing capture namespace depends on DeepCenter; preserve its established
    # acceptance-score use, without interpreting its peak coordinate convention.
    plan = read(C058/'plan.json')
    paths.update(ROOT/p for p in plan['hashes'] if ('deepcenter' in p.lower() or
                 'vendor' in p.lower() or 'official' in p.lower()))
    for stem in STEMS:
        paths.update([model_path(stem), model_path(stem).with_suffix('.json'),
            C058/'baseline'/f'{stem}.npz', C058/'labels'/f'{stem}.csv',
            pp.DEST/'graphs/heldout12'/f'{stem}_control.npz', pp.DEST/'study/heldout12.csv'])
        for folder in [ROOT/'data/train'/f'{stem}.geff', ROOT/'data/train'/f'{stem}.zarr',
                       pp.pred_path('heldout12', stem)]:
            paths.update(p for p in folder.rglob('*') if p.is_file())
        paths.update(p for p in (pp.run_dir('heldout12')/'edge_cache').glob(stem+'*') if p.is_file())
    assert all(p.is_file() for p in paths)
    return sorted(paths)


def select_groups(rec, final, shape):
    """No GT arguments: topology/original image support select whole groups."""
    ids, txyz, edges = final['ids'], final['txyz'], final['edges']
    at = {int(i):j for j,i in enumerate(ids)}
    incoming, outgoing = collections.defaultdict(list), collections.defaultdict(list)
    for a,b in edges:
        incoming[int(b)].append(int(a)); outgoing[int(a)].append(int(b))
    protected = set()
    for a,b in edges:
        a,b=int(a),int(b)
        if len(outgoing[a])!=1 or len(incoming[b])!=1 or txyz[at[b],0]!=txyz[at[a],0]+1:
            protected.update([a,b])
    half=np.array(POLICY['crop'])//2; crop=np.array(POLICY['crop'])
    counts=collections.Counter(total_recorded_groups=len(rec.groups))
    valid={}
    for i in ids:
        i=int(i);j=at[i]
        if i not in rec.nodes or i in protected or bool(final['gap_synthetic'][j]):
            valid[i]=False;continue
        n=rec.nodes[i]
        # C035 patches() first stores tzyx as float32, then Python-rounds each
        # coordinate. Mirror those exact centers for support and provenance.
        float32_center=np.asarray([n[k] for k in ['z','y','x']],np.float32)
        center=np.array([int(round(float(v))) for v in float32_center])
        if not np.array_equal(center,txyz[j,1:]):
            counts['excluded_original_center_mismatch_nodes']+=1
            valid[i]=False;continue
        valid[i]=bool(int(n['t'])==int(txyz[j,0]) and
                      ((center-half>=0)&(center-half+crop<=np.array(shape[1:]))).all())

    def context(i,direction):
        seq=[i]
        for _ in range(2):
            neighbours=(incoming if direction<0 else outgoing)[seq[-1]]
            if len(neighbours)!=1:return None
            j=neighbours[0]
            if txyz[at[j],0]!=txyz[at[seq[-1]],0]+direction:return None
            seq.append(j)
        return seq if all(valid.get(j,False) for j in seq) else None

    accepted=[]
    needed=set()
    for sid,(tids,geometry,chosen,phase) in sorted(rec.groups.items()):
        sid=int(sid); tids=np.asarray(tids,np.int64)
        if sid not in at or sid in protected or len(outgoing[sid])!=1 or len(tids)<2:
            counts['c038_source_ineligible']+=1;continue
        current=outgoing[sid][0]
        if current not in tids or current in protected or any(int(t) not in at for t in tids):
            counts['c038_target_ineligible']+=1;continue
        if any(txyz[at[int(t)],0]!=txyz[at[sid],0]+1 for t in tids):
            counts['c038_time_ineligible']+=1;continue
        counts['c038_original_eligible_groups']+=1
        history=context(sid,-1)
        if history is None:
            counts['history_unavailable_groups']+=1;continue
        futures=[context(int(t),1) for t in tids]
        if any(x is None for x in futures):
            counts['any_candidate_future_unavailable_groups']+=1;continue
        group=dict(source=sid, targets=tids.tolist(), current=current,
                   geometry=np.asarray(geometry).tolist(), history=history,
                   futures=futures, capture_chosen=int(chosen), phase=str(phase))
        accepted.append(group);needed.update(history)
        for seq in futures:needed.update(seq)
    counts.update(eligible_groups=len(accepted), candidate_pairs=sum(len(g['targets']) for g in accepted),
                  unique_encoded_nodes=len(needed))
    return accepted, sorted(needed),dict(counts)


def capture(out,stem):
    assert stem in STEMS
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    folder=out/stem;folder.mkdir(exist_ok=True)
    assert not (folder/'groups.json').exists(), 'No overwrite of a recorded probe'
    start=time.perf_counter()
    ns=harness.build_namespace(pp.BASE,{},pp.run_dir('heldout12')/'edge_cache')
    ns['TEST_DIR']=ROOT/'data/train'
    rec=appearance.AllNodeRecorder(ns);rec.enabled=True;install_recorder(ns,rec)
    path=frozen.capture(stem,'heldout12',out,ns=ns)
    with np.load(path) as z:final={k:z[k].copy() for k in z.files}
    with np.load(C058/'baseline'/f'{stem}.npz') as old:
        for k in ['ids','txyz','float_txyz','edges','gap_synthetic']:
            assert np.array_equal(final[k],old[k]),('C058 exact capture control',stem,k)
    meta=images.metadata(stem)
    groups,needed,counts=select_groups(rec,final,meta[1]['shape'])
    assert groups and needed,'No usable whole candidate groups; record coverage before continuing'
    # Persist the prediction-only selection before consulting known identities.
    save_json(folder/'runtime_groups.json',dict(policy=POLICY,stem=stem,groups=groups,coverage=counts))
    labels=pd.read_csv(C058/'labels'/f'{stem}.csv')
    assert not labels.node_id.duplicated().any() and not labels.gt_id.duplicated().any()
    p2g=dict(zip(labels.node_id.astype(int),labels.gt_id.astype(int)))
    gt,gt_edges=ns['graph_to_plain'](ns['graph_from_geff'](ROOT/'data/train'/f'{stem}.geff'))
    children=collections.defaultdict(list)
    for a,b in gt_edges:children[int(a)].append(int(b))
    for g in groups:
        gid=p2g.get(g['source']);kids=children.get(gid,[])
        known=bool(len(kids)==1 and gid in gt and gt[kids[0]][0]==gt[gid][0]+1)
        g['labels']=[int(p2g[t]==kids[0]) if known and t in p2g else -1 for t in g['targets']]
        g['source_gt']=gid
        g['expected_gt']=kids[0] if known else None
        g['target_gt']=[p2g.get(t) for t in g['targets']]
        g['known_single_successor']=known
        g['positive_reachable']=1 in g['labels']
        g['known_competition']=1 in g['labels'] and 0 in g['labels']
    save_json(folder/'groups.json',dict(policy=POLICY,stem=stem,groups=groups,coverage=counts,
        runtime_selection_sha256=sha(folder/'runtime_groups.json'),gt_used_after_selection=True))
    tzyx=np.array([[int(rec.nodes[i]['t']),*[float(rec.nodes[i][k]) for k in ['z','y','x']]] for i in needed],np.float64)
    np.savez_compressed(folder/'needed_nodes.npz',ids=np.array(needed,np.int64),tzyx=tzyx)
    save_json(folder/'capture.json',dict(status='passed',stem=stem,seconds=time.perf_counter()-start,
        exact_c058_graph=True,coverage=counts,groups_sha256=sha(folder/'groups.json'),
        needed_sha256=sha(folder/'needed_nodes.npz'),graphs_modified=False))
    print(json.dumps(dict(stage='capture',stem=stem,**counts)),flush=True)


def load_encoder(stem):
    path=model_path(stem);source=path.name[:4]
    state=torch.load(path,map_location='cpu',weights_only=True)
    assert sha(path)==read(path.with_suffix('.json'))['model_sha256']
    assert state['recipe']==APPEARANCE_RECIPE and state['arm']=='weak_aug'
    assert {s[:4] for s in state['train_movies']}=={source} and source!=stem[:4]
    assert state['c048_plan_sha256']==sha(C048/'plan.json')
    fold=next(f for f in read(C048/'plan.json')['folds'] if f['test_embryo']==stem[:4])
    assert set(state['train_movies'])==set(fold['train_movies'])
    torch.set_num_threads(4)
    model=AppearanceEncoder().cpu().eval();model.load_state_dict(state['state_dict'],strict=True)
    return model,source,sha(path)


def vector_pass(stem,ids,coords,model):
    nodes={int(i):dict(zip(['t','z','y','x'],row)) for i,row in zip(ids,coords)}
    meta=images.metadata(stem);reads=[]
    def existing_reader(dataset,t,frames):
        assert dataset==stem
        if t not in frames:
            frames[t]=read_frame(meta[0],int(t),meta[1]['shape'],np.dtype(meta[1]['data_type']))
            reads.append(int(t))
        return frames[t]
    start=time.perf_counter()
    vectors=appearance.appearance_vectors({'read_test_frame':existing_reader},stem,nodes,ids,model)
    values=np.stack([vectors[int(i)] for i in ids])
    assert values.shape==(len(ids),32) and np.isfinite(values).all()
    assert np.allclose(np.linalg.norm(values,axis=1),1,rtol=0,atol=2e-6)
    assert len(reads)==len(set(coords[:,0]))==len(set(reads))
    return values,dict(seconds=time.perf_counter()-start,unique_nodes=len(ids),
        frame_reads=len(reads),frames=reads,device='cpu',threads=4)


def saved_nodes(out,stem):
    folder=Path(out)/stem;receipt=read(folder/'capture.json')
    assert sha(folder/'groups.json')==receipt['groups_sha256']
    assert sha(folder/'needed_nodes.npz')==receipt['needed_sha256']
    with np.load(folder/'needed_nodes.npz') as z:
        return folder,z['ids'].copy(),z['tzyx'].copy()


def benchmark(out,stem):
    """At most 3 frames x64 already captured nodes; no fit or saved predictions."""
    assert stem in STEMS
    folder,all_ids,all_coords=saved_nodes(out,stem)
    assert not (folder/'benchmark.json').exists(),'Do not replace a timing receipt'
    times=sorted(set(all_coords[:,0]))
    chosen_times=[times[i] for i in np.unique(np.linspace(0,len(times)-1,min(3,len(times)),dtype=int))]
    index=np.concatenate([np.flatnonzero(all_coords[:,0]==t)[:64] for t in chosen_times])
    model,source,digest=load_encoder(stem)
    _,timing=vector_pass(stem,all_ids[index],all_coords[index],model)
    # Conservative measured scale covers whichever of frame IO or node encoding
    # grows faster, and includes 20 percent slack. It is not a scientific knob.
    scale=max(len(all_ids)/len(index),len(times)/len(chosen_times))
    estimate=float(timing['seconds']*scale*1.2)
    save_json(folder/'benchmark.json',dict(status='passed',stem=stem,source_embryo=source,
        checkpoint_sha256=digest,**timing,total_unique_nodes=len(all_ids),total_frames=len(times),
        estimated_full_encoding_seconds=estimate,estimate_rule='measured seconds * max(node ratio,frame ratio) *1.2',
        sample_policy='first64 nodes of each of up to3 evenly spaced captured frame times',
        no_fit=True,no_predictions_saved=True))
    print(json.dumps(read(folder/'benchmark.json')),flush=True)


def encode(out,stem):
    assert stem in STEMS
    folder,ids,coords=saved_nodes(out,stem)
    assert not (folder/'vectors.npz').exists(),'Do not overwrite frozen inference'
    model,source,digest=load_encoder(stem)
    values,timing=vector_pass(stem,ids,coords,model)
    np.savez_compressed(folder/'vectors.npz',ids=ids,vectors=values)
    save_json(folder/'encode.json',dict(status='passed',stem=stem,source_embryo=source,
        source_checkpoint_sha256=digest,**timing,
        vectors_sha256=sha(folder/'vectors.npz'),no_fit=True))
    print(json.dumps(read(folder/'encode.json')),flush=True)


def mean_vector(vectors,ids):
    value=np.asarray([vectors[int(i)] for i in ids],np.float64).mean(axis=0)
    norm=float(np.linalg.norm(value))
    assert np.isfinite(norm) and norm>1e-8,'Undefined trajectory mean; do not omit group'
    return value/norm


def proposed(group,similarity):
    """Unchanged C048 per-group proposal conditions; no graph assignment/edit."""
    targets=np.asarray(group['targets']);geo=np.asarray(group['geometry'])
    order=np.argsort(-np.asarray(similarity),kind='stable');top,second=order[:2]
    old=int(np.flatnonzero(targets==group['current'])[0]);target=int(targets[top])
    if target==group['current']:return group['current']
    if similarity[top]-similarity[old]<APPEARANCE_RECIPE['cosine_gain']:return group['current']
    if similarity[top]-similarity[second]<APPEARANCE_RECIPE['cosine_margin']:return group['current']
    if geo[top,3]>geo[old,3]+APPEARANCE_RECIPE['max_cost_increase']:return group['current']
    return target


def analyse(out):
    out=Path(out);rows=[];choice_rows=[];summaries=[]
    for stem in STEMS:
        folder=out/stem;capture_receipt=read(folder/'capture.json');enc=read(folder/'encode.json')
        assert sha(folder/'groups.json')==capture_receipt['groups_sha256']
        assert sha(folder/'vectors.npz')==enc['vectors_sha256']
        recorded=read(folder/'groups.json');groups=recorded['groups']
        assert recorded['policy']==POLICY
        assert sha(folder/'runtime_groups.json')==recorded['runtime_selection_sha256']
        with np.load(folder/'vectors.npz') as z:vectors={int(i):v for i,v in zip(z['ids'],z['vectors'])}
        counts=collections.Counter();duplicate_error=0.;unique=collections.defaultdict(set)
        for g in groups:
            source=g['source'];targets=g['targets'];labels=np.asarray(g['labels'])
            single=np.array([np.dot(vectors[source],vectors[t]) for t in targets],np.float64)
            history=mean_vector(vectors,g['history'])
            multi=np.array([np.dot(history,mean_vector(vectors,seq)) for seq in g['futures']])
            repeated=np.array([np.dot(mean_vector(vectors,[source]*3),mean_vector(vectors,[t]*3)) for t in targets])
            duplicate_error=max(duplicate_error,float(np.max(np.abs(repeated-single))))
            assert np.max(np.abs(repeated-single))<2e-6
            assert proposed(g,single)==proposed(g,repeated),'Duplicate control changes a C048 proposal'
            counts['groups']+=1;counts['candidate_pairs']+=len(targets)
            counts['unknown_candidates']+=int((labels<0).sum())
            counts['known_competition_groups']+=int(g['known_competition'])
            picks={name:int(np.argsort(-values,kind='stable')[0]) for name,values in [('single',single),('three',multi)]}
            choices=dict(baseline=g['current'],single=proposed(g,single),three=proposed(g,multi))
            choice_labels={name:int(labels[targets.index(target)]) for name,target in choices.items()}
            edge_identity=(stem,g['source_gt'],g['expected_gt'])
            if g['known_single_successor']:
                unique['represented_gt_links'].add(edge_identity)
            if g['known_competition']:
                unique['strict_competition_gt_links'].add(edge_identity)
            for name,label in choice_labels.items():
                counts[name+'_choice_known_correct']+=int(label==1)
                counts[name+'_choice_known_distinct']+=int(label==0)
                counts[name+'_choice_unknown']+=int(label<0)
                if label==0:unique[name+'_choice_known_distinct'].add(edge_identity)
            for reference in ['single','baseline']:
                before,after=choice_labels[reference],choice_labels['three']
                events=dict(rescue=before==0 and after==1,harm=before==1 and after==0,
                            known_to_unknown=before>=0 and after<0,
                            correct_to_unknown=before==1 and after<0,
                            unknown_to_known=before<0 and after>=0)
                for event,active in events.items():
                    key='admission_three_vs_'+reference+'_'+event
                    counts[key]+=int(active)
                    if active and g['known_single_successor']:unique[key].add(edge_identity)
            both=choice_labels['baseline']==0 and choice_labels['single']==0 and choice_labels['three']==1
            counts['admission_same_example_rescue_vs_both']+=int(both)
            if both:unique['admission_same_example_rescue_vs_both'].add(edge_identity)
            choice_rows.append(dict(stem=stem,source=source,source_gt=g['source_gt'],expected_gt=g['expected_gt'],
                known_competition=g['known_competition'],**{k+'_target':v for k,v in choices.items()},
                **{k+'_label':v for k,v in choice_labels.items()}))
            for name,values in [('single',single),('three',multi)]:
                top=picks[name];choice=choices[name];j=targets.index(choice)
                counts[name+'_top_unknown']+=int(labels[top]<0)
                counts[name+'_proposals']+=int(choice!=g['current'])
                counts[name+'_proposal_unknown']+=int(choice!=g['current'] and labels[j]<0)
            if g['known_competition']:
                known=np.flatnonzero(labels>=0)
                a=int(known[np.argsort(-single[known],kind='stable')[0]])
                b=int(known[np.argsort(-multi[known],kind='stable')[0]])
                counts['known_only_single_correct']+=int(labels[a]==1)
                counts['known_only_three_correct']+=int(labels[b]==1)
                counts['known_only_rescue']+=int(labels[a]==0 and labels[b]==1)
                counts['known_only_harm']+=int(labels[a]==1 and labels[b]==0)
            a,b=picks['single'],picks['three']
            counts['all_candidate_top_changed']+=int(a!=b)
            counts['all_candidate_confirmed_rescue']+=int(labels[a]==0 and labels[b]==1)
            counts['all_candidate_confirmed_harm']+=int(labels[a]==1 and labels[b]==0)
            counts['all_candidate_changed_unknown']+=int(a!=b and (labels[a]<0 or labels[b]<0))
            for j,t in enumerate(targets):
                rows.append(dict(stem=stem,source=source,target=t,current=g['current'],label=int(labels[j]),
                    source_gt=g['source_gt'],expected_gt=g['expected_gt'],target_gt=g['target_gt'][j],
                    known_competition=g['known_competition'],single=float(single[j]),three=float(multi[j]),
                    single_top=j==a,three_top=j==b,single_proposed=t==choices['single'],
                    three_proposed=t==choices['three'],geometry_cost=float(g['geometry'][j][3])))
        rescue=counts['admission_three_vs_single_rescue'];harm=counts['admission_three_vs_single_harm']
        unique_rescue=len(unique['admission_three_vs_single_rescue'])
        unique_harm=len(unique['admission_three_vs_single_harm'])
        no_increase=all(counts['three_choice_known_distinct']<=counts[name+'_choice_known_distinct']
                        for name in ['single','baseline'])
        mistakes=counts['single_choice_known_distinct']
        gate=dict(actual_known_rescues=rescue,known_harms=harm,net=rescue-harm,
            unique_rescues=unique_rescue,unique_harms=unique_harm,
            no_known_distinct_increase_vs_single_and_baseline=no_increase,
            status='inconclusive_no_actual_known_mistakes' if not mistakes else 'passed' if rescue>harm and unique_rescue>unique_harm and rescue>0 and no_increase else 'failed',
            same_examples_rescue_both=counts['admission_same_example_rescue_vs_both'],
            baseline_rescues=counts['admission_three_vs_baseline_rescue'],baseline_harms=counts['admission_three_vs_baseline_harm'])
        summary=dict(stem=stem,embryo=stem[:4],coverage=recorded['coverage'],counts=dict(counts),
            unique_gt_link_counts={k:len(v) for k,v in unique.items()},
            unique_gt_link_identities={k:[list(x) for x in sorted(v)] for k,v in unique.items()},
            admission_gate=gate,duplicate_vector_max_abs_error=duplicate_error,capture=capture_receipt,encoding=enc)
        summaries.append(summary)
    pd.DataFrame(rows).to_csv(out/'candidate_scores.csv',index=False)
    pd.DataFrame(choice_rows).to_csv(out/'admission_choices.csv',index=False)
    signals=[s['admission_gate']['status']=='passed' for s in summaries]
    save_json(out/'review.json',dict(status='complete_review_required',policy=POLICY,rows=summaries,
        both_embryos_admission_signal_passed=all(signals),no_graph_modified=True,
        official_score_computed=False,unknown_candidates_remain_unknown=True,
        limitation='Rank/proposal information only; no collision-safe graph acceptance and no independent cell-count inference.',
        next='A positive signal only earns separately reviewed integration; no automatic sweep or submission.'))
    print(json.dumps(dict(both_embryos_admission_signal_passed=all(signals),rows=[s['admission_gate'] for s in summaries])),flush=True)


def controls(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(6202);v=rng.normal(size=(10,32)).astype(np.float32)
    v/=np.linalg.norm(v,axis=1,keepdims=True);vectors=dict(enumerate(v))
    original=np.array([np.dot(v[0],v[i]) for i in range(1,10)])
    repeated=np.array([np.dot(mean_vector(vectors,[0]*3),mean_vector(vectors,[i]*3)) for i in range(1,10)])
    assert np.max(np.abs(original-repeated))<2e-6
    group=dict(targets=[1,2],current=1,geometry=[[0,0,0,1],[0,0,0,1]])
    assert proposed(group,[.2,.5])==2
    assert proposed(group,[.49,.5])==1
    assert proposed(group,[.5,.5])==1
    group['geometry'][1][3]=3
    assert proposed(group,[.2,.5])==1
    # Pure prediction-only context check: alternate child lacks an incoming
    # edge but has its own two-node future, so both candidates remain present.
    from types import SimpleNamespace
    times=np.array([0,1,2,3,4,5,3,4,5])
    points=np.column_stack([times,np.full((9,3),24)])
    fake=dict(ids=np.arange(9),txyz=points,edges=np.array([[0,1],[1,2],[2,3],[3,4],[4,5],[6,7],[7,8]]),
              gap_synthetic=np.zeros(9,bool))
    recorder=SimpleNamespace(nodes={i:dict(zip(['t','z','y','x'],points[i])) for i in range(9)},
        groups={2:(np.array([3,6]),np.zeros((2,7)),3,'tight')})
    selected,needed,coverage=select_groups(recorder,fake,[6,64,64,64])
    assert len(selected)==1 and selected[0]['targets']==[3,6]
    assert selected[0]['history']==[2,1,0] and selected[0]['futures']==[[3,4,5],[6,7,8]]
    assert needed==list(range(9))
    fake['gap_synthetic'][7]=True
    assert not select_groups(recorder,fake,[6,64,64,64])[0], 'Do not prune only an inconvenient candidate'
    fake['gap_synthetic'][7]=False
    recorder.nodes[7]['z']=25
    selected,_,coverage=select_groups(recorder,fake,[6,64,64,64])
    assert not selected and coverage['excluded_original_center_mismatch_nodes']==1
    save_json(out/'controls.json',dict(status='passed',policy=POLICY,
        duplicate_vector_max_abs_error=float(np.max(np.abs(original-repeated))),
        same_C048_gain_margin_geometry_proposals=True,whole_group_topology_and_synthetic_controls=True,
        no_model_inference=True,no_fit=True))
    print(json.dumps(read(out/'controls.json')),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('verb',choices=['inputs','controls','capture','benchmark','encode','analyse'])
    parser.add_argument('--out',type=Path,default=DEST)
    parser.add_argument('--stem',choices=STEMS)
    args=parser.parse_args()
    if args.verb=='inputs':
        args.out.mkdir(parents=True,exist_ok=True)
        save_json(args.out/'input_paths.json',[str(p.relative_to(ROOT)) for p in dependencies()])
        print('Wrote finite input path list; no input hashing or execution',flush=True)
    elif args.verb in ['capture','benchmark','encode']:
        assert args.stem,'--stem is required'
        globals()[args.verb](args.out,args.stem)
    else:globals()[args.verb](args.out)


if __name__=='__main__':main()
