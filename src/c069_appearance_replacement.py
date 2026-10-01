"""Two bounded fixed appearance combinations after the user's slot-replacement instruction.

Reuse verified frozen raw predictions. No training, threshold choice or routing.
Historical replica replay metrics are diagnostic; advancement uses actual organizer97.
"""
from pathlib import Path
import argparse, copy, json, os, shutil, sys
os.environ.setdefault('POLARS_MAX_THREADS', '4')
import numpy as np
import pandas as pd
from reid_probe_local import ROOT, sha, save_json, stamp
from c047_hard_example_study import read, verify_hashes
from build_fixed_model_candidates import asset_source, DATASET as APPEARANCE_DATASET
from run_last_days_local import Queue, replay_command
import c058_localizer_baseline as official
import evaluate_local
from c066_transfer97 import summaries

PARENTS = {
    'c069': ROOT/'experiments/candidates/c067_single_source_44b6',
    'c070': ROOT/'experiments/candidates/c065_pooled_transformer/deploy',
}
FIXED=ROOT/'experiments/candidates/c041_fixed_models'

def root(key): return ROOT/'experiments/candidates'/f'{key}_fixed_appearance'
def notebook(key): return root(key)/f'biohub-{key}-fixed-appearance.ipynb'

def prepare(key):
    out=root(key);parent=PARENTS[key];phase=out/'study'
    assert not (phase/'plan.json').exists(), 'Registered phase is immutable'
    phase.mkdir(parents=True,exist_ok=True)
    parent_meta=read(parent/'kernel-metadata.json');parent_nb=read(parent/parent_meta['code_file'])
    assert read(parent/'local_verification.json')['status']=='passed'
    hashes=dict(read(parent/'plan.json')['hashes'])
    if key=='c069':
        hashes.update(read(parent/'local/plan.json')['hashes'])
        hashes.update(read(parent/'local/output_hashes.json'))
        review=ROOT/'state/c067_local_review.json'
        parent_cases=read(parent/'local/plan.json')['cases']
        by_stem={c['stem']:c['graph'] for c in parent_cases}
    else:
        hashes.update({str((parent/p).relative_to(ROOT)):h for p,h in read(parent/'artifact_hashes.json').items()})
        finish=parent.parent/'finish'
        hashes.update(read(finish/'plan.json')['hashes']);hashes.update(read(finish/'output_hashes.json'))
        review=ROOT/'state/c065_deploy_review.json'
        by_stem={s:str((parent/'graphs'/('combined_'+j['split'])/(s+'_final.npz')).relative_to(ROOT)) for j in read(parent/'plan.json')['jobs'] for s in j['stems']}
    assert read(review)['status']=='passed'
    assert len(by_stem)==97
    assets=out/'dataset';assets.mkdir(exist_ok=True)
    for src in [parent/'dataset/transformer_mean600.pt',FIXED/'appearance_mean_cosine.pt']:
        shutil.copy2(src,assets/src.name);assert sha(src)==sha(assets/src.name)
    nb=copy.deepcopy(parent_nb)
    helper=asset_source({'appearance_mean_cosine.pt':sha(assets/'appearance_mean_cosine.pt')})
    helper=helper.replace('_fixed_asset','_appearance_asset').replace('_FIXED_HASHES','_APPEARANCE_HASHES')
    nb['cells'][2]['source']=(''.join(nb['cells'][2]['source'])+helper).splitlines(True)
    source=(FIXED/'portable_appearance.py').read_text(encoding='utf-8')
    anchor='\nwrite_test_submission("base")\n';code=''.join(nb['cells'][5]['source']);assert code.count(anchor)==1
    payload=f'\nexec(compile({source!r}, "c041_portable_appearance", "exec"), globals())\n'
    payload+="install_complementary(globals(), str(_appearance_asset('appearance_mean_cosine.pt')))\nC038_MODE = 'appearance'\n"
    nb['cells'][5]['source']=code.replace(anchor,payload+anchor,1).splitlines(True)
    nb['metadata']['title']=notebook(key).stem
    for i,c in enumerate(nb['cells']):
        if c['cell_type']=='code':
            text=''.join(c['source']);compile(text,f'{key}:{i}','exec')
            assert 'H:/' not in text and 'H:\\' not in text
            c['outputs']=[];c['execution_count']=None
    assert nb['cells'][4]['source']==parent_nb['cells'][4]['source'], 'Inference changed'
    notebook(key).write_text(json.dumps(nb,indent=1)+'\n',encoding='utf-8')
    meta=copy.deepcopy(parent_meta);meta.update(id='taeyangg4/'+notebook(key).stem,title=notebook(key).stem,code_file=notebook(key).name)
    meta['dataset_sources']=sorted(set(meta['dataset_sources']+[APPEARANCE_DATASET]))
    save_json(out/'kernel-metadata.json',meta)
    save_json(phase/'appearance.json',{'appearance':{'C038_MODE':'appearance'}})
    save_json(phase/'off.json',{'off':{'C038_MODE':'off'}})
    save_json(phase/'as_configured.json',{'as_configured':{}})
    jobs=copy.deepcopy(read(parent/'plan.json')['jobs']);cases=[]
    for job in jobs:
        split=job['split'];job['run']=str(parent/'e2e'/split)
        audit=copy.deepcopy(nb);code=''.join(audit['cells'][5]['source'])
        hook='\nfrom c038_followup_local import install_graph_audit\n'+f'install_graph_audit(globals(),{str(phase/"graphs"/split)!r})\n'
        audit['cells'][5]['source']=code.replace(anchor,hook+anchor,1).splitlines(True)
        path=phase/('audit_'+split+'.ipynb');path.write_text(json.dumps(audit,indent=1),encoding='utf-8');job['notebook']=str(path)
        for stem in job['stems']:
            cases.append(dict(stem=stem,split=split,embryo=stem[:4],parent_graph=by_stem[stem]))
            hashes[by_stem[stem]]=sha(ROOT/by_stem[stem])
            run=Path(job['run']);cache=run/'edge_cache'/(stem+'.npz')
            raw=next((run/'predictions').rglob(stem+'.geff'))
            for p in [cache,*[p for p in raw.rglob('*') if p.is_file()]]:
                hashes[str(p.relative_to(ROOT))]=sha(p)
    controls=[next(c['stem'] for c in cases if c['embryo']==g) for g in ['44b6','6bba']]
    paths=[Path(__file__),ROOT/'tools/launch_appearance_replacement.ps1',review,parent/'local_verification.json',parent/'official_per_movie97.csv',FIXED/'portable_appearance.py',FIXED/'appearance_mean_cosine.pt',ROOT/'src/build_fixed_model_candidates.py',ROOT/'src/c038_followup_local.py',ROOT/'src/c066_transfer97.py',ROOT/'state/replacement_scope_20260929.json']
    paths += [p for p in out.rglob('*') if p.is_file()]
    hashes.update({str(p.relative_to(ROOT)):sha(p) for p in paths})
    verify_hashes(hashes)
    save_json(phase/'plan.json',dict(candidate=key,parent=str(parent.relative_to(ROOT)),created=stamp(),hashes=hashes,jobs=jobs,cases=cases,control_stems=controls,total_jobs=5,estimated_minutes=50,
        estimate_basis='Historical appearance97 off+on59min; appearance-only plus2off controls expected40-45min, actual97/review margin5min. Reassess measured progress.',
        recipe='Unchanged C041 mean-cosine model and runtime on uniform verified parent outputs; no fitting/routing/sweep',
        no_primary_inference=True,deepcenter_can_use_cuda=True,actual_organizer_required=True,not_yet_submission_eligible=True))
    print(json.dumps(dict(candidate=key,inputs=len(hashes),movies=97)),flush=True)

def analyse(key):
    out=root(key);phase=out/'study';plan=read(phase/'plan.json');rows=[]
    for case in plan['cases']:
        stem=case['stem'];folder=phase/'graphs'/case['split']
        with np.load(ROOT/case['parent_graph']) as parent,np.load(folder/(stem+'_appearance.npz')) as graph:
            assert np.array_equal(parent['ids'],graph['ids']) and np.array_equal(parent['txyz'],graph['txyz'])
            assert len(parent['edges'])==len(graph['edges'])
            if stem in plan['control_stems']:
                with np.load(folder/(stem+'_off.npz')) as off: assert all(np.array_equal(parent[n],off[n]) for n in ['ids','txyz','edges']),(stem,'off control')
            row=official.official_score(graph['ids'],graph['txyz'],graph['edges'],ROOT/'data/train'/(stem+'.geff'))
            changed=len(set(map(tuple,parent['edges']))-set(map(tuple,graph['edges'])))
        rows.append(dict(candidate=key.upper(),stem=stem,split=case['split'],embryo=case['embryo'],changed_edges=changed,**row))
    data=pd.DataFrame(rows);data.to_csv(phase/'official_per_movie97.csv',index=False)
    result=summaries(data);aggregate=evaluate_local._load_official()[-1]
    references={'parent':pd.read_csv(PARENTS[key]/'official_per_movie97.csv'),'C065':pd.read_csv(PARENTS['c070']/'official_per_movie97.csv'),'C067':pd.read_csv(PARENTS['c069']/'official_per_movie97.csv')}
    for row in result:
        group=row['group'];part=data if group=='all97' else data[data.split!='extension75'] if group=='all22' else data[(data.embryo==group)|(data.split==group)]
        for name,ref in references.items():
            before=ref[ref.stem.isin(part.stem)];assert len(before)==len(part)
            row['delta_vs_'+name]=row['score']-aggregate(before.to_dict('records'))['score']
            for count in ['edge_tp','edge_fp','division_tp','division_fp']:
                row[count+'_delta_vs_'+name]=int(part[count].sum()-before[count].sum())
        row['changed_edges']=int(part.changed_edges.sum())
    pd.DataFrame(result).to_csv(phase/'official_summary.csv',index=False)
    save_json(phase/'analysis.json',dict(status='complete_review_required',rows=result,off_controls_exact=2,unchanged_all97_nodes=True,actual_organizer_metric=True,
        limitations='Repeated97 correlated crops/two embryos and parent fit-domain; no hidden-score guarantee; portable/T4 still required'))
    print(json.dumps(result),flush=True)

def run(key):
    out=root(key);phase=out/'study';assert not (phase/'status.json').exists();plan=read(phase/'plan.json')
    os.environ['BIOHUB_FIXED_ASSET_DIR']=str(out/'dataset');q=Queue(phase,2)
    try:
        q.state.update(total_jobs=5,plan_sha256=sha(phase/'plan.json'));q.save();verify_hashes(plan['hashes'])
        j=plan['jobs'][0]
        q.run('off_controls',replay_command(Path(j['notebook']),Path(j['run']),plan['control_stems'],phase/'off.json',phase/'replay/off.csv'))
        for j in plan['jobs']:
            q.run('appearance_'+j['split'],replay_command(Path(j['notebook']),Path(j['run']),j['stems'],phase/'appearance.json',phase/'replay'/(j['split']+'.csv')))
        q.run('actual97',[sys.executable,'-X','utf8','-u',Path(__file__),'analyse','--candidate',key])
        verify_hashes(plan['hashes'])
        files=[p for folder in ['graphs','replay'] for p in (phase/folder).rglob('*') if p.is_file()]
        files += [phase/n for n in ['analysis.json','official_per_movie97.csv','official_summary.csv']]
        save_json(phase/'output_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in files})
        q.state['total_jobs']=5;q.close('complete_review_required')
    except BaseException as exc:q.close('failed',exc);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','analyse']);p.add_argument('--candidate',choices=list(PARENTS),required=True)
    a=p.parse_args();globals()[a.command](a.candidate)
