"""Exact unchanged-inference portable proof for fixed appearance replacement candidates."""
from pathlib import Path
import argparse,copy,json,os,shutil,sys
os.environ.setdefault('POLARS_MAX_THREADS','4')
import numpy as np
import pandas as pd
from c069_appearance_replacement import root,notebook,PARENTS,FIXED
from c047_hard_example_study import read,verify_hashes
from reid_probe_local import ROOT,sha,save_json,stamp
from eval_pp_variants_local import build_namespace,STEM_SETS
from run_last_days_local import Queue,replay_command
from c067_collect import csv_graphs

def prepare(key):
    out=root(key);study=out/'study';phase=out/'local';phase.mkdir(exist_ok=True)
    assert not (phase/'plan.json').exists()
    assert read(study/'status.json')['status']=='complete_review_required'
    assert read(ROOT/f'state/{key}_study_review.json')['status']=='passed'
    assert read(study/'decision.json')['status']=='advance_portable'
    plan=read(study/'plan.json');hashes=dict(plan['hashes']);hashes.update(read(study/'output_hashes.json'))
    nb=read(notebook(key));audit=copy.deepcopy(nb);code=''.join(nb['cells'][5]['source']);anchor='\nwrite_test_submission("base")\n'
    assert code.count(anchor)==1
    hook='\nfrom c038_followup_local import install_graph_audit\n'+f'install_graph_audit(globals(),{str(phase/"graphs")!r})\n'
    audit['cells'][5]['source']=code.replace(anchor,hook+anchor,1).splitlines(True)
    (phase/'audit12.ipynb').write_text(json.dumps(audit,indent=1),encoding='utf-8')
    save_json(phase/'as_configured.json',{'as_configured':{}})
    parent=PARENTS[key];parent_nb=read(parent/read(parent/'kernel-metadata.json')['code_file'])
    assert nb['cells'][4]['source']==parent_nb['cells'][4]['source']
    # The exact same primary inference is reused, not regenerated from training data.
    shutil.copytree(parent/'tracking_repo',out/'tracking_repo',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
    paths=[Path(__file__),ROOT/'tools/launch_appearance_portable.ps1',ROOT/f'state/{key}_study_review.json',study/'decision.json',study/'plan.json',study/'output_hashes.json']
    paths += [p for p in (out/'tracking_repo').rglob('*') if p.is_file()]
    paths += [phase/'audit12.ipynb',phase/'as_configured.json']
    hashes.update({str(p.relative_to(ROOT)):sha(p) for p in paths})
    verify_hashes(hashes)
    # Compatibility input manifest for shared exact-version submit auditing.
    save_json(out/'plan.json',plan)
    save_json(phase/'plan.json',dict(candidate=key,hashes=hashes,created=stamp(),total_jobs=4,estimated_minutes=10,
        parent=str(parent.relative_to(ROOT)),stems=plan['jobs'][0]['stems'],source_primary_execution_cell_exact=True,
        portable_instrumentation='One actual as-configured12 execution with the existing read-only graph recorder; exact full notebook equality after removing only that hook is checked. Actual production writer4 runs separately.'))
    print(json.dumps(dict(prepared=key,inputs=len(hashes))),flush=True)

def write_visible(key):
    out=root(key);work=out/key;work.mkdir(exist_ok=True);run=PARENTS[key]/'e2e/heldout12'
    os.environ['BIOHUB_FIXED_ASSET_DIR']=str(out/'dataset')
    ns=build_namespace(notebook(key),{},run/'edge_cache');assert ns['C038_MODE']=='appearance'
    ns.update(TEST_DIR=ROOT/'data/train',REPO_DIR=work/'writer_repo',test_stems=STEM_SETS['vis4'],SUBMISSION_PATH=work/'submission.csv',RUN_STATS_PATH=work/'run_stats.csv',predict_seconds=0)
    for stem in STEM_SETS['vis4']:
        source=next((run/'predictions').rglob(stem+'.geff'))
        shutil.copytree(source,work/'writer_repo/predictions/portable'/ns['METHOD']/'split_0'/source.name,dirs_exist_ok=True)
    ns['write_test_submission']('portable_visible4')
    graphs=csv_graphs(pd.read_csv(work/'submission.csv'))
    for stem,g in graphs.items():
        with np.load(out/'study/graphs/heldout12'/(stem+'_appearance.npz')) as expected:
            assert all(np.array_equal(g[n],expected[n]) for n in ['ids','txyz','edges']),(stem,'writer graph')
    stats=pd.read_csv(work/'run_stats.csv')
    assert len(stats)==4 and (stats.repair_fallback==0).all() and (stats.deadline_degraded==0).all()

def verify(key):
    out=root(key);phase=out/'local';plan=read(phase/'plan.json');parent=PARENTS[key]
    for stem in plan['stems']:
        with np.load(phase/'graphs'/(stem+'_appearance.npz')) as a,np.load(out/'study/graphs/heldout12'/(stem+'_appearance.npz')) as b:
            assert all(np.array_equal(a[n],b[n]) for n in ['ids','txyz','edges']),(stem,'portable12')
    audit=read(phase/'audit12.ipynb');clean=read(notebook(key))
    hook='\nfrom c038_followup_local import install_graph_audit\n'+f'install_graph_audit(globals(),{str(phase/"graphs")!r})\n'
    text=''.join(audit['cells'][5]['source']);assert text.count(hook)==1
    audit['cells'][5]['source']=text.replace(hook,'',1).splitlines(True)
    assert audit==clean,'Only the existing read-only recorder may differ in portable12'
    data=pd.read_csv(phase/'replay/audit12.csv');assert len(data)==12 and set(data.config)=={'as_configured'}
    stats=pd.read_csv(out/key/'run_stats.csv');assert len(stats)==4
    assert (stats.repair_fallback==0).all() and (stats.deadline_degraded==0).all()
    nb=read(notebook(key));old=read(parent/read(parent/'kernel-metadata.json')['code_file'])
    assert nb['cells'][4]['source']==old['cells'][4]['source']
    runtime=(FIXED/'portable_appearance.py').read_text(encoding='utf-8')
    assert repr(runtime) in ''.join(nb['cells'][5]['source'])
    scripts={}
    for name in ['predict_unet_transformer.py','c037_transformer_runtime.py','v1284_coordinate_refinement.py']:
        a=out/'tracking_repo/scripts'/name;b=parent/'tracking_repo/scripts'/name
        assert sha(a)==sha(b);scripts[name]=sha(a)
    summary=read(out/key/'evaluation/summary.json')
    save_json(out/'local_verification.json',dict(status='passed',verified=stamp(),portable12='exact',writer4='exact',repair_fallback=0,deadline_degraded=0,
        notebook_sha256=sha(notebook(key)),model_sha256=sha(out/'dataset/transformer_mean600.pt'),csv_sha256=sha(out/key/'submission.csv'),
        appearance_model_sha256=sha(out/'dataset/appearance_mean_cosine.pt'),appearance_runtime_sha256=sha(FIXED/'portable_appearance.py'),
        local_visible4=summary['total_score'],primary_execution_cell_exact=True,parent=str(parent.relative_to(ROOT)),actual_source_hashes=scripts,
        portable12_readonly_capture=True,portable12_source_without_hook_exact=True,actual_T4_required=True))

def run(key):
    out=root(key);phase=out/'local';assert not (phase/'status.json').exists();plan=read(phase/'plan.json')
    os.environ['BIOHUB_FIXED_ASSET_DIR']=str(out/'dataset');q=Queue(phase,1)
    try:
        q.state.update(total_jobs=4,plan_sha256=sha(phase/'plan.json'));q.save();verify_hashes(plan['hashes'])
        run=PARENTS[key]/'e2e/heldout12';cmd=[sys.executable,'-X','utf8','-u',Path(__file__)]
        q.run('audit12',replay_command(phase/'audit12.ipynb',run,plan['stems'],phase/'as_configured.json',phase/'replay/audit12.csv'))
        q.run('writer4',cmd+['write_visible','--candidate',key])
        q.run('actual_visible4',[sys.executable,'-X','utf8','-u',ROOT/'src/evaluate_local.py','--csv',out/key/'submission.csv','--gt-dir',ROOT/'data/visible_gt/train','--out-dir',out/key/'evaluation'])
        q.run('verify',cmd+['verify','--candidate',key]);verify_hashes(plan['hashes'])
        files=[p for folder in [phase/'graphs',phase/'replay',out/key] for p in folder.rglob('*') if p.is_file()]
        files += [out/'local_verification.json']
        save_json(phase/'output_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in files});q.close('complete_review_required')
    except BaseException as exc:q.close('failed',exc);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','write_visible','verify']);p.add_argument('--candidate',choices=list(PARENTS),required=True)
    a=p.parse_args();globals()[a.command](a.candidate)
