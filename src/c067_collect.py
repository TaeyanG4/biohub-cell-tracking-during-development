"""Read exact research versions, merge same-checkpoint evidence, verify portable output."""
from pathlib import Path
import argparse,ast,json,os,re,shutil,sys,zipfile
os.environ.setdefault('POLARS_MAX_THREADS','4')
import numpy as np
import pandas as pd
import c067_single_source as candidate
import c052_division_transformer as pilot
import c058_localizer_baseline as official
import evaluate_local
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
from c047_hard_example_study import read,verify_hashes
from reid_probe_local import ROOT,sha,save_json,stamp
from run_last_days_local import Queue,replay_command
from eval_pp_variants_local import load_raw_graph
from final_four_download import fetch_required

def get_version(api,ref):
    owner,slug=ref.split('/')
    with api.build_kaggle_client() as client:
        request=ApiGetKernelRequest();request.user_name=owner;request.kernel_slug=slug
        response=client.kernels.kernels_api_client.get_kernel(request)
    assert response.metadata.current_version_number==1 and response.metadata.is_private
    assert response.metadata.enable_gpu and not response.metadata.enable_internet
    return response

def csv_graphs(table):
    assert set(table.row_type)<= {'node','edge'}
    result={}
    for stem,part in table.groupby('dataset',sort=False):
        nodes=part[part.row_type=='node'].sort_values('node_id');edges=part[part.row_type=='edge']
        for block in [nodes[['node_id','t','z','y','x']],edges[['source_id','target_id']]]:
            values=block.to_numpy();assert np.isfinite(values).all() and np.array_equal(values,np.rint(values)),stem
        ids=nodes.node_id.to_numpy(np.int64);txyz=nodes[['t','z','y','x']].to_numpy(np.int64)
        links=np.array(sorted(map(tuple,edges[['source_id','target_id']].to_numpy(np.int64))),np.int64).reshape(-1,2)
        assert len(ids)==len(set(ids)) and np.isin(links,ids).all()
        assert len(links)==len(set(map(tuple,links)))
        result[stem]=dict(ids=ids,txyz=txyz,edges=links)
    return result

def fetch(key):
    out=candidate.root(key);folder=out/'remote_audit';action=read(folder/'push_action.json')
    assert read(folder/'status.json')['status']=='complete_review_required'
    assert action['version']==1 and action['research_only_never_submit']
    api=KaggleApi();api.authenticate();response=get_version(api,action['ref'])
    remote=json.loads(response.blob.source);meta=read(folder/'kernel-metadata.json');local=read(folder/meta['code_file'])
    assert len(remote['cells'])==len(local['cells'])
    assert all(a['cell_type']==b['cell_type'] and ''.join(a['source'])==''.join(b['source']) for a,b in zip(remote['cells'],local['cells']))
    dest=folder/'v1';assert not (dest/'fetch_proof.json').exists(),'Already fetched; do not repeat'
    dest.mkdir(exist_ok=True);(dest/'source.ipynb').write_text(response.blob.source,encoding='utf-8')
    download=fetch_required(api,action['ref'],dest,['research_artifacts.zip'])
    get_version(api,action['ref'])
    archive=dest/'research_artifacts.zip';assert archive.is_file()
    extracted=dest/'output'
    if not extracted.exists():
        extracted.mkdir()
        with zipfile.ZipFile(archive) as stream:
            for info in stream.infolist():
                relative=Path(info.filename)
                assert not relative.is_absolute() and '..' not in relative.parts and ':' not in info.filename
                (extracted/relative).resolve().relative_to(extracted.resolve())
            stream.extractall(extracted)
    hashes=read(extracted/'research_output_hashes.json')
    for p,h in hashes.items():
        path=(extracted/p).resolve();path.relative_to(extracted.resolve())
        assert sha(path)==h,p
    logfiles=list(dest.glob('*.log'));assert len(logfiles)==1
    chunks=read(logfiles[0]);log=''.join(c['data'] for c in chunks)
    assert read(out/'plan.json')['model_sha256'] in log
    records=[];hardware=[];asset_records=[]
    for line in log.splitlines():
        if line.startswith('FIXED_ASSET transformer_mean600.pt '):
            assert line.split()[-1]==read(out/'plan.json')['model_sha256'];asset_records.append(line)
        if line.startswith('C037_PRIMARY '):
            value=ast.literal_eval(line.split(' ',1)[1]);assert value['train_embryo']==candidate.CONFIG[key] and value['step']==600 and value['alpha']==1.
            assert value['path'].endswith('/transformer_mean600.pt');records.append(value)
        if line.startswith('LOCAL_RUNTIME '):
            value=ast.literal_eval(line.split(' ',1)[1]);assert 'T4' in value['gpu'] and value['math_sdp']
            assert not any(value[k] for k in ['cudnn_tf32','matmul_tf32','flash_sdp','mem_efficient_sdp']);hardware.append(value)
    # Production loads once per shard, not once per movie. Kaggle's combined
    # stream logs each of the two shard startup records twice (also in C053).
    plan=read(out/'plan.json');assert len(records)==len(hardware)==4 and len(asset_records)==1
    assert len(re.findall(r'^GPU shard [01]: CUDA_VISIBLE_DEVICES=[01] ',log,re.M))==2
    for i in (0,1):
        lines=[s for s in log.splitlines() if s.startswith(f'GPU shard {i}: CUDA_VISIBLE_DEVICES={i} ')]
        assert len(lines)==1 and lines[0].endswith(f'--slice {i}::2')
    sizes=sorted(int(s) for s in re.findall(r'^Fold 0: (\d+) datasets \|',log,re.M))
    n=len(plan['remote_stems']);assert sizes==sorted([n//2,(n+1)//2]*2),sizes
    scripts={}
    for name in ['predict_unet_transformer.py','c037_transformer_runtime.py','v1284_coordinate_refinement.py']:
        path=extracted/'tracking_repo/scripts'/name
        assert path.read_text(encoding='utf-8')==(out/'tracking_repo/scripts'/name).read_text(encoding='utf-8'),name
        scripts[name]=sha(path)
    receipt=read(extracted/'research_input_receipt.json')
    assert receipt['status']=='passed' and receipt['stems']==plan['remote_stems']
    assert receipt['manifest_sha256']==sha(out/'dataset/research_inputs.json')
    stats=pd.read_csv(extracted/'run_stats.csv');assert set(stats.dataset)==set(plan['remote_stems'])
    assert (stats.repair_fallback==0).all() and (stats.deadline_degraded==0).all()
    save_json(dest/'fetch_proof.json',dict(status='passed',ref=action['ref'],version=1,source_cells_exact=True,
        notebook_sha256=action['notebook_sha256'],model_sha256=plan['model_sha256'],archive_sha256=sha(archive),
        hashed_outputs=len(hashes),input_receipt=receipt,runtime_records=len(records),hardware=hardware,
        research_only_never_submit=True,download_receipt=download,actual_script_hashes=scripts,
        runtime_record_scope='Two uniform inference shards; startup records duplicated by combined log, not per-movie records',
        shard_movie_counts=sizes,covered_movies=n))
    print(json.dumps(dict(status='fetched_verified',candidate=key,files=len(hashes),runtime_records=len(records))),flush=True)

def historical(stem,split):
    base=pilot.DEST/('extension75' if split=='extension75' else '')
    name=('extension_' if split=='extension75' else 'division600_')+stem[:4]
    return base/'e2e'/name,base/'graphs'/name/(stem+'_final.npz')

def prepare(key):
    out=candidate.root(key);phase=out/'local';assert not (phase/'plan.json').exists()
    data=out/'remote_audit/v1/output';proof=read(out/'remote_audit/v1/fetch_proof.json');assert proof['status']=='passed'
    plan=read(out/'plan.json');module,_=candidate.module_for(key)
    table=pd.read_csv(data/'submission.csv');assert set(table.dataset)==set(plan['remote_stems'])
    remote_graphs=csv_graphs(table)
    # Verify all remote provenance again before reusing/copying outputs.
    verify_hashes(plan['hashes'])
    for path,digest in read(data/'research_output_hashes.json').items():
        resolved=(data/path).resolve();resolved.relative_to(data.resolve());assert sha(resolved)==digest
    ns=pilot.ns_for();cases=[];smoke=[];extra=set()
    for job in plan['jobs']:
        split=job['split']
        for stem in job['stems']:
            previous,old_final=historical(stem,split)
            if stem in plan['remote_stems']:
                cache=data/'edge_cache'/(stem+'.npz')
                with np.load(cache) as new,np.load(pilot.control_run(stem)/'edge_cache'/(stem+'.npz')) as base:
                    for name in ['coords','low_coords','low_score']:assert np.array_equal(new[name],base[name],equal_nan=True),(stem,name)
            if stem in plan['smoke_stems']:
                with np.load(cache) as new,np.load(previous/'edge_cache'/(stem+'.npz')) as old:
                    assert set(new.files)==set(old.files)
                    assert all(np.array_equal(new[n],old[n],equal_nan=True) for n in old.files),(stem,'full cache')
                current_raw=next((data/'tracking_repo/predictions').rglob(stem+'.geff'))
                a,b=[load_raw_graph(ns,p) for p in [current_raw,pilot.graph_path(previous,stem)]]
                sort=lambda e:(e['source_id'],e['target_id'])
                assert a[0]==b[0] and sorted(a[1],key=sort)==sorted(b[1],key=sort),(stem,'ILP')
                with np.load(old_final) as old:
                    assert all(np.array_equal(remote_graphs[stem][n],old[n]) for n in ['ids','txyz','edges']),(stem,'final')
                smoke.append(dict(stem=stem,exact_all_cache=True,exact_ilp=True,exact_final=True))
            reuse=stem in plan['reused_stems']
            source_cache=previous/'edge_cache'/(stem+'.npz') if reuse else data/'edge_cache'/(stem+'.npz')
            with np.load(source_cache) as current,np.load(pilot.control_run(stem)/'edge_cache'/(stem+'.npz')) as base:
                for name in ['coords','low_coords','low_score']:assert np.array_equal(current[name],base[name],equal_nan=True),(stem,name)
            source_raw=pilot.graph_path(previous,stem) if reuse else next((data/'tracking_repo/predictions').rglob(stem+'.geff'))
            canonical=out/'e2e'/split
            (canonical/'edge_cache').mkdir(parents=True,exist_ok=True);(canonical/'predictions').mkdir(exist_ok=True)
            shutil.copy2(source_cache,canonical/'edge_cache'/(stem+'.npz'))
            shutil.copytree(source_raw,canonical/'predictions'/(stem+'.geff'))
            reference=out/'reference_graphs'/('combined_'+split)/(stem+'_final.npz');reference.parent.mkdir(parents=True,exist_ok=True)
            if reuse:shutil.copy2(old_final,reference)
            else:np.savez_compressed(reference,**remote_graphs[stem])
            target=out/'graphs'/('combined_'+split)/reference.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(reference,target)
            cases.append(dict(stem=stem,split=split,embryo=stem[:4],reused=reuse,graph=str(reference.relative_to(ROOT)),
                              raw_source=str(source_raw.relative_to(ROOT)),cache_source=str(source_cache.relative_to(ROOT))))
            extra.add(source_cache);extra.add(old_final) if reuse else None
            extra.update(p for p in source_raw.rglob('*') if p.is_file())
    assert len(cases)==97 and len(smoke)==2
    save_json(out/'smoke.json',dict(status='passed',rows=smoke,exact_frozen_detector_movies=97))
    phase.mkdir(parents=True,exist_ok=True)
    inputs={Path(__file__),ROOT/'src/final_four_download.py',phase/'launch.ps1',ROOT/'tools/launch_c067_local.ps1',out/'plan.json',out/'remote_audit/v1/fetch_proof.json',out/'smoke.json',
            ROOT/'state/c067_prepare_review.json',ROOT/'src/c066_transfer97.py',ROOT/'src/c058_localizer_baseline.py',ROOT/'src/evaluate_local.py',
            ROOT/'state/validation_audit_20260929/actual_per_movie.csv'}|extra
    for folder in [data,out/'e2e',out/'reference_graphs',evaluate_local.VENDOR_SRC]:
        inputs.update(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    hashes=dict(plan['hashes']);hashes.update({str(p.relative_to(ROOT)):sha(p) for p in inputs})
    save_json(phase/'plan.json',dict(created=stamp(),hashes=hashes,cases=cases,total_jobs=6,estimated_minutes=30,
        reused_movies=len(plan['reused_stems']),new_movies=len(plan['new_stems']),no_new_inference=True,
        scientific_evidence='Own embryo fit-domain; opposite embryo component transfer. All outputs use one exact checkpoint.'))
    print(json.dumps(dict(prepared=True,candidate=key,inputs=len(hashes),movies=97)),flush=True)

def analyse(key):
    from c066_transfer97 import summaries
    out=candidate.root(key);plan=read(out/'local/plan.json');rows=[]
    for case in plan['cases']:
        with np.load(ROOT/case['graph']) as graph:
            row=official.official_score(graph['ids'],graph['txyz'],graph['edges'],ROOT/'data/train'/(case['stem']+'.geff'))
        rows.append(dict(candidate=key.upper(),stem=case['stem'],split=case['split'],embryo=case['embryo'],**row))
    data=pd.DataFrame(rows);data.to_csv(out/'official_per_movie97.csv',index=False)
    result=summaries(data);pd.DataFrame(result).to_csv(out/'official_summary.csv',index=False)
    module,_=candidate.module_for(key);visible=data[data.stem.isin(module.STEM_SETS['vis4'])]
    expected=evaluate_local._load_official()[-1](visible.to_dict('records'))['score']
    save_json(out/'analysis.json',dict(status='complete_review_required',rows=result,visible4_expected=expected,
        metric_commit=evaluate_local.METRIC_COMMIT,actual_organizer_metric=True,uniform_fixed_source_model=candidate.CONFIG[key],
        warning='Opposite embryo component transfer and own embryo fit-domain; two biological domains only.'))

def verify(key):
    out=candidate.root(key);module,_=candidate.module_for(key);module.verify_local(out)
    for stem in (out/'heldout12.txt').read_text().split():
        with np.load(out/'graphs/combined_heldout12'/(stem+'_final.npz')) as actual,np.load(out/'reference_graphs/combined_heldout12'/(stem+'_final.npz')) as expected:
            assert all(np.array_equal(actual[n],expected[n]) for n in ['ids','txyz','edges']),(stem,'portable final graph')

def run(key):
    out=candidate.root(key);phase=out/'local';assert not (phase/'status.json').exists()
    plan=read(phase/'plan.json');module,_=candidate.module_for(key);os.environ['BIOHUB_FIXED_ASSET_DIR']=str(out/'dataset');os.environ['PYTHONUTF8']='1'
    q=Queue(phase,2)
    try:
        q.state.update(total_jobs=6,plan_sha256=sha(phase/'plan.json'));q.save();verify_hashes(plan['hashes'])
        command=[sys.executable,'-X','utf8','-u',Path(__file__)]
        q.run('actual97',command+['analyse','--candidate',key])
        stems=(out/'heldout12.txt').read_text().split()
        q.run('audit12',replay_command(out/'audit_heldout12.ipynb',out/'e2e/heldout12',stems,out/'variants.json',out/'replay/heldout12.csv'))
        q.run('portable12',replay_command(module.nbpath(out),out/'e2e/heldout12',stems,out/'variants.json',out/'replay/portable_heldout12.csv'))
        q.run('writer4',command+['write_visible','--candidate',key])
        q.run('actual_visible4',[sys.executable,'-X','utf8','-u',ROOT/'src/evaluate_local.py','--csv',out/key/'submission.csv','--gt-dir',ROOT/'data/visible_gt/train','--out-dir',out/key/'evaluation'])
        q.run('verify_local',command+['verify','--candidate',key]);verify_hashes(plan['hashes'])
        files=[p for folder in ['graphs','replay',key] for p in (out/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
        files += [out/name for name in ['analysis.json','official_per_movie97.csv','official_summary.csv','local_verification.json','smoke.json']]
        save_json(phase/'output_hashes.json',{str(p.relative_to(ROOT)):sha(p) for p in files});q.close('complete_review_required')
    except BaseException as exc:q.close('failed',exc);raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['fetch','prepare','run','analyse','verify','write_visible']);parser.add_argument('--candidate',choices=list(candidate.CONFIG),required=True)
    args=parser.parse_args()
    if args.command=='write_visible':candidate.module_for(args.candidate)[0].write_visible(candidate.root(args.candidate))
    else:globals()[args.command](args.candidate)
