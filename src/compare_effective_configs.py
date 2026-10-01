import nbformat
import re
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

notebooks = {
    'B0_reyhan': 'kaggle_notebooks/latest_review/reyhan_0947/biohub-cell-tracking-0-947-lb.ipynb',
    'C003': 'experiments/candidates/c003_medal_frontier/biohub-c003-medal-frontier.ipynb',
    'flexonafft_hf': 'kaggle_notebooks/flexonafft_harmonic_fusion/biohub-harmonic-fusion.ipynb',
    'raunak_hfv3': 'kaggle_notebooks/harmonic_fusion_v3/biohub-harmonic-fusion-v3.ipynb',
    'sjlee_v020': 'kaggle_notebooks/sjlee_dctta/biohub-lf-dctta-v020.ipynb',
    'sjlee_sister16': 'kaggle_notebooks/sjlee_sister16/biohub-lf-dctta020-sectta1-sister16.ipynb',
    'thtennant_gapfill': 'kaggle_notebooks/thtennant_gapfill/biohub-frontier947-gapfill-v1.ipynb',
}

effective_configs = {}
for name, path in notebooks.items():
    nb = nbformat.read(path, as_version=4)
    code = '\n'.join(c.source for c in nb.cells if c.cell_type == 'code')
    
    # 1. find all os.environ[key] = val
    set_envs = dict(re.findall(r'os\.environ\[["\'](BIOHUB_[A-Z0-9_]+)["\']\]\s*=\s*["\']([^"\']+)["\']', code))
    
    # 2. find all os.environ.get(key, default)
    get_envs = dict(re.findall(r'os\.environ\.get\(["\'](BIOHUB_[A-Z0-9_]+)["\']\s*,\s*["\']([^"\']+)["\']\)', code))
    
    # effective is set_envs overriding get_envs default
    effective = {}
    for k, default in get_envs.items():
        effective[k] = set_envs.get(k, default)
    for k, val in set_envs.items():
        if k not in effective:
            effective[k] = val
            
    effective_configs[name] = effective

keys = sorted(set(k for c in effective_configs.values() for k in c.keys()))
names = list(notebooks.keys())
header = f"{'KEY':<42} " + " ".join(f"{n[:10]:<10}" for n in names)
print(header)
print("-" * len(header))
for k in keys:
    vals = [effective_configs[n].get(k, "-") for n in names]
    if len(set(vals)) > 1:
        print(f"{k:<42} " + " ".join(f"{v[:10]:<10}" for v in vals))
