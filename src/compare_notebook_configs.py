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

all_envs = {}
for name, path in notebooks.items():
    nb = nbformat.read(path, as_version=4)
    code = '\n'.join(c.source for c in nb.cells if c.cell_type == 'code')
    matches = re.findall(r'os\.environ\[["\'](BIOHUB_[A-Z0-9_]+)["\']\]\s*=\s*["\']([^"\']+)["\']', code)
    envs = {}
    for k, v in matches:
        envs[k] = v
    all_envs[name] = envs

keys = sorted(set(k for envs in all_envs.values() for k in envs.keys()))
names = list(notebooks.keys())
header = f"{'KEY':<42} " + " ".join(f"{n[:10]:<10}" for n in names)
print(header)
print("-" * len(header))
for k in keys:
    vals = [all_envs[n].get(k, "-") for n in names]
    if len(set(vals)) > 1:
        print(f"{k:<42} " + " ".join(f"{v[:10]:<10}" for v in vals))
