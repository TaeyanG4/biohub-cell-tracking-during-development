import nbformat
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

notebooks = [
    "kaggle_notebooks/latest_review/reyhan_0947/biohub-cell-tracking-0-947-lb.ipynb",
    "kaggle_notebooks/sjlee_sister16/biohub-lf-dctta020-sectta1-sister16.ipynb",
    "kaggle_notebooks/thtennant_gapfill/biohub-frontier947-gapfill-v1.ipynb",
    "kaggle_notebooks/flexonafft_harmonic_fusion/biohub-harmonic-fusion.ipynb",
    "kaggle_notebooks/harmonic_fusion_v3/biohub-harmonic-fusion-v3.ipynb",
    "kaggle_notebooks/sjlee_hoctveto_div/biohub-lf-hoctveto-div-b.ipynb",
    "kaggle_notebooks/latest_review/haideptry_fast0947/biohub-sota-0-947-fast-2xt4-22m-divnet-3d.ipynb",
    "experiments/candidates/c003_medal_frontier/biohub-c003-medal-frontier.ipynb",
    "experiments/candidates/c004_adaptive_lineage/biohub-c004-adaptive-lineage.ipynb",
]

keys_of_interest = [
    "BIOHUB_MOTION_RELINK_TIGHT_UM",
    "BIOHUB_SAFE_DIV_SISTER_MAX_UM",
    "BIOHUB_SAFE_DIV_MAX_UM",
    "BIOHUB_SAFE_DIV_DIVERGE_UM",
    "BIOHUB_SAFE_DIV_SISTER_SYMMETRY_TAU",
    "BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM",
    "BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP",
    "BIOHUB_PPSWEEP_SELECT_MARGIN",
    "BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD",
    "BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT",
    "BIOHUB_VALIDATOR_ENABLE",
    "BIOHUB_VALIDATOR_N_PER_TYPE",
]

for p in notebooks:
    path = Path(p)
    if not path.exists():
        continue
    nb = nbformat.read(path, as_version=4)
    full_code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    print(f"=== {path.name} ({len(nb.cells)} cells, {len(full_code)} chars) ===")
    
    envs = {}
    for match in re.finditer(r'os\.environ(?:\[(?:\'|")([^\'"]+)(?:\'|")\]|\.setdefault\((?:\'|")([^\'"]+)(?:\'|"))\s*=\s*(?:\'|")([^\'"]+)(?:\'|")', full_code):
        k = match.group(1) or match.group(2)
        v = match.group(3)
        envs[k] = v
        
    for k in keys_of_interest:
        print(f"  {k}: {envs.get(k)}")
        
    pp_match = re.search(r"PP_CANDIDATES\s*(?::\s*dict\[str,\s*dict\])?\s*=\s*(\{.*?\n\s*\})", full_code, re.DOTALL)
    if pp_match:
        try:
            pp = eval(pp_match.group(1))
            print("  PP_CANDIDATES:", pp)
        except Exception as e:
            print("  PP_CANDIDATES raw lines:", [line.strip() for line in pp_match.group(1).splitlines()[:10]])
    else:
        print("  PP_CANDIDATES: None")
    print()
