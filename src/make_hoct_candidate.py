import json
from pathlib import Path

p = Path('experiments/exp_dctta_hoct_det096/biohub-lf-hoctveto-div-b.ipynb')
nb = json.loads(p.read_text(encoding='utf-8'))

def text(i):
    s = nb['cells'][i]['source']
    return ''.join(s) if isinstance(s, list) else s

s = text(0)
for a, b in [
    ('os.environ["BIOHUB_DET_THRESHOLD"] = "0.965"', 'os.environ["BIOHUB_DET_THRESHOLD"] = "0.96"'),
    ('os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD"] = "0.20"', 'os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD"] = "0.25"'),
]:
    assert s.count(a) == 1, (a, s.count(a))
    s = s.replace(a, b, 1)
anchor = 'os.environ["BIOHUB_PPSWEEP_MAX_ADJ_LOSS"] = "0.0005"\n'
assert s.count(anchor) == 1
s = s.replace(anchor, anchor + 'os.environ["BIOHUB_VALIDATOR_ENABLE"] = "0"\nos.environ["BIOHUB_MOTION_RELINK_TIGHT_UM"] = "5.5"\n', 1)
nb['cells'][0]['source'] = s

s = text(1)
assert s.count('"BIOHUB_DET_THRESHOLD": 0.965,') == 1
nb['cells'][1]['source'] = s.replace('"BIOHUB_DET_THRESHOLD": 0.965,', '"BIOHUB_DET_THRESHOLD": 0.96,', 1)

s = text(4)
old = 'os.environ["BIOHUB_SECONDARY_EDGE_FEATURE_TTA_WEIGHT"] = "1.0"'
assert s.count(old) == 1
nb['cells'][4]['source'] = s.replace(old, old.replace('1.0', '0.75'), 1)

p.write_text(json.dumps(nb, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
print('prepared controlled DCTTA + HOCT candidate')
