# Candidate C011: x138 without the private V1284 head

- **Parent**: `anvithpothula/biohub-x138` v1 — public LB 0.953, submitted 2026-09-21 (after the 2026-09-14 metric patch). Local pull: `state/notebook_radar/pulled/biohub-x138/biohub-x138.ipynb`.
- **Notebook**: `biohub-c011-x138-zero.ipynb` (built by `src/build_c011_candidate.py`, verified by `src/verify_c011.py --replay`)
- **Kernel id (not pushed)**: `taeyangg4/biohub-c011-x138-zero`
- **Status**: STAGED, verified offline. Not pushed, not submitted — submission only on the user's explicit instruction.
- **Supersedes**: C010 (`c010_x138_flow_fusion`), which added C004's wide division envelope on top of this.

## What changed relative to x138

Only code cell 4's private-head block (plus the preset/label strings printed by cell 0):

```python
# x138
_myhead = sorted(Path('/kaggle/input').rglob('biohub-v1284-head-s075/v1284_head.pt'))
if len(_myhead) != 1:
    raise RuntimeError(('my V1284 head mount mismatch', ...))   # every fork dies here
os.environ['V1284_MODE']='candidate'
os.environ['V1284_HEAD']=str(_myhead[0])

# C011
os.environ['V1284_MODE'] = 'zero'
```

`zero` is the author's own pass-through mode. `verify_c011.py` checks it numerically: `refine()` returns the detector coordinates unchanged, and the module's trilinear feature lookup equals the support pack's native integer gather exactly (CPU and CUDA, fp32 and fp16, boundary coordinates, masked slots). So C011 is x138's pipeline minus the head's sub-voxel shift. All environment settings in cell 0 are identical to x138:

- neighbourhood-flow motion prior (`FLOW_MODE=seed`, K=12, radius 40 µm, flow tight gate 7 µm)
- discarded-detection readmit (4 µm, score ≥ 0.965)
- low-detection gap filler (gap ≤ 3, score ≥ 0.5, no synthetic nodes)
- runtime guards (validator off, ILP timeout 1200 s, repair deadline 7.5 h)
- x138's division geometry: max 9 / sister 14 / tau 0.6 / diverge 2.25 / child 10 / global cap 0.00375

## Why not C010's C004 division envelope

C004's LB 0.948 did **not** come from sister 16 / diverge 0.5. C004's in-notebook validator rejected that envelope and rewrote the submission with `combo(div_base_strict+tight52)`, i.e. B0's strict division geometry plus a 5.2 µm tight relink gate (`tmp/c004_log/ppsweep_selected.json`, log line "VALIDATOR rewriting submission with config: combo(div_base_strict+tight52)").

Replaying post-processing on the 12 ILP graphs C004's Kaggle run saved (`src/eval_pp_variants_local.py`; it reproduces C004's validator table to 6 decimals — 0.935924 / 0.951094 / 0.952350):

| variant on x138 post-processing | 12 movies | val8 | vis4 | 44b6 | 6bba | division tp/fp/fn |
|---|---:|---:|---:|---:|---:|---|
| x138 as configured (= C011) | **0.9424** | 0.9538 | 0.9081 | 0.9554 | 0.9361 | 3/3/12 |
| + C004 wide envelope (= C010) | 0.9354 | 0.9443 | 0.9072 | 0.9325 | 0.9337 | 3/16/12 |
| flow prior off | 0.9347 | 0.9511 | 0.8915 | 0.9488 | 0.9269 | 3/3/12 |
| tight gate 5.2 µm | 0.9394 | 0.9511 | 0.9043 | 0.9520 | 0.9332 | 3/3/12 |

Score = size-weighted adjusted edge Jaccard + 0.1 × division Jaccard (official formula). The wide envelope leaves edge Jaccard unchanged (0.9258 both) and multiplies division false positives by five, costing 0.007 on every split. Readmit and gap filler are idle in this replay (the low-detection dump is only written during inference), and the ILP graphs come from C004's Reyhan inference, not x138's; the comparison is between post-processing settings on identical inputs.

## Expected LB and risks

- The only thing removed from the scored x138 is the V1284 head. Its author reports a 10.4 % reduction in centre error on held-out train movies; its LB effect is unknown, so C011 should land somewhere below or at 0.953, not above.
- Node coordinates are rounded to integers when the submission is written, so the head's sub-voxel shift mainly acts through edge features and linking geometry.
- Runtime: identical to x138 except no head inference; the 7.5 h repair deadline and 1200 s ILP cap stay in place.
