> **SUPERSEDED 2026-09-23 — do not push or submit. Use C011 (`../c011_x138_zero/`).**
> The "C004 winning cytokinesis" premise is wrong: C004's own validator rejected sister 16 / diverge 0.5 and
> rewrote its 0.948 submission with `combo(div_base_strict+tight52)` (B0 strict division + 5.2 um tight gate).
> On x138's post-processing this envelope costs -0.0070 (0.9424 -> 0.9354 on 12 replayed movies; division
> FP 3 -> 16, TP unchanged). See `../c011_x138_zero/README.md` and `reports/pp_replay/`.

# Candidate C010: x138 Flow Fusion

**Candidate ID**: C010  
**Target Score**: >= 0.955 (Silver/Gold Frontier)  
**Parent Anchor**: `anvithpothula/biohub-x138` (public 0.953) + C004 calibrated cytokinesis  
**Status**: INITIALIZED & VERIFIED (Submission strictly held per user directive)  

---

## 1. Hypothesis & Architectural Overview

The top public notebook `anvithpothula/biohub-x138` achieved a post-patch score of **0.953** by introducing three complementary tracking improvements on top of the dual-seed harmonic baseline:
1. **Neighbourhood-flow motion prior**: estimates local velocity consensus across $K=12$ nearest neighbours within 40 um radius to guide motion re-linking in dense regions.
2. **Discarded detection re-admission**: re-admits high-confidence peaks ($\ge 0.965$) within 4.0 um of unlinked track endpoints before association.
3. **Low-detection sub-threshold gap filler**: bridges gaps up to 3 frames using detector sub-threshold peaks ($\ge 0.50$).

However, the original notebook contains a private dataset check (`biohub-v1284-head-s075/v1284_head.pt`) that raises `RuntimeError: ('my V1284 head mount mismatch', [])` on public forks. Analysis revealed that the coordinate refinement module natively implements `V1284_MODE = 'zero'`, which disables the head displacement and returns coordinates cleanly.

Candidate C010 fuses:
- **x138's 3 algorithmic pillars** with `V1284_MODE = 'zero'` (no private dataset required).
- **C004's proven winning cytokinesis division parameters** (`sister=16.0`, `existing_child=12.0`, `diverge=0.5`, `tau=0.85`), which drove our 0.948 breakthrough.
- **Conservative short-track filtering** (`min_len=4`, `min_prob=0.88`), avoiding the noise that caused C005/C006 regressions.
- **Strict exclusion of offline model weights** (eliminating C008's feature-space domain mismatch).

---

## 2. Key Parameter Comparison

| Parameter | B0 (Reyhan 0.947) | C004 (0.948) | x138 (0.953 public) | C010 (Flow Fusion) | Rationale |
|---|---|---|---|---|---|
| `MOTION_RELINK_FLOW_MODE` | `off` | `off` | `seed` | `seed` | Preserves x138 flow prior |
| `MOTION_RELINK_FLOW_K` | - | - | `12` | `12` | Consensus neighbourhood size |
| `MOTION_RELINK_FLOW_RADIUS_UM` | - | - | `40.0` | `40.0` | Spatial neighbourhood radius |
| `MOTION_RELINK_FLOW_TIGHT_UM` | - | - | `7.0` | `7.0` | Gated tight matching radius |
| `READMIT_RADIUS_UM` | `0` | `0` | `4` | `4` | Discarded detection recovery |
| `READMIT_MIN_SCORE` | - | - | `0.965` | `0.965` | High-confidence gate |
| `GAPFILL_MAX_GAP` | `0` | `0` | `3` | `3` | Multi-frame gap filler |
| `GAPFILL_MIN_SCORE` | - | - | `0.5` | `0.5` | Sub-threshold peak cutoff |
| `SAFE_DIV_SISTER_MAX_UM` | `14.0` | `16.0` | `14.0` | `16.0` | C004 cytokinesis envelope |
| `SAFE_DIV_EXISTING_CHILD_MAX_UM` | `10.0` | `12.0` | `10.0` | `12.0` | C004 division geometry |
| `SAFE_DIV_DIVERGE_UM` | `2.25` | `0.5` | `2.25` | `0.5` | C004 dynamic daughter separation |
| `SAFE_DIV_SISTER_SYMMETRY_TAU` | `0.6` | `0.85` | `0.6` | `0.85` | C004 cleavage asymmetry tolerance |
| `SHORT_TRACK_RESCUE_MIN_LEN` | `4` | `4` | `4` | `4` | Conservative short track hurdle |
| `SHORT_TRACK_RESCUE_MIN_PROB` | `0.88` | `0.88` | `0.88` | `0.88` | High precision threshold |
| `V1284_MODE` | - | - | `candidate` (fails) | `zero` | Eliminates private dataset dependency |
| Offline 4070Ti Weights | None | None | None | None | Strict exclusion of local weights |

---

## 3. Dataset Attachments

`kernel-metadata.json` attaches only the canonical, public, fully accessible datasets:
- `pilkwang/biohub-deepcenter-unet3d-center-prior-v1`
- `pilkwang/biohub-temporal-unet3d-seed314159-v1`
- `pilkwang/biohub-tracking-support-pack-50ep-v1`
- Competition data: `biohub-cell-tracking-during-development`

Zero private datasets (`biohub-v1284-head-s075` excluded). Zero local weights.

---

## 4. Submission Directive

**CRITICAL USER CONSTRAINT**: "제출은 하지말고 내가 하라고 하면 해" (Do not submit; only submit when told to).  
Submission is NOT authorized until explicitly commanded by the user.
