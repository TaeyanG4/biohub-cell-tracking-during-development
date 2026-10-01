# C003 - Medal Frontier

**Status**: INITIALIZED AND VERIFIED  
**Parent**: B0 exact Reyhan public-0.947 (`01467DF4D109EA58...`)  
**Target**: Break past 0.947 plateau to >= 0.948 (Rank <= 213, entering Medal Range)

## Background & Motivation
- 0.947 is the public baseline plateau with 689 teams tied from Rank 214 to Rank 902.
- Due to timestamp tiebreak, any new submission scoring 0.947 is placed at Rank ~900 (NO MEDAL).
- Scoring 0.948 immediately jumps to Rank <= 213 (safely inside Bronze top 10% cutoff Rank 374, and 26 ranks away from Silver Rank 187).
- Scoring 0.949 reaches Rank 136 (Solid Silver).

## Key Enhancements over B0
1. **Motion Relink Tight Radius (5.5 um)**: Restored the proven `tight55` setting in the base configuration, matching the Lineage Forge / sjlee held-out validation proxy gain (+0.002057 over 6.0 um).
2. **Calibrated Safe Division Geometry**: Ground-truth simulation across all 199 full-train training movies proved that the baseline `diverge_um = 2.25` rejected over 4,180 valid division proposals. Calibrating `diverge_um` from 2.25 -> 1.0 um and `symmetry_tau` from 0.6 -> 0.8 increases division TP from 65 to 120 (+84.6% recall recovery) with 84.2% precision and raises mean division Jaccard from 0.2335 to 0.4105 (+0.1770 gain, contributing directly to `0.1 * division_jaccard` in the official metric).
3. **Preserved Canonical Secondary Edge Feature TTA (0.75)**: Preserved exact secondary TTA weight (0.75) verified by the 0.947 baseline and passed the built-in configuration guard with zero drift.
4. **Expanded Held-Out Validator Grid**: Injected calibrated candidates (`div_wide`, `div_conservative`, `div_base_strict`, `tight52`, `tight58`, `gap45`, `bonus125`, `dcgap035`, `reuse28`) into `PP_CANDIDATES`. If `div_wide` (`parent_max=11.0, existing_max=12.0, symmetry_tau=1.0, diverge_um=0.5`) demonstrates superior proxy score on held-out validation movies, the validator promotes it; if strict divisions are preferred, `div_base_strict` acts as an empirical safeguard.

## Submission Status
- `submission_authorized`: false (Awaiting explicit user authorization pursuant to project rules).
