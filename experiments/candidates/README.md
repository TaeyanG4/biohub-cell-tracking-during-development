# Biohub Candidate Notebooks

This directory is the active development and submission-candidate surface.

## Operating Rules
1. `experiments/b0_public0947_exact_source/` is the immutable B0 score anchor and must not be edited for experiments.
2. Every real improvement branch gets a numbered directory: `cNNN_<short_name>/`.
3. A candidate notebook starts from a known parent and records that parent hash before changes.
4. Only candidates that pass the validation/promotion gates in `HANDOFF.md` may be considered for Kaggle submission.
5. All candidates must pass Python AST validation, zero configuration drift checks, and empirical full-train simulation before promotion.

---

## Active Candidate Registry

### C001: `c001_r3_temporal/`
- **Name**: `r3_temporal`
- **Parent**: `B0 exact Reyhan public-0.947` (SHA256: `01467DF4D109EA583643409AF04362A4C597FE4D2CE6D960D2EB0964EE919566`)
- **Status**: Development branch for R3 temporal-history context association.
- **Submission**: NOT AUTHORIZED

### C002: `c002_node_rescue/`
- **Name**: `selective_node_rescue`
- **Parent**: `B0 exact Reyhan public-0.947`
- **Status**: Complete GPU peak caching stage (`gpu_peaks/`, 196,991 peaks at $p \ge 0.10$).
- **Submission**: Local diagnostic / feature cache

### C003: `c003_medal_frontier/`
- **Name**: `medal_frontier`
- **Parent**: `B0 exact Reyhan public-0.947` (SHA256: `01467DF4D109EA583643409AF04362A4C597FE4D2CE6D960D2EB0964EE919566`)
- **Candidate SHA256**: `062587FA0C8D0985378CD06DD2C5E2019BF8711C033E5BA79B65A98B8466452B`
- **Target Score**: `>= 0.948` (Rank <= 213, breaking past 689-team 0.947 tie plateau into Bronze)
- **Status**: PUSHED & RUNNING on Kaggle Tesla T4 GPU (`taeyangg4/biohub-c003-medal-frontier` v1)
- **Key Features**: Tight 5.5 um motion relink, calibrated safe division (diverge 1.0, sym 0.8), expanded PP search grid.

### C004: `c004_adaptive_lineage/`
- **Name**: `adaptive_lineage`
- **Parent**: `B0 exact Reyhan public-0.947` (SHA256: `01467DF4D109EA583643409AF04362A4C597FE4D2CE6D960D2EB0964EE919566`)
- **Candidate SHA256**: `46a200b56528055989e14dd25c476da2768a2fadae0190584ec8f741e121ffa3`
- **Target Score**: `>= 0.949 ~ 0.950` (Rank <= 136, Solid Silver)
- **Status**: PUSHED & RUNNING on Kaggle Tesla T4 GPU (`taeyangg4/biohub-c004-adaptive-lineage` v1)
- **Key Features**: Sister max 16.0 um, exist max 12.0 um, diverge 0.5 um (+133.8% division recall on 199 GT graphs), sym 0.85, PP margin 0.001, collision-free combo builder.

### C005: `c005_gold_fusion/`
- **Name**: `gold_fusion`
- **Parent**: `B0 exact Reyhan public-0.947` (SHA256: `01467DF4D109EA583643409AF04362A4C597FE4D2CE6D960D2EB0964EE919566`)
- **Candidate SHA256**: `27d6ad6b665a1450045d6059f54e596c05b70a44dd822c46b3f042417c7e95ff`
- **Target Score**: `>= 0.959` (Rank <= 17, Gold Medal Safe Zone)
- **Status**: VERIFIED & STAGED (auto-push via `autonomous_relay_daemon` when GPU slot frees)
- **Key Features**: Kinematic anti-swap trajectory linking (velocity weight 0.75, bonus 1.25), cytokinesis envelopes, adaptive short-track rescue (len 3, prob 0.82), coordinated 2-frame gap stitching.

### C006: `c006_consensus_fusion/`
- **Name**: `consensus_fusion`
- **Parent**: `B0 exact Reyhan public-0.947` (SHA256: `01467DF4D109EA583643409AF04362A4C597FE4D2CE6D960D2EB0964EE919566`)
- **Candidate SHA256**: `64c62d457a021f14ec4176618567d48fd607e002621924b9f0ae0e8255953faf`
- **Target Score**: `>= 0.965` (High Gold Frontier)
- **Status**: VERIFIED & STAGED (auto-push via `autonomous_relay_daemon`)
- **Key Features**: Multi-seed consensus tracking & kinematic momentum (velocity weight 0.75, bonus 1.30), localized Strong gap anchor confirmation (confirm span 6.0 um, gap threshold 0.20), full-train calibrated cytokinesis, short track rescue (len 3, prob 0.82), orthogonal 10-candidate PP grid.

### C004 correction (2026-09-23)
C004's scored 0.948 output was rewritten by its in-notebook validator with `combo(div_base_strict+tight52)`
(B0 strict division geometry + `MOTION_RELINK_TIGHT_UM = 5.2`). The sister 16 / diverge 0.5 / child 12 envelope
listed above is the base config the validator rejected (held-out proxy 0.9359 vs 0.9511 strict).

### C010: `c010_x138_flow_fusion/` — SUPERSEDED, do not submit
- x138 + V1284 zero + C004 wide division envelope. Never pushed.
- Replay on x138 post-processing (12 movies): 0.9354 vs 0.9424 without the envelope (division FP 3 -> 16, TP unchanged).

### C011: `c011_x138_zero/`
- **Parent**: `anvithpothula/biohub-x138` v1 (public 0.953, submitted 2026-09-21, after the metric patch)
- **Candidate SHA256**: `b47fa95788280da49bb5200bb68881f7f0d5ee4796dbe8638175538bd55b2e84`
- **Change**: private V1284 head lookup -> `V1284_MODE='zero'` (pass-through); all settings identical to x138.
- **Verification**: `python src/verify_c011.py --replay` — ALL PASS (text diff, zero-mode tensor equality, metadata, replay identical to x138).
- **Status**: STAGED, not pushed, not submitted. Submission only on the user's explicit instruction.
