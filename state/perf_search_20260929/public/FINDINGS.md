# Public-ideas lens (2026-09-29, read-only; no crawl, no GPU, no Kaggle writes)

Inputs: radar.sqlite3 (272 notebooks, last crawl 2026-09-28 01:35 UTC; 91 refs >= 0.945), reports/untried_ideas_20260924.md,
reports/untried_ideas_discussions_20260924.md, reports/public_ideas_audit_20260924_{A,B,C,D}.md, HANDOFF sections 17/18/21/22/31/34/42/43,
state/public_lineage_review_20260929/REVIEW.md, pulled sources (zhincez, review_20260927_continuation), C055 diagnostics.
Scratch: radar_top.txt (score-sorted dump), short_track_boundary_proxy.txt (new computation).

## Score landscape (all post-metric-patch unless noted)
- Every public notebook >= 0.948 is x138 (anvithpothula, 0.953, run 2026-09-21) or a copy: kunaldesale2408 0.953 (whitespace copy), raunakdey07 HF-V3 0.953 (x138 minus comments), amanatar optimized 0.953 (V4 inaccessible; V6 errored, official visible4 0.925 in our T4 audit), amanatar geometric-fusion 0.948, kksky9k det955 0.948 (403).
- Our stack already exceeds all of them: C023/C024/C042/C043 = 0.954. Nothing in the radar (last crawl 09-28) is above 0.953.
- Newest since the 09-24 sweep: anhadmahajan06 (0.946, HF + validator sweep), mtoshidesu (0.947, LF-DCTTA copy), amanatar V6 (audited, C039 closed), denpugovkin exp003 (0.885, exploit lineage), sarveshchhetri (0.43), flexonafft Lineage Forge V12 (0.947 belongs to v8; V12 unscored; tested as C055, closed).
- Metric-patch caveat: HANDOFF cites a 17 July patch (host) and a 2026-09-14 rescoring; all anchors quoted here were scored after 2026-09-14 (x138 09-21, C012 09-23, C023/C024 09-25/26, C042-C045 09-27/28). The radar 'stale' tag on 0.953 rows is a heuristic, not evidence of pre-patch scoring.

## Public ideas already tested by us (do not repeat)
jump-stabilized relink (C017-C022 +0.001-0.002 LB), V1057 ILP-edge restore (C021/C022), x138 head (C023 0.954), head ensemble (C024),
readmit off (C015 0.951 < 0.952), readmit3/gap4/linefit0.6 (C025/C028/C029 0.950-0.952), JS log-pool TTA (negative), AdaBN (closed),
structured trajectory (C033 negative), appearance ReID (C034/C035/C038/C041/C046: <= +0.0003, not submitted), local registration (C036),
temporal-context lookahead (C032 negative), amanatar V6 primary-max fusion + published division rule (C039 -0.002/-0.007),
weak-leaf/weak-edge filter (-0.004), Lineage Forge V12 guarded readmit (C055 -0.00036 vs readmit_off; 44b6 -0.0028), StrongUNet readmit (C056),
division CNN/scorer (C016/C031/C049/C050 precision 0.04-0.25 < 0.3 bar), ILP fork re-injection, division weight 0.7/0.5, det 0.96, sister16, tau 0.4.

## Untested public ideas that survive the audit (ranked; none has positive isolated LB evidence)
1. Time-boundary-exempt short-track filter (denpugovkin, audit D #5): NEW quantification below -> ceiling +0.0014 vs node-term cost ~-0.0008..-0.0011 on 22 movies. Not worth a slot.
2. Temporal nucleus-size division feature (zhincez, no LB; volume -27% at +3 frames, CI clears zero at 7/9 lags; peak brightness unchanged): only route with a physical signal not yet in any tested division scorer (C016 used intensity, C031/C049 raw crops). Needs precision >= 0.3 at recall >= 0.3 on whole-embryo opposite folds (HANDOFF 43 arithmetic). Untested; ~2 h AUC check on existing crops.
3. andnyu node-count-neutral synthetic-detector peak swap (0.946 vs 0.947 base, -0.001): needs a 3rd model download + re-inference. Low.
4. yudaiyamauchi secondary 'adaptive' blend w0.30 (0.946 vs its unlisted base; 'adaptive' was the 0.940->0.941 CONTROL mode): pre-ILP logit change, superseded by relink. Low.
5. indarkarhana AR(2) endpoint-union reconnect (code only in dataset, never fetched; LB 0.946 = base): our C055/C056 already showed endpoint readmission recovers ~0 annotated edges. Low.
6. pilkwang 22-feature association ranker in relink cost (lineage 0.915, needs dataset download): our own R3 rankers changed 0/0 top-1. Low.
7. Forward-acceleration t+2 bonus in relink (lonnieqin/yusuketogashi, flat at 0.915): overlaps x138 flow prior; capped by 88 wrong_association edges of 97 movies. Low.
8. DAE input pre-filter (0.942/0.946/0.935/0.938 non-monotone in alpha): noise. Low.
9. External-embryo validation data (7th place; Ultrack dorado / Zebrahub): not a score idea, but the only out-of-sample check; needs GB downloads and user approval. Open.
10. Test-time self-supervised fine-tune (hengck23 D6): idea only; AdaBN variant lost -0.02 on 44b6. Open/untested.
11. Lineage Forge V12 ILP division weight 0.4 / DeepCenter div 0.25 (unscored): 0.7/0.5 already produced many false forks (HANDOFF 21). Low.

## New computation: boundary-exempt short-track filter proxy (short_track_boundary_proxy.txt)
- Pool (C055 diagnostics, 22 movies): 28 missed GT nodes / 48 GT edges were in the ILP output but deleted by the short-track filter (min len 6). 10 nodes (15 GT edges) sit at t<=2; a further 4 nodes/8 edges (6bba_32db13fc t=3,4) belong to the same t=0 start chain -> <= ~23 GT edges recoverable by a start/end exemption; 16.5k GT edges on 22 movies -> ceiling +0.0014 raw edge Jaccard, if every pooled edge were recovered.
- Cost (ILP-graph proxy, components <6 nodes without fork, starting t<=2 or ending t>=T-3): 44b6 1,836 nodes of 160,676 (1.14%), 6bba 3,008 of 357,986 (0.84%) across 22 movies. Linear node term 0.1*K/T_true*J -> about -0.0011 (44b6) / -0.0008 (6bba). Post-relink counts will be lower than the ILP proxy, but net <= +0.0006 at best, far below the 0.005 LB-proof bar; consistent with HANDOFF 21 (short tracks match GT at 0.66%) and the C055 readmit_off sign flip on 44b6.
