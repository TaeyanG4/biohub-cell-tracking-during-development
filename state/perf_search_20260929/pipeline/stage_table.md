# C023 pipeline stages (from state/public_lineage_review_20260929/c023_code.py), run_stats magnitudes from c054/run_stats.csv (4 visible movies, C054 = 50/50 logit mean of C052 models on the same PP)

Order in filter_output_graph (base l.3542, V1057 wrapper l.3973):
1 detector (UNet, DET_THRESHOLD 0.965, 8-view D4 TTA patch l.1120) + dual-seed detection blend (secondary det weight 0.80, edge weight 0.15, low_margin_consensus 0.35, edge cand thr 0.48, retention guard 0.90; l.1067-1078)
2 V1284 head (x138 s075, candidate mode, <2um shift; l.1609-1660)
3 bidirectional harmonic association (weight 0.15; l.1279-1291)
4 ILP (edge -1.0*prob, appearance 0.0, disappearance 2, division 1.2, timeout 1200s; l.199-202,1327) -> raw_nodes/raw_edges
5 edge filter (next-frame, OUTPUT_EDGE_MAX_UM 14): dropped_long_edges 0-3/movie
6 stabilized motion relink (STAB_MIN_UM .001 every pair, MIN_PROB .5, MIN_EDGES 8; l.3924) wrapping Hungarian relink (seed gate 5.5, flow tight 7.0, relaxed 10, flow K12/R40/exclude1.5, LEARNED_BONUS 1.0, RAW_COST 0): replaces 100% of ILP edges (motion_relink_replaced_raw_edges == raw_edges); one-to-one => all ILP forks removed. Relaxed pass 25-369 edges/movie.
7 readmit (radius 4, MIN_SCORE 0.965 == DET_THRESHOLD; inside relink stage, then full re-relink): 80-550 nodes/movie; relink stats double-counted by second call.
8 single-parent repair: 0 dropped (relink already 1-1)
9 close_single_frame_gaps (GAP_CLOSE_UM 5.0, MAX_GAP env 2 but clamped effective 1 l.2568, density adaptive gain .04 step cap .125, reuse existing 3.2um, DeepCenter gap veto thr .25 only when span>=8.5um): 15-245 synthetic nodes/movie; gap_reused_existing 0 in 4/4; DC checked 3-41, rejected 2-36, bypassed(strong motion) 41-238
10 recover_strict_gap2 (10.2 total, 4.4 step, cap .0045 edges/180, context): 8-59 pairs, 16-118 nodes
11 fill_gaps_from_low_detections (pool >= .5, gap<=3, step 5, peak radius 3.5, no synthetic, budget 3%): 3-84 nodes
12 add_safe_divisions_postlink (parent<=9.0, sister<=14.0, existing child<=10.0, symmetry tau .6, diverge 2.25 (needs both children single successor at t+2), mutual NN, DC veto .25 with TTA, frame cap .0076, global cap .00375): geometric 15-212 -> DC rejects 3-172 -> added 5-33 = ALL output divisions (division_like_sources == safe_divisions_added)
13 division geometry filter: OFF (env 0)
14 prune isolated: 2-24 nodes
15 short-track filter (MIN_TRACK_LEN 6, keep division components; adaptive rescue trigger .10 removed frac, never triggered): removes 141-1716 nodes (0.8%-8.5%; 44b6_0b24845f 1716/20146 = 8.5%, just under the .10 rescue trigger)
16 linefit smoothing (weight .8, window 2): touches ~100% of nodes
17 V1057 ILP-edge restore (p >= .7; l.3969): restored 4-150, displaced 6-225 (6bba_05db0fb1 net -75 edges); runs after linefit/short-track, no re-prune; final graphs contain 66 isolated nodes across 97 movies (9/10/47 per split; c055 control npz)
Deadline degrade (27000 s) switches off relink/gap/gap2/safe-div/linefit but keeps readmit? (readmit lives inside relink -> off too) and short-track filter.
