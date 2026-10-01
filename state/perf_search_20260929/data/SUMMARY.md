# DATA & GT lens summary (2026-09-29, read-only)
Scripts: gt_census.py (199 geff + zarr metadata), score_vs_gt.py (C023 97-movie control rows joined to census), gt_vs_pred_depth_intensity.py (22-movie C023 final graphs vs GT depth, 6-movie intensity sample, track boundaries).
Outputs: gt_census_per_movie.csv, gt_divisions.csv, c023_control97_with_gt.csv, gt_vs_pred_depth_22.csv, gt_vs_pred_intensity_sample.csv, gt_track_boundaries.csv.

- 199 train movies: 71 x 44b6, 128 x 6bba; every image zarr is (100,64,256,256) uint16, scale (1, 1.625, 0.40625, 0.40625) um.
- The 4 visible-test movies are byte-identical train movies (chunk md5 match). Hidden test = other embryos (HANDOFF l.926), size "roughly similar" to train, public LB = 29 %.
- GT: 133,318 nodes / 128,883 edges / 4,435 track fragments / 151 divisions. Organiser t_true (estimated_number_of_nodes) totals 4.73 M -> annotated node fraction 2.8 % overall; 44b6 0.77 % (median 2 GT nodes per frame vs 327 estimated), 6bba 5.4 % (8 vs 97).
- All GT edges are dt=1; 76 % of GT tracks start and end mid-movie (median fragment length 29 frames) -> GT is a sample of track segments, not lineages; only 12 movies have all tracks spanning the full t range.
- GT edge sources are flat over time (deciles 11.5k..13.7k) and over z (13-20k per 8-slice bin; GT slightly under-samples z 40-55 relative to C023 nodes: 0.106/0.078 vs 0.126/0.106).
- Divisions: 26 (44b6) / 125 (6bba), 87 movies have >=1, 112 have 0; all 151 lie in the 97 evaluation movies, 0 in the other 102. Sister distance p50 10.6 um (p90 14.4, max 20.3); parent->daughter p50 5.8 um (max 13.5).
- C023 97-movie control (size-weighted): 6bba carries 89.8 % of the weight; 29 movies = 50 % of the weight; the 10 worst movies hold 48 % of the weighted edge loss. Node term: +0.0153 for 44b6 (t_pred/t_true median 0.79), -0.00001 for 6bba.
- Per-movie raw J correlates with annotation density (Spearman +0.48; 6bba +0.67) and anti-correlates with organiser cell count (-0.54): dense, late-stage movies are both harder and less annotated, so the local aggregate is dominated by sparse, easy movies.
- Binomial annotation-sampling SE of the pooled 97 raw J = 0.00093 (single movie median 0.0083); a public-LB-sized set (~58 movies) has SE ~0.0012 from sparsity alone.
- Intensity sample (6 movies x 3 frames): GT nodes are not systematically brighter than C023 nodes (median rank 0.0-0.7 among predicted-node intensities); 6bba_474be664 GT nodes are dimmer than all predictions in 2/3 frames.
