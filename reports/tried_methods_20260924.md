# Methods already tried or analysed by team taeyangg4 (as of 2026-09-24)

Our best public LB: 0.952 (C012 = x138 pipeline + our own V1284 coordinate head). Public frontier notebook: anvithpothula/biohub-x138 0.953.
Baseline stack of almost every public notebook >= 0.94: pilkwang UNet3D + node transformer (two seeds, logit blend) -> ILP (tracksdata / SCIP) -> post-processing.

## Pipeline components already in our submissions (the x138 / harmonic-fusion lineage)
- dual-seed UNet3D + transformer, harmonic (bidirectional) fusion of forward/reverse edge softmax, secondary-seed TTA, DeepCenter veto (DeepCenter UNet heatmap at 8-view TTA), detection-logit TTA
- ILP (edge weight -1, appearance 0, disappearance 2, division weight 1.2), ILP timeout, validator off
- post-processing: Hungarian motion relink (tight 5.5 um + relaxed pass) with neighbourhood-flow motion prior (thtennant flow2), discarded-detection readmit, low-detection gap filler (single-frame + gap-2), rule-based safe-division repair (parent <= 9 um, sister <= 14 um, symmetry 0.6, orphan daughter only, DeepCenter veto, global cap 0.375 %), short-track filter, line-fit smoothing
- V1284 coordinate refinement head (frozen 224-d UNet features -> bounded shift < 2 um); we trained our own (heads v1/v2/v3, strength x0.5/x1.25/x1.5) and compared with x138's public head (anvithpothula/biohub-v1284-head-s075)

## Tried and rejected (LB or local evidence)
- detection threshold 0.95 / 0.96 / 0.965; DCTTA; HOCT veto (dual-HOCT); no-linefit
- edge-threshold / velocity-weight / momentum changes; aggressive short-track rescue (min_len 3)
- wide division envelopes (sister 16 / 18, parent 12, divboth, divmax), division symmetry tau 0.4 (thtennant divprec), division weight 0.5 / 0.7 in the old pipeline
- amanatar weak-leaf pruning + prefix guard + extended validator sweep (geometric fusion)
- sub-voxel centroid shift (takaito gshift), mutual-best association, DivNet gate (haideptry), density-adaptive association (haideptry)
- andnyu synthetic third-model edge tiebreak (LB = base)
- readmit off (C014/C015 lost on LB), readmit / gap-filler ablations
- locally trained UNet weights (RTX 4070 Ti) fused with the public ones (C008: LB 0.926, feature-scale mismatch)
- metric exploits (fake lineage hubs, negative-time nodes, synthetic forks): excluded on purpose

## Analysed on 2026-09-24 and closed (local, official metric, held-out movies)
- learned division scorer over (P, D1, D2) candidates after x138's rule (48 features: geometry, transformer edge probabilities, detection-peak scores, DeepCenter, raw intensity, track context; logistic / MLP; movie-grouped CV): per-parent precision 6-23 %, far below the ~35 % needed
- node-count term: T_true (organiser's estimate) not predictable from our counts; no node group with near-zero GT-match rate; removing readmitted / low-probability nodes +0.001 local (noise)
- ILP division weight 0.7 / 0.5 (578 / 1,838 ILP forks vs 15 true divisions) and re-injecting ILP forks after the relink: all worse
- transformer P->D2 probability for true divisions: median 0.11 vs competing parent 0.50 (the transformer does not see far daughters)
- head routing by embryo prefix (44b6 -> x138 head, 6bba -> ours): no end-to-end gain

## Not tried (known gaps)
- retraining / fine-tuning the detector or the edge transformer on an out-of-sample split
- any external or synthetic data (José Freitas CC0 synthetic set, Zebrahub, Cell Tracking Challenge)
- appearance-based division detection from image patches (CNN on t-1..t+2 crops), nucleus size features
- alternative linkers (Trackastra, Ultrack, end-to-end learned linkers, GNN)
