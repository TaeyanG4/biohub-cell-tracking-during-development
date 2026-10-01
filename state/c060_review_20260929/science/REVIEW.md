# Independent completed-C060 science review

**Close this fixed C060 recipe without a 97-movie extension, upstream integration, T4 run or submission.** It preserves the official score but does not demonstrate transferable localization improvement while protecting already-good points. This is a disposition of the registered recipe, not a conclusion that the user's localization research direction is exhausted.

This review aggregates retained outputs only. No fit, model inference, threshold/scale search or pinned-source edit occurred. The one affected movie was re-matched with the existing actual official matcher to identify its precise assignment exchange; this is not a new scorer or a graph modification. Root separately verifies manifests, fold provenance, source state and completion notifications.

## Official score: exact tie, including every movie

The original writer's actual official summaries are identical: **0.9431126093954155 total**, adjusted edge0.9293195059471396, edge TP/FP/FN15,940/648/611, division TP/FP/FN4/5/20,517,966 output nodes. Both embryos tie exactly. Independent comparison confirms all22 per-movie edge/division counts, node counts, node recall, raw/adjusted edge and node-ratio fields are identical. This is not a gain hidden by displayed decimal precision. IDs, times, counts, topology and all excluded coordinates are unchanged in every saved output graph.

## Actual intervention was sparse

| Saved runtime outcome | all22 | 44b6 | 6bba |
|---|---:|---:|---:|
| Eligible predicted triples |254,579|84,610|169,969|
| Eight-view unique-mode agreement |9,267|3,447|5,820|
| Disagreement/ambiguity abstentions |245,312|81,163|164,149|
| Agreement on zero correction |5,182|1,610|3,572|
| Ownership rejections after agreement |0|0|0|
| Nodes actually moved |4,085|1,837|2,248|
| Originally known matched nodes moved |109|20|89|

Only3.64% of eligible nodes pass the eight-view agreement guard;1.60% move. The spatial ownership guard is **not** the observed bottleneck: it rejects none of the agreed proposals. This does not establish that relaxing reflection agreement would help; the unaccepted proposals have not shown useful localization and must not be promoted by a post-result guard change.

Accepted moves are mostly small x/y corrections: median0.40625µm, mean0.50391µm,90th percentile0.8125µm, maximum2.60127µm. Across4,085 moved nodes, only **5 change z**, and **none of the109 originally known matched moved nodes changes z**. The mechanism therefore barely acts on the z-localization issue highlighted by earlier diagnostics. The cause cannot be assigned uniquely to prior, mode quantization, learned evidence or orientation agreement from these retained arrays. No unguarded counterfactual was evaluated.

The two fixed fits did complete1,200 steps and38,400 sample presentations each. Their per-batch source losses are not held-out fit-quality measurements, and no claim that the model learned a transferable center follows from them.

## Original-pair residuals, including abstentions

Positive change below means worse error. Every comparison keeps the original baseline GT identity, including the one subsequently unmatched node.

| Group | Count | Mean change (µm) | Improved / worsened / equal |
|---|---:|---:|---:|
| All eligible known pairs |8,594|+0.000024294|55 /51 /8,488|
| 44b6 eligible pairs |1,112|−0.000456026|12 /7 /1,093|
| 6bba eligible pairs |7,482|+0.000095681|43 /44 /7,395|
| Eligible tail>3.5µm |699|−0.002073252|6 /3 /690|
| Persistent eligible tail |523|−0.001854297|4 /1 /518|
| Broken-link eligible tail |168|−0.002853608|2 /2 /164|
| Eligible good≤2.5µm |7,102|+0.000177536|48 /44 /7,010|
| All originally good≤2.5µm |14,041|+0.000089798|48 /44 /13,949|

Tail means decrease on both embryos, but this comes from only9 changed tail points:4 improved44b6 points and2 improved/3 worsened6bba points. Persistent-tail means likewise decrease, based on3 improved44b6 points and1 improved/1 worsened6bba point. The source-tail objective receives only sparse positive evidence, not broad demonstrated recovery.

The critical fixed gate fails:6bba overall eligible error increases, and good-point mean error increases in **both** embryos (+0.000908654µm44b6, +0.000065047µm6bba among eligible good points). The large denominators make the aggregate changes tiny. Among the94 good points actually moved, the mean error increases0.013413µm; the largest individual harm is0.878425µm. These data justify neither portraying C060 as a major degradation nor claiming a useful improvement.

All originally good points retain their known identity, satisfying that part of the safety gate. Official metrics also satisfy nonregression. Those controls do not compensate for the missing transferable localization benefit.

## The one match exchange is not new GT recovery

All16,931 original matched predictions:16,930 retain their GT identity,1 becomes unmatched,0 change to another GT identity, and1 formerly unmatched prediction gains a match. Reusing the actual official matcher on saved44b6_12dfb391 graphs proves the GT ID set is **exactly unchanged**:

- At time80, unchanged predicted node39643 loses GT186000000051. Its coordinates remain(z,y,x)=(43,75,178), its original fixed-pair residual remains4.542013µm, and its model proposal abstained.
- Predicted node39577 moves(39,61,179)→(39,62,181) and gains that **same** GT186000000051. Its agreed move stays inside its original spatial ownership cell.
- Both original predicted paths and all edges remain unchanged. There is no new annotated GT coverage and no scored edge/division count change.

This also illustrates the limit of the ownership guard: keeping a prediction in its own nearest-center region does not guarantee preservation of global one-to-one GT matching. Sparse annotations do not establish which path is biologically correct. The formerly unmatched prediction must not be labelled a false biological cell.

## Scientific disposition and evidence boundary

The broad75.6% finding remains an association between large residuals and missed links, not a recoverable-error ceiling. C058 demonstrated harmful domain-dependent offsets; C060 greatly limits those interventions but mostly abstains, leaves every evaluated z coordinate unchanged, and supplies no consistent net localization gain. Those are different failures. The current evidence does not support merely lowering the agreement requirement, shrinking the prior, repeating epochs or changing a correction scale.

Keep localization as the user's research priority, but require a distinct measured mechanism before another fixed experiment. In particular, useful evidence must establish what can localize the annotated point across embryos and which uncertainty/identity information is missing. It should continue to separate persistent coordinate offset from per-frame matching changes. Full-pipeline gains still require a separate dependency-correct upstream test after localization itself passes; smaller residuals cannot create missing edges in this frozen-topology study when identities remain unchanged.

Root reported all3,824 input hashes matched. A `watcher.stdout.log` append after the original output manifest was written is an expected observer-side completion-notification event, not an experiment-result artifact. Root is verifying its parsed receipt/time equality and will record that exception explicitly without rewriting the original manifest. This science review does not independently certify the manifest exception.

Artifacts: `recount.py/json`, `runtime_movies.csv`, `changed_shifts.csv`, and `mapping_exchange.py/json` in this directory. All original studies and model checkpoints remain unchanged.
