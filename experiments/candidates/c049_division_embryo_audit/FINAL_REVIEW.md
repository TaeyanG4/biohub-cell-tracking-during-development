# C049 reviewed: no graph integration

Finished2026-09-28 12:22:09KST. Four jobs passed;4388 input and10 output hashes independently verified. All8111 stored labels checked against GT coordinates/degrees,real gradients/update/reload passed,and each25-epoch model trained on only the opposite embryo from its test set.

At fixed.5: train44b6/test6bba5TP27FP120FN,precision.15625,recall.04;train6bba/test44b613TP40FP13FN,precision.24528,recall.5. Correct precision at attained recall>=.5 is.04026 and.30233 respectively,with AP.07872/.29822. These PR operating points are descriptive only,never deployment thresholds. Classifier performance on sampled GT-centre negatives does not support valid predicted-graph edits,particularly at the much lower division rate at inference.

Close this fixed corrected CNN recipe without integration,notebook,T4 or submission. Original C031 raw artifacts remain;its purported.579precision at.5recall was actually44/76precision at44/151recall=.291. No saved old probabilities permit reconstructing the exact correct old PR point. No blanket claim that all division learning fails.

Next bounded historical correction is C050 for the separate C016 geometric/intensity candidate scorer,using its existing loader,fit,predict and per-parent metric functions with whole-embryo folds. This resolves a different invalid movie-CV conclusion and does not repeat the failed image CNN or change thresholds to rescue it.
