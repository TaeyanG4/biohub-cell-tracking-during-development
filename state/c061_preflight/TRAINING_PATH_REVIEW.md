# Independent C061 training-path review

The reviewed `source_data`, `sample`, `train`, `load_model` and numeric policy
match the fixed model recipe. No blocker was found in this training path.
This review does not approve the separate scientific analysis/promotion gate.

The driver samples one source movie uniformly for each batch, then36 point
indices uniformly with replacement within that movie. Thus each example has
the required marginal movie/point distribution; examples within a batch
share the selected movie. Every batch has exactly four examples of each of
nine shifts and only XY reflections. Two separate source-only fits use seed6101,
1200 steps, FP32, fixed AdamW/cosine parameters, and only the final checkpoint.
The stored class counts must equal4800 per class/43,200 presentations per fit.
No opposite-embryo data enter `sample` or `train`. Direct GEFF coordinates
are checked integral before extraction, so `target=-shift` is valid for this
registered dataset; fractional-label support is unused.

`training_path_controls.py` executed the actual sampler eight times for each
of the existing benchmark movies. Both sources produced288 examples, exactly
32 of each class, from their own movie only. Independent reproduction of the
NumPy RNG sequence matched source selection, point indices and shift labels.
All576 actual returned crops matched independently sliced saved pixels with
one of the four permitted XY reflections; no synthetic pad or z flip appeared.

For both movies, independently reading original GEFF arrays and image metadata
reproduced the full-support pool and hash-first32 selected points. Six directly
read normalized image points plus54 shifted views matched the stored float16
pixels exactly after the declared conversion. GEFF identity, t/zyx center,
integer geometry and target sign all agree. These checks reuse the established
frame reader but calculate expected slices without the C061 crop functions.

An explicitly marked nonzero synthetic weight fixture under this preflight
directory exercises the actual `load_model` function. No optimizer or scientific
fit was run. State dictionaries and same-batch logits reproduce bit exactly;
train/eval outputs are exact. Reversing the batch changes nothing; evaluating
single examples differs by at most1.49e-8 from the full batch, with identical
decoded modes/tie flags. This is ordinary FP32 batch-shape arithmetic, not
learned cross-example dependence. The fixture's nominal steps1200 field exists
only to exercise the production loader; its actual_training_steps is0 and
control_only is true. It is not a candidate checkpoint.

The complete proof, relevant source/function hashes and exact counts are in
`training_path_controls.json`. All files created by this review are under
`state/c061_preflight`; no experiment source, candidate data, queue or graph was
modified. Later driver edits require checking whether the reviewed functions
changed before treating this proof as covering the revised source.

Follow-up static check: driver revision SHA256
`3fb0271d2553fd075cbb18b1b69cbc9c94f6ea9719ac4c45fa1a9b8fbb899a9c`
has identical reviewed `policy/source_data/sample/train/load_model/predict`
function hashes to the executed proof. Changes outside these functions do not
alter the bounded training-path controls above.
