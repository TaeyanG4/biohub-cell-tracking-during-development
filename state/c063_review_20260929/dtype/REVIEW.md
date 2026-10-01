# C063 saved-prediction dtype diagnosis

The failed analysis assertion is a CSV dtype reconstruction issue, not missing
predictions or altered labels. This read-only audit covers all33,862 saved rows
(16,931 per source) and verifies their prediction receipt hashes.

`pair_diagnostics` subtracts FP32 decoder shifts from FP64 original physical
targets. Pandas writes FP32 shift columns with their FP32 decimal formatting.
Default `read_csv` then infers those decimal values as FP64; using them without
restoring FP32 in the algebra assertion changes the represented displacement
by up to2.3114e-7um. Saved residual diagnostics were computed before that CSV
round trip and retain their original FP64 result.

| Source fit | Raw CSV64 max3D discrepancy | Rows failing1e-9 | Restored FP32 max3D discrepancy | Rows failing1e-9 |
|---|---:|---:|---:|---:|
|44b6|1.9624974800791506e-7um|7,540|8.881784197001252e-16um|0|
|6bba|1.1768054886118762e-7um|7,832|8.881784197001252e-16um|0|

Restoring only `shifts = group[shift_columns].to_numpy(dtype=np.float32)`
before subtracting it from original FP64 targets also verifies abs-z and all
signed-axis residuals within8.88e-16um for every row. Independently recovering
the proposals from original targets plus saved FP64 signed errors yields
exactly the same FP32 proposals. No loosened tolerance is required.

Original node/GT IDs, row/time identities, targets, support eligibility and
source/opposite domain identities all match the unchanged C058/capture data.
Each source file has9,113 eligible and7,818 excluded rows. Every excluded
shift is exact0; excluded before/after3D and abs-z metrics are exactly equal.

The root recovery adapter can restore this dtype solely for the existing
algebra checks and retain the saved diagnostics and all original gates.
No source/model/prediction file was edited, no fit/inference ran, and no GPU
was used. `audit.json` contains the per-axis counts and input/source hashes;
`row_errors.csv` preserves both numerical discrepancy calculations per row.
