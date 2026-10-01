# C034 cross-embryo appearance diagnostic

Existing C023 candidates, unchanged original costs/assignments; no new official-score claim.

| Test embryo | Groups | Appearance net vs geometry | Fixed-margin net vs C023 |
|---|---:|---:|---:|
| 44b6 | insufficient known labels | — | — |
| 6bba | insufficient known labels | — | — |

Advance to official-replay implementation review: False

This gate requires both cross-embryo directions to show >=5 net gains in each comparison; it is a compute rule, not proof.

- Public detector already trained on both embryos.
- Unknown pairs excluded from fitting; ranking includes them, assessed only for reachable annotated successors.
- Capture-time node matching can differ from final matching after later stages.
- Per-source ranking ignores assignment competition and divisions; not an official-score improvement.
- Public DeepCenter descriptor disabled; context/centre features excluded to isolate image appearance.
