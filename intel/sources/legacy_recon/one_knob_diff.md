# One-knob configuration diff

The effective environment assignments are extracted from the pulled notebook source. This complements the notebook author's provenance table; it does not assume titles are scores.

## nusrati_0940 -> analytica_0941

- left: `nusrati/0-940`
- right: `analyticaobscura/biohub-lb-941`
- changed effective BIOHUB variables: **2**

| Variable | Left | Right |
|---|---:|---:|
| `BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD` | `<unset>` | `0.25` |
| `BIOHUB_GAP_CLOSE_UM` | `5.8` | `5.0` |

## analytica_0941 -> busyaprime_0942

- left: `analyticaobscura/biohub-lb-941`
- right: `busyaprime/biohub-0-942-lb-one-knob-past-the-public-line`
- changed effective BIOHUB variables: **1**

| Variable | Left | Right |
|---|---:|---:|
| `BIOHUB_DET_THRESHOLD` | `0.965` | `0.96` |

## Controlled one-knob result documented in busyaprime_0942

`BIOHUB_DET_THRESHOLD`: published 0.941 value `0.965` -> `0.96`.

Documented leaderboard sweep: `0.94 -> 0.938`, `0.95 -> 0.940`, `0.96 -> 0.942`, `0.965 -> 0.941`.

Independent second route: `BIOHUB_SECONDARY_EDGE_WEIGHT` `0.15 -> 0.25` also reached 0.942 with detection threshold held at the published value.

Robustness warning: the notebook states its held-out proxy moved in the opposite direction for the 0.965 -> 0.96 change, so the leaderboard optimum is not yet a validated CV optimum.
