# C055 REVIEW — Lineage Forge V12 guarded endpoint readmission on C023 (2026-09-29)

## 결정

**보류 확정 (97개 영상 완료).**
V12 방식은 C023 대조군 대비 총점이 오르지만, 그 이득은 전부 **C023 기존 readmission을 끈 효과**이며
(`readmit_off` 진단 arm이 더 높음), V12가 추가한 노드·링크 자체는 22개 영상 중 20개에서 소폭 손해다.
readmit-off의 로컬 이득은 과거 C015(C012 + readmit off)에서 LB 0.951 < 0.952로 전이되지 않았다(HANDOFF 18절).
portable/T4/제출로 진행하지 않는다.

## 구현 범위

- 새 후보 디렉터리 `experiments/candidates/c055_guarded_readmit/`. C023 원본 노트북, C054 소스·모델·캐시, pinned harness는 수정하지 않았다.
- `src/c055_guarded_readmit_stage.py` (sha `6515ed1f…`): V12 `readmit_open_track_ends`를 그대로 옮긴 stage.
  고정 조건 그대로: 점수 ≥ 0.94, 이동 ≤ 3 µm, 예측 위치 오차 ≤ 1.5 µm, 기존 노드와 ≥ 1.5 µm, 후보 비용 차 ≥ 0.5,
  양쪽 궤적 연결 우선, 추가 노드 ≤ min(100, floor(0.002·노드수)). V12와 같은 위치(motion relink + 단일 부모/자식
  정리 후, gap closer 전)에 삽입. C023 readmission은 `READMIT_RADIUS_UM = 0`으로 꺼서 중복 적용하지 않는다.
- `src/c055_guarded_readmit.py` (sha `af1abc10…`): `prepare | run | check_cache | study | analyse`.
  기존 `src/eval_pp_variants_local.py`(sha `0dc4d9c0…`, C054 pinned 입력이라 무수정)의 `build_namespace /
  install_frame_caches / load_raw_graph / apply_overrides`와 노트북 자체 `score_sample / aggregate_official`을 그대로 사용.
  기존 `Queue`(`src/run_last_days_local.py`) 재사용. 검출기·V1284 head·ILP 분열 가중치 1.2·나머지 후처리는 C023 그대로.
  재학습·임계값 탐색 없음. GPU 추론 없음(C054와 충돌 없음).
- 유일한 의미 적응(문서화): C023 노드 좌표는 V1284 head로 보정되어 원 peak에서 최대 1.64 µm 떨어진다.
  V12는 노드가 peak 위에 있어 1.5 µm 분리 검사가 자기 peak를 자동 배제하지만, 여기서는 기존 노드의 원 peak를
  프레임별 일대일 identity(≤ 2.5 µm, greedy)로 먼저 제외한 뒤 V12 분리 검사를 그대로 적용한다.

## 재현 명령

```bash
python src/c055_guarded_readmit.py prepare
python -X utf8 -u src/c055_guarded_readmit.py run      # 숨김 프로세스로 실행됨 (launch.json)
python src/c055_guarded_readmit.py analyse
python state/c055_diagnostics_20260929/missed_nodes_vs_peaks.py
```

Arms(`variants.json`): `control` = C023 as configured, `readmit_off` = `{"READMIT_RADIUS_UM": 0}` (진단),
`v12_guarded` = `{"READMIT_RADIUS_UM": 0, "GUARDED_READMIT": 1}`.

## 검증 1 — 저점수 검출 캐시 ≡ V12 후보 (97개 영상, `cache_semantics.json` passed)

- 같은 `_detect_cells_pooled`: 국소최대 마스크 `(logits == maxpool) & (sigmoid > thr)`는 임계값과 무관 → 0.3 덤프를
  ≥ 0.94로 거르면 V12의 0.94 캡처와 같은 집합. 점수는 둘 다 peak 위치의 sigmoid. 좌표는 downsample [1,4,4]로
  노드 격자에 맞춰 저장됨(y, x는 4의 배수 확인).
- 영상마다 `peaks > 0.965 수 == production 검출 수`(97/97), 덤프 행 == 고유 (t,z,y,x) 키(V12 dedup은 no-op).
- 보정 노드→원 peak 최대 1.64 µm; 일대일 배정에서 배정 실패 0, 최근접 충돌 2건(다음 후보로 해소).
- 0.94–0.965 구간의 추가 후보: 97개 영상 합계 67,132개.

## 검증 2 — C023 대조군 정확 재현

| split | harness CLI == 2026-09-26 저장 행 | in-process == CLI | 최종 그래프 == C046 `off` 그래프 |
|---|---:|---:|---:|
| heldout12 | 12/12 | 12/12 | 12/12 |
| confirm10 | 10/10 | 10/10 | 10/10 |
| extension75 | (대기) | (대기) | (대기) |

정수·점수 컬럼 완전 일치(부동소수 1e-12), 그래프는 ids/txyz/edges 배열 동일.

## 22개 영상 결과 (official 공식, 좌표 반올림)

| 그룹 | arm | 총점 | Δ총점 vs control | Δedge | edge TP/FP/FN Δ | div TP/FP/FN | 승/패 | 추가 노드 → 최종 생존 | 생존 링크 |
|---|---|---:|---:|---:|---|---|---|---|---|
| all22 | control | 0.945986 | — | — | 15940/648/611 | 5/6/19 | — | C023 readmit 6,157 | — |
| all22 | readmit_off | 0.947896 | +0.001910 | +0.001335 | −15/−17/+15 | 5/5/19 | 16/6 | 0 | — |
| all22 | v12_guarded | 0.947539 | +0.001553 | +0.000979 | −16/−15/+16 | 5/5/19 | 17/5 | 923 → 874 | 931 (bridge 57) |
| heldout12 | v12_guarded | 0.957553 | +0.002391 | +0.002391 | −1/−14/+1 | 3/2/12 | 10/2 | 561 → 532 | 582 |
| confirm10 | v12_guarded | 0.938307 | +0.001000 | −0.000282 | −15/−1/+15 | 2/3/7 | 7/3 | 362 → 342 | 349 |
| 44b6 (6) | v12_guarded | 0.938768 | +0.001754 | +0.001754 | 0/−2/0 | 1/1/3 | 4/2 | 308 → 289 | 323 |
| 6bba (16) | v12_guarded | 0.948193 | +0.001562 | +0.000895 | −16/−13/+16 | 4/4/16 | 13/3 | 615 → 585 | 608 |
| heldout12 | readmit_off | 0.957610 | +0.002448 | +0.002448 | −2/−14/+2 | 3/2/12 | 9/3 | 0 | — |
| confirm10 | readmit_off | 0.938932 | +0.001624 | +0.000342 | −13/−3/+13 | 2/3/7 | 7/3 | 0 | — |

**V12 고유 효과 = v12_guarded − readmit_off (all22): −0.000357 총점, edge TP −1 / FP +2 / FN +1.**
영상별로 20/22에서 −0.0001 ~ −0.0004(raw edge Jaccard 불변, 노드 항 `1 − 0.1·(t_pred − t_true)/t_true`의 선형
페널티만 발생), 2개에서 +0.0010/+0.0016(각 TP 1개), 1개(6bba_474be664)에서 −0.0139(TP −2, FP +3, FN +2).
추가 노드·링크는 최종 후처리까지 거의 다 살아남는다(874/923 노드, 931 링크). 살아남지만 주석된 GT edge를 회수하지 못한다.

## 왜 효과가 없는가 — 미주석 검출을 음성으로 보지 않는 진단 (`state/c055_diagnostics_20260929/`)

C023 대조군이 놓친 GT 노드 213개(22개 영상, 관련 GT edge 379개)의 최근접 검출 peak 점수:

| 구간 | 노드 수 | 관련 GT edge |
|---|---:|---:|
| 3 µm 안에 peak 없음 (≥ 0.3에서도 미검출) | 101 | 188 |
| > 0.965 (production 검출인데 파이프라인이 버림) | 91 | 155 |
| 0.94–0.965 (**V12 추가 구간**) | 6 | 12 |
| 0.5–0.94 | 14 | 22 |
| 0.3–0.5 | 1 | 2 |

V12가 새로 여는 점수 구간(0.94–0.965, 후보 67k개)에 주석된 미회수 세포는 22개 영상에서 6개뿐이다.
따라서 이 구간의 노드 추가는 공식 지표에서 노드 항 페널티만 낳는다. 남은 큰 몫은 (a) 검출기 자체가 못 보는
세포(101)와 (b) production 검출이 있는데 버려진 노드(91)다. (b)의 단계별 원인은 `dropped_detections_stages_22.csv` 참고.

## 97개 확장 결과 (22개 gate 통과 → 75개 추가, `official_summary.csv` / `analysis.json`)

extension75 대조군 재현: harness CLI == 2026-09-26 저장 행 75/75, in-process == CLI 75/75, 최종 그래프 == C046 `off` 75/75.

| 그룹 | arm | 총점 | Δ vs control | Δedge | edge TP/FP/FN Δ | div TP/FP/FN | 승/패 | 추가→생존 노드 | 생존 링크 |
|---|---|---:|---:|---:|---|---|---|---|---|
| all97 | control | 0.940490 | — | — | — | 27/46/124 | — | — | — |
| all97 | readmit_off | 0.940921 | +0.000431 | +0.000361 | −105/−40/+105 | 27/45/124 | 70/27 | 0 | — |
| all97 | v12_guarded | 0.940750 | +0.000260 | +0.000190 | −103/−38/+103 | 27/45/124 | 69/28 | 3,420 → 3,209 | 3,434 (bridge 226) |
| extension75 | readmit_off | 0.939243 | +0.000063 | +0.000063 | −90/−23/+90 | 22/40/105 | 54/21 | 0 | — |
| extension75 | v12_guarded | 0.939129 | **−0.000051** | −0.000051 | −87/−23/+87 | 22/40/105 | 52/23 | 2,497 → 2,335 | 2,503 |
| 44b6 (27) | readmit_off | 0.939165 | **−0.002695** | −0.002695 | −21/+8/+21 | 7/15/19 | 17/10 | 0 | — |
| 44b6 (27) | v12_guarded | 0.939020 | **−0.002840** | −0.002840 | −21/+8/+21 | 7/15/19 | 17/10 | 1,262 → 1,179 | 1,352 |
| 6bba (70) | readmit_off | 0.940624 | +0.000792 | +0.000710 | −84/−48/+84 | 20/30/105 | 53/17 | 0 | — |
| 6bba (70) | v12_guarded | 0.940450 | +0.000618 | +0.000536 | −82/−46/+82 | 20/30/105 | 52/18 | 2,158 → 2,030 | 2,082 |

V12 고유 효과(v12_guarded − readmit_off): extension75에서 72/75 영상이 음수(합 −0.0124, 최소 −0.00224, 최대 +0.00193),
raw edge Jaccard가 바뀐 영상은 7개뿐(나머지는 노드 항 페널티만). 97개 전체에서 추가 노드 3,420개·생존 링크 3,434개가
회수한 주석 edge는 TP +2(vs readmit_off)에 그친다.

## 진행/보류 판단: **보류 확정**

1. `v12_guarded`는 확장 75개에서 대조군보다 낮고(−0.000051), 44b6 배아에서는 −0.00284로 뚜렷이 낮다. 22개에서의 +0.00155는
   전이되지 않는 fit-domain 노이즈였다.
2. 대조군 대비 이득이 있던 곳도 전부 C023 readmission을 끈 효과(`readmit_off`)이며, 그 효과조차 44b6에서 −0.0027로
   배아 간 방향이 갈린다(HANDOFF 21절의 node-budget 레버 특성). C015(readmit off)는 LB에서 −0.001이었다.
3. 따라서 V12 방식 교체·`readmit off` 모두 portable/T4/제출 후보가 아니다. C023/C024 0.954 final pick 유지.
4. 근본 원인은 진단으로 확정: V12의 0.94–0.965 후보 구간에 주석된 미회수 세포가 거의 없고(6/213), 공식 지표의 선형
   노드 항이 회수 없는 노드 추가를 항상 벌한다. 후속 C056(보완 검출기 StrongUNet 후보 + 같은 guard)도 22개 gate 실패
   (`experiments/candidates/c056_complementary_readmit/REVIEW.md`).

## 입력/출력 해시

- 입력: `plan.json` `hashes` 5,450개(C023 노트북 `aa6aaaaf…`, harness `0dc4d9c0…`, predict script `ecaa548e…`,
  V12 cell 5 원문 `bd3e1a28…`, 97개 edge_cache npz, 97개 ILP geff, 97개 GT geff, 2026-09-26 대조 CSV 3개,
  C046 off 그래프 97개, DeepCenter checkpoint/manifest). 실행 시작·종료 시 재검증.
- 출력: `artifact_hashes.json` 316개 파일(replay/, study/, graphs/, 상위 csv/json), manifest sha256 `f86e73057c7a3fa2943e75679dd612da1e18840a2a08a764fee6cd20fda3c3ae`; 큐 종료 2026-09-28T18:43:54.242610+00:00 status `complete_review_required`, 9 jobs, 입력 5,450개 실행 전후 재검증 통과.
- 첫 실행(02:32 KST)은 cache 검사에서 실패(일대일 identity 이전 버전) → `state/c055_failed_launch_20260929/`에 보관, 과학적 출력 없음.
