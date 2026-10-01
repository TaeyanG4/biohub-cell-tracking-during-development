# C056 REVIEW — complementary-detector (StrongUNet) guarded readmission on C023 (2026-09-29)

## 결정: 보류 (22개 영상 사전 등록 gate 실패, 75개 확장 미실행)

`strongunet_guarded`는 C023 대조군보다 +0.001571 높지만 `readmit_off` 진단 arm보다 **−0.000339 낮고**,
44b6에서는 대조군보다 −0.001573 낮다. 추가된 56개 노드 중 48개가 최종 그래프까지 남지만 주석된 GT edge를
하나도 회수하지 못했고(edge TP −17 = readmit_off와 같음, 44b6에서 TP −2/FP +2), 이득은 전부 C023 readmission을
끈 효과다(C055와 동일 패턴; C015에서 LB로 전이되지 않았음). portable/T4/제출로 진행하지 않는다.

## 구현 범위

- `src/c056_complementary_readmit_stage.py`: C055 stage의 V12 guard(`make_readmit`, 일대일 identity 제외)를
  그대로 재사용하고 후보만 StrongUNet peaks(고정 공개 임계 p ≥ 0.5)로 교체. 삽입 위치·상수 동일.
- `src/c056_complementary_readmit.py prepare|run|cache_peaks|diagnose|study|analyse`: 기존
  `src/cache_strongunet_gpu_peaks.py`의 모델·전처리(`artifacts/hengck_point_detector/00000030.pth`, `model_v5.py`,
  [::4, ::4] 격자, ×4 좌표 복원)를 `data/train` zarr에 적용해 영상당 2.5 s로 peak 캐시(`peaks/`);
  C055 driver의 study/analyse 프로토콜과 pinned harness `src/eval_pp_variants_local.py`를 재사용.
  대조군 재현은 C055의 harness CLI 행과 2026-09-26 저장 행, C046 `off` 최종 그래프에 대해 정확 일치(22/22).
  `readmit_off` 행은 C055의 행과 정확 일치(22/22). 학습·임계값 탐색·GPU 추론(캐시 외)·Kaggle 쓰기 없음.
- 입력 15,353개 pinned(`plan.json`), 실행 전후 재검증. 첫 실행이 곧 유일한 실행(PID 57480, 02:59 KST 시작).

## 재현 명령

```bash
python src/c056_complementary_readmit.py prepare
python -X utf8 -u src/c056_complementary_readmit.py run
python src/c056_complementary_readmit.py analyse
python state/c055_diagnostics_20260929/strongunet_burden_ceiling.py
```

## 후보 상한(진단, `diagnose.json`) — GT 미주석 검출은 음성이 아니다

C023 대조군이 놓친 GT 노드 213개(22개 영상) 중 3 µm 안에 StrongUNet p ≥ 0.5 peak가 있는 노드 102개
(1차 검출기에 peak 없음 101개 중 30, production 검출이 버려진 91개 중 65, 저점수 21개 중 7); 관련 GT edge 173/379.
그러나 guard(열린 궤적 끝 + 선형 예측 오차 ≤ 1.5 µm + 비모호 + 예산)는 후보 258,787개 중 57개만 제안했고
56개를 추가했다. 놓친 세포는 motion-consistent한 열린 끝 옆에 있지 않다.

## 22개 영상 결과 (official 공식, 좌표 반올림)

| 그룹 | arm | 총점 | Δ vs control | Δedge | edge TP/FP/FN Δ | div TP/FP/FN | 승/패 | 추가→생존 노드 | 생존 링크 |
|---|---|---:|---:|---:|---|---|---|---|---|
| all22 | control | 0.945986 | — | — | 15940/648/611 | 5/6/19 | — | — | — |
| all22 | readmit_off | 0.947896 | +0.001910 | +0.001335 | −15/−17/+15 | 5/5/19 | 16/6 | 0 | — |
| all22 | strongunet_guarded | 0.947557 | +0.001571 | +0.000996 | −17/−13/+17 | 5/5/19 | 15/7 | 56 → 48 | 49 (bridge 1) |
| heldout12 | strongunet_guarded | 0.956891 | +0.001729 | +0.001729 | −4/−10/+4 | 3/2/12 | 8/4 | 55 → 48 | 49 |
| confirm10 | strongunet_guarded | 0.938932 | +0.001624 | +0.000342 | −13/−3/+13 | 2/3/7 | 7/3 | 1 → 0 | 0 |
| 44b6 (6) | strongunet_guarded | 0.935441 | **−0.001573** | −0.001573 | −2/+2/+2 | 1/1/3 | 3/3 | 49 → 43 | 43 |
| 6bba (16) | strongunet_guarded | 0.948571 | +0.001940 | +0.001274 | −15/−15/+15 | 4/4/16 | 12/4 | 7 → 5 | 6 |

V12 고유 효과(strongunet_guarded − readmit_off, all22): **−0.000339**; 한 영상(44b6_2a2eff9f)에서 23개 추가 노드가
TP −2 / FP +4 / FN +2(raw −0.0233)를 만들었고 나머지 영상은 노드 항 페널티만 있다.

## 왜 안 되는가 — 부담 대 상한 (`state/c055_diagnostics_20260929/strongunet_burden_ceiling_summary.csv`)

1차 peak(≥ 0.3)와 최종 노드 모두에서 1.7 µm 이상 떨어진 "보완적" StrongUNet peak:

| p ≥ | 보완 peak 수(22편) | 최종 노드 대비 | 커버되는 놓친 GT 노드 / edge |
|---|---:|---:|---|
| 0.5 | 196,198 | 37.5 % | 40 / 73 |
| 0.7 | 135,026 | 23.6 % | 32 / 60 |
| 0.8 | 71,267 | 11.3 % | 11 / 20 |
| 0.9 | 10,615 | 1.5 % | 0 / 0 |

공식 지표의 노드 항은 선형(`jac·(1 − 0.1·(t_pred − t_true)/t_true)`)이므로 ILP 수준의 후보 합집합도
p ≥ 0.8에서 약 −0.011, p ≥ 0.5에서 약 −0.038의 노드 비용 대비 최대 +0.004의 edge 이득으로 성립하지 않는다.
놓친 세포를 덮는 peak는 저신뢰(0.5–0.8)이며 GT 없이 다른 수만 개 보완 peak와 구분되지 않는다.

## 종합 판단 (C055·C056)

- 궤적 끝 복구(readmission)는 후보 출처(1차 0.94–0.965, 2차 StrongUNet)와 무관하게 주석 edge를 회수하지 못한다.
- 22개 영상에서 두 arm이 대조군을 이기는 유일한 이유는 C023 readmission을 끈 것(+0.0019)이며, 이 로컬 효과는
  과거 C015에서 LB −0.001로 전이되지 않았다. 따라서 "readmit off" 자체도 새 후보로 제출하지 않는다.
- 남은 미회수 풀은 1차 검출기 미검출(101/213)과 ILP/짧은 궤적 필터의 노드 선택(91/213)이며, 후자는 HANDOFF 21절에서
  닫힌 node-budget 레버다. 새 근거 없이는 이 계열을 재개하지 않는다.

## 해시

- 입력: `plan.json` `hashes` 15,353개(driver/stage/C055 driver·stage/harness/peak tool/StrongUNet 가중치·모델 소스/
  C023 노트북/97 캐시·GT geff·zarr/C046 off 그래프/C055 대조 CSV·study CSV/2026-09-26 대조 CSV).
- 출력: `artifact_hashes.json` 102개 파일(peaks/, study/, graphs/, 상위 csv/json), manifest sha256 `2a44c2a58f82607299f9d2acdc89bd9595ac087102edfce9b22f83b3b0872f0a`; 큐 종료 2026-09-28T18:25:52.210680+00:00 status `complete_22_gate_failed_review_required`, 6 jobs.
