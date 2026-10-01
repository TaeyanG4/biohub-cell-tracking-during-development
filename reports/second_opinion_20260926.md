# 2026-09-26 독립 재검토: C023/C024 이후에 남은 시도

## 판단

C023 + C024를 현재 최종 선택으로 유지하는 데 동의한다. 다만 “모든 개선 방향이 실험으로 기각됐다”는 표현은 근거보다 강하다. 기존 설정 조정과 여러 학습/후처리 시도는 충분히 기각됐지만, 과거·미래 시간 문맥을 합치는 검출과 미래 궤적을 이용한 재할당은 C023 위에서 검증된 결과를 찾지 못했다.

추천은 **최종 후보를 보존하고, 하루 이내로 제한한 시간 문맥 융합 실험 하나를 우선 검토**하는 것이다. +0.001을 노릴 새로운 가설은 남아 있다. +0.01을 기대할 수 있다는 실증 근거는 없다. 이번 작업은 재검토와 진단이며, 개선 후보를 만들거나 Kaggle에 push/제출하지 않았다.

## 확인한 자료와 해석의 한계

- `AGENTS.md`, `HANDOFF.md` 특히 20–24절, `experiments/submission_log.csv`, C023/C024 README 및 실제 notebook.
- `reports/untried_ideas_20260924.md`, 네 public-ideas audit, 기존 replay/잔여 오류/프레임 이동 보고서.
- notebook radar SQLite. 저장된 0.950 이상 목록에 새 방법의 근거는 없었다. 웹 전체를 다시 크롤링하지 않았다.
- 실제 C023 predictor 및 TemporalUNet 구현, 기존 로컬 replay harness.

97개 movie는 독립 배아 97개가 아니다. 공개 검출기가 학습한 두 배아의 movie들이다. C025/C028/C029의 로컬 개선과 LB 하락은 이 검증 한계를 직접 보여준다. 과거 6bba local→LB 회귀식으로 새 실험의 LB를 예측하지 않는다. 0.001의 공개 점수 차이는 반올림된 관측값이며 private 효과나 통계적 유의성을 뜻하지 않는다.

이전 잔여 오류 비율의 직접 출처인 `reports/gap_headroom_c022.txt`는 **C022의 22개 movie**다. 이를 C023/C024의 97개 movie 진단으로 해석해서는 안 된다. 또한 `endpoint not detected/matched`에는 실제 미검출과 매칭 실패가 함께 들어 있어, 그 비율 전체를 검출 recall 개선 가능량으로 볼 수 없다.

## 현재 최종 후보를 다시 실행한 진단

기존 `eval_pp_variants_local.py`의 namespace/graph loader와 각 finalist notebook의 후처리·평가 함수를 그대로 사용했다. 올바른 head별 ILP graph와 lowdet dump, DeepCenter gate, 정수 좌표를 사용한 12개 movie 재생이다. 두 후보 모두 기존 저장 CSV와 **모든 movie의 adjusted edge 점수 및 edge/division TP·FP·FN이 정확히 일치**했다.

| 12개 movie 진단 | C023 | C024 |
|---|---:|---:|
| 공식식 로컬 총점 | 0.9546997382 | 0.9593167349 |
| GT edge 수 | 7,878 | 7,878 |
| 복원한 GT edge | 7,626 | 7,635 |
| 미복원: endpoint 미매칭 | 90 | 80 |
| 미복원: 양쪽 endpoint가 다른 링크에 연결 | 95 | 95 |
| 미복원: 한쪽만 다른 링크에 연결 | 66 | 67 |
| 미복원: 양쪽 endpoint가 모두 자유로운 순수 break | 1 | 1 |
| 양쪽 diverted 중 1:1 구조와 과거/미래 이웃이 있는 경우 | 89 | 86 |

89/86은 향후 재할당을 검토할 **구조적 대상의 수**일 뿐, 복구 가능한 수나 점수 상승의 상한 계산이 아니다. 현재 잘못 붙은 상대의 후속 구조, division 주변 보호, 모호한 영상 증거 등으로 실제 수정은 실패할 수 있다.

두 후보가 함께 맞힌 GT edge는 7,600개, C023만 맞힌 것은 26개, C024만 맞힌 것은 35개, 함께 놓친 것은 217개였다. 같은 기반 모델을 공유해 잔여 오류도 많이 겹친다. 이 합집합은 GT를 본 진단이며 실현 가능한 ensemble graph나 공식 점수가 아니다. node ID, 좌표, division, 소유권 충돌을 무시하고 CSV를 합칠 수 없다. 두 최종 선택을 유지하는 판단과, 새 graph ensemble에 큰 개선을 기대하는 판단은 구분해야 한다.

상세: `reports/finalist_residual_audit_20260926.json`, 같은 이름의 `.log`. 이 표는 **12개 train movie의 로컬 진단**으로, 97개 전체나 hidden test의 오류 분포가 아니다. C024의 로컬 우위를 이유로 C023을 제외하지 않는다.

## 1순위: 같은 프레임의 두 시간 문맥 활용

### 코드에서 확인한 사실

`tmp/c023_output/tracking_repo/scripts/predict_unet_transformer.py`:

- 2-frame window, stride 1로 `(t-1,t)`, `(t,t+1)`을 이미 모두 인코딩한다.
- `seen_frames`로 검출을 최초 한 번만 저장한다. 내부 프레임 t의 검출과 V1284 보정 좌표는 `(t-1,t)`에서 결정된다.
- 다음 `(t,t+1)`에서 계산한 t의 검출 map은 사용하지 않는다. t→t+1 transformer는 현재 window의 feature를 사용한다.
- TemporalUNet에 시간 attention이 있으므로 같은 t라도 이웃 프레임에 따라 예측이 달라질 수 있다.

이는 버그 판정이 아니다. 현재 모델이 정확히 이 처리 방식에 잘 맞을 수도 있다. 다만 공간 D4 TTA, edge probability pooling, head ensemble과는 다른 미검증 정보원이다. 같은 두 프레임의 순서를 뒤집는 것과도 다르다. 여기서는 **서로 다른 이웃 프레임**을 사용한다.

### 실제 모델로 실시한 작은 진단

두 train movie에서 미리 고정한 t=25,50,75를 사용했다. C023의 실제 dual-seed D4 encode/blend 코드를 실행하고 retention guard도 유지했다. TF32는 껐다. head, transformer, ILP, 후처리는 실행하지 않았다.

| 관측값: 총 6개 프레임 | 결과 |
|---|---:|
| 과거 문맥의 검출 후보 합계 | 3,819 |
| 미래 문맥의 검출 후보 합계 | 3,882 |
| 두 logit map 단순 평균의 후보 합계 | 3,856 |
| 미래 후보 중 과거 후보가 1.7µm 안에 없는 수 | 162 |
| 과거 후보 중 미래 후보가 1.7µm 안에 없는 수 | 99 |
| 과거 문맥 후보 수가 저장된 C023 inference와 일치 | 6/6 프레임 |

서로 다른 예측이 실제로 존재한다. **162개가 새로 찾은 진짜 세포라는 뜻은 아니다.** 후보 좌표 이동, false positive, NMS 변화도 포함한다. 평균 map의 후보 수가 약 1% 늘어난 것 역시 개선 증거가 아니다. 공식 metric의 node-count 항 때문에 불리할 수 있다.

상세: `reports/temporal_context_probe_20260926.json`, 같은 이름의 `.log`.
인코딩 소스 SHA256: `8140a92d916e0d65877d51ee53cc6bd0438b36538cda3365fa99fffa91d6eb25`.

### 시험한다면

1. C023을 기준으로 과거/미래 **검출 logit만 고정 1:1로 평균**하는 단일 변형부터 시작한다. head, edge model, 후처리 설정을 동시에 조정하지 않는다. 양쪽 문맥의 head feature까지 평균하는 것은 별개 실험이다.
2. 기존 predictor를 확장해 이미 계산하는 map을 보관한다. 내부 프레임의 검출 확정과 링크 예측을 지연하는 bounded buffer가 필요하다. 추가 UNet 인코딩을 반드시 두 배 수행할 필요는 없지만, 실제 메모리와 런타임은 구현 후 측정해야 한다.
3. off일 때 C023 재현을 먼저 확인한다. 이 변경은 검출 단계에 있으므로 기존 ILP graph의 후처리 replay만으로 평가할 수 없다. `src/run_kaggle_predict_local.py`로 다시 추론한 뒤 `src/eval_pp_variants_local.py`에서 정수 좌표·lowdet dump를 포함해 평가한다.
4. 기존 12개 movie로 먼저 기각 여부를 본다. 유망할 때만 동일한 C023 head를 사용하는 비교군과 함께 confirm-10, 나머지 75개로 확대한다. 기존 75개 head-v1 graph를 C023의 대조군으로 대신 쓰지 않는다.
5. 총점 외에 edge TP/FN, node 수, 두 배아별 방향, division TP/FP를 확인한다. 한 division 변화나 node-count 보상만으로 생긴 상승을 개선으로 승격하지 않는다. 작은 로컬 상승으로 weight/threshold sweep을 시작하지 않는다.

처음부터 연구를 늘리기보다 하루 예산 안에서 구현·초기 비교의 성공 여부를 판단하는 것이 적절하다. 로컬 검증이 좋아도 hidden 배아 개선은 미확인이다. Kaggle push/제출은 별도 사용자 명령이 필요하다.

## 2순위: 과거·미래 궤적을 이용한 제한적 재할당

현재 relink는 시간 순서로 실행하며 이전 속도와 이웃 이동장을 사용한다. 자식의 다음 이동까지 비교하는 양방향 궤적 목적함수는 별도 정보다. 기존 C019는 같은 frame pair의 ILP 이동장으로 **초기 prior**를 바꾼 것이므로, 미래 궤적을 사용한 최종 재할당의 실험을 대신하지 않는다.

`reports/public_ideas_audit_20260924_B_association.md`의 indarkarhana structured-trajectory는 공개 소스와 고정 계수가 있으며 기존 graph/edge cache로 시험 가능하다고 기록되어 있다. 그러나 그 전체 pipeline의 LB는 0.946이고 개선을 분리한 LB 증거가 없다. `lookahead`도 다른 계보에서는 LB가 같았다. 따라서 낮은 우선순위다.

노드·좌표·edge 수와 division 주변을 고정하고, 확실한 1:1 구간에서만 부모를 교환하는 진단부터 할 수 있다. acquisition jump를 제거한 좌표계에서 미래/과거 잔차를 비교해야 기존 stabilization 이득을 되돌리지 않는다. 실행은 기존 replay harness에 연결해야 하며 새 scorer를 만들지 않는다. 이 재검토에서는 교환 알고리즘이나 score 개선을 검증하지 않았다.

## 지금 우선하지 않을 방향

- **검출 threshold, relink gate, readmit/gap/linefit, ILP 비용, head 혼합비의 추가 sweep:** 반복된 local/LB 불일치와 기존 음성 결과가 충분하다.
- **단순 frozen-frame 또는 gap 복구:** 기존 C022 진단에서 frozen pair의 누락 34개 중 24개는 endpoint 미매칭이며 순수 gap도 거의 없다. 이 방향만으로 큰 상승을 기대할 근거가 약하다.
- **영상 optical flow:** 비강체 local flow는 미검증이나, 단순 phase correlation은 기존 199개 movie audit에서 jump의 GT 이동 대비 median 오차 3.30µm였다. 이미 작동하는 ILP median stabilization을 곧바로 대체할 이유가 없다.
- **검출기 재학습/새 좌표 head/새 division CNN:** 사용자 지정으로 종료한 시도를 반복하지 않는다. frozen-UNet edge transformer를 detector 후보 분포에서 학습하는 것은 C008과 다른 미검증 방향이지만, 구현·학습·검증 비용과 두 배아 검증 한계를 감안하면 이번 우선순위는 낮다.
- **외부 배아 검증:** 일반화 문제에 가장 직접적인 대응이지만 데이터·좌표·GT 포맷을 맞추는 비용이 있다. 기존 프로토콜로 손쉽게 검증할 수 있는 외부 데이터가 확보되지 않은 상태에서 남은 일정을 여기에 먼저 쓰지는 않겠다.

## 현재 결론

Claude의 **C023/C024 최종 선택 권고는 합리적**이다. 다만 완전 중단만이 유일한 선택은 아니다. 이 조사에서 확인한 가장 구체적인 미검증 가설은 **이미 계산하지만 버리는 미래 시간 문맥의 검출 map 활용**이다. 이를 고정된 작은 실험으로 검증할 가치는 있다. 결과가 나오기 전까지 최종 후보 추천이나 기대 private 점수는 바꾸지 않는다.
