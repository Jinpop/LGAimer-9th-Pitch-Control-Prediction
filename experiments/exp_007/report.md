# exp_007 — 결합 실험 (F구체제 제외 + trackman 프로파일 + 보정C) · 첫 제출 스펙 확정

> 작성: 2026-08-05 · base=exp_001 · 코드: `src/train.py`, `src/finalize.py` · 수치: `metrics.json`, `metrics_folds.json`, `metrics_final.json`

**V1 채택 · 신규 베스트 fold_2024 valid-R Brier 0.249061 (BSS +0.00334, AUC 0.5338)** — 직전 베스트 exp_005(0.249104) 대비 −0.000043.
한 줄 결론: 3변수는 상쇄 없이 결합되나 실현율 71%. 미달분은 변수 간 간섭이 아니라 **"ES→P4" 정직한 iter 정책의 대가**다.

## 1. V1/V2 비교 (fold_2024 valid-R, n=223,497, base rate 0.4897)

| 스펙 | iter(P4) | 보정 전 | 보정 후 | BSS | AUC | Δ vs exp_001 |
|---|---|---|---|---|---|---|
| exp_001 기준선 (참고, ES iter 45) | 45 | 0.249337 | — | +0.00223 | 0.5341 | — |
| exp_005 직전 베스트 (참고) | 45 | — | 0.249104 | +0.00316 | 0.5341 | −0.000233 |
| **V1 마스크+프로파일+보정C** | **23** | 0.249288 | **0.249061** | **+0.00334** | 0.5338 | **−0.000276** |
| V2 마스크+보정C | 23 | 0.249379 | 0.249149 | +0.00298 | 0.5313 | −0.000188 |

전 fold(보정 전): fold_2022 V1 0.249320 / V2 0.249329 · fold_2023 V1 0.249506 / V2 0.249504.

⚠️ 이 비교는 **V1에 불리**하다 — exp_001/005의 iter 45는 valid 2024를 본 ES값(P0, 본번 불가)이고, V1의 23은 2024 무참조 P4 승계값이다.

## 2. 가산성 판정: 약한 준가산(sub-additive, 실현율 ~71%)

| 항목 | 값 |
|---|---|
| exp_001 기준선 | 0.249337 |
| 개별 delta 합 (−0.000088 −0.000070 −0.000233) | −0.000391 |
| 완전가산 기대치 | 0.248946 |
| V1 실측(보정 후) | 0.249061 |
| 실측 delta / 가산 대비 갭 | −0.000276 / **+0.000115 미달** |

분해:
- **보정C −0.000227** (0.249288→0.249061) — 단독 −0.000233의 97% 유지. 결합해도 보정 갭은 그대로 남는다(δ_fold=−0.1006).
- **프로파일 −0.000088** (V1−V2) — 단독 −0.000070보다 **커짐**. 마스크가 프로파일 S/N을 개선한 정황(exp_003 다음제안 2 적중).
- **마스크는 이번 스펙에서 회수 안 됨**: V2 보정 전 0.249379 vs exp_001 0.249337 = +0.000042. 단 iter 정책과 교락 — V2를 자기 argmin(41)으로 보면 0.249247로 **exp_004(0.249249)를 사실상 재현**한다. 갭 +0.000115의 대부분은 **P4 정직성 비용**(V2 +0.000132, V1 +0.000151, exp_006 예상 +0.0001과 정합)이다. 역으로 exp_001/004/005 개별 수치가 낙관 편향돼 있었다는 뜻.

세그먼트(V1 보정 후): seen_R 0.248892 / **cold_start_R 0.249737(n=44,672)** — 격차 +0.000845로 exp_001(+0.000136)의 6배. 개인 신호를 실제로 쓰기 시작했다는 방증이자 2025 신규 투수 비중 리스크. 프로파일 커버리지 80.7%(有 0.249244 / 無 0.249473) — 본번은 컷오프 2024라 ~99% 기대되므로 기여 과소평가 여지 유지. pred_mean 0.4817 vs 실제 0.4897로 이번엔 **−0.8pp 과보정**.

## 3. 확정 제출 스펙

| 항목 | 값 |
|---|---|
| 학습 | ≤2024 전체, `season≤2022 & game_type=="F"` 제외 → **1,369,784행** |
| 피처 | 제공 47 + as-of 프로파일 65 = **112** (프로파일 컷오프 2024, 685투수) |
| iter | **49** (exp_006 P4: fold_2024 valid-R Brier argmin 무스케일 승계, ES 없음) |
| 파라미터 | exp_001 동일 + `deterministic:true`, `force_row_wise:true`, `num_threads:8`, seed 42 |
| 보정 | δ = **−0.07401895011552417** |
| δ 근거 | R 연도별 성공률 2019~2024(0.54949/0.526925/0.512763/0.503691/0.503118/0.489707) 선형추세 **−0.010840** → **r̂₂₀₂₅=0.476341**; 앵커 = 최종모델의 2024 R train 223,497행 예측평균 **0.494829**; δ=logit(r̂)−logit(m) |
| vocab | **train.csv 전용** (pitcher 792/batter 830/팀 13) |

결정론 검증 통과: `deterministic=true` 재학습 → fold_2024 valid-R 0.249288, train.py 헤드라인과 **gap 0.0**.

## 4. 스모크 결과 (`submission/inference.py`, 로컬 5행 스텁, 서버 venv = 평가서버 스택)

utf-8-sig 읽기 → `./output/submission.csv` **utf-8 BOM 없음**·index 없음·`row_id,control_success` ✔ / **sample_submission row_id 순서 완전 일치** ✔ / 5행, 값 전부 (0,1) 개구간, NaN·inf 0 ✔ / 예측평균 0.4627 / 스텁이 **미등장 투수 3행·프로파일 없음 4행·미등장 타자 1행**을 포함해 cold-start·결측 경로 실통과 ✔ / **2회 실행 산출물 byte 동일** ✔ / **스케일 리허설 245,789행 → 4.7초** (제한 600초).

## 5. 특이사항·감사 유의 (자기신고)

1. **vocab 비대칭**: fold 검증은 vocab을 train.csv 전체(valid 연도 포함)로 만들어 2024 신규 투수도 고유 코드를 받았으나, 제출 경로는 train-only vocab이라 2025 신규 투수가 결측이 된다. 제출 경로가 규칙상 옳고 리크도 아니지만 **fold 수치가 cold-start 구간에서 미세 낙관**일 수 있다 — 미실측.
2. fold 헤드라인 iter(23)와 최종 모델 iter(49)가 다르다 — 둘 다 P4의 올바른 적용(전자는 fold_2023 argmin, 후자는 마지막 가용 fold인 fold_2024 argmin 승계). 최종 모델 자체는 검증 불가.
3. 스케일 리허설 입력은 train 2024행 합성물(시간 확인용, 점수 아님). 서버에 60MB `~/SungjinPark/experiments/exp_007/scale/` 잔존 — 홈 디스크 98% 사용 중, 정리는 사용자 판단.
4. zip 포장 미수행(packager 소관). `requirements.txt`는 `lightgbm==4.7.0`만 초안.
5. 에러·경고 없음. train.py의 곡선 계산 assert(raw 누적 vs 직접 predict 1e-8, numpy-코드 경로 vs pandas 경로 1e-10) 전부 통과.

## 6. 산출물

- `metrics.json` (표준형, V1 기준 + final_model 블록) / `metrics_folds.json` (V1/V2 전수) / `metrics_final.json` / `train.log` / `finalize.log`
- `src/train.py`, `src/finalize.py`
- `submission/inference.py`, `submission/requirements.txt`
- `submission/model/{final_model.txt (1.29MB), profile_2024.csv (617KB), feature_meta.json, calibration.json}`
- `submission/smoke/` (샌드박스 + `output/submission.csv`)
- 서버: `a5000:~/SungjinPark/experiments/exp_007/` — `model_V{1,2}_fold_*_1000.txt`(미회수), `submission_assets/`, `smoke/`, `scale/`
- SHA256: final_model `59592e67…f84c40` / profile_2024 `4712e1b3…7d6dc107` / feature_meta `a5d76e56…079e4422` / calibration `0ab27ee7…2ded6e5f`

## 7. 다음 제안

1. **leak-auditor 감사 필수** — vocab 비대칭, profile_2024가 trackman ≤2024만 집계했는지, δ가 train 통계만인지.
2. cold-start 보강(`has_profile` 결측 플래그 또는 pitcher_id 수축 target encoding) 1변수 실험.
3. **P4 비용 회수 우선순위 상향** — 정직한 iter 비용(+0.00013~0.00015)이 결합 이득의 절반 규모. lr 0.01~0.02로 곡선 평탄화 시 감소 여지.
4. 보정 −0.8pp 과보정 경향 → 2-파라미터 보정 또는 최근 3년 추세 병행 후 δ 캡.
