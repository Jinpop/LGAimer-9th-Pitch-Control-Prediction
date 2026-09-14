# exp_003 — 투수 정적 프로파일 조인 (fold별 as-of)

> 2026-08-05 · A5000 실행 (CPU, 총 2.05분) · base = exp_001 (변경 변수 1개: 프로파일 피처 +65)
> 가설: "trackman 투수 정적 프로파일 조인이 fold_2024 valid-R Brier를 개선한다" → **소폭 개선 (경계선 채택 후보)**

## 1. 변경점

- exp_001의 47피처 + **투수 정적 프로파일 65피처** = 112피처. 그 외 전부 동일 (LGBM 파라미터·seed 42·fold·categorical 처리).
- **fold별 as-of 프로파일** (config 리크 규칙 준수): valid 연도 Y인 fold는 trackman `season <= Y-1`만 집계.
  exp_002의 `02_build_profile.py` 로직을 컷오프 파라미터화해 재사용 — `career_*`(컷오프까지 전체) + `recent_*`(컷오프 시즌만, exp_002의 s2024 일반화) + `last_season_gap`.
  exp_002의 2024년 말 산출물은 fold 검증에 **미사용** (valid 연도 물리량 유입 차단). 코드 내 assert로 컷오프 강제.
- 조인 키 `pitcher_id` (exp_002 `out/pitcher_id_map.csv`, 685명). 미매핑·프로파일 無 투수는 NaN 유지 (sentinel 금지).
- `num_threads=8` (exp_004~006 병렬 대비). **통제 확인**: exp_001 피처셋을 8스레드로 fold_2024 재실행 →
  exp_001 원본(24스레드)과 **전 세그먼트 6자리 완전 일치** (best_iter 45 동일). 스레드 효과 0 — 아래 delta는 순수 프로파일 효과.

## 2. 결과 (valid-R Brier, exp_001 대비)

| fold | exp_001 | exp_003 | delta | 방향 |
|---|---|---|---|---|
| fold_2022 | 0.249461 | 0.249511 | +0.000050 | 악화 |
| fold_2023 | 0.250527 | 0.250442 | −0.000085 | 개선 |
| **fold_2024 (주지표)** | **0.249337** | **0.249267** | **−0.000070** | **개선** |

- fold_2024 상세: BSS 0.002227→**0.002507** (+0.00028), AUC 0.534062→0.534903. valid_all Brier 0.249450→0.249368.
- LB 환산(대략, r≈0.49): ΔBrier 0.00007 ≈ **LB +28점** 수준.
- 채점 결정론 확정(08-05 결정 로그) 전제이므로 delta는 실제 차이로 서술. 단 fold 간 방향 혼재(2/3 개선) — 효과는 실재하나 약함.
- best_iteration: 49/6/49 (exp_001: 48/5/45).

## 3. 프로파일 피처 중요도 (fold_2024)

- **프로파일 gain 점유율 1.5%** (fold_2022 2.6%, fold_2023 1.8%). 상위권 진입 없음 — 최고 순위 29위 `career_breaking_rel_speed_mean`(366.1), 이어 `career_zone_speed_mean`, `career_offspeed_share`, `career_fastball_ivb_mean`, `career_horz_break_std`, `career_rel_side_mean`, `recent_breaking_rel_speed_mean` 등.
- 전체 상위는 exp_001과 동일 구도 (batter_id·pitcher_id·game_type·season·asof_pitcher_success_rate). 프로파일은 보조 신호로만 흡수됨.

## 4. Cold-start / 커버리지 세그먼트 (fold_2024, valid-R)

| 세그먼트 | n | exp_001 Brier | exp_003 Brier | delta |
|---|---|---|---|---|
| cold_start_R (train 미등장 투수) | 44,033 | 0.249447 | 0.249406 | −0.000041 |
| seen_R | 179,464 | 0.249311 | 0.249233 | −0.000078 |
| 프로파일 有 (80.7%) | 180,906 | — | 0.249244 | — |
| 프로파일 無 (19.3%) | 42,591 | — | 0.249369 | — |
| cold-start ∩ 프로파일 有 | 1,442 | — | 0.250512 (AUC 0.493) | 소표본, 개선 실패 |

- **주의**: as-of(≤2023) 프로파일의 2024 valid 행 커버리지는 **80.7%** — exp_002의 2024년 말 기준 99.4%보다 크게 낮다 (2024 신규 투수는 as-of 이력 없음). **test(2025) 시점에는 2024년 말 프로파일로 커버리지 ~99%가 기대**되므로, 이 fold 검증은 본번 효과를 **과소평가**했을 가능성 있음 [구조적 관찰, 실측 불가].
- cold-start의 핵심 기대(이력 없는 투수를 물리 프로파일로 보완)는 fold_2024에서 교집합이 1,442행뿐이라 사실상 검증 불가 — test에서는 "train 미등장이지만 2024 trackman 보유" 투수가 더 많을 수 있음.

## 5. 결론

- **경계선 채택 후보**: fold_2024 valid-R −0.000070 실개선 + test 시점 커버리지 상승 여지. 단 fold_2022 악화·약한 신호라 단독 확신은 부족.
- 제출 후보로 올릴 경우 leak-auditor 감사 필수 (as-of 컷오프·매핑 사전의 2024 정보 경유 여부 포함).
- 본번(test 2025) 추론 시 프로파일은 **컷오프 2024로 재산출** (이 코드의 `build_asof_profile(tm, pmap, 2024)` = exp_002 산출물과 동일 스코프; 컬럼명은 s2024_→recent_ 대응).

## 6. 다음 제안

1. 프로파일을 **파생 상호작용**으로 압축 (예: 구속 p90−평균, fastball_share×카운트 상태) — 원시 65열은 gain 분산으로 흡수력 약함.
2. exp_004(F리그 제외) 결과와 결합 시 재평가 — F 행 제거가 프로파일 신호 대 잡음비를 바꿀 수 있음.
3. cold-start 전용 경로: 프로파일 有/無 분기 모델 또는 has_profile 명시 플래그(결측 플래그 규약 준수 형태) 실험.

## 산출물·경로

- 로컬: `experiments/exp_003/{config.yaml, metrics.json, report.md, train.log, src/train.py}`
- 서버: `a5000:~/SungjinPark/experiments/exp_003/` — 모델 가중치 `model_fold_2022/2023/2024.txt`, `model_exp001ref_fold_2024.txt` (서버 보관, 로컬 미회수)
- 재현: seed 42, 전 파라미터·데이터 SHA256은 metrics.json에 자동 기록 (train.csv d20811…, trackman_history 포함)
