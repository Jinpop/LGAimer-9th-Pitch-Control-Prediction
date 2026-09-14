# exp_012 — 투수 커맨드(제구 일관성) 프로파일 · **기각**

> 작성: 2026-08-05 · 총 27.3분 · 추론 5.23초 · 제출 슬롯 0 · 주 지표 **fold_2024 valid_all**

## 1. trackman 실제 30컬럼 — 투구 위치 없음 [실측]

```
trackman_id, season, game_date, game_month, game_dayofweek, trackman_game_id,
pitch_no, inning, top_bottom, balls_before, strikes_before, outs_before,
pitch_of_pa, pitcher_trackman_id, batter_trackman_id, pitcher_hand, batter_hand,
pitcher_team, batter_team, tagged_pitch_type, auto_pitch_type, pitch_type_group,
rel_speed, spin_rate, induced_vert_break, horz_break, extension, rel_height,
rel_side, zone_speed
```

`plate_x`/`plate_z`/zone 계열 **없음**. config가 요구한 "투구 위치 std·분위폭", "존 중앙 투입률·존 이탈률"은 직접 산출 불가 → 두 가지로 대체:

**대체① 카운트 진행에서 볼/스트라이크 역산** (핵심 아이디어). 같은 타석의 다음 투구 카운트가 현재 투구의 판정을 알려준다: `balls+1 → 볼`(=존 이탈, 라벨 실패 ②의 직접 관측) / `strikes+1 → 스트라이크` / `2S에서 카운트 동결 → 파울`. 게이트는 (경기·투수·타자·이닝·초말) 동일 + `pitch_of_pa` 정확히 +1.
→ **1,732,645행 중 1,281,750행(74.0%) 판정 복원, 이상 전이 폐기 85행(0.007%)**. 타석 종결구는 후속 행이 없어 제외(파생 볼율 0.469는 실제보다 높으나 투수 간 비교엔 일관).
→ 주최측 `asof_pitcher_ball_rate`는 전체 평균 하나뿐인데, 이건 카운트 상태·구종·좌우타자·경기별로 쪼갤 수 있다는 점이 exp_003 대비 새 축.

**대체② 구종 내 물리량 산포 = 반복성.** exp_003의 std는 구종을 뭉쳐서 사실상 "구종 믹스"를 인코딩했다. 여기선 모든 std를 (투수 × pitch_type_group) 셀 안에서 계산 후 투구수 가중.

**외부 검증**: 파생 `cmd_ball_rate` vs 주최측 `asof_pitcher_ball_rate` **corr = 0.9023** (500구+ 463명). 매핑·역산이 옳다는 강한 증거인 동시에 **기각 사유의 핵심**.

## 2. 산출 스키마 — `out/pitcher_command_profile_2024.csv` (685투수 × 60열)

| 계열 | 열 수 | 내용 |
|---|---|---|
| `cmd_*` | 31 | 파생 볼율: 전체 / 유리·불리·동률 카운트 / 3B·2S / 초구 스트라이크율 / 좌우타자별 / 구종별+격차 / **경기별 볼율의 std·IQR·p10-p90 폭·최악일(p90)** / 타석 단위(투구수·3볼 도달률·2S 도달률) / 최근시즌·추세 |
| `cons_*` | 29 | 구종 내 가중 std: IVB·HB·구속·회전·릴리스높이/옆·익스텐션 + 산포 합성 / 구종별 std·IQR / **압박(불리) std − 유리 std** / 좌우타자별 릴리스 이동·std / 최근 추세 |

as-of: valid 연도 Y fold는 `season <= Y-1`만 집계(런타임 assert). 정적 테이블은 cutoff 2024(2025 부재 assert) → test 재현 가능. 결측은 NaN 유지 + `has_profile2` 플래그, median imputer는 train 행에만 fit.

## 3. 결과 — 전 변형이 fold_2024에서 악화

A_base가 exp_008 R1 앵커를 **완전 재현**(fold_2024 valid_R 0.248487, abs_diff **0.0**).

| run | 추가열 | fold_2022 | fold_2023 | **fold_2024 valid_all** | **Δ vs base** | valid_R | valid_F | 신규 중요도 |
|---|---|---|---|---|---|---|---|---|
| **A_base** | 0 | 0.243987 | 0.251651 | **0.248769** | — | 0.248487 | 0.250870 | — |
| B_cons | 30 | 0.244163 | 0.251443 | 0.248918 | **+0.000149** | 0.248546 | 0.251688 | 14.5% |
| C_cmd | 32 | 0.244170 | 0.251512 | 0.249051 | **+0.000282** | 0.248654 | 0.252008 | 20.1% |
| D_all | 61 | 0.244217 | 0.251573 | 0.249045 | **+0.000276** | 0.248661 | 0.251899 | **24.5%** |
| E_compact | 13 | 0.244039 | 0.251569 | 0.248835 | **+0.000066** | 0.248554 | 0.250925 | 11.4% |
| G_all_mf6(진단) | 61 | — | 0.251018 | 0.249429 | **+0.000660** | 0.248857 | 0.253689 | 30.1% |

fold_2024 valid_all: base_rate 0.486105 · A_base **BSS 0.004156**(환산 415.6). 목표선 LB 1,061 = valid_all Brier 0.247155, base와 격차 0.001614 — 이번 실험은 반대 방향.

**신규 지표 중요도는 exp_003의 1.5% → 24.5%로 16배.** 모델은 강하게 쓰는데 점수는 나빠진다.

⚠️ fold_2023은 base_rate 0.49996이고 A_base BSS가 **음수(−0.0066, 환산 0점)**인 병리적 fold다. 여기서만 전 변형이 개선되나 fold_2022·2024와 부호가 반대 → **신호가 아니라 불안정성**.

### 기각 근거 세 갈래
1. **중복.** RF 중요도 상위가 `asof_pitcher_success_rate`(0.105)·`asof_pitcher_reverse_rate`(0.074)·`asof_pitcher_ball_rate`(0.012) — **라벨에서 직접 뽑은** 투수 집계. 우리 커맨드 지표는 같은 양의 trackman 쪽 **잡음 많은 대리 변수**(corr 0.90)라, 분기를 정확한 측정에서 대리 변수로 옮기기만 한다.
2. **희석 아님(G 진단).** `max_features`를 base와 같은 6으로 고정하니 fold_2024가 **가장 크게 악화**(+0.000660). 신규 열 의존을 강제로 높이면 더 나빠진다 = 열 개수 희석이 아니라 **신호가 이전되지 않는다**. 손해가 열 수에 비례(13열 +0.000066 → 61열 +0.000276)하는 것도 같은 결론.
3. **커버된 행이 나빠진다.** D_all fold_2024에서 프로파일 **보유** 행 Brier 0.248567→0.248684, AUC 0.5413→0.5398. **미보유** 행은 사실상 불변. 손해가 프로파일이 붙은 바로 그 행에서 발생.

## 4. F리그 커버리지 — 우려 기각

| 기준 | R | F |
|---|---|---|
| 정적 cutoff-2024 (=본번 test 시점) | **99.85%** | **95.48%** |
| fold_2024 as-of(cutoff 2023) | 80.94% | 78.91% |
| 시즌별 F 최저 | — | 92.83% (2019) |

**trackman은 1군 편향이 아니다** (exp_002 매핑이 F 경기 146건 포함). `has_profile2`가 R/F 판별자로 작동할 위험 없고, test의 F 구간이 통째로 median-imputed 될 위험도 없다. F 투수는 실제로 제구가 나쁘다(중앙값 볼율 F 0.482 vs R 0.462, 릴리스 산포 0.115 vs 0.099) — 실재 신호지만 점수로 환산되지 않았다.
단 검증이 본번을 과소평가(검증 79~81% vs 본번 95~100%). 그럼에도 **커버된 행에서 악화**가 확인되어 커버리지를 올려도 부호가 뒤집힐 것으로 보지 않는다.

## 5. 결론 / 다음 제안

- **기각.** 어떤 변형도 제출 후보 아님.
- 부정 결과의 값어치: "투수 개인 커맨드 성향"이라는 레버는 **이미 주최측 asof 컬럼이 다 먹었다.** 투수 단위 정적 집계 계열은 두 번 실패(exp_003 stuff 1.5%, exp_012 command 24.5%) → **이 방향은 닫을 것을 권고.**
- 남은 레버는 **투수 개인 축 밖**:
  1. **보정 — 가장 저렴한 다음 수.** fold_2024 pred_mean 0.5012 vs base_rate 0.4861, **+1.5pp 과대.** exp_005 C안을 RF base에 valid_all 기준으로 재적용.
  2. **타자 축 미개척.** 중요도 5위가 `asof_batter_success_rate`(0.058)인데 타자를 전혀 안 건드렸다. exp_002 `batter_id_map.csv`(714명, 행 99.6%) 미사용 자산.
  3. **투수×타자 상호작용 곱셈항** — RF가 스스로 못 만드는 항.
  4. **`game_type`이 단일 최대 중요도(0.128)** — test에 F가 섞였다면 이 분기 취급이 점수를 크게 좌우. 별도 진단 가치.

## 6. 특이사항

- 중간에 주 지표가 valid_R → valid_all로 변경됐으나 코드가 매 fold·매 run에 세 눈금을 모두 산출해 **재학습 없이 판정 기준만 전환**. 두 눈금 모두 결론 동일(전 변형 악화).
- exp_009와 병렬 실행, 스레드 6 준수.
- 리크 점검: as-of cutoff 런타임 assert, trackman 2025 부재 assert, fold_2024 valid 행수 assert(253,507/223,497 — exp_001/007/008과 동일 행), 인코더·imputer·RF 전부 train 행에만 fit. 현재 행 trackman 물리량 미사용(조인 키는 pitcher_id뿐). test 미참조.
- **미검증**: leak-auditor 독립 감사 미실시(기각 결과라 제출 경로 아님). 채택 시 필수.

## 7. 산출물

- `metrics.json` — 6 run × fold 전체 지표·중요도 전량
- `coverage_by_league.json` — R/F 커버리지·imputation 격차
- `out/pitcher_command_profile_2024.csv` — 정적 커맨드 프로파일 685×60 (**본번 재현 가능 자산, 타자 실험 등에서 재사용 가능**)
- `src/{profile2.py, train.py, coverage_check.py}`
- 서버: `a5000:~/SungjinPark/experiments/exp_012/` (`train.log`, `coverage.log`, `pred_*_fold_2024.npy` 4종)
