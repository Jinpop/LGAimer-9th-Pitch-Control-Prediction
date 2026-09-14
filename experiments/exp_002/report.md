# exp_002 — trackman↔train 매핑 사전 + 투수 정적 프로파일 (데이터 자산 구축)

> 2026-08-05 · A5000 실행 (GPU 미사용, 총 ~50초) · 코드 `src/01~03`, 산출물 `out/`
> 가설: "경기 시그니처 매핑을 전파하면 train 투수 90%+에 trackman 프로파일 부착 가능" → **채택** (투구 행 기준 99.4%)

## 1. 방법 (exp_000_eda §4 로직의 확장)

1. **train 경기 블록 복원**: (season, month, dow, game_type, 팀페어) 변화점 + **블록 내 inning 역행 분할**(더블헤더 병합 방지) → 4,868블록 (EDA의 4,821 대비 +47 = 분할 규칙 효과)
2. **풀-시퀀스 지문 매칭**: 경기 전체의 (이닝, 초말, B/S/O, 투수손) 배열 md5 해시.
   Stage A = (season, month, dow, 투구수, 지문) 유일-유일 2,671건 / Stage B = (season, 지문)만으로 5건 추가
3. **행 단위 검증 게이트** (실패 시 매칭 폐기 — 오매핑 > 무매핑):
   - 경기 내 train↔tm 투수·타자 대응 1:1 + 타자손 시퀀스 일치 → **116건 폐기**
   - 전역 ID 1:1 충돌(한 train ID가 경기마다 다른 tm ID) → 관련 경기 **144건 폐기**
   - 투수 손 최빈값 불일치 쌍 2건, 타자 1건 폐기
4. 최종 수용 2,418경기에서 ID 매핑 사전 집계 (전역 1:1 bijection assert 통과)
5. 프로파일: 매핑 투수의 **tm 전체 이력**(매칭 실패 경기 포함 — 투수 신원이 확정되었으므로 사용 가능)으로 career_*(2019–24)·s2024_*(2024만) 집계. 결측 NaN 유지.

결정론적 (해시·정렬만, 난수는 검증 표본 추출 seed=42뿐).

## 2. 커버리지 (config primary_metric)

| 기준 | 값 |
|---|---|
| 경기 (train 블록) | **2,418 / 4,868 = 49.7%** (R 2,272 / F 146) |
| 투구 행 (매칭 경기 내) | 726,816 / 1,475,092 = **49.3%** |
| 투수 (ID 매핑) | **685 / 792 = 86.5%** |
| **투구 행 기준 프로파일 보유 투수** | **99.37%** (R전용 99.85%, 2024시즌 99.36%) |
| 타자 (ID 매핑) | 714 / 830 = 86.0% (행 기준 99.6%) |

- 미매핑 투수 107명은 대부분 극소투구 선수 → 행 기준 0.6% 손실뿐. **cold-start fallback은 여전히 필수** (exp_000 §3: test에 신규 투수 존재).
- 경기 매칭이 50%에 그친 원인: tm 쪽 투구 행 누락(트래킹 실패)으로 풀-시퀀스가 어긋나는 경기가 다수 [추측]. 지문 완화 없이 의도적으로 미매칭 유지.

## 3. 검증 (config.validation.checks 3항목 — 전부 통과)

1. **표본 30경기 풀-시퀀스 재확인**: 30/30 완전 일치 (지문과 독립적으로 원자료 배열 직접 비교 + 매핑 사전과 행 단위 대응 일치 확인) — `src/03_validate.py`
2. **투수 손 일치율**: 685/685 = **100%**
3. **행 기준 프로파일 커버리지**: 99.37% (위 표)

### 검증 실패(폐기) 사례 분석
- within-game 116건: 지문(투수손 포함)은 일치했으나 **타자손 시퀀스 불일치**가 주 원인으로 추정 — 스위치히터 타석 기록 방식이 양측에서 다름 [실측: 손 가변 타자 train 32명 vs tm 95명]. 즉 **진짜 매칭을 보수적으로 버린 것**으로 커버리지 손실이지 오매핑 위험 아님. 회수하면 경기 커버리지 ~+2.4%p 가능 (헤드룸).
- 전역 충돌 144건: 동일 train ID가 경기별로 다른 tm ID 대응 — tm ID의 시즌 간 불안정 가능성 [추측]. 통째 폐기로 사전 순도 유지.

## 4. 산출물 스키마

| 파일 | 스키마 |
|---|---|
| `out/pitcher_id_map.csv` (685행) | pitcher_id, pitcher_trackman_id, n_games, hand_train/tm/match, n_pitches_evidence, confidence (A: ≥3경기 564 / B: 2경기 37 / C: 1경기 84) |
| `out/batter_id_map.csv` (714행) | 동일 구조 (A 609 / B 41 / C 64) |
| `out/game_date_map.csv` (2,418행) | train_block, row_id_start/end, season/month/dow/game_type/n_pitches, trackman_game_id, **real_date_ANALYSIS_ONLY, stadium_ANALYSIS_ONLY**(16개 구장), match_stage |
| `out/pitcher_static_profile_2024.csv` (685행 × 68열) | pitcher_id, pitcher_trackman_id, confidence + **career_***: n_pitches/n_games, rel_speed·spin_rate·IVB·HB·rel_height·rel_side mean/std, extension·zone_speed mean, rel_speed_p90, {fastball,breaking,offspeed}별 share·구속·회전·무브먼트 + **s2024_*** 동일 세트(2024시즌만, 2024 미등판 투수는 NaN — 685명 중 393명 보유) + last_season |
| `out/coverage.json`, `out/validation.json` | 수치 원본 |

## 5. 사용 규칙 (필수)

- **`real_date_ANALYSIS_ONLY`·`stadium_ANALYSIS_ONLY`는 train 분석 전용. 본번(제출) 피처 금지** — test에서 재현 불가 (규칙 2-4, 전략 §12). 컬럼명에 금지 표기 내장.
- `pitcher_static_profile_2024.csv`는 **2024년 말까지의 이력만 사용** (tm에 2025 없음 assert 확인) → test(2025) 시점에 재현 가능한 **본번 피처 후보**. 단 매핑 로직 코드 검증 대비 투명 보존, scout으로 허용 범위 토크 모니터링 유지 (exp_000 §4 ⚠️ 동일).
- 프로파일 조인 시 미매핑 투수(행 0.6%) NaN — sentinel 채움 금지, GBDT native NaN 처리.
- confidence C(1경기 근거) 84명은 민감 실험 시 별도 플래그 가능.

## 6. 다음 제안

1. **exp_003**: exp_001 베이스라인 + 프로파일 피처 조인 → fold_2024 Brier 개선 측정 (변경 변수 1개).
2. 헤드룸(선택): 타자손 게이트 완화(투수손+1:1만) 재실행으로 +116경기 회수, 스위치히터 stance 피처 검토.
3. game_date_map으로 train 전용 분석(등판 간격·연투)은 피처가 아닌 **가설 탐색용**으로만.

## 서버 경로
- 코드·산출물: `a5000:~/SungjinPark/experiments/exp_002/` (out/ 전체 로컬 회수 완료, 대용량 중간물 없음 — `_train_blocks.csv` 0.4MB뿐)
