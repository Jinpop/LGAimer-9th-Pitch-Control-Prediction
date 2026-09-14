# exp_001 — LightGBM 기준선 (제공 48컬럼, 피처 엔지니어링 없음)

> 2026-08-05 · base: exp_000_eda · 코드 `src/train.py` · 서버 실행 로그 `train.log` · 수치 전체 `metrics.json`

## 가설
제공 48컬럼 그대로 LightGBM만으로 수료컷(549.51 = RF 베이스라인)을 초과하고, fold_2024 valid-R Brier 기준선을 확립한다.

## 셋업
- rolling-origin 3 fold (train≤Y → valid Y+1), 주 판단 fold_2024, 주 지표 valid-R Brier
- 범주형 9개(LightGBM native category), asof 결측 NaN 유지(sentinel 없음), seed 42
- lr 0.05 / leaves 127 / min_leaf 200 / ff·bf 0.8 / ES 200 — 전 파라미터 config.yaml·metrics.json 기록
- A5000 CPU 24스레드, venv(3.11.15 / np 1.26.4 / pd 2.0.3 / lgbm 4.7.0 — 평가 서버 호환 스택). 총 1.07분
- 코드·데이터 SHA256은 metrics.json `sha256`에 자동 기록

## 결과

| fold | valid | valid-R Brier | valid-R BSS | valid-R AUC | 전체 Brier | 전체 BSS | pred_mean − base (R) | best_iter |
|---|---|---|---|---|---|---|---|---|
| fold_2022 | 2022 | 0.249461 | +0.0021 | 0.532 | 0.244479 | +0.0188 | +0.012 | 48 |
| fold_2023 | 2023 | 0.250527 | **−0.0021** | 0.526 | 0.251361 | −0.0054 | +0.030 | **5** |
| **fold_2024** | 2024 | **0.249337** | **+0.0022** | **0.534** | 0.249450 | +0.0014 | **+0.017** | 45 |

**주 지표: fold_2024 valid-R Brier = 0.249337 (BSS +0.0022, AUC 0.534, n=223,497)**

Cold-start (train 미등장 투수, fold_2024: 81명/50,348행):

| 구간 | Brier (R) | n |
|---|---|---|
| cold-start R | 0.249447 | 44,033 |
| seen R | 0.249311 | 179,464 |
| 격차 | **+0.000136** (사실상 무격차) | |

## 관찰·해석
1. **신호 극약 확인**: valid AUC 0.53대, BSS +0.002 수준 — 전략 §0 "저신호" 그대로이나 MLB 유사문제(0.60+)보다도 낮음. 제공 asof 컬럼만으로는 순위 신호가 미약.
2. **보정 갭이 최대 즉효 지점**: pred_mean이 실제 base rate보다 R 기준 +1.7pp 높음 (base rate 하강 추세를 모델이 미반영). 산술적으로 (0.017)² ≈ 0.0003 Brier ≈ **BSS +0.001** — 현 BSS의 절반 규모. 보정(상수 오프셋/isotonic) 실험 우선순위 상향 근거.
3. **극조기 early stopping** (5~48 iter): 연도 분포 이동 아래서 lr 0.05·leaves 127은 몇 스텝 만에 valid 악화. 더 강한 정규화/낮은 lr 실험 필요.
4. **fold_2023 BSS 음수**: 2023은 F리그 라벨 단절(−24pp) 연도 — 2019~2022 F 데이터(성공률 0.70대)가 학습을 왜곡. F 제외/가중 실험(exp_002+) 근거.
5. **season이 gain 3위**: valid 연도는 미학습 값이라 외삽 — 기준선 위험 요인. pitcher_id/batter_id 원핫성 암기가 gain 1·2위인 것도 이동 환경에서 취약. (top30은 metrics.json)
6. **cold-start 무격차**: fallback 부담이 현 시점 작음 — 단, 모델이 투수 개인 신호를 거의 못 쓰는 상태의 방증이기도 함. 개인화 피처가 강해지면 재측정 필요.

## 결론 — **부분 채택**
- fold_2024 기준선 확립: **valid-R Brier 0.249337 / BSS +0.0022** (이후 실험 비교 기준).
- 수료컷 초과 여부는 **판정 보류**: RF 베이스라인 LB 549.51 ≈ BSS 0.0055 (test 2025, r 비공개)로 로컬 BSS +0.0022와 스케일 비교 시 **미달 가능성**. 단 학습 범위·연도가 달라 직접 비교 불가 — LB 제출 캘리브레이션으로 확정해야 함.

## 다음 제안 (우선순위순)
1. base rate 보정 실험 (로짓 상수 오프셋, fold_2024로 적합 검증) — 즉효 +0.001 BSS 예상
2. F리그 처리 (제외/가중) + season 컬럼 제거·변환 — fold_2023 왜곡·외삽 제거
3. 정규화 강화 스윕 (lr 0.01~0.02, leaves 31~63, depth 제한)
4. 동일 fold에서 베이스라인 RF 스펙 재학습 비교 (수료컷 로컬 재현) 또는 첫 LB 제출로 CV↔LB 캘리브레이션
5. exp_002 trackman 정적 프로파일 피처 (진행 중 캠페인)

## 산출물·특이사항
- 모델: 서버 `~/SungjinPark/experiments/exp_001/model_fold_{2022,2023,2024}.txt` (로컬 미회수 — 규약)
- 특이: 발사 직후 stdout에 정체불명 `nohup: '../env/bin/python' 실행 실패` 1줄 (본 실험 명령과 무관한 경로). train.log 무결·metrics.json 버전 검증으로 실행 정상 확인. 서버 공용 환경 잔재로 추정, 재발 시 조사.
