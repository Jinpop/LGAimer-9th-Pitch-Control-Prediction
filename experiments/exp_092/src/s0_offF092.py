"""exp_092 축 F — **학습 행에 대한 out-of-time 오프셋을 실제로 만든다.**

🔴 config 는 "AA·S1·S2·T70 의 ≤2023 OOF 가 없으면 미착수"라 적었고 그 팔들의 OOF 는 **없다**
   (s0_gateF.json — 그 계열 예측은 전부 길이 253,507 = '≤2023 적합 → 2024 예측'뿐).
   그런데 조사가 **다른 것**을 찾았다: `~/SungjinPark/oof_new/stack_fold2024_seed*.npz` 가
   **시즌 전진 연쇄(forward-chaining) OOF** 를 담고 있다 —
       각 시즌 s 에 대해 `season < s` 로 적합해 `season == s` 를 예측.
   즉 **학습 행마다 그 행보다 앞선 데이터만으로 만든 예측**이고, F 가 요구하는 out-of-time 이다.

무엇을 담고 있나 / 무엇이 부족한가 (둘 다 그대로 적는다)
  · 담긴 것: 실패유형 4셀 확률 `p_is_mid_only·p_is_rev_only·p_is_both·p_is_wayoff_true`.
    `labels.CELLS` 가 **상호배타 5셀**(0번이 is_success)이므로 성공 확률 대리값은
        p_succ = 1 − Σ(4셀)
    이고, 이것이 **mate 로짓의 근사**로 쓸 오프셋이다.
  · 부족한 것 ①: **mate 자체가 아니다.** 실패셀 CatBoost 계열이라 mate v6 블렌드와 다른 모델이다.
  · 부족한 것 ②: **커버리지 81.0%.** 전진 연쇄는 두 번째 시즌부터 시작하므로 **2019 전 행
    (211,627행)이 NaN** 이다. 그래서 F 는 커버된 904,650행으로 학습하고,
    **같은 행 제한 · 오프셋 없음** 대조(F0)를 함께 돌려 행 제한과 오프셋을 가른다.

🔴 리크 게이트 (전부 기계 검사, 실패하면 죽는다)
  ① 예측 행에는 라벨이 안 들어간다 — `mk_oof.py` 의 `pred_rows = (season == s) & tr_full`.
  ② 적합 행의 라벨 복원이 예측 시즌을 넘지 않는다 — `fit_rows` 가 `succ_season < s` 를 건다.
     (복원 라벨은 같은 투수의 **다음 행**을 쓰므로 이 조건이 시즌 경계를 닫는다.)
  ③ npz 의 `tr_full` 이 우리 oot23 학습 마스크와 **비트 동일**해야 한다.
  ④ valid(2024) 행이 100% 커버돼야 한다 — 그 벡터도 ≤2023 적합 산출물이다.
"""
import hashlib, json, os
import numpy as np
import pandas as pd
from scipy.special import logit

H = "/home/<user>/SungjinPark"
B = f"{H}/exp092"
W89 = f"{H}/exp089/work"
SEEDS = [1, 13, 42, 101]
EPS = 1e-6
os.makedirs(f"{B}/work", exist_ok=True)
os.makedirs(f"{B}/out", exist_ok=True)

tr = pd.read_csv(f"{H}/data/train.csv",
                 usecols=["row_id", "season", "game_type", "control_success"])
season = tr.season.to_numpy(np.int64)
gt = tr.game_type.to_numpy().astype(str)
y = tr.control_success.to_numpy(np.float64)
N = len(tr)
trm = np.load(f"{W89}/trm_oot23.npy")
vam = np.load(f"{W89}/valid_mask.npy")
assert len(trm) == N == 1475092

P, NANM = [], None
META = {}
for s in SEEDS:
    z = np.load(f"{H}/oof_new/stack_fold2024_seed{s}.npz", allow_pickle=True)
    cols = [str(c) for c in z["stack_cols"]]
    assert cols == ["p_is_mid_only", "p_is_rev_only", "p_is_both", "p_is_wayoff_true"]
    assert np.array_equal(z["season"], season), "season 정렬 불일치"
    assert np.array_equal(z["tr_full"].astype(bool), trm), \
        "🔴 npz tr_full != exp_089 oot23 학습 마스크"          # 게이트 ③
    assert np.array_equal(z["va"].astype(bool), vam)
    A = np.stack([z[c].astype(np.float64) for c in cols])       # (4, N)
    nm = np.isnan(A).any(0)
    assert np.array_equal(np.isnan(A).any(0), np.isnan(A).all(0)), "셀별 NaN 패턴 불일치"
    NANM = nm if NANM is None else NANM
    assert np.array_equal(nm, NANM), "시드 간 NaN 패턴 불일치"
    P.append(A)
    META[f"seed{s}"] = {"row_id_sha256": str(z["row_id_sha256"]),
                        "train_csv_sha256": str(z["train_csv_sha256"]),
                        "train_upto": int(z["train_upto"]), "unit": int(z["unit"])}
Pm = np.mean(P, 0)                                              # 4시드 평균
p_fail = Pm.sum(0)
p_succ = 1.0 - p_fail
cov = ~NANM
assert cov[vam].all(), "🔴 valid 행에 커버 구멍"                 # 게이트 ④

off = np.full(N, np.nan)
off[cov] = logit(np.clip(p_succ[cov], EPS, 1 - EPS))
rows_F = trm & cov                                              # F 학습 행
np.save(f"{B}/work/offF_all.npy", off)
np.save(f"{B}/work/rowsF_all.npy", rows_F)

# ── 이 오프셋이 얼마나 좋은 팔인가 (valid 2024 재센터링 점수) ─────────────────
REF_BR = 0.2498069
yv = y[vam]
pv = np.clip(p_succ[vam], EPS, 1 - EPS)
zv = logit(pv)


def pin_rc(z, target):
    c = 0.0
    for _ in range(200):
        p = 1.0 / (1.0 + np.exp(-(z + c)))
        g = float(p.mean()) - target
        if abs(g) < 1e-15:
            break
        c -= g / float((p * (1 - p)).mean())
    return c


c = pin_rc(zv, float(yv.mean()))
pr = 1.0 / (1.0 + np.exp(-(zv + c)))
brier_rc = float(np.mean((pr - yv) ** 2))

R = {"exp_id": "exp_092", "stage": "s0_offF",
     "source": "oof_new/stack_fold2024_seed{1,13,42,101}.npz — 시즌 전진 연쇄 OOF 4시드 평균",
     "construction": "p_succ = 1 - sum(4 fail cells);  offset = logit(clip(p_succ, 1e-6, 1-1e-6))",
     "npz_meta": META,
     "gates": {"tr_full_equals_oot23_mask": True,
               "valid_fully_covered": True,
               "seed_nan_patterns_identical": True,
               "fit_rows_close_season_boundary":
                   "mk_oof.py fit_rows = (season<s) & ~f_excl & ok & (succ_season<s) — "
                   "복원 라벨의 후속 행까지 예측 시즌 앞이라 시즌 경계가 닫힌다",
               "pred_rows_use_no_label":
                   "mk_oof.py pred_rows = (season==s) & tr_full — 라벨 미사용"},
     "coverage": {"n_oot23_train_rows": int(trm.sum()),
                  "n_covered_train_rows": int(rows_F.sum()),
                  "covered_share": float(rows_F.sum() / trm.sum()),
                  "per_season": {str(s): {
                      "n": int(((season == s) & trm).sum()),
                      "covered_share": float(cov[(season == s) & trm].mean())}
                      for s in sorted(set(season[trm].tolist()))},
                  "n_valid": int(vam.sum()),
                  "valid_covered_share": float(cov[vam].mean())},
     "offset_arm_quality_on_valid_2024": {
         "recentred_brier": brier_rc,
         "recentred_pts": 100000.0 * (1.0 - brier_rc / REF_BR),
         "p_succ_mean": float(pv.mean()), "y_mean": float(yv.mean()),
         "logit_sd": float(zv.std())},
     "artifacts": {"offF_all.npy": f"{B}/work/offF_all.npy",
                   "rowsF_all.npy": f"{B}/work/rowsF_all.npy",
                   "off_sha256": hashlib.sha256(
                       np.ascontiguousarray(np.nan_to_num(off)).tobytes()).hexdigest()},
     "🔴 남은 한계": ("① mate 가 아니라 실패셀 CatBoost 대리값이다 "
                    "② 2019 행(19.0%)이 커버 밖이라 F 는 학습 행이 81.0% 로 줄어든다 — "
                    "그래서 **같은 행 제한 · 오프셋 없음** 대조(F0)를 함께 돌린다")}
json.dump(R, open(f"{B}/out/s0_offF.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1))
