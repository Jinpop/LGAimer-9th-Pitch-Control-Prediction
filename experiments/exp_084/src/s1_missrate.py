#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp_084 stage1 — 개인 셀 미적중률의 연도별 추세와 2025 외삽.  학습 0건.

표 생성 규약은 출하 `mate/model_q/meta.json` 의 offset.key_cols=[pitcher_id,batter_hand]
lookup_d 생성 절차(= build_BR/src/solve_BR.py build_tables)를 그대로 따른다:
  마스크 exp_004: season<=maxseason & ~(season<=2022 & game_type=='F') & control_success.notna()
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "6"
import json, time                                                  # noqa: E402
import numpy as np                                                 # noqa: E402
import pandas as pd                                                # noqa: E402
from scipy.stats import binom                                      # noqa: E402

H = "/home/<user>/SungjinPark"
OUT = f"{H}/exp084/out"
T0 = time.time()
R = {"exp_id": "exp_084", "stage": 1, "date": time.strftime("%Y-%m-%d %H:%M:%S")}

tr = pd.read_csv(f"{H}/data/train.csv", encoding="utf-8-sig",
                 usecols=["row_id", "season", "game_type", "control_success",
                          "pitcher_id", "batter_hand", "pitcher_hand"])
print("train rows", len(tr), "seasons", sorted(tr.season.unique()), flush=True)
R["train_seasons"] = {int(s): int(c) for s, c in tr.season.value_counts().sort_index().items()}


def table_keys(maxseason):
    """출하 lookup_d 의 키 집합 = build_tables(maxseason) 의 (pitcher_id|batter_hand)."""
    m = (tr.season <= maxseason) & ~((tr.season <= 2022) & (tr.game_type == "F"))
    d = tr[m & tr.control_success.notna()]
    g = d.groupby(["pitcher_id", "batter_hand"]).size().reset_index()
    return set(f"{int(a)}|{int(b)}" for a, b in zip(g.pitcher_id, g.batter_hand)), int(len(d))


def rowkey(df):
    ka = pd.to_numeric(df.pitcher_id, errors="coerce")
    kb = pd.to_numeric(df.batter_hand, errors="coerce")
    ok = (ka.notna() & kb.notna()).to_numpy()
    out = np.array([""] * len(df), dtype=object)
    out[ok] = [f"{int(x)}|{int(y)}" for x, y in zip(ka[ok].astype(int), kb[ok].astype(int))]
    return out


def cellkey(df):
    ka = pd.to_numeric(df.pitcher_hand, errors="coerce")
    kb = pd.to_numeric(df.batter_hand, errors="coerce")
    ok = (ka.notna() & kb.notna()).to_numpy()
    out = np.array([""] * len(df), dtype=object)
    out[ok] = [f"{int(x)}|{int(y)}" for x, y in zip(ka[ok].astype(int), kb[ok].astype(int))]
    return out


CELLS = ["1|1", "1|2", "2|1", "2|2"]
SEASONS = sorted(int(s) for s in tr.season.unique())
rows = {}
for s in SEASONS[1:]:                      # 첫 시즌은 앞선 표가 없다
    keys, n_fit = table_keys(s - 1)
    if not keys:
        continue
    fr = tr[tr.season == s]
    rk = rowkey(fr)
    ck = cellkey(fr)
    miss = np.array([k not in keys for k in rk])
    ent = {"season": s, "table_max_season": s - 1,
           "n_table_keys": len(keys), "n_fit_rows_table": n_fit,
           "n_rows": int(len(fr)), "n_miss": int(miss.sum()),
           "miss_frac": float(miss.mean()),
           "n_seasons_in_table": int(((np.array(SEASONS) <= s - 1)).sum()),
           "per_cell": {}}
    for c in CELLS:
        m = (ck == c)
        ent["per_cell"][c] = {
            "n_rows_in_cell": int(m.sum()),
            "cell_share_of_frame": float(m.mean()),
            "n_fire": int((m & miss).sum()),
            "fire_frac_of_frame": float((m & miss).mean()),
            "miss_frac_within_cell": float(miss[m].mean()) if m.sum() else None}
    rows[s] = ent
    pc = ent["per_cell"]["2|2"]
    print(f"[{s}]  표≤{s-1} ({ent['n_seasons_in_table']}시즌·{len(keys)}키)  행 {len(fr)}  "
          f"미적중 {ent['n_miss']} = {100*ent['miss_frac']:.4f}%   "
          f"2|2 발동 {pc['n_fire']} = {100*pc['fire_frac_of_frame']:.4f}% of frame "
          f"(셀 내 {100*pc['miss_frac_within_cell']:.3f}%)", flush=True)
R["per_season"] = rows

# ── 대조: ≤2023 표 → 2024 행에서 21.4988%(54,501/253,507) 가 재현되는가 ──
g = rows.get(2024, {})
R["control_le2023_to_2024"] = {
    "recorded_n_rows": 253507, "recorded_n_miss": 54501,
    "recorded_miss_frac": 0.21498814628392904,
    "measured_n_rows": g.get("n_rows"), "measured_n_miss": g.get("n_miss"),
    "measured_miss_frac": g.get("miss_frac"),
    "n_rows_match": bool(g.get("n_rows") == 253507),
    "n_miss_match": bool(g.get("n_miss") == 54501),
    "recorded_n_fire_22_BS4": 17114,
    "measured_n_fire_22": g.get("per_cell", {}).get("2|2", {}).get("n_fire"),
    "n_fire_22_match": bool(g.get("per_cell", {}).get("2|2", {}).get("n_fire") == 17114),
    "REPRODUCED": bool(g.get("n_rows") == 253507 and g.get("n_miss") == 54501)}
print("\n=== 대조 ===", json.dumps(R["control_le2023_to_2024"], ensure_ascii=False), flush=True)

# ── 추세 · 2025 외삽 ──────────────────────────────────────────────────
def fit_extrap(xs, ys, xnew, label):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    n = len(xs)
    A = np.vstack([np.ones(n), xs]).T
    coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
    pred = A @ coef
    res = ys - pred
    dof = n - 2
    s2 = float(res @ res / dof) if dof > 0 else 0.0
    xbar = xs.mean(); sxx = float(((xs - xbar) ** 2).sum())
    yhat = float(coef[0] + coef[1] * xnew)
    # 예측구간(새 관측 1개) — t 는 dof 에 맞춰 보수적으로 df=n-2
    from scipy.stats import t as tdist
    se_pred = float(np.sqrt(s2 * (1.0 + 1.0 / n + (xnew - xbar) ** 2 / sxx))) if dof > 0 else 0.0
    se_mean = float(np.sqrt(s2 * (1.0 / n + (xnew - xbar) ** 2 / sxx))) if dof > 0 else 0.0
    tc = float(tdist.ppf(0.975, dof)) if dof > 0 else float("nan")
    ss_tot = float(((ys - ys.mean()) ** 2).sum())
    return {"label": label, "n_points": n, "x": xs.tolist(), "y": ys.tolist(),
            "intercept": float(coef[0]), "slope_per_year": float(coef[1]),
            "r2": float(1.0 - (res @ res) / ss_tot) if ss_tot > 0 else None,
            "resid_sd": float(np.sqrt(s2)),
            f"pred_{xnew}": yhat,
            "se_mean": se_mean, "se_pred": se_pred, "t_crit_95": tc,
            "ci95_mean": [yhat - tc * se_mean, yhat + tc * se_mean],
            "pi95_new": [yhat - tc * se_pred, yhat + tc * se_pred]}


TREND = {}
yrs = [s for s in rows if s >= 2020]
TREND["miss_frac_all_2020_2024"] = fit_extrap(yrs, [rows[s]["miss_frac"] for s in yrs],
                                              2025, "전체 미적중률")
TREND["fire22_frac_2020_2024"] = fit_extrap(
    yrs, [rows[s]["per_cell"]["2|2"]["fire_frac_of_frame"] for s in yrs], 2025,
    "2|2 발동 비중 (프레임 대비)")
TREND["cellshare22_2020_2024"] = fit_extrap(
    yrs, [rows[s]["per_cell"]["2|2"]["cell_share_of_frame"] for s in yrs], 2025,
    "2|2 셀 비중")
# 전 시즌(가용 전부) 대조 적합
allyrs = sorted(rows)
if len(allyrs) > len(yrs):
    TREND["miss_frac_all_seasons"] = fit_extrap(
        allyrs, [rows[s]["miss_frac"] for s in allyrs], 2025, "전체 미적중률 (전 시즌)")
    TREND["fire22_frac_all_seasons"] = fit_extrap(
        allyrs, [rows[s]["per_cell"]["2|2"]["fire_frac_of_frame"] for s in allyrs], 2025,
        "2|2 발동 비중 (전 시즌)")
R["trend"] = TREND
for k, v in TREND.items():
    print(f"[추세 {k}] 기울기 {v['slope_per_year']:+.6f}/년 · R² {v['r2']} · "
          f"2025 예측 {v['pred_2025']:.6f} · 95% 예측구간 "
          f"[{v['pi95_new'][0]:.6f}, {v['pi95_new'][1]:.6f}]", flush=True)

# ── real5: 21.5% 하에서 4/5 이상 미적중일 확률 ─────────────────────────
p23 = float(rows[2024]["miss_frac"])
p25 = float(TREND["miss_frac_all_2020_2024"]["pred_2025"])
R["real5_binomial"] = {
    "observed": "실제 대회 표본 5행 중 4행 미적중 (S38)",
    "n": 5, "k_observed": 4,
    "under_p_le2023_frame": {
        "p": p23, "P(X>=4)": float(binom.sf(3, 5, p23)),
        "P(X==4)": float(binom.pmf(4, 5, p23)),
        "P(X==5)": float(binom.pmf(5, 5, p23)),
        "P(X>=4)_pct": 100 * float(binom.sf(3, 5, p23))},
    "under_p_2025_extrapolated": {
        "p": p25, "P(X>=4)": float(binom.sf(3, 5, p25))},
    "MLE_p_from_real5": 0.8,
    "clopper_pearson_95_from_4of5": [
        float(__import__("scipy.stats", fromlist=["beta"]).beta.ppf(0.025, 4, 2)),
        float(__import__("scipy.stats", fromlist=["beta"]).beta.ppf(0.975, 5, 1))],
    "판독": ("P(X>=4) 가 5% 를 넘으면 real5 의 4/5 는 추세와 양립한다 — "
             "즉 추세를 버릴 근거가 아니다.")}
print("\n[real5]", json.dumps(R["real5_binomial"], ensure_ascii=False, indent=1), flush=True)

R["wall_sec"] = time.time() - T0
os.makedirs(OUT, exist_ok=True)
json.dump(R, open(f"{OUT}/s1_missrate.json", "w"), indent=1, ensure_ascii=False)
print(f"\n기록 {OUT}/s1_missrate.json · {R['wall_sec']:.1f}s\nS1_OK")
