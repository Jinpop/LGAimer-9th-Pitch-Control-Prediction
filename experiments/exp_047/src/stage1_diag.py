"""exp_047 STAGE 1 — diagnostic only.  ZERO model fits.

Pre-registered in experiments/exp_047/config.yaml.  Everything printed or saved
here is a MEASUREMENT on train.csv plus the already-stored exp_030 C5_B3
predictions.  Nothing is trained.

ORDER OF OPERATIONS IS PART OF THE PROTOCOL
-------------------------------------------
Selection (which axes proceed, which EB constant each uses) is computed on the
PRECEDING BLOCKS ONLY — rows of seasons 2022 and 2023, scored against exp_030's
stored block predictions.  The selection dict is written to disk and logged
BEFORE the first fold_2024 number is computed, so the ordering is auditable
rather than asserted.

Integrity gates:
  G1  0 <= S_season <= n_season on every row, per column
  G2a per column: the recovered per-row increment is in {0,1} on every row that
      has a successor  (the "round(n*rate) is an exact integer count" model)
  G2b DECISIVE: the SUCCESS column run through the NEW pre-boundary path is
      checked against ground-truth label cumulation
  G2c per column: pre-boundary decomposition == direct within-season cumulation
      of the recovered increments
  G3  denominators: pitchmix_n vs pitcher_n, the +1 step of all three counters,
      and whether the three pitch-mix rates sum to 1 (collinearity)
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import form_decomp2 as f2                                       # noqa: E402
from form_decomp2 import AXES, axis_col, sha256_file            # noqa: E402

HOME = os.path.expanduser("~/SungjinPark")
DATA = os.path.join(HOME, "data/train.csv")
TEST = os.path.join(HOME, "data/test.csv")
PRED_DIR = os.path.join(HOME, "experiments/exp_030/pred")
OUT_DIR = os.path.join(HOME, "experiments/exp_047")

BRIER_REF = 0.2498069
PTS = 100000.0 / BRIER_REF                       # 400309.19883 points / Brier

TARGET = "control_success"
SUCC_RATE = "asof_pitcher_success_rate"
PITCH_N = "asof_pitcher_n"
BAT_N = "asof_batter_n"

# ---- PRE-REGISTERED constants (fixed before any value was looked at) ------- #
K_GRID = [25, 50, 100, 200, 400, 800, 1600]
K_TRAJ_GRID = [50, 100, 200, 400, 800, 1600, 3200]
K_DEV_GRID = [25, 50, 100, 200, 400, 800]
K_FORM_F1 = 100                       # exp_046 A1's chosen constant, unchanged
SEL_MIN_INCR_R2_RESIDUAL = 0.30       # residual < 30% -> drop
SEL_MIN_BLOCK_GAIN_PTS = 10.0         # block gain < 10 pts -> drop
SEL_MAX_ABS_CORR = 0.70               # |r| > .7 -> keep the larger-gain one
SEL_MAX_AXES = 4                      # > 5 pass -> top 4 by block gain
BLOCK_SEASONS = (2022, 2023)
FOLD_SEASON = 2024

PRED = {
    2022: {42: "pred_C5B3_u2022_seed42_st2.npy", 7: "pred_C5B3_u2022_seed7_st2.npy"},
    2023: {42: "pred_C5B3_u2023_seed42_st1.npy", 7: "pred_C5B3_u2023_seed7_st2.npy"},
    2024: {42: "pred_C5B3_u2024_seed42_st2.npy", 7: "pred_C5B3_u2024_seed7_st2.npy",
           2024: "pred_C5B3_u2024_seed2024_st2.npy"},
}


def r5(x):
    if x is None:
        return None
    x = float(x)
    return None if not np.isfinite(x) else round(x, 5)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def ols_r2(y, X):
    """R^2 of an intercept-bearing least-squares fit, on finite rows only."""
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    m = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if m.sum() < 100:
        return None, int(m.sum())
    yy = y[m]
    A = np.column_stack([np.ones(m.sum()), X[m]])
    beta, *_ = np.linalg.lstsq(A, yy, rcond=None)
    resid = yy - A @ beta
    sst = float(np.sum((yy - yy.mean()) ** 2))
    if sst <= 0:
        return None, int(m.sum())
    return float(1.0 - np.sum(resid ** 2) / sst), int(m.sum())


def offset_gain(pred, y, d, beta):
    """Delta of p' = clip(p + beta*d).  NaN in d means 'no correction here'."""
    dz = np.where(np.isfinite(d), d, 0.0)
    pp = np.clip(pred + float(beta) * dz, 1e-6, 1 - 1e-6)
    db = float(np.mean((pp - y) ** 2) - np.mean((pred - y) ** 2))
    return {"dBrier": db, "dScore": -db * PTS,
            "sd_correction": float(np.std(float(beta) * dz))}


def solve_beta(rows):
    """Pooled closed-form beta for a 1-term offset over a list of (p, y, d)."""
    num = sum(float(np.sum(np.where(np.isfinite(d), d, 0.0) * (y - p)))
              for p, y, d in rows)
    den = sum(float(np.sum(np.where(np.isfinite(d), d, 0.0) ** 2))
              for p, y, d in rows)
    return (num / den) if den > 0 else 0.0


def solve_beta2(rows):
    """Pooled closed-form (b1, b2) for a 2-term offset over (p, y, d1, d2)."""
    A = np.zeros((2, 2))
    b = np.zeros(2)
    for p, y, d1, d2 in rows:
        z1 = np.where(np.isfinite(d1), d1, 0.0)
        z2 = np.where(np.isfinite(d2), d2, 0.0)
        A[0, 0] += float(z1 @ z1)
        A[0, 1] += float(z1 @ z2)
        A[1, 1] += float(z2 @ z2)
        b[0] += float(z1 @ (y - p))
        b[1] += float(z2 @ (y - p))
    A[1, 0] = A[0, 1]
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.zeros(2)


def gain2(pred, y, d1, d2, b):
    z1 = np.where(np.isfinite(d1), d1, 0.0)
    z2 = np.where(np.isfinite(d2), d2, 0.0)
    corr = b[0] * z1 + b[1] * z2
    pp = np.clip(pred + corr, 1e-6, 1 - 1e-6)
    db = float(np.mean((pp - y) ** 2) - np.mean((pred - y) ** 2))
    return {"dBrier": db, "dScore": -db * PTS,
            "sd_correction": float(np.std(corr))}


def true_sd(n_fin, s_fin, n_pri, s_pri, min_n=100):
    """exp_018 deconvolution: observed var minus expected binomial noise."""
    keep = (n_fin >= min_n) & (n_pri >= min_n)
    if keep.sum() < 30:
        return None
    p_s = s_fin[keep] / n_fin[keep]
    p_p = s_pri[keep] / n_pri[keep]
    d = p_s - p_p
    obs = float(np.var(d, ddof=1))
    noise = float(np.mean(p_s * (1 - p_s) / n_fin[keep]
                          + p_p * (1 - p_p) / n_pri[keep]))
    tv = max(obs - noise, 0.0)
    return {"n_cells": int(keep.sum()), "obs_sd_pp": r5(100 * np.sqrt(obs)),
            "noise_sd_pp": r5(100 * np.sqrt(noise)),
            "true_sd_pp": r5(100 * np.sqrt(tv)),
            "reliability": r5(tv / obs) if obs > 0 else None}


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))
    M = {"exp_id": "exp_047", "stage": 1, "fits_run": 0, "brier_ref": BRIER_REF,
         "points_per_brier": r5(PTS),
         "pre_registered": {
             "K_GRID": K_GRID, "K_TRAJ_GRID": K_TRAJ_GRID,
             "K_DEV_GRID": K_DEV_GRID,
             "incr_r2_residual_min": SEL_MIN_INCR_R2_RESIDUAL,
             "block_gain_min_pts": SEL_MIN_BLOCK_GAIN_PTS,
             "max_abs_corr": SEL_MAX_ABS_CORR, "max_axes": SEL_MAX_AXES,
             "selection_rows": "seasons 2022+2023 only (fold_2024 is never "
                               "consulted for any choice)"},
         "sha256": {"train.csv": sha256_file(DATA),
                    "stage1_diag.py": sha256_file(os.path.abspath(__file__)),
                    "form_decomp2.py": sha256_file(
                        os.path.join(here, "form_decomp2.py")),
                    "exp_046/form_decomp.py": sha256_file(f2.EXP046_FORM_DECOMP)},
         "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                      "pandas": pd.__version__}}

    log("loading train.csv ...")
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    log(f"loaded {df.shape}")
    assert df["season"].is_monotonic_increasing
    assert df["row_id"].is_unique
    n_rows = len(df)
    season = df["season"].to_numpy()
    y = df[TARGET].to_numpy(dtype=np.float64)
    base_feats = [c for c in df.columns if c not in ("row_id", TARGET)]
    assert len(base_feats) == 47, len(base_feats)

    pcode_s, puniq = pd.factorize(df["pitcher_id"], sort=True)
    pcode = pcode_s.astype(np.int64)
    bcode_s, buniq = pd.factorize(df["batter_id"], sort=True)
    bcode = bcode_s.astype(np.int64)
    ENT = {"pitcher_id": (pcode, len(puniq)), "batter_id": (bcode, len(buniq))}
    M["entities"] = {"n_pitchers": len(puniq), "n_batters": len(buniq)}

    # ==================================================================== #
    #  G3 — denominators and collinearity (must be settled before anything)
    # ==================================================================== #
    g3 = {}
    pn = df[PITCH_N].to_numpy(dtype=np.float64)
    mixn = df["asof_pitcher_pitchmix_n"].to_numpy(dtype=np.float64)
    bn = df[BAT_N].to_numpy(dtype=np.float64)
    g3["pitchmix_n_equals_pitcher_n_pct"] = r5(100.0 * float((mixn == pn).mean()))
    g3["denominator_note"] = (
        "config asked which denominator the pitch-mix rates use.  Measured: "
        "asof_pitcher_pitchmix_n is elementwise EQUAL to asof_pitcher_n on the "
        "share above, so the question is moot for this file — both give the "
        "same counts.  The code uses asof_pitcher_n for every pitcher axis.")
    for tag, ent, col in (("pitcher", "pitcher_id", PITCH_N),
                          ("pitchmix", "pitcher_id", "asof_pitcher_pitchmix_n"),
                          ("batter", "batter_id", BAT_N)):
        g = df.groupby(ent, sort=False)[col]
        st = (g.shift(-1) - df[col]).to_numpy(dtype=np.float64)
        hs = np.isfinite(st)
        g3[f"step_plus1_pct_{tag}"] = r5(100.0 * float((st[hs] == 1.0).mean()))
        g3[f"step_violations_{tag}"] = int((st[hs] != 1.0).sum())
    mix3 = f2.MIX3
    ssum = df[mix3].to_numpy(dtype=np.float64).sum(axis=1)
    defined = pn > 0
    g3["pitchmix_sum_on_defined_rows"] = {
        "n": int(defined.sum()), "mean": r5(float(ssum[defined].mean())),
        "min": r5(float(ssum[defined].min())), "max": r5(float(ssum[defined].max())),
        "pct_within_2e-6_of_1": r5(100.0 * float(
            (np.abs(ssum[defined] - 1.0) < 2e-6).mean())),
        "verdict": "the three pitch-mix rates are an exact partition up to "
                   "6-decimal rounding -> at most TWO of the three axes may "
                   "enter a model (pre-registered in config)"}
    g3["PASS"] = bool(g3["step_violations_pitcher"] == 0
                      and g3["step_violations_batter"] == 0)
    M["G3"] = g3
    log("G3:", json.dumps(g3, ensure_ascii=False))

    # ==================================================================== #
    #  G2a — recovered per-row increments in {0,1}, per column
    # ==================================================================== #
    g2a = {}
    inc = {}
    for tag, spec in AXES.items():
        ent = spec["entity"]
        cnt, _, _ = f2.career_successes(df[spec["n"]].to_numpy(np.float64),
                                        df[spec["rate"]].to_numpy(np.float64))
        s = pd.Series(cnt)
        nxt = s.groupby(df[ent].to_numpy(), sort=False).shift(-1).to_numpy()
        d = nxt - cnt
        hs = np.isfinite(d)
        inc[tag] = np.where(hs, d, np.nan)
        bad = int((hs & ~np.isin(d, [0.0, 1.0])).sum())
        g2a[tag] = {"n_with_successor": int(hs.sum()), "out_of_01": bad,
                    "PASS": bool(bad == 0)}
    # the same check on the success column, whose truth we independently know
    cnt_s, max_resid, min_half = f2.career_successes(
        pn, df[SUCC_RATE].to_numpy(np.float64))
    s = pd.Series(cnt_s)
    nxt = s.groupby(df["pitcher_id"].to_numpy(), sort=False).shift(-1).to_numpy()
    d_s = nxt - cnt_s
    hs = np.isfinite(d_s)
    g2a["success_anchor"] = {
        "n_with_successor": int(hs.sum()),
        "out_of_01": int((hs & ~np.isin(d_s, [0.0, 1.0])).sum()),
        "agree_with_control_success_pct": r5(
            100.0 * float((d_s[hs] == y[hs]).mean())),
        "PASS": bool((hs & ~np.isin(d_s, [0.0, 1.0])).sum() == 0)}
    M["G2a"] = g2a
    M["rounding"] = {"max_abs_residual_raw_minus_round": max_resid,
                     "min_distance_of_raw_frac_to_0.5": min_half,
                     "bankers_rounding_can_bite": bool(min_half < 1e-9)}
    log("G2a:", json.dumps(g2a, ensure_ascii=False))

    # ==================================================================== #
    #  exp_046 success decomposition (F1's four columns), verbatim path
    # ==================================================================== #
    log("exp_046 success decomposition (unchanged code) ...")
    tabs46 = f2.build_boundary_tables(df, pcode, len(puniq))
    zero46 = (np.zeros(len(puniq)), np.zeros(len(puniq)))
    succ = {c: np.full(n_rows, np.nan) for c in
            ("n_end", "S_end", "n_season", "S_season", "d_form", "d_form_eb",
             "rate_season", "rate_prior")}
    hf = np.zeros(n_rows, dtype=np.int8)
    rate_succ = df[SUCC_RATE].to_numpy(np.float64)
    for s_ in np.sort(np.unique(season)):
        m = season == s_
        t = tabs46.get(int(s_) - 1, zero46)
        o = f2.apply_decomposition(pn[m], rate_succ[m], pcode[m], t[0], t[1],
                                   K_FORM_F1)
        for c in succ:
            succ[c][m] = o[c]
        hf[m] = o["has_form"]
    M["F1_reference"] = {"k_form": K_FORM_F1,
                         "has_form_pct_all_rows": r5(100.0 * hf.mean())}

    # ==================================================================== #
    #  nine axes: pre-boundary tables + row-wise application
    # ==================================================================== #
    log("nine axes: pre-boundary tables + row-wise application ...")
    d_eb = {}          # tag -> {k: array}
    axis_state = {}    # tag -> raw decomposition arrays (for gates / true SD)
    for tag, spec in AXES.items():
        ecode, n_ent = ENT[spec["entity"]]
        ncol = df[spec["n"]].to_numpy(np.float64)
        rcol = df[spec["rate"]].to_numpy(np.float64)
        tabs = f2.build_pre_boundary_tables(season, ecode, ncol, rcol, n_ent)
        zero = (np.zeros(n_ent), np.zeros(n_ent))
        st = {c: np.full(n_rows, np.nan) for c in
              ("n_end", "S_end", "n_season", "S_season")}
        has = np.zeros(n_rows, dtype=np.int8)
        d_eb[tag] = {k: np.full(n_rows, np.nan) for k in K_GRID}
        for s_ in np.sort(np.unique(season)):
            m = season == s_
            t = tabs.get(int(s_) - 1, zero)
            for k in K_GRID:
                o = f2.apply_decomposition(ncol[m], rcol[m], ecode[m],
                                           t[0], t[1], k)
                d_eb[tag][k][m] = o["d_form_eb"]
            for c in st:
                st[c][m] = o[c]
            has[m] = o["has_form"]
        st["has"] = has
        st["tabs"] = tabs
        axis_state[tag] = st
        log(f"  axis {tag}: coverage all-rows "
            f"{100.0*has.mean():.3f}%")

    # ---- G1 ------------------------------------------------------------- #
    g1 = {}
    for tag, st in axis_state.items():
        v = int(((st["S_season"] < -1e-9)
                 | (st["S_season"] > st["n_season"] + 1e-9)
                 | (st["n_season"] < -1e-9)).sum())
        g1[tag] = {"violations": v, "PASS": bool(v == 0)}
    v46 = int(((succ["S_season"] < -1e-9)
               | (succ["S_season"] > succ["n_season"] + 1e-9)
               | (succ["n_season"] < -1e-9)).sum())
    g1["success_exp046_convention"] = {"violations": v46, "PASS": bool(v46 == 0)}
    g1["ALL_PASS"] = bool(all(v["PASS"] for v in g1.values()
                              if isinstance(v, dict)))
    M["G1"] = g1
    log("G1:", json.dumps(g1, ensure_ascii=False))

    # ---- G2b : DECISIVE.  success column through the NEW pre-boundary path  #
    # Expected, from labels alone:
    #   n_season = (within-season row index) + 1{pitcher has prior history}
    #   S_season = (within-season label cumsum, exclusive)
    #              + label of the pitcher's last row at or before the boundary
    log("G2b: pre-boundary machinery vs ground-truth label cumulation ...")
    tabs_pre_s = f2.build_pre_boundary_tables(season, pcode, pn, rate_succ,
                                              len(puniq))
    zero = (np.zeros(len(puniq)), np.zeros(len(puniq)))
    pre_s = {c: np.full(n_rows, np.nan) for c in
             ("n_end", "S_end", "n_season", "S_season")}
    for s_ in np.sort(np.unique(season)):
        m = season == s_
        t = tabs_pre_s.get(int(s_) - 1, zero)
        o = f2.apply_decomposition(pn[m], rate_succ[m], pcode[m], t[0], t[1], 100)
        for c in pre_s:
            pre_s[c][m] = o[c]
    # boundary-row label per (pitcher, boundary season), carried forward
    key = pd.DataFrame({"p": pcode, "s": season})
    last_ix = key.groupby(["p", "s"], sort=False).tail(1).index.to_numpy()
    ylast = np.zeros(len(puniq))
    ylast_tab = {}
    seen_tab = {}
    seen = np.zeros(len(puniq), dtype=bool)
    ls = season[last_ix]
    for s_ in np.sort(np.unique(season)):
        ii = last_ix[ls == s_]
        ylast = ylast.copy()
        seen = seen.copy()
        ylast[pcode[ii]] = y[ii]
        seen[pcode[ii]] = True
        ylast_tab[int(s_)] = ylast
        seen_tab[int(s_)] = seen
    g2b = {"checked_seasons": []}
    mm_n = mm_s = 0
    n_chk = 0
    for s_ in sorted(set(season.tolist())):
        m = season == s_
        sub = pd.DataFrame({"p": pcode[m], "y": y[m]})
        gg = sub.groupby("p", sort=False)
        idx_in = gg.cumcount().to_numpy(np.float64)
        cum_in = gg["y"].cumsum().to_numpy(np.float64) - sub["y"].to_numpy()
        prior = seen_tab.get(int(s_) - 1, np.zeros(len(puniq), bool))[pcode[m]]
        yb = ylast_tab.get(int(s_) - 1, np.zeros(len(puniq)))[pcode[m]]
        exp_n = idx_in + prior.astype(np.float64)
        exp_s = cum_in + np.where(prior, yb, 0.0)
        dn = int((np.abs(pre_s["n_season"][m] - exp_n) > 1e-9).sum())
        ds = int((np.abs(pre_s["S_season"][m] - exp_s) > 1e-9).sum())
        mm_n += dn
        mm_s += ds
        n_chk += int(m.sum())
        g2b["checked_seasons"].append({"season": int(s_), "n_rows": int(m.sum()),
                                       "mismatch_n": dn, "mismatch_S": ds})
    g2b.update({"rule": "pre-boundary decomposition of the SUCCESS column == "
                        "within-season label cumulation + the boundary row's "
                        "own label; the ONE-PITCH offset is explicit here",
                "n_rows_checked": n_chk, "mismatch_n_season": mm_n,
                "mismatch_S_season": mm_s,
                "PASS": bool(mm_n == 0 and mm_s == 0)})
    M["G2b"] = g2b
    log("G2b:", json.dumps({k: v for k, v in g2b.items()
                            if k != "checked_seasons"}, ensure_ascii=False))

    # ---- G2c : per column, decomposition == cumulation of the increments -- #
    log("G2c ...")
    g2c = {}
    for tag, spec in AXES.items():
        ecode, _ = ENT[spec["entity"]]
        st = axis_state[tag]
        bad_n = bad_s = 0
        for s_ in sorted(set(season.tolist())):
            m = season == s_
            iv = inc[tag][m]
            sub = pd.DataFrame({"e": ecode[m], "v": np.where(np.isfinite(iv),
                                                             iv, 0.0)})
            gg = sub.groupby("e", sort=False)
            idx_in = gg.cumcount().to_numpy(np.float64)
            cum_in = gg["v"].cumsum().to_numpy(np.float64) - sub["v"].to_numpy()
            has_prior = st["n_end"][m] > 0
            # boundary row's own increment == S_season at the season's first row
            off_n = st["n_season"][m] - idx_in
            off_s = st["S_season"][m] - cum_in
            # both offsets must be CONSTANT within (entity, season): the
            # boundary row contributes the same 1 pitch / same 0-or-1 event to
            # every row of the season.
            k2 = pd.DataFrame({"e": ecode[m], "a": off_n, "b": off_s})
            gv = k2.groupby("e", sort=False).agg(["min", "max"])
            bad_n += int((gv[("a", "min")] != gv[("a", "max")]).sum())
            bad_s += int((gv[("b", "min")] != gv[("b", "max")]).sum())
            _ = has_prior
        g2c[tag] = {"entity_seasons_with_nonconstant_n_offset": bad_n,
                    "entity_seasons_with_nonconstant_S_offset": bad_s,
                    "PASS": bool(bad_n == 0 and bad_s == 0)}
    g2c["ALL_PASS"] = bool(all(v["PASS"] for v in g2c.values()
                               if isinstance(v, dict)))
    M["G2c"] = g2c
    log("G2c:", json.dumps(g2c, ensure_ascii=False))

    gates_ok = (M["G1"]["ALL_PASS"] and M["G2b"]["PASS"] and M["G2c"]["ALL_PASS"]
                and M["G3"]["PASS"]
                and all(v["PASS"] for v in M["G2a"].values()))
    M["gates_all_pass"] = bool(gates_ok)

    # ==================================================================== #
    #  B axis — trajectory / deviation split of the success axis
    # ==================================================================== #
    log("B axis: trajectory tables ...")
    traj_tabs = f2.build_traj_table(season, pcode, len(puniq), succ["n_end"],
                                    succ["S_end"], succ["n_season"],
                                    succ["S_season"], y)
    traj_eb = {k: np.full(n_rows, np.nan) for k in K_TRAJ_GRID}
    dev_eb = {(kt, kd): np.full(n_rows, np.nan)
              for kt in K_TRAJ_GRID for kd in K_DEV_GRID}
    has_traj = np.zeros(n_rows, dtype=np.int8)
    traj_raw_col = np.full(n_rows, np.nan)
    ntraj_col = np.zeros(n_rows)
    zt = (np.full(len(puniq), np.nan), np.zeros(len(puniq)))
    for s_ in np.sort(np.unique(season)):
        m = season == s_
        t = traj_tabs.get(int(s_) - 1, zt)
        traj_raw_col[m] = t[0][pcode[m]]
        ntraj_col[m] = t[1][pcode[m]]
        for kt in K_TRAJ_GRID:
            for kd in K_DEV_GRID:
                o = f2.apply_traj(pcode[m], t[0], t[1], succ["d_form"][m],
                                  succ["n_season"][m], kt, kd)
                dev_eb[(kt, kd)][m] = o["dev_eb"]
            traj_eb[kt][m] = o["traj_eb"]
        has_traj[m] = o["has_traj"]

    # ==================================================================== #
    #  SELECTION — blocks 2022+2023 only.  fold_2024 is NOT read below this
    #  line until the selection dict has been written and logged.
    # ==================================================================== #
    log("=== SELECTION on blocks %s (fold_2024 untouched) ===" % (BLOCK_SEASONS,))
    block_rows = []      # (season_mask, seed, pred)
    for vy in BLOCK_SEASONS:
        mb = season == vy
        for seed, fn in PRED[vy].items():
            p = np.load(os.path.join(PRED_DIR, fn))
            assert len(p) == int(mb.sum()), (vy, seed, len(p), int(mb.sum()))
            block_rows.append((mb, seed, p))
    blk_mask = np.isin(season, list(BLOCK_SEASONS))

    sel = {"rule": "block-selected; fold_2024 not consulted", "axes": {}}
    for tag in AXES:
        per_k = {}
        for k in K_GRID:
            rows = [(p, y[mb], d_eb[tag][k][mb]) for mb, _s, p in block_rows]
            b = solve_beta(rows)
            gs = [offset_gain(p, yv, dd, b)["dScore"] for p, yv, dd in rows]
            per_k[k] = {"beta": r5(b), "mean_block_dScore": r5(np.mean(gs)),
                        "per_unit": [r5(v) for v in gs]}
        kstar = max(K_GRID, key=lambda k: per_k[k]["mean_block_dScore"])
        gain = per_k[kstar]["mean_block_dScore"]
        r2, n_r2 = ols_r2(d_eb[tag][kstar][blk_mask],
                          df.loc[blk_mask, AXES[tag]["related"]].to_numpy())
        num_base = [c for c in base_feats
                    if pd.api.types.is_numeric_dtype(df[c])]
        r2_all, _ = ols_r2(d_eb[tag][kstar][blk_mask],
                           df.loc[blk_mask, num_base].to_numpy())
        sel["axes"][tag] = {
            "k_star": kstar, "beta_block": per_k[kstar]["beta"],
            "block_gain_pts": gain, "per_k": {str(k): per_k[k] for k in K_GRID},
            "related_cols": AXES[tag]["related"],
            "r2_on_related": r5(r2), "incr_r2_residual": r5(None if r2 is None
                                                            else 1.0 - r2),
            "r2_on_all_numeric_base": r5(r2_all), "n_r2_rows": n_r2,
            "coverage_block_pct": r5(100.0 * float(
                np.isfinite(d_eb[tag][kstar][blk_mask]).mean())),
        }
        log(f"  {tag}: k*={kstar} gain={gain} r2_rel={r5(r2)} "
            f"resid={r5(None if r2 is None else 1-r2)} r2_all={r5(r2_all)}")

    # F1's own column, as the incumbent every new axis must be distinct from
    d_f1 = succ["d_form_eb"]
    rows = [(p, y[mb], d_f1[mb]) for mb, _s, p in block_rows]
    b_f1 = solve_beta(rows)
    sel["F1_incumbent"] = {
        "column": "d_form_eb (k=100, exp_046 A1)", "beta_block": r5(b_f1),
        "block_gain_pts": r5(np.mean([offset_gain(p, yv, dd, b_f1)["dScore"]
                                      for p, yv, dd in rows]))}

    # correlation matrix on block rows (incumbent included)
    corr_names = ["d_form_eb"] + list(AXES.keys())
    corr_mat = {}
    vecs = {"d_form_eb": d_f1[blk_mask]}
    for tag in AXES:
        vecs[tag] = d_eb[tag][sel["axes"][tag]["k_star"]][blk_mask]
    for a in corr_names:
        corr_mat[a] = {}
        for bnm in corr_names:
            m = np.isfinite(vecs[a]) & np.isfinite(vecs[bnm])
            corr_mat[a][bnm] = (r5(np.corrcoef(vecs[a][m], vecs[bnm][m])[0, 1])
                                if m.sum() > 100 else None)
    sel["corr_matrix_block"] = corr_mat

    # ---- pre-registered filters, applied in the registered order ---------- #
    steps = []
    surv = list(AXES.keys())
    dropped = {}
    for tag in list(surv):
        a = sel["axes"][tag]
        if a["incr_r2_residual"] is None or a["incr_r2_residual"] < SEL_MIN_INCR_R2_RESIDUAL:
            dropped[tag] = f"incr_r2_residual {a['incr_r2_residual']} < {SEL_MIN_INCR_R2_RESIDUAL}"
            surv.remove(tag)
    steps.append({"filter": "incremental R2 residual", "survivors": list(surv)})
    for tag in list(surv):
        if sel["axes"][tag]["block_gain_pts"] < SEL_MIN_BLOCK_GAIN_PTS:
            dropped[tag] = f"block_gain {sel['axes'][tag]['block_gain_pts']} < {SEL_MIN_BLOCK_GAIN_PTS}"
            surv.remove(tag)
    steps.append({"filter": "block gain >= 10 pts", "survivors": list(surv)})
    # pitch-mix partition: at most two of the three
    mixs = [t for t in surv if AXES[t]["group"] == "pitchmix"]
    while len(mixs) > 2:
        worst = min(mixs, key=lambda t: sel["axes"][t]["block_gain_pts"])
        dropped[worst] = "pitch-mix partition sums to 1: at most 2 of 3 may enter"
        surv.remove(worst)
        mixs.remove(worst)
    steps.append({"filter": "pitch-mix collinearity (<=2 of 3)",
                  "survivors": list(surv)})
    # against the incumbent d_form_eb first, then pairwise
    for tag in list(surv):
        c = corr_mat["d_form_eb"][tag]
        if c is not None and abs(c) > SEL_MAX_ABS_CORR:
            dropped[tag] = f"|r| {c} with incumbent d_form_eb > {SEL_MAX_ABS_CORR}"
            surv.remove(tag)
    steps.append({"filter": "|r| vs incumbent d_form_eb", "survivors": list(surv)})
    changed = True
    while changed:
        changed = False
        for i, a in enumerate(surv):
            for bnm in surv[i + 1:]:
                c = corr_mat[a][bnm]
                if c is not None and abs(c) > SEL_MAX_ABS_CORR:
                    lo = min((a, bnm), key=lambda t: sel["axes"][t]["block_gain_pts"])
                    dropped[lo] = f"|r| {c} with {a if lo==bnm else bnm} > {SEL_MAX_ABS_CORR}"
                    surv.remove(lo)
                    changed = True
                    break
            if changed:
                break
    steps.append({"filter": "pairwise |r|", "survivors": list(surv)})
    if len(surv) > 5:
        keep = sorted(surv, key=lambda t: -sel["axes"][t]["block_gain_pts"])[:SEL_MAX_AXES]
        for t in surv:
            if t not in keep:
                dropped[t] = "column budget: >5 passed, only top 4 by block gain proceed"
        surv = keep
    steps.append({"filter": "column budget", "survivors": list(surv)})
    sel["filter_steps"] = steps
    sel["dropped"] = dropped
    sel["SELECTED_AXES"] = list(surv)
    sel["SELECTED_COLUMNS"] = [axis_col(t) for t in surv]
    sel["SELECTED_K"] = {t: sel["axes"][t]["k_star"] for t in surv}
    sel["needs_has_form_bat"] = bool(any(AXES[t]["entity"] == "batter_id"
                                         for t in surv))

    # ---- B axis selection, also on blocks --------------------------------- #
    bsel = {}
    best = None
    for kt in K_TRAJ_GRID:
        for kd in K_DEV_GRID:
            rows2 = [(p, y[mb], traj_eb[kt][mb], dev_eb[(kt, kd)][mb])
                     for mb, _s, p in block_rows]
            bb = solve_beta2(rows2)
            gs = [gain2(p, yv, d1, d2, bb)["dScore"] for p, yv, d1, d2 in rows2]
            v = float(np.mean(gs))
            if best is None or v > best[0]:
                best = (v, kt, kd, bb, gs)
    bsel["k_traj"], bsel["k_dev"] = best[1], best[2]
    bsel["beta_traj"], bsel["beta_dev"] = r5(best[3][0]), r5(best[3][1])
    bsel["block_gain_2term_pts"] = r5(best[0])
    bsel["block_gain_1term_incumbent_pts"] = sel["F1_incumbent"]["block_gain_pts"]
    bsel["incremental_over_incumbent_pts"] = r5(
        best[0] - sel["F1_incumbent"]["block_gain_pts"])
    mt = np.isfinite(traj_eb[best[1]][blk_mask]) & np.isfinite(
        dev_eb[(best[1], best[2])][blk_mask])
    bsel["corr_traj_dev_block"] = r5(np.corrcoef(
        traj_eb[best[1]][blk_mask][mt],
        dev_eb[(best[1], best[2])][blk_mask][mt])[0, 1]) if mt.sum() > 100 else None
    bsel["has_traj_pct_block"] = r5(100.0 * float(has_traj[blk_mask].mean()))
    r2t, nr2t = ols_r2(traj_eb[best[1]][blk_mask],
                       df.loc[blk_mask, [SUCC_RATE,
                                         "asof_pitcher_prev1_game_success_rate",
                                         "asof_pitcher_prev3_game_success_rate",
                                         "asof_pitcher_prev5_game_success_rate",
                                         PITCH_N]].to_numpy())
    bsel["traj_r2_on_existing"] = r5(r2t)
    bsel["traj_incr_r2_residual"] = r5(None if r2t is None else 1 - r2t)
    bsel["n_r2_rows"] = nr2t
    bsel["PASS"] = bool(
        bsel["incremental_over_incumbent_pts"] is not None
        and bsel["incremental_over_incumbent_pts"] >= SEL_MIN_BLOCK_GAIN_PTS
        and bsel["corr_traj_dev_block"] is not None
        and abs(bsel["corr_traj_dev_block"]) <= SEL_MAX_ABS_CORR
        and bsel["traj_incr_r2_residual"] is not None
        and bsel["traj_incr_r2_residual"] >= SEL_MIN_INCR_R2_RESIDUAL)
    sel["B_axis"] = bsel
    sel["PROCEED_TO_STAGE2"] = bool(gates_ok and (len(surv) > 0 or bsel["PASS"]))
    M["selection"] = sel
    with open(os.path.join(OUT_DIR, "stage1_selection.json"), "w") as f:
        json.dump(json.loads(json.dumps(sel, default=str)), f, indent=1,
                  ensure_ascii=False)
    log("SELECTION FROZEN ->", json.dumps(
        {"axes": sel["SELECTED_AXES"], "k": sel["SELECTED_K"],
         "B_axis_PASS": bsel["PASS"], "dropped": dropped,
         "proceed": sel["PROCEED_TO_STAGE2"]}, ensure_ascii=False))

    # ==================================================================== #
    #  fold_2024 DESCRIPTIVES — computed only after the selection is frozen
    # ==================================================================== #
    log("fold_2024 descriptives (post-selection) ...")
    m24 = season == FOLD_SEASON
    is_r = (df.loc[m24, "game_type"] == "R").to_numpy()
    desc = {"note": "computed AFTER stage1_selection.json was written; nothing "
                    "here influenced any choice"}
    desc["n_rows"] = int(m24.sum())
    # dilution weight w = n_season / n  (share of the career column that this
    # season contributes) — one per entity
    for label, st, ncol in (("pitcher_pre_boundary", axis_state["rev"], pn),
                            ("batter_pre_boundary", axis_state["bsuc"], bn),
                            ("pitcher_exp046", succ, pn)):
        mm = m24 & (ncol > 0) & np.isfinite(st["n_season"])
        w = st["n_season"][mm] / ncol[mm]
        desc[f"dilution_w_{label}"] = {
            "mean": r5(w.mean()), "median": r5(np.median(w)),
            "q90": r5(np.quantile(w, 0.9)),
            "interpretation": "1-w is the share of this season's form that the "
                              "existing cumulative column dilutes away"}
    per_axis = {}
    for tag, spec in AXES.items():
        st = axis_state[tag]
        k = sel["axes"][tag]["k_star"]
        cov = np.isfinite(d_eb[tag][k][m24])
        # true SD of the axis's season-vs-prior deviation, seasons <= 2023
        tw = season <= 2023
        ecode, _ = ENT[spec["entity"]]
        kk = pd.DataFrame({"e": ecode, "s": season})[tw]
        li = kk.groupby(["e", "s"], sort=False).tail(1).index.to_numpy()
        n_fin = st["n_season"][li] + 1.0
        s_fin = st["S_season"][li] + np.where(np.isfinite(inc[tag][li]),
                                              inc[tag][li], 0.0)
        n_pri = st["n_end"][li]
        s_pri = st["S_end"][li]
        okc = (n_pri > 0) & (n_fin > 0) & np.isfinite(s_fin)
        tsd = true_sd(n_fin[okc], s_fin[okc], n_pri[okc], s_pri[okc], 100)
        per_axis[tag] = {
            "rate_col": spec["rate"], "entity": spec["entity"],
            "k_star": k,
            "coverage_fold2024_pct": r5(100.0 * cov.mean()),
            "coverage_fold2024_R_pct": r5(100.0 * cov[is_r].mean()),
            "coverage_fold2024_F_pct": r5(100.0 * cov[~is_r].mean()),
            "true_sd_min_n100": tsd,
            "sd_of_block_correction": r5(
                abs(sel["axes"][tag]["beta_block"])
                * float(np.nanstd(np.where(np.isfinite(d_eb[tag][k][m24]),
                                           d_eb[tag][k][m24], 0.0)))),
        }
        per_axis[tag]["gain_ceiling_pts_from_true_sd"] = (
            r5(PTS * (tsd["true_sd_pp"] / 100.0) ** 2 * cov.mean())
            if tsd and tsd["true_sd_pp"] else None)
        per_axis[tag]["gain_estimate_from_block_beta_pts"] = r5(
            PTS * per_axis[tag]["sd_of_block_correction"] ** 2)
    desc["per_axis"] = per_axis
    desc["B_axis"] = {
        "has_traj_pct_fold2024": r5(100.0 * float(has_traj[m24].mean())),
        "has_form_pct_fold2024": r5(100.0 * float(hf[m24].mean())),
        "traj_and_form_pct_fold2024": r5(100.0 * float(
            ((has_traj[m24] == 1) & (hf[m24] == 1)).mean())),
    }

    # honest fold_2024 read-out of the block-selected offsets (context only)
    fold = {}
    for tag in AXES:
        k = sel["axes"][tag]["k_star"]
        b = sel["axes"][tag]["beta_block"]
        per = {}
        for seed, fn in PRED[2024].items():
            p = np.load(os.path.join(PRED_DIR, fn))
            per[str(seed)] = r5(offset_gain(p, y[m24], d_eb[tag][k][m24],
                                            b)["dScore"])
        fold[tag] = {"per_seed_dScore": per,
                     "mean_dScore": r5(np.mean(list(per.values())))}
    p42 = np.load(os.path.join(PRED_DIR, PRED[2024][42]))
    fold["_F1_incumbent"] = {"mean_dScore": r5(np.mean([
        offset_gain(np.load(os.path.join(PRED_DIR, fn)), y[m24], d_f1[m24],
                    b_f1)["dScore"] for fn in PRED[2024].values()]))}
    kt, kd = bsel["k_traj"], bsel["k_dev"]
    bb = np.array([bsel["beta_traj"], bsel["beta_dev"]])
    fold["_B_axis_2term"] = {"mean_dScore": r5(np.mean([
        gain2(np.load(os.path.join(PRED_DIR, fn)), y[m24], traj_eb[kt][m24],
              dev_eb[(kt, kd)][m24], bb)["dScore"]
        for fn in PRED[2024].values()]))}
    desc["fold2024_offset_readout_context_only"] = fold
    _ = p42

    # ---- batter interpretation risk (config ⚠️_타자_컬럼_주의) ------------- #
    log("batter interpretation probe ...")
    tw = season <= 2023
    risk = {}
    for tag in ("bsuc", "bmid"):
        k = sel["axes"][tag]["k_star"]
        v = d_eb[tag][k]
        m = tw & np.isfinite(v) & np.isfinite(rate_succ)
        risk[f"{tag}_corr_with_opposing_pitcher_career_success_rate"] = r5(
            np.corrcoef(v[m], rate_succ[m])[0, 1])
        # season level: does the batter's in-season deviation track the quality
        # of the pitchers they happened to face?
        st = axis_state[tag]
        kk = pd.DataFrame({"e": bcode, "s": season})[tw]
        li = kk.groupby(["e", "s"], sort=False).tail(1).index.to_numpy()
        n_fin = st["n_season"][li] + 1.0
        s_fin = st["S_season"][li] + np.where(np.isfinite(inc[tag][li]),
                                              inc[tag][li], 0.0)
        n_pri = st["n_end"][li]
        s_pri = st["S_end"][li]
        okc = (n_pri > 0) & (n_fin > 0)
        d_season = np.where(okc, s_fin / np.where(n_fin > 0, n_fin, 1)
                            - s_pri / np.where(n_pri > 0, n_pri, 1), np.nan)
        opp = (pd.DataFrame({"e": bcode[tw], "s": season[tw],
                             "q": rate_succ[tw]})
               .groupby(["e", "s"], sort=False)["q"].mean())
        opp_v = opp.to_numpy()
        mm = np.isfinite(d_season) & np.isfinite(opp_v)
        risk[f"{tag}_corr_season_deviation_vs_mean_opposing_pitcher_rate"] = (
            r5(np.corrcoef(d_season[mm], opp_v[mm])[0, 1]) if mm.sum() > 30
            else None)
        risk[f"{tag}_n_batter_seasons"] = int(mm.sum())
    risk["reading"] = (
        "asof_batter_* is the success rate of the PITCHES THROWN TO that "
        "batter, so a batter facing weak pitchers inherits their rate.  A "
        "large |corr| with the opposing-pitcher quality means the axis is a "
        "schedule proxy rather than a batter-skill signal; it does not by "
        "itself make the axis useless (schedule composition is legal, as-of "
        "information) but it changes what the column MEANS and is reported "
        "here rather than left implicit.")
    desc["batter_interpretation_risk"] = risk

    M["fold2024_descriptives"] = desc
    M["minutes"] = round((time.time() - t0) / 60, 3)
    with open(os.path.join(OUT_DIR, "stage1_diag.json"), "w") as f:
        json.dump(json.loads(json.dumps(M, default=str)), f, indent=1,
                  ensure_ascii=False)
    log("wrote stage1_diag.json,", M["minutes"], "min")


if __name__ == "__main__":
    main()
