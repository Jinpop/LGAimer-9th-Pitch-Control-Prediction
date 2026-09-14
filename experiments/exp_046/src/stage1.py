"""exp_046 STAGE 1 — diagnostic only.  ZERO model fits.

Pre-registered in experiments/exp_046/config.yaml (stage1_diagnostic).
Everything this script prints/saves is a MEASUREMENT on train.csv plus the
already-stored exp_030 C5_B3 predictions.  Nothing is trained here.

Integrity gates (config: 🔴_무결성_게이트) — any failure stops stage 2:
  G1  0 <= S_season <= n_season on every row
  G2  train-2024 rows: decomposition == label-cumulated ground truth
  G3  n_season strictly +1 within (pitcher, season)
"""

import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from form_decomp import (N_COL, PITCHER, RATE_COL, TARGET,  # noqa: E402
                         apply_decomposition, build_boundary_tables,
                         career_successes, extend_table_for_unseen)

HOME = os.path.expanduser("~/SungjinPark")
DATA = os.path.join(HOME, "data/train.csv")
TEST = os.path.join(HOME, "data/test.csv")
PRED_DIR = os.path.join(HOME, "experiments/exp_030/pred")
OUT_DIR = os.path.join(HOME, "experiments/exp_046")

BRIER_REF = 0.2498069            # CLAUDE.md 환산 상수
PTS_PER_BRIER = 100000.0 / BRIER_REF

F_EXCLUDE_MAX_SEASON = 2022
EXPECTED_2024 = {"valid_all": 253507, "valid_R": 223497}
K_GRID = [100, 200, 400, 800, 1600]     # config stage2_arms.k_grid
K_REPORT = 400                          # bin/coverage reporting k

# stored exp_030 C5_B3 predictions: block/fold -> {seed: filename}
PRED = {
    2022: {42: "pred_C5B3_u2022_seed42_st2.npy", 7: "pred_C5B3_u2022_seed7_st2.npy"},
    2023: {42: "pred_C5B3_u2023_seed42_st1.npy", 7: "pred_C5B3_u2023_seed7_st2.npy"},
    2024: {42: "pred_C5B3_u2024_seed42_st2.npy", 7: "pred_C5B3_u2024_seed7_st2.npy",
           2024: "pred_C5B3_u2024_seed2024_st2.npy"},
}


def sha256_file(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def score(brier):
    return 100000.0 * (1.0 - brier / BRIER_REF)


def r5(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), 5)


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    M = {"exp_id": "exp_046", "stage": 1, "fits_run": 0,
         "brier_ref": BRIER_REF,
         "sha256": {"train.csv": sha256_file(DATA),
                    "stage1.py": sha256_file(os.path.abspath(__file__)),
                    "form_decomp.py": sha256_file(os.path.join(
                        os.path.dirname(os.path.abspath(__file__)), "form_decomp.py"))},
         "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                      "pandas": pd.__version__}}

    log("loading train.csv ...")
    use = ["row_id", "season", "game_type", PITCHER, N_COL, RATE_COL, TARGET]
    df = pd.read_csv(DATA, usecols=use, encoding="utf-8-sig")
    assert list(df.columns) == use or set(df.columns) == set(use), list(df.columns)
    log(f"loaded {df.shape}")
    assert df["season"].is_monotonic_increasing, "file order assumption broken"

    season = df["season"].to_numpy()
    n = df[N_COL].to_numpy(dtype=np.float64)
    rate = df[RATE_COL].to_numpy(dtype=np.float64)
    y = df[TARGET].to_numpy(dtype=np.float64)
    codes, uniq = pd.factorize(df[PITCHER], sort=True)
    pcode = codes.astype(np.int64)
    npit = len(uniq)

    # ---- rounding hazard (audit item: banker's rounding) ------------------ #
    s_car, max_resid, min_dist_half = career_successes(n, rate)
    M["rounding"] = {
        "max_abs_residual_raw_minus_round": max_resid,
        "min_distance_of_raw_frac_to_0.5": min_dist_half,
        "bankers_rounding_can_bite": bool(min_dist_half < 1e-9),
        "note": "np.round is half-to-even; it only differs from half-up when "
                "the raw product lands exactly on .5.  min_distance above is "
                "the measured margin over all 1.47M train rows.",
    }
    log("rounding:", M["rounding"])

    # ---- frozen boundary tables (TRAIN ONLY) ------------------------------ #
    log("building frozen boundary tables (train only) ...")
    tables = build_boundary_tables(df, pcode, npit)
    zero = (np.zeros(npit), np.zeros(npit))
    M["frozen_tables"] = {
        "boundaries": sorted(tables.keys()),
        "n_pitchers": int(npit),
        "pitchers_with_state_at_boundary": {
            str(b): int((tables[b][0] > 0).sum()) for b in sorted(tables)},
        "built_from": "ALL train rows (incl. season<=2022 game_type==F), because "
                      "asof_pitcher_n is the organisers' counter over all rows; "
                      "excluding F would desynchronise n_end from the column.",
    }

    # ---- row-wise decomposition, per season, using the S-1 table ---------- #
    log("applying row-wise decomposition ...")
    cols = {c: np.full(len(df), np.nan) for c in
            ("S_career", "n_end", "S_end", "n_season", "S_season",
             "rate_season", "rate_prior", "d_form")}
    hf = np.zeros(len(df), dtype=np.int8)
    d_eb = {k: np.full(len(df), np.nan) for k in K_GRID}
    for s in np.sort(np.unique(season)):
        m = season == s
        tab = tables.get(int(s) - 1, zero)
        for k in K_GRID:
            out = apply_decomposition(n[m], rate[m], pcode[m], tab[0], tab[1], k)
            d_eb[k][m] = out["d_form_eb"]
        for c in cols:
            cols[c][m] = out[c]
        hf[m] = out["has_form"]
    for c in cols:
        df[c] = cols[c]
    df["has_form"] = hf

    # =================================================================== #
    #  G1 / G2 / G3
    # =================================================================== #
    gates = {}

    # ---- G1 -------------------------------------------------------------- #
    v_lo = int((cols["S_season"] < -1e-9).sum())
    v_hi = int((cols["S_season"] > cols["n_season"] + 1e-9).sum())
    v_nneg = int((cols["n_season"] < -1e-9).sum())
    gates["G1"] = {
        "rule": "0 <= S_season <= n_season on every train row",
        "n_rows_checked": int(len(df)),
        "violations_S_season_negative": v_lo,
        "violations_S_season_gt_n_season": v_hi,
        "violations_n_season_negative": v_nneg,
        "PASS": bool(v_lo == 0 and v_hi == 0 and v_nneg == 0),
    }
    log("G1:", gates["G1"])

    # ---- G2 : the decisive gate ------------------------------------------ #
    # train-2024 rows carry labels, so n_season / S_season can be built
    # DIRECTLY by cumulating control_success inside season 2024 per pitcher.
    # (groupby/cumsum here is train-side diagnostic ONLY -- never on test.)
    m24 = season == 2024
    sub = pd.DataFrame({"p": pcode[m24], "y": y[m24]})
    g = sub.groupby("p", sort=False)
    n_dir = g.cumcount().to_numpy(dtype=np.float64)
    s_dir = (g["y"].cumsum().to_numpy(dtype=np.float64) - sub["y"].to_numpy())
    dn = np.abs(cols["n_season"][m24] - n_dir)
    ds = np.abs(cols["S_season"][m24] - s_dir)
    gates["G2"] = {
        "rule": "train-2024: decomposition (n_season,S_season) == label-cumulated truth",
        "n_rows_checked": int(m24.sum()),
        "mismatch_rows_n_season": int((dn > 1e-9).sum()),
        "mismatch_rows_S_season": int((ds > 1e-9).sum()),
        "max_abs_diff_n_season": float(np.max(dn)) if m24.sum() else None,
        "max_abs_diff_S_season": float(np.max(ds)) if m24.sum() else None,
        "PASS": bool((dn > 1e-9).sum() == 0 and (ds > 1e-9).sum() == 0),
    }
    log("G2:", gates["G2"])

    # same check for every season with a prior table, for completeness
    per_season_g2 = {}
    for s in (2020, 2021, 2022, 2023):
        ms = season == s
        sb = pd.DataFrame({"p": pcode[ms], "y": y[ms]})
        gg = sb.groupby("p", sort=False)
        nd = gg.cumcount().to_numpy(dtype=np.float64)
        sd = gg["y"].cumsum().to_numpy(dtype=np.float64) - sb["y"].to_numpy()
        per_season_g2[str(s)] = {
            "mismatch_n": int((np.abs(cols["n_season"][ms] - nd) > 1e-9).sum()),
            "mismatch_S": int((np.abs(cols["S_season"][ms] - sd) > 1e-9).sum()),
            "n_rows": int(ms.sum())}
    gates["G2_other_seasons"] = per_season_g2
    log("G2 other seasons:", per_season_g2)

    # ---- G3 -------------------------------------------------------------- #
    key = pd.DataFrame({"p": pcode, "s": season})
    step = pd.Series(cols["n_season"]).groupby([key["p"], key["s"]]).diff()
    bad3 = int((step.notna() & (step != 1.0)).sum())
    gates["G3"] = {
        "rule": "n_season increases by exactly +1 within (pitcher, season)",
        "n_steps_checked": int(step.notna().sum()),
        "violations": bad3,
        "PASS": bool(bad3 == 0),
    }
    log("G3:", gates["G3"])
    M["gates"] = gates

    gates_ok = gates["G1"]["PASS"] and gates["G2"]["PASS"] and gates["G3"]["PASS"]
    M["gates_all_pass"] = bool(gates_ok)

    # =================================================================== #
    #  coverage / distribution  (fold_2024 valid = season 2024)
    # =================================================================== #
    assert int(m24.sum()) == EXPECTED_2024["valid_all"], int(m24.sum())
    is_r = (df.loc[m24, "game_type"] == "R").to_numpy()
    assert int(is_r.sum()) == EXPECTED_2024["valid_R"]
    hf24 = df["has_form"].to_numpy()[m24].astype(bool)
    ns24 = cols["n_season"][m24]
    M["coverage_fold2024_valid"] = {
        "n_rows": int(m24.sum()),
        "has_form_pct": r5(100.0 * hf24.mean()),
        "has_form_pct_valid_R": r5(100.0 * hf24[is_r].mean()),
        "has_form_pct_valid_F": r5(100.0 * hf24[~is_r].mean()),
        "rows_n_season_eq_0_pct": r5(100.0 * float((ns24 == 0).mean())),
        "rows_no_prior_season_pct": r5(100.0 * float((cols["n_end"][m24] == 0).mean())),
        "n_season_quantiles": {q: float(np.quantile(ns24, q))
                               for q in (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)},
        "n_season_mean": r5(float(ns24.mean())),
        "share_of_rows_with_n_season_lt": {
            str(t): r5(100.0 * float((ns24 < t).mean())) for t in (10, 50, 100, 200, 400, 800)},
    }
    log("coverage:", json.dumps(M["coverage_fold2024_valid"], ensure_ascii=False))

    # ---- dilution: the algebraic identity, measured --------------------- #
    # asof_rate = rate_prior + (n_season/n) * d_form   (exact when n_end>0,n>0)
    mm = m24 & (df["has_form"].to_numpy() == 1)
    w = cols["n_season"][mm] / n[mm]
    ident = np.abs(rate[mm] - (cols["rate_prior"][mm] + w * cols["d_form"][mm]))
    dfe = d_eb[K_REPORT]
    M["dilution"] = {
        "identity": "asof_pitcher_success_rate == rate_prior + (n_season/n)*d_form",
        "identity_max_abs_error": float(np.nanmax(ident)),
        "weight_w_mean_fold2024": r5(float(w.mean())),
        "weight_w_median_fold2024": r5(float(np.median(w))),
        "weight_w_q90": r5(float(np.quantile(w, 0.90))),
        "interpretation": "w is the fraction of the career-rate column that the "
                          "current season contributes.  1-w is the share of this "
                          "season's form that the existing feature dilutes away.",
        f"corr_d_form_eb_k{K_REPORT}__vs__asof_success_rate":
            r5(float(np.corrcoef(dfe[mm], rate[mm])[0, 1])),
        f"ols_slope_asof_rate_on_d_form_eb_k{K_REPORT}":
            r5(float(np.polyfit(dfe[mm], rate[mm], 1)[0])),
        "corr_d_form_vs_rate_prior": r5(float(np.corrcoef(
            cols["d_form"][mm], cols["rate_prior"][mm])[0, 1])),
    }
    log("dilution:", json.dumps(M["dilution"], ensure_ascii=False))

    # ---- true SD of d_form (exp_018 deconvolution procedure) ------------- #
    # season-level, TRAINING WINDOW ONLY (season <= 2023): never touches 2024.
    tw = season <= 2023
    kk = pd.DataFrame({"p": pcode, "s": season})[tw]
    last_tw = kk.groupby(["p", "s"], sort=False).tail(1).index.to_numpy()
    ns_f = cols["n_season"][last_tw] + 1.0
    ss_f = cols["S_season"][last_tw] + y[last_tw]
    npr = cols["n_end"][last_tw]
    spr = cols["S_end"][last_tw]
    truesd = {}
    for min_n in (50, 100, 200):
        keep = (ns_f >= min_n) & (npr >= min_n)
        p_s = ss_f[keep] / ns_f[keep]
        p_p = spr[keep] / npr[keep]
        d = p_s - p_p
        obs_var = float(np.var(d, ddof=1))
        noise_var = float(np.mean(p_s * (1 - p_s) / ns_f[keep]
                                  + p_p * (1 - p_p) / npr[keep]))
        true_var = max(obs_var - noise_var, 0.0)
        pbar = float(y[tw].mean())
        truesd[f"min_n_{min_n}"] = {
            "n_pitcher_seasons": int(keep.sum()),
            "obs_sd_pp": r5(100 * np.sqrt(obs_var)),
            "noise_sd_pp": r5(100 * np.sqrt(noise_var)),
            "true_sd_pp": r5(100 * np.sqrt(true_var)),
            "reliability": r5(true_var / obs_var) if obs_var > 0 else None,
            "obs_over_noise": r5(np.sqrt(obs_var / noise_var)) if noise_var > 0 else None,
            "eb_k_implied": r5(pbar * (1 - pbar) / true_var) if true_var > 0 else None,
        }
    M["true_sd_d_form"] = {
        "window": "seasons <= 2023 only (fold_2024 training window)",
        "method": "obs_var - E[binomial noise of both terms]; exp_018 prelim_stats",
        "pbar_train_le_2023": r5(float(y[tw].mean())),
        "by_min_n": truesd,
    }
    log("true SD:", json.dumps(M["true_sd_d_form"], ensure_ascii=False))

    # =================================================================== #
    #  signal check + gain: uses the STORED exp_030 C5_B3 predictions
    # =================================================================== #
    def load_pred(vy, seed):
        p = np.load(os.path.join(PRED_DIR, PRED[vy][seed]))
        m = season == vy
        assert len(p) == int(m.sum()), (vy, seed, len(p), int(m.sum()))
        return p, m

    # --- binned residual on fold_2024 (seed 42) --------------------------- #
    p42, _ = load_pred(2024, 42)
    yy = y[m24]
    dv = dfe[m24]
    cov = hf24
    edges = np.array([-1, -0.10, -0.06, -0.03, -0.01, 0.01, 0.03, 0.06, 0.10, 1])
    bins = []
    for i in range(len(edges) - 1):
        b = cov & (dv >= edges[i]) & (dv < edges[i + 1])
        if b.sum() == 0:
            continue
        bins.append({"lo": float(edges[i]), "hi": float(edges[i + 1]),
                     "n": int(b.sum()),
                     "mean_d_form_eb": r5(float(dv[b].mean())),
                     "actual_success_rate": r5(float(yy[b].mean())),
                     "mean_model_pred": r5(float(p42[b].mean())),
                     "residual_y_minus_p": r5(float((yy[b] - p42[b]).mean()))})
    M["binned_fold2024_seed42"] = {
        "k": K_REPORT,
        "note": "residual_y_minus_p is what the model has NOT already captured. "
                "A monotone residual across bins is the incremental signal.",
        "bins": bins,
    }
    log("bins:", json.dumps(bins, ensure_ascii=False))

    # --- A2 (offset) closed-form gain ------------------------------------- #
    def a2_gain(pred, yv, dvec, beta=None):
        """Δ of the 1-term offset p' = clip(p + beta*d).  beta=None -> oracle."""
        dz = np.where(np.isfinite(dvec), dvec, 0.0)
        num = float(np.sum(dz * (yv - pred)))
        den = float(np.sum(dz * dz))
        if den <= 0:
            return {"beta": None, "dBrier": 0.0, "dScore": 0.0}
        b = num / den if beta is None else float(beta)
        pp = np.clip(pred + b * dz, 1e-6, 1 - 1e-6)
        db = float(np.mean((pp - yv) ** 2) - np.mean((pred - yv) ** 2))
        return {"beta": b, "beta_oracle_closed_form": num / den,
                "dBrier": db, "dScore": -db * PTS_PER_BRIER,
                "sd_of_correction": float(np.std(b * dz))}

    gain = {"oracle_fold2024": {}, "blocks": {}, "honest": {}}
    for k in K_GRID:
        per_seed = {}
        for seed, _f in PRED[2024].items():
            pr, _ = load_pred(2024, seed)
            per_seed[str(seed)] = a2_gain(pr, yy, d_eb[k][m24])
        gain["oracle_fold2024"][str(k)] = {
            "per_seed": {s: {"beta": r5(v["beta"]), "dBrier": v["dBrier"],
                             "dScore": r5(v["dScore"]),
                             "sd_of_correction": r5(v["sd_of_correction"])}
                         for s, v in per_seed.items()},
            "mean_dScore": r5(np.mean([v["dScore"] for v in per_seed.values()])),
            "mean_beta": r5(np.mean([v["beta"] for v in per_seed.values()])),
            "sign_agreement": int(sum(1 for v in per_seed.values() if v["dScore"] > 0)),
        }

    # --- honest selection: k and beta chosen on blocks 2022+2023 only ------ #
    for k in K_GRID:
        rows = []
        for vy in (2022, 2023):
            mb = season == vy
            for seed in PRED[vy]:
                pr, _ = load_pred(vy, seed)
                rows.append((pr, y[mb], d_eb[k][mb]))
        num = sum(float(np.sum(np.nan_to_num(d) * (yv - pr))) for pr, yv, d in rows)
        den = sum(float(np.sum(np.nan_to_num(d) ** 2)) for pr, yv, d in rows)
        b_blk = num / den if den > 0 else 0.0
        dsc = []
        for pr, yv, d in rows:
            g_ = a2_gain(pr, yv, d, beta=b_blk)
            dsc.append(g_["dScore"])
        gain["blocks"][str(k)] = {"beta_block": r5(b_blk),
                                  "mean_block_dScore": r5(np.mean(dsc)),
                                  "per_unit_dScore": [r5(v) for v in dsc]}
    best_k = max(K_GRID, key=lambda k: gain["blocks"][str(k)]["mean_block_dScore"])
    b_star = gain["blocks"][str(best_k)]["beta_block"]
    hon = {}
    for seed in PRED[2024]:
        pr, _ = load_pred(2024, seed)
        hon[str(seed)] = a2_gain(pr, yy, d_eb[best_k][m24], beta=b_star)
    orc_best_k = max(K_GRID, key=lambda k: gain["oracle_fold2024"][str(k)]["mean_dScore"])
    gain["honest"] = {
        "k_chosen_on_blocks": best_k, "beta_chosen_on_blocks": r5(b_star),
        "fold2024_per_seed_dScore": {s: r5(v["dScore"]) for s, v in hon.items()},
        "fold2024_mean_dScore": r5(np.mean([v["dScore"] for v in hon.values()])),
        "sign_agreement": int(sum(1 for v in hon.values() if v["dScore"] > 0)),
        "oracle_best_k": orc_best_k,
        "oracle_mean_dScore": gain["oracle_fold2024"][str(orc_best_k)]["mean_dScore"],
        "cost_of_honest_selection_pts": r5(
            np.mean([v["dScore"] for v in hon.values()])
            - gain["oracle_fold2024"][str(orc_best_k)]["mean_dScore"]),
    }
    M["gain_A2_offset"] = gain
    log("gain honest:", json.dumps(gain["honest"], ensure_ascii=False))

    # --- theoretical ceiling from the true SD ----------------------------- #
    # 이득(점) = 400309 * (sd of the probability-space correction)^2
    ts = M["true_sd_d_form"]["by_min_n"]["min_n_100"]["true_sd_pp"] / 100.0
    cov_frac = hf24.mean()
    M["gain_ceiling"] = {
        "formula": "points = (100000/0.2498069) * sd_correction^2",
        "pts_per_brier": r5(PTS_PER_BRIER),
        "true_sd_d_form": r5(ts),
        "ceiling_if_beta_1_and_nothing_captured_pts":
            r5(PTS_PER_BRIER * (ts ** 2) * cov_frac),
        "sd_of_oracle_correction_best_k": r5(np.mean(
            [gain["oracle_fold2024"][str(orc_best_k)]["per_seed"][s]["sd_of_correction"]
             for s in gain["oracle_fold2024"][str(orc_best_k)]["per_seed"]])),
        "⚠️": "This is a CEILING, not a deployment expectation (exp_042 -> cand Z). "
              "The measured A2 numbers above are the real magnitude.",
    }
    log("ceiling:", json.dumps(M["gain_ceiling"], ensure_ascii=False))

    # =================================================================== #
    #  test.csv sanity (5 public rows) — row-wise path only
    # =================================================================== #
    te = pd.read_csv(TEST, encoding="utf-8-sig")
    pmap = {v: i for i, v in enumerate(uniq)}
    tn_ext, ts_ext, unseen_ix = extend_table_for_unseen(*tables[2024])
    tp = np.array([pmap.get(v, unseen_ix) for v in te[PITCHER]], dtype=np.int64)
    seen = tp != unseen_ix
    tn = te[N_COL].to_numpy(dtype=np.float64)
    tr = te[RATE_COL].to_numpy(dtype=np.float64)
    o = apply_decomposition(tn, tr, tp, tn_ext, ts_ext, K_REPORT)
    M["test_public_rows"] = {
        "n_rows": int(len(te)), "pitchers_seen_in_train": int(seen.sum()),
        "unseen_pitcher_rule": "pitchers absent from train.csv get (n_end,S_end)=(0,0), "
                               "the same state build_boundary_tables() already stores "
                               "for a train rookie.  Their rate_season is still defined "
                               "(cold-start resolves itself); has_form=0.",
        "rows": [{"row_id": te["row_id"][i], "pitcher_id": int(te[PITCHER][i]),
                  "asof_n": float(tn[i]), "asof_rate": (None if not np.isfinite(tr[i])
                                                        else float(tr[i])),
                  "n_end_2024": float(o["n_end"][i]), "S_end_2024": float(o["S_end"][i]),
                  "n_season": float(o["n_season"][i]),
                  "S_season": float(o["S_season"][i]),
                  "rate_season": (None if not np.isfinite(o["rate_season"][i])
                                  else r5(o["rate_season"][i])),
                  "rate_prior": (None if not np.isfinite(o["rate_prior"][i])
                                 else r5(o["rate_prior"][i])),
                  "has_form": int(o["has_form"][i])} for i in range(len(te))],
        "G1_on_test_rows_violations": int(((o["S_season"] < -1e-9)
                                           | (o["S_season"] > o["n_season"] + 1e-9)
                                           | (o["n_season"] < -1e-9)).sum()),
    }
    log("test rows:", json.dumps(M["test_public_rows"], ensure_ascii=False))

    # =================================================================== #
    #  pre-registered stop rules (config stage1_diagnostic.중단_기준)
    # =================================================================== #
    est = M["gain_A2_offset"]["honest"]["fold2024_mean_dScore"]
    orc = M["gain_A2_offset"]["honest"]["oracle_mean_dScore"]
    covpct = M["coverage_fold2024_valid"]["has_form_pct"]
    M["stop_rules"] = {
        "gates_all_pass": bool(gates_ok),
        "coverage_pct": covpct, "coverage_ge_50": bool(covpct >= 50.0),
        "estimated_gain_pts_honest_A2": est,
        "oracle_gain_pts_A2": orc,
        "gain_ge_30_honest": bool(est >= 30.0),
        "gain_ge_30_oracle": bool(orc >= 30.0),
        "PROCEED_TO_STAGE2": bool(gates_ok and covpct >= 50.0 and orc >= 30.0),
        "note": "A2 is the offset arm only; A1 (tree feature) cannot be measured "
                "without fitting and is not bounded by these numbers.",
    }
    log("STOP RULES:", json.dumps(M["stop_rules"], ensure_ascii=False))

    with open(os.path.join(OUT_DIR, "stage1.json"), "w") as f:
        json.dump(json.loads(json.dumps(M, default=str)), f, indent=2, ensure_ascii=False)
    log("wrote stage1.json")


if __name__ == "__main__":
    main()
