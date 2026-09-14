"""exp_007 — combination of the three validated single-variable changes.

Spec: experiments/exp_007/config.yaml (2026-08-05, campaign 3)
Combines (each individually validated in campaign 2):
  1. exp_004 training mask : drop season<=2022 & game_type=="F" rows (train only)
  2. exp_003 profile join  : pitcher TrackMan as-of profile, fold-wise cutoff
  3. exp_005 calibration C : train-only base-rate trend anchor, constant logit
                             offset RECOMPUTED for this model (exp_005 rule)

Variants (both measured on fold_2024, train<=2023 -> valid 2024):
  V1 (main)    : mask + profile + calibration C
  V2 (ablation): mask + calibration C (no profile)
Pre-calibration numbers are also reported (decomposition).

Iteration policy (exp_006 P4, re-measured for THIS spec per variant):
  - each fold is trained ONCE for 1000 rounds without early stopping (fixed
    seed => the first k trees equal a k-round model);
  - the per-iteration valid-R Brier curve is computed via cumulative raw-score
    increments (verified against direct predict at checkpoints);
  - fold_2024 headline iter = argmin of the SAME variant's fold_2023 valid-R
    Brier curve (no 2024 peeking). fold_2023 headline iter = fold_2022 argmin.
  - fold_2024's own argmin is recorded only as (a) reference and (b) the P4
    carry-over for the final <=2024 submission model.

Calibration C (per variant, fold_2024; NO fold_2024 valid usage):
  r_hat_2024 = linear trend of R-league yearly success rate over seasons
  2019..2023 (all fold_2024 TRAIN years) extrapolated to 2024; anchor m =
  this variant's fold_2024 model predicted mean on season==2023 R TRAIN rows
  (at the P4 iter); delta = logit(r_hat) - logit(m), applied as a per-row
  constant (rule 2-4 safe).

Leak rules honoured: as-of profile cutoff == train horizon (asserted); no
current-row TrackMan physics; category codes from train.csv only; valid sets
unchanged vs exp_001 (full valid year incl. F).

Runtime: A5000 CPU, venv Python 3.11.15 / numpy 1.26.4 / pandas 2.0.3 /
lightgbm 4.7.0 / scipy 1.15.3. num_threads=8 (campaign-2 convention;
exp_006 observed thread invariance on this data).
"""

import hashlib
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import log_loss, roc_auc_score

SEED = 42
DATA_DIR = os.path.expanduser("~/SungjinPark/data")
DATA_PATH = os.path.join(DATA_DIR, "train.csv")
TM_PATH = os.path.join(DATA_DIR, "trackman_history.csv")
PMAP_PATH = os.path.expanduser(
    "~/SungjinPark/experiments/exp_002/out/pitcher_id_map.csv")
OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_007")

TARGET = "control_success"
DROP_COLS = ["row_id", TARGET]
CAT_COLS = [
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
    "pitcher_id",
    "batter_id",
]
FOLDS = [
    ("fold_2022", 2021, 2022),
    ("fold_2023", 2022, 2023),
    ("fold_2024", 2023, 2024),
]
F_EXCLUDE_MAX_SEASON = 2022  # exp_004 mask

LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": SEED,
    "num_threads": 8,
    "verbose": -1,
}
NUM_BOOST_ROUND = 1000
ES_PATIENCE = 200  # ES-equivalent reference only (policy is P4, not ES)
EPS = 1e-7

# TrackMan profile definition (identical to exp_003 / exp_002 builder)
PHYS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
        "extension", "rel_height", "rel_side", "zone_speed"]
STD_COLS = ("rel_speed", "spin_rate", "rel_height", "rel_side",
            "induced_vert_break", "horz_break")

# additivity references (campaign-2 measured, fold_2024 valid-R Brier)
REF = {
    "exp_001_baseline": 0.249337,
    "exp_003_profile_delta": -0.000070,
    "exp_004_mask_delta": -0.000088,
    "exp_005_calibC_delta": -0.000233,
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


def seg_metrics(y, p):
    """Brier / BSS(r = actual valid rate) / AUC / logloss / means."""
    n = int(len(y))
    if n == 0:
        return {"n": 0}
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    brier = float(np.mean((p - y) ** 2))
    r = float(np.mean(y))
    denom = r * (1.0 - r)
    bss = float(1.0 - brier / denom) if denom > 0 else None
    out = {
        "n": n,
        "brier": round(brier, 6),
        "bss": round(bss, 6) if bss is not None else None,
        "base_rate": round(r, 6),
        "pred_mean": round(float(np.mean(p)), 6),
    }
    if 0.0 < r < 1.0:
        out["auc"] = round(float(roc_auc_score(y, p)), 6)
        out["logloss"] = round(float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15))), 6)
    return out


def es_equivalent_best_iter(losses, patience=ES_PATIENCE):
    """lgb.early_stopping semantics on a recorded loss history (exp_006)."""
    best_loss = float("inf")
    best_it = 0
    for i, v in enumerate(losses, start=1):
        if v < best_loss:
            best_loss = v
            best_it = i
        elif i - best_it >= patience:
            return best_it, True
    return best_it, False


# ---------------- profile (exp_003 logic, unchanged) ----------------------
def sub_profile(df, prefix):
    g = df.groupby("pitcher_trackman_id")
    out = pd.DataFrame(index=g.size().index)
    out[f"{prefix}_n_pitches"] = g.size()
    out[f"{prefix}_n_games"] = g["trackman_game_id"].nunique()
    for c in PHYS:
        out[f"{prefix}_{c}_mean"] = g[c].mean()
        if c in STD_COLS:
            out[f"{prefix}_{c}_std"] = g[c].std()
    out[f"{prefix}_rel_speed_p90"] = g["rel_speed"].quantile(0.9)
    n = g.size()
    for pg in ("fastball", "breaking", "offspeed"):
        sub = df[df["pitch_type_group"] == pg].groupby("pitcher_trackman_id")
        out[f"{prefix}_{pg}_share"] = (sub.size() / n).fillna(0.0)
        out[f"{prefix}_{pg}_rel_speed_mean"] = sub["rel_speed"].mean()
        out[f"{prefix}_{pg}_spin_rate_mean"] = sub["spin_rate"].mean()
        out[f"{prefix}_{pg}_ivb_mean"] = sub["induced_vert_break"].mean()
        out[f"{prefix}_{pg}_hb_mean"] = sub["horz_break"].mean()
    return out


def build_asof_profile(tm, pmap, cutoff):
    """As-of profile from trackman seasons <= cutoff ONLY (asserted)."""
    sub = tm[tm["season"] <= cutoff]
    assert len(sub) > 0 and int(sub["season"].max()) <= cutoff, "as-of cutoff violated"
    career = sub_profile(sub, "career")
    recent = sub_profile(sub[sub["season"] == cutoff], "recent")
    prof = career.join(recent, how="left")
    prof["last_season_gap"] = cutoff - sub.groupby("pitcher_trackman_id")["season"].max()
    prof = prof.reset_index()
    prof = pmap[["pitcher_id", "pitcher_trackman_id"]].merge(
        prof, on="pitcher_trackman_id", how="inner")
    assert prof["pitcher_id"].is_unique
    return prof.drop(columns=["pitcher_trackman_id"]).set_index("pitcher_id")


# ---------------- helpers -------------------------------------------------
def make_X(df, pid, feat_cols, mask, prof):
    X = df.loc[mask, feat_cols]
    if prof is not None:
        blk = prof.reindex(pid[mask])
        blk.index = X.index
        X = pd.concat([X, blk], axis=1)
    return X


def to_numpy_codes(X):
    """DataFrame -> float64 matrix with categorical codes (-1 -> NaN).

    Codes equal what lgb.Dataset used at train time (same category dtype),
    so predicting on this matrix bypasses (and must equal) the pandas path —
    asserted against direct DataFrame predict at checkpoints.
    """
    cols = []
    for c in X.columns:
        if isinstance(X[c].dtype, pd.CategoricalDtype):
            codes = X[c].cat.codes.to_numpy(dtype=np.float64)
            codes[codes == -1] = np.nan
            cols.append(codes)
        else:
            cols.append(X[c].to_numpy(dtype=np.float64))
    return np.column_stack(cols)


def brier_curve(booster, X_np, X_df, y, is_r, n_iter):
    """Per-iteration valid-R Brier via one pred_leaf pass + leaf-value lookup.

    raw_k = init_const + sum_{t<k} leaf_value[t, leaf_idx[:, t]].
    Verified against direct raw predict at iter 1, 50 and n_iter (1e-8), and
    the numpy-codes path against the pandas DataFrame path (1e-10).
    """
    leaf = booster.predict(X_np, pred_leaf=True)
    assert leaf.shape == (X_np.shape[0], n_iter)
    y_r = y[is_r]
    curve = np.empty(n_iter)

    def leaf_values(t):
        n_leaves = int(leaf[:, t].max()) + 1
        return np.array([booster.get_leaf_output(t, l) for l in range(n_leaves)])

    # init-score constant: direct 1-iter raw minus tree-0 leaf sums must be
    # a constant across rows (0 if the init is absorbed into tree 0).
    r1_direct = booster.predict(X_np, num_iteration=1, raw_score=True)
    r1_leaf = leaf_values(0)[leaf[:, 0]]
    c = r1_direct - r1_leaf
    assert float(c.max() - c.min()) < 1e-10, "init offset not constant"
    init_const = float(c[0])

    raw = np.full(X_np.shape[0], init_const)
    checkpoints = {}
    for t in range(n_iter):
        raw += leaf_values(t)[leaf[:, t]]
        p_r = expit(raw[is_r])
        curve[t] = float(np.mean((p_r - y_r) ** 2))
        if t + 1 in (50, n_iter):
            checkpoints[t + 1] = raw.copy()
    for k, raw_k in checkpoints.items():
        direct = booster.predict(X_np, num_iteration=k, raw_score=True)
        gap = float(np.max(np.abs(direct - raw_k)))
        assert gap < 1e-8, f"leaf-sum raw mismatch at iter {k}: {gap}"
    # numpy-codes path == pandas path (category mapping identity)
    k0 = min(50, n_iter)
    p_df = booster.predict(X_df, num_iteration=k0)
    p_np = expit(checkpoints[k0])
    gap2 = float(np.max(np.abs(p_df - p_np)))
    assert gap2 < 1e-10, f"pandas/numpy predict mismatch: {gap2}"
    return curve


def apply_offset(p, delta):
    return expit(logit(np.clip(p, EPS, 1 - EPS)) + delta)


def main():
    np.random.seed(SEED)
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] loading {DATA_PATH}", flush=True)
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    assert df["row_id"].is_unique
    print(f"train loaded: {df.shape}", flush=True)

    pmap = pd.read_csv(PMAP_PATH)
    tm = pd.read_csv(
        TM_PATH, encoding="utf-8-sig",
        usecols=["season", "trackman_game_id", "pitcher_trackman_id",
                 "pitch_type_group"] + PHYS)
    assert int(tm["season"].max()) <= 2024, "trackman contains 2025 — violation"
    tm = tm[tm["pitcher_trackman_id"].isin(set(pmap["pitcher_trackman_id"]))]
    print(f"trackman rows for mapped pitchers: {len(tm):,}", flush=True)

    feat_cols = [c for c in df.columns if c not in DROP_COLS]
    pid = df["pitcher_id"].to_numpy()  # numeric copy (join key / cold-start)
    for c in CAT_COLS:  # train-only vocabulary (train.csv is the only input)
        df[c] = df[c].astype("category")

    f_excl = ((df["season"] <= F_EXCLUDE_MAX_SEASON)
              & (df["game_type"] == "F")).to_numpy()
    print(f"exp_004 mask rows (season<={F_EXCLUDE_MAX_SEASON} & F): "
          f"{int(f_excl.sum()):,}", flush=True)

    metrics = {
        "exp_id": "exp_007",
        "base_exp": "exp_001",
        "seed": SEED,
        "lgb_params": LGB_PARAMS,
        "num_boost_round": NUM_BOOST_ROUND,
        "iter_policy": "exp_006 P4 — prev fold valid-R Brier argmin, re-measured per variant",
        "mask": f"training excludes season<={F_EXCLUDE_MAX_SEASON} & game_type=='F'",
        "n_mask_rows_total": int(f_excl.sum()),
        "references": REF,
        "sha256": {
            "train.csv": sha256_file(DATA_PATH),
            "trackman_history.csv": sha256_file(TM_PATH),
            "pitcher_id_map.csv": sha256_file(PMAP_PATH),
            "train.py": sha256_file(os.path.abspath(__file__)),
        },
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "lightgbm": lgb.__version__,
        },
    }

    # pre-build fold profiles once (V1 reuses; V2 ignores)
    profiles = {}
    for _, train_until, valid_year in FOLDS:
        cutoff = valid_year - 1
        assert cutoff == train_until
        tp = time.time()
        profiles[cutoff] = build_asof_profile(tm, pmap, cutoff)
        print(f"as-of profile cutoff={cutoff}: {profiles[cutoff].shape[0]} "
              f"pitchers x {profiles[cutoff].shape[1]} feats "
              f"({time.time()-tp:.1f}s)", flush=True)

    for variant, use_profile in (("V1", True), ("V2", False)):
        vt = time.time()
        vout = {"spec": ("mask + profile + calibC" if use_profile
                         else "mask + calibC (no profile)")}
        prev_argmin = None
        boosters = {}
        curves = {}
        for fold_name, train_until, valid_year in FOLDS:
            ft = time.time()
            prof = profiles[train_until] if use_profile else None
            tr_mask = ((df["season"] <= train_until).to_numpy()) & ~f_excl
            va_mask = (df["season"] == valid_year).to_numpy()
            X_tr = make_X(df, pid, feat_cols, tr_mask, prof)
            y_tr = df.loc[tr_mask, TARGET].to_numpy()
            X_va = make_X(df, pid, feat_cols, va_mask, prof)
            y_va = df.loc[va_mask, TARGET].to_numpy()
            print(f"[{time.strftime('%H:%M:%S')}] {variant} {fold_name}: "
                  f"train {X_tr.shape[0]:,} (<= {train_until}, masked) -> "
                  f"valid {X_va.shape[0]:,} ({valid_year}), "
                  f"{X_tr.shape[1]} feats, {NUM_BOOST_ROUND} rounds no-ES",
                  flush=True)

            dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=CAT_COLS)
            dva = lgb.Dataset(X_va, label=y_va, reference=dtr,
                              categorical_feature=CAT_COLS)
            evals = {}
            booster = lgb.train(
                LGB_PARAMS, dtr, num_boost_round=NUM_BOOST_ROUND,
                valid_sets=[dva], valid_names=["valid"],
                callbacks=[lgb.record_evaluation(evals),
                           lgb.log_evaluation(200)],
            )
            train_sec = time.time() - ft
            losses = list(map(float, evals["valid"]["binary_logloss"]))
            es_it, es_fired = es_equivalent_best_iter(losses)

            is_r = (df.loc[va_mask, "game_type"] == "R").to_numpy()
            tc = time.time()
            X_va_np = to_numpy_codes(X_va)
            curve = brier_curve(booster, X_va_np, X_va, y_va, is_r,
                                NUM_BOOST_ROUND)
            argmin_it = int(np.argmin(curve)) + 1
            print(f"  {variant} {fold_name}: trained {train_sec/60:.2f} min, "
                  f"curve {time.time()-tc:.1f}s; R-Brier argmin iter="
                  f"{argmin_it} ({curve[argmin_it-1]:.6f}), ES-equiv={es_it} "
                  f"(fired={es_fired}, R-Brier {curve[es_it-1]:.6f})",
                  flush=True)

            # headline iter: P4 inheritance from previous fold (fold_2022: own
            # argmin, reference only — no earlier fold exists)
            head_it = prev_argmin if prev_argmin is not None else argmin_it
            p_head = booster.predict(X_va, num_iteration=head_it)

            seen = set(pid[tr_mask])
            cold = ~np.isin(pid[va_mask], list(seen))
            fold_out = {
                "train_until": train_until,
                "valid_year": valid_year,
                "n_train": int(tr_mask.sum()),
                "n_features": int(X_tr.shape[1]),
                "train_time_min": round(train_sec / 60, 2),
                "headline_iter": head_it,
                "headline_iter_source": ("prev_fold_R_brier_argmin (P4)"
                                         if prev_argmin is not None
                                         else "own argmin (reference — first fold)"),
                "own_R_brier_argmin": {"iter": argmin_it,
                                       "brier": round(float(curve[argmin_it - 1]), 6)},
                "es_equivalent": {"iter": es_it, "triggered": bool(es_fired),
                                  "R_brier_at_es": round(float(curve[es_it - 1]), 6)},
                "valid_all": seg_metrics(y_va, p_head),
                "valid_R": seg_metrics(y_va[is_r], p_head[is_r]),
                "valid_F": seg_metrics(y_va[~is_r], p_head[~is_r]),
                "cold_start_R": seg_metrics(y_va[cold & is_r], p_head[cold & is_r]),
                "seen_R": seg_metrics(y_va[~cold & is_r], p_head[~cold & is_r]),
                "curve_head60_R_brier": [round(float(v), 6) for v in curve[:60]],
            }
            if use_profile:
                has_prof = X_va["career_n_pitches"].notna().to_numpy()
                fold_out["valid_R_has_profile"] = seg_metrics(
                    y_va[is_r & has_prof], p_head[is_r & has_prof])
                fold_out["valid_R_no_profile"] = seg_metrics(
                    y_va[is_r & ~has_prof], p_head[is_r & ~has_prof])
                fold_out["valid_profile_row_coverage"] = round(
                    float(has_prof.mean()), 4)
            vout[fold_name] = fold_out
            boosters[fold_name] = booster
            curves[fold_name] = curve
            booster.save_model(
                os.path.join(OUT_DIR, f"model_{variant}_{fold_name}_1000.txt"))
            prev_argmin = argmin_it

        # ---- calibration C on fold_2024 (train-only fit) ------------------
        i24 = vout["fold_2024"]["headline_iter"]  # = fold_2023 argmin (P4)
        b24 = boosters["fold_2024"]
        tr_seasons_mask = (df["season"] <= 2023) & (df["game_type"] == "R")
        r_hist = df.loc[tr_seasons_mask].groupby("season")[TARGET].mean()
        slope, intercept = np.polyfit(r_hist.index.to_numpy(dtype=float),
                                      r_hist.to_numpy(dtype=float), 1)
        r_hat = float(slope * 2024 + intercept)
        anchor_mask = ((df["season"] == 2023)
                       & (df["game_type"] == "R")).to_numpy()
        prof24 = profiles[2023] if use_profile else None
        X_anchor = make_X(df, pid, feat_cols, anchor_mask, prof24)
        m_anchor = float(np.mean(b24.predict(X_anchor, num_iteration=i24)))
        delta_c = float(logit(r_hat) - logit(m_anchor))

        va_mask24 = (df["season"] == 2024).to_numpy()
        X_va24 = make_X(df, pid, feat_cols, va_mask24, prof24)
        y_va24 = df.loc[va_mask24, TARGET].to_numpy()
        is_r24 = (df.loc[va_mask24, "game_type"] == "R").to_numpy()
        p24 = b24.predict(X_va24, num_iteration=i24)
        p24_c = apply_offset(p24, delta_c)
        seen24 = set(pid[((df["season"] <= 2023).to_numpy()) & ~f_excl])
        cold24 = ~np.isin(pid[va_mask24], list(seen24))
        vout["calibration_C"] = {
            "fit_on": "train-only: R yearly rates 2019..2023 + fold_2024 model "
                      "pred mean on season==2023 R train rows (at P4 iter)",
            "iter_used": i24,
            "r_history": {int(s): round(float(r), 6)
                          for s, r in r_hist.items()},
            "trend_slope_per_year": round(float(slope), 6),
            "r_hat_2024": round(r_hat, 6),
            "anchor_pred_mean_2023R_train": round(m_anchor, 6),
            "delta": round(delta_c, 6),
            "fold_2024_valid_R_precalib": seg_metrics(y_va24[is_r24], p24[is_r24]),
            "fold_2024_valid_R_postcalib": seg_metrics(y_va24[is_r24], p24_c[is_r24]),
            "fold_2024_valid_all_postcalib": seg_metrics(y_va24, p24_c),
            "fold_2024_cold_start_R_postcalib": seg_metrics(
                y_va24[cold24 & is_r24], p24_c[cold24 & is_r24]),
            "fold_2024_seen_R_postcalib": seg_metrics(
                y_va24[~cold24 & is_r24], p24_c[~cold24 & is_r24]),
        }
        vout["train_time_min"] = round((time.time() - vt) / 60, 2)
        metrics[variant] = vout
        print(json.dumps({variant: {
            "fold_2024_headline_iter": i24,
            "precalib_R_brier": vout["calibration_C"]["fold_2024_valid_R_precalib"]["brier"],
            "postcalib_R_brier": vout["calibration_C"]["fold_2024_valid_R_postcalib"]["brier"],
            "delta_c": vout["calibration_C"]["delta"],
            "fold_2024_own_argmin": vout["fold_2024"]["own_R_brier_argmin"],
        }}, indent=2), flush=True)

    # ---- additivity verdict ----------------------------------------------
    base = REF["exp_001_baseline"]
    indiv_sum = (REF["exp_003_profile_delta"] + REF["exp_004_mask_delta"]
                 + REF["exp_005_calibC_delta"])
    v1_post = metrics["V1"]["calibration_C"]["fold_2024_valid_R_postcalib"]["brier"]
    v2_post = metrics["V2"]["calibration_C"]["fold_2024_valid_R_postcalib"]["brier"]
    v1_pre = metrics["V1"]["calibration_C"]["fold_2024_valid_R_precalib"]["brier"]
    metrics["additivity"] = {
        "exp_001_baseline": base,
        "individual_delta_sum": round(indiv_sum, 6),
        "expected_if_additive": round(base + indiv_sum, 6),
        "V1_postcalib": v1_post,
        "V1_precalib": v1_pre,
        "V2_postcalib": v2_post,
        "V1_delta_vs_baseline": round(v1_post - base, 6),
        "V1_gap_vs_additive": round(v1_post - (base + indiv_sum), 6),
        "profile_contribution_in_combo": round(v1_post - v2_post, 6),
    }
    metrics["train_time_min"] = round((time.time() - t0) / 60, 2)
    with open(os.path.join(OUT_DIR, "metrics_folds.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(json.dumps({"additivity": metrics["additivity"]}, indent=2), flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] DONE total {metrics['train_time_min']} min",
          flush=True)


if __name__ == "__main__":
    main()
