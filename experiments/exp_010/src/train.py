"""exp_010 — properly regularised LightGBM: can it beat the organizers' RF?

Spec: experiments/exp_010/config.yaml (2026-08-05, campaign 4).

Context (exp_008 diagnosis): on the SAME rolling-origin split, the organizers'
47-feature RandomForest reached fold_2024 valid-R Brier 0.248487 (AUC 0.5434)
while our 112-feature LightGBM + calibration reached only 0.249061 (AUC 0.5338).
Root cause diagnosed: pitcher_id / batter_id fed as native LightGBM categoricals
with num_leaves=127 -> the trees carve per-player leaves within 23~49 iterations
and gain is dominated by the IDs. This experiment changes the ID handling and
the regularisation, and measures four variants.

Variants (all share the exp_004 training mask; features = the 47 provided
test.csv columns, no TrackMan profile, so the comparison against exp_008 R1/R2
is like-for-like on the feature side):

  A  IDs removed from `categorical_feature` (kept as plain numeric columns),
     num_leaves 31, min_data_in_leaf 2000, learning_rate 0.02,
     feature_fraction 0.7.
  B  A, but pitcher_id / batter_id dropped from the feature matrix entirely
     (situation + as-of aggregates only, no player identity).
  C  A + causal target encoding of pitcher_id / batter_id.
  D  A's feature set with the strongest regularisation: num_leaves 15,
     min_data_in_leaf 5000, learning_rate 0.01, up to 3000 rounds.

Causal target encoding (C) — leakage-free AND reproducible at test time:
  EDA §2 established that the organizers' as-of columns are a file-order
  expanding, shift(1) career aggregate, i.e.
      asof_pitcher_success_rate[t] * asof_pitcher_n[t] == sum(y[<t])
  over that pitcher.  We recompute that expanding shift(1) sum ourselves from
  the labels (`_expanding_prev_sum`), ASSERT it equals the reconstruction from
  the provided columns, and then shrink towards the league prior:
      te = (prev_success + k * prior_t) / (prev_n + k)
  with prior_t = the file-order expanding shift(1) league mean at that row.
  A row's encoding therefore uses only strictly earlier rows — the same
  boundary the organizers used, so fold_2024 valid rows are treated exactly as
  2025 test rows will be.
  At inference the encoding is recomputed per row from the columns that exist
  in test.csv (`rate * n`), with prior frozen at the end-of-2024 league mean;
  a 2025 debutant has n = 0 / rate = NaN and falls back to te = prior.
  No test row ever looks at another test row (rule 2-4 safe).
  k is tuned on fold_2023 only; 2024 is never consulted for k.

Iteration policy: exp_006 P4 — the number of boosting rounds used to report a
fold is the argmin of the PREVIOUS fold's valid-R Brier curve, carried over
without scaling. Never chosen on the fold being reported. Each fold is trained
once without early stopping and the whole valid-R Brier-vs-iteration curve is
obtained by accumulating raw scores chunk by chunk (a fixed-seed booster's
first k trees are exactly the k-round model). fold_2022 has no predecessor, so
it is reported at its own argmin and flagged `oracle` — it exists only to feed
fold_2023.

Environment: A5000, ~/SungjinPark/venv (Python 3.11.15 / numpy 1.26.4 /
pandas 2.0.3 / sklearn 1.8.0 / lightgbm 4.7.0). num_threads=6 (three other
experiments share the box).
"""

import hashlib
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

SEED = 42
N_THREADS = 6

DATA_PATH = os.path.expanduser("~/SungjinPark/data/train.csv")
OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_010")

TARGET = "control_success"
DROP_COLS = ["row_id", TARGET]

# Only genuine (string) categoricals stay categorical. The exp_001 lineage also
# declared pitcher_hand / batter_hand / *_team_id / pitcher_id / batter_id as
# categoricals; those are integer-coded already and the IDs are the diagnosed
# over-fitting channel, so this run declares only the three string columns —
# exactly the set the organizers' RF ordinal-encodes.
CAT_COLS = ["top_bottom", "game_type", "base_state"]
ID_COLS = ["pitcher_id", "batter_id"]

FOLDS = [
    ("fold_2022", 2021, 2022),
    ("fold_2023", 2022, 2023),
    ("fold_2024", 2023, 2024),
]
F_EXCLUDE_MAX_SEASON = 2022  # exp_004 mask: drop season<=2022 & game_type=="F"

# exp_001 / exp_007 / exp_008 valid-row counts for fold_2024 (identity check)
EXPECTED_2024 = {"valid_all": 253507, "valid_R": 223497}

BASE_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.02,
    "num_leaves": 31,
    "min_data_in_leaf": 2000,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": SEED,
    "num_threads": N_THREADS,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}
D_PARAMS = dict(
    BASE_PARAMS, learning_rate=0.01, num_leaves=15, min_data_in_leaf=5000
)

N_ROUNDS_ABC = 2000
N_ROUNDS_D = 3000

# SMOKE=1 runs the identical code path with tiny round counts to validate the
# plumbing before the real launch. It never produces reportable numbers.
SMOKE = os.environ.get("EXP010_SMOKE") == "1"
if SMOKE:
    N_ROUNDS_ABC, N_ROUNDS_D = 20, 25
    OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_010_smoke")

TE_K_GRID = [50, 200, 500]
TE_K_DEFAULT = 200  # replaced by the fold_2023-tuned value before reporting

# reference numbers this experiment is measured against (exp_008 / exp_007)
REFERENCES = {
    "rf_R1_fold2024_valid_R_brier": 0.248487,
    "rf_R1_fold2024_valid_R_auc": 0.5434,
    # PRIMARY as of 2026-08-05: test.csv is now known to contain F-league rows
    # (245,789 test rows sit inside the season-total R+F range 237k-253k and are
    # ~10% above the R-only range 211k-223k; KBO's 720-game regular season makes
    # a 10% R-only increase impossible), so the target population is R+F.
    "rf_R1_fold2024_valid_all_brier": 0.248769,
    "rf_R1_fold2024_valid_all_score": 415.6,
    "rf_R2_mask_fold2024_valid_all_score": 551.1,
    "rf_R1_actual_LB": 549.51,
    "target_line_1061_points_bss_valid_all": 0.01061,
    "rf_R2_mask_fold2024_valid_R_brier": 0.248455,
    "exp_007_V1_fold2024_valid_R_brier": 0.249061,
    "exp_001_fold2024_valid_R_brier": 0.249337,
    "target_line_1061_points_brier": 0.24724,
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def sha256_file(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def lb_score(brier, r):
    denom = r * (1.0 - r)
    if denom <= 0:
        return None
    return float(max(0.0, 100000.0 * (1.0 - brier / denom)))


def seg_metrics(y, p):
    """Identical definitions to exp_001 / exp_004 / exp_008."""
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
        "score": round(lb_score(brier, r), 3) if denom > 0 else None,
        "base_rate": round(r, 6),
        "pred_mean": round(float(np.mean(p)), 6),
    }
    if 0.0 < r < 1.0:
        out["auc"] = round(float(roc_auc_score(y, p)), 6)
        out["logloss"] = round(float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15))), 6)
    return out


def curve_grid(n_max):
    g = list(range(1, 21))
    g += list(range(25, 101, 5))
    g += list(range(110, 301, 10))
    g += list(range(325, min(n_max, 1000) + 1, 25))
    if n_max > 1000:
        g += list(range(1050, n_max + 1, 50))
    g = sorted(set([k for k in g if k <= n_max] + [n_max]))
    return g


def raw_score_curve(booster, X, grid):
    """Yield (k, raw_margin_at_k) by accumulating chunks of trees.

    A fixed-seed booster's first k trees ARE the k-round model, and raw margins
    are additive across tree ranges. LightGBM may or may not fold the
    boost-from-average constant into every predicted range, so the constant is
    measured once and removed from every chunk after the first.
    """
    sub = X.iloc[: min(2000, len(X))]
    a = booster.predict(sub, start_iteration=0, num_iteration=5, raw_score=True)
    b = booster.predict(sub, start_iteration=5, num_iteration=5, raw_score=True)
    full = booster.predict(sub, start_iteration=0, num_iteration=10, raw_score=True)
    resid = full - (a + b)
    const = float(np.median(resid))
    assert np.allclose(resid, const, atol=1e-8), (
        "raw margins are not additive across tree ranges",
        float(np.abs(resid - const).max()),
    )

    acc = np.zeros(len(X), dtype=np.float64)
    prev = 0
    for k in grid:
        acc = acc + booster.predict(
            X, start_iteration=prev, num_iteration=k - prev, raw_score=True
        )
        if prev > 0:
            acc = acc + const
        prev = k
        yield k, acc


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _expanding_prev_sum(keys, values):
    """Per-key expanding sum of `values` EXCLUDING the current row (shift(1)),
    in the given (file) row order. Returns (prev_sum, prev_count)."""
    s = pd.Series(values, dtype="float64")
    g = s.groupby(keys, sort=False)
    prev_sum = (g.cumsum() - s).to_numpy()
    prev_cnt = g.cumcount().to_numpy().astype("float64")
    return prev_sum, prev_cnt


def build_target_encoding(df):
    """Causal (file-order expanding, shift(1)) target encoding inputs.

    The per-row (prev_success, prev_n) pair is taken from the ORGANIZERS'
    as-of columns via `rate * n`, because that is the exact form available at
    test time — training and inference then compute the identical quantity.
    Our own expanding shift(1) recomputation from the labels is run alongside
    purely as a verification of the EDA §2 identity, and any mismatch is
    reported loudly rather than silently absorbed.
    """
    y = df[TARGET].to_numpy(dtype="float64")
    n_rows = len(df)

    # expanding league mean, shift(1)
    league_prev_sum = np.cumsum(y) - y
    idx = np.arange(n_rows, dtype="float64")
    prior = np.divide(
        league_prev_sum, idx, out=np.full(n_rows, np.nan), where=idx > 0
    )
    prior[0] = 0.5  # first row of the file has no history
    prior = pd.Series(prior).ffill().to_numpy()

    out = {"prior": prior, "prior_final": float(y.mean())}
    checks = {}
    for who in ("pitcher", "batter"):
        # --- test-reproducible form (from provided columns only) -----------
        given_n = df[f"asof_{who}_n"].to_numpy(dtype="float64")
        given_rate = df[f"asof_{who}_success_rate"].to_numpy(dtype="float64")
        prev_n = np.nan_to_num(given_n, nan=0.0)
        prev_sum = np.nan_to_num(given_rate, nan=0.0) * prev_n
        out[f"{who}_prev_sum"] = prev_sum
        out[f"{who}_prev_n"] = prev_n

        # --- verification: our own expanding shift(1) over file order ------
        ps, pc = _expanding_prev_sum(df[f"{who}_id"].to_numpy(), y)
        ok = (~np.isnan(given_rate)) & (given_n > 0)
        checks[f"{who}_n_max_abs_diff"] = float(np.abs(prev_n - pc).max())
        checks[f"{who}_sum_max_abs_diff"] = float(np.abs(prev_sum[ok] - ps[ok]).max())
        checks[f"{who}_n_rows_checked"] = int(ok.sum())
        checks[f"{who}_n_cold_start_rows"] = int((~ok).sum())
        if checks[f"{who}_n_max_abs_diff"] != 0.0 or checks[f"{who}_sum_max_abs_diff"] > 1e-2:
            print(f"  !! WARNING: as-of identity check FAILED for {who}: "
                  f"{json.dumps(checks)}", flush=True)
    out["checks"] = checks
    return out


def te_columns(te, k):
    """Shrunk causal target encodings at smoothing strength k.

    Test-time equivalent (per row, from test.csv columns only):
        prev_n   = asof_{who}_n                       (0 for a debutant)
        prev_sum = asof_{who}_success_rate * prev_n   (0 for a debutant)
        te       = (prev_sum + k * prior) / (prev_n + k),  prior frozen at the
                   end-of-2024 league mean -> a debutant gets exactly `prior`.
    """
    cols = {}
    for who in ("pitcher", "batter"):
        cols[f"te_{who}_success_k{k}"] = (
            te[f"{who}_prev_sum"] + k * te["prior"]
        ) / (te[f"{who}_prev_n"] + k)
    return cols


# --------------------------------------------------------------------------- #
# fold training
# --------------------------------------------------------------------------- #
def train_fold(df, feat_cols, cat_cols, params, n_rounds, fold_name,
               train_until, valid_year, f_excl, tag):
    tr_mask = (df["season"] <= train_until) & ~f_excl
    va_mask = df["season"] == valid_year

    X_tr = df.loc[tr_mask, feat_cols]
    y_tr = df.loc[tr_mask, TARGET].to_numpy()
    X_va = df.loc[va_mask, feat_cols]
    y_va = df.loc[va_mask, TARGET].to_numpy()
    is_r = (df.loc[va_mask, "game_type"] == "R").to_numpy()

    print(
        f"[{time.strftime('%H:%M:%S')}] {tag}/{fold_name}: train {X_tr.shape[0]:,} "
        f"x {X_tr.shape[1]} -> valid {X_va.shape[0]:,} ({valid_year}), "
        f"{n_rounds} rounds no-ES",
        flush=True,
    )
    if valid_year == 2024:
        assert int(va_mask.sum()) == EXPECTED_2024["valid_all"], int(va_mask.sum())
        assert int(is_r.sum()) == EXPECTED_2024["valid_R"], int(is_r.sum())

    t0 = time.time()
    dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols)
    booster = lgb.train(params, dtr, num_boost_round=n_rounds)
    train_sec = time.time() - t0

    grid = curve_grid(n_rounds)
    t1 = time.time()
    curve_r, curve_all, margins = {}, {}, {}
    for k, raw in raw_score_curve(booster, X_va, grid):
        p = sigmoid(raw)
        yr, pr = y_va[is_r], p[is_r]
        curve_r[k] = float(np.mean((pr - yr) ** 2))
        curve_all[k] = float(np.mean((p - y_va) ** 2))
        margins[k] = raw.copy()
    curve_sec = time.time() - t1

    argmin_it = min(curve_r, key=lambda k: curve_r[k])
    argmin_all_it = min(curve_all, key=lambda k: curve_all[k])
    print(
        f"  trained {train_sec/60:.2f} min, curve {curve_sec:.0f}s; "
        f"valid-R Brier argmin iter={argmin_it} -> {curve_r[argmin_it]:.6f}; "
        f"valid-ALL Brier argmin iter={argmin_all_it} -> {curve_all[argmin_all_it]:.6f}",
        flush=True,
    )
    return {
        "fold_name": fold_name,
        "train_until": train_until,
        "valid_year": valid_year,
        "n_train": int(tr_mask.sum()),
        "n_features": len(feat_cols),
        "train_time_min": round(train_sec / 60, 2),
        "curve_time_sec": round(curve_sec, 1),
        "curve_valid_R_brier": {str(k): round(v, 6) for k, v in curve_r.items()},
        "curve_valid_all_brier": {str(k): round(v, 6) for k, v in curve_all.items()},
        "argmin_iter": int(argmin_it),
        "argmin_brier": round(curve_r[argmin_it], 6),
        "argmin_iter_all": int(argmin_all_it),
        "argmin_brier_all": round(curve_all[argmin_all_it], 6),
        "_margins": margins,
        "_grid": grid,
        "_y_va": y_va,
        "_is_r": is_r,
        "_booster": booster,
        "_tr_mask": tr_mask,
        "_va_mask": va_mask,
    }


def nearest_grid_iter(grid, k):
    return min(grid, key=lambda g: (abs(g - k), g))


def report_at(fold, iter_used, iter_source):
    grid = fold["_grid"]
    k = nearest_grid_iter(grid, iter_used)
    p = sigmoid(fold["_margins"][k])
    y_va, is_r = fold["_y_va"], fold["_is_r"]
    out = {
        "iter_used": int(k),
        "iter_requested": int(iter_used),
        "iter_source": iter_source,
        "n_train": fold["n_train"],
        "valid_all": seg_metrics(y_va, p),
        "valid_R": seg_metrics(y_va[is_r], p[is_r]),
        "valid_F": seg_metrics(y_va[~is_r], p[~is_r]),
        "oracle_iter": fold["argmin_iter"],
        "oracle_brier_valid_R": fold["argmin_brier"],
        "policy_cost_vs_oracle": round(
            fold["curve_valid_R_brier"][str(k)] - fold["argmin_brier"], 6
        ),
        "oracle_iter_all": fold["argmin_iter_all"],
        "oracle_brier_valid_all": fold["argmin_brier_all"],
        "policy_cost_vs_oracle_all": round(
            fold["curve_valid_all_brier"][str(k)] - fold["argmin_brier_all"], 6
        ),
        "train_time_min": fold["train_time_min"],
    }
    return out, p


def calibrate_c(df, fold, p_va, iter_used):
    """exp_005 candidate C: train-only constant logit offset.

    r_hat  = linear trend of the R-league season success rate over the fold's
             TRAINING seasons, extrapolated one year forward.
    m      = this model's mean prediction on the previous season's R TRAINING
             rows (never on valid).
    delta  = logit(r_hat) - logit(m), applied as a per-row constant shift.
    """
    train_until = fold["train_until"]
    seasons = sorted(df.loc[df["season"] <= train_until, "season"].unique())
    rates = [
        float(df.loc[(df["season"] == s) & (df["game_type"] == "R"), TARGET].mean())
        for s in seasons
    ]
    coef = np.polyfit(np.array(seasons, dtype=float), np.array(rates), 1)
    r_hat = float(np.polyval(coef, fold["valid_year"]))

    anchor_mask = (
        (df["season"] == train_until) & (df["game_type"] == "R") & fold["_tr_mask"]
    )
    Xa = df.loc[anchor_mask, fold["_feat_cols"]]
    pa = fold["_booster"].predict(Xa, num_iteration=iter_used)
    m = float(np.mean(pa))

    eps = 1e-7
    lg = lambda v: np.log(np.clip(v, eps, 1 - eps) / (1 - np.clip(v, eps, 1 - eps)))
    delta = float(lg(r_hat) - lg(m))
    p_cal = sigmoid(lg(p_va) + delta)
    return {
        "train_seasons": [int(s) for s in seasons],
        "train_R_rates": [round(v, 6) for v in rates],
        "r_hat": round(r_hat, 6),
        "anchor_pred_mean_on_train_prev_season_R": round(m, 6),
        "delta_logit": round(delta, 6),
    }, p_cal


# --------------------------------------------------------------------------- #
def run_variant(df, name, spec, f_excl, metrics):
    feat_cols = spec["feat_cols"]
    cat_cols = [c for c in CAT_COLS if c in feat_cols]
    params = spec["params"]
    n_rounds = spec["n_rounds"]

    folds = {}
    for fold_name, tu, vy in FOLDS:
        fd = train_fold(df, feat_cols, cat_cols, params, n_rounds, fold_name,
                        tu, vy, f_excl, name)
        fd["_feat_cols"] = feat_cols
        folds[fold_name] = fd

    # ---- exp_006 P4: carry the PREVIOUS fold's valid-R argmin ---------------
    out = {
        "spec": spec["desc"],
        "n_features": len(feat_cols),
        "features": feat_cols,
        "categorical": cat_cols,
        "lgb_params": params,
        "num_boost_round_cap": n_rounds,
        "iter_policy": "exp_006 P4 (previous fold's valid-R Brier argmin, no scaling)",
    }
    prev_argmin = None
    fold2024_preds = None
    for fold_name, _, _ in FOLDS:
        fd = folds[fold_name]
        if prev_argmin is None:
            rep, p = report_at(fd, fd["argmin_iter"], "own argmin (no predecessor; ORACLE, feeder only)")
            rep["oracle_flag"] = True
        else:
            rep, p = report_at(fd, prev_argmin[1], f"P4 from {prev_argmin[0]} argmin")
            rep["oracle_flag"] = False
        out[fold_name] = rep
        prev_argmin = (fold_name, fd["argmin_iter"])
        if fold_name == "fold_2024":
            fold2024_preds = p

    # ---- P4 measured on valid_ALL (primary metric as of the 2026-08-05 change:
    #      test.csv is now known to contain F-league rows, so R+F is the target
    #      population). Same boosters, same seeds — only the iteration a fold is
    #      REPORTED at changes: it is carried from the PREVIOUS fold's valid_all
    #      Brier argmin instead of its valid-R argmin. Still never chosen on the
    #      fold being reported. ------------------------------------------------
    prev_all = None
    for fold_name, _, _ in FOLDS:
        fd = folds[fold_name]
        if prev_all is None:
            rep_a, _ = report_at(fd, fd["argmin_iter_all"],
                                 "own valid_all argmin (no predecessor; ORACLE, feeder only)")
            rep_a["oracle_flag"] = True
        else:
            rep_a, _ = report_at(fd, prev_all[1], f"P4-all from {prev_all[0]} valid_all argmin")
            rep_a["oracle_flag"] = False
        out[fold_name + "_P4all"] = rep_a
        prev_all = (fold_name, fd["argmin_iter_all"])

    # ---- persist the full iteration curves so any future re-ranking (e.g. a
    #      further change of primary metric) needs no retraining ---------------
    out["curves"] = {
        fn: {
            "grid": folds[fn]["_grid"],
            "valid_R_brier": folds[fn]["curve_valid_R_brier"],
            "valid_all_brier": folds[fn]["curve_valid_all_brier"],
        }
        for fn, _, _ in FOLDS
    }

    # ---- secondary: exp_005-C constant offset on fold_2024 ------------------
    fd24 = folds["fold_2024"]
    cal_info, p_cal = calibrate_c(df, fd24, fold2024_preds, out["fold_2024"]["iter_used"])
    y_va, is_r = fd24["_y_va"], fd24["_is_r"]
    out["fold_2024_calibrated_exp005C"] = {
        "note": "SECONDARY — post-hoc constant logit offset, fit on training rows only. "
                "The headline comparison against the RF baseline uses the RAW numbers.",
        "calibration": cal_info,
        "valid_R": seg_metrics(y_va[is_r], p_cal[is_r]),
        "valid_all": seg_metrics(y_va, p_cal),
    }

    # ---- gain importance at the reported fold_2024 iter ---------------------
    booster = fd24["_booster"]
    imp = sorted(zip(feat_cols, booster.feature_importance("gain",
                                                          iteration=out["fold_2024"]["iter_used"])),
                 key=lambda t: -t[1])
    tot = sum(g for _, g in imp) or 1.0
    out["fold_2024_gain_top20"] = [
        [f, round(float(g), 1), round(100.0 * g / tot, 2)] for f, g in imp[:20]
    ]
    out["fold_2024_gain_share_ids_pct"] = round(
        100.0 * sum(g for f, g in imp if f in ID_COLS or f.startswith("te_")) / tot, 2
    )

    np.save(os.path.join(OUT_DIR, f"pred_{name}_fold_2024.npy"), fold2024_preds)
    booster.save_model(
        os.path.join(OUT_DIR, f"model_{name}_fold_2024.txt"),
        num_iteration=out["fold_2024"]["iter_used"],
    )
    metrics[name] = out
    print(json.dumps({name: {k: v for k, v in out.items()
                             if k.startswith("fold") or k == "spec"}},
                     indent=2, ensure_ascii=False), flush=True)
    # free boosters / margins
    for fd in folds.values():
        fd.pop("_margins", None)
        fd.pop("_booster", None)
    return out


def main():
    np.random.seed(SEED)
    t_start = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] loading {DATA_PATH}", flush=True)
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    print(f"loaded: {df.shape}", flush=True)
    assert df["row_id"].is_unique
    assert df["season"].is_monotonic_increasing, "file order assumption broken"

    base_feats = [c for c in df.columns if c not in DROP_COLS]
    assert len(base_feats) == 47, len(base_feats)

    # causal target encoding inputs (file order, expanding, shift(1))
    print(f"[{time.strftime('%H:%M:%S')}] building causal target encodings", flush=True)
    te = build_target_encoding(df)
    print("  as-of identity check: " + json.dumps(te["checks"]), flush=True)
    for k in TE_K_GRID:
        for cname, vals in te_columns(te, k).items():
            df[cname] = vals

    for c in CAT_COLS:
        df[c] = df[c].astype("category")

    f_excl = (df["season"] <= F_EXCLUDE_MAX_SEASON) & (df["game_type"] == "F")
    print(f"exp_004 mask: {int(f_excl.sum()):,} rows excluded from training", flush=True)

    metrics = {
        "exp_id": "exp_010",
        "base_exp": "exp_008 R1 (organizers' RF) — target: beat valid-R Brier 0.248487",
        "seed": SEED,
        "num_threads": N_THREADS,
        "mask": "training rows with season<=2022 & game_type=='F' dropped (exp_004), valid untouched",
        "n_rows_masked": int(f_excl.sum()),
        "references": REFERENCES,
        "te_identity_check": te["checks"],
        "te_prior_final_end_of_2024": round(te["prior_final"], 6),
        "te_prior_note": "train uses the file-order expanding shift(1) league mean; "
                         "test inference freezes it at the end-of-2024 value above",
        "sha256": {
            "train.csv": sha256_file(DATA_PATH),
            "train.py": sha256_file(os.path.abspath(__file__)),
        },
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "lightgbm": lgb.__version__,
        },
    }

    feats_A = list(base_feats)
    feats_B = [c for c in base_feats if c not in ID_COLS]

    # ---- C: pick the smoothing strength k on fold_2023 ONLY ----------------
    print(f"[{time.strftime('%H:%M:%S')}] === C: tuning target-encoding k on fold_2023 ===",
          flush=True)
    k_tune = {}
    for k in TE_K_GRID:
        feats = base_feats + [f"te_pitcher_success_k{k}", f"te_batter_success_k{k}"]
        fd = train_fold(df, feats, CAT_COLS, BASE_PARAMS, N_ROUNDS_ABC,
                        "fold_2023", 2022, 2023, f_excl, f"C_tune_k{k}")
        k_tune[k] = {"argmin_iter": fd["argmin_iter"], "argmin_brier": fd["argmin_brier"]}
        del fd
    k_best = min(k_tune, key=lambda k: k_tune[k]["argmin_brier"])
    metrics["C_k_tuning_on_fold_2023"] = {
        "grid": {str(k): v for k, v in k_tune.items()},
        "k_selected": k_best,
        "note": "selected on fold_2023 valid-R Brier only; fold_2024 never consulted",
    }
    print(f"  k selected on fold_2023: {k_best}", flush=True)
    feats_C = base_feats + [f"te_pitcher_success_k{k_best}", f"te_batter_success_k{k_best}"]

    variants = {
        "A": {
            "desc": "IDs numeric (not categorical), num_leaves 31, min_data 2000, lr 0.02, ff 0.7",
            "feat_cols": feats_A, "params": BASE_PARAMS, "n_rounds": N_ROUNDS_ABC,
        },
        "B": {
            "desc": "A minus pitcher_id/batter_id (no player identity columns)",
            "feat_cols": feats_B, "params": BASE_PARAMS, "n_rounds": N_ROUNDS_ABC,
        },
        "C": {
            "desc": f"A + causal target encoding of pitcher_id/batter_id (k={k_best}, "
                    f"file-order expanding shift(1), league-prior smoothing)",
            "feat_cols": feats_C, "params": BASE_PARAMS, "n_rounds": N_ROUNDS_ABC,
        },
        "D": {
            "desc": "A features, strongest regularisation: num_leaves 15, min_data 5000, lr 0.01, cap 3000",
            "feat_cols": feats_A, "params": D_PARAMS, "n_rounds": N_ROUNDS_D,
        },
    }

    for name, spec in variants.items():
        print(f"[{time.strftime('%H:%M:%S')}] ===== variant {name} =====", flush=True)
        run_variant(df, name, spec, f_excl, metrics)
        metrics["train_time_min"] = round((time.time() - t_start) / 60, 2)
        with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

    # ---- summary -----------------------------------------------------------
    rf = REFERENCES["rf_R1_fold2024_valid_R_brier"]
    rf_all = REFERENCES["rf_R1_fold2024_valid_all_brier"]
    summary = {}
    for name in variants:
        m = metrics[name]["fold_2024"]["valid_R"]
        c = metrics[name]["fold_2024_calibrated_exp005C"]["valid_R"]
        summary[name] = {
            "iter": metrics[name]["fold_2024"]["iter_used"],
            "brier": m["brier"], "bss": m["bss"], "auc": m["auc"], "score": m["score"],
            "delta_vs_rf_R1": round(m["brier"] - rf, 6),
            "beats_rf_R1": bool(m["brier"] < rf),
            "brier_calibrated": c["brier"],
            "delta_calibrated_vs_rf_R1": round(c["brier"] - rf, 6),
            "beats_rf_R1_calibrated": bool(c["brier"] < rf),
        }
    metrics["summary_fold_2024_valid_R"] = summary

    # PRIMARY table (metric change 2026-08-05: test contains F rows -> R+F)
    summary_all = {}
    for name in variants:
        for key, tag in (("fold_2024", "P4-R"), ("fold_2024_P4all", "P4-all")):
            a = metrics[name][key]["valid_all"]
            fseg = metrics[name][key]["valid_F"]
            rseg = metrics[name][key]["valid_R"]
            summary_all[f"{name}_{tag}"] = {
                "iter": metrics[name][key]["iter_used"],
                "iter_source": metrics[name][key]["iter_source"],
                "valid_all_brier": a["brier"], "valid_all_bss": a["bss"],
                "valid_all_score": a["score"], "valid_all_auc": a.get("auc"),
                "delta_vs_rf_R1_all": round(a["brier"] - rf_all, 6),
                "beats_rf_R1_all": bool(a["brier"] < rf_all),
                "valid_R_brier": rseg["brier"],
                "valid_F_brier": fseg["brier"],
                "valid_F_score": fseg["score"],
                "valid_F_pred_mean": fseg["pred_mean"],
                "valid_F_base_rate": fseg["base_rate"],
                "valid_F_calib_gap_pp": round(
                    100.0 * (fseg["pred_mean"] - fseg["base_rate"]), 3),
            }
    metrics["summary_fold_2024_valid_ALL_PRIMARY"] = summary_all
    metrics["train_time_min"] = round((time.time() - t_start) / 60, 2)
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print("--- SECONDARY: fold_2024 valid_R ---", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print("--- PRIMARY: fold_2024 valid_all (R+F) ---", flush=True)
    print(json.dumps(summary_all, indent=2), flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] DONE total {metrics['train_time_min']} min",
          flush=True)


if __name__ == "__main__":
    main()
