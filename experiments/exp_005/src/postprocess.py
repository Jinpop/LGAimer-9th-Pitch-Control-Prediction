"""exp_005 — post-hoc calibration of exp_001 baseline (no training).

Spec: experiments/exp_005/config.yaml (2026-08-05, campaign 2)
Hypothesis: the +1.7pp over-prediction of the exp_001 baseline on fold_2024
valid-R (base-rate downtrend not captured by the model) can be removed by a
post-processing correction, improving fold_2024 valid-R Brier.

Candidates (all fit WITHOUT touching fold_2024 valid labels/preds):
  A. constant logit offset  — delta fit by Brier minimisation on fold_2023
     valid-R (preds of the fold_2023 model), applied to fold_2024 preds.
  B. isotonic regression    — fit on fold_2023 valid-R (p -> y), applied to
     fold_2024 preds (out_of_bounds="clip").
  C. base-rate trend anchor — linear trend of R-league yearly success rate
     over seasons 2019..2023 (all fold_2024 TRAIN years) extrapolated to
     2024 -> r_hat; anchor mean m = fold_2024 model's predicted mean on
     season==2023 R rows (TRAIN rows); delta_C = logit(r_hat) - logit(m).
     Per-row constant shift => no reference to valid/test prediction
     distribution (rule 2-4 safe, reproducible for 2025 inference).

Reference only (clearly NOT a candidate metric): oracle offset fit in-sample
on fold_2024 valid-R, to bound the headroom of any constant-offset scheme.

Integrity rules honoured:
  - no calibration parameter is fit on fold_2024 valid (candidates A/B/C);
  - evaluation identical to exp_001 conventions (Brier, BSS with r = actual
    valid rate, AUC), primary segment = fold_2024 valid-R.

Runtime: A5000 CPU, prediction only, num_threads=4 (other experiments are
running in parallel — keep CPU load minimal).
"""

import hashlib
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

SEED = 42
DATA_PATH = os.path.expanduser("~/SungjinPark/data/train.csv")
EXP001_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_001")
OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_005")

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
N_THREADS = 4
EPS = 1e-7

# expected exp_001 numbers for reproduction check (from exp_001/metrics.json)
EXPECTED = {
    "fold_2023_valid_R_brier": 0.250527,
    "fold_2024_valid_R_brier": 0.249337,
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
    """Same conventions as exp_001: Brier / BSS(r=actual) / AUC / means."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    n = int(len(y))
    brier = float(np.mean((p - y) ** 2))
    r = float(np.mean(y))
    denom = r * (1.0 - r)
    out = {
        "n": n,
        "brier": round(brier, 6),
        "bss": round(float(1.0 - brier / denom), 6),
        "base_rate": round(r, 6),
        "pred_mean": round(float(np.mean(p)), 6),
        "auc": round(float(roc_auc_score(y, p)), 6),
    }
    return out


def calib_deciles(y, p):
    """10 equal-frequency bins by predicted prob: mean pred vs mean actual."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    qs = np.quantile(p, np.linspace(0, 1, 11))
    qs[0], qs[-1] = -np.inf, np.inf
    idx = np.searchsorted(qs, p, side="right") - 1
    idx = np.clip(idx, 0, 9)
    rows = []
    for b in range(10):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append(
            {
                "bin": b,
                "n": int(m.sum()),
                "pred_mean": round(float(p[m].mean()), 5),
                "actual_rate": round(float(y[m].mean()), 5),
            }
        )
    return rows


def apply_offset(p, delta):
    return expit(logit(np.clip(p, EPS, 1 - EPS)) + delta)


def fit_offset_brier(y, p):
    """delta minimising Brier of expit(logit(p)+delta)."""
    y = np.asarray(y, dtype=float)
    lo = logit(np.clip(p, EPS, 1 - EPS))

    def obj(d):
        return float(np.mean((expit(lo + d) - y) ** 2))

    res = minimize_scalar(obj, bounds=(-1.0, 1.0), method="bounded",
                          options={"xatol": 1e-8})
    return float(res.x), float(res.fun)


def main():
    np.random.seed(SEED)
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] loading {DATA_PATH}", flush=True)
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    print(f"loaded: {df.shape}", flush=True)
    assert df["row_id"].is_unique

    # identical preprocessing to exp_001/train.py (category codes over full file)
    feat_cols = [c for c in df.columns if c not in DROP_COLS]
    for c in CAT_COLS:
        df[c] = df[c].astype("category")

    metrics = {
        "exp_id": "exp_005",
        "base_exp": "exp_001",
        "seed": SEED,
        "task": "postprocess (no training; exp_001 saved models reused)",
        "num_threads": N_THREADS,
        "sha256": {
            "train.csv": sha256_file(DATA_PATH),
            "postprocess.py": sha256_file(os.path.abspath(__file__)),
            "model_fold_2023.txt": sha256_file(
                os.path.join(EXP001_DIR, "model_fold_2023.txt")),
            "model_fold_2024.txt": sha256_file(
                os.path.join(EXP001_DIR, "model_fold_2024.txt")),
        },
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "lightgbm": lgb.__version__,
        },
    }

    # ---- regenerate valid predictions from saved exp_001 models -------------
    preds = {}
    t_inf0 = time.time()
    for fold, valid_year in (("fold_2023", 2023), ("fold_2024", 2024)):
        booster = lgb.Booster(
            model_file=os.path.join(EXP001_DIR, f"model_{fold}.txt"))
        va_mask = df["season"] == valid_year
        X_va = df.loc[va_mask, feat_cols]
        p = booster.predict(X_va, num_threads=N_THREADS)
        y = df.loc[va_mask, TARGET].to_numpy(dtype=float)
        is_r = (df.loc[va_mask, "game_type"] == "R").to_numpy()
        preds[fold] = {"p": p, "y": y, "is_r": is_r}
        print(f"[{time.strftime('%H:%M:%S')}] {fold}: predicted "
              f"{int(va_mask.sum()):,} rows ({valid_year})", flush=True)
    inference_sec = time.time() - t_inf0

    # reproduction check vs exp_001 metrics.json
    repro = {}
    for fold in ("fold_2023", "fold_2024"):
        d = preds[fold]
        b = float(np.mean((d["p"][d["is_r"]] - d["y"][d["is_r"]]) ** 2))
        exp_key = f"{fold}_valid_R_brier"
        repro[exp_key] = {
            "regenerated": round(b, 6),
            "exp_001": EXPECTED[exp_key],
            "match": bool(abs(b - EXPECTED[exp_key]) < 5e-6),
        }
    metrics["reproduction_check"] = repro
    print("reproduction_check:", json.dumps(repro), flush=True)
    if not all(v["match"] for v in repro.values()):
        print("WARNING: regenerated predictions do not match exp_001 metrics",
              flush=True)

    # primary segment arrays
    y23, p23 = (preds["fold_2023"]["y"][preds["fold_2023"]["is_r"]],
                preds["fold_2023"]["p"][preds["fold_2023"]["is_r"]])
    y24, p24 = (preds["fold_2024"]["y"][preds["fold_2024"]["is_r"]],
                preds["fold_2024"]["p"][preds["fold_2024"]["is_r"]])

    results = {"original": seg_metrics(y24, p24)}
    calib = {"original": calib_deciles(y24, p24)}

    # ---- A. constant logit offset (fit fold_2023 valid-R) -------------------
    delta_a, brier23_a = fit_offset_brier(y23, p23)
    p24_a = apply_offset(p24, delta_a)
    results["A_logit_offset"] = seg_metrics(y24, p24_a)
    results["A_logit_offset"]["fit"] = {
        "fit_on": "fold_2023 valid-R (fold_2023 model preds)",
        "delta": round(delta_a, 6),
        "fold_2023_valid_R_brier_after_fit_insample": round(brier23_a, 6),
        "fold_2023_valid_R_brier_before": round(
            float(np.mean((p23 - y23) ** 2)), 6),
    }
    calib["A_logit_offset"] = calib_deciles(y24, p24_a)

    # ---- B. isotonic regression (fit fold_2023 valid-R) ---------------------
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(p23, y23)
    p24_b = iso.predict(p24)
    results["B_isotonic"] = seg_metrics(y24, p24_b)
    results["B_isotonic"]["fit"] = {
        "fit_on": "fold_2023 valid-R (fold_2023 model preds)",
        "n_thresholds": int(len(iso.X_thresholds_)),
        "fold_2023_valid_R_brier_after_fit_insample": round(
            float(np.mean((iso.predict(p23) - y23) ** 2)), 6),
    }
    calib["B_isotonic"] = calib_deciles(y24, p24_b)

    # ---- C. base-rate trend anchor (train-side only, valid untouched) -------
    tr_mask_2024 = df["season"] <= 2023  # fold_2024 train years
    r_hist = (
        df.loc[tr_mask_2024 & (df["game_type"] == "R")]
        .groupby("season")[TARGET].mean()
    )
    seasons = r_hist.index.to_numpy(dtype=float)
    rates = r_hist.to_numpy(dtype=float)
    slope, intercept = np.polyfit(seasons, rates, 1)
    r_hat_2024 = float(slope * 2024 + intercept)
    # anchor: fold_2024 model's predicted mean on season==2023 R rows (TRAIN)
    anchor_mask = (df["season"] == 2023) & (df["game_type"] == "R")
    booster24 = lgb.Booster(
        model_file=os.path.join(EXP001_DIR, "model_fold_2024.txt"))
    p_anchor = booster24.predict(df.loc[anchor_mask, feat_cols],
                                 num_threads=N_THREADS)
    m_anchor = float(np.mean(p_anchor))
    delta_c = float(logit(r_hat_2024) - logit(m_anchor))
    p24_c = apply_offset(p24, delta_c)
    results["C_trend_anchor"] = seg_metrics(y24, p24_c)
    results["C_trend_anchor"]["fit"] = {
        "fit_on": "train-only: R yearly rates 2019..2023 + fold_2024 model "
                  "pred mean on season==2023 R train rows",
        "r_history": {int(s): round(float(r), 6)
                      for s, r in zip(seasons, rates)},
        "trend_slope_per_year": round(float(slope), 6),
        "r_hat_2024": round(r_hat_2024, 6),
        "anchor_pred_mean_2023R_train": round(m_anchor, 6),
        "delta": round(delta_c, 6),
    }
    calib["C_trend_anchor"] = calib_deciles(y24, p24_c)

    # ---- reference: oracle offset (fit IN-SAMPLE on fold_2024 valid-R) -----
    # NOT a candidate — optimistic upper bound for any constant-offset scheme.
    delta_o, _ = fit_offset_brier(y24, p24)
    p24_o = apply_offset(p24, delta_o)
    results["reference_oracle_offset_IN_SAMPLE"] = seg_metrics(y24, p24_o)
    results["reference_oracle_offset_IN_SAMPLE"]["fit"] = {
        "fit_on": "fold_2024 valid-R ITSELF — in-sample upper bound, "
                  "not a valid candidate metric",
        "delta": round(delta_o, 6),
    }

    metrics["fold_2024_valid_R"] = results
    metrics["calibration_deciles_fold_2024_valid_R"] = calib
    metrics["train_time_min"] = 0.0
    metrics["inference_time_sec"] = round(inference_sec, 2)
    metrics["total_time_min"] = round((time.time() - t0) / 60, 2)

    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "fit"}
                      for k, v in results.items()}, indent=2), flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] DONE "
          f"{metrics['total_time_min']} min", flush=True)


if __name__ == "__main__":
    main()
