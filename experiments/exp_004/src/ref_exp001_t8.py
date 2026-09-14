"""Matched-thread reference: exp_001 config (no F exclusion), fold_2024 only,
num_threads=8.

Purpose: exp_001 official metrics were produced with num_threads=24. exp_004
runs with num_threads=8 (shared server, parallel campaign). LightGBM results
can shift slightly with thread count, so this run re-produces the exp_001
fold_2024 numbers under identical thread conditions to exp_004 — the fair
control for delta reporting. Everything else (data, features, params, seed,
fold definition, category encoding) is byte-identical to exp_001/src/train.py.

Output: ~/SungjinPark/experiments/exp_004/metrics_ref_exp001_t8.json
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
DATA_PATH = os.path.expanduser("~/SungjinPark/data/train.csv")
OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_004")

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
NUM_BOOST_ROUND = 5000
EARLY_STOPPING_ROUNDS = 200
TRAIN_UNTIL, VALID_YEAR = 2023, 2024


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


def main():
    np.random.seed(SEED)
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] REF exp_001@t8 loading {DATA_PATH}", flush=True)
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    assert df["row_id"].is_unique

    feat_cols = [c for c in df.columns if c not in DROP_COLS]
    for c in CAT_COLS:
        df[c] = df[c].astype("category")

    tr_mask = df["season"] <= TRAIN_UNTIL
    va_mask = df["season"] == VALID_YEAR
    X_tr = df.loc[tr_mask, feat_cols]
    y_tr = df.loc[tr_mask, TARGET].to_numpy()
    X_va = df.loc[va_mask, feat_cols]
    y_va = df.loc[va_mask, TARGET].to_numpy()
    print(f"fold_2024: train {X_tr.shape[0]:,} -> valid {X_va.shape[0]:,}", flush=True)

    dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=CAT_COLS)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr, categorical_feature=CAT_COLS)
    booster = lgb.train(
        LGB_PARAMS,
        dtr,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dva],
        valid_names=["valid"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True),
            lgb.log_evaluation(100),
        ],
    )
    p_va = booster.predict(X_va, num_iteration=booster.best_iteration)

    is_r = (df.loc[va_mask, "game_type"] == "R").to_numpy()
    seen_pitchers = set(df.loc[tr_mask, "pitcher_id"].unique())
    cold = ~df.loc[va_mask, "pitcher_id"].isin(seen_pitchers).to_numpy()

    metrics = {
        "ref_of": "exp_001",
        "note": "exp_001 config re-run with num_threads=8, fold_2024 only (matched-thread control for exp_004)",
        "seed": SEED,
        "lgb_params": LGB_PARAMS,
        "sha256": {"train.csv": sha256_file(DATA_PATH)},
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "lightgbm": lgb.__version__,
        },
        "fold_2024": {
            "train_until": TRAIN_UNTIL,
            "valid_year": VALID_YEAR,
            "n_train": int(tr_mask.sum()),
            "best_iteration": int(booster.best_iteration),
            "valid_all": seg_metrics(y_va, p_va),
            "valid_R": seg_metrics(y_va[is_r], p_va[is_r]),
            "valid_F": seg_metrics(y_va[~is_r], p_va[~is_r]),
            "cold_start_R": seg_metrics(y_va[cold & is_r], p_va[cold & is_r]),
        },
        "run_time_min": round((time.time() - t0) / 60, 2),
    }
    with open(os.path.join(OUT_DIR, "metrics_ref_exp001_t8.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(json.dumps(metrics["fold_2024"]["valid_R"], indent=2), flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] REF DONE {metrics['run_time_min']} min", flush=True)


if __name__ == "__main__":
    main()
