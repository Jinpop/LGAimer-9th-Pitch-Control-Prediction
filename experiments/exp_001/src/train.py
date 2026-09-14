"""exp_001 — LightGBM baseline on provided 48 columns (no feature engineering).

Spec: experiments/exp_001/config.yaml (2026-08-05)
- rolling-origin 3 folds: (train<=2021 -> 2022), (<=2022 -> 2023), (<=2023 -> 2024)
- primary metric: fold_2024 valid R-league Brier
- categorical: LightGBM native categories; missing (asof cold-start) left as NaN
  (no sentinel filling — project rule)
- seed fixed 42; effective params + code/data SHA256 dumped into metrics.json

Runtime target: A5000 server, CPU training (lightgbm 4.7.0, pandas 2.0.3,
numpy 1.26.4, Python 3.11).
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
OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_001")

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
    "num_threads": 24,
    "verbose": -1,
}
NUM_BOOST_ROUND = 5000
EARLY_STOPPING_ROUNDS = 200


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
    """Brier / BSS(r = actual valid rate) / AUC / logloss / means for a segment."""
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
    print(f"[{time.strftime('%H:%M:%S')}] loading {DATA_PATH}", flush=True)
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    print(f"loaded: {df.shape}", flush=True)
    assert df["row_id"].is_unique

    feat_cols = [c for c in df.columns if c not in DROP_COLS]
    # Category mapping built once over full file (label encoding only, no target
    # info). Valid-fold-only categories are unseen codes at train time -> the
    # cold-start path LightGBM handles natively.
    for c in CAT_COLS:
        df[c] = df[c].astype("category")

    metrics = {
        "exp_id": "exp_001",
        "seed": SEED,
        "features": {"n": len(feat_cols), "columns": feat_cols, "categorical": CAT_COLS},
        "lgb_params": LGB_PARAMS,
        "num_boost_round": NUM_BOOST_ROUND,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
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

    total_inference_sec = 0.0
    for fold_name, train_until, valid_year in FOLDS:
        ft = time.time()
        tr_mask = df["season"] <= train_until
        va_mask = df["season"] == valid_year
        X_tr = df.loc[tr_mask, feat_cols]
        y_tr = df.loc[tr_mask, TARGET].to_numpy()
        X_va = df.loc[va_mask, feat_cols]
        y_va = df.loc[va_mask, TARGET].to_numpy()
        print(
            f"[{time.strftime('%H:%M:%S')}] {fold_name}: train {X_tr.shape[0]:,} "
            f"(<= {train_until}) -> valid {X_va.shape[0]:,} ({valid_year})",
            flush=True,
        )

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
        train_sec = time.time() - ft

        ti = time.time()
        p_va = booster.predict(X_va, num_iteration=booster.best_iteration)
        infer_sec = time.time() - ti
        total_inference_sec += infer_sec

        # segments
        is_r = (df.loc[va_mask, "game_type"] == "R").to_numpy()
        seen_pitchers = set(df.loc[tr_mask, "pitcher_id"].unique())
        cold = ~df.loc[va_mask, "pitcher_id"].isin(seen_pitchers).to_numpy()

        fold_out = {
            "train_until": train_until,
            "valid_year": valid_year,
            "n_train": int(tr_mask.sum()),
            "best_iteration": int(booster.best_iteration),
            "train_time_min": round(train_sec / 60, 2),
            "inference_time_sec": round(infer_sec, 2),
            "valid_all": seg_metrics(y_va, p_va),
            "valid_R": seg_metrics(y_va[is_r], p_va[is_r]),
            "valid_F": seg_metrics(y_va[~is_r], p_va[~is_r]),
            "cold_start_all": seg_metrics(y_va[cold], p_va[cold]),
            "cold_start_R": seg_metrics(y_va[cold & is_r], p_va[cold & is_r]),
            "seen_R": seg_metrics(y_va[~cold & is_r], p_va[~cold & is_r]),
            "n_cold_start_pitchers": int(
                df.loc[va_mask, "pitcher_id"][cold].nunique()
            ),
        }
        metrics[fold_name] = fold_out
        booster.save_model(
            os.path.join(OUT_DIR, f"model_{fold_name}.txt"),
            num_iteration=booster.best_iteration,
        )
        print(json.dumps({fold_name: fold_out}, indent=2), flush=True)

        # feature importance (gain) top 30
        imp = sorted(
            zip(feat_cols, booster.feature_importance("gain")),
            key=lambda t: -t[1],
        )[:30]
        metrics[fold_name]["feature_importance_gain_top30"] = [
            [f, round(float(g), 1)] for f, g in imp
        ]

    metrics["train_time_min"] = round((time.time() - t0) / 60, 2)
    metrics["inference_time_sec"] = round(total_inference_sec, 2)
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[{time.strftime('%H:%M:%S')}] DONE total {metrics['train_time_min']} min", flush=True)


if __name__ == "__main__":
    main()
