"""exp_006 — iteration-fixing policy experiment (no-ES submission pipeline).

Spec: experiments/exp_006/config.yaml (2026-08-05, campaign 2)
Base: exp_001 (identical data / features / params / seed; only the way the
number of boosting iterations is chosen changes).

Question: at final-submission time we retrain on ALL data (<=2024) and predict
2025 test, so early stopping against the target year is impossible. Which
iter-fixing policy loses the least vs the (optimistic) ES reference?

Simulation A (main, mimics submission one year earlier — 2024 is "unseen"):
  model: train<=2023, evaluated on valid 2024 (valid-R Brier primary)
  P0  reference : iter = ES best_iter chosen ON valid 2024 (optimistic; exp_001)
  P1  carry-over: iter = ES best_iter of fold_2023 (train<=2022 -> valid 2023)
  P2  row-scale : iter = round(P1 * n_rows(<=2023) / n_rows(<=2022))
  P3  fixed grid: iter in {100, 200, 400} (safety-margin fixed values)

Simulation B (supporting, same policies shifted one year back):
  model: train<=2022, evaluated on valid 2023
  P1' = ES best_iter of fold_2022 (train<=2021 -> valid 2022)
  P2' = round(P1' * n_rows(<=2022) / n_rows(<=2021)), P3 same grid,
  P0' = ES best_iter chosen on valid 2023.

Implementation: with a fixed seed and no early stopping, LightGBM boosting is
sequential, so a single 1000-round model's first k trees are exactly the model
one would get by training k rounds. We therefore train ONCE per fold (1000
rounds, valid eval history recorded), derive the ES-equivalent best_iter from
the logloss history (patience-200 rule identical to lgb.early_stopping), and
score valid predictions at every candidate iteration via num_iteration=k.

No leakage: policies P1/P2/P3 never touch the valid year of their simulation;
P0 is reported only as the optimistic reference. Current-row TrackMan physics
are not in the feature set (provided 48 columns, as exp_001).

Runtime: A5000 CPU, venv (Python 3.11.15 / numpy 1.26.4 / pandas 2.0.3 /
lightgbm 4.7.0). num_threads=8 (exp_003~005 may run in parallel — exp_001
used 24; any tiny P0 reproduction gap is reported as thread nondeterminism).
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
OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_006")

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
    "num_threads": 8,  # parallel-experiment constraint (exp_001 used 24)
    "verbose": -1,
}
NUM_BOOST_ROUND = 1000
ES_PATIENCE = 200  # same as exp_001 early_stopping_rounds

# folds: name, train_until, valid_year
FOLDS = [
    ("fold_2022", 2021, 2022),
    ("fold_2023", 2022, 2023),
    ("fold_2024", 2023, 2024),
]

# diagnostic curve grid (policy iters are added dynamically)
CURVE_ITERS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 40, 45, 48, 50, 60, 75,
    100, 150, 200, 300, 400, 600, 800, 1000,
]
P3_GRID = [100, 200, 400]


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


def es_equivalent_best_iter(losses, patience=ES_PATIENCE):
    """Replicate lgb.early_stopping semantics on a recorded loss history.

    losses: list of per-iteration valid losses (1-based iterations).
    Returns (best_iter_1based, triggered) where triggered=False means the
    stopping condition never fired within the trained rounds (best = argmin).
    Only strict improvements update the best, matching the callback.
    """
    best_loss = float("inf")
    best_it = 0
    for i, v in enumerate(losses, start=1):
        if v < best_loss:
            best_loss = v
            best_it = i
        elif i - best_it >= patience:
            return best_it, True
    return best_it, False


def main():
    np.random.seed(SEED)
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] loading {DATA_PATH}", flush=True)
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    print(f"loaded: {df.shape}", flush=True)
    assert df["row_id"].is_unique

    feat_cols = [c for c in df.columns if c not in DROP_COLS]
    for c in CAT_COLS:
        df[c] = df[c].astype("category")

    metrics = {
        "exp_id": "exp_006",
        "seed": SEED,
        "base_exp": "exp_001",
        "lgb_params": LGB_PARAMS,
        "num_boost_round": NUM_BOOST_ROUND,
        "es_patience": ES_PATIENCE,
        "features": {"n": len(feat_cols), "categorical": CAT_COLS},
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

    n_rows = {}  # train_until -> n_train rows
    fold_data = {}  # fold_name -> dict with booster, curves, es info
    total_infer_sec = 0.0

    for fold_name, train_until, valid_year in FOLDS:
        ft = time.time()
        tr_mask = df["season"] <= train_until
        va_mask = df["season"] == valid_year
        X_tr = df.loc[tr_mask, feat_cols]
        y_tr = df.loc[tr_mask, TARGET].to_numpy()
        X_va = df.loc[va_mask, feat_cols]
        y_va = df.loc[va_mask, TARGET].to_numpy()
        n_rows[train_until] = int(tr_mask.sum())
        print(
            f"[{time.strftime('%H:%M:%S')}] {fold_name}: train {X_tr.shape[0]:,} "
            f"(<= {train_until}) -> valid {X_va.shape[0]:,} ({valid_year}), "
            f"{NUM_BOOST_ROUND} rounds no-ES",
            flush=True,
        )

        dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=CAT_COLS)
        dva = lgb.Dataset(X_va, label=y_va, reference=dtr, categorical_feature=CAT_COLS)
        evals = {}
        booster = lgb.train(
            LGB_PARAMS,
            dtr,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[dva],
            valid_names=["valid"],
            callbacks=[lgb.record_evaluation(evals), lgb.log_evaluation(200)],
        )
        train_sec = time.time() - ft
        losses = list(map(float, evals["valid"]["binary_logloss"]))
        es_it, es_fired = es_equivalent_best_iter(losses)
        print(
            f"  trained in {train_sec/60:.2f} min; ES-equiv best_iter={es_it} "
            f"(triggered={es_fired})",
            flush=True,
        )

        is_r = (df.loc[va_mask, "game_type"] == "R").to_numpy()
        fold_data[fold_name] = {
            "train_until": train_until,
            "valid_year": valid_year,
            "n_train": int(tr_mask.sum()),
            "train_time_min": round(train_sec / 60, 2),
            "es_best_iter": es_it,
            "es_triggered": bool(es_fired),
            "booster": booster,
            "X_va": X_va,
            "y_va": y_va,
            "is_r": is_r,
            "logloss_curve_head": [round(v, 6) for v in losses[:60]],
        }
        booster.save_model(os.path.join(OUT_DIR, f"model_{fold_name}_1000.txt"))

    # ----- derive policy iterations (no valid-year peeking within each sim) -
    b22 = fold_data["fold_2022"]["es_best_iter"]
    b23 = fold_data["fold_2023"]["es_best_iter"]
    b24 = fold_data["fold_2024"]["es_best_iter"]
    ratio_a = n_rows[2023] / n_rows[2022]
    ratio_b = n_rows[2022] / n_rows[2021]

    sims = {
        "sim_A_valid2024": {
            "fold": "fold_2024",
            "policies": {
                "P0_es_reference": b24,
                "P1_carry_prev_fold": b23,
                "P2_row_scaled": max(1, round(b23 * ratio_a)),
                **{f"P3_fixed_{k}": k for k in P3_GRID},
            },
            "policy_inputs": {
                "prev_fold_es_best_iter": b23,
                "row_ratio": round(ratio_a, 6),
            },
        },
        "sim_B_valid2023": {
            "fold": "fold_2023",
            "policies": {
                "P0_es_reference": b23,
                "P1_carry_prev_fold": b22,
                "P2_row_scaled": max(1, round(b22 * ratio_b)),
                **{f"P3_fixed_{k}": k for k in P3_GRID},
            },
            "policy_inputs": {
                "prev_fold_es_best_iter": b22,
                "row_ratio": round(ratio_b, 6),
            },
        },
    }

    # ----- evaluate curves + policies ------------------------------------
    for sim_name, sim in sims.items():
        fd = fold_data[sim["fold"]]
        booster, X_va, y_va, is_r = fd["booster"], fd["X_va"], fd["y_va"], fd["is_r"]
        iters = sorted(set(CURVE_ITERS) | set(sim["policies"].values()))
        curve = {}
        ti = time.time()
        for k in iters:
            p = booster.predict(X_va, num_iteration=k)
            curve[k] = {
                "valid_R": seg_metrics(y_va[is_r], p[is_r]),
                "valid_all": seg_metrics(y_va, p),
            }
        infer_sec = time.time() - ti
        total_infer_sec += infer_sec

        pol_out = {}
        p0_brier = curve[sim["policies"]["P0_es_reference"]]["valid_R"]["brier"]
        for pname, k in sim["policies"].items():
            m = curve[k]
            pol_out[pname] = {
                "iter": k,
                "valid_R": m["valid_R"],
                "valid_all_brier": m["valid_all"]["brier"],
                "delta_brier_R_vs_P0": round(m["valid_R"]["brier"] - p0_brier, 6),
            }
        best_curve_it = min(curve, key=lambda k: curve[k]["valid_R"]["brier"])
        sim_out = {
            "fold": sim["fold"],
            "valid_year": fd["valid_year"],
            "n_train": fd["n_train"],
            "policy_inputs": sim["policy_inputs"],
            "policies": pol_out,
            "curve_argmin_valid_R": {
                "iter": best_curve_it,
                "brier": curve[best_curve_it]["valid_R"]["brier"],
            },
            "curve_valid_R_brier": {
                str(k): curve[k]["valid_R"]["brier"] for k in iters
            },
            "curve_valid_R_bss": {str(k): curve[k]["valid_R"]["bss"] for k in iters},
            "eval_time_sec": round(infer_sec, 1),
        }
        metrics[sim_name] = sim_out
        print(json.dumps({sim_name: {p: v["valid_R"]["brier"] for p, v in pol_out.items()}},
                         indent=2), flush=True)

    # ----- standard-format fold summaries (compat with ledger conventions) -
    for fold_name in fold_data:
        fd = fold_data[fold_name]
        metrics[fold_name] = {
            "train_until": fd["train_until"],
            "valid_year": fd["valid_year"],
            "n_train": fd["n_train"],
            "es_best_iter": fd["es_best_iter"],
            "es_triggered": fd["es_triggered"],
            "train_time_min": fd["train_time_min"],
            "logloss_curve_head": fd["logloss_curve_head"],
        }
    # headline: sim A metrics at ES reference (P0) and at each policy live in
    # sim_A_valid2024; fold_2024 brier/bss/auc = P0 reference for comparability
    p0 = metrics["sim_A_valid2024"]["policies"]["P0_es_reference"]["valid_R"]
    metrics["fold_2024"].update(
        {"brier": p0["brier"], "bss": p0["bss"], "auc": p0.get("auc")}
    )
    metrics["exp001_reference"] = {
        "fold_2024_valid_R_brier": 0.249337,
        "best_iters": {"fold_2022": 48, "fold_2023": 5, "fold_2024": 45},
        "note": "exp_001 used num_threads=24; this run uses 8 — any gap at the "
                "same iter is thread nondeterminism (deterministic flag off).",
    }
    metrics["train_time_min"] = round((time.time() - t0) / 60, 2)
    metrics["inference_time_sec"] = round(total_infer_sec, 2)

    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[{time.strftime('%H:%M:%S')}] DONE total {metrics['train_time_min']} min",
          flush=True)


if __name__ == "__main__":
    main()
