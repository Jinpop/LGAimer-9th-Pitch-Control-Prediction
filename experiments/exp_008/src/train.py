"""exp_008 — CV<->LB anchor: reproduce the organizers' public RF baseline on OUR split.

Spec: experiments/exp_008/config.yaml (2026-08-05).

Purpose is NOT performance. The organizers' RF baseline scored 549.51193 on the
public LB (trained on all seasons <= 2024, predicting the 2025 test). We do not
know how our local BSS scale maps onto that LB score. This run puts the SAME RF
on OUR rolling-origin split (train <= 2023 -> valid 2024) so the mapping can be
estimated without spending a submission slot.

Runs
  R1: organizers' notebook verbatim (preprocessing / hyperparameters / feature
      selection untouched), train season <= 2023 -> valid season == 2024.
      Folds 2022 / 2023 / 2024 for the standard rolling-origin report.
  R2: same RF, but training rows with (season <= 2022 & game_type == "F") are
      dropped (the exp_004 mask) -> is the mask effect model-independent?
      Folds 2023 / 2024.

Evaluation (both gauges, on the exact same valid rows as exp_001 / exp_007):
  (a) valid_all  — every season == 2024 row (organizers' notebook definition)
  (b) valid_R    — game_type == "R" only (our primary gauge)
  score = max(0, 100000 * (1 - brier / (r * (1 - r)))),  r = valid base rate

Leakage: the whole pipeline (OrdinalEncoder categories, SimpleImputer medians,
RF) is fit on training rows only; valid rows only ever go through .transform /
.predict_proba. No current-row TrackMan quantity is used (the 47 features are
exactly the test.csv columns). No test-set information is touched.

Environment: A5000, ~/SungjinPark/venv (Python 3.11.15 / numpy 1.26.4 /
pandas 2.0.3 / scikit-learn 1.8.0) — matches the evaluation server.
"""

import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

SEED = 42
N_JOBS = 8  # organizers use n_jobs=-1; capped to 8 because the box is shared.
# (n_jobs only affects wall time, not the fitted forest: trees are seeded from
#  random_state and built independently.)

DATA_DIR = os.path.expanduser("~/SungjinPark/data")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_008")

TARGET = "control_success"

# ---------------------------------------------------------------- organizers'
# The block below is the public baseline, character for character in substance:
CAT_COLS = ["top_bottom", "game_type", "base_state"]


def build_model(n_jobs=N_JOBS):
    test_cols = pd.read_csv(TEST_PATH, encoding="utf-8-sig", nrows=0).columns
    features = [c for c in test_cols if c != "row_id"]
    num_cols = [c for c in features if c not in CAT_COLS]
    preprocessor = ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CAT_COLS,
            ),
            ("num", SimpleImputer(strategy="median"), num_cols),
        ]
    )
    model = Pipeline(
        [
            ("pre", preprocessor),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_leaf=200,
                    n_jobs=n_jobs,
                    random_state=42,
                ),
            ),
        ]
    )
    return model, features, num_cols


# --------------------------------------------------------------------------- #

FOLDS = [
    ("fold_2022", 2021, 2022),
    ("fold_2023", 2022, 2023),
    ("fold_2024", 2023, 2024),
]
# exp_001 / exp_007 valid-row counts for fold_2024 (identity check).
EXPECTED_2024 = {"valid_all": 253507, "valid_R": 223497}


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
    """Organizers' score formula: max(0, 100000 * (1 - brier / (r(1-r))))."""
    denom = r * (1.0 - r)
    if denom <= 0:
        return None
    return float(max(0.0, 100000.0 * (1.0 - brier / denom)))


def seg_metrics(y, p):
    """Same helper as exp_001 (identical definitions), plus the LB-style score."""
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


def run_fold(df, features, fold_name, train_until, valid_year, mask_F, tag):
    tr_mask = df["season"] <= train_until
    if mask_F:
        tr_mask = tr_mask & ~((df["season"] <= 2022) & (df["game_type"] == "F"))
    va_mask = df["season"] == valid_year

    X_tr = df.loc[tr_mask, features]
    y_tr = df.loc[tr_mask, TARGET].to_numpy()
    X_va = df.loc[va_mask, features]
    y_va = df.loc[va_mask, TARGET].to_numpy()
    print(
        f"[{time.strftime('%H:%M:%S')}] {tag}/{fold_name}: train {X_tr.shape[0]:,} "
        f"(<= {train_until}{', mask F<=2022' if mask_F else ''}) "
        f"-> valid {X_va.shape[0]:,} ({valid_year})",
        flush=True,
    )

    model, _, _ = build_model()
    t0 = time.time()
    model.fit(X_tr, y_tr)
    train_sec = time.time() - t0

    t1 = time.time()
    p_va = model.predict_proba(X_va)[:, 1]
    infer_sec = time.time() - t1

    is_r = (df.loc[va_mask, "game_type"] == "R").to_numpy()
    seen = set(df.loc[tr_mask, "pitcher_id"].unique())
    cold = ~df.loc[va_mask, "pitcher_id"].isin(seen).to_numpy()

    out = {
        "train_until": train_until,
        "valid_year": valid_year,
        "n_train": int(tr_mask.sum()),
        "train_time_min": round(train_sec / 60, 2),
        "inference_time_sec": round(infer_sec, 2),
        "valid_all": seg_metrics(y_va, p_va),
        "valid_R": seg_metrics(y_va[is_r], p_va[is_r]),
        "valid_F": seg_metrics(y_va[~is_r], p_va[~is_r]),
        "cold_start_R": seg_metrics(y_va[cold & is_r], p_va[cold & is_r]),
        "seen_R": seg_metrics(y_va[~cold & is_r], p_va[~cold & is_r]),
    }
    if valid_year == 2024:
        assert out["valid_all"]["n"] == EXPECTED_2024["valid_all"], out["valid_all"]["n"]
        assert out["valid_R"]["n"] == EXPECTED_2024["valid_R"], out["valid_R"]["n"]
        np.save(os.path.join(OUT_DIR, f"pred_{tag}_fold_2024.npy"), p_va)
        np.save(
            os.path.join(OUT_DIR, "valid_2024_row_id.npy"),
            df.loc[va_mask, "row_id"].to_numpy(),
        )
    print(json.dumps({f"{tag}/{fold_name}": out}, indent=2), flush=True)
    return out


def main():
    np.random.seed(SEED)
    t_start = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] loading {TRAIN_PATH}", flush=True)
    df = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
    print(f"loaded: {df.shape}", flush=True)
    assert df["row_id"].is_unique

    _, features, num_cols = build_model()
    assert TARGET not in features
    print(f"n_features={len(features)} (cat={len(CAT_COLS)}, num={len(num_cols)})", flush=True)
    missing = [c for c in features if c not in df.columns]
    assert not missing, missing

    metrics = {
        "exp_id": "exp_008",
        "purpose": "CV<->LB anchor (organizers' RF baseline on our split)",
        "seed": SEED,
        "n_jobs": N_JOBS,
        "features": {"n": len(features), "columns": features, "categorical": CAT_COLS},
        "rf_params": {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_leaf": 200,
            "random_state": 42,
        },
        "preprocessing": "ColumnTransformer[OrdinalEncoder(unknown=-1) on 3 cats, SimpleImputer(median) on 44 nums]",
        "sha256": {
            "train.csv": sha256_file(TRAIN_PATH),
            "train.py": sha256_file(os.path.abspath(__file__)),
        },
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
        },
        "season_gametype_counts": {
            f"{int(s)}_{g}": int(n)
            for (s, g), n in df.groupby(["season", "game_type"]).size().items()
        },
    }

    metrics["R1"] = {"spec": "organizers' RF baseline verbatim, no mask"}
    for fold_name, tu, vy in FOLDS:
        metrics["R1"][fold_name] = run_fold(df, features, fold_name, tu, vy, False, "R1")

    metrics["R2"] = {"spec": "same RF + exp_004 mask (drop season<=2022 & game_type=='F' from training)"}
    for fold_name, tu, vy in FOLDS[1:]:
        metrics["R2"][fold_name] = run_fold(df, features, fold_name, tu, vy, True, "R2")

    # ---- anchor arithmetic -------------------------------------------------
    rf_all = metrics["R1"]["fold_2024"]["valid_all"]
    rf_r = metrics["R1"]["fold_2024"]["valid_R"]
    lb_rf = 549.51193  # measured LB of this RF (trained <=2024 -> 2025 test)
    refs = {
        "exp_001_LGBM_valid_R_brier": 0.249337,
        "exp_001_LGBM_valid_R_base_rate": 0.489707,
        "exp_007_V1_postcalib_valid_R_brier": 0.249061,
        "exp_007_V1_precalib_valid_R_brier": 0.249288,
        "exp_001_LGBM_valid_all_brier": 0.24945,
        "exp_001_LGBM_valid_all_base_rate": 0.486105,
        "exp_007_V1_precalib_valid_all_brier": 0.249303,
    }
    anchor = {
        "rf_lb_score_measured": lb_rf,
        "rf_local_score_valid_all": rf_all["score"],
        "rf_local_score_valid_R": rf_r["score"],
        "factor_lb_over_local_all": (
            round(lb_rf / rf_all["score"], 4) if rf_all["score"] else None
        ),
        "factor_lb_over_local_R": (
            round(lb_rf / rf_r["score"], 4) if rf_r["score"] else None
        ),
    }
    for name, brier, r in [
        ("exp_001_LGBM", refs["exp_001_LGBM_valid_R_brier"], refs["exp_001_LGBM_valid_R_base_rate"]),
        ("exp_007_V1_postcalib", refs["exp_007_V1_postcalib_valid_R_brier"], refs["exp_001_LGBM_valid_R_base_rate"]),
        ("exp_007_V1_precalib", refs["exp_007_V1_precalib_valid_R_brier"], refs["exp_001_LGBM_valid_R_base_rate"]),
    ]:
        s = lb_score(brier, r)
        anchor[f"{name}_local_score_valid_R"] = round(s, 3)
        if anchor["factor_lb_over_local_R"]:
            anchor[f"{name}_lb_expected_via_R_factor"] = round(
                s * anchor["factor_lb_over_local_R"], 1
            )
    metrics["references"] = refs
    metrics["anchor"] = anchor

    metrics["train_time_min"] = round((time.time() - t_start) / 60, 2)
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(json.dumps(anchor, indent=2), flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] DONE total {metrics['train_time_min']} min", flush=True)


if __name__ == "__main__":
    main()
