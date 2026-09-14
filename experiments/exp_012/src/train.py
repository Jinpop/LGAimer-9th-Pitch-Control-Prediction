"""exp_012 — does a COMMAND profile beat exp_003's stuff profile on the RF base?

Spec: experiments/exp_012/config.yaml (2026-08-05, campaign 4).
Profile construction and its leak argument: see src/profile2.py.

BASE MODEL: exp_008 R1 — the organizers' RandomForest, verbatim (47 test.csv
features, OrdinalEncoder on 3 cats + median SimpleImputer, n_estimators=100,
max_depth=10, min_samples_leaf=200, random_state=42).  It is the strongest
thing we have: fold_2024 valid-R Brier 0.248487, ahead of our own LGBM line.
Judging a new feature family on a weak base is how exp_003 got a misleading
read, so the delta is measured on the strong base.

RUNS (single changed variable each, all vs A_base)
  A_base     47 features, byte-identical to exp_008 R1     -> anchor, must
             reproduce 0.248487 on fold_2024 valid_R
  B_cons     + consistency family only  (within-pitch-type physics dispersion)
  C_cmd      + command family only      (derived ball-rate, the new signal)
  D_all      + both families
  E_compact  + 12 hand-picked columns, chosen A PRIORI in profile2.COMPACT,
             to separate "the signal is weak" from "RF got diluted"
             (max_features='sqrt': 7 of 47 columns per split becomes 10 of 97,
             so appending ~50 weak columns can hurt even if they carry signal)

Every run gets an explicit `has_profile2` 0/1 flag so the median imputation
that follows is never mistaken for a real value (project rule: no sentinel).
Medians are fit on training rows only.

Validation: rolling-origin, folds 2022 / 2023 / 2024, primary gauge = fold_2024
valid_R Brier on the exact rows exp_001 / exp_007 / exp_008 used (row counts
asserted).  Model selection metric is Brier, never AUC.

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profile2 import (  # noqa: E402
    COMPACT, TM_USECOLS, build_profile2, derive_outcomes, feature_families,
    to_pitcher_id,
)

SEED = 42
N_JOBS = 6  # 3 experiments share this box (orchestrator cap)

DATA_DIR = os.path.expanduser("~/SungjinPark/data")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
TM_PATH = os.path.join(DATA_DIR, "trackman_history.csv")
PMAP_PATH = os.path.expanduser("~/SungjinPark/experiments/exp_002/out/pitcher_id_map.csv")
OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_012")

TARGET = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state"]
FLAG = "has_profile2"

FOLDS = [("fold_2022", 2021, 2022), ("fold_2023", 2022, 2023), ("fold_2024", 2023, 2024)]
EXPECTED_2024 = {"valid_all": 253507, "valid_R": 223497}
RF_PARAMS = dict(n_estimators=100, max_depth=10, min_samples_leaf=200, random_state=42)


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
    d = r * (1.0 - r)
    return None if d <= 0 else float(max(0.0, 100000.0 * (1.0 - brier / d)))


def seg_metrics(y, p):
    n = int(len(y))
    if n == 0:
        return {"n": 0}
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    brier = float(np.mean((p - y) ** 2))
    r = float(np.mean(y))
    d = r * (1.0 - r)
    out = {
        "n": n,
        "brier": round(brier, 6),
        "bss": round(float(1.0 - brier / d), 6) if d > 0 else None,
        "score": round(lb_score(brier, r), 3) if d > 0 else None,
        "base_rate": round(r, 6),
        "pred_mean": round(float(np.mean(p)), 6),
    }
    if 0.0 < r < 1.0:
        out["auc"] = round(float(roc_auc_score(y, p)), 6)
        out["logloss"] = round(float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15))), 6)
    return out


def build_model(num_cols, max_features=None):
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_COLS),
        ("num", SimpleImputer(strategy="median"), num_cols),
    ])
    extra = {} if max_features is None else {"max_features": max_features}
    return Pipeline([("pre", pre),
                     ("clf", RandomForestClassifier(n_jobs=N_JOBS, **RF_PARAMS, **extra))])


def run_fold(df, base_features, prof, add_cols, tag, fold_name, train_until, valid_year,
             save_pred, max_features=None):
    """One rolling-origin fold.  prof=None / add_cols=[] reproduces exp_008 R1."""
    t0 = time.time()
    tr_mask = (df["season"] <= train_until).to_numpy()
    va_mask = (df["season"] == valid_year).to_numpy()
    pid = df["pitcher_id"]

    if add_cols:
        blk = prof[add_cols].reindex(pid.to_numpy())
        blk.index = df.index
        blk[FLAG] = prof["cmd_n"].reindex(pid.to_numpy()).notna().to_numpy().astype(np.int8)
        X = pd.concat([df[base_features], blk], axis=1)
        use_cols = add_cols + [FLAG]
    else:
        X = df[base_features]
        use_cols = []

    num_cols = [c for c in X.columns if c not in CAT_COLS]
    X_tr, X_va = X.loc[tr_mask], X.loc[va_mask]
    y_tr = df.loc[tr_mask, TARGET].to_numpy()
    y_va = df.loc[va_mask, TARGET].to_numpy()

    print(f"[{time.strftime('%H:%M:%S')}] {tag}/{fold_name}: train {X_tr.shape[0]:,} "
          f"(<= {train_until}) -> valid {X_va.shape[0]:,} ({valid_year}), "
          f"{X.shape[1]} feats (+{len(use_cols)}), max_features={max_features or 'sqrt'}",
          flush=True)

    model = build_model(num_cols, max_features)
    model.fit(X_tr, y_tr)
    train_sec = time.time() - t0
    t1 = time.time()
    p_va = model.predict_proba(X_va)[:, 1]
    infer_sec = time.time() - t1

    is_r = (df.loc[va_mask, "game_type"] == "R").to_numpy()
    seen = set(pid[tr_mask].unique())
    cold = ~pid[va_mask].isin(seen).to_numpy()

    out = {
        "tag": tag, "train_until": train_until, "valid_year": valid_year,
        "max_features": max_features or "sqrt",
        "n_train": int(tr_mask.sum()), "n_features": int(X.shape[1]),
        "n_added": len(use_cols),
        "train_time_min": round(train_sec / 60, 2),
        "inference_time_sec": round(infer_sec, 2),
        "valid_all": seg_metrics(y_va, p_va),
        "valid_R": seg_metrics(y_va[is_r], p_va[is_r]),
        "valid_F": seg_metrics(y_va[~is_r], p_va[~is_r]),
        "cold_start_R": seg_metrics(y_va[cold & is_r], p_va[cold & is_r]),
        "seen_R": seg_metrics(y_va[~cold & is_r], p_va[~cold & is_r]),
    }

    if use_cols:
        has = X_va[FLAG].to_numpy().astype(bool)
        out["valid_R_has_profile2"] = seg_metrics(y_va[is_r & has], p_va[is_r & has])
        out["valid_R_no_profile2"] = seg_metrics(y_va[is_r & ~has], p_va[is_r & ~has])
        out["profile2_row_coverage_valid"] = round(float(has.mean()), 4)
        out["profile2_row_coverage_train"] = round(float(X_tr[FLAG].mean()), 4)

    # RF impurity importance; ColumnTransformer output order = CAT_COLS then num_cols
    imp = list(zip(CAT_COLS + num_cols, model.named_steps["clf"].feature_importances_))
    imp.sort(key=lambda t: -t[1])
    out["importance_top25"] = [[f, round(float(g), 6)] for f, g in imp[:25]]
    if use_cols:
        new = set(use_cols)
        out["new_feature_importance"] = [
            [r + 1, f, round(float(g), 6)] for r, (f, g) in enumerate(imp) if f in new
        ]
        out["new_feature_importance_share"] = round(
            float(sum(g for f, g in imp if f in new) / max(sum(g for _, g in imp), 1e-12)), 4)

    if valid_year == 2024:
        assert out["valid_all"]["n"] == EXPECTED_2024["valid_all"], out["valid_all"]["n"]
        assert out["valid_R"]["n"] == EXPECTED_2024["valid_R"], out["valid_R"]["n"]
        if save_pred:
            np.save(os.path.join(OUT_DIR, f"pred_{tag}_fold_2024.npy"), p_va)

    print(json.dumps({f"{tag}/{fold_name}": {
        k: v for k, v in out.items() if not k.endswith("importance") and k != "importance_top25"}},
        indent=2), flush=True)
    return out


def main():
    np.random.seed(SEED)
    t_start = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "out"), exist_ok=True)

    # ---------------- load ---------------------------------------------------
    print(f"[{time.strftime('%H:%M:%S')}] loading train", flush=True)
    df = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
    assert df["row_id"].is_unique
    test_cols = pd.read_csv(TEST_PATH, encoding="utf-8-sig", nrows=0).columns
    base_features = [c for c in test_cols if c != "row_id"]
    assert TARGET not in base_features
    assert not [c for c in base_features if c not in df.columns]
    print(f"train {df.shape}, base features {len(base_features)}", flush=True)

    pmap = pd.read_csv(PMAP_PATH)
    print(f"[{time.strftime('%H:%M:%S')}] loading trackman ({len(TM_USECOLS)} cols)", flush=True)
    tm_head = pd.read_csv(TM_PATH, encoding="utf-8-sig", nrows=0)
    tm_all_cols = list(tm_head.columns)
    tm = pd.read_csv(TM_PATH, encoding="utf-8-sig", usecols=TM_USECOLS)
    assert int(tm["season"].max()) <= 2024, "trackman contains 2025 — violation"
    tm = tm[tm["pitcher_trackman_id"].isin(set(pmap["pitcher_trackman_id"]))]
    print(f"trackman rows for mapped pitchers: {len(tm):,}", flush=True)

    tm, diag = derive_outcomes(tm)
    print("[outcome derivation] " + json.dumps(diag), flush=True)

    # ---------------- static 2024 table (the inference-time artifact) --------
    t = time.time()
    static = build_profile2(tm, 2024)
    static_pid = to_pitcher_id(static, pmap)
    static_pid.to_csv(os.path.join(OUT_DIR, "out", "pitcher_command_profile_2024.csv"))
    fams = feature_families(list(static_pid.columns))
    print(f"[{time.strftime('%H:%M:%S')}] static cutoff=2024 profile: "
          f"{static_pid.shape[0]} pitchers x {static_pid.shape[1]} cols "
          f"(cmd {len(fams['command'])}, cons {len(fams['consistency'])}) "
          f"{time.time()-t:.1f}s", flush=True)

    # sanity: our derived career ball rate vs the organizers' asof_pitcher_ball_rate
    last = df.sort_values("row_id").groupby("pitcher_id").tail(1)
    j = last[["pitcher_id", "asof_pitcher_ball_rate", "asof_pitcher_n"]].merge(
        static_pid[["cmd_ball_rate"]], left_on="pitcher_id", right_index=True, how="inner")
    j = j[(j["asof_pitcher_n"] >= 500) & j["cmd_ball_rate"].notna()]
    corr = float(j["asof_pitcher_ball_rate"].corr(j["cmd_ball_rate"]))
    print(f"[sanity] corr(derived cmd_ball_rate, organizers' asof_pitcher_ball_rate) "
          f"= {corr:.4f} over n={len(j)} pitchers", flush=True)

    metrics = {
        "exp_id": "exp_012",
        "base_exp": "exp_008 R1 (organizers' RF)",
        "seed": SEED, "n_jobs": N_JOBS, "rf_params": RF_PARAMS,
        "trackman_columns_actual": tm_all_cols,
        "trackman_has_plate_location": False,
        "outcome_derivation": diag,
        "profile2": {
            "n_pitchers": int(static_pid.shape[0]),
            "n_cols": int(static_pid.shape[1]),
            "families": {k: len(v) for k, v in fams.items()},
            "columns": list(static_pid.columns),
            "asof_rule": "fold validating year Y -> aggregate seasons <= Y-1 only",
            "static_artifact": "out/pitcher_command_profile_2024.csv (cutoff 2024, test-reproducible)",
            "missing": "NaN below min-denominator; has_profile2 flag + train-fit median impute",
        },
        "sanity_corr_ball_rate_vs_organizers_asof": round(corr, 4),
        "sha256": {
            "train.csv": sha256_file(TRAIN_PATH),
            "trackman_history.csv": sha256_file(TM_PATH),
            "pitcher_id_map.csv": sha256_file(PMAP_PATH),
            "train.py": sha256_file(os.path.abspath(__file__)),
            "profile2.py": sha256_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "profile2.py")),
        },
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__},
    }

    # ---------------- as-of profiles, one per fold cutoff --------------------
    profiles = {}
    for _, train_until, valid_year in FOLDS:
        cutoff = valid_year - 1
        assert cutoff == train_until, "profile horizon must equal train horizon"
        p = to_pitcher_id(build_profile2(tm, cutoff), pmap)
        profiles[cutoff] = p
        print(f"[{time.strftime('%H:%M:%S')}] as-of profile cutoff={cutoff}: "
              f"{p.shape[0]} pitchers x {p.shape[1]} cols", flush=True)
    assert list(profiles[2023].columns) == list(static_pid.columns)

    RUNS = [
        ("A_base", []),
        ("B_cons", fams["consistency"]),
        ("C_cmd", fams["command"]),
        ("D_all", fams["all"]),
        ("E_compact", COMPACT),
    ]
    assert not [c for c in COMPACT if c not in static_pid.columns]

    for tag, add_cols in RUNS:
        metrics[tag] = {"added_columns": add_cols, "n_added": len(add_cols)}
        for fold_name, tu, vy in FOLDS:
            metrics[tag][fold_name] = run_fold(
                df, base_features, profiles[vy - 1], add_cols, tag, fold_name, tu, vy,
                save_pred=(tag != "A_base"))
        with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

    # ---------------- dilution diagnostic (NOT a candidate model) ------------
    # RF picks max_features=sqrt(n) columns per split: 6 of 47 in A_base, ~10 of
    # 114 in D_all.  If D_all loses, it must be possible to say whether the new
    # columns carry no signal or whether the base columns simply got sampled
    # less often.  G pins max_features to the base's 6 with D_all's columns, so
    # it differs from A_base by BOTH the columns and the sampling rate -> it is
    # a diagnostic only, never a submission candidate.
    metrics["G_all_mf6"] = {
        "note": "diagnostic: D_all columns with max_features pinned to A_base's 6",
        "added_columns": fams["all"], "n_added": len(fams["all"]),
    }
    for fold_name, tu, vy in FOLDS[1:]:
        metrics["G_all_mf6"][fold_name] = run_fold(
            df, base_features, profiles[vy - 1], fams["all"], "G_all_mf6",
            fold_name, tu, vy, save_pred=False, max_features=6)
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # ---------------- deltas vs the base ------------------------------------
    base24 = metrics["A_base"]["fold_2024"]["valid_R"]["brier"]
    metrics["anchor_check"] = {
        "exp_008_R1_fold_2024_valid_R_brier": 0.248487,
        "reproduced": base24,
        "abs_diff": round(abs(base24 - 0.248487), 8),
    }
    metrics["summary_delta_vs_A_base"] = {
        tag: {
            f: round(metrics[tag][f]["valid_R"]["brier"]
                     - metrics["A_base"][f]["valid_R"]["brier"], 6)
            for f, _, _ in FOLDS if f in metrics[tag]
        } | {"new_feature_importance_share_2024":
             metrics[tag]["fold_2024"].get("new_feature_importance_share")}
        for tag in [t for t, _ in RUNS[1:]] + ["G_all_mf6"]
    }
    metrics["target_line"] = {
        "rf_base_fold_2024_valid_R_brier": 0.248487,
        "LB_1061_needs_valid_R_brier": 0.24724,
        "best_run": min(
            ((t, metrics[t]["fold_2024"]["valid_R"]["brier"])
             for t, _ in RUNS if "fold_2024" in metrics[t]), key=lambda x: x[1]),
    }
    metrics["train_time_min"] = round((time.time() - t_start) / 60, 2)
    metrics["inference_time_sec"] = round(
        sum(metrics[t]["fold_2024"]["inference_time_sec"] for t, _ in RUNS), 2)
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(json.dumps({"anchor": metrics["anchor_check"],
                      "delta": metrics["summary_delta_vs_A_base"]}, indent=2), flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] DONE {metrics['train_time_min']} min", flush=True)


if __name__ == "__main__":
    main()
