"""exp_007 finalize — submission artifacts for the winning variant.

Run AFTER train.py, with the winner decided from metrics_folds.json:
    python finalize.py --variant V1 --p4-iter <fold_2023 argmin> \
        --final-iter <fold_2024 argmin>
(CLI values are cross-checked against metrics_folds.json — mismatch aborts.)

Produces (in ~/SungjinPark/experiments/exp_007/submission_assets/):
  - final_model.txt      LightGBM trained on ALL <=2024 rows (exp_004 mask),
                         num_boost_round = fold_2024 valid-R Brier argmin of
                         the winning spec (exp_006 P4 carry-over),
                         deterministic=true, force_row_wise=true, num_threads=8
  - profile_2024.csv     as-of pitcher profile, cutoff=2024 (V1 only)
  - feature_meta.json    feature order, train-only category vocab, delta,
                         num_boost_round — everything inference.py needs
  - calibration.json     delta provenance (r trend 2019..2024 -> r_hat_2025,
                         anchor = final model pred mean on 2024 R train rows)
  - metrics_final.json   deterministic re-run check, timings, sha256

Also runs a determinism-impact check: fold_2024 spec retrained with
deterministic=true/force_row_wise=true at the P4 iter, valid-R Brier compared
against the (non-deterministic-flag) train.py headline.

Rule 2-4 safe: no test file is read here; calibration uses train rows only.
"""

import argparse
import hashlib
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import (  # noqa: E402  (same-dir import of the audited logic)
    CAT_COLS, DATA_PATH, DROP_COLS, F_EXCLUDE_MAX_SEASON, LGB_PARAMS, OUT_DIR,
    PHYS, PMAP_PATH, SEED, TARGET, TM_PATH, build_asof_profile, make_X,
    seg_metrics, sha256_file,
)

ASSETS_DIR = os.path.join(OUT_DIR, "submission_assets")
EPS = 1e-7


def np_logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["V1", "V2"])
    ap.add_argument("--p4-iter", type=int, required=True)
    ap.add_argument("--final-iter", type=int, required=True)
    args = ap.parse_args()
    use_profile = args.variant == "V1"

    with open(os.path.join(OUT_DIR, "metrics_folds.json")) as f:
        folds = json.load(f)
    v = folds[args.variant]
    assert v["fold_2024"]["headline_iter"] == args.p4_iter, "p4-iter mismatch"
    assert v["fold_2024"]["own_R_brier_argmin"]["iter"] == args.final_iter, \
        "final-iter mismatch vs fold_2024 argmin"
    headline_brier = v["calibration_C"]["fold_2024_valid_R_precalib"]["brier"]

    os.makedirs(ASSETS_DIR, exist_ok=True)
    np.random.seed(SEED)
    t0 = time.time()

    print(f"[{time.strftime('%H:%M:%S')}] loading data", flush=True)
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    assert df["row_id"].is_unique
    pmap = pd.read_csv(PMAP_PATH)
    tm = pd.read_csv(
        TM_PATH, encoding="utf-8-sig",
        usecols=["season", "trackman_game_id", "pitcher_trackman_id",
                 "pitch_type_group"] + PHYS)
    assert int(tm["season"].max()) <= 2024, "trackman contains 2025 — violation"
    tm = tm[tm["pitcher_trackman_id"].isin(set(pmap["pitcher_trackman_id"]))]

    feat_cols = [c for c in df.columns if c not in DROP_COLS]
    pid = df["pitcher_id"].to_numpy()
    # train-only vocabulary (input = train.csv only), saved for inference
    vocab = {}
    for c in CAT_COLS:
        df[c] = df[c].astype("category")
        vocab[c] = [x if isinstance(x, str) else int(x)
                    for x in df[c].cat.categories]
    f_excl = ((df["season"] <= F_EXCLUDE_MAX_SEASON)
              & (df["game_type"] == "F")).to_numpy()

    det_params = dict(LGB_PARAMS)
    det_params.update({"deterministic": True, "force_row_wise": True})

    out = {
        "exp_id": "exp_007_finalize",
        "variant": args.variant,
        "seed": SEED,
        "lgb_params_final": det_params,
        "iter_policy": "exp_006 P4 — fold_2024 valid-R Brier argmin carried to <=2024 model",
        "num_boost_round_final": args.final_iter,
        "sha256_inputs": {
            "train.csv": sha256_file(DATA_PATH),
            "trackman_history.csv": sha256_file(TM_PATH),
            "pitcher_id_map.csv": sha256_file(PMAP_PATH),
            "finalize.py": sha256_file(os.path.abspath(__file__)),
        },
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "lightgbm": lgb.__version__,
        },
    }

    # ---- determinism-impact check on fold_2024 ---------------------------
    print(f"[{time.strftime('%H:%M:%S')}] deterministic check: fold_2024 "
          f"{args.variant} @ iter {args.p4_iter}", flush=True)
    prof23 = build_asof_profile(tm, pmap, 2023) if use_profile else None
    tr24 = ((df["season"] <= 2023).to_numpy()) & ~f_excl
    va24 = (df["season"] == 2024).to_numpy()
    X_tr = make_X(df, pid, feat_cols, tr24, prof23)
    y_tr = df.loc[tr24, TARGET].to_numpy()
    X_va = make_X(df, pid, feat_cols, va24, prof23)
    y_va = df.loc[va24, TARGET].to_numpy()
    is_r = (df.loc[va24, "game_type"] == "R").to_numpy()
    booster_chk = lgb.train(
        det_params, lgb.Dataset(X_tr, label=y_tr, categorical_feature=CAT_COLS),
        num_boost_round=args.p4_iter)
    p_chk = booster_chk.predict(X_va)
    b_chk = float(np.mean((p_chk[is_r] - y_va[is_r]) ** 2))
    out["deterministic_check_fold_2024"] = {
        "valid_R_brier_deterministic": round(b_chk, 6),
        "valid_R_brier_trainpy_headline": headline_brier,
        "gap": round(b_chk - headline_brier, 6),
    }
    print(json.dumps(out["deterministic_check_fold_2024"]), flush=True)
    del X_tr, X_va, booster_chk, prof23

    # ---- final profile (cutoff 2024) -------------------------------------
    prof24 = None
    profile_cols = []
    if use_profile:
        prof24 = build_asof_profile(tm, pmap, 2024)
        profile_cols = list(prof24.columns)
        prof24.reset_index().to_csv(
            os.path.join(ASSETS_DIR, "profile_2024.csv"), index=False,
            encoding="utf-8")
        out["profile_2024"] = {"n_pitchers": int(prof24.shape[0]),
                               "n_features": len(profile_cols),
                               "cutoff": 2024}
        print(f"profile_2024: {prof24.shape}", flush=True)

    # ---- final model: ALL <=2024, masked, P4 iter ------------------------
    tr_fin = ((df["season"] <= 2024).to_numpy()) & ~f_excl
    X_fin = make_X(df, pid, feat_cols, tr_fin, prof24)
    y_fin = df.loc[tr_fin, TARGET].to_numpy()
    print(f"[{time.strftime('%H:%M:%S')}] final train: {X_fin.shape[0]:,} rows"
          f" x {X_fin.shape[1]} feats, {args.final_iter} rounds", flush=True)
    tt = time.time()
    booster = lgb.train(
        det_params, lgb.Dataset(X_fin, label=y_fin, categorical_feature=CAT_COLS),
        num_boost_round=args.final_iter)
    out["final_train_time_min"] = round((time.time() - tt) / 60, 2)
    out["final_n_train"] = int(tr_fin.sum())
    out["final_n_features"] = int(X_fin.shape[1])
    model_path = os.path.join(ASSETS_DIR, "final_model.txt")
    booster.save_model(model_path)

    # ---- calibration constant for 2025 (train-only) ----------------------
    r_hist = (df.loc[(df["season"] <= 2024) & (df["game_type"] == "R")]
              .groupby("season")[TARGET].mean())
    slope, intercept = np.polyfit(r_hist.index.to_numpy(dtype=float),
                                  r_hist.to_numpy(dtype=float), 1)
    r_hat_2025 = float(slope * 2025 + intercept)
    anchor_mask = ((df["season"] == 2024) & (df["game_type"] == "R")).to_numpy()
    X_anchor = make_X(df, pid, feat_cols, anchor_mask, prof24)
    ta = time.time()
    p_anchor = booster.predict(X_anchor)
    anchor_pred_sec = time.time() - ta
    m_anchor = float(np.mean(p_anchor))
    delta = float(np_logit(np.array([r_hat_2025]))[0]
                  - np_logit(np.array([m_anchor]))[0])
    calib = {
        "method": "exp_005 C — train-only base-rate trend anchor, constant logit offset",
        "fit_on": "train R yearly rates 2019..2024 (linear) + final model pred "
                  "mean on season==2024 R train rows",
        "r_history": {int(s): round(float(r), 6) for s, r in r_hist.items()},
        "trend_slope_per_year": round(float(slope), 6),
        "r_hat_2025": round(r_hat_2025, 6),
        "anchor_pred_mean_2024R_train": round(m_anchor, 6),
        "delta": delta,
        "anchor_n_rows": int(anchor_mask.sum()),
        "anchor_predict_sec": round(anchor_pred_sec, 2),
    }
    with open(os.path.join(ASSETS_DIR, "calibration.json"), "w") as f:
        json.dump(calib, f, indent=2)
    out["calibration"] = calib
    print(json.dumps({k: calib[k] for k in
                      ("r_hat_2025", "anchor_pred_mean_2024R_train", "delta")}),
          flush=True)

    # ---- feature meta for inference --------------------------------------
    meta = {
        "variant": args.variant,
        "base_feat_cols": feat_cols,
        "cat_cols": CAT_COLS,
        "cat_categories": vocab,
        "profile_cols": profile_cols,
        "num_boost_round": args.final_iter,
        "delta": delta,
        "seed": SEED,
    }
    with open(os.path.join(ASSETS_DIR, "feature_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    out["artifacts"] = {
        os.path.basename(p): sha256_file(p)
        for p in [model_path,
                  os.path.join(ASSETS_DIR, "feature_meta.json"),
                  os.path.join(ASSETS_DIR, "calibration.json")]
        + ([os.path.join(ASSETS_DIR, "profile_2024.csv")] if use_profile else [])
    }
    out["total_time_min"] = round((time.time() - t0) / 60, 2)
    with open(os.path.join(OUT_DIR, "metrics_final.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[{time.strftime('%H:%M:%S')}] DONE {out['total_time_min']} min",
          flush=True)


if __name__ == "__main__":
    main()
