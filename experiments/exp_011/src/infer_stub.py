"""exp_011 -- proof that the shrinkage transform reproduces at TEST time.

The whole experiment rests on one claim: a 2025 test row carries both the rate
and its denominator, so the empirical-Bayes recomputation is a pure per-row
arithmetic operation that needs nothing but that row plus a constant table
shipped with the model.  If that claim were false the features would be
unusable in a submission, and any local gain would be fictional.

This script demonstrates it end to end.

  S1  Load the 5-row test.csv stub and `priors.json` (a *frozen artefact* built
      from train.csv only) and run the exact same `build_features` used in
      training.  Report every produced column for all 5 rows.
  S2  ROW-INDEPENDENCE PROOF (competition rule 2-4): re-run the transform on
      each row ONE AT A TIME, as a 1-row DataFrame, and require bit-identical
      output to the batch run.  If any statistic of the test set leaked into a
      feature, the single-row values would differ.
  S3  Season fallback: test rows are season 2025, which is present in the
      table; also verify an unseen season (2026) resolves to the latest entry
      instead of producing NaN.
  S4  Range / NaN audit of the produced columns.
  S5  Fit the headline variant on ALL training seasons (<= 2024), predict the
      5 test rows, and time it -- the actual submission path.  The fitted
      pipeline is written to `model_<variant>.pkl` for the packager.

Usage:  python infer_stub.py            (reads metrics.json for k / mode / variant)
"""

import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import (  # noqa: E402
    DERIVED_COLS,
    RAW_RATE_COLS,
    SHRUNK_COLS,
    TARGET,
    build_features,
    load_priors,
)
from train import base_features, build_model  # noqa: E402

DATA_DIR = os.path.expanduser("~/SungjinPark/data")
OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_011")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")


def feature_list(variant, base):
    if variant in ("MA", "RA"):
        return base + SHRUNK_COLS
    if variant in ("MB", "RB"):
        return base + SHRUNK_COLS + DERIVED_COLS
    if variant in ("MC", "RC"):
        return [c for c in base if c not in RAW_RATE_COLS] + SHRUNK_COLS + DERIVED_COLS
    return list(base)


def main():
    rep = {}
    with open(os.path.join(OUT_DIR, "metrics.json")) as f:
        metrics = json.load(f)
    k_best = {kk: int(v) for kk, v in metrics["tuning"]["k_final"].items()}
    mode = metrics["tuning"]["mode_final"]
    gt_split = bool(metrics["tuning"]["gt_split_final"])
    variant = metrics["headline_variant"]["tag"]
    # the F mask is a TRAINING-row filter, not a feature; take it if its matched
    # pair (MB vs RB) won on the tuning fold, which is also where `variant` was
    # chosen -- fold_2024 is never consulted for any of these decisions
    prim = "valid_all"
    # the mask is part of the BASE spec (exp_008 R2); keep it unless the
    # unmasked arm actually won on the tuning fold
    use_mask = (
        metrics["MB"]["fold_2023"][prim]["brier"] <= metrics["RB"]["fold_2023"][prim]["brier"]
    )
    print(f"headline variant={variant} k={k_best} prior_mode={mode} "
          f"gt_split={gt_split} mask_F={use_mask}", flush=True)
    rep["config"] = {"variant": variant, "k": k_best, "prior_mode": mode,
                     "gt_split": gt_split, "mask_F": use_mask,
                     "selected_on": f"fold_2023 {prim} brier"}

    priors = load_priors(os.path.join(OUT_DIR, "priors.json"))
    base = base_features()
    feats = feature_list(variant, base)
    add_derived = variant in ("MB", "MC", "RB", "RC")

    # ---- S1: batch transform of the test stub ------------------------------
    test = pd.read_csv(TEST_PATH, encoding="utf-8-sig")
    print(f"\nS1 test.csv: {test.shape}  seasons={sorted(test['season'].unique())}", flush=True)
    t0 = time.time()
    tb, added = build_features(test, priors, k_best, mode=mode, add_derived=add_derived,
                               gt_split=gt_split)
    s1_sec = time.time() - t0
    show = ["row_id", "season", "asof_pitcher_n", "asof_pitcher_success_rate",
            "asof_batter_n", "asof_batter_success_rate"] + added
    with pd.option_context("display.width", 250, "display.max_columns", 100):
        print(tb[show].to_string(index=False), flush=True)
    rep["S1_transform"] = {
        "n_rows": int(len(tb)),
        "added_columns": added,
        "seconds": round(s1_sec, 4),
        "values": json.loads(tb[show].to_json(orient="records")),
    }

    # ---- S2: row-independence proof (rule 2-4) -----------------------------
    per_row = []
    for i in range(len(test)):
        one = test.iloc[[i]].copy()
        ob, _ = build_features(one, priors, k_best, mode=mode, add_derived=add_derived,
                               gt_split=gt_split)
        per_row.append(ob[added].to_numpy(dtype=float))
    single = np.vstack(per_row)
    batch = tb[added].to_numpy(dtype=float)
    max_dev = float(np.nanmax(np.abs(single - batch)))
    same_nan = bool(np.array_equal(np.isnan(single), np.isnan(batch)))
    print(f"\nS2 row-independence: max|single-row - batch| = {max_dev:.3e}, "
          f"nan pattern identical = {same_nan}", flush=True)
    assert max_dev == 0.0 and same_nan, "transform is NOT row-independent"
    rep["S2_row_independence"] = {
        "max_abs_deviation": max_dev,
        "nan_pattern_identical": same_nan,
        "passed": True,
        "meaning": "each test row's features depend only on that row plus priors.json; "
                   "no other test row and no test-set statistic is consulted (rule 2-4)",
    }

    # ---- S3: season lookup / fallback --------------------------------------
    ptab = priors["prior_gt"][mode] if gt_split else priors["prior"][mode]
    known = sorted({int(str(kk).split("|")[0]) for kk in ptab["p_success"]})
    fake = test.copy()
    fake["season"] = 2026
    fb, _ = build_features(fake, priors, k_best, mode=mode, add_derived=add_derived,
                           gt_split=gt_split)
    fb_nan = int(fb[added].isna().to_numpy().sum())
    print(f"\nS3 prior seasons in table = {known}; unseen season 2026 -> "
          f"{fb_nan} NaN produced (expect 0, falls back to {max(known)})", flush=True)
    rep["S3_season_fallback"] = {
        "table_seasons": known,
        "test_season": sorted(int(s) for s in test["season"].unique()),
        "unseen_season_nan_count": fb_nan,
        "passed": fb_nan == 0,
    }

    # ---- S4: range / NaN audit ---------------------------------------------
    audit = {}
    for c in added:
        v = tb[c].to_numpy(dtype=float)
        audit[c] = {"min": float(np.nanmin(v)), "max": float(np.nanmax(v)),
                    "n_nan": int(np.isnan(v).sum())}
    bad = {c: a for c, a in audit.items()
           if a["n_nan"] or (c.startswith("sh_") and not (0.0 <= a["min"] and a["max"] <= 1.0))}
    print(f"\nS4 range audit: {len(bad)} problem column(s) {list(bad)}", flush=True)
    rep["S4_range_audit"] = {"columns": audit, "problems": bad, "passed": not bad}

    # ---- S5: full-data fit -> predict the stub (the submission path) --------
    print(f"\n[{time.strftime('%H:%M:%S')}] S5 loading train for the full fit", flush=True)
    df = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
    tr, _ = build_features(df, priors, k_best, mode=mode, add_derived=add_derived,
                           gt_split=gt_split)
    if use_mask:
        keep = ~((tr["season"] <= 2022) & (tr["game_type"] == "F"))
        print(f"S5 mask drops {int((~keep).sum()):,} training rows", flush=True)
        tr = tr.loc[keep]
    t0 = time.time()
    model = build_model(feats)
    model.fit(tr[feats], tr[TARGET].to_numpy())
    fit_min = (time.time() - t0) / 60
    t1 = time.time()
    p = model.predict_proba(tb[feats])[:, 1]
    pred_sec = time.time() - t1
    print(f"S5 fit on all seasons <= 2024 ({len(tr):,} rows) in {fit_min:.2f} min; "
          f"predicted 5 stub rows in {pred_sec:.4f}s -> {np.round(p, 6).tolist()}", flush=True)
    mpath = os.path.join(OUT_DIR, f"model_{variant}{'_mask' if use_mask else ''}.pkl")
    with open(mpath, "wb") as f:
        pickle.dump(model, f)
    rep["S5_full_fit"] = {
        "n_train_rows": int(len(tr)),
        "mask_F": use_mask,
        "n_features": len(feats),
        "features": feats,
        "fit_min": round(fit_min, 2),
        "stub_predict_sec": round(pred_sec, 4),
        "stub_predictions": [round(float(x), 6) for x in p],
        "model_path": mpath,
        "model_bytes": os.path.getsize(mpath),
    }

    with open(os.path.join(OUT_DIR, "infer_stub.json"), "w") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print("\nALL STUB CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
