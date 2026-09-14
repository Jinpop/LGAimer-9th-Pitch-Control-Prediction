"""exp_011 -- empirical-Bayes shrinkage of the as-of rate features, on the RF base.

Hypothesis
    `asof_*_rate` are RAW ratios whose denominators span 0..6000+, so the low-n
    rows are noise the model currently takes at face value.  Replacing them with
    the Beta-posterior mean (rate*n + k*prior)/(n + k) should raise
    DISCRIMINATION (AUC) and therefore lower Brier.

Base
    exp_008 R1 = the organizers' RandomForest, verbatim, on OUR rolling-origin
    split.  fold_2024 valid_all Brier 0.248769 (valid_R 0.248487) -- the
    strongest base we have.
    Every run below changes ONLY the feature matrix; preprocessing, forest
    hyper-parameters, folds, seed and gauge are identical to exp_008.

Runs (folds 2022 / 2023 / 2024; PRIMARY GAUGE = valid_all, R+F -- see below)
    R0   base 47 features                       -> identity check vs exp_008 R1
    R0b  base 47 with RB's max_features         -> isolates the mtry side effect
                                                   of merely having more columns
    RA   base + 10 shrunk rates
    RB   RA + 7 derived (matchup log-odds, failure composite, form deviations,
         prev-game missing flag)
    RC   RB with the 10 RAW rates dropped       -> are the raw ratios harmful?
    M0   base 47 + exp_004 F mask = exp_008 R2  -> matched control for MB
    MB   RB features + the same F mask          -> MB-vs-M0 isolates the feature
                                                   block, MB-vs-RB the mask

Hyper-parameter selection (k, prior mode, gt_split, variant, mask)
    Tuned on fold_2023 (train <= 2022 -> valid 2023) ONLY.  fold_2024 is never
    consulted during tuning, so its number is an honest held-out result while
    the reported fold_2023 number is optimistic by construction and labelled so.

Leakage
    * No current-row TrackMan quantity: every feature is one of the 48 test.csv
      columns or a function of them plus scalar constants.
    * Priors read only seasons strictly EARLIER than the row's season, so a
      valid-year row sees exactly the fold's training window -- see features.py.
    * The transform is row-wise; `infer_stub.py` proves it reproduces on
      test.csv one row at a time, touching no other row (rule 2-4).

Environment: A5000, ~/SungjinPark/venv (Python 3.11.15 / numpy 1.26.4 /
pandas 2.0.3 / scikit-learn 1.8.0) -- matches the evaluation server.
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
from features import (  # noqa: E402
    DERIVED_COLS,
    PRIOR_MODES,
    RAW_RATE_COLS,
    SHRUNK_COLS,
    TARGET,
    build_features,
    build_prior_table,
    save_priors,
)

SEED = 42
N_JOBS = 6  # shared box; three experiments run in parallel today

DATA_DIR = os.path.expanduser("~/SungjinPark/data")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_011")

CAT_COLS = ["top_bottom", "game_type", "base_state"]
RF_PARAMS = dict(n_estimators=100, max_depth=10, min_samples_leaf=200, random_state=42)

FOLDS = [("fold_2022", 2021, 2022), ("fold_2023", 2022, 2023), ("fold_2024", 2023, 2024)]
EXPECTED_2024 = {"valid_all": 253507, "valid_R": 223497}

# PRIMARY GAUGE: valid_all (R+F).  Changed 2026-08-05 by the orchestrator --
# the 2025 test set has 245,789 rows, inside the per-season ALL-rows range
# (237k..253k) and ~10% above the R-only range (211k..223k); KBO's regular
# season is a fixed 720 games, so R alone cannot produce that count.  The
# earlier "test is R-only" reading rested on the 5-row stub, which is no
# evidence (P(all 5 rows R) = 0.56 even with 11% F).
PRIMARY = "valid_all"

# BASE: exp_008 R2 = the organizers' RF + the exp_004 mask (drop training rows
# with season<=2022 & game_type=="F").  Changed from R1 2026-08-05 by the
# orchestrator: with valid_all primary, the untouched-F base collapses on the
# F segment (fold_2023 valid_all 0.251651 vs 0.248713 masked, -0.002938), so
# measuring shrinkage on R1 would measure it in a regime we would never ship.
# EVERY tuning decision below is therefore made with the mask applied.
MASK_DESC = 'drop training rows with season<=2022 & game_type=="F" (exp_004 mask)'
BASE_REF = {  # exp_008 R1 -- kept only for the R0 identity check
    "valid_all": {"fold_2022": 0.243987, "fold_2023": 0.251651, "fold_2024": 0.248769},
    "valid_R": {"fold_2022": 0.248951, "fold_2023": 0.248582, "fold_2024": 0.248487},
}
# exp_008 R2 absolutes: valid_R 0.248455 (fold_2024) measured directly; the
# remaining cells are R1 + the deltas reported in exp_008 section 5.  These are
# CROSS-CHECKS only -- M0 is recomputed in-process and is the authoritative base.
R2_REF = {
    "valid_all": {"fold_2023": 0.248713, "fold_2024": 0.248430},
    "valid_R": {"fold_2023": 0.248564, "fold_2024": 0.248455},
}

K_GRID = [25, 100, 400, 1600]  # coarse log sweep, refined below
K_REFINE = [12, 35, 50, 70, 140, 200, 280, 560, 800, 1100, 2400, 5000]
TUNE_FOLD = ("fold_2023", 2022, 2023)


# --------------------------------------------------------------------------- #
# plumbing
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


def base_features():
    cols = pd.read_csv(TEST_PATH, encoding="utf-8-sig", nrows=0).columns
    return [c for c in cols if c != "row_id"]


def build_model(features, max_features=None, n_jobs=N_JOBS):
    cats = [c for c in CAT_COLS if c in features]
    nums = [c for c in features if c not in cats]
    pre = ColumnTransformer(
        [
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cats),
            ("num", SimpleImputer(strategy="median"), nums),
        ]
    )
    params = dict(RF_PARAMS)
    params["n_jobs"] = n_jobs
    if max_features is not None:
        params["max_features"] = max_features
    return Pipeline([("pre", pre), ("clf", RandomForestClassifier(**params))])


def lb_score(brier, r):
    d = r * (1.0 - r)
    return float(max(0.0, 100000.0 * (1.0 - brier / d))) if d > 0 else None


def seg_metrics(y, p):
    n = int(len(y))
    if n == 0:
        return {"n": 0}
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    brier = float(np.mean((p - y) ** 2))
    r = float(np.mean(y))
    d = r * (1.0 - r)
    out = {
        "n": n,
        "brier": round(brier, 6),
        "bss": round(1.0 - brier / d, 6) if d > 0 else None,
        "score": round(lb_score(brier, r), 3) if d > 0 else None,
        "base_rate": round(r, 6),
        "pred_mean": round(float(np.mean(p)), 6),
    }
    if 0.0 < r < 1.0:
        out["auc"] = round(float(roc_auc_score(y, p)), 6)
        out["logloss"] = round(float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15))), 6)
    return out


def fit_predict(frame, features, train_until, valid_year, max_features=None, mask_F=False):
    tr = frame["season"] <= train_until
    if mask_F:
        tr = tr & ~((frame["season"] <= 2022) & (frame["game_type"] == "F"))
    va = frame["season"] == valid_year
    model = build_model(features, max_features=max_features)
    t0 = time.time()
    model.fit(frame.loc[tr, features], frame.loc[tr, TARGET].to_numpy())
    fit_sec = time.time() - t0
    t1 = time.time()
    p = model.predict_proba(frame.loc[va, features])[:, 1]
    inf_sec = time.time() - t1
    return {
        "p": p,
        "y": frame.loc[va, TARGET].to_numpy(),
        "va": va,
        "fit_sec": fit_sec,
        "inf_sec": inf_sec,
        "n_train": int(tr.sum()),
        "max_features_resolved": int(model.named_steps["clf"].estimators_[0].max_features_),
    }


def gauges(frame, res):
    """All three gauges at once; `valid_all` is the primary decision metric."""
    is_r = (frame.loc[res["va"], "game_type"] == "R").to_numpy()
    y, p = res["y"], res["p"]
    return {
        "valid_all": seg_metrics(y, p),
        "valid_R": seg_metrics(y[is_r], p[is_r]),
        "valid_F": seg_metrics(y[~is_r], p[~is_r]),
    }, is_r


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main():
    np.random.seed(SEED)
    t_start = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] loading train", flush=True)
    df = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
    print(f"loaded {df.shape}", flush=True)
    assert df["row_id"].is_unique

    BASE = base_features()
    assert TARGET not in BASE and len(BASE) == 47, len(BASE)

    # ---- priors (identical function used by infer_stub.py) -----------------
    seasons_out = sorted(int(s) for s in df["season"].unique()) + [2025]
    priors = build_prior_table(df, seasons_out)
    priors["meta"] = {
        "exp_id": "exp_011",
        "built_from": "train.csv (full)",
        "rule": "prior[mode][f][s] uses ONLY rows with season < s; "
                "s = first season falls back to that season itself",
        "modes": list(PRIOR_MODES),
        "seasons": [str(s) for s in seasons_out],
    }
    save_priors(priors, os.path.join(OUT_DIR, "priors.json"))
    print("prior[cum]:", json.dumps(
        {k: {s: round(v, 5) for s, v in d.items()} for k, d in priors["prior"]["cum"].items()},
        indent=1), flush=True)

    metrics = {
        "exp_id": "exp_011",
        "hypothesis": "empirical-Bayes shrinkage of asof_* rates raises discrimination on the RF base",
        "base_exp": "exp_008 R1 (organizers' RF, our rolling-origin split)",
        "seed": SEED,
        "n_jobs": N_JOBS,
        "rf_params": RF_PARAMS,
        "preprocessing": "ColumnTransformer[OrdinalEncoder(unknown=-1) on 3 cats, SimpleImputer(median) on the rest]",
        "base_reference_exp008_R1_valid_R_brier": BASE_REF,
        "sha256": {
            "train.csv": sha256_file(TRAIN_PATH),
            "train.py": sha256_file(os.path.abspath(__file__)),
            "features.py": sha256_file(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "features.py")
            ),
        },
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
        },
        "priors": priors["prior"],
        "prior_windows": priors["windows"],
        "per_season_rate": priors["per_season_rate"],
    }

    # ---------------------------------------------------------------- tuning
    # k and the prior mode are chosen on fold_2023 (train <= 2022 -> valid 2023).
    # fold_2024 is NOT touched here.  Gauge: valid_R Brier = the decision metric.
    tune_name, tune_tu, tune_vy = TUNE_FOLD
    print(f"\n[{time.strftime('%H:%M:%S')}] === tuning on {tune_name} "
          f"(gauge = {PRIMARY} brier, MASKED base) ===", flush=True)
    df_tune = df.loc[df["season"] <= tune_vy].copy()  # only rows this fold can see
    tune = {
        "gauge": f"{tune_name} {PRIMARY} brier",
        "note": "fold_2024 never consulted during tuning; all tuning fits use mask_F=True",
        "mask_F": True,
        "grid_global_k": [],
        "refine_pitcher": [],
        "prior_mode": [],
        "gt_split": [],
    }

    def eval_cfg(kdict, mode, tag, gt_split=False):
        frame, _added = build_features(
            df_tune, priors, kdict, mode=mode, add_derived=False, gt_split=gt_split
        )
        feats = BASE + SHRUNK_COLS
        res = fit_predict(frame, feats, tune_tu, tune_vy, mask_F=True)
        g, _ = gauges(frame, res)
        del frame
        print(
            f"  {tag:16s} k={kdict} mode={mode} gt_split={int(gt_split)} -> "
            f"{PRIMARY} brier={g[PRIMARY]['brier']:.6f} auc={g[PRIMARY]['auc']:.6f} "
            f"| valid_R {g['valid_R']['brier']:.6f} valid_F {g['valid_F']['brier']:.6f} "
            f"({res['fit_sec']/60:.2f} min)",
            flush=True,
        )
        return {"brier": g[PRIMARY]["brier"], "auc": g[PRIMARY]["auc"],
                "valid_R_brier": g["valid_R"]["brier"], "valid_F_brier": g["valid_F"]["brier"]}

    mode_best, gt_best = "cum", False
    for k in K_GRID:
        kd = {"pitcher": k, "batter": k, "mix": k}
        tune["grid_global_k"].append({"k": k, "mode": mode_best, **eval_cfg(kd, mode_best, "grid")})
    best = min(tune["grid_global_k"], key=lambda r: r["brier"])
    k_best = {"pitcher": best["k"], "batter": best["k"], "mix": best["k"]}
    best_brier = best["brier"]
    tune["best_global_k"] = best["k"]
    print(f"  -> best global k={best['k']} {PRIMARY} brier={best_brier:.6f}", flush=True)

    # one coordinate pass on the dominant family (pitcher); batter/mix carry far
    # less signal (EDA sec.1) and a full 3-family grid is not worth the wall time
    cands = sorted(
        [k for k in K_REFINE if k != k_best["pitcher"]],
        key=lambda k: abs(np.log(k) - np.log(k_best["pitcher"])),
    )[:3]
    for k in cands:
        kd = dict(k_best)
        kd["pitcher"] = k
        m = eval_cfg(kd, mode_best, "refine[pitcher]")
        tune["refine_pitcher"].append({"k": k, **m})
        if m["brier"] < best_brier:
            best_brier, k_best["pitcher"] = m["brier"], k
    print(f"  -> after refine k={k_best} brier={best_brier:.6f}", flush=True)

    # prior mode: the league base rate falls ~1pp/season, so `cum` is biased high
    tune["prior_mode"].append({"mode": "cum", "brier": best_brier, "note": "incumbent"})
    for mode in [m for m in PRIOR_MODES if m != "cum"]:
        m = eval_cfg(k_best, mode, f"prior[{mode}]")
        tune["prior_mode"].append({"mode": mode, **m})
        if m["brier"] < best_brier:
            best_brier, mode_best = m["brier"], mode
    print(f"  -> prior mode={mode_best} brier={best_brier:.6f}", flush=True)

    # game_type-split prior: the F league changed label regime in 2023 (EDA
    # sec.1.4), so an F row's prior may need to be the F league value.  Only the
    # two target-kind specs can be split exactly -- see features.py.
    tune["gt_split"].append({"gt_split": False, "brier": best_brier, "note": "incumbent"})
    m = eval_cfg(k_best, mode_best, "gt_split", gt_split=True)
    tune["gt_split"].append({"gt_split": True, **m})
    if m["brier"] < best_brier:
        best_brier, gt_best = m["brier"], True

    tune["k_final"] = dict(k_best)
    tune["mode_final"] = mode_best
    tune["gt_split_final"] = gt_best
    tune["best_tuning_brier"] = best_brier
    metrics["tuning"] = tune
    del df_tune
    print(f"[{time.strftime('%H:%M:%S')}] FINAL k={k_best} mode={mode_best} "
          f"gt_split={gt_best}", flush=True)

    # ------------------------------------------------------------ final runs
    df_aug, added_A = build_features(df, priors, k_best, mode=mode_best,
                                     add_derived=False, gt_split=gt_best)
    df_full, added_AB = build_features(df, priors, k_best, mode=mode_best,
                                       add_derived=True, gt_split=gt_best)
    assert added_A == SHRUNK_COLS, added_A
    assert added_AB == SHRUNK_COLS + DERIVED_COLS, added_AB

    feats_MB = BASE + SHRUNK_COLS + DERIVED_COLS
    feats_MC = [c for c in BASE if c not in RAW_RATE_COLS] + SHRUNK_COLS + DERIVED_COLS
    mf_mb = int(np.sqrt(len(feats_MB)))
    # (description, frame, features, max_features, mask_F, folds)
    F3, F23 = FOLDS, FOLDS[1:]
    variants = {
        # --- the base and its arms, ALL masked (this is the shipping regime) --
        "M0": (f"BASE: base 47 + mask [{MASK_DESC}] = exp_008 R2", df, BASE, None, True, F3),
        "M0b": (f"base 47 + mask, max_features={mf_mb} (mtry control for MB)",
                df, BASE, mf_mb, True, F23),
        "MA": ("base + 10 shrunk rates, masked", df_aug, BASE + SHRUNK_COLS, None, True, F3),
        "MB": ("base + shrunk + 7 derived, masked", df_full, feats_MB, None, True, F3),
        "MC": ("shrunk + derived, 10 RAW rates dropped, masked", df_full, feats_MC, None, True, F3),
        # --- unmasked controls -------------------------------------------------
        # R0 reproduces exp_008 R1 exactly -> proves the harness is unchanged.
        # RB is MB without the mask -> isolates the mask's effect on the feature
        # arm, so MB-vs-M0 (features) and MB-vs-RB (mask) are both 1-variable.
        "R0": ("base 47, NO mask (exp_008 R1 identity check)", df, BASE, None, False, F3),
        "RB": ("base + shrunk + derived, NO mask", df_full, feats_MB, None, False, F23),
    }
    order = ["M0", "M0b", "MA", "MB", "MC", "R0", "RB"]
    preds_2024 = {}
    for tag in order:
        desc, frame, feats, mf, mask_F, folds = variants[tag]
        metrics[tag] = {"spec": desc, "n_features": len(feats), "features": feats,
                        "mask_F": mask_F}
        print(f"\n[{time.strftime('%H:%M:%S')}] === {tag}: {desc} ({len(feats)} feats) ===",
              flush=True)
        for fold_name, tu, vy in folds:
            res = fit_predict(frame, feats, tu, vy, max_features=mf, mask_F=mask_F)
            g, is_r = gauges(frame, res)
            seen = set(frame.loc[frame["season"] <= tu, "pitcher_id"].unique())
            cold = ~frame.loc[res["va"], "pitcher_id"].isin(seen).to_numpy()
            y, p = res["y"], res["p"]
            out = {
                "train_until": tu,
                "valid_year": vy,
                "n_train": res["n_train"],
                "max_features_resolved": res["max_features_resolved"],
                "train_time_min": round(res["fit_sec"] / 60, 2),
                "inference_time_sec": round(res["inf_sec"], 2),
                "cold_start_all": seg_metrics(y[cold], p[cold]),
                "seen_all": seg_metrics(y[~cold], p[~cold]),
                "cold_start_R": seg_metrics(y[cold & is_r], p[cold & is_r]),
                "delta_vs_exp008_R1": {
                    gg: round(g[gg]["brier"] - BASE_REF[gg][fold_name], 6)
                    for gg in ("valid_all", "valid_R")
                },
                "delta_vs_base_M0": {
                    gg: (round(g[gg]["brier"] - metrics["M0"][fold_name][gg]["brier"], 6)
                         if "M0" in metrics and fold_name in metrics["M0"] else 0.0)
                    for gg in ("valid_all", "valid_R")
                },
            }
            out.update(g)
            if vy == 2024:
                assert out["valid_all"]["n"] == EXPECTED_2024["valid_all"], out["valid_all"]["n"]
                assert out["valid_R"]["n"] == EXPECTED_2024["valid_R"], out["valid_R"]["n"]
                preds_2024[tag] = p
                np.save(os.path.join(OUT_DIR, f"pred_{tag}_fold_2024.npy"), p)
            metrics[tag][fold_name] = out
            print(
                f"  {fold_name}: {PRIMARY} brier={g[PRIMARY]['brier']:.6f} "
                f"bss={g[PRIMARY]['bss']:+.6f} score={g[PRIMARY]['score']:.1f} "
                f"auc={g[PRIMARY]['auc']:.6f} "
                f"(vs base M0 {out['delta_vs_base_M0'][PRIMARY]:+.6f}, "
                f"vs exp_008 R1 {out['delta_vs_exp008_R1'][PRIMARY]:+.6f}) "
                f"| R {g['valid_R']['brier']:.6f} F {g['valid_F']['brier']:.6f}",
                flush=True,
            )

    # R0 must reproduce exp_008 R1 -- same code path, same seed, same rows.
    ident = {}
    for gg in ("valid_all", "valid_R"):
        got = {f: metrics["R0"][f][gg]["brier"] for f, _, _ in FOLDS}
        ident[gg] = {
            "R0": got,
            "exp_008_R1": BASE_REF[gg],
            "max_abs_diff": round(max(abs(got[f] - BASE_REF[gg][f]) for f in got), 8),
        }
    maxdiff = max(ident[gg]["max_abs_diff"] for gg in ident)
    metrics["identity_check"] = {
        "gauges": ident,
        "max_abs_diff": maxdiff,
        "passed": bool(maxdiff < 1e-6),
        "note": "exp_008 ran with n_jobs=8, this run with 6; RF trees are seeded "
                "from random_state and built independently, so the fit is identical",
    }
    print(f"\nidentity check (R0 vs exp_008 R1): passed={metrics['identity_check']['passed']} "
          f"max_abs_diff={maxdiff}", flush=True)
    metrics["base_crosscheck_M0_vs_exp008_R2"] = {
        gg: {
            f: {"M0": metrics["M0"][f][gg]["brier"], "exp_008_R2_reported": R2_REF[gg][f],
                "diff": round(metrics["M0"][f][gg]["brier"] - R2_REF[gg][f], 6)}
            for f in R2_REF[gg]
        }
        for gg in R2_REF
    }
    print("base crosscheck M0 vs exp_008 R2:",
          json.dumps(metrics["base_crosscheck_M0_vs_exp008_R2"]), flush=True)

    # ------------------------------------------------- low-n segment analysis
    # Selection of the headline variant uses fold_2023 (tuning fold), NOT 2024.
    cand = ["MA", "MB", "MC"]
    best_tag = min(cand, key=lambda t: metrics[t]["fold_2023"][PRIMARY]["brier"])
    metrics["headline_variant"] = {
        "tag": best_tag,
        "selected_on": f"fold_2023 {PRIMARY} brier (tuning fold)",
        "fold_2023_brier": {t: metrics[t]["fold_2023"][PRIMARY]["brier"] for t in cand},
        "base": "M0 (masked RF) -- recomputed in-process, authoritative",
        "unmasked_counterpart": "RB (same features, no mask); RB-vs-MB isolates the mask",
    }
    va24 = df["season"] == 2024
    is_r24 = (df.loc[va24, "game_type"] == "R").to_numpy()
    y24 = df.loc[va24, TARGET].to_numpy()
    n24 = df.loc[va24, "asof_pitcher_n"].to_numpy()
    edges = [(-0.5, 0.5), (0.5, 100), (100, 300), (300, 1000), (1000, 3000), (3000, 1e9)]
    seg = {"variant": best_tag, "gauge": f"fold_2024 {PRIMARY}", "buckets": [], "by_league": []}
    print(f"\nlow-n segments (fold_2024 {PRIMARY}), M0 vs {best_tag}:", flush=True)
    for lo, hi in edges:
        m = (n24 > lo) & (n24 <= hi)
        if m.sum() == 0:
            continue
        row = {"asof_pitcher_n": f"({lo:g}, {hi:g}]", "n_rows": int(m.sum()),
               "frac_F": round(float(np.mean(~is_r24[m])), 4)}
        for t in ["M0", best_tag]:
            row[t] = seg_metrics(y24[m], preds_2024[t][m])
        row["delta_brier"] = round(row[best_tag]["brier"] - row["M0"]["brier"], 6)
        seg["buckets"].append(row)
        print(
            f"  n in ({lo:g},{hi:g}] rows={row['n_rows']:>7} F={row['frac_F']:.3f} "
            f"M0={row['M0']['brier']:.6f} {best_tag}={row[best_tag]['brier']:.6f} "
            f"delta={row['delta_brier']:+.6f}",
            flush=True,
        )
    # same split inside each league (does the F segment hold more low-n rows?)
    for lab, lm in [("R", is_r24), ("F", ~is_r24)]:
        for lo, hi in [(-0.5, 100), (100, 1000), (1000, 1e9)]:
            m = lm & (n24 > lo) & (n24 <= hi)
            if m.sum() == 0:
                continue
            row = {"league": lab, "asof_pitcher_n": f"({lo:g}, {hi:g}]", "n_rows": int(m.sum())}
            for t in ["M0", best_tag]:
                row[t] = seg_metrics(y24[m], preds_2024[t][m])
            row["delta_brier"] = round(row[best_tag]["brier"] - row["M0"]["brier"], 6)
            seg["by_league"].append(row)
            print(f"  [{lab}] n in ({lo:g},{hi:g}] rows={row['n_rows']:>7} "
                  f"M0={row['M0']['brier']:.6f} {best_tag}={row[best_tag]['brier']:.6f} "
                  f"delta={row['delta_brier']:+.6f}", flush=True)
    metrics[f"lown_segments_fold_2024_{PRIMARY}"] = seg

    # -------------------------------------------------------- standard block
    for fold_name, _, _ in FOLDS:
        b = metrics[best_tag][fold_name][PRIMARY]
        metrics[fold_name] = {
            "variant": best_tag,
            "gauge": PRIMARY,
            "brier": b["brier"],
            "bss": b["bss"],
            "auc": b["auc"],
            "score": b["score"],
            "n_valid": b["n"],
            "brier_valid_R": metrics[best_tag][fold_name]["valid_R"]["brier"],
            "brier_valid_F": metrics[best_tag][fold_name]["valid_F"]["brier"],
            "base_variant": "M0 (masked RF = exp_008 R2)",
            "base_brier": metrics["M0"][fold_name][PRIMARY]["brier"],
            "delta_vs_base": round(
                b["brier"] - metrics["M0"][fold_name][PRIMARY]["brier"], 6),
            "base_brier_exp008_R1_unmasked": BASE_REF[PRIMARY][fold_name],
            "delta_vs_exp008_R1": round(b["brier"] - BASE_REF[PRIMARY][fold_name], 6),
            "unmasked_arm_RB": (metrics["RB"][fold_name][PRIMARY]["brier"]
                                if fold_name in metrics["RB"] else None),
            "note": "fold_2023 was used to tune k / prior mode / variant -> optimistic; "
                    "fold_2024 is held out",
        }
    metrics["best_fold_2024_variant_observed"] = min(
        order, key=lambda t: metrics[t]["fold_2024"][PRIMARY]["brier"]
    )
    metrics["summary_fold_2024"] = {
        t: {
            "brier_all": metrics[t]["fold_2024"]["valid_all"]["brier"],
            "bss_all": metrics[t]["fold_2024"]["valid_all"]["bss"],
            "score_all": metrics[t]["fold_2024"]["valid_all"]["score"],
            "auc_all": metrics[t]["fold_2024"]["valid_all"]["auc"],
            "brier_R": metrics[t]["fold_2024"]["valid_R"]["brier"],
            "brier_F": metrics[t]["fold_2024"]["valid_F"]["brier"],
        }
        for t in order
    }
    metrics["train_time_min"] = round((time.time() - t_start) / 60, 2)
    metrics["inference_time_sec"] = metrics[best_tag]["fold_2024"]["inference_time_sec"]

    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    np.save(os.path.join(OUT_DIR, "valid_2024_row_id.npy"), df.loc[va24, "row_id"].to_numpy())
    print(f"\nfold_2024 {PRIMARY} summary:", flush=True)
    for t in order:
        s = metrics["summary_fold_2024"][t]
        print(f"  {t:4s} brier={s['brier_all']:.6f} bss={s['bss_all']:+.6f} "
              f"score={s['score_all']:.1f} auc={s['auc_all']:.6f}", flush=True)
    print(
        f"\n[{time.strftime('%H:%M:%S')}] DONE total {metrics['train_time_min']} min "
        f"headline={best_tag} fold_2024 {PRIMARY} brier="
        f"{metrics['fold_2024']['brier']:.6f} (base {BASE_REF[PRIMARY]['fold_2024']:.6f})",
        flush=True,
    )


if __name__ == "__main__":
    main()
