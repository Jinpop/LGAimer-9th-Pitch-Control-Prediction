"""exp_014 — failure-SHAPE features on top of the exp_010 D base.

Spec: experiments/exp_014/config.yaml (campaign 5, user proposal).

Idea
----
EDA §1 measured the career identity  success + reverse + middle ~= 0.900
with corr(success, reverse) = -0.874.  Two pitchers with the SAME career
success rate can therefore fail in completely different ways:

    pitcher A  success .50  reverse .35  middle .05   -> misses to the sides
    pitcher B  success .50  reverse .05  middle .35   -> leaves it over the middle

`asof_pitcher_success_rate` alone cannot tell them apart.  The ratio
    fail_shape = reverse / (reverse + middle)
is an oblique boundary in the (reverse, middle) plane, which an axis-aligned
tree can only approximate with a staircase of many splits; handing it over as
an explicit column lets one split do the job.  `residual` is the part of the
0.90 identity that is left over — see the preliminary check (src/prelim.py):
it turns out to be the rate of a THIRD failure mode that is neither reverse
nor middle, and it varies more across pitchers (CV 0.24) than the success rate
itself does (CV 0.10).

Base
----
exp_010 D — the current best: LightGBM, 47 provided columns, IDs as plain
numerics, num_leaves 15 / min_data_in_leaf 5000 / lr 0.01 / ff 0.7, cap 3000
rounds, exp_004 training mask (season<=2022 & game_type=='F' dropped from
TRAIN only).  fold_2024 valid_all Brier 0.248261 at iter 750 under P4-all.
A0 reproduces that number exactly before anything else is believed; the run
aborts if it does not.

The base was deliberately moved off the RandomForest (exp_009 P0b) because
exp_009 and exp_012 both showed features that looked fine on a weak base and
died on a strong one.

Arms (cumulative)
-----------------
  A0  control, 47 features
  A1  + fail_shape                                              (48)
  A2  + residual                            <- the proposal body (49)
  A3  + middle_share_of_fail, recent_shape_prev{1,3,5}, shape_drift  (54)
  A4  + batter_shape, matchup_shape                             (56)

Iteration policy: exp_006 P4 measured on valid_all — each arm re-measures its
OWN fold_2023 valid_all Brier argmin and carries it to fold_2024.  The number
of rounds is never chosen on the fold being reported.

Seed repeat: after the seed-42 pass, A0 / A2 / the best arm are re-run at
seeds 7 and 2024 so the arm-vs-control delta can be compared against the seed
spread rather than asserted from one draw.  A seed repeat only needs fold_2023
(to measure that seed's argmin) and fold_2024 (to report at it).

Environment: A5000, ~/SungjinPark/venv (Python 3.11.15 / numpy 1.26.4 /
pandas 2.0.3 / sklearn 1.8.0 / lightgbm 4.7.0).
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import ARM_FEATURES, ALL_NEW, add_shape_features  # noqa: E402

SEED = 42
SEED_REPEATS = [7, 2024]
N_THREADS = 6

DATA_PATH = os.path.expanduser("~/SungjinPark/data/train.csv")
OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_014")

TARGET = "control_success"
DROP_COLS = ["row_id", TARGET]
CAT_COLS = ["top_bottom", "game_type", "base_state"]

FOLDS = [
    ("fold_2022", 2021, 2022),
    ("fold_2023", 2022, 2023),
    ("fold_2024", 2023, 2024),
]
F_EXCLUDE_MAX_SEASON = 2022
EXPECTED_2024 = {"valid_all": 253507, "valid_R": 223497}

# exp_010 D, verbatim (metrics.json -> D.lgb_params, with seed swapped per run)
D_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.01,
    "num_leaves": 15,
    "min_data_in_leaf": 5000,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": SEED,
    "num_threads": N_THREADS,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}
N_ROUNDS = 3000

# A0 must land exactly here, otherwise the whole comparison is meaningless.
BASE_REFERENCE = {
    "source": "exp_010 metrics.json -> summary_fold_2024_valid_ALL_PRIMARY.D_P4-all",
    "fold_2024_valid_all_brier": 0.248261,
    "fold_2024_valid_all_score": 619.031,
    "fold_2024_valid_all_auc": 0.545191,
    "fold_2024_iter": 750,
    "fold_2023_valid_all_argmin_iter": 750,
    "fold_2023_valid_all_brier_at_own_argmin": 0.248805,
    "tolerance_brier": 5e-6,
}
TARGET_LINE = {
    "lb_1061_local_brier_range": [0.24776, 0.24780],
    "conversion_factor_measured": 1.293,
}

N_BUCKET_EDGES = [-1, 0, 100, 500, 2000, 10000, 10 ** 9]
N_BUCKET_LABELS = ["n=0", "1-100", "101-500", "501-2000", "2001-10000", ">10000"]

SMOKE = os.environ.get("EXP014_SMOKE") == "1"
if SMOKE:
    N_ROUNDS = 30
    SEED_REPEATS = [7]
    OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_014_smoke")


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
    return sorted(set([k for k in g if k <= n_max] + [n_max]))


def raw_score_curve(booster, X, grid):
    """(k, raw_margin_of_first_k_trees) by accumulating tree chunks.

    Identical to exp_010: a fixed-seed booster's first k trees ARE the k-round
    model and raw margins are additive across tree ranges, so the whole
    Brier-vs-iteration curve costs one pass over the trees instead of one
    full predict per grid point.
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


def nearest_grid_iter(grid, k):
    return min(grid, key=lambda g: (abs(g - k), g))


# --------------------------------------------------------------------------- #
def train_fold(df, feat_cols, params, fold_name, train_until, valid_year,
               f_excl, tag):
    tr_mask = (df["season"] <= train_until) & ~f_excl
    va_mask = df["season"] == valid_year

    X_tr = df.loc[tr_mask, feat_cols]
    y_tr = df.loc[tr_mask, TARGET].to_numpy()
    X_va = df.loc[va_mask, feat_cols]
    y_va = df.loc[va_mask, TARGET].to_numpy()
    is_r = (df.loc[va_mask, "game_type"] == "R").to_numpy()

    print(
        f"[{time.strftime('%H:%M:%S')}] {tag}/{fold_name}: train {X_tr.shape[0]:,}"
        f" x {X_tr.shape[1]} -> valid {X_va.shape[0]:,} ({valid_year})",
        flush=True,
    )
    if valid_year == 2024:
        assert int(va_mask.sum()) == EXPECTED_2024["valid_all"], int(va_mask.sum())
        assert int(is_r.sum()) == EXPECTED_2024["valid_R"], int(is_r.sum())

    cat = [c for c in CAT_COLS if c in feat_cols]
    t0 = time.time()
    dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat)
    booster = lgb.train(params, dtr, num_boost_round=N_ROUNDS)
    train_sec = time.time() - t0

    grid = curve_grid(N_ROUNDS)
    t1 = time.time()
    curve_all, curve_r, margins = {}, {}, {}
    for k, raw in raw_score_curve(booster, X_va, grid):
        p = sigmoid(raw)
        curve_all[k] = float(np.mean((p - y_va) ** 2))
        curve_r[k] = float(np.mean((p[is_r] - y_va[is_r]) ** 2))
        margins[k] = raw.copy()
    curve_sec = time.time() - t1

    argmin_all = min(curve_all, key=lambda k: curve_all[k])
    print(
        f"  trained {train_sec/60:.2f} min, curve {curve_sec:.0f}s; "
        f"valid_all argmin iter={argmin_all} -> {curve_all[argmin_all]:.6f}",
        flush=True,
    )
    return {
        "fold_name": fold_name,
        "valid_year": valid_year,
        "n_train": int(tr_mask.sum()),
        "n_features": len(feat_cols),
        "train_time_min": round(train_sec / 60, 2),
        "curve_time_sec": round(curve_sec, 1),
        "curve_valid_all_brier": {str(k): round(v, 6) for k, v in curve_all.items()},
        "curve_valid_R_brier": {str(k): round(v, 6) for k, v in curve_r.items()},
        "argmin_iter_all": int(argmin_all),
        "argmin_brier_all": round(curve_all[argmin_all], 6),
        "_grid": grid,
        "_margins": margins,
        "_y_va": y_va,
        "_is_r": is_r,
        "_va_mask": va_mask,
        "_booster": booster,
    }


def report_at(fold, iter_requested, iter_source):
    grid = fold["_grid"]
    k = nearest_grid_iter(grid, iter_requested)
    p = sigmoid(fold["_margins"][k])
    y_va, is_r = fold["_y_va"], fold["_is_r"]
    return {
        "iter_used": int(k),
        "iter_requested": int(iter_requested),
        "iter_source": iter_source,
        "n_train": fold["n_train"],
        "n_features": fold["n_features"],
        "valid_all": seg_metrics(y_va, p),
        "valid_R": seg_metrics(y_va[is_r], p[is_r]),
        "valid_F": seg_metrics(y_va[~is_r], p[~is_r]),
        "oracle_iter_all": fold["argmin_iter_all"],
        "oracle_brier_valid_all": fold["argmin_brier_all"],
        "policy_cost_vs_oracle_all": round(
            fold["curve_valid_all_brier"][str(k)] - fold["argmin_brier_all"], 6
        ),
        "train_time_min": fold["train_time_min"],
    }, p


def bucket_brier(y, p, nb):
    """Per asof_pitcher_n bucket Brier — ratio features are pure noise at low n,
    so an overall improvement that lives entirely in the low-n buckets would be
    small-sample over-fitting, not signal."""
    out = {}
    for lab in N_BUCKET_LABELS:
        sel = (nb == lab).to_numpy()
        if sel.sum() == 0:
            out[lab] = {"n": 0}
            continue
        out[lab] = {
            "n": int(sel.sum()),
            "brier": round(float(np.mean((p[sel] - y[sel]) ** 2)), 6),
            "base_rate": round(float(np.mean(y[sel])), 6),
        }
    return out


# --------------------------------------------------------------------------- #
def run_arm(df, arm, feat_cols, f_excl, seed, folds_needed):
    params = dict(D_PARAMS, seed=seed)
    tag = f"{arm}/s{seed}"
    folds = {}
    for fold_name, tu, vy in FOLDS:
        if fold_name not in folds_needed:
            continue
        folds[fold_name] = train_fold(df, feat_cols, params, fold_name, tu, vy,
                                      f_excl, tag)

    out = {
        "arm": arm,
        "seed": seed,
        "n_features": len(feat_cols),
        "new_features": [c for c in feat_cols if c in ALL_NEW],
        "lgb_params": params,
        "num_boost_round_cap": N_ROUNDS,
        "iter_policy": "exp_006 P4 measured on valid_all "
                       "(previous fold's valid_all Brier argmin, no scaling)",
    }
    prev = None
    for fold_name, _, _ in FOLDS:
        if fold_name not in folds:
            prev = None if fold_name != "fold_2023" else prev
            continue
        fd = folds[fold_name]
        if prev is None:
            rep, p = report_at(fd, fd["argmin_iter_all"],
                               "own valid_all argmin (no predecessor / no "
                               "feeder trained; ORACLE, not a reportable number)")
            rep["oracle_flag"] = True
        else:
            rep, p = report_at(fd, prev[1],
                               f"P4-all from {prev[0]} valid_all argmin")
            rep["oracle_flag"] = False
        out[fold_name] = rep
        prev = (fold_name, fd["argmin_iter_all"])
        if fold_name == "fold_2024":
            out["_p2024"] = p

    out["curves"] = {
        fn: {"grid": folds[fn]["_grid"],
             "valid_all_brier": folds[fn]["curve_valid_all_brier"],
             "valid_R_brier": folds[fn]["curve_valid_R_brier"]}
        for fn in folds
    }

    if "fold_2024" in folds:
        fd24 = folds["fold_2024"]
        it = out["fold_2024"]["iter_used"]
        imp = sorted(
            zip(feat_cols, fd24["_booster"].feature_importance("gain", iteration=it)),
            key=lambda t: -t[1],
        )
        tot = sum(g for _, g in imp) or 1.0
        rank = {f: i + 1 for i, (f, _) in enumerate(imp)}
        out["fold_2024_gain_top15"] = [
            [f, round(float(g), 1), round(100.0 * g / tot, 2)] for f, g in imp[:15]
        ]
        out["fold_2024_gain_new_features"] = {
            f: {"rank": rank[f], "gain": round(float(g), 1),
                "share_pct": round(100.0 * g / tot, 2)}
            for f, g in imp if f in ALL_NEW
        }
        out["fold_2024_gain_share_new_pct"] = round(
            100.0 * sum(g for f, g in imp if f in ALL_NEW) / tot, 2)
        out["_va_mask_2024"] = fd24["_va_mask"]
        out["_y2024"] = fd24["_y_va"]

    for fd in folds.values():
        fd.pop("_margins", None)
        fd.pop("_booster", None)
    return out


# --------------------------------------------------------------------------- #
def main():
    t_start = time.time()
    np.random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] loading {DATA_PATH}", flush=True)
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    print(f"loaded: {df.shape}", flush=True)
    assert df["row_id"].is_unique
    assert df["season"].is_monotonic_increasing, "file order assumption broken"

    base_feats = [c for c in df.columns if c not in DROP_COLS]
    assert len(base_feats) == 47, len(base_feats)

    add_shape_features(df)
    nan_report = {
        c: {"n_nan": int(df[c].isna().sum()),
            "pct_nan": round(100.0 * float(df[c].isna().mean()), 4),
            "mean": round(float(df[c].mean()), 6),
            "std": round(float(df[c].std()), 6)}
        for c in ALL_NEW
    }
    print("new-feature NaN / moments: " + json.dumps(nan_report), flush=True)

    for c in CAT_COLS:
        df[c] = df[c].astype("category")

    f_excl = (df["season"] <= F_EXCLUDE_MAX_SEASON) & (df["game_type"] == "F")
    print(f"exp_004 mask: {int(f_excl.sum()):,} training rows excluded", flush=True)

    nb_all = pd.cut(df["asof_pitcher_n"], bins=N_BUCKET_EDGES,
                    labels=N_BUCKET_LABELS)

    metrics = {
        "exp_id": "exp_014",
        "base_exp": "exp_010 D (strongly regularised LightGBM, 47 features)",
        "hypothesis": "explicit failure-SHAPE ratios separate pitchers that share "
                      "a success rate but fail in different ways",
        "seed_primary": SEED,
        "seed_repeats": SEED_REPEATS,
        "num_threads": N_THREADS,
        "mask": "training rows with season<=2022 & game_type=='F' dropped "
                "(exp_004); valid untouched",
        "n_rows_masked": int(f_excl.sum()),
        "base_reference": BASE_REFERENCE,
        "targets": TARGET_LINE,
        "new_feature_nan_and_moments": nan_report,
        "arms": {a: {"n_features": 47 + len(v), "added": v}
                 for a, v in ARM_FEATURES.items()},
        "sha256": {
            "train.csv": sha256_file(DATA_PATH),
            "train.py": sha256_file(os.path.abspath(__file__)),
            "features.py": sha256_file(
                os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "features.py")),
        },
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "lightgbm": lgb.__version__,
        },
    }

    def save():
        clean = json.loads(json.dumps(metrics, default=str))
        with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
            json.dump(clean, f, indent=2, ensure_ascii=False)

    # ---------------- seed-42 pass over every arm ------------------------- #
    results, preds = {}, {}
    all_folds = {"fold_2022", "fold_2023", "fold_2024"}
    for arm in ["A0", "A1", "A2", "A3", "A4"]:
        feat_cols = base_feats + ARM_FEATURES[arm]
        print(f"[{time.strftime('%H:%M:%S')}] ===== {arm} "
              f"({len(feat_cols)} features: +{ARM_FEATURES[arm]}) =====", flush=True)
        r = run_arm(df, arm, feat_cols, f_excl, SEED, all_folds)
        preds[arm] = r.pop("_p2024")
        y2024 = r.pop("_y2024")
        va_mask = r.pop("_va_mask_2024")
        nb24 = nb_all[va_mask]
        r["fold_2024_brier_by_asof_pitcher_n"] = bucket_brier(y2024, preds[arm], nb24)
        results[arm] = r
        metrics[arm] = r

        # ---- A0 control-reproduction gate -------------------------------- #
        if arm == "A0":
            got = r["fold_2024"]["valid_all"]["brier"]
            exp = BASE_REFERENCE["fold_2024_valid_all_brier"]
            d = abs(got - exp)
            metrics["A0_reproduction_check"] = {
                "expected_fold_2024_valid_all_brier": exp,
                "got": got,
                "abs_diff": round(d, 8),
                "expected_iter": BASE_REFERENCE["fold_2024_iter"],
                "got_iter": r["fold_2024"]["iter_used"],
                "expected_fold_2023_argmin": BASE_REFERENCE[
                    "fold_2023_valid_all_argmin_iter"],
                "got_fold_2023_argmin": r["fold_2023"]["oracle_iter_all"],
                "passed": bool(d <= BASE_REFERENCE["tolerance_brier"]),
            }
            print("A0 reproduction: " +
                  json.dumps(metrics["A0_reproduction_check"]), flush=True)
            save()
            if SMOKE:
                print("  (SMOKE: reproduction gate not enforced — 30 rounds is "
                      "not the real configuration)", flush=True)
            else:
                assert d <= BASE_REFERENCE["tolerance_brier"], (
                    f"A0 failed to reproduce exp_010 D: got {got}, expected "
                    f"{exp}. Everything downstream would be uninterpretable "
                    "— aborting."
                )
        np.save(os.path.join(OUT_DIR, f"pred_{arm}_s{SEED}_fold_2024.npy"), preds[arm])
        metrics["elapsed_min"] = round((time.time() - t_start) / 60, 2)
        save()

    # ---------------- seed-42 comparison table ---------------------------- #
    base_b = results["A0"]["fold_2024"]["valid_all"]["brier"]
    y2024 = None
    table = {}
    for arm in ["A0", "A1", "A2", "A3", "A4"]:
        v = results[arm]["fold_2024"]["valid_all"]
        table[arm] = {
            "n_features": results[arm]["n_features"],
            "iter": results[arm]["fold_2024"]["iter_used"],
            "fold_2023_argmin_iter": results[arm]["fold_2023"]["oracle_iter_all"],
            "valid_all_brier": v["brier"],
            "valid_all_bss": v["bss"],
            "valid_all_score": v["score"],
            "valid_all_auc": v.get("auc"),
            "valid_R_brier": results[arm]["fold_2024"]["valid_R"]["brier"],
            "valid_F_brier": results[arm]["fold_2024"]["valid_F"]["brier"],
            "delta_vs_A0": round(v["brier"] - base_b, 6),
            "score_delta_vs_A0": round(
                v["score"] - results["A0"]["fold_2024"]["valid_all"]["score"], 2),
            "fold_2023_valid_all_brier": results[arm]["fold_2023"]["valid_all"]["brier"],
            "fold_2022_valid_all_brier": results[arm]["fold_2022"]["valid_all"]["brier"],
            "gain_share_new_pct": results[arm].get("fold_2024_gain_share_new_pct", 0.0),
        }
    metrics["summary_seed42_fold_2024_valid_all"] = table
    print("--- seed 42 summary ---", flush=True)
    print(json.dumps(table, indent=2), flush=True)
    save()

    # ---- n-bucket delta decomposition ------------------------------------ #
    decomp = {}
    for arm in ["A1", "A2", "A3", "A4"]:
        a0 = results["A0"]["fold_2024_brier_by_asof_pitcher_n"]
        ax = results[arm]["fold_2024_brier_by_asof_pitcher_n"]
        decomp[arm] = {
            lab: {
                "n": a0[lab]["n"],
                "A0_brier": a0[lab].get("brier"),
                f"{arm}_brier": ax[lab].get("brier"),
                "delta": (round(ax[lab]["brier"] - a0[lab]["brier"], 6)
                          if a0[lab]["n"] else None),
                "share_of_total_delta_pct": None,
            }
            for lab in N_BUCKET_LABELS
        }
        tot_n = sum(a0[l]["n"] for l in N_BUCKET_LABELS)
        tot_d = table[arm]["delta_vs_A0"]
        for lab in N_BUCKET_LABELS:
            e = decomp[arm][lab]
            if e["n"] and tot_d != 0:
                e["share_of_total_delta_pct"] = round(
                    100.0 * (e["delta"] * e["n"] / tot_n) / tot_d, 1)
    metrics["fold_2024_delta_decomposition_by_asof_pitcher_n"] = decomp
    print("--- n-bucket decomposition ---", flush=True)
    print(json.dumps(decomp, indent=2), flush=True)
    save()

    # ---------------- seed repeats ---------------------------------------- #
    ranked = sorted(["A1", "A2", "A3", "A4"],
                    key=lambda a: table[a]["valid_all_brier"])
    repeat_arms = sorted(set(["A0", "A2", ranked[0]]),
                         key=lambda a: ["A0", "A1", "A2", "A3", "A4"].index(a))
    metrics["seed_repeat_arms"] = repeat_arms
    print(f"[{time.strftime('%H:%M:%S')}] ===== seed repeats on {repeat_arms} "
          f"(best seed-42 arm = {ranked[0]}) =====", flush=True)

    seed_tab = {a: {str(SEED): table[a]["valid_all_brier"]} for a in repeat_arms}
    seed_iter = {a: {str(SEED): table[a]["iter"]} for a in repeat_arms}
    for s in SEED_REPEATS:
        for arm in repeat_arms:
            feat_cols = base_feats + ARM_FEATURES[arm]
            r = run_arm(df, arm, feat_cols, f_excl, s, {"fold_2023", "fold_2024"})
            r.pop("_p2024", None)
            r.pop("_y2024", None)
            r.pop("_va_mask_2024", None)
            r.pop("curves", None)
            metrics[f"{arm}_seed{s}"] = r
            seed_tab[arm][str(s)] = r["fold_2024"]["valid_all"]["brier"]
            seed_iter[arm][str(s)] = r["fold_2024"]["iter_used"]
            metrics["elapsed_min"] = round((time.time() - t_start) / 60, 2)
            save()

    seeds_all = [str(SEED)] + [str(s) for s in SEED_REPEATS]
    seed_summary = {"brier_by_seed": seed_tab, "iter_by_seed": seed_iter}
    for arm in repeat_arms:
        vals = [seed_tab[arm][s] for s in seeds_all]
        seed_summary.setdefault("spread", {})[arm] = {
            "mean": round(float(np.mean(vals)), 6),
            "std": round(float(np.std(vals, ddof=1)), 6),
            "min": round(float(min(vals)), 6),
            "max": round(float(max(vals)), 6),
            "range": round(float(max(vals) - min(vals)), 6),
        }
    for arm in repeat_arms:
        if arm == "A0":
            continue
        d = [seed_tab[arm][s] - seed_tab["A0"][s] for s in seeds_all]
        seed_summary.setdefault("paired_delta_vs_A0", {})[arm] = {
            "per_seed": {s: round(seed_tab[arm][s] - seed_tab["A0"][s], 6)
                         for s in seeds_all},
            "mean_delta": round(float(np.mean(d)), 6),
            "std_delta": round(float(np.std(d, ddof=1)), 6),
            "all_seeds_same_sign": bool(all(x < 0 for x in d) or all(x > 0 for x in d)),
            "mean_delta_abs_over_A0_seed_std": (
                round(abs(float(np.mean(d))) / seed_summary["spread"]["A0"]["std"], 2)
                if seed_summary["spread"]["A0"]["std"] > 0 else None),
        }
    metrics["seed_summary"] = seed_summary
    print("--- seed summary ---", flush=True)
    print(json.dumps(seed_summary, indent=2), flush=True)

    metrics["train_time_min"] = round((time.time() - t_start) / 60, 2)
    save()
    print(f"[{time.strftime('%H:%M:%S')}] DONE total "
          f"{metrics['train_time_min']} min", flush=True)


if __name__ == "__main__":
    main()
