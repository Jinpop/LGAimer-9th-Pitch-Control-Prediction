"""exp_021 — domain-hierarchy feature ladder (plan N step 1).

Spec: experiments/exp_021/config.yaml.  ONE process = ONE (arm, seed) job, three
folds; results land in res_{arm}_seed{seed}.json so that several jobs can run
concurrently (efficiency rule #2, 2026-08-07: up to 3 concurrent fits, each
pinned to num_threads=6 so the numbers stay comparable with every earlier run).

Question
--------
The 47 provided columns are "everything the organisers handed over", not
"everything with a mechanism".  Arms A/B/C are pre-registered TIERS (config
feature_tiers, assigned by baseball domain reasoning BEFORE any gain was
looked at); D is the current 47-column control and the reproduction gate.

  A  T3 only                       8 cols
  B  T3 + T2                      24 cols
  C  T3 + T2 + T1                 36 cols   <- the candidate; C - D is the metric
  D  all 47 provided columns              <- control, reproduction gate

The ONLY variable across arms is the column set.  Hyper-parameters, seed,
training mask, iteration policy and categorical encoding are byte-identical.

Iteration policy: P10 (exp_017) — each arm carries its OWN previous fold's
valid_all Brier argmin, scaled by c* = 2.75.  fold_2022 has no predecessor and
is reported at its own argmin (oracle, flagged, not comparable across
experiments).  Every arm additionally reports the unscaled P4 shadow because the
reproduction gate for D is stated in P4 terms (exp_010 D = 0.248261 at iter
750).  valid 2024 is never consulted to choose anything.

Categorical encoding
--------------------
top_bottom / game_type / base_state are mapped through the vocabulary of the
fold's TRAINING rows (sorted unique) into integer codes carried as float64, and
declared to LightGBM via `categorical_feature` — this is byte-for-byte the code
path of submissions/D_b3_stack/build/train_final.py (the deployed D/F builder)
and is provably identical to the pandas-category path exp_010 D / exp_018 A0
used, because the training vocabulary equals the full-file vocabulary for every
column that is not constant in the training window.  See the NOTE in report.md:
config.yaml's `encoding_unchanged` prose says "no categorical_feature
declaration", but the deployed F builder DOES declare it (train_final.py:265)
and the 0.248261 reference was produced with it declared, so the declaration is
kept and the alternative is measured separately as arm D_NOCAT_DIAG.

Environment: A5000, ~/SungjinPark/venv (Python 3.11.15 / numpy 1.26.4 /
pandas 2.0.3 / lightgbm 4.7.0) == evaluation server versions.
"""

import argparse
import gc
import hashlib
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

N_THREADS = 6
DATA_PATH = os.path.expanduser("~/SungjinPark/data/train.csv")
OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_021")

TARGET = "control_success"
DROP_COLS = ["row_id", TARGET]
CAT_COLS = ["top_bottom", "game_type", "base_state"]

FOLDS = [("fold_2022", 2021, 2022), ("fold_2023", 2022, 2023),
         ("fold_2024", 2023, 2024)]
F_EXCLUDE_MAX_SEASON = 2022                       # exp_004 mask
EXPECTED_2024 = {"valid_all": 253507, "valid_R": 223497}

D_PARAMS = {                                      # exp_010 D verbatim
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.01,
    "num_leaves": 15,
    "min_data_in_leaf": 5000,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": 42,
    "num_threads": N_THREADS,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}
P10_C = 2.75

# --------------------------------------------------------------------------- #
# pre-registered tiers (config.yaml feature_tiers, 2026-08-07).  NOT re-derived
# from gain or performance — copied verbatim.
# --------------------------------------------------------------------------- #
T3_CORE = [
    "balls_before", "strikes_before",
    "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
]
T2_USEFUL = [
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
    "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate", "asof_pitcher_pitchmix_n",
    "pitcher_hand", "batter_hand", "game_type", "season", "base_state",
    "num_runners_on",
]
T1_WEAK = [
    "inning", "li", "score_diff_pitcher_team", "run_total_before", "outs_before",
    "top_bottom", "game_month", "asof_batter_success_rate",
    "asof_batter_middle_rate", "asof_batter_n", "batter_id", "pitcher_id",
]
EXCLUDED_11 = [
    "pitcher_team_id", "batter_team_id", "game_dayofweek",
    "runner_on_1b", "runner_on_2b", "runner_on_3b",
    "run_top_before", "run_bot_before", "score_diff_home",
    "home_win_expectancy", "away_win_expectancy",
]
TIER_OF_ARM = {
    "A": set(T3_CORE),
    "B": set(T3_CORE + T2_USEFUL),
    "C": set(T3_CORE + T2_USEFUL + T1_WEAK),
    "D": set(T3_CORE + T2_USEFUL + T1_WEAK + EXCLUDED_11),
}
TIER_OF_ARM["D_NOCAT_DIAG"] = TIER_OF_ARM["D"]
EXPECTED_N_FEATURES = {"A": 8, "B": 24, "C": 36, "D": 47, "D_NOCAT_DIAG": 47}

GATE = {
    "source": "exp_010 metrics.json -> summary_fold_2024_valid_ALL_PRIMARY.D_P4-all",
    "arm": "D",
    "policy": "P4 (unscaled carry) — the policy the reference was measured under",
    "fold_2024_valid_all_brier": 0.248261,
    "fold_2024_iter": 750,
    "tolerance_brier": 5e-6,
    "reference_argmins_valid_all": {"fold_2022": 300, "fold_2023": 750,
                                    "fold_2024": 1550},
}


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
    out = {"n": n, "brier": round(brier, 6),
           "bss": round(float(1.0 - brier / d), 6) if d > 0 else None,
           "score": round(lb_score(brier, r), 3) if d > 0 else None,
           "base_rate": round(r, 6), "pred_mean": round(float(np.mean(p)), 6)}
    if 0.0 < r < 1.0:
        out["auc"] = round(float(roc_auc_score(y, p)), 6)
        out["logloss"] = round(float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15))), 6)
    return out


def curve_grid(n_max):
    g = list(range(1, 21)) + list(range(25, 101, 5)) + list(range(110, 301, 10))
    g += list(range(325, min(n_max, 1000) + 1, 25))
    if n_max > 1000:
        g += list(range(1050, n_max + 1, 50))
    return sorted(set([k for k in g if k <= n_max] + [n_max]))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def nearest_grid_iter(grid, k):
    return min(grid, key=lambda g: (abs(g - k), g))


def raw_margin_curve(booster, X, grid):
    """{k: raw margin of the first k rounds} in one pass over the trees.

    exp_010 / exp_016 / exp_018 machinery, unchanged: for a fixed booster the
    first k rounds ARE the k-round model and raw margins add across ranges.
    """
    sub = X[: min(2000, len(X))]
    a = booster.predict(sub, start_iteration=0, num_iteration=5, raw_score=True)
    b = booster.predict(sub, start_iteration=5, num_iteration=5, raw_score=True)
    full = booster.predict(sub, start_iteration=0, num_iteration=10, raw_score=True)
    resid = full - (a + b)
    const = float(np.median(resid))
    assert np.allclose(resid, const, atol=1e-8), (
        "raw margins are not additive across round ranges",
        float(np.abs(resid - const).max()))
    out, acc, prev = {}, None, 0
    for k in grid:
        chunk = booster.predict(X, start_iteration=prev, num_iteration=k - prev,
                                raw_score=True)
        acc = chunk if acc is None else acc + chunk + const
        prev = k
        out[k] = acc.copy()
    return out


def argmin_of(curve):
    k = min(curve, key=lambda x: curve[x])
    return int(k), float(curve[k])


def build_matrix(df, feats, vocab):
    """float64 design matrix; categoricals -> train-window vocabulary codes.

    Unseen / missing category -> NaN (never a sentinel value).  Identical to
    submissions/D_b3_stack/build/train_final.py::build_matrix except that the
    'no row lost' assertion is relaxed to a COUNT that is reported, because a
    fold whose training window is all-R legitimately has no 'F' in its
    game_type vocabulary (that column is then constant in training and LightGBM
    drops it, so the valid-side NaN cannot change a prediction).
    """
    X = np.empty((len(df), len(feats)), dtype=np.float64)
    unseen = {}
    for j, c in enumerate(feats):
        s = df[c]
        if c in vocab:
            codes = pd.Categorical(s.astype(object),
                                   categories=vocab[c]).codes.astype(np.float64)
            unseen[c] = int(((codes < 0) & s.notna().to_numpy()).sum())
            codes[codes < 0] = np.nan
            X[:, j] = codes
        else:
            X[:, j] = pd.to_numeric(s, errors="raise").to_numpy(dtype=np.float64)
    return X, unseen


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=list(TIER_OF_ARM))
    ap.add_argument("seed", type=int)
    ap.add_argument("--cap", type=int, default=3000)
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()
    arm, seed, cap, out_dir = args.arm, args.seed, args.cap, args.out
    declare_cat = arm != "D_NOCAT_DIAG"

    t_start = time.time()
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "pred"), exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] {arm}/seed{seed} loading {DATA_PATH}",
          flush=True)
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    assert df["row_id"].is_unique
    assert df["season"].is_monotonic_increasing, "file order assumption broken"

    base_feats = [c for c in df.columns if c not in DROP_COLS]
    assert len(base_feats) == 47, len(base_feats)
    # every pre-registered name must exist exactly once, and the four tiers must
    # partition the 47 columns — otherwise the tier table is stale.
    allnames = T3_CORE + T2_USEFUL + T1_WEAK + EXCLUDED_11
    assert len(allnames) == 47 and len(set(allnames)) == 47, len(set(allnames))
    assert set(allnames) == set(base_feats), (
        sorted(set(allnames) ^ set(base_feats)))

    feats = [c for c in base_feats if c in TIER_OF_ARM[arm]]   # file order kept
    assert len(feats) == EXPECTED_N_FEATURES[arm], (len(feats), arm)

    season = df["season"].to_numpy()
    game_type = df["game_type"].to_numpy()
    f_excl = (season <= F_EXCLUDE_MAX_SEASON) & (game_type == "F")
    y_all = df[TARGET].to_numpy(dtype=np.float64)
    assert np.isin(y_all, (0.0, 1.0)).all()

    params = dict(D_PARAMS, seed=seed)
    grid = curve_grid(cap)
    cat_in_arm = [c for c in CAT_COLS if c in feats]

    res = {
        "exp_id": "exp_021", "arm": arm, "seed": seed,
        "n_features": len(feats), "features": feats,
        "categorical_declared": cat_in_arm if declare_cat else [],
        "categorical_present_but_numeric": [] if declare_cat else cat_in_arm,
        "lgb_params": params, "num_boost_round_cap": cap, "p10_c": P10_C,
        "mask": "training rows with season<=2022 & game_type=='F' dropped "
                "(exp_004); validation untouched",
        "n_rows_masked": int(f_excl.sum()),
        "sha256": {"train.csv": sha256_file(DATA_PATH),
                   "train.py": sha256_file(os.path.abspath(__file__))},
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__, "lightgbm": lgb.__version__},
        "gate_spec": GATE,
        "curves": {}, "vocab_per_fold": {}, "unseen_category_rows": {},
    }

    curves, oracles, meta, margins = {}, {}, {}, {}
    for fold_name, tu, vy in FOLDS:
        tr = (season <= tu) & ~f_excl
        va = season == vy
        y_va = y_all[va]
        is_r = (game_type[va] == "R")
        if vy == 2024:
            assert int(va.sum()) == EXPECTED_2024["valid_all"], int(va.sum())
            assert int(is_r.sum()) == EXPECTED_2024["valid_R"], int(is_r.sum())

        # ---- TRAIN-WINDOW category vocabulary (never sees valid rows) ------- #
        fit_df = df.loc[tr]
        vocab = {c: sorted(fit_df[c].dropna().unique().tolist())
                 for c in cat_in_arm}
        for c in cat_in_arm:
            assert fit_df[c].notna().all(), f"unexpected NaN in categorical '{c}'"
        del fit_df
        res["vocab_per_fold"][fold_name] = {c: list(map(str, v))
                                            for c, v in vocab.items()}

        X_tr, u_tr = build_matrix(df.loc[tr], feats, vocab)
        X_va, u_va = build_matrix(df.loc[va], feats, vocab)
        assert all(v == 0 for v in u_tr.values()), u_tr
        # a valid row can legitimately carry a category absent from the training
        # window; record it.  Such a column is constant in training, so LightGBM
        # gives it a single bin and never splits on it.
        res["unseen_category_rows"][fold_name] = u_va
        const_cols = [c for c in cat_in_arm if len(vocab[c]) < 2]
        res.setdefault("constant_in_train_cols", {})[fold_name] = const_cols

        t0 = time.time()
        ds = lgb.Dataset(X_tr, label=y_all[tr], feature_name=feats,
                         categorical_feature=(cat_in_arm if declare_cat else []),
                         free_raw_data=False)
        bst = lgb.train(params, ds, num_boost_round=cap)
        assert bst.num_trees() == cap, (arm, fold_name, bst.num_trees())
        t1 = time.time()
        marg = raw_margin_curve(bst, X_va, grid)
        cv = {k: float(np.mean((sigmoid(marg[k]) - y_va) ** 2)) for k in grid}
        a_min, a_val = argmin_of(cv)
        print(f"  [{arm} s{seed} {fold_name}] fit {(t1-t0)/60:.2f} min, curve "
              f"{time.time()-t1:.0f}s ({int(tr.sum()):,} x {len(feats)}) -> "
              f"valid_all argmin {a_min} = {a_val:.6f}"
              f"{'  *** ARGMIN AT CAP ***' if a_min >= cap else ''}", flush=True)

        curves[fold_name] = cv
        oracles[fold_name] = (a_min, a_val)
        margins[fold_name] = marg
        meta[fold_name] = {"y_va": y_va, "is_r": is_r, "n_train": int(tr.sum()),
                           "fit_min": round((t1 - t0) / 60, 2)}
        res["curves"][fold_name] = {str(k): round(v, 6) for k, v in cv.items()}
        if fold_name == "fold_2024":
            bst24 = bst
        else:
            del bst
        del ds, X_tr, X_va
        gc.collect()

    res["oracle_iters"] = {f: oracles[f][0] for f in oracles}
    res["argmin_at_cap"] = {f: bool(oracles[f][0] >= cap) for f in oracles}
    res["any_argmin_at_cap"] = bool(any(res["argmin_at_cap"].values()))

    def report(fold_name, k, source, oracle_flag):
        m = meta[fold_name]
        p = sigmoid(margins[fold_name][k])
        a_min, a_val = oracles[fold_name]
        return {"iter_used": int(k), "iter_source": source,
                "oracle_flag": oracle_flag, "n_train": m["n_train"],
                "fit_min": m["fit_min"],
                "valid_all": seg_metrics(m["y_va"], p),
                "valid_R": seg_metrics(m["y_va"][m["is_r"]], p[m["is_r"]]),
                "valid_F": seg_metrics(m["y_va"][~m["is_r"]], p[~m["is_r"]]),
                "oracle_iter": a_min, "oracle_brier": round(a_val, 6),
                "argmin_at_cap": bool(a_min >= cap),
                "policy_cost_vs_oracle": round(curves[fold_name][k] - a_val, 6)}, p

    # ---- P10 (own previous fold argmin x 2.75) ---------------------------- #
    prev, p24_p10 = None, None
    for fold_name, _, _ in FOLDS:
        if prev is None:
            rep, p = report(fold_name, oracles[fold_name][0],
                            "own argmin (no predecessor; ORACLE, feeder only)",
                            True)
        else:
            want = int(round(P10_C * oracles[prev][0]))
            k = nearest_grid_iter(grid, want)
            rep, p = report(fold_name, k,
                            f"P10: {oracles[prev][0]} (own {prev} valid_all "
                            f"argmin) x {P10_C}", False)
            rep["iter_requested"] = want
            rep["iter_capped_by_num_boost_round"] = bool(want > cap)
        res[fold_name] = rep
        if fold_name == "fold_2024":
            p24_p10 = p
        prev = fold_name

    # ---- P4 shadow (unscaled carry) — the gate is stated in P4 terms ------- #
    prev, p24_p4 = None, None
    for fold_name, _, _ in FOLDS:
        if prev is None:
            rep, p = report(fold_name, oracles[fold_name][0],
                            "own argmin (ORACLE, feeder only)", True)
        else:
            rep, p = report(fold_name, nearest_grid_iter(grid, oracles[prev][0]),
                            f"P4: {oracles[prev][0]} (own {prev} argmin), "
                            f"no scaling", False)
        res[fold_name + "_P4"] = rep
        if fold_name == "fold_2024":
            p24_p4 = p
        prev = fold_name

    # ---- reproduction gate (arm D, seed 42, P4) ---------------------------- #
    if arm == "D" and seed == 42:
        got = res["fold_2024_P4"]["valid_all"]["brier"]
        d = abs(got - GATE["fold_2024_valid_all_brier"])
        res["reproduction_gate"] = {
            "expected": GATE["fold_2024_valid_all_brier"], "got": got,
            "abs_diff": float(f"{d:.10g}"),
            "expected_iter": GATE["fold_2024_iter"],
            "got_iter": res["fold_2024_P4"]["iter_used"],
            "expected_argmins": GATE["reference_argmins_valid_all"],
            "got_argmins": res["oracle_iters"],
            "policy": GATE["policy"],
            "passed": bool(d <= GATE["tolerance_brier"])}
        print("GATE: " + json.dumps(res["reproduction_gate"]), flush=True)

    # ---- gain importance at the reported (P10) fold_2024 iteration --------- #
    imp = sorted(zip(feats, bst24.feature_importance(
        "gain", iteration=res["fold_2024"]["iter_used"])), key=lambda t: -t[1])
    tot = sum(g for _, g in imp) or 1.0
    res["fold_2024_gain_pct"] = {f: round(100.0 * g / tot, 3) for f, g in imp}
    del bst24

    pd_ = os.path.join(out_dir, "pred")
    np.save(os.path.join(pd_, f"pred_{arm}_seed{seed}_fold_2024_P10.npy"), p24_p10)
    np.save(os.path.join(pd_, f"pred_{arm}_seed{seed}_fold_2024_P4.npy"), p24_p4)
    res["pred_files"] = {
        "P10": f"pred/pred_{arm}_seed{seed}_fold_2024_P10.npy",
        "P4": f"pred/pred_{arm}_seed{seed}_fold_2024_P4.npy"}
    res["elapsed_min"] = round((time.time() - t_start) / 60, 2)

    path = os.path.join(out_dir, f"res_{arm}_seed{seed}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"[{time.strftime('%H:%M:%S')}] {arm}/seed{seed} DONE "
          f"{res['elapsed_min']} min -> {path}", flush=True)


if __name__ == "__main__":
    main()
