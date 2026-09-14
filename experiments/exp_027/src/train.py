"""exp_027 — base-model FAMILY screen: LightGBM vs XGBoost vs CatBoost, flat.

Spec: experiments/exp_027/config.yaml.  ONE process = ONE (arm, seed, fold) job.

What is inherited BYTE-FOR-BYTE from exp_021/src/train.py
  (sha256 c10c5d01dfa91105d1272fd61598a2d686c1acc0124637927350428793b46d28)
      sha256_file · lb_score · seg_metrics · curve_grid · sigmoid ·
      nearest_grid_iter · argmin_of · build_matrix · the fold masks (FOLDS) ·
      the exp_004 F-mask · the train-window categorical vocabulary rule ·
      the raw-margin chunk-accumulation curve (generalised below to a callable
      so all three families share ONE curve routine)

What is NEW
      exactly two thin family adapters, `fit_xgboost` and `fit_catboost`,
      plus `fit_lightgbm` which is exp_021's fit unchanged.  Each adapter takes
      the SAME design matrix and returns the SAME thing: a callable
      predict_range(a, b) -> raw margin of trees [a, b).  Nothing outside the
      adapters differs between families.

Stage A   --fold fold_2023  (train <=2022, 870,752 rows), seed 42, cap 3000.
          Reported at the fold's OWN valid_all argmin.  Selection happens here
          and only here.
Stage B   --fold fold_2024  (train <=2023, 1,116,277 rows), seeds 42/7/2024,
          --iter K fixed from Stage A's fold_2023 argmin x 1.281969.  The model
          is trained for exactly K rounds; fold_2024 valid never chooses K.

Environment: A5000, ~/SungjinPark/venv — Python 3.11.15 / numpy 1.26.4 /
pandas 2.0.3 / lightgbm 4.7.0 / xgboost 3.2.0 / catboost 1.2.10.  CPU only,
6 threads per fit (efficiency rule #2), deterministic settings for all three.
"""

import argparse
import gc
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

N_THREADS = 6
DATA_PATH = os.path.expanduser("~/SungjinPark/data/train.csv")
OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_027")

TARGET = "control_success"
DROP_COLS = ["row_id", TARGET]
CAT_COLS = ["top_bottom", "game_type", "base_state"]
ID_COLS = ["pitcher_id", "batter_id"]

FOLDS = {"fold_2022": (2021, 2022), "fold_2023": (2022, 2023),
         "fold_2024": (2023, 2024)}
F_EXCLUDE_MAX_SEASON = 2022                        # exp_004 mask
EXPECTED_N_TRAIN = {"fold_2022": 653728, "fold_2023": 870752,
                    "fold_2024": 1116277}
EXPECTED_2024 = {"valid_all": 253507, "valid_R": 223497}
SCALE_2024_OVER_2023 = 1.281969                    # 1116277 / 870752

INHERITED_FROM = {
    "file": "experiments/exp_021/src/train.py",
    "sha256": "c10c5d01dfa91105d1272fd61598a2d686c1acc0124637927350428793b46d28",
    "functions": ["sha256_file", "lb_score", "seg_metrics", "curve_grid",
                  "sigmoid", "nearest_grid_iter", "argmin_of", "build_matrix",
                  "FOLDS masks", "exp_004 F-mask", "train-window vocabulary",
                  "raw-margin chunk accumulation"],
}

# free internal reproduction gate: exp_021 arm D == exp_027 arm L1, same seed,
# same fold, same 47 columns, same params.  metrics.json records fold_2023
# argmin 750 and report-at-825 Brier 0.248809 with policy_cost 4e-06.
L1_GATE = {"source": "exp_021 metrics.json per_arm_seed.D_seed42",
           "fold_2023_argmin_iter": 750,
           "fold_2023_brier_at_825": 0.248809,
           "tolerance_brier": 5e-6}

# --------------------------------------------------------------------------- #
# arm table.  `changed_vs` lists EVERY parameter that differs from the stated
# reference arm — the variable ledger the discipline rule demands.
# --------------------------------------------------------------------------- #
LGB_BASE = {                                       # == exp_010 D == exp_021 D
    "objective": "binary", "metric": "binary_logloss",
    "learning_rate": 0.01, "num_leaves": 15, "min_data_in_leaf": 5000,
    "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 1,
    "num_threads": N_THREADS, "deterministic": True, "force_row_wise": True,
    "verbose": -1,
}
XGB_BASE = {
    "objective": "binary:logistic", "eval_metric": "logloss",
    "tree_method": "hist", "device": "cpu", "nthread": N_THREADS,
    "max_depth": 4, "min_child_weight": 5000, "eta": 0.01, "reg_lambda": 10,
}
CB_BASE = {
    "loss_function": "Logloss", "task_type": "CPU", "thread_count": N_THREADS,
    "boosting_type": "Plain", "bootstrap_type": "Bernoulli", "subsample": 0.8,
    "learning_rate": 0.03, "depth": 6, "l2_leaf_reg": 10,
    "one_hot_max_size": 10, "has_time": True,
    "allow_writing_files": False, "verbose": False,
}

ARMS = {
    # ---- LightGBM ---------------------------------------------------------- #
    "L1": {"family": "lightgbm", "params": {},
           "role": "CONTROL — exp_010 D reproduced", "changed_vs": {}},
    "L2": {"family": "lightgbm",
           "params": {"num_leaves": 7, "min_data_in_leaf": 20000},
           "role": "regularisation axis, step 1",
           "changed_vs": {"L1": ["num_leaves 15->7", "min_data_in_leaf 5000->20000"]}},
    "L3": {"family": "lightgbm",
           "params": {"num_leaves": 3, "min_data_in_leaf": 50000,
                      "learning_rate": 0.02},
           "role": "regularisation axis, step 2 (extreme)",
           "changed_vs": {"L1": ["num_leaves 15->3", "min_data_in_leaf 5000->50000",
                                 "learning_rate 0.01->0.02"]}},
    "L4": {"family": "lightgbm",
           "params": {"feature_fraction": 0.4, "bagging_fraction": 0.7,
                      "bagging_freq": 1},
           "role": "variance reduction, not regularisation",
           "changed_vs": {"L1": ["feature_fraction 0.7->0.4",
                                 "bagging_fraction 0.8->0.7"]}},
    "L5": {"family": "lightgbm",
           "params": {"lambda_l2": 100, "min_gain_to_split": 1e-3},
           "role": "leaf-value shrinkage",
           "changed_vs": {"L1": ["lambda_l2 0->100", "min_gain_to_split 0->1e-3"]}},
    "L6": {"family": "lightgbm", "params": {"linear_tree": True},
           "role": "linear fit in the leaves",
           "changed_vs": {"L1": ["linear_tree False->True"]}},
    # ---- XGBoost ----------------------------------------------------------- #
    "X1": {"family": "xgboost", "params": {},
           "role": "XGB reference", "changed_vs": {}},
    "X2": {"family": "xgboost",
           "params": {"max_depth": 3, "min_child_weight": 20000},
           "role": "regularisation axis",
           "changed_vs": {"X1": ["max_depth 4->3", "min_child_weight 5000->20000"]}},
    "X3": {"family": "xgboost",
           "params": {"colsample_bytree": 0.5, "subsample": 0.7},
           "role": "variance reduction",
           "changed_vs": {"X1": ["colsample_bytree 1.0->0.5", "subsample 1.0->0.7"]}},
    "X4": {"family": "xgboost",
           "params": {"grow_policy": "lossguide", "max_leaves": 15,
                      "max_depth": 0, "min_child_weight": 5000, "eta": 0.01},
           "role": "PURE-FAMILY CONTRAST vs L1 (leaf-wise, 15 leaves)",
           "changed_vs": {"X1": ["grow_policy depthwise->lossguide",
                                 "max_leaves 0->15", "max_depth 4->0"]}},
    "X5": {"family": "xgboost",
           "params": {"gamma": 1.0, "max_delta_step": 1},
           "role": "split-gain threshold + step cap",
           "changed_vs": {"X1": ["gamma 0->1.0", "max_delta_step 0->1"]}},
    # X4H is a DIAGNOSTIC arm added by the runner, NOT one of the 17 pre-
    # registered arms and NOT eligible for promotion (same status exp_021 gave
    # D_NOCAT_DIAG).  Reason: XGBoost's min_child_weight is a sum of HESSIANS,
    # and for binary:logistic at p~0.5 the hessian is ~0.25 per row, so
    # min_child_weight 5000 is a ~20,000-ROW constraint — 4x stricter than
    # LightGBM's min_data_in_leaf 5000.  Without this arm the "pure family"
    # delta L1-X4 silently conflates family with a 4x leaf-size constraint.
    "X4H": {"family": "xgboost",
            "params": {"grow_policy": "lossguide", "max_leaves": 15,
                       "max_depth": 0, "min_child_weight": 1250, "eta": 0.01,
                       "colsample_bytree": 0.7, "subsample": 0.8},
            "role": "DIAGNOSTIC (not promotable) — X4 with the leaf-size and "
                    "sampling constraints converted to LightGBM's scale",
            "changed_vs": {"X4": ["min_child_weight 5000->1250 (~5000 rows)",
                                  "colsample_bytree 1.0->0.7 (== L1 ff)",
                                  "subsample 1.0->0.8 (== L1 bagging)"]}},
    # ---- CatBoost ---------------------------------------------------------- #
    "C1": {"family": "catboost", "params": {}, "cat_ids": True,
           "role": "PRIMARY — ordered TS (has_time) + native ID categoricals",
           "changed_vs": {}},
    "C2": {"family": "catboost", "params": {"depth": 4, "l2_leaf_reg": 50},
           "cat_ids": True, "role": "regularisation axis",
           "changed_vs": {"C1": ["depth 6->4", "l2_leaf_reg 10->50"]}},
    "C3": {"family": "catboost", "params": {"one_hot_max_size": 2},
           "cat_ids": True, "role": "one-hot path closed -> more ordered TS",
           "changed_vs": {"C1": ["one_hot_max_size 10->2 (base_state now CTR)"]}},
    "C4": {"family": "catboost", "params": {"has_time": False},
           "cat_ids": True, "role": "ATTRIBUTION — C1-C4 = the has_time share",
           "changed_vs": {"C1": ["has_time True->False"]}},
    "C5": {"family": "catboost", "params": {}, "cat_ids": False,
           "role": "ATTRIBUTION — C1-C5 = the native-ID share",
           "changed_vs": {"C1": ["pitcher_id/batter_id: cat_features -> numeric"]}},
    "C6": {"family": "catboost", "params": {"rsm": 0.5, "random_strength": 2},
           "cat_ids": True, "role": "variance reduction",
           "changed_vs": {"C1": ["rsm 1.0->0.5", "random_strength 1->2"]}},
    # C5N is a DIAGNOSTIC arm added AFTER Stage A (train.py sha at Stage A was
    # ffa1ef6cd0ac1eb302fa0aa5cd7075039491036f5c3345743c28a891d2b7d5ae; the only
    # diff is this dict entry, which touches no executed path of any other arm).
    # Not promotable.  Reason: C1-C4 measures has_time WITH native ID
    # categoricals, i.e. exactly where the ordered target statistics do their
    # work.  The winning configuration C5 has no high-cardinality categorical at
    # all, and its remaining cat features (2, 1 and 8 levels) are all at or below
    # one_hot_max_size=10, so has_time may well be inert there.  "May well be"
    # is not a measurement — this arm makes it one.
    "C5N": {"family": "catboost", "params": {"has_time": False},
            "cat_ids": False,
            "role": "DIAGNOSTIC (not promotable) — C5 with has_time=False; "
                    "C5 - C5N = the has_time share AT THE WINNING CONFIG",
            "changed_vs": {"C5": ["has_time True->False"]}},
}
PROMOTABLE = [a for a in ARMS if a not in ("X4H", "C5N")]


# --------------------------------------------------------------------------- #
# inherited helpers (byte-for-byte from exp_021/src/train.py)
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
           "brier_full": float(f"{brier:.12g}"),
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


def argmin_of(curve):
    k = min(curve, key=lambda x: curve[x])
    return int(k), float(curve[k])


def build_matrix(df, feats, vocab):
    """float64 design matrix; categoricals -> train-window vocabulary codes.

    exp_021 build_matrix, unchanged.  Unseen / missing category -> NaN, never a
    sentinel.  Used by the LightGBM and XGBoost adapters.
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


def build_frame_native(df, feats, cat_native):
    """CatBoost design matrix: identical 47 columns in identical file order,
    but the categorical columns keep their NATIVE values (strings stay strings,
    IDs stay int64) so CatBoost's own target-statistic machinery handles them.
    That native handling is the whole point of the CatBoost arms."""
    data = {}
    for c in feats:
        if c in cat_native:
            s = df[c]
            assert s.notna().all(), f"NaN in CatBoost cat feature '{c}'"
            data[c] = (s.to_numpy(dtype=np.int64) if pd.api.types.is_numeric_dtype(s)
                       else s.astype(str).to_numpy())
        else:
            data[c] = pd.to_numeric(df[c], errors="raise").to_numpy(dtype=np.float64)
    return pd.DataFrame(data, columns=feats)


def chunked_margin_curve(predict_range, n_rows, grid):
    """{k: raw margin of the first k rounds}, one pass over the trees.

    exp_010 / exp_016 / exp_018 / exp_021 machinery, generalised from a
    LightGBM Booster to any callable `predict_range(a, b) -> raw margin of
    trees [a, b)`.  A fixed-seed booster's first k rounds ARE the k-round
    model and raw margins add across ranges; whatever constant intercept the
    library folds into every predicted range is measured once and removed.
    The assert is what makes this safe for a library we have not used before.
    """
    a = predict_range(0, 5)
    b = predict_range(5, 10)
    full = predict_range(0, 10)
    resid = full - (a + b)
    const = float(np.median(resid))
    assert np.allclose(resid, const, atol=1e-8), (
        "raw margins are not additive across round ranges",
        float(np.abs(resid - const).max()))
    out, acc, prev = {}, None, 0
    for k in grid:
        chunk = predict_range(prev, k)
        acc = chunk if acc is None else acc + chunk + const
        prev = k
        out[k] = acc.copy()
    return out, const


# --------------------------------------------------------------------------- #
# family adapters — the ONLY new code.  Same matrix in, same contract out.
# --------------------------------------------------------------------------- #
def fit_lightgbm(spec, seed, X_tr, y_tr, X_va, feats, cat_cols, n_round):
    import lightgbm as lgb
    params = dict(LGB_BASE, seed=seed, **spec["params"])
    declared = list(cat_cols)
    ds = lgb.Dataset(X_tr, label=y_tr, feature_name=feats,
                     categorical_feature=declared, free_raw_data=False)
    bst = lgb.train(params, ds, num_boost_round=n_round)
    assert bst.num_trees() == n_round, (bst.num_trees(), n_round)

    def predict_range(a, b):
        return bst.predict(X_va, start_iteration=a, num_iteration=b - a,
                           raw_score=True)

    return predict_range, params, {"categorical_declared": declared}


def fit_xgboost(spec, seed, X_tr, y_tr, X_va, feats, cat_cols, n_round):
    import xgboost as xgb
    params = dict(XGB_BASE, seed=seed, **spec["params"])
    dtr = xgb.DMatrix(X_tr, label=y_tr, feature_names=feats, nthread=N_THREADS)
    dva = xgb.DMatrix(X_va, feature_names=feats, nthread=N_THREADS)
    bst = xgb.train(params, dtr, num_boost_round=n_round)

    def predict_range(a, b):
        return bst.predict(dva, iteration_range=(a, b), output_margin=True
                           ).astype(np.float64)

    note = {"categorical_declared": [],
            "categoricals_as": "train-window integer codes, plain numeric "
                               "(XGBoost is NOT told they are categorical) — "
                               "this is a variable in any X-vs-L comparison; "
                               "exp_021 D vs D_NOCAT_DIAG sizes it at 1.1e-05",
            "min_child_weight_semantics": "sum of hessians; binary:logistic at "
                                          "p~0.5 gives ~0.25/row, so the value "
                                          "is ~4x the row count it names"}
    return predict_range, params, note


def fit_catboost(spec, seed, F_tr, y_tr, F_va, feats, cat_native, n_round):
    import catboost as cb
    params = dict(CB_BASE, random_seed=seed, iterations=n_round, **spec["params"])
    cat_idx = [feats.index(c) for c in cat_native]
    # NOTE: has_time is a TRAINING parameter (it lives in `params`), not a Pool
    # argument.  It makes CatBoost use the pool's row order as the time axis for
    # its ordered target statistics instead of a random permutation.
    pool_tr = cb.Pool(F_tr, label=y_tr, cat_features=cat_idx)
    pool_va = cb.Pool(F_va, cat_features=cat_idx)
    model = cb.CatBoostClassifier(**params)
    model.fit(pool_tr)
    assert model.tree_count_ == n_round, (model.tree_count_, n_round)

    def predict_range(a, b):
        return model.predict(pool_va, prediction_type="RawFormulaVal",
                             ntree_start=a, ntree_end=b,
                             thread_count=N_THREADS).astype(np.float64)

    return predict_range, params, {
        "cat_features": list(cat_native),
        "scale_and_bias": [float(v) for v in np.ravel(model.get_scale_and_bias())],
    }


FITTERS = {"lightgbm": fit_lightgbm, "xgboost": fit_xgboost,
           "catboost": fit_catboost}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm", choices=list(ARMS))
    ap.add_argument("seed", type=int)
    ap.add_argument("--fold", default="fold_2023", choices=list(FOLDS))
    ap.add_argument("--cap", type=int, default=3000)
    ap.add_argument("--iter", type=int, default=0,
                    help="Stage B: train exactly this many rounds and report "
                         "there.  0 = Stage A (train to --cap, report at the "
                         "fold's own valid_all argmin).")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()
    arm, seed, fold, out_dir = args.arm, args.seed, args.fold, args.out
    spec = ARMS[arm]
    family = spec["family"]
    stage_b = args.iter > 0
    n_round = args.iter if stage_b else args.cap

    t_start = time.time()
    np.random.seed(seed)
    for d in ("", "pred", "curves", "res"):
        os.makedirs(os.path.join(out_dir, d), exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] {arm}/{family}/seed{seed}/{fold} "
          f"n_round={n_round} stage={'B' if stage_b else 'A'}", flush=True)

    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    assert df["row_id"].is_unique
    assert df["season"].is_monotonic_increasing, "file order assumption broken"
    feats = [c for c in df.columns if c not in DROP_COLS]
    assert len(feats) == 47, len(feats)

    season = df["season"].to_numpy()
    game_type = df["game_type"].to_numpy()
    f_excl = (season <= F_EXCLUDE_MAX_SEASON) & (game_type == "F")
    y_all = df[TARGET].to_numpy(dtype=np.float64)
    assert np.isin(y_all, (0.0, 1.0)).all()

    tu, vy = FOLDS[fold]
    tr = (season <= tu) & ~f_excl
    va = season == vy
    assert int(tr.sum()) == EXPECTED_N_TRAIN[fold], (int(tr.sum()), fold)
    y_tr, y_va = y_all[tr], y_all[va]
    is_r = game_type[va] == "R"
    if vy == 2024:
        assert int(va.sum()) == EXPECTED_2024["valid_all"], int(va.sum())
        assert int(is_r.sum()) == EXPECTED_2024["valid_R"], int(is_r.sum())

    # P1 re-check ON THE ACTUAL TRAINING POOL — this is the order has_time uses
    tkey = (season.astype(np.int64) * 100 + df["game_month"].to_numpy(np.int64))[tr]
    n_time_inversions = int((np.diff(tkey) < 0).sum())
    if family == "catboost" and spec["params"].get("has_time", True):
        assert n_time_inversions == 0, (
            "has_time=True but the training pool is not in time order",
            n_time_inversions)

    res = {
        "exp_id": "exp_027", "arm": arm, "family": family, "seed": seed,
        "fold": fold, "stage": "B" if stage_b else "A", "tag": args.tag,
        "role": spec["role"], "changed_vs": spec["changed_vs"],
        "n_round": n_round, "cap": args.cap,
        "n_train": int(tr.sum()), "n_valid_all": int(va.sum()),
        "train_pool_time_inversions": n_time_inversions,
        "n_rows_masked": int(f_excl.sum()),
        "inherited_from": INHERITED_FROM,
        "sha256": {"train.csv": sha256_file(DATA_PATH),
                   "train.py": sha256_file(os.path.abspath(__file__))},
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__},
    }
    for mod in ("lightgbm", "xgboost", "catboost"):
        try:
            res["versions"][mod] = __import__(mod).__version__
        except Exception:
            pass

    # ---- train-window categorical vocabulary (never sees valid rows) ------- #
    fit_df = df.loc[tr]
    vocab = {c: sorted(fit_df[c].dropna().unique().tolist()) for c in CAT_COLS}
    for c in CAT_COLS:
        assert fit_df[c].notna().all(), f"unexpected NaN in categorical '{c}'"
    del fit_df
    res["vocab"] = {c: list(map(str, v)) for c, v in vocab.items()}
    res["constant_in_train_cols"] = [c for c in CAT_COLS if len(vocab[c]) < 2]

    t_prep = time.time()
    if family == "catboost":
        cat_native = list(CAT_COLS) + (list(ID_COLS) if spec["cat_ids"] else [])
        cat_native = [c for c in feats if c in cat_native]        # file order
        A_tr = build_frame_native(df.loc[tr], feats, cat_native)
        A_va = build_frame_native(df.loc[va], feats, cat_native)
        res["unseen_category_rows"] = {
            c: int((~A_va[c].isin(set(A_tr[c].unique()))).sum()) for c in cat_native}
        fit_args = (A_tr, y_tr, A_va, feats, cat_native, n_round)
    else:
        X_tr, u_tr = build_matrix(df.loc[tr], feats, vocab)
        X_va, u_va = build_matrix(df.loc[va], feats, vocab)
        assert all(v == 0 for v in u_tr.values()), u_tr
        res["unseen_category_rows"] = u_va
        fit_args = (X_tr, y_tr, X_va, feats, list(CAT_COLS), n_round)
    res["prep_min"] = round((time.time() - t_prep) / 60, 2)
    del df
    gc.collect()

    t0 = time.time()
    predict_range, params, note = FITTERS[family](spec, seed, *fit_args)
    fit_min = (time.time() - t0) / 60
    res["fit_min"] = round(fit_min, 3)
    res["fit_sec_per_round"] = round(60.0 * fit_min / n_round, 5)
    res["params"] = {k: (v if isinstance(v, (int, float, str, bool, type(None)))
                         else str(v)) for k, v in params.items()}
    res["family_note"] = note
    print(f"  fit {fit_min:.2f} min ({res['fit_sec_per_round']:.4f} s/round)",
          flush=True)

    t1 = time.time()
    grid = curve_grid(n_round)
    marg, const = chunked_margin_curve(predict_range, int(va.sum()), grid)
    cv = {k: float(np.mean((sigmoid(marg[k]) - y_va) ** 2)) for k in grid}
    a_min, a_val = argmin_of(cv)
    res["curve_min"] = round((time.time() - t1) / 60, 2)
    res["margin_const"] = const
    res["curve"] = {str(k): float(f"{v:.10g}") for k, v in cv.items()}
    res["argmin_iter"] = a_min
    res["argmin_brier"] = float(f"{a_val:.10g}")
    res["argmin_at_cap"] = bool((not stage_b) and a_min >= n_round)

    k_rep = a_min if not stage_b else n_round
    p = sigmoid(marg[k_rep])
    res["iter_used"] = int(k_rep)
    res["iter_source"] = ("own fold valid_all argmin (Stage A screening — the "
                          "same rule for every arm)" if not stage_b else
                          f"Stage A fold_2023 argmin x {SCALE_2024_OVER_2023} "
                          f"(fixed before this fit; fold_2024 valid never "
                          f"consulted)")
    res["valid_all"] = seg_metrics(y_va, p)
    res["valid_R"] = seg_metrics(y_va[is_r], p[is_r])
    res["valid_F"] = seg_metrics(y_va[~is_r], p[~is_r])
    res["policy_cost_vs_curve_min"] = float(f"{cv[k_rep] - a_val:.6g}")
    if stage_b:
        res["curve_argmin_is_interior"] = bool(a_min < n_round)

    if arm == "L1" and seed == 42 and fold == "fold_2023" and not stage_b:
        got825 = cv[nearest_grid_iter(grid, 825)]
        res["reproduction_gate"] = {
            **L1_GATE, "got_argmin_iter": a_min,
            "got_brier_at_825": round(got825, 6),
            "argmin_matches": bool(a_min == L1_GATE["fold_2023_argmin_iter"]),
            "abs_diff_at_825": float(f"{abs(got825 - L1_GATE['fold_2023_brier_at_825']):.10g}"),
            "passed": bool(a_min == L1_GATE["fold_2023_argmin_iter"] and
                           abs(got825 - L1_GATE["fold_2023_brier_at_825"])
                           <= L1_GATE["tolerance_brier"])}
        print("GATE: " + json.dumps(res["reproduction_gate"]), flush=True)

    stem = f"{arm}_seed{seed}_{fold}" + (f"_it{n_round}" if stage_b else "")
    np.save(os.path.join(out_dir, "pred", f"pred_{stem}.npy"), p)
    with open(os.path.join(out_dir, "curves", f"curve_{stem}.json"), "w") as f:
        json.dump({"arm": arm, "seed": seed, "fold": fold, "grid": grid,
                   "valid_all_brier": res["curve"], "argmin_iter": a_min,
                   "argmin_brier": res["argmin_brier"]}, f)
    res["pred_file"] = f"pred/pred_{stem}.npy"
    res["curve_file"] = f"curves/curve_{stem}.json"
    res["elapsed_min"] = round((time.time() - t_start) / 60, 2)

    path = os.path.join(out_dir, "res", f"res_{stem}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"[{time.strftime('%H:%M:%S')}] {stem} DONE {res['elapsed_min']} min | "
          f"argmin {a_min} = {res['argmin_brier']} | reported iter {k_rep} "
          f"brier {res['valid_all']['brier_full']}"
          f"{'  *** ARGMIN AT CAP ***' if res['argmin_at_cap'] else ''}",
          flush=True)


if __name__ == "__main__":
    main()
