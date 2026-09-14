"""exp_016 — model the three failure MODES separately and recombine.

Spec: experiments/exp_016/config.yaml (campaign 6, user proposal).
Gate:  src/gate.py -> gate.json  (must have run and passed first).

Why
---
The competition defines failure as three physically different events (대회정보
§1): (1) over the middle of the zone, (2) way off the zone, (3) opposite to
the catcher's target.  Training one binary model on their union averages three
different mechanisms together.  This experiment recovers a per-row mode label
(src/labels.py) and asks whether splitting the target and recombining beats
the single binary model on the SAME 47 provided columns.

What the D2 gate changed about the plan
---------------------------------------
config D4 assumed the modes were mutually exclusive and specified a 4-class
multiclass arm (B1) plus a "1 - sum of three" arm (B2).  The gate measured:

  * success never co-occurs with middle or reverse          (0 / 1,474,300)
  * middle and reverse co-occur on 50,266 rows              (3.41%)

so neither the 4-class form nor "1 - (P_mid + P_rev + P_way)" is well posed.
The overlap is between two failure modes only, which means the five cells

  success | middle-only | reverse-only | middle&reverse | way-off

ARE a partition (asserted in labels.recover).  Every arm below is built on
that partition instead; see report.md "추가 설계 판단" for the reasoning.

Arms
----
  B0    control: single binary on `control_success`, exp_010 D verbatim.
        Must reproduce fold_2024 valid_all Brier 0.248261 or the run aborts.
  B0b   same model on the mode-models' row set (row-matched control).  This
        separates "the rows we had to drop" from "the target decomposition".
  B2c   four independent binaries, one per failure cell.
        P(success) = 1 - clip(sum of the four, 0, 1).
  B2n   normalised five-way: P = p_s / (p_s + sum of four), p_s from B0b.
  B2m   marginal form: binaries on `middle` and `reverse` as the organisers
        define them (overlapping) plus `both` and `way-off`, recombined by
        inclusion-exclusion: P = 1 - clip(p_m + p_r + p_w - p_both, 0, 1).
  B1    5-class multiclass over the partition; P = softmax[success].
  B3    stacking: the four cell probabilities as extra columns of a binary
        model.  Train-segment cell probabilities are out-of-fold, built with
        a forward-chaining season split (config D4 warning).

Iteration policy: exp_006 P4 on valid_all.  Every arm re-measures its OWN
fold_2023 composed-Brier argmin and carries it to fold_2024.  fold_2022 has no
predecessor and uses its own argmin (oracle, flagged, not comparable).
valid 2024 is never consulted to choose anything.

Environment: A5000, ~/SungjinPark/venv (Python 3.11.15 / numpy 1.26.4 /
pandas 2.0.3 / lightgbm 4.7.0).
"""

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labels import (CELLS, FAIL_CELLS, assert_no_recovered_in_features,  # noqa: E402
                    recover)

SEED = 42
SEED_REPEATS = [7, 2024]
N_THREADS = 6

DATA_PATH = os.path.expanduser("~/SungjinPark/data/train.csv")
OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_016")

TARGET = "control_success"
DROP_COLS = ["row_id", TARGET]
CAT_COLS = ["top_bottom", "game_type", "base_state"]

FOLDS = [("fold_2022", 2021, 2022), ("fold_2023", 2022, 2023),
         ("fold_2024", 2023, 2024)]
F_EXCLUDE_MAX_SEASON = 2022
EXPECTED_2024 = {"valid_all": 253507, "valid_R": 223497}

# exp_010 D, verbatim.  config D5: the same params for every sub-model, so the
# comparison measures the decomposition and not a tuning difference.
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
EPS = 1e-6

BASE_REFERENCE = {
    "source": "exp_010 metrics.json -> summary_fold_2024_valid_ALL_PRIMARY.D_P4-all",
    "fold_2024_valid_all_brier": 0.248261,
    "fold_2024_iter": 750,
    "fold_2023_valid_all_argmin_iter": 750,
    "tolerance_brier": 5e-6,
}

SMOKE = os.environ.get("EXP016_SMOKE") == "1"
if SMOKE:
    N_ROUNDS = 40
    SEED_REPEATS = [7]
    OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_016_smoke")


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
    out = {
        "n": n, "brier": round(brier, 6),
        "bss": round(float(1.0 - brier / d), 6) if d > 0 else None,
        "score": round(lb_score(brier, r), 3) if d > 0 else None,
        "base_rate": round(r, 6), "pred_mean": round(float(np.mean(p)), 6),
    }
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


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nearest_grid_iter(grid, k):
    return min(grid, key=lambda g: (abs(g - k), g))


def raw_margin_curve(booster, X, grid):
    """{k: raw margin of the first k rounds} in one pass over the trees.

    Same trick as exp_010/exp_014: for a fixed booster the first k rounds ARE
    the k-round model and raw margins add across round ranges, so the whole
    Brier-vs-iteration curve costs one pass instead of one predict per point.
    Generalised here to the multiclass case, where the margin is (n, C).
    """
    sub = X.iloc[: min(2000, len(X))]
    a = booster.predict(sub, start_iteration=0, num_iteration=5, raw_score=True)
    b = booster.predict(sub, start_iteration=5, num_iteration=5, raw_score=True)
    full = booster.predict(sub, start_iteration=0, num_iteration=10, raw_score=True)
    resid = full - (a + b)
    const = np.median(resid, axis=0)
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


def fit_curve(X_tr, y_tr, X_va, params, n_rounds, cat, grid, tag):
    t0 = time.time()
    dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat)
    booster = lgb.train(params, dtr, num_boost_round=n_rounds)
    t1 = time.time()
    margins = raw_margin_curve(booster, X_va, grid) if grid else None
    print(f"    [{tag}] fit {(t1-t0)/60:.2f} min, curve {time.time()-t1:.0f}s "
          f"({X_tr.shape[0]:,} rows, pos {float(np.mean(y_tr)):.4f})", flush=True)
    return booster, margins


# --------------------------------------------------------------------------- #
def brier_curve(grid, fn):
    return {k: float(fn(k)) for k in grid}


def argmin_of(curve):
    k = min(curve, key=lambda x: curve[x])
    return int(k), float(curve[k])


# --------------------------------------------------------------------------- #
def main():
    t_start = time.time()
    np.random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] loading {DATA_PATH}", flush=True)
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    print(f"loaded {df.shape}", flush=True)
    assert df["row_id"].is_unique
    assert df["season"].is_monotonic_increasing, "file order assumption broken"

    base_feats = [c for c in df.columns if c not in DROP_COLS]
    assert len(base_feats) == 47, len(base_feats)
    # D3: the feature set is frozen.  The only thing this experiment changes is
    # the TARGET structure.
    assert_no_recovered_in_features(base_feats, "base_feats (before recovery)")

    diag = recover(df)
    assert diag["success_recovery"]["agree_pct"] == 100.0, diag["success_recovery"]

    for c in CAT_COLS:
        df[c] = df[c].astype("category")

    f_excl = (df["season"] <= F_EXCLUDE_MAX_SEASON) & (df["game_type"] == "F")
    ok = df["recov_ok"].to_numpy()
    # LEAK BOUNDARY, second half: is_X[t] is built from row t+1.  If row t is
    # the last TRAIN row of a pitcher and row t+1 lands in the VALID season,
    # then that training label encodes a validation outcome.  Those rows are
    # dropped from every model whose target is a recovered label.
    succ_season = df.groupby("pitcher_id", sort=False)["season"].shift(-1)
    season = df["season"].to_numpy()
    y_all = df[TARGET].to_numpy(dtype=np.float64)

    metrics = {
        "exp_id": "exp_016",
        "base_exp": "exp_010 D (47 provided columns, strongly regularised LGBM)",
        "hypothesis": "splitting the failure target into its three physical "
                      "modes and recombining extracts more signal from the "
                      "same 47 columns",
        "gate": "src/gate.py -> gate.json (items 1&2 passed, modes NOT "
                "exclusive: middle&reverse overlap 3.41%)",
        "partition": CELLS,
        "seed_primary": SEED, "seed_repeats": SEED_REPEATS,
        "num_threads": N_THREADS, "lgb_params_binary": D_PARAMS,
        "num_boost_round_cap": N_ROUNDS,
        "recovery_diagnostics": diag,
        "feature_list_used_by_every_submodel": base_feats,
        "leak_boundary": {
            "recovered_labels_used_as": "TARGET ONLY",
            "feature_list_assert": "labels.assert_no_recovered_in_features() is "
                                   "called before every Dataset build",
            "fold_boundary_rows_dropped_from_recovered-label_models":
                {f"train_until_{tu}": int(((df['season'] <= tu)
                                           & (succ_season > tu)).sum())
                 for tu in (2021, 2022, 2023)},
        },
        "sha256": {"train.csv": sha256_file(DATA_PATH),
                   "train.py": sha256_file(os.path.abspath(__file__)),
                   "labels.py": sha256_file(os.path.join(
                       os.path.dirname(os.path.abspath(__file__)), "labels.py"))},
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__, "lightgbm": lgb.__version__},
    }

    def save():
        with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
            json.dump(json.loads(json.dumps(metrics, default=str)), f,
                      indent=2, ensure_ascii=False)

    grid = curve_grid(N_ROUNDS)
    cat = [c for c in CAT_COLS if c in base_feats]

    # ---------------------------------------------------------------- #
    #  one seed pass                                                    #
    # ---------------------------------------------------------------- #
    def run_seed(seed, folds_wanted, arms_wanted, stash=None):
        params = dict(D_PARAMS, seed=seed)
        mparams = dict(params, objective="multiclass", num_class=len(CELLS),
                       metric="multi_logloss")
        res = {a: {} for a in arms_wanted}
        cell_res = {}
        prev_arg = {}          # arm -> iter carried forward (P4)
        prev_cell_arg = {}     # cell -> iter carried forward

        for fold_name, tu, vy in FOLDS:
            if fold_name not in folds_wanted:
                continue
            tr_full = ((season <= tu) & ~f_excl.to_numpy())
            tr_mode = tr_full & ok & (succ_season.to_numpy() <= tu)
            va = season == vy
            y_va = y_all[va]
            is_r = (df.loc[va, "game_type"] == "R").to_numpy()
            va_ok = ok[va]
            if vy == 2024:
                assert int(va.sum()) == EXPECTED_2024["valid_all"], int(va.sum())
                assert int(is_r.sum()) == EXPECTED_2024["valid_R"]

            print(f"[{time.strftime('%H:%M:%S')}] === seed {seed} {fold_name}: "
                  f"train_full {tr_full.sum():,} / train_mode {tr_mode.sum():,}"
                  f" -> valid {va.sum():,} ===", flush=True)

            assert_no_recovered_in_features(base_feats, f"{fold_name} design matrix")
            X_full = df.loc[tr_full, base_feats]
            X_mode = df.loc[tr_mode, base_feats]
            X_va = df.loc[va, base_feats]

            marg = {}
            need_cells = bool({"B2c", "B2n", "B2m", "B3"} & set(arms_wanted))

            keep = (seed == SEED and fold_name == "fold_2024")
            if "B0" in arms_wanted:
                _, marg["B0"] = fit_curve(X_full, y_all[tr_full], X_va, params,
                                          N_ROUNDS, cat, grid, f"B0 {fold_name}")
            if "B0b" in arms_wanted or need_cells:
                bst, marg["B0b"] = fit_curve(X_mode, y_all[tr_mode], X_va, params,
                                             N_ROUNDS, cat, grid, f"B0b {fold_name}")
                if keep:
                    bst.save_model(os.path.join(OUT_DIR, "model_B0b_fold_2024.txt"))
            if need_cells:
                for c in FAIL_CELLS:
                    yc = df.loc[tr_mode, c].to_numpy(dtype=np.float64)
                    assert not np.isnan(yc).any()
                    bst, marg[c] = fit_curve(X_mode, yc, X_va, params, N_ROUNDS,
                                             cat, grid, f"{c} {fold_name}")
                    if keep:
                        bst.save_model(os.path.join(
                            OUT_DIR, f"model_{c}_fold_2024.txt"))
            if "B2m" in arms_wanted:
                for c in ["is_middle", "is_reverse"]:
                    yc = df.loc[tr_mode, c].to_numpy(dtype=np.float64)
                    assert not np.isnan(yc).any()
                    _, marg[c] = fit_curve(X_mode, yc, X_va, params, N_ROUNDS,
                                           cat, grid, f"{c} {fold_name}")
            if "B1" in arms_wanted:
                cls = np.zeros(int(tr_mode.sum()), dtype=np.int32)
                sub = df.loc[tr_mode, CELLS].to_numpy(dtype=np.float64)
                cls = np.argmax(sub, axis=1).astype(np.int32)
                assert (sub.sum(axis=1) == 1).all()
                _, marg["B1"] = fit_curve(X_mode, cls, X_va, mparams, N_ROUNDS,
                                          cat, grid, f"B1 {fold_name}")

            # ---------- composed probability curves ------------------------ #
            def p_cells(k):
                return {c: sigmoid(marg[c][k]) for c in FAIL_CELLS}

            curves, extra = {}, {}
            if "B0" in arms_wanted:
                curves["B0"] = brier_curve(
                    grid, lambda k: np.mean((sigmoid(marg["B0"][k]) - y_va) ** 2))
            if "B0b" in arms_wanted:
                curves["B0b"] = brier_curve(
                    grid, lambda k: np.mean((sigmoid(marg["B0b"][k]) - y_va) ** 2))
            if "B2c" in arms_wanted:
                def _b2c(k):
                    s = sum(p_cells(k).values())
                    return np.mean((np.clip(1.0 - s, EPS, 1 - EPS) - y_va) ** 2)
                curves["B2c"] = brier_curve(grid, _b2c)
            if "B2n" in arms_wanted:
                def _b2n(k):
                    ps = sigmoid(marg["B0b"][k])
                    s = sum(p_cells(k).values())
                    return np.mean((np.clip(ps / (ps + s), EPS, 1 - EPS) - y_va) ** 2)
                curves["B2n"] = brier_curve(grid, _b2n)
            if "B2m" in arms_wanted:
                def _b2m(k):
                    pm = sigmoid(marg["is_middle"][k])
                    pr = sigmoid(marg["is_reverse"][k])
                    pw = sigmoid(marg["is_wayoff_true"][k])
                    pb = sigmoid(marg["is_both"][k])
                    return np.mean((np.clip(1.0 - (pm + pr + pw - pb), EPS, 1 - EPS)
                                    - y_va) ** 2)
                curves["B2m"] = brier_curve(grid, _b2m)
            if "B1" in arms_wanted:
                curves["B1"] = brier_curve(
                    grid, lambda k: np.mean((np.clip(softmax(marg["B1"][k])[:, 0],
                                                     EPS, 1 - EPS) - y_va) ** 2))

            # per-cell own curves (which mode is predictable at all?)
            cell_curves = {}
            if need_cells:
                for c in FAIL_CELLS:
                    yc = df.loc[va, c].to_numpy(dtype=np.float64)[va_ok]
                    cell_curves[c] = brier_curve(
                        grid, lambda k, c=c, yc=yc:
                        np.mean((sigmoid(marg[c][k])[va_ok] - yc) ** 2))

            # ---------- report each arm at its P4 iteration ---------------- #
            for arm in arms_wanted:
                if arm not in curves:
                    continue
                a_min, a_val = argmin_of(curves[arm])
                if arm in prev_arg:
                    k = nearest_grid_iter(grid, prev_arg[arm])
                    src, oracle = f"P4-all carried from previous fold", False
                else:
                    k, src, oracle = a_min, "own argmin (no predecessor; ORACLE)", True
                if arm == "B0":
                    p = sigmoid(marg["B0"][k])
                elif arm == "B0b":
                    p = sigmoid(marg["B0b"][k])
                elif arm == "B2c":
                    p = np.clip(1.0 - sum(p_cells(k).values()), EPS, 1 - EPS)
                elif arm == "B2n":
                    ps = sigmoid(marg["B0b"][k]); s = sum(p_cells(k).values())
                    p = np.clip(ps / (ps + s), EPS, 1 - EPS)
                elif arm == "B2m":
                    p = np.clip(1.0 - (sigmoid(marg["is_middle"][k])
                                       + sigmoid(marg["is_reverse"][k])
                                       + sigmoid(marg["is_wayoff_true"][k])
                                       - sigmoid(marg["is_both"][k])),
                                EPS, 1 - EPS)
                elif arm == "B1":
                    p = np.clip(softmax(marg["B1"][k])[:, 0], EPS, 1 - EPS)
                rep = {
                    "iter_used": int(k), "iter_source": src, "oracle_flag": oracle,
                    "n_train": int(tr_full.sum() if arm == "B0" else tr_mode.sum()),
                    "valid_all": seg_metrics(y_va, p),
                    "valid_R": seg_metrics(y_va[is_r], p[is_r]),
                    "valid_F": seg_metrics(y_va[~is_r], p[~is_r]),
                    "oracle_iter": a_min, "oracle_brier": round(a_val, 6),
                    "policy_cost_vs_oracle": round(curves[arm][k] - a_val, 6),
                }
                if arm == "B2c":
                    s_raw = sum(p_cells(k).values())
                    rep["raw_sum_over_1_pct"] = round(
                        100.0 * float(np.mean(s_raw > 1.0)), 4)
                    rep["raw_sum_mean"] = round(float(np.mean(s_raw)), 6)
                res[arm][fold_name] = rep
                res[arm].setdefault("curves", {})[fold_name] = {
                    str(kk): round(vv, 6) for kk, vv in curves[arm].items()}
                prev_arg[arm] = a_min

            # ---------- per-cell sub-model performance --------------------- #
            if need_cells:
                for c in FAIL_CELLS:
                    a_min, a_val = argmin_of(cell_curves[c])
                    k = (nearest_grid_iter(grid, prev_cell_arg[c])
                         if c in prev_cell_arg else a_min)
                    yc = df.loc[va, c].to_numpy(dtype=np.float64)[va_ok]
                    pc = sigmoid(marg[c][k])[va_ok]
                    cell_res.setdefault(c, {})[fold_name] = {
                        "iter_used": int(k),
                        "oracle_iter_own_mode": a_min,
                        "own_mode": seg_metrics(yc, pc),
                    }
                    prev_cell_arg[c] = a_min
                # per-mode-iteration variant of B2c
                kk = {c: nearest_grid_iter(grid, cell_res[c][fold_name]["iter_used"])
                      for c in FAIL_CELLS}
                s = sum(sigmoid(marg[c][kk[c]]) for c in FAIL_CELLS)
                p = np.clip(1.0 - s, EPS, 1 - EPS)
                res.setdefault("B2c_permode_iter", {})[fold_name] = {
                    "iters": kk, "valid_all": seg_metrics(y_va, p)}

            if stash is not None and fold_name in stash:
                # cell probabilities on the VALID rows at the arm's chosen
                # iteration, reused by B3 so it need not refit these models
                k = res["B2c"][fold_name]["iter_used"]
                stash[fold_name] = {c: sigmoid(marg[c][k]) for c in FAIL_CELLS}
                stash[fold_name]["_iter"] = k

            del marg, X_full, X_mode, X_va
            gc.collect()

        return res, cell_res

    # ---------------- seed-42 main pass ----------------------------------- #
    arms = ["B0", "B0b", "B2c", "B2n", "B2m", "B1"]
    stash = {"fold_2023": None, "fold_2024": None}
    res42, cells42 = run_seed(SEED, {"fold_2022", "fold_2023", "fold_2024"},
                              arms, stash=stash)
    for a in res42:
        metrics[a] = res42[a]
    metrics["cell_submodels_seed42"] = cells42

    # ---- B0 reproduction gate -------------------------------------------- #
    got = res42["B0"]["fold_2024"]["valid_all"]["brier"]
    d = abs(got - BASE_REFERENCE["fold_2024_valid_all_brier"])
    metrics["B0_reproduction_check"] = {
        "expected": BASE_REFERENCE["fold_2024_valid_all_brier"], "got": got,
        "abs_diff": round(d, 8),
        "expected_iter": BASE_REFERENCE["fold_2024_iter"],
        "got_iter": res42["B0"]["fold_2024"]["iter_used"],
        "passed": bool(d <= BASE_REFERENCE["tolerance_brier"]),
    }
    print("B0 reproduction: " + json.dumps(metrics["B0_reproduction_check"]),
          flush=True)
    metrics["elapsed_min"] = round((time.time() - t_start) / 60, 2)
    save()
    if not SMOKE:
        assert d <= BASE_REFERENCE["tolerance_brier"], (
            f"B0 failed to reproduce exp_010 D ({got} vs "
            f"{BASE_REFERENCE['fold_2024_valid_all_brier']}) — everything "
            "downstream would be uninterpretable, aborting.")

    base_b = res42["B0"]["fold_2024"]["valid_all"]["brier"]
    baseb_b = res42["B0b"]["fold_2024"]["valid_all"]["brier"]
    table = {}
    for a in arms + ["B2c_permode_iter"]:
        if a not in res42 or "fold_2024" not in res42[a]:
            continue
        v = res42[a]["fold_2024"]["valid_all"]
        table[a] = {
            "iter": res42[a]["fold_2024"].get("iter_used"),
            "valid_all_brier": v["brier"], "valid_all_bss": v["bss"],
            "valid_all_score": v["score"], "valid_all_auc": v.get("auc"),
            "pred_mean": v["pred_mean"],
            "delta_vs_B0": round(v["brier"] - base_b, 6),
            "delta_vs_B0b": round(v["brier"] - baseb_b, 6),
            "fold_2023_valid_all_brier":
                res42[a].get("fold_2023", {}).get("valid_all", {}).get("brier"),
        }
    metrics["summary_seed42_fold_2024_valid_all"] = table
    print("--- seed 42 summary ---\n" + json.dumps(table, indent=2), flush=True)
    save()

    # ---------------- B3: stacking with forward-chaining OOF -------------- #
    try:
        metrics["B3"] = run_b3(df, base_feats, cat, season, f_excl.to_numpy(), ok,
                               succ_season.to_numpy(), y_all, SEED, grid, stash,
                               EXPECTED_2024, base_b)
    except Exception as exc:                                  # noqa: BLE001
        metrics["B3"] = {"error": repr(exc)}
        print(f"B3 FAILED: {exc!r}", flush=True)
    metrics["elapsed_min"] = round((time.time() - t_start) / 60, 2)
    save()

    # ---------------- seed repeats on the best arm ------------------------ #
    cand = {a: table[a]["valid_all_brier"] for a in table if a != "B0"}
    if metrics.get("B3", {}).get("fold_2024"):
        cand["B3"] = metrics["B3"]["fold_2024"]["valid_all"]["brier"]
    best = min(cand, key=lambda a: cand[a])
    metrics["best_arm_seed42"] = best
    print(f"[{time.strftime('%H:%M:%S')}] ===== seed repeats: B0 + {best} =====",
          flush=True)

    need = {"B0", "B0b"} | ({"B2c", "B2n", "B2m", "B1"} & {best})
    if best in ("B2c", "B2n", "B2m", "B2c_permode_iter", "B3"):
        need |= {"B2c"}
    seed_tab = {a: {str(SEED): table[a]["valid_all_brier"]}
                for a in table if a in need or a == best}
    seed_tab.setdefault("B0", {str(SEED): base_b})
    if best == "B3":
        seed_tab["B3"] = {str(SEED): cand["B3"]}
    for s in SEED_REPEATS:
        st = {"fold_2023": None, "fold_2024": None} if best == "B3" else None
        r, c = run_seed(s, {"fold_2023", "fold_2024"}, sorted(need), stash=st)
        metrics[f"seed{s}"] = r
        metrics[f"cell_submodels_seed{s}"] = c
        if best == "B3":
            try:
                b3s = run_b3(df, base_feats, cat, season, f_excl.to_numpy(), ok,
                             succ_season.to_numpy(), y_all, s, grid, st,
                             EXPECTED_2024, r["B0"]["fold_2024"]["valid_all"]["brier"])
                metrics[f"B3_seed{s}"] = b3s
                r["B3"] = b3s
            except Exception as exc:                          # noqa: BLE001
                print(f"B3 seed {s} FAILED: {exc!r}", flush=True)
        for a in seed_tab:
            if a in r and "fold_2024" in r[a]:
                seed_tab[a][str(s)] = r[a]["fold_2024"]["valid_all"]["brier"]
        metrics["elapsed_min"] = round((time.time() - t_start) / 60, 2)
        save()

    seeds_all = [str(SEED)] + [str(s) for s in SEED_REPEATS]
    spread = {}
    for a, d_ in seed_tab.items():
        vals = [d_[s] for s in seeds_all if s in d_]
        if len(vals) < 2:
            continue
        spread[a] = {"per_seed": d_, "mean": round(float(np.mean(vals)), 6),
                     "std": round(float(np.std(vals, ddof=1)), 6),
                     "range": round(float(max(vals) - min(vals)), 6)}
    paired = {}
    for a in seed_tab:
        if a == "B0" or len(seed_tab[a]) < 2:
            continue
        ds = [seed_tab[a][s] - seed_tab["B0"][s] for s in seeds_all
              if s in seed_tab[a] and s in seed_tab["B0"]]
        paired[a] = {
            "per_seed": {s: round(seed_tab[a][s] - seed_tab["B0"][s], 6)
                         for s in seeds_all if s in seed_tab[a]},
            "mean_delta": round(float(np.mean(ds)), 6),
            "std_delta": round(float(np.std(ds, ddof=1)), 6) if len(ds) > 1 else None,
            "all_seeds_same_sign": bool(all(x < 0 for x in ds)
                                        or all(x > 0 for x in ds)),
        }
    metrics["seed_summary"] = {"brier_by_seed": seed_tab, "spread": spread,
                               "paired_delta_vs_B0": paired}
    print("--- seed summary ---\n" + json.dumps(metrics["seed_summary"], indent=2),
          flush=True)

    metrics["train_time_min"] = round((time.time() - t_start) / 60, 2)
    save()
    print(f"[{time.strftime('%H:%M:%S')}] DONE {metrics['train_time_min']} min",
          flush=True)


# --------------------------------------------------------------------------- #
def run_b3(df, base_feats, cat, season, f_excl, ok, succ_season, y_all, seed,
           grid, stash, expected_2024, base_b):
    """B3 — the four cell probabilities as extra columns of a binary model.

    config D4 warning: predicting the TRAIN segment with a model fitted on the
    TRAIN segment is in-sample, so the stacker would be looking at its own
    answer.  Train-segment cell probabilities are therefore out-of-fold, built
    forward-chaining by season: rows of season s are predicted by cell models
    fitted on seasons < s only.  The earliest season has no predecessor and
    keeps NaN (LightGBM handles NaN natively; a sentinel fill is banned by the
    project rules and would also be a lie about what is known).
    """
    params = dict(D_PARAMS, seed=seed)
    stack_cols = [f"p_{c}" for c in FAIL_CELLS]
    out = {"stack_features": stack_cols,
           "oof_scheme": "forward-chaining by season; season s predicted by "
                         "cell models fitted on seasons < s; earliest season NaN"}
    prev_arg = None

    for fold_name, tu, vy in [("fold_2023", 2022, 2023), ("fold_2024", 2023, 2024)]:
        t0 = time.time()
        tr_full = (season <= tu) & ~f_excl
        va = season == vy
        y_va = y_all[va]
        is_r = (df.loc[va, "game_type"] == "R").to_numpy()
        seasons = sorted(set(season[tr_full].tolist()))
        cell_iter = stash[fold_name]["_iter"]
        print(f"[{time.strftime('%H:%M:%S')}] B3 {fold_name}: OOF blocks "
              f"{seasons[1:]} at iter {cell_iter}", flush=True)

        stack = {c: np.full(len(df), np.nan) for c in stack_cols}
        for s in seasons[1:]:
            fit_rows = ((season < s) & ~f_excl & ok & (succ_season < s))
            pred_rows = (season == s) & tr_full
            Xf = df.loc[fit_rows, base_feats]
            Xp = df.loc[pred_rows, base_feats]
            for c in FAIL_CELLS:
                assert_no_recovered_in_features(base_feats, f"B3 OOF {s} {c}")
                yc = df.loc[fit_rows, c].to_numpy(dtype=np.float64)
                dtr = lgb.Dataset(Xf, label=yc, categorical_feature=cat)
                bst = lgb.train(params, dtr, num_boost_round=int(cell_iter))
                stack[f"p_{c}"][pred_rows] = bst.predict(Xp)
            print(f"    OOF season {s}: fit {int(fit_rows.sum()):,} -> "
                  f"pred {int(pred_rows.sum()):,} ({time.time()-t0:.0f}s)",
                  flush=True)
        for c in FAIL_CELLS:
            stack[f"p_{c}"][va] = stash[fold_name][c]

        Xs = df[base_feats].copy()
        for c in stack_cols:
            Xs[c] = stack[c]
        feats = base_feats + stack_cols
        assert_no_recovered_in_features(feats, f"B3 stacker {fold_name}")
        _, marg = fit_curve(Xs.loc[tr_full], y_all[tr_full], Xs.loc[va], params,
                            N_ROUNDS, cat, grid, f"B3 stacker {fold_name}")
        curve = brier_curve(grid, lambda k: np.mean((sigmoid(marg[k]) - y_va) ** 2))
        a_min, a_val = argmin_of(curve)
        k = nearest_grid_iter(grid, prev_arg) if prev_arg else a_min
        p = sigmoid(marg[k])
        out[fold_name] = {
            "iter_used": int(k), "oracle_iter": a_min,
            "oracle_brier": round(a_val, 6),
            "iter_source": ("P4-all from fold_2023" if prev_arg
                            else "own argmin (ORACLE)"),
            "oracle_flag": prev_arg is None,
            "cell_model_iter_used_for_stack_features": int(cell_iter),
            "n_train": int(tr_full.sum()),
            "nan_stack_rows_in_train_pct": round(
                100.0 * float(np.isnan(stack[stack_cols[0]][tr_full]).mean()), 3),
            "valid_all": seg_metrics(y_va, p),
            "valid_R": seg_metrics(y_va[is_r], p[is_r]),
            "valid_F": seg_metrics(y_va[~is_r], p[~is_r]),
            "delta_vs_B0": None,
            "minutes": round((time.time() - t0) / 60, 2),
        }
        out[fold_name]["curve"] = {str(kk): round(vv, 6) for kk, vv in curve.items()}
        prev_arg = a_min
        del Xs, stack, marg
        gc.collect()
    if "fold_2024" in out:
        out["fold_2024"]["delta_vs_B0"] = round(
            out["fold_2024"]["valid_all"]["brier"] - base_b, 6)
    return out


if __name__ == "__main__":
    main()
