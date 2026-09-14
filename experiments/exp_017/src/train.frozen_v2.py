"""exp_017 — iteration-carry policy repair for the exp_016 B3 stack.

Two jobs, selected with EXP017_JOB:

  extend  (lr 0.01, cap 3000 — exp_016 spec, unchanged)
      Adds the folds exp_016 never ran, so that every policy that needs two
      previous folds (P8 trend) can be scored on B3 as well as B0, and so that
      the 2022->2023 transition exists for all three seeds:
          B0  fold_2022  seeds 42 / 7 / 2024   (seed 42 doubles as a repro gate)
          B3  fold_2022  seeds 42 / 7 / 2024
      Also emits the F2 diagnostics: per-season OOF stack-column AUC and NaN
      share, plus the cell-model iteration actually used for the stack columns.

  lowlr   (P9: lr 0.005, cap 6000, everything else identical)
      Full three-fold chain for B0 and B3, seed 42 first; seeds 7 / 2024 follow
      in the same process so the seed spread is measurable without a relaunch.

Both jobs write curves in the shape policy_offline.py consumes:
    metrics["runs"] = [{arm, seed, fold, curve, n_train, ...}, ...]

Nothing here reads the fold it is scored on.  The oracle (argmin of the target
fold) is recorded for reference only; no policy consumes it.
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

JOB = os.environ.get("EXP017_JOB", "extend")
SMOKE = os.environ.get("EXP017_SMOKE") == "1"

N_THREADS = 6
DATA_PATH = os.path.expanduser("~/SungjinPark/data/train.csv")
OUT_DIR = os.path.expanduser(f"~/SungjinPark/experiments/exp_017_{JOB}")

TARGET = "control_success"
DROP_COLS = ["row_id", TARGET]
CAT_COLS = ["top_bottom", "game_type", "base_state"]
FOLDS = [("fold_2022", 2021, 2022), ("fold_2023", 2022, 2023),
         ("fold_2024", 2023, 2024)]
F_EXCLUDE_MAX_SEASON = 2022
EXPECTED_2024 = {"valid_all": 253507, "valid_R": 223497}
EPS = 1e-6

# exp_010 D / exp_016, verbatim except learning_rate in the lowlr job.
D_PARAMS = {
    "objective": "binary", "metric": "binary_logloss",
    "learning_rate": 0.01, "num_leaves": 15, "min_data_in_leaf": 5000,
    "feature_fraction": 0.7, "bagging_fraction": 0.8, "bagging_freq": 1,
    "seed": 42, "num_threads": N_THREADS, "deterministic": True,
    "force_row_wise": True, "verbose": -1,
}

if JOB == "extend":
    LR, N_ROUNDS = 0.01, 3000
    SEEDS = [42, 7, 2024]
    FOLDS_WANTED = ["fold_2022"]
elif JOB == "lowlr":
    LR, N_ROUNDS = 0.005, 6000
    # seed 42 only by default: the full three-fold B3 chain at 6000 rounds is
    # ~90 min per seed.  Repeat seeds are added with EXP017_SEEDS only if P9
    # turns out to be worth repeating.
    SEEDS = [42]
    FOLDS_WANTED = ["fold_2022", "fold_2023", "fold_2024"]
else:
    raise SystemExit(f"unknown EXP017_JOB={JOB}")

if os.environ.get("EXP017_SEEDS"):
    SEEDS = [int(s) for s in os.environ["EXP017_SEEDS"].split(",")]
if os.environ.get("EXP017_FOLDS"):
    FOLDS_WANTED = os.environ["EXP017_FOLDS"].split(",")
if os.environ.get("EXP017_TAG"):
    OUT_DIR = os.path.expanduser(
        f"~/SungjinPark/experiments/exp_017_{os.environ['EXP017_TAG']}")

if SMOKE:
    N_ROUNDS = 40
    SEEDS = [42]
    OUT_DIR += "_smoke"

# exp_016 reproduction gate (extend job only): B0 fold_2022 seed 42.
REPRO = {"fold": "fold_2022", "arm": "B0", "seed": 42,
         "argmin_iter": 300, "argmin_brier": 0.247197, "tol": 5e-6}


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
           "score": round(max(0.0, 100000.0 * (1.0 - brier / d)), 3) if d > 0 else None,
           "base_rate": round(r, 6), "pred_mean": round(float(np.mean(p)), 6)}
    if 0.0 < r < 1.0:
        out["auc"] = round(float(roc_auc_score(y, p)), 6)
        out["logloss"] = round(float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15))), 6)
    return out


def curve_grid(n_max):
    """exp_016 grid, extended with the same 50-step spacing past 3000."""
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
    sub = X.iloc[: min(2000, len(X))]
    a = booster.predict(sub, start_iteration=0, num_iteration=5, raw_score=True)
    b = booster.predict(sub, start_iteration=5, num_iteration=5, raw_score=True)
    full = booster.predict(sub, start_iteration=0, num_iteration=10, raw_score=True)
    resid = full - (a + b)
    const = np.median(resid, axis=0)
    assert np.allclose(resid, const, atol=1e-8), "raw margins not additive"
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


def brier_curve(grid, fn):
    return {k: float(fn(k)) for k in grid}


def argmin_of(curve):
    k = min(curve, key=lambda x: curve[x])
    return int(k), float(curve[k])


# --------------------------------------------------------------------------- #
def main():
    t_start = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[{time.strftime('%H:%M:%S')}] JOB={JOB} lr={LR} cap={N_ROUNDS} "
          f"seeds={SEEDS} folds={FOLDS_WANTED}", flush=True)

    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    assert df["row_id"].is_unique
    assert df["season"].is_monotonic_increasing
    base_feats = [c for c in df.columns if c not in DROP_COLS]
    assert len(base_feats) == 47, len(base_feats)
    assert_no_recovered_in_features(base_feats, "base_feats (before recovery)")

    diag = recover(df, verbose=False)
    assert diag["success_recovery"]["agree_pct"] == 100.0
    for c in CAT_COLS:
        df[c] = df[c].astype("category")

    f_excl = ((df["season"] <= F_EXCLUDE_MAX_SEASON)
              & (df["game_type"] == "F")).to_numpy()
    ok = df["recov_ok"].to_numpy()
    succ_season = df.groupby("pitcher_id", sort=False)["season"].shift(-1).to_numpy()
    season = df["season"].to_numpy()
    y_all = df[TARGET].to_numpy(dtype=np.float64)

    grid = curve_grid(N_ROUNDS)
    cat = [c for c in CAT_COLS if c in base_feats]

    metrics = {
        "exp_id": "exp_017", "job": JOB, "base_exp": "exp_016 (B0 / B3 arms)",
        "learning_rate": LR, "num_boost_round_cap": N_ROUNDS,
        "lgb_params": dict(D_PARAMS, learning_rate=LR),
        "seeds": SEEDS, "folds_wanted": FOLDS_WANTED,
        "grid_size": len(grid),
        "purpose": "score iteration-carry policies P4..P9 without ever reading "
                   "the fold being scored",
        "sha256": {"train.csv": sha256_file(DATA_PATH),
                   "train.py": sha256_file(os.path.abspath(__file__))},
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__, "lightgbm": lgb.__version__},
        "runs": [], "oof_diagnostics": [], "repro_gate": None,
    }

    def save():
        with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
            json.dump(json.loads(json.dumps(metrics, default=str)), f,
                      indent=1, ensure_ascii=False)

    # ---------------------------------------------------------------------- #
    for seed in SEEDS:
        params = dict(D_PARAMS, seed=seed, learning_rate=LR)
        prev_arg = {}          # arm -> previous fold argmin (P4 carry)
        prev_cell_arg = None   # B2c composite argmin, drives the stack columns

        for fold_name, tu, vy in FOLDS:
            if fold_name not in FOLDS_WANTED:
                # still needs to walk the chain in the lowlr job; in the extend
                # job only fold_2022 is requested and it has no predecessor.
                continue
            t_fold = time.time()
            tr_full = (season <= tu) & ~f_excl
            tr_mode = tr_full & ok & (succ_season <= tu)
            va = season == vy
            y_va = y_all[va]
            is_r = (df.loc[va, "game_type"] == "R").to_numpy()
            va_ok = ok[va]
            if vy == 2024:
                assert int(va.sum()) == EXPECTED_2024["valid_all"]
                assert int(is_r.sum()) == EXPECTED_2024["valid_R"]
            print(f"[{time.strftime('%H:%M:%S')}] === seed {seed} {fold_name}: "
                  f"train_full {tr_full.sum():,} / train_mode {tr_mode.sum():,} "
                  f"-> valid {va.sum():,} ===", flush=True)

            assert_no_recovered_in_features(base_feats, f"{fold_name} design")
            X_full = df.loc[tr_full, base_feats]
            X_mode = df.loc[tr_mode, base_feats]
            X_va = df.loc[va, base_feats]

            # ---------------- B0 ------------------------------------------ #
            _, m0 = fit_curve(X_full, y_all[tr_full], X_va, params, N_ROUNDS,
                              cat, grid, f"B0 {fold_name}")
            c0 = brier_curve(grid, lambda k: np.mean((sigmoid(m0[k]) - y_va) ** 2))
            a0, v0 = argmin_of(c0)
            k0 = nearest_grid_iter(grid, prev_arg["B0"]) if "B0" in prev_arg else a0
            metrics["runs"].append({
                "arm": "B0", "seed": seed, "fold": fold_name,
                "n_train": int(tr_full.sum()),
                "iter_used_P4": int(k0), "oracle_iter": a0,
                "oracle_brier": round(v0, 6),
                "oracle_flag": "B0" not in prev_arg,
                "valid_all": seg_metrics(y_va, sigmoid(m0[k0])),
                "valid_R": seg_metrics(y_va[is_r], sigmoid(m0[k0])[is_r]),
                "valid_F": seg_metrics(y_va[~is_r], sigmoid(m0[k0])[~is_r]),
                "curve": {str(k): round(v, 7) for k, v in c0.items()},
            })
            prev_arg["B0"] = a0
            del m0
            gc.collect()

            if REPRO and JOB == "extend" and seed == REPRO["seed"] \
                    and fold_name == REPRO["fold"]:
                d = abs(v0 - REPRO["argmin_brier"])
                metrics["repro_gate"] = {
                    "expected": REPRO, "got_argmin_iter": a0,
                    "got_argmin_brier": round(v0, 6), "abs_diff": round(d, 8),
                    "pass": bool(d <= REPRO["tol"] and a0 == REPRO["argmin_iter"])}
                print(f"  REPRO GATE {metrics['repro_gate']}", flush=True)

            # ---------------- cell models (feed B2c iter + B3 stack) ------- #
            mc = {}
            for c in FAIL_CELLS:
                yc = df.loc[tr_mode, c].to_numpy(dtype=np.float64)
                assert not np.isnan(yc).any()
                assert_no_recovered_in_features(base_feats, f"cell {c} {fold_name}")
                _, mc[c] = fit_curve(X_mode, yc, X_va, params, N_ROUNDS, cat,
                                     grid, f"{c} {fold_name}")

            def _b2c(k):
                s = sum(sigmoid(mc[c][k]) for c in FAIL_CELLS)
                return np.mean((np.clip(1.0 - s, EPS, 1 - EPS) - y_va) ** 2)
            c2c = brier_curve(grid, _b2c)
            a2c, v2c = argmin_of(c2c)
            cell_iter = (nearest_grid_iter(grid, prev_cell_arg)
                         if prev_cell_arg is not None else a2c)
            metrics["runs"].append({
                "arm": "B2c", "seed": seed, "fold": fold_name,
                "n_train": int(tr_mode.sum()),
                "iter_used_P4": int(cell_iter), "oracle_iter": a2c,
                "oracle_brier": round(v2c, 6),
                "oracle_flag": prev_cell_arg is None,
                "curve": {str(k): round(v, 7) for k, v in c2c.items()},
            })

            # cell-model quality on VALID at the iteration the stack will use
            cell_valid = {}
            for c in FAIL_CELLS:
                yc = df.loc[va, c].to_numpy(dtype=np.float64)[va_ok]
                cell_valid[c] = seg_metrics(yc, sigmoid(mc[c][cell_iter])[va_ok])
            stash = {c: sigmoid(mc[c][cell_iter]) for c in FAIL_CELLS}
            del mc
            gc.collect()

            # ---------------- B3: forward-chaining OOF stack -------------- #
            stack_cols = [f"p_{c}" for c in FAIL_CELLS]
            seasons = sorted(set(season[tr_full].tolist()))
            print(f"[{time.strftime('%H:%M:%S')}] B3 {fold_name}: OOF blocks "
                  f"{seasons[1:]} at cell iter {cell_iter}", flush=True)
            stack = {c: np.full(len(df), np.nan) for c in stack_cols}
            oof_q = []
            for s in seasons[1:]:
                fit_rows = (season < s) & ~f_excl & ok & (succ_season < s)
                pred_rows = (season == s) & tr_full
                Xf = df.loc[fit_rows, base_feats]
                Xp = df.loc[pred_rows, base_feats]
                blk = {"fold": fold_name, "seed": seed, "oof_season": int(s),
                       "n_fit": int(fit_rows.sum()), "n_pred": int(pred_rows.sum()),
                       "cell_iter": int(cell_iter), "auc": {}, "base_rate": {}}
                for c in FAIL_CELLS:
                    assert_no_recovered_in_features(base_feats, f"OOF {s} {c}")
                    yc = df.loc[fit_rows, c].to_numpy(dtype=np.float64)
                    dtr = lgb.Dataset(Xf, label=yc, categorical_feature=cat)
                    bst = lgb.train(params, dtr, num_boost_round=int(cell_iter))
                    pr = bst.predict(Xp)
                    stack[f"p_{c}"][pred_rows] = pr
                    # F2 diagnostic: how good is this stack column where it is
                    # defined?  Only rows whose recovered label is valid count.
                    m_ok = ok[pred_rows]
                    yt = df.loc[pred_rows, c].to_numpy(dtype=np.float64)[m_ok]
                    blk["auc"][c] = (round(float(roc_auc_score(yt, pr[m_ok])), 6)
                                     if 0 < yt.mean() < 1 else None)
                    blk["base_rate"][c] = round(float(yt.mean()), 6)
                oof_q.append(blk)
                print(f"    OOF season {s}: fit {int(fit_rows.sum()):,} -> pred "
                      f"{int(pred_rows.sum()):,} auc "
                      f"{ {k: v for k, v in blk['auc'].items()} } "
                      f"({time.time()-t_fold:.0f}s)", flush=True)
            for c in FAIL_CELLS:
                stack[f"p_{c}"][va] = stash[c]
            metrics["oof_diagnostics"].extend(oof_q)

            Xs = df[base_feats].copy()
            for c in stack_cols:
                Xs[c] = stack[c]
            feats = base_feats + stack_cols
            assert_no_recovered_in_features(feats, f"B3 stacker {fold_name}")
            _, m3 = fit_curve(Xs.loc[tr_full], y_all[tr_full], Xs.loc[va], params,
                              N_ROUNDS, cat, grid, f"B3 stacker {fold_name}")
            c3 = brier_curve(grid, lambda k: np.mean((sigmoid(m3[k]) - y_va) ** 2))
            a3, v3 = argmin_of(c3)
            k3 = nearest_grid_iter(grid, prev_arg["B3"]) if "B3" in prev_arg else a3
            nan_pct = round(100.0 * float(
                np.isnan(stack[stack_cols[0]][tr_full]).mean()), 3)
            metrics["runs"].append({
                "arm": "B3", "seed": seed, "fold": fold_name,
                "n_train": int(tr_full.sum()),
                "iter_used_P4": int(k3), "oracle_iter": a3,
                "oracle_brier": round(v3, 6),
                "oracle_flag": "B3" not in prev_arg,
                "cell_model_iter_used_for_stack_features": int(cell_iter),
                "nan_stack_rows_in_train_pct": nan_pct,
                "cell_valid_quality_at_stack_iter": cell_valid,
                "valid_all": seg_metrics(y_va, sigmoid(m3[k3])),
                "valid_R": seg_metrics(y_va[is_r], sigmoid(m3[k3])[is_r]),
                "valid_F": seg_metrics(y_va[~is_r], sigmoid(m3[k3])[~is_r]),
                "curve": {str(k): round(v, 7) for k, v in c3.items()},
                "minutes": round((time.time() - t_fold) / 60, 2),
            })
            prev_arg["B3"] = a3
            prev_cell_arg = a2c
            del Xs, stack, m3, X_full, X_mode, X_va
            gc.collect()
            metrics["elapsed_min"] = round((time.time() - t_start) / 60, 2)
            save()
            print(f"[{time.strftime('%H:%M:%S')}] {fold_name} seed {seed} done "
                  f"({(time.time()-t_fold)/60:.1f} min) B0 argmin {a0} / "
                  f"B2c argmin {a2c} / B3 argmin {a3}", flush=True)

    metrics["elapsed_min"] = round((time.time() - t_start) / 60, 2)
    save()
    print(f"[{time.strftime('%H:%M:%S')}] DONE {metrics['elapsed_min']} min",
          flush=True)


if __name__ == "__main__":
    main()
