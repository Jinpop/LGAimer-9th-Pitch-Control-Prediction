"""exp_019 — two-stage offset for the (pitcher x batter_hand) deviation.

QUESTION (the only one this run answers)
----------------------------------------
exp_018 showed the conditional deviation carries information the tree does not
absorb: the A0 residual regressed on d_s has slope 0.614 (0 = fully absorbed),
and a post-hoc linear correction recovers 2.4x what handing the column to the
tree recovered.  But that 0.614 was fitted ON the 2024 validation rows.  It is a
headroom diagnostic, not a result.

    Does a slope fitted only on EARLIER seasons transfer to the held-out season?

STAGE 1 is identical across every arm — literally the same booster, the same
iteration.  Only the offset differs, so no iteration confound is possible.

  C0  p                                        <- control, = exp_018 A0
  C1  clip(p + beta*d_s)                       beta from earlier-season OOF
  C2  sigmoid(logit(p) + gamma*d_s)            gamma from earlier-season OOF
  C3  clip(p + beta*d_r)                       the variable exp_018 A1 used

C1<->C2 differ only in scale, C1<->C3 only in the variable.  d_s and d_r are
correlated -0.79, so they are never combined (exp_018 A2 lost by doing that).

HOW THE SLOPE IS FITTED HONESTLY
--------------------------------
Fitting on training rows biases the slope towards 0: the tree has already
over-absorbed the pitcher effect on rows it was fitted to.  So the residuals
must be out-of-sample.  Season forward-chaining, same machinery as exp_016 B3:

    block s  :=  a model fitted on seasons < s, predicting season s

Blocks 2020..2024 are fitted in order.  Blocks 2022/2023/2024 ARE the three
rolling-origin folds (train<=2021/2022/2023), so only 2020 and 2021 are extra
fits.  For a fold whose validation season is V, the slope pools blocks s < V
only; a runtime assert re-checks max(block season) < V and max(lookup source
season) < V on every fold.  Season 2024 never participates in fitting anything.

ITERATION (P10, chained one block further back than exp_018)
------------------------------------------------------------
iter(block s) = nearest_grid(2.75 x argmin(block s-1)).  Block 2020 has no
predecessor and is reported at its own argmin (ORACLE, flagged, feeder only —
the same treatment exp_018 gave fold_2022).  Because an argmin is a property of
a curve and not of a reported iteration, extending the chain backwards leaves
fold_2023 and fold_2024 on exactly the iterations exp_018 used.  fold_2022 is
now on a real P10 iteration instead of its own argmin, which is a CHANGE vs
exp_018 and is recorded as one.

CHANGED vs exp_018 (all of it, not just the headline)
-----------------------------------------------------
  1. stage 2 offset added (C1/C2/C3); C0 is spec-identical to exp_018 A0
  2. platoon columns are NOT in the design matrix at all (exp_018 A1/A2/A3 put
     them there; that is the arm this experiment replaces)
  3. two extra stage-1 blocks (2020, 2021) exist, purely to source OOF residuals
  4. fold_2022 is reported at a chained P10 iteration, not at its own argmin
  5. prelim_stats (exp_018 §4) is not re-run — it measured the lookup, which is
     unchanged here (features_platoon.py is byte-identical, sha 68d63063...)

CHANGED vs exp_010 D additionally: iteration policy P4 -> P10.

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
import features_platoon as fp                                    # noqa: E402
import offset as off                                             # noqa: E402
from labels import assert_no_recovered_in_features, recover      # noqa: E402

SEED = 42
SEED_REPEATS = [7, 2024]
N_THREADS = 6

DATA_PATH = os.path.expanduser("~/SungjinPark/data/train.csv")
OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_019")
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
EXP018_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_018")
EXP016_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_016")

TARGET = "control_success"
DROP_COLS = ["row_id", TARGET]
CAT_COLS = ["top_bottom", "game_type", "base_state"]

BLOCKS = [2020, 2021, 2022, 2023, 2024]      # fitted in this order
FOLD_SEASONS = [2022, 2023, 2024]            # the rolling-origin folds
SLOPE_BLOCKS = [2020, 2021, 2022, 2023]      # blocks eligible to fit a slope
F_EXCLUDE_MAX_SEASON = 2022
EXPECTED_2024 = {"valid_all": 253507, "valid_R": 223497}

D_PARAMS = {                       # exp_010 D verbatim
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
P10_C = 2.75

GATE = {
    "source": "exp_010 metrics.json / exp_018 A0_reproduction_gate",
    "arm": "C0",
    "policy": "P4 (unscaled carry) — the policy the reference was measured under",
    "fold_2024_valid_all_brier": 0.248261,
    "fold_2024_iter": 750,
    "tolerance_brier": 5e-6,
}
# second, stronger gate: the exp_018 A0 prediction vector itself
GATE_PRED = os.path.join(EXP018_DIR, "pred_A0_seed{seed}_fold_2024.npy")

# pre-registered choices, written down before any result is seen
PREREG = {
    "slope_estimator": "no-intercept least squares of (y - p) on d, pooled over "
                       "all rows of every eligible earlier block, unweighted",
    "why_no_intercept": "an intercept is a global probability shift, i.e. the "
                        "exp_005 season-drift correction, a different mechanism "
                        "that was measured to hurt this base (+3.0e-5)",
    "rows_used_in_the_slope_fit": "ALL rows of the block, F-league included",
    "why_all_rows": "the correction is applied to all rows at validation and at "
                    "deployment, so the fitting population is matched to the "
                    "application population. The variant that drops the exp_004 "
                    "masked rows is reported as a sensitivity, NOT as a fallback",
    "residual_iteration": "each block's own P10 iteration (block 2020: its own "
                          "argmin, no predecessor exists)",
    "decision_rule": "adopt iff all 3 seeds share the sign AND |mean delta| >= "
                     "3 x C0 seed std",
}

SMOKE = os.environ.get("EXP019_SMOKE") == "1"
if SMOKE:
    N_ROUNDS = 40
    SEED_REPEATS = []
    OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_019_smoke")


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
    """{k: raw margin of the first k rounds}.  exp_010/exp_016/exp_018 machinery."""
    sub = X.iloc[: min(2000, len(X))]
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
    assert_no_recovered_in_features(base_feats, "base_feats (before recovery)")

    diag = recover(df)
    assert diag["success_recovery"]["agree_pct"] == 100.0, diag["success_recovery"]

    season = df["season"].to_numpy()
    ok = df["recov_ok"].to_numpy()
    succ_season = df.groupby("pitcher_id", sort=False)["season"].shift(-1).to_numpy()
    f_excl = ((df["season"] <= F_EXCLUDE_MAX_SEASON)
              & (df["game_type"] == "F")).to_numpy()
    y_all = df[TARGET].to_numpy(dtype=np.float64)
    is_r_all = (df["game_type"] == "R").to_numpy()

    # ---- lookups (exp_018, unchanged) + deviations ------------------------- #
    print(f"[{time.strftime('%H:%M:%S')}] building platoon lookups", flush=True)
    t_lu = time.time()
    lu_diag, src_fp = fp.build_season_forward_chained(df, ok, succ_season, f_excl)
    dev, hit, src_dev, dev_diag = off.build_deviations(df, ok, succ_season, f_excl)
    dev_diag["build_seconds"] = round(time.time() - t_lu, 1)
    print(f"  lookups + deviations built in {dev_diag['build_seconds']}s "
          f"(cross-check max |diff| vs features_platoon = "
          f"{dev_diag['cross_check_max_abs_dev_vs_features_platoon']:g})", flush=True)
    assert np.array_equal(np.isnan(src_fp), np.isnan(src_dev))
    assert np.allclose(src_fp[~np.isnan(src_fp)], src_dev[~np.isnan(src_dev)])

    for c in CAT_COLS:
        df[c] = df[c].astype("category")

    grid = curve_grid(N_ROUNDS)
    cat = [c for c in CAT_COLS if c in base_feats]
    feats = list(base_feats)                      # 47 columns, every arm, always
    assert not (set(feats) & set(fp.ALL_COLS)), "platoon columns leaked into X"
    assert not (set(feats) & set(off.DEV_SPEC)), "deviation columns leaked into X"
    assert_no_recovered_in_features(feats, "exp_019 design matrix")

    metrics = {
        "exp_id": "exp_019",
        "base_exp": "exp_010 D (47 provided columns, strongly regularised LGBM); "
                    "diagnosis source exp_018",
        "question": "does a deviation-offset slope fitted only on earlier seasons "
                    "transfer to the held-out season?",
        "changed_vs_exp_018": [
            "stage-2 offset added (C1/C2/C3); C0 is spec-identical to exp_018 A0",
            "platoon columns are NOT in the design matrix at all",
            "two extra stage-1 blocks (2020, 2021) exist to source OOF residuals",
            "fold_2022 is reported at a chained P10 iteration, not its own argmin",
            "prelim_stats not re-run (the lookup is byte-identical)",
        ],
        "changed_vs_exp_010_D": [
            "iteration policy P4 -> P10 (x2.75)",
            "stage-2 offset (C1/C2/C3)",
        ],
        "prereg": PREREG,
        "seed_primary": SEED, "seed_repeats": SEED_REPEATS,
        "num_threads": N_THREADS, "lgb_params": D_PARAMS,
        "num_boost_round_cap": N_ROUNDS, "p10_c": P10_C,
        "n_features_every_arm": len(feats),
        "mask": "training rows with season<=2022 & game_type=='F' dropped "
                "(exp_004); validation untouched",
        "n_rows_masked": int(f_excl.sum()),
        "eb_k_used": {k: v[1] for k, v in fp.RATE_SPEC.items()},
        "recovery_diagnostics": diag,
        "platoon_lookup": lu_diag,
        "deviation_diag": dev_diag,
        "leak_boundary": {
            "recovered_labels_used_as": "LOOKUP-TABLE AGGREGATION INPUT ONLY",
            "design_matrix": "47 provided columns only, in every arm",
            "as_of": "row of season s uses a table built from seasons <= s-1",
            "slope_fit": "only blocks with season < the fold's validation season",
            "d_on_lookup_miss": "exactly 0 — the estimator's own no-op, not a "
                                "sentinel fill of a missing value",
        },
        "sha256": {"train.csv": sha256_file(DATA_PATH),
                   "train.py": sha256_file(os.path.abspath(__file__)),
                   "offset.py": sha256_file(os.path.join(SRC_DIR, "offset.py")),
                   "labels.py": sha256_file(os.path.join(SRC_DIR, "labels.py")),
                   "features_platoon.py": sha256_file(
                       os.path.join(SRC_DIR, "features_platoon.py"))},
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__, "lightgbm": lgb.__version__},
        "gate_spec": GATE,
    }
    diagnostics = {"exp_id": "exp_019", "per_seed": {}}

    def save():
        with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
            json.dump(json.loads(json.dumps(metrics, default=str)), f,
                      indent=2, ensure_ascii=False)
        with open(os.path.join(OUT_DIR, "diagnostics.json"), "w") as f:
            json.dump(json.loads(json.dumps(diagnostics, default=str)), f,
                      indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------ #
    def fit_blocks(seed):
        """Fit blocks 2020..2024 in order; return per-block predictions."""
        params = dict(D_PARAMS, seed=seed)
        blocks, prev_argmin, chain = {}, None, []
        for s in BLOCKS:
            tr = (season <= s - 1) & ~f_excl
            va = season == s
            y_va = y_all[va]

            # iterations are known BEFORE the fit (they come from block s-1)
            if prev_argmin is None:
                k10 = k4 = None
            else:
                k10 = nearest_grid_iter(grid, int(round(P10_C * prev_argmin)))
                k4 = nearest_grid_iter(grid, prev_argmin)

            # as-of assert for every row this block touches
            used = src_fp[tr | va]
            mx = np.nanmax(used) if np.any(~np.isnan(used)) else -np.inf
            assert mx < s, f"AS-OF VIOLATION block {s}: lookup used season {mx}"

            if s == 2024:
                assert int(va.sum()) == EXPECTED_2024["valid_all"], int(va.sum())
                assert int(is_r_all[va].sum()) == EXPECTED_2024["valid_R"]

            t0 = time.time()
            dtr = lgb.Dataset(df.loc[tr, feats], label=y_all[tr],
                              categorical_feature=cat)
            bst = lgb.train(params, dtr, num_boost_round=N_ROUNDS)
            t1 = time.time()
            marg = raw_margin_curve(bst, df.loc[va, feats], grid)
            cv = {k: float(np.mean((sigmoid(marg[k]) - y_va) ** 2)) for k in grid}
            a_min, a_val = argmin_of(cv)
            if k10 is None:
                k10 = k4 = a_min
                src_txt = "own argmin (no predecessor; ORACLE, feeder only)"
            else:
                src_txt = f"P10: argmin(block {s-1}) x {P10_C}"

            sens_iters = sorted({a_min, k10,
                                 nearest_grid_iter(grid, max(1, k10 // 2)),
                                 nearest_grid_iter(grid, min(N_ROUNDS, k10 * 2))})
            blocks[s] = {
                "season": s, "n_train": int(tr.sum()), "n_valid": int(va.sum()),
                "iter_P10": int(k10), "iter_P4": int(k4), "iter_source": src_txt,
                "oracle_iter": a_min, "oracle_brier": round(a_val, 6),
                "fit_min": round((t1 - t0) / 60, 2),
                "curve": {str(k): round(v, 6) for k, v in cv.items()},
                "_y": y_va, "_is_r": is_r_all[va], "_hit": hit[va],
                "_d_s": dev["d_s"][va], "_d_r": dev["d_r"][va],
                "_f_excl": f_excl[va],
                "_p10": sigmoid(marg[k10]), "_p4": sigmoid(marg[k4]),
                "_sens": {int(k): sigmoid(marg[k]) for k in sens_iters},
            }
            chain.append({"block": s, "argmin": a_min, "iter_P10": int(k10),
                          "capped": bool(round(P10_C * (prev_argmin or 0)) > N_ROUNDS)})
            print(f"    [s{seed} block {s}] fit {(t1-t0)/60:.2f} min "
                  f"({int(tr.sum()):,} x {len(feats)}) -> argmin {a_min} = "
                  f"{a_val:.6f}, reported iter {k10} ({src_txt})", flush=True)
            prev_argmin = a_min
            del bst, marg, dtr
            gc.collect()
        return blocks, chain

    def slope_pack(blocks, valid_season, pred_key="_p10", dname="d_s",
                   drop_masked=False, r_only=False):
        """Pool the eligible earlier blocks and fit the offset coefficients."""
        use = [s for s in SLOPE_BLOCKS if s < valid_season]
        assert use, valid_season
        assert max(use) < valid_season, "SLOPE LEAK: block season >= valid season"
        r_l, d_l, p_l, y_l = [], [], [], []
        for s in use:
            b = blocks[s]
            keep = np.ones(len(b["_y"]), dtype=bool)
            if drop_masked:
                keep &= ~b["_f_excl"]
            if r_only:
                keep &= b["_is_r"]
            p = b[pred_key][keep]
            y = b["_y"][keep]
            d = b["_" + dname][keep]
            r_l.append(y - p)
            d_l.append(d)
            p_l.append(p)
            y_l.append(y)
            # the lookup feeding these rows can only be older still
            assert np.nanmax(src_fp[season == s]) < valid_season
        return (np.concatenate(r_l), np.concatenate(d_l),
                np.concatenate(p_l), np.concatenate(y_l), use)

    def eval_arms(b, betas, valid_season):
        """C0..C3 on one fold's validation rows."""
        y, is_r = b["_y"], b["_is_r"]
        p0 = b["_p10"]
        preds = {
            "C0": p0,
            "C1": off.apply_linear(p0, b["_d_s"], betas["beta_s"]),
            "C2": off.apply_logit(p0, b["_d_s"], betas["gamma_s"]),
            "C3": off.apply_linear(p0, b["_d_r"], betas["beta_r"]),
        }
        out = {}
        for a, p in preds.items():
            out[a] = {"valid_all": seg_metrics(y, p),
                      "valid_R": seg_metrics(y[is_r], p[is_r]),
                      "valid_F": seg_metrics(y[~is_r], p[~is_r]),
                      "delta_vs_C0": round(
                          float(np.mean((p - y) ** 2) - np.mean((p0 - y) ** 2)), 9)}
        return out, preds

    # ------------------------------------------------------------------ #
    def run_seed(seed):
        print(f"[{time.strftime('%H:%M:%S')}] ===== seed {seed} =====", flush=True)
        blocks, chain = fit_blocks(seed)
        res = {"seed": seed, "iteration_chain": chain,
               "blocks": {str(s): {k: v for k, v in blocks[s].items()
                                   if not k.startswith("_")}
                          for s in BLOCKS}}
        dgn = {"seed": seed}

        # ---- per-fold: fit offsets on earlier blocks only, then apply ---- #
        for V in FOLD_SEASONS:
            b = blocks[V]
            r, d_s, p_oof, y_oof, used = slope_pack(blocks, V, dname="d_s")
            _, d_r, _, _, _ = slope_pack(blocks, V, dname="d_r")
            fs = off.fit_slope(r, d_s)
            fr = off.fit_slope(r, d_r)
            fg = off.fit_gamma(p_oof, y_oof, d_s)
            betas = {"beta_s": fs["beta"], "beta_r": fr["beta"],
                     "gamma_s": fg["gamma"]}
            arms, preds = eval_arms(b, betas, V)

            res[f"fold_{V}"] = {
                "iter_used": b["iter_P10"], "iter_source": b["iter_source"],
                "oracle_iter": b["oracle_iter"], "oracle_brier": b["oracle_brier"],
                "n_train": b["n_train"], "fit_min": b["fit_min"],
                "slope_blocks_used": used,
                "max_block_season_used": int(max(used)),
                "valid_season": V,
                "assert_max_block_lt_valid": bool(max(used) < V),
                "offsets": {"linear_d_s": fs, "linear_d_r": fr, "logit_d_s": fg},
                "betas_applied": betas,
                "arms": arms,
            }
            print(f"  [s{seed} fold_{V}] blocks {used} -> beta_s {fs['beta']} "
                  f"(se {fs['se']}), beta_r {fr['beta']}, gamma_s {fg['gamma']}",
                  flush=True)
            print("    " + json.dumps(
                {a: {"brier": arms[a]["valid_all"]["brier"],
                     "d": arms[a]["delta_vs_C0"]} for a in arms}), flush=True)

            # ---- P4 shadow of C0 (the reproduction gate lives here) ------ #
            res[f"fold_{V}_C0_P4"] = {
                "iter_used": b["iter_P4"],
                "valid_all": seg_metrics(b["_y"], b["_p4"]),
            }
            if V == 2024:
                np.save(os.path.join(OUT_DIR, f"pred_C0_seed{seed}_fold_2024.npy"),
                        preds["C0"])
                for a in ["C1", "C2", "C3"]:
                    np.save(os.path.join(
                        OUT_DIR, f"pred_{a}_seed{seed}_fold_2024.npy"), preds[a])

        # ================= diagnostics ================================== #
        b24 = blocks[2024]
        d_s24, d_r24, y24, p24, hit24 = (b24["_d_s"], b24["_d_r"], b24["_y"],
                                         b24["_p10"], b24["_hit"])

        # D0 — replicate the orchestrator's headroom arithmetic on this seed
        r24 = y24 - p24
        e_d2 = float(np.mean(d_s24 ** 2))
        dgn["D0_headroom_replication"] = {
            "lookup_hit_rate_pct": round(100.0 * float(hit24.mean()), 3),
            "sd_d_s_hit_rows": round(float(np.std(d_s24[hit24])), 6),
            "sd_d_s_all_rows": round(float(np.std(d_s24)), 6),
            "E_d_s_squared_all_rows": round(e_d2, 10),
            "ceiling_if_perfect": round(e_d2, 10),
            "oracle_slope_on_valid_2024": round(
                float(np.dot(d_s24, r24) / np.dot(d_s24, d_s24)), 6),
            "corr_d_s_d_r_hit_rows": round(
                float(np.corrcoef(d_s24[hit24], d_r24[hit24])[0, 1]), 4),
            "orchestrator_values_for_comparison": {
                "lookup_hit_rate_pct": 78.5, "sd_d_s": 0.01997,
                "ceiling_if_perfect": 3.13e-4, "residual_slope": 0.614,
                "corr_d_s_d_r": -0.787},
        }

        # D1 — slope stability across seasons
        d1 = {}
        for dname in ["d_s", "d_r"]:
            rows = []
            for s in SLOPE_BLOCKS:
                bb = blocks[s]
                f = off.fit_slope(bb["_y"] - bb["_p10"], bb["_" + dname])
                f["season"] = s
                f["iter_used"] = bb["iter_P10"]
                f["oracle_iter_flag"] = bool(s == BLOCKS[0])
                rows.append(f)
            ok_rows = [x for x in rows if x["se"]]
            trend = (off.weighted_trend([x["season"] for x in ok_rows],
                                        [x["beta"] for x in ok_rows],
                                        [x["se"] for x in ok_rows])
                     if len(ok_rows) >= 3 else None)
            pooled = off.fit_slope(
                *[np.concatenate(z) for z in zip(
                    *[(blocks[s]["_y"] - blocks[s]["_p10"], blocks[s]["_" + dname])
                      for s in SLOPE_BLOCKS])])
            d1[dname] = {"per_season": rows, "trend": trend,
                         "pooled_2020_2023": pooled}
            if trend:
                d1[dname]["extrapolation_gap_vs_pooled"] = round(
                    pooled["beta"] - trend["extrapolated_2024"], 6)
        dgn["D1_slope_stability"] = d1

        # D2 — oracle gap on fold_2024
        f24 = res["fold_2024"]
        d2 = {}
        for dname, key, applier in [("d_s", "beta_s", off.apply_linear),
                                    ("d_r", "beta_r", off.apply_linear)]:
            dd = b24["_" + dname]
            b_or = float(np.dot(dd, r24) / np.dot(dd, dd))
            p_or = applier(p24, dd, b_or)
            p_oof = applier(p24, dd, f24["betas_applied"][key])
            base = float(np.mean((p24 - y24) ** 2))
            d2[dname] = {
                "beta_oof": f24["betas_applied"][key],
                "beta_oracle_fitted_on_valid_2024": round(b_or, 6),
                "ratio_oof_over_oracle": round(
                    f24["betas_applied"][key] / b_or, 4) if b_or else None,
                "brier_delta_oof": round(float(np.mean((p_oof - y24) ** 2)) - base, 9),
                "brier_delta_oracle": round(
                    float(np.mean((p_or - y24) ** 2)) - base, 9),
                "transfer_efficiency_pct": None,
            }
            do, dr_ = d2[dname]["brier_delta_oof"], d2[dname]["brier_delta_oracle"]
            if dr_ < 0:
                d2[dname]["transfer_efficiency_pct"] = round(100.0 * do / dr_, 2)
        dgn["D2_oracle_gap"] = d2

        # D3 — the same regression on the exp_016 B3 fold_2024 prediction
        b3 = None
        for cand in ["pred_B3_seed42_fold_2024.npy", "pred_B3_fold_2024.npy",
                     "pred_B3_seed%d_fold_2024.npy" % seed]:
            pth = os.path.join(EXP016_DIR, cand)
            if os.path.exists(pth):
                b3 = pth
                break
        if b3:
            pb3 = np.load(b3)
            dgn["D3_b3_headroom"] = {
                "source": b3,
                "slope_d_s": off.fit_slope(y24 - pb3, d_s24),
                "slope_d_r": off.fit_slope(y24 - pb3, d_r24),
                "note": "ORACLE slope (fitted on valid 2024) — a headroom "
                        "measurement on the B3 lineage, not an achievable gain",
            }
        else:
            dgn["D3_b3_headroom"] = {
                "status": "UNAVAILABLE — not measured",
                "checked": EXP016_DIR,
                "found_instead": sorted(
                    [f for f in os.listdir(EXP016_DIR)
                     if f.endswith((".npy", ".txt"))]) if os.path.isdir(EXP016_DIR)
                    else "directory missing",
                "note": "config.yaml D3.if_missing says do not regenerate. "
                        "Reconstructing B3 needs its forward-chaining OOF stage, "
                        "not a single fit.",
            }

        # D4 — coverage decomposition on fold_2024
        d4 = {}
        for arm in ["C1", "C2", "C3"]:
            pa = np.load(os.path.join(OUT_DIR, f"pred_{arm}_seed{seed}_fold_2024.npy"))
            se_a = (pa - y24) ** 2
            se_0 = (p24 - y24) ** 2
            n = len(y24)
            d4[arm] = {
                "n_hit": int(hit24.sum()), "n_miss": int((~hit24).sum()),
                "hit_pct": round(100.0 * float(hit24.mean()), 3),
                "brier_delta_overall": round(float(se_a.mean() - se_0.mean()), 9),
                "brier_delta_within_hit_rows": round(
                    float(se_a[hit24].mean() - se_0[hit24].mean()), 9),
                "contribution_of_hit_rows_to_overall": round(
                    float((se_a[hit24] - se_0[hit24]).sum() / n), 9),
                "contribution_of_miss_rows_to_overall": round(
                    float((se_a[~hit24] - se_0[~hit24]).sum() / n), 9),
                "extrapolated_if_coverage_100pct": round(
                    float((se_a[hit24] - se_0[hit24]).mean()), 9),
            }
        dgn["D4_coverage"] = d4

        # D5 — sensitivity of the pooled slope to the residual iteration and
        #      to the rows included.  Reported, never used to choose anything.
        sens = {"iteration": {}, "row_population": {}}
        common = sorted(set.intersection(*[set(blocks[s]["_sens"]) for s in SLOPE_BLOCKS]))
        for k in common:
            rr, dd = [], []
            for s in SLOPE_BLOCKS:
                rr.append(blocks[s]["_y"] - blocks[s]["_sens"][k])
                dd.append(blocks[s]["_d_s"])
            sens["iteration"][str(k)] = off.fit_slope(
                np.concatenate(rr), np.concatenate(dd))["beta"]
        for tag, kw in [("all_rows_PREREG", {}),
                        ("drop_exp004_masked", {"drop_masked": True}),
                        ("R_league_only", {"r_only": True})]:
            r_, d_, _, _, _ = slope_pack(blocks, 2024, dname="d_s", **kw)
            sens["row_population"][tag] = off.fit_slope(r_, d_)
        # which blocks are pooled.  Block 2020 is fitted on 2019 alone and is
        # therefore the weakest stage-1 model in the chain; if it biases the
        # pooled slope it will show up here.
        sens["block_subset"] = {}
        for subset in [(2020, 2021, 2022, 2023), (2021, 2022, 2023),
                       (2022, 2023), (2023,)]:
            rr = np.concatenate([blocks[s]["_y"] - blocks[s]["_p10"] for s in subset])
            dd = np.concatenate([blocks[s]["_d_s"] for s in subset])
            sens["block_subset"]["+".join(str(s) for s in subset)] = \
                off.fit_slope(rr, dd)["beta"]
        dgn["D5_sensitivity"] = sens

        # D6 — what fold_2024 Brier OTHER beta choices would have given.
        # POST-HOC EXPLORATION ONLY.  The reported result is the pre-registered
        # pooled-OOF beta in `arms`; nothing here selects anything.
        base24 = float(np.mean((p24 - y24) ** 2))
        alts = {"prereg_pooled_2020_2023": f24["betas_applied"]["beta_s"]}
        for tag in ["2021+2022+2023", "2022+2023", "2023"]:
            alts["pooled_" + tag] = sens["block_subset"][tag]
        tr_ = d1["d_s"]["trend"]
        if tr_:
            alts["trend_extrapolated_2024"] = tr_["extrapolated_2024"]
        alts["oracle_on_valid_2024"] = \
            dgn["D2_oracle_gap"]["d_s"]["beta_oracle_fitted_on_valid_2024"]
        dgn["D6_alternative_betas_POSTHOC"] = {
            "warning": "diagnostic only — these numbers touched valid 2024 after "
                       "the fact and must never be used to pick beta",
            "C0_brier": round(base24, 9),
            "linear_d_s": {t: round(float(np.mean(
                (off.apply_linear(p24, d_s24, bt) - y24) ** 2)) - base24, 9)
                for t, bt in alts.items()},
            "betas": alts,
        }

        del blocks
        gc.collect()
        return res, dgn

    # ---------------- seeds ------------------------------------------- #
    seeds = [SEED] + list(SEED_REPEATS)
    per_seed = {}
    for sd in seeds:
        r, g = run_seed(sd)
        per_seed[sd] = r
        metrics[f"seed{sd}"] = r
        diagnostics["per_seed"][str(sd)] = g
        metrics["elapsed_min"] = round((time.time() - t_start) / 60, 2)
        save()

        if sd == SEED:
            got = r["fold_2024_C0_P4"]["valid_all"]["brier"]
            dgap = abs(got - GATE["fold_2024_valid_all_brier"])
            gate = {"expected": GATE["fold_2024_valid_all_brier"], "got": got,
                    "abs_diff": round(dgap, 8),
                    "expected_iter": GATE["fold_2024_iter"],
                    "got_iter": r["fold_2024_C0_P4"]["iter_used"],
                    "policy": GATE["policy"],
                    "passed": bool(dgap <= GATE["tolerance_brier"])}
            # stronger: compare the prediction vector to exp_018 A0
            pth = GATE_PRED.format(seed=sd)
            if os.path.exists(pth):
                a0 = np.load(pth)
                mine = np.load(os.path.join(
                    OUT_DIR, f"pred_C0_seed{sd}_fold_2024.npy"))
                gate["exp018_A0_pred_max_abs_diff"] = float(
                    np.max(np.abs(a0 - mine))) if a0.shape == mine.shape else None
                gate["exp018_A0_pred_shape_match"] = bool(a0.shape == mine.shape)
            metrics["C0_reproduction_gate"] = gate
            print("C0 gate: " + json.dumps(gate), flush=True)
            save()
            if not SMOKE:
                assert gate["passed"], (
                    f"C0 failed to reproduce exp_010 D / exp_018 A0 ({got} vs "
                    f"{GATE['fold_2024_valid_all_brier']}) — everything "
                    "downstream would be uninterpretable, aborting.")

    # ---------------- seed summary / verdict --------------------------- #
    arms = ["C0", "C1", "C2", "C3"]
    tab = {a: {str(s): per_seed[s]["fold_2024"]["arms"][a]["valid_all"]["brier"]
               for s in seeds} for a in arms}
    spread = {}
    for a in arms:
        v = [tab[a][str(s)] for s in seeds]
        spread[a] = {"per_seed": tab[a], "mean": round(float(np.mean(v)), 6),
                     "std": round(float(np.std(v, ddof=1)), 6) if len(v) > 1 else None,
                     "range": round(float(max(v) - min(v)), 6)}
    c0_std = spread["C0"]["std"]
    verdicts = {}
    for a in ["C1", "C2", "C3"]:
        ds = [tab[a][str(s)] - tab["C0"][str(s)] for s in seeds]
        md = float(np.mean(ds))
        same = bool(all(x < 0 for x in ds) or all(x > 0 for x in ds))
        verdicts[a] = {
            "per_seed_delta": {str(s): round(tab[a][str(s)] - tab["C0"][str(s)], 6)
                               for s in seeds},
            "mean_delta": round(md, 8),
            "std_delta": round(float(np.std(ds, ddof=1)), 8) if len(ds) > 1 else None,
            "all_seeds_same_sign": same,
            "C0_seed_std": c0_std,
            "abs_mean_delta_over_C0_std": (round(abs(md) / c0_std, 2)
                                           if c0_std else None),
            "criterion": PREREG["decision_rule"],
            "measured": bool(same and c0_std and abs(md) >= 3 * c0_std),
        }
    metrics["seed_summary"] = {"fold_2024_valid_all_brier_by_seed": tab,
                               "spread": spread, "verdicts": verdicts}
    metrics["beta_by_seed"] = {
        str(s): {V: per_seed[s][f"fold_{V}"]["betas_applied"] for V in FOLD_SEASONS}
        for s in seeds}
    print("--- seed summary ---\n" + json.dumps(metrics["seed_summary"], indent=2),
          flush=True)
    print("--- betas ---\n" + json.dumps(metrics["beta_by_seed"], indent=2),
          flush=True)

    metrics["train_time_min"] = round((time.time() - t_start) / 60, 2)
    save()
    print(f"[{time.strftime('%H:%M:%S')}] DONE {metrics['train_time_min']} min",
          flush=True)


if __name__ == "__main__":
    main()
