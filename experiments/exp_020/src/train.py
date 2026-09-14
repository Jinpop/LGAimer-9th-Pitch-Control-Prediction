"""exp_020 — does exp_016 B3 already absorb the conditional deviation d_s?

THE ONE QUESTION
----------------
exp_019 C1 measured a two-stage offset worth -1.04e-04, but ON TOP OF exp_010 D
(a single binary model).  The submission lineage is exp_016 B3 (mode stacking).
exp_019 D3 could not answer whether B3 already carries that information because
no B3 prediction file existed.

    beta_B3 := slope of (y - p_B3) on d_s, fitted on EARLIER blocks only.

      beta_B3 ~ 0     -> B3 absorbed it, the offset adds nothing
      beta_B3 ~ beta_A0 -> B3 absorbed none of it, the two are additive
      in between      -> partial absorption; the ratio IS the headroom left

PRE-REGISTERED DEVIATIONS (config.yaml preregistered_deviations)
----------------------------------------------------------------
DEV1  seed 42 only.  The pre-registered adoption rule (3 seeds same sign AND
      |mean delta| >= 3 x seed std) CANNOT be evaluated.  This run estimates a
      MAGNITUDE; it does not adopt anything.
DEV2  the slope pools blocks 2022+2023 only (exp_019 pooled 2020..2023).
      exp_019 D5 showed the 2022+23 pooled slope on A0 residuals is 0.525 vs
      0.450 for the pre-registered pooling, so this convention can inflate the
      estimate.  The honest control is therefore beta_A0 RE-FITTED ON THE SAME
      TWO BLOCKS, which this script computes from its own A0 fits.
DEV3  the cell-model iterations are inherited from exp_016 per fold.

WHAT IS REUSED AND HOW
----------------------
  src/exp016_ref.py       byte-identical copy of exp_016/src/train.py
                          (sha 4f24deb7...).  Folds 2023 and 2024 of B3 are
                          produced by CALLING `exp016_ref.run_b3` — the literal
                          exp_016 code path, not a re-implementation.  The
                          reproduction gate is what proves it.
  src/offset.py           byte-identical exp_019 (sha 68d6007d...)
  src/features_platoon.py byte-identical exp_018/exp_019 (sha 68d63063...)
  src/labels.py           byte-identical exp_016/exp_019 (sha 439cd504...)

Block 2022 has no exp_016 counterpart (exp_016 ran B3 on two folds only), so
`b3_one_block()` below is a TRANSCRIPTION of `exp016_ref.run_b3`'s loop body for
a single fold.  It is flagged as such; the per-block slopes of 2022 and 2023 are
reported separately so a transcription error would show up as a disagreement.

ARMS (fold_2024 valid_all is the only reported metric)
------------------------------------------------------
  E0   B3, no offset                          <- reproduction gate lives here
  E1   clip(p_B3 + beta_B3 * d_s), beta pooled over blocks 2022+2023
  E2   the same with beta fitted ON valid 2024 — UPPER BOUND DISPLAY ONLY,
       never used to choose anything (config validation.never)

Environment: A5000, ~/SungjinPark/venv (Python 3.11.15 / numpy 1.26.4 /
pandas 2.0.3 / lightgbm 4.7.0).
"""

import gc
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

SMOKE = os.environ.get("EXP020_SMOKE") == "1"
if SMOKE:                       # shrink exp016_ref's N_ROUNDS before importing
    os.environ["EXP016_SMOKE"] = "1"

import lightgbm as lgb                                            # noqa: E402
import exp016_ref as ref                                          # noqa: E402
import features_platoon as fp                                     # noqa: E402
import offset as off                                              # noqa: E402
from labels import (CELLS, FAIL_CELLS,                             # noqa: E402
                    assert_no_recovered_in_features, recover)

SEED = 42
N_THREADS = 6

DATA_PATH = os.path.expanduser("~/SungjinPark/data/train.csv")
OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_020")
EXP019_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_019")

TARGET = "control_success"
DROP_COLS = ["row_id", TARGET]
CAT_COLS = ["top_bottom", "game_type", "base_state"]
F_EXCLUDE_MAX_SEASON = 2022
EXPECTED_2024 = {"valid_all": 253507, "valid_R": 223497}

BLOCKS = [2022, 2023, 2024]          # (train <= V-1) -> predict V
SLOPE_BLOCKS = [2022, 2023]          # DEV2
VALID_SEASON = 2024

# ---- inherited constants (DEV3 + exp_019 chain), all read off saved metrics -- #
INHERITED = {
    "exp_016_B2c_iter_used_per_fold": {      # = B3 cell-model iteration
        "fold_2022": 240, "fold_2023": 240, "fold_2024": 1150},
    "exp_016_B3_stacker_iter_used": {"fold_2023": 650, "fold_2024": 650},
    "exp_016_B3_stacker_argmin": {"fold_2023": 650, "fold_2024": 2800},
    "exp_019_A0_block_iter_P10": {2022: 270, 2023: 825, 2024: 2050},
    "exp_019_A0_block_iter_P4": {2022: 100, 2023: 300, 2024: 750},
    "exp_019_A0_block_argmin": {2022: 300, 2023: 750, 2024: 1550},
    "source": "experiments/exp_016/metrics.json (B2c/B3, seed 42) and "
              "experiments/exp_019/metrics.json (seed42.blocks, seed 42)",
}

GATE = {
    "arm": "E0 = exp_016 B3, fold_2024 valid_all",
    "brier_at_iter_650_P4": 0.248191,
    "iter_P4": 650,
    "brier_at_oracle_iter": 0.248176,
    "oracle_iter": 2800,
    "tolerance": 5e-6,
    "on_fail": "abort — exp_016 code has drifted and nothing below is readable",
}

PREREG = {
    "slope_estimator": "no-intercept least squares of (y - p) on d_s, pooled "
                       "over ALL rows of blocks 2022 and 2023, unweighted "
                       "(exp_019 PREREG, only the block set differs = DEV2)",
    "block_iteration_convention_B3": "each block at the iteration exp_016 used "
                                     "for that fold; block 2022 has no exp_016 "
                                     "value and uses its own argmin (ORACLE, "
                                     "feeder only, flagged).  Both blocks are "
                                     "therefore at their own argmin.",
    "block_iteration_convention_A0_primary": "own argmin, to MATCH the B3 "
                                             "convention — this is the fair "
                                             "control for the ratio",
    "A0_secondary_conventions": "P10 (reproduces exp_019 D5 = 0.525) and P4",
    "applied_to_fold_2024": "beta pooled over blocks 2022+2023, applied to the "
                            "B3 prediction at iter 650 (exp_016 P4 policy)",
    "no_adoption": "DEV1 — one seed.  Magnitude estimate only, no verdict.",
    "E2_is_display_only": "beta fitted on valid 2024 is an upper bound; it is "
                          "never applied to any reported arm choice",
}

# iterations at which the slope is re-fitted for the D5 sensitivity.  A FIXED
# common set, because exp_019's D5 used per-block candidate sets whose
# intersection came out empty (that experiment's reported code defect).
SENS_ITERS = [100, 300, 650, 1150, 2050, 2800]

if SMOKE:
    OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_020_smoke")
    INHERITED["exp_016_B2c_iter_used_per_fold"] = {
        "fold_2022": 5, "fold_2023": 5, "fold_2024": 5}
    INHERITED["exp_016_B3_stacker_iter_used"] = {"fold_2023": 10, "fold_2024": 10}
    INHERITED["exp_016_B3_stacker_argmin"] = {"fold_2023": 10, "fold_2024": 10}
    INHERITED["exp_019_A0_block_iter_P10"] = {2022: 10, 2023: 10, 2024: 10}
    INHERITED["exp_019_A0_block_iter_P4"] = {2022: 5, 2023: 5, 2024: 5}
    INHERITED["exp_019_A0_block_argmin"] = {2022: 10, 2023: 10, 2024: 10}
    SENS_ITERS = [5, 10, 20, 40]


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


def partial_corr(a, b, Z):
    """corr(a, b) after linearly removing the columns of Z (with intercept)."""
    X = np.column_stack([np.ones(len(a))] + [Z[:, j] for j in range(Z.shape[1])])
    ra = a - X @ np.linalg.lstsq(X, a, rcond=None)[0]
    rb = b - X @ np.linalg.lstsq(X, b, rcond=None)[0]
    sa, sb = float(np.std(ra)), float(np.std(rb))
    if sa <= 0 or sb <= 0:
        return None
    return float(np.mean((ra - ra.mean()) * (rb - rb.mean())) / (sa * sb))


def r2_of(y, Z):
    X = np.column_stack([np.ones(len(y))] + [Z[:, j] for j in range(Z.shape[1])])
    fit = X @ np.linalg.lstsq(X, y, rcond=None)[0]
    ss = float(np.sum((y - y.mean()) ** 2))
    return float(1.0 - np.sum((y - fit) ** 2) / ss) if ss > 0 else None


# --------------------------------------------------------------------------- #
def b3_one_block(df, base_feats, cat, season, f_excl, ok, succ_season, y_all,
                 seed, grid, stash, fold_name, tu, vy, cell_iter):
    """TRANSCRIPTION of exp016_ref.run_b3's per-fold body, for ONE fold.

    exp_016 hard-codes the fold list [fold_2023, fold_2024] inside run_b3, so
    block 2022 cannot be produced by calling it.  Everything below is copied
    line-for-line from that function with the loop unrolled; the only removed
    piece is the P4 carry (this block is the head of the chain and therefore
    reports its own argmin, ORACLE, feeder only).
    """
    params = dict(ref.D_PARAMS, seed=seed)
    stack_cols = [f"p_{c}" for c in FAIL_CELLS]
    t0 = time.time()

    tr_full = (season <= tu) & ~f_excl
    va = season == vy
    y_va = y_all[va]
    is_r = (df.loc[va, "game_type"] == "R").to_numpy()
    seasons = sorted(set(season[tr_full].tolist()))
    print(f"[{time.strftime('%H:%M:%S')}] B3 {fold_name} (transcribed): OOF "
          f"blocks {seasons[1:]} at iter {cell_iter}", flush=True)

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
              f"pred {int(pred_rows.sum()):,} ({time.time()-t0:.0f}s)", flush=True)
    for c in FAIL_CELLS:
        stack[f"p_{c}"][va] = stash[c]

    Xs = df[base_feats].copy()
    for c in stack_cols:
        Xs[c] = stack[c]
    feats = base_feats + stack_cols
    assert_no_recovered_in_features(feats, f"B3 stacker {fold_name}")
    _, marg = ref.fit_curve(Xs.loc[tr_full], y_all[tr_full], Xs.loc[va], params,
                            ref.N_ROUNDS, cat, grid, f"B3 stacker {fold_name}")
    curve = ref.brier_curve(grid, lambda k: np.mean((ref.sigmoid(marg[k]) - y_va) ** 2))
    a_min, a_val = ref.argmin_of(curve)
    k = a_min
    p = ref.sigmoid(marg[k])
    out = {
        "iter_used": int(k), "oracle_iter": a_min, "oracle_brier": round(a_val, 6),
        "iter_source": "own argmin (no exp_016 predecessor; ORACLE, feeder only)",
        "oracle_flag": True,
        "cell_model_iter_used_for_stack_features": int(cell_iter),
        "n_train": int(tr_full.sum()),
        "nan_stack_rows_in_train_pct": round(
            100.0 * float(np.isnan(stack[stack_cols[0]][tr_full]).mean()), 3),
        "valid_all": ref.seg_metrics(y_va, p),
        "valid_R": ref.seg_metrics(y_va[is_r], p[is_r]),
        "valid_F": ref.seg_metrics(y_va[~is_r], p[~is_r]),
        "minutes": round((time.time() - t0) / 60, 2),
        "curve": {str(kk): round(vv, 6) for kk, vv in curve.items()},
        "provenance": "TRANSCRIBED from exp016_ref.run_b3 (single-fold unroll)",
    }
    del Xs, stack
    gc.collect()
    return out, marg


# --------------------------------------------------------------------------- #
def main():
    t_start = time.time()
    np.random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "pred"), exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] loading {DATA_PATH}", flush=True)
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    print(f"loaded {df.shape}", flush=True)
    assert df["row_id"].is_unique
    assert df["season"].is_monotonic_increasing, "file order assumption broken"

    base_feats = [c for c in df.columns if c not in DROP_COLS]
    assert len(base_feats) == 47, len(base_feats)
    assert_no_recovered_in_features(base_feats, "base_feats (before recovery)")

    rec_diag = recover(df)
    assert rec_diag["success_recovery"]["agree_pct"] == 100.0

    season = df["season"].to_numpy()
    ok = df["recov_ok"].to_numpy()
    succ_season = df.groupby("pitcher_id", sort=False)["season"].shift(-1).to_numpy()
    f_excl = ((df["season"] <= F_EXCLUDE_MAX_SEASON)
              & (df["game_type"] == "F")).to_numpy()
    y_all = df[TARGET].to_numpy(dtype=np.float64)
    is_r_all = (df["game_type"] == "R").to_numpy()

    # ---- platoon lookups (exp_018 machinery) + deviations (exp_019) -------- #
    print(f"[{time.strftime('%H:%M:%S')}] building platoon lookups", flush=True)
    t_lu = time.time()
    lu_diag, src_fp = fp.build_season_forward_chained(df, ok, succ_season, f_excl)
    dev, hit, src_dev, dev_diag = off.build_deviations(df, ok, succ_season, f_excl)
    dev_diag["build_seconds"] = round(time.time() - t_lu, 1)
    assert np.array_equal(np.isnan(src_fp), np.isnan(src_dev))
    assert np.allclose(src_fp[~np.isnan(src_fp)], src_dev[~np.isnan(src_dev)])
    print(f"  lookups + deviations in {dev_diag['build_seconds']}s (cross-check "
          f"max|diff| = {dev_diag['cross_check_max_abs_dev_vs_features_platoon']:g})",
          flush=True)

    for c in CAT_COLS:
        df[c] = df[c].astype("category")

    grid = ref.curve_grid(ref.N_ROUNDS)
    cat = [c for c in CAT_COLS if c in base_feats]
    assert not (set(base_feats) & set(fp.ALL_COLS)), "platoon cols in X"
    assert not (set(base_feats) & set(off.DEV_SPEC)), "deviation cols in X"

    # ---- as-of asserts, once, for every season this run touches ------------ #
    asof = {}
    for V in BLOCKS:
        used = src_fp[season <= V]
        mx = float(np.nanmax(used)) if np.any(~np.isnan(used)) else float("-inf")
        assert mx < V, f"AS-OF VIOLATION block {V}: lookup used season {mx}"
        asof[str(V)] = mx
    assert float(np.nanmax(src_fp[season == VALID_SEASON])) < VALID_SEASON

    metrics = {
        "exp_id": "exp_020",
        "base_exp": "exp_016 B3 (mode stacking); offset machinery from exp_019 C1",
        "question": "beta_B3 — does exp_016 B3 already absorb the conditional "
                    "deviation d_s that exp_019 C1 exploited on top of exp_010 D?",
        "preregistered_deviations": {
            "DEV1": "seed 42 only -> NO adoption verdict is possible",
            "DEV2": "slope blocks = 2022+2023 only; beta_A0 is re-fitted on the "
                    "SAME blocks here as the honest control",
            "DEV3": "cell-model iterations inherited from exp_016 per fold",
        },
        "changed_vs_exp_016_B3": [
            "a third B3 block (fold_2022, train<=2021) is added — feeder only",
            "stage-2 linear offset on d_s (E1/E2) applied to the B3 prediction",
            "seed 42 only (exp_016 ran 3 seeds)",
            "cell-model and stacker iterations are INHERITED, not re-derived, "
            "for folds 2023/2024; block 2022 uses its own argmin",
        ],
        "changed_vs_exp_019_C1": [
            "stage 1 is B3 (mode stacking) instead of exp_010 D single binary",
            "slope pooled over blocks 2022+2023 instead of 2020..2023 (DEV2)",
            "seed 42 only instead of 3 seeds (DEV1)",
            "block iteration convention: own argmin instead of P10",
        ],
        "prereg": PREREG, "inherited_constants": INHERITED, "gate_spec": GATE,
        "seed": SEED, "num_threads": N_THREADS, "lgb_params": ref.D_PARAMS,
        "num_boost_round_cap": ref.N_ROUNDS,
        "mask": "training rows with season<=2022 & game_type=='F' dropped "
                "(exp_004); validation untouched",
        "n_rows_masked": int(f_excl.sum()),
        "eb_k_used": {k: v[1] for k, v in fp.RATE_SPEC.items()},
        "recovery_diagnostics": rec_diag,
        "platoon_lookup": lu_diag,
        "deviation_diag": dev_diag,
        "asof_max_lookup_season_by_block": asof,
        "leak_boundary": {
            "recovered_labels_used_as": "TARGET (cell models) and LOOKUP-TABLE "
                                        "AGGREGATION INPUT (platoon) ONLY",
            "design_matrix": "47 provided columns (+ 4 OOF cell probabilities "
                             "for the B3 stacker); no recovered column, no "
                             "platoon column, no deviation column",
            "as_of": "row of season s uses a table built from seasons <= s-1",
            "slope_fit": "blocks 2022 and 2023 only; valid season is 2024",
            "d_on_lookup_miss": "exactly 0 — the estimator's own no-op, not a "
                                "sentinel fill (exp_019 wording, same code)",
        },
        "sha256": {
            "train.csv": sha256_file(DATA_PATH),
            "train.py": sha256_file(os.path.abspath(__file__)),
            "exp016_ref.py": sha256_file(os.path.join(SRC_DIR, "exp016_ref.py")),
            "offset.py": sha256_file(os.path.join(SRC_DIR, "offset.py")),
            "labels.py": sha256_file(os.path.join(SRC_DIR, "labels.py")),
            "features_platoon.py": sha256_file(
                os.path.join(SRC_DIR, "features_platoon.py"))},
        "sha256_expected": {
            "exp016_ref.py": "4f24deb71d9eef3d626900f34a71aecd2d6acbae646eb9c92"
                             "c51189a38a3ece1",
            "offset.py": "68d6007db03ab850e2a5181383a4148fae8947c711558282c98"
                         "46391f13e0488",
            "labels.py": "439cd504478b4bc8162e9dedc32be27bfadd5d1178472d6b384"
                         "b60f46ed7a9f2",
            "features_platoon.py": "68d63063e8ce659cf2876fbca4f5adef7ee1e720620"
                                   "ae63bb78ea496c8dadf09"},
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__, "lightgbm": lgb.__version__},
    }
    for k, want in metrics["sha256_expected"].items():
        assert metrics["sha256"][k] == want, (k, metrics["sha256"][k], want)
    diagnostics = {"exp_id": "exp_020"}

    def save():
        with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
            json.dump(json.loads(json.dumps(metrics, default=str)), f,
                      indent=2, ensure_ascii=False)
        with open(os.path.join(OUT_DIR, "diagnostics.json"), "w") as f:
            json.dump(json.loads(json.dumps(diagnostics, default=str)), f,
                      indent=2, ensure_ascii=False)

    save()
    params = dict(ref.D_PARAMS, seed=SEED)

    # ------------------------------------------------------------------ #
    #  cell models on the valid rows (the `stash` exp_016 B3 consumes)    #
    # ------------------------------------------------------------------ #
    cell_probs = {}          # fold -> {cell: p on that fold's valid rows}
    cell_own = {}

    def build_stash(fold_name, tu, vy):
        cell_iter = INHERITED["exp_016_B2c_iter_used_per_fold"][fold_name]
        tr_full = (season <= tu) & ~f_excl
        tr_mode = tr_full & ok & (succ_season <= tu)
        va = season == vy
        assert_no_recovered_in_features(base_feats, f"{fold_name} cell models")
        X_mode = df.loc[tr_mode, base_feats]
        X_va = df.loc[va, base_feats]
        st, own = {"_iter": int(cell_iter)}, {}
        for c in FAIL_CELLS:
            yc = df.loc[tr_mode, c].to_numpy(dtype=np.float64)
            assert not np.isnan(yc).any()
            _, marg = ref.fit_curve(X_mode, yc, X_va, params, ref.N_ROUNDS, cat,
                                    grid, f"{c} {fold_name}")
            st[c] = ref.sigmoid(marg[cell_iter])
            yv = df.loc[va, c].to_numpy(dtype=np.float64)
            vok = ok[va]
            own[c] = ref.seg_metrics(yv[vok], st[c][vok])
            del marg
            gc.collect()
        cell_probs[fold_name] = {c: st[c] for c in FAIL_CELLS}
        cell_own[fold_name] = own
        print(f"  [stash {fold_name}] cell iter {cell_iter}, "
              f"n_mode {int(tr_mode.sum()):,}", flush=True)
        return st

    # ---------------- STEP 1: B3 folds 2023 + 2024 via exp_016 code ------- #
    print(f"[{time.strftime('%H:%M:%S')}] ===== B3 folds 2023, 2024 "
          f"(exp016_ref.run_b3) =====", flush=True)
    stash = {"fold_2023": build_stash("fold_2023", 2022, 2023),
             "fold_2024": build_stash("fold_2024", 2023, 2024)}

    captured = {}
    _orig_fit_curve = ref.fit_curve

    def fit_curve_capture(X_tr, y_tr, X_va, prm, n_rounds, ct, gd, tag):
        bst, marg = _orig_fit_curve(X_tr, y_tr, X_va, prm, n_rounds, ct, gd, tag)
        if tag.startswith("B3 stacker"):
            captured[tag.split()[-1]] = marg
        return bst, marg

    ref.fit_curve = fit_curve_capture
    try:
        b3 = ref.run_b3(df, base_feats, cat, season, f_excl, ok, succ_season,
                        y_all, SEED, grid, stash, EXPECTED_2024, 0.248261)
    finally:
        ref.fit_curve = _orig_fit_curve
    metrics["B3_exp016_path"] = {k: v for k, v in b3.items()}
    save()

    # ---------------- reproduction gate ----------------------------------- #
    f24 = b3["fold_2024"]
    got_p4 = f24["valid_all"]["brier"]
    got_or = f24["oracle_brier"]
    gate = {
        "brier_at_iter_650": got_p4,
        "expected_at_iter_650": GATE["brier_at_iter_650_P4"],
        "abs_diff_P4": round(abs(got_p4 - GATE["brier_at_iter_650_P4"]), 9),
        "iter_used": f24["iter_used"], "expected_iter": GATE["iter_P4"],
        "oracle_brier": got_or, "expected_oracle_brier": GATE["brier_at_oracle_iter"],
        "abs_diff_oracle": round(abs(got_or - GATE["brier_at_oracle_iter"]), 9),
        "oracle_iter": f24["oracle_iter"], "expected_oracle_iter": GATE["oracle_iter"],
        "fold_2023_brier": b3["fold_2023"]["valid_all"]["brier"],
        "fold_2023_expected": 0.248756,
        "tolerance": GATE["tolerance"],
    }
    gate["passed"] = bool(
        gate["abs_diff_P4"] <= GATE["tolerance"]
        and gate["abs_diff_oracle"] <= GATE["tolerance"]
        and gate["iter_used"] == GATE["iter_P4"])
    metrics["reproduction_gate"] = gate
    print("REPRODUCTION GATE: " + json.dumps(gate), flush=True)
    metrics["elapsed_min"] = round((time.time() - t_start) / 60, 2)
    save()
    if not SMOKE:
        assert gate["passed"], (
            "exp_016 B3 did NOT reproduce — everything downstream is "
            "uninterpretable, aborting.  " + json.dumps(gate))

    # ---------------- STEP 2: B3 block 2022 (feeder) ---------------------- #
    print(f"[{time.strftime('%H:%M:%S')}] ===== B3 block 2022 =====", flush=True)
    st22 = build_stash("fold_2022", 2021, 2022)
    b3_22, marg22 = b3_one_block(
        df, base_feats, cat, season, f_excl, ok, succ_season, y_all, SEED, grid,
        st22, "fold_2022", 2021, 2022,
        INHERITED["exp_016_B2c_iter_used_per_fold"]["fold_2022"])
    metrics["B3_block_2022"] = b3_22
    captured["fold_2022"] = marg22
    metrics["elapsed_min"] = round((time.time() - t_start) / 60, 2)
    save()

    # ---------------- STEP 3: A0 blocks (the DEV2 control) ---------------- #
    print(f"[{time.strftime('%H:%M:%S')}] ===== A0 blocks (exp_010 D) =====",
          flush=True)
    a0 = {}
    for V in BLOCKS:
        tr = (season <= V - 1) & ~f_excl
        va = season == V
        y_va = y_all[va]
        assert_no_recovered_in_features(base_feats, f"A0 block {V}")
        t0 = time.time()
        _, marg = _orig_fit_curve(df.loc[tr, base_feats], y_all[tr],
                                  df.loc[va, base_feats], params, ref.N_ROUNDS,
                                  cat, grid, f"A0 block {V}")
        cv = {k: float(np.mean((ref.sigmoid(marg[k]) - y_va) ** 2)) for k in grid}
        a_min, a_val = ref.argmin_of(cv)
        want = sorted({a_min, INHERITED["exp_019_A0_block_iter_P10"][V],
                       INHERITED["exp_019_A0_block_iter_P4"][V],
                       INHERITED["exp_019_A0_block_argmin"][V],
                       ref.nearest_grid_iter(grid, max(1, a_min // 2)),
                       ref.nearest_grid_iter(grid, min(ref.N_ROUNDS, a_min * 2))}
                      | set(SENS_ITERS))
        a0[V] = {"n_train": int(tr.sum()), "n_valid": int(va.sum()),
                 "own_argmin": a_min, "own_argmin_brier": round(a_val, 6),
                 "exp019_argmin": INHERITED["exp_019_A0_block_argmin"][V],
                 "argmin_matches_exp019": bool(
                     a_min == INHERITED["exp_019_A0_block_argmin"][V]),
                 "curve": {str(k): round(v, 6) for k, v in cv.items()},
                 "fit_min": round((time.time() - t0) / 60, 2),
                 "_p": {int(k): ref.sigmoid(marg[k]) for k in want}}
        print(f"    [A0 block {V}] argmin {a_min} = {a_val:.6f} "
              f"(exp_019 said {INHERITED['exp_019_A0_block_argmin'][V]})",
              flush=True)
        del marg
        gc.collect()

    # cross-check against exp_019's saved C0 prediction vector
    xchk = {"checked": EXP019_DIR}
    p19 = os.path.join(EXP019_DIR, "pred_C0_seed42_fold_2024.npy")
    if os.path.exists(p19):
        v19 = np.load(p19)
        mine = a0[2024]["_p"][INHERITED["exp_019_A0_block_iter_P10"][2024]]
        xchk["exp019_C0_pred_max_abs_diff"] = (
            float(np.max(np.abs(v19 - mine))) if v19.shape == mine.shape else None)
        xchk["shape_match"] = bool(v19.shape == mine.shape)
        p19c1 = os.path.join(EXP019_DIR, "pred_C1_seed42_fold_2024.npy")
        if os.path.exists(p19c1):
            xchk["exp019_C1_available"] = True
    else:
        xchk["status"] = "exp_019 prediction file not found"
    metrics["A0_blocks"] = {str(V): {k: v for k, v in a0[V].items()
                                     if not k.startswith("_")} for V in BLOCKS}
    metrics["A0_crosscheck_vs_exp019"] = xchk
    print("A0 cross-check: " + json.dumps(xchk), flush=True)
    metrics["elapsed_min"] = round((time.time() - t_start) / 60, 2)
    save()

    # ------------------------------------------------------------------ #
    #  slopes                                                            #
    # ------------------------------------------------------------------ #
    d_s = {V: dev["d_s"][season == V] for V in BLOCKS}
    d_r = {V: dev["d_r"][season == V] for V in BLOCKS}
    y_b = {V: y_all[season == V] for V in BLOCKS}
    hit_b = {V: hit[season == V] for V in BLOCKS}
    isr_b = {V: is_r_all[season == V] for V in BLOCKS}

    # B3 predictions per block at every iteration we might need
    b3_iter_primary = {2022: b3_22["iter_used"],
                       2023: INHERITED["exp_016_B3_stacker_iter_used"]["fold_2023"],
                       2024: INHERITED["exp_016_B3_stacker_iter_used"]["fold_2024"]}
    b3_argmin = {2022: b3_22["oracle_iter"],
                 2023: b3["fold_2023"]["oracle_iter"],
                 2024: b3["fold_2024"]["oracle_iter"]}
    IT_MAIN = INHERITED["exp_016_B3_stacker_iter_used"]["fold_2024"]
    b3_p = {V: {} for V in BLOCKS}
    for V in BLOCKS:
        mg = captured[f"fold_{V}"]
        want = ({b3_iter_primary[V], b3_argmin[V], IT_MAIN,
                 ref.nearest_grid_iter(grid, max(1, b3_argmin[V] // 2)),
                 ref.nearest_grid_iter(grid, min(ref.N_ROUNDS, b3_argmin[V] * 2)),
                 ref.nearest_grid_iter(grid, b3_argmin[2022])}
                | set(SENS_ITERS))
        for k in sorted(want):
            b3_p[V][k] = ref.sigmoid(mg[k])

    # runtime leak asserts on the slope fit
    slope_assert = {
        "blocks_used": SLOPE_BLOCKS,
        "valid_season": VALID_SEASON,
        "max_block_season_lt_valid": bool(max(SLOPE_BLOCKS) < VALID_SEASON),
        "max_lookup_source_season_in_slope_blocks": float(np.nanmax(
            src_fp[np.isin(season, SLOPE_BLOCKS)])),
    }
    assert slope_assert["max_block_season_lt_valid"], "SLOPE LEAK"
    assert slope_assert["max_lookup_source_season_in_slope_blocks"] < VALID_SEASON
    metrics["slope_leak_assert"] = slope_assert

    def pooled_slope(pred_by_block, dname="d_s", blocks=SLOPE_BLOCKS):
        dd = {"d_s": d_s, "d_r": d_r}[dname]
        r = np.concatenate([y_b[V] - pred_by_block[V] for V in blocks])
        d = np.concatenate([dd[V] for V in blocks])
        return off.fit_slope(r, d)

    # ---- B3 slopes -------------------------------------------------------- #
    b3_primary_pred = {V: b3_p[V][b3_iter_primary[V]] for V in SLOPE_BLOCKS}
    beta_b3 = pooled_slope(b3_primary_pred, "d_s")
    beta_b3_r = pooled_slope(b3_primary_pred, "d_r")
    beta_b3_per_block = {
        str(V): off.fit_slope(y_b[V] - b3_p[V][b3_iter_primary[V]], d_s[V])
        for V in SLOPE_BLOCKS}

    # ---- A0 slopes on the SAME blocks, three iteration conventions -------- #
    conv = {
        "own_argmin_PRIMARY_matches_B3": {V: a0[V]["own_argmin"] for V in BLOCKS},
        "P10_exp019": INHERITED["exp_019_A0_block_iter_P10"],
        "P4": INHERITED["exp_019_A0_block_iter_P4"],
    }
    beta_a0 = {}
    beta_a0_per_block = {}
    for tag, iters in conv.items():
        pred = {V: a0[V]["_p"][iters[V]] for V in BLOCKS}
        beta_a0[tag] = pooled_slope(pred, "d_s")
        beta_a0_per_block[tag] = {
            str(V): off.fit_slope(y_b[V] - pred[V], d_s[V]) for V in SLOPE_BLOCKS}

    ratio = (round(beta_b3["beta"] / beta_a0["own_argmin_PRIMARY_matches_B3"]["beta"], 4)
             if beta_a0["own_argmin_PRIMARY_matches_B3"]["beta"] else None)

    metrics["D1_beta"] = {
        "beta_B3_pooled_2022_2023": beta_b3,
        "beta_B3_per_block": beta_b3_per_block,
        "beta_B3_d_r_pooled": beta_b3_r,
        "beta_A0_pooled_2022_2023_by_iteration_convention": beta_a0,
        "beta_A0_per_block": beta_a0_per_block,
        "ratio_beta_B3_over_beta_A0_own_argmin": ratio,
        "block_iters_B3": {str(k): v for k, v in b3_iter_primary.items()},
        "block_iters_A0_primary": {str(k): v for k, v in
                                   conv["own_argmin_PRIMARY_matches_B3"].items()},
        "exp019_reference_pooled_2022_2023_P10": 0.525,
        "note": "beta_A0 under the P10 convention is the number exp_019 D5 "
                "reported (0.525); the PRIMARY comparison is own-argmin vs "
                "own-argmin because that is the convention the B3 blocks are on.",
    }
    print("--- D1 ---\n" + json.dumps(metrics["D1_beta"], indent=2, default=str),
          flush=True)
    save()

    # ------------------------------------------------------------------ #
    #  arms on fold_2024                                                 #
    # ------------------------------------------------------------------ #
    y24, d24, hit24 = y_b[2024], d_s[2024], hit_b[2024]
    isr24 = isr_b[2024]
    assert len(y24) == EXPECTED_2024["valid_all"]
    assert int(isr24.sum()) == EXPECTED_2024["valid_R"]

    p_b3_650 = b3_p[2024][IT_MAIN]
    p_b3_or = b3_p[2024][b3_argmin[2024]]
    beta_or = float(np.dot(d24, y24 - p_b3_650) / np.dot(d24, d24))

    arms_pred = {
        "E0_B3_P4iter": p_b3_650,
        "E1_B3_P4iter_plus_offset": off.apply_linear(p_b3_650, d24,
                                                     beta_b3["beta"]),
        "E2_B3_P4iter_plus_ORACLE_offset": off.apply_linear(p_b3_650, d24,
                                                            beta_or),
        "E0b_B3_oracle_iter": p_b3_or,
        "E1b_B3_oracle_iter_plus_offset": off.apply_linear(
            p_b3_or, d24, beta_b3["beta"]),
    }
    for tag, iters in conv.items():
        pa = a0[2024]["_p"][iters[2024]]
        arms_pred[f"A0_{tag}"] = pa
        arms_pred[f"A0_{tag}_plus_offset"] = off.apply_linear(
            pa, d24, beta_a0[tag]["beta"])

    base0 = float(np.mean((p_b3_650 - y24) ** 2))
    arms = {}
    for a, p in arms_pred.items():
        arms[a] = {"valid_all": ref.seg_metrics(y24, p),
                   "valid_R": ref.seg_metrics(y24[isr24], p[isr24]),
                   "valid_F": ref.seg_metrics(y24[~isr24], p[~isr24]),
                   "delta_vs_E0": round(float(np.mean((p - y24) ** 2) - base0), 9)}
        np.save(os.path.join(OUT_DIR, "pred", f"pred_{a}_seed42_fold_2024.npy"), p)
    metrics["arms_fold_2024"] = arms
    metrics["beta_applied"] = {
        "E1": beta_b3["beta"],
        "E2_ORACLE_display_only": round(beta_or, 6),
        "A0_by_convention": {t: beta_a0[t]["beta"] for t in conv},
    }
    print("--- arms fold_2024 ---\n" + json.dumps(
        {a: {"brier": arms[a]["valid_all"]["brier"],
             "d_vs_E0": arms[a]["delta_vs_E0"]} for a in arms}, indent=2),
        flush=True)
    save()

    # ------------------------------------------------------------------ #
    #  D2 absorption mechanism                                           #
    # ------------------------------------------------------------------ #
    stack24 = np.column_stack([cell_probs["fold_2024"][c] for c in FAIL_CELLS])
    r_b3 = y24 - p_b3_650
    r_a0 = y24 - a0[2024]["_p"][conv["own_argmin_PRIMARY_matches_B3"][2024]]
    d2 = {"stack_columns": [f"p_{c}" for c in FAIL_CELLS],
          "n_rows": int(len(y24)),
          "corr_d_s_with_stack_col": {},
          "corr_d_s_with_stack_col_hit_rows_only": {},
          "R2_of_d_s_on_all_four_stack_cols": r2_of(d24, stack24),
          "R2_of_d_s_on_all_four_stack_cols_hit_rows":
              r2_of(d24[hit24], stack24[hit24]),
          "corr_d_s_with_p_B3": round(float(np.corrcoef(d24, p_b3_650)[0, 1]), 4),
          "corr_d_s_with_p_A0": round(float(np.corrcoef(
              d24, a0[2024]["_p"][conv["own_argmin_PRIMARY_matches_B3"][2024]]
          )[0, 1]), 4),
          "partial_corr_d_s_resid_B3_given_stack":
              partial_corr(d24, r_b3, stack24),
          "raw_corr_d_s_resid_B3": round(float(np.corrcoef(d24, r_b3)[0, 1]), 5),
          "raw_corr_d_s_resid_A0": round(float(np.corrcoef(d24, r_a0)[0, 1]), 5),
          }
    for j, c in enumerate(FAIL_CELLS):
        d2["corr_d_s_with_stack_col"][f"p_{c}"] = round(
            float(np.corrcoef(d24, stack24[:, j])[0, 1]), 4)
        d2["corr_d_s_with_stack_col_hit_rows_only"][f"p_{c}"] = round(
            float(np.corrcoef(d24[hit24], stack24[hit24, j])[0, 1]), 4)
    d2["corr_d_s_d_r_hit_rows"] = round(
        float(np.corrcoef(d24[hit24], d_r[2024][hit24])[0, 1]), 4)
    d2["cell_submodel_own_mode_metrics"] = cell_own
    diagnostics["D2_absorption"] = d2

    # ------------------------------------------------------------------ #
    #  D3 additivity                                                     #
    # ------------------------------------------------------------------ #
    def br(p):
        return float(np.mean((p - y24) ** 2))

    d3 = {}
    for tag in conv:
        pa = arms_pred[f"A0_{tag}"]
        pao = arms_pred[f"A0_{tag}_plus_offset"]
        dl_b3 = br(p_b3_650) - br(pa)
        dl_off_a0 = br(pao) - br(pa)
        dl_off_b3 = br(arms_pred["E1_B3_P4iter_plus_offset"]) - br(p_b3_650)
        dl_comb = br(arms_pred["E1_B3_P4iter_plus_offset"]) - br(pa)
        pred_add = dl_b3 + dl_off_a0
        d3[tag] = {
            "A0_reference_iter": conv[tag][2024],
            "A0_brier": round(br(pa), 6),
            "B3_brier_P4iter": round(br(p_b3_650), 6),
            "delta_B3_alone": round(dl_b3, 9),
            "delta_offset_on_A0": round(dl_off_a0, 9),
            "delta_offset_on_B3": round(dl_off_b3, 9),
            "delta_combined_vs_A0": round(dl_comb, 9),
            "additive_prediction": round(pred_add, 9),
            "observed_minus_additive": round(dl_comb - pred_add, 9),
            "additivity_ratio_observed_over_predicted": (
                round(dl_comb / pred_add, 4) if pred_add else None),
            "offset_retention_on_B3_vs_on_A0": (
                round(dl_off_b3 / dl_off_a0, 4) if dl_off_a0 else None),
            "verdict_scale": "ratio ~1 additive; <1 partial overlap; >1 "
                             "super-additive; delta_offset_on_B3 ~0 = absorbed",
        }
    d3["exp019_C1_reference_seed42_P10"] = {
        "C0_brier": 0.248257, "C1_brier": 0.248147, "delta": -0.000110248,
        "note": "exp_019 seed-42 numbers, quoted for context; the internally "
                "computed A0_P10_exp019 row above is the like-for-like control",
    }
    diagnostics["D3_combination"] = d3

    # ------------------------------------------------------------------ #
    #  D4 coverage                                                       #
    # ------------------------------------------------------------------ #
    d4 = {}
    se0 = (p_b3_650 - y24) ** 2
    n = len(y24)
    for a in ["E1_B3_P4iter_plus_offset", "E2_B3_P4iter_plus_ORACLE_offset"]:
        sea = (arms_pred[a] - y24) ** 2
        d4[a] = {
            "n_hit": int(hit24.sum()), "n_miss": int((~hit24).sum()),
            "hit_pct": round(100.0 * float(hit24.mean()), 3),
            "brier_delta_overall": round(float(sea.mean() - se0.mean()), 9),
            "brier_delta_within_hit_rows": round(
                float(sea[hit24].mean() - se0[hit24].mean()), 9),
            "contribution_of_hit_rows_to_overall": round(
                float((sea[hit24] - se0[hit24]).sum() / n), 9),
            "contribution_of_miss_rows_to_overall": round(
                float((sea[~hit24] - se0[~hit24]).sum() / n), 9),
            "extrapolated_if_coverage_100pct": round(
                float((sea[hit24] - se0[hit24]).mean()), 9),
        }
    d4["miss_rows_are_d_equals_zero_not_sentinel"] = bool(
        np.all(d24[~hit24] == 0.0))
    diagnostics["D4_coverage"] = d4

    # ------------------------------------------------------------------ #
    #  D5 sensitivity (iteration + block convention)                     #
    # ------------------------------------------------------------------ #
    d5 = {"B3_beta_by_block_iteration": {}, "A0_beta_by_block_iteration": {}}
    common_b3 = sorted(set(b3_p[2022]) & set(b3_p[2023]))
    for k in common_b3:
        d5["B3_beta_by_block_iteration"][str(k)] = off.fit_slope(
            np.concatenate([y_b[V] - b3_p[V][k] for V in SLOPE_BLOCKS]),
            np.concatenate([d_s[V] for V in SLOPE_BLOCKS]))["beta"]
    common_a0 = sorted(set(a0[2022]["_p"]) & set(a0[2023]["_p"]))
    for k in common_a0:
        d5["A0_beta_by_block_iteration"][str(k)] = off.fit_slope(
            np.concatenate([y_b[V] - a0[V]["_p"][k] for V in SLOPE_BLOCKS]),
            np.concatenate([d_s[V] for V in SLOPE_BLOCKS]))["beta"]
    # B3 under a P4 chain instead of exp_016's per-fold argmin
    k23_chain = ref.nearest_grid_iter(grid, b3_argmin[2022])
    pred_chain = {2022: b3_p[2022][b3_iter_primary[2022]], 2023: b3_p[2023][k23_chain]}
    d5["B3_beta_P4_chained_block2023"] = {
        "iter_block_2023": int(k23_chain),
        "slope": pooled_slope(pred_chain, "d_s")}
    d5["applied_beta_alternatives_on_fold_2024_POSTHOC"] = {
        "warning": "post-hoc display; nothing here selected the reported beta",
        "E0_brier": round(base0, 9),
        "deltas": {
            t: round(br(off.apply_linear(p_b3_650, d24, bv)) - base0, 9)
            for t, bv in {
                "prereg_pooled_2022_2023": beta_b3["beta"],
                "P4_chained": d5["B3_beta_P4_chained_block2023"]["slope"]["beta"],
                "block_2022_only": beta_b3_per_block["2022"]["beta"],
                "block_2023_only": beta_b3_per_block["2023"]["beta"],
                "beta_A0_own_argmin": beta_a0[
                    "own_argmin_PRIMARY_matches_B3"]["beta"],
                "oracle_on_valid_2024": round(beta_or, 6),
            }.items()},
    }
    diagnostics["D5_sensitivity"] = d5

    diagnostics["DEV1_no_verdict"] = {
        "seeds_run": [SEED],
        "prereg_adoption_rule": "3 seeds same sign AND |mean delta| >= 3 x "
                                "control seed std",
        "evaluable": False,
        "statement": "This run estimates the MAGNITUDE of beta_B3. It does not, "
                     "and cannot, adopt or reject anything.",
    }

    metrics["train_time_min"] = round((time.time() - t_start) / 60, 2)
    save()
    print(f"[{time.strftime('%H:%M:%S')}] DONE {metrics['train_time_min']} min",
          flush=True)


if __name__ == "__main__":
    main()
