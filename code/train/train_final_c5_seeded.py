"""Deployment training for candidates L and M — CatBoost C5 on the B3
mode-stacking architecture, fitted on ALL of season <= 2024.

ONE RUN PRODUCES ONE SET OF FIVE BOOSTERS.  Candidates L and M share them:
L ships them raw, M ships the same files plus a stage-3 offset.  The offset is a
post-hoc transform of the stacker's probability, so it cannot change the fit.
Training twice would only introduce a chance for the two candidates to diverge.

WHAT THIS IS A RE-RUN OF
------------------------
`experiments/exp_030/src/b3_cat.py`, unit 2024 (train <= 2023 -> valid 2024),
with the validation year folded back into the training pool.  Every mask, pool
and ordering below is that file's; the differences are exactly three and they
are the differences DEPLOYMENT forces:

  1. `tr_full` is season <= 2024 instead of season <= 2023, so there is no held
     out season and NO VALIDATION CURVE IS COMPUTED HERE AT ALL.  Nothing in
     this file can be selected on; every number that governs was fixed before it
     ran.
  2. the forward-chaining OOF loop therefore runs one more block (2020..2024
     instead of 2020..2023).
  3. iteration counts are the stage-1 argmins scaled by the DEPLOYMENT row
     ratio rather than the fold_2024 one — the same rule, a larger pool.

⚠️ THE ITERATION INPUT IS KNOWN TO BE CENSORED.  exp_030's stage-1 stacker
argmin (375 on fold_2023 under cap 1650) is interior and sound, but the
fold_2024 unit that MEASURED this architecture's 724.98404 ran its stacker at a
FIXED 481 rounds, and at seeds 7 and 2024 that curve was still descending there
(exp_030 metrics, `fold_2024_boundary_diagnostic`).  So the recorded score is a
LOWER BOUND, and a stacker iteration derived from that regime may leave the
deployed model under-trained by an amount NOBODY HAS MEASURED.  exp_034 measured
the cost in the other direction — cutting the stacker 481 -> 240 cost 26.85795
points — which says the curve is steep on the low side.
`--stacker-base-argmin` exists so that a re-measured base argmin can replace the
censored one WITHOUT changing the scaling rule, and whatever is used is written
into the report as `stacker_iter_source`.

ITERATION POLICY (executed here, not transcribed)
-------------------------------------------------
    cell k   = round(k_base_cell  * n_mode_deploy / n_mode_base)
    stacker  = round(k_base_stack * n_full_deploy / n_full_base)
The four base cell argmins, the base stacker argmin and both base row counts are
READ out of exp_030/stage1.json.  Nothing is typed in.

LEAK BOUNDARY
-------------
  * recovered `is_*` columns are TARGETS ONLY; `assert_no_recovered_in_features`
    fires before every single design matrix is built.
  * the stacker's four extra columns are out-of-fold and forward-chaining by
    season: rows of season s are predicted by cell models fitted on seasons < s
    only.  No random KFold, no train-segment self-prediction.  The earliest
    season keeps NaN — CatBoost handles NaN in a float feature natively
    (nan_mode Min); a sentinel fill is banned.
  * `has_time=True` uses the pool's row order as its time axis, so every fit
    pool is re-checked at run time for (season, game_month) inversions.
  * nothing here reads the test file.

Usage:
    python train_final_c5.py --out ~/SungjinPark/submissions/L_c5b3
    python train_final_c5.py --out ... --stacker-base-argmin 900 \
        --stacker-base-note "exp_0NN re-measured argmin at cap 2000"
"""

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

from labels import (FAIL_CELLS, assert_no_recovered_in_features,     # noqa: E402
                    recover)

DATA_PATH = os.path.expanduser("~/SungjinPark/data/train.csv")
EXP027_TRAIN = os.path.expanduser(
    "~/SungjinPark/experiments/exp_027/src/train.py")
STAGE1_JSON = os.path.expanduser(
    "~/SungjinPark/experiments/exp_030/stage1.json")
EXP030_METRICS = os.path.expanduser(
    "~/SungjinPark/experiments/exp_030/metrics.json")

EXP027_SHA = "e4b308a5f6e4564ed43aed6715d05c0aa617b5e9d22e1bfd51e71dd127416f0f"
LABELS_SHA = "439cd504478b4bc8162e9dedc32be27bfadd5d1178472d6b384b60f46ed7a9f2"
TRAIN_CSV_SHA = "d2081186b458b49f60b082be480c273135833e15ba59a76d033af28bcf8763ff"

TARGET = "control_success"
DROP_COLS = ["row_id", TARGET]
CAT_COLS = ["top_bottom", "game_type", "base_state"]
ID_COLS = ["pitcher_id", "batter_id"]        # C5: NUMERIC, never cat_features
F_EXCLUDE_MAX_SEASON = 2022                  # exp_004 mask
MAX_TRAIN_SEASON = 2024
N_THREADS = 6
DEFAULT_SEED = 42          # candidate Q: --seed overrides; 42 reproduces

# exp_016 / exp_024 / candidate D all fitted on these two counts.  They are
# asserted, not discovered: if a mask silently changed, the run stops.
EXPECTED_N_FULL = 1369784
EXPECTED_N_MODE = 1369186
BASE_UNIT = 2023


def sha256_file(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_exp027():
    """Import exp_027/src/train.py by path, but only after its sha matches."""
    got = sha256_file(EXP027_TRAIN)
    assert got == EXP027_SHA, ("exp_027 train.py changed", got, EXP027_SHA)
    spec = importlib.util.spec_from_file_location("exp027_train", EXP027_TRAIN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                     # main() is __main__-guarded
    c5 = mod.ARMS["C5"]
    assert c5["family"] == "catboost", c5
    assert c5["params"] == {}, c5                    # C5 == CB_BASE exactly
    assert c5["cat_ids"] is False, c5                # IDs stay numeric
    assert mod.CB_BASE["has_time"] is True
    assert mod.CB_BASE["thread_count"] == N_THREADS
    return mod, c5


def assert_time_ordered(tkey, mask, where):
    t = tkey[mask]
    inv = int((np.diff(t) < 0).sum())
    assert inv == 0, (f"has_time=True but pool '{where}' is not in time order",
                      inv)
    return inv


def fit_and_save(e27, C5, F_tr, y_tr, feats, cat_native, n_round, out_path, tag,
                 seed):
    """Fit one CatBoost model to a FIXED round count and save it as .cbm.

    exp_027's `fit_catboost` returns a prediction closure, not the model, so the
    fit is repeated here with the SAME parameter dict it builds — the dict is
    read out of the module (`CB_BASE`), never retyped — and the model object is
    kept.  `assert model.tree_count_ == n_round` is exp_027's own check and is
    what makes "the shipped booster has exactly the rounds the policy chose" a
    verified statement rather than an intention.
    """
    import catboost as cb
    params = dict(e27.CB_BASE, random_seed=seed, iterations=n_round,
                  **C5["params"])
    cat_idx = [feats.index(c) for c in cat_native]
    pool = cb.Pool(F_tr, label=y_tr, cat_features=cat_idx)
    t0 = time.time()
    model = cb.CatBoostClassifier(**params)
    model.fit(pool)
    mins = (time.time() - t0) / 60.0
    assert model.tree_count_ == n_round, (model.tree_count_, n_round)
    assert list(model.feature_names_) == list(feats), (
        "saved feature names differ from the fit order",
        list(model.feature_names_)[:5], list(feats)[:5])
    model.save_model(out_path, format="cbm")
    print(f"  [{tag}] {n_round} rounds on {len(F_tr):,} rows -> "
          f"{os.path.basename(out_path)} ({mins:.2f} min)", flush=True)
    return model, params, mins


def predict_prob(e27, model, F, cat_native, feats):
    """sigmoid(RawFormulaVal) — the arithmetic script.py ships and exp_030
    measured.  Kept identical on both sides so the OOF stack columns the stacker
    trains on and the ones script.py feeds it are the same function."""
    import catboost as cb
    cat_idx = [feats.index(c) for c in cat_native]
    raw = model.predict(cb.Pool(F, cat_features=cat_idx),
                        prediction_type="RawFormulaVal",
                        thread_count=N_THREADS).astype(np.float64)
    return 1.0 / (1.0 + np.exp(-raw))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True,
                    help="candidate directory; models go to <out>/model/")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help="CatBoost random_seed for every fit in this run "
                         "(cells, OOF cells, stacker) and the numpy seed. "
                         "THE ONLY THING candidate Q varies.")
    ap.add_argument("--stage1-json", default=STAGE1_JSON)
    ap.add_argument("--exp030-metrics", default=EXP030_METRICS)
    ap.add_argument("--stacker-base-argmin", type=int, default=0,
                    help="replace the stage-1 stacker argmin (see the module "
                         "docstring on censoring).  Requires --stacker-base-note.")
    ap.add_argument("--stacker-iter", type=int, default=0,
                    help="deploy the stacker at EXACTLY this many rounds.  No "
                         "scaling law is applied.  Requires --stacker-base-note "
                         "carrying the reason, because a number that no rule in "
                         "this file produced is the single easiest thing to lose "
                         "the provenance of.")
    ap.add_argument("--cell-policy", default="stage1_scaled",
                    choices=["stage1_scaled", "fold2024_scaled"],
                    help="stage1_scaled: stage-1 (unit 2023) cell argmins x "
                         "n_mode_deploy/n_mode_2023.  fold2024_scaled: exp_030's "
                         "fold_2024 cell iterations x n_mode_deploy/n_mode_2024. "
                         "The two differ by at most one tree per cell (double "
                         "rounding); both are computed and both are recorded.")
    ap.add_argument("--stacker-base-note", default="")
    ap.add_argument("--smoke", type=int, default=0,
                    help="fit every model to this many rounds instead of the "
                         "policy's, for a plumbing test.  NEVER ship a smoke "
                         "model: the report records smoke=True.")
    args = ap.parse_args()

    if (args.stacker_base_argmin or args.stacker_iter) \
            and not args.stacker_base_note:
        raise SystemExit("--stacker-base-argmin / --stacker-iter requires "
                         "--stacker-base-note — an unexplained iteration "
                         "override is exactly the kind of unlisted variable "
                         "this project bans")
    if args.stacker_base_argmin and args.stacker_iter:
        raise SystemExit("--stacker-base-argmin and --stacker-iter both given; "
                         "pick one, they mean different things (scale a base "
                         "vs pin the deployed count)")

    out_dir = os.path.abspath(os.path.expanduser(args.out))
    model_dir = os.path.join(out_dir, "model")
    os.makedirs(model_dir, exist_ok=True)
    t_start = time.time()
    SEED = int(args.seed)
    np.random.seed(SEED)

    csv_sha = sha256_file(DATA_PATH)
    assert csv_sha == TRAIN_CSV_SHA, ("train.csv SHA MISMATCH", csv_sha)
    assert sha256_file(os.path.join(SRC_DIR, "labels.py")) == LABELS_SHA
    e27, C5 = load_exp027()
    build_frame_native = e27.build_frame_native
    print("sha gates OK (train.csv, labels.py, exp_027/train.py)", flush=True)

    with open(args.stage1_json) as f:
        base = json.load(f)
    assert base["stage"] == 1 and base["unit"] == BASE_UNIT, base["unit"]
    assert not base["any_argmin_at_cap"], "stage 1 was invalid (cap hit)"
    base_cell = dict(base["cell_argmin_iter"])
    base_stack = int(base["stack_argmin_iter"])
    stacker_iter_source = (
        f"exp_030 stage-1 (unit {BASE_UNIT} seed 42) stacker argmin "
        f"{base_stack} x the deployment training-row ratio")
    if args.stacker_base_argmin:
        stacker_iter_source = (
            f"OVERRIDE: base stacker argmin {base_stack} -> "
            f"{args.stacker_base_argmin} x the deployment training-row ratio. "
            f"reason: {args.stacker_base_note}")
        base_stack = int(args.stacker_base_argmin)

    # exp_030's fold_2024 iterations and row counts, READ from its metrics.
    # The cell rule has two defensible routes and they disagree by at most one
    # tree because of double rounding; both are computed so the ledger can say
    # which was shipped instead of leaving a reader to re-derive it and land on
    # the other one.
    with open(args.exp030_metrics) as f:
        m30 = json.load(f)
    assert m30["exp_id"] == "exp_030"
    pol30 = m30["stage_2_iter_policy"]
    f24_cell = {c: int(v) for c, v in pol30["resulting_cell_iter"]["2024"].items()}
    f24_stack = int(pol30["resulting_stacker_iter"]["2024"])
    n_mode_f24 = int(round(base["n_train_mode"]
                           * pol30["cell_scale_is_n_mode_ratio"]["2024"]))
    n_full_f24 = int(round(base["n_train_full"]
                           * pol30["stacker_scale_is_n_train_ratio"]["2024"]))
    assert set(f24_cell) == set(FAIL_CELLS), f24_cell

    # ------------------------------------------------------------------ data
    print(f"[{time.strftime('%H:%M:%S')}] loading {DATA_PATH}", flush=True)
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    assert df["row_id"].is_unique
    assert df["season"].is_monotonic_increasing
    base_feats = [c for c in df.columns if c not in DROP_COLS]
    assert len(base_feats) == 47, len(base_feats)
    assert_no_recovered_in_features(base_feats, "base_feats (before recovery)")

    rec = recover(df, verbose=False)
    assert rec["success_recovery"]["agree_pct"] == 100.0

    season = df["season"].to_numpy()
    ok = df["recov_ok"].to_numpy()
    succ_season = df.groupby("pitcher_id", sort=False)["season"].shift(-1).to_numpy()
    f_excl = ((df["season"] <= F_EXCLUDE_MAX_SEASON)
              & (df["game_type"] == "F")).to_numpy()
    y_all = df[TARGET].to_numpy(dtype=np.float64)
    assert np.isin(y_all, (0.0, 1.0)).all()
    tkey = (season.astype(np.int64) * 100
            + df["game_month"].to_numpy(np.int64))

    tr_full = (season <= MAX_TRAIN_SEASON) & ~f_excl
    tr_mode = tr_full & ok & (succ_season <= MAX_TRAIN_SEASON)
    # for the deployment pool the succ_season clause degenerates: a row whose
    # pitcher has no NEXT row cannot have its cell label recovered at all, so
    # `ok` is already False there.  Asserted rather than assumed.
    assert np.array_equal(tr_mode, tr_full & ok), (
        "succ_season <= 2024 removed rows that `ok` had kept — the label "
        "recovery boundary is not where it was believed to be")
    n_full = int(tr_full.sum())
    n_mode = int(tr_mode.sum())
    assert n_full == EXPECTED_N_FULL, (n_full, EXPECTED_N_FULL)
    assert n_mode == EXPECTED_N_MODE, (n_mode, EXPECTED_N_MODE)

    cat_native = [c for c in base_feats if c in CAT_COLS]
    assert len(cat_native) == 3, cat_native
    assert not any(c in cat_native for c in ID_COLS), "C5 keeps IDs numeric"

    s_mode = n_mode / base["n_train_mode"]
    s_full = n_full / base["n_train_full"]
    s_mode_f24 = n_mode / n_mode_f24
    s_full_f24 = n_full / n_full_f24
    cell_via_stage1 = {c: int(round(base_cell[c] * s_mode)) for c in FAIL_CELLS}
    cell_via_fold24 = {c: int(round(f24_cell[c] * s_mode_f24))
                       for c in FAIL_CELLS}
    cell_round = (cell_via_fold24 if args.cell_policy == "fold2024_scaled"
                  else cell_via_stage1)
    cell_iter_source = (
        f"exp_030 fold_2024 cell iterations {f24_cell} x n_mode ratio "
        f"{s_mode_f24:.5f} = {n_mode}/{n_mode_f24}"
        if args.cell_policy == "fold2024_scaled" else
        f"exp_030 stage-1 (unit {BASE_UNIT}) cell argmins {base_cell} x n_mode "
        f"ratio {s_mode:.5f} = {n_mode}/{base['n_train_mode']}")

    if args.stacker_iter:
        stack_round = int(args.stacker_iter)
        stacker_iter_source = (
            f"PINNED to exactly {stack_round}.  This is NOT the output of the "
            f"row-ratio law (that would give "
            f"{int(round(base_stack * s_full))} from stage-1 base {base_stack}, "
            f"or {int(round(f24_stack * s_full_f24))} from the fold_2024 "
            f"count {f24_stack}).  reason: {args.stacker_base_note}")
    else:
        stack_round = int(round(base_stack * s_full))
    if args.smoke:
        cell_round = {c: args.smoke for c in FAIL_CELLS}
        stack_round = args.smoke

    report = {
        "candidate_family": "Q (3-seed average) — one seed's boosters",
        "arm": "C5_B3", "seed": SEED, "smoke": bool(args.smoke),
        "train_upto": MAX_TRAIN_SEASON,
        "n_train_full": n_full, "n_train_mode": n_mode,
        "n_masked_rows_exp004": int(f_excl.sum()),
        "n_rows_dropped_unrecoverable_label": n_full - n_mode,
        "cat_features": list(cat_native), "id_cols_numeric": list(ID_COLS),
        "iteration_policy": {
            "rule": "stage-1 fold_2023 argmin x the ratio of TRAINING ROWS in "
                    "the pool that fits the model; NO validation curve exists "
                    "in this run and none was consulted",
            "base_unit": BASE_UNIT, "base_seed": 42,
            "base_n_train_mode": base["n_train_mode"],
            "base_n_train_full": base["n_train_full"],
            "base_cell_argmin": base_cell,
            "base_stack_argmin_used": base_stack,
            "base_stack_argmin_in_stage1_json": int(base["stack_argmin_iter"]),
            "scale_mode": s_mode, "scale_full": s_full,
            "cell_round": dict(cell_round), "stack_round": int(stack_round),
            "stacker_iter_source": stacker_iter_source,
            "cell_iter_source": cell_iter_source,
            "cell_policy": args.cell_policy,
            "THE_TWO_RULES_ARE_DIFFERENT": (
                "the stacker count was CHOSEN on a measured curve (exp_036 "
                "measured the uncensored fold_2024 argmin on three seeds and a "
                "law was picked from it); the cell counts were NOT — no cell "
                "curve was ever measured at or above the deployment size, so "
                "they ride the row-ratio law on faith.  Listing them under one "
                "'iteration policy' heading would hide that."),
            "KNOWN_WEAKNESS_cell_law_unverified_upward": (
                "exp_034 already saw the row-linear law break for the cells in "
                "the DOWNWARD direction (`is_rev_only` 788 predicted vs 230/240 "
                "measured).  The upward direction is untested.  If the law is "
                "similarly wrong going up, the cells are OVER-trained here and "
                "nothing in this build would detect it."),
            "exp030_fold2024_cell_iter": f24_cell,
            "exp030_fold2024_stack_iter": f24_stack,
            "n_mode_fold2024": n_mode_f24, "n_full_fold2024": n_full_f24,
            "scale_mode_from_fold2024": s_mode_f24,
            "scale_full_from_fold2024": s_full_f24,
            "cell_round_via_stage1_route": cell_via_stage1,
            "cell_round_via_fold2024_route": cell_via_fold24,
            "cell_routes_disagree_on": {
                c: [cell_via_stage1[c], cell_via_fold24[c]]
                for c in FAIL_CELLS if cell_via_stage1[c] != cell_via_fold24[c]},
            "stack_round_via_stage1_route": int(round(base_stack * s_full)),
            "stack_round_via_fold2024_route": int(round(f24_stack * s_full_f24)),
            "KNOWN_DEFECT_censored_fold2024_stacker": (
                "exp_030's fold_2024 unit ran its stacker at a FIXED 481 rounds "
                "and at seeds 7 and 2024 the curve was still descending there, "
                "so 724.98404 is a LOWER BOUND and the true argmin is known "
                "only as '>= 481'.  The deployed count inherits that regime."),
        },
        "sha256": {
            "train.csv": csv_sha,
            "train_final_c5.py": sha256_file(os.path.abspath(__file__)),
            "labels.py": LABELS_SHA, "exp_027_train.py": EXP027_SHA,
        },
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__},
        "time_inversions": {},
    }
    import catboost as cb                                            # noqa: F401
    report["versions"]["catboost"] = cb.__version__
    report["time_inversions"]["tr_full"] = assert_time_ordered(tkey, tr_full,
                                                               "tr_full")
    report["time_inversions"]["tr_mode"] = assert_time_ordered(tkey, tr_mode,
                                                               "tr_mode")
    print(f"masks: f_excl {int(f_excl.sum()):,} · stacker fit {n_full:,} · "
          f"cell fit {n_mode:,} · cell rounds {cell_round} · "
          f"stacker rounds {stack_round}", flush=True)

    # the training vocabulary of the three categorical columns.  CatBoost does
    # not need it (it handles unseen categories itself); script.py uses it ONLY
    # to detect a column that matches nothing, which is the signature of a dtype
    # mismatch.  Frozen on TRAINING rows.
    vocab = {c: sorted(pd.unique(df.loc[tr_full, c].astype(str)).tolist())
             for c in cat_native}
    report["vocab_source"] = ("training rows only (season<=2024 minus the "
                              "exp_004 mask); used by script.py for a "
                              "wipe-out guard, never to map a value")

    # ------------------------------------------------------- cell models
    print(f"[{time.strftime('%H:%M:%S')}] cell models on {n_mode:,} rows",
          flush=True)
    assert_no_recovered_in_features(base_feats, "cell models")
    F_mode = build_frame_native(df.loc[tr_mode], base_feats, cat_native)
    cell_models, cell_info = {}, {}
    for c in FAIL_CELLS:
        yc = df.loc[tr_mode, c].to_numpy(dtype=np.float64)
        assert not np.isnan(yc).any()
        path = os.path.join(model_dir, f"cell_{c}.cbm")
        m, params, mins = fit_and_save(e27, C5, F_mode, yc, base_feats,
                                       cat_native, int(cell_round[c]), path, c,
                                       SEED)
        cell_models[c] = m
        cell_info[c] = {"file": f"cell_{c}.cbm", "n_round": int(cell_round[c]),
                        "n_fit_rows": n_mode,
                        "pos_rate_train": float(np.mean(yc)),
                        "fit_min": round(mins, 3),
                        "sha256": sha256_file(path)}
    report["cells"] = cell_info
    report["catboost_effective_params"] = {
        k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
        for k, v in params.items()}
    del F_mode
    gc.collect()

    # --------------------------------------- forward-chaining OOF by season
    stack_cols = [f"p_{c}" for c in FAIL_CELLS]
    seasons = sorted(set(season[tr_full].tolist()))
    print(f"[{time.strftime('%H:%M:%S')}] OOF stack columns, blocks "
          f"{seasons[1:]}", flush=True)
    stack = {c: np.full(len(df), np.nan) for c in stack_cols}
    t_oof = time.time()
    oof_log = []
    for s in seasons[1:]:
        fit_rows = (season < s) & ~f_excl & ok & (succ_season < s)
        pred_rows = (season == s) & tr_full
        assert_time_ordered(tkey, fit_rows, f"OOF fit {s}")
        assert_no_recovered_in_features(base_feats, f"B3 OOF {s}")
        Ff = build_frame_native(df.loc[fit_rows], base_feats, cat_native)
        Fp = build_frame_native(df.loc[pred_rows], base_feats, cat_native)
        for c in FAIL_CELLS:
            yc = df.loc[fit_rows, c].to_numpy(dtype=np.float64)
            assert not np.isnan(yc).any()
            import catboost as cbm
            params_o = dict(e27.CB_BASE, random_seed=SEED,
                            iterations=int(cell_round[c]), **C5["params"])
            cat_idx = [base_feats.index(x) for x in cat_native]
            mo = cbm.CatBoostClassifier(**params_o)
            mo.fit(cbm.Pool(Ff, label=yc, cat_features=cat_idx))
            assert mo.tree_count_ == int(cell_round[c])
            stack[f"p_{c}"][pred_rows] = predict_prob(e27, mo, Fp, cat_native,
                                                      base_feats)
            del mo
            gc.collect()
        oof_log.append({"season": int(s), "n_fit": int(fit_rows.sum()),
                        "n_pred": int(pred_rows.sum()),
                        "elapsed_s": round(time.time() - t_oof, 1)})
        print(f"  OOF season {s}: fit {int(fit_rows.sum()):,} -> pred "
              f"{int(pred_rows.sum()):,} ({time.time()-t_oof:.0f}s)", flush=True)
        del Ff, Fp
        gc.collect()
    first = seasons[0]
    for c in stack_cols:
        assert np.isnan(stack[c][(season == first) & tr_full]).all(), (
            "the earliest season must keep NaN — a sentinel fill is banned")
        assert not np.isnan(stack[c][(season > first) & tr_full]).any(), (
            "a later-season training row is missing its OOF stack columns")
    report["oof_blocks"] = oof_log
    report["oof_min"] = round((time.time() - t_oof) / 60, 2)
    report["nan_stack_rows_in_train_pct"] = round(
        100.0 * float(np.isnan(stack[stack_cols[0]][tr_full]).mean()), 3)

    # ------------------------------------------------------------- stacker
    for c in stack_cols:
        df[c] = stack[c]
    feats = base_feats + stack_cols
    assert_no_recovered_in_features(feats, "B3 stacker")
    assert len(feats) == 51, len(feats)
    Fs = build_frame_native(df.loc[tr_full], feats, cat_native)
    path = os.path.join(model_dir, "stacker.cbm")
    stacker, params_s, mins_s = fit_and_save(
        e27, C5, Fs, y_all[tr_full], feats, cat_native, int(stack_round), path,
        "stacker", SEED)
    report["stacker"] = {"file": "stacker.cbm", "n_round": int(stack_round),
                         "n_fit_rows": n_full, "fit_min": round(mins_s, 3),
                         "sha256": sha256_file(path)}
    del Fs
    gc.collect()

    report["base_features"] = base_feats
    report["stack_features"] = stack_cols
    report["features"] = feats
    report["cell_model_order"] = list(FAIL_CELLS)
    report["vocab"] = vocab
    report["minutes"] = round((time.time() - t_start) / 60, 2)

    with open(os.path.join(out_dir, "build", "train_report.json"), "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print("RESULT: " + json.dumps({
        "n_full": n_full, "n_mode": n_mode, "cell_round": cell_round,
        "stack_round": int(stack_round), "smoke": bool(args.smoke),
        "minutes": report["minutes"]}), flush=True)


if __name__ == "__main__":
    main()
