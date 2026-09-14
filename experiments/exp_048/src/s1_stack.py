"""exp_048 STAGE 1 — the R-branch stacker at 1400 rounds, 51 vs 67 columns.

WHAT IS INHERITED AND WHAT IS NEW
---------------------------------
INHERITED, byte-checked before use:
  * `~/SungjinPark/build_AH/localref/oof/stack_fold2024_seed{S}.npz` — the four
    forward-chained OOF cell columns, the `tr_full` / `va` masks and `season`,
    written yesterday by candidate AH's local reference run.  Their sha256 must
    equal the digests AH recorded in `local_reference_AH.json`.  No cell model
    is refitted here.
  * exp_027's `build_frame_native`, `CB_BASE`, `ARMS["C5"]`, and exp_016's
    `sigmoid` / `seg_metrics`, all sha-pinned.

NEW: the 16 `ins_*` columns (arm A1 only) and nothing else.

THE CONTROL IS A GATE, NOT A NUMBER
-----------------------------------
Arm A0 re-fits the 51-column stacker at 1400 rounds and its prediction vector
must be sha256-IDENTICAL to AH's `pred_C5B3_u2024_seed{S}_localref_at1400.npy`.
If it is, then the OOF matrix, the masks, the feature frame, the fitter and the
library are AH's, so A1 - A0 is a reading of the 16 columns and of nothing else.

`fit_cb` below is exp_027's `fit_catboost` with the model object also returned
(for feature importance).  The A0 sha gate is what proves the two are the same
fit; nothing else here depends on that being true by inspection.

Usage:
    python s1_stack.py --seed 42 --refit-check
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
sys.path.insert(0, os.path.join(SRC_DIR, "ins"))

import exp016_ref as ref                                            # noqa: E402
from labels import assert_no_recovered_in_features                  # noqa: E402
import inseason as ins                                              # noqa: E402
from ins_feats import build_ins_matrix                               # noqa: E402

EXP_ID = "exp_048"
DATA_PATH = os.path.expanduser("~/SungjinPark/data/train.csv")
AH_DIR = os.path.expanduser("~/SungjinPark/build_AH")
OOF_DIR = os.path.join(AH_DIR, "localref", "oof")
AH_REF_JSON = os.path.join(AH_DIR, "local_reference_AH.json")
AH_PRED_DIR = os.path.join(AH_DIR, "localref", "pred")
PACK_DIR = os.path.join(SRC_DIR, "ins")

EXP027_TRAIN = os.path.expanduser(
    "~/SungjinPark/experiments/exp_027/src/train.py")
EXP027_SHA = "e4b308a5f6e4564ed43aed6715d05c0aa617b5e9d22e1bfd51e71dd127416f0f"
LABELS_SHA = "439cd504478b4bc8162e9dedc32be27bfadd5d1178472d6b384b60f46ed7a9f2"
REF016_SHA = "4f24deb71d9eef3d626900f34a71aecd2d6acbae646eb9c92c51189a38a3ece1"

TARGET = "control_success"
DROP_COLS = ["row_id", TARGET]
CAT_COLS = ["top_bottom", "game_type", "base_state"]
ID_COLS = ["pitcher_id", "batter_id"]
FAIL_CELLS_ORDER = ["is_mid_only", "is_rev_only", "is_both", "is_wayoff_true"]
N_THREADS = 6
REF_BRIER = 0.2498069
ROUNDS = [655, 1000, 1400]
CAP = 1400
UNIT = 2024
TRAIN_UPTO = 2023
N_TRAIN_EXPECT = 1116277
N_VALID_EXPECT = 253507
N_VALID_R_EXPECT = 223497


def sha256_file(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def score_of(brier):
    return 100000.0 * (1.0 - brier / REF_BRIER)


def load_exp027():
    got = sha256_file(EXP027_TRAIN)
    assert got == EXP027_SHA, ("exp_027 train.py changed", got)
    spec = importlib.util.spec_from_file_location("exp027_train", EXP027_TRAIN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    c5 = mod.ARMS["C5"]
    assert c5["family"] == "catboost" and c5["params"] == {} and not c5["cat_ids"]
    assert mod.CB_BASE["has_time"] is True
    assert mod.CB_BASE["thread_count"] == N_THREADS
    return mod, c5


def fit_cb(e27, spec, seed, F_tr, y_tr, F_va, feats, cat_native, n_round):
    """exp_027 `fit_catboost`, verbatim, plus the model object."""
    import catboost as cb
    params = dict(e27.CB_BASE, random_seed=seed, iterations=n_round,
                  **spec["params"])
    cat_idx = [feats.index(c) for c in cat_native]
    pool_tr = cb.Pool(F_tr, label=y_tr, cat_features=cat_idx)
    pool_va = cb.Pool(F_va, cat_features=cat_idx)
    model = cb.CatBoostClassifier(**params)
    model.fit(pool_tr)
    assert model.tree_count_ == n_round, (model.tree_count_, n_round)

    def predict_range(a, b):
        return model.predict(pool_va, prediction_type="RawFormulaVal",
                             ntree_start=a, ntree_end=b,
                             thread_count=N_THREADS).astype(np.float64)

    return predict_range, params, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--refit-check", action="store_true",
                    help="also fit A1 independently at 655 and 1000 and check "
                         "the staged evaluation reproduces it bit for bit")
    ap.add_argument("--out", default=os.path.expanduser(
        "~/SungjinPark/experiments/exp_048"))
    args = ap.parse_args()
    SEED = args.seed
    t_start = time.time()
    np.random.seed(SEED)
    for d in ("res", "pred"):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)

    assert sha256_file(os.path.join(SRC_DIR, "labels.py")) == LABELS_SHA
    assert sha256_file(os.path.join(SRC_DIR, "exp016_ref.py")) == REF016_SHA
    e27, C5 = load_exp027()
    build_frame_native = e27.build_frame_native

    tag = f"s{SEED}"
    print(f"[{time.strftime('%H:%M:%S')}] {tag}: exp_048 stage 1 "
          f"(train<={TRAIN_UPTO} -> valid {UNIT}) seed {SEED}", flush=True)

    # ---------------------------------------------------------------- data
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    assert df["row_id"].is_unique
    assert df["season"].is_monotonic_increasing
    base_feats = [c for c in df.columns if c not in DROP_COLS]
    assert len(base_feats) == 47, len(base_feats)
    assert_no_recovered_in_features(base_feats, "base_feats")
    season = df["season"].to_numpy()
    y_all = df[TARGET].to_numpy(dtype=np.float64)
    assert np.isin(y_all, (0.0, 1.0)).all()

    cat_native = [c for c in base_feats if c in CAT_COLS]
    assert len(cat_native) == 3, cat_native
    assert not any(c in cat_native for c in ID_COLS)

    # ------------------------------------------------- inherited OOF matrix
    with open(AH_REF_JSON) as f:
        ahref = json.load(f)
    ah_seed = ahref["per_seed"][str(SEED)]
    oof_path = os.path.join(OOF_DIR, f"stack_fold2024_seed{SEED}.npz")
    oof_sha = sha256_file(oof_path)
    assert oof_sha == ah_seed["run"]["oof_sha256"], (
        "inherited OOF sha mismatch", oof_sha, ah_seed["run"]["oof_sha256"])
    z = np.load(oof_path, allow_pickle=False)
    assert int(z["seed"]) == SEED and int(z["unit"]) == UNIT
    assert int(z["train_upto"]) == TRAIN_UPTO
    assert str(z["train_csv_sha256"].item()) == sha256_file(DATA_PATH), (
        "the OOF was built on a different train.csv")
    tr_full = z["tr_full"].astype(bool)
    va = z["va"].astype(bool)
    assert np.array_equal(z["season"], season)
    assert int(tr_full.sum()) == N_TRAIN_EXPECT, int(tr_full.sum())
    assert int(va.sum()) == N_VALID_EXPECT, int(va.sum())
    stack_cols = [f"p_{c}" for c in FAIL_CELLS_ORDER]
    for c in stack_cols:
        df[c] = z[c]
        assert not np.isnan(z[c][va]).any(), ("valid stack has NaN", c)
    cell_round = {c: int(v) for c, v in zip(FAIL_CELLS_ORDER,
                                            z["cell_round"].tolist())}
    assert cell_round == ah_seed["run"]["cell_iter_used"], cell_round

    y_va = y_all[va]
    game_type_va = df.loc[va, "game_type"].to_numpy()
    is_r = game_type_va == "R"
    assert int(is_r.sum()) == N_VALID_R_EXPECT, int(is_r.sum())
    hand_va = df.loc[va, "pitcher_hand"].to_numpy()

    res = {
        "exp_id": EXP_ID, "stage": 1, "seed": SEED, "unit": UNIT,
        "arm_base": "C5_B3", "train_upto": TRAIN_UPTO, "valid_season": UNIT,
        "n_train_full": int(tr_full.sum()), "n_valid": int(va.sum()),
        "n_valid_R": int(is_r.sum()), "n_valid_F": int((~is_r).sum()),
        "cell_iter_inherited": cell_round,
        "stacker_rounds_fixed_at": CAP,
        "rounds_reported": ROUNDS,
        "inherited": {
            "oof_file": oof_path, "oof_sha256": oof_sha,
            "oof_bytes": os.path.getsize(oof_path),
            "source": "candidate AH local reference run, 2026-08-13",
            "cells_refitted": False,
        },
        "sha256": {
            "train.csv": sha256_file(DATA_PATH),
            "s1_stack.py": sha256_file(os.path.abspath(__file__)),
            "ins_feats.py": sha256_file(os.path.join(SRC_DIR, "ins_feats.py")),
            "inseason.py": sha256_file(os.path.join(PACK_DIR, "inseason.py")),
            "anchor_lt2024.npz": sha256_file(
                os.path.join(PACK_DIR, "anchor_lt2024.npz")),
            "labels.py": LABELS_SHA, "exp016_ref.py": REF016_SHA,
            "exp_027_train.py": EXP027_SHA,
        },
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__},
    }
    import catboost as cb                                            # noqa: F401
    res["versions"]["catboost"] = cb.__version__

    # ----------------------------------------------------- the 16 ins columns
    print(f"[{time.strftime('%H:%M:%S')}] {tag}: building ins_* columns",
          flush=True)
    t_ins = time.time()
    V_ins, ins_cols, ins_info = build_ins_matrix(df, ins, PACK_DIR,
                                                 valid_season=UNIT)
    for j, c in enumerate(ins_cols):
        df[c] = V_ins[:, j]
    ins_info["build_sec"] = round(time.time() - t_ins, 1)
    ins_info["nan_pct_in_train_rows"] = round(
        100.0 * float(np.isnan(V_ins[tr_full, 0]).mean()), 4)
    ins_info["nan_pct_in_valid_rows"] = round(
        100.0 * float(np.isnan(V_ins[va, 0]).mean()), 4)
    assert ins_info["nan_pct_in_valid_rows"] == 0.0
    res["ins_columns"] = ins_info

    # coverage, by segment, on the validation rows
    npn = V_ins[va, ins_cols.index("ins_p_n")]
    nbn = V_ins[va, ins_cols.index("ins_b_n")]
    res["coverage_valid_2024"] = {
        "ins_p_n_gt0_pct_all": round(100.0 * float((npn > 0).mean()), 4),
        "ins_p_n_gt0_pct_R": round(100.0 * float((npn[is_r] > 0).mean()), 4),
        "ins_p_n_gt0_pct_F": round(100.0 * float((npn[~is_r] > 0).mean()), 4),
        "ins_b_n_gt0_pct_all": round(100.0 * float((nbn > 0).mean()), 4),
        "ins_b_n_gt0_pct_R": round(100.0 * float((nbn[is_r] > 0).mean()), 4),
        "ins_b_n_gt0_pct_F": round(100.0 * float((nbn[~is_r] > 0).mean()), 4),
        "ins_p_n_mean_R": round(float(npn[is_r].mean()), 3),
        "ins_p_n_mean_F": round(float(npn[~is_r].mean()), 3),
    }
    del V_ins
    gc.collect()

    # --------------------------------------------------------------- arms
    def evaluate(prob):
        b = float(np.mean((prob - y_va) ** 2))
        return {
            "valid_all_brier": b,
            "valid_all_score_converted": float(f"{score_of(b):.5f}"),
            "valid_all": ref.seg_metrics(y_va, prob),
            "valid_R": ref.seg_metrics(y_va[is_r], prob[is_r]),
            "valid_F": ref.seg_metrics(y_va[~is_r], prob[~is_r]),
        }

    arms = {"A0": base_feats + stack_cols,
            "A1": base_feats + stack_cols + ins_cols}
    assert len(arms["A0"]) == 51 and len(arms["A1"]) == 67
    out = {}
    probs_cap = {}
    for name in ("A0", "A1"):
        feats = arms[name]
        assert_no_recovered_in_features(feats, f"{name} stacker")
        print(f"[{time.strftime('%H:%M:%S')}] {tag} {name}: {len(feats)} cols, "
              f"stacker fit to {CAP}", flush=True)
        F_tr = build_frame_native(df.loc[tr_full], feats, cat_native)
        F_va = build_frame_native(df.loc[va], feats, cat_native)
        t0 = time.time()
        pr, params, model = fit_cb(e27, C5, SEED, F_tr, y_all[tr_full], F_va,
                                   feats, cat_native, CAP)
        fit_min = (time.time() - t0) / 60
        a = {"n_features": len(feats), "features": list(feats),
             "fit_min": round(fit_min, 3),
             "params": {k: (v if isinstance(v, (int, float, str, bool,
                                                type(None))) else str(v))
                        for k, v in params.items()},
             "by_round": {}}
        for r in ROUNDS:
            p_r = ref.sigmoid(pr(0, r))
            a["by_round"][str(r)] = evaluate(p_r)
            pth = os.path.join(args.out, "pred",
                               f"pred_{name}_u2024_seed{SEED}_at{r}.npy")
            np.save(pth, p_r)
            a["by_round"][str(r)]["pred_sha256"] = sha256_file(pth)
            a["by_round"][str(r)]["pred_file"] = pth
            if r == CAP:
                probs_cap[name] = p_r
            print(f"    {name} @{r}: brier "
                  f"{a['by_round'][str(r)]['valid_all_brier']:.12f}  score "
                  f"{a['by_round'][str(r)]['valid_all_score_converted']}",
                  flush=True)
        imp = model.get_feature_importance(type="PredictionValuesChange")
        order = np.argsort(-imp)
        a["importance_PredictionValuesChange"] = {
            feats[i]: float(f"{imp[i]:.6g}") for i in order}
        a["importance_rank"] = {feats[i]: int(k + 1)
                                for k, i in enumerate(order)}
        a["by_pitcher_hand"] = {
            str(h): ref.seg_metrics(y_va[hand_va == h],
                                    probs_cap[name][hand_va == h])
            for h in sorted(pd.unique(hand_va).tolist())}

        if name == "A1" and args.refit_check:
            chk = {}
            for r in (655, 1000):
                t0 = time.time()
                pr_r, _p, _m = fit_cb(e27, C5, SEED, F_tr, y_all[tr_full],
                                      F_va, feats, cat_native, r)
                p_ind = ref.sigmoid(pr_r(0, r))
                p_st = np.load(os.path.join(
                    args.out, "pred",
                    f"pred_{name}_u2024_seed{SEED}_at{r}.npy"))
                d = np.abs(p_ind - p_st)
                chk[str(r)] = {
                    "bit_identical": bool(np.array_equal(p_ind, p_st)),
                    "max_abs_pred_diff": float(d.max()),
                    "n_rows_differing": int(np.count_nonzero(d)),
                    "independent_fit_brier": float(
                        np.mean((p_ind - y_va) ** 2)),
                    "fit_min": round((time.time() - t0) / 60, 3),
                }
                print(f"    A1 refit-check @{r}: bit_identical "
                      f"{chk[str(r)]['bit_identical']}", flush=True)
                del pr_r
                gc.collect()
            a["prefix_identity_check_67col"] = {
                "what": ("an independently fitted r-round 67-column stacker vs "
                         "the first r trees of the 1400-round fit"),
                "per_round": chk}

        out[name] = a
        del pr, model, F_tr, F_va
        gc.collect()

    res["arms"] = out

    # ------------------------------------------------- the control is a gate
    ah1400 = ah_seed["by_round"]["1400"]
    ah_pred = os.path.join(
        AH_PRED_DIR, f"pred_C5B3_u2024_seed{SEED}_localref_at1400.npy")
    a0 = out["A0"]["by_round"]["1400"]
    p_ah = np.load(ah_pred)
    d = np.abs(probs_cap["A0"] - p_ah)
    gate = {
        "what": ("arm A0 re-fits AH's 51-column stacker at 1400 rounds from "
                 "the inherited OOF.  Its prediction must be byte-identical to "
                 "AH's stored vector; if it is, A1 - A0 isolates the 16 "
                 "columns."),
        "ah_reference_file": ah_pred,
        "ah_reference_brier": ah1400["brier_route_A_independent_fit"],
        "observed_brier": a0["valid_all_brier"],
        "abs_diff_brier": abs(a0["valid_all_brier"]
                              - ah1400["brier_route_A_independent_fit"]),
        "tol": 5e-6,
        "ah_pred_sha256": sha256_file(ah_pred),
        "observed_pred_sha256": a0["pred_sha256"],
        "bit_identical": bool(np.array_equal(probs_cap["A0"], p_ah)),
        "max_abs_pred_diff": float(d.max()),
        "n_rows_differing": int(np.count_nonzero(d)),
    }
    gate["PASS"] = bool(gate["abs_diff_brier"] < gate["tol"])
    res["A0_reproduction_gate_vs_AH"] = gate
    print(f"    A0 GATE: bit_identical {gate['bit_identical']}, abs_diff "
          f"{gate['abs_diff_brier']:.3e}, PASS {gate['PASS']}", flush=True)
    assert gate["PASS"], ("A0 reproduction gate FAILED", gate)

    # ------------------------------------------------------------- the delta
    deltas = {}
    for r in ROUNDS:
        b0 = out["A0"]["by_round"][str(r)]["valid_all_brier"]
        b1 = out["A1"]["by_round"][str(r)]["valid_all_brier"]
        deltas[str(r)] = {
            "brier_A0": b0, "brier_A1": b1, "delta_brier": b1 - b0,
            "delta_score_converted": float(
                f"{-100000.0 * (b1 - b0) / REF_BRIER:.5f}"),
            "delta_score_valid_R_only": float(
                f"{-100000.0 * (out['A1']['by_round'][str(r)]['valid_R']['brier'] - out['A0']['by_round'][str(r)]['valid_R']['brier']) / REF_BRIER:.5f}"),
            "delta_score_valid_F_only": float(
                f"{-100000.0 * (out['A1']['by_round'][str(r)]['valid_F']['brier'] - out['A0']['by_round'][str(r)]['valid_F']['brier']) / REF_BRIER:.5f}"),
        }
    res["delta_A1_minus_A0"] = deltas
    res["headline"] = {
        "rounds": CAP,
        "brier_A0": deltas[str(CAP)]["brier_A0"],
        "brier_A1": deltas[str(CAP)]["brier_A1"],
        "delta_score_converted": deltas[str(CAP)]["delta_score_converted"],
    }
    res["minutes"] = round((time.time() - t_start) / 60, 2)

    p = os.path.join(args.out, "res", f"res_exp048_s1_seed{SEED}.json")
    with open(p, "w") as f:
        json.dump(res, f, indent=1)
    print("RESULT: " + json.dumps({
        "seed": SEED, "gate_pass": gate["PASS"],
        "gate_bit_identical": gate["bit_identical"],
        "delta_score_at_1400": deltas["1400"]["delta_score_converted"],
        "brier_A0_1400": deltas["1400"]["brier_A0"],
        "brier_A1_1400": deltas["1400"]["brier_A1"],
        "minutes": res["minutes"]}), flush=True)


if __name__ == "__main__":
    main()
