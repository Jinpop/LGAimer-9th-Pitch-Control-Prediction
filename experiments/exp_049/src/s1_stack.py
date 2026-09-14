"""exp_049 STAGE 1 — the R-branch stacker at 1400 rounds, 67 vs 81 columns.

INHERITED, byte-checked before use
----------------------------------
  * `build_AH/localref/oof/stack_fold2024_seed{S}.npz` — the four forward-chained
    OOF cell columns and the masks.  sha256 must equal AH's recorded digest.
    NO CELL MODEL IS REFITTED.
  * exp_048's `ins_feats.build_ins_matrix` + the `ins/` pack (16 `ins_*` columns).
  * exp_027's `build_frame_native` / `CB_BASE` / `ARMS["C5"]`, exp_016's
    `sigmoid` / `seg_metrics`, all sha-pinned.
  * exp_037's featdir: the 14 as-of columns, read from `feat_main.parquet`.

NEW: the 14 columns (`p6_*` 6, `idea_matchup_pitch`, `i3_anom_*` 7) and nothing
else.  Stacker rounds stay pinned at 1400; seeds stay 42 / 7 / 2024.

THE CONTROL IS A GATE, NOT A NUMBER
-----------------------------------
Arm A0 is exp_048's A1: the 67-column stacker at 1400.  It is REFITTED here and
its prediction vector must be sha256-identical to exp_048's stored
`pred_A1_u2024_seed{S}_at1400.npy`.  If it is, the OOF matrix, the masks, the
frame builder, the fitter and the library are exp_048's, so A1 - A0 reads the 14
columns and nothing else.

ARMS
----
  A0          67 = 47 base + 4 stack + 16 ins            (control = exp_048 A1)
  A1          81 = A0 + 14
  L_no_p6     75 = A1 - p6_* (6)
  L_no_idea   80 = A1 - idea_matchup_pitch (1)
  L_no_i3     74 = A1 - i3_anom_* (7)

The three L_* arms are the leave-one-family-out decomposition exp_047 taught us
to demand: a staircase alone can give a family a sign it does not keep once the
others are present.

Usage:  python s1_stack.py --seed 42
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
from ins_feats import build_ins_matrix                              # noqa: E402
import feat14 as F14                                                # noqa: E402

EXP_ID = "exp_049"
DATA_PATH = os.path.expanduser("~/SungjinPark/data/train.csv")
AH_DIR = os.path.expanduser("~/SungjinPark/build_AH")
OOF_DIR = os.path.join(AH_DIR, "localref", "oof")
AH_REF_JSON = os.path.join(AH_DIR, "local_reference_AH.json")
E048_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_048")
FEATDIR = os.path.expanduser("~/SungjinPark/experiments/exp_037/featdir")
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

PITCHMIX = ["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
            "asof_pitcher_offspeed_rate", "asof_pitcher_pitchmix_n"]


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
    ap.add_argument("--out", default=os.path.expanduser(
        "~/SungjinPark/experiments/exp_049"))
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
    print(f"[{time.strftime('%H:%M:%S')}] {tag}: exp_049 stage 1 "
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
    assert str(z["train_csv_sha256"].item()) == sha256_file(DATA_PATH)
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
        "stacker_rounds_fixed_at": CAP, "rounds_reported": ROUNDS,
        "inherited": {
            "oof_file": oof_path, "oof_sha256": oof_sha,
            "source": "candidate AH local reference run, 2026-08-13",
            "cells_refitted": False,
            "featdir": FEATDIR,
            "feat_parquet_sha256": sha256_file(
                os.path.join(FEATDIR, "feat", "feat_main.parquet")),
        },
        "sha256": {
            "train.csv": sha256_file(DATA_PATH),
            "s1_stack.py": sha256_file(os.path.abspath(__file__)),
            "feat14.py": sha256_file(os.path.join(SRC_DIR, "feat14.py")),
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
    V_ins, ins_cols, ins_info = build_ins_matrix(df, ins, PACK_DIR,
                                                 valid_season=UNIT)
    for j, c in enumerate(ins_cols):
        df[c] = V_ins[:, j]
    assert len(ins_cols) == 16
    del V_ins
    gc.collect()

    # -------------------------------------------------------- the 14 columns
    print(f"[{time.strftime('%H:%M:%S')}] {tag}: attaching the 14 as-of columns",
          flush=True)
    feat = pd.read_parquet(os.path.join(FEATDIR, "feat", "feat_main.parquet"),
                           columns=["row_id"] + F14.COLS14)
    assert len(feat) == len(df)
    feat = feat.set_index("row_id").loc[df["row_id"].to_numpy()]
    assert np.array_equal(feat.index.to_numpy(), df["row_id"].to_numpy())
    new_cols = list(F14.COLS14)
    for c in new_cols:
        df[c] = feat[c].to_numpy(dtype=np.float64)
    nan_tr = {c: round(100.0 * float(np.isnan(df[c].to_numpy()[tr_full]).mean()), 4)
              for c in new_cols}
    nan_va = {c: round(100.0 * float(np.isnan(df[c].to_numpy()[va]).mean()), 4)
              for c in new_cols}
    res["new_columns"] = {
        "n": len(new_cols), "names": new_cols, "groups": F14.GROUPS14,
        "nan_pct_in_train_rows": nan_tr, "nan_pct_in_valid_rows": nan_va,
        "note": ("2019 rows carry NaN by design (no antecedent season); those "
                 "rows are 237,413 of the 1,116,277 training rows, hence the "
                 "~21.3% NaN share on the train side and 0% on valid."),
    }
    assert max(nan_va.values()) == 0.0, nan_va
    del feat
    gc.collect()

    # ------------------------------------- correlation with what we already had
    corr_targets = [c for c in PITCHMIX if c in df.columns] + ins_cols
    sub_va = df.loc[va]
    cm = {}
    for c in new_cols:
        a = sub_va[c].to_numpy(dtype=np.float64)
        row = {}
        for b_name in corr_targets:
            b = sub_va[b_name].to_numpy(dtype=np.float64)
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() < 1000 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
                row[b_name] = None
            else:
                row[b_name] = float(f"{np.corrcoef(a[ok], b[ok])[0, 1]:.4f}")
        cm[c] = row
    res["corr_new14_vs_existing"] = cm
    res["corr_max_abs_vs_pitchmix"] = {
        c: max((abs(v) for k, v in cm[c].items()
                if k in PITCHMIX and v is not None), default=None)
        for c in new_cols}
    # and among the 14 themselves
    M = sub_va[new_cols].to_numpy(dtype=np.float64)
    with np.errstate(invalid="ignore"):
        CC = np.corrcoef(M, rowvar=False)
    res["corr_within_new14"] = {
        new_cols[i]: {new_cols[j]: float(f"{CC[i, j]:.4f}")
                      for j in range(len(new_cols)) if j != i}
        for i in range(len(new_cols))}
    del sub_va, M
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

    A0 = base_feats + stack_cols + ins_cols
    A1 = A0 + new_cols
    arms = {
        "A0": A0,
        "A1": A1,
        "L_no_p6": [c for c in A1 if c not in F14.PC],
        "L_no_idea": [c for c in A1 if c != F14.IDEA],
        "L_no_i3": [c for c in A1 if c not in F14.ANOM],
    }
    assert len(arms["A0"]) == 67, len(arms["A0"])
    assert len(arms["A1"]) == 81, len(arms["A1"])
    assert len(arms["L_no_p6"]) == 75
    assert len(arms["L_no_idea"]) == 80
    assert len(arms["L_no_i3"]) == 74

    out = {}
    probs_cap = {}
    for name, feats in arms.items():
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
             "fit_min": round(fit_min, 3), "by_round": {}}
        if name in ("A0", "A1"):
            a["params"] = {k: (v if isinstance(v, (int, float, str, bool,
                                                   type(None))) else str(v))
                           for k, v in params.items()}
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
        out[name] = a
        del pr, model, F_tr, F_va
        gc.collect()

    res["arms"] = out

    # ------------------------------------------------- the control is a gate
    e048_json = os.path.join(E048_DIR, "res", f"res_exp048_s1_seed{SEED}.json")
    with open(e048_json) as f:
        e048 = json.load(f)
    ref1400 = e048["arms"]["A1"]["by_round"]["1400"]
    ref_pred = os.path.join(E048_DIR, "pred",
                            f"pred_A1_u2024_seed{SEED}_at1400.npy")
    a0 = out["A0"]["by_round"]["1400"]
    p_ref = np.load(ref_pred)
    d = np.abs(probs_cap["A0"] - p_ref)
    gate = {
        "what": ("arm A0 re-fits exp_048's 67-column stacker at 1400 rounds "
                 "from the same inherited OOF.  Its prediction must be "
                 "byte-identical to exp_048's stored vector; if it is, "
                 "A1 - A0 isolates the 14 columns."),
        "reference_file": ref_pred,
        "reference_brier": ref1400["valid_all_brier"],
        "reference_pred_sha256": ref1400["pred_sha256"],
        "observed_brier": a0["valid_all_brier"],
        "observed_pred_sha256": a0["pred_sha256"],
        "abs_diff_brier": abs(a0["valid_all_brier"]
                              - ref1400["valid_all_brier"]),
        "tol": 5e-6,
        "sha_on_disk_reference": sha256_file(ref_pred),
        "bit_identical": bool(np.array_equal(probs_cap["A0"], p_ref)),
        "max_abs_pred_diff": float(d.max()),
        "n_rows_differing": int(np.count_nonzero(d)),
    }
    gate["PASS"] = bool(gate["abs_diff_brier"] < gate["tol"])
    res["A0_reproduction_gate_vs_exp048"] = gate
    print(f"    A0 GATE: bit_identical {gate['bit_identical']}, abs_diff "
          f"{gate['abs_diff_brier']:.3e}, PASS {gate['PASS']}", flush=True)
    assert gate["PASS"], ("A0 reproduction gate FAILED", gate)

    # ------------------------------------------------------------- the deltas
    deltas = {}
    for r in ROUNDS:
        b0 = out["A0"]["by_round"][str(r)]["valid_all_brier"]
        row = {}
        for name in ("A1", "L_no_p6", "L_no_idea", "L_no_i3"):
            b = out[name]["by_round"][str(r)]["valid_all_brier"]
            row[name] = {
                "brier": b, "delta_brier_vs_A0": b - b0,
                "delta_score_vs_A0": float(
                    f"{-100000.0 * (b - b0) / REF_BRIER:.5f}")}
        b1 = out["A1"]["by_round"][str(r)]["valid_all_brier"]
        for name in ("L_no_p6", "L_no_idea", "L_no_i3"):
            b = out[name]["by_round"][str(r)]["valid_all_brier"]
            # the family's own worth = what is lost by removing it
            row[name]["family_worth_score"] = float(
                f"{-100000.0 * (b1 - b) / REF_BRIER:.5f}")
        row["brier_A0"] = b0
        row["delta_score_valid_R_only_A1"] = float(
            f"{-100000.0 * (out['A1']['by_round'][str(r)]['valid_R']['brier'] - out['A0']['by_round'][str(r)]['valid_R']['brier']) / REF_BRIER:.5f}")
        row["delta_score_valid_F_only_A1"] = float(
            f"{-100000.0 * (out['A1']['by_round'][str(r)]['valid_F']['brier'] - out['A0']['by_round'][str(r)]['valid_F']['brier']) / REF_BRIER:.5f}")
        deltas[str(r)] = row
    res["deltas"] = deltas
    res["headline"] = {
        "rounds": CAP,
        "brier_A0": deltas[str(CAP)]["brier_A0"],
        "brier_A1": deltas[str(CAP)]["A1"]["brier"],
        "delta_score_converted": deltas[str(CAP)]["A1"]["delta_score_vs_A0"],
    }
    res["minutes"] = round((time.time() - t_start) / 60, 2)

    p = os.path.join(args.out, "res", f"res_exp049_s1_seed{SEED}.json")
    with open(p, "w") as f:
        json.dump(res, f, indent=1)
    print("RESULT: " + json.dumps({
        "seed": SEED, "gate_pass": gate["PASS"],
        "gate_bit_identical": gate["bit_identical"],
        "delta_score_at_1400": res["headline"]["delta_score_converted"],
        "brier_A0_1400": res["headline"]["brier_A0"],
        "brier_A1_1400": res["headline"]["brier_A1"],
        "minutes": res["minutes"]}), flush=True)


if __name__ == "__main__":
    main()
