"""exp_050 — stacker-seed replication of the exp_049 A1(81) vs L_no_p6(75) ordering.

🔴 THIS IS NOT THE PRE-REGISTERED REPLICATION exp_050/config.yaml ASKED FOR.
--------------------------------------------------------------------------
config.yaml asked for seeds 13 / 101 / 777.  The fold_2024 OOF exists on the
server for seeds 42 / 7 / 2024 ONLY (`build_AH/localref/oof/`), and
`local_reference_AH.json` has `per_seed` keys {42, 7, 2024}.  Regenerating an
OOF for a new seed costs 17-21 min/seed (measured from
`build_AH/localref/logs/`: 19.67 / 22.37 / 23.38 min total per seed, of which
the stacker fit is 2.10-2.45 min) AND would break config.yaml's own invariant
list ("cells frozen, OOF inherited").  So it was not done.

What is done instead, holding EVERY config invariant:
  * the OOF is FIXED to the inherited seed-42 file (cells frozen, OOF inherited)
  * only the STACKER's `random_seed` moves: 42 -> 13 / 101 / 777
  * arms are A1 (81 cols) and L_no_p6 (75 cols) only

⚠️ The noise source is NARROWER than exp_049's.  In exp_049 the "seed" moved the
cell OOF and the stacker together (the OOF files are per-seed).  Here only the
stacker's randomness moves, so the seed-to-seed sd is structurally smaller and
MUST NOT be compared head-to-head against exp_049's sd 3.68 / 22.72.

IDENTITY ANCHOR
---------------
Running with --stacker-seed 42 --oof-seed 42 is exactly exp_049's seed-42 run
restricted to two arms.  Its A1 and L_no_p6 prediction vectors must be
sha256-identical to exp_049's stored
`pred_{arm}_u2024_seed42_at{r}.npy` at r in {655, 1000, 1400}.  If they are, the
frame builder, the fitter, the inherited OOF, the 14 columns, the masks and the
library are exp_049's, so the new stacker seeds differ in the seed and nothing
else.

Usage:  python s1_seedvar.py --stacker-seed 13 [--oof-seed 42]
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

EXP_ID = "exp_050"
DATA_PATH = os.path.expanduser("~/SungjinPark/data/train.csv")
AH_DIR = os.path.expanduser("~/SungjinPark/build_AH")
OOF_DIR = os.path.join(AH_DIR, "localref", "oof")
AH_REF_JSON = os.path.join(AH_DIR, "local_reference_AH.json")
E049_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_049")
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
    """exp_027 `fit_catboost`, verbatim (identical to exp_049's copy)."""
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
    ap.add_argument("--stacker-seed", type=int, required=True)
    ap.add_argument("--oof-seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.expanduser(
        "~/SungjinPark/experiments/exp_050"))
    args = ap.parse_args()
    SEED = args.stacker_seed
    OOF_SEED = args.oof_seed
    t_start = time.time()
    np.random.seed(SEED)
    for d in ("res", "pred"):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)

    assert sha256_file(os.path.join(SRC_DIR, "labels.py")) == LABELS_SHA
    assert sha256_file(os.path.join(SRC_DIR, "exp016_ref.py")) == REF016_SHA
    e27, C5 = load_exp027()
    build_frame_native = e27.build_frame_native

    tag = f"st{SEED}_oof{OOF_SEED}"
    print(f"[{time.strftime('%H:%M:%S')}] {tag}: exp_050 "
          f"(train<={TRAIN_UPTO} -> valid {UNIT}) stacker seed {SEED}, "
          f"OOF seed {OOF_SEED}", flush=True)

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
    assert str(OOF_SEED) in ahref["per_seed"], (
        "no fold_2024 OOF reference for this seed",
        OOF_SEED, sorted(ahref["per_seed"].keys()))
    ah_seed = ahref["per_seed"][str(OOF_SEED)]
    oof_path = os.path.join(OOF_DIR, f"stack_fold2024_seed{OOF_SEED}.npz")
    oof_sha = sha256_file(oof_path)
    assert oof_sha == ah_seed["run"]["oof_sha256"], (
        "inherited OOF sha mismatch", oof_sha, ah_seed["run"]["oof_sha256"])
    z = np.load(oof_path, allow_pickle=False)
    assert int(z["seed"]) == OOF_SEED and int(z["unit"]) == UNIT
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

    res = {
        "exp_id": EXP_ID,
        "🔴_not_the_preregistered_replication": (
            "config.yaml asked for seeds 13/101/777 with their own fold_2024 "
            "OOF.  Only seeds 42/7/2024 have one on the server.  Here the OOF "
            "is FIXED at seed 42 and only the STACKER seed moves.  The noise "
            "source is narrower than exp_049's, so the sd is NOT comparable to "
            "exp_049's 3.68 / 22.72."),
        "stacker_seed": SEED, "oof_seed": OOF_SEED, "unit": UNIT,
        "arm_base": "C5_B3", "train_upto": TRAIN_UPTO, "valid_season": UNIT,
        "n_train_full": int(tr_full.sum()), "n_valid": int(va.sum()),
        "n_valid_R": int(is_r.sum()), "n_valid_F": int((~is_r).sum()),
        "cell_iter_inherited": cell_round,
        "stacker_rounds_fixed_at": CAP, "rounds_reported": ROUNDS,
        "inherited": {
            "oof_file": oof_path, "oof_sha256": oof_sha,
            "source": "candidate AH local reference run, 2026-08-13",
            "cells_refitted": False, "featdir": FEATDIR,
            "feat_parquet_sha256": sha256_file(
                os.path.join(FEATDIR, "feat", "feat_main.parquet")),
        },
        "sha256": {
            "train.csv": sha256_file(DATA_PATH),
            "s1_seedvar.py": sha256_file(os.path.abspath(__file__)),
            "exp_049_s1_stack.py": sha256_file(
                os.path.join(E049_DIR, "src", "s1_stack.py")),
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
    nan_va = {c: round(100.0 * float(np.isnan(df[c].to_numpy()[va]).mean()), 4)
              for c in new_cols}
    assert max(nan_va.values()) == 0.0, nan_va
    del feat
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
        "A1": A1,
        "L_no_p6": [c for c in A1 if c not in F14.PC],
    }
    assert len(A0) == 67, len(A0)
    assert len(arms["A1"]) == 81, len(arms["A1"])
    assert len(arms["L_no_p6"]) == 75, len(arms["L_no_p6"])

    out = {}
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
             "fit_min": round(fit_min, 3), "by_round": {},
             "params": {k: (v if isinstance(v, (int, float, str, bool,
                                                type(None))) else str(v))
                        for k, v in params.items()}}
        for r in ROUNDS:
            p_r = ref.sigmoid(pr(0, r))
            a["by_round"][str(r)] = evaluate(p_r)
            pth = os.path.join(
                args.out, "pred",
                f"pred_{name}_u2024_st{SEED}_oof{OOF_SEED}_at{r}.npy")
            np.save(pth, p_r)
            a["by_round"][str(r)]["pred_sha256"] = sha256_file(pth)
            a["by_round"][str(r)]["pred_file"] = pth
            print(f"    {name} @{r}: brier "
                  f"{a['by_round'][str(r)]['valid_all_brier']:.12f}  score "
                  f"{a['by_round'][str(r)]['valid_all_score_converted']}",
                  flush=True)
        imp = model.get_feature_importance(type="PredictionValuesChange")
        order = np.argsort(-imp)
        a["importance_rank"] = {feats[i]: int(k + 1)
                                for k, i in enumerate(order)}
        out[name] = a
        del pr, model, F_tr, F_va
        gc.collect()

    res["arms"] = out

    # --------------------------------------------------- the identity anchor
    anchor = {"applicable": bool(SEED == 42 and OOF_SEED == 42),
              "what": ("stacker seed 42 on OOF seed 42 IS exp_049's seed-42 "
                       "run restricted to two arms; the prediction vectors "
                       "must be sha256-identical to exp_049's stored ones.")}
    if anchor["applicable"]:
        per = {}
        ok_all = True
        for name in arms:
            for r in ROUNDS:
                refp = os.path.join(
                    E049_DIR, "pred", f"pred_{name}_u2024_seed42_at{r}.npy")
                mine = out[name]["by_round"][str(r)]["pred_file"]
                same = sha256_file(refp) == sha256_file(mine)
                d = float(np.abs(np.load(refp) - np.load(mine)).max())
                per[f"{name}@{r}"] = {
                    "exp049_file": refp,
                    "exp049_sha256": sha256_file(refp),
                    "exp050_sha256": sha256_file(mine),
                    "sha_identical": bool(same), "max_abs_pred_diff": d}
                ok_all = ok_all and same and d == 0.0
        anchor["per_arm_round"] = per
        anchor["PASS"] = bool(ok_all)
        print(f"    IDENTITY ANCHOR: PASS={anchor['PASS']}", flush=True)
        assert anchor["PASS"], ("identity anchor FAILED", anchor)
    res["identity_anchor_vs_exp049_seed42"] = anchor

    # ------------------------------------- L_no_p6 - A1 at every round
    contrast = {}
    for r in ROUNDS:
        b1 = out["A1"]["by_round"][str(r)]["valid_all_brier"]
        bl = out["L_no_p6"]["by_round"][str(r)]["valid_all_brier"]
        contrast[str(r)] = {
            "brier_A1": b1, "brier_L_no_p6": bl,
            "delta_brier_L_minus_A1": bl - b1,
            "delta_score_L_minus_A1": float(
                f"{-100000.0 * (bl - b1) / REF_BRIER:.5f}"),
            "L_no_p6_better": bool(bl < b1),
        }
    res["contrast_L_no_p6_minus_A1"] = contrast
    res["minutes"] = round((time.time() - t_start) / 60, 2)

    p = os.path.join(args.out, "res",
                     f"res_exp050_st{SEED}_oof{OOF_SEED}.json")
    with open(p, "w") as f:
        json.dump(res, f, indent=1)
    print("RESULT: " + json.dumps({
        "stacker_seed": SEED, "oof_seed": OOF_SEED,
        "anchor": anchor.get("PASS", None),
        "contrast": {r: contrast[r]["delta_score_L_minus_A1"] for r in contrast},
        "minutes": res["minutes"]}), flush=True)


if __name__ == "__main__":
    main()
