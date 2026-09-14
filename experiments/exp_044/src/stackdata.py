"""exp_044 — the fold_2023 STACKER design matrices of C5_B3.

Same relationship to the pipeline as celldata.py has for the cells: the pools,
the feature list and the row masks below are b3_cat.py's stacker section lifted
out so the search can fit that one model many times.  The four extra columns are
NOT recomputed here — they are read from the `--save-stack` npz that a real
stage-1 run wrote, and the npz's sha256 is checked against the sha the stage-1
result json recorded.  That is what makes "these are the pipeline's own
forward-chaining OOF columns" a checked claim.

Fold: unit 2023 (train <= 2022 -> valid 2023).  fold_2024 is never touched.
Train pool: tr_full = (season <= 2022) & ~f_excl        (b3_cat.py line for line)
Valid     : every season-2023 row (not just recov_ok) — the stacker is scored on
            valid_all, which is the fold's real metric.
"""

import hashlib
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

from labels import FAIL_CELLS, assert_no_recovered_in_features, recover  # noqa: E402

DATA_PATH = os.path.expanduser("~/SungjinPark/data/train.csv")
EXP027_TRAIN = os.path.expanduser("~/SungjinPark/experiments/exp_027/src/train.py")
EXP027_SHA = "e4b308a5f6e4564ed43aed6715d05c0aa617b5e9d22e1bfd51e71dd127416f0f"
LABELS_SHA = "439cd504478b4bc8162e9dedc32be27bfadd5d1178472d6b384b60f46ed7a9f2"
REF016_SHA = "4f24deb71d9eef3d626900f34a71aecd2d6acbae646eb9c92c51189a38a3ece1"
B3CAT_SHA = "0984c57e5af66600f9fca5532a984ddd6d4c22996c9e0105d3e798bc7bfa76ca"

TARGET = "control_success"
DROP_COLS = ["row_id", TARGET]
CAT_COLS = ["top_bottom", "game_type", "base_state"]
F_EXCLUDE_MAX_SEASON = 2022
SEARCH_TRAIN_UPTO = 2022
SEARCH_VALID = 2023
N_TRAIN_FULL_EXPECTED = 870752      # UNIT_SPEC[2023]["n_train"] in b3_cat.py
N_VALID_EXPECTED = 245525           # UNIT_SPEC[2023]["n_valid"]


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
    got = sha256_file(EXP027_TRAIN)
    assert got == EXP027_SHA, ("exp_027 train.py changed", got)
    spec = importlib.util.spec_from_file_location("exp027_train", EXP027_TRAIN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build(stack_npz, stage1_json):
    """Design matrices for the fold_2023 stacker, OOF columns read from disk."""
    import json
    assert sha256_file(os.path.join(SRC_DIR, "labels.py")) == LABELS_SHA
    assert sha256_file(os.path.join(SRC_DIR, "exp016_ref.py")) == REF016_SHA
    assert sha256_file(os.path.join(SRC_DIR, "b3_cat.py")) == B3CAT_SHA

    stack_npz = os.path.expanduser(stack_npz)
    stage1_json = os.path.expanduser(stage1_json)
    with open(stage1_json) as f:
        s1 = json.load(f)
    # the OOF columns must come from a stage-1 run of THIS fold and seed
    assert s1["stage"] == 1 and s1["unit"] == 2023 and s1["seed"] == 42, s1["unit"]
    got = sha256_file(stack_npz)
    assert got == s1["stack_sha256"], ("stack npz is not the one stage 1 wrote",
                                       got, s1["stack_sha256"])
    z = np.load(stack_npz)
    stack_cols = [f"p_{c}" for c in FAIL_CELLS]
    assert list(s1["stack_cols"]) == stack_cols, s1["stack_cols"]
    assert sorted(z.files) == sorted(stack_cols), z.files

    e27 = load_exp027()
    build_frame_native = e27.build_frame_native

    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    assert df["row_id"].is_unique
    assert df["season"].is_monotonic_increasing
    base_feats = [c for c in df.columns if c not in DROP_COLS]
    assert len(base_feats) == 47, len(base_feats)
    assert_no_recovered_in_features(base_feats, "base_feats (before recovery)")
    rec = recover(df, verbose=False)
    assert rec["success_recovery"]["agree_pct"] == 100.0

    season = df["season"].to_numpy()
    f_excl = ((df["season"] <= F_EXCLUDE_MAX_SEASON)
              & (df["game_type"] == "F")).to_numpy()
    y_all = df[TARGET].to_numpy(dtype=np.float64)
    assert np.isin(y_all, (0.0, 1.0)).all()
    tkey = season.astype(np.int64) * 100 + df["game_month"].to_numpy(np.int64)

    tu, vy = SEARCH_TRAIN_UPTO, SEARCH_VALID
    tr_full = (season <= tu) & ~f_excl
    va = season == vy
    assert int(tr_full.sum()) == N_TRAIN_FULL_EXPECTED, int(tr_full.sum())
    assert int(va.sum()) == N_VALID_EXPECTED, int(va.sum())
    assert int(tr_full.sum()) == s1["n_train_full"], s1["n_train_full"]
    inv = int((np.diff(tkey[tr_full]) < 0).sum())
    assert inv == 0, ("tr_full not time ordered", inv)

    for c in stack_cols:
        col = np.asarray(z[c], dtype=np.float64)
        assert col.shape[0] == len(df), (c, col.shape, len(df))
        df[c] = col
    nan_va = int(np.isnan(df.loc[va, stack_cols[0]].to_numpy()).sum())
    assert nan_va == 0, ("valid stack column has NaN", nan_va)
    nan_tr_pct = round(100.0 * float(np.isnan(
        df.loc[tr_full, stack_cols[0]].to_numpy()).mean()), 3)

    feats = base_feats + stack_cols
    assert len(feats) == 51, len(feats)
    assert_no_recovered_in_features(feats, "B3 stacker")
    cat_native = [c for c in base_feats if c in CAT_COLS]
    assert len(cat_native) == 3, cat_native

    F_tr = build_frame_native(df.loc[tr_full], feats, cat_native)
    F_va = build_frame_native(df.loc[va], feats, cat_native)
    out = {
        "model": "stacker",
        "feats": feats,
        "cat_native": cat_native,
        "cat_idx": [feats.index(c) for c in cat_native],
        "F_tr": F_tr, "y_tr": y_all[tr_full],
        "F_va": F_va, "y_va": y_all[va],
        "n_tr": int(tr_full.sum()), "n_va": int(va.sum()),
        "n_train_full": int(tr_full.sum()), "n_valid_all": int(va.sum()),
        "pos_rate_train": float(y_all[tr_full].mean()),
        "pos_rate_valid": float(y_all[va].mean()),
        "nan_stack_rows_in_train_pct": nan_tr_pct,
        "sha_train_csv": sha256_file(DATA_PATH),
        "sha_stack_npz": got,
        "note_stack_source": (f"{os.path.basename(stack_npz)} written by "
                              f"{os.path.basename(stage1_json)} "
                              f"(unit 2023, seed 42, stage 1)"),
        "note_cell_params": json.dumps(s1.get("params_map", {}), sort_keys=True),
    }
    del df
    return out
