"""exp_042 — the fold_2023 data for ONE binary cell model of C5_B3.

This is the exact pool/mask/target construction that
`experiments/exp_030/src/b3_cat.py` uses for its cell models, lifted out so
that the hyper-parameter search can fit that one model many times without
re-deriving the masks.  Nothing here is new: the masks are copied line for
line from b3_cat.py's `main()` and the sha of that file is asserted, so
"same pool as the pipeline" is a checked claim rather than a remembered one.

Search fold: unit 2023 = train <= 2022 -> valid 2023.
Cell pool  : tr_mode = (season <= 2022) & ~f_excl & recov_ok & (succ_season <= 2022)
Cell valid : rows of season 2023 that are recov_ok  (`va_ok`), same as b3_cat.py.
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
N_TRAIN_MODE_EXPECTED = 870542      # observed in exp_030 res_C5B3_u2023_seed42_st1.json
N_TRAIN_FULL_EXPECTED = 870752


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


def build(cell):
    """Return dict with the CatBoost design matrices for one cell, fold_2023."""
    assert cell in FAIL_CELLS, (cell, FAIL_CELLS)
    assert sha256_file(os.path.join(SRC_DIR, "labels.py")) == LABELS_SHA
    assert sha256_file(os.path.join(SRC_DIR, "exp016_ref.py")) == REF016_SHA
    assert sha256_file(os.path.join(SRC_DIR, "b3_cat.py")) == B3CAT_SHA
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
    ok = df["recov_ok"].to_numpy()
    succ_season = df.groupby("pitcher_id", sort=False)["season"].shift(-1).to_numpy()
    f_excl = ((df["season"] <= F_EXCLUDE_MAX_SEASON)
              & (df["game_type"] == "F")).to_numpy()
    tkey = season.astype(np.int64) * 100 + df["game_month"].to_numpy(np.int64)

    tu, vy = SEARCH_TRAIN_UPTO, SEARCH_VALID
    tr_full = (season <= tu) & ~f_excl
    tr_mode = tr_full & ok & (succ_season <= tu)
    va = season == vy
    va_ok = ok[va]

    # has_time=True uses the pool's row order as the time axis
    inv = int((np.diff(tkey[tr_mode]) < 0).sum())
    assert inv == 0, ("tr_mode not time ordered", inv)

    cat_native = [c for c in base_feats if c in CAT_COLS]
    assert len(cat_native) == 3, cat_native
    assert_no_recovered_in_features(base_feats, "cell model")

    F_tr = build_frame_native(df.loc[tr_mode], base_feats, cat_native)
    F_va_all = build_frame_native(df.loc[va], base_feats, cat_native)
    F_va = F_va_all.loc[va_ok].reset_index(drop=True)

    y_tr = df.loc[tr_mode, cell].to_numpy(dtype=np.float64)
    y_va = df.loc[va, cell].to_numpy(dtype=np.float64)[va_ok]
    assert not np.isnan(y_tr).any() and not np.isnan(y_va).any()

    out = {
        "cell": cell,
        "feats": base_feats,
        "cat_native": cat_native,
        "cat_idx": [base_feats.index(c) for c in cat_native],
        "F_tr": F_tr, "y_tr": y_tr,
        "F_va": F_va, "y_va": y_va,
        "n_train_mode": int(tr_mode.sum()),
        "n_train_full": int(tr_full.sum()),
        "n_valid_all": int(va.sum()),
        "n_valid_ok": int(va_ok.sum()),
        "pos_rate_train": float(y_tr.mean()),
        "pos_rate_valid": float(y_va.mean()),
        "train_csv_sha256": sha256_file(DATA_PATH),
    }
    assert out["n_train_mode"] == N_TRAIN_MODE_EXPECTED, out["n_train_mode"]
    assert out["n_train_full"] == N_TRAIN_FULL_EXPECTED, out["n_train_full"]
    del df, F_va_all
    return out
