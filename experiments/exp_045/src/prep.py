"""exp_045 — fold-wise preprocessing cache builder (CPU only).

Produces cache/fold_{YEAR}.npz + cache/fold_{YEAR}.json for YEAR in {2023, 2024}.

    python prep.py --fold 2023 --fold 2024

EVERY fit-time quantity is derived from the fold's TRAIN slice only:
  vocab (entity index maps), quantile transformer, missing-value medians.
The valid slice is only ever *transformed*.  See the leak-boundary block in
common.py.

Layout written per fold:
    num_tr / num_va : float32 [N, D]  = [continuous | isna flags | binaries]
    emb_tr / emb_va : int64   [N, 5]  = vocab indices, 0 == OOV
    y_tr  / y_va    : float32 [N]
    is_r_va         : bool    [N]
    row_id_va       : the valid slice's row_id strings (alignment key)
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

QT_KW = dict(n_quantiles=1000, output_distribution="normal",
             subsample=2_000_000, random_state=0)   # subsample > N -> no subsampling


def build(df, inh, valid_year, out_dir, verbose=True):
    folds = {f[2]: f for f in inh["folds"]}
    assert valid_year in folds, valid_year
    _, train_until, _ = folds[valid_year]

    tr, va = C.masks_for_fold(df, train_until, valid_year,
                              inh["f_exclude_max_season"])
    n_tr, n_va = int(tr.sum()), int(va.sum())
    is_r_va = (df.loc[va, "game_type"] == "R").to_numpy()
    C.assert_fold_shapes(inh, valid_year, n_tr, n_va, int(is_r_va.sum()))
    if verbose:
        print(f"[prep] fold_{valid_year}: train<= {train_until} n={n_tr:,} "
              f"-> valid {valid_year} n={n_va:,} (R={int(is_r_va.sum()):,})",
              flush=True)

    feat_cols = [c for c in df.columns if c not in (C.ROW_ID, C.TARGET)]
    assert len(feat_cols) == 47, len(feat_cols)
    cont_cols = [c for c in feat_cols
                 if c not in C.EMB_COLS and c not in C.BIN_COLS]
    assert len(cont_cols) == 35, (len(cont_cols), cont_cols)

    meta = {"exp_id": "exp_045", "fold": f"fold_{valid_year}",
            "train_until": train_until, "valid_year": valid_year,
            "n_train": n_tr, "n_valid": n_va,
            "n_valid_R": int(is_r_va.sum()),
            "feature_cols_47": feat_cols,
            "emb_cols": C.EMB_COLS, "bin_cols": C.BIN_COLS,
            "cont_cols": cont_cols,
            "qt_kwargs": {k: v for k, v in QT_KW.items()},
            "inherited": inh}

    # ---------------- (b) vocabularies: TRAIN SLICE ONLY -------------------
    emb_tr = np.zeros((n_tr, len(C.EMB_COLS)), dtype=np.int64)
    emb_va = np.zeros((n_va, len(C.EMB_COLS)), dtype=np.int64)
    vocab_info = {}
    for j, col in enumerate(C.EMB_COLS):
        s_tr = df.loc[tr, col]
        s_va = df.loc[va, col]
        assert s_tr.notna().all() and s_va.notna().all(), col
        uniq = np.sort(pd.unique(s_tr.to_numpy()))          # TRAIN ONLY
        vmap = {v: i + 1 for i, v in enumerate(uniq)}       # 0 reserved for OOV
        emb_tr[:, j] = s_tr.map(vmap).to_numpy(dtype=np.int64)
        va_idx = s_va.map(vmap)
        n_oov = int(va_idx.isna().sum())
        emb_va[:, j] = va_idx.fillna(0).to_numpy(dtype=np.int64)
        assert emb_tr[:, j].min() >= 1, col                 # no train row is OOV
        n_slots = len(uniq) + 1
        assert n_slots <= C.EMB_CARD[col] + 1, (col, n_slots)
        vocab_info[col] = {
            "n_train_levels": int(len(uniq)),
            "n_slots_incl_oov": int(n_slots),
            "global_cardinality": C.EMB_CARD[col],
            "n_valid_rows_hitting_oov": n_oov,
            "pct_valid_rows_oov": round(100.0 * n_oov / n_va, 4),
            "n_distinct_unseen_levels_in_valid":
                int(len(set(pd.unique(s_va.to_numpy())) - set(uniq.tolist()))),
        }
        if verbose:
            print(f"[prep]   vocab {col}: {len(uniq)} train levels, "
                  f"{n_oov:,} valid rows -> OOV "
                  f"({vocab_info[col]['pct_valid_rows_oov']}%)", flush=True)
    meta["vocab"] = vocab_info

    # ---------------- binaries: raw 0/1, no scaling ------------------------
    bin_tr = np.empty((n_tr, len(C.BIN_COLS)), dtype=np.float32)
    bin_va = np.empty((n_va, len(C.BIN_COLS)), dtype=np.float32)
    for j, col in enumerate(C.BIN_COLS):
        m = C.BIN_MAP[col]
        a = df.loc[tr, col].map(m)
        b = df.loc[va, col].map(m)
        assert a.notna().all() and b.notna().all(), (col, "unmapped level")
        bin_tr[:, j] = a.to_numpy(dtype=np.float32)
        bin_va[:, j] = b.to_numpy(dtype=np.float32)

    # ---------------- (c) missing flags: decided on TRAIN ONLY -------------
    raw_tr = df.loc[tr, cont_cols].to_numpy(dtype=np.float64)
    raw_va = df.loc[va, cont_cols].to_numpy(dtype=np.float64)
    na_tr = np.isnan(raw_tr)
    na_va = np.isnan(raw_va)
    flag_idx = [j for j in range(len(cont_cols)) if na_tr[:, j].any()]
    flag_cols = [f"{cont_cols[j]}_isna" for j in flag_idx]
    # A column that is NaN in valid but never in train would get filled without
    # a flag.  Refuse silently doing that: record it and fail if it happens.
    valid_only_na = [cont_cols[j] for j in range(len(cont_cols))
                     if na_va[:, j].any() and not na_tr[:, j].any()]
    assert not valid_only_na, ("column missing in valid but never in train -> "
                               "no train-derived flag exists", valid_only_na)
    meta["missing"] = {
        "flag_cols": flag_cols,
        "train_na_counts": {cont_cols[j]: int(na_tr[:, j].sum()) for j in flag_idx},
        "valid_na_counts": {cont_cols[j]: int(na_va[:, j].sum()) for j in flag_idx},
        "valid_only_na_cols": valid_only_na,
        "fill_rule": "post-transform 0.0, which is the TRAIN median by "
                     "construction of QuantileTransformer(output_distribution="
                     "'normal'): the train empirical 0.5 quantile maps to "
                     "norm.ppf(0.5) == 0.0",
    }

    # ---------------- (a) quantile transformer: FIT ON TRAIN ONLY ----------
    t0 = time.time()
    qt = QuantileTransformer(**QT_KW)
    qt.fit(raw_tr)                                   # <-- train slice only
    cont_tr = qt.transform(raw_tr).astype(np.float32)
    cont_va = qt.transform(raw_va).astype(np.float32)
    meta["qt_fit_sec"] = round(time.time() - t0, 2)

    # machine check that 0.0 really is the train median in the output space
    med = np.nanmedian(raw_tr, axis=0, keepdims=True)
    med_t = qt.transform(med)[0]
    meta["missing"]["train_median_maps_to"] = {
        "max_abs_deviation_from_zero": float(np.max(np.abs(med_t))),
        "per_col": {cont_cols[j]: float(med_t[j]) for j in flag_idx},
    }

    # fill: flags first, then the value slot
    flag_tr = na_tr[:, flag_idx].astype(np.float32)
    flag_va = na_va[:, flag_idx].astype(np.float32)
    cont_tr[na_tr] = 0.0
    cont_va[na_va] = 0.0
    assert np.isfinite(cont_tr).all() and np.isfinite(cont_va).all()

    num_tr = np.concatenate([cont_tr, flag_tr, bin_tr], axis=1)
    num_va = np.concatenate([cont_va, flag_va, bin_va], axis=1)
    num_names = cont_cols + flag_cols + C.BIN_COLS
    meta["num_cols"] = num_names
    meta["n_num"] = int(num_tr.shape[1])
    assert num_tr.shape[1] == len(num_names)

    y_tr = df.loc[tr, C.TARGET].to_numpy(dtype=np.float32)
    y_va = df.loc[va, C.TARGET].to_numpy(dtype=np.float32)
    meta["base_rate_train"] = float(y_tr.mean())
    meta["base_rate_valid"] = float(y_va.mean())

    os.makedirs(out_dir, exist_ok=True)
    npz = os.path.join(out_dir, f"fold_{valid_year}.npz")
    np.savez(npz, num_tr=num_tr, num_va=num_va, emb_tr=emb_tr, emb_va=emb_va,
             y_tr=y_tr, y_va=y_va, is_r_va=is_r_va,
             row_id_va=df.loc[va, C.ROW_ID].to_numpy().astype("U16"))
    meta["npz_sha256"] = C.sha256_file(npz)
    C.jdump(meta, os.path.join(out_dir, f"fold_{valid_year}.json"))
    if verbose:
        print(f"[prep] wrote {npz} ({os.path.getsize(npz)/1e6:.0f} MB), "
              f"D_num={num_tr.shape[1]}", flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, action="append", required=True)
    ap.add_argument("--out", default=os.path.expanduser(
        "~/SungjinPark/experiments/exp_045/cache"))
    args = ap.parse_args()

    inh = C.load_inherited()
    print("[prep] inherited:", json.dumps(
        {k: v for k, v in inh.items() if k != "ancestor_sha256"},
        ensure_ascii=False), flush=True)

    t0 = time.time()
    df = pd.read_csv(C.DATA_PATH, encoding="utf-8-sig")
    assert df.shape == (1475092, 49), df.shape
    assert df[C.ROW_ID].is_unique
    print(f"[prep] loaded {df.shape} in {time.time()-t0:.1f}s", flush=True)

    summary = {"train_csv_sha256": C.sha256_file(C.DATA_PATH),
               "src_sha256": {f: C.sha256_file(os.path.join(C.REPO_SRC, f))
                              for f in ("common.py", "prep.py")},
               "versions": {"python": sys.version.split()[0],
                            "numpy": np.__version__, "pandas": pd.__version__},
               "folds": {}}
    for y in args.fold:
        summary["folds"][f"fold_{y}"] = build(df, inh, y, args.out)
    C.jdump(summary, os.path.join(args.out, "prep_summary.json"))
    print(f"[prep] DONE {(time.time()-t0)/60:.2f} min", flush=True)


if __name__ == "__main__":
    main()
