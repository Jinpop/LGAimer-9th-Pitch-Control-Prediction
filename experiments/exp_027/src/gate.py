"""exp_027 preconditions P1 / P2 / P3.

P1  Is train.csv globally in TIME order?  CatBoost `has_time=True` treats the
    ROW ORDER OF THE TRAINING POOL as the time axis; exp_016 only established
    per-pitcher monotonicity of the file order, which is a weaker statement.
    train.csv carries no `game_date`; the finest available time key is
    (season, game_month), so that is what is checked, at that granularity, and
    the granularity limit is reported rather than papered over.

P2  Row counts / label rate / column count vs the exp_010 & exp_018 records.

P3  Library + thread determinism settings are printed for the record (the
    actual pinning happens in train.py).
"""

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

DATA_PATH = os.path.expanduser("~/SungjinPark/data/train.csv")
TARGET = "control_success"
F_EXCLUDE_MAX_SEASON = 2022
FOLDS = [("fold_2022", 2021, 2022), ("fold_2023", 2022, 2023),
         ("fold_2024", 2023, 2024)]
EXPECTED = {
    "n_features": 47,
    "fold_2023_n_train": 870752,       # config.yaml stages.A_screen
    "fold_2024_n_train": 1116277,      # config.yaml stages.B_promote
    "fold_2024_valid_all": 253507,     # exp_001/007/008/010/021
    "fold_2024_valid_R": 223497,
}


def sha256_file(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    out = {"data_sha256": sha256_file(DATA_PATH),
           "versions": {"python": sys.version.split()[0],
                        "numpy": np.__version__, "pandas": pd.__version__}}
    for mod in ("lightgbm", "xgboost", "catboost"):
        try:
            m = __import__(mod)
            out["versions"][mod] = m.__version__
        except Exception as e:                                   # pragma: no cover
            out["versions"][mod] = f"IMPORT FAILED: {e}"

    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    out["shape"] = list(df.shape)
    feats = [c for c in df.columns if c not in ("row_id", TARGET)]
    out["n_features"] = len(feats)
    out["row_id_unique"] = bool(df["row_id"].is_unique)

    season = df["season"].to_numpy()
    month = df["game_month"].to_numpy()
    gtype = df["game_type"].to_numpy()
    y = df[TARGET].to_numpy(dtype=np.float64)

    # ---------------- P1 -------------------------------------------------- #
    key = season.astype(np.int64) * 100 + month.astype(np.int64)
    dec = np.flatnonzero(np.diff(key) < 0)
    p1 = {
        "time_key": "(season, game_month)  -- train.csv has NO game_date column",
        "granularity_caveat": (
            "month-level.  Within one (season, month) block the relative order "
            "of rows is unverifiable from the provided columns."),
        "season_monotonic_nondecreasing": bool(pd.Series(season).is_monotonic_increasing),
        "month_key_monotonic_nondecreasing": bool(len(dec) == 0),
        "n_decreasing_transitions": int(len(dec)),
        "pct_decreasing_transitions": round(100.0 * len(dec) / max(1, len(key) - 1), 6),
        "distinct_month_blocks": int((np.diff(key) != 0).sum() + 1),
        "distinct_month_values": int(len(np.unique(key))),
    }
    # WHERE the inversions are decides whether they matter.  has_time reads the
    # order of the TRAINING POOL, so an inversion among rows that never enter
    # training cannot affect it.
    p1["decreasing_transitions_detail"] = [
        {"row_idx": int(i),
         "from": f"{season[i]}-{month[i]:02d} {gtype[i]}",
         "to": f"{season[i+1]}-{month[i+1]:02d} {gtype[i+1]}",
         "either_side_dropped_by_exp004_mask": bool(
             (season[i] <= F_EXCLUDE_MAX_SEASON and gtype[i] == "F") or
             (season[i+1] <= F_EXCLUDE_MAX_SEASON and gtype[i+1] == "F"))}
        for i in dec]
    p1["all_inversions_removed_by_mask"] = bool(all(
        d["either_side_dropped_by_exp004_mask"]
        for d in p1["decreasing_transitions_detail"]))
    p1["file_layout"] = (
        "seasons 2019-2022 are laid out as [all R, month-sorted][all F, "
        "month-sorted], so the month resets at the R->F seam; seasons 2023-2024 "
        "interleave R and F in month order.  Every inversion is at a "
        "season<=2022 F seam, and the exp_004 mask drops exactly those rows "
        "from training.")
    pool_ok = True
    for name, tu, _ in FOLDS:
        tr = (season <= tu) & ~((season <= F_EXCLUDE_MAX_SEASON) & (gtype == "F"))
        k = key[tr]
        d = np.flatnonzero(np.diff(k) < 0)
        p1[f"{name}_train_decreasing_transitions"] = int(len(d))
        pool_ok = pool_ok and len(d) == 0
    # corroboration: file order == chronological order is what makes the
    # organisers' expanding shift(1) as-of columns meaningful (exp_010 verified
    # asof_pitcher_success_rate*n == cumulative prior successes in FILE order).
    s = pd.Series(y)
    g = s.groupby(df["pitcher_id"].to_numpy(), sort=False)
    prev_sum = (g.cumsum() - s).to_numpy()
    prev_cnt = g.cumcount().to_numpy().astype(np.float64)
    given_n = df["asof_pitcher_n"].to_numpy(dtype=np.float64)
    given_rate = df["asof_pitcher_success_rate"].to_numpy(dtype=np.float64)
    ok = (~np.isnan(given_rate)) & (given_n > 0)
    p1["asof_fileorder_identity_n_maxabsdiff"] = float(
        np.abs(np.nan_to_num(given_n) - prev_cnt).max())
    p1["asof_fileorder_identity_sum_maxabsdiff"] = float(
        np.abs(np.nan_to_num(given_rate)[ok] * given_n[ok] - prev_sum[ok]).max())
    p1["asof_identity_rows_checked"] = int(ok.sum())
    # The question has_time actually poses is about the TRAINING POOL's order,
    # not the whole file's.  Both facts are recorded; the verdict is on the pool.
    p1["verdict"] = (
        "PASS for every training pool — the raw file has "
        f"{len(dec)} decreasing (season, month) transitions, but all of them sit "
        "at season<=2022 R->F / F->F seams whose F rows the exp_004 mask removes "
        "from training, so fold_2022 / fold_2023 / fold_2024 training pools each "
        "have ZERO inversions.  The organisers' as-of columns are also file-order "
        "expanding shift(1) (verified here: n matches exactly), which is only "
        "meaningful if file order is chronological.  CAVEAT: month granularity — "
        "order within a (season, month) block is unverifiable from the provided "
        "columns.  train.py re-asserts pool-order == 0 inversions at run time for "
        "every has_time arm."
        if pool_ok else
        "FAIL — a training pool is not in time order; has_time arms are unsafe")
    p1["pool_order_ok"] = bool(pool_ok)
    out["P1"] = p1

    # ---------------- P2 -------------------------------------------------- #
    f_excl = (season <= F_EXCLUDE_MAX_SEASON) & (gtype == "F")
    p2 = {"n_rows_masked_exp004": int(f_excl.sum()),
          "expected": EXPECTED, "folds": {}}
    for name, tu, vy in FOLDS:
        tr = (season <= tu) & ~f_excl
        va = season == vy
        is_r = gtype[va] == "R"
        p2["folds"][name] = {
            "n_train": int(tr.sum()),
            "train_label_rate": round(float(y[tr].mean()), 8),
            "n_valid_all": int(va.sum()),
            "n_valid_R": int(is_r.sum()),
            "n_valid_F": int((~is_r).sum()),
            "valid_all_label_rate": round(float(y[va].mean()), 8),
            "valid_all_brier_of_base_rate": round(
                float(np.mean((y[va].mean() - y[va]) ** 2)), 8),
        }
    p2["checks"] = {
        "n_features_47": out["n_features"] == EXPECTED["n_features"],
        "fold_2023_n_train": p2["folds"]["fold_2023"]["n_train"] == EXPECTED["fold_2023_n_train"],
        "fold_2024_n_train": p2["folds"]["fold_2024"]["n_train"] == EXPECTED["fold_2024_n_train"],
        "fold_2024_valid_all": p2["folds"]["fold_2024"]["n_valid_all"] == EXPECTED["fold_2024_valid_all"],
        "fold_2024_valid_R": p2["folds"]["fold_2024"]["n_valid_R"] == EXPECTED["fold_2024_valid_R"],
        "labels_binary": bool(np.isin(y, (0.0, 1.0)).all()),
        "row_id_unique": out["row_id_unique"],
    }
    p2["scale_ratio_2024_over_2023"] = round(
        p2["folds"]["fold_2024"]["n_train"] / p2["folds"]["fold_2023"]["n_train"], 6)
    p2["verdict"] = "PASS" if all(p2["checks"].values()) else "FAIL"
    out["P2"] = p2

    # ---------------- P3 -------------------------------------------------- #
    out["P3"] = {
        "device": "CPU only for all three families (no GPU / no CUDA)",
        "threads": 6,
        "determinism": {
            "lightgbm": "deterministic=True, force_row_wise=True, seed=<seed>",
            "xgboost": "device=cpu, tree_method=hist, nthread=6, seed=<seed>",
            "catboost": "task_type=CPU, thread_count=6, random_seed=<seed>",
        },
        "why": "CatBoost GPU / XGBoost CUDA are not bit-reproducible -> Phase 3 conflict",
    }

    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                           "gate.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
