"""exp_048 — how much of exp_046's `d_form_eb` is the same quantity as ins_*.

Both are "this pitcher's season so far minus his prior baseline", built from
the same raw material (`asof_pitcher_n` x `asof_pitcher_success_rate`, minus a
frozen train-side boundary table).  This measures the overlap on the fold_2024
VALIDATION rows: correlation, coverage, and where the two disagree.

exp_046's module is imported from its own directory and its sha is recorded --
no copy is made, so there is nothing to drift.
"""

import hashlib
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.join(SRC_DIR, "ins"))
import inseason as ins                                              # noqa: E402
from ins_feats import COLS16                                        # noqa: E402

DATA_PATH = os.path.expanduser("~/SungjinPark/data/train.csv")
PACK_DIR = os.path.join(SRC_DIR, "ins")
FD_PATH = os.path.expanduser(
    "~/SungjinPark/experiments/exp_046/src/form_decomp.py")
OUT = os.path.expanduser("~/SungjinPark/experiments/exp_048/res")
KS = [100, 400]


def sha256_file(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def r5(x):
    return None if x is None or not np.isfinite(x) else float(f"{x:.5f}")


def main():
    os.makedirs(OUT, exist_ok=True)
    fd = load_mod(FD_PATH, "form_decomp")
    res = {"exp_id": "exp_048", "what": "ins_* vs exp_046 d_form_eb overlap",
           "sha256": {"form_decomp.py": sha256_file(FD_PATH),
                      "inseason.py": sha256_file(
                          os.path.join(PACK_DIR, "inseason.py"))}}

    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    season = df["season"].to_numpy()
    codes, uniq = pd.factorize(df["pitcher_id"], sort=True)
    pcode = codes.astype(np.int64)
    npit = int(len(uniq))

    # exp_046's boundary tables, built on TRAIN ONLY (<=2023 for this fold)
    tr = season <= 2023
    tables = fd.build_boundary_tables(df.loc[tr].reset_index(drop=True),
                                      pcode[tr], npit)
    n_end, s_end = tables[2023]

    va = season == 2024
    n = df.loc[va, "asof_pitcher_n"].to_numpy(np.float64)
    rate = df.loc[va, "asof_pitcher_success_rate"].to_numpy(np.float64)
    pc = pcode[va]
    y = df.loc[va, "control_success"].to_numpy(np.float64)

    A = ins.load(os.path.join(PACK_DIR, "anchor_lt2024.npz"))
    I = ins.apply(df.loc[va], A).to_numpy(np.float64)
    assert I.shape[1] == 16

    res["n_valid_rows"] = int(va.sum())
    res["per_k"] = {}
    for k in KS:
        o = fd.apply_decomposition(n, rate, pc, n_end, s_end, k)
        d = o["d_form_eb"]
        hf = np.isfinite(d)
        row = {"k": k,
               "d_form_eb_defined_pct": r5(100.0 * float(hf.mean())),
               "d_form_eb_std": r5(float(np.nanstd(d))),
               "corr_with_each_ins_col": {}}
        for j, c in enumerate(COLS16):
            v = I[:, j]
            m = hf & np.isfinite(v)
            row["corr_with_each_ins_col"][c] = (
                r5(float(np.corrcoef(d[m], v[m])[0, 1]))
                if m.sum() > 10 and np.std(v[m]) > 0 else None)
        # the headline pair: exp_046's d_form_eb vs the pack's success deviation
        j = COLS16.index("ins_p_success_dev")
        v = I[:, j]
        m = hf & np.isfinite(v)
        row["headline_vs_ins_p_success_dev"] = {
            "pearson_r": r5(float(np.corrcoef(d[m], v[m])[0, 1])),
            "r_squared": r5(float(np.corrcoef(d[m], v[m])[0, 1] ** 2)),
            "ols_slope_ins_on_dform": r5(float(
                np.polyfit(d[m], v[m], 1)[0])),
            "n_rows_both_defined": int(m.sum()),
        }
        # coverage disagreement: where does one exist and the other not
        npn = I[:, COLS16.index("ins_p_n")]
        row["coverage"] = {
            "d_form_eb_defined_pct": r5(100.0 * float(hf.mean())),
            "ins_p_n_gt0_pct": r5(100.0 * float((npn > 0).mean())),
            "d_defined_but_ins_zero_pct": r5(
                100.0 * float((hf & (npn <= 0)).mean())),
            "ins_positive_but_d_undefined_pct": r5(
                100.0 * float((~hf & (npn > 0)).mean())),
        }
        # each side's own single-column AUC on the same rows
        from sklearn.metrics import roc_auc_score
        row["single_column_auc_on_common_rows"] = {
            "d_form_eb": r5(float(roc_auc_score(y[m], d[m]))),
            "ins_p_success_dev": r5(float(roc_auc_score(y[m], v[m]))),
        }
        res["per_k"][str(k)] = row
        print(f"  k={k}: r(d_form_eb, ins_p_success_dev) = "
              f"{row['headline_vs_ins_p_success_dev']['pearson_r']}, "
              f"defined {row['d_form_eb_defined_pct']}%", flush=True)

    p = os.path.join(OUT, "stage1_corr_dform.json")
    with open(p, "w") as f:
        json.dump(res, f, indent=1)
    print("CORR done -> " + p, flush=True)


if __name__ == "__main__":
    main()
