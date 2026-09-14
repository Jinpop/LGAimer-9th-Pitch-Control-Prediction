"""exp_046 — SECONDARY, NOT A PRE-REGISTERED ARM.

Question: is the in-season-form offset orthogonal to the offset machine that is
already in the submitted candidates?  The current best submissions carry an
`S400` season/league offset (exp_033), so an A2 gain measured on the RAW C5_B3
control does not by itself say the two add up.

Method, zero fits and zero refits:
  * base   = exp_033/pred/regen_C5B3_offset_S400_self_fold2024_seed*.npy
             (C5_B3 with the existing S400_self offset at beta 0.59072)
  * apply the SAME k and beta_f that a2_offset.py selected on blocks 2022+2023
    against the RAW control.  Nothing is refitted here, so fold_2024 still
    selects nothing.

🔴 Read this as an ORTHOGONALITY PROBE, not as an adoption test.  beta_f is
not the optimum for this base, so the number below is a LOWER bound on what a
properly refitted combination would give, and it is measured on a base whose
own selection history is exp_033's, not this experiment's.
"""

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)
import exp016_ref as ref                                            # noqa: E402
import form_decomp as fd                                            # noqa: E402

HOME = os.path.expanduser("~/SungjinPark")
DATA = os.path.join(HOME, "data/train.csv")
E33 = os.path.join(HOME, "experiments/exp_033/pred")
E30 = os.path.join(HOME, "experiments/exp_030/pred")
OUT = os.path.join(HOME, "experiments/exp_046/res/A2_on_existing_offset.json")
BRIER_REF = 0.2498069
PTS = 100000.0 / BRIER_REF
EPS = 1e-6
SEEDS = [42, 7, 2024]


def sha256_file(p, b=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            c = f.read(b)
            if not c:
                break
            h.update(c)
    return h.hexdigest()


def r5(x):
    return None if x is None else round(float(x), 5)


def main():
    with open(os.path.join(HOME, "experiments/exp_046/res/A2_offset.json")) as f:
        a2 = json.load(f)
    k = int(a2["A2_primary_registered_grid"]["k"])
    beta = float(a2["A2_primary_registered_grid"]["beta"])

    df = pd.read_csv(DATA, usecols=["season", "game_type", "pitcher_id",
                                    fd.N_COL, fd.RATE_COL, fd.TARGET],
                     encoding="utf-8-sig")
    season = df["season"].to_numpy()
    y = df[fd.TARGET].to_numpy(dtype=np.float64)
    codes, uniq = pd.factorize(df["pitcher_id"], sort=True)
    pcode = codes.astype(np.int64)
    tabs = fd.build_boundary_tables(df, pcode, len(uniq))
    m = season == 2024
    o = fd.apply_decomposition(df[fd.N_COL].to_numpy(np.float64)[m],
                               df[fd.RATE_COL].to_numpy(np.float64)[m],
                               pcode[m], tabs[2023][0], tabs[2023][1], k)
    d = np.nan_to_num(o["d_form_eb"])
    y24 = y[m]
    is_r = (df.loc[m, "game_type"] == "R").to_numpy()

    M = {"exp_id": "exp_046", "probe": "A2 on top of the existing S400 offset",
         "NOT_A_PREREGISTERED_ARM": True,
         "k_form": k, "beta_f_from_blocks": beta,
         "beta_f_refit_here": False,
         "brier_ref": BRIER_REF, "fits_run": 0, "per_seed": {}}
    ds_raw, ds_off = [], []
    for s in SEEDS:
        p_raw = np.load(os.path.join(E30, f"pred_C5B3_u2024_seed{s}_st2.npy"))
        f33 = os.path.join(E33, f"regen_C5B3_offset_S400_self_fold2024_seed{s}.npy")
        p_off = np.load(f33)
        assert len(p_off) == len(y24) == len(p_raw)
        b_raw = float(np.mean((p_raw - y24) ** 2))
        b_off = float(np.mean((p_off - y24) ** 2))
        b_raw_f = float(np.mean((np.clip(p_raw + beta * d, EPS, 1 - EPS) - y24) ** 2))
        p_both = np.clip(p_off + beta * d, EPS, 1 - EPS)
        b_both = float(np.mean((p_both - y24) ** 2))
        ds_raw.append(-(b_raw_f - b_raw) * PTS)
        ds_off.append(-(b_both - b_off) * PTS)
        M["per_seed"][str(s)] = {
            "exp033_pred_sha256": sha256_file(f33),
            "brier_C5B3_raw": b_raw,
            "brier_C5B3_plus_S400": b_off,
            "brier_C5B3_plus_form": b_raw_f,
            "brier_C5B3_plus_S400_plus_form": b_both,
            "dScore_form_on_raw": r5(-(b_raw_f - b_raw) * PTS),
            "dScore_form_on_S400_base": r5(-(b_both - b_off) * PTS),
            "dScore_S400_alone": r5(-(b_off - b_raw) * PTS),
            "dScore_both_vs_raw": r5(-(b_both - b_raw) * PTS),
            "valid_all_both": ref.seg_metrics(y24, p_both),
            "valid_R_both": ref.seg_metrics(y24[is_r], p_both[is_r]),
            "valid_F_both": ref.seg_metrics(y24[~is_r], p_both[~is_r]),
        }
    M["mean_dScore_form_on_raw"] = r5(np.mean(ds_raw))
    M["mean_dScore_form_on_S400_base"] = r5(np.mean(ds_off))
    M["retention_ratio"] = r5(np.mean(ds_off) / np.mean(ds_raw)) if np.mean(ds_raw) else None
    M["sign_agreement_on_S400_base"] = int(sum(1 for v in ds_off if v > 0))
    M["interpretation_limits"] = (
        "retention_ratio near 1 means the two offsets correct nearly independent "
        "directions; near 0 means they overlap.  beta_f was NOT refitted on this "
        "base, so a refit could only raise the second number -- this is a lower "
        "bound.  No adoption decision is made from this file.")
    with open(OUT, "w") as f:
        json.dump(json.loads(json.dumps(M, default=str)), f, indent=1,
                  ensure_ascii=False)
    print(json.dumps({kk: M[kk] for kk in
                      ("mean_dScore_form_on_raw", "mean_dScore_form_on_S400_base",
                       "retention_ratio", "sign_agreement_on_S400_base")},
                     indent=1))


if __name__ == "__main__":
    main()
