"""exp_046 — SECONDARY PROBE, NOT A PRE-REGISTERED ARM.

Question: does the tree arm A1 already extract everything the offset arm A2
extracts?  If the 51-column model has fully absorbed d_form_eb, then adding the
A2 offset on top of A1's predictions should do nothing.

Zero fits.  beta_f and k are the SAME values a2_offset.py selected on supply
blocks 2022+2023 against the RAW control -- they are NOT refitted on A1's
predictions (no A1 block predictions exist) and they are NOT selected on
fold_2024.  So the number below is a LOWER bound on the combination, and it is
a probe, not an adoption test.
"""

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
E46 = os.path.join(HOME, "experiments/exp_046")
DATA = os.path.join(HOME, "data/train.csv")
PTS = 100000.0 / 0.2498069
EPS = 1e-6
SEEDS = [42, 7, 2024]


def r5(x):
    return round(float(x), 5)


def main():
    with open(os.path.join(E46, "res/A2_offset.json")) as f:
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

    M = {"exp_id": "exp_046", "probe": "A2 offset on top of A1 tree arm",
         "NOT_A_PREREGISTERED_ARM": True, "fits_run": 0,
         "k_form": k, "beta_f_from_blocks_on_RAW_control": beta,
         "beta_f_refit_on_A1": False, "per_seed": {}}
    inc, tot = [], []
    for s in SEEDS:
        p1 = np.load(os.path.join(E46, f"pred/pred_C5B3form_u2024_seed{s}_st2.npy"))
        p0 = np.load(os.path.join(HOME, "experiments/exp_030/pred",
                                 f"pred_C5B3_u2024_seed{s}_st2.npy"))
        assert len(p1) == len(y24)
        b_r0 = float(np.mean((p0 - y24) ** 2))
        b_a1 = float(np.mean((p1 - y24) ** 2))
        pc = np.clip(p1 + beta * d, EPS, 1 - EPS)
        b_c = float(np.mean((pc - y24) ** 2))
        inc.append(-(b_c - b_a1) * PTS)
        tot.append(-(b_c - b_r0) * PTS)
        M["per_seed"][str(s)] = {
            "brier_R0": b_r0, "brier_A1": b_a1, "brier_A1_plus_A2": b_c,
            "dScore_A1_vs_R0": r5(-(b_a1 - b_r0) * PTS),
            "dScore_A2_added_on_A1": r5(-(b_c - b_a1) * PTS),
            "dScore_A1_plus_A2_vs_R0": r5(-(b_c - b_r0) * PTS),
            "corr_A1_resid_with_d_form_eb": r5(
                float(np.corrcoef(y24 - p1, d)[0, 1])),
            "corr_R0_resid_with_d_form_eb": r5(
                float(np.corrcoef(y24 - p0, d)[0, 1])),
            "valid_all": ref.seg_metrics(y24, pc),
            "valid_R": ref.seg_metrics(y24[is_r], pc[is_r]),
            "valid_F": ref.seg_metrics(y24[~is_r], pc[~is_r]),
        }
    M["mean_dScore_A2_added_on_A1"] = r5(np.mean(inc))
    M["mean_dScore_A1_plus_A2_vs_R0"] = r5(np.mean(tot))
    M["sign_agreement_added"] = int(sum(1 for v in inc if v > 0))
    M["reading"] = (
        "A large residual increment means the tree did NOT absorb the signal "
        "even though it was handed the column; a near-zero increment means it "
        "did.  beta_f is not the optimum for this base, so the increment is a "
        "lower bound.")
    with open(os.path.join(E46, "res/A1_plus_A2_probe.json"), "w") as f:
        json.dump(json.loads(json.dumps(M, default=str)), f, indent=1,
                  ensure_ascii=False)
    print(json.dumps({kk: M[kk] for kk in
                      ("mean_dScore_A2_added_on_A1",
                       "mean_dScore_A1_plus_A2_vs_R0",
                       "sign_agreement_added")}, indent=1))
    for s in SEEDS:
        v = M["per_seed"][str(s)]
        print(s, "A1vsR0", v["dScore_A1_vs_R0"], "| +A2", v["dScore_A2_added_on_A1"],
              "| total", v["dScore_A1_plus_A2_vs_R0"],
              "| resid-corr R0", v["corr_R0_resid_with_d_form_eb"],
              "A1", v["corr_A1_resid_with_d_form_eb"])


if __name__ == "__main__":
    main()
