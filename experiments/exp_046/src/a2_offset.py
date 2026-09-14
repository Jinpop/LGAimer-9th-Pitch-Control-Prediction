"""exp_046 ARM A2 — the offset form of the same in-season-form signal.

    p' = clip( p + beta_f * d_form_eb ,  eps, 1-eps )

ZERO FITS.  The base predictions are exp_030's stored C5_B3 outputs; this
script only post-processes them, exactly like the existing offset machine
(candidates F/H/I/J/K/M).

🔴 SELECTION DISCIPLINE (config pre_registered / 🔴_선택_규율)
    k and beta_f are chosen on the SUPPLY BLOCKS (2022 + 2023) ONLY.
    fold_2024 is read after they are fixed.  The cost of that honesty is
    measured against the fold-oracle choice and reported.

🔴 RULE 2-4
    d_form_eb comes from src/form_decomp.apply_decomposition, which is row-wise
    (no groupby / sort / shift / cumsum) over a table frozen on earlier seasons.
"""

import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)
import exp016_ref as ref                                            # noqa: E402
import form_decomp as fd                                            # noqa: E402

HOME = os.path.expanduser("~/SungjinPark")
DATA = os.path.join(HOME, "data/train.csv")
PRED_DIR = os.path.join(HOME, "experiments/exp_030/pred")
OUT_DIR = os.path.join(HOME, "experiments/exp_046")

BRIER_REF = 0.2498069
PTS = 100000.0 / BRIER_REF
EPS = 1e-6

K_GRID_REG = [100, 200, 400, 800, 1600]           # config stage2_arms.k_grid
K_GRID_EXT = [25, 50, 75] + K_GRID_REG            # secondary, clearly labelled
SEEDS_FOLD = [42, 7, 2024]
SIGMA_SCORE = 8.55113                             # config pre_registered
THRESHOLD_3SIGMA = 25.65339

PRED = {
    2022: {42: "pred_C5B3_u2022_seed42_st2.npy", 7: "pred_C5B3_u2022_seed7_st2.npy"},
    2023: {42: "pred_C5B3_u2023_seed42_st1.npy", 7: "pred_C5B3_u2023_seed7_st2.npy"},
    2024: {42: "pred_C5B3_u2024_seed42_st2.npy", 7: "pred_C5B3_u2024_seed7_st2.npy",
           2024: "pred_C5B3_u2024_seed2024_st2.npy"},
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


def r5(x):
    return None if x is None else round(float(x), 5)


def dscore(b_new, b_old):
    return -(b_new - b_old) * PTS


def apply_offset(p, d, beta):
    return np.clip(p + beta * np.where(np.isfinite(d), d, 0.0), EPS, 1 - EPS)


def main():
    t0 = time.time()
    M = {"exp_id": "exp_046", "arm": "A2_offset", "fits_run": 0,
         "brier_ref": BRIER_REF, "sigma_score": SIGMA_SCORE,
         "threshold_3sigma": THRESHOLD_3SIGMA,
         "sha256": {"train.csv": sha256_file(DATA),
                    "a2_offset.py": sha256_file(os.path.abspath(__file__)),
                    "form_decomp.py": sha256_file(
                        os.path.join(SRC_DIR, "form_decomp.py"))},
         "base_pred_sha256": {f"{v}_{s}": sha256_file(os.path.join(PRED_DIR, f_))
                              for v, d_ in PRED.items() for s, f_ in d_.items()}}

    df = pd.read_csv(DATA, usecols=["season", "game_type", "pitcher_id",
                                    fd.N_COL, fd.RATE_COL, fd.TARGET],
                     encoding="utf-8-sig")
    season = df["season"].to_numpy()
    y = df[fd.TARGET].to_numpy(dtype=np.float64)
    n = df[fd.N_COL].to_numpy(dtype=np.float64)
    rate = df[fd.RATE_COL].to_numpy(dtype=np.float64)
    codes, uniq = pd.factorize(df["pitcher_id"], sort=True)
    pcode = codes.astype(np.int64)
    tabs = fd.build_boundary_tables(df, pcode, len(uniq))
    zero = (np.zeros(len(uniq)), np.zeros(len(uniq)))

    d_eb, has_form = {}, None
    for k in K_GRID_EXT:
        v = np.full(len(df), np.nan)
        for s in np.sort(np.unique(season)):
            m = season == s
            t = tabs.get(int(s) - 1, zero)
            o = fd.apply_decomposition(n[m], rate[m], pcode[m], t[0], t[1], k)
            v[m] = o["d_form_eb"]
            if k == K_GRID_EXT[0]:
                if has_form is None:
                    has_form = np.zeros(len(df), dtype=np.int8)
                has_form[m] = o["has_form"]
        d_eb[k] = v

    def preds(vy):
        m = season == vy
        out = {}
        for s, f_ in PRED[vy].items():
            p = np.load(os.path.join(PRED_DIR, f_))
            assert len(p) == int(m.sum()), (vy, s)
            out[s] = p
        return m, out

    m24, p24 = preds(2024)
    y24 = y[m24]
    is_r = (df.loc[m24, "game_type"] == "R").to_numpy()
    assert int(m24.sum()) == 253507 and int(is_r.sum()) == 223497

    # ---------------- selection on the supply blocks ONLY ------------------ #
    block_rows = []
    for vy in (2022, 2023):
        m, pp = preds(vy)
        for s, p in pp.items():
            block_rows.append((vy, s, m, p))

    def fit_beta(k):
        num = den = 0.0
        for vy, s, m, p in block_rows:
            d = np.nan_to_num(d_eb[k][m])
            num += float(np.sum(d * (y[m] - p)))
            den += float(np.sum(d * d))
        return num / den if den > 0 else 0.0

    def block_score(k, beta):
        per = {}
        tot = 0.0
        for vy, s, m, p in block_rows:
            b0 = float(np.mean((p - y[m]) ** 2))
            b1 = float(np.mean((apply_offset(p, d_eb[k][m], beta) - y[m]) ** 2))
            per[f"{vy}_seed{s}"] = r5(dscore(b1, b0))
            tot += dscore(b1, b0)
        return tot / len(block_rows), per

    sel = {}
    for label, grid in (("registered", K_GRID_REG), ("extended", K_GRID_EXT)):
        tab = {}
        for k in grid:
            b = fit_beta(k)
            mean_ds, per = block_score(k, b)
            tab[str(k)] = {"beta_block": r5(b), "mean_block_dScore": r5(mean_ds),
                           "per_unit": per}
        best = max(grid, key=lambda k: tab[str(k)]["mean_block_dScore"])
        sel[label] = {"grid": grid, "table": tab, "k_selected": best,
                      "beta_selected": tab[str(best)]["beta_block"],
                      "k_at_grid_edge": bool(best in (min(grid), max(grid)))}
    M["selection_on_blocks_2022_2023"] = sel

    # ---------------- application to fold_2024 (read AFTER selection) ------ #
    def evaluate(k, beta, tag):
        rows = {}
        for s in SEEDS_FOLD:
            p0 = p24[s]
            p1 = apply_offset(p0, d_eb[k][m24], beta)
            b0 = float(np.mean((p0 - y24) ** 2))
            b1 = float(np.mean((p1 - y24) ** 2))
            rows[str(s)] = {
                "R0_valid_all": ref.seg_metrics(y24, p0),
                "A2_valid_all": ref.seg_metrics(y24, p1),
                "A2_valid_R": ref.seg_metrics(y24[is_r], p1[is_r]),
                "A2_valid_F": ref.seg_metrics(y24[~is_r], p1[~is_r]),
                "R0_valid_R": ref.seg_metrics(y24[is_r], p0[is_r]),
                "R0_valid_F": ref.seg_metrics(y24[~is_r], p0[~is_r]),
                "R0_brier_full": b0, "A2_brier_full": b1,
                "dBrier": b1 - b0, "dScore": r5(dscore(b1, b0)),
                "dScore_valid_R": r5(dscore(
                    float(np.mean((p1[is_r] - y24[is_r]) ** 2)),
                    float(np.mean((p0[is_r] - y24[is_r]) ** 2)))),
                "dScore_valid_F": r5(dscore(
                    float(np.mean((p1[~is_r] - y24[~is_r]) ** 2)),
                    float(np.mean((p0[~is_r] - y24[~is_r]) ** 2)))),
                "max_abs_shift_prob": r5(float(np.max(np.abs(p1 - p0)))),
                "mean_abs_shift_prob": r5(float(np.mean(np.abs(p1 - p0)))),
            }
        ds = [rows[str(s)]["dScore"] for s in SEEDS_FOLD]
        return {
            "tag": tag, "k": k, "beta": r5(beta), "per_seed": rows,
            "dScore_per_seed": ds,
            "mean_dScore": r5(np.mean(ds)),
            "paired_std_dScore_ddof1": r5(np.std(ds, ddof=1)),
            "sign_agreement": int(sum(1 for v in ds if v > 0)),
            "all_same_sign": bool(all(v > 0 for v in ds) or all(v < 0 for v in ds)),
            "threshold_multiple": r5(abs(np.mean(ds)) / THRESHOLD_3SIGMA),
            "ADOPTED": bool((all(v > 0 for v in ds) or all(v < 0 for v in ds))
                            and abs(np.mean(ds)) >= THRESHOLD_3SIGMA),
        }

    prim = sel["registered"]
    M["A2_primary_registered_grid"] = evaluate(
        prim["k_selected"], prim["beta_selected"], "blocks-selected, registered k grid")
    ext = sel["extended"]
    M["A2_secondary_extended_grid"] = evaluate(
        ext["k_selected"], ext["beta_selected"],
        "SECONDARY, post-hoc extended k grid (still selected on blocks only)")

    # ---------------- the price of honest selection ----------------------- #
    orc = {}
    for k in K_GRID_EXT:
        num = den = 0.0
        for s in SEEDS_FOLD:
            d = np.nan_to_num(d_eb[k][m24])
            num += float(np.sum(d * (y24 - p24[s])))
            den += float(np.sum(d * d))
        b = num / den if den > 0 else 0.0
        orc[str(k)] = evaluate(k, b, "ORACLE (k,beta from fold_2024 itself)")
    best_orc = max(K_GRID_EXT, key=lambda k: orc[str(k)]["mean_dScore"])
    M["oracle_reference"] = {
        "note": "🔴 NOT a result — the upper bound of this arm's form, used only "
                "to price the honest selection rule.",
        "best_k": best_orc, "best_beta": orc[str(best_orc)]["beta"],
        "mean_dScore": orc[str(best_orc)]["mean_dScore"],
        "per_k_mean_dScore": {k: orc[k]["mean_dScore"] for k in orc},
    }
    M["cost_of_honest_selection_pts"] = r5(
        M["A2_primary_registered_grid"]["mean_dScore"]
        - orc[str(best_orc)]["mean_dScore"])

    M["coverage_check"] = {
        "has_form_pct_fold2024_valid": r5(100.0 * float(has_form[m24].mean())),
        "rows_with_zero_correction": int((has_form[m24] == 0).sum()),
    }
    M["minutes"] = round((time.time() - t0) / 60, 3)
    with open(os.path.join(OUT_DIR, "res", "A2_offset.json"), "w") as f:
        json.dump(json.loads(json.dumps(M, default=str)), f, indent=1,
                  ensure_ascii=False)
    print("A2 PRIMARY: " + json.dumps({
        "k": M["A2_primary_registered_grid"]["k"],
        "beta": M["A2_primary_registered_grid"]["beta"],
        "dScore_per_seed": M["A2_primary_registered_grid"]["dScore_per_seed"],
        "mean_dScore": M["A2_primary_registered_grid"]["mean_dScore"],
        "threshold_multiple": M["A2_primary_registered_grid"]["threshold_multiple"],
        "ADOPTED": M["A2_primary_registered_grid"]["ADOPTED"]}), flush=True)
    print("A2 SECONDARY(ext grid): " + json.dumps({
        "k": M["A2_secondary_extended_grid"]["k"],
        "mean_dScore": M["A2_secondary_extended_grid"]["mean_dScore"],
        "ADOPTED": M["A2_secondary_extended_grid"]["ADOPTED"]}), flush=True)
    print("ORACLE(upper bound only): " + json.dumps(M["oracle_reference"]), flush=True)
    print(f"minutes {M['minutes']}", flush=True)


if __name__ == "__main__":
    main()
