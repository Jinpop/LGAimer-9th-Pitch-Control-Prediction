"""exp_016 D2 — pre-training gate on the recovered mode labels.

config.yaml D2: *verify the recovery before training anything, and stop if it
does not hold.*  Four items, all reported whatever they show:

  (1) share of rows whose cumulative-count difference falls outside {0, 1}
      (expected < 1%; rounding in the provided rate columns is the suspect)
  (2) share of rows where the recovered `is_success` equals the ground-truth
      `control_success` (expected > 99%; if this fails the whole scheme is
      wrong, not merely noisy)
  (3) MODE EXCLUSIVITY -- are there rows with is_middle + is_reverse >= 2?
      EDA §1 measured ~10% category overlap at the career level, so a
      non-exclusive labelling is a live possibility.  If the modes overlap,
      the 4-class multiclass plan (B1) is not well posed and only B2 runs.
  (4) the rate of the recovered way-off mode, against the ~0.10 residual that
      exp_014 §2-(2) recorded as "identity unexplained".  A match would
      identify that residual; a mismatch would leave it open.

Nothing here trains a model and nothing here writes a feature.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labels import RATE_COLS, recover  # noqa: E402

DATA_PATH = os.path.expanduser("~/SungjinPark/data/train.csv")
OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_016")

GATE_THRESHOLDS = {
    "max_pct_diff_outside_01": 1.0,
    "min_success_agreement_pct": 99.0,
}


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[{time.strftime('%H:%M:%S')}] loading {DATA_PATH}", flush=True)
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    print(f"loaded {df.shape}", flush=True)
    assert df["row_id"].is_unique
    assert df["season"].is_monotonic_increasing, "file order assumption broken"

    diag = recover(df)

    out = {"exp_id": "exp_016", "stage": "D2 gate", "diagnostics": diag}

    has_next = df["recov_has_next"].to_numpy()
    ok = df["recov_ok"].to_numpy()
    y = df["control_success"].to_numpy()

    # ---------------- item 1 ---------------------------------------------- #
    item1 = {
        "definition": "rows (with a successor) whose count difference is not "
                      "exactly 0 or 1, for at least one of the five modes",
        "n_rows_with_successor": int(has_next.sum()),
        "n_outside_01": int(np.sum(has_next & ~ok)),
        "pct_outside_01": diag["rows_with_diff_outside_01_any_pct"],
        "per_mode_counts": diag["rows_with_diff_outside_01_by_mode"],
        "max_abs_distance_from_integer_count": diag[
            "max_abs_distance_from_integer_count"],
        "expected": "< 1%",
        "passed": bool(diag["rows_with_diff_outside_01_any_pct"]
                       < GATE_THRESHOLDS["max_pct_diff_outside_01"]),
    }
    # what do the offending differences actually look like?
    hist = {}
    for key in RATE_COLS:
        d = df[f"draw_{key}"].to_numpy(dtype=np.float64)
        sel = has_next & ~np.isin(d, [0.0, 1.0])
        if sel.sum():
            vals, cnts = np.unique(d[sel], return_counts=True)
            order = np.argsort(-cnts)[:8]
            hist[key] = {str(vals[i]): int(cnts[i]) for i in order}
    item1["offending_difference_values_top8"] = hist
    out["gate_1_diff_outside_01"] = item1

    # ---------------- item 2 ---------------------------------------------- #
    sr = diag["success_recovery"]
    out["gate_2_success_recovery_matches_control_success"] = {
        "definition": "recovered is_success == provided control_success",
        **sr,
        "expected": "> 99%",
        "passed": bool(sr["agree_pct"] is not None
                       and sr["agree_pct"] > GATE_THRESHOLDS[
                           "min_success_agreement_pct"]),
    }

    # ---------------- item 3: exclusivity ---------------------------------- #
    sub = df.loc[ok]
    s = sub["is_success"].to_numpy()
    m = sub["is_middle"].to_numpy()
    r = sub["is_reverse"].to_numpy()
    w = sub["is_wayoff"].to_numpy()
    n_ok = int(len(sub))

    combo = {}
    for si in (0, 1):
        for mi in (0, 1):
            for ri in (0, 1):
                c = int(np.sum((s == si) & (m == mi) & (r == ri)))
                combo[f"success={si},middle={mi},reverse={ri}"] = {
                    "n": c, "pct": round(100.0 * c / n_ok, 4)}
    n_mr2 = int(np.sum((m + r) >= 2))
    n_smr2 = int(np.sum((s + m + r) >= 2))
    out["gate_3_mode_exclusivity"] = {
        "definition": "rows where two or more of {success, middle, reverse} "
                      "fire simultaneously",
        "n_recoverable_rows": n_ok,
        "n_middle_and_reverse": n_mr2,
        "pct_middle_and_reverse": round(100.0 * n_mr2 / n_ok, 4),
        "n_any_two_of_three": n_smr2,
        "pct_any_two_of_three": round(100.0 * n_smr2 / n_ok, 4),
        "joint_distribution": combo,
        "exclusive": bool(n_smr2 == 0),
        "note": "if not exclusive, the 4-class multiclass plan B1 is not well "
                "posed (config D4) and only the independent-binary plan B2 is "
                "run; B2 needs no exclusivity.",
    }

    # ---------------- item 4: way-off rate vs exp_014 residual ------------- #
    neg = int(np.sum(w < 0))
    gt1 = int(np.sum(w > 1))
    out["gate_4_wayoff_rate"] = {
        "definition": "is_wayoff = 1 - is_success - is_middle - is_reverse",
        "mean_row_level": round(float(np.mean(w)), 6),
        "n_negative": neg,
        "pct_negative": round(100.0 * neg / n_ok, 4),
        "n_above_one": gt1,
        "exp_014_career_residual_reference": 1.0 - 0.894,
        "exp_014_note": "exp_014 measured career-level "
                        "success+reverse+middle = 0.894 (0.900 for >=500-pitch "
                        "pitchers); its residual was recorded as UNIDENTIFIED.",
        "base_rates": {
            "success": round(float(np.mean(s)), 6),
            "middle": round(float(np.mean(m)), 6),
            "reverse": round(float(np.mean(r)), 6),
            "wayoff": round(float(np.mean(w)), 6),
            "ball": round(float(sub["is_ball"].mean()), 6),
            "strike": round(float(sub["is_strike"].mean()), 6),
        },
        "control_success_base_rate_all_rows": round(float(np.mean(y)), 6),
    }

    # ---- does this identify exp_014's unexplained residual? --------------- #
    # exp_014 §2-(2) reported career-level success+reverse+middle = 0.894 and
    # recorded the 0.106 remainder as UNIDENTIFIED, explicitly listing
    # "category overlap (EDA ~10%)" as one of the possibilities it could not
    # rule out.  With row-level labels the identity becomes exact arithmetic:
    #     1 - (S + M + R) = P(way-off) - P(middle AND reverse)
    # because every overlap row is counted twice on the left.
    p_wayoff_true = float(np.mean(sub["is_wayoff_true"].to_numpy()))
    p_both = float(np.mean(sub["is_both"].to_numpy()))
    out["residual_identity_check"] = {
        "row_level_1_minus_S_M_R": round(float(np.mean(w)), 6),
        "P_wayoff_true": round(p_wayoff_true, 6),
        "P_middle_AND_reverse": round(p_both, 6),
        "P_wayoff_true_minus_P_both": round(p_wayoff_true - p_both, 6),
        "identity_holds": bool(abs((p_wayoff_true - p_both)
                                   - float(np.mean(w))) < 1e-9),
        "exp_014_career_level_residual": 0.106,
        "note": "row-weighted here vs pitcher-weighted in exp_014, so the two "
                "numbers are not required to be identical.",
    }
    # pitcher-weighted version, to line up with exp_014's own averaging
    last = df.loc[df["recov_has_next"] == False]  # noqa: E712  last row/pitcher
    car = df.groupby("pitcher_id", sort=False).tail(1)
    n_car = car["asof_pitcher_n"].to_numpy(dtype=np.float64)
    smr = (car["asof_pitcher_success_rate"].to_numpy(dtype=np.float64)
           + car["asof_pitcher_middle_rate"].to_numpy(dtype=np.float64)
           + car["asof_pitcher_reverse_rate"].to_numpy(dtype=np.float64))
    sel500 = n_car >= 500
    out["residual_identity_check"]["career_S_M_R_all_pitchers"] = round(
        float(np.nanmean(smr)), 4)
    out["residual_identity_check"]["career_S_M_R_n_ge_500"] = round(
        float(np.nanmean(smr[sel500])), 4)
    out["residual_identity_check"]["n_pitchers_ge_500"] = int(sel500.sum())
    del last

    # ---- five-cell partition ---------------------------------------------- #
    out["five_cell_partition"] = {
        c: {"n": int(sub[c].sum()), "rate": round(float(sub[c].mean()), 6)}
        for c in ["is_success", "is_mid_only", "is_rev_only", "is_both",
                  "is_wayoff_true"]
    }
    out["five_cell_partition"]["_note"] = (
        "mutually exclusive and exhaustive (asserted in labels.recover); this "
        "is the partition B1/B2 are actually built on, replacing the 4-class "
        "form in config D4 that the exclusivity gate ruled out.")

    # ---- how many rows leak a VALID outcome through the recovery? --------- #
    # is_X[t] is built from row t+1.  If row t is the last training row of a
    # pitcher and row t+1 falls in the validation season, then that training
    # LABEL encodes a validation row's outcome.  Those rows must be dropped
    # from mode-model training.  Count them per fold boundary here.
    succ_season = df.groupby("pitcher_id", sort=False)["season"].shift(-1)
    out["fold_boundary_rows_to_drop_from_mode_training"] = {
        f"train_until_{tu}": int(((df["season"] <= tu)
                                  & (succ_season > tu)).sum())
        for tu in (2021, 2022, 2023)
    }

    # per-season, because the F-league label regime changed in 2023 (EDA §1-4)
    per_season = {}
    for (season, gt), grp in sub.groupby(["season", "game_type"], observed=True):
        per_season[f"{int(season)}_{gt}"] = {
            "n": int(len(grp)),
            "success": round(float(grp["is_success"].mean()), 4),
            "middle": round(float(grp["is_middle"].mean()), 4),
            "reverse": round(float(grp["is_reverse"].mean()), 4),
            "wayoff": round(float(grp["is_wayoff"].mean()), 4),
        }
    out["mode_base_rate_by_season_and_game_type"] = per_season

    # ---------------- usable-row accounting -------------------------------- #
    out["row_accounting"] = {
        "total": int(len(df)),
        "no_successor_last_row_per_pitcher": int((~has_next).sum()),
        "successor_but_diff_outside_01": int(np.sum(has_next & ~ok)),
        "recoverable": n_ok,
        "recoverable_pct": round(100.0 * n_ok / len(df), 4),
    }

    passed = (out["gate_1_diff_outside_01"]["passed"]
              and out["gate_2_success_recovery_matches_control_success"]["passed"])
    out["GATE_PASSED_items_1_2"] = bool(passed)
    out["B1_multiclass_well_posed"] = bool(out["gate_3_mode_exclusivity"]["exclusive"])
    out["elapsed_min"] = round((time.time() - t0) / 60, 2)

    with open(os.path.join(OUT_DIR, "gate.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in out.items()
                      if k != "mode_base_rate_by_season_and_game_type"},
                     indent=2, ensure_ascii=False), flush=True)
    print(f"GATE items 1&2 passed: {passed}; B1 well posed: "
          f"{out['B1_multiclass_well_posed']}", flush=True)
    if not passed:
        sys.exit(3)


if __name__ == "__main__":
    main()
