"""exp_042 step 4 — promotion aggregation and the adoption gate.

Compares two arms on fold_2024 (train <= 2023 -> valid 2024), three seeds:
  ctrl : the settings in production today, re-run through b3_cat_hpo.py with an
         EMPTY override.  Its predictions are sha-compared against exp_030's
         stored predictions — an identity anchor computed from two files on
         disk, not from a literal typed here.
  hpo  : the search winner, applied to all four cells AND the stacker.

Primary metric: valid_all (R + F) Brier.  Conversion 100000*(1-brier/0.2498069).
valid_R / valid_F are recorded and NOT used to decide (the F segment pointed the
wrong way against the leaderboard three times running, 2026-08-10~11).

No post-hoc transform of any kind is applied: no offset, no gamma, no seed
averaging into the compared quantity.  Uncalibrated base against uncalibrated
base.
"""

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)
import exp016_ref as ref                                           # noqa: E402
from labels import assert_no_recovered_in_features                 # noqa: E402

DATA_PATH = os.path.expanduser("~/SungjinPark/data/train.csv")
EXP042 = os.path.expanduser("~/SungjinPark/experiments/exp_042")
EXP030_PRED = os.path.expanduser("~/SungjinPark/experiments/exp_030/pred")
REF_BRIER = 0.2498069
SEEDS = [42, 7, 2024]
# exp_039-era registered decision boundary, n = 9
SIGMA_POINTS = 11.97419
GATE_POINTS = 35.92257


def sha256_file(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def score_of(b):
    return 100000.0 * (1.0 - b / REF_BRIER)


def r5(x):
    return float(f"{x:.5f}")


def main():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    assert df["row_id"].is_unique
    base_feats = [c for c in df.columns if c not in ("row_id", "control_success")]
    assert len(base_feats) == 47, len(base_feats)
    assert_no_recovered_in_features(base_feats, "aggregate (no recovery run)")
    season = df["season"].to_numpy()
    va = season == 2024
    y = df.loc[va, "control_success"].to_numpy(dtype=np.float64)
    gt = df.loc[va, "game_type"].to_numpy()
    is_r = gt == "R"
    hand = df.loc[va, "pitcher_hand"].to_numpy()
    n_va = int(va.sum())
    del df

    out = {"exp_id": "exp_042", "phase": "promotion",
           "fold": "fold_2024 (train <= 2023 -> valid 2024)",
           "n_valid": n_va, "n_valid_R": int(is_r.sum()),
           "n_valid_F": int((~is_r).sum()),
           "valid_R_share_pct": round(100.0 * is_r.mean(), 3),
           "ref_brier": REF_BRIER, "seeds": SEEDS,
           "primary_metric": "valid_all (R+F) Brier, converted to LB points",
           "gate": {"rule": "3-seed sign agreement AND |mean delta| >= "
                            "35.92257 points (sigma 11.97419 x 3, n=9)",
                    "sigma_points": SIGMA_POINTS,
                    "threshold_points": GATE_POINTS},
           "arms": {}, "identity_anchor": {}, "iterations": {}}

    preds = {}
    for tag in ("ctrl", "hpo"):
        out["arms"][tag] = {}
        out["iterations"][tag] = {}
        for s in SEEDS:
            stem = f"C5B3_{tag}_u2024_seed{s}_st2"
            pp = os.path.join(EXP042, "pred", f"pred_{stem}.npy")
            rp = os.path.join(EXP042, "res", f"res_{stem}.json")
            p = np.load(pp)
            assert p.shape == (n_va,), (p.shape, n_va)
            r = json.load(open(rp))
            b = float(np.mean((p - y) ** 2))
            preds[(tag, s)] = p
            out["arms"][tag][str(s)] = {
                "brier": b, "score": r5(score_of(b)),
                "valid_all": ref.seg_metrics(y, p),
                "valid_R": ref.seg_metrics(y[is_r], p[is_r]),
                "valid_F": ref.seg_metrics(y[~is_r], p[~is_r]),
                "by_pitcher_hand": {str(h): ref.seg_metrics(y[hand == h],
                                                            p[hand == h])
                                    for h in sorted(pd.unique(hand).tolist())},
                "pred_mean": float(p.mean()),
                "pred_path": pp, "pred_sha256": sha256_file(pp),
                "params_override": r["params_override"],
                "minutes": r["minutes"],
            }
            out["iterations"][tag][str(s)] = {
                "cell_round": r["cell_round"], "stack_round": r["stack_round"],
                "cell_iter_used": r["cell_iter_used"],
                "stack_iter_used": r["stacker"]["iter_used"],
                "cell_argmin_diagnostic": r["cell_argmin_iter"],
                "stack_argmin_diagnostic": r["stack_argmin_iter"],
                "scale_mode": r["scale_mode"], "scale_full": r["scale_full"],
                "base_cell_argmin": r["base_cell_argmin"],
                "base_stack_argmin": r["base_stack_argmin"],
                "iter_source": r["iter_source"],
            }

    # ---------------------------------------------------------- identity anchor
    for s in SEEDS:
        mine = os.path.join(EXP042, "pred", f"pred_C5B3_ctrl_u2024_seed{s}_st2.npy")
        parent = os.path.join(EXP030_PRED, f"pred_C5B3_u2024_seed{s}_st2.npy")
        a, b_ = sha256_file(mine), sha256_file(parent)
        pa, pb = np.load(mine), np.load(parent)
        out["identity_anchor"][str(s)] = {
            "exp042_ctrl_pred": mine, "exp042_ctrl_sha256": a,
            "exp030_pred": parent, "exp030_sha256": b_,
            "sha_equal": bool(a == b_),
            "max_abs_diff": float(np.max(np.abs(pa - pb))),
        }
    out["identity_anchor"]["all_equal"] = bool(
        all(out["identity_anchor"][str(s)]["sha_equal"] for s in SEEDS))

    # ------------------------------------------------------------------ deltas
    per_seed = {}
    for s in SEEDS:
        bc = out["arms"]["ctrl"][str(s)]["brier"]
        bh = out["arms"]["hpo"][str(s)]["brier"]
        d = bh - bc
        per_seed[str(s)] = {
            "ctrl_brier": bc, "hpo_brier": bh,
            "ctrl_score": r5(score_of(bc)), "hpo_score": r5(score_of(bh)),
            "delta_brier": d, "delta_score": r5(score_of(bh) - score_of(bc)),
            "sign": int(np.sign(score_of(bh) - score_of(bc))),
            "pred_max_abs_diff_vs_ctrl": float(
                np.max(np.abs(preds[("hpo", s)] - preds[("ctrl", s)]))),
            "pred_mean_abs_diff_vs_ctrl": float(
                np.mean(np.abs(preds[("hpo", s)] - preds[("ctrl", s)]))),
        }
    ds = [per_seed[str(s)]["delta_score"] for s in SEEDS]
    mean_d = float(np.mean(ds))
    signs = {per_seed[str(s)]["sign"] for s in SEEDS}
    agree = bool(len(signs) == 1 and 0 not in signs)
    out["per_seed_delta"] = per_seed
    out["delta_summary"] = {
        "delta_score_by_seed": [r5(x) for x in ds],
        "mean_delta_score": r5(mean_d),
        "mean_delta_brier": float(np.mean(
            [per_seed[str(s)]["delta_brier"] for s in SEEDS])),
        "sd_delta_score": r5(float(np.std(ds, ddof=1))),
        "sign_agreement_3of3": agree,
        "abs_mean_over_threshold": r5(abs(mean_d) / GATE_POINTS),
        "sigma_multiple": r5(abs(mean_d) / SIGMA_POINTS),
    }
    out["adoption"] = {
        "adopted": bool(agree and abs(mean_d) >= GATE_POINTS and mean_d > 0),
        "sign_agreement": agree,
        "magnitude_pass": bool(abs(mean_d) >= GATE_POINTS),
        "direction": ("hpo better" if mean_d > 0 else
                      "ctrl better" if mean_d < 0 else "tie"),
        "threshold_points": GATE_POINTS,
        "note": ("threshold and seeds were registered in config.yaml before any "
                 "fold_2024 number existed; valid_F was not consulted"),
    }
    # mean-of-seeds arms, reported only as a descriptive companion
    for tag in ("ctrl", "hpo"):
        bs = [out["arms"][tag][str(s)]["brier"] for s in SEEDS]
        out["arms"][tag]["mean_over_seeds"] = {
            "mean_brier": float(np.mean(bs)),
            "mean_score": r5(score_of(float(np.mean(bs)))),
            "seed_sd_brier": float(np.std(bs, ddof=1)),
        }

    p = os.path.join(EXP042, "res", "promotion.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({"identity_anchor_all_equal":
                      out["identity_anchor"]["all_equal"],
                      "delta_summary": out["delta_summary"],
                      "adoption": out["adoption"]}, indent=1))
    print("WROTE " + p)


if __name__ == "__main__":
    main()
