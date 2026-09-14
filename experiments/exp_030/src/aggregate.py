"""exp_030 aggregation — runs LOCALLY on the recovered prediction vectors.

Does four things and nothing else:

  1. re-derives the fold_2024 validation mask and labels from the local
     train.csv (sha-gated against the server's copy) — no number below is
     copied from a report;
  2. re-measures the LGBM_B3 control (exp_024 arm base, NO offset) from
     exp_024/pred/*.npy after checking each file's sha256 against the
     provenance block of exp_024/metrics.json, and asserts the re-measured
     Brier equals the `base_no_offset` value that file records;
  3. measures C5_B3 (this experiment) on the same mask for the three seeds;
  4. applies the adoption rule — 3 seeds same sign AND |mean delta| >= 3 x
     the CONTROL's seed std — and reports the paired std alongside it.

Everything is reported as absolute Brier plus the converted LB score
100000 * (1 - Brier / 0.2498069).
"""

import hashlib
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss

_HERE = os.path.dirname(os.path.abspath(__file__))            # .../exp_030/src
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))  # project root
DATA = os.path.join(ROOT, "data", "train.csv")
E024 = os.path.join(ROOT, "experiments", "exp_024")
E030 = os.path.join(ROOT, "experiments", "exp_030")
DATA_SHA = "d2081186b458b49f60b082be480c273135833e15ba59a76d033af28bcf8763ff"
REF_BRIER = 0.2498069
SEEDS = [42, 7, 2024]

C5_FLAT = {"mean_brier": 0.2481012381, "mean_score": 682.79214,
           "source": "exp_027 stage_B.arms.C5.mean_brier"}
L1_FLAT = {"mean_brier": 0.2482572037, "mean_score": 620.35768,
           "source": "exp_027 stage_B.arms.L1.mean_brier"}


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


def seg(y, p):
    b = float(np.mean((p - y) ** 2))
    r = float(np.mean(y))
    return {"n": int(len(y)), "brier": b, "score": r5(score_of(b)),
            "bss": float(1.0 - b / (r * (1 - r))),
            "auc": float(roc_auc_score(y, p)),
            "logloss": float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15))),
            "base_rate": r, "pred_mean": float(np.mean(p))}


def main():
    assert sha256_file(DATA) == DATA_SHA, "local train.csv differs from server"
    df = pd.read_csv(DATA, encoding="utf-8-sig",
                     usecols=["season", "game_type", "pitcher_hand",
                              "control_success"])
    season = df["season"].to_numpy()
    va = season == 2024
    y = df.loc[va, "control_success"].to_numpy(dtype=np.float64)
    is_r = df.loc[va, "game_type"].to_numpy() == "R"
    hand = df.loc[va, "pitcher_hand"].to_numpy()
    assert int(va.sum()) == 253507 and int(is_r.sum()) == 223497

    m024 = json.load(open(os.path.join(E024, "metrics.json")))
    out = {"exp_id": "exp_030", "reference_brier": REF_BRIER,
           "conversion": "score = 100000 * (1 - brier / 0.2498069)",
           "n_valid_all": int(va.sum()), "n_valid_R": int(is_r.sum()),
           "n_valid_F": int((~is_r).sum())}

    # ---------------- control: LGBM_B3, no offset, from exp_024 ------------- #
    ctrl = {}
    for s in SEEDS:
        f = os.path.join(E024, "pred", f"pred_B3_block_2024_seed{s}.npy")
        got = sha256_file(f)
        prov = m024["base_vector_provenance"][str(s)]["fold_2024"]
        assert got == prov["sha256"], (f, got, prov["sha256"])
        p = np.load(f)
        assert p.shape == (int(va.sum()),), p.shape
        recorded = m024["per_seed"][str(s)]["base_no_offset"]["fold_2024_brier"]
        d = seg(y, p)
        assert abs(d["brier"] - recorded) < 1e-12, (d["brier"], recorded)
        ctrl[s] = {**d, "pred_sha256": got, "recorded_in_exp024": recorded,
                   "valid_R": seg(y[is_r], p[is_r]),
                   "valid_F": seg(y[~is_r], p[~is_r]),
                   "by_pitcher_hand": {str(h): seg(y[hand == h], p[hand == h])
                                       for h in sorted(pd.unique(hand).tolist())}}
    out["LGBM_B3_control"] = {
        "what": "exp_024 arm base B3 (LightGBM), NO offset — re-measured here "
                "from the stored vectors, sha-checked, and asserted equal to "
                "exp_024 metrics.json per_seed[*].base_no_offset.fold_2024_brier",
        "per_seed": ctrl}

    # ---------------- candidate: C5_B3 ------------------------------------- #
    cand = {}
    for s in SEEDS:
        f = os.path.join(E030, "pred", f"pred_C5B3_u2024_seed{s}_st2.npy")
        p = np.load(f)
        assert p.shape == (int(va.sum()),), p.shape
        r = json.load(open(os.path.join(
            E030, "res", f"res_C5B3_u2024_seed{s}_st2.json")))
        assert abs(float(np.mean((p - y) ** 2)) - r["brier_full"]) < 1e-12
        cand[s] = {**seg(y, p), "pred_sha256": sha256_file(f),
                   "iter_used_stacker": r["stacker"]["iter_used"],
                   "cell_iter_used": r["cell_iter_used"],
                   "curve_argmin_is_interior": r["stacker"]["curve_argmin_is_interior"],
                   "diagnostic_own_argmin": r["stacker"]["argmin_iter"],
                   "minutes": r["minutes"],
                   "valid_R": seg(y[is_r], p[is_r]),
                   "valid_F": seg(y[~is_r], p[~is_r]),
                   "by_pitcher_hand": {str(h): seg(y[hand == h], p[hand == h])
                                       for h in sorted(pd.unique(hand).tolist())},
                   "cells": {c: r["cells"][c]["own_mode_at_used"]
                             for c in r["cells"]}}
    out["C5_B3"] = {"per_seed": cand}

    # ---------------- adoption test ---------------------------------------- #
    cb_ = np.array([ctrl[s]["brier"] for s in SEEDS])
    kb_ = np.array([cand[s]["brier"] for s in SEEDS])
    paired = kb_ - cb_
    ctrl_std = float(np.std(cb_, ddof=1))
    thr = 3.0 * ctrl_std
    mean_d = float(np.mean(paired))
    out["decision"] = {
        "rule": "3 seeds same sign AND |mean delta| >= 3 x CONTROL seed std",
        "seeds": SEEDS,
        "control_brier_by_seed": [float(x) for x in cb_],
        "candidate_brier_by_seed": [float(x) for x in kb_],
        "control_mean_brier": float(np.mean(cb_)),
        "control_mean_score": r5(score_of(float(np.mean(cb_)))),
        "candidate_mean_brier": float(np.mean(kb_)),
        "candidate_mean_score": r5(score_of(float(np.mean(kb_)))),
        "paired_delta_by_seed": [float(x) for x in paired],
        "paired_delta_score_by_seed": [r5(-1e5 * x / REF_BRIER) for x in paired],
        "mean_delta_brier": mean_d,
        "mean_delta_score_points": r5(-1e5 * mean_d / REF_BRIER),
        "control_seed_std_DENOMINATOR": ctrl_std,
        "candidate_seed_std": float(np.std(kb_, ddof=1)),
        "paired_delta_seed_std_REFERENCE_ONLY": float(np.std(paired, ddof=1)),
        "threshold_abs_mean_delta": thr,
        "abs_mean_delta_over_control_std": abs(mean_d) / ctrl_std if ctrl_std else None,
        "signs_agree": bool(np.all(paired < 0) or np.all(paired > 0)),
        "passes_magnitude": bool(abs(mean_d) >= thr),
        "ADOPTED": bool((np.all(paired < 0) or np.all(paired > 0))
                        and abs(mean_d) >= thr and mean_d < 0),
        "note": "negative delta = C5_B3 better.  ADOPTED requires the sign to "
                "be an improvement as well as consistent.",
    }

    # ---------------- secondary rulers ------------------------------------- #
    km, cm = float(np.mean(kb_)), float(np.mean(cb_))
    out["rulers"] = {
        "C5_flat_exp027": C5_FLAT, "L1_flat_exp027": L1_FLAT,
        "LGBM_B3_mean": {"brier": cm, "score": r5(score_of(cm))},
        "C5_B3_mean": {"brier": km, "score": r5(score_of(km))},
        "stacking_gain_on_C5": {
            "brier": km - C5_FLAT["mean_brier"],
            "score_points": r5(-1e5 * (km - C5_FLAT["mean_brier"]) / REF_BRIER),
            "meaning": "C5_B3 - C5_flat: what mode-stacking adds ON TOP of C5"},
        "stacking_gain_on_L1": {
            "brier": cm - L1_FLAT["mean_brier"],
            "score_points": r5(-1e5 * (cm - L1_FLAT["mean_brier"]) / REF_BRIER),
            "meaning": "LGBM_B3 - L1_flat: what mode-stacking added on LightGBM"},
        "flat_family_gain": {
            "brier": C5_FLAT["mean_brier"] - L1_FLAT["mean_brier"],
            "score_points": r5(-1e5 * (C5_FLAT["mean_brier"]
                                       - L1_FLAT["mean_brier"]) / REF_BRIER),
            "meaning": "C5_flat - L1_flat: exp_027's measured family gain, flat"},
        "family_gain_under_stacking": {
            "brier": km - cm,
            "score_points": r5(-1e5 * (km - cm) / REF_BRIER),
            "meaning": "C5_B3 - LGBM_B3: THE HEADLINE — what the base swap "
                       "leaves once the stack is on top of it"},
    }

    with open(os.path.join(E030, "aggregate.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["decision"], indent=1))
    print(json.dumps(out["rulers"], indent=1))


if __name__ == "__main__":
    main()
