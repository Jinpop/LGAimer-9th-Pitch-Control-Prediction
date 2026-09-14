"""exp_045 — the actual deliverable: gap / correlation / convexity vs CatBoost.

    python analyze.py --arch A0

Nothing here fits anything.  It reads the NN prediction vectors written by
train_nn.py and the frozen CatBoost vectors from exp_030 / exp_031, and
produces config.ensemble_measurement items (1)-(4).

ROW ALIGNMENT.  Both sides are the rows of data/train.csv with
`season == VALID_YEAR`, taken in file order:
  * CatBoost (exp_033/src/grid_c5.py) slices with `m24 = (season == 2024)`.
  * NN (prep.py) slices with the exp_004 valid mask `season == valid_year`.
Identical construction, so the vectors are aligned by definition.  That is
asserted, not assumed: every CatBoost file is checked by sha256 against the
value exp_033 recorded, and its Brier is recomputed against the labels the NN
cache carries and compared to exp_033's recorded Brier to ~1e-12.  A row
permutation would move that Brier.

DEPLOYMENT EPOCH.  Fixed policy, stated before any fold_2024 number is read:
    deploy_epoch = median over seeds of the fold_2023 patience-5 best epoch.
The fold_2024 number at that epoch is the honest one.  The fold_2024 number at
fold_2024's own argmin is fold-oracle and is labelled as such (config do_not #2,
candidate O's defect).
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

ROOT = os.path.expanduser("~/SungjinPark/experiments/exp_045")
EXPS = os.path.expanduser("~/SungjinPark/experiments")
SEEDS = [42, 7, 2024]

# frozen CatBoost artefacts, pinned by the sha256 exp_033 recorded for them
CAT = {
    "fold_2024_offset_S400_self": {   # exp_033 selected candidate line
        "42": (f"{EXPS}/exp_031/pred/pred_C5B3_offset_S400_self_fold2024_seed42.npy",
               "a05fe17d7fc922cbbdaf589f9e890a9e4f108094e8c1f32248b4c31b8b014af0",
               0.24794831900950637),
        "7": (f"{EXPS}/exp_031/pred/pred_C5B3_offset_S400_self_fold2024_seed7.npy",
              "7afc77a0220fcc77e193ff8cc3f622b08df8d7415b3a87ba28cc7d74a7f61131",
              0.24788834471528198),
        "2024": (f"{EXPS}/exp_031/pred/pred_C5B3_offset_S400_self_fold2024_seed2024.npy",
                 "68796ccfb2d68daf1a7ba19f7c2f74bf9ed5cff37ed13cbd50e4c199a50fb026",
                 0.24786391750337364)},
    "fold_2024_base_no_offset": {     # exp_030 stage 2, the C5_B3 base
        "42": (f"{EXPS}/exp_030/pred/pred_C5B3_u2024_seed42_st2.npy",
               "ef4bafad23f53798b60e221b497e448c70321ef14e0618233930ff54893d54bf",
               0.2480393607892575),
        "7": (f"{EXPS}/exp_030/pred/pred_C5B3_u2024_seed7_st2.npy",
              "d3c606b0433768250219de958864054294d0098fb724dc7377224b1b92153668",
              0.2479833583326968),
        "2024": (f"{EXPS}/exp_030/pred/pred_C5B3_u2024_seed2024_st2.npy",
                 "aa999bb9eee3c419a4239338330e086b72826e57c9cdf03e231768cc89ea330d",
                 0.2479648004372976)},
    "fold_2023_base_no_offset": {     # exp_031 stage 3 (own early stop) blocks
        "42": (f"{EXPS}/exp_031/pred/pred_C5B3_u2023_seed42_st3.npy",
               "bd8c17ec35e0b9c86a291d361d5fe94b5f7605bf13eff9ece15e86859298ae56",
               0.24867962923638576),
        "7": (f"{EXPS}/exp_031/pred/pred_C5B3_u2023_seed7_st3.npy",
              "a25e0733fcbab2422496f1e8cd7d8b8378ad2754ca510fd1b551a3df797936c1",
              None)},
}
W_GRID = np.round(np.linspace(0.0, 1.0, 21), 3)


def brier(y, p):
    return float(np.mean((np.asarray(p, dtype=np.float64) - y) ** 2))


def logit(p, eps=1e-6):
    q = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(q / (1.0 - q))


def load_cat(group, y, n_expect):
    out, prov = {}, {}
    for s, (path, sha, ref_b) in CAT[group].items():
        assert os.path.exists(path), path
        got = C.sha256_file(path)
        assert got == sha, ("frozen CatBoost artefact changed", path, got, sha)
        v = np.load(path).astype(np.float64)
        assert v.shape == (n_expect,), (path, v.shape, n_expect)
        b = brier(y, v)
        rec = {"path": path, "sha256": got, "brier_recomputed": b,
               "brier_recorded_by_exp033": ref_b}
        if ref_b is not None:
            rec["abs_diff"] = abs(b - ref_b)
            # ROW-ALIGNMENT PROOF: any permutation of the rows moves this.
            assert rec["abs_diff"] < 1e-12, ("row alignment / provenance "
                                             "mismatch", path, b, ref_b)
            rec["alignment_check"] = "PASS (<1e-12 vs exp_033 record)"
        else:
            rec["alignment_check"] = ("no exp_033-recorded Brier for this file; "
                                      "sha256 pin only")
        out[s], prov[s] = v, rec
    return out, prov


def w_unconstrained(y, p_nn, p_cat):
    """Exact vertex of the (always convex) quadratic Brier(w), w unrestricted.

    B(w) = mean((p_cat - y + w*(p_nn - p_cat))**2), so
    w_opt = -mean((p_cat - y) * d) / mean(d**2)   with d = p_nn - p_cat.
    A NEGATIVE w_opt means the best thing to do with the NN vector is to
    subtract it: it carries no usable additional signal at all.
    """
    d = p_nn - p_cat
    denom = float(np.mean(d * d))
    if denom <= 0:
        return None
    w = -float(np.mean((p_cat - y) * d)) / denom
    b = brier(y, p_cat + w * d)
    return {"w_opt_unconstrained": w, "brier_at_w_opt": b,
            "score_at_w_opt": round(C.score_of(b), 5),
            "gain_over_cat_only_score_points":
                round(C.score_delta(b - brier(y, p_cat)), 5),
            "mean_abs_disagreement": float(np.mean(np.abs(d)))}


def mix_curve_logit(y, p_nn, p_cat):
    """DIAGNOSTIC ONLY.  config specifies probability-space mixing; this is the
    same 21-point sweep done in logit space, to check that the probability-space
    verdict is not an artefact of the mixing geometry."""
    ln, lc = logit(p_nn), logit(p_cat)
    bs = [brier(y, 1.0 / (1.0 + np.exp(-(w * ln + (1 - w) * lc)))) for w in W_GRID]
    j = int(np.argmin(bs))
    return {"w_star": float(W_GRID[j]), "brier_at_w_star": bs[j],
            "score_at_w_star": round(C.score_of(bs[j]), 5),
            "gain_over_cat_only_score_points":
                round(C.score_delta(bs[j] - bs[0]), 5),
            "brier": bs}


def mix_curve(y, p_nn, p_cat):
    """Brier over w in W_GRID for w*NN + (1-w)*Cat, plus the convexity term."""
    bs = [brier(y, w * p_nn + (1.0 - w) * p_cat) for w in W_GRID]
    j = int(np.argmin(bs))
    w_star = float(W_GRID[j])
    b_min, b_nn, b_cat = bs[j], bs[-1], bs[0]
    b_lin = w_star * b_nn + (1.0 - w_star) * b_cat
    return {"w_grid": [float(w) for w in W_GRID],
            "brier": bs,
            "score": [round(C.score_of(b), 5) for b in bs],
            "w_star": w_star,
            "brier_at_w_star": b_min,
            "score_at_w_star": round(C.score_of(b_min), 5),
            "brier_cat_only_w0": b_cat, "score_cat_only_w0": round(C.score_of(b_cat), 5),
            "brier_nn_only_w1": b_nn, "score_nn_only_w1": round(C.score_of(b_nn), 5),
            "linear_interp_at_w_star_brier": b_lin,
            "convexity_brier": b_lin - b_min,
            "convexity_score_points": round(C.score_delta(b_min - b_lin), 5),
            "gain_over_cat_only_score_points": round(C.score_delta(b_min - b_cat), 5)}


def mean_of(d):
    return float(np.mean(list(d.values())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out-name", default="ensemble_measurement.json")
    args = ap.parse_args()
    root, arch = args.root, args.arch
    out = {"exp_id": "exp_045", "arch": arch, "seeds": SEEDS,
           "ref_brier": C.REF_BRIER,
           "conversion": "score = 100000 * (1 - Brier / 0.2498069); a score "
                         "converted from a 6-decimal Brier carries +-0.20 of "
                         "uncertainty (1e-6 Brier == 0.4003 score points)"}

    cache = {}
    for fy in (2023, 2024):
        z = np.load(os.path.join(root, "cache", f"fold_{fy}.npz"), allow_pickle=False)
        cache[fy] = {"y": z["y_va"].astype(np.float64), "is_r": z["is_r_va"],
                     "row_id": z["row_id_va"]}
    assert len(cache[2024]["y"]) == 253507
    assert int(cache[2024]["is_r"].sum()) == 223497

    # ---------------- NN side ---------------------------------------------
    nn, res = {}, {}
    for fy in (2023, 2024):
        nn[fy], res[fy] = {}, {}
        for s in SEEDS:
            tag = f"{arch}_fold{fy}_seed{s}"
            r = json.load(open(os.path.join(root, "res", f"res_{tag}.json")))
            res[fy][s] = r
            nn[fy][s] = np.load(os.path.join(root, "pred", f"pred_{tag}.npy")
                                ).astype(np.float64)
            assert len(nn[fy][s]) == len(cache[fy]["y"])

    # deployment epoch: fixed policy, fold_2023 only
    e23 = [res[2023][s]["early_stopping"]["best_epoch"] for s in SEEDS]
    deploy_epoch = int(np.median(e23))
    out["deployment_epoch"] = {
        "rule": "median over seeds of the fold_2023 patience-5 best epoch; "
                "fold_2024 is never consulted for this",
        "fold_2023_best_epochs_by_seed": {str(s): e for s, e in zip(SEEDS, e23)},
        "deploy_epoch": deploy_epoch}

    nn24_deploy = {}
    for s in SEEDS:
        allp = np.load(os.path.join(root, "pred", f"all_{arch}_fold2024_seed{s}.npy"))
        nn24_deploy[s] = allp[deploy_epoch - 1].astype(np.float64)

    def nn_block(fy, preds, label):
        y = cache[fy]["y"]; isr = cache[fy]["is_r"]
        bys = {str(s): brier(y, preds[s]) for s in SEEDS}
        m = float(np.mean(list(bys.values())))
        sd = float(np.std(list(bys.values()), ddof=1))
        return {"label": label,
                "brier_by_seed": bys,
                "score_by_seed": {k: round(C.score_of(v), 5) for k, v in bys.items()},
                "mean_brier": m, "mean_score": round(C.score_of(m), 5),
                "seed_std_brier_ddof1": sd,
                "seed_std_score_points": round(abs(C.score_delta(sd)), 5),
                "valid_all": C.seg_metrics(y, np.mean([preds[s] for s in SEEDS], axis=0)),
                "valid_R": C.seg_metrics(y[isr], np.mean([preds[s] for s in SEEDS], axis=0)[isr]),
                "valid_F": C.seg_metrics(y[~isr], np.mean([preds[s] for s in SEEDS], axis=0)[~isr])}

    out["nn"] = {
        "fold_2023_own_es": nn_block(2023, nn[2023],
                                     "fold_2023, patience-5 best on fold_2023 valid"),
        "fold_2024_own_es_FOLD_ORACLE": nn_block(
            2024, nn[2024],
            "fold_2024 at its OWN patience-5 best epoch -- fold-oracle, "
            "NOT a deployment number"),
        "fold_2024_deploy_epoch_HONEST": nn_block(
            2024, nn24_deploy,
            f"fold_2024 at epoch {deploy_epoch}, derived from fold_2023 only "
            "-- this is the honest number"),
        "epoch_curves_fold_2024": {str(s): res[2024][s]["valid_brier_curve"]
                                   for s in SEEDS},
        "epoch_curves_fold_2023": {str(s): res[2023][s]["valid_brier_curve"]
                                   for s in SEEDS},
        "arch_cfg": res[2024][SEEDS[0]]["arch_cfg"],
        "n_params": res[2024][SEEDS[0]]["n_params"],
        "fit_time_min_by_tag": {f"fold{fy}_seed{s}": res[fy][s]["fit_time_min"]
                                for fy in (2023, 2024) for s in SEEDS},
    }
    # 🔴 NN's OWN seed sigma (config: do not reuse CatBoost's 8.55113)
    out["nn_sigma"] = {
        "note": "NN seed-to-seed std, measured here.  CatBoost's sigma "
                "(8.55113 score points, exp_044) must NOT be substituted.",
        "fold_2024_honest_sigma_score_points":
            out["nn"]["fold_2024_deploy_epoch_HONEST"]["seed_std_score_points"],
        "fold_2024_own_es_sigma_score_points":
            out["nn"]["fold_2024_own_es_FOLD_ORACLE"]["seed_std_score_points"],
        "fold_2023_sigma_score_points":
            out["nn"]["fold_2023_own_es"]["seed_std_score_points"],
        "catboost_sigma_for_contrast_score_points": 8.55113}

    # ---------------- CatBoost side + alignment ---------------------------
    y24, y23 = cache[2024]["y"], cache[2023]["y"]
    cat24o, prov1 = load_cat("fold_2024_offset_S400_self", y24, len(y24))
    cat24b, prov2 = load_cat("fold_2024_base_no_offset", y24, len(y24))
    cat23b, prov3 = load_cat("fold_2023_base_no_offset", y23, len(y23))
    out["catboost_provenance"] = {"fold_2024_offset_S400_self": prov1,
                                  "fold_2024_base_no_offset": prov2,
                                  "fold_2023_base_no_offset": prov3}
    out["row_alignment"] = {
        "construction": "both sides slice train.csv by season == valid_year in "
                        "file order (exp_004 valid mask == grid_c5.py m24)",
        "verified_by": "sha256 pin on every frozen vector + Brier recomputed "
                       "against the NN cache labels, matched to <1e-12 against "
                       "the value exp_033 recorded",
        "reindex_needed": False,
        "n_valid_2024": int(len(y24)), "n_valid_2023": int(len(y23))}

    def cat_block(y, d, label):
        bys = {s: brier(y, v) for s, v in d.items()}
        m = float(np.mean(list(bys.values())))
        sd = (float(np.std(list(bys.values()), ddof=1))
              if len(bys) > 1 else None)
        return {"label": label, "n_seeds": len(bys),
                "brier_by_seed": bys,
                "score_by_seed": {k: round(C.score_of(v), 5) for k, v in bys.items()},
                "mean_brier": m, "mean_score": round(C.score_of(m), 5),
                "seed_std_score_points": (round(abs(C.score_delta(sd)), 5)
                                          if sd is not None else None)}

    out["catboost"] = {
        "fold_2024_offset_S400_self": cat_block(y24, cat24o,
                                                "exp_033 candidate line (offset applied)"),
        "fold_2024_base_no_offset": cat_block(y24, cat24b, "exp_030 C5_B3 base"),
        "fold_2023_base_no_offset": cat_block(y23, cat23b,
                                              "exp_031 block stage 3, seeds 42/7 only")}

    # ---------------- (1) gap ---------------------------------------------
    def gap_vs(cat_key, nn_key):
        nnm = out["nn"][nn_key]["mean_score"]
        cm = out["catboost"][cat_key]["mean_score"]
        return {"nn_mean_score": nnm, "cat_mean_score": cm,
                "gap_nn_minus_cat_score_points": round(nnm - cm, 5),
                "abs_gap_score_points": round(abs(nnm - cm), 5),
                "nn_mean_brier": out["nn"][nn_key]["mean_brier"],
                "cat_mean_brier": out["catboost"][cat_key]["mean_brier"]}

    out["measure_1_gap"] = {
        "definition": "config: NN 3-seed mean score - CatBoost 3-seed mean "
                      "score on fold_2024 valid_all.  The pre-registered "
                      "readings are stated in absolute value.",
        "PRIMARY_honest_vs_offset_candidate":
            gap_vs("fold_2024_offset_S400_self", "fold_2024_deploy_epoch_HONEST"),
        "honest_vs_base_no_offset":
            gap_vs("fold_2024_base_no_offset", "fold_2024_deploy_epoch_HONEST"),
        "fold_oracle_vs_offset_candidate_DIAGNOSTIC":
            gap_vs("fold_2024_offset_S400_self", "fold_2024_own_es_FOLD_ORACLE"),
        "fold_oracle_vs_base_DIAGNOSTIC":
            gap_vs("fold_2024_base_no_offset", "fold_2024_own_es_FOLD_ORACLE")}

    # ---------------- (2) correlation -------------------------------------
    def rho_block(nn_d, cat_d, seeds):
        pm_nn = np.mean([nn_d[s] for s in seeds], axis=0)
        pm_ct = np.mean([cat_d[str(s)] if str(s) in cat_d else cat_d[s]
                         for s in seeds], axis=0)
        per = {}
        for s in seeds:
            c = cat_d[str(s)] if str(s) in cat_d else cat_d[s]
            per[str(s)] = {"prob": float(np.corrcoef(nn_d[s], c)[0, 1]),
                           "logit": float(np.corrcoef(logit(nn_d[s]), logit(c))[0, 1])}
        return {"seed_matched": per,
                "seed_mean_vectors": {
                    "prob": float(np.corrcoef(pm_nn, pm_ct)[0, 1]),
                    "logit": float(np.corrcoef(logit(pm_nn), logit(pm_ct))[0, 1])},
                "mean_of_seed_matched": {
                    "prob": float(np.mean([v["prob"] for v in per.values()])),
                    "logit": float(np.mean([v["logit"] for v in per.values()]))}}

    out["measure_2_rho"] = {
        "definition": "Pearson between the two prediction vectors on fold_2024 "
                      "valid_all, in probability space and in logit space",
        "vs_offset_candidate": rho_block(nn24_deploy, cat24o, SEEDS),
        "vs_base_no_offset": rho_block(nn24_deploy, cat24b, SEEDS),
        "fold_oracle_nn_vs_offset_DIAGNOSTIC": rho_block(nn[2024], cat24o, SEEDS),
        "teammate_measured_reference": {"LGB_vs_Cat": 0.9489, "NN_vs_Cat": 0.8653},
        "nn_internal_seed_rho_prob": float(np.mean(
            [np.corrcoef(nn24_deploy[a], nn24_deploy[b])[0, 1]
             for a in SEEDS for b in SEEDS if a < b]))}

    # ---------------- (3) convexity ---------------------------------------
    seed_matched = {str(s): mix_curve(y24, nn24_deploy[s], cat24o[str(s)])
                    for s in SEEDS}
    sm_mean_curve = np.mean([seed_matched[str(s)]["brier"] for s in SEEDS], axis=0)
    j = int(np.argmin(sm_mean_curve))
    w_star_sm = float(W_GRID[j])
    b_lin_sm = w_star_sm * sm_mean_curve[-1] + (1 - w_star_sm) * sm_mean_curve[0]
    pm_nn = np.mean([nn24_deploy[s] for s in SEEDS], axis=0)
    pm_ct = np.mean([cat24o[str(s)] for s in SEEDS], axis=0)

    out["measure_3_convexity"] = {
        "definition": "w in {0, 0.05, ..., 1}; mix = w*NN + (1-w)*Cat in "
                      "PROBABILITY space.  convexity = (linear interpolation of "
                      "the two endpoints evaluated at w*) - (actual Brier at "
                      "w*), converted to score points.",
        "PRIMARY_seed_matched": {
            "note": "mix seed s with seed s, then average the 21-point Brier "
                    "curve over seeds.  Keeps the comparison on the same "
                    "footing as measure_1 (which is a mean of per-seed scores) "
                    "and isolates the mixing gain from the seed-averaging gain.",
            "per_seed": seed_matched,
            "mean_curve_brier": [float(v) for v in sm_mean_curve],
            "mean_curve_score": [round(C.score_of(v), 5) for v in sm_mean_curve],
            "w_star": w_star_sm,
            "brier_at_w_star": float(sm_mean_curve[j]),
            "score_at_w_star": round(C.score_of(sm_mean_curve[j]), 5),
            "convexity_brier": float(b_lin_sm - sm_mean_curve[j]),
            "convexity_score_points": round(
                C.score_delta(sm_mean_curve[j] - b_lin_sm), 5),
            "gain_over_cat_only_score_points": round(
                C.score_delta(sm_mean_curve[j] - sm_mean_curve[0]), 5)},
        "SECONDARY_seed_mean_vectors": mix_curve(y24, pm_nn, pm_ct),
        "unconstrained_vertex_seed_matched": {
            str(s): w_unconstrained(y24, nn24_deploy[s], cat24o[str(s)])
            for s in SEEDS},
        "unconstrained_vertex_vs_base_no_offset": {
            str(s): w_unconstrained(y24, nn24_deploy[s], cat24b[str(s)])
            for s in SEEDS},
        "logit_space_DIAGNOSTIC_seed_matched": {
            str(s): mix_curve_logit(y24, nn24_deploy[s], cat24o[str(s)])
            for s in SEEDS},
        "vs_base_no_offset_seed_matched_mean_w_star": None,
        "UPPER_BOUND_WARNING": "the w* here is chosen ON fold_2024.  This is an "
                               "upper bound, not a deployment forecast "
                               "(config ensemble_measurement.upper_bound_warning). "
                               "measure_4 is the honest estimate."}
    # same curve against the un-offset base, for the record
    sm_base = {str(s): mix_curve(y24, nn24_deploy[s], cat24b[str(s)]) for s in SEEDS}
    cb = np.mean([sm_base[str(s)]["brier"] for s in SEEDS], axis=0)
    jb = int(np.argmin(cb))
    wb = float(W_GRID[jb])
    lb = wb * cb[-1] + (1 - wb) * cb[0]
    out["measure_3_convexity"]["vs_base_no_offset_seed_matched_mean_w_star"] = {
        "w_star": wb, "brier_at_w_star": float(cb[jb]),
        "score_at_w_star": round(C.score_of(cb[jb]), 5),
        "convexity_score_points": round(C.score_delta(cb[jb] - lb), 5),
        "gain_over_cat_only_score_points": round(C.score_delta(cb[jb] - cb[0]), 5),
        "mean_curve_brier": [float(v) for v in cb]}

    # ---------------- (4) honest transfer: w* from fold_2023 --------------
    s23 = [42, 7]                       # CatBoost fold_2023 has these two seeds
    c23 = np.mean([cat23b[str(s)] for s in s23], axis=0)
    n23 = np.mean([nn[2023][s] for s in s23], axis=0)
    cur23 = mix_curve(y23, n23, c23)
    w23 = cur23["w_star"]
    # apply that w to fold_2024, seed-matched, on both CatBoost variants
    def apply_w(w, cat_d):
        bs = [brier(y24, w * nn24_deploy[s] + (1 - w) * cat_d[str(s)]) for s in SEEDS]
        m = float(np.mean(bs))
        b0 = float(np.mean([brier(y24, cat_d[str(s)]) for s in SEEDS]))
        return {"w_applied": w, "brier_by_seed": bs, "mean_brier": m,
                "mean_score": round(C.score_of(m), 5),
                "cat_only_mean_brier": b0,
                "cat_only_mean_score": round(C.score_of(b0), 5),
                "delta_vs_cat_only_score_points": round(C.score_delta(m - b0), 5)}

    out["measure_4_honest_transfer"] = {
        "definition": "fit w on fold_2023 (argmin of the 21-point curve), then "
                      "score fold_2024 with that w.  THIS is the estimate of "
                      "what deployment would actually get.",
        "fold_2023_curve": cur23,
        "fold_2023_seeds_used": s23,
        "fold_2023_unconstrained_vertex": w_unconstrained(y23, n23, c23),
        "fold_2023_gap_score_points": {
            "nn_mean_score": round(C.score_of(
                float(np.mean([brier(y23, nn[2023][s]) for s in s23]))), 5),
            "cat_mean_score": round(C.score_of(
                float(np.mean([brier(y23, cat23b[str(s)]) for s in s23]))), 5)},
        "fold_2023_caveat": "the CatBoost fold_2023 vectors are the un-offset "
                            "C5_B3 blocks (exp_031 stage 3) and cover seeds "
                            "42/7 only, while the fold_2024 candidate line has "
                            "the S400_self offset applied.  The un-offset "
                            "fold_2024 transfer below is the like-for-like one.",
        "applied_to_fold_2024_offset_candidate": apply_w(w23, cat24o),
        "applied_to_fold_2024_base_no_offset_LIKE_FOR_LIKE": apply_w(w23, cat24b),
        "for_contrast_fold_2024_oracle_w":
            out["measure_3_convexity"]["PRIMARY_seed_matched"]["w_star"]}

    out["src_sha256"] = {f: C.sha256_file(os.path.join(C.REPO_SRC, f))
                         for f in ("common.py", "prep.py", "train_nn.py",
                                   "analyze.py")}
    C.jdump(out, os.path.join(root, args.out_name))
    print(json.dumps({
        "gap_primary": out["measure_1_gap"]["PRIMARY_honest_vs_offset_candidate"],
        "rho": out["measure_2_rho"]["vs_offset_candidate"]["mean_of_seed_matched"],
        "convexity": {k: out["measure_3_convexity"]["PRIMARY_seed_matched"][k]
                      for k in ("w_star", "convexity_score_points",
                                "gain_over_cat_only_score_points")},
        "honest_transfer": out["measure_4_honest_transfer"][
            "applied_to_fold_2024_offset_candidate"],
        "nn_sigma": out["nn_sigma"],
    }, indent=1, default=C._default), flush=True)


if __name__ == "__main__":
    main()
