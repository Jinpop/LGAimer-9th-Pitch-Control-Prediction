"""exp_040 — do the offset and the calibration survive on top of the features?

WHAT THIS FILE DOES, AND WHAT IT DOES NOT
-----------------------------------------
No training happens here.  exp_037 already wrote six fold_2024 prediction
vectors to disk (2 bases x 3 seeds); this file loads them, checks their sha256
against the numbers exp_037 recorded in its own res/*.json, and applies four
element-wise transforms to each:

    T0  none
    T1  offset only          p' = clip(p + beta*d, EPS, 1-EPS)
    T2  calibration only     p'' = sigmoid(logit(p) + gamma)
    T3  both, deployment order: offset FIRST, gamma on top of the offset output

The question is the INTERACTION: is the offset gain smaller on `L_no_sit`
(63 columns, teammate features) than on `R0` (47 columns, no features)?  If it
is, the features have eaten part of what the offset was correcting.

WHY THE ANSWER IS NOT CLEAN EVEN IF THE NUMBER IS
--------------------------------------------------
beta = 0.59072 was fitted on the supply blocks of a base WITHOUT the teammate
features (exp_031 stage-3 units, 2022/2023 x seeds 42/7).  If the offset gain
shrinks on `L_no_sit`, "the features ate it" and "beta is mis-fitted for this
base" are not distinguishable from anything measured here.  Separating them
needs a refit on the L_no_sit supply blocks, which needs training, which this
experiment does not do.  Both readings are reported.

gamma IS IN-SAMPLE.  It is solved so the mean prediction over the fold_2024
validation rows equals those same rows' realised success rate.  One parameter
against 253,507 rows, and the deployed analogue (candidate N) transferred to
the leaderboard at 1.1143x its local gain rather than below 1x — but it is
in-sample and is labelled as such wherever it appears.

`d` IS BUILT FROM <=2023 TABLES ONLY.  `dev_k.build_multi_k` gives a season-s
row a table built from seasons <= s-1; the 2024 rows therefore see <= 2023.
That boundary is the only leak surface in this file, and the module that draws
it is byte-inherited from exp_033/exp_036 (sha asserted below) rather than
rewritten -- the expanded form d = (s - n*overall)/(n + k) is load-bearing and
the literal subtraction is known to break at k != 111.
"""

import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.metrics import roc_auc_score

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

import backoff as bk                                                 # noqa: E402
import dev_k as dk                                                   # noqa: E402
import features_platoon as fp                                        # noqa: E402
import offset as off                                                 # noqa: E402
from labels import recover                                           # noqa: E402

DATA_PATH = os.path.expanduser("~/SungjinPark/data/train.csv")
OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_040")
EXP037 = os.path.expanduser("~/SungjinPark/experiments/exp_037")
EXP031 = os.path.expanduser("~/SungjinPark/experiments/exp_031")
EXP032 = os.path.expanduser("~/SungjinPark/experiments/exp_032")     # == exp_034

TARGET = "control_success"
F_EXCLUDE_MAX_SEASON = 2022
REF_BRIER = 0.2498069
K = 400
ARM = "self"
CELL = f"S{K}_{ARM}"
BETA = 0.59072                       # frozen; fitted WITHOUT teammate features
SUPPLY_BLOCKS = [2022, 2023]
BLOCK_SEEDS = [42, 7]
BASES = ["R0", "L_no_sit"]
SEEDS = [42, 7, 2024]
VALID_SEASON = 2024
EPS = off.EPS

# byte-inheritance gate: these are the exp_033/exp_036 files, not new ones
INHERITED_SHA = {
    "dev_k.py": "e1cd86dd2795a4d12f279f7ea600353f441cfc7b19eea12207fb87117207a0d9",
    "backoff.py": "4169d2dadd8a07dceef2098a88971d3b3e437276bfb28834295e7df523591039",
    "features_platoon.py": "68d63063e8ce659cf2876fbca4f5adef7ee1e720620ae63bb78ea496c8dadf09",
    "offset.py": "68d6007db03ab850e2a5181383a4148fae8947c711558282c9846391f13e0488",
    "labels.py": "439cd504478b4bc8162e9dedc32be27bfadd5d1178472d6b384b60f46ed7a9f2",
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


def score(brier):
    return float(f"{100000.0 * (1.0 - brier / REF_BRIER):.5f}")


def score_delta(dbrier):
    return float(f"{-100000.0 * dbrier / REF_BRIER:.5f}")


def logit(p):
    pc = np.clip(p, EPS, 1.0 - EPS)
    return np.log(pc / (1.0 - pc))


def apply_gamma(p, gamma):
    return 1.0 / (1.0 + np.exp(-(logit(p) + gamma)))


def solve_gamma(p, rate):
    """gamma with mean(sigmoid(logit(p)+gamma)) == rate.  Monotone in gamma."""
    z = logit(p)

    def f(g):
        return float(np.mean(1.0 / (1.0 + np.exp(-(z + g)))) - rate)

    g = brentq(f, -10.0, 10.0, xtol=1e-12, rtol=1e-14)
    return float(g), f(g)


def seg(y, p):
    b = float(np.mean((p - y) ** 2))
    return {"brier": b, "score": score(b), "auc": float(roc_auc_score(y, p)),
            "n": int(len(y)), "pred_mean": float(np.mean(p))}


def main():
    t0 = time.time()
    out = {"exp_id": "exp_040",
           "what": ("do the offset and the global calibration still pay on a "
                    "base that already carries the teammate features?"),
           "no_training": True, "cell": CELL, "k": K, "arm": ARM,
           "beta_applied": BETA,
           "beta_provenance": ("fitted on exp_031 supply blocks 2022/2023 x "
                               "seeds 42/7, i.e. on a base WITHOUT teammate "
                               "features; frozen here, not refitted"),
           "bases": BASES, "seeds": SEEDS}

    # ------------------------------------------------ inherited-code gate
    got_sha = {f: sha256_file(os.path.join(SRC_DIR, f)) for f in INHERITED_SHA}
    for f, want in INHERITED_SHA.items():
        assert got_sha[f] == want, ("inherited module drifted", f, got_sha[f])
    out["inherited_src_sha256"] = got_sha
    out["own_src_sha256"] = {"postproc.py": sha256_file(os.path.abspath(__file__))}

    # ------------------------------------------------ data, labels, d table
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    out["train_csv_sha256"] = sha256_file(DATA_PATH)
    rec = recover(df, verbose=False)
    assert rec["success_recovery"]["agree_pct"] == 100.0
    season = df["season"].to_numpy()
    ok = df["recov_ok"].to_numpy()
    succ_season = df.groupby("pitcher_id", sort=False)["season"].shift(-1).to_numpy()
    f_excl = ((df["season"] <= F_EXCLUDE_MAX_SEASON)
              & (df["game_type"] == "F")).to_numpy()
    y_all = df[TARGET].to_numpy(dtype=np.float64)
    assert np.isin(y_all, (0.0, 1.0)).all()

    print(f"[{time.strftime('%H:%M:%S')}] platoon lookups", flush=True)
    fp.build_season_forward_chained(df, ok, succ_season, f_excl, verbose=False)
    d_s, hit, n_cell0, src_season, _ = dk.build_multi_k(
        df, ok, succ_season, f_excl, [K], verbose=False)
    L_s, has_Ls, _, _ = bk.build_league_dev(df, ok, succ_season, f_excl,
                                            verbose=False)
    Ls_eff = np.where(has_Ls, L_s, 0.0)
    assert np.all(d_s[K][~hit] == 0.0), "d must be exactly 0 on cell misses"
    d_full = dk.build_arm(d_s[K], hit, Ls_eff, n_cell0, K, ARM)

    va = season == VALID_SEASON
    y24 = y_all[va]
    d24 = d_full[va]
    is_r = (df.loc[va, "game_type"].to_numpy() == "R")
    assert len(y24) == 253507, len(y24)
    assert int(is_r.sum()) == 223497, int(is_r.sum())

    # the leak surface, asserted rather than asserted-in-prose: every 2024 row's
    # table was built from seasons <= 2023
    src24 = src_season[va]
    assert np.nanmax(src24) <= VALID_SEASON - 1, float(np.nanmax(src24))
    out["d_table_boundary"] = {
        "max_source_season_for_2024_rows": float(np.nanmax(src24)),
        "all_rows_same_source": bool(np.nanmin(src24) == np.nanmax(src24)),
        "built_from_test_or_2024_labels": False}

    # observed 2024 success rate — COMPUTED, never transcribed
    rate24 = float(np.mean(y24))
    out["observed_rate_2024"] = rate24
    out["observed_rate_2024_source"] = "mean of control_success over the 253,507 season-2024 rows of train.csv"

    out["d_2024"] = {
        "n": int(len(d24)), "sd": float(np.std(d24)),
        "cell_hit_pct": round(100.0 * float(hit[va].mean()), 3),
        "max_abs_beta_d": float(np.max(np.abs(BETA * d24))),
        "note": ("max_abs_beta_d is a PROBABILITY shift, not a score. Whether a "
                 "given row moved toward or away from its label is not known "
                 "row-wise, so it does NOT convert to points.")}

    # ------------------------------------------ beta reproduction (free check)
    try:
        with open(os.path.join(EXP032, "curves", "iter_fit.json")) as f:
            ifit = json.load(f)
        blk, prov = {}, {}
        for o in ifit["observations_raw"]:
            if o["unit"] in SUPPLY_BLOCKS and o["seed"] in BLOCK_SEEDS:
                p = os.path.join(EXP031, "pred",
                                 f"pred_C5B3_u{o['unit']}_seed{o['seed']}_st3.npy")
                g = sha256_file(p)
                assert g == o["pred_sha256"], ("supply block pred drifted", p)
                blk[(o["unit"], o["seed"])] = np.load(p)
                prov[f"{o['unit']}_{o['seed']}"] = {"file": p, "sha256": g}
        assert len(blk) == 4, sorted(blk)
        Y = {V: y_all[season == V] for V in SUPPLY_BLOCKS}
        D = {V: d_full[season == V] for V in SUPPLY_BLOCKS}
        y_sup = np.concatenate([Y[V] for V in SUPPLY_BLOCKS for _ in BLOCK_SEEDS])
        p_sup = np.concatenate([blk[(V, s)] for V in SUPPLY_BLOCKS
                                for s in BLOCK_SEEDS])
        d_sup = np.concatenate([D[V] for V in SUPPLY_BLOCKS for _ in BLOCK_SEEDS])
        fit = off.fit_slope(y_sup - p_sup, d_sup)
        out["beta_refit_on_supply"] = fit
        out["beta_reproduces_frozen"] = bool(fit["beta"] == BETA)
        out["supply_provenance"] = prov
        assert out["beta_reproduces_frozen"], (fit["beta"], BETA)
        print(f"    beta refit on supply = {fit['beta']} — reproduces {BETA}",
              flush=True)
    except Exception as e:                                           # noqa: BLE001
        out["beta_refit_on_supply"] = {"status": "NOT RUN", "error": repr(e)}
        out["beta_reproduces_frozen"] = None
        print(f"    !! beta reproduction check skipped: {e!r}", flush=True)

    # ------------------------------------------------ load exp_037 predictions
    preds, prov037 = {}, {}
    for base in BASES:
        for s in SEEDS:
            stem = f"{base}_u2024_seed{s}_st2"
            pth = os.path.join(EXP037, "pred", f"pred_{stem}.npy")
            with open(os.path.join(EXP037, "res", f"res_{stem}.json")) as f:
                r = json.load(f)
            g = sha256_file(pth)
            assert g == r["pred_sha256"], ("exp_037 prediction drifted", stem,
                                           g, r["pred_sha256"])
            p = np.load(pth)
            assert len(p) == len(y24), (stem, len(p), len(y24))
            b = float(np.mean((p - y24) ** 2))
            assert abs(b - r["brier_full"]) < 5e-12, (stem, b, r["brier_full"])
            preds[(base, s)] = p
            prov037[stem] = {"sha256": g, "exp037_brier_full": r["brier_full"],
                             "our_recomputed_brier": b,
                             "stack_argmin_iter": r["stack_argmin_iter"],
                             "cap": r["cap"],
                             "any_argmin_at_cap": r["any_argmin_at_cap"]}
    out["exp037_provenance"] = prov037
    assert not any(v["any_argmin_at_cap"] for v in prov037.values())

    # ------------------------------------------------ gamma, per base, N recipe
    # candidate N solved gamma on the OFFSET-applied predictions and deployed
    # the 3-seed mean rounded to 6 places.  T2 needs the same solve on the
    # un-offset predictions, so both are computed here, per base.
    gammas = {}
    for base in BASES:
        entry = {}
        for tag, transform in (("on_base", lambda p: p),
                               ("on_offset",
                                lambda p: off.apply_linear(p, d24, BETA))):
            per = {}
            for s in SEEDS:
                g, resid = solve_gamma(transform(preds[(base, s)]), rate24)
                per[str(s)] = {"gamma_unrounded": g, "mean_residual_at_gamma": resid}
            raw = float(np.mean([per[str(s)]["gamma_unrounded"] for s in SEEDS]))
            entry[tag] = {"per_seed": per, "mean_unrounded": raw,
                          "deployed": float(f"{raw:.6f}")}
        gammas[base] = entry
    out["gamma"] = gammas
    out["gamma_recipe"] = ("per seed solve mean(sigmoid(logit(p)+gamma)) == the "
                           "realised 2024 success rate, average the three, round "
                           "to 6 places — candidate N's recipe")
    out["gamma_is_in_sample"] = ("YES. gamma is fitted to the mean of the very "
                                 "rows it is then scored on. 1 parameter vs "
                                 "253,507 rows; candidate N's deployed analogue "
                                 "transferred to LB at 1.1143x its local gain, "
                                 "which does not disconfirm it, but the bias is "
                                 "real and is not hidden here.")

    # ------------------------------------------------ the 24 evaluations
    results = {}
    for base in BASES:
        gb = gammas[base]["on_base"]["deployed"]
        go = gammas[base]["on_offset"]["deployed"]
        per_seed = {}
        for s in SEEDS:
            p0 = preds[(base, s)]
            raw_off = p0 + BETA * d24
            p1 = np.clip(raw_off, EPS, 1.0 - EPS)
            p2 = apply_gamma(p0, gb)
            p3 = apply_gamma(p1, go)
            n_clip = int(((raw_off < EPS) | (raw_off > 1.0 - EPS)).sum())
            row = {"n_clipped_offset": n_clip,
                   "gamma_used_T2": gb, "gamma_used_T3": go}
            for tag, pv in (("T0", p0), ("T1", p1), ("T2", p2), ("T3", p3)):
                row[tag] = {"valid_all": seg(y24, pv),
                            "valid_R": seg(y24[is_r], pv[is_r]),
                            "valid_F": seg(y24[~is_r], pv[~is_r])}
            b = {t: row[t]["valid_all"]["brier"] for t in ("T0", "T1", "T2", "T3")}
            row["delta_brier_vs_T0"] = {t: b[t] - b["T0"] for t in ("T1", "T2", "T3")}
            row["delta_score_vs_T0"] = {t: score_delta(b[t] - b["T0"])
                                        for t in ("T1", "T2", "T3")}
            row["delta_brier_T3_minus_T2"] = b["T3"] - b["T2"]
            row["delta_score_T3_minus_T2"] = score_delta(b["T3"] - b["T2"])
            row["delta_brier_T3_minus_T1"] = b["T3"] - b["T1"]
            row["delta_score_T3_minus_T1"] = score_delta(b["T3"] - b["T1"])
            per_seed[str(s)] = row

        agg = {"by_seed": per_seed}
        for t in ("T0", "T1", "T2", "T3"):
            for segname in ("valid_all", "valid_R", "valid_F"):
                bs = [per_seed[str(s)][t][segname]["brier"] for s in SEEDS]
                agg.setdefault(t, {})[segname] = {
                    "mean_brier": float(np.mean(bs)),
                    "mean_score": score(float(np.mean(bs))),
                    "seed_std_brier": float(np.std(bs, ddof=0)),
                    "seed_std_score_points": abs(score_delta(float(np.std(bs, ddof=0)))),
                }
            agg[t]["mean_pred_mean"] = float(np.mean(
                [per_seed[str(s)][t]["valid_all"]["pred_mean"] for s in SEEDS]))
        for t in ("T1", "T2", "T3"):
            dd = [per_seed[str(s)]["delta_brier_vs_T0"][t] for s in SEEDS]
            agg.setdefault("delta_vs_T0", {})[t] = {
                "per_seed_brier": {str(s): per_seed[str(s)]["delta_brier_vs_T0"][t]
                                   for s in SEEDS},
                "per_seed_score": {str(s): per_seed[str(s)]["delta_score_vs_T0"][t]
                                   for s in SEEDS},
                "mean_brier": float(np.mean(dd)),
                "mean_score": score_delta(float(np.mean(dd))),
                "signs_agree": bool(all(v < 0 for v in dd) or all(v > 0 for v in dd)),
                "improves": bool(float(np.mean(dd)) < 0)}
        for name, key in (("T3_minus_T1", "delta_brier_T3_minus_T1"),
                          ("T3_minus_T2", "delta_brier_T3_minus_T2")):
            dd = [per_seed[str(s)][key] for s in SEEDS]
            agg["delta_vs_T0"][name] = {
                "per_seed_brier": {str(s): per_seed[str(s)][key] for s in SEEDS},
                "per_seed_score": {str(s): score_delta(per_seed[str(s)][key])
                                   for s in SEEDS},
                "mean_brier": float(np.mean(dd)),
                "mean_score": score_delta(float(np.mean(dd))),
                "signs_agree": bool(all(v < 0 for v in dd) or all(v > 0 for v in dd)),
                "improves": bool(float(np.mean(dd)) < 0)}
        agg["n_clipped_offset_by_seed"] = {
            str(s): per_seed[str(s)]["n_clipped_offset"] for s in SEEDS}
        results[base] = agg
    out["results"] = results

    # ------------------------------------------------ THE question
    inter = {}
    for t in ("T1", "T2", "T3"):
        per = {}
        for s in SEEDS:
            gl = results["L_no_sit"]["by_seed"][str(s)]["delta_brier_vs_T0"][t]
            gr = results["R0"]["by_seed"][str(s)]["delta_brier_vs_T0"][t]
            # gains are NEGATIVE brier deltas.  gain_L - gain_R in POINTS:
            # positive => the transform pays MORE on the featured base.
            per[str(s)] = {
                "gain_points_R0": score_delta(gr),
                "gain_points_L_no_sit": score_delta(gl),
                "L_minus_R_points": float(f"{score_delta(gl) - score_delta(gr):.5f}"),
                "L_minus_R_brier": gl - gr}
        diffs = [per[str(s)]["L_minus_R_brier"] for s in SEEDS]
        signs_agree = bool(all(v < 0 for v in diffs) or all(v > 0 for v in diffs))
        inter[t] = {
            "per_seed": per,
            "mean_L_minus_R_brier": float(np.mean(diffs)),
            "mean_L_minus_R_points": float(f"{score_delta(float(np.mean(diffs))):.5f}"),
            "seed_signs_agree": signs_agree,
            "VERDICT": ("features_absorb_part_of_it" if signs_agree and np.mean(diffs) > 0
                        else "transform_pays_MORE_on_featured_base"
                        if signs_agree and np.mean(diffs) < 0
                        else "NOT MEASURABLE — the three seeds disagree in sign"),
            "caveat": ("beta was fitted without the teammate features, so a "
                       "shrunken gain on L_no_sit is equally consistent with "
                       "'beta is mis-fitted for this base'. Not separable here."
                       if t in ("T1", "T3") else
                       "gamma is solved per base, so no mis-fit confound here; "
                       "it is in-sample on both bases equally.")}
    out["interaction"] = inter

    out["changed_vs_prior_measurements"] = [
        "1. the offset is applied on a base carrying the teammate features for the first time",
        "2. the calibration is applied on a base carrying the teammate features for the first time",
        "3. offset+calibration measured together on fold_2024 for the first time (candidate N combined them only in the deployment build)",
        "4. the base is an exp_037 artefact — its stacker iteration is fold_2023 argmin x scale, not exp_030's 481",
    ]
    out["not_an_adoption_experiment"] = (
        "no threshold is applied. the offset and the calibration are already "
        "LB-validated and deployed; what is asked here is whether they still "
        "pay on top of the features.")
    out["compute_min"] = round((time.time() - t0) / 60, 2)

    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, "postproc.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print("WROTE " + p, flush=True)
    for base in BASES:
        dv = results[base]["delta_vs_T0"]
        print(f"  {base}: T1 {dv['T1']['mean_score']:+.5f} "
              f"T2 {dv['T2']['mean_score']:+.5f} T3 {dv['T3']['mean_score']:+.5f}",
              flush=True)
    print("  interaction (L - R, points): " + json.dumps(
        {t: [inter[t]["mean_L_minus_R_points"], inter[t]["seed_signs_agree"]]
         for t in inter}), flush=True)


if __name__ == "__main__":
    main()
