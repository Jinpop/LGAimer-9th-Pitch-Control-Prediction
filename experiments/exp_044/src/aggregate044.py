"""exp_044 — collect every result into metrics.json and apply the registered rule.

Nothing in this file decides anything that prereg.json did not already fix.  It
reads what the runs wrote, converts Brier to the leaderboard scale, computes the
control arm's seed sigma with ddof=1, and applies the pre-registered verdict.

The identity anchor is checked here by hashing exp_042's stored predictions at
run time and comparing them with exp_044's — an observed comparison of two files
on disk, not a sha typed into a source file.
"""

import glob
import hashlib
import json
import os
import sys

import numpy as np

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)
from labels import FAIL_CELLS                                       # noqa: E402

E = os.path.expanduser("~/SungjinPark/experiments/exp_044")
E42 = os.path.expanduser("~/SungjinPark/experiments/exp_042")
REF_BRIER = 0.2498069
SEEDS = [42, 7, 2024, 1, 13, 101, 777, 2025, 31337]
ANCHOR_SEEDS = [42, 7, 2024]        # the seeds exp_042 also ran


def score_of(b):
    return 100000.0 * (1.0 - b / REF_BRIER)


def d_score(db):
    """Brier difference -> score difference (a fall in Brier is a gain)."""
    return -db * 100000.0 / REF_BRIER


def r5(x):
    """5 decimals, no trailing zeros, as the project's recording rule asks."""
    return float(f"{x:.5f}")


def sha256_file(p, block=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load(p):
    with open(p) as f:
        return json.load(f)


def search_summary(name):
    p = os.path.join(E, "res", f"search_{name}.json")
    if not os.path.exists(p):
        return {"missing": p}
    d = load(p)
    ok = d["objective"]
    return {
        "objective": ok, "eval_metric": d["eval_metric"],
        "fold": d["fold"], "n_trials": d["n_trials"],
        "cap": d["cap"], "recap": d["recap"], "od_wait": d["od_wait"],
        "workers": d["workers"], "threads_per_fit": d["threads_per_fit"],
        "schedule_counts": d["schedule_counts"], "wide_indices": d["wide_indices"],
        "censoring": {
            "n_censored_screen_at_cap": d["n_censored_screen_at_cap"],
            "censored_screen_indices": d["censored_screen_indices"],
            "pct_censored_screen_at_cap": r5(
                100.0 * d["n_censored_screen_at_cap"] / d["n_trials"]),
            "n_censored_screen_after_recap": d["n_censored_screen_after_recap"],
            "n_censored_confirm_at_cap": d["n_censored_confirm_at_cap"],
            "n_failed_screen": d["n_failed_screen"],
        },
        "param_importances": d["param_importances"],
        "pick": {"params": d["pick"]["params"], "space": d["pick"]["space"],
                 "screen_index": d["pick"]["index"],
                 "screen_rank": d["pick"]["screen_rank"],
                 f"mean3_{ok}": d["pick"][f"mean3_{ok}"],
                 f"std3_{ok}": d["pick"][f"std3_{ok}"],
                 f"per_seed_{ok}": d["pick"][f"per_seed_{ok}"],
                 "best_iteration_per_seed": d["pick"]["best_iteration_per_seed"],
                 "cap_per_seed": d["pick"]["cap_per_seed"]},
        "pick_came_from_wide_space": bool(d["pick"]["space"] == "wide"),
        "exp042_winner_screen_rank": d["exp042_winner_screen_rank"],
        "exp042_winner_confirm": d["exp042_winner_confirm"],
        "top10_screen": [{"index": r["index"], "space": r["space"],
                          ok: r[ok], "best_iteration": r["best_iteration"],
                          "cap": r["cap"], "params": r["params"]}
                         for r in d["top10_screen"]],
        "confirm_ranked": [{"confirm_rank": r["confirm_rank"],
                            "index": r["index"], "space": r["space"],
                            f"mean3_{ok}": r[f"mean3_{ok}"],
                            f"per_seed_{ok}": r[f"per_seed_{ok}"],
                            "best_iteration_per_seed":
                                r["best_iteration_per_seed"]}
                           for r in d["confirm_ranked"]],
        "timing_min": {"screen": d["screen_min"], "confirm": d["confirm_min"],
                       "total": d["total_min"],
                       "fit_min_screen": d["fit_min_screen"]},
    }


def main():
    out = {"exp_id": "exp_044",
           "ref_brier_for_conversion": REF_BRIER,
           "conversion": "score = 100000*(1 - brier/0.2498069)",
           "prereg_sha256": sha256_file(os.path.join(E, "prereg.json"))}

    # ------------------------------------------------------------- searches
    out["searches"] = {n: search_summary(n)
                       for n in list(FAIL_CELLS) + ["stacker"]}

    # -------------------------------------------- cell winners vs exp_042
    w42 = load(os.path.join(E42, "best_params.json"))
    diff = {}
    for n in list(FAIL_CELLS) + ["stacker"]:
        s = out["searches"][n]
        if "pick" not in s:
            continue
        pk = s["pick"]["params"]
        diff[n] = {k: {"exp044": pk[k], "exp042": w42[k],
                       "same": bool(pk[k] == w42[k])} for k in sorted(w42)}
        diff[n]["_space"] = s["pick"]["space"]
    out["winner_vs_exp042_table"] = diff
    out["exp042_single_setting"] = w42

    # ------------------------------------------------------- stage-1 repeat
    a = os.path.join(E, "stage1_pcA.json")
    b = os.path.join(E, "stage1_pc.json")
    if os.path.exists(a) and os.path.exists(b):
        A, B = load(a), load(b)
        out["stage1"] = {
            "pcA": {"cell_argmin_iter": A["cell_argmin_iter"],
                    "stack_argmin_iter": A["stack_argmin_iter"],
                    "any_argmin_at_cap": A["any_argmin_at_cap"],
                    "cap": A["cap"], "minutes": A["minutes"],
                    "stack_sha256": A.get("stack_sha256")},
            "pc": {"cell_argmin_iter": B["cell_argmin_iter"],
                   "stack_argmin_iter": B["stack_argmin_iter"],
                   "any_argmin_at_cap": B["any_argmin_at_cap"],
                   "cap": B["cap"], "minutes": B["minutes"],
                   "stack_sha256": B.get("stack_sha256")},
            "cell_argmin_repeats": bool(
                A["cell_argmin_iter"] == B["cell_argmin_iter"]),
            "stack_npz_sha_repeats": bool(
                A.get("stack_sha256") == B.get("stack_sha256")),
            "note": ("pcA and pc differ ONLY in the stacker slot, so the four "
                     "cell argmins and the OOF npz must repeat exactly; that "
                     "is a determinism check on the whole cell + OOF path"),
        }
        out["stage1"]["ctrl_exp042"] = {
            k: load(os.path.join(E42, "stage1_hpo.json"))[k]
            for k in ("cell_argmin_iter", "stack_argmin_iter",
                      "any_argmin_at_cap", "cap")}

    # ----------------------------------------------------------- promotion
    arms = {}
    for arm in ("ctrl", "pc"):
        rows = {}
        for s in SEEDS:
            p = os.path.join(E, "res", f"res_C5B3_{arm}_u2024_seed{s}_st2.json")
            if not os.path.exists(p):
                continue
            d = load(p)
            rows[s] = {
                "brier": d["brier_full"],
                "score": r5(score_of(d["brier_full"])),
                "valid_R_brier": d["valid_R"]["brier"],
                "valid_F_brier": d["valid_F"]["brier"],
                "auc": d["valid_all"]["auc"],
                "cell_iter_used": d["cell_iter_used"],
                "stack_iter_used": d["stacker"]["iter_used"],
                "stack_argmin_iter": d["stacker"]["argmin_iter"],
                "stack_curve_argmin_is_interior":
                    d["stacker"]["curve_argmin_is_interior"],
                "n_distinct_settings": d["n_distinct_settings"],
                "minutes": d["minutes"],
                "pred_sha256": d["pred_sha256"],
                "pred_file": d["pred_file"],
            }
        arms[arm] = rows
    out["promotion_units"] = arms

    # --------------------------------------------------- the identity anchor
    anchor = {}
    for s in ANCHOR_SEEDS:
        p42 = os.path.join(E42, "pred", f"pred_C5B3_hpo_u2024_seed{s}_st2.npy")
        p44 = os.path.join(E, "pred", f"pred_C5B3_ctrl_u2024_seed{s}_st2.npy")
        rec = {"exp042_file": p42, "exp044_file": p44,
               "exp042_present": os.path.exists(p42),
               "exp044_present": os.path.exists(p44)}
        if rec["exp042_present"] and rec["exp044_present"]:
            h42, h44 = sha256_file(p42), sha256_file(p44)
            rec.update({"exp042_sha256": h42, "exp044_sha256": h44,
                        "identical": bool(h42 == h44)})
            if h42 != h44:
                x, y = np.load(p42), np.load(p44)
                rec["max_abs_diff"] = float(np.max(np.abs(x - y)))
        anchor[s] = rec
    out["identity_anchor"] = {
        "what": ("the control arm is the exp_042 winner on all five models, so "
                 "running it through the new per-model code must reproduce "
                 "exp_042's stored fold_2024 predictions byte for byte"),
        "seeds": anchor,
        "all_identical": bool(anchor and all(
            v.get("identical") is True for v in anchor.values())),
    }

    # ------------------------------------------- sigma, threshold, verdict
    c, t = arms["ctrl"], arms["pc"]
    both = [s for s in SEEDS if s in c and s in t]
    if len(both) == len(SEEDS):
        cb = np.array([c[s]["brier"] for s in SEEDS], dtype=np.float64)
        tb = np.array([t[s]["brier"] for s in SEEDS], dtype=np.float64)
        sig_b = float(np.std(cb, ddof=1))
        sig_s = -d_score(sig_b)               # a spread, reported as magnitude
        dlt = tb - cb
        mean_d = float(dlt.mean())
        signs = [int(np.sign(x)) for x in dlt]
        all_neg = bool(all(x < 0 for x in dlt))
        all_pos = bool(all(x > 0 for x in dlt))
        out["sigma_and_verdict"] = {
            "recipe": ("sigma = std(ddof=1) of the CONTROL arm's nine fold_2024 "
                       "valid_all Brier values, converted to score units. "
                       "Registered in prereg.json before any value existed."),
            "n_seeds": len(SEEDS), "seeds": SEEDS,
            "control_brier_by_seed": {str(s): c[s]["brier"] for s in SEEDS},
            "treatment_brier_by_seed": {str(s): t[s]["brier"] for s in SEEDS},
            "control_mean_brier": float(cb.mean()),
            "treatment_mean_brier": float(tb.mean()),
            "control_mean_score": r5(score_of(float(cb.mean()))),
            "treatment_mean_score": r5(score_of(float(tb.mean()))),
            "sigma_brier_ddof1": sig_b,
            "sigma_score": r5(sig_s),
            "threshold_3sigma_brier": 3.0 * sig_b,
            "threshold_3sigma_score": r5(3.0 * sig_s),
            "delta_brier_by_seed": {str(s): float(x)
                                    for s, x in zip(SEEDS, dlt)},
            "delta_score_by_seed": {str(s): r5(d_score(float(x)))
                                    for s, x in zip(SEEDS, dlt)},
            "mean_delta_brier": mean_d,
            "mean_delta_score": r5(d_score(mean_d)),
            "sign_agreement": {"all_negative_treatment_better": all_neg,
                               "all_positive_treatment_worse": all_pos,
                               "signs": signs},
            "abs_mean_delta_over_sigma": (abs(mean_d) / sig_b) if sig_b else None,
            "verdict": ("ADOPT" if ((all_neg or all_pos)
                                    and abs(mean_d) >= 3.0 * sig_b
                                    and all_neg)
                        else "REJECT"),
            "verdict_rule": ("ADOPT requires all nine per-seed deltas to share "
                             "a sign AND |mean delta| >= 3 sigma AND that sign "
                             "to favour the treatment. Anything else rejects."),
            "forward_only": ("this sigma applies from exp_045 onward; no past "
                             "verdict is recomputed with it"),
            "scope": ("valid for this base and this feature set only"),
        }
        # paired seed-wise spread, recorded because it is cheaper information
        out["sigma_and_verdict"]["paired_delta_std_ddof1"] = float(
            np.std(dlt, ddof=1))
        out["sigma_and_verdict"]["treatment_sigma_brier_ddof1"] = float(
            np.std(tb, ddof=1))
    else:
        out["sigma_and_verdict"] = {
            "incomplete": True,
            "have_ctrl": sorted(c), "have_pc": sorted(t)}

    # ------------------------------------------------------------- timings
    tm = {}
    for f in sorted(glob.glob(os.path.join(E, "res", "search_*.json"))):
        d = load(f)
        tm[f"search_{d['name']}_min"] = d["total_min"]
    for k, f in (("stage1_pcA", "stage1_pcA.json"), ("stage1_pc", "stage1_pc.json")):
        p = os.path.join(E, f)
        if os.path.exists(p):
            tm[f"{k}_min"] = load(p)["minutes"]
    umin = [v["minutes"] for arm in arms.values() for v in arm.values()]
    if umin:
        tm["promotion_unit_min"] = {"n": len(umin), "min": min(umin),
                                    "median": float(np.median(umin)),
                                    "max": max(umin), "sum": float(sum(umin))}
    out["measured_minutes"] = tm

    p = os.path.join(E, "metrics.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print("WROTE " + p)
    if "sigma_and_verdict" in out and not out["sigma_and_verdict"].get("incomplete"):
        sv = out["sigma_and_verdict"]
        print(json.dumps({k: sv[k] for k in (
            "sigma_brier_ddof1", "sigma_score", "threshold_3sigma_score",
            "mean_delta_brier", "mean_delta_score", "sign_agreement",
            "verdict")}, indent=1))
    print("identity anchor all identical: "
          f"{out['identity_anchor']['all_identical']}")


if __name__ == "__main__":
    main()
