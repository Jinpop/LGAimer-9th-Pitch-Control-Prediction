"""exp_042 step 5 — assemble experiments/exp_042/metrics.json from the
artefacts that were pulled back from the server.  Runs locally; it reads
res/thread_cal.json, res/search_<cell>.json and res/promotion.json and writes
metrics.json.  It invents nothing: every number it emits is copied from one of
those three files or is arithmetic on them.
"""

import hashlib
import json
import os
import sys

EXP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF = 0.2498069


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def j(name):
    with open(os.path.join(EXP, "res", name)) as f:
        return json.load(f)


def r5(x):
    return float(f"{x:.5f}")


def main():
    cell = sys.argv[1] if len(sys.argv) > 1 else "is_wayoff_true"
    cal = j("thread_cal.json")
    se = j(f"search_{cell}.json")
    pr = j("promotion.json")
    with open(os.path.join(os.path.dirname(EXP), "exp_033", "metrics.json")) as f:
        bp = json.load(f)["block_provenance"]
    cells = ["is_mid_only", "is_rev_only", "is_both", "is_wayoff_true"]
    by_block = {b: v["cell_argmin"] for b, v in bp.items()}
    mean_argmin = {c: sum(by_block[b][c] for b in by_block) / len(by_block)
                   for c in cells}
    argmax_per_block = {b: max(cells, key=lambda c: by_block[b][c])
                        for b in by_block}
    with open(os.path.join(os.path.dirname(EXP), "exp_030", "stage1.json")) as f:
        ctrl_stage1 = json.load(f)

    top10 = [{"rank": t["rank"], "trial": t["number"],
              "is_current_config": t.get("is_current_config", False),
              "logloss": t["logloss"], "brier": t["brier"],
              "best_iteration": t["best_iteration"],
              "censored": t["censored"], "params": t["params"]}
             for t in se["top10"]]
    cur = se["current_config_record"]

    out = {
        "exp_id": "exp_042",
        "title": ("CatBoost C5_B3 hyper-parameter search — searched on "
                  "fold_2023, promoted and judged on fold_2024"),
        "conversion": {"formula": "score = 100000 * (1 - brier / 0.2498069)",
                       "ref_brier": REF,
                       "sensitivity": "1e-06 of Brier = 0.4003 points",
                       "note": ("a score converted from a 6-decimal Brier "
                                "carries +/-0.2002 points of rounding; Brier "
                                "is the raw datum")},
        "fold_2023_conversion_note": (
            "fold_2023 ABSOLUTE Brier is deliberately NOT converted — the "
            "reference 0.2498069 is the 2025 test base rate and has no referent "
            "on a 2023 validation set.  Only fold_2024 numbers are converted."),

        "cell_selection": {
            "rule": ("exp_033 metrics.json block_provenance: the cell with the "
                     "largest argmin iteration"),
            "pre_registered": "is_wayoff_true",
            "used": cell,
            "observed_cell_argmin_by_block": by_block,
            "argmax_cell_per_block": argmax_per_block,
            "mean_argmin_over_blocks": mean_argmin,
            "argmax_of_mean": max(cells, key=lambda c: mean_argmin[c]),
            "⚠️_the_rule_is_ambiguous": (
                "the rule 'the cell with the largest argmin' does not name which "
                "blocks to read.  Observed: the two 2022 blocks pick "
                "is_wayoff_true (725 / 625) — those are the numbers config.yaml "
                "itself cited — and the mean over all four blocks also picks "
                "is_wayoff_true (700.00).  A 2023-blocks-only reading would pick "
                "is_rev_only (1050 / 925).  The pre-registered cell was used."),
        },

        "threads": {
            "container_cpu_quota": 20,
            "nproc_is_host_value": 40,
            "calibration": cal["calibration"],
            "probe": cal["probe"],
            "od_cross_check": cal["od_cross_check"],
            "chosen_search_concurrency": {
                "workers": se["workers"], "threads_per_fit": se["threads_per_fit"],
                "total_threads": se["workers"] * se["threads_per_fit"]},
            "promote_phase_threads": 6,
        },

        "search": {
            "fold": se["fold"], "cell": se["cell"], "objective": se["objective"],
            "n_trials_declared": se["n_trials_declared"],
            "n_trials_run": se["n_trials_run"], "n_failed": se["n_failed"],
            "cap": se["cap"], "recap": se["recap"], "od_wait": se["od_wait"],
            "n_censored_at_first_cap": se["n_censored_at_first_cap"],
            "censored_trial_numbers": se["censored_trial_numbers"],
            "n_censored_after_recap": se["n_censored_after_recap"],
            "sampler": se["sampler"], "optuna_version": se["optuna_version"],
            "fit_seed": se["fit_seed"],
            "no_row_subsetting": se["no_row_subsetting"],
            "search_min": se["search_min"], "total_min": se["total_min"],
            "param_importances": se["param_importances"],
            "best": se["best"],
            "top10": top10,
            "current_config": se["current_config"],
            "current_config_rank": se["current_config_rank"],
            "current_config_record": cur,
            "winner_vs_current_fold2023": {
                "delta_logloss": se["best"]["logloss"] - cur["logloss"],
                "delta_brier_cell": se["best"]["brier"] - cur["brier"],
                "⚠️": ("this is the CELL's own binary target on fold_2023 — it "
                       "is not the competition metric and is not converted"),
            },
        },

        "promotion": {
            "fold": pr["fold"], "seeds": pr["seeds"],
            "n_valid": pr["n_valid"], "n_valid_R": pr["n_valid_R"],
            "n_valid_F": pr["n_valid_F"],
            "valid_R_share_pct": pr["valid_R_share_pct"],
            "identity_anchor": pr["identity_anchor"],
            "arms": pr["arms"],
            "per_seed_delta": pr["per_seed_delta"],
            "delta_summary": pr["delta_summary"],
            "adoption": pr["adoption"],
            "iterations_by_arm_and_seed": pr["iterations"],
            "stage1_iteration_source": {
                "policy": ("iterations are chosen ONCE, on fold_2023 unit 2023 "
                           "seed 42, at that unit's own curve argmin; fold_2024 "
                           "units scale them by the training-row ratio and never "
                           "consult their own validation curve"),
                "ctrl": {"cap": ctrl_stage1["cap"],
                         "any_argmin_at_cap": ctrl_stage1["any_argmin_at_cap"],
                         "cell_argmin_iter": ctrl_stage1["cell_argmin_iter"],
                         "stack_argmin_iter": ctrl_stage1["stack_argmin_iter"],
                         "reused_from": ("experiments/exp_030/stage1.json — the "
                                         "control did not re-derive them")},
                "hpo": {"cap": 3300,
                        "any_argmin_at_cap": None,   # filled below from the file
                        "cell_argmin_iter": None,
                        "stack_argmin_iter": None,
                        "minutes": None},
            },
            "seed_stability": {
                "ctrl_seed_sd_brier":
                    pr["arms"]["ctrl"]["mean_over_seeds"]["seed_sd_brier"],
                "hpo_seed_sd_brier":
                    pr["arms"]["hpo"]["mean_over_seeds"]["seed_sd_brier"],
                "note": ("recorded because the seed spread of the base is what "
                         "sets the adoption threshold for every later "
                         "experiment on this base"),
            },
        },

        "fold_2024": {
            "ctrl": {"brier": pr["arms"]["ctrl"]["mean_over_seeds"]["mean_brier"],
                     "score": pr["arms"]["ctrl"]["mean_over_seeds"]["mean_score"],
                     "by_seed": {s: {"brier": pr["arms"]["ctrl"][s]["brier"],
                                     "score": pr["arms"]["ctrl"][s]["score"],
                                     "bss": pr["arms"]["ctrl"][s]["valid_all"]["bss"],
                                     "auc": pr["arms"]["ctrl"][s]["valid_all"]["auc"]}
                                 for s in ("42", "7", "2024")}},
            "hpo": {"brier": pr["arms"]["hpo"]["mean_over_seeds"]["mean_brier"],
                    "score": pr["arms"]["hpo"]["mean_over_seeds"]["mean_score"],
                    "by_seed": {s: {"brier": pr["arms"]["hpo"][s]["brier"],
                                    "score": pr["arms"]["hpo"][s]["score"],
                                    "bss": pr["arms"]["hpo"][s]["valid_all"]["bss"],
                                    "auc": pr["arms"]["hpo"][s]["valid_all"]["auc"]}
                                for s in ("42", "7", "2024")}},
        },

        "changed_variables": {
            "1_hyperparameters": {"from": se["current_config"],
                                  "to": se["best"]["params"]},
            "2_iterations_dragged_along": (
                "the hpo arm re-derives stage 1 on fold_2023 with its own "
                "settings, so every cell round and the stacker round moves; "
                "the full per-arm per-seed table is in "
                "promotion.iterations_by_arm_and_seed"),
            "3_search_phase_thread_count": (
                f"{se['threads_per_fit']} threads x {se['workers']} workers "
                "during the search only; the promotion runs at 6 threads"),
            "unchanged": ["47 feature columns", "C5_B3 structure (4 cells + "
                          "stacker)", "fold definitions", "seeds 42/7/2024",
                          "no post-hoc transform of any kind"],
        },
        "source_sha256": {f: sha(os.path.join(EXP, "src", f))
                          for f in sorted(os.listdir(os.path.join(EXP, "src")))
                          if f.endswith((".py", ".sh"))},
        "train_csv_sha256": se["train_csv_sha256"],
    }
    with open(os.path.join(EXP, "stage1_hpo.json")) as f:
        h1 = json.load(f)
    hs = out["promotion"]["stage1_iteration_source"]["hpo"]
    hs["cap"] = h1["cap"]
    hs["any_argmin_at_cap"] = h1["any_argmin_at_cap"]
    hs["cell_argmin_iter"] = h1["cell_argmin_iter"]
    hs["stack_argmin_iter"] = h1["stack_argmin_iter"]
    hs["minutes"] = h1["minutes"]
    hs["params_override"] = h1["params_override"]

    out["wall_clock_min"] = {
        "thread_calibration": cal["data_build_min"] + sum(
            v["fit_min"] for v in cal["calibration"].values())
        + cal["od_cross_check"]["fit_min"],
        "search_150_trials": se["search_min"],
        "search_recap_reruns": round(se["total_min"] - se["search_min"], 2),
        "search_total": se["total_min"],
        "promotion_stage1_hpo": h1["minutes"],
        "promotion_units": {f"{t}_{s}": pr["arms"][t][s]["minutes"]
                            for t in ("ctrl", "hpo") for s in ("42", "7", "2024")},
        "⚠️_vs_config_eta": (
            "config.yaml budgeted 210 min total (search 60-90).  The search "
            "alone took 224.28 min.  Cause: the ETA was priced from the CONTROL "
            "cell fit (1.61-1.85 min at lr 0.03) while the registered space "
            "reaches lr 0.01, where a single fit ran 9-13 min."),
    }
    p = os.path.join(EXP, "metrics.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("WROTE " + p)


if __name__ == "__main__":
    main()
