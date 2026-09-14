"""exp_017 — assemble metrics.json and policy_results.json from the run files."""
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
RAW = os.path.join(EXP, "raw")
REF = 0.2498069            # r(1-r) of fold_2024 valid_all (EXPERIMENTS.md)


def score(b):
    return round(100000.0 * (1.0 - b / REF), 1)


def bss(b):
    return round(1.0 - b / REF, 6)


def main():
    off = json.load(open(os.path.join(RAW, "offline_final.json")))
    runs = {}
    meta = {}
    for tag in ["extend", "full01", "cap8k", "lowlr"]:
        m = json.load(open(os.path.join(RAW, f"{tag}_metrics.json")))
        meta[tag] = {"learning_rate": m["learning_rate"],
                     "cap": m["num_boost_round_cap"],
                     "seeds": m["seeds"], "folds": m["folds_wanted"],
                     "elapsed_min": m.get("elapsed_min"),
                     "repro_gate": m.get("repro_gate"),
                     "sha256": m["sha256"], "versions": m["versions"]}
        for r in m["runs"]:
            runs[(tag, r["arm"], r["seed"], r["fold"])] = r

    hm = off["holdout_multiplier"]
    ap = hm["applied_to_late"]

    def grp(arm, field):
        ks = sorted(k for k in ap if k.startswith(arm + "|"))
        return [ap[k][field] for k in ks]

    out = {
        "exp_id": "exp_017",
        "kind": "policy experiment (no new model architecture)",
        "base_exp": "exp_016 B3 stacking / B0 control, exp_010 D hyperparameters",
        "primary_metric": "fold_2024 valid_all Brier",
        "score_reference_r1mr": REF,
        "runtime": meta,
        "compute_min_total": round(sum(v["elapsed_min"] for v in meta.values()), 1),

        # ---- headline: fold_2024 under the control policy and the adopted one #
        "fold_2024": {
            "B3_P4_control": {
                "iter_by_seed": {"42": 650, "7": 650, "2024": 550},
                "brier_by_seed": [0.248191, 0.248215, 0.248217],
                "brier": round(st.mean([0.248191, 0.248215, 0.248217]), 6),
                "bss": bss(st.mean([0.248191, 0.248215, 0.248217])),
                "score": score(st.mean([0.248191, 0.248215, 0.248217])),
                "auc_seed42": 0.546121,
                "seed_std_e6": round(1e6 * st.stdev(
                    [0.248191, 0.248215, 0.248217]), 1),
            },
            "B3_P10_c2.75_ADOPTED": {
                "iter_by_seed": dict(zip(["2024", "42", "7"], grp("B3", "iter"))),
                "brier_by_seed": grp("B3", "brier"),
                "brier": round(st.mean(grp("B3", "brier")), 6),
                "bss": bss(st.mean(grp("B3", "brier"))),
                "score": score(st.mean(grp("B3", "brier"))),
                "auc": None,
                "auc_note": "the offline curves store Brier only; AUC at the "
                            "policy iteration was not recomputed",
                "seed_std_e6": round(1e6 * st.stdev(grp("B3", "brier")), 1),
                "recovery_pct_of_P4_gap": grp("B3", "recovery_pct"),
            },
            "B3_oracle_upper_reference_NOT_A_POLICY": {
                "iter_by_seed": {"42": 2800, "7": 2300, "2024": 2150},
                "brier": round(st.mean([0.248176, 0.248174, 0.248174]), 6),
                "bss": bss(st.mean([0.248176, 0.248174, 0.248174])),
                "score": score(st.mean([0.248176, 0.248174, 0.248174])),
            },
            "B0_P4_control": {
                "iter_by_seed": {"42": 750, "7": 775, "2024": 1000},
                "brier": round(st.mean([0.248261, 0.248266, 0.248260]), 6),
                "bss": bss(st.mean([0.248261, 0.248266, 0.248260])),
                "score": score(st.mean([0.248261, 0.248266, 0.248260])),
                "auc_seed42": 0.545191,
            },
            "B0_P10_c2.75": {
                "iter_by_seed": dict(zip(["2024", "42", "7"], grp("B0", "iter"))),
                "brier_by_seed": grp("B0", "brier"),
                "brier": round(st.mean(grp("B0", "brier")), 6),
                "bss": bss(st.mean(grp("B0", "brier"))),
                "score": score(st.mean(grp("B0", "brier"))),
                "recovery_pct_of_P4_gap": grp("B0", "recovery_pct"),
            },
            "paired_delta_B3_minus_B0_e6": {
                "P4": [-70, -51, -43],
                "P10_c2.75": [round(1e6 * (b - a), 1) for b, a in
                              zip(grp("B3", "brier"), grp("B0", "brier"))],
            },
        },

        # ---- the two earlier folds, seed 42, production spec ---------------- #
        "fold_2023": {
            "B3_P4": {"iter": 170, "brier": 0.248908,
                      "note": "first honest (non-oracle) B3 fold_2023 number; "
                              "exp_016 could only report the oracle 0.248756 "
                              "because its B3 chain started at fold_2023"},
            "B3_oracle": {"iter": 650, "brier": 0.248756},
            "B0_P4": {"iter": 300, "brier": 0.248892},
            "B0_oracle": {"iter": 750, "brier": 0.248805},
        },
        "fold_2022": {
            "B3_oracle_no_predecessor": {"iter": 170, "brier": 0.247681},
            "B0_oracle_no_predecessor": {"iter": 300, "brier": 0.247197},
        },

        "argmin_table": off["argmin_table"],
        "curve_shape": off["curve_shape"],
        "holdout_multiplier": hm,
        "cap_censoring_check": {
            "question": "were the exp_016 fold_2024 argmins (2650-2800) an "
                        "artefact of the 3000-round search cap?",
            "answer": "no",
            "evidence": {"cap_3000_argmins": {"B0": [300, 750, 1550],
                                              "B3": [170, 650, 2800]},
                         "cap_8000_argmins": {"B0": [300, 750, 1550],
                                              "B3": [170, 650, 2800]}},
        },
        "train_time_min": round(sum(v["elapsed_min"] for v in meta.values()), 1),
        "wall_clock_min_4_jobs_in_parallel": 102,
        "inference_time_sec": None,
        "inference_time_note": "not measured — this experiment produced no new "
                               "submission artefact; the model spec is exp_016 B3 "
                               "with a different num_boost_round",
    }

    with open(os.path.join(EXP, "metrics.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    with open(os.path.join(EXP, "policy_results.json"), "w") as f:
        json.dump({"rows": off["rows"], "aggregate": off["aggregate"],
                   "multiplier_scan": off["multiplier_scan"],
                   "holdout_multiplier": off["holdout_multiplier"]},
                  f, indent=1, ensure_ascii=False)
    print("wrote metrics.json and policy_results.json")
    print(json.dumps(out["fold_2024"], indent=1))


if __name__ == "__main__":
    sys.exit(main())
