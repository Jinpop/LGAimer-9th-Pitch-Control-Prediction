"""exp_008 — add the standard top-level fold keys + derived anchor arithmetic.

Runs LOCALLY on the metrics.json fetched from the server. Pure post-processing:
it only reshapes numbers already produced by train.py / calib_probe.py, it never
recomputes a model. Idempotent.

    python3 src/finalize_metrics.py experiments/exp_008/metrics.json
"""

import json
import sys

LB_RF = 549.51193  # measured LB of the organizers' RF (trained <=2024 -> 2025 test)
DEN_R = 0.489707 * (1 - 0.489707)     # valid-R 2024 base-rate variance
DEN_ALL = 0.486105 * (1 - 0.486105)   # valid-all 2024 base-rate variance


def local_score(brier, den):
    return max(0.0, 100000.0 * (1.0 - brier / den))


def main(path):
    m = json.load(open(path))
    r1 = m["R1"]
    # standard top-level fold keys (headline run = R1, our primary gauge = valid_R)
    for fold in ("fold_2022", "fold_2023", "fold_2024"):
        d = r1[fold]
        m[fold] = {
            "run": "R1 (organizers' RF verbatim)",
            "brier": d["valid_R"]["brier"],
            "bss": d["valid_R"]["bss"],
            "auc": d["valid_R"]["auc"],
            "score": d["valid_R"]["score"],
            "brier_all": d["valid_all"]["brier"],
            "bss_all": d["valid_all"]["bss"],
            "score_all": d["valid_all"]["score"],
            "n_valid_R": d["valid_R"]["n"],
            "n_valid_all": d["valid_all"]["n"],
        }
    m["inference_time_sec"] = round(
        sum(r1[f]["inference_time_sec"] for f in ("fold_2022", "fold_2023", "fold_2024"))
        + sum(m["R2"][f]["inference_time_sec"] for f in ("fold_2023", "fold_2024")),
        2,
    )

    f_r = LB_RF / r1["fold_2024"]["valid_R"]["score"]
    f_all = LB_RF / r1["fold_2024"]["valid_all"]["score"]
    targets = {}
    for lb in (549.51193, 930.0, 989.0):
        bss_r = (lb / f_r) / 1e5
        bss_all = (lb / f_all) / 1e5
        targets[f"LB_{lb:g}"] = {
            "local_valid_R_bss_needed": round(bss_r, 5),
            "local_valid_R_brier_needed": round(DEN_R * (1 - bss_r), 6),
            "local_valid_all_bss_needed": round(bss_all, 5),
            "local_valid_all_brier_needed": round(DEN_ALL * (1 - bss_all), 6),
        }
    m["anchor"]["targets"] = targets
    m["anchor"]["rf_minus_exp007V1_valid_R_brier"] = round(
        r1["fold_2024"]["valid_R"]["brier"] - m["references"]["exp_007_V1_postcalib_valid_R_brier"], 6
    )
    m["anchor"]["gap_top989_over_ours_expected"] = round(989.0 / (333.363 * f_r), 2)
    m["anchor"]["gap_top989_over_rf"] = round(989.0 / LB_RF, 2)
    m["anchor"]["note"] = (
        "LB score == 100000 * BSS by definition, so the 'factor' is purely the "
        "split shift (<=2023->2024 valid vs <=2024->2025 test), not a unit change. "
        "valid_R gauge factor ~0.98, valid_all gauge factor ~1.32: the two bracket "
        "the true transfer. Ordering between models on identical valid rows is "
        "gauge-independent."
    )
    m["mask_effect_rf"] = {
        "fold_2024_valid_R_delta": round(
            m["R2"]["fold_2024"]["valid_R"]["brier"] - r1["fold_2024"]["valid_R"]["brier"], 6
        ),
        "fold_2024_valid_all_delta": round(
            m["R2"]["fold_2024"]["valid_all"]["brier"] - r1["fold_2024"]["valid_all"]["brier"], 6
        ),
        "fold_2023_valid_R_delta": round(
            m["R2"]["fold_2023"]["valid_R"]["brier"] - r1["fold_2023"]["valid_R"]["brier"], 6
        ),
        "fold_2023_valid_all_delta": round(
            m["R2"]["fold_2023"]["valid_all"]["brier"] - r1["fold_2023"]["valid_all"]["brier"], 6
        ),
        "lgbm_reference_exp_004_valid_R_delta": -0.000088,
    }
    m["oracle_offset_probe"] = {
        "source": "src/calib_probe.py (offset fitted ON valid -> ORACLE upper bound, not usable)",
        "R1_valid_R": {"offset": -0.0403, "brier": 0.248386, "gain": -0.000101},
        "R1_valid_all": {"offset": -0.0607, "brier": 0.248540, "gain": -0.000228},
        "R2_valid_R": {"offset": -0.0376, "brier": 0.248368, "gain": -0.000087},
        "R2_valid_all": {"offset": -0.0417, "brier": 0.248322, "gain": -0.000108},
        "corr_R1_R2_preds": 0.9072,
        "R1_pred_std": 0.03244,
        "R1_pred_range": [0.4082, 0.619],
    }
    json.dump(m, open(path, "w"), indent=2, ensure_ascii=False)
    print(json.dumps({k: m[k] for k in ("fold_2022", "fold_2023", "fold_2024", "mask_effect_rf")}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
