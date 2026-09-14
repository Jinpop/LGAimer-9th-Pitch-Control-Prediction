"""exp_045 — assemble metrics.json from the fit results + ensemble_measurement.

    python make_metrics.py

Also measures inference wall time for one fold_2024 valid pass (253,507 rows)
on GPU and on CPU, since a NN would have to fit inside the competition's
inference budget if it were ever used.
"""

import glob
import json
import os
import statistics as st
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from train_nn import ARCH, EmbMLP  # noqa: E402

ROOT = os.path.expanduser("~/SungjinPark/experiments/exp_045")
PRIMARY, SECONDARY = "D0", "C3"
SEEDS = [42, 7, 2024]


def res(arch, fold, seed):
    return json.load(open(os.path.join(ROOT, "res",
                                       f"res_{arch}_fold{fold}_seed{seed}.json")))


def time_inference(arch):
    meta = json.load(open(os.path.join(ROOT, "cache", "fold_2024.json")))
    z = np.load(os.path.join(ROOT, "cache", "fold_2024.npz"), allow_pickle=False)
    cfg = ARCH[arch]
    slots = [meta["vocab"][c]["n_slots_incl_oov"] for c in C.EMB_COLS]
    dims = [max(2, int(round(C.EMB_DIM[c] * cfg["emb_mult"])))
            if c in ("pitcher_id", "batter_id") else C.EMB_DIM[c]
            for c in C.EMB_COLS]
    out = {}
    for dev_name in ("cuda", "cpu"):
        if dev_name == "cuda" and not torch.cuda.is_available():
            continue
        dev = torch.device(dev_name)
        torch.manual_seed(0)
        m = EmbMLP(slots, dims, z["num_va"].shape[1], cfg["hidden"],
                   cfg["dropout"], cfg.get("emb_dropout", 0.0)).to(dev).eval()
        vn = torch.from_numpy(z["num_va"]).to(dev)
        ve = torch.from_numpy(z["emb_va"]).to(dev)
        with torch.no_grad():                       # warm-up then measure
            for _ in range(2):
                for i in range(0, vn.shape[0], 65536):
                    torch.sigmoid(m(vn[i:i + 65536], ve[i:i + 65536]))
            if dev_name == "cuda":
                torch.cuda.synchronize()
            t = time.time()
            for i in range(0, vn.shape[0], 65536):
                torch.sigmoid(m(vn[i:i + 65536], ve[i:i + 65536]))
            if dev_name == "cuda":
                torch.cuda.synchronize()
            out[dev_name] = round(time.time() - t, 3)
        del vn, ve, m
        if dev_name == "cuda":
            torch.cuda.empty_cache()
    out["n_rows"] = int(z["num_va"].shape[0])
    return out


def main():
    em = json.load(open(os.path.join(ROOT, "ensemble_measurement.json")))
    em3 = json.load(open(os.path.join(ROOT, "ensemble_measurement_C3.json")))

    all_res = sorted(glob.glob(os.path.join(ROOT, "res", "res_*.json")))
    total_fit_min = 0.0
    recon = []
    for f in all_res:
        d = json.load(open(f))
        total_fit_min += d["fit_time_min"]
        if d["fold"] == "fold_2023" and d["seed"] == 42:
            recon.append({"arch": d["arch"], "cfg": d["arch_cfg"],
                          "n_params": d["n_params"],
                          "best_epoch": d["early_stopping"]["best_epoch"],
                          "brier": d["early_stopping"]["best_brier"],
                          "score": d["early_stopping"]["best_score"]})
    recon.sort(key=lambda r: -r["score"])

    confirm = {}
    for a in (PRIMARY, "D9", "D4", SECONDARY):
        try:
            s = [res(a, 2023, k)["early_stopping"]["best_score"] for k in SEEDS]
        except FileNotFoundError:
            continue
        confirm[a] = {"score_by_seed": dict(zip(map(str, SEEDS), s)),
                      "mean_score": round(sum(s) / 3, 5),
                      "sd_score_ddof1": round(st.stdev(s), 5)}

    nn = em["nn"]
    m1 = em["measure_1_gap"]["PRIMARY_honest_vs_offset_candidate"]
    m3 = em["measure_3_convexity"]["PRIMARY_seed_matched"]
    m4 = em["measure_4_honest_transfer"]
    gap = abs(m1["gap_nn_minus_cat_score_points"])
    conv = m3["convexity_score_points"]

    out = {
        "exp_id": "exp_045",
        "title": "NN (entity-embedding MLP) vs CatBoost C5_B3: gap, correlation, "
                 "convexity",
        "conversion_note": "score = 100000 * (1 - Brier / 0.2498069).  A score "
                           "converted from a 6-decimal Brier carries +-0.20 of "
                           "uncertainty (Brier 1e-6 == 0.4003 score points); the "
                           "Brier is the raw datum, the score is the conversion.",
        "primary_arch": PRIMARY, "secondary_arch": SECONDARY,
        "arch_cfg": nn["arch_cfg"], "n_params": nn["n_params"],
        "selection_discipline": {
            "architecture_and_epoch_selected_on": "fold_2023 ONLY",
            "fold_2024_role": "judgment fold; selects nothing",
            "deployment_epoch": em["deployment_epoch"]},

        "fold_2023": {
            "n_valid": em["row_alignment"]["n_valid_2023"],
            "nn_own_es": {k: nn["fold_2023_own_es"][k] for k in
                          ("brier_by_seed", "score_by_seed", "mean_brier",
                           "mean_score", "seed_std_score_points", "valid_all",
                           "valid_R", "valid_F")},
            "catboost_base_no_offset_seeds_42_7":
                em["catboost"]["fold_2023_base_no_offset"],
            "gap_score_points": m4["fold_2023_gap_score_points"],
            "mix_w_star": m4["fold_2023_curve"]["w_star"],
            "mix_gain_over_cat_score_points":
                m4["fold_2023_curve"]["gain_over_cat_only_score_points"],
            "convexity_score_points":
                m4["fold_2023_curve"]["convexity_score_points"]},

        "fold_2024": {
            "n_valid": em["row_alignment"]["n_valid_2024"],
            "nn_HONEST_epoch8": {k: nn["fold_2024_deploy_epoch_HONEST"][k] for k in
                                 ("brier_by_seed", "score_by_seed", "mean_brier",
                                  "mean_score", "seed_std_score_points",
                                  "valid_all", "valid_R", "valid_F")},
            "nn_own_es_FOLD_ORACLE_diagnostic":
                {k: nn["fold_2024_own_es_FOLD_ORACLE"][k] for k in
                 ("brier_by_seed", "score_by_seed", "mean_brier", "mean_score",
                  "seed_std_score_points")},
            "catboost_offset_S400_self":
                em["catboost"]["fold_2024_offset_S400_self"],
            "catboost_base_no_offset":
                em["catboost"]["fold_2024_base_no_offset"],
            "brier": nn["fold_2024_deploy_epoch_HONEST"]["mean_brier"],
            "bss": nn["fold_2024_deploy_epoch_HONEST"]["valid_all"]["bss"],
            "auc": nn["fold_2024_deploy_epoch_HONEST"]["valid_all"]["auc"],
            "score": nn["fold_2024_deploy_epoch_HONEST"]["mean_score"]},

        "ensemble_measurement": {
            "1_gap": em["measure_1_gap"],
            "2_rho": em["measure_2_rho"],
            "3_convexity": em["measure_3_convexity"],
            "4_honest_transfer": m4},

        "pre_registered_decision": {
            "readings_as_registered": [
                "gap < 58 AND rho < 0.90  -> ensemble promising",
                "gap < 58, rho >= 0.90    -> diversity too thin, change arch",
                "gap 58~150, rho < 0.85   -> ambiguous, measure convexity",
                "gap > 150                -> grow the NN; mixing loses as is"],
            "measured_abs_gap_score_points": round(gap, 5),
            "measured_rho_prob": em["measure_2_rho"]["vs_offset_candidate"][
                "mean_of_seed_matched"]["prob"],
            "measured_convexity_score_points": conv,
            "formula_check_convexity_gt_half_gap": {
                "convexity": conv, "half_gap": round(gap / 2, 5),
                "passes": bool(conv > gap / 2)},
            "reading_hit": "gap > 150 -> fourth reading",
            "NOT_EDITED_AFTER_SEEING_VALUES": True},

        "nn_sigma": em["nn_sigma"],
        "secondary_arch_C3": {
            "why": "C3 keeps the id embeddings LARGE (dim 32) where the primary "
                   "D0 shrinks them to dim 2.  Measured to see whether embedding "
                   "richness buys diversity.  NOT a selection.",
            "fold_2024_gap": em3["measure_1_gap"][
                "PRIMARY_honest_vs_offset_candidate"],
            "rho": em3["measure_2_rho"]["vs_offset_candidate"][
                "mean_of_seed_matched"],
            "convexity_score_points": em3["measure_3_convexity"][
                "PRIMARY_seed_matched"]["convexity_score_points"],
            "w_star": em3["measure_3_convexity"]["PRIMARY_seed_matched"]["w_star"],
            "nn_sigma": em3["nn_sigma"]},

        "recon_fold2023_seed42_all_configs": recon,
        "recon_3seed_confirmation_fold2023": confirm,
        "row_alignment": em["row_alignment"],
        "catboost_provenance": em["catboost_provenance"],

        "train_time_min": round(total_fit_min, 2),
        "n_fits_total": len(all_res),
        "inference_time_sec": time_inference(PRIMARY),
        "gpu": json.load(open(os.path.join(
            ROOT, "res", f"res_{PRIMARY}_fold2024_seed42.json")))["gpu"],
        "src_sha256": em["src_sha256"],
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "torch": torch.__version__},
    }
    C.jdump(out, os.path.join(ROOT, "metrics.json"))
    print(json.dumps({"train_time_min": out["train_time_min"],
                      "n_fits": out["n_fits_total"],
                      "inference": out["inference_time_sec"],
                      "reading": out["pre_registered_decision"]}, indent=1,
                     default=C._default))


if __name__ == "__main__":
    main()
