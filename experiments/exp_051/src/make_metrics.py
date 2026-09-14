"""exp_051 — assemble the local metrics.json from retrieved server artefacts.

Run LOCALLY after retrieving res/ from the server:
    python3 make_metrics.py <exp_051_local_dir>

Citation convention (EXPERIMENTS.md 산출물 인용 규약): every claim about
another file carries the path read and the value seen; nothing is transcribed
from memory.
"""

import json
import os
import sys

REF_BRIER = 0.2498069
SEEDS = (42, 7, 2024)


def score_of(b):
    return 100000.0 * (1.0 - b / REF_BRIER)


def main():
    d = sys.argv[1]
    res_dir = os.path.join(d, "res")
    with open(os.path.join(res_dir, "stage0_gates.json")) as f:
        s0 = json.load(f)
    with open(os.path.join(res_dir, "summary.json")) as f:
        s2 = json.load(f)
    per = {}
    for s in SEEDS:
        with open(os.path.join(res_dir,
                               f"res_exp051_u2024_seed{s}.json")) as f:
            per[s] = json.load(f)

    r42 = per[42]
    m = {
        "exp_id": "exp_051",
        "date": "2026-08-15",
        "title": ("cells upgraded 47 -> 63 columns (+16 ins_*): OOF "
                  "regenerated, 67-col stacker refit at 1400, fold_2024, "
                  "seeds 42/7/2024"),
        "scope": "stage 0 + stage 1 + aggregation.  No deployment package.",
        "score_conversion": {
            "formula": "100000 * (1 - Brier / 0.2498069)",
            "ref_brier": REF_BRIER, "decimals": 5,
            "sensitivity": ("Brier 1e-06 = 0.4003 points; a 6-decimal Brier "
                            "converts with +/-0.2002 of rounding uncertainty."
                            "  The Brier is the raw quantity.")},
        "unit": {
            "fold": "fold_2024", "train_upto": 2023, "valid_season": 2024,
            "n_train": r42["n_train_full"], "n_train_mode": r42["n_train_mode"],
            "n_valid": r42["n_valid"], "n_valid_R": r42["n_valid_R"],
            "n_valid_F": r42["n_valid_F"],
            "arm_base": "C5_B3", "seeds": list(SEEDS),
            "cell_n_features": r42["cell_n_features"],
            "stacker_n_features": r42["stacker_n_features"],
            "stacker_rounds": 1400},
        "cell_iter_pinned": {
            "policy": ("the four cell iterations were READ from the control "
                       "OOF's artefact, never re-chosen (config "
                       "셀_iter_규약; exp_034 precedent)"),
            "read_from": r42["cell_iter_source"],
            "values": r42["cell_iter_used"],
            "note_carried_along_variable": (
                "the 63-col cells' own grid argmins are recorded per seed in "
                "cells_vs_AH_47col (exp051_argmin_iter) as a carried-along "
                "observation only; nothing selects on them"),
        },
        "stage0": s0,
        "A0_control_gate_vs_exp048_A1": {
            str(s): per[s]["A0_control_gate_vs_exp048_A1"] for s in SEEDS},
        "fold_2024": s2["by_round"],
        "headline_fold_2024_at_1400": {
            "brier_A0_control": s2["by_round"]["1400"]["mean_brier_A0"],
            "score_A0_control": s2["by_round"]["1400"]["mean_score_A0"],
            "brier_T1": s2["by_round"]["1400"]["mean_brier_T1"],
            "score_T1": s2["by_round"]["1400"]["mean_score_T1"],
            "mean_delta_score": s2["by_round"]["1400"]["mean_delta_score"]},
        "pre_registered_decision": s2["pre_registered_decision"],
        "cells_47_vs_63": s2["cells_47_vs_63"],
        "oof_movement_vs_AH": s2["oof_movement_vs_AH"],
        "stacker_importance_top20_seed42": dict(list(
            r42["stacker_importance_PredictionValuesChange"].items())[:20]),
        "sha256_seed42_run": r42["sha256"],
        "versions": r42["versions"],
        "changed_vs_control": {
            "the_one": ("the four CELL models' input: 47 -> 63 columns "
                        "(the 16 ins_* columns)"),
            "dependent_reproductions_forced_by_the_one": [
                "the four OOF columns regenerated (same forward-chaining, "
                "same season blocks, same pinned cell iterations)",
                "the 67-column stacker refit at 1400 on the new OOF "
                "(input list unchanged: base 47 + p_* 4 + ins 16, exp_048 "
                "A1's exact order)"],
            "carried_along": [
                "cell fit time (63 cols)",
                "the OOF values themselves (measured in oof_movement_vs_AH)",
                "the OOF distribution the stacker receives",
                "63-col cell grid argmins differ from the pinned iterations "
                "(recorded, not acted on)"],
            "held_fixed": [
                "cell iterations (pinned from local_reference_AH.json)",
                "stacker rounds 1400, input column list and order (67)",
                "masks, fold definition, seeds 42/7/2024, CatBoost params, "
                "library versions",
                "the ins matrix itself (stage-0 regeneration byte-checked "
                "against exp_048's recorded input shas and value-checked "
                "against its recorded season stats)"],
        },
        "runtime_min": {str(s): per[s]["minutes"] for s in SEEDS},
        "oof_min": {str(s): per[s]["oof_min"] for s in SEEDS},
        "stage0_min": s0["minutes"],
        "train_time_min": max(per[s]["minutes"] for s in SEEDS),
        "inference_time_sec": None,
        "note_on_inference_time": ("not measured: no deployment package was "
                                   "built (out of scope)."),
        "artifact_paths_server": {
            "oof": ["~/SungjinPark/experiments/exp_051/oof/"
                    f"stack_fold2024_seed{s}.npz" for s in SEEDS],
            "pred": ["~/SungjinPark/experiments/exp_051/pred/"
                     f"pred_exp051_u2024_seed{s}_at{r}.npy"
                     for s in SEEDS for r in (655, 1000, 1400)],
            "ins_matrix": "~/SungjinPark/experiments/exp_051/"
                          "ins_matrix_fold2024.npz",
            "res": "~/SungjinPark/experiments/exp_051/res/"},
    }
    with open(os.path.join(d, "metrics.json"), "w") as f:
        json.dump(m, f, indent=1)
    print("wrote", os.path.join(d, "metrics.json"))


if __name__ == "__main__":
    main()
