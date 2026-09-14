"""exp_037 / between C and D — the selection, made on fold_2023 alone.

Reads every completed stage-1 (fold_2023, seed 42) result, ranks the non-control
arms by their fold_2023 valid_all Brier and writes selection.json.  fold_2024 is
not read here and cannot be: no fold_2024 artefact exists when this runs.

The chosen arm's own stage-1 argmins become its stage-2 iteration counts through
b3_feat.py's row-ratio rule (x1.281957 for the stacker on unit 2024).  The
control R0 does the same from ITS stage 1.  Neither consults a fold_2024 curve.

fold_2023 Brier levels are NOT converted to LB points: the conversion reference
0.2498069 is the 2025 test base rate and applying it to a 2023 validation set
would manufacture a number that means nothing.  DELTAS are converted, because a
delta of Brier maps to a delta of points at a fixed reference regardless of
level.
"""
import json
import os
import sys

from arms import ARMS, DUPLICATE_OF, EXPECTED_N

REF_BRIER = 0.2498069


def score_delta(d):
    """Brier delta -> LB point delta.  Improvement (negative Brier delta) is a
    POSITIVE point delta."""
    return -100000.0 * d / REF_BRIER


def main():
    out = os.path.abspath(sys.argv[sys.argv.index("--out") + 1])
    res_dir = os.path.join(out, "res")
    got = {}
    for a in ARMS:
        p = os.path.join(res_dir, f"res_{a}_u2023_seed42_st1.json")
        if os.path.exists(p):
            with open(p) as f:
                r = json.load(f)
            if r.get("any_argmin_at_cap"):
                r["_invalid"] = "argmin reached cap"
            got[a] = r
    # L_no_i3 is R3 by construction; alias it rather than refit it.
    for dup, src in DUPLICATE_OF.items():
        if dup not in got and src in got:
            r = dict(got[src])
            r["_alias_of"] = src
            r["arm"] = dup
            got[dup] = r

    assert "R0" in got, "the control never completed; selection is impossible"
    b0 = got["R0"]["brier_full"]

    table = []
    for a, r in got.items():
        d = r["brier_full"] - b0
        table.append({
            "arm": a, "n_feats": r.get("n_feats_used", EXPECTED_N.get(a)),
            "brier_fold2023": r["brier_full"],
            "delta_vs_R0_brier": d,
            "delta_vs_R0_score": float(f"{score_delta(d):.5f}"),
            "stack_argmin": r["stack_argmin_iter"],
            "cell_argmin": r["cell_argmin_iter"],
            "minutes": r.get("minutes"),
            "invalid": r.get("_invalid"),
            "alias_of": r.get("_alias_of"),
            "stage": ARMS[a]["stage"],
        })
    table.sort(key=lambda x: x["brier_fold2023"])

    cand = [t for t in table if t["arm"] != "R0" and not t["invalid"]]
    best = cand[0] if cand else None

    sel = {
        "selected_on": "fold_2023 (train<=2022 -> valid 2023), seed 42, "
                       "valid_all Brier",
        "fold_2024_consulted": False,
        "control": {"arm": "R0", "brier_fold2023": b0},
        "table": table,
        "best_arm": best["arm"] if best else None,
        "best_brier_fold2023": best["brier_fold2023"] if best else None,
        "best_delta_vs_R0_brier": best["delta_vs_R0_brier"] if best else None,
        "best_delta_vs_R0_score": best["delta_vs_R0_score"] if best else None,
        "n_arms_completed": len(got),
        "promote_arms": ["R0"] + ([best["arm"]] if best else []),
        "note": ("if best_arm is an alias (L_no_i3 == R3) the promotion runs "
                 "the source arm's name, since the column set is identical"),
    }
    if best and best["alias_of"]:
        sel["promote_arms"] = ["R0", best["alias_of"]]
        sel["best_arm_promoted_as"] = best["alias_of"]

    with open(os.path.join(out, "selection.json"), "w") as f:
        json.dump(sel, f, indent=1)
    print(json.dumps({"best_arm": sel["best_arm"],
                      "promote_arms": sel["promote_arms"],
                      "delta_score": sel["best_delta_vs_R0_score"]}), flush=True)
    for t in table:
        print(f"  {t['arm']:12s} n={t['n_feats']:3d} "
              f"brier {t['brier_fold2023']:.10f} "
              f"dScore {t['delta_vs_R0_score']:+9.3f} "
              f"{'INVALID:' + str(t['invalid']) if t['invalid'] else ''}"
              f"{' (alias of ' + t['alias_of'] + ')' if t['alias_of'] else ''}",
              flush=True)


if __name__ == "__main__":
    main()
