"""Emit the Stage B job list from Stage A results.

Iteration policy (config iter_policy.stage_B): each promoted arm's fold_2024
round count is its OWN fold_2023 argmin scaled by the training-row ratio
1,116,277 / 870,752 = 1.281969.  Computed PER ARM because the families run at
different learning rates.  fold_2024 valid is never consulted.

L1 is always included so every arm sits on the same yardstick.
"""
import json
import os
import sys

OUT = os.path.expanduser("~/SungjinPark/experiments/exp_027")
SCALE = 1.281969
SEEDS = [42, 7, 2024]


def main():
    arms = sys.argv[1:]
    assert arms, "usage: make_stageB.py ARM [ARM ...]   (L1 added automatically)"
    if "L1" not in arms:
        arms = ["L1"] + arms
    lines, table = [], {}
    for a in arms:
        p = os.path.join(OUT, "res", f"res_{a}_seed42_fold_2023.json")
        r = json.load(open(p))
        assert not r["argmin_at_cap"], f"{a}: argmin at cap -> arm is invalid"
        k = int(round(r["argmin_iter"] * SCALE))
        table[a] = {"fold_2023_argmin": r["argmin_iter"], "scale": SCALE,
                    "fold_2024_iter": k}
        for s in SEEDS:
            lines.append(f"{a} {s} {k}")
    json.dump(table, open(os.path.join(OUT, "stageB_iters.json"), "w"), indent=2)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
