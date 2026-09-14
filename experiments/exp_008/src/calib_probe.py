"""exp_008 supplementary — oracle constant-logit-offset headroom for the RF preds.

Run ON THE SERVER after train.py (it reads pred_R{1,2}_fold_2024.npy).
The offset is fitted ON THE VALID ROWS on purpose: this is an ORACLE upper bound
on what any constant recalibration could buy, used only to decide whether a
calibration step is worth building for the RF family. It is NOT a model and must
never be used to produce a submission.

    ~/SungjinPark/venv/bin/python calib_probe.py
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

D = "/home/<user>/SungjinPark/experiments/exp_008/"
TRAIN = "/home/<user>/SungjinPark/data/train.csv"


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def main():
    df = pd.read_csv(TRAIN, encoding="utf-8-sig", usecols=["season", "game_type", "control_success"])
    va = df["season"] == 2024
    y = df.loc[va, "control_success"].to_numpy().astype(float)
    is_r = (df.loc[va, "game_type"] == "R").to_numpy()

    preds = {tag: np.load(D + f"pred_{tag}_fold_2024.npy") for tag in ("R1", "R2")}
    for tag, p in preds.items():
        for seg, m in (("all", np.ones_like(is_r, dtype=bool)), ("R", is_r)):
            q = np.clip(p[m], 1e-6, 1 - 1e-6)
            yy = y[m]
            lo = np.log(q / (1 - q))
            res = minimize_scalar(
                lambda d: brier(1 / (1 + np.exp(-(lo + d))), yy),
                bounds=(-1, 1),
                method="bounded",
            )
            print(
                tag, seg,
                "brier", round(brier(p[m], yy), 6),
                "oracle_offset", round(res.x, 4),
                "oracle_brier", round(res.fun, 6),
                "gain", round(res.fun - brier(p[m], yy), 6),
            )
    print("corr(R1,R2)", round(float(np.corrcoef(preds["R1"], preds["R2"])[0, 1]), 4))
    print("R1 pred std", round(float(preds["R1"].std()), 5),
          "range", round(float(preds["R1"].min()), 4), round(float(preds["R1"].max()), 4))


if __name__ == "__main__":
    main()
