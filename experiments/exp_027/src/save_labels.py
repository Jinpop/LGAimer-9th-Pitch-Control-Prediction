"""Dump the validation label vectors so the Stage C blend never re-reads the
csv (and so the blend cannot accidentally touch a training row)."""
import os
import numpy as np
import pandas as pd

DATA = os.path.expanduser("~/SungjinPark/data/train.csv")
OUT = os.path.expanduser("~/SungjinPark/experiments/exp_027")
df = pd.read_csv(DATA, encoding="utf-8-sig",
                 usecols=["season", "game_type", "control_success"])
s = df["season"].to_numpy()
for yr in (2023, 2024):
    m = s == yr
    np.save(os.path.join(OUT, f"y_fold_{yr}.npy"),
            df.loc[m, "control_success"].to_numpy(dtype=np.float64))
    np.save(os.path.join(OUT, f"isR_fold_{yr}.npy"),
            (df.loc[m, "game_type"].to_numpy() == "R"))
    print(yr, int(m.sum()), float(df.loc[m, "control_success"].mean()))
