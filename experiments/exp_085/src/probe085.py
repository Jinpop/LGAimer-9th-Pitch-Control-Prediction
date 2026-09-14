"""exp_085 stage 0 probe — 프레임 정렬·결측 구조·ID 카디널리티 확인만. 학습 없음."""
import json
import os

import numpy as np
import pandas as pd

H = "/home/<user>/SungjinPark"
B = f"{H}/exp085"
SPD = f"{H}/sp_def_recv"
os.makedirs(f"{B}/out", exist_ok=True)

FSP = [l.strip() for l in open(f"{SPD}/features_sp.txt", encoding="utf-8") if l.strip()]
CATS5 = ["base_state", "game_type", "pitcher_team_id", "batter_team_id", "cnt_state"]
DENSE = [c for c in FSP if c not in CATS5]
print("FSP", len(FSP), "DENSE", len(DENSE))

df = pd.read_parquet(f"{B.replace('exp085', 'exp060_nn')}/cache/frame99.parquet")
print("frame99", df.shape)

tr = pd.read_csv(f"{H}/data/train.csv",
                 usecols=["row_id", "season", "game_type", "pitcher_id",
                          "batter_id", "pitcher_hand", "batter_hand",
                          "control_success"])
print("train.csv", tr.shape)

r_f = df.row_id.to_numpy().astype(str)
r_t = tr.row_id.to_numpy().astype(str)
same = bool(np.array_equal(r_f, r_t))
print("frame99 order == train.csv natural order:", same)

v_f = df[df.season == 2024].row_id.to_numpy().astype(str)
v_t = tr[tr.season == 2024].row_id.to_numpy().astype(str)
srt = np.sort(v_t)
print("n valid", len(v_f), len(v_t))
print("valid frame99 == valid train nat:", bool(np.array_equal(v_f, v_t)))
print("valid train nat == sorted:", bool(np.array_equal(v_t, srt)))
ids = np.asarray(np.load(f"{H}/arms/ids.npy", allow_pickle=True)).astype(str)
print("arms/ids == valid train nat:", bool(np.array_equal(ids, v_t)))

season = df.season.to_numpy()
gt = df.game_type.to_numpy()
trm = (season <= 2023) & ~((season <= 2022) & (gt == "F"))
vam = season == 2024
print("n_train", int(trm.sum()), "n_valid", int(vam.sum()),
      "masked_out", int(len(df) - trm.sum() - vam.sum()))

nanrep = {}
for c in DENSE:
    a = df[c].to_numpy(np.float64)
    ntr = float(np.isnan(a[trm]).mean())
    nva = float(np.isnan(a[vam]).mean())
    if ntr > 0 or nva > 0:
        nanrep[c] = [round(ntr, 6), round(nva, 6)]
print("dense cols with NaN:", len(nanrep), "of", len(DENSE))
print(json.dumps(nanrep, indent=0)[:3000])

# ID 카디널리티 / 미등장률
for c in ["pitcher_id", "batter_id"]:
    lv = pd.unique(tr.loc[trm, c])
    unseen = ~tr.loc[vam, c].isin(set(lv.tolist()))
    print(c, "levels(train<=2023)", len(lv),
          "valid unseen rows", int(unseen.sum()),
          f"{100*float(unseen.mean()):.4f}%",
          "valid unseen distinct ids",
          int(tr.loc[vam, c][unseen.to_numpy()].nunique()))

P6 = [f"p6_{t}" for t in ("curve", "cutter", "fastball", "offspeed", "sinker", "slider")]
p6 = df[P6].to_numpy(np.float64)
allnan = np.isnan(p6).all(axis=1)
print("p6 all-NaN rows:", int(allnan.sum()), "train:", int(allnan[trm].sum()),
      "valid:", int(allnan[vam].sum()))
print("p6 any-NaN rows:", int(np.isnan(p6).any(axis=1).sum()))

for c in ["pitcher_hand", "batter_hand"]:
    print(c, "train levels", sorted(pd.unique(tr.loc[trm, c]).tolist()),
          "valid levels", sorted(pd.unique(tr.loc[vam, c]).tolist()),
          "nan", int(tr[c].isna().sum()))

for c in CATS5:
    print(c, "train levels", int(df.loc[trm, c].nunique()),
          "valid unseen rows",
          int((~df.loc[vam, c].isin(set(df.loc[trm, c].unique().tolist()))).sum()))
