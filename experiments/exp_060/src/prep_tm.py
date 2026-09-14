"""exp_060 stage0 — trackman 행에 train pitcher_id·lab(6구종) 부여 → work/tm_mapped.parquet

매핑 출처 (전부 기존 검증 자산, 새 추정 없음):
- pitcher_trackman_id → pitcher_id : exp_052/ref_exp002/pitcher_id_map.csv (685명,
  exp_052 G0a에서 exp_002 지도와 685/685 일치·투수손 일치율 1.000 검증)
- top_bottom: Top→T, Bottom→B / hand: Left→1, Right→2 (pitcher_id_map의
  hand_train↔hand_tm 대각 일치로 확인: 1↔1 174명, 2↔2 511명)
- lab: tagged_pitch_type의 무모호 표기만 6 TYPES로. 모호·희귀 표기는 드롭(기록).
"""
import os
import time

import numpy as np
import pandas as pd

B = os.path.expanduser("~/SungjinPark/exp060_nn")
t0 = time.time()

tm = pd.read_csv(
    os.path.expanduser("~/SungjinPark/data/trackman_history.csv"),
    usecols=["season", "inning", "top_bottom", "balls_before", "strikes_before",
             "outs_before", "pitcher_trackman_id", "batter_hand", "pitcher_hand",
             "tagged_pitch_type"],
    encoding="utf-8-sig")
n_raw = len(tm)

pmap = pd.read_csv(os.path.expanduser(
    "~/SungjinPark/experiments/exp_052/ref_exp002/pitcher_id_map.csv"))
lut = dict(zip(pmap.pitcher_trackman_id, pmap.pitcher_id))
tm["pitcher_id"] = tm.pitcher_trackman_id.map(lut)

tm["top_bottom"] = tm.top_bottom.map({"Top": "T", "Bottom": "B"})
tm["pitcher_hand"] = tm.pitcher_hand.map({"Left": 1, "Right": 2})
tm["batter_hand"] = tm.batter_hand.map({"Left": 1, "Right": 2})

LAB = {"Fastball": "fastball", "Four-Seam": "fastball",
       "Sinker": "sinker", "SInker": "sinker",
       "Cutter": "cutter", "Slider": "slider", "Curveball": "curve",
       "ChangeUp": "offspeed", "Changeup": "offspeed", "Splitter": "offspeed"}
tm["lab"] = tm.tagged_pitch_type.map(LAB)

n_nopit = int(tm.pitcher_id.isna().sum())
n_nolab = int(tm.lab.isna().sum())
tm = tm.dropna(subset=["pitcher_id", "lab", "pitcher_hand", "batter_hand",
                       "top_bottom"]).copy()
tm["pitcher_id"] = tm.pitcher_id.astype(np.int64)
out = tm[["season", "inning", "top_bottom", "balls_before", "strikes_before",
          "outs_before", "batter_hand", "pitcher_hand", "pitcher_id", "lab"]]
os.makedirs(f"{B}/work", exist_ok=True)
out.to_parquet(f"{B}/work/tm_mapped.parquet")

print(f"raw {n_raw} -> kept {len(out)} "
      f"(drop nopitcher {n_nopit}, nolab {n_nolab}, overlap counted once)")
print("season dist:", out.season.value_counts().sort_index().to_dict())
print("lab dist:", out.lab.value_counts().to_dict())
print(f"n_pitchers {out.pitcher_id.nunique()}  elapsed {time.time()-t0:.1f}s")
