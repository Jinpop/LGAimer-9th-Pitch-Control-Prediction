"""feat.parquet 생성 — ins_* + 신규 6계열. as-of 규율 준수 (연도 Y 는 season<Y 만)."""
import time, sys
import numpy as np
import pandas as pd
import feats as FT

t0 = time.time()
DT = {"season": "int16", "game_month": "int8", "game_dayofweek": "int8", "inning": "int8",
      "balls_before": "int8", "strikes_before": "int8", "outs_before": "int8",
      "run_top_before": "int16", "run_bot_before": "int16", "run_total_before": "int16",
      "score_diff_home": "int16", "score_diff_pitcher_team": "int16",
      "runner_on_1b": "int8", "runner_on_2b": "int8", "runner_on_3b": "int8",
      "num_runners_on": "int8", "pitcher_id": "int32", "batter_id": "int32",
      "pitcher_hand": "int8", "batter_hand": "int8",
      "pitcher_team_id": "int16", "batter_team_id": "int16",
      "asof_pitcher_n": "int32", "asof_batter_n": "int32",
      "asof_pitcher_pitchmix_n": "int32", "control_success": "int8"}
df = pd.read_csv("data/data/train.csv", encoding="utf-8-sig", dtype=DT)
print(f"[load] {df.shape} {time.time()-t0:.0f}s")

seasons = sorted(df[FT.SEASON].unique())

# ---- ins 앵커 (구성상 season<S 만 참조 → 누수 없음)
tp = FT.build_ins_anchor(df, FT.PID, FT.P_N, FT.P_RATES, seasons)
tb = FT.build_ins_anchor(df, FT.BID, FT.B_N, FT.B_RATES, seasons)
print(f"[anchor] p={len(tp)} b={len(tb)} {time.time()-t0:.0f}s")

ins_p = FT.apply_ins(df, tp, FT.P_N, FT.P_RATES, FT.P_SHORT, "ins_p")
ins_b = FT.apply_ins(df, tb, FT.B_N, FT.B_RATES, FT.B_SHORT, "ins_b")
INS = pd.concat([ins_p, ins_b], axis=1).astype(np.float32)
print(f"[ins] {INS.shape} {time.time()-t0:.0f}s")

# ---- 타겟 유래 표(F4/F5)는 연도별로 season<Y 로만
parts = []
for Y in seasons:
    m = (df[FT.SEASON] == Y).to_numpy()
    hist = df.loc[df[FT.SEASON] < Y, [FT.SEASON, FT.PID, FT.PTEAM, FT.BTEAM, FT.TGT]]
    tt = FT.build_team_table(hist) if len(hist) else {}
    et = FT.build_eb_table(hist) if len(hist) else {}
    parts.append(FT.apply_new(df.loc[m], ins_p.loc[m], tt, et))
    print(f"  [new {Y}] {int(m.sum())} rows  {time.time()-t0:.0f}s")
NEW = pd.concat(parts).reindex(df.index)
print(f"[new] {NEW.shape} {time.time()-t0:.0f}s")

keep = [c for c in df.columns if c != "row_id"]
out = pd.concat([df[keep], INS, NEW], axis=1)
out.to_parquet("feat.parquet", index=False)
print(f"[save] feat.parquet {out.shape} {time.time()-t0:.0f}s")
print("families:", {k: len([c for c in NEW.columns if c.startswith(v)])
                    for k, v in FT.FAMILIES.items()})
