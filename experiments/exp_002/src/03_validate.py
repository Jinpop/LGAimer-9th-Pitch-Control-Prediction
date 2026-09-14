# exp_002 / 03: 독립 검증 (config.validation.checks)
#  1) 매핑 경기 표본 30개 풀-시퀀스 재확인 (01의 지문과 별개로 원자료에서 배열 직접 비교)
#  2) 투수 손 일치율 100% 확인 (최종 맵 기준, 원자료 재계산)
#  3) 커버리지: train 투구 행 기준 프로파일 보유 비율
# 표본 추출만 seed 고정 (검증 표본 선택용 — 산출물 자체는 결정론적)
import pandas as pd
import numpy as np
import json
import os
import time

DATA = os.path.expanduser("~/SungjinPark/data")
OUT = os.path.expanduser("~/SungjinPark/experiments/exp_002/out")
t0 = time.time()


def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)


gmap = pd.read_csv(os.path.join(OUT, "game_date_map.csv"))
pmap = pd.read_csv(os.path.join(OUT, "pitcher_id_map.csv"))
prof = pd.read_csv(os.path.join(OUT, "pitcher_static_profile_2024.csv"))

train = pd.read_csv(
    os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
    usecols=["row_id", "season", "game_type", "inning", "top_bottom",
             "balls_before", "strikes_before", "outs_before",
             "pitcher_id", "batter_id", "pitcher_hand", "batter_hand"])
tm = pd.read_csv(
    os.path.join(DATA, "trackman_history.csv"), encoding="utf-8-sig",
    usecols=["trackman_game_id", "pitch_no", "inning", "top_bottom",
             "balls_before", "strikes_before", "outs_before",
             "pitcher_trackman_id", "batter_trackman_id", "pitcher_hand", "batter_hand"])
tm["tb"] = tm["top_bottom"].map({"Top": "T", "Bottom": "B"})
HAND = {"Left": 1, "Right": 2}
tm["p_hand"] = tm["pitcher_hand"].map(HAND)
tm["b_hand"] = tm["batter_hand"].map(HAND)
train_pos = {r: i for i, r in enumerate(train["row_id"].values)}

# ---- check 1: 30경기 표본 풀-시퀀스 재확인 ----
rng = np.random.default_rng(42)
sample = gmap.sort_values("train_block").sample(n=30, random_state=42)
pdict = dict(zip(pmap["pitcher_id"], pmap["pitcher_trackman_id"]))
n_ok, fails = 0, []
for _, r in sample.iterrows():
    i0, i1 = train_pos[r["row_id_start"]], train_pos[r["row_id_end"]]
    A = train.iloc[i0:i1 + 1]
    B = tm[tm["trackman_game_id"] == r["trackman_game_id"]].sort_values("pitch_no")
    ok = (len(A) == len(B)
          and (A["inning"].values == B["inning"].values).all()
          and (A["top_bottom"].values == B["tb"].values).all()
          and (A["balls_before"].values == B["balls_before"].values).all()
          and (A["strikes_before"].values == B["strikes_before"].values).all()
          and (A["outs_before"].values == B["outs_before"].values).all()
          and (A["pitcher_hand"].values == B["p_hand"].values).all()
          and (A["batter_hand"].values == B["b_hand"].values).all())
    # 매핑 사전과 행 단위 대응 일치 (맵에 있는 투수만)
    map_ok = all(pdict.get(a) is None or pdict[a] == b
                 for a, b in zip(A["pitcher_id"].values, B["pitcher_trackman_id"].values))
    if ok and map_ok:
        n_ok += 1
    else:
        fails.append(str(r["train_block"]))
log(f"check1 30-game full-sequence re-verify: {n_ok}/30 pass; fails={fails}")

# ---- check 2: 투수 손 일치율 ----
tr_ph = train.groupby("pitcher_id")["pitcher_hand"].agg(lambda s: s.mode().iat[0])
tm_ph = tm.groupby("pitcher_trackman_id")["p_hand"].agg(lambda s: s.mode().iat[0])
hm = pmap.assign(h_tr=pmap["pitcher_id"].map(tr_ph),
                 h_tm=pmap["pitcher_trackman_id"].map(tm_ph))
hand_rate = float((hm["h_tr"] == hm["h_tm"]).mean())
log(f"check2 pitcher hand agreement: {hand_rate:.4f} ({(hm['h_tr']==hm['h_tm']).sum()}/{len(hm)})")

# ---- check 3: 커버리지 (train 투구 행 기준 프로파일 보유) ----
has_prof = set(prof.loc[prof["career_n_pitches"].notna(), "pitcher_id"])
res = {
    "check1_sample30_pass": n_ok,
    "check1_fails": fails,
    "check2_hand_agreement": hand_rate,
    "check3_rows_with_profile_pct": round(100 * train["pitcher_id"].isin(has_prof).mean(), 2),
    "check3_rows2024_with_profile_pct": round(
        100 * train.loc[train["season"] == 2024, "pitcher_id"].isin(has_prof).mean(), 2),
    "check3_rowsR_with_profile_pct": round(
        100 * train.loc[train["game_type"] == "R", "pitcher_id"].isin(has_prof).mean(), 2),
    "check3_pitchers_with_profile": len(has_prof),
}
log(json.dumps(res, indent=2))
with open(os.path.join(OUT, "validation.json"), "w") as f:
    json.dump(res, f, indent=2)
log("DONE 03_validate")
