# exp_000_eda / 03: 파일 순서 이상 지점·게임 블록 구조 조사
import pandas as pd
import numpy as np
import os

DATA = "<PROJECT_ROOT>/data"
OUT = "<PROJECT_ROOT>/experiments/exp_000_eda/profile"

cols = ["row_id", "season", "game_month", "game_dayofweek", "game_type", "inning",
        "top_bottom", "pitcher_team_id", "batter_team_id", "pitcher_id", "batter_id",
        "balls_before", "strikes_before", "outs_before", "asof_pitcher_n"]
train = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig", usecols=cols)

# ---- 1) 월 역행 지점 ----
train["month_drop"] = (train["game_month"].diff() < 0) & (train["season"].diff() == 0)
drops = train.index[train["month_drop"]]
print("월 역행 지점 수:", len(drops))
print("연도별:", train.loc[drops, "season"].value_counts().sort_index().to_dict())
# 앞뒤 문맥 몇 개
for i in list(drops[:3]):
    print(train.iloc[i-2:i+3][["row_id","season","game_month","game_dayofweek","game_type","inning","top_bottom","pitcher_team_id","batter_team_id"]].to_string())
    print("---")

# ---- 2) '게임 블록' 추정: (month, dayofweek, game_type, 팀 페어) 변화 지점으로 블록 나누기 ----
team_pair = train.apply(lambda r: tuple(sorted((r["pitcher_team_id"], r["batter_team_id"]))), axis=1)
train["team_pair"] = team_pair
key = train[["season", "game_month", "game_dayofweek", "game_type"]].astype(str).agg("|".join, axis=1) + "|" + train["team_pair"].astype(str)
train["block"] = (key != key.shift(1)).cumsum()
bl = train.groupby("block").agg(n=("row_id", "count"), season=("season", "first"),
                                gt=("game_type", "first"))
print("\n블록 수:", len(bl), " (6시즌 실제 경기수 대략 720x6+퓨처스 ~ 6000+)")
print("블록 크기 분포:")
print(bl["n"].describe().round(1).to_string())
print("게임당 투구수로 그럴듯한가 (250~350):", ((bl["n"] > 200) & (bl["n"] < 400)).mean().round(3))
print("시즌·타입별 블록 수:")
print(bl.groupby(["season", "gt"]).size().unstack().to_string())

# 블록 내 이닝 진행이 1->9로 자연스러운지 샘플 확인
b0 = bl[(bl["n"] > 250) & (bl["n"] < 350)].index[5]
d = train[train["block"] == b0]
print("\n샘플 블록 이닝 시퀀스 (앞 30):", d["inning"].head(30).tolist())
print("이닝 단조증가:", d["inning"].is_monotonic_increasing, " max inning:", d["inning"].max())

# ---- 3) 블록(=게임 추정) 단위로 시즌 내 순서 확인: 같은 날짜(월,요일)가 흩어져 있나 ----
bl2 = train.groupby("block").agg(season=("season","first"), month=("game_month","first"),
                                 dow=("game_dayofweek","first"), gt=("game_type","first"))
for s in [2019, 2023]:
    d = bl2[bl2.season == s]
    print(f"\nseason {s}: 블록 월 시퀀스 앞 40:", d["month"].head(40).tolist())
