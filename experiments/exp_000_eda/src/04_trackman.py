# exp_000_eda / 04: trackman_history 프로파일 + train과의 조인 가능성
import pandas as pd
import numpy as np
import os

DATA = "<PROJECT_ROOT>/data"
OUT = "<PROJECT_ROOT>/experiments/exp_000_eda/profile"

tm = pd.read_csv(os.path.join(DATA, "trackman_history.csv"), encoding="utf-8-sig")
print("shape:", tm.shape)
print(tm.dtypes.to_string())

# ---- 1) 결측·기본 프로파일 ----
prof = pd.DataFrame({
    "dtype": tm.dtypes.astype(str),
    "missing_pct": (tm.isna().mean() * 100).round(3),
    "nunique": tm.nunique(),
})
prof.to_csv(os.path.join(OUT, "column_profile_trackman.csv"))
print("\n--- profile ---")
print(prof.to_string())

# ---- 2) 시즌별 행수·경기수 (train 대비) ----
g = tm.groupby("season").agg(n=("trackman_id", "count"),
                             games=("trackman_game_id", "nunique"),
                             pitchers=("pitcher_trackman_id", "nunique"))
g.to_csv(os.path.join(OUT, "trackman_by_season.csv"))
print("\n--- by season ---")
print(g.to_string())

# ---- 3) ID 공간: train pitcher_id 와 겹치는가 ----
train_ids = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                        usecols=["pitcher_id", "batter_id", "season", "game_month",
                                 "game_dayofweek", "inning", "top_bottom",
                                 "balls_before", "strikes_before", "outs_before",
                                 "pitcher_team_id", "batter_team_id", "game_type"])
tp, trp = set(train_ids["pitcher_id"].unique()), set(tm["pitcher_trackman_id"].unique())
print("\ntrain pitchers:", len(tp), " trackman pitchers:", len(trp),
      " intersect:", len(tp & trp))
print("trackman pitcher id range:", tm["pitcher_trackman_id"].min(), "-", tm["pitcher_trackman_id"].max())
print("train pitcher id range:", train_ids["pitcher_id"].min(), "-", train_ids["pitcher_id"].max())
tb, trb = set(train_ids["batter_id"].unique()), set(tm["batter_trackman_id"].unique())
print("batters intersect:", len(tb & trb), "of train", len(tb), "/ tm", len(trb))

# ---- 4) 팀 표기 ----
print("\npitcher_team values:", sorted(tm["pitcher_team"].astype(str).unique())[:30])
print("train team ids:", sorted(train_ids["pitcher_team_id"].unique()))

# ---- 5) 구종 분포 ----
print("\npitch_type_group:", dict(tm["pitch_type_group"].value_counts(dropna=False)))
print("tagged_pitch_type top:", dict(tm["tagged_pitch_type"].value_counts(dropna=False).head(12)))
print("auto_pitch_type top:", dict(tm["auto_pitch_type"].value_counts(dropna=False).head(12)))

# ---- 6) 물리량 분포 요약 ----
phys = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break", "extension",
        "rel_height", "rel_side", "zone_speed"]
ps = tm[phys].describe().T.round(2)
ps.to_csv(os.path.join(OUT, "trackman_physics_summary.csv"))
print("\n--- physics summary ---")
print(ps.to_string())

# ---- 7) 게임 단위 상태키 조인 가능성 ----
# (season, month, dow, inning, top_bottom, balls, strikes, outs) 상태키가 양쪽에서 얼마나 유일한가
key_cols_tm = ["season", "game_month", "game_dayofweek", "inning", "top_bottom",
               "balls_before", "strikes_before", "outs_before"]
tm_key = tm[key_cols_tm].astype(str).agg("|".join, axis=1)
tr_key = train_ids[key_cols_tm].astype(str).agg("|".join, axis=1)
print("\n상태키 유일 비율: trackman %.4f / train %.4f" %
      ((~tm_key.duplicated(keep=False)).mean(), (~tr_key.duplicated(keep=False)).mean()))
print("train 상태키가 trackman에 존재하는 비율: %.4f" % tr_key.isin(set(tm_key)).mean())

# ---- 8) 투수 매핑 가능성: 시즌별 투구수 시그니처 비교 ----
# train R+F 전체 투구수 per pitcher per season vs trackman per pitcher per season
a = train_ids.groupby(["pitcher_id", "season"]).size().rename("n_train")
b = tm.groupby(["pitcher_trackman_id", "season"]).size().rename("n_tm")
# 시그니처(시즌별 투구수 벡터)가 완전히 일치하는 투수쌍 수를 근사 추정
av = a.unstack(fill_value=0)
bv = b.unstack(fill_value=0)
sig_a = av.apply(lambda r: tuple(r), axis=1)
sig_b = bv.apply(lambda r: tuple(r), axis=1)
from collections import Counter
ca, cb = Counter(sig_a), Counter(sig_b)
uniq_a = {s for s, c in ca.items() if c == 1}
uniq_b = {s for s, c in cb.items() if c == 1}
exact = uniq_a & uniq_b
print("\n투구수 시그니처 완전일치(양쪽 유일) 투수 수:", len(exact), "/", len(sig_a))

# game 단위: trackman 경기 날짜별 투구수 vs train 블록 투구수 매칭 여지 (참고 수치만)
gd = tm.groupby("trackman_game_id").size()
print("trackman 경기당 투구수: mean %.1f median %.1f" % (gd.mean(), gd.median()))
print("trackman game_date 범위:", tm["game_date"].min(), "-", tm["game_date"].max())
