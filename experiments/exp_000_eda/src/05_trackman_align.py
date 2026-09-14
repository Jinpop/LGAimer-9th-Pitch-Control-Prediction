# exp_000_eda / 05: trackman-train 정렬(매핑) 실현 가능성 정밀 조사
import pandas as pd
import numpy as np
import os

DATA = "<PROJECT_ROOT>/data"
OUT = "<PROJECT_ROOT>/experiments/exp_000_eda/profile"

tm = pd.read_csv(os.path.join(DATA, "trackman_history.csv"), encoding="utf-8-sig",
                 usecols=["season", "game_date", "game_month", "game_dayofweek",
                          "trackman_game_id", "pitch_no", "inning", "top_bottom",
                          "balls_before", "strikes_before", "outs_before",
                          "pitcher_team", "batter_team", "pitcher_hand", "batter_hand",
                          "pitcher_trackman_id"])

# ---- 1) 인코딩 차이 확인 ----
print("tm top_bottom:", tm["top_bottom"].unique())
print("tm hand:", tm["pitcher_hand"].unique())
print("tm balls values:", sorted(tm["balls_before"].unique()))
print("tm strikes values:", sorted(tm["strikes_before"].unique()))
print("tm outs values:", sorted(tm["outs_before"].unique()))
print("tm inning values:", sorted(tm["inning"].unique()))

# game_date 파싱
tm["date"] = pd.to_datetime(tm["game_date"], format="mixed")
print("date range:", tm["date"].min(), "-", tm["date"].max())
# dow 일치 확인
print("dow matches parsed date:", (tm["date"].dt.dayofweek == tm["game_dayofweek"]).mean())

# ---- 2) 정규화 후 상태키 재비교 ----
train = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                    usecols=["season", "game_month", "game_dayofweek", "game_type",
                             "inning", "top_bottom", "balls_before", "strikes_before",
                             "outs_before", "pitcher_team_id", "batter_team_id",
                             "pitcher_hand", "batter_hand", "pitcher_id"])
tb_map = {"Top": "T", "Bottom": "B", "T": "T", "B": "B"}
tm["tb"] = tm["top_bottom"].map(tb_map)
key_tm = (tm["season"].astype(str) + "|" + tm["game_month"].astype(str) + "|" +
          tm["game_dayofweek"].astype(str) + "|" + tm["inning"].astype(str) + "|" +
          tm["tb"] + "|" + tm["balls_before"].astype(str) + "|" +
          tm["strikes_before"].astype(str) + "|" + tm["outs_before"].astype(str))
key_tr = (train["season"].astype(str) + "|" + train["game_month"].astype(str) + "|" +
          train["game_dayofweek"].astype(str) + "|" + train["inning"].astype(str) + "|" +
          train["top_bottom"] + "|" + train["balls_before"].astype(str) + "|" +
          train["strikes_before"].astype(str) + "|" + train["outs_before"].astype(str))
print("\n정규화 후 train 상태키의 trackman 존재 비율: %.4f" % key_tr.isin(set(key_tm)).mean())

# ---- 3) 경기 수준 비교: 시즌·월·요일별 경기 수 (train 블록 vs trackman) ----
# train 블록 재구성
tp = train.apply(lambda r: tuple(sorted((r["pitcher_team_id"], r["batter_team_id"]))), axis=1)
train["team_pair"] = tp
bk = (train[["season", "game_month", "game_dayofweek", "game_type"]].astype(str).agg("|".join, axis=1)
      + "|" + train["team_pair"].astype(str))
train["block"] = (bk != bk.shift(1)).cumsum()
tr_games = train.groupby("block").agg(season=("season", "first"), month=("game_month", "first"),
                                      dow=("game_dayofweek", "first"), gt=("game_type", "first"),
                                      n=("season", "count"))
tm_games = tm.groupby("trackman_game_id").agg(season=("season", "first"), month=("game_month", "first"),
                                              dow=("game_dayofweek", "first"), n=("season", "count"),
                                              pteam=("pitcher_team", lambda s: tuple(sorted(s.unique()))))
print("\ntrain 게임 수:", len(tr_games), " trackman 게임 수:", len(tm_games))

# (season, month, dow, n_pitches) 로 경기 매칭 시도 — 투구수는 강한 시그니처
tr_games["key"] = (tr_games["season"].astype(str) + "|" + tr_games["month"].astype(str) + "|" +
                   tr_games["dow"].astype(str) + "|" + tr_games["n"].astype(str))
tm_games["key"] = (tm_games["season"].astype(str) + "|" + tm_games["month"].astype(str) + "|" +
                   tm_games["dow"].astype(str) + "|" + tm_games["n"].astype(str))
from collections import Counter
ctr, ctm = Counter(tr_games["key"]), Counter(tm_games["key"])
uniq_both = [k for k in ctr if ctr[k] == 1 and ctm.get(k, 0) == 1]
print("(시즌,월,요일,총투구수) 유일-유일 경기 매칭:", len(uniq_both), "/", len(tr_games))
match_1plus = sum(1 for k in ctr if ctm.get(k, 0) >= 1)
print("트랙맨에 동일키 존재하는 train 경기 비율: %.4f" % (sum(ctr[k] for k in ctr if ctm.get(k,0)>=1) / len(tr_games)))

# ---- 4) 유일-유일 매칭 경기 하나로 행 수준 시퀀스 일치 검증 ----
if uniq_both:
    k = uniq_both[0]
    b_id = tr_games.index[tr_games["key"] == k][0]
    g_id = tm_games.index[tm_games["key"] == k][0]
    A = train[train["block"] == b_id].reset_index()
    B = tm[tm["trackman_game_id"] == g_id].sort_values("pitch_no").reset_index()
    seq_cols_a = ["inning", "top_bottom", "balls_before", "strikes_before", "outs_before"]
    a_seq = A[["inning", "balls_before", "strikes_before", "outs_before"]].assign(tb=A["top_bottom"])
    b_seq = B[["inning", "balls_before", "strikes_before", "outs_before"]].assign(tb=B["tb"])
    same = (a_seq.values == b_seq.values).all()
    print(f"\n샘플 경기 매칭: train block {b_id} vs tm game {g_id}, 행수 {len(A)}={len(B)}")
    print("행 시퀀스(이닝/카운트/아웃/초말) 완전 일치:", bool(same))
    if same:
        # 투수 손 일치로 투수 매핑 힌트
        hand_map = {"R": 2, "L": 1, "Right": 2, "Left": 1}
        bh = B["pitcher_hand"].map(hand_map)
        print("pitcher_hand 시퀀스 일치율:", (A["pitcher_hand"].values == bh.values).mean())
        pair = pd.DataFrame({"train_pid": A["pitcher_id"].values,
                             "tm_pid": B["pitcher_trackman_id"].values})
        cons = pair.groupby("train_pid")["tm_pid"].nunique()
        print("경기 내 train 투수 -> tm 투수 1:1 대응:", (cons == 1).all(),
              " (투수 수:", len(cons), ")")

# ---- 5) trackman의 초과 커버리지: 시즌별 경기수 비교 ----
cmp = pd.DataFrame({
    "train_games": tr_games.groupby(["season"]).size(),
    "tm_games": tm_games.groupby(tm_games["season"].astype(str)).size().values,
})
print("\n--- 경기 수 비교 ---")
print(cmp.to_string())
cmp.to_csv(os.path.join(OUT, "games_train_vs_trackman.csv"))
