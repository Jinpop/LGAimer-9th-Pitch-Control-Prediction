# exp_000_eda / 02: 행 순서·asof 의미 검증·게임타입·선수 구성
import pandas as pd
import numpy as np
import os

DATA = "<PROJECT_ROOT>/data"
OUT = "<PROJECT_ROOT>/experiments/exp_000_eda/profile"

train = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig")
test = pd.read_csv(os.path.join(DATA, "test.csv"), encoding="utf-8-sig")

# ---- 1) row_id 순서 = 시간순인가 ----
train["rid_num"] = train["row_id"].str.replace("TRAIN_", "").astype(int)
print("row_id numeric sorted in file order:", train["rid_num"].is_monotonic_increasing)
print("season monotonic along file order:", train["season"].is_monotonic_increasing)
# 시즌 내 월 순서
mon_ok = train.groupby("season")["game_month"].apply(lambda s: s.is_monotonic_increasing)
print("game_month monotonic within season:\n", mon_ok)

# ---- 2) 투수별 asof_pitcher_n 이 파일 순서로 단조증가 + 증분 분포 ----
g = train.groupby("pitcher_id")["asof_pitcher_n"]
mono = g.apply(lambda s: s.is_monotonic_increasing)
print("\nasof_pitcher_n monotonic within pitcher (file order): all =", mono.all(),
      " / violators =", (~mono).sum())
diffs = train.groupby("pitcher_id")["asof_pitcher_n"].diff().dropna()
print("asof_pitcher_n increment distribution:")
print(diffs.value_counts().head(10).to_string())
print("increment==1 비율: %.6f" % (diffs == 1).mean())

# 타자도
gb = train.groupby("batter_id")["asof_batter_n"]
diffs_b = train.groupby("batter_id")["asof_batter_n"].diff().dropna()
print("batter increment==1 비율: %.6f" % (diffs_b == 1).mean())

# ---- 3) asof_pitcher_success_rate 갱신 정합성: rate*n 누적 = 과거 control_success 합인가 ----
# 표본 투수 몇 명으로 정확 검증
errs = []
for pid in train["pitcher_id"].drop_duplicates().sample(12, random_state=0):
    d = train[train["pitcher_id"] == pid]
    n = d["asof_pitcher_n"].to_numpy()
    r = d["asof_pitcher_success_rate"].to_numpy()
    y = d["control_success"].to_numpy()
    if (np.diff(n) != 1).any():
        errs.append((pid, "gap"))
        continue
    succ_cum = np.nancumsum(np.where(np.isnan(r * n), 0, r * n))
    # 다음 행의 r*n 은 이전까지 y 누적과 같아야 함
    expect = np.cumsum(y)[:-1]
    got = (r * n)[1:]
    ok = np.allclose(expect, got, atol=1e-6)
    errs.append((pid, "OK" if ok else f"MISMATCH max={np.abs(expect-got).max()}"))
print("\nasof rate 재구성 검증 (rate[t]*n[t] == sum(y[<t])):")
for e in errs:
    print(" ", e)

# ---- 4) 시즌 경계에서 asof 리셋 여부 ----
first_of_season = train.groupby(["pitcher_id", "season"], as_index=False).first()
fs = first_of_season.sort_values(["pitcher_id", "season"])
fs["prev_season_exists"] = fs.groupby("pitcher_id")["season"].shift(1).notna()
carry = fs[fs["prev_season_exists"] & (fs["asof_pitcher_n"] > 0)]
reset = fs[fs["prev_season_exists"] & (fs["asof_pitcher_n"] == 0)]
print("\n시즌 첫 행에서 asof_pitcher_n>0 (career 누적 유지):", len(carry),
      " / ==0 (리셋?):", len(reset))

# ---- 5) game_type 프로파일 ----
gt = train.groupby(["game_type", "season"]).agg(
    n=("control_success", "count"), rate=("control_success", "mean")).round(4)
gt.to_csv(os.path.join(OUT, "game_type_by_season.csv"))
print("\n--- game_type x season ---")
print(gt.to_string())
print("\nteam ids by game_type:")
for t in ["R", "F"]:
    d = train[train["game_type"] == t]
    print(f" {t}: pitcher_team_id={sorted(d['pitcher_team_id'].unique())}")
print("game_type F month dist:", dict(train[train.game_type=='F']["game_month"].value_counts().sort_index()))
print("test stub game_type:", test["game_type"].unique())

# ---- 6) 시즌별 선수 구성 / 신규 선수 비율 ----
rows = []
seen_p, seen_b = set(), set()
for s in sorted(train["season"].unique()):
    d = train[train["season"] == s]
    ps, bs = set(d["pitcher_id"]), set(d["batter_id"])
    new_p = ps - seen_p
    new_p_rows = d["pitcher_id"].isin(new_p).mean()
    rows.append({"season": s, "n_pitchers": len(ps), "new_pitchers": len(new_p),
                 "new_pitcher_row_pct": round(new_p_rows * 100, 2),
                 "n_batters": len(bs), "new_batters": len(bs - seen_b)})
    seen_p |= ps
    seen_b |= bs
comp = pd.DataFrame(rows)
comp.to_csv(os.path.join(OUT, "player_composition_by_season.csv"), index=False)
print("\n--- 시즌별 선수 구성 ---")
print(comp.to_string(index=False))

# ---- 7) test 스텁 선수의 train 존재 여부 ----
print("\ntest stub pitchers in train:",
      {int(p): bool(p in seen_p) for p in test["pitcher_id"].unique()})
print("test stub batters in train:",
      {int(b): bool(b in seen_b) for b in test["batter_id"].unique()})
print("train pitcher_id max:", train["pitcher_id"].max(), " test stub pitcher ids:", sorted(test["pitcher_id"].unique()))
print("train batter_id max:", train["batter_id"].max(), " test stub batter ids:", sorted(test["batter_id"].unique()))

# ---- 8) win expectancy / li 정합 ----
we_sum = train["home_win_expectancy"] + train["away_win_expectancy"]
print("\nhome+away win expectancy: min=%.2f max=%.2f mean=%.4f" % (we_sum.min(), we_sum.max(), we_sum.mean()))
print("li quantiles:", train["li"].quantile([0, .25, .5, .75, .95, .99, 1]).round(3).to_dict())

# ---- 9) prev1 결측 메커니즘: 투수의 커리어 첫 경기인가 ----
# prev1 NaN 행의 asof_pitcher_n 분포
m = train["asof_pitcher_prev1_game_success_rate"].isna()
print("\nprev1 NaN rows: asof_pitcher_n stats:")
print(train.loc[m, "asof_pitcher_n"].describe().round(2).to_string())

# ---- 10) 2024 내 월별 base rate (ABS 시즌 내 추이) ----
m24 = train[train.season >= 2023].groupby(["season", "game_month"])["control_success"].agg(["count", "mean"]).round(4)
m24.to_csv(os.path.join(OUT, "base_rate_2023_2024_by_month.csv"))
print("\n--- 2023-2024 월별 base rate ---")
print(m24.to_string())
