# exp_000_eda / 01: 스키마·결측·타깃 정의 역추적
# 실행: python3 01_schema_target.py  (cwd 무관, 절대경로 사용)
import pandas as pd
import numpy as np
import os

DATA = "<PROJECT_ROOT>/data"
OUT = "<PROJECT_ROOT>/experiments/exp_000_eda/profile"
os.makedirs(OUT, exist_ok=True)

train = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig")
test = pd.read_csv(os.path.join(DATA, "test.csv"), encoding="utf-8-sig")
print(f"train shape={train.shape}, test(stub) shape={test.shape}")

# ---- 0) train/test 컬럼 대조 ----
tr_cols, te_cols = set(train.columns), set(test.columns)
print("train-only cols:", sorted(tr_cols - te_cols))
print("test-only cols:", sorted(te_cols - tr_cols))
print("column order identical (except target):",
      list(train.drop(columns=["control_success"]).columns) == list(test.columns))

# ---- 1) dtypes + nunique + 결측률 ----
prof = pd.DataFrame({
    "dtype": train.dtypes.astype(str),
    "n_missing": train.isna().sum(),
    "missing_pct": (train.isna().mean() * 100).round(3),
    "nunique": train.nunique(),
})
prof.to_csv(os.path.join(OUT, "column_profile_train.csv"))
print("\n--- column profile ---")
print(prof.to_string())

# ---- 2) 범주형 값 확인 ----
for c in ["season", "game_type", "top_bottom", "pitcher_hand", "batter_hand",
          "base_state", "game_month", "inning", "balls_before", "strikes_before",
          "outs_before"]:
    vc = train[c].value_counts(dropna=False).sort_index()
    print(f"\n[{c}] {dict(vc)}")

# ---- 3) 연도별 base rate ----
by_year = train.groupby("season")["control_success"].agg(["count", "mean"])
by_year.columns = ["n", "success_rate"]
by_year.to_csv(os.path.join(OUT, "base_rate_by_season.csv"))
print("\n--- base rate by season ---")
print(by_year.to_string())

# ---- 4) 카운트 12상태별 base rate (라벨의 카운트 의존성) ----
ct = train.pivot_table(index="balls_before", columns="strikes_before",
                       values="control_success", aggfunc="mean")
ct_n = train.pivot_table(index="balls_before", columns="strikes_before",
                         values="control_success", aggfunc="count")
ct.round(4).to_csv(os.path.join(OUT, "success_by_count.csv"))
ct_n.to_csv(os.path.join(OUT, "n_by_count.csv"))
print("\n--- success rate by count (balls x strikes) ---")
print(ct.round(4).to_string())
print("\n--- n by count ---")
print(ct_n.to_string())

# 카운트 x 연도 (ABS 전후 카운트 구조 변화)
cty = train.groupby(["season", "balls_before", "strikes_before"])["control_success"].mean().unstack([1, 2])
cty.round(4).to_csv(os.path.join(OUT, "success_by_count_by_season.csv"))

# ---- 5) 기타 상황별 base rate ----
for key in ["base_state", "outs_before", "inning", "pitcher_hand", "batter_hand", "game_month"]:
    g = train.groupby(key)["control_success"].agg(["count", "mean"]).round(4)
    g.to_csv(os.path.join(OUT, f"success_by_{key}.csv"))
    print(f"\n--- success by {key} ---")
    print(g.to_string())

# 투타 매치업
mu = train.groupby(["pitcher_hand", "batter_hand"])["control_success"].agg(["count", "mean"]).round(4)
print("\n--- success by hand matchup ---")
print(mu.to_string())

# ---- 6) asof 피처와 타깃의 관계 (라벨 정의 힌트: strike/ball/middle rate 상관) ----
asof_cols = [c for c in train.columns if c.startswith("asof_")]
corr = train[asof_cols + ["control_success"]].corr()["control_success"].drop("control_success")
corr.round(4).to_csv(os.path.join(OUT, "asof_corr_with_target.csv"))
print("\n--- corr(asof_*, target) ---")
print(corr.round(4).sort_values().to_string())

# asof_pitcher_success_rate 의 전체 평균 vs 타깃 전체 평균 (정의 일치 확인)
print("\ntarget mean:", train["control_success"].mean())
print("asof_pitcher_success_rate weighted 근사(마지막 시점):")
last = train.groupby("pitcher_id").tail(1)
print(" last-row asof rate x n 합 / n 합 =",
      (last["asof_pitcher_success_rate"] * last["asof_pitcher_n"]).sum() / last["asof_pitcher_n"].sum())

# ball_rate + strike_rate + ? = 1 확인 (성공 정의가 스트라이크와 어떻게 다른가)
sub = train.dropna(subset=["asof_pitcher_ball_rate", "asof_pitcher_strike_rate"])
s = (sub["asof_pitcher_ball_rate"] + sub["asof_pitcher_strike_rate"])
print("\nball_rate+strike_rate: mean=%.4f, min=%.4f, max=%.4f" % (s.mean(), s.min(), s.max()))
print("success_rate vs strike_rate corr:",
      sub["asof_pitcher_success_rate"].corr(sub["asof_pitcher_strike_rate"]).round(4))
print("success_rate - strike_rate mean:",
      (sub["asof_pitcher_success_rate"] - sub["asof_pitcher_strike_rate"]).mean().round(4))

# ---- 7) sentinel 흔적 조사: 수치형 0/-1 집중, rate 컬럼의 0/1 비중 ----
rows = []
for c in train.columns:
    if train[c].dtype.kind in "if":
        col = train[c]
        rows.append({
            "col": c,
            "pct_zero": float((col == 0).mean() * 100),
            "pct_neg1": float((col == -1).mean() * 100),
            "pct_nan": float(col.isna().mean() * 100),
            "min": float(col.min()), "max": float(col.max()),
        })
sent = pd.DataFrame(rows).round(3)
sent.to_csv(os.path.join(OUT, "sentinel_scan.csv"), index=False)
print("\n--- sentinel scan ---")
print(sent.to_string())

# ---- 8) 결측 구조: asof 결측이 n==0과 정확히 일치하는가 ----
print("\n--- 결측 vs 표본수 정합성 ---")
print("asof_pitcher_n==0 rows:", (train["asof_pitcher_n"] == 0).sum())
print("asof_pitcher_success_rate NaN rows:", train["asof_pitcher_success_rate"].isna().sum())
print("n==0 <-> success_rate NaN 일치:",
      ((train["asof_pitcher_n"] == 0) == train["asof_pitcher_success_rate"].isna()).all())
print("asof_batter_n==0 rows:", (train["asof_batter_n"] == 0).sum())
print("batter rate NaN:", train["asof_batter_success_rate"].isna().sum())
print("batter 일치:", ((train["asof_batter_n"] == 0) == train["asof_batter_success_rate"].isna()).all())
print("pitchmix n==0:", (train["asof_pitcher_pitchmix_n"] == 0).sum(),
      " fastball NaN:", train["asof_pitcher_fastball_rate"].isna().sum())
# prev-game 결측은 '첫 경기' 의미 — n>0인데 prev1 NaN인 행
m = train["asof_pitcher_prev1_game_success_rate"].isna() & (train["asof_pitcher_n"] > 0)
print("prev1 NaN but career n>0:", m.sum(), "(등판 1경기차 등 게임 경계 결측)")

# 연도별 결측률 (cold start가 특정 연도에 몰리는지)
miss_year = train.groupby("season")[asof_cols].apply(lambda d: d.isna().mean() * 100).round(2)
miss_year.to_csv(os.path.join(OUT, "asof_missing_by_season.csv"))
print("\n--- asof missing % by season ---")
print(miss_year.to_string())

# ---- 9) row_id 중복·타깃 분포 ----
print("\nrow_id unique:", train["row_id"].is_unique)
print("full-row duplicates:", train.duplicated().sum())
print("dup ignoring row_id:", train.drop(columns=["row_id"]).duplicated().sum())
print("target value counts:", dict(train["control_success"].value_counts()))
