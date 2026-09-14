# exp_000_eda / 06: 라벨 구성 역추적 (success/reverse/middle/ball/strike 관계) + R전용 카운트표 + 그림
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "<PROJECT_ROOT>/data"
BASE = "<PROJECT_ROOT>/experiments/exp_000_eda"
OUT = os.path.join(BASE, "profile")
FIGS = os.path.join(BASE, "figs")
os.makedirs(FIGS, exist_ok=True)

train = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig")

# ---- 1) 투수 커리어 마지막 시점의 rate들 간 항등식 탐색 ----
last = train.groupby("pitcher_id").tail(1)
last = last[last["asof_pitcher_n"] >= 500]  # 표본 충분한 투수만
s, r, m = last["asof_pitcher_success_rate"], last["asof_pitcher_reverse_rate"], last["asof_pitcher_middle_rate"]
b, k = last["asof_pitcher_ball_rate"], last["asof_pitcher_strike_rate"]
print("n pitchers (>=500구):", len(last))
for name, expr in [
    ("success + reverse + middle", s + r + m),
    ("success + reverse", s + r),
    ("ball + strike", b + k),
    ("success - (1 - reverse - middle)", s - (1 - r - m)),
]:
    print(f"{name}: mean={expr.mean():.4f} std={expr.std():.4f} min={expr.min():.4f} max={expr.max():.4f}")
print("\ncorr matrix (career rates):")
print(last[["asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
            "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
            "asof_pitcher_strike_rate"]].corr().round(3).to_string())

# 회귀: success ~ reverse + middle + ball + strike (커리어 수준 정의 근사)
X = np.column_stack([r, m, b, k, np.ones(len(last))])
coef, res, *_ = np.linalg.lstsq(X, s, rcond=None)
pred = X @ coef
print("\nOLS success ~ reverse, middle, ball, strike:")
print(" coefs [reverse, middle, ball, strike, const]:", np.round(coef, 4))
print(" R^2:", 1 - ((s - pred) ** 2).sum() / ((s - s.mean()) ** 2).sum())

# ---- 2) R 전용 카운트표 (test는 R로 추정) ----
r_only = train[train["game_type"] == "R"]
ctR = r_only.pivot_table(index="balls_before", columns="strikes_before",
                         values="control_success", aggfunc="mean").round(4)
ctR.to_csv(os.path.join(OUT, "success_by_count_R_only.csv"))
print("\n--- R전용 카운트별 성공률 ---")
print(ctR.to_string())
ctF = train[train.game_type == "F"].pivot_table(index="balls_before", columns="strikes_before",
                                                values="control_success", aggfunc="mean").round(4)
print("\n--- F전용 ---")
print(ctF.to_string())

# 2023+ R전용 (최근 라벨 체제)
ctR23 = r_only[r_only.season >= 2023].pivot_table(index="balls_before", columns="strikes_before",
                                                  values="control_success", aggfunc="mean").round(4)
print("\n--- R전용 2023-24 ---")
print(ctR23.to_string())

# ---- 3) 연도별 base rate by game_type 그림 ----
g = train.groupby(["season", "game_type"])["control_success"].mean().unstack()
fig, ax = plt.subplots(figsize=(7, 4))
g.plot(marker="o", ax=ax)
ax.set_ylabel("control_success rate")
ax.set_title("Base rate by season / game_type")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIGS, "base_rate_by_season_gametype.png"), dpi=120)
print("\nfig saved. base rate table:")
print(g.round(4).to_string())

# ---- 4) LI·승률 피처와 타깃 ----
train["li_bin"] = pd.qcut(train["li"], 10, duplicates="drop")
libin = train.groupby("li_bin", observed=True)["control_success"].agg(["count", "mean"]).round(4)
libin.to_csv(os.path.join(OUT, "success_by_li_decile.csv"))
print("\n--- LI 십분위별 성공률 ---")
print(libin.to_string())

# ---- 5) asof_pitcher_n(경험) vs 성공률 ----
train["n_bin"] = pd.cut(train["asof_pitcher_n"], [-1, 0, 50, 200, 500, 1000, 3000, 6000, 20000])
nb = train.groupby("n_bin", observed=True)["control_success"].agg(["count", "mean"]).round(4)
nb.to_csv(os.path.join(OUT, "success_by_career_n.csv"))
print("\n--- 커리어 투구수 구간별 성공률 ---")
print(nb.to_string())
