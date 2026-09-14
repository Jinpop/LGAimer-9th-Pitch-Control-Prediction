"""exp_103 stage0 — 전제 검증.

0a  train.csv 행 순서가 시간순인가 (as-of 카운터가 그룹 안에서 단조 비감소인가)
0b  운영진 asof_batter_* 가 정확히 무엇의 집계인가 (역설계 → 재현 일치 확인)

🔴 어느 것도 라벨을 현재 행에 넣지 않는다. 재현은 shift(1) expanding 으로만 한다.
"""
import json, os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("EXP103_TRAIN", "<PROJECT_ROOT>/data/train.csv")
OUT = os.path.join(ROOT, "out")
os.makedirs(OUT, exist_ok=True)

USE = ["row_id", "season", "game_month", "game_dayofweek", "inning", "top_bottom",
       "balls_before", "strikes_before",
       "pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
       "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_middle_rate",
       "asof_pitcher_prev1_game_success_rate",
       "asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate",
       "control_success"]

df = pd.read_csv(DATA, usecols=USE, encoding="utf-8-sig")
df = df[USE]
N = len(df)
res = {"n_rows": int(N), "cols": USE}

# row_id 가 파일 순서와 같은 순번인지 (TRAIN_0000001 …)
rid = df["row_id"].str.slice(6).astype(np.int64)
res["row_id_is_sequential"] = bool((rid.values == np.arange(1, N + 1)).all())

# ── 0a  행 순서 = 시간순 증명 ────────────────────────────────────
def monotone_report(key_cols, ncol, tag):
    g = df.groupby(key_cols, sort=False)[ncol]
    prev = g.shift(1)
    valid = prev.notna()
    d = (df[ncol] - prev)[valid]
    out = {
        "n_pairs": int(valid.sum()),
        "n_groups": int(g.ngroups),
        "n_decrease": int((d < 0).sum()),
        "frac_decrease": float((d < 0).mean()),
        "n_step_ne_1": int((d != 1).sum()),
        "frac_step_ne_1": float((d != 1).mean()),
        "step_value_counts": {str(k): int(v) for k, v in
                              d.value_counts().head(6).items()},
        "first_value_is_zero_frac": float(
            (df.loc[~valid, ncol] == 0).mean()),
    }
    res[tag] = out
    return out

monotone_report(["season", "pitcher_id"], "asof_pitcher_n", "a0_pitcher_season")
monotone_report(["season", "batter_id"], "asof_batter_n", "a0_batter_season")
# 시즌 무시(통산 누적인지) 대조
monotone_report(["pitcher_id"], "asof_pitcher_n", "a0_pitcher_career")
monotone_report(["batter_id"], "asof_batter_n", "a0_batter_career")

# 시즌 경계에서 리셋되는가
tmp = df[["season", "pitcher_id", "asof_pitcher_n"]].copy()
first = tmp.groupby(["season", "pitcher_id"], sort=False)["asof_pitcher_n"].first()
res["a0_season_reset_pitcher_first_n_zero_frac"] = float((first == 0).mean())
firstb = df.groupby(["season", "batter_id"], sort=False)["asof_batter_n"].first()
res["a0_season_reset_batter_first_n_zero_frac"] = float((firstb == 0).mean())

# 시즌 블록이 파일 안에서 연속인가 (섞여 있으면 시간순이 아니다)
sc = df["season"].values
res["a0_season_blocks_contiguous"] = bool(
    (df["season"].ne(df["season"].shift()).cumsum().nunique()
     == df["season"].nunique()))
res["a0_season_order"] = [int(x) for x in
                          df["season"][df["season"].ne(df["season"].shift())].tolist()]
# 월이 시즌 안에서 비감소인가
mono_m = df.groupby("season", sort=False)["game_month"].apply(
    lambda s: int((s.diff() < 0).sum()))
res["a0_month_decrease_within_season"] = {str(k): int(v) for k, v in mono_m.items()}

# ── 0b  운영진 집계 역설계 ──────────────────────────────────────
y = df["control_success"].values.astype(np.float64)

def recon(keys, tag):
    g = df.groupby(keys, sort=False)
    cs = g["control_success"].cumsum().values - y          # shift(1) 누적 합
    cn = g.cumcount().values.astype(np.float64)            # shift(1) 누적 개수
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(cn > 0, cs / np.maximum(cn, 1), np.nan)
    return cn, rate

for keys, tag in ((["season", "batter_id"], "b_season"),
                  (["batter_id"], "b_career")):
    cn, rate = recon(keys, tag)
    obs_n = df["asof_batter_n"].values.astype(np.float64)
    obs_r = df["asof_batter_success_rate"].values.astype(np.float64)
    same_n = int((cn == obs_n).sum())
    both = np.isfinite(rate) & np.isfinite(obs_r)
    nanmatch = int((np.isnan(rate) == np.isnan(obs_r)).sum())
    dd = np.abs(rate[both] - obs_r[both]) if both.sum() else np.array([0.0])
    res[f"b0_recon_{tag}"] = {
        "n_match_asof_batter_n": same_n,
        "frac_match_asof_batter_n": float(same_n / N),
        "n_both_finite": int(both.sum()),
        "nan_pattern_match_frac": float(nanmatch / N),
        "max_abs_diff_rate": float(dd.max()),
        "frac_rate_within_1e-9": float((dd < 1e-9).mean()),
    }

# 투수 쪽도 같은 규약인지 (대조)
gp = df.groupby(["season", "pitcher_id"], sort=False)
cs = gp["control_success"].cumsum().values - y
cn = gp.cumcount().values.astype(np.float64)
with np.errstate(invalid="ignore", divide="ignore"):
    pr = np.where(cn > 0, cs / np.maximum(cn, 1), np.nan)
obs = df["asof_pitcher_success_rate"].values.astype(np.float64)
both = np.isfinite(pr) & np.isfinite(obs)
res["b0_recon_pitcher_season"] = {
    "frac_match_asof_pitcher_n": float((cn == df["asof_pitcher_n"].values).mean()),
    "max_abs_diff_rate": float(np.abs(pr[both] - obs[both]).max()),
    "frac_rate_within_1e-9": float((np.abs(pr[both] - obs[both]) < 1e-9).mean()),
}

# asof_batter_middle_rate 는 라벨로 재현 불가(중간코스는 우리에게 없다) — 확인만
res["b0_batter_middle_rate_isna_frac"] = float(
    df["asof_batter_middle_rate"].isna().mean())
res["b0_batter_n_describe"] = {k: float(v) for k, v in
                               df["asof_batter_n"].describe().items()}

json.dump(res, open(f"{OUT}/s0_premise.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
