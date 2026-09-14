# exp_002 / 02: 투수 정적 프로파일 (2024년 말 기준) — test 사용 가능 후보 피처
# 원칙 (전략 §12 / exp_000_eda §4): 2024년 말까지의 trackman 이력만 사용.
#   career_* = 2019~2024 전체, s2024_* = 2024 시즌만.
#   결측은 NaN 유지 (sentinel 채움 금지 규약).
# 결정론적: 집계만, 난수 없음.
import pandas as pd
import numpy as np
import os
import time

DATA = os.path.expanduser("~/SungjinPark/data")
OUT = os.path.expanduser("~/SungjinPark/experiments/exp_002/out")
t0 = time.time()


def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)


pmap = pd.read_csv(os.path.join(OUT, "pitcher_id_map.csv"))
log("pitcher map:", len(pmap))

tm = pd.read_csv(
    os.path.join(DATA, "trackman_history.csv"), encoding="utf-8-sig",
    usecols=["season", "trackman_game_id", "pitcher_trackman_id", "pitch_type_group",
             "rel_speed", "spin_rate", "induced_vert_break", "horz_break",
             "extension", "rel_height", "rel_side", "zone_speed"])
assert (tm["season"] <= 2024).all(), "trackman에 2025 데이터 존재 — 기준 위반"
tm = tm[tm["pitcher_trackman_id"].isin(set(pmap["pitcher_trackman_id"]))]
log("tm rows for mapped pitchers:", len(tm))

PHYS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
        "extension", "rel_height", "rel_side", "zone_speed"]


def profile(df, prefix):
    g = df.groupby("pitcher_trackman_id")
    out = pd.DataFrame(index=g.size().index)
    out[f"{prefix}_n_pitches"] = g.size()
    out[f"{prefix}_n_games"] = g["trackman_game_id"].nunique()
    for c in PHYS:
        out[f"{prefix}_{c}_mean"] = g[c].mean()
        if c in ("rel_speed", "spin_rate", "rel_height", "rel_side",
                 "induced_vert_break", "horz_break"):
            out[f"{prefix}_{c}_std"] = g[c].std()
    # 구속 상위 10% (최고 구속 근사, 이상치 완화)
    out[f"{prefix}_rel_speed_p90"] = g["rel_speed"].quantile(0.9)
    # 구종 그룹별 비율·구속·회전수
    groups = ["fastball", "breaking", "offspeed"]
    n = g.size()
    for pg in groups:
        sub = df[df["pitch_type_group"] == pg].groupby("pitcher_trackman_id")
        out[f"{prefix}_{pg}_share"] = (sub.size() / n).fillna(0.0)
        out[f"{prefix}_{pg}_rel_speed_mean"] = sub["rel_speed"].mean()
        out[f"{prefix}_{pg}_spin_rate_mean"] = sub["spin_rate"].mean()
        out[f"{prefix}_{pg}_ivb_mean"] = sub["induced_vert_break"].mean()
        out[f"{prefix}_{pg}_hb_mean"] = sub["horz_break"].mean()
    return out


log("pitch_type_group values:", dict(tm["pitch_type_group"].value_counts(dropna=False)))
career = profile(tm, "career")
s2024 = profile(tm[tm["season"] == 2024], "s2024")
prof = career.join(s2024, how="left")
prof["last_season"] = tm.groupby("pitcher_trackman_id")["season"].max()
prof = prof.reset_index()

prof = pmap[["pitcher_id", "pitcher_trackman_id", "confidence"]].merge(
    prof, on="pitcher_trackman_id", how="left")
assert prof["pitcher_id"].is_unique
prof = prof.sort_values("pitcher_id")
prof.to_csv(os.path.join(OUT, "pitcher_static_profile_2024.csv"), index=False)
log("profile table:", prof.shape, "-> pitcher_static_profile_2024.csv")
log("s2024 coverage among mapped pitchers: %.1f%%" %
    (100 * prof["s2024_n_pitches"].notna().mean()))
log("DONE 02_build_profile")
