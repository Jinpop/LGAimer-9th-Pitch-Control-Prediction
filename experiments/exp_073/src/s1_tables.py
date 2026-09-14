"""exp_073 stage 1 — as-of TrackMan 투수 집계표.

🔴 as-of 규율 (이 실험의 유일한 치명적 실패 지점)
    시즌 Y 행에 붙는 값은 **season < Y** 트랙맨 행만으로 만든다.
    시즌별 표를 따로 만들고 경계를 assert 한다.  Y=2024 는 스크린용
    (프레임이 전부 season 2024), Y=2025 는 배포 커버리지 확인용.

매핑 정본: p6_verify/src/pitcher_id_map.csv (571쌍,
    sha256 90b8d376c61b3af3753a89ea8f052b19b3ea017c0051fed32bd875eafd82c67d).
    rebuild_tables.py 로 종단 검증된 판.  685판은 커버리지 차이만 부수 기록.

사전 등록된 최소 표본 (결과를 보기 전에 정한다)
    sd 계열   : 유한 쌍 2개 이상 (ddof=1 이 정의되는 최소)
    구종 내 sd: 구종당 5개 이상인 구종만, 그 구종의 투구수로 가중평균
    speed_trend: 직전 시즌 10개 이상 **그리고** 그 이전 커리어 10개 이상
    그 밖의 하한(예: 200구)은 두지 않는다 — 두면 결과를 보고 고르는 문이 열린다.
    결측은 NaN 으로 남긴다.  sentinel 충전 금지.
"""
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

H = "/home/<user>/SungjinPark"
OUT = f"{H}/exp073_tm"
TM = f"{H}/data/trackman_history.csv"
MAP571 = f"{H}/p6_verify/src/pitcher_id_map.csv"
MAP685 = f"{H}/experiments/exp_002/out/pitcher_id_map.csv"
MAP571_SHA = "90b8d376c61b3af3753a89ea8f052b19b3ea017c0051fed32bd875eafd82c67d"
PHYS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
        "extension", "rel_height", "rel_side", "zone_speed"]
MIN_G = 5        # 구종 내 sd 를 쓰기 위한 구종당 최소 투구수
MIN_TREND = 10   # speed_trend 양쪽 최소 투구수

t0 = time.time()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


got = sha_file(MAP571)
assert got == MAP571_SHA, f"571쌍 매핑 sha 불일치 {got}"
print(f"🟢 매핑 정본 sha {got[:16]}")

m = pd.read_csv(MAP571)
assert len(m) == 571 and m["pitcher_id"].is_unique \
    and m["pitcher_trackman_id"].is_unique
tm2pid = dict(zip(m["pitcher_trackman_id"].astype(np.int64),
                  m["pitcher_id"].astype(np.int64)))
m685 = pd.read_csv(MAP685)
tm2pid685 = dict(zip(m685["pitcher_trackman_id"].astype(np.int64),
                     m685["pitcher_id"].astype(np.int64)))

print(f"trackman_history.csv 읽는 중 …")
use = ["season", "pitcher_trackman_id", "tagged_pitch_type"] + PHYS
tm = pd.read_csv(TM, encoding="utf-8-sig", usecols=use)
print(f"  {len(tm):,}행 · {time.time()-t0:.0f}s")

REP = {"trackman_sha256": sha_file(TM), "n_rows_raw": int(len(tm)),
       "seasons_raw": sorted(int(s) for s in tm["season"].unique()),
       "map571_sha256": got, "map571_pairs": int(len(m)),
       "map685_pairs": int(len(m685)),
       "missing_rate_raw": {c: float(tm[c].isna().mean()) for c in PHYS}}

tm["pitcher_id"] = tm["pitcher_trackman_id"].map(tm2pid)
REP["row_cov_571_of_trackman"] = float(tm["pitcher_id"].notna().mean())
REP["row_cov_685_of_trackman"] = float(
    tm["pitcher_trackman_id"].map(tm2pid685).notna().mean())
print(f"  트랙맨 행 기준 매핑 커버리지  571판 "
      f"{REP['row_cov_571_of_trackman']:.4f}  685판 "
      f"{REP['row_cov_685_of_trackman']:.4f}")

tm = tm[tm["pitcher_id"].notna()].copy()
tm["pitcher_id"] = tm["pitcher_id"].astype(np.int64)
tm["season"] = tm["season"].astype(np.int64)
REP["n_rows_mapped"] = int(len(tm))


def two_d_sd(g, a, b):
    """sqrt(var(a) + var(b)), 두 열이 모두 유한한 행에서 ddof=1."""
    v = g[[a, b]].to_numpy(np.float64)
    v = v[np.isfinite(v).all(axis=1)]
    if len(v) < 2:
        return np.nan
    return float(np.sqrt(v[:, 0].var(ddof=1) + v[:, 1].var(ddof=1)))


def build(Y):
    sub = tm[tm["season"] < Y]
    # 🔴 as-of 경계 assert
    assert len(sub) > 0
    assert int(sub["season"].max()) <= Y - 1, "as-of 위반: 표에 season >= Y 가 있다"
    assert bool((sub["season"].to_numpy() < Y).all()), "as-of 위반"
    rows, meta = {}, {"Y": Y, "season_max_used": int(sub["season"].max()),
                      "season_min_used": int(sub["season"].min()),
                      "n_rows_used": int(len(sub)),
                      "n_pitchers": int(sub["pitcher_id"].nunique())}
    for pid, g in sub.groupby("pitcher_id", sort=True):
        r = {"tm_n": float(len(g))}
        r["tm_rel_sd"] = two_d_sd(g, "rel_height", "rel_side")
        r["tm_break_sd"] = two_d_sd(g, "induced_vert_break", "horz_break")
        rs = g["rel_speed"].to_numpy(np.float64)
        rs = rs[np.isfinite(rs)]
        r["tm_speed_sd"] = float(rs.std(ddof=1)) if len(rs) >= 2 else np.nan
        for nm, col in (("tm_ext_mean", "extension"),
                        ("tm_spin_mean", "spin_rate")):
            v = g[col].to_numpy(np.float64)
            v = v[np.isfinite(v)]
            r[nm] = float(v.mean()) if len(v) else np.nan
        # 구종 내 릴리스 산포 — 구종 믹스 효과 제거판
        num, den = 0.0, 0.0
        for _, gg in g.groupby("tagged_pitch_type", sort=True):
            if len(gg) < MIN_G:
                continue
            s = two_d_sd(gg, "rel_height", "rel_side")
            if np.isfinite(s):
                num += s * len(gg)
                den += len(gg)
        r["tm_rel_sd_wp"] = float(num / den) if den > 0 else np.nan
        # 직전 시즌 대비 구속 추세
        prev = g.loc[g["season"] == Y - 1, "rel_speed"].to_numpy(np.float64)
        prev = prev[np.isfinite(prev)]
        old = g.loc[g["season"] < Y - 1, "rel_speed"].to_numpy(np.float64)
        old = old[np.isfinite(old)]
        r["tm_speed_trend"] = (float(prev.mean() - old.mean())
                               if len(prev) >= MIN_TREND and len(old) >= MIN_TREND
                               else np.nan)
        rows[int(pid)] = r
    df = pd.DataFrame(rows).T.sort_index()
    df.index.name = "pitcher_id"
    meta["finite_rate"] = {c: float(np.isfinite(df[c].to_numpy(np.float64)).mean())
                           for c in df.columns}
    meta["describe"] = {c: {k: float(v) for k, v in
                            df[c].describe().to_dict().items()}
                        for c in df.columns}
    return df, meta


os.makedirs(OUT, exist_ok=True)
TABLES = {}
for Y in (2024, 2025):
    df, meta = build(Y)
    TABLES[Y] = df
    REP[f"table_Y{Y}"] = meta
    df.to_csv(f"{OUT}/tm_table_Y{Y}.csv")
    print(f"\n=== Y={Y} 표 (season {meta['season_min_used']}"
          f"~{meta['season_max_used']}, {meta['n_rows_used']:,}행, "
          f"투수 {meta['n_pitchers']}명) ===")
    for c in df.columns:
        d = meta["describe"][c]
        print(f"  {c:<16} 유한 {meta['finite_rate'][c]:.4f}  "
              f"median {d.get('50%', float('nan')):+.5f}  "
              f"[{d.get('min', float('nan')):+.5f}, "
              f"{d.get('max', float('nan')):+.5f}]")

# 🔴 두 표가 실제로 다른지 — 같으면 as-of 분리가 안 된 것이다
a, b = TABLES[2024]["tm_rel_sd"], TABLES[2025]["tm_rel_sd"]
common = a.index.intersection(b.index)
same = np.allclose(a.loc[common].to_numpy(np.float64),
                   b.loc[common].to_numpy(np.float64), equal_nan=True)
REP["Y2024_vs_Y2025_identical"] = bool(same)
assert not same, "🔴 Y=2024 표와 Y=2025 표가 같다 — 시즌 분리가 안 됐다"
print(f"\n🟢 Y=2024 / Y=2025 표 상이 확인 (공통 투수 {len(common)}명)")

REP["wall_s"] = round(time.time() - t0, 1)
json.dump(REP, open(f"{OUT}/tables_report.json", "w"), indent=1,
          ensure_ascii=False)
print(f"\ns1 완료 {REP['wall_s']}s")
