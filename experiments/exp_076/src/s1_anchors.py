"""exp_076 stage 1 — 동결 앵커 제작 (라벨 복원 없음).

🔴 config 지시: `experiments/exp_037/src/labels.py` 를 부르지 않는다.
    이 실험의 축은 전부 **주최측이 행마다 주는 열**과 **주최측 as-of 열에서 읽은
    동결 앵커**의 산술이다.  복원 라벨이 한 군데도 안 들어가므로 복원 코드를
    부르면 변인만 는다.

앵커 정의 (`build_AI/src/ins/inseason.py` 의 build_anchor 와 **같은 규약**)
    시즌 Y 행의 앵커 = 그 투수의 season < Y 행 중 `asof_pitcher_n` 이 최대인
    행에서 읽은 `asof_pitcher_*_rate` 값.  = 직전 시즌 종료 시점 커리어 누적률.
    구종믹스 앵커는 `asof_pitcher_pitchmix_n` 최대 행에서 읽는다.

🔴 as-of 규율 (exp_074·075 와 같은 강도)
    ① sub["season"].max() <= Y-1
    ② (sub["season"] < Y).all()
    ③ 선택된 앵커 행 자체의 season 이 전부 < Y
    ④ Y=2024 표와 Y=2025 표가 실제로 다르다 (시즌 분리의 기계 증명)
    ⑤ 배포 앵커 `build_AI/src/ins/anchor_lt2024.npz` 의 성공률과 대조
       (같은 규약으로 만들었으면 반올림 오차 안에서 같아야 한다)

강건성 앵커 (robust, as-of 와 무관 — 신선도 점검용)
    앵커 행이 **season == Y-1** 인 투수만 남긴다.  직전 시즌에 안 던진 투수의
    묵은 앵커를 전량 폐기했을 때 판정이 바뀌는지 본다.

결측은 NaN 으로 남긴다.  sentinel 충전 금지.
"""
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

H = "/home/<user>/SungjinPark"
OUT = f"{H}/exp076_rg"
TRAIN = f"{H}/data/train.csv"
ANCHOR_LT2024 = f"{H}/build_AI/src/ins/anchor_lt2024.npz"

RATE_SUC = "asof_pitcher_success_rate"
RATE_MID = "asof_pitcher_middle_rate"
MIX = ["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
       "asof_pitcher_offspeed_rate"]
PREV = [f"asof_pitcher_prev{k}_game_{m}_rate"
        for m in ("success", "middle") for k in (1, 3, 5)]

t0 = time.time()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


USE = (["row_id", "season", "pitcher_id", "asof_pitcher_n",
        "asof_pitcher_pitchmix_n", RATE_SUC, RATE_MID] + MIX + PREV)

print("train.csv 읽는 중 …")
df = pd.read_csv(TRAIN, encoding="utf-8-sig", usecols=USE)
print(f"  {len(df):,}행 · {time.time()-t0:.0f}s")

REP = {"train_sha256": sha_file(TRAIN), "n_rows": int(len(df)),
       "labels_py_used": False,
       "labels_py_note": ("🔴 라벨 복원 코드를 부르지 않았다 — 이 실험의 축은 "
                          "주최측 행 열과 주최측 as-of 열에서 읽은 동결 앵커의 "
                          "산술뿐이다 (config 지시)."),
       "seasons": sorted(int(s) for s in df["season"].unique()),
       "n_pitchers": int(df["pitcher_id"].nunique())}

# ── 구조 점검 1: pitchmix_n 이 asof_pitcher_n 과 같은 계수기인가 ────────
n = df["asof_pitcher_n"].to_numpy(np.float64)
mn = df["asof_pitcher_pitchmix_n"].to_numpy(np.float64)
eqm = (n == mn) | (np.isnan(n) & np.isnan(mn))
REP["pitchmix_n_equals_asof_pitcher_n"] = {
    "frac_equal": float(eqm.mean()), "n_differ": int((~eqm).sum()),
    "note": ("두 계수기가 같으면 구종믹스 앵커 행 = 커리어 앵커 행이다. "
             "그래도 앵커는 각자의 계수기 argmax 로 따로 뽑는다.")}
print(f"  pitchmix_n == asof_pitcher_n : {eqm.mean():.6f} "
      f"(다른 행 {int((~eqm).sum()):,})")

# ── 구조 점검 2: prev 열은 시즌마다 리셋되는가 (커버리지 기전) ─────────
df["_ord_in_season"] = df.groupby(["pitcher_id", "season"], sort=False).cumcount()
p1 = df["asof_pitcher_prev1_game_success_rate"].to_numpy(np.float64)
first_in_season = (df["_ord_in_season"].to_numpy() == 0)
REP["prev_column_season_reset"] = {
    "nan_rate_prev1_overall": float(np.isnan(p1).mean()),
    "nan_rate_prev1_first_row_of_pitcher_season": float(
        np.isnan(p1[first_in_season]).mean()),
    "nan_rate_prev1_by_season": {
        str(int(s)): float(np.isnan(
            p1[df["season"].to_numpy() == s]).mean())
        for s in sorted(df["season"].unique())},
    "note": ("투수-시즌 첫 행에서 prev1 이 거의 전부 NaN 이면 prev 열은 "
             "시즌 내 경기 집계이고 시즌 경계에서 리셋된다는 뜻이다. "
             "→ 배포 test(2025) 의 초반 행에서 NaN 이 몰린다.")}
print(f"  prev1 NaN 전체 {REP['prev_column_season_reset']['nan_rate_prev1_overall']:.6f}"
      f" · 투수-시즌 첫 행 "
      f"{REP['prev_column_season_reset']['nan_rate_prev1_first_row_of_pitcher_season']:.6f}")

# prev1 NaN 이 시즌 내 몇 번째 행까지 지속되는가 (2024 시즌 기준)
sub24 = df[df["season"] == 2024]
o24 = sub24["_ord_in_season"].to_numpy()
p124 = sub24["asof_pitcher_prev1_game_success_rate"].to_numpy(np.float64)
bins = [(0, 0), (1, 10), (11, 30), (31, 60), (61, 120), (121, 10 ** 9)]
REP["prev1_nan_rate_by_within_season_rank_2024"] = {
    f"{a}-{b}": float(np.isnan(p124[(o24 >= a) & (o24 <= b)]).mean())
    for a, b in bins}
REP["prev1_nan_cumulative_rank_share_2024"] = {
    "share_of_rows_with_prev1_nan_that_are_rank_lt_60":
        (float((o24[np.isnan(p124)] < 60).mean())
         if np.isnan(p124).any() else None)}
print(f"  2024 시즌내 순위별 prev1 NaN: "
      f"{REP['prev1_nan_rate_by_within_season_rank_2024']}")


def build_anchor(Y, robust):
    """season < Y 행만으로 투수별 동결 앵커를 만든다."""
    m = (df["season"].to_numpy() < Y)
    assert m.sum() > 0
    sub = df.loc[m]
    assert int(sub["season"].max()) <= Y - 1, "🔴 as-of 위반: season >= Y"
    assert bool((sub["season"].to_numpy() < Y).all()), "🔴 as-of 위반"

    ok_n = sub["asof_pitcher_n"].notna()
    i_car = sub.loc[ok_n].groupby("pitcher_id")["asof_pitcher_n"].idxmax()
    car = sub.loc[i_car, ["pitcher_id", "season", "asof_pitcher_n",
                          RATE_SUC, RATE_MID]].set_index("pitcher_id")
    assert bool((car["season"].to_numpy() < Y).all()), "🔴 앵커 행 season >= Y"

    ok_m = sub["asof_pitcher_pitchmix_n"].notna()
    i_mix = sub.loc[ok_m].groupby("pitcher_id")["asof_pitcher_pitchmix_n"].idxmax()
    mix = sub.loc[i_mix, ["pitcher_id", "season", "asof_pitcher_pitchmix_n"]
                  + MIX].set_index("pitcher_id")
    assert bool((mix["season"].to_numpy() < Y).all()), "🔴 믹스 앵커 행 season >= Y"

    out = pd.DataFrame(index=car.index.union(mix.index).astype(int))
    out.index.name = "pitcher_id"
    out["anc_n"] = car["asof_pitcher_n"].reindex(out.index).to_numpy(np.float64)
    out["anc_season"] = car["season"].reindex(out.index).to_numpy(np.float64)
    out["anc_suc"] = car[RATE_SUC].reindex(out.index).to_numpy(np.float64)
    out["anc_mid"] = car[RATE_MID].reindex(out.index).to_numpy(np.float64)
    out["anc_mix_n"] = mix["asof_pitcher_pitchmix_n"].reindex(
        out.index).to_numpy(np.float64)
    out["anc_mix_season"] = mix["season"].reindex(out.index).to_numpy(np.float64)
    for c, nm in zip(MIX, ("anc_fb", "anc_br", "anc_off")):
        out[nm] = mix[c].reindex(out.index).to_numpy(np.float64)

    if robust:
        stale_c = ~(out["anc_season"].to_numpy() == Y - 1)
        stale_m = ~(out["anc_mix_season"].to_numpy() == Y - 1)
        for c in ("anc_suc", "anc_mid"):
            out.loc[stale_c, c] = np.nan
        for c in ("anc_fb", "anc_br", "anc_off"):
            out.loc[stale_m, c] = np.nan

    meta = {"Y": Y, "robust_require_prev_season": robust,
            "season_max_used": int(sub["season"].max()),
            "season_min_used": int(sub["season"].min()),
            "n_rows_used": int(m.sum()), "n_pitchers": int(len(out)),
            "anchor_row_season_max": float(np.nanmax(out["anc_season"])),
            "anchor_row_season_counts": {
                str(int(k)): int(v) for k, v in
                pd.Series(out["anc_season"]).value_counts().sort_index().items()},
            "finite_rate": {c: float(np.isfinite(
                out[c].to_numpy(np.float64)).mean()) for c in out.columns},
            "describe": {c: {k: float(v) for k, v in
                             out[c].describe().to_dict().items()}
                         for c in out.columns}}
    return out.sort_index(), meta


os.makedirs(OUT, exist_ok=True)
TAB = {}
for Y, robust, tag in ((2024, False, "Y2024"), (2025, False, "Y2025"),
                       (2024, True, "Y2024robust")):
    tb, meta = build_anchor(Y, robust)
    TAB[tag] = tb
    REP[f"anchor_{tag}"] = meta
    tb.to_csv(f"{OUT}/anchor_{tag}.csv")
    print(f"\n=== {tag} 앵커 (season {meta['season_min_used']}~"
          f"{meta['season_max_used']} · 투수 {meta['n_pitchers']}명 · "
          f"robust={robust}) ===")
    for c in ("anc_suc", "anc_mid", "anc_fb", "anc_br", "anc_off"):
        d = meta["describe"][c]
        print(f"  {c:<10} 유한 {meta['finite_rate'][c]:.4f}  "
              f"median {d.get('50%', float('nan')):+.5f}  "
              f"[{d.get('min', float('nan')):+.5f}, "
              f"{d.get('max', float('nan')):+.5f}]")

# ── ④ 시즌 분리의 기계 증명 ──────────────────────────────────────────
a, b = TAB["Y2024"]["anc_suc"], TAB["Y2025"]["anc_suc"]
common = a.index.intersection(b.index)
same = np.allclose(a.loc[common].to_numpy(np.float64),
                   b.loc[common].to_numpy(np.float64), equal_nan=True)
REP["Y2024_vs_Y2025_identical"] = bool(same)
assert not same, "🔴 Y=2024 앵커와 Y=2025 앵커가 같다 — 시즌 분리가 안 됐다"
d_ = np.abs(a.loc[common].to_numpy(np.float64) - b.loc[common].to_numpy(np.float64))
REP["Y2024_vs_Y2025_anc_suc_max_abs_diff"] = float(np.nanmax(d_))
print(f"\n🟢 Y2024 / Y2025 앵커 상이 (공통 투수 {len(common)}명, "
      f"anc_suc 최대차 {np.nanmax(d_):.6f})")

# ── ⑤ 배포 앵커와의 대조 ─────────────────────────────────────────────
z = np.load(ANCHOR_LT2024, allow_pickle=False)
dep = pd.DataFrame({"p_n": z["p_n"], "c_suc": z["p_c"][:, 0]},
                   index=pd.Index(z["p_ids"].astype(int), name="pitcher_id"))
dep["dep_rate"] = dep["c_suc"] / np.maximum(dep["p_n"], 1.0)
ci = TAB["Y2024"].index.intersection(dep.index)
mine = TAB["Y2024"].loc[ci, "anc_suc"].to_numpy(np.float64)
theirs = dep.loc[ci, "dep_rate"].to_numpy(np.float64)
nn = dep.loc[ci, "p_n"].to_numpy(np.float64)
mm = np.isfinite(mine) & np.isfinite(theirs)
diff = np.abs(mine[mm] - theirs[mm])
tol = 0.5 / np.maximum(nn[mm], 1.0)          # _counts 의 반올림 상한
REP["deployed_anchor_crosscheck"] = {
    "anchor_lt2024_sha256": sha_file(ANCHOR_LT2024),
    "n_pitchers_deployed": int(len(dep)),
    "n_pitchers_mine_Y2024": int(len(TAB["Y2024"])),
    "n_compared": int(mm.sum()),
    "max_abs_diff": float(diff.max()) if mm.sum() else None,
    "n_exceeding_rounding_bound": int((diff > tol + 1e-12).sum()),
    "pitcher_id_set_equal": bool(set(TAB["Y2024"].index) == set(dep.index)),
    "note": ("배포 inseason.build_anchor 는 rate 를 정수 카운트로 반올림해 "
             "저장한다.  따라서 c/n 과 원 rate 의 차이 상한은 0.5/n 이다. "
             "그 상한을 넘는 투수가 0명이면 같은 규약으로 만든 것이다.")}
print(f"\n🟢 배포 앵커 대조: 투수 {int(mm.sum()):,}명 · 최대차 "
      f"{diff.max():.3e} · 반올림 상한 초과 "
      f"{int((diff > tol + 1e-12).sum())}명 · id 집합 동일 "
      f"{REP['deployed_anchor_crosscheck']['pitcher_id_set_equal']}")
assert REP["deployed_anchor_crosscheck"]["n_exceeding_rounding_bound"] == 0, \
    "🔴 배포 앵커와 규약이 다르다"

# ── robust 판 차이 ───────────────────────────────────────────────────
so, ss = TAB["Y2024"], TAB["Y2024robust"]
REP["robust_vs_orig"] = {
    c: {"n_dropped": int((np.isfinite(so[c].to_numpy(np.float64))
                          & ~np.isfinite(ss[c].to_numpy(np.float64))).sum()),
        "finite_orig": float(np.isfinite(so[c].to_numpy(np.float64)).mean()),
        "finite_robust": float(np.isfinite(ss[c].to_numpy(np.float64)).mean())}
    for c in ("anc_suc", "anc_mid", "anc_fb", "anc_br", "anc_off")}
print(f"  robust 판에서 버린 투수: {REP['robust_vs_orig']['anc_suc']['n_dropped']}명 "
      f"(직전 시즌 미등판)")

REP["anchor_sha256"] = {t: sha_file(f"{OUT}/anchor_{t}.csv")
                        for t in ("Y2024", "Y2025", "Y2024robust")}
REP["wall_s_s1"] = round(time.time() - t0, 1)
json.dump(REP, open(f"{OUT}/anchors_report.json", "w"), indent=1,
          ensure_ascii=False)
print(f"\ns1 완료 {REP['wall_s_s1']}s")
