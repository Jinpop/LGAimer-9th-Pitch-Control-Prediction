"""exp_074 stage 1 — 라벨 복원 + 경기 분절 + as-of 투수 집계표.

🔴 라벨 복원 코드는 **재사용**한다.  새로 만들지 않는다.
    experiments/exp_037/src/labels.py 를 sys.path 로 그대로 import 한다
    (sha256 439cd504478b4bc8162e9dedc32be27bfadd5d1178472d6b384b60f46ed7a9f2,
     exp_030 / exp_037 / build_AA 세 곳의 사본이 바이트 동일).
    한 글자도 고치지 않는다 — 고치면 이 실험이 그 코드의 검증이 돼버린다.

🔴 as-of 규율 (이 실험의 유일한 치명적 실패 지점)
    시즌 Y 행에 붙는 값은 **season < Y** train 행만으로 만든다.
    시즌별 표를 따로 만들고 경계를 assert 한다.
    Y=2024 는 스크린용(프레임이 전부 season 2024), Y=2025 는 배포 커버리지 확인용.
    표는 투수당 스칼라 1개 — `ins_*`·`d_form_eb` 와 정확히 같은 배포 형태다.

🔴 경기 분절 — 주최측은 game_id 를 주지 않는다.
    투수 안에서 **파일 순서**로 다음 중 하나라도 바뀌면 새 경기로 본다:
      season · game_month · game_dayofweek
      asof_pitcher_prev{1,3,5}_game_{success,middle}_rate  (경기 안에서는 상수다)
    검증 게이트: 경기 k 에서 관측된 prev1_game_success_rate 가
    **우리가 분절한 경기 k-1 의 control_success 평균**과 일치하는가.
    (게이트는 복원 라벨이 아니라 **정답 열 control_success** 로 한다 —
     분절과 복원을 독립으로 검사하기 위해서다.)

사전 등록된 최소 표본 (🔴 결과를 보기 전에 정한다)
    fm_*_g5 · fm_suc_sd5 · supp_fm_suc_g5 : season<Y 경기 수 >= 5
    fm_suc_sd10 · fm_suc_trend            : season<Y 경기 수 >= 10
    경기당 최소 투구수 하한은 두지 않는다 — 두면 결과를 보고 고르는 문이 열린다.
    결측은 NaN 으로 남긴다.  sentinel 충전 금지.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

H = "/home/<user>/SungjinPark"
OUT = f"{H}/exp074_fm"
TRAIN = f"{H}/data/train.csv"
LABELS_SRC = f"{H}/experiments/exp_037/src/labels.py"
LABELS_SHA = "439cd504478b4bc8162e9dedc32be27bfadd5d1178472d6b384b60f46ed7a9f2"
INSEASON_SRC = f"{H}/build_BN/pkg_BN/inseason.py"
INSEASON_SHA = "1b4a407d2b53f2cbdbf0d2c837d0cab7c3db5e1e01bfc2e6b639813723303086"
ANCHOR_LT2024 = f"{H}/build_AI/src/ins/anchor_lt2024.npz"
ANCHOR_LT2024_SHA = \
    "0e4b586da41cd6e78c5e19d4dd656bc9df088413625945e2a88ae85d100e5eb7"

MIN_G5, MIN_G10 = 5, 10
t0 = time.time()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# ── 🔴 재사용 코드 sha 대조 (개조 금지의 기계 증명) ────────────────────
for p, want in ((LABELS_SRC, LABELS_SHA), (INSEASON_SRC, INSEASON_SHA),
                (ANCHOR_LT2024, ANCHOR_LT2024_SHA)):
    got = sha_file(p)
    assert got == want, f"sha 불일치 {p}: {got}"
    print(f"🟢 재사용 {os.path.relpath(p, H):<44} {got[:16]}")

sys.path.insert(0, os.path.dirname(LABELS_SRC))
sys.path.insert(0, os.path.dirname(INSEASON_SRC))
import labels as LB          # noqa: E402  (exp_037/src/labels.py 원본)
import inseason as INS       # noqa: E402  (pkg_BN/inseason.py 원본)

assert LB.RATE_COLS == {"success": "asof_pitcher_success_rate",
                        "reverse": "asof_pitcher_reverse_rate",
                        "middle": "asof_pitcher_middle_rate",
                        "ball": "asof_pitcher_ball_rate",
                        "strike": "asof_pitcher_strike_rate"}

# ── train.csv ─────────────────────────────────────────────────────────
GAMEKEY = ["season", "game_month", "game_dayofweek"] + [
    f"asof_pitcher_prev{k}_game_{m}_rate"
    for m in ("success", "middle") for k in (1, 3, 5)]
USE = (["row_id", "pitcher_id", "batter_id", "asof_pitcher_n",
        "asof_batter_n", "asof_batter_success_rate",
        "asof_batter_middle_rate", "control_success"]
       + list(LB.RATE_COLS.values()) + GAMEKEY)

print(f"\ntrain.csv 읽는 중 …")
df = pd.read_csv(TRAIN, encoding="utf-8-sig", usecols=USE)
print(f"  {len(df):,}행 · {time.time()-t0:.0f}s")
REP = {"train_sha256": sha_file(TRAIN), "n_rows": int(len(df)),
       "labels_py_sha256": LABELS_SHA, "labels_py_path": "experiments/exp_037/src/labels.py",
       "inseason_py_sha256": INSEASON_SHA,
       "anchor_lt2024_sha256": ANCHOR_LT2024_SHA,
       "seasons": sorted(int(s) for s in df["season"].unique()),
       "registered_min_games": {"g5_axes": MIN_G5, "g10_axes": MIN_G10,
                                "min_pitches_per_game": "없음(사전 등록)"}}

# ── 🔴 라벨 복원 (원본 코드 그대로 호출) ──────────────────────────────
print("\n=== 라벨 복원 (exp_037/src/labels.py recover) ===")
REP["recovery_diag"] = LB.recover(df, verbose=True)
assert REP["recovery_diag"]["partition_sums_to_one"]
assert REP["recovery_diag"]["success_recovery"]["agree_pct"] == 100.0, \
    "복원 success 가 control_success 와 100% 일치하지 않는다"
print("  🟢 복원 success == control_success 100.00%")

# ── 경기 분절 (투수 안 · 파일 순서) ───────────────────────────────────
print("\n=== 경기 분절 ===")
g = df.groupby("pitcher_id", sort=False)
chg = np.zeros(len(df), dtype=bool)
for k in GAMEKEY:
    cur = df[k]
    prev = g[k].shift(1)
    ne = (cur != prev) & ~(cur.isna() & prev.isna())
    chg |= ne.to_numpy()
first = (g.cumcount().to_numpy() == 0)
df["_new"] = (chg | first).astype(np.int64)
g2 = df.groupby("pitcher_id", sort=False)          # _new 를 본 새 groupby
df["game_ord"] = g2["_new"].cumsum().to_numpy() - 1
n_games_total = int(df.groupby(["pitcher_id", "game_ord"], sort=False).ngroups)
print(f"  경기 {n_games_total:,}개 · 투수 {df['pitcher_id'].nunique()}명")

# 🔴 asof_pitcher_n 이 투수 안 파일 순서로 +1 인지 (labels.py 가 이미 검증했지만
#    분절도 같은 시간축을 쓴다는 것을 여기서 다시 못박는다)
assert REP["recovery_diag"]["n_step_all_plus_one"], "파일 순서가 시간축이 아니다"

ok = df["recov_ok"].to_numpy()
gs = (df.assign(_ok=ok.astype(np.float64))
        .groupby(["pitcher_id", "game_ord"], sort=True)
        .agg(season=("season", "first"),
             n_row=("row_id", "size"),
             n_ok=("_ok", "sum"),
             y_sum=("control_success", "sum"),
             s_success=("is_success", "sum"),
             s_reverse=("is_reverse", "sum"),
             s_middle=("is_middle", "sum"),
             s_ball=("is_ball", "sum"),
             s_strike=("is_strike", "sum"))
        .reset_index())
REP["games"] = {
    "n_games": int(len(gs)),
    "pitches_per_game": {k: float(v) for k, v in
                         gs["n_row"].describe().to_dict().items()},
    "n_ok_rows": int(ok.sum()), "n_not_ok_rows": int((~ok).sum())}
print(f"  경기당 투구수 median {gs['n_row'].median():.0f} "
      f"[{gs['n_row'].min()}, {gs['n_row'].max()}]")

# ── 🔴 분절 검증 게이트 — prev1 관측값 vs 우리가 분절한 직전 경기의 정답률 ──
gg = gs.groupby("pitcher_id", sort=False)
roll = lambda col, k: gg[col].transform(
    lambda s: s.rolling(k, min_periods=k).sum().shift(1)).to_numpy(np.float64)
obs = (df.groupby(["pitcher_id", "game_ord"], sort=True)
       [["asof_pitcher_prev1_game_success_rate",
         "asof_pitcher_prev3_game_success_rate"]].first())
calc_prev1 = np.where(roll("n_row", 1) > 0, roll("y_sum", 1) / roll("n_row", 1),
                      np.nan)
obs_prev1 = obs["asof_pitcher_prev1_game_success_rate"].to_numpy(np.float64)
m = np.isfinite(calc_prev1) & np.isfinite(obs_prev1)
agree = np.abs(calc_prev1[m] - obs_prev1[m]) <= 5e-7
# prev3 = 직전 3경기 **투구 풀링** 성공률인지 (정의 확인용)
calc_prev3 = np.where(roll("n_row", 3) > 0, roll("y_sum", 3) / roll("n_row", 3),
                      np.nan)
obs_prev3 = obs["asof_pitcher_prev3_game_success_rate"].to_numpy(np.float64)
m3 = np.isfinite(calc_prev3) & np.isfinite(obs_prev3)
agree3 = np.abs(calc_prev3[m3] - obs_prev3[m3]) <= 5e-7
REP["segmentation_gate"] = {
    "prev1_n_compared": int(m.sum()),
    "prev1_agree_pct": round(100.0 * float(agree.mean()), 4),
    "prev1_max_abs_diff": float(np.abs(calc_prev1[m] - obs_prev1[m]).max()),
    "prev3_pooled_n_compared": int(m3.sum()),
    "prev3_pooled_agree_pct": round(100.0 * float(agree3.mean()), 4),
    "gate_uses": "control_success (정답 열) — 복원 라벨과 독립"}
print(f"\n  🔴 분절 게이트  prev1 일치 "
      f"{REP['segmentation_gate']['prev1_agree_pct']:.4f}% "
      f"({int(m.sum()):,}경기 비교)")
print(f"     prev3(풀링) 일치 "
      f"{REP['segmentation_gate']['prev3_pooled_agree_pct']:.4f}%")

# ── as-of 표 ──────────────────────────────────────────────────────────
AXES = ["fm_rev_g5", "fm_ball_g5", "fm_strike_g5", "fm_suc_sd5",
        "fm_suc_sd10", "fm_suc_trend"]
SUPP = ["supp_fm_suc_g5", "supp_fm_mid_g5", "fm_ngames", "fm_npitch"]


def build(Y):
    sub = gs[gs["season"] < Y]
    assert len(sub) > 0
    assert int(sub["season"].max()) <= Y - 1, "🔴 as-of 위반: 표에 season >= Y"
    assert bool((sub["season"].to_numpy() < Y).all()), "🔴 as-of 위반"
    meta = {"Y": Y, "season_max_used": int(sub["season"].max()),
            "season_min_used": int(sub["season"].min()),
            "n_games_used": int(len(sub)),
            "n_pitchers": int(sub["pitcher_id"].nunique())}
    rows = {}
    for pid, gp in sub.groupby("pitcher_id", sort=True):
        gp = gp.sort_values("game_ord")
        n = gp["n_row"].to_numpy(np.float64)
        ok_n = gp["n_ok"].to_numpy(np.float64)
        S = {k: gp["s_" + k].to_numpy(np.float64)
             for k in ("success", "reverse", "middle", "ball", "strike")}
        ng = len(gp)
        r = {"fm_ngames": float(ng), "fm_npitch": float(n.sum())}

        def pooled(key, sl):
            d = ok_n[sl].sum()
            return float(S[key][sl].sum() / d) if d > 0 else np.nan

        car = {k: pooled(k, slice(None)) for k in S}
        if ng >= MIN_G5:
            l5 = slice(ng - 5, ng)
            r["fm_rev_g5"] = pooled("reverse", l5) - car["reverse"]
            r["fm_ball_g5"] = pooled("ball", l5) - car["ball"]
            r["fm_strike_g5"] = pooled("strike", l5) - car["strike"]
            r["supp_fm_suc_g5"] = pooled("success", l5) - car["success"]
            r["supp_fm_mid_g5"] = pooled("middle", l5) - car["middle"]
            with np.errstate(invalid="ignore", divide="ignore"):
                gr5 = np.where(ok_n[l5] > 0, S["success"][l5] / ok_n[l5], np.nan)
            gr5 = gr5[np.isfinite(gr5)]
            r["fm_suc_sd5"] = float(gr5.std(ddof=1)) if len(gr5) >= 2 else np.nan
        if ng >= MIN_G10:
            l10 = slice(ng - 10, ng)
            with np.errstate(invalid="ignore", divide="ignore"):
                gr10 = np.where(ok_n[l10] > 0, S["success"][l10] / ok_n[l10],
                                np.nan)
            gr10 = gr10[np.isfinite(gr10)]
            r["fm_suc_sd10"] = (float(gr10.std(ddof=1)) if len(gr10) >= 2
                                else np.nan)
            r["fm_suc_trend"] = pooled("success", slice(ng - 3, ng)) \
                - pooled("success", l10)
        rows[int(pid)] = r
    out = pd.DataFrame(rows).T.reindex(columns=AXES + SUPP).sort_index()
    out.index.name = "pitcher_id"
    meta["finite_rate"] = {c: float(np.isfinite(out[c].to_numpy(np.float64)).mean())
                           for c in out.columns}
    meta["describe"] = {c: {k: float(v) for k, v in
                            out[c].describe().to_dict().items()}
                        for c in out.columns}
    return out, meta


os.makedirs(OUT, exist_ok=True)
TAB = {}
for Y in (2024, 2025):
    tb, meta = build(Y)
    TAB[Y] = tb
    REP[f"table_Y{Y}"] = meta
    tb.to_csv(f"{OUT}/fm_table_Y{Y}.csv")
    print(f"\n=== Y={Y} 표 (season {meta['season_min_used']}"
          f"~{meta['season_max_used']} · 경기 {meta['n_games_used']:,} · "
          f"투수 {meta['n_pitchers']}명) ===")
    for c in tb.columns:
        d = meta["describe"][c]
        print(f"  {c:<16} 유한 {meta['finite_rate'][c]:.4f}  "
              f"median {d.get('50%', float('nan')):+.5f}  "
              f"[{d.get('min', float('nan')):+.5f}, "
              f"{d.get('max', float('nan')):+.5f}]")

a, b = TAB[2024]["fm_rev_g5"], TAB[2025]["fm_rev_g5"]
common = a.index.intersection(b.index)
same = np.allclose(a.loc[common].to_numpy(np.float64),
                   b.loc[common].to_numpy(np.float64), equal_nan=True)
REP["Y2024_vs_Y2025_identical"] = bool(same)
assert not same, "🔴 Y=2024 표와 Y=2025 표가 같다 — 시즌 분리가 안 됐다"
print(f"\n🟢 Y=2024 / Y=2025 표 상이 확인 (공통 투수 {len(common)}명)")

# ── 부수: ins 앵커(lt2024) 재구축 대조 — 원본 코드로 만든 앵커가 맞는지 ──
# (부수 확인 — 실패해도 이 실험의 결론에 영향이 없으므로 기록만 한다)
hist = df[df["season"] < 2024]
A_re = INS.build_anchor(hist)
A_sv = INS.load(ANCHOR_LT2024)
per_key = {}
for k in A_sv:
    try:
        per_key[k] = bool(np.allclose(np.asarray(A_re[k], np.float64),
                                      np.asarray(A_sv[k], np.float64),
                                      equal_nan=True))
    except Exception as e:                                   # 모양 불일치 등
        per_key[k] = f"불가:{type(e).__name__}"
REP["ins_anchor_lt2024_rebuild_matches"] = per_key
print(f"🟢 ins 앵커(season<2024) 재구축 대조: {per_key}")

REP["wall_s"] = round(time.time() - t0, 1)
json.dump(REP, open(f"{OUT}/tables_report.json", "w"), indent=1,
          ensure_ascii=False)
print(f"\ns1 완료 {REP['wall_s']}s")
