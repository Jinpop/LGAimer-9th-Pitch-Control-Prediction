"""exp_075 stage 1 — 라벨 복원 + 타자 블록 분절 + as-of **타자** 집계표.

🔴 라벨 복원 코드는 **재사용**한다.  새로 만들지 않는다.
    experiments/exp_037/src/labels.py 를 sys.path 로 그대로 import 한다
    (sha256 439cd504478b4bc8162e9dedc32be27bfadd5d1178472d6b384b60f46ed7a9f2).
    한 글자도 고치지 않는다 — 고치면 이 실험이 그 코드의 검증이 돼버린다.
    exp_074 가 같은 방식으로 재사용했고 복원 success 가 control_success 와
    1,474,300행 100.00% 일치했다.

🔴 as-of 규율 (이 실험의 유일한 치명적 실패 지점)
    시즌 Y 행에 붙는 값은 **season < Y** train 행만으로 만든다.
    시즌별 표를 따로 만들고 경계를 assert 한다 (max<=Y-1 과 (<Y).all() 둘 다).
    Y=2024 는 스크린용(프레임이 전부 season 2024), Y=2025 는 배포 커버리지 확인용.
    표는 타자당 스칼라 — `asof_batter_success_rate` 가 배포되는 형태 그대로다.

🔴 복원 라벨을 **쓰지 않아도 되는 축은 쓰지 않는다**
    success 계열(supp_bt_suc · bt_suc_sd · bt_hand_dev)은 정답 열
    `control_success` 로 직접 만든다.  복원(t+1 차분)은 reverse·ball·strike
    세 축에만 쓴다.  따라서 시즌 경계 노출은 그 세 축에만 있다.

🔴 타자 블록 분절 — game_id 가 없다.
    타자의 한 경기 블록 = (season, game_month, game_dayofweek,
                          batter_team_id, pitcher_team_id).
    한 타자는 한 경기에서 한 상대 팀만 만난다.  같은 두 팀이 같은 달 같은
    요일에 두 번 붙으면 충돌한다 — 그 충돌률을 **투수 경기 분절**(exp_074 가
    주최측 prev1 열로 99.9978% 검증한 것)에 대고 측정해 보고한다.
    이름을 '경기' 라 부르되 측정된 충돌률을 함께 적는다.

사전 등록된 최소 표본 (🔴 결과를 보기 전에 정한다)
    bt_rev · bt_ball · bt_strike · supp_bt_suc · supp_bt_mid : season<Y ok행 >= 100
    bt_suc_sd                                               : season<Y 블록 >= 5
    bt_hand_dev                                             : 좌·우 각각 행 >= 50
    행 수준 축(bt_x_psr · bt_d_psr)은 표가 아니라 주최측 열의 산술이라 최소
    표본 조건이 없다 (주최측 열이 NaN 이면 NaN).
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
OUT = f"{H}/exp075_bt"
TRAIN = f"{H}/data/train.csv"
LABELS_SRC = f"{H}/experiments/exp_037/src/labels.py"
LABELS_SHA = "439cd504478b4bc8162e9dedc32be27bfadd5d1178472d6b384b60f46ed7a9f2"

MIN_OK_ROWS = 100      # 커리어 비율 축
MIN_BLOCKS = 5         # 기복 축
MIN_HAND_ROWS = 50     # 플래툰 축 (좌·우 각각)
t0 = time.time()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


# ── 🔴 재사용 코드 sha 대조 (개조 금지의 기계 증명) ────────────────────
got = sha_file(LABELS_SRC)
assert got == LABELS_SHA, f"sha 불일치 {LABELS_SRC}: {got}"
print(f"🟢 재사용 experiments/exp_037/src/labels.py  {got[:16]}")
sys.path.insert(0, os.path.dirname(LABELS_SRC))
import labels as LB          # noqa: E402  (exp_037/src/labels.py 원본)

assert LB.RATE_COLS == {"success": "asof_pitcher_success_rate",
                        "reverse": "asof_pitcher_reverse_rate",
                        "middle": "asof_pitcher_middle_rate",
                        "ball": "asof_pitcher_ball_rate",
                        "strike": "asof_pitcher_strike_rate"}

# ── train.csv ─────────────────────────────────────────────────────────
PGAMEKEY = ["season", "game_month", "game_dayofweek"] + [
    f"asof_pitcher_prev{k}_game_{m}_rate"
    for m in ("success", "middle") for k in (1, 3, 5)]
USE = (["row_id", "pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
        "pitcher_team_id", "batter_team_id", "asof_pitcher_n",
        "asof_batter_n", "asof_batter_success_rate",
        "asof_batter_middle_rate", "control_success"]
       + list(LB.RATE_COLS.values()) + PGAMEKEY)

print("\ntrain.csv 읽는 중 …")
df = pd.read_csv(TRAIN, encoding="utf-8-sig", usecols=USE)
print(f"  {len(df):,}행 · {time.time()-t0:.0f}s")
REP = {"train_sha256": sha_file(TRAIN), "n_rows": int(len(df)),
       "labels_py_sha256": LABELS_SHA,
       "labels_py_path": "experiments/exp_037/src/labels.py",
       "seasons": sorted(int(s) for s in df["season"].unique()),
       "registered_min_samples": {
           "career_rate_axes_min_ok_rows": MIN_OK_ROWS,
           "bt_suc_sd_min_blocks": MIN_BLOCKS,
           "bt_hand_dev_min_rows_per_hand": MIN_HAND_ROWS,
           "row_level_axes": "없음 (주최측 열의 산술)"},
       "n_batters": int(df["batter_id"].nunique()),
       "n_pitchers": int(df["pitcher_id"].nunique())}

# ── 🔴 라벨 복원 (원본 코드 그대로 호출) ──────────────────────────────
print("\n=== 라벨 복원 (exp_037/src/labels.py recover) ===")
REP["recovery_diag"] = LB.recover(df, verbose=True)
assert REP["recovery_diag"]["partition_sums_to_one"]
assert REP["recovery_diag"]["success_recovery"]["agree_pct"] == 100.0, \
    "복원 success 가 control_success 와 100% 일치하지 않는다"
assert REP["recovery_diag"]["n_step_all_plus_one"], "파일 순서가 시간축이 아니다"
print("  🟢 복원 success == control_success 100.00%")

# ── 🔴 asof_batter_n 도 타자 안 파일 순서로 +1 인가 (타자 시간축 확인) ──
gb = df.groupby("batter_id", sort=False)
bn = df["asof_batter_n"].to_numpy(np.float64)
bn_next = gb["asof_batter_n"].shift(-1).to_numpy(np.float64)
hn = ~np.isnan(bn_next)
bstep = bn_next[hn] - bn[hn]
REP["batter_time_axis"] = {
    "asof_batter_n_step_all_plus_one": bool(np.all(bstep == 1.0)),
    "violations": int(np.sum(bstep != 1.0)),
    "rows_compared": int(hn.sum()),
    "meaning": ("파일 순서가 타자에게도 같은 시간축인지 — 투수 쪽 검증(labels.py "
                "n_step_all_plus_one)의 타자판.  집계 자체는 season<Y 전량 합이라 "
                "순서에 의존하지 않지만, 시간축 전제를 여기서 기계로 확인한다.")}
print(f"  타자 시간축 +1 위반 {REP['batter_time_axis']['violations']:,}행 "
      f"/ {int(hn.sum()):,}")

# ── 투수 경기 분절 (exp_074 와 동일 · 타자 블록 충돌 측정용) ──────────
print("\n=== 투수 경기 분절 (충돌 측정 기준선) ===")
gp = df.groupby("pitcher_id", sort=False)
chg = np.zeros(len(df), dtype=bool)
for k in PGAMEKEY:
    cur, prev = df[k], gp[k].shift(1)
    chg |= ((cur != prev) & ~(cur.isna() & prev.isna())).to_numpy()
df["_new"] = (chg | (gp.cumcount().to_numpy() == 0)).astype(np.int64)
df["pgame_ord"] = (df.groupby("pitcher_id", sort=False)["_new"]
                   .cumsum().to_numpy() - 1)
n_pgames = int(df.groupby(["pitcher_id", "pgame_ord"], sort=False).ngroups)
print(f"  투수 경기 {n_pgames:,}개")

# ── 타자 블록 분절 ────────────────────────────────────────────────────
BKEY = ["season", "game_month", "game_dayofweek", "batter_team_id",
        "pitcher_team_id"]
df["bblock"] = (df.groupby(BKEY, sort=False).ngroup().to_numpy())
n_bblocks = int(df.groupby(["batter_id", "bblock"], sort=False).ngroups)
# 충돌 = 같은 (투수, 타자블록) 안에 투수 경기가 2개 이상 들어간 경우
coll = (df.groupby(["pitcher_id", "bblock"], sort=False)["pgame_ord"]
        .nunique())
REP["batter_blocks"] = {
    "key": BKEY,
    "n_batter_blocks": n_bblocks,
    "n_pitcher_games": n_pgames,
    "collision_frac_pitcher_block_spans_multiple_pitcher_games":
        round(float((coll > 1).mean()), 6),
    "collision_n": int((coll > 1).sum()),
    "collision_denom": int(len(coll)),
    "rows_per_batter_block": {
        k: float(v) for k, v in
        df.groupby(["batter_id", "bblock"], sort=False)["row_id"]
        .size().describe().to_dict().items()},
    "note": ("충돌 = 같은 두 팀이 같은 시즌·달·요일에 두 번 이상 붙어 서로 다른 "
             "투수 경기가 한 블록으로 합쳐진 비율.  기준선인 투수 경기 분절은 "
             "exp_074 에서 주최측 prev1 열과 99.9978% 일치가 확인된 것이다.")}
print(f"  타자 블록 {n_bblocks:,}개 · 충돌률 "
      f"{REP['batter_blocks']['collision_frac_pitcher_block_spans_multiple_pitcher_games']:.4%}")

# ── 시즌 경계 보수 마스크 (복원 쓰는 세 축에만 해당) ──────────────────
succ_season = df.groupby("pitcher_id", sort=False)["season"].shift(-1)
ok = df["recov_ok"].to_numpy()
REP["boundary"] = {}
for Y in (2024, 2025):
    cross = int((ok & (df["season"].to_numpy() < Y)
                 & ~(succ_season.to_numpy(np.float64) < Y)).sum())
    REP["boundary"][f"Y{Y}_rows_recovered_across_season_boundary"] = cross
print(f"  시즌 경계를 넘어 복원된 행: Y=2024 "
      f"{REP['boundary']['Y2024_rows_recovered_across_season_boundary']:,}")

AXES_TABLE = ["bt_rev", "bt_ball", "bt_strike", "bt_suc_sd", "bt_hand_dev"]
SUPP_TABLE = ["supp_bt_suc", "supp_bt_mid", "bt_n_rows", "bt_n_blocks"]
HANDS = sorted(df["pitcher_hand"].dropna().unique().tolist())
REP["pitcher_hand_values"] = [str(h) for h in HANDS]
assert len(HANDS) == 2, f"pitcher_hand 값이 2개가 아니다: {HANDS}"
HL, HR = HANDS[0], HANDS[1]     # 사전 등록: 정렬 첫 값 − 둘째 값
REP["bt_hand_dev_definition"] = f"success_rate(pitcher_hand=={HL}) - (=={HR})"


def build(Y, strict):
    """season < Y 행만으로 타자 표를 만든다.

    strict=True 이면 복원 라벨을 **후속 행의 season 도 < Y** 인 행으로 제한한다
    (시즌 경계를 넘는 차분 전량 폐기).  success 계열은 정답 열을 쓰므로
    strict 와 무관하다.
    """
    m_season = (df["season"].to_numpy() < Y)
    assert m_season.sum() > 0
    sub = df.loc[m_season]
    assert int(sub["season"].max()) <= Y - 1, "🔴 as-of 위반: season >= Y"
    assert bool((sub["season"].to_numpy() < Y).all()), "🔴 as-of 위반"

    okm = ok[m_season]
    if strict:
        okm = okm & (succ_season.to_numpy(np.float64)[m_season] < Y)
    bid = sub["batter_id"].to_numpy(np.int64)
    y = sub["control_success"].to_numpy(np.float64)
    okf = okm.astype(np.float64)

    A = pd.DataFrame({"batter_id": bid, "_ok": okf, "_y": y})
    for c in ("reverse", "ball", "strike", "middle"):
        A["_" + c] = np.where(okm, sub["is_" + c].to_numpy(np.float64), 0.0)
    A["_mid_y"] = np.where(okm, sub["is_middle"].to_numpy(np.float64), 0.0)
    G = A.groupby("batter_id", sort=True).agg(
        n_rows=("_y", "size"), y_sum=("_y", "sum"), ok_n=("_ok", "sum"),
        s_rev=("_reverse", "sum"), s_ball=("_ball", "sum"),
        s_strike=("_strike", "sum"), s_mid=("_mid_y", "sum"))

    out = pd.DataFrame(index=G.index.astype(int))
    out.index.name = "batter_id"
    den = G["ok_n"].to_numpy(np.float64)
    enough = den >= MIN_OK_ROWS
    for nm, col in (("bt_rev", "s_rev"), ("bt_ball", "s_ball"),
                    ("bt_strike", "s_strike"), ("supp_bt_mid", "s_mid")):
        v = np.where(enough & (den > 0), G[col].to_numpy(np.float64) / den,
                     np.nan)
        out[nm] = v
    nr = G["n_rows"].to_numpy(np.float64)
    out["supp_bt_suc"] = np.where(nr >= MIN_OK_ROWS,
                                  G["y_sum"].to_numpy(np.float64) / nr, np.nan)
    out["bt_n_rows"] = nr

    # 기복 — 블록별 성공률의 표준편차 (정답 열 사용)
    B = (pd.DataFrame({"batter_id": bid,
                       "bblock": sub["bblock"].to_numpy(np.int64), "_y": y})
         .groupby(["batter_id", "bblock"], sort=True)["_y"].mean()
         .reset_index())
    nb_ = B.groupby("batter_id")["_y"].size()
    sd_ = B.groupby("batter_id")["_y"].std(ddof=1)
    out["bt_n_blocks"] = nb_.reindex(out.index).to_numpy(np.float64)
    out["bt_suc_sd"] = np.where(out["bt_n_blocks"].to_numpy() >= MIN_BLOCKS,
                                sd_.reindex(out.index).to_numpy(np.float64),
                                np.nan)

    # 플래툰 — 좌완 상대 성공률 − 우완 상대 성공률 (정답 열 사용)
    Hd = (pd.DataFrame({"batter_id": bid,
                        "h": sub["pitcher_hand"].to_numpy(), "_y": y})
          .groupby(["batter_id", "h"], sort=True)["_y"]
          .agg(["size", "mean"]).unstack("h"))
    def hcol(stat, hv):
        try:
            return Hd[(stat, hv)].reindex(out.index).to_numpy(np.float64)
        except KeyError:
            return np.full(len(out), np.nan)
    nL, nR = hcol("size", HL), hcol("size", HR)
    mL, mR = hcol("mean", HL), hcol("mean", HR)
    out["bt_hand_dev"] = np.where((nL >= MIN_HAND_ROWS) & (nR >= MIN_HAND_ROWS),
                                  mL - mR, np.nan)

    out = out.reindex(columns=AXES_TABLE + SUPP_TABLE).sort_index()
    meta = {"Y": Y, "strict": strict,
            "season_max_used": int(sub["season"].max()),
            "season_min_used": int(sub["season"].min()),
            "n_rows_used": int(m_season.sum()),
            "n_ok_rows_used": int(okm.sum()),
            "n_batters": int(len(out)),
            "finite_rate": {c: float(np.isfinite(out[c].to_numpy(np.float64))
                                     .mean()) for c in out.columns},
            "describe": {c: {k: float(v) for k, v in
                             out[c].describe().to_dict().items()}
                         for c in out.columns}}
    return out, meta


os.makedirs(OUT, exist_ok=True)
TAB = {}
for Y, strict, tag in ((2024, False, "Y2024"), (2025, False, "Y2025"),
                       (2024, True, "Y2024strict")):
    tb, meta = build(Y, strict)
    TAB[tag] = tb
    REP[f"table_{tag}"] = meta
    tb.to_csv(f"{OUT}/bt_table_{tag}.csv")
    print(f"\n=== {tag} 표 (season {meta['season_min_used']}~"
          f"{meta['season_max_used']} · 타자 {meta['n_batters']}명 · "
          f"strict={strict}) ===")
    for c in tb.columns:
        d = meta["describe"][c]
        print(f"  {c:<14} 유한 {meta['finite_rate'][c]:.4f}  "
              f"median {d.get('50%', float('nan')):+.5f}  "
              f"[{d.get('min', float('nan')):+.5f}, "
              f"{d.get('max', float('nan')):+.5f}]")

a, b = TAB["Y2024"]["bt_rev"], TAB["Y2025"]["bt_rev"]
common = a.index.intersection(b.index)
same = np.allclose(a.loc[common].to_numpy(np.float64),
                   b.loc[common].to_numpy(np.float64), equal_nan=True)
REP["Y2024_vs_Y2025_identical"] = bool(same)
assert not same, "🔴 Y=2024 표와 Y=2025 표가 같다 — 시즌 분리가 안 됐다"
print(f"\n🟢 Y=2024 / Y=2025 표 상이 확인 (공통 타자 {len(common)}명)")

# 보수판이 원판과 실제로 다른가 (복원 쓰는 세 축에서만 달라야 한다)
so = TAB["Y2024"]; ss = TAB["Y2024strict"]
ci = so.index.intersection(ss.index)
REP["strict_vs_orig_max_abs_diff"] = {}
for c in AXES_TABLE + SUPP_TABLE:
    x = so.loc[ci, c].to_numpy(np.float64); z_ = ss.loc[ci, c].to_numpy(np.float64)
    m = np.isfinite(x) & np.isfinite(z_)
    REP["strict_vs_orig_max_abs_diff"][c] = (float(np.abs(x[m] - z_[m]).max())
                                             if m.sum() else None)
print(f"  보수판 최대차: {REP['strict_vs_orig_max_abs_diff']}")

REP["wall_s_s1"] = round(time.time() - t0, 1)
json.dump(REP, open(f"{OUT}/tables_report.json", "w"), indent=1,
          ensure_ascii=False)
print(f"\ns1 완료 {REP['wall_s_s1']}s")
