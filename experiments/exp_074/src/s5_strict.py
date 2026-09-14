"""exp_074 s5 — 🔴 시즌 경계 복원의 보수판 재실행 (리크 위생).

무엇을 닫는가
    라벨 복원은 is_X[t] = cnt_X[t+1] - cnt_X[t] 다.  따라서 어떤 투수의
    **season 2023 마지막 투구**의 결과를 읽으려면 그 다음 행 — 즉 그 투수의
    **season 2024 첫 행** — 의 asof_* 를 봐야 한다.

    그 asof_* 값 자체는 '그 2024 행 이전' 까지의 누적이라 정보 내용은 전부
    2023 이하다.  그래도 **표를 만들 때 프레임 시즌의 행을 한 번 만졌다**는
    사실은 남는다.  기억으로 안전하다고 적지 않고 기계로 닫는다.

보수판
    후속 행의 season 도 < Y 인 행만 복원 라벨을 쓴다.
    (= 시즌 경계를 넘는 차분을 전부 버린다.  투수당 최대 1투구 손실.)
    Y=2024 표를 그렇게 다시 만들어 6축을 재스크린하고 원판과 비교한다.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit

t0 = time.time()
H = "/home/<user>/SungjinPark"
OUT = f"{H}/exp074_fm"
FRAME = f"{H}/build_AC/work/tr2024"
PKG = f"{H}/build_BN/pkg_BN"
EPS, EPS_PROBE, N_ROWS = 1e-6, 0.02, 253507
MIN_G5, MIN_G10, Y = 5, 10, 2024
AXES = ["fm_rev_g5", "fm_ball_g5", "fm_strike_g5", "fm_suc_sd5",
        "fm_suc_sd10", "fm_suc_trend"]
lg = lambda p: logit(np.clip(np.asarray(p, np.float64), EPS, 1 - EPS))

sys.path.insert(0, f"{H}/experiments/exp_037/src")
import labels as LB  # noqa: E402

GAMEKEY = ["season", "game_month", "game_dayofweek"] + [
    f"asof_pitcher_prev{k}_game_{m}_rate"
    for m in ("success", "middle") for k in (1, 3, 5)]
USE = (["row_id", "pitcher_id", "asof_pitcher_n", "control_success"]
       + list(LB.RATE_COLS.values()) + GAMEKEY)
df = pd.read_csv(f"{H}/data/train.csv", encoding="utf-8-sig", usecols=USE)
LB.recover(df, verbose=False)

g = df.groupby("pitcher_id", sort=False)
chg = np.zeros(len(df), dtype=bool)
for k in GAMEKEY:
    cur, prev = df[k], g[k].shift(1)
    chg |= ((cur != prev) & ~(cur.isna() & prev.isna())).to_numpy()
df["_new"] = (chg | (g.cumcount().to_numpy() == 0)).astype(np.int64)
df["game_ord"] = df.groupby("pitcher_id", sort=False)["_new"].cumsum().to_numpy() - 1

succ_season = df.groupby("pitcher_id", sort=False)["season"].shift(-1)
ok = df["recov_ok"].to_numpy()
strict = ok & (succ_season.to_numpy(np.float64) < Y)
n_cross = int((ok & (df["season"].to_numpy() < Y)
               & ~(succ_season.to_numpy(np.float64) < Y)).sum())
print(f"🔴 시즌 경계를 넘어 복원된 행 (season<{Y} 중): {n_cross:,}")

REP = {"Y": Y, "n_rows_recovered_across_season_boundary": n_cross,
       "policy": "후속 행 season < Y 인 행만 복원 라벨 사용 (보수판)"}


def build(mask, tag):
    d = df.assign(_ok=mask.astype(np.float64))
    for c in ("success", "reverse", "middle", "ball", "strike"):
        d["is_" + c] = np.where(mask, d["is_" + c], np.nan)
    gs = (d.groupby(["pitcher_id", "game_ord"], sort=True)
          .agg(season=("season", "first"), n_row=("row_id", "size"),
               n_ok=("_ok", "sum"),
               **{f"s_{c}": (f"is_{c}", "sum") for c in
                  ("success", "reverse", "middle", "ball", "strike")})
          .reset_index())
    sub = gs[gs["season"] < Y]
    assert int(sub["season"].max()) <= Y - 1 and bool((sub["season"] < Y).all())
    rows = {}
    for pid, gp in sub.groupby("pitcher_id", sort=True):
        gp = gp.sort_values("game_ord")
        ok_n = gp["n_ok"].to_numpy(np.float64)
        S = {c: gp["s_" + c].to_numpy(np.float64) for c in
             ("success", "reverse", "middle", "ball", "strike")}
        ng = len(gp)
        r = {}

        def pooled(k, sl):
            dn = ok_n[sl].sum()
            return float(S[k][sl].sum() / dn) if dn > 0 else np.nan

        car = {k: pooled(k, slice(None)) for k in S}
        if ng >= MIN_G5:
            l5 = slice(ng - 5, ng)
            r["fm_rev_g5"] = pooled("reverse", l5) - car["reverse"]
            r["fm_ball_g5"] = pooled("ball", l5) - car["ball"]
            r["fm_strike_g5"] = pooled("strike", l5) - car["strike"]
            with np.errstate(invalid="ignore", divide="ignore"):
                gr = np.where(ok_n[l5] > 0, S["success"][l5] / ok_n[l5], np.nan)
            gr = gr[np.isfinite(gr)]
            r["fm_suc_sd5"] = float(gr.std(ddof=1)) if len(gr) >= 2 else np.nan
        if ng >= MIN_G10:
            l10 = slice(ng - 10, ng)
            with np.errstate(invalid="ignore", divide="ignore"):
                gr = np.where(ok_n[l10] > 0, S["success"][l10] / ok_n[l10], np.nan)
            gr = gr[np.isfinite(gr)]
            r["fm_suc_sd10"] = float(gr.std(ddof=1)) if len(gr) >= 2 else np.nan
            r["fm_suc_trend"] = pooled("success", slice(ng - 3, ng)) - pooled(
                "success", l10)
        rows[int(pid)] = r
    out = pd.DataFrame(rows).T.reindex(columns=AXES).sort_index()
    out.index.name = "pitcher_id"
    print(f"  [{tag}] 투수 {len(out)}명")
    return out


tb_strict = build(strict, "보수판")
tb_orig = pd.read_csv(f"{OUT}/fm_table_Y2024.csv", index_col="pitcher_id")

# ── 재스크린 (s2 와 같은 프로토콜) ────────────────────────────────────
bl = json.load(open(f"{PKG}/blend.json", encoding="utf-8"))
AXL, SW = bl["axis_offsets"], bl["smooth_weights"]
D59 = np.load(f"{H}/exp059_screen/directions.npz", allow_pickle=True)
pid_row = pd.read_csv(f"{FRAME}/test.csv", encoding="utf-8-sig",
                      usecols=["pitcher_id"])["pitcher_id"].to_numpy(np.int64)
y = D59["y"].astype(np.float64)


def unit(v):
    v = np.asarray(v, np.float64)
    fin = np.isfinite(v)
    u = np.zeros(len(v))
    u[fin] = (v[fin] - v[fin].mean()) / v[fin].std()
    return u


z = (float(bl["w_v6"]) * lg(np.load(f"{H}/build_BN/work/tr2024_p_v19_BN_wv6_070.npy"))
     + float(bl["c_aa"]) * lg(np.load(f"{H}/build_AY/work/pcal_aa.npy"))
     + float(SW["S2"]) * lg(np.load(f"{H}/build_AY/work/pcal_S2.npy"))
     + float(SW["S1"]) * lg(np.load(f"{H}/build_AY/work/pcal_S1.npy"))
     + float(SW["T70"]) * lg(np.load(f"{H}/build_AY/work/pcal_T70.npy")))
zp = z + float(AXL[0]["u"]) * unit(np.asarray(D59["beta_ins_off"], np.float64)) \
     + float(AXL[1]["u"]) * unit(np.asarray(D59["ctrl_asof_psr"], np.float64))
z_BN = float(bl["alpha"]) * zp + float(bl["beta_skew"]) * zp * zp + float(bl["d_adj"])
p_BN = expit(z_BN)
r0 = y - p_BN
blk = np.unique(pid_row, return_inverse=True)[1]
nb = int(blk.max()) + 1
IDX = np.random.default_rng(42).integers(0, nb, size=(1000, nb))


def screen(v):
    u = unit(v)
    fq = z_BN + EPS_PROBE * u
    anc = brentq(lambda d: expit(fq + d).mean() - float(p_BN.mean()), -5, 5,
                 xtol=1e-15)
    gvec = (expit(fq + anc) - p_BN) * r0
    M = float(gvec.mean())
    se = float(np.bincount(blk, weights=gvec, minlength=nb)[IDX]
               .sum(axis=1).std(ddof=1)) / N_ROWS
    return M, (M / se if se else 0.0)


def attach(tb, c):
    d = {int(k): float(v) for k, v in tb[c].items()}
    return np.array([d.get(int(p), np.nan) for p in pid_row], np.float64)


print(f"\n{'축':<16}{'t_원판':>9}{'t_보수판':>11}{'최대차(표)':>13}")
CMP = {}
for c in AXES:
    vo, vsx = attach(tb_orig, c), attach(tb_strict, c)
    Mo, to = screen(vo)
    Ms, tsx = screen(vsx)
    common = tb_orig.index.intersection(tb_strict.index)
    a = tb_orig.loc[common, c].to_numpy(np.float64)
    b = tb_strict.loc[common, c].to_numpy(np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    CMP[c] = {"t_original": round(to, 3), "t_strict": round(tsx, 3),
              "M_original": Mo, "M_strict": Ms,
              "max_abs_table_diff": float(np.abs(a[m] - b[m]).max()),
              "n_pitchers_compared": int(m.sum()),
              "both_below_threshold": bool(abs(to) < 2 and abs(tsx) < 2)}
    print(f"{c:<16}{to:>9.2f}{tsx:>11.2f}{CMP[c]['max_abs_table_diff']:>13.2e}")
REP["compare"] = CMP
REP["verdict"] = ("보수판에서도 6축 전부 |t| < 2 — 시즌 경계 차분이 판정을 "
                  "만들지 않았다" if all(v["both_below_threshold"]
                                        for v in CMP.values())
                  else "🔴 원판/보수판 판정이 갈린다 — 경계 차분이 결과에 영향")
print(f"\n{REP['verdict']}")
REP["wall_s_s5"] = round(time.time() - t0, 1)
json.dump(REP, open(f"{OUT}/strict.json", "w"), indent=1, ensure_ascii=False)
print(f"strict.json 저장 · s5 {time.time()-t0:.0f}s")
