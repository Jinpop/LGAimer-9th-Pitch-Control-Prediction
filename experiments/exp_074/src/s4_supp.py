"""exp_074 보조 진단 (🔴 등록 통계 아님 — 판정에 쓰지 않는다).

등록 스크린에서 통과 축이 0/6 이었다.  그런데 **perm_shuffle 이 t +1.96 으로
나왔다** — ≈0 이 아니다.  그러면 두 가지를 갈라야 한다:

  (A) 축들이 정말 죽었다
  (B) 이 프레임에서 |t| >= 2 가 애초에 변별을 못 한다 (스크린 자체의 문제)

그리고 신호가 없는 것인지 부모가 이미 흡수한 것인지도 갈라야 한다.

D1 축별 부트스트랩 se — 검출력을 수치로
D2 🔴 경험적 귀무분포 — **결측 패턴을 보존한** 투수 라벨 순열 200회.
   s2 의 perm_shuffle 은 NaN 까지 섞어 무이동 행이 107,500 (실축은 55,407) 이
   됐다 — 결측 구조가 다른 대조군이었다.  여기서 바로잡는다.
D3 커버 행만의 부분프레임 재스크린 (무이동 22% 희석 제거).
   🔴 배포 커버리지는 98.76% 인데 스크린 프레임은 78.14% 다 — 스크린이
   배포보다 불리한 조건에서 재고 있다.
D4 모델을 안 거친 날것: 투수 수준 corr(축, 2024 관측 성공률) · 행 수준 corr(u, y)
D5 부모가 이미 보는가: corr(u, logit(p_BN)) · corr(u, ctrl_prev3_suc)
"""
import json
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
AXES = ["fm_rev_g5", "fm_ball_g5", "fm_strike_g5", "fm_suc_sd5",
        "fm_suc_sd10", "fm_suc_trend"]
SUPP_AXES = ["supp_fm_suc_g5", "supp_fm_mid_g5"]
lg = lambda p: logit(np.clip(np.asarray(p, np.float64), EPS, 1 - EPS))

bl = json.load(open(f"{PKG}/blend.json", encoding="utf-8"))
A, B, D_BN = float(bl["alpha"]), float(bl["beta_skew"]), float(bl["d_adj"])
SW, AXL = bl["smooth_weights"], bl["axis_offsets"]
D59 = np.load(f"{H}/exp059_screen/directions.npz", allow_pickle=True)
fr = pd.read_csv(f"{FRAME}/test.csv", encoding="utf-8-sig",
                 usecols=["row_id", "pitcher_id", "asof_pitcher_success_rate",
                          "asof_pitcher_prev3_game_success_rate"])
pid_row = fr["pitcher_id"].to_numpy(np.int64)
y = D59["y"].astype(np.float64)


def unit(v):
    v = np.asarray(v, np.float64)
    fin = np.isfinite(v)
    mu, sd = float(v[fin].mean()), float(v[fin].std())
    u = np.zeros(len(v))
    u[fin] = (v[fin] - mu) / sd
    return u


RAW_AX = {"beta_ins_off": np.asarray(D59["beta_ins_off"], np.float64),
          "ctrl_asof_psr": np.asarray(D59["ctrl_asof_psr"], np.float64)}
z = (float(bl["w_v6"]) * lg(np.load(f"{H}/build_BN/work/"
                                    "tr2024_p_v19_BN_wv6_070.npy"))
     + float(bl["c_aa"]) * lg(np.load(f"{H}/build_AY/work/pcal_aa.npy"))
     + float(SW["S2"]) * lg(np.load(f"{H}/build_AY/work/pcal_S2.npy"))
     + float(SW["S1"]) * lg(np.load(f"{H}/build_AY/work/pcal_S1.npy"))
     + float(SW["T70"]) * lg(np.load(f"{H}/build_AY/work/pcal_T70.npy")))
zp = z + float(AXL[0]["u"]) * unit(RAW_AX["beta_ins_off"]) \
     + float(AXL[1]["u"]) * unit(RAW_AX["ctrl_asof_psr"])
z_BN = A * zp + B * zp * zp + D_BN
p_BN = expit(z_BN)
r0 = y - p_BN

tbl = pd.read_csv(f"{OUT}/fm_table_Y2024.csv", index_col="pitcher_id")
V = {}
for c in AXES + SUPP_AXES:
    d = {int(k): float(v) for k, v in tbl[c].items()}
    V[c] = np.array([d.get(int(p), np.nan) for p in pid_row], np.float64)

R = {"note": "🔴 보조 진단 — 사전 등록 통계가 아니다.  판정에 쓰지 않는다."}


def screen(v, zf, p0, rr, blk, nb, IDX, n):
    u = unit(v)
    fq = zf + EPS_PROBE * u
    anc = brentq(lambda d: expit(fq + d).mean() - float(p0.mean()), -5, 5,
                 xtol=1e-15)
    delta = expit(fq + anc) - p0
    g = delta * rr
    M = float(g.mean())
    bs = np.bincount(blk, weights=g, minlength=nb)
    se = float(bs[IDX].sum(axis=1).std(ddof=1)) / n
    return M, se, (M / se if se else 0.0)


blk = np.unique(pid_row, return_inverse=True)[1]
nb = int(blk.max()) + 1
IDX = np.random.default_rng(42).integers(0, nb, size=(1000, nb))

print("=== D1 축별 se (검출력) ===")
D1 = {}
for c in AXES + SUPP_AXES + ["ctrl_prev3", "ctrl_psr"]:
    v = {"ctrl_prev3": fr["asof_pitcher_prev3_game_success_rate"]
         .to_numpy(np.float64), "ctrl_psr": RAW_AX["ctrl_asof_psr"]}.get(c, V.get(c))
    M, se, t = screen(v, z_BN, p_BN, r0, blk, nb, IDX, N_ROWS)
    D1[c] = {"M": M, "se": se, "t": round(t, 3), "M_needed_for_t2": round(2 * se, 12)}
    print(f"  {c:<16} M {M:+.3e}  se {se:.3e}  t {t:+.2f}  "
          f"|t|=2 에 필요한 M {2*se:.3e}")
R["D1_power"] = D1

# ── D2 🔴 결측 패턴 보존 순열 200회 (경험적 귀무분포) ─────────────────
print("\n=== D2 경험적 귀무분포 — 결측 보존 순열 200회 ===")
D2 = {}
rng = np.random.default_rng(2024)
pid_tab = tbl.index.to_numpy(np.int64)
for c in ["fm_rev_g5", "fm_suc_trend"]:
    vals = tbl[c].to_numpy(np.float64)
    fin = np.isfinite(vals)
    ts = []
    for _ in range(200):
        pv = vals.copy()
        pv[fin] = rng.permutation(vals[fin])          # 유한값끼리만 섞는다
        d = {int(k): float(x) for k, x in zip(pid_tab, pv)}
        vv = np.array([d.get(int(p), np.nan) for p in pid_row], np.float64)
        ts.append(screen(vv, z_BN, p_BN, r0, blk, nb, IDX, N_ROWS)[2])
    ts = np.array(ts)
    D2[c] = {"n_perm": 200, "t_mean": float(ts.mean()), "t_sd": float(ts.std(ddof=1)),
             "abs_t_q50": float(np.quantile(np.abs(ts), 0.50)),
             "abs_t_q95": float(np.quantile(np.abs(ts), 0.95)),
             "frac_abs_t_ge_2": float((np.abs(ts) >= 2).mean()),
             "t_min": float(ts.min()), "t_max": float(ts.max()),
             "observed_t": None}
    print(f"  {c:<14} 귀무 t 평균 {ts.mean():+.3f} sd {ts.std(ddof=1):.3f}  "
          f"|t| 95분위 {np.quantile(np.abs(ts),0.95):.2f}  "
          f"P(|t|>=2) {(np.abs(ts)>=2).mean():.3f}")
R["D2_empirical_null_missing_preserved"] = D2

# ── D3 커버 행만 부분프레임 ──────────────────────────────────────────
print("\n=== D3 커버 행만 재스크린 (무이동 희석 제거) ===")
D3 = {}
for c in AXES + SUPP_AXES:
    m = np.isfinite(V[c])
    zb, pb, rb = z_BN[m], p_BN[m], r0[m]
    b2 = np.unique(pid_row[m], return_inverse=True)[1]
    nb2 = int(b2.max()) + 1
    I2 = np.random.default_rng(42).integers(0, nb2, size=(1000, nb2))
    M, se, t = screen(V[c][m], zb, pb, rb, b2, nb2, I2, int(m.sum()))
    D3[c] = {"n_rows": int(m.sum()), "n_blocks": nb2, "M": M, "se": se,
             "t": round(t, 3)}
    print(f"  {c:<16} 행 {int(m.sum()):,} 투수 {nb2}  t {t:+.2f}")
R["D3_covered_only"] = D3

# ── D4 모델 밖 날것 ──────────────────────────────────────────────────
print("\n=== D4 모델을 안 거친 결합 ===")
obs = pd.DataFrame({"pid": pid_row, "y": y}).groupby("pid")["y"].mean()
D4 = {}
for c in AXES + SUPP_AXES:
    d = tbl[c]
    common = obs.index.intersection(d.index)
    a = d.loc[common].to_numpy(np.float64)
    b = obs.loc[common].to_numpy(np.float64)
    mm = np.isfinite(a) & np.isfinite(b)
    u = unit(V[c])
    fin = np.isfinite(V[c])
    D4[c] = {"n_pitchers": int(mm.sum()),
             "corr_pitcher_level_vs_obs_success":
                 round(float(np.corrcoef(a[mm], b[mm])[0, 1]), 4),
             "corr_row_level_u_vs_y":
                 round(float(np.corrcoef(u[fin], y[fin])[0, 1]), 4)}
    print(f"  {c:<16} 투수수준 r {D4[c]['corr_pitcher_level_vs_obs_success']:+.4f}"
          f"   행수준 corr(u,y) {D4[c]['corr_row_level_u_vs_y']:+.4f}")
R["D4_model_free"] = D4

# ── D5 부모가 이미 보는가 ────────────────────────────────────────────
print("\n=== D5 부모/대조군과의 결합 ===")
prev3 = fr["asof_pitcher_prev3_game_success_rate"].to_numpy(np.float64)
D5 = {}
for c in AXES + SUPP_AXES:
    fin = np.isfinite(V[c])
    u = unit(V[c])
    m3 = fin & np.isfinite(prev3)
    D5[c] = {"corr_u_vs_logit_pBN":
                 round(float(np.corrcoef(u[fin], z_BN[fin])[0, 1]), 4),
             "corr_raw_vs_ctrl_prev3":
                 round(float(np.corrcoef(V[c][m3], prev3[m3])[0, 1]), 4)}
    print(f"  {c:<16} corr(u, logit p_BN) "
          f"{D5[c]['corr_u_vs_logit_pBN']:+.4f}   corr(축, prev3) "
          f"{D5[c]['corr_raw_vs_ctrl_prev3']:+.4f}")
R["D5_parent_coupling"] = D5

R["wall_s_s4"] = round(time.time() - t0, 1)
json.dump(R, open(f"{OUT}/supp.json", "w"), indent=1, ensure_ascii=False)
print(f"\nsupp.json 저장 · s4 {time.time()-t0:.0f}s")
