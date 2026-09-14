"""exp_073 보조 진단 (등록 통계 아님 — 판정에 쓰지 않는다).

등록 스크린에서 통과 축이 0 이었다.  그러면 답이 갈리는 두 갈래를 갈라야 한다:

  (A) 신호가 데이터에 **없다** — 릴리스 산포는 이 데이터에서 제구와 관계가 약하다
  (B) 신호는 있는데 블렌드가 **이미 흡수**했다 — 오프셋 축으로만 닫힌다

가르는 재료
  D1 축별 부트스트랩 se — 검출력을 수치로 남긴다
  D2 원시 결합: 투수 수준 corr(축, 2024 관측 성공률) · 행 수준 corr(u, y)
     (모델을 거치지 않은 날것.  여기서도 0 이면 (A) 다)
  D3 부모가 이미 보는가: corr(u, logit(p_BN)) · corr(u, asof_pitcher_success_rate)
  D4 커버 행만의 부분프레임 재스크린 (무이동 21% 희석 제거)
  D5 등록 문턱을 못 넘었어도 tm_rel_sd 의 ctrl 잔차화 t 를 기록
"""
import json
import time

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit

t0 = time.time()
H = "/home/<user>/SungjinPark"
OUT = f"{H}/exp073_tm"
FRAME = f"{H}/build_AC/work/tr2024"
PKG = f"{H}/build_BN/pkg_BN"
EPS, EPS_PROBE, N_ROWS = 1e-6, 0.02, 253507
TM_AXES = ["tm_rel_sd", "tm_rel_sd_wp", "tm_speed_trend", "tm_speed_sd",
           "tm_break_sd", "tm_ext_mean", "tm_spin_mean"]
lg = lambda p: logit(np.clip(np.asarray(p, np.float64), EPS, 1 - EPS))

bl = json.load(open(f"{PKG}/blend.json", encoding="utf-8"))
A, B, D_BN = float(bl["alpha"]), float(bl["beta_skew"]), float(bl["d_adj"])
SW, AXL = bl["smooth_weights"], bl["axis_offsets"]
D59 = np.load(f"{H}/exp059_screen/directions.npz", allow_pickle=True)
tt = pd.read_csv(f"{FRAME}/test.csv", encoding="utf-8-sig",
                 usecols=["row_id", "pitcher_id", "asof_pitcher_success_rate"])
pid_row = tt["pitcher_id"].to_numpy(np.int64)
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

tbl = pd.read_csv(f"{OUT}/tm_table_Y2024.csv", index_col="pitcher_id")
V = {}
for c in TM_AXES:
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
for c in TM_AXES + ["ctrl"]:
    v = RAW_AX["ctrl_asof_psr"] if c == "ctrl" else V[c]
    M, se, t = screen(v, z_BN, p_BN, r0, blk, nb, IDX, N_ROWS)
    D1[c] = {"M": M, "se": se, "t": round(t, 3),
             "M_needed_for_t2": round(2 * se, 12)}
    print(f"  {c:<16} M {M:+.3e}  se {se:.3e}  t {t:+.2f}  "
          f"|M| 이 {2*se:.3e} 이상이어야 통과")
R["D1_power"] = D1

print("\n=== D2 원시 결합 (모델을 거치지 않은 날것) ===")
D2 = {}
g24 = pd.DataFrame({"pid": pid_row, "y": y}).groupby("pid")["y"].agg(
    ["mean", "size"])
for c in TM_AXES:
    d = {int(k): float(v) for k, v in tbl[c].items()}
    xv = np.array([d.get(int(p), np.nan) for p in g24.index], np.float64)
    m = np.isfinite(xv)
    rp = float(np.corrcoef(xv[m], g24["mean"].to_numpy()[m])[0, 1])
    w = g24["size"].to_numpy()[m].astype(np.float64)
    xw = xv[m] - np.average(xv[m], weights=w)
    yw = g24["mean"].to_numpy()[m] - np.average(g24["mean"].to_numpy()[m],
                                                weights=w)
    rw = float((w * xw * yw).sum()
               / np.sqrt((w * xw * xw).sum() * (w * yw * yw).sum()))
    u = unit(V[c])
    fin = np.isfinite(V[c])
    rrow = float(np.corrcoef(u[fin], y[fin])[0, 1])
    D2[c] = {"n_pitchers": int(m.sum()),
             "corr_pitcher_unweighted": round(rp, 4),
             "corr_pitcher_pitchweighted": round(rw, 4),
             "corr_row_with_y": round(rrow, 5)}
    print(f"  {c:<16} 투수 {int(m.sum()):>3}명  r(비가중) {rp:+.4f}  "
          f"r(투구수가중) {rw:+.4f}  행수준 r(u,y) {rrow:+.5f}")
R["D2_raw_association"] = D2

print("\n=== D3 부모가 이미 그 축을 보는가 ===")
psr = tt["asof_pitcher_success_rate"].to_numpy(np.float64)
D3 = {}
for c in TM_AXES:
    fin = np.isfinite(V[c])
    r_z = float(np.corrcoef(unit(V[c])[fin], z_BN[fin])[0, 1])
    m2 = fin & np.isfinite(psr)
    r_p = float(np.corrcoef(V[c][m2], psr[m2])[0, 1])
    D3[c] = {"corr_u_logit_pBN": round(r_z, 4), "corr_raw_asof_psr": round(r_p, 4)}
    print(f"  {c:<16} corr(u, logit p_BN) {r_z:+.4f}   "
          f"corr(raw, asof_psr) {r_p:+.4f}")
R["D3_parent_sees_it"] = D3

print("\n=== D4 커버 행만의 부분프레임 재스크린 (희석 제거) ===")
sub = np.isfinite(V["tm_rel_sd"])
ns = int(sub.sum())
blks = np.unique(pid_row[sub], return_inverse=True)[1]
nbs = int(blks.max()) + 1
IDXs = np.random.default_rng(42).integers(0, nbs, size=(1000, nbs))
D4 = {"n_rows": ns, "n_pitchers": nbs,
      "note": "부분프레임이라 M 의 절대값은 전체 프레임과 비교되지 않는다 — t 만 본다"}
for c in TM_AXES + ["ctrl"]:
    v = RAW_AX["ctrl_asof_psr"] if c == "ctrl" else V[c]
    M, se, t = screen(v[sub], z_BN[sub], p_BN[sub], r0[sub], blks, nbs, IDXs, ns)
    D4[c] = {"M": M, "se": se, "t": round(t, 3)}
    print(f"  {c:<16} M {M:+.3e}  se {se:.3e}  t {t:+.2f}")
R["D4_covered_subframe"] = D4

print("\n=== D5 등록 문턱 미달이지만 tm_rel_sd 의 ctrl 잔차화 t ===")
D5 = {}
ctrl = RAW_AX["ctrl_asof_psr"]
for c in ("tm_rel_sd", "tm_rel_sd_wp"):
    m = np.isfinite(V[c]) & np.isfinite(ctrl)
    X = np.column_stack([np.ones(m.sum()), ctrl[m]])
    beta, *_ = np.linalg.lstsq(X, V[c][m], rcond=None)
    res = np.full(N_ROWS, np.nan)
    res[m] = V[c][m] - X @ beta
    M, se, t = screen(res, z_BN, p_BN, r0, blk, nb, IDX, N_ROWS)
    rc = float(np.corrcoef(V[c][m], ctrl[m])[0, 1])
    D5[c] = {"corr_with_ctrl": round(rc, 4), "t_resid_ctrl": round(t, 3),
             "M_resid_ctrl": M}
    print(f"  {c:<16} corr(ctrl) {rc:+.4f}  잔차화 후 t {t:+.2f}")
R["D5_residualized_anyway"] = D5

R["wall_s_s3"] = round(time.time() - t0, 1)
json.dump(R, open(f"{OUT}/supp.json", "w"), indent=1, ensure_ascii=False)
print(f"\nsupp.json 저장 · s3 {time.time()-t0:.0f}s")
