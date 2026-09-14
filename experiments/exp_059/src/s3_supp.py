"""exp_059 s3 — 보조 진단: 방향 간 상관 + ctrl_asof_psr 성분 제거 후 재스크린.

면 A 통과 방향 다섯이 서로 독립 신호인지, '커리어 성공률 과대반영' 하나의
공유 성분인지 가른다. 프로토콜(중심화·앵커·부트스트랩)은 s2 와 동일.
metrics.json 에 수치 블록만 추가한다.
"""
import json
import time

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logit

t0 = time.time()
H = "/home/<user>/SungjinPark"
OUT = f"{H}/exp059_screen"
EPS = 1e-6
K = 100000.0 / 0.2498069
EPS_PROBE = 0.02
lg = lambda p: logit(np.clip(np.asarray(p, np.float64), EPS, 1 - EPS))

D = np.load(f"{OUT}/directions.npz")
y = D["y"]
N = len(y)
blk = np.unique(D["pitcher_id"], return_inverse=True)[1]
nb = int(blk.max()) + 1
IDX = np.random.default_rng(42).integers(0, nb, size=(1000, nb))

C = {"aa": 0.25244, "S2": 0.0, "S1": 0.0, "T70": 0.04756}
z = (0.7 * lg(np.load(f"{H}/build_AR/work/tr2024_p_v19_AR.npy"))
     + C["aa"] * lg(np.load(f"{H}/build_AY/work/pcal_aa.npy"))
     + C["S2"] * lg(np.load(f"{H}/build_AY/work/pcal_S2.npy"))
     + C["S1"] * lg(np.load(f"{H}/build_AY/work/pcal_S1.npy"))
     + C["T70"] * lg(np.load(f"{H}/build_AY/work/pcal_T70.npy")))
bl = json.load(open(f"{H}/build_BG/pkg_BG/blend.json", encoding="utf-8"))
z2 = z * z
zA = bl["alpha"] * z + bl["beta_skew"] * z2 + bl["d_adj"]
pA = expit(zA)
r0A = y - pA


def norm_u(v):
    v = np.asarray(v, np.float64)
    fin = np.isfinite(v)
    u = np.zeros(N)
    u[fin] = (v[fin] - v[fin].mean()) / v[fin].std()
    return u


def screen_u(u):
    f = zA + EPS_PROBE * u
    tgt = float(pA.mean())
    anc = brentq(lambda d: expit(f + d).mean() - tgt, -5, 5, xtol=1e-15)
    delta = expit(f + anc) - pA
    g = delta * r0A
    M = float(g.mean())
    bs = np.bincount(blk, weights=g, minlength=nb)
    se = float(bs[IDX].sum(axis=1).std(ddof=1)) / N
    return M, (M / se if se else 0.0)


NAMES = ["sp_pit_bhand", "sp_bat_phand", "sp_pit_strk2", "sp_pit_ball3",
         "sp_pit_bhand_all", "sp_bat_phand_all", "sd_arms", "d_form_eb",
         "mode_p_is_mid_only", "mode_p_is_rev_only", "mode_p_is_both",
         "mode_p_is_wayoff_true", "beta_ins_off", "ctrl_asof_psr"]
U = {nm: norm_u(D[nm]) for nm in NAMES}

# 방향 간 상관 (u 벡터, 중심화·정규화 후)
corr = {}
for i, a in enumerate(NAMES):
    for b in NAMES[i + 1:]:
        c = float(U[a] @ U[b] / N)
        if abs(c) >= 0.10:
            corr[f"{a}~{b}"] = round(c, 3)

# ctrl_asof_psr 성분 제거 후 재스크린
uc = U["ctrl_asof_psr"]
uc2 = float(uc @ uc)
resid = {}
for nm in NAMES:
    if nm == "ctrl_asof_psr":
        continue
    u = U[nm]
    ur = u - (float(u @ uc) / uc2) * uc
    sd = ur.std()
    if sd < 1e-12:
        continue
    ur = (ur - ur.mean()) / sd
    M, t = screen_u(ur)
    resid[nm] = {"M": M, "t": round(t, 3), "sign": "+" if M > 0 else "-",
                 "corr_with_ctrl": round(float(u @ uc) / N, 3)}

print("|corr| >= 0.10 인 쌍:")
for k, v in sorted(corr.items(), key=lambda x: -abs(x[1])):
    print(f"  {k:<44} {v:+.3f}")
print("\nctrl_asof_psr 성분 제거 후 (면 A):")
print(f"{'방향':<26}{'corr(ctrl)':>11}{'t_res':>8}")
for nm, r in resid.items():
    flag = " ◀" if abs(r["t"]) >= 2 else ""
    print(f"{nm:<26}{r['corr_with_ctrl']:>+11.3f}{r['t']:>8.2f}{flag}")

m = json.load(open(f"{OUT}/metrics.json"))
m["supp_corr_u_abs_ge_010"] = corr
m["supp_face_A_residualized_vs_ctrl_asof_psr"] = resid
m["wall_s"]["s3"] = round(time.time() - t0, 1)
json.dump(m, open(f"{OUT}/metrics.json", "w"), indent=1, ensure_ascii=False)
print(f"\nmetrics.json 갱신 · s3 {time.time()-t0:.0f}s")
