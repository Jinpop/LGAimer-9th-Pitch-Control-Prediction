"""exp_072 stage3 — c_t70 은 단독 축이 아니다.  (c_aa, c_t70) 2축 · (w_out, c_aa, c_t70) 3축
동시 적합의 정직면 Δ 와 보류 블록 Δ.  exp_070 이 남긴 '2축으로 다뤄야 한다'를 실제로 잰다.
학습 0건.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "6"
import hashlib, json, time
from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

H = "/home/<user>/SungjinPark"
OOT, OOT3, OOT4 = f"{H}/exp069/oot", f"{H}/exp070/oot3", f"{H}/exp071/out"
OUT = f"{H}/exp072"
EPS, REF = 1e-6, 0.2498069
K = 100000.0 / REF
R_BOOT, SEED_BOOT = 400, 72
T0 = time.time()
DEP = np.array([0.5, -0.10, 0.03, 0.7, 0.25244, 0.04756])
NAMES = ["w_v6", "w_ebm", "beta_ins", "w_out", "c_aa", "c_t70"]
SETS = {"c_aa+c_t70": [4, 5], "w_out+c_aa+c_t70": [3, 4, 5],
        "w_v6+c_aa+c_t70": [0, 4, 5], "all6": [0, 1, 2, 3, 4, 5]}
BND = {0: (0.0, 1.0), 1: (-0.5, 0.5), 2: (-0.2, 0.3), 3: (0.3, 1.5),
       4: (-0.5, 1.0), 5: (-1.0, 0.6)}


def lg(p): return logit(np.clip(np.asarray(p, np.float64), EPS, 1.0 - EPS))
def conv(b): return 100000.0 * (1.0 - b / REF)


tr = pd.read_csv(f"{H}/data/train.csv",
                 usecols=["row_id", "season", "control_success", "pitcher_id"])
tr = tr[tr.season == 2024].sort_values("row_id", kind="stable").reset_index(drop=True)
y = tr.control_success.to_numpy(np.float64)
assert hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest() == \
    "e0483e6a48d9f699ca9c4abf7b4b81aed4d679c359c7e9c3752b3f9e6e27c9d3"
pit_codes, pu = pd.factorize(tr.pitcher_id.to_numpy(), sort=True)
G = len(pu)
V = {k: np.load(p).astype(np.float64) for k, p in {
    "p_q": f"{OOT}/p_q_oot2024.npy",
    "p_v6_3": f"{OOT}/p_v6_3arm_renorm_oot2024.npy",
    "p_ebm": f"{OOT3}/p_ebm_oot2024.npy",
    "p_v6_4": f"{OOT4}/p_v6_4arm_oot2024.npy",
    "off": f"{OOT}/off_oot2024.npy"}.items()}
AA = (42, 7, 2024)
L_aa = np.mean([lg(np.load(f"{H}/experiments/exp_049/pred/pred_A1_u2024_seed{s}_at1400.npy")) for s in AA], axis=0)
L_t70 = np.mean([lg(np.load(f"{H}/experiments/exp_055/pred/pred_T70_u2024_seed{s}_at1400.npy")) for s in AA], axis=0)
SURF = {t: np.column_stack([lg(V[k]), lg(V["p_q"]), lg(V["p_ebm"]), V["off"], L_aa, L_t70])
        for t, k in (("3arm", "p_v6_3"), ("4arm", "p_v6_4"))}


def vmap(th):
    wv, we, be, wo, ca, ct = th
    return np.array([wo*(1-we)*wv, wo*(1-we)*(1-wv), wo*we, wo*be, ca, ct])


def rc(z, m=None, c0=0.0):
    zz, yy = (z, y) if m is None else (z[m], y[m])
    yb = float(yy.mean())
    for _ in range(60):
        p = expit(zz + c0); mu = float(p.mean())
        if abs(mu - yb) < 1e-14: break
        d = (mu - yb) / float((p*(1-p)).mean()); c0 -= d
        if abs(d) < 1e-14: break
    return c0


def brier(X, th, m=None):
    z = X @ vmap(th); c = rc(z, m)
    if m is None: return float(np.mean((expit(z+c)-y)**2))
    return float(np.mean((expit(z[m]+c)-y[m])**2))


def fit_set(X, idx, m=None):
    th = DEP.copy()

    def f(x):
        t2 = th.copy(); t2[idx] = x
        return brier(X, t2, m)
    r = minimize(f, DEP[idx].copy(), method="L-BFGS-B",
                 bounds=[BND[i] for i in idx],
                 options={"maxiter": 400, "ftol": 1e-16, "gtol": 1e-12})
    return np.asarray(r.x, float)


rngf = np.random.default_rng(707)
fold = (rngf.permutation(G) % 5)[pit_codes]
MASKS = [(fold != k, fold == k) for k in range(5)]


def cv_fixed(X, th):
    z = X @ vmap(th); num, den = 0.0, 0
    for mtr, mte in MASKS:
        c = rc(z, mtr)
        num += float(np.sum((expit(z[mte]+c)-y[mte])**2)); den += int(mte.sum())
    return conv(num/den)


def cv_fitted(X, idx):
    num, den, picks = 0.0, 0, []
    for mtr, mte in MASKS:
        xv = fit_set(X, idx, mtr); picks.append([float(v) for v in xv])
        th = DEP.copy(); th[idx] = xv
        z = X @ vmap(th); c = rc(z, mtr)
        num += float(np.sum((expit(z[mte]+c)-y[mte])**2)); den += int(mte.sum())
    return conv(num/den), picks


M = {"exp_id": "exp_072", "stage": "3_multi_axis_c_t70"}
S_DEP = {t: conv(brier(SURF[t], DEP)) for t in SURF}
CVDEP = {t: cv_fixed(SURF[t], DEP) for t in SURF}
assert abs(S_DEP["3arm"] - 928.1835495351443) < 1e-8
M["ref"] = {"insample_deployed": S_DEP, "cv_deployed": CVDEP}
TAB = {}
for tag in ("3arm", "4arm"):
    X = SURF[tag]; TAB[tag] = {}
    for sname, idx in SETS.items():
        sol = fit_set(X, idx)
        th = DEP.copy(); th[idx] = sol
        cvs, picks = cv_fitted(X, idx)
        TAB[tag][sname] = {
            "axes": [NAMES[i] for i in idx],
            "deployed": [float(DEP[i]) for i in idx],
            "solution": [float(v) for v in sol],
            "insample_delta_pts": conv(brier(X, th)) - S_DEP[tag],
            "cv_holdout_delta_fixed_pts": cv_fixed(X, th) - CVDEP[tag],
            "cv_holdout_delta_fitted_procedure_pts": cvs - CVDEP[tag],
            "cv_fold_picks": picks}
        print(f"  [{tag}] {sname:<18} sol {np.round(sol,5).tolist()}  insΔ "
              f"{TAB[tag][sname]['insample_delta_pts']:+.4f}  보류Δ(고정) "
              f"{TAB[tag][sname]['cv_holdout_delta_fixed_pts']:+.4f}  보류Δ(적합절차) "
              f"{TAB[tag][sname]['cv_holdout_delta_fitted_procedure_pts']:+.4f}", flush=True)
M["multi_axis"] = TAB

# 부트스트랩 — 고정 해의 Δ CI (재적합 없음)
JOBS = [(tag, s) for tag in ("3arm", "4arm") for s in SETS]


def _sc_u(X, th, u, U, yb):
    z = X @ vmap(th); c = 0.0
    for _ in range(60):
        p = expit(z+c); mu = float((u*p).sum()/U)
        if abs(mu-yb) < 1e-14: break
        d = (mu-yb)/float((u*p*(1-p)).sum()/U); c -= d
        if abs(d) < 1e-14: break
    p = expit(z+c)
    return float((u*(p-y)**2).sum()/U)


def brep(rep):
    r = np.random.default_rng([SEED_BOOT, rep])
    cnt = np.bincount(r.integers(0, G, G), minlength=G).astype(np.float64)
    u = cnt[pit_codes]; U = float(u.sum()); yb = float((u*y).sum()/U)
    base = {t: _sc_u(SURF[t], DEP, u, U, yb) for t in SURF}
    o = []
    for tag, s in JOBS:
        th = DEP.copy(); th[SETS[s]] = np.asarray(TAB[tag][s]["solution"])
        o.append(K*(base[tag] - _sc_u(SURF[tag], th, u, U, yb)))
    return o


with Pool(5) as pool:
    B = np.array(pool.map(brep, range(R_BOOT), chunksize=8))
BT = {}
for j, (tag, s) in enumerate(JOBS):
    BT.setdefault(tag, {})[s] = {
        "boot_mean_pts": float(B[:, j].mean()),
        "ci95_pts": [float(np.percentile(B[:, j], 2.5)),
                     float(np.percentile(B[:, j], 97.5))],
        "frac_positive": float((B[:, j] > 0).mean())}
M["bootstrap_fixed_solution"] = BT
M["bootstrap_meta"] = {"R": R_BOOT, "seed": SEED_BOOT,
                       "note": "해를 전 데이터에서 고정하고 리샘플에서 평가 — 낙관 방향"}
M["wall_clock_min"] = (time.time()-T0)/60.0
json.dump(M, open(f"{OUT}/pair072.json", "w"), indent=1, ensure_ascii=False)
print(f"\n기록 {OUT}/pair072.json · {M['wall_clock_min']:.2f}분", flush=True)
