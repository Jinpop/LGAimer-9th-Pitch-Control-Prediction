"""exp_072 stage2 — 남은 축의 값 진단.  학습 0건, 캐시 벡터 위 산술.

정직면(팀원 OOT 납품 ≤2023학습→2024예측) 위에서 각 축을 **단독으로** 움직였을 때의
① 표본내 Δ  ② 투수 블록 5겹 그룹 CV 보류 Δ  ③ 투수 블록 부트스트랩 CI 를 낸다.
곡면 둘: 3종판 p_v6 (exp_070 대조군) · 4팔 완성판 p_v6 (exp_071).
🔴 판정하지 않는다.  산출은 표다.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "6"

import hashlib
import json
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

H = "/home/<user>/SungjinPark"
OOT = f"{H}/exp069/oot"
OOT3 = f"{H}/exp070/oot3"
OOT4 = f"{H}/exp071/out"
OUT = f"{H}/exp072"
os.makedirs(OUT, exist_ok=True)

EPS = 1e-6
REF = 0.2498069
K = 100000.0 / REF
N_EXPECT = 253507
R_BOOT = 400
SEED_BOOT = 72
T0 = time.time()

DEP = np.array([0.5, -0.10, 0.03, 0.7, 0.25244, 0.04756])
NAMES = ["w_v6", "w_ebm", "beta_ins", "w_out", "c_aa", "c_t70"]
BOUNDS = {"w_v6": (0.0, 1.0), "w_ebm": (-0.5, 0.5), "beta_ins": (-0.2, 0.3),
          "w_out": (0.3, 1.5), "c_aa": (-0.2, 0.8), "c_t70": (-1.0, 0.5)}
# 사전 등록 후보값 (exp_070 산출물에서 그대로 가져온다 — 결과 보고 안 바꾼다)
CAND = {
    "w_v6": {"cond_argmax_070": 0.8581420253062062, "joint_A_070": 0.7382,
             "shipped_BN": 0.70},
    "w_ebm": {"cond_argmax_070": -0.1444, "joint_A_070": -0.0544, "zero": 0.0},
    "beta_ins": {"cond_argmax_070": 0.0715, "joint_A_070": 0.0261, "zero": 0.0},
    "w_out": {"cond_argmax_070": 0.7593, "joint_A_070": 1.1033},
    "c_aa": {"cond_argmax_070": 0.291, "joint_A_070": 0.3187},
    "c_t70": {"cond_argmax_070": 0.0750, "joint_A_070": -0.4149,
              "structB_070": -0.42021, "zero": 0.0},
}
Y_SHA = "e0483e6a48d9f699ca9c4abf7b4b81aed4d679c359c7e9c3752b3f9e6e27c9d3"


def lg(p):
    return logit(np.clip(np.asarray(p, np.float64), EPS, 1.0 - EPS))


def conv(b):
    return 100000.0 * (1.0 - b / REF)


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a, np.float64).tobytes()).hexdigest()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


M = {"exp_id": "exp_072", "stage": "2_axis_value_diagnosis"}

tr = pd.read_csv(f"{H}/data/train.csv",
                 usecols=["row_id", "season", "control_success", "pitcher_id"])
tr = tr[tr.season == 2024].sort_values("row_id", kind="stable").reset_index(drop=True)
N = len(tr)
assert N == N_EXPECT
y = tr.control_success.to_numpy(np.float64)
rid = tr.row_id.to_numpy().astype(str)
assert sha_arr(y) == Y_SHA, "라벨 벡터 불일치"
oot_rid = np.asarray(np.load(f"{OOT}/row_id_oot2024.npy", allow_pickle=True)).astype(str)
assert np.array_equal(oot_rid, rid)
ours_rid = np.asarray(np.load(f"{OOT4}/row_id_oot2024_ours.npy",
                              allow_pickle=True)).astype(str)
assert np.array_equal(ours_rid, rid)
pit_codes, pit_uniques = pd.factorize(tr.pitcher_id.to_numpy(), sort=True)
G = len(pit_uniques)
M["alignment"] = {"n": N, "y_sha256": sha_arr(y), "n_pitcher_blocks": int(G),
                  "row_id_order_ok": True}

V = {k: np.load(p).astype(np.float64) for k, p in {
    "p_q": f"{OOT}/p_q_oot2024.npy",
    "p_v6_3": f"{OOT}/p_v6_3arm_renorm_oot2024.npy",
    "p_ebm": f"{OOT3}/p_ebm_oot2024.npy",
    "p_v6_4": f"{OOT4}/p_v6_4arm_oot2024.npy",
    "off": f"{OOT}/off_oot2024.npy"}.items()}
AA_SEEDS = (42, 7, 2024)
L_aa = np.mean([lg(np.load(f"{H}/experiments/exp_049/pred/"
                           f"pred_A1_u2024_seed{s}_at1400.npy").astype(np.float64))
                for s in AA_SEEDS], axis=0)
L_t70 = np.mean([lg(np.load(f"{H}/experiments/exp_055/pred/"
                            f"pred_T70_u2024_seed{s}_at1400.npy").astype(np.float64))
                 for s in AA_SEEDS], axis=0)
M["sha256_v6_4arm"] = sha_file(f"{OOT4}/p_v6_4arm_oot2024.npy")

SURF = {}
for tag, key in (("3arm", "p_v6_3"), ("4arm", "p_v6_4")):
    SURF[tag] = np.column_stack([lg(V[key]), lg(V["p_q"]), lg(V["p_ebm"]),
                                 V["off"], L_aa, L_t70])


def vmap(th):
    wv, we, be, wo, ca, ct = th
    return np.array([wo * (1 - we) * wv, wo * (1 - we) * (1 - wv),
                     wo * we, wo * be, ca, ct])


def recenter_c(z, m=None, c0=0.0):
    if m is None:
        zz, yy = z, y
    else:
        zz, yy = z[m], y[m]
    yb = float(yy.mean())
    for _ in range(60):
        p = expit(zz + c0)
        mu = float(p.mean())
        if abs(mu - yb) < 1e-14:
            break
        d = (mu - yb) / float((p * (1 - p)).mean())
        c0 -= d
        if abs(d) < 1e-14:
            break
    return c0


def score_theta(X, th):
    z = X @ vmap(th)
    c = recenter_c(z)
    return conv(float(np.mean((expit(z + c) - y) ** 2)))


# ── 시작 자격 ────────────────────────────────────────────────────────
S_DEP = {t: score_theta(SURF[t], DEP) for t in SURF}
M["start_qualification"] = {
    "3arm_deployed": S_DEP["3arm"], "3arm_expected_exp070": 928.1835495351443,
    "3arm_abs_diff": abs(S_DEP["3arm"] - 928.1835495351443),
    "4arm_deployed": S_DEP["4arm"], "4arm_expected_exp071": 925.4743659412812,
    "4arm_abs_diff": abs(S_DEP["4arm"] - 925.4743659412812)}
assert abs(S_DEP["3arm"] - 928.1835495351443) < 1e-8, "3종판 대조군 재현 실패"
assert abs(S_DEP["4arm"] - 925.4743659412812) < 1e-8, "4팔 대조군 재현 실패"
print(f"[시작자격] 🟢 3arm {S_DEP['3arm']:.6f}  4arm {S_DEP['4arm']:.6f}", flush=True)

# ── 폴드 (exp_070 followup 과 같은 난수·같은 배정) ────────────────────
rngf = np.random.default_rng(707)
fold = rngf.permutation(G)[pit_codes] % 5
fold_ = (rngf := None)
rngf = np.random.default_rng(707)
fold = (rngf.permutation(G) % 5)[pit_codes]
MASKS = [(fold != k, fold == k) for k in range(5)]


def cv_score_fixed(X, th):
    """상수를 그대로 출하했을 때의 보류 점수 (적합 0, c 만 훈련 블록 추정)."""
    z = X @ vmap(th)
    num, den = 0.0, 0
    for mtr, mte in MASKS:
        c = recenter_c(z, mtr)
        num += float(np.sum((expit(z[mte] + c) - y[mte]) ** 2))
        den += int(mte.sum())
    return conv(num / den)


def fit_axis(X, i, m=None, th0=None):
    th = (DEP if th0 is None else th0).copy()

    def f(x):
        t2 = th.copy(); t2[i] = x
        z = X @ vmap(t2)
        c = recenter_c(z, m)
        if m is None:
            return float(np.mean((expit(z + c) - y) ** 2))
        return float(np.mean((expit(z[m] + c) - y[m]) ** 2))
    r = minimize_scalar(f, bounds=BOUNDS[NAMES[i]], method="bounded",
                        options={"xatol": 1e-7})
    return float(r.x)


def cv_score_fitted(X, i):
    """축 하나를 폴드마다 적합해 보류 블록에서 평가 — '고르는 절차'의 정직한 값."""
    num, den = 0.0, 0
    picks = []
    for mtr, mte in MASKS:
        xv = fit_axis(X, i, mtr)
        picks.append(xv)
        th = DEP.copy(); th[i] = xv
        z = X @ vmap(th)
        c = recenter_c(z, mtr)
        num += float(np.sum((expit(z[mte] + c) - y[mte]) ** 2))
        den += int(mte.sum())
    return conv(num / den), picks


CVDEP = {t: cv_score_fixed(SURF[t], DEP) for t in SURF}
M["cv_deployed"] = CVDEP
print(f"[CV 배포] 3arm {CVDEP['3arm']:.6f}  4arm {CVDEP['4arm']:.6f}", flush=True)

# ── 축별 표 ──────────────────────────────────────────────────────────
TAB = {}
for tag in ("3arm", "4arm"):
    X = SURF[tag]
    TAB[tag] = {}
    for i, nm in enumerate(NAMES):
        am = fit_axis(X, i)
        th_am = DEP.copy(); th_am[i] = am
        row = {"deployed": float(DEP[i]),
               "cond_argmax": am,
               "insample_delta_at_cond_argmax_pts": score_theta(X, th_am) - S_DEP[tag],
               "cv_holdout_delta_fitted_procedure_pts": None,
               "cv_fold_picks": None,
               "candidates": {}}
        cvs, picks = cv_score_fitted(X, i)
        row["cv_holdout_delta_fitted_procedure_pts"] = cvs - CVDEP[tag]
        row["cv_fold_picks"] = picks
        for cname, cval in CAND[nm].items():
            th = DEP.copy(); th[i] = cval
            row["candidates"][cname] = {
                "value": float(cval),
                "insample_delta_pts": score_theta(X, th) - S_DEP[tag],
                "cv_holdout_delta_pts": cv_score_fixed(X, th) - CVDEP[tag]}
        TAB[tag][nm] = row
        print(f"  [{tag}] {nm:<9} argmax {am:+.5f}  표본내Δ "
              f"{row['insample_delta_at_cond_argmax_pts']:+.4f}  보류Δ(적합절차) "
              f"{row['cv_holdout_delta_fitted_procedure_pts']:+.4f}", flush=True)
M["axis_table"] = TAB
M["wall_after_table_min"] = (time.time() - T0) / 60.0

# ── 부트스트랩 (투수 블록) — 고정값 Δ 의 CI ─────────────────────────
BJOBS = []
for tag in ("3arm", "4arm"):
    for i, nm in enumerate(NAMES):
        for cname, cval in CAND[nm].items():
            BJOBS.append((tag, nm, i, cname, float(cval)))


def _sc_u(X, th, u, U, yb):
    z = X @ vmap(th)
    c = 0.0
    for _ in range(60):
        p = expit(z + c)
        mu = float((u * p).sum() / U)
        if abs(mu - yb) < 1e-14:
            break
        d = (mu - yb) / float((u * p * (1 - p)).sum() / U)
        c -= d
        if abs(d) < 1e-14:
            break
    p = expit(z + c)
    return float((u * (p - y) ** 2).sum() / U)


def boot_rep(rep):
    r = np.random.default_rng([SEED_BOOT, rep])
    cnt = np.bincount(r.integers(0, G, G), minlength=G).astype(np.float64)
    u = cnt[pit_codes]
    U = float(u.sum()); yb = float((u * y).sum() / U)
    out = []
    base = {t: _sc_u(SURF[t], DEP, u, U, yb) for t in SURF}
    for tag, nm, i, cname, cval in BJOBS:
        th = DEP.copy(); th[i] = cval
        out.append(K * (base[tag] - _sc_u(SURF[tag], th, u, U, yb)))
    return out


t = time.time()
with Pool(5) as pool:
    B = np.array(pool.map(boot_rep, range(R_BOOT), chunksize=8))
BOOT = {}
for j, (tag, nm, i, cname, cval) in enumerate(BJOBS):
    BOOT.setdefault(tag, {}).setdefault(nm, {})[cname] = {
        "value": cval,
        "boot_mean_pts": float(B[:, j].mean()),
        "ci95_pts": [float(np.percentile(B[:, j], 2.5)),
                     float(np.percentile(B[:, j], 97.5))],
        "frac_positive": float((B[:, j] > 0).mean())}
M["bootstrap_fixed_value_delta"] = BOOT
M["bootstrap_meta"] = {"R": R_BOOT, "seed": SEED_BOOT,
                       "unit": "pitcher block (391)",
                       "note": ("고정값 Δ 의 CI — 리샘플마다 재적합하지 않는다. "
                                "exp_070 의 재적합형 CI 와 다른 양이다.")}
M["wall_boot_min"] = (time.time() - t) / 60.0
M["wall_clock_min"] = (time.time() - T0) / 60.0
json.dump(M, open(f"{OUT}/axes072.json", "w"), indent=1, ensure_ascii=False)
print(f"\n기록 {OUT}/axes072.json  ·  벽시계 {M['wall_clock_min']:.2f}분", flush=True)
