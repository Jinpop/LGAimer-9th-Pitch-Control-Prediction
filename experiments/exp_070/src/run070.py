"""exp_070 — 정직면 전면 재적합. 구조 실험의 본체. 학습 0건, 캐시 벡터 위 산술.

구조 A (현행 중첩, 자유도 6):
  z_p = (1-w_ebm)*( w_v6*lg(p_v6_3arm) + (1-w_v6)*lg(p_q) ) + w_ebm*lg(p_ebm) + beta*off
  z   = w_out*z_p + c_aa*lg(AA81) + c_t70*lg(T70)
구조 B (탈중첩 평면, 자유도 8):
  z = w_lgb*L_lgb + w_cat*L_cat + w_leaf*L_leaf + w_q*L_q + w_ebm*L_ebm
      + w_aa*L_aa + w_t70*L_t70 + beta*off
구조 C (보조): 구조 A 에서 c_aa = c_t70 = 0 (자유도 4) — 우리 팔의 정직한 한계 기여.

🔴 눈금: 전부 재센터링 후 (각 블렌드 로짓에 상수 c 를 더해 가중평균 확률 == 가중 y.mean).
   점수 환산은 프로젝트 표준 REF=0.2498069. 팀원 README 는 로컬 2024 기후 참조
   ref_local=mean((ybar-y)^2)=0.24980692675678806 을 썼다 — 두 눈금 차 +0.0106점(상수).
   시작 자격 933.88(REF)/933.89(ref_local) 둘 다 재현해야 통과.
🔴 부트스트랩: 투수 블록(391 블록) 1000회, 각 구조 argmax 와 Δ의 퍼센타일 CI.
🔴 판정하지 않는다 — 산출은 곡면 + 상금 크기 + CI.
"""
import os

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "6"

import hashlib
import json
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit

H = "/home/<user>/SungjinPark"
OOT = f"{H}/exp069/oot"            # 읽기 전용 (팀원 납품 1·2차 사본)
OOT3 = f"{H}/exp070/oot3"          # 3차 납품 p_ebm (이번에 업로드)
OUT = f"{H}/exp070/out"
os.makedirs(OUT, exist_ok=True)

EPS = 1e-6
REF = 0.2498069                    # 프로젝트 표준 환산 참조 (CLAUDE.md)
K = 100000.0 / REF
N_EXPECT = 253507
R_BOOT = 1000
SEED_BOOT = 70
T0 = time.time()
WALL = {}

# 배포 상수 (config §구조_A 대조군)
DEP = np.array([0.5, -0.10, 0.03, 0.7, 0.25244, 0.04756])
DEP_NAMES = ["w_v6", "w_ebm", "beta_ins", "w_out", "c_aa", "c_t70"]
B_NAMES = ["w_lgb", "w_cat", "w_leaf", "w_q", "w_ebm", "w_aa", "w_t70", "beta_ins"]

# 로컬 scratchpad 에서 잰 파일 sha256 (양쪽 사본 대조용 — 어긋나면 중단)
SHA_EXPECT = {
    "off_oot2024.npy": "a2647b5251efd54a5beb22a5208628f22f67207608bfe39cdd036ba577fcd575",
    "p_cat_oot2024.npy": "5119b91576bdc289bc647e34370a1133676753ee8279c87797062220f801ae88",
    "p_leaf_oot2024.npy": "071b2a42726ed702c742345f6b4ae8080a2d650fb3c45ede47263ad9a41e9f26",
    "p_lgb_oot2024.npy": "e692666f4ce4f5a6f8259d30c7ad29bd087e9168ccad20ae6f03c39a1f080906",
    "p_q_oot2024.npy": "2aaed0c1d15f74d88455d346c9851afa185319a2e7c40d555d5138f720d24fdc",
    "p_v6_3arm_renorm_oot2024.npy":
        "6b047a7665033463c15d2cdb2d47eb29ab591b66d6fd1c772c9d6a71f1f0b1e4",
    "row_id_oot2024.npy": "587983eddf9b797190f0b91fa38ee4add32ecde0d0491fe803decf0a3175936f",
    "p_ebm_oot2024.npy": "477150c2421a5b0d9bbb09f87434e4a97cee7ce2cc241ebcd331b41da37b144d",
}
Y_SHA_EXPECT = "e0483e6a48d9f699ca9c4abf7b4b81aed4d679c359c7e9c3752b3f9e6e27c9d3"


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


M = {"exp_id": "exp_070",
     "type": "정직면 전면 재적합 — 학습 0건, 캐시 벡터 위 산술",
     "env": {"python": sys.version.split()[0], "numpy": np.__version__,
             "scipy": scipy.__version__, "threads_env": "6 (5종 전부)"},
     "score_conversion": {
         "primary_ref": REF,
         "formula": "100000 * (1 - Brier / 0.2498069)",
         "ref_local_climatology_2024": None,  # 아래서 채움
         "s3": "6자리 Brier 환산 ±0.2002 불확실 (S3)"},
     }

# ═══════════════════════════════════════════════════════════════════
# 0. 라벨·정렬
# ═══════════════════════════════════════════════════════════════════
t = time.time()
tr = pd.read_csv(f"{H}/data/train.csv",
                 usecols=["row_id", "season", "control_success", "pitcher_id"])
tr = tr[tr.season == 2024].sort_values("row_id", kind="stable").reset_index(drop=True)
N = len(tr)
assert N == N_EXPECT, N
y = tr.control_success.to_numpy(np.float64)
rid = tr.row_id.to_numpy().astype(str)
pit = tr.pitcher_id.to_numpy()
assert sha_arr(y) == Y_SHA_EXPECT, "라벨 벡터가 exp_065/066/069 와 다르다"
ybar = float(y.mean())
ref_local = float(np.mean((ybar - y) ** 2))
M["score_conversion"]["ref_local_climatology_2024"] = ref_local
M["score_conversion"]["scale_gap_pts_at_brier_0.2477"] = float(
    100000.0 * 0.2477 * (1.0 / REF - 1.0 / ref_local) * -1.0)

oot_rid = np.asarray(np.load(f"{OOT}/row_id_oot2024.npy", allow_pickle=True)).astype(str)
assert np.array_equal(oot_rid, rid), "납품 row_id 순서 불일치"

# 투수 블록
pit_codes, pit_uniques = pd.factorize(pit, sort=True)
G = len(pit_uniques)
M["alignment"] = {"n_rows": N, "y_mean": ybar, "y_sha256": sha_arr(y),
                  "oot_row_id_order_ok": True, "n_pitcher_blocks_2024": int(G)}
WALL["s0_align_min"] = (time.time() - t) / 60.0
print(f"[정렬] N={N} ybar={ybar:.10f} ref_local={ref_local:.17f} 투수블록 {G}", flush=True)

# ═══════════════════════════════════════════════════════════════════
# 1. 재료 로드 + sha 대조
# ═══════════════════════════════════════════════════════════════════
t = time.time()
FILES = {"p_lgb": f"{OOT}/p_lgb_oot2024.npy", "p_cat": f"{OOT}/p_cat_oot2024.npy",
         "p_leaf": f"{OOT}/p_leaf_oot2024.npy", "p_q": f"{OOT}/p_q_oot2024.npy",
         "p_v6": f"{OOT}/p_v6_3arm_renorm_oot2024.npy",
         "off": f"{OOT}/off_oot2024.npy", "p_ebm": f"{OOT3}/p_ebm_oot2024.npy"}
sha_used = {}
for k, p in FILES.items():
    s = sha_file(p)
    sha_used[os.path.basename(p)] = s
    assert s == SHA_EXPECT[os.path.basename(p)], f"sha 불일치 {p}"
sha_used["row_id_oot2024.npy"] = sha_file(f"{OOT}/row_id_oot2024.npy")
V = {k: np.load(p).astype(np.float64) for k, p in FILES.items()}
for k in ["p_lgb", "p_cat", "p_leaf", "p_q", "p_v6", "p_ebm"]:
    a = V[k]
    assert a.shape == (N,) and 0.0 < a.min() and a.max() < 1.0, k
# 🔴 S14 변형 — oot3/p_ebm 은 확률이다 (배포 arms/p_ebm 로짓과 다름). 값 범위로 확인.
assert 0.25 < V["p_ebm"].min() < 0.35 and 0.60 < V["p_ebm"].max() < 0.75, \
    f"p_ebm 범위 이상 {V['p_ebm'].min()} {V['p_ebm'].max()} — 확률이 아닐 수 있다"
assert V["off"].shape == (N,) and V["off"].min() < -0.5 and V["off"].max() < 3.0

AA_SEEDS = (42, 7, 2024)
aa_files = {s: f"{H}/experiments/exp_049/pred/pred_A1_u2024_seed{s}_at1400.npy"
            for s in AA_SEEDS}
t70_files = {s: f"{H}/experiments/exp_055/pred/pred_T70_u2024_seed{s}_at1400.npy"
             for s in AA_SEEDS}
for d in (aa_files, t70_files):
    for s, p in d.items():
        sha_used[os.path.basename(p)] = sha_file(p)
aa_seed = {s: np.load(p).astype(np.float64) for s, p in aa_files.items()}
t70_seed = {s: np.load(p).astype(np.float64) for s, p in t70_files.items()}
for d in (aa_seed, t70_seed):
    for s, a in d.items():
        assert a.shape == (N,) and 0.0 < a.min() and a.max() < 1.0

# config 사양: 로짓평균
L_aa = np.mean([lg(aa_seed[s]) for s in AA_SEEDS], axis=0)
L_t70 = np.mean([lg(t70_seed[s]) for s in AA_SEEDS], axis=0)
# 각주: exp_069 는 AA81 을 확률평균으로 썼다 — 차이 기록
aa_probmean = np.mean([aa_seed[s] for s in AA_SEEDS], axis=0)
M["aa81_probmean_vs_logitmean"] = {
    "max_abs_prob_diff": float(np.max(np.abs(expit(L_aa) - aa_probmean))),
    "note": "config 사양은 로짓평균. exp_069 의 확률평균(원자료 838.44309)과 다른 결합."}

L_v6, L_q, L_ebm = lg(V["p_v6"]), lg(V["p_q"]), lg(V["p_ebm"])
L_lgb, L_cat, L_leaf = lg(V["p_lgb"]), lg(V["p_cat"]), lg(V["p_leaf"])
off = V["off"]
M["sha256_files_used"] = sha_used
M["sha256_note"] = ("서버 사본으로 실행. 로컬 scratchpad 사본과 8개 납품 벡터 전부 "
                    "sha256 동일함을 실행 전 확인 (SHA_EXPECT 하드코딩 대조).")
WALL["s1_load_min"] = (time.time() - t) / 60.0

# ═══════════════════════════════════════════════════════════════════
# 2. 재센터링 + 점수
# ═══════════════════════════════════════════════════════════════════


def recenter_c(z, u=None, c0=0.0):
    """z 에 상수 c 를 더해 (가중) 평균 확률 == (가중) y 평균. Newton."""
    if u is None:
        yb = ybar
        for _ in range(40):
            p = expit(z + c0)
            m = float(p.mean())
            if abs(m - yb) < 1e-14:
                break
            d = (m - yb) / float((p * (1.0 - p)).mean())
            c0 -= d
            if abs(d) < 1e-14:
                break
    else:
        U = float(u.sum())
        yb = float((u * y).sum() / U)
        for _ in range(40):
            p = expit(z + c0)
            m = float((u * p).sum() / U)
            if abs(m - yb) < 1e-14:
                break
            d = (m - yb) / float((u * p * (1.0 - p)).sum() / U)
            c0 -= d
            if abs(d) < 1e-14:
                break
    return c0


def score_z(z, u=None):
    c = recenter_c(z, u)
    p = expit(z + c)
    if u is None:
        b = float(np.mean((p - y) ** 2))
    else:
        b = float((u * (p - y) ** 2).sum() / u.sum())
    return b, conv(b), c


# ═══════════════════════════════════════════════════════════════════
# 3. 시작 자격 — 배포 3축 조합 933.88(REF) / 933.89(ref_local)
# ═══════════════════════════════════════════════════════════════════
t = time.time()
z_inner_dep = (1.0 - DEP[1]) * (DEP[0] * L_v6 + (1.0 - DEP[0]) * L_q) \
    + DEP[1] * L_ebm + DEP[2] * off
b_g, s_g, c_g = score_z(z_inner_dep)
s_g_local = 100000.0 * (1.0 - b_g / ref_local)
gate_ok = (round(s_g, 2) == 933.88) and (round(s_g_local, 2) == 933.89)
M["start_qualification"] = {
    "rule": "배포 3축 (0.5, -0.10, 0.03) 재센터링 점수가 933.88~933.89 로 재현",
    "REF_scale": s_g, "ref_local_scale": s_g_local, "brier": b_g,
    "recenter_c": c_g,
    "해설": ("933.88 = 지휘자 실측(REF=0.2498069 눈금), 933.89 = 팀원 실측"
             "(ref_local 눈금). 두 값의 차이는 참조 상수 차이 하나로 전부 설명된다."),
    "PASS": bool(gate_ok)}
print(f"[시작자격] REF {s_g:.5f} / ref_local {s_g_local:.5f}  PASS={gate_ok}", flush=True)
assert gate_ok, "🔴 시작 자격 실패 — 눈금이 틀렸다. 중단."

# 단일 팔 재센터링 표 (README 2dp 대조는 ref_local 눈금)
README_REF = {"p_lgb": 892.21, "p_cat": 922.01, "p_leaf": 734.34,
              "p_v6": 926.48, "p_q": 894.64, "p_ebm": 688.00}
arms_tbl = {}
for k, Lx in [("p_lgb", L_lgb), ("p_cat", L_cat), ("p_leaf", L_leaf),
              ("p_q", L_q), ("p_v6", L_v6), ("p_ebm", L_ebm),
              ("AA81", L_aa), ("T70", L_t70)]:
    b_, s_, c_ = score_z(Lx)
    row = {"brier": b_, "score_REF": s_, "score_ref_local": 100000 * (1 - b_ / ref_local),
           "recenter_c": c_}
    if k in README_REF:
        row["README_2dp"] = README_REF[k]
        row["match_2dp"] = bool(round(row["score_ref_local"], 2) == README_REF[k])
        assert row["match_2dp"], f"{k} README 대조 실패"
    arms_tbl[k] = row
M["single_arms_recentered"] = arms_tbl
WALL["s3_gate_min"] = (time.time() - t) / 60.0
print("[단일팔] " + "  ".join(f"{k} {v['score_REF']:.2f}" for k, v in arms_tbl.items()),
      flush=True)

# ═══════════════════════════════════════════════════════════════════
# 4. 구조 정의 (설계행렬 + 파라미터 사상 + 야코비안)
# ═══════════════════════════════════════════════════════════════════
XA = np.column_stack([L_v6, L_q, L_ebm, off, L_aa, L_t70])          # A·C 공용
XB = np.column_stack([L_lgb, L_cat, L_leaf, L_q, L_ebm, L_aa, L_t70, off])


def vmap_A(th):
    wv, we, be, wo, ca, ct = th
    return np.array([wo * (1 - we) * wv, wo * (1 - we) * (1 - wv),
                     wo * we, wo * be, ca, ct])


def jac_A(th):
    wv, we, be, wo, ca, ct = th
    J = np.zeros((6, 6))
    J[0] = [wo * (1 - we), -wo * wv, 0, (1 - we) * wv, 0, 0]
    J[1] = [-wo * (1 - we), -wo * (1 - wv), 0, (1 - we) * (1 - wv), 0, 0]
    J[2] = [0, wo, 0, we, 0, 0]
    J[3] = [0, 0, wo, be, 0, 0]
    J[4] = [0, 0, 0, 0, 1, 0]
    J[5] = [0, 0, 0, 0, 0, 1]
    return J


def vmap_C(th):
    return vmap_A(np.concatenate([th, [0.0, 0.0]]))


def jac_C(th):
    return jac_A(np.concatenate([th, [0.0, 0.0]]))[:, :4]


STRUCT = {
    "A": {"X": XA, "vmap": vmap_A, "jac": jac_A, "dim": 6, "names": DEP_NAMES},
    "B": {"X": XB, "vmap": lambda th: np.asarray(th, float),
          "jac": lambda th: np.eye(8), "dim": 8, "names": B_NAMES},
    "C": {"X": XA, "vmap": vmap_C, "jac": jac_C, "dim": 4, "names": DEP_NAMES[:4]},
}


def obj_grad(st, th, u=None, c_warm=None):
    """재센터링 포함 Brier 와 θ-그래디언트 (암함수 정리로 dc/dθ 반영)."""
    X, vm, jc = st["X"], st["vmap"], st["jac"]
    v = vm(th)
    z = X @ v
    c = recenter_c(z, u, 0.0 if c_warm is None else c_warm)
    p = expit(z + c)
    r = p - y
    pq = p * (1.0 - p)
    if u is None:
        U = float(N)
        a = pq
        bv = pq * r
        B_ = float(np.mean(r * r))
    else:
        U = float(u.sum())
        a = u * pq
        bv = a * r
        B_ = float((u * r * r).sum() / U)
    sa = X.T @ a
    sb = X.T @ bv
    Sa = float(a.sum())
    Sb = float(bv.sum())
    dB_dv = (2.0 / U) * (sb - (sa / Sa) * Sb)
    g = jc(th).T @ dB_dv
    return B_, g, c


def fit_lbfgs(st, th0, u=None):
    cw = [0.0]

    def f(th):
        B_, g, c = obj_grad(st, th, u, cw[0])
        cw[0] = c
        return B_, g

    r = minimize(f, np.asarray(th0, float), jac=True, method="L-BFGS-B",
                 options={"gtol": 1e-13, "ftol": 1e-16, "maxiter": 800})
    return np.asarray(r.x), float(r.fun)


def fit_nm(st, th0, maxfev=20000):
    X, vm = st["X"], st["vmap"]
    cw = [0.0]

    def f(th):
        z = X @ vm(th)
        c = recenter_c(z, None, cw[0])
        cw[0] = c
        p = expit(z + c)
        return float(np.mean((p - y) ** 2))

    r = minimize(f, np.asarray(th0, float), method="Nelder-Mead",
                 options={"fatol": 1e-13, "xatol": 1e-7,
                          "maxiter": maxfev, "maxfev": maxfev, "adaptive": True})
    return np.asarray(r.x), float(r.fun)


# 그래디언트 검증 (유한차분 대조)
t = time.time()
rng0 = np.random.default_rng(7)
gradchk = {}
for nm, th_t in [("A", DEP * 1.1 + 0.01), ("B", np.full(8, 0.12)),
                 ("C", DEP[:4] * 0.9)]:
    st = STRUCT[nm]
    B0, g0, _ = obj_grad(st, th_t)
    h = 1e-6
    gfd = np.zeros_like(g0)
    for j in range(len(th_t)):
        e = np.zeros_like(th_t)
        e[j] = h
        bp, _, _ = obj_grad(st, th_t + e)
        bm, _, _ = obj_grad(st, th_t - e)
        gfd[j] = (bp - bm) / (2 * h)
    rel = float(np.max(np.abs(g0 - gfd) / (np.abs(gfd) + 1e-12)))
    gradchk[nm] = rel
    assert rel < 1e-4, (nm, rel, g0, gfd)
M["gradient_check_max_rel_err"] = gradchk
WALL["s4_gradcheck_min"] = (time.time() - t) / 60.0
print(f"[grad검증] {gradchk}", flush=True)

# ═══════════════════════════════════════════════════════════════════
# 5. 전 데이터 적합 — 다중 시작 (NM + L-BFGS 교차 확인)
# ═══════════════════════════════════════════════════════════════════
t = time.time()
# 배포 상수의 평면 등가 (renorm: L_v6 = (0.05 L_lgb + 0.68 L_cat + 0.12 L_leaf)/0.85)
q_dep = DEP[3] * (1 - DEP[1]) * DEP[0]
flat_dep = np.array([q_dep * 0.05 / 0.85, q_dep * 0.68 / 0.85, q_dep * 0.12 / 0.85,
                     DEP[3] * (1 - DEP[1]) * (1 - DEP[0]), DEP[3] * DEP[1],
                     DEP[4], DEP[5], DEP[3] * DEP[2]])

STARTS = {
    "A": [DEP.copy(),
          np.array([0.70, -0.10, 0.05, 0.7, 0.25244, 0.04756]),
          np.array([0.50, 0.00, 0.00, 1.0, 0.0, 0.0]),
          np.array([0.60, -0.15, 0.04, 0.8, 0.15, 0.10]),
          np.array([0.50, -0.10, 0.03, 1.0, 0.30, 0.10])],
    "B": [flat_dep.copy(),
          np.full(8, 0.14),
          np.array([0.0, 0.40, 0.0, 0.30, -0.10, 0.25, 0.05, 0.02]),
          np.array([0.1, 0.5, 0.1, 0.3, -0.05, 0.2, 0.1, 0.05])],
    "C": [DEP[:4].copy(),
          np.array([0.70, -0.10, 0.05, 0.8]),
          np.array([0.50, 0.00, 0.00, 1.0]),
          np.array([0.65, -0.12, 0.04, 0.9])],
}
# 평면 등가 검증: B(flat_dep) ≈ A(DEP)
b_dep_A, s_dep_A, c_dep_A = score_z(XA @ vmap_A(DEP))
b_dep_B, s_dep_B, _ = score_z(XB @ flat_dep)
assert abs(s_dep_B - s_dep_A) < 0.01, (s_dep_A, s_dep_B)
M["deployment_full_blend"] = {
    "theta": dict(zip(DEP_NAMES, DEP.tolist())),
    "brier": b_dep_A, "score_REF": s_dep_A, "recenter_c": c_dep_A,
    "flat_equiv_score_REF": s_dep_B,
    "flat_equiv_gap_pts": s_dep_B - s_dep_A,
    "note": "6상수 배포 조합(재센터링) — 모든 Δ의 대조군"}
print(f"[배포조합] 6상수 전체 {s_dep_A:.5f} (평면등가 {s_dep_B:.5f})", flush=True)

FIT = {}
for nm in ("A", "B", "C"):
    st = STRUCT[nm]
    best = None
    detail = []
    for i, th0 in enumerate(STARTS[nm]):
        thL, bL = fit_lbfgs(st, th0)
        detail.append({"start": np.asarray(th0).tolist(),
                       "lbfgs_brier": bL, "theta": thL.tolist()})
        if best is None or bL < best[1]:
            best = (thL, bL)
    # NM 마무리 (최적 시작만) + 독립 NM (배포 시작점) — 최적화기 합의 확인
    thN, bN = fit_nm(st, best[0])
    if bN < best[1]:
        best = (thN, bN)
    thNM, bNM = fit_nm(st, STARTS[nm][0])
    agree = abs(bNM - best[1])
    th_s, b_s = best
    b_chk, s_chk, c_chk = score_z(st["X"] @ st["vmap"](th_s))
    assert abs(b_chk - b_s) < 1e-14
    FIT[nm] = {"theta": dict(zip(st["names"], th_s.tolist())),
               "theta_vec": th_s.tolist(),
               "brier": b_s, "score_REF": s_chk, "recenter_c": c_chk,
               "starts_detail": detail,
               "optimizer_agreement_brier_gap_vs_pure_NM": agree,
               "pure_NM_from_dep_start_brier": bNM}
    print(f"[적합 {nm}] {s_chk:.5f}  θ={np.round(th_s,5).tolist()}  "
          f"NM합의 {agree:.3e}", flush=True)

sA, sB, sC = FIT["A"]["score_REF"], FIT["B"]["score_REF"], FIT["C"]["score_REF"]
assert FIT["A"]["brier"] <= b_dep_A + 1e-15, "A_opt 가 배포보다 나쁨 — 최적화 실패"
assert FIT["B"]["brier"] <= FIT["A"]["brier"] + 1e-12, "B ⊇ A 위반 — 최적화 실패"
assert FIT["C"]["brier"] >= FIT["A"]["brier"] - 1e-12, "C ⊂ A 위반"

M["fits_full_data"] = FIT
M["deltas_full_data_pts"] = {
    "Δ_A재적합_vs_배포": sA - s_dep_A,
    "구조상금_B_vs_A": sB - sA,
    "우리팔_한계기여_A_vs_C": sA - sC,
    "B_vs_배포": sB - s_dep_A,
    "C_vs_배포": sC - s_dep_A}
WALL["s5_fits_min"] = (time.time() - t) / 60.0
json.dump(M, open(f"{OUT}/metrics_partial.json", "w"), indent=1, ensure_ascii=False)

# ═══════════════════════════════════════════════════════════════════
# 6. 구조 A 1축 조건부 곡선 (나머지 배포 고정 / 나머지 argmax 고정 두 벌)
# ═══════════════════════════════════════════════════════════════════
t = time.time()
GRIDS = {
    "w_v6": np.round(np.arange(0.0, 1.0001, 0.05), 4),
    "w_ebm": np.round(np.arange(-0.30, 0.1501, 0.025), 4),
    "beta_ins": np.round(np.arange(0.0, 0.1001, 0.01), 4),
    "w_out": np.round(np.arange(0.40, 1.2001, 0.05), 4),
    "c_aa": np.round(np.arange(0.0, 0.6001, 0.04), 4),
    "c_t70": np.round(np.arange(-0.10, 0.3001, 0.025), 4),
}
thA = np.asarray(FIT["A"]["theta_vec"])


def axis_curve(center, j, grid):
    rows = {}
    for gv in grid:
        th = center.copy()
        th[j] = gv
        b_, s_, _ = score_z(XA @ vmap_A(th))
        rows[f"{gv:.4f}"] = {"score_REF": s_, "vs_dep_pts": s_ - s_dep_A}
    # 1차원 정밀 argmax
    lo, hi = float(grid[0]), float(grid[-1])

    def f1(x):
        th = center.copy()
        th[j] = x
        return score_z(XA @ vmap_A(th))[0]

    r = minimize_scalar(f1, bounds=(lo, hi), method="bounded",
                        options={"xatol": 1e-6})
    return {"curve": rows, "cond_argmax": float(r.x),
            "cond_argmax_score_REF": conv(float(r.fun)),
            "cap_touched": bool(abs(r.x - lo) < 1e-4 or abs(r.x - hi) < 1e-4)}


CURVES = {"center_deployment": {}, "center_A_argmax": {}}
for j, nm in enumerate(DEP_NAMES):
    CURVES["center_deployment"][nm] = axis_curve(DEP.copy(), j, GRIDS[nm])
    CURVES["center_A_argmax"][nm] = axis_curve(thA.copy(), j, GRIDS[nm])
M["curves_structure_A"] = CURVES
WALL["s6_curves_min"] = (time.time() - t) / 60.0
print("[곡선] 조건부 argmax(배포 중심): " +
      " ".join(f"{nm}={CURVES['center_deployment'][nm]['cond_argmax']:.4f}"
               for nm in DEP_NAMES), flush=True)
json.dump(M, open(f"{OUT}/metrics_partial.json", "w"), indent=1, ensure_ascii=False)

# ═══════════════════════════════════════════════════════════════════
# 7. 투수 블록 부트스트랩 1000회 (병렬 4)
# ═══════════════════════════════════════════════════════════════════
t = time.time()
thA_full = np.asarray(FIT["A"]["theta_vec"])
thB_full = np.asarray(FIT["B"]["theta_vec"])
thC_full = np.asarray(FIT["C"]["theta_vec"])


def one_rep(rep):
    rng = np.random.default_rng([SEED_BOOT, rep])
    cnt = np.bincount(rng.integers(0, G, G), minlength=G).astype(np.float64)
    u = cnt[pit_codes]
    if u.sum() <= 0:
        return None
    b_dep_r, _, _ = score_z(XA @ vmap_A(DEP), u)
    thA_r, bA_r = fit_lbfgs(STRUCT["A"], thA_full, u)
    thB_r, bB_r = fit_lbfgs(STRUCT["B"], thB_full, u)
    thC_r, bC_r = fit_lbfgs(STRUCT["C"], thC_full, u)
    extra = 0
    if rep % 37 == 0:      # 국소최적 진단 — 두 번째 시작점
        thA2, bA2 = fit_lbfgs(STRUCT["A"], DEP, u)
        if bA2 < bA_r - 1e-13:
            thA_r, bA_r = thA2, bA2
            extra = 1
    return (thA_r.tolist() + thB_r.tolist() + thC_r.tolist()
            + [K * (b_dep_r - bA_r), K * (bA_r - bB_r), K * (bC_r - bA_r),
               float(extra)])


NPROC = 4
with Pool(NPROC) as pool:
    rows = pool.map(one_rep, range(R_BOOT), chunksize=25)
rows = np.array([r for r in rows if r is not None])
assert rows.shape[0] == R_BOOT
A_bs = rows[:, 0:6]
B_bs = rows[:, 6:14]
C_bs = rows[:, 14:18]
dA_bs = rows[:, 18]        # A재적합 − 배포 (점)
prize_bs = rows[:, 19]     # B − A (점)
ourarm_bs = rows[:, 20]    # A − C (점)
n_alt_better = int(rows[:, 21].sum())


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


BOOT = {"n_reps": R_BOOT, "block": "pitcher_id (2024, 391블록) 다항 리샘플",
        "seed": SEED_BOOT, "engine": "L-BFGS(해석 그래디언트) 전 데이터 argmax 워밍",
        "local_opt_diag": {
            "reps_with_second_start": int(np.ceil(R_BOOT / 37)),
            "second_start_beat_warm": n_alt_better},
        "argmax_CI_A": {}, "argmax_CI_B": {}, "argmax_CI_C": {},
        "delta_CI_pts": {
            "Δ_A재적합_vs_배포": {"point": M["deltas_full_data_pts"]["Δ_A재적합_vs_배포"],
                                  "ci95": ci(dA_bs), "boot_mean": float(dA_bs.mean())},
            "구조상금_B_vs_A": {"point": M["deltas_full_data_pts"]["구조상금_B_vs_A"],
                                "ci95": ci(prize_bs), "boot_mean": float(prize_bs.mean())},
            "우리팔_한계기여_A_vs_C": {
                "point": M["deltas_full_data_pts"]["우리팔_한계기여_A_vs_C"],
                "ci95": ci(ourarm_bs), "boot_mean": float(ourarm_bs.mean())}}}
for j, nm in enumerate(DEP_NAMES):
    c_ = ci(A_bs[:, j])
    BOOT["argmax_CI_A"][nm] = {
        "point": float(thA_full[j]), "ci95": c_,
        "boot_mean": float(A_bs[:, j].mean()), "deployment_value": float(DEP[j]),
        "deployment_inside_CI": bool(c_[0] <= DEP[j] <= c_[1]),
        "판독": ("이동 근거 없음 (배포값이 CI 안)" if c_[0] <= DEP[j] <= c_[1]
                 else "배포값이 CI 밖")}
for j, nm in enumerate(B_NAMES):
    BOOT["argmax_CI_B"][nm] = {"point": float(thB_full[j]), "ci95": ci(B_bs[:, j]),
                               "boot_mean": float(B_bs[:, j].mean())}
for j, nm in enumerate(DEP_NAMES[:4]):
    BOOT["argmax_CI_C"][nm] = {"point": float(thC_full[j]), "ci95": ci(C_bs[:, j]),
                               "boot_mean": float(C_bs[:, j].mean())}
M["bootstrap"] = BOOT
WALL["s7_bootstrap_min"] = (time.time() - t) / 60.0
print(f"[부트스트랩] {R_BOOT}회 {WALL['s7_bootstrap_min']:.2f}분  "
      f"ΔA CI {BOOT['delta_CI_pts']['Δ_A재적합_vs_배포']['ci95']}  "
      f"상금 CI {BOOT['delta_CI_pts']['구조상금_B_vs_A']['ci95']}", flush=True)

# ═══════════════════════════════════════════════════════════════════
# 8. 표준 블록 + 마무리
# ═══════════════════════════════════════════════════════════════════
M["fold_2024"] = {
    "대상": "배포 6상수 전체 블렌드 (재센터링, 대조군)",
    "brier": b_dep_A, "bss": s_dep_A,
    "A_opt_score": sA, "B_opt_score": sB, "C_opt_score": sC, "n": N}
M["fold_2022"] = None
M["fold_2023"] = None
M["train_time_min"] = 0.0
M["inference_time_sec"] = None
M["편향_명시"] = {
    "p_v6": ("NN 없는 3종판 — v6 계열(안쪽 w_v6, 평면 lgb/cat/leaf) 가중이 "
             "아래로 눌린 하한"),
    "p_q": "1시드 (배포 3시드) — Q 가중이 아래로 눌린다",
    "방향": "두 편향이 서로 반대 팔을 민다 — 안쪽 w_v6 argmax 는 양쪽으로 당겨진다"}
M["판정_금지"] = "채택·출하 구조를 결정하지 않는다. 산출은 곡면·상금 크기·CI 다 (config §사전 등록)."
WALL["total_min"] = (time.time() - T0) / 60.0
M["wall_clock_min"] = WALL
json.dump(M, open(f"{OUT}/metrics_core.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({"dep": s_dep_A, "A_opt": sA, "B_opt": sB, "C_opt": sC,
                  "dA": sA - s_dep_A, "prize": sB - sA, "ourarm": sA - sC,
                  "total_min": WALL["total_min"]}, ensure_ascii=False), flush=True)
print("DONE_RUN070", flush=True)
