"""exp_070 후속 — 지휘자 지시 3건 (2026-08-21).

① 과적합 서명 정면 대응
   - c_t70 부호 반전(+0.04756 → −0.41486)·w_out > 1 을 부트스트랩 CI 와 수축 진단으로 판정
   - 🔴 수축 진단은 **투수 블록 그룹 CV** 로 한다. exp_065 면 B 는 표본내 면이라
     수축이 표본내 점수를 개선할 수 있었지만, 여기 A/B/C 적합은 평가와 같은 행 위의
     표본내 적합이라 수축은 표본내 점수를 반드시 악화시킨다 — 진단이 되지 않는다.
     대신 투수 블록을 5겹으로 갈라 **훈련 블록에서 적합 → 보류 블록에서 평가**하고
     λ 를 키운다. 보류 점수가 λ 증가와 함께 개선되면 자유 적합이 기억을 캔 것이다.
② 구조 B 상금 크기 — 명시적으로
③ 축 비직교성 — 설계행렬 상관 + 파라미터 상관(역헤시안) + 배포에서 1축 조건부 Δ
   ★ 후속 후보 BN 용: w_v6 만 0.5 → 0.70, 나머지 5개 배포 고정 Δ (단독 항목 + 부트 CI)
"""
import os

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "6"

import hashlib
import json
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, logit

H = "/home/<user>/SungjinPark"
OOT = f"{H}/exp069/oot"
OOT3 = f"{H}/exp070/oot3"
OUT = f"{H}/exp070/out"
EPS = 1e-6
REF = 0.2498069
K = 100000.0 / REF
R_BOOT = 1000
SEED_BOOT = 70
T0 = time.time()
DEP = np.array([0.5, -0.10, 0.03, 0.7, 0.25244, 0.04756])
DEP_NAMES = ["w_v6", "w_ebm", "beta_ins", "w_out", "c_aa", "c_t70"]
B_NAMES = ["w_lgb", "w_cat", "w_leaf", "w_q", "w_ebm", "w_aa", "w_t70", "beta_ins"]


def lg(p):
    return logit(np.clip(np.asarray(p, np.float64), EPS, 1.0 - EPS))


def conv(b):
    return 100000.0 * (1.0 - b / REF)


CORE = json.load(open(f"{OUT}/metrics_core.json"))

tr = pd.read_csv(f"{H}/data/train.csv",
                 usecols=["row_id", "season", "control_success", "pitcher_id"])
tr = tr[tr.season == 2024].sort_values("row_id", kind="stable").reset_index(drop=True)
y = tr.control_success.to_numpy(np.float64)
N = len(y)
ybar = float(y.mean())
pit_codes, pit_uniques = pd.factorize(tr.pitcher_id.to_numpy(), sort=True)
G = len(pit_uniques)
assert hashlib.sha256(np.ascontiguousarray(y, np.float64).tobytes()).hexdigest() == \
    "e0483e6a48d9f699ca9c4abf7b4b81aed4d679c359c7e9c3752b3f9e6e27c9d3"

V = {k: np.load(f"{OOT}/{f}").astype(np.float64) for k, f in
     [("p_lgb", "p_lgb_oot2024.npy"), ("p_cat", "p_cat_oot2024.npy"),
      ("p_leaf", "p_leaf_oot2024.npy"), ("p_q", "p_q_oot2024.npy"),
      ("p_v6", "p_v6_3arm_renorm_oot2024.npy"), ("off", "off_oot2024.npy")]}
V["p_ebm"] = np.load(f"{OOT3}/p_ebm_oot2024.npy").astype(np.float64)
AA_SEEDS = (42, 7, 2024)
L_aa = np.mean([lg(np.load(f"{H}/experiments/exp_049/pred/"
                           f"pred_A1_u2024_seed{s}_at1400.npy").astype(np.float64))
                for s in AA_SEEDS], axis=0)
L_t70 = np.mean([lg(np.load(f"{H}/experiments/exp_055/pred/"
                            f"pred_T70_u2024_seed{s}_at1400.npy").astype(np.float64))
                 for s in AA_SEEDS], axis=0)
L_v6, L_q, L_ebm = lg(V["p_v6"]), lg(V["p_q"]), lg(V["p_ebm"])
L_lgb, L_cat, L_leaf = lg(V["p_lgb"]), lg(V["p_cat"]), lg(V["p_leaf"])
off = V["off"]
XA = np.column_stack([L_v6, L_q, L_ebm, off, L_aa, L_t70])
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


STRUCT = {
    "A": {"X": XA, "vmap": vmap_A, "jac": jac_A, "names": DEP_NAMES},
    "B": {"X": XB, "vmap": lambda th: np.asarray(th, float),
          "jac": lambda th: np.eye(8), "names": B_NAMES},
    "C": {"X": XA, "vmap": lambda th: vmap_A(np.concatenate([th, [0.0, 0.0]])),
          "jac": lambda th: jac_A(np.concatenate([th, [0.0, 0.0]]))[:, :4],
          "names": DEP_NAMES[:4]},
}


def recenter_c(z, m=None, c0=0.0):
    """m = 행 마스크(bool) 또는 None(전체). 재센터링은 m 안에서만."""
    zz = z if m is None else z[m]
    yb = ybar if m is None else float(y[m].mean())
    for _ in range(40):
        p = expit(zz + c0)
        mu = float(p.mean())
        if abs(mu - yb) < 1e-14:
            break
        d = (mu - yb) / float((p * (1.0 - p)).mean())
        c0 -= d
        if abs(d) < 1e-14:
            break
    return c0


def score_z(z, m=None):
    c = recenter_c(z, m)
    zz = z if m is None else z[m]
    yy = y if m is None else y[m]
    b = float(np.mean((expit(zz + c) - yy) ** 2))
    return b, conv(b), c


def obj_grad(st, th, m=None, lam=0.0, v_target=None, c_warm=0.0):
    X, vm, jc = st["X"], st["vmap"], st["jac"]
    v = vm(th)
    z = X @ v
    c = recenter_c(z, m, c_warm)
    zz = z if m is None else z[m]
    yy = y if m is None else y[m]
    Xs = X if m is None else X[m]
    p = expit(zz + c)
    r = p - yy
    pq = p * (1.0 - p)
    U = float(len(yy))
    bv = pq * r
    sa = Xs.T @ pq
    sb = Xs.T @ bv
    dB_dv = (2.0 / U) * (sb - (sa / float(pq.sum())) * float(bv.sum()))
    B_ = float(np.mean(r * r))
    if lam > 0.0:
        dv = v if v_target is None else (v - v_target)
        B_pen = B_ + lam * float(dv @ dv)
        g = jc(th).T @ (dB_dv + 2.0 * lam * dv)
        return B_, B_pen, g, c
    return B_, B_, jc(th).T @ dB_dv, c


def fit(st, th0, m=None, lam=0.0, v_target=None):
    cw = [0.0]

    def f(th):
        B_, Bp, g, c = obj_grad(st, th, m, lam, v_target, cw[0])
        cw[0] = c
        return Bp, g

    r = minimize(f, np.asarray(th0, float), jac=True, method="L-BFGS-B",
                 options={"gtol": 1e-13, "ftol": 1e-16, "maxiter": 800})
    return np.asarray(r.x), float(r.fun)


M = {"exp_id": "exp_070_followup",
     "지시": "지휘자 2026-08-21 3건 — ① 과적합 서명 ② 구조 B 상금 ③ 축 비직교성"}
thA = np.asarray(CORE["fits_full_data"]["A"]["theta_vec"])
thB = np.asarray(CORE["fits_full_data"]["B"]["theta_vec"])
thC = np.asarray(CORE["fits_full_data"]["C"]["theta_vec"])
s_dep = CORE["deployment_full_blend"]["score_REF"]
b_dep = CORE["deployment_full_blend"]["brier"]

# ═══════════════════════════════════════════════════════════════════
# ① 과적합 서명 — (a) CI 판정  (b) 그룹 CV 수축 진단
# ═══════════════════════════════════════════════════════════════════
BCI = CORE["bootstrap"]["argmax_CI_A"]
sig = {}
for nm in ("c_t70", "w_out", "c_aa", "w_v6", "w_ebm", "beta_ins"):
    r = BCI[nm]
    lo, hi = r["ci95"]
    sig[nm] = {"point": r["point"], "ci95": [lo, hi],
               "deployment": r["deployment_value"],
               "CI_contains_0": bool(lo <= 0.0 <= hi),
               "CI_contains_deployment": bool(lo <= r["deployment_value"] <= hi),
               "CI_contains_1": bool(lo <= 1.0 <= hi)}
BCI_B = CORE["bootstrap"]["argmax_CI_B"]
sig_B_t70 = {"point": BCI_B["w_t70"]["point"], "ci95": BCI_B["w_t70"]["ci95"],
             "CI_contains_0": bool(BCI_B["w_t70"]["ci95"][0] <= 0.0
                                   <= BCI_B["w_t70"]["ci95"][1])}

# ── (b) 투수 블록 5겹 그룹 CV × λ 수축 ───────────────────────────────────
t = time.time()
rng = np.random.default_rng(707)
fold_of_block = rng.permutation(G) % 5
fold = fold_of_block[pit_codes]
LAMS = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
v_dep = vmap_A(DEP)


def cv_job(args):
    stname, lam, target = args
    st = STRUCT[stname]
    th0 = {"A": thA, "B": thB, "C": thC}[stname]
    vt = None
    if target == "dep":
        vt = v_dep if stname in ("A", "C") else np.array(
            [0.7 * 1.1 * 0.5 * 0.05 / 0.85, 0.7 * 1.1 * 0.5 * 0.68 / 0.85,
             0.7 * 1.1 * 0.5 * 0.12 / 0.85, 0.7 * 1.1 * 0.5,
             0.7 * -0.10, 0.25244, 0.04756, 0.7 * 0.03])
    num = 0.0
    den = 0
    for k in range(5):
        mtr = fold != k
        mte = fold == k
        th_k, _ = fit(st, th0, mtr, lam, vt)
        z = st["X"] @ st["vmap"](th_k)
        c = recenter_c(z, mtr)          # 🔴 재센터링 상수도 훈련 블록에서만
        p = expit(z[mte] + c)
        num += float(np.sum((p - y[mte]) ** 2))
        den += int(mte.sum())
    return (stname, lam, target, conv(num / den))


jobs = [(s, l, t_) for s in ("A", "B", "C") for l in LAMS for t_ in ("zero", "dep")]
with Pool(4) as pool:
    res = pool.map(cv_job, jobs)
CVTAB = {s: {t_: {} for t_ in ("zero", "dep")} for s in ("A", "B", "C")}
for s, l, t_, sc in res:
    CVTAB[s][t_][f"{l:g}"] = sc

# 배포 상수 자체의 CV 점수 (적합 0 — 훈련 블록에서 c 만 추정)
num = 0.0
den = 0
z_dep = XA @ v_dep
for k in range(5):
    mtr, mte = fold != k, fold == k
    c = recenter_c(z_dep, mtr)
    num += float(np.sum((expit(z_dep[mte] + c) - y[mte]) ** 2))
    den += int(mte.sum())
cv_dep = conv(num / den)


def verdict(series):
    xs = [series[f"{l:g}"] for l in LAMS]
    best_i = int(np.argmax(xs))
    return {"lambda_argmax": LAMS[best_i], "score_at_argmax": xs[best_i],
            "score_at_lam0_free": xs[0], "gain_from_shrinkage_pts": xs[best_i] - xs[0],
            "monotone_improving": bool(all(xs[i + 1] >= xs[i] - 1e-9
                                           for i in range(len(xs) - 1)))}


M["①_과적합_서명"] = {
    "관찰": {"c_t70_A": thA[5], "c_t70_B": thB[6], "배포_c_t70": 0.04756,
             "부호반전_배수": abs(thA[5] / 0.04756),
             "w_out_A": thA[3], "배포_w_out": 0.7},
    "a_부트스트랩_CI_판정": dict(
        sig, w_t70_구조B=sig_B_t70,
        판독=("c_t70 의 95% CI 가 0 을 포함하면 부호 반전은 잡음이다. "
              "포함하지 않으면 그 자체가 보고할 발견이다.")),
    "b_수축_진단": {
        "설계": ("🔴 exp_065 면 B 와 달리 여기 A/B/C 적합은 평가와 같은 행 위의 "
                 "표본내 적합이라 수축은 표본내 점수를 반드시 악화시킨다 — 진단이 안 된다. "
                 "그래서 투수 블록 5겹 그룹 CV 로 바꿨다: 훈련 블록 적합 → 보류 블록 평가. "
                 "재센터링 상수도 훈련 블록에서만 추정한다."),
        "판독법": ("보류 점수가 λ 증가와 함께 개선되면 자유 적합이 기억을 캔 것이다. "
                   "λ 를 키울 때 떨어지면 해가 실재한다."),
        "n_folds": 5, "블록": "pitcher_id 391개를 5겹에 무작위 배정",
        "lambda_grid": LAMS,
        "penalty": "λ·||v − v_target||²  (v = 유효 평면 계수, target: zero 또는 배포)",
        "배포상수_CV점수": cv_dep,
        "CV_표": CVTAB,
        "판정": {s: {t_: verdict(CVTAB[s][t_]) for t_ in ("zero", "dep")}
                 for s in ("A", "B", "C")},
        "자유적합_일반화_격차_pts": {
            s: {"표본내": CORE["fits_full_data"][s]["score_REF"],
                "보류(CV, λ=0)": CVTAB[s]["zero"]["0"],
                "격차": CORE["fits_full_data"][s]["score_REF"] - CVTAB[s]["zero"]["0"]}
            for s in ("A", "B", "C")},
        "배포_대비_CV_Δ_pts": {s: CVTAB[s]["zero"]["0"] - cv_dep for s in ("A", "B", "C")}},
    "w_out_주의": ("w_out > 1 은 그 자체로 과적합 서명이 아니다 — 로짓 확대(퍼짐 보정) 축이고, "
                   "후보 BA 가 α=1.05 로 LB +6.5034820636 을 실측한 그 축과 같은 계열이다. "
                   "⚠️ 다만 이 구조에는 α 가 들어 있지 않아 **w_out 이 α 가 이미 하는 일을 "
                   "일부 중복 흡수**한다 — 두 값을 더하지 마라."),
}
print(f"[①] CV 진단 {(time.time()-t)/60:.2f}분  배포CV {cv_dep:.4f}  "
      f"A자유CV {CVTAB['A']['zero']['0']:.4f}", flush=True)
json.dump(M, open(f"{OUT}/followup_partial.json", "w"), indent=1, ensure_ascii=False)

# ═══════════════════════════════════════════════════════════════════
# ② 구조 B 상금
# ═══════════════════════════════════════════════════════════════════
pz = CORE["bootstrap"]["delta_CI_pts"]["구조상금_B_vs_A"]
M["②_구조B_상금"] = {
    "A_opt": CORE["fits_full_data"]["A"]["score_REF"],
    "B_opt": CORE["fits_full_data"]["B"]["score_REF"],
    "상금_pts": pz["point"], "상금_CI95": pz["ci95"],
    "자유도": "A 6 → B 8 (+2)",
    "배포비용": "탈중첩 평면 출하는 mate/ 코드 수정 = C단(packager 전면 검증)",
    "결론": ("**가지 않는다.** 자유도를 8 로 늘려 얻은 것이 "
             f"{pz['point']:+.5f}점이고, 그 대가는 mate/ 코드 수정(C단)이다. "
             "상금이 비용을 못 갚는다."),
    "CV_교차확인": {"A": CVTAB["A"]["zero"]["0"], "B": CVTAB["B"]["zero"]["0"],
                    "상금_보류블록_pts": CVTAB["B"]["zero"]["0"] - CVTAB["A"]["zero"]["0"]}}

# ═══════════════════════════════════════════════════════════════════
# ③ 축 비직교성 + 1축 조건부 Δ (★ BN 용 w_v6 단독)
# ═══════════════════════════════════════════════════════════════════
t = time.time()
cols = {"L_v6": L_v6, "L_q": L_q, "L_ebm": L_ebm, "off": off,
        "L_aa": L_aa, "L_t70": L_t70, "L_lgb": L_lgb, "L_cat": L_cat, "L_leaf": L_leaf}
names = list(cols)
Cm = np.corrcoef(np.column_stack([cols[n] for n in names]).T)
M["③_축_비직교성"] = {
    "설계행렬_상관_로짓": {names[i]: {names[j]: float(Cm[i, j]) for j in range(len(names))}
                          for i in range(len(names))}}

# 파라미터 상관 (가우스-뉴턴 헤시안의 역 → 추정치 간 얽힘)
h = 1e-5
Hs = np.zeros((6, 6))
for j in range(6):
    e = np.zeros(6)
    e[j] = h
    _, _, gp, _ = obj_grad(STRUCT["A"], DEP + e)
    _, _, gm, _ = obj_grad(STRUCT["A"], DEP - e)
    Hs[:, j] = (gp - gm) / (2 * h)
Hs = 0.5 * (Hs + Hs.T)
Cov = np.linalg.pinv(Hs)
d = np.sqrt(np.abs(np.diag(Cov)))
Pc = Cov / np.outer(d, d)
M["③_축_비직교성"]["파라미터_상관_배포점_가우스뉴턴"] = {
    DEP_NAMES[i]: {DEP_NAMES[j]: float(Pc[i, j]) for j in range(6)} for i in range(6)}
M["③_축_비직교성"]["해설"] = (
    "설계행렬 상관은 '팔끼리 얼마나 겹치는가', 파라미터 상관은 '한 축을 움직이면 "
    "다른 축의 최적이 얼마나 따라 움직이는가'다. 뒤의 것이 조건부 argmax 와 "
    "동시 적합 해가 갈리는 이유를 직접 설명한다.")

# 1축 조건부 Δ (나머지 5개 배포 고정)
CD = CORE["curves_structure_A"]["center_deployment"]
one = {}
for j, nm in enumerate(DEP_NAMES):
    th = DEP.copy()
    th[j] = CD[nm]["cond_argmax"]
    _, s_, _ = score_z(XA @ vmap_A(th))
    one[nm] = {"배포값": float(DEP[j]), "조건부_argmax": CD[nm]["cond_argmax"],
               "Δ_pts_at_조건부argmax": s_ - s_dep,
               "동시적합_해": float(thA[j]),
               "cap_touched": CD[nm]["cap_touched"]}
M["③_축_비직교성"]["1축_조건부_Δ_배포에서"] = dict(
    one, 합계_naive=float(sum(v["Δ_pts_at_조건부argmax"] for v in one.values())),
    동시적합_Δ=CORE["deltas_full_data_pts"]["Δ_A재적합_vs_배포"],
    S16=("🔴 naive 합계를 동시 이동 예측으로 쓰지 마라 — 축이 직교하지 않는다. "
         "3축 실측에서 naive 합산과 재적합이 부호까지 갈렸다(S16)."))

# ★ BN 용 — w_v6 만 0.5 → 0.70, 나머지 5개 배포 고정
th70 = DEP.copy()
th70[0] = 0.70
b70, s70, c70 = score_z(XA @ vmap_A(th70))
grid_wv6 = {}
for gv in np.round(np.arange(0.50, 1.0001, 0.05), 4):
    th = DEP.copy()
    th[0] = gv
    _, sg, _ = score_z(XA @ vmap_A(th))
    grid_wv6[f"{gv:.2f}"] = {"score_REF": sg, "Δ_vs_dep_pts": sg - s_dep}


def wv6_rep(rep):
    r = np.random.default_rng([SEED_BOOT, rep])
    cnt = np.bincount(r.integers(0, G, G), minlength=G).astype(np.float64)
    u = cnt[pit_codes]

    def sc_u(th):
        z = XA @ vmap_A(th)
        c = 0.0
        U = float(u.sum())
        yb = float((u * y).sum() / U)
        for _ in range(40):
            p = expit(z + c)
            mu = float((u * p).sum() / U)
            if abs(mu - yb) < 1e-14:
                break
            dd = (mu - yb) / float((u * p * (1 - p)).sum() / U)
            c -= dd
            if abs(dd) < 1e-14:
                break
        p = expit(z + c)
        return float((u * (p - y) ** 2).sum() / U)

    b_d = sc_u(DEP)
    th_ = DEP.copy()
    th_[0] = 0.70
    b_7 = sc_u(th_)

    def f1(x):
        t2 = DEP.copy()
        t2[0] = x
        return sc_u(t2)

    rr = minimize_scalar(f1, bounds=(0.0, 1.0), method="bounded",
                         options={"xatol": 1e-5})
    return [K * (b_d - b_7), float(rr.x), K * (b_d - rr.fun)]


with Pool(4) as pool:
    wrows = np.array(pool.map(wv6_rep, range(R_BOOT), chunksize=25))
M["③_축_비직교성"]["★BN용_w_v6_단독_0.5→0.70"] = {
    "형태": "w_v6 만 0.70, 나머지 5개(w_ebm −0.10 · β 0.03 · w_out 0.7 · "
            "c_aa 0.25244 · c_t70 0.04756)는 배포값 고정",
    "배포_점수": s_dep, "이동후_점수": s70, "Δ_pts": s70 - s_dep,
    "Δ_부트_CI95": [float(np.percentile(wrows[:, 0], 2.5)),
                    float(np.percentile(wrows[:, 0], 97.5))],
    "Δ_부트_평균": float(wrows[:, 0].mean()),
    "Δ_부트_양수비율": float((wrows[:, 0] > 0).mean()),
    "조건부_argmax": CD["w_v6"]["cond_argmax"],
    "조건부_argmax_부트_CI95": [float(np.percentile(wrows[:, 1], 2.5)),
                                float(np.percentile(wrows[:, 1], 97.5))],
    "조건부_argmax_Δ_pts": one["w_v6"]["Δ_pts_at_조건부argmax"],
    "조건부_argmax_Δ_부트_CI95": [float(np.percentile(wrows[:, 2], 2.5)),
                                  float(np.percentile(wrows[:, 2], 97.5))],
    "0.50~1.00_격자": grid_wv6,
    "🔴": ("이것은 fold_2024 out-of-time 로컬 눈금이다. S1 — LB 예측치로 옮기지 마라. "
           "전이 배율은 별건(BN)이 실측한다."),
    "편향": ("p_v6 가 NN 없는 3종판이라 w_v6 는 **아래로 눌린 하한**이고, "
             "p_q 가 1시드라 Q 쪽도 눌려 w_v6 를 위로 민다 — 두 편향이 반대다.")}
M["wall_clock_min"] = {"followup_total_min": (time.time() - T0) / 60.0,
                       "s3_min": (time.time() - t) / 60.0}
json.dump(M, open(f"{OUT}/followup.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({
    "c_t70_CI": sig["c_t70"]["ci95"], "c_t70_CI_contains_0": sig["c_t70"]["CI_contains_0"],
    "w_out_CI": sig["w_out"]["ci95"],
    "cv_dep": cv_dep, "cv_A_free": CVTAB["A"]["zero"]["0"],
    "shrink_verdict_A_zero": verdict(CVTAB["A"]["zero"]),
    "prize_B": pz["point"],
    "wv6_070_delta": s70 - s_dep,
    "wv6_070_CI": M["③_축_비직교성"]["★BN용_w_v6_단독_0.5→0.70"]["Δ_부트_CI95"],
}, ensure_ascii=False), flush=True)
print("DONE_FOLLOWUP070", flush=True)
