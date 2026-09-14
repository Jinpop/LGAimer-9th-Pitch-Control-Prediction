"""exp_072 stage1 — `w_v6` 축 꼭지점 기계 (BI→BL 수순 그대로).  학습 0건.

축 파라미터 u = (파트너 내부 w_v6) - 0.5.  프로브 u = 0.20 (후보 BN).
delta = (p_BN - p_BL)/u  ← **출하 산술 그대로** (오프셋을 블렌드 로짓에 넣고
                            -> alpha/beta_skew -> d_adj 재못박기)
G = mean(delta^2),  K = 100000/0.2498069
L = (g/K + u^2 G)/(2u),  꼭지점 u* = L/G,  최대이득 K L^2 / G
🔴 K*u^2*G 를 함께 낸다 — 산술 자체 검사.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "6"

import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar
from scipy.special import expit, logit

H = "/home/<user>/SungjinPark"
PARENT = f"{H}/build_BL/pkg_BL"
FRAME = f"{H}/build_AC/work/tr2024"
PACK = f"{H}/build_AI/pack/ins"
OUT = f"{H}/exp072"
os.makedirs(OUT, exist_ok=True)

EPS = 1e-6
REF = 0.2498069
K = 100000.0 / REF
N_ROWS = 253507
U_PROBE = 0.20                      # w_v6 0.50 -> 0.70
LB_BL = 1114.1019927547
LB_BN = 1115.9165956458
T0 = time.time()

PIN = {
    f"{H}/build_AR/work/tr2024_p_v19_AR.npy":
        "99909f0692a2507d0b599b58cac2ecc0b59b086d1a800a899d5fbabd2f1755f4",
    f"{H}/build_AY/work/pcal_aa.npy":
        "3e7159abd25fbfb7441f69bf3f4022b9aac3dbdaf73af8da2765f9c1b645d5bd",
    f"{H}/build_AY/work/pcal_S2.npy":
        "f99a6240c08c53e3e40fa293a0f9efb0911d067236a8cf359d1855f11fb58b99",
    f"{H}/build_AY/work/pcal_S1.npy":
        "776cb0accc0236823b6b2d71612de92fb3711908254037b564d64c0e1c8c0905",
    f"{H}/build_AY/work/pcal_T70.npy":
        "1bac0abde366df280c28f421527066d8c132eb55a1182efc513c36ba5652ac5c",
    f"{FRAME}/test.csv":
        "4db38113d5355546339c4391fbbf8b2e152a7c248ad9261caa1071f6bb81f5d0",
    f"{PACK}/anchor_lt2024.npz":
        "0e4b586da41cd6e78c5e19d4dd656bc9df088413625945e2a88ae85d100e5eb7",
    f"{H}/exp059_screen/directions.npz":
        "5aea557dff22a18b1943419b7890e74220753cf687821d3513f8a0d0341effe6",
}
BL_ZIP = f"{H}/build_BL/submit_BL_joint_axes.zip"
BL_ZIP_SHA = "e8401d695d212ff4d47f680b9ce44fe75b546189f334f430956c38f8ed935e88"
BN_PARTNER = f"{H}/build_BN/work/tr2024_p_v19_BN_wv6_070.npy"
BL_D_ADJ = 0.02950104990829574
BN_D_ADJ = 0.028610055723290194


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def lg(p):
    return logit(np.clip(np.asarray(p, np.float64), EPS, 1.0 - EPS))


M = {"exp_id": "exp_072", "stage": "1_vertex_w_v6"}
print("=== sha 대조 ===", flush=True)
for p, want in PIN.items():
    got = sha_file(p)
    assert got == want, f"sha 불일치 {p}\n 기대 {want}\n 실측 {got}"
assert sha_file(BL_ZIP) == BL_ZIP_SHA, "부모 BL zip sha 불일치"
M["sha_pins_ok"] = True
M["sha_bn_partner_tr2024"] = sha_file(BN_PARTNER)
print(f"  🟢 전부 일치.  BN 파트너 출력 sha {M['sha_bn_partner_tr2024'][:16]}", flush=True)

bl0 = json.load(open(f"{PARENT}/blend.json", encoding="utf-8"))
A = float(bl0["alpha"]); B = float(bl0["beta_skew"])
D_BL = float(bl0["d_adj"]); T = float(bl0["target_rate"])
W_OUT = float(bl0["w_v6"]); C_AA = float(bl0["c_aa"])
SW = bl0["smooth_weights"]
C_S2 = float(SW["S2"]); C_S1 = float(SW["S1"]); C_T70 = float(SW["T70"])
AXL = bl0["axis_offsets"]
U1 = float(AXL[0]["u"]); U2 = float(AXL[1]["u"])
assert D_BL == BL_D_ADJ
mb0 = json.load(open(f"{PARENT}/mate/model/blend.json", encoding="utf-8"))
assert mb0["w_v6"] == 0.5

p_old = np.load(f"{H}/build_AR/work/tr2024_p_v19_AR.npy")
p_new = np.load(BN_PARTNER)
assert p_old.shape == (N_ROWS,) and p_new.shape == (N_ROWS,)
p_aa = np.load(f"{H}/build_AY/work/pcal_aa.npy")
p_S2 = np.load(f"{H}/build_AY/work/pcal_S2.npy")
p_S1 = np.load(f"{H}/build_AY/work/pcal_S1.npy")
p_T70 = np.load(f"{H}/build_AY/work/pcal_T70.npy")
y = np.load(f"{FRAME}/labels.npy").astype(np.float64)

sys.path.insert(0, PARENT)
import inseason                                                    # noqa: E402
tt = pd.read_csv(f"{FRAME}/test.csv", encoding="utf-8-sig")
I = inseason.apply(tt, inseason.load(f"{PACK}/anchor_lt2024.npz"))
_cur = np.clip(I["ins_p_success"].to_numpy(np.float64), 0.02, 0.98)
_pri = np.clip(_cur - I["ins_p_success_dev"].to_numpy(np.float64), 0.02, 0.98)
off_lt = np.nan_to_num(np.log(_cur / (1 - _cur)) - np.log(_pri / (1 - _pri)))
assert np.array_equal(off_lt, np.load(f"{H}/arms/off.npy"))
D59 = np.load(f"{H}/exp059_screen/directions.npz")
RAW = {"beta_ins_off": off_lt,
       "ctrl_asof_psr": np.asarray(D59["ctrl_asof_psr"], np.float64)}
DEL = {}
for a in AXL:
    nm = a["name"]
    v = np.asarray(RAW[nm], np.float64)
    fin = np.isfinite(v)
    mu, sd = float(v[fin].mean()), float(v[fin].std())
    assert mu == a["center"] and sd == a["scale"], nm
    u = np.zeros(len(v)); u[fin] = (v[fin] - mu) / sd
    DEL[nm] = u
OFFV = U1 * DEL["beta_ins_off"] + U2 * DEL["ctrl_asof_psr"]

L_aa, L_S2, L_S1, L_T70 = lg(p_aa), lg(p_S2), lg(p_S1), lg(p_T70)
Z_FIX = C_AA * L_aa + C_S2 * L_S2 + C_S1 * L_S1 + C_T70 * L_T70


def anc(f):
    return brentq(lambda d: expit(f + d).mean() - T, -5, 5, xtol=1e-15)


def build_from_logit(Lp, shift):
    """출하 산술 그대로.  Lp = 파트너 팔 로짓."""
    z = W_OUT * Lp + Z_FIX
    zp = z + OFFV
    f = A * zp + B * zp * zp
    dd = anc(f) + shift
    return expit(f + dd), dd, zp


L_old, L_new = lg(p_old), lg(p_new)
# STEP A — 솔버 자체 증명
_, d_rep, _ = build_from_logit(L_old, 0.0)
SHIFT = D_BL - d_rep
M["STEP_A"] = {"BL_shipped_d_adj": repr(D_BL), "solver": repr(d_rep),
               "abs_diff": abs(SHIFT), "PASS": bool(abs(SHIFT) < 1e-12)}
assert abs(SHIFT) < 1e-12, "STEP A 실패"

p_BL, d0, _ = build_from_logit(L_old, SHIFT)
p_BN, d2, _ = build_from_logit(L_new, SHIFT)
assert d0 == BL_D_ADJ, (d0, BL_D_ADJ)
M["d_adj_reproduction"] = {"BL": repr(d0), "BN": repr(d2),
                           "BN_recorded_in_solve_BN": repr(BN_D_ADJ),
                           "BN_abs_diff": abs(d2 - BN_D_ADJ),
                           "PASS": bool(abs(d2 - BN_D_ADJ) < 1e-15)}
assert abs(d2 - BN_D_ADJ) < 1e-15, "BN d_adj 재현 실패"
print(f"\n=== d_adj 재현 🟢  BL {d0!r}  BN {d2!r}", flush=True)

br_BL = float(np.mean((p_BL - y) ** 2))
br_BN = float(np.mean((p_BN - y) ** 2))
M["in_sample_cross_check"] = {
    "brier_BL": br_BL, "brier_BN": br_BN,
    "recorded_solve_BN": {"brier_BL": 0.24538918841703036,
                          "brier_BN": 0.24552396647966293},
    "abs_diff_BL": abs(br_BL - 0.24538918841703036),
    "abs_diff_BN": abs(br_BN - 0.24552396647966293),
    "🔴": "표본내 — 순위 근거가 아니다"}

# ══════════════════════════════════════════════════════════════════
# 꼭지점 기계
# ══════════════════════════════════════════════════════════════════
g = LB_BN - LB_BL
delta = (p_BN - p_BL) / U_PROBE
G = float(np.mean(delta ** 2))
Q = U_PROBE ** 2 * G
KQ = K * Q
L = (g / K + Q) / (2.0 * U_PROBE)
u_star = L / G
gain_max = K * L * L / G
r = g / KQ + 1.0
u_star_via_r = r * U_PROBE / 2.0

M["vertex"] = {
    "axis": "partner_inner_w_v6 (u = w_v6 - 0.5)",
    "u_probe": U_PROBE, "LB_parent_BL": LB_BL, "LB_probe_BN": LB_BN,
    "g_pts": g, "K": K,
    "G_mean_delta2": G,
    "Q_eq_u2G": Q,
    "K_times_u2_G_pts": KQ,
    "L": L,
    "r_eq_g_over_KQ_plus1": r,
    "u_star": u_star, "u_star_via_r": u_star_via_r,
    "u_star_consistency_abs": abs(u_star - u_star_via_r),
    "w_v6_star": 0.5 + u_star,
    "gain_max_pts_vs_BL": gain_max,
    "remaining_pts_vs_BN": gain_max - g,
    "delta_stats": {"max_abs": float(np.abs(delta).max()),
                    "mean_abs": float(np.abs(delta).mean()),
                    "rms": float(np.sqrt(G))},
    "pred_diff_BN_minus_BL": {"max_abs": float(np.abs(p_BN - p_BL).max()),
                              "mean_abs": float(np.abs(p_BN - p_BL).mean())},
}
print(f"\n=== 꼭지점 ===")
print(f"  g {g:+.10f}점  ·  G {G:.12e}  ·  K*u^2*G {KQ:.10f}점")
print(f"  L {L:.12e}  ·  r {r:.10f}")
print(f"  u* {u_star:+.8f}  ->  w_v6* {0.5+u_star:.6f}")
print(f"  최대이득(BL 대비) {gain_max:+.6f}점  ·  BN 대비 남은 몫 {gain_max-g:+.6f}점",
      flush=True)

# 예측 이득 곡선 (2차 모형)
grid = {}
for wv in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]:
    uu = wv - 0.5
    grid[f"{wv:.2f}"] = K * (2 * uu * L - uu * uu * G)
M["vertex"]["model_gain_curve_vs_BL_pts"] = grid
M["vertex"]["model_gain_at_honest_argmax"] = {
    "exp_070_3arm_0.8581420253062062":
        K * (2 * (0.8581420253062062 - 0.5) * L
             - (0.8581420253062062 - 0.5) ** 2 * G),
    "exp_071_4arm_0.77935483531056":
        K * (2 * (0.77935483531056 - 0.5) * L
             - (0.77935483531056 - 0.5) ** 2 * G)}

# ══════════════════════════════════════════════════════════════════
# 곡선 정확도 — 파트너 로짓이 u 에 대해 아핀이라는 성질을 쓴다
#   lgP(u) = L_old + (u/0.20)*(L_new - L_old)
# ══════════════════════════════════════════════════════════════════
dL = L_new - L_old


def p_exact(u):
    return build_from_logit(L_old + (u / U_PROBE) * dL, SHIFT)[0]


chk = p_exact(U_PROBE)
M["affine_roundtrip_at_probe"] = {
    "max_abs_vs_build_from_prob": float(np.abs(chk - p_BN).max()),
    "note": "파트너 확률에서 lg 를 취해 되돌린 것과 출하 build 결과의 차 — 0 이어야"}

lin = {}
for t in (0.25, 0.5, 1.0, 2.0, 3.0):
    uu = t * U_PROBE
    pe = p_exact(uu) - p_BL
    pl = uu * delta
    lin[f"t={t:g} (w_v6={0.5+uu:.2f})"] = {
        "corr": float(np.corrcoef(pe, pl)[0, 1]),
        "norm_ratio_over_t": float(np.linalg.norm(pe) / np.linalg.norm(pl)),
        "max_resid": float(np.abs(pe - pl).max()),
        "proj_a_u_over_u": float((pe @ delta) / (delta @ delta) / uu)}
M["linearity_of_p_in_u"] = lin

# 곡률 반영 정제 꼭지점: p(u)-p0 = a(u)*delta + e,  e ⟂ delta,  mean(e·R)=0 가정
dd = float(delta @ delta)


def g_exact_model(u):
    if u == 0.0:
        return 0.0
    d_ = p_exact(u) - p_BL
    a = float(d_ @ delta) / dd
    return K * (2.0 * a * L - float(np.mean(d_ ** 2)))


rr = minimize_scalar(lambda x: -g_exact_model(x), bounds=(0.0, 0.5),
                     method="bounded", options={"xatol": 1e-6})
M["vertex_curvature_refined"] = {
    "가정": ("p(u) 를 정확 계산하고 delta 에 사영해 선형항을 a(u)*L 로 둔다. "
             "delta 에 직교한 성분과 잔차 R 의 내적이 0 이라는 **추가 가정**이 붙는다."),
    "u_star_refined": float(rr.x), "w_v6_star_refined": 0.5 + float(rr.x),
    "gain_max_refined_pts_vs_BL": float(-rr.fun),
    "g_exact_at_probe": g_exact_model(U_PROBE),
    "g_exact_grid": {f"{0.5+u:.2f}": g_exact_model(u)
                     for u in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40)}}

# 표본내 대조 (참고 — 순위 근거 아님)
ins = {}
for wv in (0.50, 0.60, 0.70, 0.80, 0.90, 1.00):
    pe = p_exact(wv - 0.5)
    ins[f"{wv:.2f}"] = K * (float(np.mean((p_BL - y) ** 2))
                            - float(np.mean((pe - y) ** 2)))
M["in_sample_gain_curve_vs_BL_pts"] = ins
M["in_sample_gain_curve_🔴"] = ("tr2024 는 다섯 팔 전부에 표본 내 — LB 와 부호가 "
                                "반대일 수 있다. 순위 근거가 아니다.")
M["wall_clock_min"] = (time.time() - T0) / 60.0
json.dump(M, open(f"{OUT}/vertex072.json", "w"), indent=1, ensure_ascii=False)
print(f"\n기록 {OUT}/vertex072.json  ·  벽시계 {M['wall_clock_min']:.2f}분", flush=True)
