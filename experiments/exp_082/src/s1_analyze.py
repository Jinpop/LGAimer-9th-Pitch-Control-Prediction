"""exp_082 — 실패유형 오프셋 축: b 꼭지점 풀기 + c 첫 특성화.  학습 0건."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "6"
import hashlib, json, sys, time                                    # noqa: E402
import numpy as np                                                 # noqa: E402
import pandas as pd                                                # noqa: E402
from scipy.optimize import brentq, minimize_scalar                 # noqa: E402
from scipy.special import expit, logit                             # noqa: E402

sys.path.insert(0, "/home/<user>/SungjinPark/exp078/src")
sys.path.insert(0, "/home/<user>/SungjinPark/exp079/src")
import core078 as C                                                # noqa: E402

H = C.H; K = C.K; REF = C.REF; N = C.N_ROWS; T_REF = C.T_REF
OUT = f"{H}/exp082"
FRAME = C.FRAME
T0 = time.time()
MU_MR = -0.45376277509804325
MU_WO = -2.0056142032773767
B_SHIP, C_SHIP = -0.05, 0.0
D_BO = 0.02875748759679648
D_NEW = 0.028693042338963914
G_LB = 2.6766924977            # LB 1118.6680789834 − BO 1115.9913864857
LB_BO, LB_TM = 1115.9913864857, 1118.6680789834
R = {"exp_id": "exp_082", "K_pts_per_brier": K, "ref_brier": REF,
     "threads": "OMP/OPENBLAS/MKL/NUMEXPR/VECLIB = 6",
     "core078_stage0_gates": C.M["stage0_gates"],
     "constants": {"mu_mr": MU_MR, "mu_wo": MU_WO, "b_shipped": B_SHIP,
                   "c_shipped": C_SHIP, "d_adj_BO": D_BO, "d_adj_new": D_NEW,
                   "LB_BO": LB_BO, "LB_teammate": LB_TM, "g_pts": G_LB}}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def pts(b):
    return 100000.0 * (1.0 - b / REF)


IDS = pd.read_csv(f"{FRAME}/test.csv", encoding="utf-8-sig",
                  usecols=["row_id"])["row_id"].astype(str).tolist()
assert len(IDS) == N


def read_ft(path, ids):
    a = pd.read_csv(path, encoding="utf-8-sig")
    b = pd.read_csv(path, encoding="utf-8-sig", float_precision="round_trip")
    assert list(a.columns) == ["row_id", "lg_mr", "lg_wo"]
    assert a["row_id"].astype(str).tolist() == list(ids), "failtype row_id 순서 불일치"
    return {"mr_def": a.lg_mr.to_numpy(np.float64), "wo_def": a.lg_wo.to_numpy(np.float64),
            "mr_rt": b.lg_mr.to_numpy(np.float64), "wo_rt": b.lg_wo.to_numpy(np.float64)}


def read_sub(path, ids):
    d = pd.read_csv(path, encoding="utf-8-sig", float_precision="round_trip")
    assert d["row_id"].astype(str).tolist() == list(ids), f"{path} row_id 순서 불일치"
    return d["control_success"].to_numpy(np.float64)


# ══════════════════════════════════════════════════ 1. tr2024 보정 프레임
FT = read_ft(f"{OUT}/runs/ft_matecal.csv", IDS)
LGMR, LGWO = FT["mr_rt"], FT["wo_rt"]
p_part_fresh = read_sub(f"{OUT}/runs/matecal_partner.csv", IDS)
p_part_store = np.load(f"{H}/build_BO/work/tr2024_p_v19_BO_wv6_vertex.npy")
_dp = np.abs(p_part_fresh - p_part_store)
R["mate_reproduction_tr2024"] = {
    "bitwise_identical": bool(np.array_equal(p_part_fresh, p_part_store)),
    "n_rows_differ": int(np.count_nonzero(_dp)), "max_abs": float(_dp.max()),
    "note": "S13 — 세션 간 SIMD 축약 순서 차이는 1e-9급으로 알려져 있다"}
print(f"[mate 재현] 비트동일 {R['mate_reproduction_tr2024']['bitwise_identical']} "
      f"ndiff {R['mate_reproduction_tr2024']['n_rows_differ']} "
      f"max {_dp.max():.3e}", flush=True)

# S21 — 기본 파서 vs round_trip
_s21 = {}
for nm, a, b in (("lg_mr", FT["mr_def"], FT["mr_rt"]), ("lg_wo", FT["wo_def"], FT["wo_rt"])):
    d = np.abs(a - b)
    _s21[nm] = {"n_rows_differ": int(np.count_nonzero(d)),
                "share": float(np.count_nonzero(d) / len(d)),
                "max_abs_diff": float(d.max()),
                "max_rel_ulp": float((d / np.maximum(np.abs(b), 1e-300)).max())}
R["S21_float_parser"] = _s21

W_TR = C.zf(C.BO, C.ARMS18)                 # alpha*z' + beta_skew*z'^2
_d0 = C.anc(W_TR, T_REF)
p_BO_tr = expit(W_TR + _d0)
assert abs(_d0 - D_BO) < 1e-15


def off(mr, wo, b, c):
    return b * (mr - MU_MR) + c * (wo - MU_WO)


def dsolve(b, c):
    return C.anc(W_TR + off(LGMR, LGWO, b, c), T_REF)


G0 = {}
# G0b — b=0,c=0 이면 BO 예측이 정확히 재현되는가 (계산판)
d00 = dsolve(0.0, 0.0)
p00 = expit(W_TR + d00 + off(LGMR, LGWO, 0.0, 0.0))
G0["G0b"] = {"what": "b=0,c=0 → BO 예측·d_adj 정확 재현 (tr2024 보정 프레임)",
             "d_adj_solved": repr(d00), "d_adj_BO_shipped": repr(D_BO),
             "abs_diff_d": abs(d00 - D_BO),
             "pred_bitwise_identical_to_BO": bool(np.array_equal(p00, p_BO_tr)),
             "pred_max_abs_diff": float(np.abs(p00 - p_BO_tr).max()),
             "PASS": bool(abs(d00 - D_BO) < 1e-15 and np.array_equal(p00, p_BO_tr))}
# G0c — 팀원 d_adj 독립 재유도
d_bc = dsolve(B_SHIP, C_SHIP)
G0["G0c"] = {"what": "b=-0.05,c=0 에서 팀원 d_adj 독립 재유도",
             "d_adj_solved": repr(d_bc), "d_adj_shipped": repr(D_NEW),
             "abs_diff": abs(d_bc - D_NEW),
             "PASS": bool(abs(d_bc - D_NEW) < 1e-15)}
R["mu_independent_check"] = {
    "mean_lg_mr_tr2024": float(LGMR.mean()), "mu_mr_shipped": MU_MR,
    "abs_diff_mr": abs(float(LGMR.mean()) - MU_MR),
    "mean_lg_wo_tr2024": float(LGWO.mean()), "mu_wo_shipped": MU_WO,
    "abs_diff_wo": abs(float(LGWO.mean()) - MU_WO)}
print(f"[G0b] |Δd| {G0['G0b']['abs_diff_d']:.3e} · 예측 비트동일 "
      f"{G0['G0b']['pred_bitwise_identical_to_BO']}", flush=True)
print(f"[G0c] d {d_bc!r} vs {D_NEW!r} → |Δ| {G0['G0c']['abs_diff']:.3e}", flush=True)
print(f"[mu] mean(lg_mr) {LGMR.mean()!r}  |Δ| {R['mu_independent_check']['abs_diff_mr']:.3e}"
      f" · mean(lg_wo) |Δ| {R['mu_independent_check']['abs_diff_wo']:.3e}", flush=True)

# ══════════════════════════════════════════════════ 2. 배포형 프레임 3종
AUD = f"{H}/audit_BT/runs"
FRAMES = {}
FRAMES["tr2024_cal"] = {"kind": "calibration", "n": N, "w": W_TR,
                        "mr": LGMR, "wo": LGWO, "p_bo": p_BO_tr,
                        "anchor": "lt2024(보정 규약)"}
G0A = {}
for fr, fdir in (("cov", "cov"), ("band", "band"), ("real5", "real5")):
    ids_f = pd.read_csv(f"{H}/build_BT/frames/{fdir}/test.csv", encoding="utf-8-sig",
                        usecols=["row_id"])["row_id"].astype(str).tolist()
    ft = read_ft(f"{OUT}/runs/ft_bonly__{fr}.csv", ids_f)
    p_bo = read_sub(f"{AUD}/au_bo__{fr}.csv", ids_f)
    p_tm = read_sub(f"{OUT}/runs/bonly__{fr}.csv", ids_f)
    w = logit(p_bo) - D_BO
    lhs = logit(p_tm)
    rhs = w + D_NEW + off(ft["mr_rt"], ft["wo_rt"], B_SHIP, C_SHIP)
    err = np.abs(lhs - rhs)
    G0A[fr] = {"n_rows": len(ids_f), "max_abs_logit_err": float(err.max()),
               "mean_abs_logit_err": float(err.mean()),
               "max_abs_p_err": float(np.abs(expit(rhs) - p_tm).max()),
               "PASS": bool(err.max() < 1e-9)}
    FRAMES[fr] = {"kind": "deployment_standin", "n": len(ids_f), "w": w,
                  "mr": ft["mr_rt"], "wo": ft["wo_rt"], "p_bo": p_bo,
                  "anchor": "출하 앵커", "s21": {
                      k: {"n_rows_differ": int(np.count_nonzero(ft[k + "_def"] - ft[k + "_rt"])),
                          "max_abs_diff": float(np.abs(ft[k + "_def"] - ft[k + "_rt"]).max())}
                      for k in ("mr", "wo")}}
    print(f"[G0a/{fr}] 산술 항등 max|Δlogit| {err.max():.3e}  n={len(ids_f)}", flush=True)
G0["G0a"] = {"what": "팀원 최종 예측 = BO 예측 로짓 + (d_new−d_BO) + b·(lg_mr−mu_mr)",
             "frames": G0A, "PASS": all(v["PASS"] for v in G0A.values())}

# 항등 앵커 — b=0,c=0,d=d_BO 패키지가 BO 출력을 재현하는가
IDENT = {}
_pi = f"{OUT}/runs/ident__cov.csv"
_pb = f"{OUT}/runs/boref__cov.csv"
_pa = f"{AUD}/au_bo__cov.csv"
for nm, p in (("ident__cov(이번 세션)", _pi), ("boref__cov(이번 세션 BO)", _pb),
              ("au_bo__cov(이전 세션 BO)", _pa)):
    IDENT[nm] = sha(p)
v_i, v_b, v_a = (read_sub(x, pd.read_csv(f"{H}/build_BT/frames/cov/test.csv",
                          encoding="utf-8-sig", usecols=["row_id"])["row_id"].astype(str).tolist())
                 for x in (_pi, _pb, _pa))
IDENT["ident_vs_boref_same_session_bitwise"] = bool(np.array_equal(v_i, v_b))
IDENT["ident_vs_boref_sha_equal"] = IDENT["ident__cov(이번 세션)"] == IDENT["boref__cov(이번 세션 BO)"]
IDENT["ident_vs_au_bo_prev_session_ndiff"] = int(np.count_nonzero(v_i - v_a))
IDENT["ident_vs_au_bo_prev_session_max_abs"] = float(np.abs(v_i - v_a).max())
IDENT["S13_note"] = "세션이 다르면 팀원 numpy NN 의 SIMD 축약 순서로 1e-9급 차이가 날 수 있다"
G0["identity_anchor_cov"] = IDENT
G0["identity_anchor_PASS"] = bool(IDENT["ident_vs_boref_sha_equal"])
print(f"[항등앵커] 같은 세션 sha 일치 {IDENT['ident_vs_boref_sha_equal']} · "
      f"이전 세션 대비 ndiff {IDENT['ident_vs_au_bo_prev_session_ndiff']} "
      f"max {IDENT['ident_vs_au_bo_prev_session_max_abs']:.3e}", flush=True)
R["stage0_gates"] = G0
if not (G0["G0a"]["PASS"] and G0["G0b"]["PASS"] and G0["G0c"]["PASS"]):
    R["ABORTED"] = "stage0 게이트 실패"
    json.dump(R, open(f"{OUT}/out/metrics.json", "w"), indent=1, ensure_ascii=False)
    raise SystemExit("stage0 게이트 실패 — 정지")
print("[G0] 3종 통과\n", flush=True)


# ══════════════════════════════════════════════════ 3. 프레임별 δ / G
def pred(fr, b, c):
    F = FRAMES[fr]
    return expit(F["w"] + dsolve(b, c) + off(F["mr"], F["wo"], b, c))


HB = HC = 0.002
DEL, GS = {}, {}
for fr in FRAMES:
    F = FRAMES[fr]
    p0 = pred(fr, 0.0, 0.0)
    db = (pred(fr, HB, 0.0) - pred(fr, -HB, 0.0)) / (2 * HB)
    dc = (pred(fr, 0.0, HC) - pred(fr, 0.0, -HC)) / (2 * HC)
    s4b = (pred(fr, 4 * HB, 0.0) - p0) / (4 * HB)
    s4c = (pred(fr, 0.0, 4 * HC) - p0) / (4 * HC)
    DEL[fr] = {"b": db, "c": dc, "p0": p0}
    gbb, gcc, gbc = float(np.mean(db ** 2)), float(np.mean(dc ** 2)), float(np.mean(db * dc))
    GS[fr] = {"n_rows": F["n"], "anchor": F["anchor"], "kind": F["kind"],
              "rms_delta_b": float(np.sqrt(gbb)), "rms_delta_c": float(np.sqrt(gcc)),
              "G_bb": gbb, "G_cc": gcc, "G_bc": gbc,
              "K_G_bb_pts_at_u1": K * gbb, "K_G_cc_pts_at_u1": K * gcc,
              "rho_b_c": float(gbc / np.sqrt(gbb * gcc)),
              "corr_lg_mr_lg_wo": float(np.corrcoef(F["mr"], F["wo"])[0, 1]),
              "sd_lg_mr": float(F["mr"].std()), "sd_lg_wo": float(F["wo"].std()),
              "mean_lg_mr": float(F["mr"].mean()), "mean_lg_wo": float(F["wo"].mean()),
              "secant4h_norm_ratio_b": float(np.linalg.norm(s4b) / np.linalg.norm(db)),
              "secant4h_norm_ratio_c": float(np.linalg.norm(s4c) / np.linalg.norm(dc))}
    print(f"[δ/{fr:<11}] rms_b {GS[fr]['rms_delta_b']:.5e} rms_c {GS[fr]['rms_delta_c']:.5e}"
          f"  K·G_bb {K*gbb:8.4f}  K·G_cc {K*gcc:8.4f}  ρ(b,c) {GS[fr]['rho_b_c']:+.4f}"
          f"  corr(mr,wo) {GS[fr]['corr_lg_mr_lg_wo']:+.4f}", flush=True)
R["stage1_frames"] = GS

# ══════════════════════════════════════════════════ 4. stage1 — b 꼭지점
u = B_SHIP
S1 = {}
for fr in FRAMES:
    Q_exact = float(np.mean((pred(fr, u, 0.0) - DEL[fr]["p0"]) ** 2))
    G_der = GS[fr]["G_bb"]
    G_sec = Q_exact / u ** 2
    for tag, Gv, Qv in (("derivative_G", G_der, G_der * u ** 2),
                        ("secant_G", G_sec, Q_exact)):
        L = (G_LB / K + Qv) / (2 * u)
        ustar = L / Gv
        gmax = K * L ** 2 / Gv
        S1[f"{fr}/{tag}"] = {
            "G_used": Gv, "K_times_Q_at_u_pts": K * Qv, "L": L,
            "b_star": float(0.0 + ustar), "u_star": ustar,
            "max_gain_pts": gmax, "gain_already_bought_pts": G_LB,
            "remaining_headroom_pts": gmax - G_LB,
            "predicted_LB_at_vertex": LB_BO + gmax}
    S1[f"{fr}/nonlinearity"] = {"G_secant_over_G_derivative": G_sec / G_der,
                                "K_times_Q_exact_pts": K * Q_exact}
    print(f"[b*/{fr:<11}] 도함수G b* {S1[f'{fr}/derivative_G']['b_star']:+.6f} "
          f"최대 {S1[f'{fr}/derivative_G']['max_gain_pts']:+.5f}점 남은몫 "
          f"{S1[f'{fr}/derivative_G']['remaining_headroom_pts']:+.5f} | 할선G b* "
          f"{S1[f'{fr}/secant_G']['b_star']:+.6f} 남은몫 "
          f"{S1[f'{fr}/secant_G']['remaining_headroom_pts']:+.5f} | G비 {G_sec/G_der:.4f}",
          flush=True)
_bs = [S1[f"{fr}/derivative_G"]["b_star"] for fr in FRAMES]
_gm = [S1[f"{fr}/derivative_G"]["max_gain_pts"] for fr in FRAMES]
_kg = [GS[fr]["K_G_bb_pts_at_u1"] for fr in FRAMES]
R["stage1_b_vertex"] = S1
R["stage1_frame_spread"] = {
    "b_star_min": min(_bs), "b_star_max": max(_bs), "b_star_range": max(_bs) - min(_bs),
    "max_gain_min_pts": min(_gm), "max_gain_max_pts": max(_gm),
    "remaining_min_pts": min(_gm) - G_LB, "remaining_max_pts": max(_gm) - G_LB,
    "K_G_bb_min": min(_kg), "K_G_bb_max": max(_kg),
    "K_G_bb_ratio_max_over_min": max(_kg) / min(_kg)}

# ══════════════════════════════════════════════════ 5. stage2 — c 특성화
S2 = {}
for fr in FRAMES:
    gcc = GS[fr]["G_cc"]
    c1 = float(1.0 / np.sqrt(K * gcc))          # 1차 근사

    def KQ(cv):
        return K * float(np.mean((pred(fr, 0.0, cv) - DEL[fr]["p0"]) ** 2)) - 1.0
    try:
        c_exact = brentq(KQ, c1 * 0.2, c1 * 5.0, xtol=1e-12)
    except Exception:
        c_exact = c1

    def nl(cv):
        d = pred(fr, 0.0, cv) - DEL[fr]["p0"]
        lin = cv * DEL[fr]["c"]
        return float(np.linalg.norm(d - lin) / np.linalg.norm(lin)) - 0.01
    try:
        c_nl = brentq(nl, 1e-4, 5.0, xtol=1e-10)
    except Exception:
        c_nl = None
    S2[fr] = {"K_G_cc_pts_at_u1": K * gcc,
              "amp_KQ1_firstorder": c1, "amp_KQ1_exact": c_exact,
              "amp_S42_buy_0p2x": 0.2 * c_exact,
              "K_times_Q_at_0p2x_pts": K * float(np.mean(
                  (pred(fr, 0.0, 0.2 * c_exact) - DEL[fr]["p0"]) ** 2)),
              "nonlinear_1pct_amplitude": c_nl,
              "rho_with_b_axis": GS[fr]["rho_b_c"],
              "corr_lg_mr_lg_wo": GS[fr]["corr_lg_mr_lg_wo"]}
    print(f"[c/{fr:<11}] K·G_cc {K*gcc:9.4f}  KQ1 진폭 {c_exact:.6f}  0.2배 "
          f"{0.2*c_exact:.6f}  비선형1% {c_nl}", flush=True)

# ρ — 기존 15축 (tr2024 보정 프레임, core078 δ) + exp_080 후보 9축
def rho(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


dc_tr, db_tr = DEL["tr2024_cal"]["c"], DEL["tr2024_cal"]["b"]
RHO15_C = {a: rho(dc_tr, C.DELTA[C.IDX[a]]) for a in C.AXES}
RHO15_B = {a: rho(db_tr, C.DELTA[C.IDX[a]]) for a in C.AXES}
_nm9 = list(np.load(f"{H}/exp080/out/cand_delta_names.npy", allow_pickle=True))
_d9 = np.load(f"{H}/exp080/out/cand_delta.npy")
RHO9_C = {n: rho(dc_tr, _d9[i]) for i, n in enumerate(_nm9)}
RHO9_B = {n: rho(db_tr, _d9[i]) for i, n in enumerate(_nm9)}
_allc = dict(RHO15_C); _allc.update(RHO9_C); _allc["b_failtype"] = GS["tr2024_cal"]["rho_b_c"]
_mx = max(_allc.items(), key=lambda kv: abs(kv[1]))
R["stage2_c"] = S2
R["stage2_rho"] = {
    "rho_c_vs_15axes_tr2024": RHO15_C, "rho_b_vs_15axes_tr2024": RHO15_B,
    "rho_c_vs_exp080_9axes": RHO9_C, "rho_b_vs_exp080_9axes": RHO9_B,
    "c_max_abs_rho_axis": _mx[0], "c_max_abs_rho": abs(_mx[1]),
    "c_max_abs_rho_signed": _mx[1],
    "note": ("15축 δ 는 core078(tr2024 보정 프레임) · 9축은 exp_080 저장본. "
             "b 축과의 ρ 는 같은 프레임 δ 로 계산.")}
print(f"[ρ] c 축 최대 |ρ| {abs(_mx[1]):.4f} ({_mx[0]}) · c–b {GS['tr2024_cal']['rho_b_c']:+.4f}",
      flush=True)

# ══════════════════════════════════════════════════ 6. 정직면 (원값 + sd 맞춤)
import surface079 as S                                             # noqa: E402
yo = S.yo
LV0 = (S.W4["LGB"] * S._L["p_lgb"] + S.W4["CAT"] * S.lg(S.V["p_cat"])
       + S.W4["NN"] * S._L["p_nn"] + S.W4["LEAF"] * S._L["p_leaf"])
Z0 = (S.BO["w_out"] * ((1 - S.BO["w_ebm"]) * (S.BO["w_v6_inner"] * LV0
                       + (1 - S.BO["w_v6_inner"]) * S._L["p_q"])
                       + S.BO["w_ebm"] * S._L["p_ebm"]
                       + S.BO["beta_ins"] * S.V["off"])
      + S.BO["c_aa"] * S.L_aa + S.BO["c_T70"] * S.L_t70)
MU0, SD0 = float(Z0.mean()), float(Z0.std())


def z_pts(z):
    p = expit(z + S.rc(z))
    return pts(float(np.mean((p - yo) ** 2)))


def z_pts_alpha_free(z):
    r = minimize_scalar(lambda a: -z_pts(a * z), bounds=(0.3, 3.0),
                        method="bounded", options={"xatol": 1e-7})
    return float(-r.fun), float(r.x)


BASE = z_pts(Z0)
BASE_AF, A0 = z_pts_alpha_free(Z0)
DIRV = {"c_lg_wo": LGWO - MU_WO, "b_lg_mr": LGMR - MU_MR}
HS = {"base_pts": BASE, "base_alpha_free_pts": BASE_AF, "alpha_free_a_star": A0,
      "alpha_free_headroom_pts": BASE_AF - BASE,
      "🔴": ("lg_mr·lg_wo 는 ≤2024 학습 Q 셀에서 나온다 — 2024 행에는 표본 내다. "
             "정직면의 나머지 팔은 ≤2023 학습이므로 이 읽기는 위로 편향돼 있다. "
             "상한으로만 읽을 것.")}
for nm, v in DIRV.items():
    def raw(t):
        return z_pts(Z0 + t * v)

    def sdm(t):
        z = Z0 + t * v
        return z_pts((z - float(z.mean())) / float(z.std()) * SD0 + MU0)
    rr = minimize_scalar(lambda t: -raw(t), bounds=(-0.60, 0.60),
                         method="bounded", options={"xatol": 1e-7})
    rs = minimize_scalar(lambda t: -sdm(t), bounds=(-0.60, 0.60),
                         method="bounded", options={"xatol": 1e-7})
    t_raw, g_raw = float(rr.x), float(-rr.fun) - BASE
    t_sdm, g_sdm = float(rs.x), float(-rs.fun) - BASE
    af, a_at = z_pts_alpha_free(Z0 + t_raw * v)
    zz = Z0 + t_raw * v
    HS[nm] = {"raw_argmax_t": t_raw, "raw_gain_pts": g_raw,
              "sd_ratio_at_raw_argmax": float(zz.std() / SD0),
              "sd_matched_gain_at_raw_argmax_pts": sdm(t_raw) - BASE,
              "sd_matched_own_argmax_t": t_sdm, "sd_matched_own_gain_pts": g_sdm,
              "information_share": (g_sdm / g_raw) if g_raw else None,
              "alpha_free_gain_pts": af - BASE_AF, "alpha_free_a_star": a_at,
              "gain_at_shipped_b_-0.05" if nm == "b_lg_mr" else "gain_at_t_eq_0.05":
                  raw(-0.05 if nm == "b_lg_mr" else 0.05) - BASE,
              "grid_raw_delta": {f"{t:+.3f}": raw(t) - BASE
                                 for t in (-0.30, -0.20, -0.10, -0.05, -0.02, 0.0,
                                           0.02, 0.05, 0.10, 0.20, 0.30)},
              "grid_sd_matched_delta": {f"{t:+.3f}": sdm(t) - BASE
                                        for t in (-0.30, -0.20, -0.10, -0.05, -0.02,
                                                  0.0, 0.02, 0.05, 0.10, 0.20, 0.30)}}
    print(f"[정직면/{nm}] raw t {t_raw:+.5f} 이득 {g_raw:+.5f}점 | sd맞춤 own "
          f"t {t_sdm:+.5f} {g_sdm:+.5f}점 | sd맞춤@raw {HS[nm]['sd_matched_gain_at_raw_argmax_pts']:+.5f}",
          flush=True)
R["stage2_honest_surface"] = HS

# ══════════════════════════════════════════════════ 7. stage3 — 동시 해 식
S3 = {}
for fr in FRAMES:
    g = GS[fr]
    Lb = S1[f"{fr}/secant_G"]["L"]
    S3[fr] = {"L_b": Lb, "G_bb": g["G_bb"], "G_cc": g["G_cc"], "G_bc": g["G_bc"],
              "b_only_conditional_u_star": Lb / g["G_bb"],
              "b_only_conditional_gain_pts": K * Lb ** 2 / g["G_bb"],
              "joint_formula": ("u* = G^-1 L 로 (b,c) 동시.  L_c 는 아직 없다 — "
                                "c 프로브를 사서 g_c 를 재야 정해진다."),
              "joint_gain_given_Lc": ("gain = K·(Lb,Lc)ᵀ G⁻¹ (Lb,Lc);  "
                                      f"G = [[{g['G_bb']:.6e},{g['G_bc']:.6e}],"
                                      f"[{g['G_bc']:.6e},{g['G_cc']:.6e}]]"),
              "b_star_if_c_lands_at_u_c": ("b* = (G_cc·L_b − G_bc·L_c)/(G_bb·G_cc − G_bc²)")}
    det = g["G_bb"] * g["G_cc"] - g["G_bc"] ** 2
    S3[fr]["G_det"] = det
    S3[fr]["G_condition_2x2"] = float(np.linalg.cond(
        np.array([[g["G_bb"], g["G_bc"]], [g["G_bc"], g["G_cc"]]])))
    S3[fr]["extra_gain_over_b_only_per_unit_Lc"] = (
        "Δgain = K·(L_c − G_bc·L_b/G_bb)² / (G_cc − G_bc²/G_bb)")
    S3[fr]["c_effective_G_after_orthogonalising_b"] = g["G_cc"] - g["G_bc"] ** 2 / g["G_bb"]
R["stage3"] = S3
R["stage3_🔴"] = "c 의 L 은 없다 — 착륙 해에 넣지 않는다.  진폭 설계까지만."

R["wall_sec"] = time.time() - T0
R["runs"] = {os.path.basename(p): sha(f"{OUT}/runs/{p}")
             for p in sorted(os.listdir(f"{OUT}/runs"))}
R["prep"] = json.load(open(f"{OUT}/out/s0_prep.json", encoding="utf-8"))
json.dump(R, open(f"{OUT}/out/metrics.json", "w"), indent=1, ensure_ascii=False)
print(f"\n기록 {OUT}/out/metrics.json · {R['wall_sec']:.1f}s", flush=True)
