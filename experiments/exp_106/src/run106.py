"""exp_106 — 현 팔 구성에서 앙상블 가중 7축 전면 재적합. 재학습 0.

단계
  s0  배포점 재현 (항등 앵커) · 딥 팔 sha 대조
  s1  축별 1자유도 조건부 스윕 (raw / sd 맞춤 병기)
  s2  전축 동시 적합 — 2차 국소해(고윳값 절단) + 면 위 직접 최적화
  s3  sd 맞춤 · 부호 뒤집힘
  s4  잡음 — 개수 맞춤 귀무 2종 x 200 + 적합-간 sd (판본 8개)
  s5  재판독 — exp_095 (w, v) 격자를 결함면·고친 면 양쪽에서
  s6  직교성 — 축 간 rho, exp_078 15축 기록과 대조
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                                   # noqa: E402
from scipy.optimize import minimize                                  # noqa: E402
import surf106 as S                                                  # noqa: E402

t0 = time.time()
OUT = "/home/<user>/SungjinPark/exp106/out"
os.makedirs(OUT, exist_ok=True)
R = {"exp_id": "exp_106",
     "title": "현 팔 구성에서 앙상블 가중 7축 전면 재적합 (고친 정직면)",
     "K_pts_per_brier": S.K, "n_rows": int(S.N),
     "재학습": 0, "GPU": "미사용",
     "prereg": {"rule": "(결합 argmax - 배포점) >= +2.5 AND sd 맞춤 후 >= +1.5 AND t >= 2.0",
                "delta": 2.5, "delta_sd_matched": 1.5, "t_min": 2.0,
                "전이계수_고지": "팀 원장 가중 축 0.247 -> 정직면 +2.5 = LB 약 +0.62"}}


def log(*a):
    print("[%6.1fs]" % (time.time() - t0), *a, flush=True)


def _jd(o):
    import numpy as _np
    if isinstance(o, (_np.bool_,)):
        return bool(o)
    if isinstance(o, (_np.integer,)):
        return int(o)
    if isinstance(o, (_np.floating,)):
        return float(o)
    if isinstance(o, _np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def dump():
    json.dump(R, open(f"{OUT}/metrics106.json", "w"), ensure_ascii=False,
              indent=1, default=_jd)


# ══════════════════════════════════════════════ s0 — 배포점 재현 ════════
log("s0 배포점 재현")
th0 = dict(S.THETA0)
th_nodeep = dict(th0, w_deep=0.0, v_out=0.0)
z_nodeep = S.zvec(th_nodeep)
rc0 = S.score_rc(z_nodeep)
z0 = S.zvec(th0)
DEPLOY = S.score(z0, th0["alpha"], th0["beta_skew"])
BASE_ABD = S.score(z_nodeep, th0["alpha"], th0["beta_skew"])

# 딥 팔 w 격자 — exp_097 기록(S145) 재현 확인
gw = {}
for w in (0.20, 0.40, 0.43):
    gw[f"{w:.2f}"] = S.sc_theta(dict(th0, w_deep=w)) - BASE_ABD

R["s0_gates"] = {
    "G0_anchor_rc_w_deep0": {"what": "w_deep=0 · rc 관례 점수 == exp_097 이후 불변 앵커",
                             "got": rc0, "expect": S.BASE_EXPECT,
                             "abs_diff": abs(rc0 - S.BASE_EXPECT),
                             "PASS": abs(rc0 - S.BASE_EXPECT) < 1e-9},
    "G1_deep_sha": {"what": "딥 팔 z 벡터 sha256 == 적합 기록",
                    "n": len(S.SHA),
                    "PASS": all(v["MATCH"] for v in S.SHA.values()),
                    "detail": S.SHA},
    "G2_w040_matches_exp097": {"what": "고친 면 1632스텝 w=0.40 이득 == S145 기록 26.1514",
                               "got": gw["0.40"], "recorded": S.E097_GAIN_W040,
                               "abs_diff": abs(gw["0.40"] - S.E097_GAIN_W040),
                               "PASS": abs(gw["0.40"] - S.E097_GAIN_W040) < 5e-3},
    "G3_argmax_w_matches_exp097": {"what": "w 단축 argmax == S145 기록 (0.43, 26.2698)",
                                   "recorded": list(S.E097_ARGMAX)},
}
wgrid = [round(x, 4) for x in np.arange(0.0, 0.6001, 0.01)]
wcurve = [{"w": w, "gain": S.sc_theta(dict(th0, w_deep=w)) - BASE_ABD} for w in wgrid]
wbest = max(wcurve, key=lambda c: c["gain"])
R["s0_gates"]["G3_argmax_w_matches_exp097"].update(
    {"got": [wbest["w"], wbest["gain"]],
     "abs_diff_gain": abs(wbest["gain"] - S.E097_ARGMAX[1]),
     "PASS": (abs(wbest["w"] - S.E097_ARGMAX[0]) < 1e-9
              and abs(wbest["gain"] - S.E097_ARGMAX[1]) < 5e-3)})
# 치명 게이트는 항등 앵커(G0)와 sha(G1) 둘. G2·G3 은 장부 대조라 정보용으로 남긴다.
R["s0_gates"]["FATAL_PASS"] = bool(R["s0_gates"]["G0_anchor_rc_w_deep0"]["PASS"]
                                   and R["s0_gates"]["G1_deep_sha"]["PASS"])
R["s0_gates"]["ALL_PASS"] = all(v["PASS"] for k, v in R["s0_gates"].items()
                                if isinstance(v, dict) and "PASS" in v)
R["s0_deploy_point"] = {
    "theta_shipped": {k: th0[k] for k in
                      ["alpha", "beta_skew", "c_aa", "c_T70", "w_out",
                       "w_v6_inner", "w_ebm", "beta_ins", "w_deep", "v_out"]},
    "출처": "submit_CR_q12land.zip  blend.json + mate/model/blend.json (2026-08-24)",
    "score_abd_deploy": DEPLOY,
    "score_abd_w_deep0": BASE_ABD,
    "deep_gain_at_ship": DEPLOY - BASE_ABD,
    "score_rc_w_deep0": rc0,
    "gain_by_w": gw,
    "면이_안_담는_것": ["루트 _shift(축 오프셋 BI/BJ/BK 등)", "실패유형 오프셋",
                        "mate 자신의 d_adj", "AA·T70 은 3시드(배포 6시드)",
                        "딥 OOT 복제본은 1632스텝(배포 2160스텝의 에폭 맞춤)"]}
log("  rc anchor", rc0, "PASS", R["s0_gates"]["G0_anchor_rc_w_deep0"]["PASS"])
log("  deploy abd", DEPLOY, " deep gain@ship", DEPLOY - BASE_ABD)
log("  FATAL_PASS", R["s0_gates"]["FATAL_PASS"], " ALL_PASS", R["s0_gates"]["ALL_PASS"])
dump()
if not R["s0_gates"]["FATAL_PASS"]:
    log("🔴 배포점 재현(항등 앵커) 실패 — 여기서 멈춘다. 위의 모든 수치가 의미 없다.")
    R["STOPPED"] = "s0 항등 앵커 실패"
    dump()
    sys.exit(1)

# ══════════════════════════════════════════════ s1 — 1자유도 스윕 ═══════
log("s1 축별 1자유도 조건부 스윕")
SWEEP = {   # (lo, hi, n)  — 배포값을 반드시 포함하는 격자
    "alpha":       (0.80, 1.45, 131),
    "c_aa":        (0.00, 0.60, 121),
    "c_T70":       (-0.15, 0.30, 91),
    "w_out":       (0.40, 1.05, 131),
    "w_v6_inner":  (0.30, 1.00, 141),
    "w_ebm":       (-0.45, 0.25, 141),
    "w_deep":      (0.00, 0.75, 151),
}
R["s1_1dof"] = {}
for ax, (lo, hi, n) in SWEEP.items():
    grid = sorted(set([round(v, 6) for v in np.linspace(lo, hi, n)] + [round(th0[ax], 10)]))
    cur = [{"v": v, "score": S.sc_theta(dict(th0, **{ax: v}))} for v in grid]
    best = max(cur, key=lambda c: c["score"])
    zb = S.zvec(dict(th0, **{ax: best["v"]}))
    sdm = S.score(S.sd_match(zb, z0), th0["alpha"], th0["beta_skew"])
    R["s1_1dof"][ax] = {
        "위치": "루트" if ax in S.ROOT_AXES else "mate 내부",
        "shipped": th0[ax], "argmax": best["v"],
        "gain_pts": best["score"] - DEPLOY,
        "gain_sd_matched_pts": sdm - DEPLOY,
        "sd_ratio_at_argmax": float(zb.std() / z0.std()),
        "이동": best["v"] - th0[ax],
        "grid": [lo, hi, n],
        "curve": cur}
    log("  %-12s ship %+9.5f -> argmax %+9.5f   gain %+8.4f  (sd맞춤 %+8.4f)"
        % (ax, th0[ax], best["v"], best["score"] - DEPLOY, sdm - DEPLOY))
dump()

# ══════════════════════════════════════════════ s2 — 동시 적합 ═════════
log("s2 전축 동시 적합")
AX = S.AXES


def deltas_at(th, dz=None):
    """각 축의 u 벡터(= dp/dtheta | c고정 을 q 로 나눈 것)."""
    dz = S.Z_DEEP if dz is None else dz
    wi, we, wd, wo = th["w_v6_inner"], th["w_ebm"], th["w_deep"], th["w_out"]
    base_in = wi * S.LV + (1.0 - wi) * S.LQ
    zm = (1.0 - we) * base_in + we * S.LE + th["beta_ins"] * S.OFF
    zm_full = (1.0 - wd) * zm + wd * dz
    d = {"c_aa": S.L_aa, "c_T70": S.L_t70, "w_out": zm_full,
         "w_deep": wo * (dz - zm),
         "w_v6_inner": wo * (1.0 - wd) * (1.0 - we) * (S.LV - S.LQ),
         "w_ebm": wo * (1.0 - wd) * (S.LE - base_in),
         "v_out": dz}
    return d


z_th0 = S.zvec(th0)
DL = deltas_at(th0)
US = [S.raw_u(z_th0, th0["alpha"], th0["beta_skew"], a, DL.get(a)) for a in AX]
L, G = S.quad_LG(z_th0, th0["alpha"], th0["beta_skew"], US)
tq, gain_q, rank_q, cond_q, eig = S.solve_trunc(L, G)
th_q = dict(th0)
for a, dv in zip(AX, tq):
    th_q[a] = th0[a] + float(dv)
sc_q = S.sc_theta(th_q)
R["s2_quadratic"] = {
    "axes": AX, "L": L.tolist(), "G": G.tolist(),
    "eigenvalues": eig.tolist(),
    "G_condition": cond_q, "numeric_rank_rcond1e-10": rank_q,
    "solver": "고윳값 절단 사영 (rcond 1e-10) — np.linalg.solve 금지(S44)",
    "step": {a: float(v) for a, v in zip(AX, tq)},
    "predicted_gain_pts": gain_q,
    "theta_at_step": {a: th_q[a] for a in AX},
    "actual_score_at_step": sc_q,
    "actual_gain_pts": sc_q - DEPLOY,
    "note": "2차 국소 모형의 예측과 면 위 실측이 다르면 걸음이 선형 구간을 넘은 것이다."}
log("  quad: cond %.3e rank %d  pred %+8.4f  actual %+8.4f"
    % (cond_q, rank_q, gain_q, sc_q - DEPLOY))

BOUNDS = {"alpha": (0.5, 2.0), "c_aa": (-0.3, 1.0), "c_T70": (-0.3, 1.0),
          "w_out": (0.2, 1.3), "w_v6_inner": (0.0, 1.0), "w_ebm": (-0.6, 0.6),
          "w_deep": (0.0, 0.95)}


def direct_argmax(free, th_base, dz=None, l_aa=None, l_t70=None, constrained=False,
                  x0=None):
    lo = np.array([BOUNDS[a][0] for a in free])
    hi = np.array([BOUNDS[a][1] for a in free])
    x_start = np.array([th_base[a] for a in free]) if x0 is None else np.asarray(x0)

    def build(x):
        th = dict(th_base)
        for a, v in zip(free, x):
            th[a] = float(v)
        if constrained:
            th["w_out"] = 1.0 - th["c_aa"] - th["c_T70"]
        return th

    def neg(x):
        if np.any(x < lo - 1e-12) or np.any(x > hi + 1e-12):
            return 1e9
        return -S.sc_theta(build(x), dz=dz, l_aa=l_aa, l_t70=l_t70)

    res = minimize(neg, x_start, method="Powell",
                   bounds=list(zip(lo, hi)),
                   options={"xtol": 1e-7, "ftol": 1e-9, "maxfev": 40000})
    th = build(res.x)
    return th, -float(res.fun), int(res.nfev)


th_star, sc_star, nfev = direct_argmax(AX, th0)
zs = S.zvec(th_star)
sdm_star = S.score(S.sd_match(zs, z0), th0["alpha"], th0["beta_skew"])
R["s2_direct"] = {
    "method": "Powell (면 위 직접 최적화, 각 지점에서 d_adj 재유도)",
    "nfev": nfev,
    "theta_star": {a: th_star[a] for a in AX},
    "이동": {a: th_star[a] - th0[a] for a in AX},
    "score": sc_star, "gain_pts": sc_star - DEPLOY,
    "gain_sd_matched_pts": sdm_star - DEPLOY,
    "sd_ratio": float(zs.std() / z0.std())}
log("  direct: gain %+8.4f  (sd맞춤 %+8.4f)  nfev %d"
    % (sc_star - DEPLOY, sdm_star - DEPLOY, nfev))

# 출하 가능형: script.py:391 이 c_aa + c_S2 + c_S1 + c_T70 == 1 - w_v6 를 assert 한다
FREE_C = ["alpha", "c_aa", "c_T70", "w_v6_inner", "w_ebm", "w_deep"]
th_c, sc_c, nfev_c = direct_argmax(FREE_C, th0, constrained=True)
zc = S.zvec(th_c)
sdm_c = S.score(S.sd_match(zc, z0), th0["alpha"], th0["beta_skew"])
R["s2_direct_shippable"] = {
    "제약": "w_out = 1 - c_aa - c_T70  (script.py:391 assert)",
    "free_axes": FREE_C, "nfev": nfev_c,
    "theta_star": {a: th_c[a] for a in S.AXES},
    "이동": {a: th_c[a] - th0[a] for a in S.AXES},
    "score": sc_c, "gain_pts": sc_c - DEPLOY,
    "gain_sd_matched_pts": sdm_c - DEPLOY}
log("  shippable: gain %+8.4f  (sd맞춤 %+8.4f)" % (sc_c - DEPLOY, sdm_c - DEPLOY))

# 8축(바깥 딥 슬롯 v 포함) — 부수 확인. 출하 안 된 자유도라 주 판정은 아니다.
BOUNDS["v_out"] = (-0.2, 0.6)
th_8, sc_8, _ = direct_argmax(AX + ["v_out"], th0)
R["s2_direct_plus_v_out"] = {
    "note": "v_out 은 출하 패키지에 없는 자유도(exp_095 축). 부수 확인용.",
    "theta_star": {a: th_8[a] for a in AX + ["v_out"]},
    "gain_pts": sc_8 - DEPLOY}
log("  +v_out: gain %+8.4f  v*=%.4f" % (sc_8 - DEPLOY, th_8["v_out"]))

# 짝지은 투수 블록 부트스트랩 (배포점 대비)
p_dep = S.prob(z0, th0["alpha"], th0["beta_skew"])
p_star = S.prob(zs, th_star["alpha"], th_star["beta_skew"])
p_ship = S.prob(zc, th_c["alpha"], th_c["beta_skew"])
R["s2_boot_vs_deploy"] = {
    "direct_7axis": S.boot_t((p_star - S.y) ** 2 - (p_dep - S.y) ** 2),
    "shippable_6dof": S.boot_t((p_ship - S.y) ** 2 - (p_dep - S.y) ** 2)}
log("  boot t (7축) %.3f   (출하형) %.3f"
    % (R["s2_boot_vs_deploy"]["direct_7axis"]["t"] or 0,
       R["s2_boot_vs_deploy"]["shippable_6dof"]["t"] or 0))
dump()

# ══════════════════════════════════════════════ s3 — 척도 분해 ═════════
log("s3 척도 분해")
# alpha 만 자유롭게 둔 최적(순수 척도 축)
a_grid = np.linspace(0.80, 1.45, 261)
a_cur = [{"alpha": float(a), "score": S.sc_theta(dict(th0, alpha=float(a)))}
         for a in a_grid]
a_best = max(a_cur, key=lambda c: c["score"])
R["s3_scale"] = {
    "설명": "S25·S33 — 가중 축 이득의 상당 부분은 정보가 아니라 척도다. "
            "척도분은 alpha 로도 살 수 있고 그 통로는 exp_078 잔여 +0.2546 으로 거의 닫혔다.",
    "alpha_only_argmax": a_best["alpha"],
    "alpha_only_gain_pts": a_best["score"] - DEPLOY,
    "joint_gain_raw": sc_star - DEPLOY,
    "joint_gain_sd_matched": sdm_star - DEPLOY,
    "척도_몫_비율": (1.0 - (sdm_star - DEPLOY) / (sc_star - DEPLOY)
                     if abs(sc_star - DEPLOY) > 1e-12 else None),
    "부호_뒤집힘_축": [a for a, e in R["s1_1dof"].items()
                        if e["gain_pts"] * e["gain_sd_matched_pts"] < 0],
    "1dof_raw_vs_sdmatched": {a: {"raw": e["gain_pts"],
                                  "sd_matched": e["gain_sd_matched_pts"]}
                              for a, e in R["s1_1dof"].items()}}
log("  alpha 단독 argmax %.5f gain %+8.4f | 척도 몫 %s"
    % (a_best["alpha"], a_best["score"] - DEPLOY,
       R["s3_scale"]["척도_몫_비율"]))
dump()

# ══════════════════════════════════════════════ s4 — 잡음 ══════════════
log("s4 잡음: 개수 맞춤 귀무 2종 x 200")
rng = np.random.default_rng(20261026)
n_ax = len(AX)
rms = np.array([float(np.sqrt((u * u).mean())) for u in US])
real_gain_quad = gain_q


def null_gain(us):
    Ln, Gn = S.quad_LG(z_th0, th0["alpha"], th0["beta_skew"], us)
    return S.solve_trunc(Ln, Gn)[1]


NDRAW = 200
null1, null2 = [], []
for i in range(NDRAW):
    u1 = []
    for j in range(n_ax):
        v = rng.standard_normal(S.N)
        v -= v.mean()
        u1.append(v / float(np.sqrt((v * v).mean())) * rms[j])
    null1.append(null_gain(u1))
    u2 = [US[j][rng.permutation(S.N)] for j in range(n_ax)]
    null2.append(null_gain(u2))
    if (i + 1) % 50 == 0:
        log("   귀무 %d/%d" % (i + 1, NDRAW))
null1 = np.array(null1)
null2 = np.array(null2)


def nsum(a, real):
    return {"n": int(len(a)), "mean": float(a.mean()), "sd": float(a.std(ddof=1)),
            "per_dof": float(a.mean() / n_ax),
            "p05": float(np.percentile(a, 5)), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "max": float(a.max()),
            "실측_초과_비율": float((a >= real).mean()),
            "실측_minus_귀무평균": float(real - a.mean())}


R["s4_null"] = {
    "설계": "축 개수(7) 맞춤. 같은 2차 국소 절차(고윳값 절단)로 최대이득을 계산.",
    "이론_자유도당": 0.390759,
    "실측_2차예측_7축": real_gain_quad,
    "실측_직접_7축": sc_star - DEPLOY,
    "null1_gaussian_rms맞춤": nsum(null1, real_gain_quad),
    "null2_행순열_실축": nsum(null2, real_gain_quad),
    "🔴판정_뒤집힘": None}
R["s4_null"]["🔴판정_뒤집힘"] = bool(
    (real_gain_quad >= np.percentile(null1, 95))
    != (real_gain_quad >= np.percentile(null2, 95)))
log("  null1 mean %.4f sd %.4f p95 %.4f | null2 mean %.4f p95 %.4f | real %.4f"
    % (null1.mean(), null1.std(ddof=1), np.percentile(null1, 95),
       null2.mean(), np.percentile(null2, 95), real_gain_quad))
dump()

log("s4b 적합-간 sd (판본 8개)")
variants = {}
for s in S.S5:
    others = [S.DEEP_FIX_EACH[k] for k in S.S5 if k != s]
    variants[f"deepLOO_drop{s}"] = dict(dz=np.mean(others, 0))
for s in S.AA:
    variants[f"aaLOO_drop{s}"] = dict(
        l_aa=np.mean([S.AA_EACH[k] for k in S.AA if k != s], 0),
        l_t70=np.mean([S.T70_EACH[k] for k in S.AA if k != s], 0))
vres = {}
for name, kw in variants.items():
    dep_v = S.sc_theta(th0, **kw)
    th_v, sc_v, _ = direct_argmax(AX, th0, dz=kw.get("dz"),
                                  l_aa=kw.get("l_aa"), l_t70=kw.get("l_t70"),
                                  x0=[th_star[a] for a in AX])
    vres[name] = {"deploy": dep_v, "argmax_gain": sc_v - dep_v,
                  "theta": {a: th_v[a] for a in AX}}
    log("  %-18s gain %+8.4f" % (name, sc_v - dep_v))
gains = np.array([v["argmax_gain"] for v in vres.values()])
fit_sd = float(gains.std(ddof=1))
boot_sd = R["s2_boot_vs_deploy"]["direct_7axis"]["boot_se_pts"]
R["s4b_fit_to_fit"] = {
    "설계": "기댓값이 같아야 하는 판본 8개(딥 5시드 LOO · AA/T70 3시드 LOO). "
            "S178 — 부트스트랩은 시드를 고정하므로 이 성분을 못 본다.",
    "판본": vres,
    "gains": gains.tolist(),
    "fit_sd_pts": fit_sd, "boot_sd_pts": boot_sd,
    "보고_1sigma_pts": float(np.sqrt(fit_sd ** 2 + boot_sd ** 2)),
    "t_보고_1sigma": float((sc_star - DEPLOY) / np.sqrt(fit_sd ** 2 + boot_sd ** 2))}
log("  fit_sd %.4f  boot_sd %.4f  1sigma %.4f  t %.3f"
    % (fit_sd, boot_sd, R["s4b_fit_to_fit"]["보고_1sigma_pts"],
       R["s4b_fit_to_fit"]["t_보고_1sigma"]))
dump()

# ══════════════════════════════════════════════ s5 — 재판독 ════════════
log("s5 재판독 exp_095 (w, v)")
WG = [round(x, 4) for x in np.arange(0.0, 0.6001, 0.01)]
VG = [round(x, 4) for x in np.arange(0.0, 0.3001, 0.01)]
reread = {}
for surf, dz in (("결함면_K1632", S.Z_DEEP_DEF), ("고친면_F1632", S.Z_DEEP)):
    base_s = S.sc_theta(dict(th0, w_deep=0.0, v_out=0.0), dz=dz)
    grid = []
    for w in WG:
        for v in VG:
            g = S.sc_theta(dict(th0, w_deep=w, v_out=v), dz=dz) - base_s
            grid.append((w, v, g))
    b2 = max(grid, key=lambda t: t[2])
    b1 = max([g for g in grid if g[1] == 0.0], key=lambda t: t[2])
    jk = []
    for s in S.S5:
        dzj = np.mean([(S.DEEP_DEF_EACH if surf.startswith("결함") else S.DEEP_FIX_EACH)[k]
                       for k in S.S5 if k != s], 0)
        bj = S.sc_theta(dict(th0, w_deep=0.0, v_out=0.0), dz=dzj)
        gj = [(w, v, S.sc_theta(dict(th0, w_deep=w, v_out=v), dz=dzj) - bj)
              for w in WG for v in (0.0, 0.02, 0.04, 0.06, 0.10, 0.15, 0.20)]
        jk.append(max(gj, key=lambda t: t[2]))
    reread[surf] = {
        "baseline_w0v0": base_s,
        "argmax_2dof": {"w": b2[0], "v": b2[1], "gain": b2[2]},
        "argmax_1dof_v0": {"w": b1[0], "gain": b1[2]},
        "2dof_minus_1dof": b2[2] - b1[2],
        "v_star_is_zero": bool(b2[1] == 0.0),
        "jackknife_v_star": [t[1] for t in jk],
        "jackknife_all_zero": all(t[1] == 0.0 for t in jk),
        "gain_at_ship_w0.40_v0": [g for g in grid
                                  if abs(g[0] - 0.40) < 1e-9 and g[1] == 0.0][0][2]}
    log("  %s: 1자유도 argmax w=%.2f %+8.4f | 2자유도 (w=%.2f,v=%.2f) %+8.4f | v*=0? %s"
        % (surf, b1[0], b1[2], b2[0], b2[1], b2[2], b2[1] == 0.0))
R["s5_reread_exp095"] = {
    "질문": "exp_095 의 v=0 결론이 고친 면에서도 유지되는가 (S134 재판독 대상)",
    "exp_095_기록": {"argmax_2dof": [0.36, 0.0], "note": "결함면. 순위 인용 금지(S163)."},
    "surfaces": reread,
    "판정": ("유지" if reread["고친면_F1632"]["v_star_is_zero"] else "뒤집힘")}
log("  판정:", R["s5_reread_exp095"]["판정"])
dump()

# ══════════════════════════════════════════════ s6 — 직교성 ════════════
log("s6 직교성")
E = np.stack([u - u.mean() for u in US])
C = np.corrcoef(E)
pairs = []
for i in range(n_ax):
    for j in range(i + 1, n_ax):
        pairs.append([AX[i], AX[j], float(C[i, j])])
pairs.sort(key=lambda p: -abs(p[2]))
e078 = json.load(open("/home/<user>/SungjinPark/exp106/exp078_stage1.json")) \
    if os.path.exists("/home/<user>/SungjinPark/exp106/exp078_stage1.json") else None
R["s6_orthogonality"] = {
    "rho_matrix": C.tolist(), "axes": AX,
    "pairs_sorted_by_abs_rho": pairs,
    "G_condition": cond_q,
    "exp_078_15축과의_관계": (
        "🔴 우리 7축 중 6개(alpha·c_aa·c_T70·w_out·w_v6_inner·w_ebm)는 "
        "exp_078 15축 기저에 이미 있는 **같은 축**이다. 새 자유도는 `w_deep` 하나뿐이고, "
        "그것은 exp_078(08-21) 당시 딥 팔이 없어서 존재하지 않았다."),
    "exp_078_기록_rho": {"alpha|w_out": 0.9960693576044906,
                         "alpha|c_aa": 0.9823176290784889,
                         "c_aa|w_out": 0.9686829737529024,
                         "c_aa|c_T70": 0.9706851797533724,
                         "c_T70|w_out": 0.9562480628856523,
                         "출처": "exp_078 metrics.json near_collinear_pairs (calBX 프레임)"},
    "w_deep_max_abs_rho": max(abs(p[2]) for p in pairs if "w_deep" in (p[0], p[1]))}
log("  top pairs:", pairs[:3])
dump()

# ══════════════════════════════════════════════ 판정 ═══════════════════
sig = R["s4b_fit_to_fit"]["보고_1sigma_pts"]
g_raw = sc_star - DEPLOY
g_sd = sdm_star - DEPLOY
t_boot = R["s2_boot_vs_deploy"]["direct_7axis"]["t"]
R["판정"] = {
    "사전등록_문턱": R["prereg"]["rule"],
    "결합_argmax_minus_배포점": g_raw,
    "sd_맞춤": g_sd,
    "t_bootstrap_only": t_boot,
    "t_보고_1sigma(boot+fit)": R["s4b_fit_to_fit"]["t_보고_1sigma"],
    "cond1_delta>=2.5": bool(g_raw >= 2.5),
    "cond2_sd맞춤>=1.5": bool(g_sd >= 1.5),
    "cond3_t>=2.0": bool(t_boot is not None and t_boot >= 2.0),
    "귀무_대비": {"null1_p95": float(np.percentile(null1, 95)),
                  "null2_p95": float(np.percentile(null2, 95)),
                  "실측_2차": real_gain_quad},
    "PASS": None}
R["판정"]["PASS"] = bool(R["판정"]["cond1_delta>=2.5"] and R["판정"]["cond2_sd맞춤>=1.5"]
                         and R["판정"]["cond3_t>=2.0"])
R["실행시간_분"] = (time.time() - t0) / 60.0
dump()
log("판정", json.dumps(R["판정"], ensure_ascii=False, default=str))
log("done", R["실행시간_분"], "min")
