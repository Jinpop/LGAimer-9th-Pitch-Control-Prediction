"""exp_078 stage3 보충 + stage4 — 학습 0건.

stage3b  ① 축별 잔여 재고(단일축 조건부 꼭지점)  ② 능형 스윕  ③ 공선축 정리 해
stage4   파트너 v6 내부 4팔 가중 신설 3축 — **진단만**.  L 이 없으므로 착륙 금지.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "6"
import io
import json
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit

sys.path.insert(0, "/home/<user>/SungjinPark/exp078/src")
import core078 as C                                              # noqa: E402

H = C.H
OUT = C.OUT
K = C.K
GM = C.GM
IDX = C.IDX
AXES = C.AXES
BO = C.BO
ROWS = C.ROWS
T1 = time.time()
R = {"exp_id": "exp_078", "part": "stage3b+stage4",
     "core_wall_min": (time.time() - C.T0) / 60.0}

# ══════════════════════════════════════════════════════════════════════
# stage3b-① 축별 잔여 재고 — 단일축 이동인 점만
# ══════════════════════════════════════════════════════════════════════
INV = {}
for r in ROWS:
    solo = [a for a in r["v"] if a != "d_mean"]
    if len(solo) != 1:
        continue
    a = solo[0]
    L_ax = r["L_at_BO"] / r["v"][a]
    G_jj = float(GM[IDX[a], IDX[a]])
    INV.setdefault(a, {})[r["child"]] = {
        "grade": r["grade"], "parent": r["parent"],
        "L_axis_at_parent": r["L_at_parent"] / r["v"][a],
        "L_axis_at_BO": L_ax, "G_jj_at_BO": G_jj,
        "conditional_vertex_u": L_ax / G_jj,
        "conditional_vertex_gain_pts": float(K * L_ax ** 2 / G_jj),
        "delta_cos_parent_BO": r["delta_cos_parent_BO"],
        "delta_norm_ratio": r["delta_norm_ratio"],
        "unknown_term_bound_pts": r["unknown_term_bound_pts"]}
R["per_axis_inventory"] = INV
print("\n=== stage3b① 축별 잔여 재고 (단일축 점) ===", flush=True)
for a, d in INV.items():
    for ch, v in d.items():
        print(f"  {a:<14} [{ch}·{v['grade']}] L(BO) {v['L_axis_at_BO']:+.4e}  "
              f"u* {v['conditional_vertex_u']:+.6f}  이득 "
              f"{v['conditional_vertex_gain_pts']:+.5f}점  cos "
              f"{v['delta_cos_parent_BO']:.6f}  미지항 상한 "
              f"{v['unknown_term_bound_pts']:.3f}점", flush=True)
R["per_axis_inventory_note"] = (
    "unknown_term_bound_pts 는 2K·rms(δ_BO − δ_parent)·rms(y − p_BO) — 전이 "
    "공식이 무시하는 항 mean(Δδ·r0) 의 코시-슈바르츠 상한이다.  이 상한이 "
    "조건부 이득보다 크면 그 축의 재고는 부호조차 결정되지 않는다.")

# ══════════════════════════════════════════════════════════════════════
# stage3b-② 능형 스윕 — A+B 합동해가 왜 터지는지
# ══════════════════════════════════════════════════════════════════════
def recL(rows):
    A = np.array([[r["v"].get(a, 0.0) for a in AXES] for r in rows])
    b = np.array([r["L_at_BO"] for r in rows])
    Lv, _, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    return Lv, [AXES[j] for j in range(len(AXES)) if np.abs(A[:, j]).max() > 0]


ROWS_AB = [r for r in ROWS if r["grade"] in ("A", "B")]
ROWS_A = [r for r in ROWS if r["grade"] == "A"]
BND = {"alpha": (0.5, 2.0), "beta_skew": (-0.5, 0.5), "c_aa": (-0.2, 0.8),
       "c_S2": (-0.5, 0.5), "c_S1": (-0.5, 0.5), "c_T70": (-1.0, 0.5),
       "w_out": (0.3, 1.5), "w_v6_inner": (0.0, 1.0), "w_ebm": (-0.5, 0.5),
       "beta_ins": (-0.2, 0.3), "off_BI": (-0.1, 0.1), "off_BJ": (-0.1, 0.1),
       "off_BK": (-0.1, 0.1), "seg_spread_BH": (-1.0, 1.0), "d_mean": (-0.5, 0.5)}
SPEC12 = C.SPEC12


def ridge_solve(Lv, axset, lam_rel):
    ai = [IDX[a] for a in axset]
    Ls, Gs = Lv[ai], GM[np.ix_(ai, ai)]
    lam = lam_rel * float(np.trace(Gs) / len(axset))
    u = np.linalg.solve(Gs + np.eye(len(axset)) * lam, Ls)
    gain = float(K * (2 * u @ Ls - u @ Gs @ u))
    oob = {a: float(BO[a] + u[i]) for i, a in enumerate(axset)
           if not (BND[a][0] <= BO[a] + u[i] <= BND[a][1])}
    return {"lambda_rel": lam_rel, "lambda_abs": lam,
            "predicted_gain_pts": gain,
            "max_abs_u": float(np.abs(u).max()),
            "u_star": {a: float(u[i]) for i, a in enumerate(axset)},
            "out_of_range": oob, "in_range": not oob}


SWEEP = {}
for nm, rws in (("A", ROWS_A), ("AB", ROWS_AB)):
    Lv, cov = recL(rws)
    use = [a for a in SPEC12 if a in cov]
    SWEEP[nm] = {"axes": use, "points": [r["child"] for r in rws],
                 "sweep": [ridge_solve(Lv, use, x) for x in
                           (0.0, 1e-6, 1e-4, 1e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0)]}
    print(f"\n=== stage3b② 능형 스윕 [{nm}, 축 {len(use)}] ===", flush=True)
    for s in SWEEP[nm]["sweep"]:
        print(f"  λ_rel {s['lambda_rel']:<8g} 이득 {s['predicted_gain_pts']:+12.5f}점 "
              f"max|u*| {s['max_abs_u']:9.4f}  범위내 {s['in_range']}", flush=True)
R["ridge_sweep"] = SWEEP

# ══════════════════════════════════════════════════════════════════════
# stage3b-③ 공선축 정리 — |ρ| > 0.99 쌍에서 하나씩 뺀다
# ══════════════════════════════════════════════════════════════════════
CORR = GM / np.sqrt(np.outer(np.diag(GM), np.diag(GM)))
DROP = ["w_out", "off_BI", "c_S2", "c_S1"]      # alpha·beta_ins·c_T70 과 중복
PRUNED = [a for a in SPEC12 if a not in DROP]
PR = {}
for nm, rws in (("A", ROWS_A), ("AB", ROWS_AB)):
    Lv, cov = recL(rws)
    use = [a for a in PRUNED if a in cov]
    if not use:
        continue
    ai = [IDX[a] for a in use]
    Gs, Ls = GM[np.ix_(ai, ai)], Lv[ai]
    ev = np.linalg.eigvalsh(Gs)
    u = np.linalg.solve(Gs, Ls)
    gain = float(K * (2 * u @ Ls - u @ Gs @ u))
    PR[nm] = {"axes": use, "dropped_collinear": DROP,
              "G_condition": float(ev.max() / ev.min()),
              "u_star": {a: float(u[i]) for i, a in enumerate(use)},
              "new_constants": {a: float(BO[a] + u[i]) for i, a in enumerate(use)},
              "predicted_gain_pts": gain,
              "out_of_range": {a: float(BO[a] + u[i]) for i, a in enumerate(use)
                               if not (BND[a][0] <= BO[a] + u[i] <= BND[a][1])}}
    print(f"\n=== stage3b③ 공선정리 [{nm}] 축 {use} 조건수 "
          f"{PR[nm]['G_condition']:.3e} 이득 {gain:+.5f}점", flush=True)
R["pruned_solve"] = PR
R["collinear_pairs_abs_corr_gt_0.99"] = [
    (AXES[i], AXES[j], float(CORR[i, j])) for i in range(len(AXES))
    for j in range(i + 1, len(AXES)) if abs(CORR[i, j]) > 0.99]

# ══════════════════════════════════════════════════════════════════════
# stage4 — 파트너 v6 내부 4팔 가중 신설 3축 (진단만)
# ══════════════════════════════════════════════════════════════════════
print("\n=== stage4 신설축 진단 ===", flush=True)
W4 = {"LGB": 0.05, "CAT": 0.68, "NN": 0.15, "LEAF": 0.12}
PL = np.load(f"{H}/exp078/work/tr2024_v6_pl.npy")
PC = np.load(f"{H}/exp078/work/tr2024_v6_pc.npy")
PN = np.load(f"{H}/exp078/work/tr2024_v6_pn.npy")
PF = np.load(f"{H}/exp078/work/tr2024_v6_pf.npy")
LG_RE = W4["LGB"] * PL + W4["CAT"] * PC + W4["NN"] * PN + W4["LEAF"] * PF
R["stage4_capture_check"] = {
    "max_abs_diff_vs_arms_lg_p_v6": float(np.abs(LG_RE - C.LGV6).max()),
    "bitwise_equal": bool(np.array_equal(LG_RE, C.LGV6)),
    "arm_logit_ranges": {n: [float(v.min()), float(v.max())]
                         for n, v in (("pl", PL), ("pc", PC), ("pn", PN),
                                      ("pf", PF))},
    "arm_pairwise_corr": {f"{a}-{b}": float(np.corrcoef(x, z)[0, 1])
                          for (a, x), (b, z) in
                          [(("pl", PL), ("pc", PC)), (("pl", PL), ("pn", PN)),
                           (("pl", PL), ("pf", PF)), (("pc", PC), ("pn", PN)),
                           (("pc", PC), ("pf", PF)), (("pn", PN), ("pf", PF))]}}
print(f"  포획 재구성 |Δ| vs arms/lg_p_v6 = "
      f"{R['stage4_capture_check']['max_abs_diff_vs_arms_lg_p_v6']:.3e}", flush=True)

NEW = {"v6_CAT_vs_NN": PC - PN, "v6_CAT_vs_LEAF": PC - PF,
       "v6_CAT_vs_LGB": PC - PL}


def build_t(name, t, base=None):
    """LGV6 를 t 만큼 옮긴 뒤 출하 산술 전체를 통과시킨다 (CSV 왕복 포함)."""
    th = dict(BO if base is None else base)
    lgv6 = C.LGV6 + t * NEW[name]
    w_in, we, be = th["w_v6_inner"], th["w_ebm"], th["beta_ins"]
    bs = w_in * lgv6 + (1.0 - w_in) * C.LGQ
    z = (1.0 - we) * bs + we * C.PEBM
    z = z + be * C.OFFR
    p = expit(z + C.D_INNER)
    b = io.StringIO()
    pd.DataFrame({"control_success": p}).to_csv(b, index=False)
    b.seek(0)
    p = pd.read_csv(b)["control_success"].to_numpy(np.float64)
    lgp = C.lg(p)
    c = np.array([th["c_aa"], th["c_S2"], th["c_S1"], th["c_T70"]])
    zz = th["w_out"] * lgp + c @ C.ARMS18
    zp = (zz + th["off_BI"] * C.DIRS["off_BI"] + th["off_BJ"] * C.DIRS["off_BJ"]
          + th["off_BK"] * C.DIRS["off_BK"])
    f = th["alpha"] * zp + th["beta_skew"] * zp * zp
    return expit(f + C.anc(f, C.T_REF))


chk0 = build_t("v6_CAT_vs_NN", 0.0)
R["stage4_identity_at_t0"] = float(np.abs(chk0 - C.p_BO).max())
assert R["stage4_identity_at_t0"] < 1e-15, "t=0 항등 실패"
H4 = 0.002
S4 = {}
r0rms = float(np.sqrt(np.mean((C.y - C.p_BO) ** 2)))
for nm in NEW:
    dj = (build_t(nm, H4) - build_t(nm, -H4)) / (2 * H4)
    Gjj = float(np.mean(dj ** 2))
    rho = {a: float(np.corrcoef(dj, C.DELTA[IDX[a]])[0, 1]) for a in AXES}
    # LB 1점 프로브 진폭 설계
    def kq(u):
        return float(K * np.mean((build_t(nm, u) - C.p_BO) ** 2))

    def nonlin(u):
        ds = (build_t(nm, u) - C.p_BO) / u
        return float(np.linalg.norm(ds - dj) / np.linalg.norm(dj))

    lo, hi = 1e-4, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if kq(mid) < 1.0:
            lo = mid
        else:
            hi = mid
    u_kq1 = 0.5 * (lo + hi)
    lo2, hi2 = 1e-4, 3.0
    for _ in range(30):
        mid = 0.5 * (lo2 + hi2)
        if nonlin(mid) < 0.01:
            lo2 = mid
        else:
            hi2 = mid
    S4[nm] = {"G_jj": Gjj, "K_times_G_jj_pts_at_u1": float(K * Gjj),
              "rms_delta": float(np.sqrt(Gjj)),
              "max_abs_delta": float(np.abs(dj).max()),
              "mean_delta": float(dj.mean()),
              "corr_with_existing_axes": rho,
              "max_abs_corr_existing": float(max(abs(v) for v in rho.values())),
              "argmax_abs_corr_axis": max(rho, key=lambda k: abs(rho[k])),
              "probe_u_for_KQ_1pt": u_kq1,
              "KQ_at_that_u": kq(u_kq1),
              "nonlinearity_at_that_u": nonlin(u_kq1),
              "max_u_with_nonlinearity_under_1pct": lo2,
              "unknown_L_note": "이 축에는 LB 점이 없다 — L 미측정.  착륙 금지."}
    print(f"  [{nm}] rms(δ) {np.sqrt(Gjj):.6e}  K*G_jj {K*Gjj:.4f}  "
          f"기존축 최대|ρ| {S4[nm]['max_abs_corr_existing']:.4f} "
          f"({S4[nm]['argmax_abs_corr_axis']})  K*Q=1 진폭 {u_kq1:.5f}  "
          f"비선형 {S4[nm]['nonlinearity_at_that_u']:.2e}", flush=True)
R["stage4_axes_on_tr2024"] = S4

# ── 정직면 (OOT 2024) ────────────────────────────────────────────────
OOT, OOT3, OOT4 = f"{H}/exp069/oot", f"{H}/exp070/oot3", f"{H}/exp071/out"
V = {k: np.load(p).astype(np.float64) for k, p in {
    "p_lgb": f"{OOT}/p_lgb_oot2024.npy", "p_cat": f"{OOT}/p_cat_oot2024.npy",
    "p_leaf": f"{OOT}/p_leaf_oot2024.npy", "p_nn": f"{OOT4}/p_nn_oot2024.npy",
    "p_q": f"{OOT}/p_q_oot2024.npy", "p_ebm": f"{OOT3}/p_ebm_oot2024.npy",
    "off": f"{OOT}/off_oot2024.npy",
    "p_v6_4": f"{OOT4}/p_v6_4arm_oot2024.npy"}.items()}
tro = pd.read_csv(f"{H}/data/train.csv",
                  usecols=["row_id", "season", "control_success", "pitcher_id"])
tro = tro[tro.season == 2024].sort_values("row_id", kind="stable")
yo = tro.control_success.to_numpy(np.float64)
oot_rid = np.asarray(np.load(f"{OOT}/row_id_oot2024.npy",
                             allow_pickle=True)).astype(str)
assert np.array_equal(oot_rid, tro.row_id.to_numpy().astype(str))
L4 = {n: C.lg(V[n]) for n in ("p_lgb", "p_cat", "p_leaf", "p_nn")}
lg_v6_oot = (W4["LGB"] * L4["p_lgb"] + W4["CAT"] * L4["p_cat"]
             + W4["NN"] * L4["p_nn"] + W4["LEAF"] * L4["p_leaf"])
R["stage4_oot_start_qualification"] = {
    "max_abs_diff_p_v6_4arm": float(np.abs(expit(lg_v6_oot) - V["p_v6_4"]).max()),
    "oot_vector_ranges": {n: [float(V[n].min()), float(V[n].max())] for n in V}}
print("  OOT 범위:", {n: [round(float(V[n].min()), 4), round(float(V[n].max()), 4)]
                    for n in V}, flush=True)
AA = (42, 7, 2024)
L_aa_o = np.mean([C.lg(np.load(f"{H}/experiments/exp_049/pred/"
                               f"pred_A1_u2024_seed{s}_at1400.npy")) for s in AA],
                 axis=0)
L_t70_o = np.mean([C.lg(np.load(f"{H}/experiments/exp_055/pred/"
                                f"pred_T70_u2024_seed{s}_at1400.npy")) for s in AA],
                  axis=0)
NEW_O = {"v6_CAT_vs_NN": L4["p_cat"] - L4["p_nn"],
         "v6_CAT_vs_LEAF": L4["p_cat"] - L4["p_leaf"],
         "v6_CAT_vs_LGB": L4["p_cat"] - L4["p_lgb"]}
DEP = dict(w_v6=BO["w_v6_inner"], w_ebm=BO["w_ebm"], beta_ins=BO["beta_ins"],
           w_out=BO["w_out"], c_aa=BO["c_aa"], c_t70=BO["c_T70"])


def rc(z):
    c, yb = 0.0, float(yo.mean())
    for _ in range(80):
        p = expit(z + c)
        mu = float(p.mean())
        if abs(mu - yb) < 1e-14:
            break
        d = (mu - yb) / float((p * (1 - p)).mean())
        c -= d
        if abs(d) < 1e-14:
            break
    return c


def oot_pts(name, t):
    lv = lg_v6_oot + t * NEW_O[name]
    z = (DEP["w_out"] * ((1 - DEP["w_ebm"])
                         * (DEP["w_v6"] * lv + (1 - DEP["w_v6"]) * C.lg(V["p_q"]))
                         + DEP["w_ebm"] * C.lg(V["p_ebm"])
                         + DEP["beta_ins"] * V["off"])
         + DEP["c_aa"] * L_aa_o + DEP["c_t70"] * L_t70_o)
    p = expit(z + rc(z))
    return 100000.0 * (1.0 - float(np.mean((p - yo) ** 2)) / C.REF)


base_pts = oot_pts("v6_CAT_vs_NN", 0.0)
O4 = {"honest_surface_base_pts": base_pts,
      "surface": "OOT 2024 (팀원 납품 3팔 + exp_071 p_nn), BO 상수, 전역 재중심화"}
for nm in NEW_O:
    rr = minimize_scalar(lambda t: -oot_pts(nm, t), bounds=(-0.50, 0.50),
                         method="bounded", options={"xatol": 1e-6})
    O4[nm] = {"argmax_t": float(rr.x), "gain_pts": float(-rr.fun) - base_pts,
              "new_W_CAT": W4["CAT"] + float(rr.x),
              "grid": {f"{t:+.2f}": oot_pts(nm, t) - base_pts
                       for t in (-0.3, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.3)}}
    print(f"  [정직면 {nm}] argmax t {rr.x:+.5f} → W_CAT "
          f"{W4['CAT']+rr.x:.5f}  이득 {O4[nm]['gain_pts']:+.5f}점", flush=True)
R["stage4_honest_surface"] = O4
R["stage4_🔴_S12"] = (
    "🔴 S12 인용 — '부호 신탁(4/4)은 전역 스칼라 축의 실적이다. 다차원 가중을 "
    "적합하는 축에는 적용되지 않으며, 그 대용으로 교차 t 를 쓰면 틀린다"
    "(BH 실측 −3.24).'  이 3축은 v6 내부 4팔 가중을 적합하는 축이므로 정직면·"
    "프레임 부호를 방향 근거로 쓸 수 없다.  또 BN 이 실측한 정직면→LB 전이 "
    "배율 0.39~0.55 는 이 곡면 위에서 나온 값이며 부호까지 보증하지 않는다.")
R["stage4_🔴_no_landing"] = (
    "이 3축에는 LB 점이 없다 → L 미측정 → stage3 착륙 해에 넣지 않는다.")
R["wall_clock_min"] = (time.time() - T1) / 60.0
json.dump(R, open(f"{OUT}/stage34_078.json", "w"), indent=1, ensure_ascii=False)
print(f"\n기록 {OUT}/stage34_078.json · 벽시계 {R['wall_clock_min']:.2f}분", flush=True)
