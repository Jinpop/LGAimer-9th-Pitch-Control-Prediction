#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp_084 stage3·4 — 곡률 법칙 적합 → 2025 G 추정 → 이동폭 재선택 · 반대칭 축 비교."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "6"
import json, time                                                   # noqa: E402
import numpy as np                                                  # noqa: E402
from scipy.stats import t as tdist                                  # noqa: E402

H = "/home/<user>/SungjinPark"
OUT = f"{H}/exp084/out"
K = 400309.19882517256
T0 = time.time()
S1 = json.load(open(f"{OUT}/s1_missrate.json", encoding="utf-8"))
S2 = json.load(open(f"{OUT}/s2_curv.json", encoding="utf-8"))
DQ = json.load(open(f"{H}/build_BR/work/diag_qleague.json", encoding="utf-8"))
R = {"exp_id": "exp_084", "stage": "3/4", "date": time.strftime("%Y-%m-%d %H:%M:%S"),
     "K_pts_per_brier": K}

# ── LB 실측 두 점 ──────────────────────────────────────────────────────
LB_BO = 1115.9913864857
U4 = 0.004046327151797174 - (-0.0035568207147814723)      # BS4, 셀 2|2
U3 = 0.011485705117032143 - 0.003841565032476346          # BS3, 셀 2|1
G4_LB = 1.3177794334          # BS4 − BO  (1117.3091659191 − 1115.9913864857)
G3_LB = -0.19                 # BS3 − BO  (지휘자 보고, 소수점 2자리)
G3_BAND = (-0.21, -0.17)
R["LB_points"] = {"parent_BO": LB_BO,
                  "BS4_cell": "2|2", "BS4_u": U4, "BS4_gain": G4_LB,
                  "BS3_cell": "2|1", "BS3_u": U3, "BS3_gain": G3_LB,
                  "⚠️_BS3_precision": ("BS3 는 소수점 2자리로만 전달받았다 (−0.19). "
                                       "±0.02 밴드로 민감도를 병기한다."),
                  "BS4_lb_after": LB_BO + G4_LB}

# ── 곡률 법칙 적합:  K·G(f) = A · f · (1−f)  (평균 못박힘이 만드는 형태) ──
SYN = S2["synthetic_frames"]
FR = f"frame_le2023_actual"


def law_fit(cell):
    xs, ys = [], []
    for e in SYN:
        f = e["per_cell"][cell]["fire_frac_of_frame"] if cell != "antisym_ph2" else None
        if cell == "antisym_ph2":
            continue
        if f <= 0:
            continue
        xs.append(f); ys.append(e["per_cell"][cell]["K_G_jj"])
    xs = np.array(xs); ys = np.array(ys)
    phi = xs * (1.0 - xs)
    A = float(phi @ ys / (phi @ phi))
    pred = A * phi
    rel = np.abs(ys - pred) / ys
    # 대조: 순수 비례 A0·f
    A0 = float(xs @ ys / (xs @ xs))
    rel0 = np.abs(ys - A0 * xs) / ys
    return {"A_f_times_1_minus_f": A, "max_rel_resid": float(rel.max()),
            "rms_rel_resid": float(np.sqrt((rel ** 2).mean())),
            "pure_proportional_A0": A0, "pure_proportional_max_rel_resid": float(rel0.max()),
            "f": xs.tolist(), "K_G_measured": ys.tolist(), "K_G_law": pred.tolist()}


LAW = {c: law_fit(c) for c in ["1|1", "1|2", "2|1", "2|2"]}
# 반대칭 축은 두 셀의 받침이 서로소라 φ = f22(1−f22) + w²f21(1−f21) − 2w f22 f21
W = S2["frame_le2023_actual"]["antisym_ph2"]["weight_ratio_n22_over_n21"]
xs, ys = [], []
for e in SYN:
    f22 = e["per_cell"]["2|2"]["fire_frac_of_frame"]
    f21 = e["per_cell"]["2|1"]["fire_frac_of_frame"]
    if f22 <= 0:
        continue
    xs.append(f22 * (1 - f22) + W * W * f21 * (1 - f21) + 2 * W * f22 * f21)
    ys.append(e["antisym_ph2"]["K_G_jj"])
xs = np.array(xs); ys = np.array(ys)
A_anti = float(xs @ ys / (xs @ xs))
LAW["antisym_ph2"] = {"A_phi": A_anti, "phi": xs.tolist(), "K_G_measured": ys.tolist(),
                      "max_rel_resid": float((np.abs(ys - A_anti * xs) / ys).max()),
                      "phi_form": ("f22(1−f22) + w²f21(1−f21) + 2w·f22·f21 — 받침이 서로소라 "
                                   "E[1_S22·1_S21] = 0 이고, 평균 못박힘이 남기는 교차항은 "
                                   "+2w·f22·f21 로 **양수**다 (G_anti 가 G22 + w²G21 보다 크다)")}
R["curvature_law"] = LAW
for c, v in LAW.items():
    if c == "antisym_ph2":
        print(f"[법칙 {c}] A {v['A_phi']:.1f} · 최대상대잔차 {v['max_rel_resid']:.5f}")
    else:
        print(f"[법칙 {c}] A {v['A_f_times_1_minus_f']:.1f} · 최대상대잔차 {v['max_rel_resid']:.5f}"
              f"  (순수비례는 {v['pure_proportional_max_rel_resid']:.4f})", flush=True)

A22 = LAW["2|2"]["A_f_times_1_minus_f"]
A21 = LAW["2|1"]["A_f_times_1_minus_f"]
F22_FRAME = S2["frame_le2023_actual"]["per_cell"]["2|2"]["fire_frac_of_frame"]
F21_FRAME = S2["frame_le2023_actual"]["per_cell"]["2|1"]["fire_frac_of_frame"]
KG22_FRAME = S2["frame_le2023_actual"]["per_cell"]["2|2"]["K_G_jj"]
KG21_FRAME = S2["frame_le2023_actual"]["per_cell"]["2|1"]["K_G_jj"]


# ── 2025 의 f22 (2|2 발동 비중) 추정 여러 갈래 ──────────────────────────
PS = S1["per_season"]
yrs = sorted(int(y) for y in PS)


def series(getter):
    return np.array([getter(PS[str(y)]) for y in yrs], float)


def lin_extrap(x, y, xnew):
    n = len(x); A = np.vstack([np.ones(n), x]).T
    co, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ co
    dof = n - 2
    s2 = float(res @ res / dof)
    xb = x.mean(); sxx = float(((x - xb) ** 2).sum())
    yh = float(co[0] + co[1] * xnew)
    se = float(np.sqrt(s2 * (1 + 1 / n + (xnew - xb) ** 2 / sxx)))
    tc = float(tdist.ppf(0.975, dof))
    tslope = float(co[1] / np.sqrt(s2 / sxx))
    return {"pred": yh, "pi95": [yh - tc * se, yh + tc * se], "se_pred": se,
            "slope": float(co[1]), "t_slope": tslope,
            "p_slope_two_sided": float(2 * tdist.sf(abs(tslope), dof)),
            "r2": float(1 - (res @ res) / ((y - y.mean()) ** 2).sum())}


X = np.array(yrs, float)
f22 = series(lambda e: e["per_cell"]["2|2"]["fire_frac_of_frame"])
mall = series(lambda e: e["miss_frac"])
sh22 = series(lambda e: e["per_cell"]["2|2"]["cell_share_of_frame"])
in22 = series(lambda e: e["per_cell"]["2|2"]["miss_frac_within_cell"])
EX = {"f22_fire_frac": lin_extrap(X, f22, 2025),
      "miss_frac_all": lin_extrap(X, mall, 2025),
      "cell_share_22": lin_extrap(X, sh22, 2025),
      "miss_within_22": lin_extrap(X, in22, 2025)}
EX["f22_via_product"] = {
    "pred": EX["cell_share_22"]["pred"] * EX["miss_within_22"]["pred"],
    "note": "셀 비중 추세 × 셀 내 미적중 추세"}
R["extrapolation_2025"] = {"years": yrs, "f22_series": f22.tolist(),
                           "miss_all_series": mall.tolist(),
                           "cell_share_22_series": sh22.tolist(),
                           "miss_within_22_series": in22.tolist(), "fits": EX}
print("\n[외삽] f22 2025", EX["f22_fire_frac"], flush=True)

# ── real5 (n=5, 실제로는 서로 다른 키 4개) ─────────────────────────────
r5 = S2["real5_direct"]
keys = [x["personal_key"] for x in r5["rows"]]
uk = {}
for x in r5["rows"]:
    uk[x["personal_key"]] = x["hit"]
R["real5"] = {"rows": r5["rows"], "n_rows": 5, "n_miss_rows": r5["n_miss"],
              "n_distinct_keys": len(uk), "n_distinct_miss": int(sum(1 for v in uk.values() if not v)),
              "f22_fire_in_real5": float(sum(1 for x in r5["rows"]
                                             if x["cell"] == "2|2" and not x["hit"]) / 5),
              "🔴": ("5행의 개인 키는 4개뿐이다 — TEST_005332 와 TEST_035185 가 같은 "
                     "`24713|2` 다.  즉 독립 표본은 5가 아니라 4이고 그중 3이 미적중이다.")}
from scipy.stats import binom                                        # noqa: E402
p23 = float(PS["2024"]["miss_frac"])
R["real5"]["P_ge4_of_5_at_p_frame"] = float(binom.sf(3, 5, p23))
R["real5"]["P_ge3_of_4_at_p_frame"] = float(binom.sf(2, 4, p23))
R["real5"]["p_frame"] = p23

# ── 곡률 시나리오 ──────────────────────────────────────────────────────
def kg(f, A=A22):
    return A * f * (1.0 - f)


SCEN = {}
for nm, f in [("추세_2025_점추정", EX["f22_fire_frac"]["pred"]),
              ("추세_2025_PI하한_0.005바닥", max(EX["f22_fire_frac"]["pi95"][0], 0.005)),
              ("추세_2025_PI상한", EX["f22_fire_frac"]["pi95"][1]),
              ("추세_곱_2025", EX["f22_via_product"]["pred"]),
              ("최근3년_평균", float(f22[-3:].mean())),
              ("프레임_le2023", F22_FRAME),
              ("real5_1of5", 0.2),
              ("전셀_발동_상한", 0.364522)]:
    SCEN[nm] = {"f22": f, "K_G": kg(f), "c_vs_frame": kg(f) / KG22_FRAME}
R["curvature_scenarios"] = SCEN
print("\n[시나리오]")
for k, v in SCEN.items():
    print(f"  {k:<18} f22 {v['f22']:.6f}  K·G {v['K_G']:9.3f}  c {v['c_vs_frame']:.4f}", flush=True)


# ── 꼭지점 기계 ────────────────────────────────────────────────────────
def solve(u0, g0, KG):
    """gain(u) = 2K·u·L − K·u²·G,  Q0 = K u0² G."""
    Q0 = KG * u0 * u0
    L = (g0 + Q0) / (2.0 * K * u0)
    ustar = L * K / KG                      # L/G,  G = KG/K
    return {"Q0_at_probe": Q0, "L": L, "u_star": ustar,
            "max_gain": K * L * L / (KG / K) if KG > 0 else None,
            "gain": lambda d: 2 * K * d * L - KG * d * d}


DGRID = [0.02, 0.03, 0.035, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12]
TAB = {}
for nm, v in SCEN.items():
    s = solve(U4, G4_LB, v["K_G"])
    TAB[nm] = {"K_G": v["K_G"], "c": v["c_vs_frame"], "L": s["L"],
               "u_star": s["u_star"], "max_gain": s["max_gain"],
               "gain_by_delta": {f"{d:.3f}": float(s["gain"](d)) for d in DGRID}}
R["delta_table"] = TAB
print("\n[이동폭별 예상 이득]")
hdr = "  시나리오".ljust(20) + "".join(f"{d:>8.3f}" for d in DGRID) + "     u*   최대"
print(hdr)
for nm, v in TAB.items():
    print("  " + nm.ljust(18)
          + "".join(f"{v['gain_by_delta'][f'{d:.3f}']:8.2f}" for d in DGRID)
          + f"  {v['u_star']:7.4f} {v['max_gain']:7.2f}", flush=True)

# 최악의 경우가 가장 높은 Δ (추세하한~real5 범위)
HEDGE = ["추세_2025_점추정", "프레임_le2023", "real5_1of5"]
grid = np.linspace(0.005, 0.15, 2901)
worst = np.full_like(grid, np.inf)
for nm in HEDGE:
    s = solve(U4, G4_LB, SCEN[nm]["K_G"])
    worst = np.minimum(worst, np.array([s["gain"](d) for d in grid]))
i = int(np.argmax(worst))
R["minimax_over_scenarios"] = {
    "scenarios": HEDGE, "argmax_delta": float(grid[i]), "worst_case_gain": float(worst[i]),
    "worst_at_0.04": float(min(solve(U4, G4_LB, SCEN[nm]["K_G"])["gain"](0.04) for nm in HEDGE)),
    "worst_at_0.05": float(min(solve(U4, G4_LB, SCEN[nm]["K_G"])["gain"](0.05) for nm in HEDGE)),
    "worst_at_0.06": float(min(solve(U4, G4_LB, SCEN[nm]["K_G"])["gain"](0.06) for nm in HEDGE))}
print("\n[미니맥스]", json.dumps(R["minimax_over_scenarios"], ensure_ascii=False), flush=True)

R["nonlinearity"] = {
    "max_u_with_nonlinearity_under_1pct_BR_diag":
        DQ["axes"]["q_league_22"]["max_u_with_nonlinearity_under_1pct"],
    "probe_u_for_KQ_1pt": DQ["axes"]["q_league_22"]["probe_u_for_KQ_1pt"],
    "note": "권고 Δ 가 1% 한계의 절반을 넘으면 3차항 오차를 따로 재야 한다"}

# ── stage4: 반대칭 축 (합 0 유지) ──────────────────────────────────────
F21_SERIES = {e["target_m"]: e["per_cell"]["2|1"]["fire_frac_of_frame"] for e in SYN}
A_ANTI = LAW["antisym_ph2"]["A_phi"]


def kg_anti(f22v, f21v):
    return A_ANTI * (f22v * (1 - f22v) + W * W * f21v * (1 - f21v) + 2 * W * f22v * f21v)


def anti(kg22, kg21, g3, kganti=None):
    L22 = (G4_LB + kg22 * U4 * U4) / (2.0 * K * U4)
    L21 = (g3 + kg21 * U3 * U3) / (2.0 * K * U3)
    La = L22 - W * L21
    KGa = kganti if kganti is not None else (kg22 + W * W * kg21)
    return {"L22": L22, "L21": L21, "L_anti": La, "K_G_anti": KGa,
            "u_star_22": L22 * K / kg22, "max_gain_22": K * K * L22 * L22 / kg22,
            "u_star_anti": La * K / KGa, "max_gain_anti": K * K * La * La / KGa,
            "crossover_delta": float(2 * (La - L22) / ((KGa - kg22) / K)),
            "gain_22_by_delta": {f"{d:.3f}": float(2 * K * d * L22 - kg22 * d * d)
                                 for d in DGRID},
            "gain_anti_by_delta": {f"{d:.3f}": float(2 * K * d * La - KGa * d * d)
                                   for d in DGRID}}


ANTI = {}
for nm in ["프레임_le2023", "추세_2025_점추정", "real5_1of5"]:
    f = SCEN[nm]["f22"]
    # 같은 미적중률 아래 2|1 의 발동 비중도 같은 비율로 움직인다고 본다
    scale = f / F22_FRAME
    kg22 = kg(f, A22)
    f21 = min(F21_FRAME * scale, 1.0)
    kg21 = A21 * f21 * (1 - f21)
    ANTI[nm] = anti(kg22, kg21, G3_LB, kganti=kg_anti(f, f21))
    ANTI[nm]["f22"], ANTI[nm]["f21"], ANTI[nm]["K_G21"] = f, f21, kg21
ANTI["BS3_sensitivity_frame"] = {
    f"g3={g:.2f}": {kk: vv for kk, vv in anti(
        KG22_FRAME, KG21_FRAME, g,
        kganti=S2["frame_le2023_actual"]["antisym_ph2"]["K_G_jj"]).items()
                    if kk in ("L_anti", "max_gain_anti", "u_star_anti")}
    for g in (G3_BAND[0], G3_LB, G3_BAND[1])}
ANTI["🔴_zero_sum_prose"] = (
    "출하 mate/q_infer.py:117-120 이 `zero-sum over bh within each ph` 를 주장한다(S48). "
    "2|2 단독 이동은 그 문장을 거짓으로 만든다 (BS4 가 이미 그렇게 나갔고 zip 안에 기록이 "
    "없다). 반대칭 축(ΔL22 = +u, ΔL21 = −u·n22/n21)은 그 문장을 참으로 유지한다. "
    "n22/n21 = %.15f (n_by_cell 2|2 %d, 2|1 %d)." % (W, 529843, 490570))
R["stage4_antisym"] = ANTI
print("\n[stage4 반대칭]")
for nm in ["프레임_le2023", "추세_2025_점추정", "real5_1of5"]:
    v = ANTI[nm]
    print(f"  {nm:<18} L22 {v['L22']:.6e} L21 {v['L21']:+.6e} L_anti {v['L_anti']:.6e} "
          f"K·G_anti {v['K_G_anti']:8.2f}")
    print(f"      최대이득  2|2단독 {v['max_gain_22']:7.3f} (u* {v['u_star_22']:.4f})  vs "
          f"반대칭 {v['max_gain_anti']:7.3f} (u* {v['u_star_anti']:.4f})  교차점 Δ "
          f"{v['crossover_delta']:.5f}")
    print(f"      Δ=0.04    2|2단독 {v['gain_22_by_delta']['0.040']:7.3f}  vs 반대칭 "
          f"{v['gain_anti_by_delta']['0.040']:7.3f}", flush=True)

R["stage5_cross"] = S2["stage5_cross"]
R["wall_sec"] = time.time() - T0
json.dump(R, open(f"{OUT}/s3_solve.json", "w"), indent=1, ensure_ascii=False)
print(f"\n기록 {OUT}/s3_solve.json · {R['wall_sec']:.1f}s\nS3_OK")
