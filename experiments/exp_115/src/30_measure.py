#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp_115 2단 — sp_pit_ball3 의 곡률·교차항·천장.  후보 CZ 와 같은 코드·같은 프레임.
   🔴 크기·비용 측정이다.  부호 판정이 아니다."""
import json, sys
import numpy as np, pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit

H = "/home/<user>/SungjinPark"; B = f"{H}/build_pb3_probe"
PKG, WORK = f"{B}/pkg", f"{B}/work"
FRAME = f"{H}/build_AC/work/tr2024"
K = 400309.19882517256
sys.path.insert(0, f"{PKG}/PAR")
import splits as SPL

def L(p): return json.load(open(p, encoding="utf-8"))
B10 = L(f"{WORK}/build10.json"); CZ = L(f"{WORK}/solve30.json")
blp = L(f"{PKG}/PAR/blend.json"); D_PAR = float(blp["d_adj"])
U = float(B10["u"]); U_SIB = -0.014
LB_CT = 6.6519071210

p_par = np.load(f"{WORK}/calp_PAR.npy")
p_pb3 = np.load(f"{WORK}/calp_PB3.npy")
p_zi = np.load(f"{WORK}/calp_ZI.npy")
p_zp = np.load(f"{WORK}/calp_ZP.npy")
p_zs = np.load(f"{WORK}/calp_ZS.npy")
n = len(p_par)
out = {"n_rows": n, "axis": B10["axis"], "u": U, "u_sibling": U_SIB,
       "center": B10["center"], "scale": B10["scale"]}

AIM = float(p_par.mean())
out["determinism"] = {
    "aim_this_session": repr(AIM),
    "aim_recorded_in_CZ_solve": CZ["d_adj"]["aim_measured_parent_CS3_mean"],
    "identical": repr(AIM) == CZ["d_adj"]["aim_measured_parent_CS3_mean"],
    "note": ("부모 캘 실행을 이번 세션에 다시 돌려 CZ 측정과 같은 값이 나오는지 확인한다. "
             "같으면 ZI/ZP/ZS 벡터(그 세션 산출물)를 그대로 써도 된다.")}

f_ = logit(p_pb3) - D_PAR
d_ship = float(brentq(lambda d: float(expit(f_ + d).mean()) - AIM, -5.0, 5.0,
                      xtol=1e-16, rtol=4 * np.finfo(float).eps, maxiter=300))
p_ship = expit(f_ + d_ship)
dpv = p_ship - p_par
Q = float(K * (dpv ** 2).mean())
out["d_adj_dependent"] = {"parent": repr(D_PAR), "reland": repr(d_ship),
                          "step": repr(d_ship - D_PAR),
                          "residual": float(p_ship.mean() - AIM)}

cal = pd.read_csv(f"{FRAME}/test.csv", encoding="utf-8-sig", usecols=["game_type"])
isR = (cal.game_type == "R").to_numpy()
d_sp = p_par - p_zs
d_ins = p_par - p_zi
d_psr = p_par - p_zp
d_cnt = p_par - np.load(f"{H}/exp109/work/calp_CT.npy")
lg_mr = np.load(f"{WORK}/ftmr_PAR.npy")
MU_MR = float(blp["failtype_offset"]["mu_mr"])
sfac = p_par * (1 - p_par)
def pin(v): return sfac * (v - (sfac * v).sum() / sfac.sum())
d_ftb = pin(lg_mr - MU_MR)

CT_ = {}
CSUM = 0.0
for nm, dv in (("sp_pit_split", d_sp), ("beta_ins_off", d_ins),
               ("ctrl_asof_psr", d_psr), ("count_offsets_CS3", d_cnt),
               ("failtype_b_direction", d_ftb)):
    e = {"rho_all": float(np.corrcoef(dv, dpv)[0, 1]),
         "rho_R": float(np.corrcoef(dv[isR], dpv[isR])[0, 1]),
         "rho_F": float(np.corrcoef(dv[~isR], dpv[~isR])[0, 1])}
    if nm != "failtype_b_direction":
        e["cross_2K_mean_pts"] = float(2 * K * np.mean(dv * dpv))
        CSUM += e["cross_2K_mean_pts"]
    CT_[nm] = e

Qsib = float(K * (d_sp ** 2).mean())
out["amplitude"] = {"K_mean_dp2_pts": Q, "max_abs_dp": float(np.abs(dpv).max()),
                    "rms_dp": float(np.sqrt((dpv ** 2).mean())),
                    "mean_shift_after_relanding": float(dpv.mean()),
                    "n_moved": int(np.count_nonzero(dpv)),
                    "sibling_sp_pit_split_K_mean_dp2_pts": Qsib}
out["cross_terms_all_axes"] = CT_
out["cross_sum_pts"] = CSUM
COST = Q + CSUM

out["amplitude_convention_check"] = {
    "ours_over_sibling": Q / Qsib,
    "predicted_by_quadratic_model": (U / U_SIB) ** 2,
    "pct_dev": 100 * (Q / Qsib) / (U / U_SIB) ** 2 - 100,
    "bat_phand_pct_dev": 100 * 0.2495992978818122 / 0.25 - 100,
    "note": ("형제 축은 두 성분(pit_bhand/sd1 + pit_strk2/sd2)의 합이라 한 성분 축과 "
             "곡률이 정확히 같을 구조적 이유는 없다.  bat_phand 는 −0.16% 였다.")}

A_SIB = LB_CT + Qsib
LAM = {"t_ratio_bat_scale_0.5724": 2.47 / 4.315,
       "transfer_bat_scale_0.8039": 0.414 / 0.515,
       "equal_strength_1.00": 1.0,
       "OWN_transfer_0.254_over_0.515": 0.254 / 0.515,
       "OWN_f24_t_0.22_over_4.315": 0.22 / 4.315}
tab = {}
for k, lam in LAM.items():
    A = A_SIB * abs(U / U_SIB) * lam
    tab[k] = {"lambda": lam, "K_L_u": A,
              "gain_at_half_step_favourable": A - COST,
              "gain_at_half_step_opposite": -A - COST,
              "ceiling_favourable_branch": (A - CSUM) ** 2 / (4 * Q),
              "ceiling_opposite_branch": (A + CSUM) ** 2 / (4 * Q)}
out["ceiling"] = {"q_own_curvature": Q, "cross_sum": CSUM, "cost_at_half_step": COST,
                  "A_sibling_from_LB": A_SIB, "table": tab,
                  "formula": ("gain(g) = A'*g - q*g^2 ; A' = A - cross_sum ; "
                              "ceiling = A'^2/(4q).  부호를 모르므로 두 가지를 다 적는다 "
                              "— 부호는 LB 로 산다.")}

json.dump(out, open(f"{WORK}/measure30.json", "w"), ensure_ascii=False, indent=1)
print(f"q={Q:.4f}  교차합={CSUM:+.4f}  비용={COST:.4f}  "
      f"rho_sp={CT_['sp_pit_split']['rho_all']:+.6f}")
print(f"진폭 검산 {Q / Qsib:.4f} vs 예측 {(U / U_SIB) ** 2:.4f} "
      f"({100 * (Q / Qsib) / (U / U_SIB) ** 2 - 100:+.2f}%)")
for k, v in tab.items():
    print(f"  {k:32s} A={v['K_L_u']:6.3f}  천장 유리 {v['ceiling_favourable_branch']:6.3f}"
          f"  반대 {v['ceiling_opposite_branch']:7.3f}")
print("MEASURE_OK")
