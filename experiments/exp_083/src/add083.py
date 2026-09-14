"""exp_083 보강 — 생존 후보의 귀무 + '기존(이미 산) 축' 단독 최대 |ρ|."""
import os
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "6"
import json, sys
import numpy as np, pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
sys.path.insert(0, "/home/<user>/SungjinPark/exp078/src")
sys.path.insert(0, "/home/<user>/SungjinPark/exp079/src")
import core078 as C, surface079 as S
H, K, REF, N, T_REF = C.H, C.K, C.REF, C.N_ROWS, C.T_REF
O = f"{H}/exp083"; EPS = 1e-6


def lgc(p):
    return logit(np.clip(np.asarray(p, np.float64), EPS, 1 - EPS))


CM = np.load(f"{O}/out/cap_mate.npz", allow_pickle=True)
CA = np.load(f"{O}/out/cap_aa_cal.npz", allow_pickle=True)
NM = list(CA["names"])
SR = [42,7,2024,1,13,101,777,2025,31337,3,17,77,123,555,999,2019,8191,65537]
stk = np.stack([lgc(CA[f"v{NM.index(f'R|R_seed_{s}_stacker.cbm'):03d}"]) for s in SR])
AXV = {"D_nn_hidden_nact": CM["nn_hidden_nact"].astype(np.float64),
       "D_nn_hidden_pc3": CM["nn_hidden_pc"][:, 2].astype(np.float64),
       "E_aa_seed_sd18": stk.std(axis=0, ddof=1)}

W_TR = C.zf(C.BO, C.ARMS18)
p_BO = expit(W_TR + C.anc(W_TR, T_REF))
_ft = pd.read_csv(f"{H}/exp082/runs/ft_matecal.csv", encoding="utf-8-sig",
                  float_precision="round_trip")
EXIST = {a: C.DELTA[C.IDX[a]] for a in C.AXES}
_n9 = list(np.load(f"{H}/exp080/out/cand_delta_names.npy", allow_pickle=True))
_d9 = np.load(f"{H}/exp080/out/cand_delta.npy")
for i, nm in enumerate(_n9):
    EXIST[f"exp080/{nm}"] = _d9[i]
for nm, col in (("teammate_b(q_lg_mr)", "lg_mr"), ("teammate_c(q_lg_wo)", "lg_wo")):
    vv = _ft[col].to_numpy(np.float64); h = 0.002
    o1, o2 = h*(vv-vv.mean()), -h*(vv-vv.mean())
    EXIST[nm] = (expit(W_TR + C.anc(W_TR+o1, T_REF) + o1)
                 - expit(W_TR + C.anc(W_TR+o2, T_REF) + o2)) / (2*h)


def delta(v):
    sd = float(np.std(v)); h = 0.002 / max(sd, 1e-12)
    u = np.asarray(v, np.float64) - float(np.mean(v))
    a = expit(W_TR + C.anc(W_TR + h*u, T_REF) + h*u)
    b = expit(W_TR + C.anc(W_TR - h*u, T_REF) - h*u)
    return (a - b) / (2*h)


LV0 = (S.W4["LGB"]*S._L["p_lgb"] + S.W4["CAT"]*S.lg(S.V["p_cat"])
       + S.W4["NN"]*S._L["p_nn"] + S.W4["LEAF"]*S._L["p_leaf"])
Z0 = (S.BO["w_out"]*((1-S.BO["w_ebm"])*(S.BO["w_v6_inner"]*LV0
      + (1-S.BO["w_v6_inner"])*S._L["p_q"]) + S.BO["w_ebm"]*S._L["p_ebm"]
      + S.BO["beta_ins"]*S.V["off"]) + S.BO["c_aa"]*S.L_aa + S.BO["c_T70"]*S.L_t70)
yo = S.yo


def zp(z):
    p = expit(z + S.rc(z)); return 100000.0*(1.0 - float(np.mean((p-yo)**2))/REF)


BASE = zp(Z0)


def screen(v):
    u = np.asarray(v, np.float64) - float(np.mean(v))
    u = u / max(float(u.std()), 1e-300)
    r = minimize_scalar(lambda t: -zp(Z0 + t*u), bounds=(-0.35, 0.35),
                        method="bounded", options={"xatol": 1e-7})
    return float(-r.fun) - BASE


rng = np.random.default_rng(830)
R = {"base_pts": BASE, "n_perm": 60}
for k, v in AXV.items():
    d = delta(v)
    rr = {a: float(np.dot(d, dv)/(np.linalg.norm(d)*np.linalg.norm(dv)))
          for a, dv in EXIST.items() if np.linalg.norm(dv) > 0}
    mx = max(rr.items(), key=lambda kv: abs(kv[1]))
    gs = np.array([screen(v[rng.permutation(N)]) for _ in range(60)])
    obs = screen(v)
    R[k] = {"max_abs_rho_BOUGHT_axes_only": abs(mx[1]), "axis": mx[0], "signed": mx[1],
            "rho_top5_bought": dict(sorted(rr.items(), key=lambda kv: -abs(kv[1]))[:5]),
            "observed_raw_pts": obs, "xfer_pts": obs*0.05153,
            "null_mean": float(gs.mean()), "null_sd": float(gs.std(ddof=1)),
            "null_max": float(gs.max()), "null_p99": float(np.quantile(gs, 0.99)),
            "z_vs_null": float((obs-gs.mean())/gs.std(ddof=1)),
            "p_perm_ge_obs": float((gs >= obs).mean())}
    print(f"{k:<22} 산 축 대비 max|ρ| {abs(mx[1]):.4f} ({mx[0]})  obs {obs:+.4f}  "
          f"귀무 평균 {gs.mean():+.4f} sd {gs.std(ddof=1):.4f} max {gs.max():+.4f}  "
          f"z {(obs-gs.mean())/gs.std(ddof=1):.1f}  p {(gs>=obs).mean():.4f}", flush=True)
json.dump(R, open(f"{O}/out/add.json", "w"), indent=1, ensure_ascii=False)
print("->", f"{O}/out/add.json")
