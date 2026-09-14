"""exp_089 stage 2 보강 — 척도 분리(S25/S33) + 파트너 안쪽 단독 면 + 상관 진단.

왜 필요한가
  · 딥 교체는 **혼합 비율 축**이다. S33 이 못박은 대로 이 계열의 정직면 이득은
    sd 를 맞추기 전에는 읽으면 안 된다 — 이득/손실의 상당 부분이 정보가 아니라 척도다.
  · 팀원이 보고한 +10.20 은 **파트너 안쪽 면**에서 잰 값으로 보인다(우리 AA·T70 없음).
    같은 자리를 따로 재야 그 수와 우리 수가 비교 가능해진다.
"""
import json, os, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "6"
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

H = "/home/<user>/SungjinPark"
B = f"{H}/exp089"
E88 = f"{H}/exp088/out/preds"
OOT, OOT3, OOT4 = f"{H}/exp069/oot", f"{H}/exp070/oot3", f"{H}/exp071/out"
EPS, REF = 1e-6, 0.2498069
K = 100000.0 / REF
SEEDS = [42, 777, 2024, 101, 7]
AA = (42, 7, 2024)
W4 = {"LGB": 0.05, "CAT": 0.68, "NN": 0.15, "LEAF": 0.12}
BO = dict(alpha=1.07911, beta_skew=-0.059151, c_aa=0.25244, c_T70=0.04756,
          w_out=0.7, w_v6_inner=0.6668397489, w_ebm=-0.1, beta_ins=0.03)
t0 = time.time()
lg = lambda p: logit(np.clip(np.asarray(p, np.float64), EPS, 1 - EPS))
pts = lambda b: 100000.0 * (1.0 - b / REF)

tro = pd.read_csv(f"{H}/data/train.csv",
                  usecols=["row_id", "season", "pitcher_id", "control_success"])
tro = tro[tro.season == 2024].sort_values("row_id", kind="stable")
y = tro.control_success.to_numpy(np.float64)
pit = tro.pitcher_id.to_numpy()
YB = float(y.mean())
V = {k: np.load(p).astype(np.float64) for k, p in {
    "p_lgb": f"{OOT}/p_lgb_oot2024.npy", "p_cat": f"{OOT}/p_cat_oot2024.npy",
    "p_leaf": f"{OOT}/p_leaf_oot2024.npy", "p_nn": f"{OOT4}/p_nn_oot2024.npy",
    "p_q": f"{OOT}/p_q_oot2024.npy", "p_ebm": f"{OOT3}/p_ebm_oot2024.npy",
    "off": f"{OOT}/off_oot2024.npy"}.items()}
LV = sum(W4[t] * lg(V[k]) for t, k in
         (("LGB", "p_lgb"), ("CAT", "p_cat"), ("NN", "p_nn"), ("LEAF", "p_leaf")))
LQ, LE = lg(V["p_q"]), lg(V["p_ebm"])
L_aa = np.mean([lg(np.load(f"{H}/experiments/exp_049/pred/"
                           f"pred_A1_u2024_seed{s}_at1400.npy")) for s in AA], 0)
L_t70 = np.mean([lg(np.load(f"{H}/experiments/exp_055/pred/"
                            f"pred_T70_u2024_seed{s}_at1400.npy")) for s in AA], 0)
Z_MATE = ((1 - BO["w_ebm"]) * (BO["w_v6_inner"] * LV + (1 - BO["w_v6_inner"]) * LQ)
          + BO["w_ebm"] * LE + BO["beta_ins"] * V["off"])
DEEP = {"oot23_ours": np.mean([np.load(f"{B}/out/preds/z_M_oot23_s{s}.npy")
                               for s in SEEDS], 0),
        "le2024_ours": np.mean([np.load(f"{B}/out/preds/z_M_le2024_s{s}.npy")
                                for s in SEEDS], 0),
        "le2024_mate": np.mean([np.load(f"{E88}/deep_route1_s{s}_logit.npy")
                                for s in SEEDS], 0)}

R = {"exp_id": "exp_089", "stage": "s3_extra"}
R["correlations_on_honest_surface"] = {
    "rho(z_mate_oot, deep_oot23_ours)": float(np.corrcoef(Z_MATE, DEEP["oot23_ours"])[0, 1]),
    "rho(z_mate_oot, deep_le2024_mate)": float(np.corrcoef(Z_MATE, DEEP["le2024_mate"])[0, 1]),
    "rho(deep_oot23_ours, deep_le2024_ours)": float(
        np.corrcoef(DEEP["oot23_ours"], DEEP["le2024_ours"])[0, 1]),
    "sd_z_mate_oot": float(Z_MATE.std()),
    "sd_deep": {k: float(v.std()) for k, v in DEEP.items()},
    "mate_arm_scores_alone_oot": {
        "v6_4arm": pts(float(np.mean((expit(LV) - y) ** 2))),
        "Q": pts(float(np.mean((expit(LQ) - y) ** 2))),
        "EBM": pts(float(np.mean((expit(LE) - y) ** 2))),
        "z_mate_recentred": None},
}


def pin(z, f, target=YB):
    c = 0.0
    for _ in range(200):
        p = f(z, c)
        g = float(p.mean()) - target
        if abs(g) < 1e-15:
            break
        c -= g / float((p * (1 - p)).mean())
    return c


F = lambda z, c: expit(z + c)
R["correlations_on_honest_surface"]["mate_arm_scores_alone_oot"]["z_mate_recentred"] = \
    pts(float(np.mean((F(Z_MATE, pin(Z_MATE, F)) - y) ** 2)))


def score_z(z):
    return pts(float(np.mean((F(z, pin(z, F)) - y) ** 2)))


def score_alpha_free(z):
    """전역 스칼라 a 를 자유롭게 — 척도로 살 수 있는 몫을 다 준 뒤의 값."""
    r = minimize_scalar(lambda a: -score_z(a * z), bounds=(0.3, 3.0),
                        method="bounded", options={"xatol": 1e-8})
    return float(-r.fun), float(r.x)


def sd_match(znew, zref):
    """S25 — 이동 후 로짓의 (mean, sd) 를 이동 전에 맞춘다."""
    return ((znew - znew.mean()) / znew.std()) * zref.std() + zref.mean()


GRID = [round(x, 4) for x in np.arange(0.0, 0.6001, 0.02)]
SURF = {
    "root_full": lambda zm: BO["w_out"] * zm + BO["c_aa"] * L_aa + BO["c_T70"] * L_t70,
    "mate_only": lambda zm: zm,
}
R["sweeps"] = {}
for sname, sf in SURF.items():
    for dname, dz in DEEP.items():
        z0 = sf(Z_MATE)
        b_raw = score_z(z0)
        b_af, a_b = score_alpha_free(z0)
        cur = []
        for w in GRID:
            zm = (1.0 - w) * Z_MATE + w * dz
            z = sf(zm)
            raw = score_z(z)
            af, a_ = score_alpha_free(z)
            sdm = score_z(sd_match(z, z0)) if w > 0 else raw
            cur.append({"w": w, "gain_raw": raw - b_raw,
                        "gain_alpha_free": af - b_af,
                        "gain_sd_matched": sdm - b_raw, "alpha_star": a_})
        g20 = [c for c in cur if abs(c["w"] - 0.20) < 1e-9][0]
        R["sweeps"][f"{sname}__{dname}"] = {
            "base_raw_pts": b_raw, "base_alpha_free_pts": b_af,
            "base_alpha_star": a_b,
            "at_w0.20": g20,
            "argmax_raw": max(cur, key=lambda c: c["gain_raw"]),
            "argmax_alpha_free": max(cur, key=lambda c: c["gain_alpha_free"]),
            "argmax_sd_matched": max(cur, key=lambda c: c["gain_sd_matched"]),
            "curve": cur,
        }
        print(f"{sname:10s} {dname:12s} base {b_raw:9.3f} | w=.20 raw "
              f"{g20['gain_raw']:+8.3f} afree {g20['gain_alpha_free']:+8.3f} "
              f"sdm {g20['gain_sd_matched']:+8.3f}", flush=True)

R["wall_sec"] = round(time.time() - t0, 1)
json.dump(R, open(f"{B}/out/s3_extra.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in R.items() if k != "sweeps"},
                 ensure_ascii=False, indent=1))
