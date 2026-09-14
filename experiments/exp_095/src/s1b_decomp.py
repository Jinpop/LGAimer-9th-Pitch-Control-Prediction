"""exp_095 stage 1b — alpha 자유 면에서 v 축이 무엇을 사는지 가른다.

stage1 이 낸 것: 고정 alpha 에서 2자유도 argmax = 1자유도(v=0) argmax (차 0.0).
                 그런데 alpha 자유로 재면 argmax 가 (w 0.06, v 0.30) 으로 옮겨가고
                 v=0 최선보다 +2.489 크다.  그 +2.489 가 무엇인가?

가설: v 를 올리고 alpha 를 낮추면 z 안에서 c_aa·z_aa 와 c_T70·z_T70 의 몫이 함께 줄어든다.
      즉 바깥 슬롯의 alpha 자유 이득은 **딥 팔의 자리** 가 아니라 **우리 두 팔의 가중** 을 사는 것이다.
      이 가설이 맞으면 v=0 을 유지한 채 r 로 우리 두 팔만 줄여도 같은 이득이 나와야 한다.

    z(w, v, r) = w_out·[(1−w)·z_mate + w·z_deep] + r·(c_aa·z_aa + c_T70·z_T70) + v·z_deep

  ① (w, r) 격자 · v=0 · alpha 자유  →  최대가 (w,v) 격자 · r=1 의 최대 이상인가
  ② v 를 0.60 까지 늘려 alpha 자유 곡선에 내부 정점이 있는지 (격자 끝 인공물 검사, G-EDGE)
  ③ 각 지점의 **실효 계수** alpha·(a_mate, a_deep, a_aa, a_T70) 를 그대로 적는다

⚠️ 이 실험은 c_aa·c_T70 을 움직여 본다.  그것은 **출하 제안이 아니다** — exp_078 이 그 통로의
   잔여를 +0.2546 으로 쟀고(S19·S33), 여기서는 v 축의 이득을 **귀속** 시키려고만 쓴다.
"""
import json, os, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "4"
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

H = "/home/<user>/SungjinPark"
B90, B95 = f"{H}/exp090", f"{H}/exp095"
OOT, OOT3, OOT4 = f"{H}/exp069/oot", f"{H}/exp070/oot3", f"{H}/exp071/out"
EPS, REF = 1e-6, 0.2498069
K = 100000.0 / REF
S5 = [42, 777, 2024, 101, 7]
AA = (42, 7, 2024)
W4 = {"LGB": 0.05, "CAT": 0.68, "NN": 0.15, "LEAF": 0.12}
BO = dict(alpha=1.07911, beta_skew=-0.059151, c_aa=0.25244, c_T70=0.04756,
          w_out=0.7, w_v6_inner=0.6668397489, w_ebm=-0.1, beta_ins=0.03)
BASE_EXPECT = 928.501461058684
lg = lambda p: logit(np.clip(np.asarray(p, np.float64), EPS, 1 - EPS))
pts = lambda b: 100000.0 * (1.0 - b / REF)
t0 = time.time()

tro = pd.read_csv(f"{H}/data/train.csv",
                  usecols=["row_id", "season", "pitcher_id", "control_success"])
tro = tro[tro.season == 2024].sort_values("row_id", kind="stable")
y = tro.control_success.to_numpy(np.float64)
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
DZ = np.mean([np.load(f"{B90}/out/preds/z_K1632_oot23_s{s}.npy") for s in S5], 0)
Z_MATE = ((1 - BO["w_ebm"]) * (BO["w_v6_inner"] * LV + (1 - BO["w_v6_inner"]) * LQ)
          + BO["w_ebm"] * LE + BO["beta_ins"] * V["off"])
OURS = BO["c_aa"] * L_aa + BO["c_T70"] * L_t70


def pin(z, f):
    c = 0.0
    for _ in range(200):
        p = f(z, c)
        g = float(p.mean()) - YB
        if abs(g) < 1e-15:
            break
        c -= g / float((p * (1 - p)).mean())
    return c


F_RC = lambda z, c: expit(z + c)
F_ABD = lambda z, c: expit(BO["alpha"] * z + BO["beta_skew"] * z * z + c)
sc = lambda z: pts(float(np.mean((F_ABD(z, pin(z, F_ABD)) - y) ** 2)))
sc_rc = lambda z: pts(float(np.mean((F_RC(z, pin(z, F_RC)) - y) ** 2)))
root = lambda w, v, r=1.0: (BO["w_out"] * ((1 - w) * Z_MATE + w * DZ)
                            + r * OURS + v * DZ)


def af(z):
    r = minimize_scalar(lambda a: -sc(a * z), bounds=(0.3, 3.0),
                        method="bounded", options={"xatol": 1e-8})
    return float(-r.fun), float(r.x)


z0 = root(0.0, 0.0)
G0 = {"rc": sc_rc(z0), "expected": BASE_EXPECT, "abs_diff": abs(sc_rc(z0) - BASE_EXPECT)}
assert G0["abs_diff"] == 0.0, G0
base_af = af(z0)[0]
print("GATE", G0, "base_af", base_af, flush=True)
R = {"exp_id": "exp_095", "stage": "s1b_decomp", "baseline_gate": G0,
     "baseline_alpha_free_abd": base_af,
     "surface": "stage1 과 같은 면 + r 로 우리 두 팔(c_aa·z_aa + c_T70·z_T70)만 척도 조절",
     "🔴_not_a_ship_proposal": ("r 은 귀속 실험용이다.  c_aa·c_T70 재적합은 exp_078 이 "
                                "잔여 +0.2546 으로 이미 닫은 통로다(S19·S33).")}


def rec(w, v, r, tag=""):
    z = root(w, v, r)
    g, a = af(z)
    a_m = a * BO["w_out"] * (1 - w)
    a_d = a * (BO["w_out"] * w + v)
    return {"tag": tag, "w": w, "v": v, "r": r,
            "gain_alpha_free": g - base_af, "alpha_star": a,
            "gain_fixed_alpha": sc(z) - sc(z0),
            "eff_mate": a_m, "eff_deep": a_d,
            "eff_aa": a * r * BO["c_aa"], "eff_T70": a * r * BO["c_T70"],
            "ratio_deep_over_mate": a_d / a_m if a_m else None,
            "ratio_aa_over_mate": (a * r * BO["c_aa"]) / a_m if a_m else None}


# ① (w, r) 격자 · v = 0
WG = [round(x, 4) for x in np.arange(0.0, 0.5001, 0.02)]
RG = [round(x, 4) for x in np.arange(0.40, 1.3001, 0.05)]
grid_wr = []
for w in WG:
    for r in RG:
        grid_wr.append(rec(w, 0.0, r))
    print(f"  wr w={w:.2f} {time.time()-t0:.0f}s", flush=True)
best_wr = max(grid_wr, key=lambda d: d["gain_alpha_free"])
R["grid_w_r_v0"] = grid_wr
R["best_w_r_v0"] = best_wr

# ② v 를 0.60 까지 (격자 끝 인공물 검사)
long_v = []
for w in (0.00, 0.06, 0.20, 0.36):
    for v in [round(x, 4) for x in np.arange(0.0, 0.6001, 0.04)]:
        long_v.append(rec(w, v, 1.0))
    print(f"  longv w={w:.2f} {time.time()-t0:.0f}s", flush=True)
R["long_v_scan_r1"] = long_v
by_w = {}
for d in long_v:
    by_w.setdefault(d["w"], []).append(d)
R["long_v_argmax_per_w"] = {
    str(w): max(v, key=lambda d: d["gain_alpha_free"]) for w, v in by_w.items()}

# ③ 대조 지점 넷
R["anchors"] = {
    "ship_w0.20_v0_r1": rec(0.20, 0.0, 1.0, "출하점"),
    "fixed_alpha_argmax_w0.36_v0_r1": rec(0.36, 0.0, 1.0, "고정alpha 2자유도 argmax"),
    "alpha_free_argmax_w0.06_v0.30_r1": rec(0.06, 0.30, 1.0, "alpha자유 2자유도 argmax"),
    "best_w_r_v0": best_wr}
R["ATTRIBUTION"] = {
    "alpha_free_2dof_max_r1": R["anchors"]["alpha_free_argmax_w0.06_v0.30_r1"]["gain_alpha_free"],
    "alpha_free_v0_max_r_free": best_wr["gain_alpha_free"],
    "v_axis_adds_beyond_arm_reweight":
        R["anchors"]["alpha_free_argmax_w0.06_v0.30_r1"]["gain_alpha_free"]
        - best_wr["gain_alpha_free"],
    "note": ("음수/0 이면 v 축의 alpha 자유 이득은 '우리 두 팔을 줄이는 것'이 이미 사는 것이고 "
             "바깥 슬롯 자체의 기여가 아니다.")}
R["wall_sec"] = round(time.time() - t0, 1)
json.dump(R, open(f"{B95}/out/s1b_decomp.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in R.items()
                  if k not in ("grid_w_r_v0", "long_v_scan_r1")},
                 ensure_ascii=False, indent=1))
