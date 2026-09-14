"""exp_080 stage1 — v6 내부 3축의 정직면 이득에서 척도 몫을 뺀다 (S25).  학습 0건.

exp_078 은 v6_CAT_vs_NN 이 정직면 +4.80546 을 준다고 했는데 척도를 안 뺐다.
exp_079 는 v6-Cat 학습량 축에서 +3.03062 중 정보가 +0.96673(31.9%)뿐임을 실측했다.
여기서 같은 검사를 3축 전부에 한다.

척도 제거 방식 세 가지를 전부 낸다:
  ① sd맞춤(S25/exp_079 방식) — 이동 후 v6 합성팔 로짓의 (mean, sd) 를 이동 전에 맞춘다
  ② alpha자유       — z 에 전역 스칼라를 곱해 양쪽에서 각각 최적화한 뒤 비교
                       (정직면에는 alpha 가 없으므로 이것이 alpha 로 살 수 있는 몫의 상한)
  ③ 둘 다
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "6"
import json, time
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

H = "/home/<user>/SungjinPark"
OUT = f"{H}/exp080/out"
os.makedirs(OUT, exist_ok=True)
OOT, OOT3, OOT4 = f"{H}/exp069/oot", f"{H}/exp070/oot3", f"{H}/exp071/out"
EPS, REF = 1e-6, 0.2498069
K = 100000.0 / REF
W4 = {"LGB": 0.05, "CAT": 0.68, "NN": 0.15, "LEAF": 0.12}
AA = (42, 7, 2024)
BO = dict(alpha=1.07911, beta_skew=-0.059151, c_aa=0.25244, c_T70=0.04756,
          w_out=0.7, w_v6_inner=0.6668397489, w_ebm=-0.1, beta_ins=0.03)
BASE_EXPECT = 928.501461058684
t0 = time.time()
R = {"exp_id": "exp_080", "part": "stage1",
     "K_pts_per_brier": K, "ref_brier": REF}


def lg(p):
    return logit(np.clip(np.asarray(p, np.float64), EPS, 1 - EPS))


def pts(b):
    return 100000.0 * (1.0 - b / REF)


V = {k: np.load(p).astype(np.float64) for k, p in {
    "p_lgb": f"{OOT}/p_lgb_oot2024.npy", "p_cat": f"{OOT}/p_cat_oot2024.npy",
    "p_leaf": f"{OOT}/p_leaf_oot2024.npy", "p_nn": f"{OOT4}/p_nn_oot2024.npy",
    "p_q": f"{OOT}/p_q_oot2024.npy", "p_ebm": f"{OOT3}/p_ebm_oot2024.npy",
    "off": f"{OOT}/off_oot2024.npy"}.items()}
R["S14_ranges"] = {n: {"min": float(v.min()), "max": float(v.max()),
                       "mean": float(v.mean()), "sd": float(v.std()),
                       "in_prob_range": bool(v.min() >= 0 and v.max() <= 1)}
                   for n, v in V.items()}
R["S14_verdict"] = ("p_lgb·p_cat·p_leaf·p_nn·p_q·p_ebm 는 전부 [0,1] 안이므로 확률 — "
                    "lg() 를 씌운다.  off 는 로짓 차분이라 음수를 포함하며 씌우지 않는다. "
                    "🔴 배포 프레임의 arms/p_ebm 는 이름과 달리 로짓이지만(S14) OOT 의 "
                    "exp070/oot3/p_ebm_oot2024.npy 는 확률이다 — 같은 이름 다른 물건.")
assert V["p_ebm"].min() >= 0.0 and V["p_ebm"].max() <= 1.0
assert V["off"].min() < 0.0

tro = pd.read_csv(f"{H}/data/train.csv",
                  usecols=["row_id", "season", "control_success"])
tro = tro[tro.season == 2024].sort_values("row_id", kind="stable")
yo = tro.control_success.to_numpy(np.float64)
rid = np.asarray(np.load(f"{OOT}/row_id_oot2024.npy", allow_pickle=True)).astype(str)
assert np.array_equal(rid, tro.row_id.to_numpy().astype(str))
assert len(yo) == 253507
YB = float(yo.mean())

L4 = {n: lg(V[n]) for n in ("p_lgb", "p_cat", "p_leaf", "p_nn")}
LV0 = (W4["LGB"] * L4["p_lgb"] + W4["CAT"] * L4["p_cat"]
       + W4["NN"] * L4["p_nn"] + W4["LEAF"] * L4["p_leaf"])
LQ, LE = lg(V["p_q"]), lg(V["p_ebm"])
L_aa = np.mean([lg(np.load(f"{H}/experiments/exp_049/pred/"
                           f"pred_A1_u2024_seed{s}_at1400.npy")) for s in AA], axis=0)
L_t70 = np.mean([lg(np.load(f"{H}/experiments/exp_055/pred/"
                            f"pred_T70_u2024_seed{s}_at1400.npy")) for s in AA], axis=0)
NEW = {"v6_CAT_vs_NN": L4["p_cat"] - L4["p_nn"],
       "v6_CAT_vs_LEAF": L4["p_cat"] - L4["p_leaf"],
       "v6_CAT_vs_LGB": L4["p_cat"] - L4["p_lgb"]}


def rc(z):
    c = 0.0
    for _ in range(80):
        p = expit(z + c)
        mu = float(p.mean())
        if abs(mu - YB) < 1e-14:
            break
        d = (mu - YB) / float((p * (1 - p)).mean())
        c -= d
        if abs(d) < 1e-14:
            break
    return c


def zof(lv):
    return (BO["w_out"] * ((1 - BO["w_ebm"])
                           * (BO["w_v6_inner"] * lv + (1 - BO["w_v6_inner"]) * LQ)
                           + BO["w_ebm"] * LE + BO["beta_ins"] * V["off"])
            + BO["c_aa"] * L_aa + BO["c_T70"] * L_t70)


def pts_of_z(z):
    p = expit(z + rc(z))
    return pts(float(np.mean((p - yo) ** 2)))


def sfp(lv):
    return pts_of_z(zof(lv))


def pts_alpha_free(lv):
    """z 에 전역 스칼라 a 를 곱해 최적화 — 정직면에서 alpha 가 살 수 있는 몫."""
    z = zof(lv)
    r = minimize_scalar(lambda a: -pts_of_z(a * z), bounds=(0.3, 3.0),
                        method="bounded", options={"xatol": 1e-7})
    return float(-r.fun), float(r.x)


MU0, SD0 = float(LV0.mean()), float(LV0.std())
base = sfp(LV0)
R["base"] = {"honest_surface_pts": base, "expected_exp078": BASE_EXPECT,
             "abs_diff": abs(base - BASE_EXPECT),
             "lv_mean": MU0, "lv_sd": SD0}
assert abs(base - BASE_EXPECT) < 1e-9, f"정직면 기준 불일치 {base}"
base_af, a0 = pts_alpha_free(LV0)
R["base"]["alpha_free_pts"] = base_af
R["base"]["alpha_free_a_star"] = a0
R["base"]["alpha_free_headroom_pts"] = base_af - base
print(f"기준 {base:.9f} (기대 {BASE_EXPECT}) · alpha자유 {base_af:.5f} (a*={a0:.6f})",
      flush=True)


def raw_pts(nm, t):
    return sfp(LV0 + t * NEW[nm])


def sdm_lv(nm, t):
    lv = LV0 + t * NEW[nm]
    mu, sd = float(lv.mean()), float(lv.std())
    return MU0 + (lv - mu) * (SD0 / sd), sd / SD0


def sdm_pts(nm, t):
    lv, _ = sdm_lv(nm, t)
    return sfp(lv)


S1 = {}
GRID = [-0.40, -0.30, -0.20, -0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10,
        0.15, 0.20, 0.25, 0.30, 0.40]
for nm in NEW:
    rr = minimize_scalar(lambda t: -raw_pts(nm, t), bounds=(-0.50, 0.50),
                         method="bounded", options={"xatol": 1e-6})
    t_raw = float(rr.x)
    g_raw = float(-rr.fun) - base
    rs = minimize_scalar(lambda t: -sdm_pts(nm, t), bounds=(-0.50, 0.50),
                         method="bounded", options={"xatol": 1e-6})
    t_sdm = float(rs.x)
    g_sdm_own = float(-rs.fun) - base
    g_sdm_at_raw = sdm_pts(nm, t_raw) - base
    _, sdr = sdm_lv(nm, t_raw)
    af_raw, a_raw = pts_alpha_free(LV0 + t_raw * NEW[nm])
    g_af = af_raw - base_af
    S1[nm] = {
        "raw_argmax_t": t_raw, "raw_gain_pts": g_raw,
        "new_W_CAT": W4["CAT"] + t_raw,
        "sd_ratio_lv_at_raw_argmax": sdr,
        "sd_matched_gain_at_raw_argmax_pts": g_sdm_at_raw,
        "information_share": (g_sdm_at_raw / g_raw) if g_raw != 0 else None,
        "sd_matched_own_argmax_t": t_sdm,
        "sd_matched_own_gain_pts": g_sdm_own,
        "alpha_free_gain_at_raw_argmax_pts": g_af,
        "alpha_free_a_star": a_raw,
        "alpha_free_share": (g_af / g_raw) if g_raw != 0 else None,
        "grid_raw_delta": {f"{t:+.2f}": raw_pts(nm, t) - base for t in GRID},
        "grid_sd_matched_delta": {f"{t:+.2f}": sdm_pts(nm, t) - base for t in GRID},
        "grid_sd_ratio": {f"{t:+.2f}": sdm_lv(nm, t)[1] for t in GRID}}
    print(f"[{nm}] raw t {t_raw:+.5f} 이득 {g_raw:+.5f}점 | sd비 {sdr:.5f} → "
          f"sd맞춤 {g_sdm_at_raw:+.5f}점 (정보 {100*g_sdm_at_raw/g_raw:.1f}%) | "
          f"alpha자유 {g_af:+.5f}점 ({100*g_af/g_raw:.1f}%)", flush=True)
R["stage1_axes"] = S1
R["stage1_note"] = (
    "sd맞춤 = 이동 후 v6 합성팔 로짓 lv 의 (mean, sd) 를 이동 전 값에 맞춘 뒤 다시 잰 이득. "
    "exp_079 가 v6-Cat 팔 로짓에 쓴 것과 같은 산술(mu0 + (lv-mu)*sd0/sd). "
    "alpha자유 = z 에 전역 스칼라 a 를 곱해 t=0 과 t=t* 양쪽에서 각각 최적화한 뒤의 차. "
    "정직면에는 alpha 가 없으므로 이 둘은 '척도로 설명되는 몫'을 위아래에서 재는 두 자다.")
R["wall_sec"] = time.time() - t0
json.dump(R, open(f"{OUT}/s1_stage1.json", "w"), indent=1, ensure_ascii=False)
print(f"\n기록 {OUT}/s1_stage1.json · {R['wall_sec']:.1f}s", flush=True)
