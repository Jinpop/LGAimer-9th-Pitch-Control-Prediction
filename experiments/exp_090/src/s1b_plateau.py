"""exp_090 stage 1b — 고원의 폭과 argmax 의 신뢰도.

물음 셋:
  ① 1632 의 +11.05 가 이웃(1088·1360·2176)보다 **정말** 큰가, 아니면 시드 잡음인가
     → 같은 행 위의 **짝지은** 투수 블록 부트스트랩으로 두 지점의 차를 직접 잰다.
  ② 지점 간 변동 중 얼마가 시드 잡음인가
     → 각 지점에서 **1시드 딥**으로 잰 w=0.20 이득 5개의 표준편차, 그리고
       5시드평균 통계의 leave-one-seed-out 잭나이프 표준오차.
  ③ 고원의 경계는 어디인가 → 최고점 대비 유의하게 낮아지는 첫 지점.
"""
import json, os, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "6"
import numpy as np
import pandas as pd
from scipy.special import expit, logit

H = "/home/<user>/SungjinPark"
B89, B90 = f"{H}/exp089", f"{H}/exp090"
OOT, OOT3, OOT4 = f"{H}/exp069/oot", f"{H}/exp070/oot3", f"{H}/exp071/out"
EPS, REF = 1e-6, 0.2498069
K = 100000.0 / REF
S5 = [42, 777, 2024, 101, 7]
AA = (42, 7, 2024)
W4 = {"LGB": 0.05, "CAT": 0.68, "NN": 0.15, "LEAF": 0.12}
BO = dict(alpha=1.07911, beta_skew=-0.059151, c_aa=0.25244, c_T70=0.04756,
          w_out=0.7, w_v6_inner=0.6668397489, w_ebm=-0.1, beta_ins=0.03)
LB_W = 0.20
lg = lambda p: logit(np.clip(np.asarray(p, np.float64), EPS, 1 - EPS))
pts = lambda b: 100000.0 * (1.0 - b / REF)
t0 = time.time()

REUSE = {544: (B89, "M_oot23_s{s}"), 720: (B89, "S720_oot23_s{s}"),
         816: (B89, "E3_oot23_s{s}"), 1088: (B89, "S1088_oot23_s{s}")}
STEPS = sorted(list(REUSE) + [1360, 1632, 2176, 2720, 3264, 4080, 5440, 8160])
Z = {}
for n in STEPS:
    if n in REUSE:
        root_, pat = REUSE[n]
        Z[n] = [np.load(f"{root_}/out/preds/z_{pat.format(s=s)}.npy") for s in S5]
    else:
        Z[n] = [np.load(f"{B90}/out/preds/z_K{n}_oot23_s{s}.npy") for s in S5]

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
F = lambda z, c: expit(BO["alpha"] * z + BO["beta_skew"] * z * z + c)


def pin(z):
    c = 0.0
    for _ in range(200):
        p = F(z, c)
        g = float(p.mean()) - YB
        if abs(g) < 1e-15:
            break
        c -= g / float((p * (1 - p)).mean())
    return c


def probs(dz, w=LB_W):
    z = BO["w_out"] * ((1 - w) * Z_MATE + w * dz) + BO["c_aa"] * L_aa + BO["c_T70"] * L_t70
    return F(z, pin(z))


base_p = probs(np.zeros_like(y), 0.0)
BASE = pts(float(np.mean((base_p - y) ** 2)))
rng = np.random.default_rng(20260824)
uq, inv = np.unique(pit, return_inverse=True)
nb = len(uq)
order = np.argsort(inv, kind="stable")
st = np.searchsorted(inv[order], np.arange(nb))
en = np.append(st[1:], len(order))
BSIZE = (en - st).astype(np.float64)
PICKS = np.stack([rng.integers(0, nb, nb) for _ in range(1000)])
IDX = [order[st[j]:en[j]] for j in range(nb)]


def blockstat(d):
    bs_ = np.array([d[IDX[j]].sum() for j in range(nb)])
    return bs_[PICKS].sum(1) / BSIZE[PICKS].sum(1)


R = {"exp_id": "exp_090", "stage": "s1b_plateau", "baseline_abd_w0": BASE,
     "note": "abd 관례·w=0.20 고정. 짝지은 블록 부트스트랩은 같은 1000개 추출을 쓴다."}
SQ = {}
for n in STEPS:
    dz = np.mean(Z[n], 0)
    SQ[n] = (probs(dz) - y) ** 2
GAIN = {n: pts(float(SQ[n].mean())) - BASE for n in STEPS}
R["gain_at_w0.20"] = {str(n): GAIN[n] for n in STEPS}

# ① 지점 쌍 비교 (짝지은 블록 부트스트랩)
peak = max(STEPS, key=lambda n: GAIN[n])
R["peak_steps"] = peak
R["paired_vs_peak"] = {}
for n in STEPS:
    if n == peak:
        continue
    d = SQ[n] - SQ[peak]                     # 양수면 peak 가 낫다
    bs = blockstat(d)
    se = float(bs.std(ddof=1))
    R["paired_vs_peak"][str(n)] = {
        "delta_gain_pts_peak_minus_this": (GAIN[peak] - GAIN[n]),
        "paired_t": float(d.mean() / se) if se > 0 else None,
        "p_share_peak_worse": float((bs < 0).mean())}

# ② 시드 잡음
R["seed_noise"] = {}
for n in STEPS:
    per = [pts(float(np.mean((probs(z) - y) ** 2))) - BASE for z in Z[n]]
    jk = [pts(float(np.mean((probs(np.mean([Z[n][i] for i in range(5) if i != k], 0))
                             - y) ** 2))) - BASE for k in range(5)]
    jk = np.array(jk)
    R["seed_noise"][str(n)] = {
        "single_seed_gains": per, "single_seed_mean": float(np.mean(per)),
        "single_seed_sd": float(np.std(per, ddof=1)),
        "mean5_gain": GAIN[n],
        "jackknife_se_of_mean5": float(np.sqrt(4 / 5 * np.sum((jk - jk.mean()) ** 2))),
        "loo_gains": jk.tolist()}

# ③ 고원 경계
sig = {k: v for k, v in R["paired_vs_peak"].items() if (v["paired_t"] or 0) >= 2.0}
plateau = sorted([peak] + [int(k) for k in R["paired_vs_peak"]
                           if (R["paired_vs_peak"][k]["paired_t"] or 0) < 2.0
                           and GAIN[int(k)] > 0])
R["plateau"] = {"definition": "최고점과 짝지은 t < 2.0 이고 이득이 양수인 지점",
                "members": plateau,
                "significantly_below_peak_t>=2": sorted(int(k) for k in sig)}
R["wall_sec"] = round(time.time() - t0, 1)
json.dump(R, open(f"{B90}/out/s1b_plateau.json", "w"), ensure_ascii=False, indent=1)
print(f"baseline {BASE:.4f}  peak {peak}")
print("steps   gain5   ss_mean  ss_sd   jk_se | paired_t_vs_peak  p(peak worse)")
for n in STEPS:
    s_ = R["seed_noise"][str(n)]
    pv = R["paired_vs_peak"].get(str(n))
    print("%6d %+8.3f %+8.3f %7.3f %7.3f | %s" % (
        n, GAIN[n], s_["single_seed_mean"], s_["single_seed_sd"],
        s_["jackknife_se_of_mean5"],
        "peak" if pv is None else "%+6.2f   %.3f" % (pv["paired_t"], pv["p_share_peak_worse"])))
print(json.dumps(R["plateau"], ensure_ascii=False))
