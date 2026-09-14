"""exp_089 stage 3 — 눈금 확정. 딥 팔 판본 4종을 같은 정직면에 꽂아 비교하고
LB 실측 +5.0813180433 과의 비를 낸다."""
import itertools, json, os, time
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
S5 = [42, 777, 2024, 101, 7]
AA = (42, 7, 2024)
W4 = {"LGB": 0.05, "CAT": 0.68, "NN": 0.15, "LEAF": 0.12}
BO = dict(alpha=1.07911, beta_skew=-0.059151, c_aa=0.25244, c_T70=0.04756,
          w_out=0.7, w_v6_inner=0.6668397489, w_ebm=-0.1, beta_ins=0.03)
LB_GAIN, LB_W = 5.0813180433, 0.20
lg = lambda p: logit(np.clip(np.asarray(p, np.float64), EPS, 1 - EPS))
pts = lambda b: 100000.0 * (1.0 - b / REF)
t0 = time.time()

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

mean5 = lambda pat: np.mean([np.load(pat.format(s=s)) for s in S5], 0)
DEEP = {
    "oot23_ours_ep2": mean5(B + "/out/preds/z_M_oot23_s{s}.npy"),
    "oot23_ours_step720": mean5(B + "/out/preds/z_S720_oot23_s{s}.npy"),
    "oot23_ours_ep3_step816": mean5(B + "/out/preds/z_E3_oot23_s{s}.npy"),
    "oot23_ours_step1088": mean5(B + "/out/preds/z_S1088_oot23_s{s}.npy"),
    "le2024_ours_ep2": mean5(B + "/out/preds/z_M_le2024_s{s}.npy"),
    "le2024_mate_ship": mean5(E88 + "/deep_route1_s{s}_logit.npy"),
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


F_RC = lambda z, c: expit(z + c)
F_ABD = lambda z, c: expit(BO["alpha"] * z + BO["beta_skew"] * z * z + c)


def prob(z, conv):
    f = F_RC if conv == "rc" else F_ABD
    return f(z, pin(z, f))


def sc(z, conv):
    return pts(float(np.mean((prob(z, conv) - y) ** 2)))


def root(zm):
    return BO["w_out"] * zm + BO["c_aa"] * L_aa + BO["c_T70"] * L_t70


def alpha_free(z, conv):
    r = minimize_scalar(lambda a: -sc(a * z, conv), bounds=(0.3, 3.0),
                        method="bounded", options={"xatol": 1e-8})
    return float(-r.fun), float(r.x)


rng = np.random.default_rng(20260824)
uq, inv = np.unique(pit, return_inverse=True)
order = np.argsort(inv, kind="stable")
st = np.searchsorted(inv[order], np.arange(len(uq)))
en = np.append(st[1:], len(order))
BOOT = [np.concatenate([order[st[j]:en[j]] for j in rng.integers(0, len(uq), len(uq))])
        for _ in range(1000)]

R = {"exp_id": "exp_089", "stage": "s4_final", "K_pts_per_brier": K,
     "LB_gain_measured": LB_GAIN, "LB_w_deep": LB_W, "BO_constants": BO,
     "surface": ("모든 팔 out-of-time. mate = OOT v6 4팔 + Q + EBM + off, "
                 "ours = AA(exp_049 A1@1400 3시드) + T70(exp_055 3시드) — 둘 다 "
                 "split_A out-of-time 벡터. 딥만 판본을 갈아끼운다."),
     "n_rows": int(len(y)), "y_mean": YB}
z0 = root(Z_MATE)
base = {c: sc(z0, c) for c in ("rc", "abd")}
base_af = {c: alpha_free(z0, c) for c in ("rc", "abd")}
R["baseline_w0"] = {"rc": base["rc"], "abd": base["abd"],
                    "rc_expected": 928.501461058684,
                    "rc_abs_diff": abs(base["rc"] - 928.501461058684),
                    "alpha_free": {c: base_af[c][0] for c in base_af}}
assert R["baseline_w0"]["rc_abs_diff"] < 1e-9

GRID = [round(x, 4) for x in np.arange(0.0, 0.6001, 0.01)]
R["variants"] = {}
for name, dz in DEEP.items():
    ent = {"alone_score": pts(float(np.mean((expit(dz) - y) ** 2))),
           "z_sd": float(dz.std()), "z_mean": float(dz.mean()),
           "rho_vs_z_mate_oot": float(np.corrcoef(dz, Z_MATE)[0, 1])}
    for conv in ("rc", "abd"):
        cur = []
        for w in GRID:
            z = root((1 - w) * Z_MATE + w * dz)
            cur.append({"w": w, "gain": sc(z, conv) - base[conv]})
        g20 = [c for c in cur if abs(c["w"] - LB_W) < 1e-9][0]["gain"]
        z20 = root((1 - LB_W) * Z_MATE + LB_W * dz)
        af20 = alpha_free(z20, conv)[0] - base_af[conv][0]
        d = (prob(z20, conv) - y) ** 2 - (prob(z0, conv) - y) ** 2
        bs = np.array([d[ix].mean() for ix in BOOT])
        se = float(bs.std(ddof=1))
        best = max(cur, key=lambda c: c["gain"])
        ent[conv] = {
            "gain_at_w0.20": g20,
            "gain_at_w0.20_alpha_free": af20,
            "argmax_w": best["w"], "gain_at_argmax": best["gain"],
            "boot_t_at_w0.20": float(-d.mean() / se),
            "boot_gain_ci95_pts": [float(-np.percentile(bs, 97.5) * K),
                                   float(-np.percentile(bs, 2.5) * K)],
            "B_K_mean_delta2": K * float(np.mean(
                (prob(z20, conv) - prob(z0, conv)) ** 2)),
            "curve": cur,
        }
    R["variants"][name] = ent
    print(f"{name:18s} alone {ent['alone_score']:8.2f} rho_mate "
          f"{ent['rho_vs_z_mate_oot']:.4f} | abd w=.20 {ent['abd']['gain_at_w0.20']:+8.3f} "
          f"(afree {ent['abd']['gain_at_w0.20_alpha_free']:+7.3f}, t "
          f"{ent['abd']['boot_t_at_w0.20']:+.2f}, argmax w {ent['abd']['argmax_w']:.2f})",
          flush=True)

P = R["variants"]["oot23_ours_ep2"]
R["STAGE3_CALIBRATION"] = {
    "honest_surface_gain_at_w0.20_abd": P["abd"]["gain_at_w0.20"],
    "honest_surface_gain_at_w0.20_rc": P["rc"]["gain_at_w0.20"],
    "LB_gain": LB_GAIN,
    "TRANSFER_RATIO_surface_over_LB_abd": P["abd"]["gain_at_w0.20"] / LB_GAIN,
    "TRANSFER_RATIO_surface_over_LB_rc": P["rc"]["gain_at_w0.20"] / LB_GAIN,
    "w_star_honest_abd": P["abd"]["argmax_w"],
    "w_star_honest_rc": P["rc"]["argmax_w"],
    "w_star_from_LB_reported": 0.1757,
    "B_honest_abd_at_w0.20": P["abd"]["B_K_mean_delta2"],
    "B_cov_frame_reported": 6.717,
    "robustness_step_matched_720": R["variants"]["oot23_ours_step720"]["abd"]["gain_at_w0.20"],
    "robustness_step_816": R["variants"]["oot23_ours_ep3_step816"]["abd"]["gain_at_w0.20"],
    "robustness_step_1088": R["variants"]["oot23_ours_step1088"]["abd"]["gain_at_w0.20"],
    "TRANSFER_RATIO_step_matched_720_over_LB":
        R["variants"]["oot23_ours_step720"]["abd"]["gain_at_w0.20"] / LB_GAIN,
    "w_star_step_matched_720_abd": R["variants"]["oot23_ours_step720"]["abd"]["argmax_w"],
    "identity_control": ("STEPS089=720 을 le2024 에 걸면 epoch=2 판본과 예측 sha 동일 "
                         "(max|Δ| 0.0) — 스텝 못박기 코드 경로 자체는 아무것도 안 바꾼다"),
    "in_sample_deep_ours_gain_abd": R["variants"]["le2024_ours_ep2"]["abd"]["gain_at_w0.20"],
    "in_sample_deep_mate_gain_abd": R["variants"]["le2024_mate_ship"]["abd"]["gain_at_w0.20"],
    "teammate_reported_matched_f24_gain": 10.20,
}
R["wall_sec"] = round(time.time() - t0, 1)
json.dump(R, open(f"{B}/out/s4_final.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(R["STAGE3_CALIBRATION"], ensure_ascii=False, indent=1))
