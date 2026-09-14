"""exp_097b stageB — 정직면 재판독 (독립 구현).

면은 exp_090 s1_curve.py 의 정의를 그대로 쓰되 코드는 새로 쓴다.
게이트: w=0 · rc 관례가 928.501461058684 를 abs_diff < 1e-9 로 재현해야 한다.

두 면을 나란히 잰다.
  def : 딥 팔이 exp_089 결함 입력(X_oot23, f32 표준화 결함) 위에서 적합된 판본
  fix : 같은 구조·같은 시드가 f64 수리 입력(X_f64o) 위에서 적합된 판본
"""
import hashlib
import itertools
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

H = "/home/<user>/SungjinPark"
OOT, OOT3, OOT4 = f"{H}/exp069/oot", f"{H}/exp070/oot3", f"{H}/exp071/out"
EPS, REF = 1e-6, 0.2498069
K = 100000.0 / REF
S5 = [42, 777, 2024, 101, 7]
AA = (42, 7, 2024)
W4 = {"LGB": 0.05, "CAT": 0.68, "NN": 0.15, "LEAF": 0.12}
BO = dict(alpha=1.07911, beta_skew=-0.059151, c_aa=0.25244, c_T70=0.04756,
          w_out=0.7, w_v6_inner=0.6668397489, w_ebm=-0.1, beta_ins=0.03)
BASE_EXPECT = 928.501461058684
CE_LB_GAIN, CE_W = 4.8280402096, 0.20
W_DEPLOY = 0.40                # 🔴 부모 CH 의 mate/model/blend.json 값 (0.20 은 CE 시절)
DEPLOY_STEPS_OOT = 1632        # CH 배포 2160스텝의 OOT 대응 지점
lgf = lambda p: logit(np.clip(np.asarray(p, np.float64), EPS, 1 - EPS))
pts = lambda b: 100000.0 * (1.0 - b / REF)

# ── 면 재구성 ──────────────────────────────────────────────────────────
tro = pd.read_csv(f"{H}/data/train.csv",
                  usecols=["row_id", "season", "pitcher_id", "control_success"])
tro = tro[tro.season == 2024].sort_values("row_id", kind="stable")
y = tro.control_success.to_numpy(np.float64)
pit = tro.pitcher_id.to_numpy()
assert len(y) == 253507, len(y)
YB = float(y.mean())
V = {k: np.load(p).astype(np.float64) for k, p in {
    "p_lgb": f"{OOT}/p_lgb_oot2024.npy", "p_cat": f"{OOT}/p_cat_oot2024.npy",
    "p_leaf": f"{OOT}/p_leaf_oot2024.npy", "p_nn": f"{OOT4}/p_nn_oot2024.npy",
    "p_q": f"{OOT}/p_q_oot2024.npy", "p_ebm": f"{OOT3}/p_ebm_oot2024.npy",
    "off": f"{OOT}/off_oot2024.npy"}.items()}
LV = sum(W4[t] * lgf(V[k]) for t, k in
         (("LGB", "p_lgb"), ("CAT", "p_cat"), ("NN", "p_nn"), ("LEAF", "p_leaf")))
LQ, LE = lgf(V["p_q"]), lgf(V["p_ebm"])
L_aa = np.mean([lgf(np.load(f"{H}/experiments/exp_049/pred/"
                            f"pred_A1_u2024_seed{s}_at1400.npy")) for s in AA], 0)
L_t70 = np.mean([lgf(np.load(f"{H}/experiments/exp_055/pred/"
                             f"pred_T70_u2024_seed{s}_at1400.npy")) for s in AA], 0)
Z_MATE = ((1 - BO["w_ebm"]) * (BO["w_v6_inner"] * LV + (1 - BO["w_v6_inner"]) * LQ)
          + BO["w_ebm"] * LE + BO["beta_ins"] * V["off"])


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
prob = lambda z, conv: (F_RC if conv == "rc" else F_ABD)(
    z, pin(z, F_RC if conv == "rc" else F_ABD))
sc = lambda z, conv: pts(float(np.mean((prob(z, conv) - y) ** 2)))
root = lambda zm: BO["w_out"] * zm + BO["c_aa"] * L_aa + BO["c_T70"] * L_t70

z0 = root(Z_MATE)
base = {c: sc(z0, c) for c in ("rc", "abd")}
base_p = {c: prob(z0, c) for c in ("rc", "abd")}
GATE = abs(base["rc"] - BASE_EXPECT)
assert GATE < 1e-9, ("기준점 게이트 실패", base["rc"], GATE)

# ── 부트스트랩 (exp_090 과 같은 시드·같은 추출 순서) ────────────────────
rng = np.random.default_rng(20260824)
uq, inv = np.unique(pit, return_inverse=True)
nb = len(uq)
order = np.argsort(inv, kind="stable")
st = np.searchsorted(inv[order], np.arange(nb))
en = np.append(st[1:], len(order))
BSIZE = (en - st).astype(np.float64)
PICKS = np.stack([rng.integers(0, nb, nb) for _ in range(1000)])


def boot_t(pnew, pref):
    d = (pnew - y) ** 2 - (pref - y) ** 2
    bs_ = np.array([d[order[st[j]:en[j]]].sum() for j in range(nb)])
    bs = bs_[PICKS].sum(1) / BSIZE[PICKS].sum(1)
    se = float(bs.std(ddof=1))
    obs = float(d.mean())
    return {"gain_pts": -obs * K, "t": float(-obs / se) if se > 0 else None,
            "ci95_pts": [float(-np.percentile(bs, 97.5) * K),
                         float(-np.percentile(bs, 2.5) * K)]}


# ── 딥 예측 적재 ───────────────────────────────────────────────────────
DEF_SRC = {544: (f"{H}/exp089", "M_oot23_s{s}"),
           720: (f"{H}/exp089", "S720_oot23_s{s}"),
           816: (f"{H}/exp089", "E3_oot23_s{s}"),
           1088: (f"{H}/exp089", "S1088_oot23_s{s}")}
for n in (1360, 1632, 2176, 2720, 3264, 4080):
    DEF_SRC[n] = (f"{H}/exp090", "K%d_oot23_s{s}" % n)
FIX_DIR = f"{H}/exp097/out/preds"


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


SHA = {}
DZ = {"def": {}, "fix": {}}
for n, (rt, pat) in sorted(DEF_SRC.items()):
    zs = []
    for s in S5:
        p = f"{rt}/out/preds/z_{pat.format(s=s)}.npy"
        z = np.load(p)
        SHA["def|%d|%d" % (n, s)] = sha_arr(z)
        zs.append(z.astype(np.float64))
    DZ["def"][n] = np.mean(zs, 0)
for n in sorted(DEF_SRC):
    ps = [f"{FIX_DIR}/z_F{n}_f64o_s{s}.npy" for s in S5]
    if not all(os.path.exists(p) for p in ps):
        continue
    zs = []
    for s, p in zip(S5, ps):
        z = np.load(p)
        SHA["fix|%d|%d" % (n, s)] = sha_arr(z)
        rec = json.load(open(f"{H}/exp097/out/fit_F{n}_f64o_s{s}.json"))["z_sha256"]
        assert sha_arr(z) == rec, ("fix 예측 sha 불일치", n, s)
        zs.append(z.astype(np.float64))
    DZ["fix"][n] = np.mean(zs, 0)

GRID = [round(x, 4) for x in np.arange(0.0, 0.7001, 0.01)]
R = {"exp_id": "exp_097b", "stage": "sB_indep", "K": K,
     "baseline_rc": base["rc"], "baseline_rc_expected": BASE_EXPECT,
     "baseline_gate_abs_diff": GATE, "baseline_abd": base["abd"],
     "n_rows": int(len(y)), "y_mean": YB,
     "w_deployed_CH": W_DEPLOY, "deploy_steps_oot_equiv": DEPLOY_STEPS_OOT,
     "n_pred_sha_recorded": len(SHA), "surfaces": {}}

for surf in ("def", "fix"):
    out = {}
    for n, dz in sorted(DZ[surf].items()):
        cur = {}
        for w in GRID:
            cur[w] = sc(root((1 - w) * Z_MATE + w * dz), "abd") - base["abd"]
        wbest = max(cur, key=lambda w: cur[w])
        out[n] = {"alone_recentred": sc(dz, "rc"),
                  "rho_vs_z_mate": float(np.corrcoef(dz, Z_MATE)[0, 1]),
                  "z_sd": float(dz.std()),
                  "gain_w0.20": cur[0.20], "gain_w0.40": cur[0.40],
                  "argmax_w": wbest, "argmax_gain": cur[wbest],
                  "curve": {str(w): cur[w] for w in GRID}}
    R["surfaces"][surf] = out

# ── 배포점 대비 사전등록 판정 ───────────────────────────────────────────
dep = R["surfaces"]["fix"][DEPLOY_STEPS_OOT]["gain_w0.40"]
p_dep = prob(root((1 - W_DEPLOY) * Z_MATE + W_DEPLOY * DZ["fix"][DEPLOY_STEPS_OOT]), "abd")
qual = []
best_alt = None
for n, dz in sorted(DZ["fix"].items()):
    for w in GRID:
        g = R["surfaces"]["fix"][n]["curve"][str(w)]
        if best_alt is None or g > best_alt[2]:
            best_alt = (n, w, g)
        if g >= dep + 2.0:
            pn = prob(root((1 - w) * Z_MATE + w * dz), "abd")
            b = boot_t(pn, p_dep)
            qual.append({"steps": n, "w": w, "gain": g,
                         "delta_vs_deployed": g - dep, "boot_t_vs_deployed": b["t"]})
R["deployed_point_fix"] = {"steps": DEPLOY_STEPS_OOT, "w": W_DEPLOY, "gain": dep}
R["best_point_fix"] = {"steps": best_alt[0], "w": best_alt[1], "gain": best_alt[2],
                       "delta_vs_deployed": best_alt[2] - dep}
bb = boot_t(prob(root((1 - best_alt[1]) * Z_MATE + best_alt[1] * DZ["fix"][best_alt[0]]),
                 "abd"), p_dep)
R["best_point_fix"]["boot_t_vs_deployed"] = bb["t"]
R["best_point_fix"]["boot_ci95_pts"] = bb["ci95_pts"]
R["PREREG"] = {"rule": "정직면 이득 >= 배포점 + 2.0 AND 짝지은 투수블록 부트스트랩 t >= 2.0",
               "n_qualifying": len(qual), "qualifying": qual,
               "PASS": bool(qual)}

# ── CE 앵커: 같은 이동을 두 면에서 다시 재 전이비를 새로 유도 ─────────────
def delta_pair(surf, lo, hi, w):
    a = sc(root((1 - w) * Z_MATE + w * DZ[surf][lo]), "abd") - base["abd"]
    b = sc(root((1 - w) * Z_MATE + w * DZ[surf][hi]), "abd") - base["abd"]
    return b - a


ce = {}
for lo, hi in [(720, 1632), (544, 1632)]:
    d_def = delta_pair("def", lo, hi, CE_W)
    d_fix = delta_pair("fix", lo, hi, CE_W) if (lo in DZ["fix"] and hi in DZ["fix"]) else None
    ce["%d->%d" % (lo, hi)] = {
        "delta_def_w0.20": d_def, "delta_fix_w0.20": d_fix,
        "LB_gain": CE_LB_GAIN,
        "transfer_def": CE_LB_GAIN / d_def if d_def else None,
        "transfer_fix": CE_LB_GAIN / d_fix if d_fix else None}
R["CE_anchor"] = {"recorded_S111_delta_def": 10.102, "recorded_S111_transfer": 0.4779,
                  "pairs": ce}
tr = ce["720->1632"]["transfer_fix"]
R["LB_prediction_for_best_move"] = {
    "honest_delta_pts": R["best_point_fix"]["delta_vs_deployed"],
    "transfer_ratio_used": tr,
    "predicted_LB_pts": R["best_point_fix"]["delta_vs_deployed"] * tr,
    "LB_detection_limit": 1.4,
    "ABOVE_DETECTION_LIMIT": bool(R["best_point_fix"]["delta_vs_deployed"] * tr >= 1.4)}
R["pred_sha"] = SHA
json.dump(R, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in R.items() if k not in ("pred_sha", "surfaces")},
                 ensure_ascii=False, indent=1))
for surf in ("def", "fix"):
    print("=== surface", surf)
    for n, e in sorted(R["surfaces"][surf].items()):
        print("  steps=%5d alone=%9.3f rho=%.6f  w20=%8.4f w40=%8.4f  argmax w=%.2f gain=%8.4f"
              % (n, e["alone_recentred"], e["rho_vs_z_mate"], e["gain_w0.20"],
                 e["gain_w0.40"], e["argmax_w"], e["argmax_gain"]))
