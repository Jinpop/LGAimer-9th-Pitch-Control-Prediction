"""exp_103 stage2 — 정직면 OOT 평가.

🔴 면은 exp_097 sB1_rebase.py 의 것을 **글자 그대로** 승계한다(상수·못박기·부트스트랩 rng 포함).
   기준선 26.2698 = 고친 f64 면, 딥 1632스텝, argmax w=0.43. **결함면 13.7143 은 쓰지 않는다.**
   두 재현 게이트가 면이 같음을 증명한다:
     R1  w=0 재중심 점수 == 928.501461058684  (|Δ| < 1e-9)
     R2  옛 딥(94열) 1632 argmax 이득 == 26.269817382706947  (|Δ| < 1e-6)

사전 등록 문턱(S8, 발사 전 확정): 혼합 이득 − 26.2698 >= +2.0 AND 짝지은 투수블록 부트스트랩 t >= 2.0
"""
import itertools, json, os, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "3"
import numpy as np, pandas as pd
from scipy.special import expit, logit

H = "/home/<user>/SungjinPark"
B, B97 = f"{H}/exp103", f"{H}/exp097"
OOT, OOT3, OOT4 = f"{H}/exp069/oot", f"{H}/exp070/oot3", f"{H}/exp071/out"
EPS, REF = 1e-6, 0.2498069
K = 100000.0 / REF
S5 = [42, 777, 2024, 101, 7]
AA = (42, 7, 2024)
W4 = {"LGB": 0.05, "CAT": 0.68, "NN": 0.15, "LEAF": 0.12}
BO = dict(alpha=1.07911, beta_skew=-0.059151, c_aa=0.25244, c_T70=0.04756,
          w_out=0.7, w_v6_inner=0.6668397489, w_ebm=-0.1, beta_ins=0.03)
BASE_EXPECT = 928.501461058684
REF_GAIN = 26.269817382706947
OPER, W_REF = 1632, 0.43
lg = lambda p: logit(np.clip(np.asarray(p, np.float64), EPS, 1 - EPS))
pts = lambda b: 100000.0 * (1.0 - b / REF)
t0 = time.time()

tro = pd.read_csv(f"{H}/data/train.csv",
                  usecols=["row_id", "season", "pitcher_id", "control_success"])
tro = tro[tro.season == 2024].sort_values("row_id", kind="stable")
y = tro.control_success.to_numpy(np.float64)
pit = tro.pitcher_id.to_numpy()
rid = np.asarray(np.load(f"{OOT}/row_id_oot2024.npy", allow_pickle=True)).astype(str)
assert np.array_equal(rid, tro.row_id.to_numpy().astype(str))
assert len(y) == 253507
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


def pin(z, f, target=YB):
    c = 0.0
    for _ in range(200):
        p = f(z, c)
        gg = float(p.mean()) - target
        if abs(gg) < 1e-15:
            break
        c -= gg / float((p * (1 - p)).mean())
    return c


F_RC = lambda z, c: expit(z + c)
F_ABD = lambda z, c: expit(BO["alpha"] * z + BO["beta_skew"] * z * z + c)
prob = lambda z, conv: (F_RC if conv == "rc" else F_ABD)(
    z, pin(z, F_RC if conv == "rc" else F_ABD))
sc = lambda z, conv: pts(float(np.mean((prob(z, conv) - y) ** 2)))
root = lambda zm: BO["w_out"] * zm + BO["c_aa"] * L_aa + BO["c_T70"] * L_t70
sd_match = lambda zn, zr: ((zn - zn.mean()) / zn.std()) * zr.std() + zr.mean()

rng = np.random.default_rng(20260824)
uq, inv = np.unique(pit, return_inverse=True)
nb = len(uq)
order = np.argsort(inv, kind="stable")
st = np.searchsorted(inv[order], np.arange(nb))
en = np.append(st[1:], len(order))
BSIZE = (en - st).astype(np.float64)
PICKS = np.stack([rng.integers(0, nb, nb) for _ in range(1000)])


def boot_t(d):
    bs_ = np.array([d[order[st[j]:en[j]]].sum() for j in range(nb)])
    bs = bs_[PICKS].sum(1) / BSIZE[PICKS].sum(1)
    se = float(bs.std(ddof=1))
    return {"gain_pts": -float(d.mean()) * K, "boot_se_brier": se,
            "t": (-float(d.mean()) / se) if se > 0 else None,
            "ci95_pts": [float(-np.percentile(bs, 97.5) * K),
                         float(-np.percentile(bs, 2.5) * K)]}


z0 = root(Z_MATE)
base_abd, base_rc = sc(z0, "abd"), sc(z0, "rc")
p_base = prob(z0, "abd")
R = {"exp_id": "exp_103", "stage": "s2_eval", "n_rows": int(len(y)),
     "prereg": "혼합 이득 − 26.2698 >= +2.0 AND 짝지은 투수블록 부트스트랩 t >= 2.0 (발사 전 확정)",
     "R1_baseline_rc": {"got": base_rc, "expected": BASE_EXPECT,
                        "abs_diff": abs(base_rc - BASE_EXPECT),
                        "PASS": bool(abs(base_rc - BASE_EXPECT) < 1e-9)},
     "baseline_abd": base_abd}
assert R["R1_baseline_rc"]["PASS"], base_rc

Z_OLD = np.mean([np.load(f"{B97}/out/preds/z_F{OPER}_f64o_s{s}.npy") for s in S5], 0)
NEWTAG = os.environ.get("NEWTAG", "N1632")
Z_NEW = np.mean([np.load(f"{B}/out/preds/z_{NEWTAG}_s{s}.npy") for s in S5], 0)
GRID = [round(x, 4) for x in np.arange(0.0, 0.6001, 0.01)]


def curve(zd):
    return [{"w": w, "gain": sc(root((1 - w) * Z_MATE + w * zd), "abd") - base_abd}
            for w in GRID]


cur_old = curve(Z_OLD)
best_old = max(cur_old, key=lambda c: c["gain"])
R["R2_old_deep_1632"] = {"argmax": best_old, "expected_gain": REF_GAIN,
                         "abs_diff": abs(best_old["gain"] - REF_GAIN),
                         "PASS": bool(abs(best_old["gain"] - REF_GAIN) < 1e-6)}
assert R["R2_old_deep_1632"]["PASS"], best_old
z_ref = root((1 - W_REF) * Z_MATE + W_REF * Z_OLD)
p_ref = prob(z_ref, "abd")
sq_ref = (p_ref - y) ** 2

cur_new = curve(Z_NEW)
best_new = max(cur_new, key=lambda c: c["gain"])
R["new_arm"] = {
    "tag": NEWTAG, "n_seeds": len(S5),
    "alone_recentred_score": sc(Z_NEW, "rc"),
    "old_alone_recentred_score": sc(Z_OLD, "rc"),
    "z_sd": float(Z_NEW.std()),
    "rho_vs_z_mate": float(np.corrcoef(Z_NEW, Z_MATE)[0, 1]),
    "rho_vs_z_old_deep": float(np.corrcoef(Z_NEW, Z_OLD)[0, 1]),
    "old_rho_vs_z_mate": float(np.corrcoef(Z_OLD, Z_MATE)[0, 1]),
    "curve_2arm": cur_new, "argmax_2arm": best_new,
    "gain_sd_matched_at_argmax": sc(sd_match(
        root((1 - best_new["w"]) * Z_MATE + best_new["w"] * Z_NEW), z0), "abd") - base_abd,
}

# 3자유도: (1−w−w2)·Z_MATE + w·Z_OLD + w2·Z_NEW
G3 = [round(x, 4) for x in np.arange(0.0, 0.6001, 0.02)]
tri = []
for w in G3:
    for w2 in G3:
        if w + w2 > 0.75:
            continue
        g = sc(root((1 - w - w2) * Z_MATE + w * Z_OLD + w2 * Z_NEW), "abd") - base_abd
        tri.append({"w_old": w, "w_new": w2, "gain": g})
best3 = max(tri, key=lambda c: c["gain"])
R["tri_arm"] = {"argmax": best3, "n_grid": len(tri),
                "grid": sorted(tri, key=lambda c: -c["gain"])[:25]}


def vs_ref(z):
    return boot_t((prob(z, "abd") - y) ** 2 - sq_ref)


cands = {
    "2arm_new_argmax": root((1 - best_new["w"]) * Z_MATE + best_new["w"] * Z_NEW),
    "2arm_new_w0.43": root((1 - W_REF) * Z_MATE + W_REF * Z_NEW),
    "tri_argmax": root((1 - best3["w_old"] - best3["w_new"]) * Z_MATE
                       + best3["w_old"] * Z_OLD + best3["w_new"] * Z_NEW),
}
R["vs_reference_26.2698"] = {k: vs_ref(z) for k, z in cands.items()}
R["gain_minus_ref"] = {
    "2arm_new_argmax": best_new["gain"] - REF_GAIN,
    "tri_argmax": best3["gain"] - REF_GAIN}
bestk = max(R["gain_minus_ref"], key=lambda k: R["gain_minus_ref"][k])
tt = R["vs_reference_26.2698"][{"2arm_new_argmax": "2arm_new_argmax",
                               "tri_argmax": "tri_argmax"}[bestk]]["t"]
R["VERDICT"] = {"best_variant": bestk,
                "gain_minus_ref": R["gain_minus_ref"][bestk], "t": tt,
                "PASS": bool(R["gain_minus_ref"][bestk] >= 2.0 and tt is not None
                             and tt >= 2.0)}
R["wall_sec"] = round(time.time() - t0, 1)
json.dump(R, open(f"{B}/out/s2_ship.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in R.items()
                  if k not in ("new_arm", "tri_arm")}, ensure_ascii=False, indent=1))
print(json.dumps({k: v for k, v in R["new_arm"].items() if k != "curve_2arm"},
                 ensure_ascii=False, indent=1))
print(json.dumps(R["tri_arm"]["argmax"], ensure_ascii=False))
