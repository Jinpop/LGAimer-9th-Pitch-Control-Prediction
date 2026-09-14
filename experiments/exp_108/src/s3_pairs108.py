"""exp_108 보충 판독 — 짝지은 비교를 **스텝을 맞춰서** 다시 낸다.

s2_mix108.py 의 `귀무_대조` 는 k=1632 고정이라 작은 스케일 지점(자기 argmax 가 816)에게
불리한 자리에서 읽는다. 여기서는 ① 각 지점의 자기 argmax 스텝 ② 공통 816 두 가지로 낸다.
🔴 사전 등록 판정은 s2 의 것이 정본이다. 이 파일은 **진단**이다(문턱을 다시 정하지 않는다).
"""
import hashlib, json, os, time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np
import pandas as pd
from scipy.special import expit, logit

H = "/home/<user>/SungjinPark"
B, B97 = f"{H}/exp108", f"{H}/exp097"
EPS, REF = 1e-6, 0.2498069
K = 100000.0 / REF
S5 = [42, 777, 2024, 101, 7]
AA = (42, 7, 2024)
W4 = {"LGB": 0.05, "CAT": 0.68, "NN": 0.15, "LEAF": 0.12}
BO = dict(alpha=1.07911, beta_skew=-0.059151, c_aa=0.25244, c_T70=0.04756,
          w_out=0.7, w_v6_inner=0.6668397489, w_ebm=-0.1, beta_ins=0.03)
OOT, OOT3, OOT4 = f"{H}/exp069/oot", f"{H}/exp070/oot3", f"{H}/exp071/out"
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
Z_MATE = ((1 - BO["w_ebm"]) * (BO["w_v6_inner"] * LV + (1 - BO["w_v6_inner"]) * lg(V["p_q"]))
          + BO["w_ebm"] * lg(V["p_ebm"]) + BO["beta_ins"] * V["off"])
L_aa = np.mean([lg(np.load(f"{H}/experiments/exp_049/pred/"
                          f"pred_A1_u2024_seed{s}_at1400.npy")) for s in AA], 0)
L_t70 = np.mean([lg(np.load(f"{H}/experiments/exp_055/pred/"
                            f"pred_T70_u2024_seed{s}_at1400.npy")) for s in AA], 0)


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
    m, se = float(d.mean()), float(bs.std(ddof=1))
    return {"gain_pts": -m * K, "boot_se_pts": se * K,
            "t": (-m / se) if se > 0 else None,
            "ci95_pts": [float(-np.percentile(bs, 97.5) * K),
                         float(-np.percentile(bs, 2.5) * K)]}


GRID = [round(x, 4) for x in np.arange(0.0, 0.6001, 0.01)]
z0 = root(Z_MATE)
BASE_ABD = sc(z0, "abd")


def arm(nm, k):
    return np.mean([np.load(f"{B}/out/preds/z_{nm}__k{k}_s{s}.npy") for s in S5], 0)


def best(dz):
    cur = [(w, sc(root((1 - w) * Z_MATE + w * dz), "abd") - BASE_ABD) for w in GRID]
    return max(cur, key=lambda c: c[1])


R = json.load(open(f"{B}/out/s2_mix108.json"))
T = R["표"]
OUT = {"왜": "s2 의 귀무 대조는 k=1632 고정이라 자기 argmax 가 816 인 지점을 불리한 자리에서 "
             "읽는다. 여기서는 자기 argmax 와 공통 816 두 자리에서 다시 낸다. "
             "🔴 사전 등록 판정은 s2 가 정본이고 이것은 진단이다(S8).",
       "쌍": {}}

PAIRS = [("A_u0p125", "N3_identity"), ("A_u0p25", "N3_identity"),
         ("A_u0p06", "N3_identity"), ("A_u0p5", "N3_identity"),
         ("A_u0p125", "N4_A_u0p25"), ("A_u0p25", "N4_A_u0p25"),
         ("N4_A_u0p25", "N3_identity"),
         ("C_bonly_0p25", "N3_identity"), ("C_wonly_0p25", "N3_identity"),
         ("A_u0p25", "C_bonly_0p25"), ("A_u0p25", "C_wonly_0p25")]

for a_, b_ in PAIRS:
    rec = {}
    for mode in ("own_argmax", "common_k816"):
        if mode == "own_argmax":
            ka, kb = T[a_]["own_argmax_step"], T[b_]["own_argmax_step"]
        else:
            ka = kb = 816
            if "816" not in T[a_]["per_step"] or "816" not in T[b_]["per_step"]:
                continue
        za, zb = arm(a_, ka), arm(b_, kb)
        wa, ga = best(za)
        wb, gb = best(zb)
        pa = prob(root((1 - wa) * Z_MATE + wa * za), "abd")
        pb = prob(root((1 - wb) * Z_MATE + wb * zb), "abd")
        rec[mode] = {"a_step": ka, "b_step": kb, "a_w": wa, "b_w": wb,
                     "a_gain": ga, "b_gain": gb, "a_minus_b": ga - gb,
                     "a_alone": sc(za, "rc"), "b_alone": sc(zb, "rc"),
                     "a_rho": float(np.corrcoef(za, Z_MATE)[0, 1]),
                     "b_rho": float(np.corrcoef(zb, Z_MATE)[0, 1]),
                     "paired": boot_t((pa - y) ** 2 - (pb - y) ** 2)}
        d_rho = rec[mode]["a_rho"] - rec[mode]["b_rho"]
        d_al = rec[mode]["a_alone"] - rec[mode]["b_alone"]
        rec[mode]["S103"] = {
            "d_rho": d_rho, "d_alone": d_al,
            "pred_defectfit": -308.92 * d_rho + 0.19533 * d_al,
            "pred_fixedfit": -289.78 * d_rho + 0.186 * d_al,
            "observed": ga - gb,
            "rho_단가_점_per_0.01": (d_al / (d_rho / 0.01)) if abs(d_rho) > 1e-9 else None,
            "🔴": "S103 은 순위 진단이다 — 절대값 편의 +4.483 실측(exp_107)"}
    OUT["쌍"][f"{a_}_minus_{b_}"] = rec
    m = rec.get("own_argmax", {})
    print("%-30s own %+8.3f (t %+6.3f)  k816 %+8.3f (t %+6.3f)" % (
        f"{a_} − {b_}", m.get("a_minus_b", float("nan")),
        m.get("paired", {}).get("t") or float("nan"),
        rec.get("common_k816", {}).get("a_minus_b", float("nan")),
        (rec.get("common_k816", {}).get("paired", {}) or {}).get("t") or float("nan")),
        flush=True)

json.dump(OUT, open(f"{B}/out/s3_pairs108.json", "w"), ensure_ascii=False, indent=1)
print("wrote s3_pairs108.json")
