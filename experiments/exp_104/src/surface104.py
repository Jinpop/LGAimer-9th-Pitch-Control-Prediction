"""exp_104 — 정직면 혼합 화면 공용 모듈. 코드는 `sA_mix104.py` 의 면 블록과 글자 그대로 같다.

(exp_103b `s2_mixeval.py` / exp_102 `sA_mix.py` / exp_097 `sB1_rebase.py` 승계)
불러 쓰는 쪽에서 반드시 `assert abs(BASE_RC - BASE_EXPECT) < 1e-9` 를 확인한다.
"""
import numpy as np
import pandas as pd
from scipy.special import expit, logit

H = "/home/<user>/SungjinPark"
B97, B102 = f"{H}/exp097", f"{H}/exp102"
OOT, OOT3, OOT4 = f"{H}/exp069/oot", f"{H}/exp070/oot3", f"{H}/exp071/out"
EPS, REF = 1e-6, 0.2498069
K = 100000.0 / REF
S5 = [42, 777, 2024, 101, 7]
AA = (42, 7, 2024)
W4 = {"LGB": 0.05, "CAT": 0.68, "NN": 0.15, "LEAF": 0.12}
BO = dict(alpha=1.07911, beta_skew=-0.059151, c_aa=0.25244, c_T70=0.04756,
          w_out=0.7, w_v6_inner=0.6668397489, w_ebm=-0.1, beta_ins=0.03)
BASE_EXPECT = 928.501461058684
PREREG_BASELINE = 26.2698
PREREG_DELTA = 2.5
PREREG_T = 2.0
lg = lambda p: logit(np.clip(np.asarray(p, np.float64), EPS, 1 - EPS))
pts = lambda b: 100000.0 * (1.0 - b / REF)

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

Z0 = root(Z_MATE)
BASE_RC, BASE_ABD = sc(Z0, "rc"), sc(Z0, "abd")
assert abs(BASE_RC - BASE_EXPECT) < 1e-9, BASE_RC

Z_DEEP = np.mean([np.load(f"{B97}/out/preds/z_F1632_f64o_s{s}.npy") for s in S5], 0)

_rng = np.random.default_rng(20260824)
_uq, _inv = np.unique(pit, return_inverse=True)
NB = len(_uq)
ORDER = np.argsort(_inv, kind="stable")
ST = np.searchsorted(_inv[ORDER], np.arange(NB))
EN = np.append(ST[1:], len(ORDER))
BSIZE = (EN - ST).astype(np.float64)
PICKS = np.stack([_rng.integers(0, NB, NB) for _ in range(1000)])


def boot_t(d):
    bs_ = np.array([d[ORDER[ST[j]:EN[j]]].sum() for j in range(NB)])
    bs = bs_[PICKS].sum(1) / BSIZE[PICKS].sum(1)
    m = float(d.mean())
    se = float(bs.std(ddof=1))
    return {"gain_pts": -m * K, "boot_se_pts": se * K,
            "t": (-m / se) if se > 0 else None,
            "ci95_pts": [float(-np.percentile(bs, 97.5) * K),
                         float(-np.percentile(bs, 2.5) * K)]}
