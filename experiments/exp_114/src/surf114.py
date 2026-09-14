"""exp_114 — 정직면 혼합 화면. exp_113 f4_eval.py / exp_097 sB1_rebase.py 를 **글자 그대로** 승계한다.
바꾼 것 없음 — 재사용을 위해 모듈로 뺐을 뿐이다. 게이트가 그것을 증명한다:
  G1 baseline_rc == 928.501461058684
  G2 bundle0 @1632 혼합 argmax gain == 26.269817382706947  (exp_097 sB_indep best_point_fix)
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "3"
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import expit, logit  # noqa: E402

H = "/home/<user>/SungjinPark"
B = f"{H}/exp114"
OOT, OOT3, OOT4 = f"{H}/exp069/oot", f"{H}/exp070/oot3", f"{H}/exp071/out"
EPS, REF = 1e-6, 0.2498069
K = 100000.0 / REF
BUNDLE = {0: [42, 777, 2024, 101, 7], 1: [11, 13, 17, 19, 23], 2: [29, 31, 37, 41, 43],
          3: [47, 53, 59, 61, 67], 4: [71, 73, 79, 83, 89]}
AA = (42, 7, 2024)
W4 = {"LGB": 0.05, "CAT": 0.68, "NN": 0.15, "LEAF": 0.12}
BO = dict(alpha=1.07911, beta_skew=-0.059151, c_aa=0.25244, c_T70=0.04756,
          w_out=0.7, w_v6_inner=0.6668397489, w_ebm=-0.1, beta_ins=0.03)
BASE_EXPECT = 928.501461058684
REF_GAIN_1632_B0 = 26.269817382706947      # exp_097 sB_indep best_point_fix
CKPTS = [544, 720, 816, 952, 1088, 1224, 1360, 1496, 1632, 1768, 1904, 2040,
         2160, 2176, 2312, 2448, 2720, 2992, 3264, 3536, 3808, 4080, 4352]
GRID = [round(x, 4) for x in np.arange(0.0, 0.6001, 0.01)]
TRANSFER = 0.34817768252705894             # S111 정정판 (exp_097 CE_anchor.transfer_fix 720->1632)
LB_FLOOR = 1.4

lg = lambda p: logit(np.clip(np.asarray(p, np.float64), EPS, 1 - EPS))
pts = lambda b: 100000.0 * (1.0 - b / REF)


def build():
    tro = pd.read_csv(f"{H}/data/train.csv",
                      usecols=["row_id", "season", "pitcher_id", "control_success"])
    tro = tro[tro.season == 2024].sort_values("row_id", kind="stable")
    y = tro.control_success.to_numpy(np.float64)
    pit = tro.pitcher_id.to_numpy()
    rid = np.asarray(np.load(f"{OOT}/row_id_oot2024.npy", allow_pickle=True)).astype(str)
    assert np.array_equal(rid, tro.row_id.to_numpy().astype(str))
    assert len(y) == 253507
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
    return y, pit, Z_MATE, L_aa, L_t70


class Surface:
    def __init__(self):
        self.y, self.pit, self.Z_MATE, self.L_aa, self.L_t70 = build()
        self.YB = float(self.y.mean())
        self.z0 = self.root(self.Z_MATE)
        self.base_abd, self.base_rc = self.sc(self.z0, "abd"), self.sc(self.z0, "rc")

    def root(self, zm):
        return BO["w_out"] * zm + BO["c_aa"] * self.L_aa + BO["c_T70"] * self.L_t70

    def _pin(self, z, f):
        c = 0.0
        for _ in range(200):
            p = f(z, c)
            g = float(p.mean()) - self.YB
            if abs(g) < 1e-15:
                break
            c -= g / float((p * (1 - p)).mean())
        return c

    def prob(self, z, conv):
        f = (lambda z_, c: expit(z_ + c)) if conv == "rc" else (
            lambda z_, c: expit(BO["alpha"] * z_ + BO["beta_skew"] * z_ * z_ + c))
        return f(z, self._pin(z, f))

    def sc(self, z, conv):
        return pts(float(np.mean((self.prob(z, conv) - self.y) ** 2)))

    def curve(self, zd):
        cur = {w: self.sc(self.root((1 - w) * self.Z_MATE + w * zd), "abd") - self.base_abd
               for w in GRID}
        wb = max(cur, key=lambda w: cur[w])
        return cur, {"w": wb, "gain": cur[wb]}

    def sd_match(self, zn):
        return ((zn - zn.mean()) / zn.std()) * self.z0.std() + self.z0.mean()


def zbundle(tagf, b, k):
    """tagf(seed) -> tag 접두. 5시드 로짓 평균."""
    return np.mean([np.load(f"{B}/out/preds/z_{tagf(s)}__k{k}.npy") for s in BUNDLE[b]], 0)


def bootstrapper(pit, seed=20260824, nrep=1000):
    rng = np.random.default_rng(seed)
    uq, inv = np.unique(pit, return_inverse=True)
    nb = len(uq)
    order = np.argsort(inv, kind="stable")
    st = np.searchsorted(inv[order], np.arange(nb))
    en = np.append(st[1:], len(order))
    BSIZE = (en - st).astype(np.float64)
    PICKS = np.stack([rng.integers(0, nb, nb) for _ in range(nrep)])

    def boot(d):
        bs_ = np.array([d[order[st[j]:en[j]]].sum() for j in range(nb)])
        bs = bs_[PICKS].sum(1) / BSIZE[PICKS].sum(1)
        se = float(bs.std(ddof=1))
        return {"gain_pts": -float(d.mean()) * K, "boot_se_pts": se * K,
                "boot_t": (-float(d.mean()) / se) if se > 0 else None,
                "ci95_pts": [float(-np.percentile(bs, 97.5) * K),
                             float(-np.percentile(bs, 2.5) * K)]}
    return boot
