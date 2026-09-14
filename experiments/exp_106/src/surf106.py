"""exp_106 공용 면 — exp_104 `surface104.py` 의 면 블록을 글자 그대로 승계한다.

달라지는 것은 **딥 팔이 이미 들어가 있다는 것** 하나다:
exp_104 는 `Z_MATE`(딥 없음)를 기준선으로 쓰고 `w` 를 격자로 훑었다.
exp_106 은 배포 상수 7개를 전부 변수로 열고, 그중 하나가 `w_deep` 이다.

산술은 출하 패키지에서 읽어 확정했다 (submit_CR_q12land.zip, 2026-08-24):
  mate/script.py:95-128
     base = w_v6_inner*lg(p_v6) + (1-w_v6_inner)*lg(p_q)
     z    = (1-w_ebm)*base + w_ebm*p_ebm            # p_ebm 은 이미 로짓 (S14)
     z    = z + beta_ins*off
     z    = (1-w_deep)*z + w_deep*lg(p_deep)
  script.py:410-502
     logit_mix = w_v6*lg(p_partner) + c_aa*lg(p_aa) + c_T70*lg(p_T70)
     p         = expit(alpha*lm + beta_skew*lm^2 + d_adj + failtype)
  → 루트 `w_v6`(=여기서 w_out) 0.7 · c_aa 0.25244 · c_T70 0.04756 · alpha 1.07911
  → mate 안쪽 `w_v6` 0.6668397489 · w_ebm -0.1 · w_deep 0.4

면이 안 담는 것(승계된 관례, 지어내지 않고 적는다):
  - 루트 `_shift`(축 오프셋 BI/BJ/BK 등)·실패유형 오프셋·mate 자신의 d_adj
  - AA·T70 은 3시드(42/7/2024)뿐 — 배포는 AA 6시드다. OOT 복제본이 3개뿐이다.
  - 딥 팔 OOT 복제본은 1632스텝(=배포 2160스텝의 에폭 맞춤, exp_095 주석)
"""
import hashlib
import json
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np                                                   # noqa: E402
import pandas as pd                                                  # noqa: E402
from scipy.special import expit, logit                               # noqa: E402

H = "/home/<user>/SungjinPark"
B90, B97, B102 = f"{H}/exp090", f"{H}/exp097", f"{H}/exp102"
OOT, OOT3, OOT4 = f"{H}/exp069/oot", f"{H}/exp070/oot3", f"{H}/exp071/out"
EPS, REF = 1e-6, 0.2498069
K = 100000.0 / REF
S5 = [42, 777, 2024, 101, 7]
AA = (42, 7, 2024)
W4 = {"LGB": 0.05, "CAT": 0.68, "NN": 0.15, "LEAF": 0.12}
BO = dict(alpha=1.07911, beta_skew=-0.059151, c_aa=0.25244, c_T70=0.04756,
          w_out=0.7, w_v6_inner=0.6668397489, w_ebm=-0.1, beta_ins=0.03)
SHIP_W_DEEP = 0.4                       # mate/model/blend.json  w_deep (후보 CH~CR)
SHIP_V_OUT = 0.0                        # 바깥 딥 슬롯 — 출하 안 됨(exp_095 축)
BASE_EXPECT = 928.501461058684          # w_deep=0 · v=0 · rc 관례 (exp_097 이후 불변)
E097_GAIN_W040 = 26.1514                # S145 (고친 면, 1632스텝, w=0.40)
E097_ARGMAX = (0.43, 26.2698)           # S145 / exp_104 PREREG_BASELINE
lg = lambda p: logit(np.clip(np.asarray(p, np.float64), EPS, 1 - EPS))
pts = lambda b: 100000.0 * (1.0 - b / REF)

# ── 라벨·행 정렬 ──────────────────────────────────────────────────────────
tro = pd.read_csv(f"{H}/data/train.csv",
                  usecols=["row_id", "season", "pitcher_id", "control_success"])
tro = tro[tro.season == 2024].sort_values("row_id", kind="stable")
y = tro.control_success.to_numpy(np.float64)
pit = tro.pitcher_id.to_numpy()
rid = np.asarray(np.load(f"{OOT}/row_id_oot2024.npy", allow_pickle=True)).astype(str)
assert np.array_equal(rid, tro.row_id.to_numpy().astype(str))
assert len(y) == 253507
N = len(y)
YB = float(y.mean())

# ── 팔 로짓 ──────────────────────────────────────────────────────────────
V = {k: np.load(p).astype(np.float64) for k, p in {
    "p_lgb": f"{OOT}/p_lgb_oot2024.npy", "p_cat": f"{OOT}/p_cat_oot2024.npy",
    "p_leaf": f"{OOT}/p_leaf_oot2024.npy", "p_nn": f"{OOT4}/p_nn_oot2024.npy",
    "p_q": f"{OOT}/p_q_oot2024.npy", "p_ebm": f"{OOT3}/p_ebm_oot2024.npy",
    "off": f"{OOT}/off_oot2024.npy"}.items()}
assert all(V[k].min() >= 0 and V[k].max() <= 1 for k in
           ("p_lgb", "p_cat", "p_leaf", "p_nn", "p_q", "p_ebm"))      # S14
assert V["off"].min() < 0.0
LV = sum(W4[t] * lg(V[k]) for t, k in
         (("LGB", "p_lgb"), ("CAT", "p_cat"), ("NN", "p_nn"), ("LEAF", "p_leaf")))
LQ, LE = lg(V["p_q"]), lg(V["p_ebm"])
OFF = V["off"]
AA_EACH = {s: lg(np.load(f"{H}/experiments/exp_049/pred/"
                         f"pred_A1_u2024_seed{s}_at1400.npy")) for s in AA}
T70_EACH = {s: lg(np.load(f"{H}/experiments/exp_055/pred/"
                          f"pred_T70_u2024_seed{s}_at1400.npy")) for s in AA}
L_aa = np.mean(list(AA_EACH.values()), 0)
L_t70 = np.mean(list(T70_EACH.values()), 0)

SHA = {}


def _load_deep(tag_fmt, base, seeds, fitjson=None):
    out = {}
    for s in seeds:
        p_ = f"{base}/out/preds/{tag_fmt.format(s=s)}.npy"
        z_ = np.load(p_)
        h_ = hashlib.sha256(np.ascontiguousarray(z_).tobytes()).hexdigest()
        rec = None
        if fitjson:
            fj = f"{base}/out/{fitjson.format(s=s)}.json"
            if os.path.exists(fj):
                rec = json.load(open(fj)).get("z_sha256")
        SHA[tag_fmt.format(s=s)] = {"sha256": h_, "recorded": rec,
                                    "MATCH": (rec is None or h_ == rec)}
        out[s] = z_
    return out


DEEP_FIX_EACH = _load_deep("z_F1632_f64o_s{s}", B97, S5, "fit_F1632_f64o_s{s}")
DEEP_DEF_EACH = _load_deep("z_K1632_oot23_s{s}", B90, S5, "fit_K1632_oot23_s{s}")
assert all(v["MATCH"] for v in SHA.values()), SHA
Z_DEEP = np.mean(list(DEEP_FIX_EACH.values()), 0)          # 고친 f64 면
Z_DEEP_DEF = np.mean(list(DEEP_DEF_EACH.values()), 0)      # 결함면 (§5 재판독 전용)

AXES = ["alpha", "c_aa", "c_T70", "w_out", "w_v6_inner", "w_ebm", "w_deep"]
ROOT_AXES = {"alpha", "c_aa", "c_T70", "w_out"}
INNER_AXES = {"w_v6_inner", "w_ebm", "w_deep"}
THETA0 = dict(BO)
THETA0["w_deep"] = SHIP_W_DEEP
THETA0["v_out"] = SHIP_V_OUT


def zvec(th, dz=None, l_aa=None, l_t70=None):
    """배포 산술 그대로. th 는 THETA0 를 갱신한 dict."""
    dz = Z_DEEP if dz is None else dz
    l_aa = L_aa if l_aa is None else l_aa
    l_t70 = L_t70 if l_t70 is None else l_t70
    wi, we = th["w_v6_inner"], th["w_ebm"]
    zm = (1.0 - we) * (wi * LV + (1.0 - wi) * LQ) + we * LE + th["beta_ins"] * OFF
    wd = th["w_deep"]
    zm = (1.0 - wd) * zm + wd * dz
    return (th["w_out"] * zm + th["c_aa"] * l_aa + th["c_T70"] * l_t70
            + th.get("v_out", 0.0) * dz)


def pin(z, f, target=YB):
    c = 0.0
    for _ in range(200):
        p = f(z, c)
        g = float(p.mean()) - target
        if abs(g) < 1e-15:
            break
        c -= g / float((p * (1 - p)).mean())
    return c


def prob(z, alpha, beta_skew):
    f = (lambda zz, c: expit(alpha * zz + beta_skew * zz * zz + c))
    return f(z, pin(z, f))


def prob_rc(z):
    f = (lambda zz, c: expit(zz + c))
    return f(z, pin(z, f))


def score(z, alpha, beta_skew):
    return pts(float(np.mean((prob(z, alpha, beta_skew) - y) ** 2)))


def score_rc(z):
    return pts(float(np.mean((prob_rc(z) - y) ** 2)))


def sc_theta(th, **kw):
    return score(zvec(th, **kw), th["alpha"], th["beta_skew"])


def sd_match(znew, zref):
    return ((znew - znew.mean()) / znew.std()) * zref.std() + zref.mean()


# ── 짝지은 투수 블록 부트스트랩 (exp_097/102/103b/104 와 같은 rng·호출 순서) ──
_rng = np.random.default_rng(20260824)
_uq, _inv = np.unique(pit, return_inverse=True)
NB = len(_uq)
ORDER = np.argsort(_inv, kind="stable")
ST = np.searchsorted(_inv[ORDER], np.arange(NB))
EN = np.append(ST[1:], len(ORDER))
BSIZE = (EN - ST).astype(np.float64)
PICKS = np.stack([_rng.integers(0, NB, NB) for _ in range(1000)])


def boot_t(d):
    """d = 행별 (brier_A - brier_B). 양수 이득이면 A 가 낫다."""
    bs_ = np.array([d[ORDER[ST[j]:EN[j]]].sum() for j in range(NB)])
    bs = bs_[PICKS].sum(1) / BSIZE[PICKS].sum(1)
    m = float(d.mean())
    se = float(bs.std(ddof=1))
    return {"gain_pts": -m * K, "boot_se_pts": se * K,
            "t": (-m / se) if se > 0 else None,
            "ci95_pts": [float(-np.percentile(bs, 97.5) * K),
                         float(-np.percentile(bs, 2.5) * K)]}


# ── 2차 국소 모형 (exp_078/081 틀) ───────────────────────────────────────
#   z -> z + sum_j t_j delta_j,  c 는 매 지점 다시 못박음(pinning)
#   p = expit(alpha*z + beta*z^2 + c),  q = p(1-p),  g = alpha + 2*beta*z
#   dp/dz = q*g,  dp/dc = q,  제약 mean(p)=ybar  =>  dc_j = -mean(q*g*d_j)/mean(q)
#   e_j = q*(g*d_j - mean(q*g*d_j)/mean(q))
#   L_j = -mean(s*e_j)   (s = p - y),   G_jk = mean(e_j*e_k)      (가우스-뉴턴)
#   dScore = K*(2 t.L - t.G t),  최대 = K * L' G^+ L  (고윳값 절단, S44)
#   축이 z 를 미는 것(가중류)이면 u = g*delta,  alpha 면 u = z,  beta_skew 면 u = z^2.
def raw_u(z0, alpha, beta_skew, name, delta=None):
    if name == "alpha":
        return z0.copy()
    if name == "beta_skew":
        return z0 * z0
    g = alpha + 2.0 * beta_skew * z0
    return g * delta


def quad_LG(z0, alpha, beta_skew, us):
    """us = [dp/dtheta_j | c 고정, 을 q 로 나눈 벡터] 목록."""
    p0 = prob(z0, alpha, beta_skew)
    q = p0 * (1.0 - p0)
    s = p0 - y
    mq = float(q.mean())
    E = np.stack([q * (u - float((q * u).mean()) / mq) for u in us])
    L = -(E @ s) / len(y)
    G = (E @ E.T) / len(y)
    return L, G


def solve_trunc(L, G, rcond=1e-10):
    """S44: 고윳값 절단 사영. np.linalg.solve 금지."""
    w_, U = np.linalg.eigh(G)
    keep = w_ > rcond * max(w_.max(), 0.0)
    Lp = U[:, keep].T @ L
    t = U[:, keep] @ (Lp / w_[keep])
    gain = float(K * (Lp @ (Lp / w_[keep])))
    cond = float(w_.max() / w_.min()) if w_.min() > 0 else float("inf")
    return t, gain, int(keep.sum()), cond, w_
