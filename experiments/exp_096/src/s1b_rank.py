"""exp_096 stage1b — A3(rank) 후속: 격자 확장 + 강건성.

s1_arch 가 사전 등록 격자에서 rank 를 PASS 로 냈는데 argmax 가 **격자 두 모서리에 다 붙었다**
(w = 0.20 하단, w2 = 0.30 상단).  즉 최적이 격자 밖이다.  여기서 넓혀 잰다.

🔴 사전 등록 판정은 s1_arch 의 config 격자 결과가 정본이다 (S8).  이 파일은 **진단**이다 —
   넓힌 격자의 argmax 로 문턱을 다시 판정하지 않는다.

강건성으로 묻는 것 넷:
  ① 시드 잭나이프 — 5시드 중 하나를 빼도 이득이 남는가
  ② 3시드 부분집합 10개 — 이득이 5시드 평균에만 얹혀 있는가
  ③ argmax 가 아닌 **미리 정한 점**에서의 짝지은 t (선택 편의 없는 값)
  ④ 척도 몫 (S25·S33) — sd 맞춤 이득과 alpha 자유 이득
"""
import hashlib, itertools, json, os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

H = "/home/<user>/SungjinPark"
B90, B96 = f"{H}/exp090", f"{H}/exp096"
OOT, OOT3, OOT4 = f"{H}/exp069/oot", f"{H}/exp070/oot3", f"{H}/exp071/out"
EPS, REF = 1e-6, 0.2498069
K = 100000.0 / REF
S5 = [42, 777, 2024, 101, 7]
AA = (42, 7, 2024)
W4 = {"LGB": 0.05, "CAT": 0.68, "NN": 0.15, "LEAF": 0.12}
BO = dict(alpha=1.07911, beta_skew=-0.059151, c_aa=0.25244, c_T70=0.04756,
          w_out=0.7, w_v6_inner=0.6668397489, w_ebm=-0.1, beta_ins=0.03)
BASE_EXPECT = 928.501461058684
REF_2ARM = 13.714309769419287
ARM = sys.argv[1] if len(sys.argv) > 1 else "rank"
OUTN = sys.argv[2] if len(sys.argv) > 2 else "s1b_rank.json"
lg = lambda p: logit(np.clip(np.asarray(p, np.float64), EPS, 1 - EPS))
pts = lambda b: 100000.0 * (1.0 - b / REF)
t0 = time.time()

DEEP = []
for s in S5:
    tag = f"K1632_oot23_s{s}"
    z_ = np.load(f"{B90}/out/preds/z_{tag}.npy")
    assert (hashlib.sha256(np.ascontiguousarray(z_).tobytes()).hexdigest()
            == json.load(open(f"{B90}/out/fit_{tag}.json"))["z_sha256"])
    DEEP.append(z_)
DZ = np.mean(DEEP, 0)
DFIT = [json.load(open(f"{B90}/out/fit_K1632_oot23_s{s}.json")) for s in S5]

tro = pd.read_csv(f"{H}/data/train.csv",
                  usecols=["row_id", "season", "pitcher_id", "control_success"])
tro = tro[tro.season == 2024].sort_values("row_id", kind="stable")
y = tro.control_success.to_numpy(np.float64)
pit = tro.pitcher_id.to_numpy()
assert np.array_equal(
    np.asarray(np.load(f"{OOT}/row_id_oot2024.npy", allow_pickle=True)).astype(str),
    tro.row_id.to_numpy().astype(str))
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
FIX = BO["c_aa"] * L_aa + BO["c_T70"] * L_t70


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
ZERO = np.zeros_like(DZ)


def root(w, w2, nz=ZERO, dz=DZ):
    return BO["w_out"] * ((1.0 - w - w2) * Z_MATE + w * dz + w2 * nz) + FIX


def alpha_free(z):
    r = minimize_scalar(lambda a: -sc(a * z, "abd"), bounds=(0.3, 3.0),
                        method="bounded", options={"xatol": 1e-8})
    return float(-r.fun), float(r.x)


sd_match = lambda a, b: ((a - a.mean()) / a.std()) * b.std() + b.mean()
rng = np.random.default_rng(20260824)
uq, inv = np.unique(pit, return_inverse=True)
nb = len(uq)
order = np.argsort(inv, kind="stable")
st = np.searchsorted(inv[order], np.arange(nb))
en = np.append(st[1:], len(order))
BSIZE = (en - st).astype(np.float64)
PICKS = np.stack([rng.integers(0, nb, nb) for _ in range(1000)])
IDX = [order[st[j]:en[j]] for j in range(nb)]


def boot(d):
    bs_ = np.array([d[IDX[j]].sum() for j in range(nb)])
    bs = bs_[PICKS].sum(1) / BSIZE[PICKS].sum(1)
    se = float(bs.std(ddof=1))
    obs = float(d.mean())
    return {"gain_pts": -obs * K, "t": float(-obs / se) if se > 0 else None,
            "p_share_new_worse": float((bs > 0).mean()),
            "ci95_pts": [float(-np.percentile(bs, 97.5) * K),
                         float(-np.percentile(bs, 2.5) * K)]}


z0 = root(0.0, 0.0)
base = sc(z0, "abd")
base_rc = sc(z0, "rc")
assert abs(base_rc - BASE_EXPECT) == 0.0, base_rc
assert abs(sc(root(0.36, 0.0), "abd") - base - REF_2ARM) == 0.0
SQ_ref = (prob(root(0.36, 0.0), "abd") - y) ** 2

NZS = [np.load(f"{B96}/out/preds/z_{ARM}_oot23_s{s}.npy") for s in S5]
NZ = np.mean(NZS, 0)
rho = lambda a, b: float(np.corrcoef(a, b)[0, 1])

R = {"exp_id": "exp_096", "stage": "s1b", "arm": ARM,
     "🔴_prereg_verdict_lives_in": "out/s1_arch*.json (config 격자). 여기는 진단이다.",
     "deep_reference": {
         "alone_per_seed": [f["valid_score_raw"] for f in DFIT],
         "alone_seedmean_rc": sc(DZ, "rc"),
         "seed_rho_pairs": [rho(DEEP[i], DEEP[j])
                            for i in range(5) for j in range(i + 1, 5)],
         "z_sd": float(DZ.std())},
     "arm_stats": {
         "alone_per_seed": [json.load(open(
             f"{B96}/out/fit_{ARM}_oot23_s{s}.json"))["valid_score_raw"] for s in S5],
         "alone_seedmean_rc": sc(NZ, "rc"),
         "seed_rho_pairs": [rho(NZS[i], NZS[j])
                            for i in range(5) for j in range(i + 1, 5)],
         "rho_vs_z_mate": rho(NZ, Z_MATE), "rho_vs_z_deep": rho(NZ, DZ),
         "z_sd": float(NZ.std())}}

# ── ① 넓힌 격자 (고정 alpha) ──────────────────────────────────────────
WG = [round(x, 4) for x in np.arange(0.0, 0.4401, 0.02)]
W2G = [round(x, 4) for x in np.arange(0.0, 0.6001, 0.02)]
G = {}
for w in WG:
    for w2 in W2G:
        if w + w2 > 0.90:
            continue
        G[(w, w2)] = sc(root(w, w2, NZ), "abd") - base
bw = max(G, key=lambda t: G[t])
top = sorted(G, key=lambda t: -G[t])[:12]
R["wide_grid"] = {
    "spec": {"w": [WG[0], WG[-1], 0.02], "w2": [W2G[0], W2G[-1], 0.02],
             "constraint": "w + w2 <= 0.90", "n_points": len(G)},
    "argmax": {"w": bw[0], "w2": bw[1], "gain_fixed_alpha": G[bw]},
    "argmax_on_boundary": bool(bw[0] in (WG[0], WG[-1]) or bw[1] in (W2G[0], W2G[-1])),
    "top12": [{"w": t[0], "w2": t[1], "gain": G[t]} for t in top],
    "gain_at_config_argmax(0.20,0.30)": G[(0.20, 0.30)],
    "residual_wide_vs_ref": G[bw] - REF_2ARM}
gaf, al = alpha_free(root(*bw, NZ))
R["wide_grid"]["argmax_alpha_free_gain"] = gaf - alpha_free(z0)[0]
R["wide_grid"]["argmax_alpha_star"] = al
R["wide_grid"]["argmax_sd_matched_gain"] = (
    sc(sd_match(root(*bw, NZ), z0), "abd") - base)
print(f"wide argmax {bw} gain {G[bw]:.4f}  {time.time()-t0:.0f}s", flush=True)

# ── ③ 미리 정한 점의 짝지은 t (선택 편의 없음) ────────────────────────
FIXED_PTS = [(0.20, 0.10), (0.20, 0.20), (0.20, 0.30), (0.24, 0.20),
             (0.36, 0.10), (0.36, 0.20), (0.0, 0.30), (0.0, 0.44)]
R["paired_boot_at_points"] = {}
for w, w2 in FIXED_PTS:
    P_ = prob(root(w, w2, NZ), "abd")
    R["paired_boot_at_points"][f"w{w}_w2{w2}"] = {
        "gain_fixed_alpha": sc(root(w, w2, NZ), "abd") - base,
        **boot((P_ - y) ** 2 - SQ_ref)}
R["paired_boot_at_points"]["🔴_note"] = (
    "🔴 정정(감사 지적) — 이 목록은 s1_arch 결과를 **본 뒤에** 만들었고 그 argmax "
    "(0.20, 0.30) 자신을 포함한다. 따라서 '선택 편의가 없다'고 적으면 거짓이다. "
    "옳은 서술: argmax 선택 편의는 남아 있다. 다만 argmax 가 아닌 인접 점 "
    "(0.20,0.20) t=3.11 · (0.24,0.20) t=2.83 에서도 t 가 2 를 넘으므로 "
    "판정이 argmax 한 점에만 얹혀 있지는 않다. "
    "config 사전 등록 판정은 argmax 의 t 이고 그 값은 s1_arch 에 있다.")

# ── ① 시드 잭나이프 ───────────────────────────────────────────────────
JK = []
for k in range(5):
    nzk = np.mean([NZS[i] for i in range(5) if i != k], 0)
    gk = {t: sc(root(t[0], t[1], nzk), "abd") - base for t in G}
    bk = max(gk, key=lambda t: gk[t])
    JK.append({"left_out_seed": S5[k], "argmax": [bk[0], bk[1]],
               "gain_at_argmax": gk[bk],
               "gain_at_config_argmax": gk[(0.20, 0.30)],
               "residual_vs_ref": gk[bk] - REF_2ARM})
jr = np.array([j["residual_vs_ref"] for j in JK])
jc = np.array([j["gain_at_config_argmax"] - REF_2ARM for j in JK])
R["jackknife_over_arm_seeds"] = {
    "per_loo": JK,
    "se_of_residual": float(np.sqrt(4 / 5 * np.sum((jr - jr.mean()) ** 2))),
    "se_of_residual_at_config_argmax":
        float(np.sqrt(4 / 5 * np.sum((jc - jc.mean()) ** 2))),
    "min_residual": float(jr.min()), "max_residual": float(jr.max())}
print(f"jackknife resid {jr.min():.3f}~{jr.max():.3f}  {time.time()-t0:.0f}s",
      flush=True)

# ── ② 3시드 부분집합 ──────────────────────────────────────────────────
SUB3 = []
for c in itertools.combinations(range(5), 3):
    nz3 = np.mean([NZS[i] for i in c], 0)
    SUB3.append({"seeds": [S5[i] for i in c],
                 "gain_at_config_argmax": sc(root(0.20, 0.30, nz3), "abd") - base,
                 "gain_at_0.20_0.20": sc(root(0.20, 0.20, nz3), "abd") - base})
g3 = np.array([s["gain_at_config_argmax"] for s in SUB3])
R["three_seed_subsets"] = {
    "n": len(SUB3), "per_subset": SUB3,
    "mean": float(g3.mean()), "min": float(g3.min()), "max": float(g3.max()),
    "share_above_ref": float((g3 > REF_2ARM).mean()),
    "5seed_value": G[(0.20, 0.30)]}

# ── ④ 딥 팔을 rank 로 갈아끼우는 경우 (치환) ──────────────────────────
SUBC = [{"w2": round(x, 4),
         "gain": sc(root(0.0, round(x, 4), NZ), "abd") - base}
        for x in np.arange(0.02, 0.7001, 0.02)]
bs = max(SUBC, key=lambda c: c["gain"])
R["substitution_w0"] = {"curve": SUBC, "best": bs,
                        "note": "w=0 이면 딥 팔이 빠지고 rank 팔만 들어간다"}
R["wall_sec"] = round(time.time() - t0, 1)
json.dump(R, open(f"{B96}/out/{OUTN}", "w"), ensure_ascii=False, indent=1)
slim = {k: v for k, v in R.items()
        if k not in ("wide_grid", "substitution_w0", "three_seed_subsets")}
slim["wide_grid"] = {k: v for k, v in R["wide_grid"].items() if k != "top12"}
slim["wide_top12"] = R["wide_grid"]["top12"]
slim["three_seed_subsets"] = {k: v for k, v in R["three_seed_subsets"].items()
                              if k != "per_subset"}
slim["substitution_best"] = R["substitution_w0"]["best"]
print(json.dumps(slim, ensure_ascii=False, indent=1))
