"""exp_102 stage A — learning-amount MIXED deep arm on the FIXED (float64) OOT input.

Hypothesis (S99): a deep arm trained longer gets worse alone but overlaps mate less.
Averaging checkpoints taken at DIFFERENT learning amounts should therefore hold the
alone score up while pulling rho down -- the one knob S103 asks for.

Ship path (why this is not a toy): teammate `mate/deep_infer.py:predict_deep` does
    zs = [_one(np.load(p), codes, X) for p in npz_paths]
    return 1/(1+exp(-np.mean(zs, axis=0)))          # <- LOGIT average
so a list of npz files trained for different numbers of steps goes down exactly the
same code path.  Nothing in the .py changes; the S101 shape hard-coding is untouched.

Surface: exp_097 sB1_rebase.py, letter for letter (same constants, same pinning,
same pitcher-block bootstrap rng and call order).  Only the arm z changes.

Pre-registered verdict (config.yaml, fixed before launch, S8):
    grid argmax gain - 26.2698 >= +2.0   AND   paired pitcher-block bootstrap
    t >= 2.0 against the COUNT-MATCHED null.
"""
import hashlib
import itertools
import json
import os
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "4"
import numpy as np
import pandas as pd
from scipy.special import expit, logit

H = "/home/<user>/SungjinPark"
B97, B102 = f"{H}/exp097", f"{H}/exp102"
OOT, OOT3, OOT4 = f"{H}/exp069/oot", f"{H}/exp070/oot3", f"{H}/exp071/out"
EPS, REF = 1e-6, 0.2498069
K = 100000.0 / REF
S5 = [42, 777, 2024, 101, 7]
EXTRA_SEEDS = [11, 13, 17, 19, 23, 29, 31, 37, 41, 43,
               47, 53, 59, 61, 67, 71, 73, 79, 83, 89]
AA = (42, 7, 2024)
W4 = {"LGB": 0.05, "CAT": 0.68, "NN": 0.15, "LEAF": 0.12}
BO = dict(alpha=1.07911, beta_skew=-0.059151, c_aa=0.25244, c_T70=0.04756,
          w_out=0.7, w_v6_inner=0.6668397489, w_ebm=-0.1, beta_ins=0.03)
BASE_EXPECT = 928.501461058684
PREREG_BASELINE = 26.2698          # exp_097 fix surface, {1632}x5, argmax w=0.43
PREREG_MARGIN = 2.0
PREREG_T = 2.0
lg = lambda p: logit(np.clip(np.asarray(p, np.float64), EPS, 1 - EPS))
pts = lambda b: 100000.0 * (1.0 - b / REF)
t0 = time.time()

SHA = {}


def zload(step, seed):
    """Load one deep-arm logit vector by (total_steps, seed), hashing it."""
    for base in (B97, B102):
        p = f"{base}/out/preds/z_F{step}_f64o_s{seed}.npy"
        if os.path.exists(p):
            z = np.load(p)
            h = hashlib.sha256(np.ascontiguousarray(z).tobytes()).hexdigest()
            rec = None
            fj = f"{base}/out/fit_F{step}_f64o_s{seed}.json"
            if os.path.exists(fj):
                rec = json.load(open(fj))["z_sha256"]
            SHA[f"{step}|{seed}"] = {"src": base.split("/")[-1],
                                     "file_sha256": h, "recorded": rec,
                                     "MATCH": (rec is None or h == rec)}
            return z
    raise FileNotFoundError(f"z_F{step}_f64o_s{seed}.npy")


POOL = {}


def get(step, seed):
    if (step, seed) not in POOL:
        POOL[(step, seed)] = zload(step, seed)
    return POOL[(step, seed)]


# ── surface (exp_097 sB1_rebase.py verbatim) ─────────────────────────────
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


def sd_match(znew, zref):
    return ((znew - znew.mean()) / znew.std()) * zref.std() + zref.mean()


rng = np.random.default_rng(20260824)
uq, inv = np.unique(pit, return_inverse=True)
nb = len(uq)
order = np.argsort(inv, kind="stable")
st = np.searchsorted(inv[order], np.arange(nb))
en = np.append(st[1:], len(order))
BSIZE = (en - st).astype(np.float64)
PICKS = np.stack([rng.integers(0, nb, nb) for _ in range(1000)])


def boot_t(d):
    """d = per-row (brier_A - brier_B); positive gain means A is better."""
    bs_ = np.array([d[order[st[j]:en[j]]].sum() for j in range(nb)])
    bs = bs_[PICKS].sum(1) / BSIZE[PICKS].sum(1)
    m = float(d.mean())
    se = float(bs.std(ddof=1))
    return {"gain_pts": -m * K, "boot_se_brier": se,
            "t": (-m / se) if se > 0 else None,
            "ci95_pts": [float(-np.percentile(bs, 97.5) * K),
                         float(-np.percentile(bs, 2.5) * K)]}


# ── arm definitions ──────────────────────────────────────────────────────
# each arm = list of (step, seed) checkpoints; the arm z is their LOGIT mean.
def combo(steps, seeds):
    return [(t, s) for t in steps for s in seeds]


ARMS = {}
# --- mixes: 5 fixed seeds, several learning amounts ---------------------
MIXSETS = {
    "M01_base_1632":        [1632],
    "M02_816_1632":         [816, 1632],
    "M03_1088_1632":        [1088, 1632],
    "M04_1632_2720":        [1632, 2720],
    "M05_816_1632_2720":    [816, 1632, 2720],
    "M06_1088_1632_2176":   [1088, 1632, 2176],
    "M07_816_1632_3264":    [816, 1632, 3264],
    "M08_1360_1632_2176":   [1360, 1632, 2176],
    "M09_1088_1360_1632":   [1088, 1360, 1632],
    "M10_4pt":              [816, 1088, 1632, 2176],
    "M11_5pt":              [816, 1088, 1632, 2176, 2720],
    "M12_5pt_wide":         [544, 1088, 1632, 2720, 4080],
    "M13_816_1088_1632_2176_3264": [816, 1088, 1632, 2176, 3264],
}
for k, ss in MIXSETS.items():
    ARMS[k] = {"kind": "mix", "steps": ss, "seeds": S5,
               "cks": combo(ss, S5), "n_ck": len(ss) * len(S5)}
# --- null A: SAME learning amount 1632, more independent seeds ----------
ALL1632 = S5 + EXTRA_SEEDS
for n in (5, 10, 15, 20, 25):
    ARMS[f"N0A_1632x{n}seeds"] = {"kind": "nullA", "steps": [1632],
                                  "seeds": ALL1632[:n],
                                  "cks": combo([1632], ALL1632[:n]), "n_ck": n}
# --- null B: SAME 5 trajectories, negligible step spread ----------------
ARMS["N0B_1568_1632_1696"] = {"kind": "nullB", "steps": [1568, 1632, 1696],
                              "seeds": S5,
                              "cks": combo([1568, 1632, 1696], S5), "n_ck": 15}

R = {"exp_id": "exp_102", "stage": "sA_mix",
     "K_pts_per_brier": K, "n_rows": int(len(y)),
     "surface": "exp_097 sB1_rebase.py 면 그대로 (고친 f64 OOT 입력).",
     "prereg": {"baseline_gain": PREREG_BASELINE, "margin": PREREG_MARGIN,
                "t_min": PREREG_T,
                "rule": "격자 argmax 이득 - 26.2698 >= +2.0 AND 개수 맞춤 귀무 대조 "
                        "대비 짝지은 투수 블록 부트스트랩 t >= 2.0"},
     "ship_path": "mate/deep_infer.py predict_deep -> np.mean(zs,0) 로짓 평균. "
                  "학습량이 다른 npz 를 넣는 것은 같은 코드 경로."}

z0 = root(Z_MATE)
base = {c: sc(z0, c) for c in ("rc", "abd")}
R["baseline_w0"] = {"rc": base["rc"], "expected": BASE_EXPECT,
                    "abs_diff": abs(base["rc"] - BASE_EXPECT), "abd": base["abd"]}
assert R["baseline_w0"]["abs_diff"] < 1e-9, base["rc"]

GRID = [round(x, 4) for x in np.arange(0.0, 0.6001, 0.01)]
ZARM, PBEST = {}, {}
R["arms"] = {}
for name, spec in ARMS.items():
    dz = np.mean([get(t, s) for t, s in spec["cks"]], 0)
    ZARM[name] = dz
    cur = [{"w": w, "gain": sc(root((1 - w) * Z_MATE + w * dz), "abd") - base["abd"]}
           for w in GRID]
    best = max(cur, key=lambda c: c["gain"])
    zb = root((1 - best["w"]) * Z_MATE + best["w"] * dz)
    PBEST[name] = prob(zb, "abd")
    within = [np.corrcoef(get(*a), get(*b))[0, 1]
              for a, b in itertools.combinations(spec["cks"], 2)]
    ent = {"kind": spec["kind"], "steps": spec["steps"], "n_seeds": len(spec["seeds"]),
           "n_checkpoints": spec["n_ck"],
           "alone_recentred_score": sc(dz, "rc"),
           "z_sd": float(dz.std()), "z_mean": float(dz.mean()),
           "rho_vs_z_mate": float(np.corrcoef(dz, Z_MATE)[0, 1]),
           "within_checkpoint_rho_mean": float(np.mean(within)),
           "argmax": {"w": best["w"], "gain": best["gain"]},
           "gain_at_w0.40": [c for c in cur if abs(c["w"] - 0.40) < 1e-9][0]["gain"],
           "gain_at_w0.43": [c for c in cur if abs(c["w"] - 0.43) < 1e-9][0]["gain"],
           "gain_sd_matched_at_argmax": sc(sd_match(zb, z0), "abd") - base["abd"],
           "curve_abd": cur}
    R["arms"][name] = ent
    print("%-28s ck %2d  alone %8.2f  rho %.5f | argmax w=%.2f %+8.4f  (w.40 %+8.4f)"
          % (name, spec["n_ck"], ent["alone_recentred_score"],
             ent["rho_vs_z_mate"], best["w"], best["gain"],
             ent["gain_at_w0.40"]), flush=True)

# gate: M01 must reproduce the exp_097 fix-surface {1632}x5 point
R["gate_M01_reproduces_exp097"] = {
    "M01_argmax": R["arms"]["M01_base_1632"]["argmax"],
    "exp097_fix_1632_argmax": {"w": 0.43, "gain": 26.269975},
    "abs_diff_gain": abs(R["arms"]["M01_base_1632"]["argmax"]["gain"] - 26.269975),
    "same_as_N0A_5": (R["arms"]["M01_base_1632"]["argmax"]
                      == R["arms"]["N0A_1632x5seeds"]["argmax"])}

# ── paired comparisons vs the count-matched null ─────────────────────────
NULL_FOR = {5: "N0A_1632x5seeds", 10: "N0A_1632x10seeds", 15: "N0A_1632x15seeds",
            20: "N0A_1632x20seeds", 25: "N0A_1632x25seeds"}
R["paired_vs_null"] = {}
for name, ent in R["arms"].items():
    if ent["kind"] not in ("mix", "nullB"):
        continue
    n = ent["n_checkpoints"]
    nl = NULL_FOR.get(n)
    if nl is None:
        continue
    rec = {"null": nl, "n_checkpoints": n,
           "mix_argmax": ent["argmax"], "null_argmax": R["arms"][nl]["argmax"],
           "gain_minus_null": ent["argmax"]["gain"] - R["arms"][nl]["argmax"]["gain"],
           "gain_minus_prereg_baseline": ent["argmax"]["gain"] - PREREG_BASELINE,
           "d_rho": ent["rho_vs_z_mate"] - R["arms"][nl]["rho_vs_z_mate"],
           "d_alone": (ent["alone_recentred_score"]
                       - R["arms"][nl]["alone_recentred_score"])}
    d_own = (PBEST[name] - y) ** 2 - (PBEST[nl] - y) ** 2
    rec["boot_own_argmax"] = boot_t(d_own)
    w = ent["argmax"]["w"]
    pn = prob(root((1 - w) * Z_MATE + w * ZARM[nl]), "abd")
    rec["boot_common_w"] = {"w": w, **boot_t((PBEST[name] - y) ** 2 - (pn - y) ** 2)}
    R["paired_vs_null"][name] = rec
    print("PAIR %-28s vs %-18s dgain %+8.4f  t(own) %+6.3f  t(w=%.2f) %+6.3f"
          % (name, nl, rec["gain_minus_null"], rec["boot_own_argmax"]["t"] or 0.0,
             w, rec["boot_common_w"]["t"] or 0.0), flush=True)

# ── S103 diagnostic: gain ~ rho + alone (in-sample OLS on these arms) ────
xs = np.array([[R["arms"][k]["rho_vs_z_mate"],
                R["arms"][k]["alone_recentred_score"]] for k in R["arms"]])
gs = np.array([R["arms"][k]["argmax"]["gain"] for k in R["arms"]])
Xd = np.column_stack([np.ones(len(xs)), xs])
cf, *_ = np.linalg.lstsq(Xd, gs, rcond=None)
res = gs - Xd @ cf
dof = len(gs) - 3
s2 = float(res @ res) / dof
se = np.sqrt(np.diag(np.linalg.inv(Xd.T @ Xd)) * s2)
R["S103_diag"] = {
    "note": "진단으로만 (S83·S103). 판정에 쓰지 않는다.",
    "coef": {"intercept": cf[0], "rho": cf[1], "alone": cf[2]},
    "t": {"rho": cf[1] / se[1], "alone": cf[2] / se[2]},
    "r2": 1 - float(res @ res) / float(((gs - gs.mean()) ** 2).sum()),
    "residual_pts": {k: float(r) for k, r in zip(R["arms"], res)}}

# ── pre-registered verdict ───────────────────────────────────────────────
q = []
for name, rec in R["paired_vs_null"].items():
    if R["arms"][name]["kind"] != "mix":
        continue
    t = rec["boot_own_argmax"]["t"]
    if (rec["gain_minus_prereg_baseline"] >= PREREG_MARGIN
            and t is not None and t >= PREREG_T):
        q.append({"arm": name, "gain": R["arms"][name]["argmax"]["gain"],
                  "gain_minus_baseline": rec["gain_minus_prereg_baseline"],
                  "t": t})
R["PREREGISTERED_VERDICT"] = {"rule": R["prereg"]["rule"], "qualifying": q,
                              "PASS": bool(q)}
print("\nPREREG axisA: qualifying %d -> PASS=%s" % (len(q), bool(q)))

R["pred_sha"] = SHA
R["all_sha_match"] = all(v["MATCH"] for v in SHA.values())
R["wall_sec"] = round(time.time() - t0, 1)
os.makedirs(f"{B102}/out", exist_ok=True)
json.dump(R, open(f"{B102}/out/sA_mix.json", "w"), ensure_ascii=False, indent=1)
print("WROTE", f"{B102}/out/sA_mix.json", "wall", R["wall_sec"])
