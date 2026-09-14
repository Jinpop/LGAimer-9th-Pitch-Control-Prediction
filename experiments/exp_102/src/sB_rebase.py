"""exp_102 stage B — re-read exp_091/092's top deep-slot points on the FIXED input.

exp_091 (22 points) and exp_092 (19 points) rejected the whole deep-slot diversity
family, but every one of those readings sat on the defective float32 surface
(S134).  exp_097 showed the SIZE of deep readings was 1.92x low there; nobody
checked whether the ORDER changes.  This re-runs the handful of points that were
at the top (or interestingly at the bottom) on the corrected float64 input.

Points (all 1632 steps, 5 seeds, harness = exp_092 train092.py verbatim):
  Bdr02   dropout 0.2   -- was alone +14.3 but mixed -3.25 (the sharpest S103 case)
  Bm16    random 16-col mask (rng 9212) -- exp_092's BEST mixed point (+14.892)
  Bls005  label smoothing 0.05          -- exp_092 #2 (+13.911)
  Bw1dm1  hidden [256,128]   (depth-1)  -- exp_091 (+14.604)  [S101: NOT shippable]
  Bw2dm1  hidden [512,256]   (depth-1)  -- exp_091 (+14.672)  [S101: NOT shippable]

Threshold identical to axis A: gain - 26.2698 >= +2.0 AND paired pitcher-block
bootstrap t >= 2.0 versus the same-seed-count baseline.
"""
import hashlib
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
AA = (42, 7, 2024)
W4 = {"LGB": 0.05, "CAT": 0.68, "NN": 0.15, "LEAF": 0.12}
BO = dict(alpha=1.07911, beta_skew=-0.059151, c_aa=0.25244, c_T70=0.04756,
          w_out=0.7, w_v6_inner=0.6668397489, w_ebm=-0.1, beta_ins=0.03)
BASE_EXPECT = 928.501461058684
PREREG_BASELINE, PREREG_MARGIN, PREREG_T = 26.2698, 2.0, 2.0
lg = lambda p: logit(np.clip(np.asarray(p, np.float64), EPS, 1 - EPS))
pts = lambda b: 100000.0 * (1.0 - b / REF)
t0 = time.time()

# defective-surface readings, copied from experiments/exp_09{1,2}/metrics.json
DEF = {
    "BASE_1632":  {"gain": 13.714309769419287, "w": 0.36, "rho": 0.9141957764385082,
                   "alone": 862.5652996988365, "src": "exp_091 A_n05_seeds5 / exp_092 D0"},
    "Bm16":       {"gain": 14.891798100835672, "w": 0.38, "rho": 0.90882,
                   "alone": 865.30, "src": "exp_092 D5m16_rand"},
    "Bw2dm1":     {"gain": 14.672181774055844, "w": 0.38, "rho": 0.9069759754417659,
                   "alone": 863.1762081707906, "src": "exp_091 B_w2_dm1"},
    "Bw1dm1":     {"gain": 14.60402787439864, "w": 0.36, "rho": 0.9032524261657744,
                   "alone": 857.365244636954, "src": "exp_091 B_w1_dm1"},
    "Bls005":     {"gain": 13.911, "w": 0.38, "rho": 0.91432,
                   "alone": 871.16, "src": "exp_092 E_ls005"},
    "Bdr02":      {"gain": 10.459739770762894, "w": 0.36, "rho": 0.9267898231991485,
                   "alone": 876.8753407579877, "src": "exp_091 C_dr02_wd001"},
}
SHIPPABLE = {"Bm16": "예 — 마스킹은 A 행 0 으로 구워진다. deep_infer.py 불변",
             "Bls005": "예 — 손실만 바뀐다. 추론 경로 불변",
             "Bdr02": "예 — dropout 은 추론 시 항등",
             "Bw1dm1": "아니오 (S101) — deep_infer._one 이 m.1/m.4/m.5/m.8/m.10 깊이 3 을 못박는다",
             "Bw2dm1": "아니오 (S101) — 위 + _fold 가 첫 은닉 폭 256 을 reshape 로 못박는다"}

SHA = {}


def zload(tag):
    for base in (B102, B97):
        p = f"{base}/out/preds/z_{tag}.npy"
        if os.path.exists(p):
            z = np.load(p)
            h = hashlib.sha256(np.ascontiguousarray(z).tobytes()).hexdigest()
            fj = f"{base}/out/fit_{tag}.json"
            rec = json.load(open(fj))["z_sha256"] if os.path.exists(fj) else None
            SHA[tag] = {"src": base.split("/")[-1], "file_sha256": h,
                        "recorded": rec, "MATCH": (rec is None or h == rec)}
            return z
    raise FileNotFoundError(tag)


# ── surface (exp_097 sB1_rebase.py verbatim) ─────────────────────────────
tro = pd.read_csv(f"{H}/data/train.csv",
                  usecols=["row_id", "season", "pitcher_id", "control_success"])
tro = tro[tro.season == 2024].sort_values("row_id", kind="stable")
y = tro.control_success.to_numpy(np.float64)
pit = tro.pitcher_id.to_numpy()
rid = np.asarray(np.load(f"{OOT}/row_id_oot2024.npy", allow_pickle=True)).astype(str)
assert np.array_equal(rid, tro.row_id.to_numpy().astype(str))
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
    bs_ = np.array([d[order[st[j]:en[j]]].sum() for j in range(nb)])
    bs = bs_[PICKS].sum(1) / BSIZE[PICKS].sum(1)
    m = float(d.mean())
    se = float(bs.std(ddof=1))
    return {"gain_pts": -m * K, "boot_se_brier": se,
            "t": (-m / se) if se > 0 else None,
            "ci95_pts": [float(-np.percentile(bs, 97.5) * K),
                         float(-np.percentile(bs, 2.5) * K)]}


ARMS = {"BASE_1632": [f"F1632_f64o_s{s}" for s in S5]}
for nm in ("Bdr02", "Bm16", "Bls005", "Bw1dm1", "Bw2dm1"):
    ARMS[nm] = [f"{nm}_f64o_s{s}" for s in S5]

R = {"exp_id": "exp_102", "stage": "sB_rebase", "K_pts_per_brier": K,
     "surface": "exp_097 sB1_rebase.py 면 (고친 f64 OOT 입력)",
     "prereg": {"baseline_gain": PREREG_BASELINE, "margin": PREREG_MARGIN,
                "t_min": PREREG_T},
     "defective_surface_readings": DEF, "shippable": SHIPPABLE}

z0 = root(Z_MATE)
base = {c: sc(z0, c) for c in ("rc", "abd")}
R["baseline_w0"] = {"rc": base["rc"], "expected": BASE_EXPECT,
                    "abs_diff": abs(base["rc"] - BASE_EXPECT)}
assert R["baseline_w0"]["abs_diff"] < 1e-9

GRID = [round(x, 4) for x in np.arange(0.0, 0.6001, 0.01)]
PB, ZA = {}, {}
R["arms"] = {}
for nm, tags in ARMS.items():
    dz = np.mean([zload(t) for t in tags], 0)
    ZA[nm] = dz
    cur = [{"w": w, "gain": sc(root((1 - w) * Z_MATE + w * dz), "abd") - base["abd"]}
           for w in GRID]
    bst = max(cur, key=lambda c: c["gain"])
    zb = root((1 - bst["w"]) * Z_MATE + bst["w"] * dz)
    PB[nm] = prob(zb, "abd")
    R["arms"][nm] = {
        "n_seeds": len(tags), "steps": 1632,
        "alone_recentred_score": sc(dz, "rc"), "z_sd": float(dz.std()),
        "rho_vs_z_mate": float(np.corrcoef(dz, Z_MATE)[0, 1]),
        "argmax": {"w": bst["w"], "gain": bst["gain"]},
        "gain_at_w0.43": [c for c in cur if abs(c["w"] - 0.43) < 1e-9][0]["gain"],
        "gain_sd_matched_at_argmax": sc(sd_match(zb, z0), "abd") - base["abd"],
        "curve_abd": cur}
    print("%-10s alone %8.2f  rho %.5f | argmax w=%.2f %+8.4f   (def surface %+8.4f)"
          % (nm, R["arms"][nm]["alone_recentred_score"],
             R["arms"][nm]["rho_vs_z_mate"], bst["w"], bst["gain"],
             DEF[nm]["gain"]), flush=True)

# ── paired vs baseline + rank-flip detection ────────────────────────────
R["vs_baseline"] = {}
for nm in ARMS:
    if nm == "BASE_1632":
        continue
    d = (PB[nm] - y) ** 2 - (PB["BASE_1632"] - y) ** 2
    w = R["arms"][nm]["argmax"]["w"]
    pb = prob(root((1 - w) * Z_MATE + w * ZA["BASE_1632"]), "abd")
    R["vs_baseline"][nm] = {
        "d_gain_fix": R["arms"][nm]["argmax"]["gain"] - R["arms"]["BASE_1632"]["argmax"]["gain"],
        "d_gain_def": DEF[nm]["gain"] - DEF["BASE_1632"]["gain"],
        "boot_own_argmax": boot_t(d),
        "boot_common_w": {"w": w, **boot_t((PB[nm] - y) ** 2 - (pb - y) ** 2)},
        "d_rho_fix": R["arms"][nm]["rho_vs_z_mate"] - R["arms"]["BASE_1632"]["rho_vs_z_mate"],
        "d_rho_def": DEF[nm]["rho"] - DEF["BASE_1632"]["rho"]}

order_def = sorted(DEF, key=lambda k: -DEF[k]["gain"])
order_fix = sorted(R["arms"], key=lambda k: -R["arms"][k]["argmax"]["gain"])
flips = []
for i, a in enumerate(order_def):
    for b in order_def[i + 1:]:
        if order_fix.index(a) > order_fix.index(b):
            flips.append({"pair": [a, b], "def_order": "a>b", "fix_order": "b>a",
                          "def_gains": [DEF[a]["gain"], DEF[b]["gain"]],
                          "fix_gains": [R["arms"][a]["argmax"]["gain"],
                                        R["arms"][b]["argmax"]["gain"]]})
sign_flips = [k for k in R["vs_baseline"]
              if np.sign(R["vs_baseline"][k]["d_gain_fix"])
              != np.sign(R["vs_baseline"][k]["d_gain_def"])]
R["rank"] = {"order_def": order_def, "order_fix": order_fix,
             "pairwise_flips": flips, "n_pairwise_flips": len(flips),
             "sign_flips_vs_baseline": sign_flips,
             "size_ratio_fix_over_def": {
                 k: (R["arms"][k]["argmax"]["gain"] / DEF[k]["gain"]) for k in DEF}}

q = [{"arm": k, "gain": R["arms"][k]["argmax"]["gain"],
      "gain_minus_baseline": R["arms"][k]["argmax"]["gain"] - PREREG_BASELINE,
      "t": R["vs_baseline"][k]["boot_own_argmax"]["t"]}
     for k in R["vs_baseline"]
     if R["arms"][k]["argmax"]["gain"] - PREREG_BASELINE >= PREREG_MARGIN
     and (R["vs_baseline"][k]["boot_own_argmax"]["t"] or -9) >= PREREG_T]
R["PREREGISTERED_VERDICT"] = {"qualifying": q, "PASS": bool(q)}
print("\nrank def:", order_def)
print("rank fix:", order_fix)
print("pairwise flips:", len(flips), "sign flips vs baseline:", sign_flips)
print("PREREG axisB: qualifying %d -> PASS=%s" % (len(q), bool(q)))

R["pred_sha"] = SHA
R["all_sha_match"] = all(v["MATCH"] for v in SHA.values())
R["wall_sec"] = round(time.time() - t0, 1)
json.dump(R, open(f"{B102}/out/sB_rebase.json", "w"), ensure_ascii=False, indent=1)
print("WROTE", f"{B102}/out/sB_rebase.json", "wall", R["wall_sec"])
