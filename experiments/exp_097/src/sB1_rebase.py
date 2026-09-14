"""exp_097 stageB1/B2 — re-read the deep-arm learning-amount axis on the
float64-standardised input, at the DEPLOYED weight w_deep = 0.40.

The surface is exp_090 s1_curve.py's, letter for letter (same constants, same
pinning, same pitcher-block bootstrap rng and call order).  Only the deep arm's
INPUT changes:
    def  = exp_089/090 predictions, trained on X_oot23      (season sd 19.616)
    fix  = exp_097 predictions,     trained on X_f64_oot23  (season sd 1.4238)

Everything else -- z_mate, L_aa, L_T70, the alpha/beta_skew/d convention -- is
shared, so a def/fix difference at the same step and the same w is the defect.
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
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

H = "/home/<user>/SungjinPark"
B89, B90, B97 = f"{H}/exp089", f"{H}/exp090", f"{H}/exp097"
OOT, OOT3, OOT4 = f"{H}/exp069/oot", f"{H}/exp070/oot3", f"{H}/exp071/out"
EPS, REF = 1e-6, 0.2498069
K = 100000.0 / REF
S5 = [42, 777, 2024, 101, 7]
AA = (42, 7, 2024)
W4 = {"LGB": 0.05, "CAT": 0.68, "NN": 0.15, "LEAF": 0.12}
BO = dict(alpha=1.07911, beta_skew=-0.059151, c_aa=0.25244, c_T70=0.04756,
          w_out=0.7, w_v6_inner=0.6668397489, w_ebm=-0.1, beta_ins=0.03)
BASE_EXPECT = 928.501461058684
N_TRAIN_OOT, N_TRAIN_DEPLOY = 1116277, 1475092
W_CE, W_SHIP = 0.20, 0.40           # CE-era weight / candidate CH's own w_deep
lg = lambda p: logit(np.clip(np.asarray(p, np.float64), EPS, 1 - EPS))
pts = lambda b: 100000.0 * (1.0 - b / REF)
t0 = time.time()

# ── points ───────────────────────────────────────────────────────────────
DEF_REUSE = {544: (B89, "M_oot23_s{s}"), 720: (B89, "S720_oot23_s{s}"),
             816: (B89, "E3_oot23_s{s}"), 1088: (B89, "S1088_oot23_s{s}")}
DEF_NEW = (1360, 1632, 2176, 2720, 3264, 4080)
FIX_STEPS = (544, 720, 816, 1088, 1360, 1632, 2176, 2720, 3264, 4080)
SHA = {}


def _load(path, rec_path=None, rec_key="z_sha256", tag=None):
    z = np.load(path)
    h = hashlib.sha256(np.ascontiguousarray(z).tobytes()).hexdigest()
    rec = None
    if rec_path and os.path.exists(rec_path):
        rec = json.load(open(rec_path))[rec_key]
    SHA[tag] = {"file_sha256": h, "recorded": rec,
                "MATCH": (rec is None or h == rec)}
    return z


POINTS = {"def": {}, "fix": {}}
for n, (r, pat) in DEF_REUSE.items():
    POINTS["def"][n] = [_load(f"{r}/out/preds/z_{pat.format(s=s)}.npy",
                              f"{r}/out/fit_{pat.format(s=s)}.json",
                              tag=f"def|{n}|{s}") for s in S5]
for n in DEF_NEW:
    if all(os.path.exists(f"{B90}/out/preds/z_K{n}_oot23_s{s}.npy") for s in S5):
        POINTS["def"][n] = [_load(f"{B90}/out/preds/z_K{n}_oot23_s{s}.npy",
                                  f"{B90}/out/fit_K{n}_oot23_s{s}.json",
                                  tag=f"def|{n}|{s}") for s in S5]
for n in FIX_STEPS:
    if all(os.path.exists(f"{B97}/out/preds/z_F{n}_f64o_s{s}.npy") for s in S5):
        POINTS["fix"][n] = [_load(f"{B97}/out/preds/z_F{n}_f64o_s{s}.npy",
                                  f"{B97}/out/fit_F{n}_f64o_s{s}.json",
                                  tag=f"fix|{n}|{s}") for s in S5]
assert all(v["MATCH"] for v in SHA.values()), "pred sha mismatch"

# ── surface (exp_090 s1_curve.py verbatim) ───────────────────────────────
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
    bs_ = np.array([d[order[st[j]:en[j]]].sum() for j in range(nb)])
    num = bs_[PICKS].sum(1)
    den = BSIZE[PICKS].sum(1)
    bs = num / den
    m = float(d.mean())
    se = float(bs.std(ddof=1))
    return {"gain_pts": -m * K, "boot_se_brier": se,
            "t": (-m / se) if se > 0 else None,
            "ci95_pts": [float(-np.percentile(bs, 97.5) * K),
                         float(-np.percentile(bs, 2.5) * K)]}


R = {"exp_id": "exp_097", "stage": "sB1_rebase", "K_pts_per_brier": K,
     "surface": "exp_090 s1_curve.py 면 그대로. 딥 팔 입력만 f32 결함/f64 수리로 갈린다.",
     "w_CE_era": W_CE, "w_deployed_CH": W_SHIP,
     "pred_sha": SHA, "n_rows": int(len(y))}
z0 = root(Z_MATE)
base = {c: sc(z0, c) for c in ("rc", "abd")}
base_p = {c: prob(z0, c) for c in ("rc", "abd")}
R["baseline_w0"] = {"rc": base["rc"], "expected": BASE_EXPECT,
                    "abs_diff": abs(base["rc"] - BASE_EXPECT), "abd": base["abd"]}
assert R["baseline_w0"]["abs_diff"] < 1e-9, base["rc"]

GRID = [round(x, 4) for x in np.arange(0.0, 0.6001, 0.01)]
R["points"] = {}
for surf in ("def", "fix"):
    R["points"][surf] = {}
    for n in sorted(POINTS[surf]):
        zs = POINTS[surf][n]
        dz = np.mean(zs, 0)
        ent = {"total_steps": n, "surface": surf,
               "alone_recentred_score": sc(dz, "rc"),
               "z_sd": float(dz.std()), "z_mean": float(dz.mean()),
               "rho_vs_z_mate": float(np.corrcoef(dz, Z_MATE)[0, 1]),
               "within_seed_rho": float(np.mean(
                   [np.corrcoef(zs[i], zs[j])[0, 1]
                    for i, j in itertools.combinations(range(5), 2)]))}
        cur = [{"w": w, "gain": sc(root((1 - w) * Z_MATE + w * dz), "abd")
                - base["abd"]} for w in GRID]
        best = max(cur, key=lambda c: c["gain"])
        for lab, w in (("w0.20", W_CE), ("w0.40", W_SHIP)):
            zw = root((1 - w) * Z_MATE + w * dz)
            d = (prob(zw, "abd") - y) ** 2 - (base_p["abd"] - y) ** 2
            ent[lab] = {"gain": [c for c in cur
                                 if abs(c["w"] - w) < 1e-9][0]["gain"],
                        "gain_sd_matched": sc(sd_match(zw, z0), "abd")
                        - base["abd"],
                        "boot": boot_t(d)}
        ent["argmax"] = {"w": best["w"], "gain": best["gain"]}
        ent["curve_abd"] = cur
        R["points"][surf][str(n)] = ent
        print("%-4s steps %5d  alone_rc %8.2f  rho %0.5f | w=.20 %+8.3f (t %+.2f)"
              "  w=.40 %+8.3f (t %+.2f) | argmax w=%.2f %+8.3f"
              % (surf, n, ent["alone_recentred_score"], ent["rho_vs_z_mate"],
                 ent["w0.20"]["gain"], ent["w0.20"]["boot"]["t"],
                 ent["w0.40"]["gain"], ent["w0.40"]["boot"]["t"],
                 ent["argmax"]["w"], ent["argmax"]["gain"]), flush=True)

# ── CE anchor: which step pair reproduces the +10.102 recorded in S111? ──
if R["points"]["def"]:
    g = {int(k): v["w0.20"]["gain"] for k, v in R["points"]["def"].items()}
    pairs = [{"lo": a, "hi": b, "delta_def_w0.20": g[b] - g[a]}
             for a in g for b in g if b > a]
    pairs.sort(key=lambda p: abs(p["delta_def_w0.20"] - 10.102))
    R["CE_anchor_pair_search"] = {
        "recorded_S111_delta_on_def_surface": 10.102,
        "recorded_LB_gain": 4.8280402096,
        "recorded_transfer_ratio": 0.4779,
        "closest_pairs": pairs[:5]}
    print("\nCE anchor search (def surface, w=0.20), target delta 10.102:")
    for p in pairs[:5]:
        print("   %5d -> %5d   delta %+8.4f" % (p["lo"], p["hi"],
                                                p["delta_def_w0.20"]))
    if R["points"]["fix"]:
        gf = {int(k): v["w0.20"]["gain"] for k, v in R["points"]["fix"].items()}
        best_pair = pairs[0]
        lo, hi = best_pair["lo"], best_pair["hi"]
        if lo in gf and hi in gf:
            d_fix = gf[hi] - gf[lo]
            R["CE_transfer_ratio_on_fixed_surface"] = {
                "pair": [lo, hi], "delta_fix_w0.20": d_fix,
                "LB_gain": 4.8280402096,
                "transfer_ratio_fix": 4.8280402096 / d_fix if d_fix else None,
                "transfer_ratio_def_recomputed":
                    4.8280402096 / best_pair["delta_def_w0.20"]}
            print("CE transfer ratio on fixed surface: delta %+0.4f -> ratio %0.5f"
                  % (d_fix, 4.8280402096 / d_fix))

# ── B2: exp_095's outer deep slot v, re-read on the fixed surface ────────
OPER = 1632
for surf in ("def", "fix"):
    if OPER not in POINTS[surf]:
        continue
    dz = np.mean(POINTS[surf][OPER], 0)
    WG = [round(x, 4) for x in np.arange(0.0, 0.5001, 0.02)]
    VG = [round(x, 4) for x in np.arange(0.0, 0.3001, 0.02)]
    gmap = []
    for w in WG:
        for v in VG:
            z = root((1 - w) * Z_MATE + w * dz) + v * dz
            gmap.append({"w": w, "v": v, "gain": sc(z, "abd") - base["abd"]})
    bst = max(gmap, key=lambda c: c["gain"])
    v0 = max([c for c in gmap if c["v"] == 0.0], key=lambda c: c["gain"])
    rows_with_v = sum(1 for w in WG if max(
        [c for c in gmap if c["w"] == w], key=lambda c: c["gain"])["v"] > 0)
    R.setdefault("B2_outer_slot_v", {})[surf] = {
        "step": OPER, "argmax_2dof": bst, "argmax_v0": v0,
        "argmax_v_is_zero": bool(bst["v"] == 0.0),
        "n_w_rows_with_positive_v_star": rows_with_v, "n_w_rows": len(WG)}
    print("B2 %-4s step %d: 2dof argmax w=%.2f v=%.2f gain %+0.4f | v=0 argmax "
          "w=%.2f gain %+0.4f | rows with v*>0: %d/%d"
          % (surf, OPER, bst["w"], bst["v"], bst["gain"], v0["w"], v0["gain"],
             rows_with_v, len(WG)))

# ── pre-registered verdict (config.yaml, fixed before launch, S8) ────────
if R["points"]["fix"]:
    fx = R["points"]["fix"]
    ref_step = str(OPER) if str(OPER) in fx else sorted(fx, key=int)[0]
    ref = fx[ref_step]["w0.40"]["gain"]
    cands = []
    for k, v in fx.items():
        for lab, w in (("w0.20", W_CE), ("w0.40", W_SHIP)):
            gg, tt = v[lab]["gain"], v[lab]["boot"]["t"]
            if gg - ref >= 2.0 and tt is not None and tt >= 2.0:
                cands.append({"steps": int(k), "w": w, "gain": gg,
                              "gain_minus_deployed": gg - ref, "t": tt})
    R["PREREGISTERED_VERDICT"] = {
        "rule": "정직면 이득 >= 배포점 + 2.0  AND  짝지은 투수블록 부트스트랩 t >= 2.0",
        "deployed_point": {"steps": OPER, "w": W_SHIP, "gain": ref},
        "qualifying": cands, "PASS": bool(cands)}
    print("\nPREREG: deployed point (steps %s, w=%.2f) gain %+0.4f -> "
          "qualifying candidates %d" % (ref_step, W_SHIP, ref, len(cands)))

R["wall_sec"] = round(time.time() - t0, 1)
json.dump(R, open(f"{B97}/out/sB1_rebase.json", "w"), ensure_ascii=False, indent=1)
print("WROTE", f"{B97}/out/sB1_rebase.json")
