"""exp_096 — 3자유도 혼합 격자 평가 (stage1·stage2 공용).

사용: eval096.py <출력파일명> <ARM_TAG_PREFIX> [<ARM_TAG_PREFIX> ...]
  예: eval096.py s1_arch.json attn cos rank ctrl35

면은 exp_090 s1_curve / exp_095 s1_wv 를 **글자 그대로** 승계한다.
바뀌는 것은 내부 슬롯이 3자유도가 된 것뿐이다:

    z_inner(w, w2) = (1 - w - w2)*z_mate + w*z_deep + w2*z_new
    z(w, w2)       = w_out*z_inner + c_aa*z_aa + c_T70*z_T70
    p              = expit(alpha*z + beta_skew*z^2 + d)      (abd 관례 = 주 판정면)

w2 = 0 이면 exp_090 식과 글자 그대로 같다 → G1/G2/G3 로 매 실행 재확인한다.

🔴 사전 등록 문턱 (config 원본, 결과 보고 바꾸지 않았다 — S8):
     PASS = (격자 argmax 이득 − 13.714309769419287 >= +2.5)  AND  (짝지은 t >= 2.0)
🔴 판정은 혼합 이득 하나다.  ρ·단독점수는 **진단으로만** 적는다 (S83).
🔴 정직면 크기를 LB 로 환산하지 않는다 (S54·S78·S111).
"""
import hashlib, json, os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

OUTNAME = sys.argv[1]
ARMS = sys.argv[2:]
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
REF_2ARM = 13.714309769419287        # (w=0.36, w2=0) — 사전 등록 기준점
G3_EXPECT = 11.05028248077349        # (w=0.20, w2=0)
THR_GAIN, THR_T = 2.5, 2.0
STEPS = 1632
lg = lambda p: logit(np.clip(np.asarray(p, np.float64), EPS, 1 - EPS))
pts = lambda b: 100000.0 * (1.0 - b / REF)
t0 = time.time()

ZS = []
for s in S5:
    tag = f"K{STEPS}_oot23_s{s}"
    z_ = np.load(f"{B90}/out/preds/z_{tag}.npy")
    assert (hashlib.sha256(np.ascontiguousarray(z_).tobytes()).hexdigest()
            == json.load(open(f"{B90}/out/fit_{tag}.json"))["z_sha256"])
    ZS.append(z_)
DZ = np.mean(ZS, 0)

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
assert all(V[k].min() >= 0 and V[k].max() <= 1 for k in
           ("p_lgb", "p_cat", "p_leaf", "p_nn", "p_q", "p_ebm"))
assert V["off"].min() < 0.0
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


def root(w, w2, nz=ZERO):
    inner = (1.0 - w - w2) * Z_MATE + w * DZ + w2 * nz
    return BO["w_out"] * inner + FIX


def alpha_free(z, conv="abd"):
    r = minimize_scalar(lambda a: -sc(a * z, conv), bounds=(0.3, 3.0),
                        method="bounded", options={"xatol": 1e-8})
    return float(-r.fun), float(r.x)


def sd_match(znew, zref):
    return ((znew - znew.mean()) / znew.std()) * zref.std() + zref.mean()


# ── 투수 블록 부트스트랩 (exp_089/090/095 와 같은 시드·같은 추출 순서) ────
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
    return {"gain_pts": -obs * K, "boot_se_brier": se,
            "t": float(-obs / se) if se > 0 else None,
            "p_share_new_worse": float((bs > 0).mean()),
            "ci95_pts": [float(-np.percentile(bs, 97.5) * K),
                         float(-np.percentile(bs, 2.5) * K)]}


R = {"exp_id": "exp_096", "eval_of": ARMS, "K_pts_per_brier": K, "ref_brier": REF,
     "BO_constants": BO, "n_rows": int(len(y)), "y_mean": YB,
     "mix": "z_inner(w,w2) = (1-w-w2)*z_mate + w*z_deep + w2*z_new",
     "🔴_no_LB_conversion": ("정직면 크기를 LB 로 환산하지 않는다 (S54·S78). "
                             "순위·부호·상대 비교만 쓴다.")}

z0 = root(0.0, 0.0)
base = {c: sc(z0, c) for c in ("rc", "abd")}
base_af = alpha_free(z0)[0]
base_p = prob(z0, "abd")
SQ_base = (base_p - y) ** 2
g036 = sc(root(0.36, 0.0), "abd") - base["abd"]
g020 = sc(root(0.20, 0.0), "abd") - base["abd"]
R["gates"] = {
    "G1_rc": {"here": base["rc"], "expected": BASE_EXPECT,
              "abs_diff": abs(base["rc"] - BASE_EXPECT),
              "PASS": abs(base["rc"] - BASE_EXPECT) == 0.0},
    "G2_w0.36": {"here": g036, "expected": REF_2ARM,
                 "abs_diff": abs(g036 - REF_2ARM),
                 "PASS": abs(g036 - REF_2ARM) == 0.0},
    "G3_w0.20": {"here": g020, "expected": G3_EXPECT,
                 "abs_diff": abs(g020 - G3_EXPECT),
                 "PASS": abs(g020 - G3_EXPECT) == 0.0}}
R["gates"]["ALL_PASS"] = all(R["gates"][k]["PASS"] for k in
                             ("G1_rc", "G2_w0.36", "G3_w0.20"))
assert R["gates"]["ALL_PASS"], R["gates"]
print("GATES OK", flush=True)

P_ref = prob(root(0.36, 0.0), "abd")
SQ_ref = (P_ref - y) ** 2
WG = [round(x, 4) for x in np.arange(0.20, 0.4401, 0.04)]
W2G = [round(x, 4) for x in np.arange(0.00, 0.3001, 0.02)]
rho = lambda a, b: float(np.corrcoef(a, b)[0, 1])

ARMR = {}
for a in ARMS:
    zs, fits = [], []
    for s in S5:
        pth = f"{B96}/out/preds/z_{a}_oot23_s{s}.npy"
        if not os.path.exists(pth):
            print(f"  !! missing {pth}", flush=True)
            continue
        zs.append(np.load(pth))
        fits.append(json.load(open(f"{B96}/out/fit_{a}_oot23_s{s}.json")))
    if not zs:
        continue
    NZ = np.mean(zs, 0)
    pr = [[rho(zs[i], zs[j]) for j in range(len(zs))] for i in range(len(zs))]
    off = [pr[i][j] for i in range(len(zs)) for j in range(i + 1, len(zs))]
    d = {"arm": a, "n_seeds": len(zs), "seeds": [f["seed"] for f in fits],
         "n_params": fits[0]["n_params"], "lr_schedule": fits[0]["lr_schedule"],
         "X_path": fits[0]["X_path"], "total_steps": fits[0]["total_steps"],
         "n_train": fits[0]["n_train"],
         "fit_wall_sec_mean": float(np.mean([f["fit_wall_sec"] for f in fits])),
         "DIAG_alone_score_per_seed": [f["valid_score_raw"] for f in fits],
         "DIAG_alone_score_seedmean_rc": sc(NZ, "rc"),
         "DIAG_rho_vs_z_mate": rho(NZ, Z_MATE),
         "DIAG_rho_vs_z_deep": rho(NZ, DZ),
         "DIAG_seed_rho_mean": float(np.mean(off)) if off else None,
         "DIAG_seed_rho_min": float(np.min(off)) if off else None,
         "DIAG_seed_rho_max": float(np.max(off)) if off else None,
         "DIAG_S115_band": [0.91064, 0.93932],
         "DIAG_z_sd": float(NZ.std())}
    G = {}
    for w in WG:
        for w2 in W2G:
            z = root(w, w2, NZ)
            gfix = sc(z, "abd") - base["abd"]
            gaf, al = alpha_free(z)
            G[(w, w2)] = {"w": w, "w2": w2, "gain_fixed_alpha": gfix,
                          "gain_alpha_free": gaf - base_af, "alpha_star": al,
                          "gain_sd_matched": sc(sd_match(z, z0), "abd") - base["abd"],
                          "z_sd": float(z.std())}
    # 진단: 치환형 (w=0, w2 자유) — 딥을 통째로 갈아끼우는 경우
    SUB = []
    for w2 in [round(x, 4) for x in np.arange(0.02, 0.4401, 0.02)]:
        z = root(0.0, w2, NZ)
        SUB.append({"w": 0.0, "w2": w2,
                    "gain_fixed_alpha": sc(z, "abd") - base["abd"]})
    best = max(G.values(), key=lambda c: c["gain_fixed_alpha"])
    best_af = max(G.values(), key=lambda c: c["gain_alpha_free"])
    best_sd = max(G.values(), key=lambda c: c["gain_sd_matched"])
    best_sub = max(SUB, key=lambda c: c["gain_fixed_alpha"])
    P_b = prob(root(best["w"], best["w2"], NZ), "abd")
    bt = boot((P_b - y) ** 2 - SQ_ref)
    # w2=0 단면 (같은 격자에서의 2팔 최적) — 새 팔이 없을 때와의 대조
    cut0 = max([G[(w, 0.0)] for w in WG], key=lambda c: c["gain_fixed_alpha"])
    resid = best["gain_fixed_alpha"] - REF_2ARM
    d.update({
        "grid": [G[k] for k in sorted(G)], "grid_spec": {"w": WG, "w2": W2G},
        "substitution_curve_w0": SUB, "substitution_best": best_sub,
        "argmax_fixed_alpha": best, "argmax_alpha_free": best_af,
        "argmax_sd_matched": best_sd,
        "best_at_w2_zero(2arm in this grid)": cut0,
        "residual_vs_prereg_ref": resid,
        "paired_boot_vs_ref_w0.36": bt,
        "VERDICT": {
            "rule": ("argmax 이득 − %.12f >= %.1f  AND  짝지은 투수블록 t >= %.1f"
                     % (REF_2ARM, THR_GAIN, THR_T)),
            "residual_pts": resid, "threshold_residual": THR_GAIN,
            "paired_t": bt["t"], "threshold_t": THR_T,
            "residual_PASS": bool(resid >= THR_GAIN),
            "t_PASS": bool(bt["t"] is not None and bt["t"] >= THR_T),
            "PASS": bool(resid >= THR_GAIN and bt["t"] is not None
                         and bt["t"] >= THR_T)}})
    ARMR[a] = d
    _f = lambda v, n=4: ("%.*f" % (n, v)) if isinstance(v, float) else str(v)
    print(f"[{a}] n_seeds={d['n_seeds']} "
          f"alone={_f(d['DIAG_alone_score_seedmean_rc'], 2)} "
          f"rho_mate={_f(d['DIAG_rho_vs_z_mate'])} "
          f"rho_deep={_f(d['DIAG_rho_vs_z_deep'])} "
          f"argmax(w={best['w']},w2={best['w2']}) "
          f"gain={_f(best['gain_fixed_alpha'])} "
          f"resid={_f(resid)} t={_f(bt['t'])} "
          f"PASS={d['VERDICT']['PASS']}  {time.time()-t0:.0f}s", flush=True)

R["arms"] = ARMR
R["prereg_reference"] = {"point": "(w=0.36, w2=0)", "gain": REF_2ARM,
                         "baseline_abd": base["abd"], "baseline_rc": base["rc"]}
R["ANY_PASS"] = bool(any(v["VERDICT"]["PASS"] for v in ARMR.values()))
R["wall_sec"] = round(time.time() - t0, 1)
os.makedirs(f"{B96}/out", exist_ok=True)
json.dump(R, open(f"{B96}/out/{OUTNAME}", "w"), ensure_ascii=False, indent=1)
slim = {k: v for k, v in R.items() if k != "arms"}
slim["arms"] = {k: {kk: vv for kk, vv in v.items()
                    if kk not in ("grid", "substitution_curve_w0")}
                for k, v in ARMR.items()}
print(json.dumps(slim, ensure_ascii=False, indent=1))
