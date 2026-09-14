"""exp_071 stage 3 — p_v6 완성(4팔) + w_v6 1축 조건부 곡선 재실행.

exp_070 followup070.py 의 ★BN용 절차를 산술까지 그대로 옮겼다.
대조군으로 3종판(p_v6_3arm_renorm)도 같은 코드로 돌려 exp_070 의
Δ(0.70) = +4.618141264278734 를 재현하는지 확인한다 — 재현되면 파이프라인이 같은 것이다.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "6"
import hashlib
import json
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

H = "/home/<user>/SungjinPark"
E = f"{H}/exp071"
OOT, OOT3, OUT = f"{H}/exp069/oot", f"{H}/exp070/oot3", f"{E}/out"
EPS = 1e-6
REF = 0.2498069
K = 100000.0 / REF
R_BOOT = 1000
SEED_BOOT = 70
DEP = np.array([0.5, -0.10, 0.03, 0.7, 0.25244, 0.04756])
T0 = time.time()
M = {}

lg = lambda p: logit(np.clip(np.asarray(p, np.float64), EPS, 1.0 - EPS))
conv = lambda b: 100000.0 * (1.0 - b / REF)

tr = pd.read_csv(f"{H}/data/train.csv",
                 usecols=["row_id", "season", "control_success", "pitcher_id"])
tr = tr[tr.season == 2024].sort_values("row_id", kind="stable").reset_index(drop=True)
y = tr.control_success.to_numpy(np.float64)
N = len(y)
ybar = float(y.mean())
pit_codes, pit_u = pd.factorize(tr.pitcher_id.to_numpy(), sort=True)
G = len(pit_u)
assert hashlib.sha256(np.ascontiguousarray(y, np.float64).tobytes()).hexdigest() == \
    "e0483e6a48d9f699ca9c4abf7b4b81aed4d679c359c7e9c3752b3f9e6e27c9d3"
M["y_sha_assert"] = "PASS"
M["N"], M["n_pitchers"] = int(N), int(G)

V = {k: np.load(f"{OOT}/{f}").astype(np.float64) for k, f in
     [("p_lgb", "p_lgb_oot2024.npy"), ("p_cat", "p_cat_oot2024.npy"),
      ("p_leaf", "p_leaf_oot2024.npy"), ("p_q", "p_q_oot2024.npy"),
      ("p_v6_3arm", "p_v6_3arm_renorm_oot2024.npy"), ("off", "off_oot2024.npy")]}
V["p_ebm"] = np.load(f"{OOT3}/p_ebm_oot2024.npy").astype(np.float64)
p_nn = np.load(f"{OUT}/p_nn_oot2024.npy").astype(np.float64)
assert len(p_nn) == N
rid_ours = np.load(f"{OUT}/row_id_oot2024_ours.npy", allow_pickle=True).astype(str)
M["정렬대조_row_id_동일"] = bool((rid_ours == tr.row_id.to_numpy().astype(str)).all())
assert M["정렬대조_row_id_동일"], "row_id 정렬 불일치"

W_LGB, W_CAT, W_NN, W_LEAF = 0.05, 0.68, 0.15, 0.12

# ── 🔴 항등 대조: NN 자리를 빼고 0.85 로 재정규화 → 팀원 3종판과 일치해야 한다 ──
z3 = (W_LGB * lg(V["p_lgb"]) + W_CAT * lg(V["p_cat"]) + W_LEAF * lg(V["p_leaf"])) / 0.85
p3 = expit(z3)
d3 = np.abs(p3 - V["p_v6_3arm"])
ulp = np.spacing(np.abs(V["p_v6_3arm"]))
M["항등대조_3종판_재정규화"] = {
    "max_abs": float(d3.max()), "mean_abs": float(d3.mean()),
    "max_ulp": float(np.max(d3 / ulp)), "n_초과_1ulp": int((d3 > ulp).sum()),
    "판정": "PASS" if float(np.max(d3 / ulp)) <= 1.0 else "확인필요"}
print(f"[항등] 3종판 재정규화 max|Δ| = {d3.max():.3e}  max_ulp = "
      f"{np.max(d3/ulp):.3f}  {M['항등대조_3종판_재정규화']['판정']}", flush=True)

# ── p_v6 완성 (4팔) ────────────────────────────────────────────────
z4 = (W_LGB * lg(V["p_lgb"]) + W_CAT * lg(V["p_cat"]) +
      W_NN * lg(p_nn) + W_LEAF * lg(V["p_leaf"]))
p_v6_4 = expit(z4)
np.save(f"{OUT}/p_v6_4arm_oot2024.npy", p_v6_4)


def recenter_c(z, c0=0.0):
    for _ in range(40):
        p = expit(z + c0)
        mu = float(p.mean())
        if abs(mu - ybar) < 1e-14:
            break
        d = (mu - ybar) / float((p * (1.0 - p)).mean())
        c0 -= d
        if abs(d) < 1e-14:
            break
    return c0


def score_z(z):
    c = recenter_c(z)
    b = float(np.mean((expit(z + c) - y) ** 2))
    return b, conv(b)


def vmap_A(th):
    wv, we, be, wo, ca, ct = th
    return np.array([wo * (1 - we) * wv, wo * (1 - we) * (1 - wv),
                     wo * we, wo * be, ca, ct])


AA_SEEDS = (42, 7, 2024)
L_aa = np.mean([lg(np.load(f"{H}/experiments/exp_049/pred/"
                           f"pred_A1_u2024_seed{s}_at1400.npy").astype(np.float64))
                for s in AA_SEEDS], axis=0)
L_t70 = np.mean([lg(np.load(f"{H}/experiments/exp_055/pred/"
                            f"pred_T70_u2024_seed{s}_at1400.npy").astype(np.float64))
                 for s in AA_SEEDS], axis=0)
L_q, L_ebm, off = lg(V["p_q"]), lg(V["p_ebm"]), V["off"]
XA3 = np.column_stack([lg(V["p_v6_3arm"]), L_q, L_ebm, off, L_aa, L_t70])
XA4 = np.column_stack([lg(p_v6_4), L_q, L_ebm, off, L_aa, L_t70])

# 단일 팔 재센터링 점수
M["팔_단독_재센터링점수"] = {
    nm: score_z(lg(v))[1] for nm, v in
    [("p_nn_ours(k고정)", p_nn), ("p_v6_3arm", V["p_v6_3arm"]), ("p_v6_4arm", p_v6_4),
     ("p_lgb", V["p_lgb"]), ("p_cat", V["p_cat"]), ("p_leaf", V["p_leaf"]),
     ("p_q", V["p_q"]), ("p_ebm", V["p_ebm"])]}
M["상관_로짓"] = {
    "ρ(nn,cat)": float(np.corrcoef(lg(p_nn), lg(V["p_cat"]))[0, 1]),
    "ρ(nn,v6_3arm)": float(np.corrcoef(lg(p_nn), lg(V["p_v6_3arm"]))[0, 1]),
    "ρ(v6_4arm,v6_3arm)": float(np.corrcoef(lg(p_v6_4), lg(V["p_v6_3arm"]))[0, 1]),
    "ρ(v6_4arm,q)": float(np.corrcoef(lg(p_v6_4), L_q)[0, 1]),
    "ρ(v6_3arm,q)": float(np.corrcoef(lg(V["p_v6_3arm"]), L_q)[0, 1])}
print(json.dumps(M["팔_단독_재센터링점수"], ensure_ascii=False), flush=True)


def axis_pack(XA, tag):
    s_dep = score_z(XA @ vmap_A(DEP))[1]
    grid = {}
    for gv in np.round(np.arange(0.0, 1.0001, 0.05), 4):
        th = DEP.copy(); th[0] = gv
        grid[f"{gv:.2f}"] = {"score_REF": score_z(XA @ vmap_A(th))[1],
                             "Δ_vs_dep_pts": score_z(XA @ vmap_A(th))[1] - s_dep}

    def f1(x):
        th = DEP.copy(); th[0] = x
        return score_z(XA @ vmap_A(th))[0]

    r = minimize_scalar(f1, bounds=(0.0, 1.0), method="bounded",
                        options={"xatol": 1e-6})
    th70 = DEP.copy(); th70[0] = 0.70
    s70 = score_z(XA @ vmap_A(th70))[1]
    return {"배포_점수": s_dep, "이동후_점수_w_v6=0.70": s70, "Δ_pts": s70 - s_dep,
            "조건부_argmax": float(r.x), "조건부_argmax_점수": conv(float(r.fun)),
            "조건부_argmax_Δ_pts": conv(float(r.fun)) - s_dep,
            "cap_touched": bool(abs(r.x) < 1e-4 or abs(r.x - 1.0) < 1e-4),
            "0.00~1.00_격자": grid, "tag": tag}


t = time.time()
A3 = axis_pack(XA3, "3종판(exp_070 대조군)")
A4 = axis_pack(XA4, "4팔 완성판")
M["점추정"] = {"3종판": A3, "4팔완성판": A4}
M["exp_070_재현대조"] = {
    "exp_070_Δ_pts": 4.618141264278734,
    "우리_3종판_Δ_pts": A3["Δ_pts"],
    "abs_diff": abs(A3["Δ_pts"] - 4.618141264278734),
    "exp_070_조건부argmax": 0.8581420253062062,
    "우리_3종판_조건부argmax": A3["조건부_argmax"],
    "판정": "PASS" if abs(A3["Δ_pts"] - 4.618141264278734) < 1e-9 else "FAIL"}
print(f"[점추정] 3종판 Δ(0.70) = {A3['Δ_pts']:+.5f} (exp_070 +4.61814, "
      f"재현 {M['exp_070_재현대조']['판정']})   4팔 Δ(0.70) = {A4['Δ_pts']:+.5f}  "
      f"argmax {A3['조건부_argmax']:.4f} → {A4['조건부_argmax']:.4f}", flush=True)
M["wall"] = {"점추정_min": (time.time() - t) / 60.0}
json.dump(M, open(f"{OUT}/stage3_partial.json", "w"), indent=1, ensure_ascii=False)

# ── k 민감도 (점추정만) ────────────────────────────────────────────
KS = {}
EP = {s: np.load(f"{E}/cache/p24_epochs_seed{s}.npy").astype(np.float64)
      for s in (42, 777, 2024, 101)}
TI = json.load(open(f"{OUT}/train071.json"))
for kk in [2, 4, 6, 8, 10, 12, 14, 16, 20, 25]:
    pk = expit(np.mean([lg(EP[s][kk - 1]) for s in (42, 777, 2024, 101)], axis=0))
    zk = (W_LGB * lg(V["p_lgb"]) + W_CAT * lg(V["p_cat"]) +
          W_NN * lg(pk) + W_LEAF * lg(V["p_leaf"]))
    Xk = np.column_stack([lg(expit(zk)), L_q, L_ebm, off, L_aa, L_t70])
    sd_ = score_z(Xk @ vmap_A(DEP))[1]
    th = DEP.copy(); th[0] = 0.70
    s7 = score_z(Xk @ vmap_A(th))[1]

    def fk(x, Xk=Xk):
        t2 = DEP.copy(); t2[0] = x
        return score_z(Xk @ vmap_A(t2))[0]

    rk = minimize_scalar(fk, bounds=(0.0, 1.0), method="bounded",
                         options={"xatol": 1e-6})
    KS[str(kk)] = {"Δ_pts_0.70": s7 - sd_, "조건부_argmax": float(rk.x),
                   "nn단독_재센터링점수": score_z(lg(pk))[1]}
best_ep = {int(a): b for a, b in TI["best_epoch_per_seed"].items()}
pb = expit(np.mean([lg(EP[s][best_ep[s] - 1]) for s in (42, 777, 2024, 101)], axis=0))
zb = (W_LGB * lg(V["p_lgb"]) + W_CAT * lg(V["p_cat"]) +
      W_NN * lg(pb) + W_LEAF * lg(V["p_leaf"]))
Xb = np.column_stack([lg(expit(zb)), L_q, L_ebm, off, L_aa, L_t70])
sdb = score_z(Xb @ vmap_A(DEP))[1]
thb = DEP.copy(); thb[0] = 0.70

def fb(x):
    t2 = DEP.copy(); t2[0] = x
    return score_z(Xb @ vmap_A(t2))[0]

rb = minimize_scalar(fb, bounds=(0.0, 1.0), method="bounded", options={"xatol": 1e-6})
KS["best_epoch판"] = {"Δ_pts_0.70": score_z(Xb @ vmap_A(thb))[1] - sdb,
                      "조건부_argmax": float(rb.x),
                      "nn단독_재센터링점수": score_z(lg(pb))[1],
                      "epochs": best_ep}
M["k_민감도"] = KS
print("[k민감도] " + " ".join(f"{k_}:{v['Δ_pts_0.70']:+.3f}/{v['조건부_argmax']:.3f}"
                              for k_, v in KS.items()), flush=True)
json.dump(M, open(f"{OUT}/stage3_partial.json", "w"), indent=1, ensure_ascii=False)

# ── 투수 블록 부트스트랩 (3종판·4팔 짝지어 같은 리샘플 위에서) ─────
t = time.time()


def rep(r_):
    rg = np.random.default_rng([SEED_BOOT, r_])
    cnt = np.bincount(rg.integers(0, G, G), minlength=G).astype(np.float64)
    u = cnt[pit_codes]
    U = float(u.sum())
    yb = float((u * y).sum() / U)

    def sc(XA, th):
        z = XA @ vmap_A(th)
        c = 0.0
        for _ in range(40):
            p = expit(z + c)
            mu = float((u * p).sum() / U)
            if abs(mu - yb) < 1e-14:
                break
            dd = (mu - yb) / float((u * p * (1 - p)).sum() / U)
            c -= dd
            if abs(dd) < 1e-14:
                break
        p = expit(z + c)
        return float((u * (p - y) ** 2).sum() / U)

    o = []
    for XA in (XA3, XA4):
        b_d = sc(XA, DEP)
        th_ = DEP.copy(); th_[0] = 0.70
        b_7 = sc(XA, th_)

        def f1(x, XA=XA):
            t2 = DEP.copy(); t2[0] = x
            return sc(XA, t2)

        rr = minimize_scalar(f1, bounds=(0.0, 1.0), method="bounded",
                             options={"xatol": 1e-5})
        o += [K * (b_d - b_7), float(rr.x), K * (b_d - rr.fun)]
    return o


with Pool(8) as pool:
    W_ = np.array(pool.map(rep, range(R_BOOT), chunksize=25))
ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
BS = {}
for j, nm in enumerate(("3종판", "4팔완성판")):
    c = j * 3
    BS[nm] = {"Δ(0.70)_부트_CI95": ci(W_[:, c]), "Δ(0.70)_부트_평균": float(W_[:, c].mean()),
              "Δ(0.70)_부트_양수비율": float((W_[:, c] > 0).mean()),
              "조건부_argmax_부트_CI95": ci(W_[:, c + 1]),
              "조건부_argmax_부트_중앙": float(np.median(W_[:, c + 1])),
              "조건부argmax_Δ_부트_CI95": ci(W_[:, c + 2])}
dd = W_[:, 3] - W_[:, 0]
BS["짝지은_차이_4팔−3종판"] = {"Δ차_부트_CI95": ci(dd), "Δ차_부트_평균": float(dd.mean()),
                              "Δ차_양수비율": float((dd > 0).mean()),
                              "점추정_차": A4["Δ_pts"] - A3["Δ_pts"],
                              "해설": "같은 리샘플 위의 짝지은 차 — 두 CI 겹침보다 예민하다"}
M["부트스트랩"] = dict(BS, R_BOOT=R_BOOT, SEED_BOOT=SEED_BOOT,
                       방식="투수 블록 재표집(가중), exp_070 followup070 과 동일 RNG")
M["부트_exp070_재현대조"] = {
    "exp_070_CI": [0.28391923018694304, 8.87803124864863],
    "우리_3종판_CI": BS["3종판"]["Δ(0.70)_부트_CI95"],
    "exp_070_양수비율": 0.981,
    "우리_3종판_양수비율": BS["3종판"]["Δ(0.70)_부트_양수비율"]}
M["wall"]["부트_min"] = (time.time() - t) / 60.0
M["wall"]["stage3_총_min"] = (time.time() - T0) / 60.0
json.dump(M, open(f"{OUT}/stage3.json", "w"), indent=1, ensure_ascii=False)
print(f"[부트] {R_BOOT}회 {M['wall']['부트_min']:.2f}분  "
      f"3종판 CI {BS['3종판']['Δ(0.70)_부트_CI95']}  "
      f"4팔 CI {BS['4팔완성판']['Δ(0.70)_부트_CI95']}  "
      f"짝지은차 {BS['짝지은_차이_4팔−3종판']['Δ차_부트_평균']:+.4f} "
      f"양수 {BS['짝지은_차이_4팔−3종판']['Δ차_양수비율']:.3f}", flush=True)
print("DONE_STAGE3", flush=True)
