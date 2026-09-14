"""exp_113 — 정직면 혼합 화면 평가. 면은 exp_102 sA_mix.py / exp_097 sB1_rebase.py 그대로.

🔴 사전 등록 (fit 결과를 하나도 보기 전에 확정 — S8):
  판정량 = argmax_{w,steps} gain(실제 열) − argmax_{w,steps} gain(**N1 개수맞춤 무작위 열**)
           🔴 무변형 기준선 대비가 아니다 — 열을 더하는 것 자체에 고정 대가가 있다 (S183).
  1σ     = sqrt( 짝지은 투수블록 부트스트랩 se² + 적합간 sd² )
           적합간 sd 는 이 화면에서 **서로소 5시드 묶음 3개**로 직접 재고,
           exp_107 실측 두 점 차 **2.1571** 보다 작게 나오면 2.1571 을 쓴다 (보수적).
  승격   = t = 판정량 / 1σ >= 2.0  AND  판정량 >= **+2.4390 점**
           (= LB 검출한계 +1.4 ÷ 팀 원장 '팔 추가' 전이계수 0.574)
  귀무를 셋 쓴다 — N1(기준선) · N2(같은 열 (season,pitcher_id) 순열) ·
  N2b(pitcher_id 안 순열, 시즌 가로지름). **둘 이상에서 판정이 뒤집히는지 본다** (S179).
"""
import hashlib
import itertools
import json
import os
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "3"
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import expit, logit  # noqa: E402

H = "/home/<user>/SungjinPark"
B97, B102, B = f"{H}/exp097", f"{H}/exp102", f"{H}/exp113"
OOT, OOT3, OOT4 = f"{H}/exp069/oot", f"{H}/exp070/oot3", f"{H}/exp071/out"
EPS, REF = 1e-6, 0.2498069
K = 100000.0 / REF
BUNDLE = {0: [42, 777, 2024, 101, 7], 1: [11, 13, 17, 19, 23], 2: [29, 31, 37, 41, 43],
          3: [47, 53, 59, 61, 67], 4: [71, 73, 79, 83, 89]}
S5 = BUNDLE[0]
AA = (42, 7, 2024)
W4 = {"LGB": 0.05, "CAT": 0.68, "NN": 0.15, "LEAF": 0.12}
BO = dict(alpha=1.07911, beta_skew=-0.059151, c_aa=0.25244, c_T70=0.04756,
          w_out=0.7, w_v6_inner=0.6668397489, w_ebm=-0.1, beta_ins=0.03)
BASE_EXPECT = 928.501461058684
REF_GAIN = 26.269817382706947
STEPS = [1088, 1632, 2176, 2720]   # 1088 은 argmax 가 격자 아래 경계에 붙지 않게 넓힌 것
MAIN = ["P", "R", "PR"]
TRANSFER = 0.574
LB_FLOOR = 1.4
PREREG_MIN_PTS = LB_FLOOR / TRANSFER
PREREG_T = 2.0
SIGMA_FIT_FLOOR = 2.1570652801242445          # exp_107 실측 두 점 차
lg = lambda p: logit(np.clip(np.asarray(p, np.float64), EPS, 1 - EPS))
pts = lambda b: 100000.0 * (1.0 - b / REF)
t0 = time.time()

# ── 면 ───────────────────────────────────────────────────────────
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
sd_match = lambda zn, zr: ((zn - zn.mean()) / zn.std()) * zr.std() + zr.mean()

rng = np.random.default_rng(20260824)
uq, inv = np.unique(pit, return_inverse=True)
nb = len(uq)
order = np.argsort(inv, kind="stable")
st = np.searchsorted(inv[order], np.arange(nb))
en = np.append(st[1:], len(order))
BSIZE = (en - st).astype(np.float64)
PICKS = np.stack([rng.integers(0, nb, nb) for _ in range(1000)])


def boot(d):
    bs_ = np.array([d[order[st[j]:en[j]]].sum() for j in range(nb)])
    bs = bs_[PICKS].sum(1) / BSIZE[PICKS].sum(1)
    se = float(bs.std(ddof=1))
    return {"gain_pts": -float(d.mean()) * K, "boot_se_pts": se * K,
            "boot_t": (-float(d.mean()) / se) if se > 0 else None,
            "ci95_pts": [float(-np.percentile(bs, 97.5) * K),
                         float(-np.percentile(bs, 2.5) * K)]}


# ── 팔 로드 ──────────────────────────────────────────────────────
def base94(seed):
    for b in (B97, B102):
        p = f"{b}/out/preds/z_F1632_f64o_s{seed}.npy"
        if os.path.exists(p):
            return np.load(p)
    raise FileNotFoundError(seed)


def zvar(tag, k, seeds):
    return np.mean([np.load(f"{B}/out/preds/z_{tag}__k{k}_s{s}.npy") for s in seeds], 0)


R = {"exp_id": "exp_113", "stage": "f4_eval",
     "면": "exp_102 sA_mix.py / exp_097 sB1_rebase.py (고친 f64 OOT 입력, S134)",
     "prereg": {
         "판정량": "argmax_{w,steps} gain(실제) − argmax_{w,steps} gain(N1 개수맞춤 무작위)",
         "왜_N1_기준선": "열을 더하는 것 자체에 고정 대가가 있다 (S183: 3열에 약 −4점)",
         "min_pts": PREREG_MIN_PTS, "min_t": PREREG_T,
         "전이비": TRANSFER, "LB_검출한계": LB_FLOOR,
         "sigma": "sqrt(부트스트랩 se² + 적합간 sd²), 적합간 sd 는 서로소 5시드 묶음 3개로 "
                  f"직접 재되 exp_107 실측 {SIGMA_FIT_FLOOR:.4f} 보다 작으면 그것을 쓴다",
         "귀무_셋": ["N1(기준선)", "N2 (season,pitcher_id) 순열", "N2b pitcher_id 순열"]},
     "열_선택": json.load(open(f"{B}/out/f3_screen.json"))["SEL_PR"]}

z0 = root(Z_MATE)
base_abd, base_rc = sc(z0, "abd"), sc(z0, "rc")
R["G1_surface"] = {"w0_rc": base_rc, "expected": BASE_EXPECT,
                   "abs_diff": abs(base_rc - BASE_EXPECT),
                   "PASS": bool(abs(base_rc - BASE_EXPECT) < 1e-9)}
assert R["G1_surface"]["PASS"], base_rc

GRID = [round(x, 4) for x in np.arange(0.0, 0.6001, 0.01)]


def curve(zd):
    cur = [{"w": w, "gain": sc(root((1 - w) * Z_MATE + w * zd), "abd") - base_abd}
           for w in GRID]
    return cur, max(cur, key=lambda c: c["gain"])


# ── 기준선 94열: 서로소 5묶음 (S194 재현) ───────────────────────
ZB = {b: np.mean([base94(s) for s in BUNDLE[b]], 0) for b in range(5)}
bnd_gain = {}
for b in range(5):
    _, best = curve(ZB[b])
    bnd_gain[b] = best
R["G2_base94_bundles"] = {
    "gains": {b: bnd_gain[b]["gain"] for b in bnd_gain},
    "argmax_w": {b: bnd_gain[b]["w"] for b in bnd_gain},
    "mean": float(np.mean([bnd_gain[b]["gain"] for b in bnd_gain])),
    "sd_ddof1": float(np.std([bnd_gain[b]["gain"] for b in bnd_gain], ddof=1)),
    "bundle0_expected": REF_GAIN,
    "abs_diff_bundle0": abs(bnd_gain[0]["gain"] - REF_GAIN),
    "PASS": bool(abs(bnd_gain[0]["gain"] - REF_GAIN) < 1e-6)}
assert R["G2_base94_bundles"]["PASS"], bnd_gain[0]
Z_B94 = ZB[0]

# ── 판본 × 스텝 ─────────────────────────────────────────────────
TAGS = (MAIN + [f"N1s_{t}" for t in MAIN] + [f"N2s_{t}" for t in MAIN] + ["N1g_PR"]
        + [f"N1_{t}" for t in MAIN] + [f"N2_{t}" for t in MAIN] + ["N2b_PR"])
# 🔴 N1_*·N2_*·N2b_PR 는 **결함 귀무**다 — (season, pitcher_id) 안 순열이라
#    라벨 상관을 거의 그대로 남긴다(f3b_nulls.json). 버리지 않고 보고하되 판정에는 안 쓴다.
NULLS_OK = {t: [f"N1s_{t}", f"N2s_{t}"] + (["N1g_PR"] if t == "PR" else []) for t in MAIN}
NULLS_BAD = {t: [f"N1_{t}", f"N2_{t}"] + (["N2b_PR"] if t == "PR" else []) for t in MAIN}
R["variants"] = {}
BESTZ = {}
for tag in TAGS:
    per_step = {}
    for k in STEPS:
        if not all(os.path.exists(f"{B}/out/preds/z_{tag}__k{k}_s{s}.npy") for s in S5):
            continue
        zd = zvar(tag, k, S5)
        cur, best = curve(zd)
        per_step[k] = {
            "argmax": best, "alone_recentred_score": sc(zd, "rc"),
            "z_sd": float(zd.std()),
            "rho_vs_z_mate": float(np.corrcoef(zd, Z_MATE)[0, 1]),
            "rho_vs_base94": float(np.corrcoef(zd, Z_B94)[0, 1]),
            "gain_sd_matched": sc(sd_match(
                root((1 - best["w"]) * Z_MATE + best["w"] * zd), z0), "abd") - base_abd,
            "gain_at_w0.43": [c for c in cur if abs(c["w"] - 0.43) < 1e-9][0]["gain"]}
        BESTZ[(tag, k)] = zd
    if not per_step:
        continue
    kb = max(per_step, key=lambda k: per_step[k]["argmax"]["gain"])
    R["variants"][tag] = {"per_step": per_step, "own_argmax_step": kb,
                          "own_argmax": per_step[kb]["argmax"]}
    print("%-8s k=%d w=%.2f gain %+9.4f alone %8.2f rho_mate %.5f"
          % (tag, kb, per_step[kb]["argmax"]["w"], per_step[kb]["argmax"]["gain"],
             per_step[kb]["alone_recentred_score"], per_step[kb]["rho_vs_z_mate"]),
          flush=True)


def mix_at(tag):
    v = R["variants"][tag]
    k, w = v["own_argmax_step"], v["own_argmax"]["w"]
    return root((1 - w) * Z_MATE + w * BESTZ[(tag, k)])


# ── 적합간 sd: 서로소 묶음 3개 (PR vs N1_PR @1632) ──────────────
diffs, bg = [], {}
for b in (0, 1, 2):
    ok = all(os.path.exists(f"{B}/out/preds/z_{t}__k1632_s{s}.npy")
             for t in ("PR", "N1s_PR") for s in BUNDLE[b])
    if not ok:
        continue
    gp = curve(zvar("PR", 1632, BUNDLE[b]))[1]["gain"]
    gn = curve(zvar("N1s_PR", 1632, BUNDLE[b]))[1]["gain"]
    bg[b] = {"PR": gp, "N1s_PR": gn, "diff": gp - gn}
    diffs.append(gp - gn)
sd_meas = float(np.std(diffs, ddof=1)) if len(diffs) >= 2 else None
SIGMA_FIT = max(sd_meas, SIGMA_FIT_FLOOR) if sd_meas is not None else SIGMA_FIT_FLOOR
R["sigma_fit"] = {"bundles": bg, "diffs": diffs,
                  "sd_measured_ddof1": sd_meas, "floor_exp107": SIGMA_FIT_FLOOR,
                  "sigma_fit_used": SIGMA_FIT,
                  "mean_diff_over_bundles": float(np.mean(diffs)) if diffs else None}

# ── 판정 ─────────────────────────────────────────────────────────
R["verdict"] = {}
for t in MAIN:
    if t not in R["variants"]:
        continue
    row = {"own_argmax": R["variants"][t]["own_argmax"],
           "own_argmax_step": R["variants"][t]["own_argmax_step"]}
    sq_real = (prob(mix_at(t), "abd") - y) ** 2
    for nul in NULLS_OK[t] + NULLS_BAD[t]:
        if nul not in R["variants"]:
            continue
        d = sq_real - (prob(mix_at(nul), "abd") - y) ** 2
        bt = boot(d)
        delta = (R["variants"][t]["own_argmax"]["gain"]
                 - R["variants"][nul]["own_argmax"]["gain"])
        sig = float(np.hypot(bt["boot_se_pts"], SIGMA_FIT))
        row[nul] = {"delta_pts": delta, "boot_se_pts": bt["boot_se_pts"],
                    "boot_only_t": bt["boot_t"], "sigma_combined_pts": sig,
                    "t_combined": delta / sig,
                    "lb_equivalent": delta * TRANSFER,
                    "결함_귀무": nul in NULLS_BAD[t],
                    "PASS": bool(delta >= PREREG_MIN_PTS and delta / sig >= PREREG_T)}
    row["vs_base94_bundle_mean"] = (R["variants"][t]["own_argmax"]["gain"]
                                    - R["G2_base94_bundles"]["mean"])
    R["verdict"][t] = row

prim = [(t, R["verdict"][t].get(f"N1s_{t}", {})) for t in R["verdict"]]
prim = [(t, v) for t, v in prim if v]
best = max(prim, key=lambda kv: kv[1]["delta_pts"]) if prim else None
R["VERDICT"] = {
    "best_variant": best[0] if best else None,
    "delta_vs_N1_pts": best[1]["delta_pts"] if best else None,
    "t_combined": best[1]["t_combined"] if best else None,
    "lb_equivalent": best[1]["lb_equivalent"] if best else None,
    "PASS": bool(best and best[1]["PASS"]),
    "primary_null": "N1s_* (season 안 순열) — 1차 N1 은 결함이라 판정에서 뺐다",
    "null_flip_check": {t: {n: R["verdict"][t][n]["delta_pts"]
                            for n in R["verdict"][t] if n.startswith("N")}
                        for t in R["verdict"]},
    "결함_귀무_기록": json.load(open(f"{B}/out/f3b_nulls.json"))["왜"]}
R["wall_sec"] = round(time.time() - t0, 1)
json.dump(R, open(f"{B}/out/f4_eval.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in R.items() if k != "variants"},
                 ensure_ascii=False, indent=1))
