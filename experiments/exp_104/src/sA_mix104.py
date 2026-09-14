"""exp_104 축A 평가 — 결함면 기각 30지점을 **고친 면 정직 혼합 화면**에서 다시 읽는다.

면은 exp_103b `s2_mixeval.py` / exp_102 `sA_mix.py` / exp_097 `sB1_rebase.py` 를 그대로 쓴다.
게이트: 기준선(94열·1632스텝·5시드) 치환 argmax 가 **26.2698** 을 재현해야 한다.

🔴 사전 등록(발사 전 확정, S8 — 결과 보고 바꾸지 않는다):
      혼합 이득 − 26.2698 ≥ **+2.5**   AND   짝지은 투수 블록 부트스트랩 **t ≥ 2.0**
🔴 ρ 로 판정하지 않는다(S83). ρ·단독점수는 진단 열로만 적는다.
🔴 학습량 동반(S183·S94): 각 지점을 {816, 1632, 2720} 세 스텝에서 굽고 **자기 argmax 스텝**으로
   비교한다. 기준선도 같은 세 스텝에서 자기 argmax 를 잡아 함께 적는다.
🔴 불확실성 두 성분(S178): ① 짝지은 부트스트랩 se ② **적합-간 sd** —
   exp_102 가 고친 면에서 구운 1632스텝 25시드를 **서로소 5시드 묶음 5개**로 갈라 직접 잰다.
   같은 하니스·같은 레시피라 기댓값이 같아야 하는 판본 5개다.
"""
import hashlib, json, os, time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "4"
import numpy as np
import pandas as pd
from scipy.special import expit, logit

H = "/home/<user>/SungjinPark"
B97, B102, B = f"{H}/exp097", f"{H}/exp102", f"{H}/exp104"
OOT, OOT3, OOT4 = f"{H}/exp069/oot", f"{H}/exp070/oot3", f"{H}/exp071/out"
EPS, REF = 1e-6, 0.2498069
K = 100000.0 / REF
S5 = [42, 777, 2024, 101, 7]
STEPS = [816, 1632, 2720]
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
t0 = time.time()

# ── 면 (exp_103b s2_mixeval.py 그대로) ───────────────────────────────────
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
    bs = bs_[PICKS].sum(1) / BSIZE[PICKS].sum(1)
    m = float(d.mean())
    se = float(bs.std(ddof=1))
    return {"gain_pts": -m * K, "boot_se_pts": se * K,
            "t": (-m / se) if se > 0 else None,
            "ci95_pts": [float(-np.percentile(bs, 97.5) * K),
                         float(-np.percentile(bs, 2.5) * K)]}


SHA = {}


def zload(path):
    z = np.load(path)
    SHA[os.path.basename(path)] = hashlib.sha256(
        np.ascontiguousarray(z).tobytes()).hexdigest()[:16]
    return z


GRID = [round(x, 4) for x in np.arange(0.0, 0.6001, 0.01)]


def replace_curve(dz):
    cur = [{"w": w, "gain": sc(root((1 - w) * Z_MATE + w * dz), "abd") - BASE_ABD}
           for w in GRID]
    return cur, max(cur, key=lambda c: c["gain"])


z0 = root(Z_MATE)
BASE_RC, BASE_ABD = sc(z0, "rc"), sc(z0, "abd")
assert abs(BASE_RC - BASE_EXPECT) < 1e-9, BASE_RC

META = json.load(open(f"{B}/out/jobsA_meta.json"))
POINTS = META["points"]
R = {"exp_id": "exp_104", "stage": "sA_mix104",
     "면": "exp_103b s2_mixeval.py / exp_102 sA_mix.py 면 그대로 (고친 f64 OOT 입력, S134)",
     "n_rows": int(len(y)), "seeds": S5, "steps": STEPS,
     "baseline_w0": {"rc": BASE_RC, "expected": BASE_EXPECT, "abd": BASE_ABD},
     "prereg": {"rule": f"gain - {PREREG_BASELINE} >= +{PREREG_DELTA} AND paired "
                        f"pitcher-block bootstrap t >= {PREREG_T}",
                "baseline_gain": PREREG_BASELINE, "delta_min": PREREG_DELTA,
                "t_min": PREREG_T,
                "source": "exp_104 config.yaml — 발사 전 확정, 결과 보고 안 바꿨다 (S8)"},
     "🔴판정에_ρ_미사용": "S83. ρ·단독점수는 진단 열이다."}

# ── 기준선 팔: 고친 면 94열, 세 스텝 ────────────────────────────────────
BASE_ARM, BASE_TAB = {}, {}
for k in STEPS:
    dz = np.mean([zload(f"{B97}/out/preds/z_F{k}_f64o_s{s}.npy") for s in S5], 0)
    cur, best = replace_curve(dz)
    BASE_ARM[k] = dz
    BASE_TAB[k] = {"argmax": best, "alone_recentred": sc(dz, "rc"),
                   "z_sd": float(dz.std()),
                   "rho_vs_z_mate": float(np.corrcoef(dz, Z_MATE)[0, 1])}
    print(f"BASE k={k:5d} alone {BASE_TAB[k]['alone_recentred']:8.2f} "
          f"argmax w={best['w']:.2f} gain {best['gain']:+9.4f}", flush=True)
BK = max(STEPS, key=lambda k: BASE_TAB[k]["argmax"]["gain"])
R["baseline_arm"] = {
    "per_step": BASE_TAB, "own_argmax_step": BK,
    "own_argmax_gain": BASE_TAB[BK]["argmax"]["gain"],
    "gate_1632_reproduces_26.2698": {
        "got": BASE_TAB[1632]["argmax"]["gain"], "expected": PREREG_BASELINE,
        "abs_diff": abs(BASE_TAB[1632]["argmax"]["gain"] - PREREG_BASELINE),
        "PASS": abs(BASE_TAB[1632]["argmax"]["gain"] - PREREG_BASELINE) < 0.01}}
assert R["baseline_arm"]["gate_1632_reproduces_26.2698"]["PASS"], BASE_TAB[1632]

W_REF = BASE_TAB[1632]["argmax"]["w"]
P_REF = prob(root((1 - W_REF) * Z_MATE + W_REF * BASE_ARM[1632]), "abd")
BKW = BASE_TAB[BK]["argmax"]["w"]
P_REF_OWN = prob(root((1 - BKW) * Z_MATE + BKW * BASE_ARM[BK]), "abd")

# ── 적합-간 sd (S178) — 고친 면 1632스텝 25시드를 서로소 5묶음으로 ────────
EXTRA25 = [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73,
           79, 83, 89]
GROUPS = [S5] + [EXTRA25[i * 5:(i + 1) * 5] for i in range(4)]
rep_gain = []
for gi, grp in enumerate(GROUPS):
    zs = []
    for s in grp:
        p1, p2 = (f"{B97}/out/preds/z_F1632_f64o_s{s}.npy",
                  f"{B102}/out/preds/z_F1632_f64o_s{s}.npy")
        zs.append(zload(p1 if os.path.exists(p1) else p2))
    _, bb = replace_curve(np.mean(zs, 0))
    rep_gain.append(bb["gain"])
R["적합간_sd_S178"] = {
    "what": "같은 레시피(94열·1632스텝·5시드)의 서로소 시드 묶음 5개 — 기댓값이 같아야 하는 판본",
    "groups": GROUPS, "gains": rep_gain,
    "mean": float(np.mean(rep_gain)), "sd": float(np.std(rep_gain, ddof=1)),
    "se_of_a_5seed_point": float(np.std(rep_gain, ddof=1)),
    "🔴": "부트스트랩 se 는 같은 예측 벡터 위의 표본 잡음만 잰다. 이 sd 는 시드 뽑기가 "
          "바뀔 때 5시드 평균 팔 자체가 움직이는 몫이고, 두 성분은 더해야 한다."}
print("적합간 sd:", R["적합간_sd_S178"]["sd"], rep_gain, flush=True)

# ── 후보 지점 ───────────────────────────────────────────────────────────
OFF_VALID = None
TAB, MISSING = {}, []
for nm, meta in POINTS.items():
    per = {}
    for k in STEPS:
        try:
            zs = [zload(f"{B}/out/preds/z_{nm}__k{k}_s{s}.npy") for s in S5]
        except FileNotFoundError:
            MISSING.append(f"{nm}__k{k}")
            continue
        dz = np.mean(zs, 0)
        cur, best = replace_curve(dz)
        per[k] = {"argmax": best, "alone_recentred": sc(dz, "rc"),
                  "z_sd": float(dz.std()),
                  "rho_vs_z_mate": float(np.corrcoef(dz, Z_MATE)[0, 1]),
                  "rho_vs_base94_samestep": float(
                      np.corrcoef(dz, BASE_ARM[k])[0, 1]),
                  "gain_at_w0.43": [c for c in cur
                                    if abs(c["w"] - 0.43) < 1e-9][0]["gain"],
                  "_z": dz}
    if not per:
        continue
    ok = max(per, key=lambda k: per[k]["argmax"]["gain"])
    dz, w = per[ok]["_z"], per[ok]["argmax"]["w"]
    zb = root((1 - w) * Z_MATE + w * dz)
    pb = prob(zb, "abd")
    g = per[ok]["argmax"]["gain"]
    cmpx = boot_t((pb - y) ** 2 - (P_REF - y) ** 2)
    cmpo = boot_t((pb - y) ** 2 - (P_REF_OWN - y) ** 2)
    pc = prob(root((1 - W_REF) * Z_MATE + W_REF * dz), "abd")
    cmpc = boot_t((pc - y) ** 2 - (P_REF - y) ** 2)
    TAB[nm] = {
        "axis": meta["axis"], "desc": meta["desc"], "hid": meta["hid"],
        "drop": meta["drop"], "opt": meta["opt"], "n_masked": meta["n_masked"],
        "loss": meta["loss"], "target": meta["target"] or None,
        "own_argmax_step": ok, "own_argmax_w": w, "gain": g,
        "delta_vs_prereg_baseline": g - PREREG_BASELINE,
        "gain_sd_matched_at_argmax": sc(sd_match(zb, z0), "abd") - BASE_ABD,
        "vs_base94_1632_paired": cmpx,
        "vs_base94_ownstep_paired": cmpo,
        "vs_base94_common_w0.43_paired": cmpc,
        "alone_recentred": per[ok]["alone_recentred"],
        "rho_vs_z_mate": per[ok]["rho_vs_z_mate"],
        "per_step": {str(k): {kk: vv for kk, vv in v.items() if kk != "_z"}
                     for k, v in per.items()},
        "PASS_size": bool(g - PREREG_BASELINE >= PREREG_DELTA),
        "PASS_t": bool(cmpx["t"] is not None and cmpx["t"] >= PREREG_T)}
    TAB[nm]["PASS"] = bool(TAB[nm]["PASS_size"] and TAB[nm]["PASS_t"])
    print("%-18s step %5d w %.2f gain %+9.4f (d %+8.4f) t %+6.3f  "
          "alone %8.2f rho %.5f %s"
          % (nm, ok, w, g, g - PREREG_BASELINE, cmpx["t"] or float("nan"),
             TAB[nm]["alone_recentred"], TAB[nm]["rho_vs_z_mate"],
             "PASS" if TAB[nm]["PASS"] else ""), flush=True)

    # 🔴 축 F 의 두 번째 읽기: 같은 적합에서 z + off (잔차에 기저를 되돌린 판본)
    if nm == "F1_resid_off":
        if OFF_VALID is None:
            offa = np.load(f"{H}/exp092/work/offF_all.npy")
            vam = np.load(f"{H}/exp102/work/valid_mask.npy")
            OFF_VALID = offa[vam]
            assert np.isfinite(OFF_VALID).all() and len(OFF_VALID) == len(y)
        per2 = {}
        for k, v in per.items():
            cur2, best2 = replace_curve(v["_z"] + OFF_VALID)
            per2[k] = {"argmax": best2,
                       "alone_recentred": sc(v["_z"] + OFF_VALID, "rc")}
        ok2 = max(per2, key=lambda k: per2[k]["argmax"]["gain"])
        w2 = per2[ok2]["argmax"]["w"]
        z2 = root((1 - w2) * Z_MATE + w2 * (per[ok2]["_z"] + OFF_VALID))
        c2 = boot_t((prob(z2, "abd") - y) ** 2 - (P_REF - y) ** 2)
        g2 = per2[ok2]["argmax"]["gain"]
        TAB["F1_resid_plus_off"] = {
            "axis": "F_residual",
            "desc": "F1 과 같은 적합, 평가에서 z + out-of-time 오프셋 (기저 되돌림 읽기)",
            "own_argmax_step": ok2, "own_argmax_w": w2, "gain": g2,
            "delta_vs_prereg_baseline": g2 - PREREG_BASELINE,
            "vs_base94_1632_paired": c2,
            "alone_recentred": per2[ok2]["alone_recentred"],
            "per_step": {str(k): v["argmax"] for k, v in per2.items()},
            "PASS_size": bool(g2 - PREREG_BASELINE >= PREREG_DELTA),
            "PASS_t": bool(c2["t"] is not None and c2["t"] >= PREREG_T)}
        TAB["F1_resid_plus_off"]["PASS"] = bool(
            TAB["F1_resid_plus_off"]["PASS_size"]
            and TAB["F1_resid_plus_off"]["PASS_t"])
        print("%-18s step %5d w %.2f gain %+9.4f (d %+8.4f) t %+6.3f"
              % ("F1_resid_plus_off", ok2, w2, g2, g2 - PREREG_BASELINE,
                 c2["t"] or float("nan")), flush=True)

R["missing_artifacts"] = MISSING
R["표"] = TAB

# ── S104 개수 맞춤 귀무 대조 — 이름 붙인 마스크 vs 같은 개수 무작위 마스크 ──
PAIRS = [("D1_no_sp", "D5m06_rand", 6), ("D2_no_ins", "Bm16(exp_102)", 16),
         ("D3_no_i3anom_p6", "D5m13_rand", 13), ("D4_sit_cat_only", "D5m83_rand", 83)]
CM = {}
for a_, b_, k in PAIRS:
    if a_ not in TAB:
        continue
    if b_.startswith("Bm16"):
        try:
            bz = np.mean([zload(f"{B102}/out/preds/z_Bm16_f64o_s{s}.npy")
                          for s in S5], 0)
            _, bb = replace_curve(bz)
            bg, bw = bb["gain"], bb["w"]
        except FileNotFoundError:
            continue
    else:
        if b_ not in TAB:
            continue
        bg = TAB[b_]["per_step"]["1632"]["argmax"]["gain"]
        bw = TAB[b_]["per_step"]["1632"]["argmax"]["w"]
        bz = np.mean([zload(f"{B}/out/preds/z_{b_}__k1632_s{s}.npy") for s in S5], 0)
    az = np.mean([zload(f"{B}/out/preds/z_{a_}__k1632_s{s}.npy") for s in S5], 0)
    _, ab = replace_curve(az)
    pa = prob(root((1 - ab["w"]) * Z_MATE + ab["w"] * az), "abd")
    pb_ = prob(root((1 - bw) * Z_MATE + bw * bz), "abd")
    CM[f"{a_}_minus_{b_}"] = {
        "n_masked": k, "named_gain": ab["gain"], "random_gain": bg,
        "named_minus_random": ab["gain"] - bg,
        "paired": boot_t((pa - y) ** 2 - (pb_ - y) ** 2),
        "step": 1632}
R["S104_개수맞춤_귀무대조"] = {
    "왜": "이름 붙인 계열을 빼는 것이 같은 개수 무작위보다 나은가. exp_092 는 결함면에서 "
          "`ins_` 16열이 무작위 16열보다 −10.891 이라 했다(S104). 고친 면에서 다시 잰다.",
    "표": CM}

# ── 무작위 47열 3연발의 퍼짐 — 이 화면의 '귀무 가족' 분산 ────────────────
R47 = [TAB[n]["per_step"]["1632"]["argmax"]["gain"]
       for n in ("D5h1_rand47", "D5h2_rand47", "D5h3_rand47") if n in TAB]
if len(R47) == 3:
    R["🔴귀무가족_퍼짐_무작위47열"] = {
        "gains": R47, "mean": float(np.mean(R47)), "sd": float(np.std(R47, ddof=1)),
        "range": float(max(R47) - min(R47)),
        "🔴": "기댓값이 같은 세 판본(무작위 47열 마스킹)이 이만큼 벌어진다. "
              "**팔을 크게 바꾸는 변형에는 5시드 적합-간 sd(94열 기준선에서 잰 값)를 "
              "그대로 쓰면 안 된다** — 그 계열의 귀무 퍼짐이 훨씬 크다. "
              "사전 등록 크기 문턱 +2.5 는 이 계열에서 무작위 마스크 하나가 그냥 넘는다."}
qual = [k for k, v in TAB.items() if v["PASS"]]
near = sorted(TAB.items(), key=lambda kv: -kv[1]["delta_vs_prereg_baseline"])[:6]
R["VERDICT"] = {
    "rule": R["prereg"]["rule"],
    "n_points_read": len(TAB), "qualifying": qual,
    "PASS": bool(qual) and not MISSING,
    "closest_points": [{"point": k, "delta": v["delta_vs_prereg_baseline"],
                        "t": v["vs_base94_1632_paired"]["t"],
                        "step": v["own_argmax_step"]} for k, v in near],
    "🔴": "산출물 누락이 있으면 PASS 를 주장하지 않는다" if MISSING else "산출물 전량 존재"}
_bse = float(np.median([v["vs_base94_1632_paired"]["boot_se_pts"]
                        for v in TAB.values()]))
_fsd = R["적합간_sd_S178"]["sd"]
R["검출한계"] = {
    "boot_se_pts_median": _bse,
    "적합간_sd_pts_한_점": _fsd,
    "적합간_sd_pts_두_점_차": _fsd * np.sqrt(2.0),
    "결합_1sigma_pts": float(np.hypot(_bse, _fsd * np.sqrt(2.0))),
    "🔴_두_성분": "① 부트스트랩 se 는 **고정된 예측 벡터 위의 표본 잡음**만 잰다. "
                  "② 적합-간 sd 는 5시드 묶음을 다시 뽑으면 팔 자체가 움직이는 몫이다. "
                  "후보 − 기준선은 서로 독립인 두 뽑기의 차이라 ②에 √2 가 붙는다. "
                  "둘은 독립이므로 분산으로 더한다(S178).",
    "🔴_기준선_26.2698_도_한_뽑기다": {
        "replicate_mean": R["적합간_sd_S178"]["mean"],
        "reference_group_S5": R["적합간_sd_S178"]["gains"][0],
        "reference_minus_replicate_mean":
            R["적합간_sd_S178"]["gains"][0] - R["적합간_sd_S178"]["mean"],
        "뜻": "사전 등록 기준선 26.2698 은 시드 묶음 5개 중 하나이고 평균보다 위에 있다. "
              "후보의 델타는 그만큼 낮게 읽힌다 — 문턱은 그대로 두고(S8) 이 편의를 함께 적는다."},
    "요구_효과크기_t2_결합": 2.0 * float(np.hypot(_bse, _fsd * np.sqrt(2.0)))}
comb = R["검출한계"]["결합_1sigma_pts"]
for k, v in TAB.items():
    v["t_combined_1sigma"] = v["delta_vs_prereg_baseline"] / comb
R["z_sha16_count"] = len(SHA)
R["wall_sec"] = round(time.time() - t0, 1)
json.dump(R, open(f"{B}/out/sA_mix104.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({"VERDICT": R["VERDICT"], "검출한계": R["검출한계"],
                  "baseline_arm": {k: vv for k, vv in R["baseline_arm"].items()
                                   if k != "per_step"}},
                 ensure_ascii=False, indent=1))
