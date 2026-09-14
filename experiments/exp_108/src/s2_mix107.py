"""exp_107 평가 — 고친 정직면 혼합 화면. 면·게이트는 exp_104 `sA_mix104.py` 를 그대로 승계한다.

🔴 사전 등록(발사 전 확정, S8 — 결과 보고 바꾸지 않는다):
      혼합 이득 − 26.2698 >= **+9.31점**  AND  짝지은 투수 블록 부트스트랩 **t >= 2.0**
   (+9.31 = 2 × 결합 1σ 4.654 = exp_104 가 이 화면에서 실측한 값, S187)
🔴 ρ 로 판정하지 않는다(S83). ρ·단독점수·S103 교환율은 **진단**이다 — 다만 이 실험의 절반이라
   미달이어도 전부 적는다.
🔴 각 지점을 자기 argmax 스텝·자기 argmax `w_deep` 에서 읽는다(S183·S97).
⚠️ 기준선 26.2698 은 5뽑기 중 하나다 — 서로소 5시드 묶음으로 다시 뽑아 평균을 함께 적는다.
"""
import hashlib, json, os, time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np
import pandas as pd
from scipy.special import expit, logit

H = "/home/<user>/SungjinPark"
B97, B102, B = f"{H}/exp097", f"{H}/exp102", f"{H}/exp107"
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
PREREG_DELTA = 9.31          # 2 × 4.654 (exp_104 실측 결합 1σ, S187)
PREREG_T = 2.0
PREREG_SIGMA = 4.654
RHO_UNIT_PRICE = 15.82       # S103 — ρ 0.01 내리기의 값어치(팔 단독 점수)
lg = lambda p: logit(np.clip(np.asarray(p, np.float64), EPS, 1 - EPS))
pts = lambda b: 100000.0 * (1.0 - b / REF)
t0 = time.time()

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


def boot_t(d):
    bs_ = np.array([d[order[st[j]:en[j]]].sum() for j in range(nb)])
    bs = bs_[PICKS].sum(1) / BSIZE[PICKS].sum(1)
    m, se = float(d.mean()), float(bs.std(ddof=1))
    return {"gain_pts": -m * K, "boot_se_pts": se * K,
            "t": (-m / se) if se > 0 else None,
            "ci95_pts": [float(-np.percentile(bs, 97.5) * K),
                         float(-np.percentile(bs, 2.5) * K)]}


SHA = {}


def zload(p):
    z = np.load(p)
    SHA[os.path.basename(p)] = hashlib.sha256(
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

R = {"exp_id": "exp_107", "stage": "s2_mix107",
     "면": "exp_104 sA_mix104.py 면 그대로 (고친 f64 OOT 입력, S134)",
     "n_rows": int(len(y)), "seeds": S5, "steps": STEPS,
     "baseline_w0": {"rc": BASE_RC, "expected": BASE_EXPECT, "abd": BASE_ABD},
     "prereg": {"rule": f"gain - {PREREG_BASELINE} >= +{PREREG_DELTA} AND paired "
                        f"pitcher-block bootstrap t >= {PREREG_T}",
                "baseline_gain": PREREG_BASELINE, "delta_min": PREREG_DELTA,
                "t_min": PREREG_T, "combined_1sigma_used": PREREG_SIGMA,
                "source": "exp_107 config.yaml (exp_104 실측 S187) — 발사 전 확정, "
                          "결과 보고 안 바꿨다 (S8)"}}

# ── 기준선 팔 ────────────────────────────────────────────────────────────
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
RHO_BASE = BASE_TAB[1632]["rho_vs_z_mate"]
ALONE_BASE = BASE_TAB[1632]["alone_recentred"]

# ── 기준선 재추첨: 서로소 5시드 묶음들 (config 요구 — 26.2698 은 한 뽑기다) ──
EXTRA20 = [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73,
           79, 83, 89]
GROUPS = [S5] + [EXTRA20[i * 5:(i + 1) * 5] for i in range(4)]
rep = []
for grp in GROUPS:
    zs = []
    for s in grp:
        p1, p2 = (f"{B97}/out/preds/z_F1632_f64o_s{s}.npy",
                  f"{B102}/out/preds/z_F1632_f64o_s{s}.npy")
        zs.append(zload(p1 if os.path.exists(p1) else p2))
    _, bb = replace_curve(np.mean(zs, 0))
    rep.append(bb["gain"])
R["🔴기준선_재추첨"] = {
    "왜": "사전 등록 기준선 26.2698 은 서로소 5시드 묶음 중 **한 뽑기**다. 평균을 함께 쓴다.",
    "groups": GROUPS, "gains_1632": rep,
    "mean": float(np.mean(rep)), "sd_ddof1": float(np.std(rep, ddof=1)),
    "prereg_draw_minus_mean": float(rep[0] - np.mean(rep)),
    "적합간_sd_한_점": float(np.std(rep, ddof=1)),
    "적합간_sd_두_점_차": float(np.std(rep, ddof=1) * np.sqrt(2.0))}
BASE_MEAN = float(np.mean(rep))
print("기준선 재추첨:", rep, "평균", BASE_MEAN, flush=True)

# ── 후보 지점 ───────────────────────────────────────────────────────────
META = {}
for ph in ("1", "2"):
    p = f"{B}/out/jobs{ph}_meta.json"
    if os.path.exists(p):
        META.update(json.load(open(p))["points"])
TAB, MISSING = {}, []
for nm, meta in META.items():
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
                  "rho_vs_base94_samestep": float(np.corrcoef(dz, BASE_ARM[k])[0, 1]),
                  "gain_at_w0.43": [c for c in cur
                                    if abs(c["w"] - 0.43) < 1e-9][0]["gain"], "_z": dz}
    if not per:
        continue
    ok = max(per, key=lambda k: per[k]["argmax"]["gain"])
    dz, w = per[ok]["_z"], per[ok]["argmax"]["w"]
    zb = root((1 - w) * Z_MATE + w * dz)
    pb = prob(zb, "abd")
    g = per[ok]["argmax"]["gain"]
    cmpx = boot_t((pb - y) ** 2 - (P_REF - y) ** 2)
    pc = prob(root((1 - W_REF) * Z_MATE + W_REF * dz), "abd")
    cmpc = boot_t((pc - y) ** 2 - (P_REF - y) ** 2)
    # S103 교환율 진단 — 1632 스텝(기준선과 같은 스텝)에서 읽는다
    d_rho = per[1632]["rho_vs_z_mate"] - RHO_BASE if 1632 in per else None
    d_al = per[1632]["alone_recentred"] - ALONE_BASE if 1632 in per else None
    s103 = None
    if d_rho is not None:
        s103 = {"d_rho_vs_base_1632": d_rho, "d_alone_vs_base_1632": d_al,
                "pred_dgain_defectfit_-308.92rho_+0.18533alone":
                    -308.92 * d_rho + 0.19533 * d_al,
                "pred_dgain_fixedfit_-289.78rho_+0.186alone":
                    -289.78 * d_rho + 0.186 * d_al,
                "obs_dgain_1632": per[1632]["argmax"]["gain"] - PREREG_BASELINE,
                "rho_unit_price_pts_per_0.01":
                    (d_al / (d_rho / 0.01)) if abs(d_rho) > 1e-9 else None,
                "cheaper_than_15.82": (
                    None if abs(d_rho) > 1e-9 and d_rho >= 0 else
                    (bool(-d_al / (-d_rho / 0.01) < RHO_UNIT_PRICE)
                     if d_rho < -1e-9 else None)),
                "🔴": "ρ 가 올라가면(d_rho>0) 단가 개념이 성립하지 않는다 — 그 경우 None"}
    TAB[nm] = {
        "axis": meta["axis"], "desc": meta["desc"], "target_file": meta["target_file"],
        "own_argmax_step": ok, "own_argmax_w": w, "gain": g,
        "delta_vs_prereg_baseline": g - PREREG_BASELINE,
        "delta_vs_redrawn_baseline_mean": g - BASE_MEAN,
        "t_combined_prereg_sigma": (g - PREREG_BASELINE) / PREREG_SIGMA,
        "gain_sd_matched_at_argmax": sc(sd_match(zb, z0), "abd") - BASE_ABD,
        "vs_base94_1632_paired": cmpx,
        "vs_base94_common_w0.43_paired": cmpc,
        "alone_recentred": per[ok]["alone_recentred"],
        "rho_vs_z_mate": per[ok]["rho_vs_z_mate"],
        "S103_교환율": s103,
        "per_step": {str(k): {kk: vv for kk, vv in v.items() if kk != "_z"}
                     for k, v in per.items()},
        "PASS_size": bool(g - PREREG_BASELINE >= PREREG_DELTA),
        "PASS_t": bool(cmpx["t"] is not None and cmpx["t"] >= PREREG_T)}
    TAB[nm]["PASS"] = bool(TAB[nm]["PASS_size"] and TAB[nm]["PASS_t"])
    print("%-14s step %5d w %.2f gain %+9.4f (d %+8.4f) t %+6.3f  alone %8.2f "
          "rho %.5f %s" % (nm, ok, w, g, g - PREREG_BASELINE,
                           cmpx["t"] or float("nan"), TAB[nm]["alone_recentred"],
                           TAB[nm]["rho_vs_z_mate"],
                           "PASS" if TAB[nm]["PASS"] else ""), flush=True)
R["missing_artifacts"] = MISSING
R["표"] = TAB

# ── 귀무 대조 쌍 (S104·S175·S179 — 판정이 귀무 선택에 따라 뒤집히면 판정이 아니다) ──
def pair(a_, b_, k=1632):
    if a_ not in TAB or b_ not in TAB:
        return None
    if str(k) not in TAB[a_]["per_step"] or str(k) not in TAB[b_]["per_step"]:
        return None
    az = np.mean([zload(f"{B}/out/preds/z_{a_}__k{k}_s{s}.npy") for s in S5], 0)
    bz = np.mean([zload(f"{B}/out/preds/z_{b_}__k{k}_s{s}.npy") for s in S5], 0)
    _, ab = replace_curve(az)
    _, bb = replace_curve(bz)
    pa = prob(root((1 - ab["w"]) * Z_MATE + ab["w"] * az), "abd")
    pbv = prob(root((1 - bb["w"]) * Z_MATE + bb["w"] * bz), "abd")
    return {"step": k, "a_gain": ab["gain"], "b_gain": bb["gain"],
            "a_minus_b": ab["gain"] - bb["gain"],
            "paired": boot_t((pa - y) ** 2 - (pbv - y) ** 2)}


NULLS = {}
for e in ("0p1", "0p2", "0p4"):
    NULLS[f"B{e}_minus_N2_B{e}"] = pair(f"B{e}", f"N2_B{e}")
for nm in list(TAB):
    if nm.startswith("A"):
        NULLS[f"{nm}_minus_N3"] = pair(nm, "N3_identity")
NULLS["N1b_minus_N1a"] = pair("N1b_perm_eA", "N1a_perm_e0")
NULLS["C_minus_N3"] = pair("C_mix", "N3_identity")
R["귀무_대조"] = {
    "왜": "N2 는 축 B 의 개수·크기 맞춤 귀무(타깃 잡음만 남고 정보는 0). "
          "N3 는 무변형 항등 복제본이라 축 A 의 귀무이자 사다리 맨 아래(S135). "
          "N1 쌍(b−a)은 신호 없는 팔 위에서 스무딩만의 몫이다 — 바닥 대조이지 등기댓값 귀무가 아니다.",
    "표": {k: v for k, v in NULLS.items() if v}}

qual = [k for k, v in TAB.items() if v["PASS"]]
near = sorted(TAB.items(), key=lambda kv: -kv[1]["delta_vs_prereg_baseline"])[:8]
_bse = float(np.median([v["vs_base94_1632_paired"]["boot_se_pts"] for v in TAB.values()]))
_fsd = R["🔴기준선_재추첨"]["sd_ddof1"]
R["검출한계"] = {
    "boot_se_pts_median": _bse, "적합간_sd_pts_한_점": _fsd,
    "결합_1sigma_pts_재측정": float(np.hypot(_bse, _fsd * np.sqrt(2.0))),
    "사전등록_1sigma": PREREG_SIGMA,
    "요구_효과크기_사전등록": PREREG_DELTA}
R["VERDICT"] = {
    "rule": R["prereg"]["rule"], "n_points_read": len(TAB), "qualifying": qual,
    "PASS": bool(qual) and not MISSING,
    "closest_points": [{"point": k, "delta": v["delta_vs_prereg_baseline"],
                        "t": v["vs_base94_1632_paired"]["t"],
                        "step": v["own_argmax_step"], "w": v["own_argmax_w"],
                        "rho": v["rho_vs_z_mate"], "alone": v["alone_recentred"]}
                       for k, v in near],
    "🔴": "산출물 누락이 있으면 PASS 를 주장하지 않는다" if MISSING else "산출물 전량 존재"}
R["z_sha16_count"] = len(SHA)
R["wall_sec"] = round(time.time() - t0, 1)
json.dump(R, open(f"{B}/out/s2_mix107.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({"VERDICT": R["VERDICT"], "검출한계": R["검출한계"],
                  "기준선_재추첨": R["🔴기준선_재추첨"]}, ensure_ascii=False, indent=1))
