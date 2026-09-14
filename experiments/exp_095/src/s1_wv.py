"""exp_095 stage 1 — 딥 팔 가중의 2자유도 기하 (w_inner, v_outer).

면은 exp_090 s1_curve.py 를 **글자 그대로** 승계한다 (모든 팔 out-of-time, season<=2023 학습,
season 2024 253,507행 채점).  달라지는 것은 딥 팔이 들어가는 **자리**뿐이다:

    exp_090 :  z(w)   = w_out*[(1-w)*z_mate + w*z_deep] + c_aa*z_aa + c_T70*z_T70
    exp_095 :  z(w,v) = w_out*[(1-w)*z_mate + w*z_deep] + c_aa*z_aa + c_T70*z_T70 + v*z_deep

    즉 딥 계수 = w_out*w + v,  mate 계수 = w_out*(1-w).
    w 를 올리면 딥이 오르는 만큼 mate 가 깎이고, v 는 mate 를 안 깎고 딥만 올린다.

관례 둘:
    rc  : p = expit(z + c)                        (기준점 대조용)
    abd : p = expit(alpha*z + beta_skew*z^2 + d)  (배포 관례 — ★주 판정면★)
`c`/`d` 는 평균을 2024 실제 성공률에 못박는 상수(측정면 전용, 제출물 아님).

🔴 재현 게이트 셋 (하나라도 실패하면 stage1 무효):
   G1 기준점  : w=0,v=0 · rc == 928.501461058684
   G2 예측파일: K1632 5시드 z 의 sha256 == exp_090 fit_*.json 기록값
   G3 곡선    : v=0 단면에서 abd gain(w=0.20) == 11.05028248077349 이고
                0.01 격자 argmax == (w 0.36, gain 13.714309769419287)

⚠️ v>0 은 z 의 전체 척도도 함께 올린다.  그래서 고정-alpha 이득에는 척도 몫이 섞인다(S25·S33).
   alpha 자유 이득과 sd 맞춤 이득을 **같은 격자 전체에서** 병기한다.
"""
import hashlib, itertools, json, os, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "4"
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

H = "/home/<user>/SungjinPark"
B89, B90, B95 = f"{H}/exp089", f"{H}/exp090", f"{H}/exp095"
OOT, OOT3, OOT4 = f"{H}/exp069/oot", f"{H}/exp070/oot3", f"{H}/exp071/out"
EPS, REF = 1e-6, 0.2498069
K = 100000.0 / REF
S5 = [42, 777, 2024, 101, 7]
AA = (42, 7, 2024)
W4 = {"LGB": 0.05, "CAT": 0.68, "NN": 0.15, "LEAF": 0.12}
BO = dict(alpha=1.07911, beta_skew=-0.059151, c_aa=0.25244, c_T70=0.04756,
          w_out=0.7, w_v6_inner=0.6668397489, w_ebm=-0.1, beta_ins=0.03)
BASE_EXPECT = 928.501461058684
E090_GAIN_W020 = 11.05028248077349          # exp_090 abd, K1632, w=0.20, v=0
E090_ARGMAX_W = 0.36
E090_GAIN_ARGMAX = 13.714309769419287
SHIP_W, SHIP_V = 0.20, 0.0
STEPS = 1632                                  # OOT argmax(=배포 2160스텝)
lg = lambda p: logit(np.clip(np.asarray(p, np.float64), EPS, 1 - EPS))
pts = lambda b: 100000.0 * (1.0 - b / REF)
t0 = time.time()
os.makedirs(f"{B95}/out", exist_ok=True)

# ── 팔 벡터 ───────────────────────────────────────────────────────────
SHA = {}
ZS = []
for s in S5:
    tag = f"K{STEPS}_oot23_s{s}"
    p_ = f"{B90}/out/preds/z_{tag}.npy"
    z_ = np.load(p_)
    h_ = hashlib.sha256(np.ascontiguousarray(z_).tobytes()).hexdigest()
    rec = json.load(open(f"{B90}/out/fit_{tag}.json"))["z_sha256"]
    SHA[str(s)] = {"file_sha256": h_, "recorded_in_exp090": rec,
                   "MATCH": bool(h_ == rec)}
    ZS.append(z_)
assert all(v["MATCH"] for v in SHA.values()), "G2 실패: 딥 예측 sha 불일치"
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
           ("p_lgb", "p_cat", "p_leaf", "p_nn", "p_q", "p_ebm"))        # S14
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
FIX = BO["c_aa"] * L_aa + BO["c_T70"] * L_t70


def root(w, v, dz=DZ, zm=Z_MATE):
    """★이 실험의 유일한 변경점★ — v*z_deep 이 바깥에 더해진다."""
    return BO["w_out"] * ((1.0 - w) * zm + w * dz) + FIX + v * dz


def alpha_free(z, conv):
    r = minimize_scalar(lambda a: -sc(a * z, conv), bounds=(0.3, 3.0),
                        method="bounded", options={"xatol": 1e-8})
    return float(-r.fun), float(r.x)


def sd_match(znew, zref):                       # S25/S33 — 척도 몫 제거
    return ((znew - znew.mean()) / znew.std()) * zref.std() + zref.mean()


# ── 투수 블록 부트스트랩 (exp_089·090 과 같은 시드·같은 추출 순서) ─────────
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
    """d = 행별 (Brier_new − Brier_ref).  음수면 new 가 낫다."""
    bs_ = np.array([d[IDX[j]].sum() for j in range(nb)])
    bs = bs_[PICKS].sum(1) / BSIZE[PICKS].sum(1)
    se = float(bs.std(ddof=1))
    obs = float(d.mean())
    return {"gain_pts": -obs * K, "boot_se_brier": se,
            "t": float(-obs / se) if se > 0 else None,
            "p_share_new_worse": float((bs > 0).mean()),
            "ci95_pts": [float(-np.percentile(bs, 97.5) * K),
                         float(-np.percentile(bs, 2.5) * K)]}


R = {"exp_id": "exp_095", "stage": "s1_wv", "K_pts_per_brier": K, "ref_brier": REF,
     "BO_constants": BO, "deep_steps_oot": STEPS, "n_rows": int(len(y)),
     "y_mean": YB, "pred_sha_check": SHA,
     "surface": "exp_090 s1_curve 면 그대로 + 바깥 슬롯 v*z_deep",
     "deep_coef_formula": "deep = w_out*w + v ;  mate = w_out*(1-w)",
     "🔴_no_LB_conversion": ("정직면 크기를 LB 로 환산하지 않는다. S54(19.4배)·S78(54배). "
                             "이 계열의 유일한 눈금은 S111 딥 학습량 축 전이비 0.4779 이고, "
                             "그것도 같은 면·같은 w·같은 학습량 비교에만 쓴다.")}

# ── G1/G3 재현 게이트 ──────────────────────────────────────────────────
z0 = root(0.0, 0.0)
base = {c: sc(z0, c) for c in ("rc", "abd")}
base_af = {c: alpha_free(z0, c) for c in ("rc", "abd")}
base_p = {c: prob(z0, c) for c in ("rc", "abd")}
R["G1_baseline"] = {"rc": base["rc"], "rc_expected": BASE_EXPECT,
                    "abs_diff": abs(base["rc"] - BASE_EXPECT),
                    "abd": base["abd"],
                    "alpha_free": {c: base_af[c][0] for c in base_af},
                    "PASS": bool(abs(base["rc"] - BASE_EXPECT) == 0.0)}
assert abs(base["rc"] - BASE_EXPECT) < 1e-9, base["rc"]

fine = [round(x, 4) for x in np.arange(0.0, 0.6001, 0.01)]
curve_v0 = [{"w": w, "gain": sc(root(w, 0.0), "abd") - base["abd"]} for w in fine]
bestf = max(curve_v0, key=lambda c: c["gain"])
g020 = [c for c in curve_v0 if abs(c["w"] - 0.20) < 1e-9][0]["gain"]
R["G3_curve_repro"] = {
    "gain_at_w0.20_here": g020, "gain_at_w0.20_exp090": E090_GAIN_W020,
    "abs_diff_w0.20": abs(g020 - E090_GAIN_W020),
    "argmax_w_here": bestf["w"], "argmax_w_exp090": E090_ARGMAX_W,
    "gain_at_argmax_here": bestf["gain"], "gain_at_argmax_exp090": E090_GAIN_ARGMAX,
    "abs_diff_argmax": abs(bestf["gain"] - E090_GAIN_ARGMAX),
    "PASS": bool(abs(g020 - E090_GAIN_W020) < 1e-9
                 and abs(bestf["w"] - E090_ARGMAX_W) < 1e-9
                 and abs(bestf["gain"] - E090_GAIN_ARGMAX) < 1e-9)}
assert R["G3_curve_repro"]["PASS"], R["G3_curve_repro"]
R["G2_pred_sha_PASS"] = True
R["GATES_ALL_PASS"] = bool(R["G1_baseline"]["PASS"] and R["G3_curve_repro"]["PASS"])
print("GATES", json.dumps({k: R[k] for k in
                           ("G1_baseline", "G3_curve_repro", "GATES_ALL_PASS")},
                          ensure_ascii=False), flush=True)

# ── 2자유도 격자 ──────────────────────────────────────────────────────
WG = [round(x, 4) for x in np.arange(0.0, 0.5001, 0.02)]
VG = [round(x, 4) for x in np.arange(0.0, 0.3001, 0.02)]
G = {}
for w in WG:
    for v in VG:
        z = root(w, v)
        gfix = sc(z, "abd") - base["abd"]
        gaf, a_ = alpha_free(z, "abd")
        gsd = sc(sd_match(z, z0), "abd") - base["abd"]
        G[(w, v)] = {"w": w, "v": v,
                     "deep_coef": BO["w_out"] * w + v,
                     "mate_coef": BO["w_out"] * (1.0 - w),
                     "gain_fixed_alpha": gfix,
                     "gain_alpha_free": gaf - base_af["abd"][0],
                     "alpha_star": a_,
                     "gain_sd_matched": gsd,
                     "z_sd": float(z.std())}
    print(f"  w={w:.2f} done  {time.time()-t0:.0f}s", flush=True)
R["grid"] = [G[k] for k in sorted(G)]
R["grid_spec"] = {"w": WG, "v": VG, "n_points": len(G)}

ship = G[(SHIP_W, SHIP_V)]
best_fix = max(G.values(), key=lambda c: c["gain_fixed_alpha"])
best_af = max(G.values(), key=lambda c: c["gain_alpha_free"])
best_sd = max(G.values(), key=lambda c: c["gain_sd_matched"])
# 1자유도 단면
cut_v0 = [G[(w, 0.0)] for w in WG]
cut_w20 = [G[(SHIP_W, v)] for v in VG]
b_cut_v0 = max(cut_v0, key=lambda c: c["gain_fixed_alpha"])
b_cut_w20 = max(cut_w20, key=lambda c: c["gain_fixed_alpha"])
b_cut_v0_af = max(cut_v0, key=lambda c: c["gain_alpha_free"])
b_cut_w20_af = max(cut_w20, key=lambda c: c["gain_alpha_free"])
R["ship_point"] = ship
R["argmax_2dof"] = {"fixed_alpha": best_fix, "alpha_free": best_af,
                    "sd_matched": best_sd}
R["cut_1dof"] = {
    "v0_w_curve_argmax": b_cut_v0, "v0_w_curve_argmax_alpha_free": b_cut_v0_af,
    "w020_v_curve_argmax": b_cut_w20,
    "w020_v_curve_argmax_alpha_free": b_cut_w20_af,
    "v0_curve": cut_v0, "w020_curve": cut_w20}
R["residual_vs_ship"] = {
    "fixed_alpha": best_fix["gain_fixed_alpha"] - ship["gain_fixed_alpha"],
    "alpha_free": best_af["gain_alpha_free"] - ship["gain_alpha_free"],
    "sd_matched": best_sd["gain_sd_matched"] - ship["gain_sd_matched"],
    "2dof_minus_best_1dof_v0_fixed_alpha":
        best_fix["gain_fixed_alpha"] - b_cut_v0["gain_fixed_alpha"],
    "2dof_minus_best_1dof_v0_alpha_free":
        best_af["gain_alpha_free"] - b_cut_v0_af["gain_alpha_free"]}

# ── 짝지은 투수 블록 부트스트랩 ────────────────────────────────────────
P_ship = prob(root(SHIP_W, SHIP_V), "abd")
SQ_ship = (P_ship - y) ** 2
SQ_base = (base_p["abd"] - y) ** 2


def cmp_pt(w, v, tag):
    P_ = prob(root(w, v), "abd")
    SQ = (P_ - y) ** 2
    return {"w": w, "v": v, "tag": tag,
            "vs_ship": boot(SQ - SQ_ship), "vs_baseline": boot(SQ - SQ_base)}


R["paired_boot"] = {
    "argmax_fixed_alpha": cmp_pt(best_fix["w"], best_fix["v"], "2dof_argmax_fixed"),
    "argmax_alpha_free": cmp_pt(best_af["w"], best_af["v"], "2dof_argmax_af"),
    "best_1dof_v0": cmp_pt(b_cut_v0["w"], 0.0, "1dof_w_argmax"),
    "best_1dof_w020": cmp_pt(SHIP_W, b_cut_w20["v"], "1dof_v_argmax"),
    "exp090_prereg_point_w0.36_v0": cmp_pt(0.36, 0.0, "exp090_argmax(선택편의 없음)"),
    "ship_vs_baseline": boot(SQ_ship - SQ_base),
    "🔴_selection_note": ("argmax 는 이 표본에서 고른 점이라 그 t 에는 선택 편의가 있다. "
                          "exp090_prereg_point 는 다른 실험이 이미 고른 점이라 그 편의가 없다.")}

# ── 시드 잭나이프 (딥 5시드 leave-one-out) ────────────────────────────
JK = []
for k in range(5):
    dzk = np.mean([ZS[i] for i in range(5) if i != k], 0)
    gk = {}
    for w in WG:
        for v in VG:
            gk[(w, v)] = sc(root(w, v, dz=dzk), "abd") - base["abd"]
    bk = max(gk, key=lambda t: gk[t])
    JK.append({"left_out_seed": S5[k], "argmax_w": bk[0], "argmax_v": bk[1],
               "gain_at_argmax": gk[bk],
               "gain_at_ship": gk[(SHIP_W, SHIP_V)],
               "gain_at_2dof_argmax_point": gk[(best_fix["w"], best_fix["v"])],
               "residual": gk[bk] - gk[(SHIP_W, SHIP_V)]})
    print(f"  jk drop s{S5[k]} argmax {bk} resid {JK[-1]['residual']:+.3f}", flush=True)
jk_res = np.array([j["residual"] for j in JK])
jk_pt = np.array([j["gain_at_2dof_argmax_point"] - j["gain_at_ship"] for j in JK])
R["jackknife_over_deep_seeds"] = {
    "per_loo": JK,
    "se_of_residual": float(np.sqrt(4 / 5 * np.sum((jk_res - jk_res.mean()) ** 2))),
    "se_of_fixed_point_residual":
        float(np.sqrt(4 / 5 * np.sum((jk_pt - jk_pt.mean()) ** 2))),
    "argmax_w_set": sorted({j["argmax_w"] for j in JK}),
    "argmax_v_set": sorted({j["argmax_v"] for j in JK})}

# ── 사전 등록 판정 (config 그대로, 결과 보고 바꾸지 않았다 — S8) ─────────
res = R["residual_vs_ship"]["fixed_alpha"]
tt = R["paired_boot"]["argmax_fixed_alpha"]["vs_ship"]["t"]
R["PREREGISTERED_VERDICT"] = {
    "rule": "2자유도 argmax 이득 − 출하점(0.20,0) 이득 >= 2.0 이고 짝지은 부트스트랩 t >= 2.0",
    "residual_pts": res, "threshold_residual": 2.0,
    "paired_t_vs_ship": tt, "threshold_t": 2.0,
    "residual_PASS": bool(res >= 2.0), "t_PASS": bool(tt is not None and tt >= 2.0),
    "PASS": bool(res >= 2.0 and tt is not None and tt >= 2.0)}
R["wall_sec"] = round(time.time() - t0, 1)
json.dump(R, open(f"{B95}/out/s1_wv.json", "w"), ensure_ascii=False, indent=1)
slim = {k: v for k, v in R.items() if k not in ("grid", "cut_1dof", "pred_sha_check")}
slim["cut_1dof_argmax"] = {k: v for k, v in R["cut_1dof"].items()
                           if not k.endswith("curve")}
print(json.dumps(slim, ensure_ascii=False, indent=1))
