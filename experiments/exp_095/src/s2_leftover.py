"""exp_095 stage 2 — 배포 경로에서 계산되고 최종 로짓에 안 실리는 per-row 벡터의 전수 조사.

면은 stage1 과 같다 (exp_090 s1_curve 승계, 모든 팔 out-of-time, 기준점 928.501461058684).
기준점은 **출하점** z_ship = root(w=0.20, v=0) 이고 abd 관례다.

축 만드는 법 (config 그대로):
    x_std = (x − mean(x)) / sd(x)
    z(s)  = z_ship + s · x_std,   d 는 매 s 에서 다시 못박는다
    gain(s) = K·(Brier(0) − Brier(s)) = 2K·L·s − K·G·s²
    → L = (gain(+h) − gain(−h)) / (4Kh),  G = −(gain(+h) + gain(−h)) / (2Kh²)
    → 꼭지점 이득 = K·L²/G,  s* = L/G
h = 0.01 (표준화 벡터이므로 로짓 이동 sd 가 0.01).  h/2 쌍으로 2차 근사의 잔차를 함께 잰다.

🔴 귀무 대조 (S104) — 사전 등록:
    귀무 A(주 판정) : 정직면 잔차와 무관하게 생성한 iid 정규 방향 200개
    귀무 B(보조)    : 후보 벡터의 행 순열 200개 (주변분포 보존, 행 정렬 파괴)
    후보는 귀무 A 의 상위 5% 지점을 넘어야 보고한다.  안 넘으면 '없음'.

🔴 정직면 크기를 LB 로 환산하지 않는다.  S54(실패유형 계열 19.4배)·S78(정직면 유도 축 54배).
   이 계열에서 유일하게 눈금이 잡힌 것은 S111 딥 학습량 축의 0.4779 뿐이고,
   그것도 같은 면·같은 w·같은 학습량 비교에만 쓴다.  여기서는 **순위와 부호만** 쓴다.
"""
import json, os, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "4"
import numpy as np
import pandas as pd
from scipy.special import expit, logit

H = "/home/<user>/SungjinPark"
B90, B95 = f"{H}/exp090", f"{H}/exp095"
OOT, OOT3, OOT4 = f"{H}/exp069/oot", f"{H}/exp070/oot3", f"{H}/exp071/out"
EPS, REF = 1e-6, 0.2498069
K = 100000.0 / REF
S5 = [42, 777, 2024, 101, 7]
AA = (42, 7, 2024)
W4 = {"LGB": 0.05, "CAT": 0.68, "NN": 0.15, "LEAF": 0.12}
BO = dict(alpha=1.07911, beta_skew=-0.059151, c_aa=0.25244, c_T70=0.04756,
          w_out=0.7, w_v6_inner=0.6668397489, w_ebm=-0.1, beta_ins=0.03)
BASE_EXPECT = 928.501461058684
SHIP_W, STEPS, HH = 0.20, 1632, 0.01
N_NULL = 200
lg = lambda p: logit(np.clip(np.asarray(p, np.float64), EPS, 1 - EPS))
pts = lambda b: 100000.0 * (1.0 - b / REF)
t0 = time.time()
os.makedirs(f"{B95}/out", exist_ok=True)

# ── 면 (stage1 과 글자 그대로 같다) ────────────────────────────────────
tro = pd.read_csv(f"{H}/data/train.csv",
                  usecols=["row_id", "season", "pitcher_id", "batter_hand",
                           "control_success", "asof_pitcher_n",
                           "asof_pitcher_success_rate"])
tr_all = tro
tro = tr_all[tr_all.season == 2024].sort_values("row_id", kind="stable")
y = tro.control_success.to_numpy(np.float64)
pit = tro.pitcher_id.to_numpy()
rid = np.asarray(np.load(f"{OOT}/row_id_oot2024.npy", allow_pickle=True)).astype(str)
assert np.array_equal(rid, tro.row_id.to_numpy().astype(str))
n = len(y)
assert n == 253507
YB = float(y.mean())
V = {k: np.load(p).astype(np.float64) for k, p in {
    "p_lgb": f"{OOT}/p_lgb_oot2024.npy", "p_cat": f"{OOT}/p_cat_oot2024.npy",
    "p_leaf": f"{OOT}/p_leaf_oot2024.npy", "p_nn": f"{OOT4}/p_nn_oot2024.npy",
    "p_q": f"{OOT}/p_q_oot2024.npy", "p_ebm": f"{OOT3}/p_ebm_oot2024.npy",
    "off": f"{OOT}/off_oot2024.npy"}.items()}
L_LGB, L_CAT, L_NN, L_LEAF = (lg(V["p_lgb"]), lg(V["p_cat"]),
                              lg(V["p_nn"]), lg(V["p_leaf"]))
LV = W4["LGB"] * L_LGB + W4["CAT"] * L_CAT + W4["NN"] * L_NN + W4["LEAF"] * L_LEAF
LQ, LE = lg(V["p_q"]), lg(V["p_ebm"])
AA_SEEDS = [lg(np.load(f"{H}/experiments/exp_049/pred/"
                       f"pred_A1_u2024_seed{s}_at1400.npy")) for s in AA]
T70_SEEDS = [lg(np.load(f"{H}/experiments/exp_055/pred/"
                        f"pred_T70_u2024_seed{s}_at1400.npy")) for s in AA]
L_aa, L_t70 = np.mean(AA_SEEDS, 0), np.mean(T70_SEEDS, 0)
DEEP_SEEDS = [np.load(f"{B90}/out/preds/z_K{STEPS}_oot23_s{s}.npy") for s in S5]
DZ = np.mean(DEEP_SEEDS, 0)
Z_MATE = ((1 - BO["w_ebm"]) * (BO["w_v6_inner"] * LV + (1 - BO["w_v6_inner"]) * LQ)
          + BO["w_ebm"] * LE + BO["beta_ins"] * V["off"])
Z_INNER = ((1 - SHIP_W) * Z_MATE + SHIP_W * DZ)
Z_SHIP = BO["w_out"] * Z_INNER + BO["c_aa"] * L_aa + BO["c_T70"] * L_t70


def pin(z, target=YB):
    c = 0.0
    for _ in range(200):
        p = expit(BO["alpha"] * z + BO["beta_skew"] * z * z + c)
        g = float(p.mean()) - target
        if abs(g) < 1e-15:
            break
        c -= g / float((p * (1 - p)).mean())
    return c


def prob(z):
    return expit(BO["alpha"] * z + BO["beta_skew"] * z * z + pin(z))


P0 = prob(Z_SHIP)
SQ0 = (P0 - y) ** 2
BASE = pts(float(SQ0.mean()))
# 기준점 게이트: w=0 · rc 관례가 928.501461058684 를 재현하는지 (stage1 과 같은 값)
z_rc0 = BO["w_out"] * Z_MATE + BO["c_aa"] * L_aa + BO["c_T70"] * L_t70
_c = 0.0
for _ in range(200):
    _p = expit(z_rc0 + _c)
    _g = float(_p.mean()) - YB
    if abs(_g) < 1e-15:
        break
    _c -= _g / float((_p * (1 - _p)).mean())
RC0 = pts(float(np.mean((expit(z_rc0 + _c) - y) ** 2)))
GATE = {"rc_baseline": RC0, "expected": BASE_EXPECT,
        "abs_diff": abs(RC0 - BASE_EXPECT), "PASS": bool(abs(RC0 - BASE_EXPECT) == 0.0)}
assert GATE["PASS"], GATE
print("BASELINE GATE", GATE, "| ship abd", BASE, flush=True)

# ── 블록 부트스트랩 ───────────────────────────────────────────────────
rng = np.random.default_rng(20260824)
uq, inv = np.unique(pit, return_inverse=True)
nb = len(uq)
order = np.argsort(inv, kind="stable")
st = np.searchsorted(inv[order], np.arange(nb))
PICKS = np.stack([rng.integers(0, nb, nb) for _ in range(1000)])
BSIZE = np.append(st[1:], len(order)) - st
BSIZE = BSIZE.astype(np.float64)


def boot_t(d):
    bs_ = np.add.reduceat(d[order], st)
    bs = bs_[PICKS].sum(1) / BSIZE[PICKS].sum(1)
    se = float(bs.std(ddof=1))
    obs = float(d.mean())
    return (float(-obs / se) if se > 0 else None), se


# ── 축 하나 재기 ──────────────────────────────────────────────────────
def measure(x, h=HH):
    x = np.asarray(x, np.float64)
    m, s = float(x.mean()), float(x.std())
    if not np.isfinite(s) or s <= 0:
        return None
    xs = (x - m) / s
    out = {}
    g = {}
    P = {}
    for k_ in (+1, -1, +0.5, -0.5):
        z = Z_SHIP + (k_ * h) * xs
        p_ = prob(z)
        P[k_] = p_
        g[k_] = BASE - pts(float(np.mean((p_ - y) ** 2)))     # = −gain
    gp, gm = -g[+1], -g[-1]                                   # gain(+h), gain(−h)
    gp2, gm2 = -g[+0.5], -g[-0.5]
    L = (gp - gm) / (4.0 * K * h)
    G = -(gp + gm) / (2.0 * K * h * h)
    L2 = (gp2 - gm2) / (4.0 * K * (h / 2))
    G2 = -(gp2 + gm2) / (2.0 * K * (h / 2) ** 2)
    vertex = (K * L * L / G) if G > 0 else float("nan")
    d_lin = (P[+1] - y) ** 2 - (P[-1] - y) ** 2               # mean = −4Lh
    t_, se_ = boot_t(d_lin)
    out.update(L=L, G=G, K_G=K * G, vertex_gain_pts=vertex,
               s_star=(L / G if G > 0 else float("nan")),
               L_halfstep=L2, G_halfstep=G2,
               L_rel_resid=(abs(L2 - L) / abs(L) if L != 0 else float("nan")),
               G_rel_resid=(abs(G2 - G) / abs(G) if G != 0 else float("nan")),
               gain_at_plus_h=gp, gain_at_minus_h=gm,
               boot_t_of_L=t_, boot_se=se_,
               delta_p=(P[+1] - P0))
    return out


# ── 후보: 코드에서 열거한 '계산되고 안 실리는' per-row 벡터 ──────────────
def sd_of(lst):
    return np.stack(lst).std(axis=0, ddof=1)


# ≤2023 개인 셀 표 (out-of-time): 2024 행에 붙인다.  exp_083 F 갈래가
# "재측정하려면 ≤2023 표가 필요하다"고 남긴 자리다 (S32/S47).
pre = tr_all[tr_all.season <= 2023]
key_pre = (pre.pitcher_id.astype(str) + "|" + pre.batter_hand.astype(str))
gcnt = pre.groupby(key_pre).control_success.agg(["count", "sum"])
key_24 = (tro.pitcher_id.astype(str) + "|" + tro.batter_hand.astype(str)).to_numpy()
idx = pd.Index(gcnt.index)
pos = idx.get_indexer(pd.Index(key_24))
hit = (pos >= 0)
ncell = np.where(hit, gcnt["count"].to_numpy()[np.maximum(pos, 0)], 0.0).astype(np.float64)
scell = np.where(hit, gcnt["sum"].to_numpy()[np.maximum(pos, 0)], 0.0).astype(np.float64)
overall = float(pre.control_success.mean())
d_s400 = (scell - ncell * overall) / (ncell + 400.0)

CAND = {
    # ── 딥 팔 (exp_083 조사 당시 존재하지 않던 갈래) ──────────────────
    "L1_deep_seed_sd5": (sd_of(DEEP_SEEDS),
                         "mate/deep_infer.py:64 zs=[...] → :65 np.mean(zs) 로 산포가 버려진다",
                         "NEW"),
    "L2_deep_seed_absdev_s42": (np.abs(DEEP_SEEDS[0] - DZ),
                                "mate/deep_infer.py:64-65 (같은 자리, 다른 축약)", "NEW"),
    "L2b_deep_seed_range": (np.max(DEEP_SEEDS, 0) - np.min(DEEP_SEEDS, 0),
                            "mate/deep_infer.py:64-65 (같은 자리, 다른 축약)", "NEW"),
    "L10_deep_vs_mate_absdisagree": (np.abs(DZ - Z_MATE),
                                     "mate/script.py:127 z=(1−w_deep)z+w_deep·lg(p_deep) "
                                     "— 불일치의 크기는 안 쓰인다", "NEW"),
    "L10s_deep_vs_mate_signed": (DZ - Z_MATE,
                                 "mate/script.py:127 (부호 있는 방향 = w_deep 축 그 자체)",
                                 "STAGE1_AXIS"),
    # ── 시드 산포 (exp_083 E 갈래의 다른 팔·다른 시드수) ───────────────
    "L4_aa_seed_sd3": (sd_of(AA_SEEDS),
                       "aa_infer.py:1039 seed_spread(per_seed) — CH 에서 u=0.0 으로 꺼져 있다",
                       "BOUGHT_THEN_ZEROED"),
    "L4b_t70_seed_sd3": (sd_of(T70_SEEDS),
                         "aa_infer.py:1041-1046 sm_per_seed → average_over_seeds", "NEW"),
    # ── v6 내부 4팔 (exp_080 이 '가중 축'으로 본 자리, 여기선 산포·불일치) ──
    "L6_v6_arm_sd4": (np.stack([L_LGB, L_CAT, L_NN, L_LEAF]).std(axis=0, ddof=1),
                      "mate/v6_infer.py:149,163,177,189 → :196 가중합만 남는다", "NEW"),
    "L7_v6_NN_minus_CAT": (L_NN - L_CAT, "mate/v6_infer.py:163,177", "EXP080_MEASURED"),
    "L7b_v6_LGB_minus_CAT": (L_LGB - L_CAT, "mate/v6_infer.py:149,163", "EXP080_MEASURED"),
    "L7c_v6_LEAF_minus_CAT": (L_LEAF - L_CAT, "mate/v6_infer.py:163,189", "EXP080_MEASURED"),
    "L7d_v6_arm_absdisagree_NN_CAT": (np.abs(L_NN - L_CAT),
                                      "mate/v6_infer.py:163,177 (크기만)", "NEW"),
    "L8_Q_minus_v6": (LQ - LV, "mate/script.py:95 base=w·lg(p_v)+(1−w)·lg(p_q)",
                      "EXP080_MEASURED"),
    "L8b_Q_v6_absdisagree": (np.abs(LQ - LV), "mate/script.py:95 (크기만)", "NEW"),
    "L9_EBM_minus_base": (LE - (BO["w_v6_inner"] * LV + (1 - BO["w_v6_inner"]) * LQ),
                          "mate/script.py:96 z=(1−w_ebm)·base+w_ebm·p_e", "EXP080_MEASURED"),
    # ── 백오프 표시자·표본수 (exp_083 F 갈래가 프레임 축퇴로 못 잰 자리) ──
    "L11_personal_miss_flag": ((~hit).astype(np.float64),
                               "mate/q_infer.py:621 / aa_infer.py:1217 `hit` — d_m 안에서만 쓰이고 "
                               "표시자 자체는 안 실린다", "NEW(≤2023표)"),
    "L12_personal_n_cell": (ncell,
                            "mate/q_infer.py:622 / aa_infer.py:1218 `n_cell` — assert 에서만 쓰인다",
                            "NEW(≤2023표)"),
    "L12b_personal_log1p_n": (np.log1p(ncell), "같은 자리, 로그 축약", "NEW(≤2023표)"),
    "L12c_shrunk_dev_s400": (d_s400,
                             "mate/q_infer.py:102 d_s=(s−n·o)/(n+k) — ≤2023 빈티지 재구성",
                             "BOUGHT(beta 축)"),
    # ── 양성 대조: 이미 산 방향 ────────────────────────────────────────
    "C1_ins_off_BOUGHT": (V["off"],
                          "script.py:449 beta_ins 축 + axis_offsets[0] beta_ins_off — 이미 샀다",
                          "POSITIVE_CONTROL"),
    "C2_asof_psr_BOUGHT": (tro.asof_pitcher_success_rate.to_numpy(np.float64),
                           "script.py:465 axis_offsets[1] ctrl_asof_psr — 이미 샀다",
                           "POSITIVE_CONTROL"),
    "C3_asof_pitcher_n": (np.log1p(tro.asof_pitcher_n.to_numpy(np.float64)),
                          "test.csv 원열 (모델 입력) — 대조용", "POSITIVE_CONTROL"),
}
CAND = {k: v for k, v in CAND.items()
        if np.isfinite(np.asarray(v[0], np.float64)).all()}

R = {"exp_id": "exp_095", "stage": "s2_leftover", "K_pts_per_brier": K,
     "baseline_gate": GATE, "ship_point_abd_score": BASE,
     "ship_point": {"w_deep_inner": SHIP_W, "v_outer": 0.0,
                    "deep_steps_oot": STEPS},
     "axis_recipe": "x_std=(x−mean)/sd ; z=z_ship+s·x_std ; d 재못박음 ; h=0.01",
     "prereg_null": f"iid 정규 방향 {N_NULL}개의 꼭지점 이득 상위 5% 지점",
     "🔴_no_LB_conversion": ("S54 19.4배 · S78 54배 · S111 딥 계열 0.4779. "
                             "순위와 부호만 쓴다.")}

# ── 후보 측정 ─────────────────────────────────────────────────────────
MEAS, DP = {}, {}
for k, (x, cite, tagv) in CAND.items():
    m = measure(x)
    if m is None:
        MEAS[k] = {"code_citation": cite, "status": tagv, "SKIP": "sd=0"}
        continue
    DP[k] = m.pop("delta_p")
    m["code_citation"] = cite
    m["status"] = tagv
    MEAS[k] = m
    print(f"  {k:34s} vtx {m['vertex_gain_pts']:+9.4f}  t {m['boot_t_of_L']:+7.3f}"
          f"  KG {m['K_G']:8.4f}  s* {m['s_star']:+.4f}"
          f"  quad_resid L {m['L_rel_resid']:.4f}", flush=True)

# ── 귀무 A (사전 등록, 주 판정): iid 정규 방향 ──────────────────────────
rngN = np.random.default_rng(950195)
nullA = []
for i in range(N_NULL):
    m = measure(rngN.standard_normal(n))
    m.pop("delta_p")
    nullA.append(m["vertex_gain_pts"])
    if (i + 1) % 50 == 0:
        print(f"  nullA {i+1}/{N_NULL}  {time.time()-t0:.0f}s", flush=True)
nullA = np.array(nullA)

# ── 귀무 B (보조): 후보 벡터의 행 순열 ─────────────────────────────────
rngP = np.random.default_rng(950295)
keys = [k for k in CAND if k in DP]
nullB = []
for i in range(N_NULL):
    k = keys[i % len(keys)]
    xv = np.asarray(CAND[k][0], np.float64)
    m = measure(xv[rngP.permutation(n)])
    m.pop("delta_p")
    nullB.append(m["vertex_gain_pts"])
nullB = np.array(nullB)

qA = float(np.percentile(nullA, 95))
qB = float(np.percentile(nullB, 95))
R["null_A_iid_normal"] = {
    "n": N_NULL, "mean": float(nullA.mean()), "sd": float(nullA.std(ddof=1)),
    "median": float(np.median(nullA)), "p95": qA, "max": float(nullA.max()),
    "theory_mean_K_brier_over_N": K * float(SQ0.mean()) / n}
R["null_B_row_permutation"] = {
    "n": N_NULL, "mean": float(nullB.mean()), "sd": float(nullB.std(ddof=1)),
    "median": float(np.median(nullB)), "p95": qB, "max": float(nullB.max())}

# ── 이미 산 축과의 상관 (δp 기준) ──────────────────────────────────────
BOUGHT = [k for k in DP if MEAS[k]["status"].startswith(("POSITIVE", "STAGE1", "BOUGHT"))]
for k in MEAS:
    if k not in DP:
        continue
    rr = {b: float(np.corrcoef(DP[k], DP[b])[0, 1]) for b in BOUGHT if b != k}
    MEAS[k]["rho_vs_bought_axes"] = rr
    MEAS[k]["max_abs_rho_vs_bought"] = (max(abs(v) for v in rr.values()) if rr else None)
    MEAS[k]["exceeds_null_A_p95"] = bool(MEAS[k]["vertex_gain_pts"] > qA)
    MEAS[k]["exceeds_null_B_p95"] = bool(MEAS[k]["vertex_gain_pts"] > qB)
R["candidates"] = MEAS

passers = [{"name": k, "vertex_gain_pts": v["vertex_gain_pts"],
            "boot_t_of_L": v["boot_t_of_L"], "s_star": v["s_star"],
            "K_G": v["K_G"], "status": v["status"],
            "code_citation": v["code_citation"],
            "max_abs_rho_vs_bought": v.get("max_abs_rho_vs_bought"),
            "exceeds_null_B_p95": v["exceeds_null_B_p95"]}
           for k, v in MEAS.items()
           if v.get("exceeds_null_A_p95") and not v["status"].startswith(
               ("POSITIVE", "STAGE1"))]
passers.sort(key=lambda d: -d["vertex_gain_pts"])
R["PREREGISTERED_VERDICT"] = {
    "rule": "무작위 귀무 200개(A)의 상위 5% 지점을 넘는 후보만 보고. 안 넘으면 '없음'.",
    "null_A_p95_pts": qA, "null_B_p95_pts": qB,
    "n_passing": len(passers), "passing": passers,
    "VERDICT": ("없음" if not passers else f"{len(passers)}건")}
R["wall_sec"] = round(time.time() - t0, 1)
json.dump(R, open(f"{B95}/out/s2_leftover.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in R.items() if k != "candidates"},
                 ensure_ascii=False, indent=1))
