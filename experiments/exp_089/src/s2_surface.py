"""exp_089 stage1 게이트 + stage2 정직면 + stage3 눈금.

정직면 = **모든 팔이 out-of-time**. 지금까지 이 면이 없었다 — 팀원 딥 팔이 ≤2024 학습이라
2024 가 그 팔에게 표본 안이었기 때문이다. 딥 팔의 ≤2023 판본을 우리가 만들어 그 구멍을 메운다.

면의 산술 (exp_078/080 이 쓴 BO 관례 그대로, 그래서 기준점 928.501461058684 를 assert 한다):
    LV      = .05·lg(p_lgb) + .68·lg(p_cat) + .15·lg(p_nn) + .12·lg(p_leaf)     (OOT 4팔)
    z_mate  = (1−w_ebm)·( w_v6i·LV + (1−w_v6i)·LQ ) + w_ebm·LE + beta_ins·off   (OOT)
    z_mate' = (1−w)·z_mate + w·lg(p_deep)            ← 배포가 딥을 꽂는 바로 그 자리
    z       = w_out·z_mate' + c_aa·L_aa + c_T70·L_T70                            (OOT)
    p       = expit(z + c),  c 는 평균을 못박는 상수 (배포 d_adj 재유도의 정직면 대응)

두 관례를 다 낸다:
  ① `rc`   — exp_080 관례. alpha·beta_skew 없음. 기존 정직면 수치와 직접 비교 가능.
  ② `abd`  — 배포 관례. p = expit(alpha·z + beta_skew·z² + d), d 로 평균 못박음.
     LB 이득이 이 산술을 통과해 측정된 값이므로 전이비의 분모와 형태가 맞는 쪽이다.

딥 팔 세 판본을 같은 면에 꽂아 비교한다:
  · `oot23_ours`  우리 ≤2023 복제본 5시드      ← ★정직 (이 실험의 산출물)
  · `le2024_ours` 우리 ≤2024 복제본 5시드      ← 표본 안 (부풀림 크기 대조)
  · `le2024_mate` 팀원 배포본 5시드            ← 표본 안 (팀원 로컬 +10.20 과 대응)
"""
import itertools, json, os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "6"
import numpy as np
import pandas as pd
from scipy.special import expit, logit

H = "/home/<user>/SungjinPark"
B = f"{H}/exp089"
E88 = f"{H}/exp088/out/preds"
OOT, OOT3, OOT4 = f"{H}/exp069/oot", f"{H}/exp070/oot3", f"{H}/exp071/out"
EPS, REF = 1e-6, 0.2498069
K = 100000.0 / REF
SEEDS = [42, 777, 2024, 101, 7]
AA = (42, 7, 2024)
W4 = {"LGB": 0.05, "CAT": 0.68, "NN": 0.15, "LEAF": 0.12}
BO = dict(alpha=1.07911, beta_skew=-0.059151, c_aa=0.25244, c_T70=0.04756,
          w_out=0.7, w_v6_inner=0.6668397489, w_ebm=-0.1, beta_ins=0.03)
BASE_EXPECT = 928.501461058684
LB_GAIN = 5.0813180433          # 부모 BX 1129.42909358 → BX+deep 1134.5104116233
LB_W = 0.20
t0 = time.time()
R = {"exp_id": "exp_089", "K_pts_per_brier": K, "ref_brier": REF,
     "BO_constants": BO, "LB_gain_measured": LB_GAIN, "LB_w_deep": LB_W}


def lg(p):
    return logit(np.clip(np.asarray(p, np.float64), EPS, 1 - EPS))


def pts(b):
    return 100000.0 * (1.0 - b / REF)


# ── 정렬 ───────────────────────────────────────────────────────────────
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
for n_, v in V.items():                      # S14 — 이름이 아니라 값의 범위로 확인
    assert len(v) == 253507
assert all(V[k].min() >= 0 and V[k].max() <= 1
           for k in ("p_lgb", "p_cat", "p_leaf", "p_nn", "p_q", "p_ebm"))
assert V["off"].min() < 0.0

LV = sum(W4[t] * lg(V[k]) for t, k in
         (("LGB", "p_lgb"), ("CAT", "p_cat"), ("NN", "p_nn"), ("LEAF", "p_leaf")))
LQ, LE = lg(V["p_q"]), lg(V["p_ebm"])
L_aa = np.mean([lg(np.load(f"{H}/experiments/exp_049/pred/"
                           f"pred_A1_u2024_seed{s}_at1400.npy")) for s in AA], axis=0)
L_t70 = np.mean([lg(np.load(f"{H}/experiments/exp_055/pred/"
                            f"pred_T70_u2024_seed{s}_at1400.npy")) for s in AA], axis=0)
Z_MATE = ((1 - BO["w_ebm"]) * (BO["w_v6_inner"] * LV + (1 - BO["w_v6_inner"]) * LQ)
          + BO["w_ebm"] * LE + BO["beta_ins"] * V["off"])

# ── 딥 팔 세 판본 ──────────────────────────────────────────────────────
Zo = {s: np.load(f"{B}/out/preds/z_M_oot23_s{s}.npy") for s in SEEDS}
Zi = {s: np.load(f"{B}/out/preds/z_M_le2024_s{s}.npy") for s in SEEDS}
Zt = {s: np.load(f"{E88}/deep_route1_s{s}_logit.npy") for s in SEEDS}
DEEP = {"oot23_ours": np.mean([Zo[s] for s in SEEDS], 0),
        "le2024_ours": np.mean([Zi[s] for s in SEEDS], 0),
        "le2024_mate": np.mean([Zt[s] for s in SEEDS], 0)}


def wrho(Z):
    return float(np.mean([np.corrcoef(Z[a], Z[b])[0, 1]
                          for a, b in itertools.combinations(SEEDS, 2)]))


rt, ri, ro = wrho(Zt), wrho(Zi), wrho(Zo)
rho55 = float(np.corrcoef(DEEP["le2024_ours"], DEEP["le2024_mate"])[0, 1])
R["stage1_replication_gate"] = {
    "rho_ours_le2024_5seed_vs_mate_5seed": rho55,
    "gate_threshold": 0.98,
    "PASS": bool(rho55 >= 0.98),
    "within_seed_rho_mate": rt, "within_seed_rho_ours_le2024": ri,
    "within_seed_rho_ours_oot23": ro,
    "ceiling_5v5_perfect_replica": 5 * rt / np.sqrt((1 + 4 * rt) * (1 + 4 * ri)),
    "attainment_vs_ceiling": rho55 / (5 * rt / np.sqrt((1 + 4 * rt) * (1 + 4 * ri))),
    "per_seed_rho_ours_vs_mate_matched_label": {
        str(s): float(np.corrcoef(Zi[s], Zt[s])[0, 1]) for s in SEEDS},
    "rho_ours_oot23_vs_mate_5seed": float(
        np.corrcoef(DEEP["oot23_ours"], DEEP["le2024_mate"])[0, 1]),
    "note": ("천장은 시드 잡음이 정한다 — 구조가 완벽해도 5시드평균끼리는 "
             "5r/√((1+4r_a)(1+4r_b)) 를 못 넘는다"),
}
R["deep_arm_alone"] = {
    k: {"score": pts(float(np.mean((expit(z) - y) ** 2))), "z_sd": float(z.std()),
        "z_mean": float(z.mean())} for k, z in DEEP.items()}


# ── 면 산술 ────────────────────────────────────────────────────────────
def pin(z, f, target):
    """f(z, c) 의 평균이 target 이 되게 하는 c 를 뉴턴법으로 푼다."""
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


def surface(w, deep, conv):
    zm = (1.0 - w) * Z_MATE + w * deep
    z = BO["w_out"] * zm + BO["c_aa"] * L_aa + BO["c_T70"] * L_t70
    f = F_RC if conv == "rc" else F_ABD
    return f(z, pin(z, f, YB))


base_p = {c: surface(0.0, DEEP["oot23_ours"], c) for c in ("rc", "abd")}
base_pts = {c: pts(float(np.mean((base_p[c] - y) ** 2))) for c in base_p}
R["baseline_w0"] = {"rc_pts": base_pts["rc"], "abd_pts": base_pts["abd"],
                    "rc_expected_exp078_080": BASE_EXPECT,
                    "rc_abs_diff": abs(base_pts["rc"] - BASE_EXPECT)}
assert abs(base_pts["rc"] - BASE_EXPECT) < 1e-9, f"정직면 기준 불일치 {base_pts['rc']}"

GRID = [round(x, 4) for x in np.arange(0.0, 0.6001, 0.01)]
R["sweep"] = {}
for name, dz in DEEP.items():
    for conv in ("rc", "abd"):
        cur = []
        for w in GRID:
            p = surface(w, dz, conv)
            cur.append({"w": w, "pts": pts(float(np.mean((p - y) ** 2)))})
        b0 = base_pts[conv]
        for c_ in cur:
            c_["gain"] = c_["pts"] - b0
        best = max(cur, key=lambda c_: c_["gain"])
        g20 = [c_ for c_ in cur if abs(c_["w"] - LB_W) < 1e-9][0]["gain"]
        # 2차 모형: gain(w) = 2K L w − K G w²  →  꼭지점 w* = L/G
        ws = np.array([c_["w"] for c_ in cur])
        gs = np.array([c_["gain"] for c_ in cur])
        m = ws <= 0.30
        q = np.polyfit(ws[m], gs[m], 2)
        R["sweep"][f"{name}__{conv}"] = {
            "gain_at_w0.20": g20,
            "argmax_w": best["w"], "gain_at_argmax": best["gain"],
            "quad_fit_on_w<=0.30": {"a2": float(q[0]), "a1": float(q[1]),
                                    "a0": float(q[2]),
                                    "vertex_w": float(-q[1] / (2 * q[0]))},
            "curve": cur,
        }

# ── 투수 블록 부트스트랩 (w=0.20 vs w=0, 정직 딥) ───────────────────────
rng = np.random.default_rng(20260824)
uq, inv = np.unique(pit, return_inverse=True)
order = np.argsort(inv, kind="stable")
starts = np.searchsorted(inv[order], np.arange(len(uq)))
ends = np.append(starts[1:], len(order))
R["bootstrap_pitcher_block"] = {}
for conv in ("rc", "abd"):
    p20 = surface(LB_W, DEEP["oot23_ours"], conv)
    p0 = base_p[conv]
    d = (p20 - y) ** 2 - (p0 - y) ** 2          # 음수면 이득
    obs = float(d.mean())
    bs = np.empty(1000)
    for r_ in range(1000):
        pick = rng.integers(0, len(uq), len(uq))
        idx = np.concatenate([order[starts[j]:ends[j]] for j in pick])
        bs[r_] = d[idx].mean()
    se = float(bs.std(ddof=1))
    R["bootstrap_pitcher_block"][conv] = {
        "n_blocks": int(len(uq)), "n_reps": 1000,
        "mean_delta_brier": obs, "gain_pts": -obs * K,
        "boot_se_brier": se, "t": float(-obs / se) if se > 0 else None,
        "share_negative_delta": float((bs < 0).mean()),
        "boot_gain_pts_p2.5": float(-np.percentile(bs, 97.5) * K),
        "boot_gain_pts_p97.5": float(-np.percentile(bs, 2.5) * K),
    }

# ── stage 3 눈금 ───────────────────────────────────────────────────────
tr = {}
for conv in ("rc", "abd"):
    for name in DEEP:
        g = R["sweep"][f"{name}__{conv}"]["gain_at_w0.20"]
        tr[f"{name}__{conv}"] = {
            "honest_surface_gain_at_w0.20": g,
            "LB_gain": LB_GAIN,
            "transfer_ratio_LB_over_surface": LB_GAIN / g if g != 0 else None,
            "surface_over_LB": g / LB_GAIN,
        }
d20 = surface(LB_W, DEEP["oot23_ours"], "abd") - base_p["abd"]
Bq = K * float(np.mean(d20 ** 2))
R["stage3_calibration"] = {
    "ratios": tr,
    "PRIMARY_axis_transfer_ratio_surface_over_LB": tr["oot23_ours__abd"]["surface_over_LB"],
    "B_KmeanDelta2_honest_abd_at_w0.20": Bq,
    "B_cov_frame_reported": 6.717,
    "w_star_from_LB_reported": 0.1757,
    "w_star_honest_rc": R["sweep"]["oot23_ours__rc"]["argmax_w"],
    "w_star_honest_abd": R["sweep"]["oot23_ours__abd"]["argmax_w"],
    "w_star_honest_abd_quadratic_vertex":
        R["sweep"]["oot23_ours__abd"]["quad_fit_on_w<=0.30"]["vertex_w"],
    "in_sample_inflation_ours": (
        R["sweep"]["le2024_ours__abd"]["gain_at_w0.20"]
        / R["sweep"]["oot23_ours__abd"]["gain_at_w0.20"]),
    "in_sample_inflation_mate": (
        R["sweep"]["le2024_mate__abd"]["gain_at_w0.20"]
        / R["sweep"]["oot23_ours__abd"]["gain_at_w0.20"]),
}
R["wall_sec"] = round(time.time() - t0, 1)
json.dump(R, open(f"{B}/out/s2_surface.json", "w"), ensure_ascii=False, indent=1)
pr = {k: v for k, v in R.items() if k != "sweep"}
pr["sweep_summary"] = {k: {kk: vv for kk, vv in v.items() if kk != "curve"}
                       for k, v in R["sweep"].items()}
print(json.dumps(pr, ensure_ascii=False, indent=1))
