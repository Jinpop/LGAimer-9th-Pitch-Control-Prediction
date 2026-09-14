"""exp_090 stage 1 — OOT 학습량(총 옵티마이저 스텝) 축의 정직면 곡선.

면은 exp_089 s4_final.py 의 것을 **글자 그대로 승계**한다 (모든 팔 out-of-time):
    LV      = .05·lg(p_lgb) + .68·lg(p_cat) + .15·lg(p_nn) + .12·lg(p_leaf)   (OOT 4팔)
    z_mate  = (1−w_ebm)·( w_v6i·LV + (1−w_v6i)·LQ ) + w_ebm·LE + beta_ins·off (OOT)
    z       = w_out·[(1−w)·z_mate + w·z_deep] + c_aa·L_AA + c_T70·L_T70       (OOT)
    rc  : p = expit(z + c)                       (exp_078/080 관례 — 기준점 대조용)
    abd : p = expit(alpha·z + beta_skew·z² + d)  (배포 관례 — ★주 판정면★)
`c`/`d` 는 평균을 2024 실제 성공률에 못박는 상수(측정면 전용, 제출물 아님).

🔴 기준점 게이트: w=0 · rc 가 928.501461058684 를 abs_diff < 1e-9 로 재현해야 한다.
🔴 재사용 게이트: 544/720/816/1088 은 exp_089 산출물을 그대로 쓰고 `.npy` 의 sha256 을
   exp_089 가 적어 둔 `fit_*.json:z_sha256` 과 대조한다. 다시 학습하지 않는다.
부트스트랩은 exp_089 와 **같은 rng 시드·같은 호출 순서**(투수 블록 1000회)라 공유 지점의 t 가
그대로 재현돼야 한다 — 그 자체가 측정 코드 승계의 검사다.
"""
import itertools, json, os, sys, time, hashlib
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "6"
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

H = "/home/<user>/SungjinPark"
B89, B90 = f"{H}/exp089", f"{H}/exp090"
E88 = f"{H}/exp088/out/preds"
OOT, OOT3, OOT4 = f"{H}/exp069/oot", f"{H}/exp070/oot3", f"{H}/exp071/out"
EPS, REF = 1e-6, 0.2498069
K = 100000.0 / REF
S5 = [42, 777, 2024, 101, 7]
AA = (42, 7, 2024)
W4 = {"LGB": 0.05, "CAT": 0.68, "NN": 0.15, "LEAF": 0.12}
BO = dict(alpha=1.07911, beta_skew=-0.059151, c_aa=0.25244, c_T70=0.04756,
          w_out=0.7, w_v6_inner=0.6668397489, w_ebm=-0.1, beta_ins=0.03)
LB_GAIN, LB_W = 5.0813180433, 0.20
BASE_EXPECT = 928.501461058684
N_TRAIN_OOT, N_TRAIN_DEPLOY = 1116277, 1475092
lg = lambda p: logit(np.clip(np.asarray(p, np.float64), EPS, 1 - EPS))
pts = lambda b: 100000.0 * (1.0 - b / REF)
t0 = time.time()

# ── 스텝 지점 → 예측 파일 (재사용 4개 + 이번 실험 신규) ────────────────────
REUSE = {544: (B89, "M_oot23_s{s}"), 720: (B89, "S720_oot23_s{s}"),
         816: (B89, "E3_oot23_s{s}"), 1088: (B89, "S1088_oot23_s{s}")}
NEW_STEPS = [n for n in (1360, 1632, 2176, 2720, 3264, 4080, 5440, 8160,
                         10880, 13600, 16320)
             if all(os.path.exists(f"{B90}/out/preds/z_K{n}_oot23_s{s}.npy")
                    for s in S5)]
POINTS = {}
SHA = {}
for n, (root, pat) in REUSE.items():
    zs = []
    for s in S5:
        tag = pat.format(s=s)
        p = f"{root}/out/preds/z_{tag}.npy"
        z = np.load(p)
        h = hashlib.sha256(np.ascontiguousarray(z).tobytes()).hexdigest()
        rec = json.load(open(f"{root}/out/fit_{tag}.json"))["z_sha256"]
        SHA[f"{n}|{s}"] = {"file_sha256": h, "recorded_in_exp089": rec,
                           "MATCH": bool(h == rec)}
        zs.append(z)
    POINTS[n] = ("reused_exp089", zs)
for n in NEW_STEPS:
    zs = []
    for s in S5:
        tag = f"K{n}_oot23_s{s}"
        z = np.load(f"{B90}/out/preds/z_{tag}.npy")
        h = hashlib.sha256(np.ascontiguousarray(z).tobytes()).hexdigest()
        rec = json.load(open(f"{B90}/out/fit_{tag}.json"))["z_sha256"]
        SHA[f"{n}|{s}"] = {"file_sha256": h, "recorded_in_exp090": rec,
                           "MATCH": bool(h == rec)}
        zs.append(z)
    POINTS[n] = ("new_exp090", zs)
assert all(v["MATCH"] for v in SHA.values()), "예측 파일 sha 불일치"

# ── 면 ────────────────────────────────────────────────────────────────
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
           ("p_lgb", "p_cat", "p_leaf", "p_nn", "p_q", "p_ebm"))    # S14
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
root = lambda zm: BO["w_out"] * zm + BO["c_aa"] * L_aa + BO["c_T70"] * L_t70


def alpha_free(z, conv):
    r = minimize_scalar(lambda a: -sc(a * z, conv), bounds=(0.3, 3.0),
                        method="bounded", options={"xatol": 1e-8})
    return float(-r.fun), float(r.x)


def sd_match(znew, zref):                       # S25/S33 — 척도 몫 제거
    return ((znew - znew.mean()) / znew.std()) * zref.std() + zref.mean()


# ── 투수 블록 부트스트랩 (exp_089 와 같은 시드·같은 추출 순서) ──────────────
rng = np.random.default_rng(20260824)
uq, inv = np.unique(pit, return_inverse=True)
nb = len(uq)
order = np.argsort(inv, kind="stable")
st = np.searchsorted(inv[order], np.arange(nb))
en = np.append(st[1:], len(order))
BSIZE = (en - st).astype(np.float64)
PICKS = np.stack([rng.integers(0, nb, nb) for _ in range(1000)])


def boot_t(d):
    """d = per-row (Brier_new − Brier_base). 음수면 이득. 블록합으로 정확히 같은 통계."""
    bs_ = np.array([d[order[st[j]:en[j]]].sum() for j in range(nb)])
    num = bs_[PICKS].sum(1)
    den = BSIZE[PICKS].sum(1)
    bs = num / den
    se = float(bs.std(ddof=1))
    obs = float(d.mean())
    return {"gain_pts": -obs * K, "boot_se_brier": se,
            "t": float(-obs / se) if se > 0 else None,
            "ci95_pts": [float(-np.percentile(bs, 97.5) * K),
                         float(-np.percentile(bs, 2.5) * K)]}


R = {"exp_id": "exp_090", "stage": "s1_curve", "K_pts_per_brier": K,
     "ref_brier": REF, "BO_constants": BO, "LB_gain_measured": LB_GAIN,
     "LB_w_deep": LB_W, "n_rows": int(len(y)), "y_mean": YB,
     "n_train_oot23": N_TRAIN_OOT, "n_train_deploy_le2024": N_TRAIN_DEPLOY,
     "steps_per_epoch": {"oot23": N_TRAIN_OOT // 4096,
                         "le2024": N_TRAIN_DEPLOY // 4096},
     "pred_sha_check": SHA, "pred_sha_all_match": True,
     "surface": ("모든 팔 out-of-time. exp_089 s4_final 면 그대로. 딥만 학습량을 바꿔 꽂는다."),
     "lr_schedule_note": ("스케줄러 없음(lr 1e-3 상수) — 총 스텝을 바꿔도 LR 궤적이 안 바뀐다. "
                          "따라서 이 축의 변인은 학습량 하나다."),
     }
z0 = root(Z_MATE)
base = {c: sc(z0, c) for c in ("rc", "abd")}
base_af = {c: alpha_free(z0, c) for c in ("rc", "abd")}
base_p = {c: prob(z0, c) for c in ("rc", "abd")}
R["baseline_w0"] = {"rc": base["rc"], "abd": base["abd"],
                    "rc_expected": BASE_EXPECT,
                    "rc_abs_diff": abs(base["rc"] - BASE_EXPECT),
                    "alpha_free": {c: base_af[c][0] for c in base_af},
                    "z_mate_alone_recentred": sc(Z_MATE, "rc")}
assert R["baseline_w0"]["rc_abs_diff"] < 1e-9, base["rc"]
R["baseline_gate_PASS"] = True

GRID = [round(x, 4) for x in np.arange(0.0, 0.6001, 0.01)]
R["points"] = {}
for n in sorted(POINTS):
    src, zs = POINTS[n]
    dz = np.mean(zs, 0)
    ent = {"total_steps": n, "epochs_equiv_oot23": n / (N_TRAIN_OOT // 4096),
           "source": src,
           "alone_raw_score": pts(float(np.mean((expit(dz) - y) ** 2))),
           "alone_recentred_score": sc(dz, "rc"),
           "recentring_c": pin(dz, F_RC),
           "z_sd": float(dz.std()), "z_mean": float(dz.mean()),
           "rho_vs_z_mate_oot": float(np.corrcoef(dz, Z_MATE)[0, 1]),
           "within_seed_rho": float(np.mean(
               [np.corrcoef(zs[i], zs[j])[0, 1]
                for i, j in itertools.combinations(range(5), 2)])),
           "per_seed_alone_recentred": [sc(z, "rc") for z in zs]}
    for conv in ("rc", "abd"):
        cur = []
        for w in GRID:
            cur.append({"w": w, "gain": sc(root((1 - w) * Z_MATE + w * dz), conv)
                        - base[conv]})
        best = max(cur, key=lambda c: c["gain"])
        z20 = root((1 - LB_W) * Z_MATE + LB_W * dz)
        zbe = root((1 - best["w"]) * Z_MATE + best["w"] * dz)
        d20 = (prob(z20, conv) - y) ** 2 - (base_p[conv] - y) ** 2
        b20 = boot_t(d20)
        if best["w"] > 0:
            dbe = (prob(zbe, conv) - y) ** 2 - (base_p[conv] - y) ** 2
            bbe = boot_t(dbe)
        else:
            bbe = {"gain_pts": 0.0, "t": 0.0, "boot_se_brier": None,
                   "ci95_pts": [0.0, 0.0]}
        ent[conv] = {
            "gain_at_w0.20": [c for c in cur if abs(c["w"] - LB_W) < 1e-9][0]["gain"],
            "gain_at_w0.20_alpha_free": alpha_free(z20, conv)[0] - base_af[conv][0],
            "gain_at_w0.20_sd_matched": sc(sd_match(z20, z0), conv) - base[conv],
            "boot_t_at_w0.20": b20["t"], "boot_ci95_at_w0.20": b20["ci95_pts"],
            "argmax_w": best["w"], "gain_at_argmax": best["gain"],
            "boot_t_at_argmax": bbe["t"], "boot_ci95_at_argmax": bbe["ci95_pts"],
            "B_K_mean_delta2_at_w0.20": K * float(np.mean(
                (prob(z20, conv) - base_p[conv]) ** 2)),
            "curve": cur,
        }
    R["points"][str(n)] = ent
    print(f"steps {n:6d} ({ent['epochs_equiv_oot23']:5.2f} ep) alone_rc "
          f"{ent['alone_recentred_score']:8.2f} | abd w*={ent['abd']['argmax_w']:.2f} "
          f"gmax {ent['abd']['gain_at_argmax']:+8.3f} (t {ent['abd']['boot_t_at_argmax']:+.2f}) "
          f"| w=.20 {ent['abd']['gain_at_w0.20']:+8.3f} "
          f"(t {ent['abd']['boot_t_at_w0.20']:+.2f}, sdm "
          f"{ent['abd']['gain_at_w0.20_sd_matched']:+7.3f})", flush=True)

# ── 사전 등록 판정 ─────────────────────────────────────────────────────
ref = R["points"]["720"]["abd"]["gain_at_w0.20"]
cand = []
for k, v in R["points"].items():
    if int(k) == 720:
        continue
    g, t = v["abd"]["gain_at_w0.20"], v["abd"]["boot_t_at_w0.20"]
    if t is not None and t >= 2.0 and g >= 3.0 * ref:
        cand.append({"steps": int(k), "gain_at_w0.20": g, "t": t,
                     "ratio_vs_720": g / ref if ref else None})
best_pt = max(R["points"].items(),
              key=lambda kv: kv[1]["abd"]["gain_at_w0.20"])
R["PREREGISTERED_VERDICT"] = {
    "rule": "abd 면 w=0.20 에서 t >= 2.0 이고 이득이 720스텝 지점의 3배 이상인 지점이 있으면 stage2",
    "ref_720_gain_at_w0.20": ref, "threshold_gain": 3.0 * ref,
    "qualifying_points": cand,
    "PASS": bool(len(cand) > 0),
    "argmax_point_steps": int(best_pt[0]),
    "argmax_point_gain_at_w0.20": best_pt[1]["abd"]["gain_at_w0.20"],
    "monotone_to_end": bool(
        int(best_pt[0]) == max(int(k) for k in R["points"])),
}
# 고원 지점 후보와 배포 환산 (두 해석 나란히)
s_star = int(best_pt[0])
R["DEPLOY_STEP_CONVERSION"] = {
    "s_star_oot_steps": s_star,
    "interpretation_A_same_steps": s_star,
    "interpretation_B_same_epochs_rowratio": s_star * N_TRAIN_DEPLOY / N_TRAIN_OOT,
    "interpretation_B_same_epochs_integer_spe":
        s_star * (N_TRAIN_DEPLOY // 4096) / (N_TRAIN_OOT // 4096),
    "row_ratio": N_TRAIN_DEPLOY / N_TRAIN_OOT,
    "spe_ratio": (N_TRAIN_DEPLOY // 4096) / (N_TRAIN_OOT // 4096),
    "deployed_arm_steps": 720,
}
R["wall_sec"] = round(time.time() - t0, 1)
json.dump(R, open(f"{B90}/out/s1_curve.json", "w"), ensure_ascii=False, indent=1)
slim = {k: v for k, v in R.items() if k not in ("points", "pred_sha_check")}
slim["points_summary"] = {
    k: {kk: vv for kk, vv in v.items() if kk not in ("rc", "abd")}
    | {"abd": {a: b for a, b in v["abd"].items() if a != "curve"}}
    for k, v in R["points"].items()}
print(json.dumps(slim, ensure_ascii=False, indent=1))
