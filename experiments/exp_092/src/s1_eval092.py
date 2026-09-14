"""exp_092 stage 1 — 축 D(피처 부분집합)·E(손실)·F(잔차 타깃)의 정직면 곡선과 **짝지은** t.

면은 exp_089 s4_final / exp_090 s1_curve / exp_091 s1_eval 의 것을 **글자 그대로 승계**한다:
    LV      = .05·lg(p_lgb) + .68·lg(p_cat) + .15·lg(p_nn) + .12·lg(p_leaf)   (OOT 4팔)
    z_mate  = (1−w_ebm)·( w_v6i·LV + (1−w_v6i)·LQ ) + w_ebm·LE + beta_ins·off (OOT)
    z       = w_out·[(1−w)·z_mate + w·z_deep] + c_aa·L_AA + c_T70·L_T70       (OOT)
    rc  : p = expit(z + c)                       (기준점 대조용)
    abd : p = expit(alpha·z + beta_skew·z² + d)  (배포 관례 — ★주 판정면★)

🔴 게이트 셋
  ① 기준점  : w=0 · rc 가 928.501461058684 를 abs_diff < 1e-9 로 재현
  ② 기준선  : 1632·5시드·현행구조의 abd argmax 이득이 13.714309769419287 · argmax w 0.36
  ③ sha    : 모든 z 벡터가 자기 fit json 의 z_sha256 과 일치
  ②는 부트스트랩 PICKS(rng 20260824·같은 추출 순서)까지 함께 검사한다 — 짝지은 비교의 전제.

사전 등록 판정 (S8 — 결과 보고 바꾸지 않는다):
  주 기준 P1 = 각 지점을 **자기 argmax w**, 기준선을 **자기 argmax w(0.36)** 에 놓고
               같은 행 위에서 짝지은 투수 블록 부트스트랩 1000회. t >= 2.0 이면 통과.
  t 부호: **양수면 후보가 낫다**.

🔑 이 실험의 머리 지표는 `rho(z_mate)` 다. 전 지점의 (rho, 혼합이득) 산점을 내고,
   exp_091 의 22지점까지 합쳐(같은 면·같은 기준선) 관계가 단조인지 어디서 꺾이는지 본다.
"""
import itertools, json, os, sys, time, hashlib
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "6"
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from scipy.stats import spearmanr

H = "/home/<user>/SungjinPark"
B89, B90, B91, B92 = f"{H}/exp089", f"{H}/exp090", f"{H}/exp091", f"{H}/exp092"
OOT, OOT3, OOT4 = f"{H}/exp069/oot", f"{H}/exp070/oot3", f"{H}/exp071/out"
EPS, REF_BR = 1e-6, 0.2498069
K = 100000.0 / REF_BR
S5 = [42, 777, 2024, 101, 7]
AA = (42, 7, 2024)
W4 = {"LGB": 0.05, "CAT": 0.68, "NN": 0.15, "LEAF": 0.12}
BO = dict(alpha=1.07911, beta_skew=-0.059151, c_aa=0.25244, c_T70=0.04756,
          w_out=0.7, w_v6_inner=0.6668397489, w_ebm=-0.1, beta_ins=0.03)
LB_W = 0.20
BASE_EXPECT = 928.501461058684
REF_GAIN_EXPECT = 13.714309769419287
REF_W_EXPECT = 0.36
lg = lambda p: logit(np.clip(np.asarray(p, np.float64), EPS, 1 - EPS))
pts = lambda b: 100000.0 * (1.0 - b / REF_BR)
t0 = time.time()
PLAN = json.load(open(f"{B92}/out/plan.json"))

SHA = {}


def zload(root, tag):
    z = np.load(f"{root}/out/preds/z_{tag}.npy")
    h = hashlib.sha256(np.ascontiguousarray(z).tobytes()).hexdigest()
    rec = json.load(open(f"{root}/out/fit_{tag}.json"))["z_sha256"]
    SHA[f"{os.path.basename(root)}:{tag}"] = {"file_sha256": h, "recorded": rec,
                                              "MATCH": bool(h == rec)}
    assert h == rec, f"sha 불일치 {root} {tag}"
    return z


Z_REF5 = [zload(B90, f"K1632_oot23_s{s}") for s in S5]      # D0 = 기준선 (exp_090 재사용)

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

# 축 F 오프셋의 valid 조각 (out-of-time — s0_offF092.py 게이트 참조)
OFF_V = np.load(f"{B92}/work/offF_all.npy")[np.load(f"{B89}/work/valid_mask.npy")]
assert np.isfinite(OFF_V).all() and len(OFF_V) == len(y)

# ── 지점 조립 ─────────────────────────────────────────────────────────
POINTS = {}
for name, m in PLAN["points"].items():
    meta = {"axis": m["axis"], "n_masked": m["n_masked"], "loss": m["loss"],
            "desc": m["desc"], "rows_restrict": bool(m.get("rows")),
            "offset_used": bool(m.get("off")), "IS_BASELINE": m["BASELINE"]}
    zs = list(Z_REF5) if m["BASELINE"] else [zload(B92, f"{name}_s{s}") for s in S5]
    POINTS[name] = (meta, zs)
# 🔑 F1 은 한 적합에서 두 읽기가 나온다: 잔차만 / 잔차+오프셋 (오프셋도 out-of-time)
if "F1_resid_off" in POINTS:
    m, zs = POINTS["F1_resid_off"]
    POINTS["F1_resid_plus_off"] = (dict(m, desc=m["desc"] + " — **평가 시 오프셋을 더한 읽기**",
                                        variant="z = model(x) + offset_valid"),
                                   [z + OFF_V for z in zs])


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
mixz = lambda dz, w: root((1 - w) * Z_MATE + w * dz)


def alpha_free(z, conv):
    r = minimize_scalar(lambda a: -sc(a * z, conv), bounds=(0.3, 3.0),
                        method="bounded", options={"xatol": 1e-8})
    return float(-r.fun), float(r.x)


def sd_match(znew, zref):                       # S25/S33 — 척도 몫 제거
    return ((znew - znew.mean()) / znew.std()) * zref.std() + zref.mean()


# ── 투수 블록 부트스트랩 (exp_089/090/091 과 같은 시드·같은 추출 순서) ─────────
rng = np.random.default_rng(20260824)
uq, inv = np.unique(pit, return_inverse=True)
nb = len(uq)
order = np.argsort(inv, kind="stable")
st = np.searchsorted(inv[order], np.arange(nb))
en = np.append(st[1:], len(order))
BSIZE = (en - st).astype(np.float64)
PICKS = np.stack([rng.integers(0, nb, nb) for _ in range(1000)])
IDX = [order[st[j]:en[j]] for j in range(nb)]


def blockstat(d):
    bs_ = np.array([d[IDX[j]].sum() for j in range(nb)])
    return bs_[PICKS].sum(1) / BSIZE[PICKS].sum(1)


def boot_gain(d):
    bs = blockstat(d)
    se = float(bs.std(ddof=1))
    obs = float(d.mean())
    return {"gain_pts": -obs * K, "boot_se_brier": se,
            "t": float(-obs / se) if se > 0 else None,
            "ci95_pts": [float(-np.percentile(bs, 97.5) * K),
                         float(-np.percentile(bs, 2.5) * K)]}


def paired_t(sq_cand, sq_ref):
    d = sq_ref - sq_cand
    bs = blockstat(d)
    se = float(bs.std(ddof=1))
    return {"delta_pts_cand_minus_ref": float(d.mean()) * K,
            "t": float(d.mean() / se) if se > 0 else None,
            "p_share_cand_worse": float((bs < 0).mean()),
            "ci95_pts": [float(np.percentile(bs, 2.5) * K),
                         float(np.percentile(bs, 97.5) * K)]}


R = {"exp_id": "exp_092", "stage": "s1_eval", "K_pts_per_brier": K,
     "ref_brier": REF_BR, "BO_constants": BO, "n_rows": int(len(y)),
     "y_mean": YB, "fixed_total_steps": 1632,
     "surface": "exp_089 정직면(모든 팔 out-of-time). 딥 팔만 설정을 바꿔 꽂는다.",
     "t_sign_convention": "양수면 후보가 기준선보다 낫다",
     "pred_sha_check": SHA}

z0 = root(Z_MATE)
base = {c: sc(z0, c) for c in ("rc", "abd")}
base_af = {c: alpha_free(z0, c) for c in ("rc", "abd")}
base_p = {c: prob(z0, c) for c in ("rc", "abd")}
R["baseline_w0"] = {"rc": base["rc"], "abd": base["abd"],
                    "rc_expected": BASE_EXPECT,
                    "rc_abs_diff": abs(base["rc"] - BASE_EXPECT),
                    "z_mate_alone_recentred": sc(Z_MATE, "rc")}
assert R["baseline_w0"]["rc_abs_diff"] < 1e-9, base["rc"]
R["gate_baseline_point_PASS"] = True
R["offset_arm_on_surface"] = {
    "rho_vs_z_mate_oot": float(np.corrcoef(OFF_V, Z_MATE)[0, 1]),
    "alone_recentred_pts": sc(OFF_V, "rc"), "sd": float(OFF_V.std())}

GRID = [round(x, 4) for x in np.arange(0.0, 0.6001, 0.01)]


def measure(name, meta, zs):
    dz = np.mean(zs, 0)
    n = len(zs)
    ent = dict(meta)
    ent.update({
        "name": name, "n_seeds": n,
        "alone_raw_score": pts(float(np.mean((expit(dz) - y) ** 2))),
        "alone_recentred_score": sc(dz, "rc"),
        "z_sd": float(dz.std()), "z_mean": float(dz.mean()),
        "rho_vs_z_mate_oot": float(np.corrcoef(dz, Z_MATE)[0, 1]),
        "within_seed_rho": float(np.mean(
            [np.corrcoef(zs[i], zs[j])[0, 1]
             for i, j in itertools.combinations(range(n), 2)])),
        "per_seed_alone_recentred": [sc(z, "rc") for z in zs],
    })
    for conv in ("rc", "abd"):
        cur = [{"w": w, "gain": sc(mixz(dz, w), conv) - base[conv]} for w in GRID]
        best = max(cur, key=lambda c: c["gain"])
        z20 = mixz(dz, LB_W)
        ent[conv] = {
            "argmax_w": best["w"], "gain_at_argmax": best["gain"],
            "gain_at_w0.20": [c for c in cur if abs(c["w"] - LB_W) < 1e-9][0]["gain"],
            "gain_at_w0.36": [c for c in cur if abs(c["w"] - 0.36) < 1e-9][0]["gain"],
            "curve": cur,
        }
        if conv == "abd":
            ent[conv]["gain_at_w0.20_sd_matched"] = sc(sd_match(z20, z0), conv) - base[conv]
            ent[conv]["gain_at_w0.20_alpha_free"] = alpha_free(z20, conv)[0] - base_af[conv][0]
    ent["_dz"] = dz
    return ent


ENT = {}
for name in POINTS:
    meta, zs = POINTS[name]
    ENT[name] = measure(name, meta, zs)
    e = ENT[name]
    print(f"{name:22s} rho {e['rho_vs_z_mate_oot']:.4f} alone_rc "
          f"{e['alone_recentred_score']:8.2f} | abd w*={e['abd']['argmax_w']:.2f} "
          f"gmax {e['abd']['gain_at_argmax']:+8.3f} | w.20 {e['abd']['gain_at_w0.20']:+8.3f}",
          flush=True)

REFNAME = "D0_all99_bce"
ref = ENT[REFNAME]
R["gate_reference_reproduction"] = {
    "what": "exp_090/091 의 1632스텝·5시드·현행구조 abd argmax 이득/argmax w 재현",
    "gain_at_argmax": ref["abd"]["gain_at_argmax"], "expected": REF_GAIN_EXPECT,
    "abs_diff": abs(ref["abd"]["gain_at_argmax"] - REF_GAIN_EXPECT),
    "argmax_w": ref["abd"]["argmax_w"], "argmax_w_expected": REF_W_EXPECT,
    "gain_at_w0.20": ref["abd"]["gain_at_w0.20"],
    "rho_vs_z_mate_oot": ref["rho_vs_z_mate_oot"],
    "PASS": bool(abs(ref["abd"]["gain_at_argmax"] - REF_GAIN_EXPECT) < 1e-9
                 and abs(ref["abd"]["argmax_w"] - REF_W_EXPECT) < 1e-12)}
assert R["gate_reference_reproduction"]["PASS"], R["gate_reference_reproduction"]

SQ = lambda dz, w: (prob(mixz(dz, w), "abd") - y) ** 2
sq_ref_argmax = SQ(ref["_dz"], ref["abd"]["argmax_w"])
sq_ref_020 = SQ(ref["_dz"], LB_W)
sq_ref_036 = SQ(ref["_dz"], 0.36)
R["points"] = {}
for name, e in ENT.items():
    dz, wstar = e["_dz"], e["abd"]["argmax_w"]
    out = {k: v for k, v in e.items() if k != "_dz"}
    out["abd"] = {k: v for k, v in e["abd"].items() if k != "curve"}
    out["rc"] = {k: v for k, v in e["rc"].items() if k != "curve"}
    out["curve_abd"] = e["abd"]["curve"]
    if name == REFNAME:
        out["paired_vs_reference"] = {
            "P1_own_argmax": {"t": 0.0, "delta_pts_cand_minus_ref": 0.0,
                              "note": "기준선 자기 자신"},
            "P2_w0.20": {"t": 0.0}, "P3_w0.36": {"t": 0.0}}
    else:
        out["paired_vs_reference"] = {
            "P1_own_argmax": dict(paired_t(SQ(dz, wstar), sq_ref_argmax),
                                  cand_w=wstar, ref_w=ref["abd"]["argmax_w"]),
            "P2_w0.20": dict(paired_t(SQ(dz, LB_W), sq_ref_020), w=LB_W),
            "P3_w0.36": dict(paired_t(SQ(dz, 0.36), sq_ref_036), w=0.36)}
    out["boot_vs_w0_at_w0.20"] = boot_gain(SQ(dz, LB_W) - (base_p["abd"] - y) ** 2)
    out["boot_vs_w0_at_argmax"] = (
        boot_gain(SQ(dz, wstar) - (base_p["abd"] - y) ** 2) if wstar > 0 else
        {"gain_pts": 0.0, "t": 0.0})
    zs = POINTS[name][1]
    n = len(zs)
    per = [pts(float(np.mean((prob(mixz(z, LB_W), "abd") - y) ** 2))) - base["abd"]
           for z in zs]
    jk = np.array([pts(float(np.mean((prob(mixz(np.mean(
        [zs[i] for i in range(n) if i != k], 0), LB_W), "abd") - y) ** 2)))
        - base["abd"] for k in range(n)])
    out["seed_noise"] = {
        "single_seed_gain_mean": float(np.mean(per)),
        "single_seed_gain_sd": float(np.std(per, ddof=1)),
        "jackknife_se_of_mean": float(np.sqrt((n - 1) / n * np.sum((jk - jk.mean()) ** 2))),
        "single_seed_gains": per}
    R["points"][name] = out
    p1 = out["paired_vs_reference"]["P1_own_argmax"]
    print(f"  {name:22s} P1_t {(p1['t'] if p1['t'] is not None else 0):+6.2f}  "
          f"P2_t {out['paired_vs_reference']['P2_w0.20'].get('t', 0) or 0:+6.2f}  "
          f"P3_t {out['paired_vs_reference']['P3_w0.36'].get('t', 0) or 0:+6.2f}",
          flush=True)

# ── D5 귀무 대조와의 대조표 (개수 맞춤) ──────────────────────────────────
NULLPAIR = {"D1_no_sp": "D5m06_rand", "D2_no_ins": "D5m16_rand",
            "D3_no_i3anom_p6": "D5m13_rand", "D4_sit_cat_only": "D5m83_rand"}
R["D5_null_comparison"] = {}
for fam, nul in NULLPAIR.items():
    if fam not in ENT or nul not in ENT:
        continue
    a, b = ENT[fam], ENT[nul]
    pr = paired_t(SQ(a["_dz"], a["abd"]["argmax_w"]),
                  SQ(b["_dz"], b["abd"]["argmax_w"]))
    R["D5_null_comparison"][fam] = {
        "null_point": nul, "n_masked": a["n_masked"],
        "family_rho": a["rho_vs_z_mate_oot"], "null_rho": b["rho_vs_z_mate_oot"],
        "d_rho_family_minus_null": a["rho_vs_z_mate_oot"] - b["rho_vs_z_mate_oot"],
        "family_gain": a["abd"]["gain_at_argmax"], "null_gain": b["abd"]["gain_at_argmax"],
        "d_gain_family_minus_null": a["abd"]["gain_at_argmax"] - b["abd"]["gain_at_argmax"],
        "paired_t_family_vs_null": pr["t"],
        "🔑": ("계열 마스킹이 같은 개수 무작위 마스킹보다 rho 를 더 내리고 이득을 더 올리면 "
              "'그 계열이 겹침의 원인'이다. 아니면 '그냥 정보를 줄인 것'이다")}

# ── (rho, 이득) 산점 — exp_091 22지점까지 합친다 (같은 면·같은 기준선) ────────
SCAT = [{"exp": "exp_092", "point": n, "axis": ENT[n]["axis"],
         "rho": ENT[n]["rho_vs_z_mate_oot"],
         "gain_at_argmax": ENT[n]["abd"]["gain_at_argmax"],
         "argmax_w": ENT[n]["abd"]["argmax_w"],
         "alone_rc": ENT[n]["alone_recentred_score"]} for n in ENT]
try:
    E91 = json.load(open(f"{B91}/out/s1_eval.json"))
    assert abs(E91["baseline_w0"]["rc"] - base["rc"]) < 1e-9, "exp_091 면 불일치"
    for n, v in E91["points"].items():
        SCAT.append({"exp": "exp_091", "point": n, "axis": v["axis"],
                     "rho": v["rho_vs_z_mate_oot"],
                     "gain_at_argmax": v["abd"]["gain_at_argmax"],
                     "argmax_w": v["abd"]["argmax_w"],
                     "alone_rc": v["alone_recentred_score"]})
except FileNotFoundError:
    pass
rh = np.array([s["rho"] for s in SCAT])
gn = np.array([s["gain_at_argmax"] for s in SCAT])
al = np.array([s["alone_rc"] for s in SCAT])
own = np.array([s["exp"] == "exp_092" for s in SCAT])
lin = np.polyfit(rh, gn, 1)
R["rho_gain_scatter"] = {
    "n_points_total": int(len(SCAT)), "n_points_exp092": int(own.sum()),
    "pearson_rho_vs_gain": float(np.corrcoef(rh, gn)[0, 1]),
    "spearman_rho_vs_gain": float(spearmanr(rh, gn).statistic),
    "pearson_rho_vs_gain_exp092_only": float(np.corrcoef(rh[own], gn[own])[0, 1]),
    "spearman_rho_vs_gain_exp092_only": float(spearmanr(rh[own], gn[own]).statistic),
    "pearson_alone_vs_gain": float(np.corrcoef(al, gn)[0, 1]),
    "spearman_alone_vs_gain": float(spearmanr(al, gn).statistic),
    "ols_gain_on_rho": {"slope_pts_per_unit_rho": float(lin[0]),
                        "intercept": float(lin[1])},
    "rho_range": [float(rh.min()), float(rh.max())],
    "gain_range": [float(gn.min()), float(gn.max())],
    "points": sorted(SCAT, key=lambda s: s["rho"])}

cand = []
for name, v in R["points"].items():
    if name == REFNAME:
        continue
    t = v["paired_vs_reference"]["P1_own_argmax"]["t"]
    if t is not None and t >= 2.0:
        cand.append({"point": name, "P1_t": t,
                     "gain_at_argmax": v["abd"]["gain_at_argmax"],
                     "argmax_w": v["abd"]["argmax_w"],
                     "rho": v["rho_vs_z_mate_oot"]})
best_mix = max(R["points"].items(), key=lambda kv: kv[1]["abd"]["gain_at_argmax"])
best_alone = max(R["points"].items(), key=lambda kv: kv[1]["alone_recentred_score"])
R["PREREGISTERED_VERDICT"] = {
    "rule": ("주 기준 P1 — 각 지점을 자기 argmax w 에서, 기준선(D0 = 1632·5시드·현행구조·BCE·전체 99열)"
             "을 argmax w 0.36 에서 놓고 짝지은 투수 블록 부트스트랩 t >= 2.0"),
    "reference_point": REFNAME,
    "reference_gain_at_argmax": ref["abd"]["gain_at_argmax"],
    "reference_argmax_w": ref["abd"]["argmax_w"],
    "reference_rho": ref["rho_vs_z_mate_oot"],
    "qualifying_points": cand, "PASS": bool(len(cand) > 0),
    "S99_best_mix_point": best_mix[0],
    "S99_best_mix_gain_at_argmax": best_mix[1]["abd"]["gain_at_argmax"],
    "S99_best_alone_point": best_alone[0],
    "S99_best_alone_recentred_score": best_alone[1]["alone_recentred_score"],
    "S99_same_point": bool(best_mix[0] == best_alone[0])}
R["wall_sec"] = round(time.time() - t0, 1)
json.dump(R, open(f"{B92}/out/s1_eval.json", "w"), ensure_ascii=False, indent=1)
slim = {k: v for k, v in R.items() if k not in ("points", "pred_sha_check")}
slim["rho_gain_scatter"] = {k: v for k, v in R["rho_gain_scatter"].items()
                            if k != "points"}
print(json.dumps(slim, ensure_ascii=False, indent=1))
