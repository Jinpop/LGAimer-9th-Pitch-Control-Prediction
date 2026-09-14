"""exp_083 분석 — 이미 계산되고 버려지는 per-row 중간값 전수 조사.  학습 0건."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "6"
import json, sys, time                                              # noqa: E402
import numpy as np                                                  # noqa: E402
import pandas as pd                                                 # noqa: E402
from scipy.optimize import brentq, minimize_scalar                  # noqa: E402
from scipy.special import expit, logit                              # noqa: E402

sys.path.insert(0, "/home/<user>/SungjinPark/exp078/src")
sys.path.insert(0, "/home/<user>/SungjinPark/exp079/src")
import core078 as C                                                 # noqa: E402

H = C.H; K = C.K; REF = C.REF; N = C.N_ROWS; T_REF = C.T_REF
OUT = f"{H}/exp083"
FRAME = C.FRAME
EPS = 1e-6
XFER = 0.05153          # S54 — 팀원 b 축 실측 전이비 (정직면 +51.94414 → LB +2.6766924977)
T0 = time.time()
R = {"exp_id": "exp_083", "K_pts_per_brier": K, "ref_brier": REF, "n_rows": N,
     "threads": 6, "frame": FRAME, "transfer_ratio_S54": XFER,
     "no_training": True, "no_submission": True}


def pts(b):
    return 100000.0 * (1.0 - b / REF)


def lgc(p):
    return logit(np.clip(np.asarray(p, np.float64), EPS, 1.0 - EPS))


CA = np.load(f"{OUT}/out/cap_aa_cal.npz", allow_pickle=True)
CM = np.load(f"{OUT}/out/cap_mate.npz", allow_pickle=True)
NAMES = list(CA["names"])
IDS = pd.read_csv(f"{FRAME}/test.csv", encoding="utf-8-sig",
                  usecols=["row_id"])["row_id"].astype(str).to_numpy()
assert np.array_equal(CA["row_id"].astype(str), IDS)
assert np.array_equal(CM["row_id"].astype(str), IDS)
GT = pd.read_csv(f"{FRAME}/test.csv", encoding="utf-8-sig",
                 usecols=["game_type"])["game_type"].astype(str).to_numpy()

# ── 캡처 무결성 게이트 ────────────────────────────────────────────────────────
GATE = {"v6_replication_ndiff": int(CM["gate_v6_ndiff"][0]),
        "v6_replication_max_abs": float(CM["gate_v6_max"][0]),
        "n_booster_calls_aa": len(NAMES),
        "aa_p_final_vs_stored": {}}
for tag, p in (("build_AV/work/tr2024_p_aa_AV.npy", f"{H}/build_AV/work/tr2024_p_aa_AV.npy"),
               ("build_AJ/work/tr2024_p_aa_off_fresh_AJ.npy",
                f"{H}/build_AJ/work/tr2024_p_aa_off_fresh_AJ.npy")):
    if os.path.exists(p):
        v = np.load(p).astype(np.float64)
        if len(v) == N:
            for nm, mine in (("p_final", CA["p_final"]), ("p_off", CA["p_off"])):
                d = np.abs(mine - v)
                GATE["aa_p_final_vs_stored"][f"{tag}|{nm}"] = {
                    "ndiff": int(np.count_nonzero(d)), "max_abs": float(d.max())}
R["capture_gates"] = GATE
print(f"[gate] v6 재현 ndiff={GATE['v6_replication_ndiff']} · AA 부스터 호출 {len(NAMES)}",
      flush=True)


def pick(branch, seed, leaf):
    i = NAMES.index(f"{branch}|{branch}_seed_{seed}_{leaf}")
    return CA[f"v{i:03d}"].astype(np.float64)


SEEDS_R = [42, 7, 2024, 1, 13, 101, 777, 2025, 31337, 3, 17, 77, 123, 555, 999,
           2019, 8191, 65537]
SEEDS_F = [42, 7, 2024]
CELLS = ["cell_is_mid_only.cbm", "cell_is_rev_only.cbm", "cell_is_both.cbm",
         "cell_is_wayoff_true.cbm"]


def failtype_logits(branch, seeds):
    """팀원 판례와 ★같은 규약★: p_mr=(mid+rev+both)/S, p_wo=wo/S,
    S=mr+wo+max(0,1-mr-wo); clip 1e-6; 시드 방향 로짓 평균."""
    amr = np.zeros(N); awo = np.zeros(N); nclip = 0
    for s in seeds:
        c = [pick(branch, s, f) for f in CELLS]
        mr = c[0] + c[1] + c[2]
        wo = c[3]
        S = mr + wo + np.maximum(0.0, 1.0 - mr - wo)
        pm, pw = mr / S, wo / S
        nclip += int(((pm < EPS) | (pm > 1 - EPS) | (pw < EPS) | (pw > 1 - EPS)).sum())
        amr += lgc(pm); awo += lgc(pw)
    return amr / len(seeds), awo / len(seeds), nclip


AA_MR_R, AA_WO_R, NCL_R = failtype_logits("R", SEEDS_R)
AA_MR_F, AA_WO_F, NCL_F = failtype_logits("F", SEEDS_F)
# 배포 분기 혼합과 같은 규약: R 행은 R 분기, F 행은 0.5/0.5
wF = np.where(GT == "F", 0.5, 0.0)
AA_MR = (1 - wF) * AA_MR_R + wF * AA_MR_F
AA_WO = (1 - wF) * AA_WO_R + wF * AA_WO_F

# 시드 산포 (E) — 시드별 stage-2 로짓의 표본 sd
STK_R = np.stack([lgc(pick("R", s, "stacker.cbm")) for s in SEEDS_R])
STK_F = np.stack([lgc(pick("F", s, "stacker.cbm")) for s in SEEDS_F])
AA_SD18 = STK_R.std(axis=0, ddof=1)
AA_SDF = STK_F.std(axis=0, ddof=1)
CELL_SD = {}
for ci, f in enumerate(CELLS):
    v = np.stack([lgc(pick("R", s, f)) for s in SEEDS_R])
    CELL_SD[f.replace("cell_", "").replace(".cbm", "")] = v.std(axis=0, ddof=1)
del STK_R, STK_F

# 팀원 b/c 축 (Q 셀) — exp_082 산출물 재사용
_ft = pd.read_csv(f"{OUT.replace('exp083','exp082')}/runs/ft_matecal.csv",
                  encoding="utf-8-sig", float_precision="round_trip")
assert _ft["row_id"].astype(str).tolist() == list(IDS)
Q_MR = _ft["lg_mr"].to_numpy(np.float64)
Q_WO = _ft["lg_wo"].to_numpy(np.float64)

NNPC = CM["nn_hidden_pc"]
EBT = CM["ebm_term_top"]
EBN = list(CM["ebm_term_names"])

# ══════════════════════════════ 후보 축 사전 ═════════════════════════════════
AX = {}
AX["A_aa_lg_mr"] = AA_MR
AX["A_aa_lg_wo"] = AA_WO
AX["A_aa_lg_mr_Ronly"] = AA_MR_R
AX["A_aa_lg_wo_Ronly"] = AA_WO_R
AX["A_aa_minus_q_mr"] = AA_MR - Q_MR          # 두 모델의 ★불일치★ — 순수 새 자유도 후보
AX["A_aa_minus_q_wo"] = AA_WO - Q_WO
for i in range(3):
    AX[f"D_nn_hidden_pc{i+1}"] = NNPC[:, i]
AX["D_nn_hidden_l2"] = CM["nn_hidden_l2"]
AX["D_nn_hidden_nact"] = CM["nn_hidden_nact"]
for i in range(6):
    AX[f"B_ebm_term_{EBN[i]}"] = EBT[:, i].astype(np.float64)
AX["C_leaf_absmass"] = CM["leaf_abs"]
AX["C_leaf_pos"] = CM["leaf_pos"]
AX["C_leaf_neg"] = CM["leaf_neg"]
AX["E_aa_seed_sd18"] = AA_SD18
AX["E_aa_seed_sdF"] = AA_SDF
for k, v in CELL_SD.items():
    AX[f"E_aa_cellsd_{k}"] = v
AX["E_v6_cat_seed_sd"] = CM["seed_cat_sd"]
AX["E_v6_nn_seed_sd"] = CM["seed_nn_sd"]
AX["E_v6_lgb_seed_sd"] = CM["seed_lgb_sd"]
AX["E_v6_leaf_seed_sd"] = CM["seed_leaf_sd"]
AX["E_v6_ebm_seed_sd"] = CM["seed_ebm_sd"]
AX["F_leaf_oov"] = CM["leaf_oov"]
for j in range(3):
    AX[f"F_lut{j}_hit"] = CA[f"lut{j}_has"].astype(np.float64)

DEAD = {}
for k in list(AX):
    v = np.asarray(AX[k], np.float64)
    if not np.isfinite(v).all():
        DEAD[k] = "비유한값 포함"; del AX[k]; continue
    if float(v.std()) < 1e-12:
        DEAD[k] = (f"이 프레임에서 상수다 (sd={float(v.std()):.3e}, "
                   f"값={float(v[0]):.6g}) → δ 가 항등적으로 0, 측정 불가")
        del AX[k]
R["axes_dropped_degenerate"] = DEAD
print(f"[축] 살아있는 후보 {len(AX)} · 축퇴 탈락 {len(DEAD)}", flush=True)

# ══════════════════════════════ δ / G / ρ ═══════════════════════════════════
W_TR = C.zf(C.BO, C.ARMS18)
D_BO = C.anc(W_TR, T_REF)
p_BO = expit(W_TR + D_BO)
R["base"] = {"d_adj_solved": repr(D_BO), "mean_p_BO": float(p_BO.mean())}

MU = {k: float(np.asarray(v, np.float64).mean()) for k, v in AX.items()}
SD = {k: float(np.asarray(v, np.float64).std()) for k, v in AX.items()}


def pred(k, t):
    o = t * (np.asarray(AX[k], np.float64) - MU[k])
    return expit(W_TR + C.anc(W_TR + o, T_REF) + o)


# 기존 축 δ (같은 프레임)
EXIST = {a: C.DELTA[C.IDX[a]] for a in C.AXES}
_n9 = list(np.load(f"{H}/exp080/out/cand_delta_names.npy", allow_pickle=True))
_d9 = np.load(f"{H}/exp080/out/cand_delta.npy")
for i, nm in enumerate(_n9):
    EXIST[f"exp080/{nm}"] = _d9[i]
# 팀원 b·c 축 δ (같은 산술)
for nm, vv in (("teammate_b(q_lg_mr)", Q_MR), ("teammate_c(q_lg_wo)", Q_WO)):
    h = 0.002
    o1, o2 = h * (vv - vv.mean()), -h * (vv - vv.mean())
    EXIST[nm] = (expit(W_TR + C.anc(W_TR + o1, T_REF) + o1)
                 - expit(W_TR + C.anc(W_TR + o2, T_REF) + o2)) / (2 * h)


def rho(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


DEL, RES = {}, {}
for k in AX:
    h = 0.002 / max(SD[k], 1e-12)           # t·sd(v) ≈ 0.002 로짓
    d = (pred(k, h) - pred(k, -h)) / (2 * h)
    DEL[k] = d
    g = float(np.mean(d ** 2))

    def KQ(t):
        return K * float(np.mean((pred(k, t) - p_BO) ** 2)) - 1.0
    a1 = 1.0 / np.sqrt(K * g) if g > 0 else np.nan
    try:
        aex = brentq(KQ, a1 * 0.05, a1 * 20.0, xtol=1e-14)
    except Exception:
        aex = a1
    rr = {}
    for a, dv in EXIST.items():
        r = rho(d, dv)
        if np.isfinite(r):
            rr[a] = r
    mx = max(rr.items(), key=lambda kv: abs(kv[1])) if rr else ("", float("nan"))
    RES[k] = {"sd_v": SD[k], "mean_v": MU[k], "rms_delta": float(np.sqrt(g)),
              "G_jj": g, "K_G_jj_pts_at_u1": K * g,
              "amp_KQ1": float(aex), "amp_KQ1_times_0p2": float(0.2 * aex),
              "K_Q_at_0p2x_pts": K * float(np.mean((pred(k, 0.2 * aex) - p_BO) ** 2)),
              "max_abs_rho_axis": mx[0], "max_abs_rho": abs(mx[1]),
              "max_abs_rho_signed": mx[1],
              "rho_top5": dict(sorted(rr.items(), key=lambda kv: -abs(kv[1]))[:5])}
    print(f"[δ] {k:<34} rms {RES[k]['rms_delta']:.4e}  K·G {K*g:9.3f}  "
          f"KQ1 {aex:.5g}  max|ρ| {abs(mx[1]):.4f} ({mx[0]})", flush=True)

# 형제(exp_083 내부) 축과의 ρ 는 ★판정 기준이 아니다★ — config 가 사전 등록한
# "기존 축"은 exp_078 15축 + q_beta + q_league 4 + 셀 8 + 팀원 b·c 이고 형제는 그 목록에 없다.
# 그래도 S16(다축 동시 적재 위험) 때문에 별도 필드로 기록한다.
for k in AX:
    sib = {}
    for k2 in AX:
        if k2 == k:
            continue
        r = rho(DEL[k], DEL[k2])
        if np.isfinite(r):
            sib[k2] = r
    mxs = max(sib.items(), key=lambda kv: abs(kv[1])) if sib else ("", float("nan"))
    RES[k]["max_abs_rho_sibling_exp083"] = abs(mxs[1])
    RES[k]["max_abs_rho_sibling_axis"] = mxs[0]
    RES[k]["rho_top5"] = dict(sorted(RES[k]["rho_top5"].items(),
                                     key=lambda kv: -abs(kv[1]))[:6])
R["axes"] = RES

# ══════════════════════════════ 정직면 ══════════════════════════════════════
import surface079 as S                                              # noqa: E402
yo = S.yo
LV0 = (S.W4["LGB"] * S._L["p_lgb"] + S.W4["CAT"] * S.lg(S.V["p_cat"])
       + S.W4["NN"] * S._L["p_nn"] + S.W4["LEAF"] * S._L["p_leaf"])
Z0 = (S.BO["w_out"] * ((1 - S.BO["w_ebm"]) * (S.BO["w_v6_inner"] * LV0
                       + (1 - S.BO["w_v6_inner"]) * S._L["p_q"])
                       + S.BO["w_ebm"] * S._L["p_ebm"]
                       + S.BO["beta_ins"] * S.V["off"])
      + S.BO["c_aa"] * S.L_aa + S.BO["c_T70"] * S.L_t70)
MU0, SD0 = float(Z0.mean()), float(Z0.std())


def z_pts(z):
    p = expit(z + S.rc(z))
    return pts(float(np.mean((p - yo) ** 2)))


BASE = z_pts(Z0)
R["honest_surface_base_pts"] = BASE
# ≤2023 학습 여부 — 정직면 누출의 유일한 기전 (S54)
CLEAN = {"A_": True, "E_aa": True, "B_": False, "C_": False, "D_": False,
         "E_v6": False, "F_": False}


def cleanliness(k):
    if k.startswith("A_") or k.startswith("E_aa"):
        return True
    return False


HS = {}
for k in AX:
    v = (np.asarray(AX[k], np.float64) - MU[k]) / max(SD[k], 1e-300)   # 단위 sd

    def raw(t):
        return z_pts(Z0 + t * v)

    def sdm(t):
        z = Z0 + t * v
        return z_pts((z - float(z.mean())) / float(z.std()) * SD0 + MU0)
    rr = minimize_scalar(lambda t: -raw(t), bounds=(-0.35, 0.35),
                         method="bounded", options={"xatol": 1e-7})
    t_raw, g_raw = float(rr.x), float(-rr.fun) - BASE
    cl = cleanliness(k)
    HS[k] = {"unit": "축을 단위 sd 로 표준화한 t",
             "raw_argmax_t": t_raw, "raw_gain_pts": g_raw,
             "sd_matched_gain_at_argmax_pts": sdm(t_raw) - BASE,
             "sd_ratio_at_argmax": float((Z0 + t_raw * v).std() / SD0),
             "models_trained_le2023": bool(cl),
             "xfer_converted_pts_S54": g_raw * XFER,
             "read_to_judge_on": ("raw (모델이 ≤2023 학습이라 2024 정직면에 표본 외)"
                                 if cl else "xfer (S54 전이비 0.05153 적용)"),
             "judged_pts": g_raw if cl else g_raw * XFER}
    print(f"[정직면] {k:<34} t {t_raw:+.4f}  raw {g_raw:+9.4f}  "
          f"×0.05153 {g_raw*XFER:+8.4f}  clean={cl}", flush=True)
R["honest_surface"] = HS

R["wall_sec"] = time.time() - T0
os.makedirs(f"{OUT}/out", exist_ok=True)
json.dump(R, open(f"{OUT}/out/analyze_cal.json", "w"), indent=1, ensure_ascii=False)
print(f"\n기록 -> {OUT}/out/analyze_cal.json  ({R['wall_sec']:.1f}s)", flush=True)
