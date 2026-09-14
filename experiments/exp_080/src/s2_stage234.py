"""exp_080 stage2·3·4 — 새 상수 축 재고 · 보정셀 잔여 · 방향 집합 확정.  학습 0건.

core078 을 그대로 import 해 배포 산술·15축 DELTA·G 를 재사용한다 (파일 출력 없음).
새로 만드는 축은 전부 **배포 패키지에서 실제로 바꿀 수 있는 상수**여야 한다.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "6"
import io, json, sys, time
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

sys.path.insert(0, "/home/<user>/SungjinPark/exp078/src")
import core078 as C                                              # noqa: E402

H, K, EPS, REF = C.H, C.K, C.EPS, C.REF
OUT = "/home/<user>/SungjinPark/exp080/out"
os.makedirs(OUT, exist_ok=True)
N = C.N_ROWS
BO, AXES, IDX = C.BO, C.AXES, C.IDX
T1 = time.time()
R = {"exp_id": "exp_080", "part": "stage2+3+4", "K_pts_per_brier": K,
     "core078_wall_min": (time.time() - C.T0) / 60.0}
print(f"\n### core078 적재 {R['core078_wall_min']:.2f}분", flush=True)

# ══════════════════════════════════════════════════════════════════════
# 준비 — 배포 프레임의 Q 오프셋 편차 d_m 재구성 (배포 표 그대로)
# ══════════════════════════════════════════════════════════════════════
MQ = json.load(open(f"{H}/build_BO/pkg_BO/mate/model_q/meta.json"))
OFS, BKO = MQ["offset"], MQ["backoff"]
BETA_Q, EBK = float(OFS["beta"]), float(OFS["eb_k"])
LOOK_D, LOOK_N, LEAGUE = OFS["lookup_d"], OFS["lookup_n"], BKO["league_dev"]
GT_CELLS = ["1|1", "1|2", "2|1", "2|2"]

fr = pd.read_csv(f"{H}/build_AC/work/tr2024/test.csv", encoding="utf-8-sig",
                 usecols=["row_id", "game_type", "pitcher_id", "batter_hand",
                          "pitcher_hand"])
assert len(fr) == N


def _key(a, b):
    ka = pd.to_numeric(a, errors="coerce")
    kb = pd.to_numeric(b, errors="coerce")
    ok = ka.notna() & kb.notna()
    out = np.array([""] * len(a), dtype=object)
    out[ok.to_numpy()] = [f"{int(x)}|{int(y)}"
                          for x, y in zip(ka[ok].astype(int), kb[ok].astype(int))]
    return out


kp = _key(fr.pitcher_id, fr.batter_hand)
kl = _key(fr.pitcher_hand, fr.batter_hand)
d_p = np.array([LOOK_D.get(k, 0.0) for k in kp], np.float64)
hit = np.array([k in LOOK_D for k in kp])
n_cell = np.array([LOOK_N.get(k, 0) for k in kp], np.float64)
Lv_row = np.array([LEAGUE.get(k, 0.0) for k in kl], np.float64)
has_L = np.array([k in LEAGUE for k in kl])
D_M = d_p + np.where(hit, 0.0, Lv_row)
NUM = np.where(hit, d_p * (n_cell + EBK), 0.0)          # 확장 분자 (s - n*overall)
R["q_offset_reconstruction"] = {
    "n_hit": int(hit.sum()), "hit_frac": float(hit.mean()),
    "n_miss": int((~hit).sum()), "miss_frac": float((~hit).mean()),
    "meta_says_miss_about": "~22%",
    "n_league_present": int(has_L.sum()),
    "d_m_stats": {"min": float(D_M.min()), "max": float(D_M.max()),
                  "mean": float(D_M.mean()), "sd": float(D_M.std())},
    "lookup_diag_d_abs_max": OFS["lookup_diag"]["d_abs_max"],
    "hard_backoff_exact_hit": bool(np.array_equal(D_M[hit], d_p[hit])),
    "hard_backoff_exact_miss": bool(np.array_equal(D_M[~hit], Lv_row[~hit])),
    "d_p_zero_on_miss": bool(np.all(d_p[~hit] == 0.0))}
print("Q 오프셋 재구성:", {k: v for k, v in R["q_offset_reconstruction"].items()
                      if not isinstance(v, dict)}, flush=True)

# 🔴 arms/lg_p_q 는 블렌딩 빌드 산출 = logit(clip(p + beta*d_m)) — gamma 미적용
#    (q_infer.py L649 "블렌딩 빌드: stage 4(gamma)를 적용하지 않는다")
P_OFF = expit(C.LGQ)
P_PRE = P_OFF - BETA_Q * D_M
y = C.y
R["q_pre_offset_recovery"] = {
    "p_off_min": float(P_OFF.min()), "p_off_max": float(P_OFF.max()),
    "p_pre_min": float(P_PRE.min()), "p_pre_max": float(P_PRE.max()),
    "n_p_pre_out_of_01": int(((P_PRE <= 0) | (P_PRE >= 1)).sum()),
    "n_clip_active_at_deployed_beta":
        int(((P_PRE + BETA_Q * D_M < EPS) | (P_PRE + BETA_Q * D_M > 1 - EPS)).sum()),
    "residual_slope_on_deploy_frame_in_sample":
        float(np.mean(D_M * (y - P_OFF)) / np.mean(D_M ** 2)),
    "note": ("배포 프레임은 ≤2024 적합이라 표본 내다 — 이 기울기는 정직면의 "
             "beta 적합값(0.59072)과 같을 이유가 없다. 재구성이 산술적으로 "
             "닫히는지(p_pre 가 (0,1) 안인지)를 보는 용도다.")}
assert R["q_pre_offset_recovery"]["n_p_pre_out_of_01"] == 0
print("Q 사전확률 복원:", R["q_pre_offset_recovery"], flush=True)


# ══════════════════════════════════════════════════════════════════════
# 배포 산술 — LGQ 를 갈아끼울 수 있는 build (core078 과 같은 경로)
# ══════════════════════════════════════════════════════════════════════
def build_q(lgq, th=None):
    th = dict(BO if th is None else th)
    base = th["w_v6_inner"] * C.LGV6 + (1.0 - th["w_v6_inner"]) * lgq
    z = (1.0 - th["w_ebm"]) * base + th["w_ebm"] * C.PEBM + th["beta_ins"] * C.OFFR
    p = expit(z + C.D_INNER)
    b = io.StringIO()
    pd.DataFrame({"control_success": p}).to_csv(b, index=False)
    b.seek(0)
    p = pd.read_csv(b)["control_success"].to_numpy(np.float64)
    c = np.array([th["c_aa"], th["c_S2"], th["c_S1"], th["c_T70"]])
    zz = th["w_out"] * C.lg(p) + c @ C.ARMS18
    zp = (zz + th["off_BI"] * C.DIRS["off_BI"] + th["off_BJ"] * C.DIRS["off_BJ"]
          + th["off_BK"] * C.DIRS["off_BK"])
    f = th["alpha"] * zp + th["beta_skew"] * zp * zp
    return expit(f + C.anc(f, C.T_REF))


def lgq_of(beta=BETA_Q, k=EBK, league=None):
    lg_ = LEAGUE if league is None else league
    Lr = np.array([lg_.get(kk, 0.0) for kk in kl], np.float64) if league is not None else Lv_row
    ds = np.where(hit, NUM / (n_cell + k), 0.0)
    dm = ds + np.where(hit, 0.0, Lr)
    return C.lg(np.clip(P_PRE + beta * dm, EPS, 1 - EPS))


chk = build_q(lgq_of())
R["stage2_identity_at_deployed_constants"] = float(np.abs(chk - C.p_BO).max())
print(f"항등 앵커(배포 상수 재구성) max|Δ| = "
      f"{R['stage2_identity_at_deployed_constants']:.3e}", flush=True)
assert R["stage2_identity_at_deployed_constants"] < 1e-12, "Q 재구성 항등 실패"

# ── v6 내부 3축도 같은 프레임에서 δ 재생성 (직교화에 필요) ──────────
PL = np.load(f"{H}/exp078/work/tr2024_v6_pl.npy")
PC = np.load(f"{H}/exp078/work/tr2024_v6_pc.npy")
PN = np.load(f"{H}/exp078/work/tr2024_v6_pn.npy")
PF = np.load(f"{H}/exp078/work/tr2024_v6_pf.npy")
W4 = {"LGB": 0.05, "CAT": 0.68, "NN": 0.15, "LEAF": 0.12}
LG_RE = W4["LGB"] * PL + W4["CAT"] * PC + W4["NN"] * PN + W4["LEAF"] * PF
R["v6_capture_check_max_abs"] = float(np.abs(LG_RE - C.LGV6).max())
V6NEW = {"v6_CAT_vs_NN": PC - PN, "v6_CAT_vs_LEAF": PC - PF,
         "v6_CAT_vs_LGB": PC - PL}


def build_v6(nm, t):
    th = dict(BO)
    lgv6 = C.LGV6 + t * V6NEW[nm]
    base = th["w_v6_inner"] * lgv6 + (1.0 - th["w_v6_inner"]) * C.LGQ
    z = (1.0 - th["w_ebm"]) * base + th["w_ebm"] * C.PEBM + th["beta_ins"] * C.OFFR
    p = expit(z + C.D_INNER)
    b = io.StringIO()
    pd.DataFrame({"control_success": p}).to_csv(b, index=False)
    b.seek(0)
    p = pd.read_csv(b)["control_success"].to_numpy(np.float64)
    c = np.array([th["c_aa"], th["c_S2"], th["c_S1"], th["c_T70"]])
    zz = th["w_out"] * C.lg(p) + c @ C.ARMS18
    zp = (zz + th["off_BI"] * C.DIRS["off_BI"] + th["off_BJ"] * C.DIRS["off_BJ"]
          + th["off_BK"] * C.DIRS["off_BK"])
    f = th["alpha"] * zp + th["beta_skew"] * zp * zp
    return expit(f + C.anc(f, C.T_REF))


# ══════════════════════════════════════════════════════════════════════
# stage2 — 새 상수 축의 δ · G_jj · 기존축 상관 · 프로브 진폭
# ══════════════════════════════════════════════════════════════════════
CAND = {}          # name -> delta vector (배포 프레임)
META = {}


def add_axis(name, fplus, fminus, h, files, note, nonlin_fn=None):
    dj = (fplus - fminus) / (2 * h)
    Gjj = float(np.mean(dj ** 2))
    rho = {a: float(np.corrcoef(dj, C.DELTA[IDX[a]])[0, 1]) for a in AXES}
    for vn, vv in V6D.items():
        if vn != name:
            rho[vn] = float(np.corrcoef(dj, vv)[0, 1])
    mx = max(rho, key=lambda k: abs(rho[k]))
    CAND[name] = dj
    META[name] = {"rms_delta": float(np.sqrt(Gjj)), "G_jj": Gjj,
                  "K_times_G_jj_pts_at_u1": float(K * Gjj),
                  "max_abs_delta": float(np.abs(dj).max()),
                  "mean_delta": float(dj.mean()),
                  "corr_with_existing": rho,
                  "max_abs_corr_existing": float(abs(rho[mx])),
                  "argmax_abs_corr_axis": mx,
                  "probe_u_for_KQ_1pt_analytic": float(1.0 / np.sqrt(K * Gjj)),
                  "changed_files": files, "note": note}
    return dj


V6D = {}
H6 = 0.002
for nm in V6NEW:
    V6D[nm] = (build_v6(nm, H6) - build_v6(nm, -H6)) / (2 * H6)
for nm in V6NEW:
    dj = V6D[nm]
    Gjj = float(np.mean(dj ** 2))
    rho = {a: float(np.corrcoef(dj, C.DELTA[IDX[a]])[0, 1]) for a in AXES}
    mx = max(rho, key=lambda k: abs(rho[k]))
    CAND[nm] = dj
    META[nm] = {"rms_delta": float(np.sqrt(Gjj)), "G_jj": Gjj,
                "K_times_G_jj_pts_at_u1": float(K * Gjj),
                "max_abs_delta": float(np.abs(dj).max()),
                "mean_delta": float(dj.mean()),
                "corr_with_existing": rho,
                "max_abs_corr_existing": float(abs(rho[mx])),
                "argmax_abs_corr_axis": mx,
                "probe_u_for_KQ_1pt_analytic": float(1.0 / np.sqrt(K * Gjj)),
                "changed_files": ["mate/model/blend.json (v6 4팔 가중)",
                                  "blend.json (d_adj 재고정)"],
                "note": "exp_078 stage4 신설축.  파트너 v6 내부 가중이라 우리가 바꾸려면 "
                        "파트너 패키지의 v6 가중 블록을 건드린다."}
    print(f"[{nm}] rms(δ) {np.sqrt(Gjj):.4e} K*G_jj {K*Gjj:.4f} "
          f"max|ρ| {abs(rho[mx]):.4f} ({mx})", flush=True)

# q_beta
HB = 0.01
add_axis("q_beta", build_q(lgq_of(beta=BETA_Q + HB)),
         build_q(lgq_of(beta=BETA_Q - HB)), HB,
         ["mate/model_q/meta.json (offset.beta)", "blend.json (d_adj 재고정)"],
         "선형 상수.  q_infer.py L507 이 meta.json 에서 그대로 읽는다.")
# q_k  (수치미분 — 표가 k 의 비선형 함수라 δ 를 미분으로 잡는다)
HK = 20.0
add_axis("q_k", build_q(lgq_of(k=EBK + HK)), build_q(lgq_of(k=EBK - HK)), HK,
         ["mate/model_q/meta.json (offset.eb_k **그리고 lookup_d 1392개 전부 재생성**)",
          "blend.json (d_adj 재고정)"],
         "🔴 eb_k 는 추론에서 안 쓰인다(rebuilt_at_inference=False) — 출하되는 것은 "
         "lookup_d 표다.  k 를 바꾸면 표 1392개 값을 전부 다시 계산해 넣어야 하고, "
         "그래서 '상수 하나'가 아니라 '표 하나'가 바뀐다.  단위는 그루 아닌 k 1 이므로 "
         "δ 도 그 단위다.")
# league 4셀
HL = 0.002
for cc in GT_CELLS:
    lp = dict(LEAGUE); lp[cc] = LEAGUE[cc] + HL
    lm = dict(LEAGUE); lm[cc] = LEAGUE[cc] - HL
    add_axis(f"q_league_{cc.replace('|','')}", build_q(lgq_of(league=lp)),
             build_q(lgq_of(league=lm)), HL,
             ["mate/model_q/meta.json (backoff.league_dev)", "blend.json (d_adj 재고정)"],
             "개인 셀 MISS 행에만 닿는다.  4셀 중 공통이동은 앵커가 흡수하므로 "
             "실효 자유도는 3.")
for nm in ("q_beta", "q_k", "q_league_11", "q_league_12", "q_league_21", "q_league_22"):
    m = META[nm]
    print(f"[{nm}] rms(δ) {m['rms_delta']:.4e} K*G_jj {m['K_times_G_jj_pts_at_u1']:.4f} "
          f"max|ρ| {m['max_abs_corr_existing']:.4f} ({m['argmax_abs_corr_axis']})",
          flush=True)

# aa_fmix — 자산 없음
R["stage2_aa_fmix"] = {
    "status": "δ 미측정 — 분기별 예측 벡터가 캐시에 없다",
    "form": "F 행에서 p_aa = w*p_R분기 + (1-w)*p_F분기, w=0.5 (model/meta.json "
            "branch_weights_by_segment.F)",
    "delta_formula": "δ ∝ (p_R분기 - p_F분기)·1[game_type=='F'] 를 배포 산술에 통과",
    "n_F_rows": int((fr.game_type == "F").sum()),
    "F_frac": float((fr.game_type == "F").mean()),
    "changed_files": ["model/meta.json (branch_weights_by_segment.F)",
                      "blend.json (d_adj 재고정)"],
    "cost_to_measure": "AA 추론 1회(9시드 47열 + 3시드 63열 분기별 저장)가 필요하다. "
                       "캐시에는 혼합 후 벡터(pcal_aa)만 있다.",
    "🔴": "정직면에는 F 혼합 구조 자체가 없다(OOT AA 팔은 exp_049 단일 분기) — "
          "이 축의 정직면 사전근거는 원리적으로 측정 불가.  S7 도 F쪽 판정을 "
          "valid_F 로 하지 말라고 한다."}

# ── 프로브 진폭·비선형 (수치) ────────────────────────────────────────
def probe_curves(name, mover):
    def kq(u):
        return float(K * np.mean((mover(u) - C.p_BO) ** 2))

    lo, hi = 1e-6, None
    u1 = META[name]["probe_u_for_KQ_1pt_analytic"]
    lo, hi = u1 * 1e-3, u1 * 50
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        if kq(mid) < 1.0:
            lo = mid
        else:
            hi = mid
    uk = 0.5 * (lo + hi)
    dj = CAND[name]

    def nonlin(u):
        return float(np.linalg.norm((mover(u) - C.p_BO) / u - dj)
                     / np.linalg.norm(dj))

    lo2, hi2 = u1 * 1e-3, u1 * 200
    for _ in range(35):
        mid = 0.5 * (lo2 + hi2)
        if nonlin(mid) < 0.01:
            lo2 = mid
        else:
            hi2 = mid
    META[name]["probe_u_for_KQ_1pt"] = uk
    META[name]["KQ_at_that_u"] = kq(uk)
    META[name]["nonlinearity_at_that_u"] = nonlin(uk)
    META[name]["max_u_with_nonlinearity_under_1pct"] = lo2


probe_curves("q_beta", lambda u: build_q(lgq_of(beta=BETA_Q + u)))
probe_curves("q_k", lambda u: build_q(lgq_of(k=EBK + u)))
for cc in GT_CELLS:
    nm = f"q_league_{cc.replace('|','')}"
    probe_curves(nm, lambda u, cc=cc: build_q(lgq_of(
        league={**LEAGUE, cc: LEAGUE[cc] + u})))
for nm in V6NEW:
    probe_curves(nm, lambda u, nm=nm: build_v6(nm, u))
R["stage2_axes"] = META
print("\n프로브 진폭 완료", flush=True)
json.dump(R, open(f"{OUT}/s2_partial.json", "w"), indent=1, ensure_ascii=False)
np.save(f"{OUT}/cand_delta_names.npy", np.array(list(CAND.keys())))
np.save(f"{OUT}/cand_delta.npy", np.stack([CAND[k] for k in CAND]))
print(f"중간기록 완료 · {(time.time()-T1)/60:.2f}분", flush=True)
