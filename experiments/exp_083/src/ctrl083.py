"""exp_083 대조 — A 갈래 정직면 +176 이 무엇의 몫인지 가른다.  학습 0건.

셋을 잰다.
 (1) 행 순서 정합 assert (정직면 행 == tr2024 프레임 행)
 (2) 귀무 — 축 벡터를 전체 순열해 같은 스크린을 돌린다.  이 스크린의 잡음바닥.
 (3) 귀속 대조 — 우리 AA 팔의 ★자기 출력★(stage-2 로짓·최종 로짓)을 같은 축으로
     넣는다.  정직면 Z0 는 c_aa*L_aa 로 AA 를 이미 담고 있지만 그 L_aa 는
     exp_049 A1(3시드·1400라운드) 이고 우리가 포획한 것은 배포 18시드다.
     자기 출력이 큰 이득을 내면 +176 은 '실패유형 방향'이 아니라
     '더 좋은 AA 팔' 의 몫이다.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "6"
import json, sys, time                                              # noqa: E402
import numpy as np                                                  # noqa: E402
import pandas as pd                                                 # noqa: E402
from scipy.optimize import minimize_scalar                          # noqa: E402
from scipy.special import expit, logit                              # noqa: E402

sys.path.insert(0, "/home/<user>/SungjinPark/exp078/src")
sys.path.insert(0, "/home/<user>/SungjinPark/exp079/src")
import core078 as C                                                 # noqa: E402
import surface079 as S                                              # noqa: E402

H = C.H; K = C.K; REF = C.REF; N = C.N_ROWS
OUT = f"{H}/exp083"
EPS = 1e-6
T0 = time.time()
R = {}


def pts(b):
    return 100000.0 * (1.0 - b / REF)


def lgc(p):
    return logit(np.clip(np.asarray(p, np.float64), EPS, 1 - EPS))


IDS = pd.read_csv(f"{C.FRAME}/test.csv", encoding="utf-8-sig",
                  usecols=["row_id"])["row_id"].astype(str).to_numpy()
_rid = np.asarray(np.load(f"{H}/exp069/oot/row_id_oot2024.npy",
                          allow_pickle=True)).astype(str)
R["row_order_gate"] = {"honest_surface_ids_equal_tr2024_ids": bool(np.array_equal(_rid, IDS)),
                       "n": int(len(IDS))}
assert np.array_equal(_rid, IDS), "정직면 행 순서가 tr2024 프레임과 다르다"
print(f"[gate] 행 순서 일치 n={len(IDS)}", flush=True)

CA = np.load(f"{OUT}/out/cap_aa_cal.npz", allow_pickle=True)
NAMES = list(CA["names"])
SEEDS_R = [42, 7, 2024, 1, 13, 101, 777, 2025, 31337, 3, 17, 77, 123, 555, 999,
           2019, 8191, 65537]
CELLS = ["cell_is_mid_only.cbm", "cell_is_rev_only.cbm", "cell_is_both.cbm",
         "cell_is_wayoff_true.cbm"]


def pick(b, s, f):
    return CA[f"v{NAMES.index(f'{b}|{b}_seed_{s}_{f}'):03d}"].astype(np.float64)


amr = np.zeros(N); awo = np.zeros(N)
stk = np.zeros(N)
for s in SEEDS_R:
    c = [pick("R", s, f) for f in CELLS]
    mr = c[0] + c[1] + c[2]; wo = c[3]
    Ssum = mr + wo + np.maximum(0.0, 1.0 - mr - wo)
    amr += lgc(mr / Ssum); awo += lgc(wo / Ssum)
    stk += pick("R", s, "stacker.cbm")
AA_MR = amr / len(SEEDS_R); AA_WO = awo / len(SEEDS_R)
AA_STK = stk / len(SEEDS_R)                    # stage-2 확률 평균 (배포 규약)

LV0 = (S.W4["LGB"] * S._L["p_lgb"] + S.W4["CAT"] * S.lg(S.V["p_cat"])
       + S.W4["NN"] * S._L["p_nn"] + S.W4["LEAF"] * S._L["p_leaf"])
Z0 = (S.BO["w_out"] * ((1 - S.BO["w_ebm"]) * (S.BO["w_v6_inner"] * LV0
                       + (1 - S.BO["w_v6_inner"]) * S._L["p_q"])
                       + S.BO["w_ebm"] * S._L["p_ebm"]
                       + S.BO["beta_ins"] * S.V["off"])
      + S.BO["c_aa"] * S.L_aa + S.BO["c_T70"] * S.L_t70)
yo = S.yo


def z_pts(z):
    p = expit(z + S.rc(z))
    return pts(float(np.mean((p - yo) ** 2)))


BASE = z_pts(Z0)
R["base_pts"] = BASE
print(f"[base] 정직면 {BASE:.5f}점", flush=True)


def screen(v, bound=0.35):
    u = (np.asarray(v, np.float64) - float(np.mean(v)))
    u = u / max(float(u.std()), 1e-300)
    r = minimize_scalar(lambda t: -z_pts(Z0 + t * u), bounds=(-bound, bound),
                        method="bounded", options={"xatol": 1e-7})
    return float(r.x), float(-r.fun) - BASE


# ── (3) 귀속 대조 ────────────────────────────────────────────────────────────
CTRL = {}
for nm, v in (("aa_stage2_logit_18seed", lgc(AA_STK)),
              ("aa_p_off_logit", lgc(CA["p_off"])),
              ("aa_p_final_logit", lgc(CA["p_final"])),
              ("surface_own_L_aa(exp049 A1 3seed)", S.L_aa),
              ("A_aa_lg_mr", AA_MR),
              ("A_aa_lg_wo", AA_WO)):
    t, g = screen(v)
    CTRL[nm] = {"argmax_t_unit_sd": t, "raw_gain_pts": g}
    print(f"[대조] {nm:<36} t {t:+.5f}  이득 {g:+10.4f}점", flush=True)
CTRL["rho_lg_mr_vs_aa_stage2"] = float(np.corrcoef(AA_MR, lgc(AA_STK))[0, 1])
CTRL["rho_lg_mr_vs_L_aa"] = float(np.corrcoef(AA_MR, S.L_aa)[0, 1])
CTRL["rho_aa_stage2_vs_L_aa"] = float(np.corrcoef(lgc(AA_STK), S.L_aa)[0, 1])
R["attribution_controls"] = CTRL

# ── (3b) L_aa 를 배포 18시드로 갈아끼운 면 위에서 다시 재기 ──────────────────
# +176 이 '실패유형 방향'인지 '더 좋은 AA 팔'인지 가르는 결정적 대조.
Z1 = Z0 - S.BO["c_aa"] * S.L_aa + S.BO["c_aa"] * lgc(AA_STK)
B1 = z_pts(Z1)


def z_pts1(z):
    return z_pts(z)


def screen1(v, bound=0.35):
    u = (np.asarray(v, np.float64) - float(np.mean(v)))
    u = u / max(float(u.std()), 1e-300)
    r = minimize_scalar(lambda t: -z_pts1(Z1 + t * u), bounds=(-bound, bound),
                        method="bounded", options={"xatol": 1e-7})
    return float(r.x), float(-r.fun) - B1


t1, g1 = screen1(AA_MR)
t2, g2 = screen1(AA_WO)
R["surface_with_deployed_AA_arm"] = {
    "what": "정직면의 c_aa 항을 exp_049 A1(3시드) 대신 배포 18시드 stage-2 로짓으로 교체",
    "base_pts_swapped": B1, "gain_of_swap_alone_pts": B1 - BASE,
    "A_aa_lg_mr_gain_on_swapped_surface_pts": g1, "A_aa_lg_mr_argmax_t": t1,
    "A_aa_lg_wo_gain_on_swapped_surface_pts": g2}
print(f"[교체면] 팔 교체만으로 {B1-BASE:+.4f}점 · 그 위에서 lg_mr {g1:+.4f}점",
      flush=True)

# ── (2) 귀무 — 전체 순열 ─────────────────────────────────────────────────────
rng = np.random.default_rng(83)
NULL = {}
for nm, v in (("A_aa_lg_mr", AA_MR), ("E_aa_seed_sd18", None)):
    if v is None:
        stk2 = np.stack([lgc(pick("R", s, "stacker.cbm")) for s in SEEDS_R])
        v = stk2.std(axis=0, ddof=1)
    gs = []
    for i in range(30):
        gs.append(screen(v[rng.permutation(N)])[1])
    gs = np.array(gs)
    NULL[nm] = {"n_perm": 30, "mean_pts": float(gs.mean()), "sd_pts": float(gs.std(ddof=1)),
                "max_pts": float(gs.max()), "p95_pts": float(np.quantile(gs, 0.95))}
    print(f"[귀무] {nm:<20} 평균 {gs.mean():+.5f} sd {gs.std(ddof=1):.5f} "
          f"max {gs.max():+.5f}", flush=True)
R["permutation_null"] = NULL

R["wall_sec"] = time.time() - T0
json.dump(R, open(f"{OUT}/out/ctrl_cal.json", "w"), indent=1, ensure_ascii=False)
print(f"\n기록 -> {OUT}/out/ctrl_cal.json ({R['wall_sec']:.1f}s)", flush=True)
