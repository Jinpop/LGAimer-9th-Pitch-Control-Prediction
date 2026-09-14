"""exp_081 공통 — 평가면 E(정직 OOT 2024)와 대리면 P(보정 tr2024·≤2023 Q표) 구성.

🔴 이 실험에서 E 는 "정직면 이득을 재는 면"이 아니라 **채점 서버 역할**이다.
   E 에서 밖으로 나오는 정보는 축당 스칼라 1개(점수)뿐이다.  라벨 벡터는
   score_E() 안에서만 쓰이고 그 밖으로 나가지 않는다.

두 면의 관계 (실전과 동형)
  E  정직 OOT 2024 — 팔 전부 ≤2023 학습(표본 밖) · Q 오프셋 표 ≤2023 · 실제 라벨 보유
  P  보정 tr2024   — 팔 배포(≤2024 학습) · Q 오프셋 표만 ≤2023 재구성(build_BR 덤프)
                     라벨 안 씀.  G_jj 는 **여기서만** 계산한다.
  🔴 P 를 build_AC 대역 프레임 그대로 쓰면 q_league 4축의 δ 가 정확히 0 이다(S32).
     그래서 build_BR 이 만든 ≤2023 표 덤프(미적중 21.4988%)를 P 의 Q 경로로 쓴다.

산술은 양쪽 다 배포 BO 그대로:
  z   = w_out*((1-w_ebm)*(w_v6*lv + (1-w_v6)*lq) + w_ebm*le + beta_ins*off + d_inner)
        + c_aa*l_aa + c_T70*l_t70
  zp  = z + off_BI*dirBI + off_BJ*dirBJ + off_BK*dirBK      (= script.py 의 logit_mix)
  x   = zp                                                   (셀 경계용 x, S26·S29)
  f   = alpha*zp + beta_skew*zp^2
  p   = expit(f + d)      d 는 평균을 목표에 맞추는 앵커
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "6"

import io                                                        # noqa: E402
import json                                                      # noqa: E402
import sys                                                       # noqa: E402
import time                                                      # noqa: E402

import numpy as np                                               # noqa: E402
import pandas as pd                                              # noqa: E402
from scipy.special import expit, logit                           # noqa: E402

sys.path.insert(0, "/home/<user>/SungjinPark/exp078/src")
import core078 as C                                              # noqa: E402

H = C.H
K = C.K
REF = C.REF
EPS = C.EPS
N = C.N_ROWS
BO = dict(C.BO)
T_REF = C.T_REF
D_INNER = C.D_INNER
OUT = f"{H}/exp081"
os.makedirs(f"{OUT}/out", exist_ok=True)

OOT, OOT3, OOT4 = f"{H}/exp069/oot", f"{H}/exp070/oot3", f"{H}/exp071/out"
BRD = f"{H}/build_BR/work/dump_le23_bo_dump"
FRAME = f"{H}/build_AC/work/tr2024"
W4 = {"LGB": 0.05, "CAT": 0.68, "NN": 0.15, "LEAF": 0.12}
AA_SEEDS = (42, 7, 2024)
GT_CELLS = ["1|1", "1|2", "2|1", "2|2"]
GTS = ["R", "F"]
NCELL = 8
GATES = {}


def lg(p):
    return C.lg(p)


def pts(b):
    return 100000.0 * (1.0 - b / REF)


def _key(a, b):
    ka = pd.to_numeric(a, errors="coerce")
    kb = pd.to_numeric(b, errors="coerce")
    ok = (ka.notna() & kb.notna()).to_numpy()
    out = np.array([""] * len(ka), dtype=object)
    out[ok] = [f"{int(x)}|{int(y)}"
               for x, y in zip(ka[ok].astype(int), kb[ok].astype(int))]
    return out


def anchor(f, target, tol=1e-14, itmax=100, d0=0.0):
    """mean(expit(f+d)) == target 을 푸는 d (뉴턴).  d0 는 따뜻한 출발점."""
    d = float(d0)
    for _ in range(itmax):
        p = expit(f + d)
        mu = float(p.mean())
        if abs(mu - target) < tol:
            break
        step = (mu - target) / float((p * (1.0 - p)).mean())
        d -= step
        if abs(step) < tol:
            break
    return d


# ══════════════════════════════════════════════════════════════════════
# 1. 평가면 E — 정직 OOT 2024
# ══════════════════════════════════════════════════════════════════════
T0 = time.time()
_tr = pd.read_csv(f"{H}/data/train.csv",
                  usecols=["row_id", "season", "game_type", "control_success",
                           "pitcher_id", "batter_hand", "pitcher_hand"])
_tro = _tr[_tr.season == 2024].sort_values("row_id", kind="stable")
Y_E = _tro.control_success.to_numpy(np.float64)
YB = float(Y_E.mean())
_rid = np.asarray(np.load(f"{OOT}/row_id_oot2024.npy", allow_pickle=True)).astype(str)
assert np.array_equal(_rid, _tro.row_id.to_numpy().astype(str)), "OOT 행 순서 불일치"
assert len(Y_E) == N

_frP = pd.read_csv(f"{FRAME}/test.csv", encoding="utf-8-sig",
                   usecols=["row_id", "game_type", "pitcher_id", "batter_hand",
                            "pitcher_hand"])
assert len(_frP) == N
assert np.array_equal(_frP.row_id.to_numpy().astype(str), _rid), "P/E 행 정렬 불일치"
GT = _frP.game_type.to_numpy().astype(str)

# ── ≤2023 개인/리그 표 (정직 관계: ≤2023 표 → 2024 행) ────────────────
_m23 = (_tr.season <= 2023) & ~((_tr.season <= 2022) & (_tr.game_type == "F"))
_d23 = _tr[_m23 & _tr.control_success.notna()].copy()
for _c in ("pitcher_id", "batter_hand", "pitcher_hand"):
    _d23[_c] = _d23[_c].astype("Int64")
_ov = _d23.groupby("pitcher_id").control_success.mean()
_g = (_d23.groupby(["pitcher_id", "batter_hand"]).control_success
      .agg(["sum", "count"]).reset_index())
_g["overall"] = _g.pitcher_id.map(_ov)
_num = (_g["sum"].to_numpy(np.float64)
        - _g["count"].to_numpy(np.float64) * _g.overall.to_numpy(np.float64))
_kk = [f"{int(a)}|{int(b)}" for a, b in zip(_g.pitcher_id, _g.batter_hand)]
NUM23 = dict(zip(_kk, _num))
NC23 = dict(zip(_kk, _g["count"].to_numpy(np.int64)))
_lgp = _d23.groupby("pitcher_hand").control_success.mean()
_lpb = (_d23.groupby(["pitcher_hand", "batter_hand"]).control_success
        .agg(["mean", "count"]))
LD23 = {f"{int(a)}|{int(b)}": float(r["mean"] - _lgp.loc[a])
        for (a, b), r in _lpb.iterrows()}

KP_E = _key(_tro.pitcher_id, _tro.batter_hand)
KL_E = _key(_tro.pitcher_hand, _tro.batter_hand)
HIT_E = np.array([k in NUM23 for k in KP_E])
NUM_E = np.array([NUM23.get(k, 0.0) for k in KP_E], np.float64)
NCT_E = np.array([NC23.get(k, 0) for k in KP_E], np.float64)

MQ = json.load(open(f"{H}/build_BO/pkg_BO/mate/model_q/meta.json",
                    encoding="utf-8"))
BETA_Q = float(MQ["offset"]["beta"])
EBK = float(MQ["offset"]["eb_k"])
DS_E = np.where(HIT_E, NUM_E / (NCT_E + EBK), 0.0)


def d_E(league=None):
    lgd = LD23 if league is None else league
    Lr = np.array([lgd.get(k, 0.0) for k in KL_E], np.float64)
    return DS_E + np.where(HIT_E, 0.0, Lr)


VE = {n: np.load(p).astype(np.float64) for n, p in {
    "p_lgb": f"{OOT}/p_lgb_oot2024.npy", "p_cat": f"{OOT}/p_cat_oot2024.npy",
    "p_leaf": f"{OOT}/p_leaf_oot2024.npy", "p_nn": f"{OOT4}/p_nn_oot2024.npy",
    "p_q": f"{OOT}/p_q_oot2024.npy", "p_ebm": f"{OOT3}/p_ebm_oot2024.npy",
    "off": f"{OOT}/off_oot2024.npy"}.items()}
S14_E = {n: {"min": float(v.min()), "max": float(v.max()), "mean": float(v.mean())}
         for n, v in VE.items()}
LCAT_E, LNN_E = lg(VE["p_cat"]), lg(VE["p_nn"])
LV_E = (W4["LGB"] * lg(VE["p_lgb"]) + W4["CAT"] * LCAT_E
        + W4["NN"] * LNN_E + W4["LEAF"] * lg(VE["p_leaf"]))
LE_E = lg(VE["p_ebm"])
LAA_E = np.mean([lg(np.load(f"{H}/experiments/exp_049/pred/"
                            f"pred_A1_u2024_seed{s}_at1400.npy"))
                 for s in AA_SEEDS], axis=0)
LT70_E = np.mean([lg(np.load(f"{H}/experiments/exp_055/pred/"
                             f"pred_T70_u2024_seed{s}_at1400.npy"))
                  for s in AA_SEEDS], axis=0)

# 축 방향 벡터 — center/scale 은 출하 동결 상수(BL blend.json).  E 에는 그 상수를
# **E 의 raw** 에 적용한다(배포가 test raw 에 적용하는 것과 같은 관계).
_BLJ = json.load(open(f"{H}/build_BL/pkg_BL/blend.json", encoding="utf-8"))
_AXL = {a["name"]: a for a in _BLJ["axis_offsets"]}
_D59 = np.load(f"{H}/exp059_screen/directions.npz")


def _froz(raw, name):
    v = np.asarray(raw, np.float64)
    fin = np.isfinite(v)
    u = np.zeros(len(v))
    u[fin] = (v[fin] - _AXL[name]["center"]) / _AXL[name]["scale"]
    return u


DIR_E = {"BI": _froz(VE["off"], "beta_ins_off"),
         "BJ": _froz(_D59["ctrl_asof_psr"], "ctrl_asof_psr"),
         "BK": C.DIRS["off_BK"]}
DIR_P = {"BI": C.DIRS["off_BI"], "BJ": C.DIRS["off_BJ"], "BK": C.DIRS["off_BK"]}


def lq_E(beta=BETA_Q, league=None):
    return lg(np.clip(VE["p_q"] + beta * d_E(league), EPS, 1.0 - EPS))


def zp_E(dbeta=0.0, dleague=None, dv6=0.0, th=None):
    """E 의 logit_mix (alpha/beta_skew 적용 전, 축 오프셋 포함)."""
    t = BO if th is None else th
    lq = lq_E(BETA_Q + dbeta,
              None if not dleague else {**LD23, **dleague})
    lv = LV_E if dv6 == 0.0 else LV_E + dv6 * (LCAT_E - LNN_E)
    z = (t["w_out"] * ((1 - t["w_ebm"])
                       * (t["w_v6_inner"] * lv + (1 - t["w_v6_inner"]) * lq)
                       + t["w_ebm"] * LE_E + t["beta_ins"] * VE["off"] + D_INNER)
         + t["c_aa"] * LAA_E + t["c_T70"] * LT70_E)
    return (z + t["off_BI"] * DIR_E["BI"] + t["off_BJ"] * DIR_E["BJ"]
            + t["off_BK"] * DIR_E["BK"])


def f_of(zp, th=None):
    t = BO if th is None else th
    return t["alpha"] * zp + t["beta_skew"] * zp * zp


ZP_E = zp_E()
F_E = f_of(ZP_E)
D_E_ANC = anchor(F_E, YB)
P_E = expit(F_E + D_E_ANC)
W_E = P_E * (1.0 - P_E)
BRIER_E = float(np.mean((P_E - Y_E) ** 2))
BASE_E_PTS = pts(BRIER_E)

# ── E 게이트: exp_080 이 기록한 "간이 정직면"(alpha=1·오프셋 0) 재현 ──
_zsimple = (BO["w_out"] * ((1 - BO["w_ebm"])
                           * (BO["w_v6_inner"] * LV_E
                              + (1 - BO["w_v6_inner"]) * lg(VE["p_q"]))
                           + BO["w_ebm"] * LE_E + BO["beta_ins"] * VE["off"])
            + BO["c_aa"] * LAA_E + BO["c_T70"] * LT70_E)
_ps = expit(_zsimple + anchor(_zsimple, YB))
GATES["E1_simple_honest_base_no_qoffset"] = {
    "got": pts(float(np.mean((_ps - Y_E) ** 2))),
    "expected_exp078_079_080": 928.501461058684,
    "abs_diff": abs(pts(float(np.mean((_ps - Y_E) ** 2))) - 928.501461058684)}
GATES["E1_simple_honest_base_no_qoffset"]["PASS"] = bool(
    GATES["E1_simple_honest_base_no_qoffset"]["abs_diff"] < 1e-9)
_zsimple2 = _zsimple + (BO["w_out"] * (1 - BO["w_ebm"]) * (1 - BO["w_v6_inner"])
                        * (lq_E() - lg(VE["p_q"])))
_ps2 = expit(_zsimple2 + anchor(_zsimple2, YB))
GATES["E2_simple_honest_base_with_le2023_qoffset"] = {
    "got": pts(float(np.mean((_ps2 - Y_E) ** 2))),
    "expected_exp080": 937.8351186346334,
    "abs_diff": abs(pts(float(np.mean((_ps2 - Y_E) ** 2))) - 937.8351186346334)}
GATES["E2_simple_honest_base_with_le2023_qoffset"]["PASS"] = bool(
    GATES["E2_simple_honest_base_with_le2023_qoffset"]["abs_diff"] < 1e-9)
GATES["E3_honest_frame_miss_rate"] = {
    "n_miss": int((~HIT_E).sum()), "miss_frac": float((~HIT_E).mean()),
    "expected_exp080_BR": 0.21498814628392904,
    "PASS": bool(int((~HIT_E).sum()) == 54501)}

# ══════════════════════════════════════════════════════════════════════
# 2. 대리면 P — 보정 tr2024 · 팔 배포 · Q 오프셋 표만 ≤2023
# ══════════════════════════════════════════════════════════════════════
P_PRE = np.load(f"{BRD}/p_pre.npy")
D_P23 = np.load(f"{BRD}/d_p.npy")
HIT_P = np.load(f"{BRD}/hit.npy")
D_M23 = np.load(f"{BRD}/d_m.npy")
P_OFF23 = np.load(f"{BRD}/p_off.npy")
KL_P = _key(_frP.pitcher_hand, _frP.batter_hand)
LD23_BR = json.load(open(f"{H}/build_BR/work/solve_BR.json",
                         encoding="utf-8"))["le2023_tables"]["league_dev"]
PC_P = np.load(f"{H}/exp078/work/tr2024_v6_pc.npy")
PN_P = np.load(f"{H}/exp078/work/tr2024_v6_pn.npy")
PL_P = np.load(f"{H}/exp078/work/tr2024_v6_pl.npy")
PF_P = np.load(f"{H}/exp078/work/tr2024_v6_pf.npy")
GATES["P0_v6_capture_max_abs"] = float(np.abs(
    W4["LGB"] * PL_P + W4["CAT"] * PC_P + W4["NN"] * PN_P
    + W4["LEAF"] * PF_P - C.LGV6).max())


def d_P(league=None):
    lgd = LD23_BR if league is None else league
    Lr = np.array([lgd.get(k, 0.0) for k in KL_P], np.float64)
    return D_P23 + np.where(HIT_P, 0.0, Lr)


def lq_P(beta=BETA_Q, league=None):
    return lg(np.clip(P_PRE + beta * d_P(league), EPS, 1.0 - EPS))


def zp_P(dbeta=0.0, dleague=None, dv6=0.0, th=None, roundtrip=False, lgq=None):
    t = BO if th is None else th
    if lgq is None:
        lgq = lq_P(BETA_Q + dbeta,
                   None if not dleague else {**LD23_BR, **dleague})
    lv = C.LGV6 if dv6 == 0.0 else C.LGV6 + dv6 * (PC_P - PN_P)
    zin = ((1 - t["w_ebm"]) * (t["w_v6_inner"] * lv + (1 - t["w_v6_inner"]) * lgq)
           + t["w_ebm"] * C.PEBM + t["beta_ins"] * C.OFFR)
    if roundtrip:                       # S21 — 배포 CSV 왕복 (게이트 전용)
        p = expit(zin + D_INNER)
        b = io.StringIO()
        pd.DataFrame({"control_success": p}).to_csv(b, index=False)
        b.seek(0)
        lgp = lg(pd.read_csv(b)["control_success"].to_numpy(np.float64))
    else:
        lgp = zin + D_INNER
    cvec = np.array([t["c_aa"], t["c_S2"], t["c_S1"], t["c_T70"]])
    z = t["w_out"] * lgp + cvec @ C.ARMS18
    return (z + t["off_BI"] * DIR_P["BI"] + t["off_BJ"] * DIR_P["BJ"]
            + t["off_BK"] * DIR_P["BK"])


# ── P 게이트: ≤2024 배포 표(=arms/lg_p_q)를 넣으면 core078 의 p_BO 재현 ──
_zp24 = zp_P(lgq=C.LGQ, roundtrip=True)
_f24 = f_of(_zp24)
_p24 = expit(_f24 + anchor(_f24, T_REF))
GATES["P1_reproduce_p_BO_max_abs"] = float(np.abs(_p24 - C.p_BO).max())
GATES["P1_PASS"] = bool(GATES["P1_reproduce_p_BO_max_abs"] < 1e-12)
GATES["P2_lgq23_matches_BR_dump"] = {
    "max_abs": float(np.abs(lq_P() - lg(P_OFF23)).max()),
    "bitwise_equal": bool(np.array_equal(lq_P(), lg(P_OFF23))),
    "d_m_bitwise_equal": bool(np.array_equal(d_P(), D_M23))}
GATES["P3_le2023_miss_rate"] = {"n_miss": int((~HIT_P).sum()),
                                "miss_frac": float((~HIT_P).mean()),
                                "PASS": bool(int((~HIT_P).sum()) == 54501)}

ZP_P = zp_P()
F_P = f_of(ZP_P)
D_P_ANC = anchor(F_P, T_REF)
P_P = expit(F_P + D_P_ANC)
W_P = P_P * (1.0 - P_P)
DFDZ_P = BO["alpha"] + 2.0 * BO["beta_skew"] * ZP_P     # ∂f/∂zp on P
DFDZ_E = BO["alpha"] + 2.0 * BO["beta_skew"] * ZP_E     # ∂f/∂zp on E

# ══════════════════════════════════════════════════════════════════════
# 3. 셀 — 경계는 출하 동결(BQ2 xdef_solver.npz), x = zp (S26·S29)
# ══════════════════════════════════════════════════════════════════════
XD = np.load(f"{H}/build_BPBQ2/work/xdef_solver.npz", allow_pickle=True)
BOUNDS = {"R": np.asarray(XD["bounds_R"], np.float64),
          "F": np.asarray(XD["bounds_F"], np.float64)}
_shift_bq2 = np.asarray(XD["shift"], np.float64)
GATES["C1_shift_equals_axis_offsets_on_P"] = float(np.abs(
    _shift_bq2 - (BO["off_BI"] * DIR_P["BI"] + BO["off_BJ"] * DIR_P["BJ"])).max())


def cell_index(x):
    idx = np.full(N, -1, np.int64)
    fin = np.isfinite(x)
    for gi, g in enumerate(GTS):
        m = (GT == g) & fin
        idx[m] = gi * NCELL + np.searchsorted(BOUNDS[g], x[m], side="right")
    return idx


CIDX_E = cell_index(ZP_E)
CIDX_P = cell_index(ZP_P)
assert (CIDX_E >= 0).all() and (CIDX_P >= 0).all()
CELL_NAMES = [f"cell_{GTS[c // NCELL]}{c % NCELL}" for c in range(16)]
GATES["C2_cell_counts"] = {"E": np.bincount(CIDX_E, minlength=16).tolist(),
                           "P": np.bincount(CIDX_P, minlength=16).tolist()}


def score_E(dzsum=None):
    """🔴 E 에서 나오는 유일한 정보 — 스칼라 1개(환산 점수).

    dzsum 은 logit_mix(zp) 에 더할 오프셋 벡터.  앵커는 매번 다시 푼다."""
    zp = ZP_E if dzsum is None else ZP_E + dzsum
    f = f_of(zp)
    p = expit(f + anchor(f, YB))
    return pts(float(np.mean((p - Y_E) ** 2)))


def delta_of(dz, face):
    """logit_mix 방향 dz 에 대한 ∂p/∂u (앵커 재해 포함)."""
    if face == "E":
        w, df = W_E, DFDZ_E * dz
    else:
        w, df = W_P, DFDZ_P * dz
    return w * (df - float(np.sum(w * df) / np.sum(w)))


LOAD_MIN = (time.time() - T0) / 60.0
