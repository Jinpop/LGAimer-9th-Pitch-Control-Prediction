"""exp_081 — 수확 캠페인 종단 시뮬레이션.  학습 0건 · 제출물 0건 · 슬롯 0.

절차는 실전(LB)과 동형이다.
  · 평가면 E 에서 나오는 정보는 **축당 스칼라 1개(환산 점수)** 뿐이다.
  · G_jj 는 **대리면 P 에서만** 계산한다.  직교화 계수도 P 에서만 정한다.
  · L_j = [g_j/K + u_j^2 * G_jj^P] / (2 u_j),  착륙 u*_j = L_j / G_jj^P.
  · 착륙점의 점수는 **E 에서 다시 실측**한다 (예측값으로 대체하지 않는다).

stage1 실축 15방향 · stage2 위약 200회(🔴 본론) · stage3 슬롯 m 민감도
stage4 g 10자리 반올림 + G^P ±25% 오차의 대가
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "6"

import json                                                      # noqa: E402
import sys                                                       # noqa: E402
import time                                                      # noqa: E402

import numpy as np                                               # noqa: E402
from scipy.special import expit                                  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import frames081 as F                                            # noqa: E402
import core078 as C                                              # noqa: E402

T0 = time.time()
N, K, BO = F.N, F.K, F.BO
OUTJ = f"{F.OUT}/out/metrics.json"
R = {"exp_id": "exp_081", "date": time.strftime("%Y-%m-%d %H:%M:%S"),
     "title": "수확 캠페인 종단 시뮬레이션 — 실제 라벨 위에서 절차를 끝까지 돌린다",
     "type": "시뮬레이션 (학습 0건 · 제출물 0건 · 슬롯 0)",
     "K_pts_per_brier": K, "ref_brier": F.REF, "n_rows": N,
     "threads": "OMP/OPENBLAS/MKL/NUMEXPR/VECLIB = 6",
     "preregistered_verdict": {
         "A": "위약 평균 >= 5.0 이고 P(총액>=5) >= 0.6 → 캠페인 입증",
         "B": "위약 평균 3.0~5.0 또는 P(>=5) 0.3~0.6 → 부분 입증",
         "C": "위약 평균 < 3.0 → 환율 이론이 틀렸다",
         "🔴_S8": "결과를 보고 문턱을 바꾸지 않는다"},
     "load_min": F.LOAD_MIN, "gates": F.GATES,
     "S14_range_check_E_arms": F.S14_E,
     "S14_range_check_P_arms": C.M["S14_range_check"],
     "S14_verdict": C.M["S14_verdict"]}
for _k in ("E1_simple_honest_base_no_qoffset",
           "E2_simple_honest_base_with_le2023_qoffset"):
    assert F.GATES[_k]["PASS"], f"게이트 실패 {_k}: {F.GATES[_k]}"
assert F.GATES["P1_PASS"], f"게이트 실패 P1: {F.GATES['P1_reproduce_p_BO_max_abs']}"
assert F.GATES["E3_honest_frame_miss_rate"]["PASS"]
assert F.GATES["P3_le2023_miss_rate"]["PASS"]
R["gates"]["LD23_E_vs_BR_max_abs"] = float(max(
    abs(F.LD23[c] - F.LD23_BR[c]) for c in F.GT_CELLS))
print(f"적재 {F.LOAD_MIN:.2f}분 · 게이트 통과 "
      f"(E1 {F.GATES['E1_simple_honest_base_no_qoffset']['abs_diff']:.1e} · "
      f"E2 {F.GATES['E2_simple_honest_base_with_le2023_qoffset']['abs_diff']:.1e} · "
      f"P1 {F.GATES['P1_reproduce_p_BO_max_abs']:.1e} · "
      f"LD23 {R['gates']['LD23_E_vs_BR_max_abs']:.1e})", flush=True)

FLOOR = K * F.BRIER_E / N
R["surfaces"] = {
    "E_evaluation_surface": {
        "what": ("정직 OOT 2024 · 253,507행 · 실제 라벨 · 팔 전부 ≤2023 학습 · "
                 "Q 오프셋 표 ≤2023 · 산술은 배포 BO 그대로"
                 "(alpha·beta_skew·축오프셋 포함)"),
        "base_pts": F.BASE_E_PTS, "base_brier": F.BRIER_E,
        "anchor_target": F.YB, "anchor_d": F.D_E_ANC,
        "🔴_앵커": ("E 의 앵커는 라벨 평균에 맞춘다(exp_078/079/080·BQ2 정직면 관례). "
                  "즉 상수 방향 {1} 은 이미 꼭지점에 있고, 그 방향은 매수 기저에 "
                  "들어 있어 새 축에서 제거된다 — 이 실험의 비교에는 영향이 없다."),
        "🔴": "E 에서 밖으로 나오는 것은 축당 점수 스칼라 1개뿐이다."},
    "P_proxy_surface": {
        "what": ("보정 tr2024 · 253,507행 · 라벨 안 씀 · 팔 배포(≤2024 학습) · "
                 "Q 오프셋 표만 ≤2023 재구성(build_BR 덤프, 미적중 21.4988%)"),
        "base_mean_pred": float(F.P_P.mean()), "anchor_target": F.T_REF,
        "🔴_S32": ("build_AC 대역 프레임을 그대로 쓰면 q_league 4축의 δ 가 정확히 0 "
                   "이라 축 4개가 조용히 죽는다.  그래서 P 의 Q 경로만 ≤2023 표로 "
                   "바꿨다 — 실제 배포(≤2024 표 → 2025 행)가 갖는 표–행 관계다."),
        "differs_from_E_in": ["팔 학습 구간 ≤2024 vs ≤2023", "앵커 목표 T_REF vs 라벨평균",
                              "v6·EBM·off·AA·T70 팔 벡터 전부"]},
    "noise_floor_per_dof_pts_on_E": FLOOR,
    "noise_floor_per_dof_pts_on_LB": 0.40231289196975534}
print(f"E base {F.BASE_E_PTS:.6f}점 (Brier {F.BRIER_E:.9f}) · 1자유도 잡음바닥 "
      f"{FLOOR:.6f}점 · P base mean {F.P_P.mean():.6f}", flush=True)

# ══════════════════════════════════════════════════════════════════════
# 방향 표현 — (dz, df, s):  zp += u*dz ;  f += u*df ;  p = expit(f+anc+u*s)
# ══════════════════════════════════════════════════════════════════════
Z = np.zeros(N)


class Dirs:
    def __init__(self):
        self.names, self._dz, self._df, self._s = [], [], [], []

    def add(self, name, dz=None, df=None, s=0.0):
        self.names.append(name)
        self._dz.append(Z if dz is None else np.asarray(dz, np.float64))
        self._df.append(Z if df is None else np.asarray(df, np.float64))
        self._s.append(float(s))

    def freeze(self):
        self.DZ = np.stack(self._dz)
        self.DF = np.stack(self._df)
        self.S = np.array(self._s, np.float64)
        self._dz = self._df = self._s = None
        return self


def deltas(DZ, DF, S, face):
    """(dz, df, s) 묶음 -> δ 행렬 (m x N).  앵커 재해 포함."""
    w = F.W_E if face == "E" else F.W_P
    dfz = F.DFDZ_E if face == "E" else F.DFDZ_P
    D = dfz * DZ if DF is None else dfz * DZ + DF
    D = D - (D @ w / w.sum())[:, None]
    return w * (D + np.asarray(S, np.float64)[:, None])


def rownorm2(D):
    return np.einsum("ij,ij->i", D, D) / N


def eval_E(dz, df, s):
    """🔴 E 위의 평가 — 점수 스칼라 1개만 돌려준다."""
    zp = F.ZP_E + dz
    f = F.f_of(zp) + df
    d = F.anchor(f, F.YB, d0=F.D_E_ANC)
    p = expit(f + d + s)
    return F.pts(float(np.mean((p - F.Y_E) ** 2)))


# ══════════════════════════════════════════════════════════════════════
# 매수 기저 13개 — exp_078 의 15축 중 정직면에서도 정의되는 것 전부.
#   {1, x, x^2} = d_mean · alpha · beta_skew 가 그 안에 명시적으로 있다.
# ══════════════════════════════════════════════════════════════════════
def bought_dirs(face):
    B = Dirs()
    if face == "E":
        zp, lv, lq, le, off = F.ZP_E, F.LV_E, F.lq_E(), F.LE_E, F.VE["off"]
        laa, lt70, dirs = F.LAA_E, F.LT70_E, F.DIR_E
    else:
        zp, lv, lq, le, off = F.ZP_P, C.LGV6, F.lq_P(), C.PEBM, C.OFFR
        laa, lt70, dirs = C.ARMS18[0], C.ARMS18[3], F.DIR_P
    wo, we, wv, bi = BO["w_out"], BO["w_ebm"], BO["w_v6_inner"], BO["beta_ins"]
    base = wv * lv + (1 - wv) * lq
    B.add("d_mean", s=1.0)                                  # {1}
    B.add("alpha", df=zp)                                   # {x}
    B.add("beta_skew", df=zp * zp)                          # {x^2}
    B.add("c_aa", dz=laa)
    B.add("c_T70", dz=lt70)
    B.add("w_out", dz=(1 - we) * base + we * le + bi * off + F.D_INNER)
    B.add("w_v6_inner", dz=wo * (1 - we) * (lv - lq))
    B.add("w_ebm", dz=wo * (le - base))
    B.add("beta_ins", dz=wo * off)
    B.add("off_BI", dz=dirs["BI"])
    B.add("off_BJ", dz=dirs["BJ"])
    B.add("off_BK", dz=dirs["BK"])
    B.add("seg_spread_BH", df=C.CON * zp)
    return B.freeze()


BE, BP = bought_dirs("E"), bought_dirs("P")
NB = len(BE.names)
R["bought_basis"] = {
    "names": BE.names, "n": NB,
    "note": ("exp_078 의 15축 중 c_S2·c_S1 만 뺐다(정직면에 그 팔 벡터가 없다; "
             "배포 계수도 0 이고 c 블록끼리 ρ 0.989~0.992 라 c_aa 가 대표한다). "
             "{1, x, x^2} = d_mean · alpha · beta_skew 로 명시적으로 들어 있다."),
    "🔴_S33": ("정직면 기준점이 전역 척도의 최적이 아니라는 것이 이 기저가 "
               "필요한 이유다 — x 방향을 안 빼면 아무 방향이나 그 척도분을 긁어온다.")}

# ══════════════════════════════════════════════════════════════════════
# 새 15축 — q_beta · q_league x4 · cellres_u1 · 셀 8 · v6_CAT_vs_NN
# ══════════════════════════════════════════════════════════════════════
E80 = json.load(open(f"{F.H}/exp080/out/metrics.json", encoding="utf-8"))
U1W = np.array(E80["stage3_cell_info_allocation"]["u1_weights"], np.float64)
CELL8 = ["cell_R2", "cell_F5", "cell_F4", "cell_R5",
         "cell_R1", "cell_R4", "cell_F6", "cell_R6"]
LGQW = BO["w_out"] * (1 - BO["w_ebm"]) * (1 - BO["w_v6_inner"])
V6W = BO["w_out"] * (1 - BO["w_ebm"]) * BO["w_v6_inner"]


def new_dirs(face):
    A = Dirs()
    if face == "E":
        dvec = F.d_E()
        r = F.VE["p_q"] + F.BETA_Q * dvec
        hit, kl, cidx = F.HIT_E, F.KL_E, F.CIDX_E
        v6d = F.LCAT_E - F.LNN_E
    else:
        dvec = F.d_P()
        r = F.P_PRE + F.BETA_Q * dvec
        hit, kl, cidx = F.HIT_P, F.KL_P, F.CIDX_P
        v6d = F.PC_P - F.PN_P
    live = ((r > F.EPS) & (r < 1 - F.EPS)).astype(np.float64)
    inv = live / np.clip(r * (1 - r), 1e-12, None)
    A.add("q_beta", dz=LGQW * dvec * inv)
    for cc in F.GT_CELLS:
        ind = ((~hit) & (kl == cc)).astype(np.float64)
        A.add(f"q_league_{cc.replace('|', '')}", dz=LGQW * F.BETA_Q * ind * inv)
    A.add("cellres_u1", dz=U1W[cidx])
    for nm in CELL8:
        A.add(nm, dz=(cidx == F.CELL_NAMES.index(nm)).astype(np.float64))
    A.add("v6_CAT_vs_NN", dz=V6W * v6d)
    return A.freeze()


AE, AP = new_dirs("E"), new_dirs("P")
AXN = list(AE.names)
NA = len(AXN)
assert NA == 15, AXN

# ── 게이트: 해석식 dz 가 파라미터 수치미분과 같은가 (양면) ────────────
GN = {}
for face in ("E", "P"):
    zpf = F.zp_E if face == "E" else F.zp_P
    A = AE if face == "E" else AP
    bld = F.LD23 if face == "E" else F.LD23_BR
    h = 0.002
    cases = {"q_beta": ({"dbeta": h}, {"dbeta": -h}),
             "q_league_21": ({"dleague": {"2|1": bld["2|1"] + h}},
                             {"dleague": {"2|1": bld["2|1"] - h}}),
             "v6_CAT_vs_NN": ({"dv6": h}, {"dv6": -h})}
    for nm, (kp, km) in cases.items():
        num = (zpf(**kp) - zpf(**km)) / (2 * h)
        ana = A.DZ[AXN.index(nm)]
        GN[f"{face}/{nm}"] = {
            "rel_rms_diff": float(np.sqrt(np.mean((num - ana) ** 2))
                                  / max(np.sqrt(np.mean(ana ** 2)), 1e-300)),
            "corr": float(np.corrcoef(num, ana)[0, 1])}
R["gates"]["G_analytic_vs_numeric_dz"] = GN
for k, v in GN.items():
    assert v["rel_rms_diff"] < 5e-3, (k, v)
print("해석 dz vs 수치미분: "
      + "  ".join(f"{k} {v['rel_rms_diff']:.1e}" for k, v in GN.items()), flush=True)

# ══════════════════════════════════════════════════════════════════════
# 직교화 — 🔴 계수는 P 에서만 정한다.  자기 계수는 1 로 고정(u 단위 보존).
# ══════════════════════════════════════════════════════════════════════
DB_P = deltas(BP.DZ, BP.DF, BP.S, "P")
GBB = DB_P @ DB_P.T / N
# 🔴 S19 — 매수 기저는 사실상 저차원이다(exp_078 조건수 7.96e+14).  역행렬을
#    풀면 안 되고, 고윳값 절단으로 **span 위로의 직교사영**만 쓴다.
_ev, _evec = np.linalg.eigh(GBB)
_keep = _ev > _ev.max() * 1e-10
GBB_PINV = (_evec[:, _keep] / _ev[_keep]) @ _evec[:, _keep].T
R["bought_basis"]["gram_condition_number"] = float(_ev.max() / max(_ev.min(), 1e-300))
R["bought_basis"]["numeric_rank_rcond_1e-10"] = int(_keep.sum())
R["bought_basis"]["eigenvalues"] = _ev.tolist()


def orth(DC_P):
    """후보 δ(P) 를 매수 기저 + 앞선 후보에 순차 직교화.  (Tb, Tc, Gdiag)."""
    m = DC_P.shape[0]
    GBC = DB_P @ DC_P.T / N
    GCC = DC_P @ DC_P.T / N
    Tb, Tc, Gd = np.zeros((m, NB)), np.eye(m), np.zeros(m)
    cb = GBB_PINV @ GBC
    for j in range(m):
        tb = -cb[:, j].copy()
        tc = np.zeros(m)
        tc[j] = 1.0
        for i in range(j):
            if not np.isfinite(Gd[i]) or Gd[i] <= GCC[j, j] * 1e-12:
                continue
            num = float(tb @ (GBB @ Tb[i] + GBC @ Tc[i])
                        + tc @ (GBC.T @ Tb[i] + GCC @ Tc[i]))
            tb -= num / Gd[i] * Tb[i]
            tc -= num / Gd[i] * Tc[i]
        Tb[j], Tc[j] = tb, tc
        Gd[j] = float(tb @ (GBB @ tb + GBC @ tc) + tc @ (GBC.T @ tb + GCC @ tc))
    return Tb, Tc, Gd


def apply_T(Tb, Tc, BD, CDZ, CDF=None, CDS=None):
    dz = Tb @ BD.DZ + Tc @ CDZ
    df = Tb @ BD.DF if CDF is None else Tb @ BD.DF + Tc @ CDF
    s = Tb @ BD.S if CDS is None else Tb @ BD.S + Tc @ CDS
    return dz, df, s


def probe(GdP, dirE, kq=1.0):
    """🔴 실전 동형 — 축당 E 에서 스칼라 1개(점수)만 읽는다.

    kq 는 프로브 진폭 관례: K * u^2 * G_jj^P = kq 점.  kq=1 은 exp_080·S37 이
    쓴 특성화 진폭, kq=0.04 는 그 0.2배(S22 의 '작은 프로브')다."""
    dzE, dfE, sE = dirE
    up = np.sqrt(kq / (K * GdP))
    g = np.array([eval_E(up[j] * dzE[j], up[j] * dfE[j], up[j] * sE[j])
                  - F.BASE_E_PTS for j in range(len(GdP))])
    return {"u_probe": up, "g_probe": g, "dzE": dzE, "dfE": dfE, "sE": sE}


def solve_and_land(pk, Gp, subsets, ground=None):
    up = pk["u_probe"]
    g = pk["g_probe"] if ground is None else np.round(pk["g_probe"], ground)
    L = (g / K + up * up * Gp) / (2.0 * up)
    us = L / Gp
    out = {"L_hat": L, "u_star": us, "pred_gain": K * L * L / Gp, "measured": {}}
    for tag, idx in subsets.items():
        out["measured"][tag] = eval_E(us[idx] @ pk["dzE"][idx],
                                      us[idx] @ pk["dfE"][idx],
                                      float(us[idx] @ pk["sE"][idx])) - F.BASE_E_PTS
    return out


def offdiag(G):
    d = np.sqrt(np.diag(G))
    Cm = G / np.outer(d, d)
    return float(np.abs(Cm - np.eye(len(G))).max()), Cm


# ══════════════════════════════════════════════════════════════════════
# stage1 — 실축 15방향
# ══════════════════════════════════════════════════════════════════════
T1 = time.time()
DC_P = deltas(AP.DZ, AP.DF, AP.S, "P")
Tb, Tc, GdP = orth(DC_P)
dzE, dfE, sE = apply_T(Tb, Tc, BE, AE.DZ, AE.DF, AE.S)
DO_E = deltas(dzE, dfE, sE, "E")
DO_P = deltas(*apply_T(Tb, Tc, BP, AP.DZ, AP.DF, AP.S), "P")
GP_full, GE_full = DO_P @ DO_P.T / N, DO_E @ DO_E.T / N
_gchk = float(np.abs(np.diag(GP_full) - GdP).max() / max(np.abs(GdP).max(), 1e-300))
R["gates"]["orth_coeff_vs_materialised_G_rel"] = _gchk
assert _gchk < 1e-6, f"직교화 계수와 실체화 δ 의 G 대각이 어긋난다 {_gchk:.3e}"
odP, CP = offdiag(GP_full)
odE, CE = offdiag(GE_full)
GdE = np.diag(GE_full).copy()

DIR1 = (dzE, dfE, sE)
KQS = [1.0, 0.04]
SUB1 = {"all15": np.arange(NA)}
SUB1.update({f"solo_{AXN[j]}": np.array([j]) for j in range(NA)})
PK1 = {kq: probe(GdP, DIR1, kq=kq) for kq in KQS}
SOL1 = {kq: solve_and_land(PK1[kq], GdP, SUB1) for kq in KQS}
pk1, solP = PK1[1.0], SOL1[1.0]
solE = solve_and_land(pk1, GdE, {"all15": np.arange(NA)})
L_true = DO_E @ (F.Y_E - F.P_E) / N          # 🔴 감사 전용 — 절차 밖

rows = []
for j, nm in enumerate(AXN):
    rows.append({
        "axis": nm, "u_probe": float(pk1["u_probe"][j]),
        "g_probe_pts": float(pk1["g_probe"][j]),
        "L_hat": float(solP["L_hat"][j]), "u_star": float(solP["u_star"][j]),
        "predicted_gain_pts": float(solP["pred_gain"][j]),
        "measured_gain_pts_on_E": float(solP["measured"][f"solo_{nm}"]),
        "K_times_G_jj_P_at_u1": float(K * GdP[j]),
        "K_times_G_jj_E_at_u1": float(K * GdE[j]),
        "G_P_over_G_E": float(GdP[j] / GdE[j]),
        "AUDIT_ONLY_L_true_on_E": float(L_true[j]),
        "AUDIT_ONLY_L_hat_over_L_true": float(solP["L_hat"][j] / L_true[j])
        if L_true[j] != 0 else None,
        "orth_norm_ratio_vs_raw_P": float(np.sqrt(
            GdP[j] / float(DC_P[j] @ DC_P[j] / N))),
        "max_abs_corr_with_others_on_P": float(np.abs(np.delete(CP[j], j)).max()),
        "max_abs_corr_with_others_on_E": float(np.abs(np.delete(CE[j], j)).max()),
        "small_probe_kq0.04": {
            "u_probe": float(SOL1[0.04]["u_star"][j] * 0.0
                             + PK1[0.04]["u_probe"][j]),
            "g_probe_pts": float(PK1[0.04]["g_probe"][j]),
            "L_hat": float(SOL1[0.04]["L_hat"][j]),
            "u_star": float(SOL1[0.04]["u_star"][j]),
            "predicted_gain_pts": float(SOL1[0.04]["pred_gain"][j]),
            "measured_gain_pts_on_E": float(SOL1[0.04]["measured"][f"solo_{nm}"])}})
    print(f"  [{nm:<14}] u_p {rows[-1]['u_probe']:+.5g}  g {rows[-1]['g_probe_pts']:+.6f}"
          f"  L {rows[-1]['L_hat']:+.4e}  u* {rows[-1]['u_star']:+.5g}"
          f"  G^P/G^E {rows[-1]['G_P_over_G_E']:.3f}"
          f"  예측 {rows[-1]['predicted_gain_pts']:+.5f}  실측 "
          f"{rows[-1]['measured_gain_pts_on_E']:+.5f}"
          f"  |작은프로브 실측 "
          f"{rows[-1]['small_probe_kq0.04']['measured_gain_pts_on_E']:+.5f}", flush=True)

R["stage1"] = {
    "axes": rows,
    "probe_convention": "K * u_probe^2 * G_jj^P = 1점 (exp_080·S37 관례)",
    "orthogonalisation": {
        "metric": "P 의 Gram (라벨 안 씀)",
        "projected_out": "매수 기저 13개 = exp_078 15축(−c_S2·c_S1), {1, x, x^2} 포함",
        "max_abs_offdiag_corr_on_P": odP, "max_abs_offdiag_corr_on_E": odE,
        "🔴": ("직교화 계수를 P 에서 정했으므로 E 에서는 완전 직교가 아니다 — "
               "두 값의 차가 대리면을 쓴 대가의 일부다.")},
    "simultaneous_landing": {
        "predicted_total_pts": float(solP["pred_gain"].sum()),
        "measured_total_pts_on_E": float(solP["measured"]["all15"]),
        "sum_of_solo_measured_pts": float(sum(r["measured_gain_pts_on_E"]
                                              for r in rows)),
        "realisation_ratio": float(solP["measured"]["all15"]
                                   / max(solP["pred_gain"].sum(), 1e-30))},
    "cost_of_using_G_from_P": {
        "predicted_total_with_G_from_E_pts": float(solE["pred_gain"].sum()),
        "measured_total_with_G_from_E_pts": float(solE["measured"]["all15"]),
        "measured_P_minus_E_pts": float(solP["measured"]["all15"]
                                        - solE["measured"]["all15"]),
        "G_P_over_G_E_min": float((GdP / GdE).min()),
        "G_P_over_G_E_max": float((GdP / GdE).max()),
        "G_P_over_G_E_median": float(np.median(GdP / GdE))},
    "noise_floor_comparison": {
        "per_dof_pts": FLOOR, "x15_pts": 15 * FLOOR,
        "measured_over_floor": float(solP["measured"]["all15"] / (15 * FLOOR))},
    "probe_amplitude_sensitivity": {
        "kq_1.0": {"predicted_total_pts": float(SOL1[1.0]["pred_gain"].sum()),
                   "measured_total_pts": float(SOL1[1.0]["measured"]["all15"]),
                   "n_axes_with_negative_measured_solo": int(sum(
                       1 for a in AXN if SOL1[1.0]["measured"][f"solo_{a}"] < 0))},
        "kq_0.04": {"predicted_total_pts": float(SOL1[0.04]["pred_gain"].sum()),
                    "measured_total_pts": float(SOL1[0.04]["measured"]["all15"]),
                    "n_axes_with_negative_measured_solo": int(sum(
                        1 for a in AXN if SOL1[0.04]["measured"][f"solo_{a}"] < 0))},
        "🔴": ("L_hat = L^E + u_probe*(G^P - G^E)/2 이므로 **G^P 오차가 프로브 "
               "진폭에 비례해 L 로 들어온다.**  큰 진폭은 정보를 더 주지 않으면서"
               "(S22) 편의만 키운다 — 이 실험이 그것을 실측한다.")},
    "🔴_선택_표본내": (
        "stage1 의 15축은 **정직면(=E) 정보로 골라진 축**이다 — exp_080 이 정직면 "
        "이득 순으로 정렬했고 cellres_u1 은 정직면 잔차에서 적합됐다.  따라서 "
        "stage1 의 실측 총액은 선택에 대해 표본 내이고 상한이지 LB 예보가 아니다. "
        "기계가 옳은지는 stage2 위약이 판정한다.")}
print(f"\n[stage1] 예측 총액 {solP['pred_gain'].sum():+.5f}점 · 실측 총액 "
      f"{solP['measured']['all15']:+.5f}점 · 잡음바닥×15 {15*FLOOR:.5f}점 "
      f"· 비대각 최대 |ρ| P {odP:.4f} / E {odE:.4f} ({time.time()-T1:.0f}s)", flush=True)

# 파라미터 비선형성 — 원(직교화 전) 축을 정확 파라미터 지도로 다시 평가
NL = {}
for nm, mk in (("q_beta", lambda u: {"dbeta": u}),
               ("q_league_21", lambda u: {"dleague": {"2|1": F.LD23["2|1"] + u}}),
               ("v6_CAT_vs_NN", lambda u: {"dv6": u})):
    j = AXN.index(nm)
    graw = float(DC_P[j] @ DC_P[j] / N)
    u = float(np.sqrt(1.0 / (K * graw)))
    zp = F.zp_E(**mk(u))
    f = F.f_of(zp)
    exact = F.pts(float(np.mean(
        (expit(f + F.anchor(f, F.YB, d0=F.D_E_ANC)) - F.Y_E) ** 2))) - F.BASE_E_PTS
    lin = eval_E(u * AE.DZ[j], 0.0, 0.0) - F.BASE_E_PTS
    NL[nm] = {"probe_u_raw_axis": u, "exact_parametric_g_pts": exact,
              "logit_linear_g_pts": lin, "abs_diff_pts": abs(exact - lin)}
R["stage1_parametric_nonlinearity"] = {
    "cases": NL,
    "note": ("직교화 전 원축을 K·G=1 프로브 진폭에서 정확 파라미터 지도와 로짓선형 "
             "근사로 각각 평가한 차이.  이 실험의 평가 모형이 로짓선형이라 그 "
             "근사가 얼마나 비싼지를 보여준다.")}
print("  파라미터 비선형성: "
      + "  ".join(f"{k} {v['abs_diff_pts']:.2e}점" for k, v in NL.items()), flush=True)
json.dump(R, open(OUTJ, "w"), indent=1, ensure_ascii=False, default=float)

# ══════════════════════════════════════════════════════════════════════
# stage2 — 위약 대조 (🔴 본론)
# ══════════════════════════════════════════════════════════════════════
T2 = time.time()
NRAND = 20
REPS = int(os.environ.get("EXP081_REPS", "200"))
MSET = [1, 5, 10, 13, 15, 20]
SUBM = {f"m{m}": np.arange(m) for m in MSET}
ZDS = np.zeros(NRAND)
SEED = 20260822


def placebo_once(kind, rng):
    raw = (rng.standard_normal((NRAND, N)) if kind == "gauss"
           else (rng.random((NRAND, N)) < 1.0 / 16.0).astype(np.float64))
    dcp = deltas(raw, None, ZDS, "P")
    tb, tc, gd = orth(dcp)
    del dcp
    dirE = apply_T(tb, tc, BE, raw, None, None)
    del raw
    return dirE, gd


def stat(v):
    v = np.asarray(v, np.float64)
    return {"n": int(len(v)), "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
            "q05": float(np.quantile(v, 0.05)), "q50": float(np.quantile(v, 0.50)),
            "q95": float(np.quantile(v, 0.95)), "P_ge_5": float((v >= 5.0).mean()),
            "min": float(v.min()), "max": float(v.max())}


ST2 = {}
CONFIGS = [("gauss", 1.0, REPS), ("gauss", 0.04, REPS),
           ("cellish", 1.0, max(50, REPS // 4))]
for kind, kq, nrep in CONFIGS:
    tag = f"{kind}_kq{kq:g}"
    rng = np.random.default_rng(SEED)      # 🔴 설정마다 같은 난수 — 짝지은 비교
    tot = {m: [] for m in MSET}
    pre = {m: [] for m in MSET}
    perax, ratio = [], []
    v4 = {"g_round10": [], "G075": [], "G125": [], "both": []}
    do4 = (kind == "gauss" and kq == 1.0)
    for it in range(nrep):
        dirE, gd = placebo_once(kind, rng)
        pk = probe(gd, dirE, kq=kq)
        gdE = rownorm2(deltas(pk["dzE"], pk["dfE"], pk["sE"], "E"))
        ratio.append(gd / gdE)
        so = solve_and_land(pk, gd, SUBM)
        for m in MSET:
            tot[m].append(so["measured"][f"m{m}"])
            pre[m].append(float(so["pred_gain"][:m].sum()))
        perax.append(so["pred_gain"].copy())
        if do4:
            i15 = {"m15": np.arange(15)}
            v4["g_round10"].append(
                solve_and_land(pk, gd, i15, ground=10)["measured"]["m15"])
            v4["G075"].append(solve_and_land(pk, 0.75 * gd, i15)["measured"]["m15"])
            v4["G125"].append(solve_and_land(pk, 1.25 * gd, i15)["measured"]["m15"])
            v4["both"].append(solve_and_land(pk, 0.75 * gd, i15,
                                             ground=10)["measured"]["m15"])
        del pk, dirE
        if (it + 1) % 25 == 0:
            print(f"  [{tag}] {it+1}/{nrep}회 · m15 평균 {np.mean(tot[15]):+.4f}점 "
                  f"· {time.time()-T2:.0f}s", flush=True)
    ST2[tag] = {
        "kind": kind, "probe_kq": kq, "n_reps": nrep, "n_directions_drawn": NRAND,
        "realised_total_by_m": {f"m{m}": stat(tot[m]) for m in MSET},
        "predicted_total_by_m": {f"m{m}": stat(pre[m]) for m in MSET},
        "theory_mean_by_m": {f"m{m}": m * FLOOR for m in MSET},
        "mean_minus_theory_m15": float(np.mean(tot[15]) - 15 * FLOOR),
        "realised_per_direction_m1_mean_pts": float(np.mean(tot[1])),
        "frac_reps_with_negative_m1": float(np.mean(np.asarray(tot[1]) < 0)),
        "per_direction_mean_pred_gain_pts": float(np.mean(np.array(perax))),
        "G_P_over_G_E": {"mean": float(np.mean(ratio)), "min": float(np.min(ratio)),
                         "max": float(np.max(ratio))}}
    if do4:
        ST2[tag]["stage4_variants_m15"] = {k: stat(v) for k, v in v4.items()}
    st = ST2[tag]["realised_total_by_m"]["m15"]
    print(f"[stage2/{tag}] m15 평균 {st['mean']:+.5f} · sd {st['sd']:.5f} · "
          f"P(>=5) {st['P_ge_5']:.3f} · 이론 {15*FLOOR:.5f} · m1 평균 "
          f"{ST2[tag]['realised_per_direction_m1_mean_pts']:+.5f} · "
          f"{(time.time()-T2)/60:.2f}분", flush=True)
    json.dump({**R, "stage2": ST2}, open(OUTJ, "w"), indent=1,
              ensure_ascii=False, default=float)

HEAD = "gauss_kq1"
R["stage2"] = ST2
R["stage2"]["seed"] = SEED
R["stage2"]["headline_config"] = HEAD
R["stage2"]["draw_note"] = (
    "gauss = 표준정규 로짓방향(팔 가중·연속 축과 같은 모양). "
    "cellish = 무작위 1/16 지시자(셀 축과 같은 모양). 둘 다 같은 절차를 밟는다. "
    "두 경우 모두 매수 기저 13개({1,x,x^2} 포함)와 앞선 방향에 직교화했다. "
    "kq 는 프로브 진폭 관례(K·u²·G^P = kq 점); 세 설정 모두 같은 난수열을 쓴다. "
    "🔴 판정은 사전 등록대로 gauss_kq1 로 한다 — config 가 참조한 exp_080·S37 관례다.")
R["stage2"]["probe_amplitude_comparison"] = {
    "kq_1.0_m15_mean": ST2[HEAD]["realised_total_by_m"]["m15"]["mean"],
    "kq_0.04_m15_mean": ST2["gauss_kq0.04"]["realised_total_by_m"]["m15"]["mean"],
    "kq_1.0_frac_reps_single_direction_negative":
        ST2[HEAD]["frac_reps_with_negative_m1"],
    "kq_0.04_frac_reps_single_direction_negative":
        ST2["gauss_kq0.04"]["frac_reps_with_negative_m1"],
    "🔴": ("L_hat = L^E + u_probe*(G^P − G^E)/2.  프로브를 키우면 정보는 안 늘고"
           "(S22) G^P 오차만 L 로 들어온다.  두 설정은 같은 난수열이라 차이가 "
           "전적으로 진폭 때문이다.")}
R["stage2"]["🔴_왜_본론인가"] = (
    "위약 방향에는 설계상 신호가 없다.  그래도 평균이 이론값 근처에서 나온다면 "
    "'아무 방향이나 m 개 사면 m×바닥이 나온다'가 실측으로 입증된 것이다.")

# ══════════════════════════════════════════════════════════════════════
# stage3 — 슬롯 민감도
# ══════════════════════════════════════════════════════════════════════
R["stage3_slot_sensitivity"] = {
    "table": [{"m": m,
               "mean_pts": ST2[HEAD]["realised_total_by_m"][f"m{m}"]["mean"],
               "sd_pts": ST2[HEAD]["realised_total_by_m"][f"m{m}"]["sd"],
               "P_ge_5": ST2[HEAD]["realised_total_by_m"][f"m{m}"]["P_ge_5"],
               "q05_pts": ST2[HEAD]["realised_total_by_m"][f"m{m}"]["q05"],
               "q95_pts": ST2[HEAD]["realised_total_by_m"][f"m{m}"]["q95"],
               "theory_mean_pts": m * FLOOR,
               "cellish_mean_pts":
                   ST2["cellish_kq1"]["realised_total_by_m"][f"m{m}"]["mean"],
               "cellish_P_ge_5":
                   ST2["cellish_kq1"]["realised_total_by_m"][f"m{m}"]["P_ge_5"],
               "small_probe_kq0.04_mean_pts":
                   ST2["gauss_kq0.04"]["realised_total_by_m"][f"m{m}"]["mean"],
               "small_probe_kq0.04_P_ge_5":
                   ST2["gauss_kq0.04"]["realised_total_by_m"][f"m{m}"]["P_ge_5"]}
              for m in MSET],
    "config_requested_rows": [5, 10, 13, 15, 20],
    "m1_row_note": "m=1 은 config 밖 추가 행 — 방향 1개의 실현 이득 분포를 보려는 것",
    "note": ("같은 200회의 중첩 부분집합(앞 m 개)에서 읽었다 — 직교화가 순차라 "
             "앞 m 개는 m 방향만 샀을 때와 정확히 같다."),
    "🔴_S22": "꼭지점 이득은 L 의 부호와 무관하게 >= 0 이므로 하방은 0 이 아니라 바닥이다."}

# ══════════════════════════════════════════════════════════════════════
# stage4 — 정밀도·G 오차의 대가
# ══════════════════════════════════════════════════════════════════════
IA = {"a": np.arange(NA)}
s4 = {"stage1_15axes": {
    "baseline_pts": float(solP["measured"]["all15"]),
    "g_rounded_10dp_pts": float(solve_and_land(pk1, GdP, IA,
                                               ground=10)["measured"]["a"]),
    "G_P_x0.75_pts": float(solve_and_land(pk1, 0.75 * GdP, IA)["measured"]["a"]),
    "G_P_x1.25_pts": float(solve_and_land(pk1, 1.25 * GdP, IA)["measured"]["a"]),
    "both_pts": float(solve_and_land(pk1, 0.75 * GdP, IA,
                                     ground=10)["measured"]["a"])}}
b = s4["stage1_15axes"]["baseline_pts"]
for k in ("g_rounded_10dp", "G_P_x0.75", "G_P_x1.25", "both"):
    s4["stage1_15axes"][k + "_loss_pts"] = b - s4["stage1_15axes"][k + "_pts"]
s4["stage1_15axes"]["sum_of_two_single_losses_pts"] = (
    s4["stage1_15axes"]["g_rounded_10dp_loss_pts"]
    + s4["stage1_15axes"]["G_P_x0.75_loss_pts"])
pl = ST2[HEAD]["stage4_variants_m15"]
pb = ST2[HEAD]["realised_total_by_m"]["m15"]["mean"]
s4["stage2_placebo_m15_mean"] = {
    "baseline_pts": pb,
    "g_rounded_10dp_loss_pts": pb - pl["g_round10"]["mean"],
    "G_P_x0.75_loss_pts": pb - pl["G075"]["mean"],
    "G_P_x1.25_loss_pts": pb - pl["G125"]["mean"],
    "both_loss_pts": pb - pl["both"]["mean"],
    "sum_of_two_single_losses_pts": (2 * pb - pl["g_round10"]["mean"]
                                     - pl["G075"]["mean"])}
s4["analytic"] = {
    "formula": "G^P = c·G 로 풀면 이득 배율 (2c−1)/c^2 · 손실률 (c−1)^2/c^2",
    "loss_frac_c_0.75": (0.25 / 0.75) ** 2, "loss_frac_c_1.25": (0.25 / 1.25) ** 2,
    "config_quoted": 0.0625,
    "🔴": ("config 가 적은 (1−0.75)^2 = 6.25% 는 어림이다. 정확식은 (c−1)^2/c^2 라 "
           "c=0.75 이면 11.11%, c=1.25 이면 4.00% 로 **비대칭**이다 — G 를 낮게 "
           "잡는 쪽이 높게 잡는 쪽보다 2.78배 비싸다.")}
s4["lb_precision"] = {
    "lb_score_decimals": 10, "g_quantisation_pts": 5e-11,
    "implied_L_quantisation": float(5e-11 / (2 * K * float(pk1["u_probe"].mean()))),
    "verdict": "LB 정밀도는 제약이 아니다"}
R["stage4"] = s4
print(f"[stage4] 위약 m15 — 반올림 손실 "
      f"{s4['stage2_placebo_m15_mean']['g_rounded_10dp_loss_pts']:+.3e}점 · "
      f"G×0.75 {s4['stage2_placebo_m15_mean']['G_P_x0.75_loss_pts']:+.5f}점 · "
      f"G×1.25 {s4['stage2_placebo_m15_mean']['G_P_x1.25_loss_pts']:+.5f}점", flush=True)

# ══════════════════════════════════════════════════════════════════════
mu = ST2[HEAD]["realised_total_by_m"]["m15"]["mean"]
pg = ST2[HEAD]["realised_total_by_m"]["m15"]["P_ge_5"]
V = "A" if (mu >= 5.0 and pg >= 0.6) else ("C" if mu < 3.0 else "B")
R["verdict"] = {
    "grade": V, "placebo_mean_m15_pts": mu, "P_ge_5_m15": pg,
    "theory_mean_m15_pts": 15 * FLOOR,
    "stage1_real15_measured_pts": float(solP["measured"]["all15"]),
    "stage1_over_placebo_sigma": float(
        (solP["measured"]["all15"] - mu)
        / ST2[HEAD]["realised_total_by_m"]["m15"]["sd"])}
R["wall_min"] = (time.time() - T0) / 60.0
json.dump(R, open(OUTJ, "w"), indent=1, ensure_ascii=False, default=float)
print(f"\n판정 {V} · 위약 평균 {mu:+.5f}점 · P(>=5) {pg:.3f} · 이론 {15*FLOOR:.5f}점")
print(f"stage1 실축 15방향 실측 {solP['measured']['all15']:+.5f}점 "
      f"(위약 대비 {R['verdict']['stage1_over_placebo_sigma']:+.2f}σ)")
print(f"기록 {OUTJ} · {R['wall_min']:.2f}분")
print("EXP081_OK")
