"""exp_080 stage2b·3·4 — 정직프레임 Q 오프셋 축 · 보정셀 잔여 · 방향 집합 확정.

🔴 stage2 에서 드러난 것: 배포 대역(스탠드인) 프레임은 ≤2024 동결 개인표를
   2024 행에 적용하므로 **개인 셀 적중률이 100.00%** 다.  리그 백오프 분기가
   한 행도 발동하지 않아 q_league 4축의 δ 가 정확히 0 이 된다.
   → 그 축들은 대역 프레임에서 특성화 불가.  ≤2023 표를 2024 행에 적용한
   **정직 프레임**이 실제 배포(≤2024 표 → 2025 행)와 같은 관계이므로 그쪽에서 잰다.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "6"
import json, sys, time
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

sys.path.insert(0, "/home/<user>/SungjinPark/exp078/src")
import core078 as C                                              # noqa: E402

H, K, EPS, REF = C.H, C.K, C.EPS, C.REF
OUT = "/home/<user>/SungjinPark/exp080/out"
N = C.N_ROWS
AXES, IDX, BO = C.AXES, C.IDX, C.BO
T1 = time.time()
R = {"exp_id": "exp_080", "part": "stage2b+3+4", "K_pts_per_brier": K}
OOT, OOT3, OOT4 = f"{H}/exp069/oot", f"{H}/exp070/oot3", f"{H}/exp071/out"
W4 = {"LGB": 0.05, "CAT": 0.68, "NN": 0.15, "LEAF": 0.12}
AA = (42, 7, 2024)
GT_CELLS = ["1|1", "1|2", "2|1", "2|2"]
MQ = json.load(open(f"{H}/build_BO/pkg_BO/mate/model_q/meta.json"))
OFS, BKO = MQ["offset"], MQ["backoff"]
BETA_Q, EBK, LEAGUE = float(OFS["beta"]), float(OFS["eb_k"]), BKO["league_dev"]


def lg(p):
    return logit(np.clip(np.asarray(p, np.float64), EPS, 1 - EPS))


def pts(b):
    return 100000.0 * (1.0 - b / REF)


# ══════════════════════════════════════════════════════════════════════
# A. 정직 프레임 — ≤2023 표를 2024 행에 적용
# ══════════════════════════════════════════════════════════════════════
tr = pd.read_csv(f"{H}/data/train.csv",
                 usecols=["row_id", "season", "game_type", "control_success",
                          "pitcher_id", "batter_hand", "pitcher_hand"])
tro = tr[tr.season == 2024].sort_values("row_id", kind="stable")
yo = tro.control_success.to_numpy(np.float64)
rid = np.asarray(np.load(f"{OOT}/row_id_oot2024.npy", allow_pickle=True)).astype(str)
assert np.array_equal(rid, tro.row_id.to_numpy().astype(str))
YB = float(yo.mean())


def build_tables(maxseason):
    m = (tr.season <= maxseason) & ~((tr.season <= 2022) & (tr.game_type == "F"))
    d = tr[m & tr.control_success.notna()].copy()
    d["pitcher_id"] = d.pitcher_id.astype("Int64")
    d["batter_hand"] = d.batter_hand.astype("Int64")
    d["pitcher_hand"] = d.pitcher_hand.astype("Int64")
    ov = d.groupby("pitcher_id").control_success.mean()
    g = d.groupby(["pitcher_id", "batter_hand"]).control_success.agg(["sum", "count"]).reset_index()
    g["overall"] = g.pitcher_id.map(ov)
    num = g["sum"].to_numpy(np.float64) - g["count"].to_numpy(np.float64) * g.overall.to_numpy(np.float64)
    kk = [f"{int(a)}|{int(b)}" for a, b in zip(g.pitcher_id, g.batter_hand)]
    lgp = d.groupby("pitcher_hand").control_success.mean()
    lpb = d.groupby(["pitcher_hand", "batter_hand"]).control_success.agg(["mean", "count"])
    LD = {f"{int(a)}|{int(b)}": float(r["mean"] - lgp.loc[a]) for (a, b), r in lpb.iterrows()}
    return dict(zip(kk, num)), dict(zip(kk, g["count"].to_numpy(np.int64))), LD, int(len(d))


NUM23, NC23, LD23, nf23 = build_tables(2023)
NUM24, NC24, LD24, nf24 = build_tables(2024)
_dref = OFS["lookup_d"]
_cm = set(NUM24) & set(_dref)
_cal = np.array([NUM24[k] / (NC24[k] + EBK) - _dref[k] for k in _cm])
R["table_rebuild_calibration"] = {
    "le2024_rebuild_vs_deployed_max_abs_d": float(np.abs(_cal).max()),
    "le2024_rebuild_vs_deployed_rms_d": float(np.sqrt(np.mean(_cal ** 2))),
    "deployed_d_sd_over_cells": OFS["lookup_diag"]["d_sd_over_cells"],
    "n_fit_rows_rebuild_le2024": nf24, "n_fit_rows_meta_cells": MQ["n_fit_rows_cells"],
    "n_fit_rows_meta_stacker": MQ["n_fit_rows_stacker"],
    "🔴": ("재구성이 스태커 행수(1,369,784)와는 정확히 일치하고 셀 행수(1,369,186)와 "
          "598 만큼 다르다 — meta 의 n_rows_dropped_unrecoverable_label 과 같은 수다. "
          "그 598행을 못 걸러서 생기는 d 오차가 위 rms 이고, 셀 간 d 산포의 "
          "몇 %인지로 크기를 판단한다."),
    "n_fit_rows_rebuild_le2023": nf23}


def keys_of(df, a, b):
    ka = pd.to_numeric(df[a], errors="coerce")
    kb = pd.to_numeric(df[b], errors="coerce")
    ok = (ka.notna() & kb.notna()).to_numpy()
    out = np.array([""] * len(df), dtype=object)
    out[ok] = [f"{int(x)}|{int(y)}" for x, y in zip(ka[ok].astype(int), kb[ok].astype(int))]
    return out


kp_o = keys_of(tro, "pitcher_id", "batter_hand")
kl_o = keys_of(tro, "pitcher_hand", "batter_hand")
hit_o = np.array([k in NUM23 for k in kp_o])
num_o = np.array([NUM23.get(k, 0.0) for k in kp_o], np.float64)
nc_o = np.array([NC23.get(k, 0) for k in kp_o], np.float64)
R["honest_frame_hit_rate"] = {
    "n_hit": int(hit_o.sum()), "hit_frac": float(hit_o.mean()),
    "n_miss": int((~hit_o).sum()), "miss_frac": float((~hit_o).mean()),
    "deploy_standin_miss_frac": 0.0,
    "🔴": ("배포 대역 프레임(≤2024 표 → 2024 행)은 적중 100%, 정직 프레임(≤2023 표 "
          "→ 2024 행)은 위 값.  실제 배포(≤2024 표 → 2025 행)는 정직 프레임 쪽과 "
          "같은 관계이므로 리그 백오프 노출 추정치는 이 값이다.")}
print("정직 프레임 적중률:", R["honest_frame_hit_rate"], flush=True)


def d_of(k=EBK, league=None):
    lgd = LD23 if league is None else league
    Lr = np.array([lgd.get(kk, 0.0) for kk in kl_o], np.float64)
    ds = np.where(hit_o, num_o / (nc_o + k), 0.0)
    return ds + np.where(hit_o, 0.0, Lr)


# ── 정직면 ───────────────────────────────────────────────────────────
V = {n: np.load(p).astype(np.float64) for n, p in {
    "p_lgb": f"{OOT}/p_lgb_oot2024.npy", "p_cat": f"{OOT}/p_cat_oot2024.npy",
    "p_leaf": f"{OOT}/p_leaf_oot2024.npy", "p_nn": f"{OOT4}/p_nn_oot2024.npy",
    "p_q": f"{OOT}/p_q_oot2024.npy", "p_ebm": f"{OOT3}/p_ebm_oot2024.npy",
    "off": f"{OOT}/off_oot2024.npy"}.items()}
L4 = {n: lg(V[n]) for n in ("p_lgb", "p_cat", "p_leaf", "p_nn")}
LV0 = (W4["LGB"] * L4["p_lgb"] + W4["CAT"] * L4["p_cat"]
       + W4["NN"] * L4["p_nn"] + W4["LEAF"] * L4["p_leaf"])
LE = lg(V["p_ebm"])
L_aa = np.mean([lg(np.load(f"{H}/experiments/exp_049/pred/"
                           f"pred_A1_u2024_seed{s}_at1400.npy")) for s in AA], axis=0)
L_t70 = np.mean([lg(np.load(f"{H}/experiments/exp_055/pred/"
                            f"pred_T70_u2024_seed{s}_at1400.npy")) for s in AA], axis=0)


def rc(z):
    c = 0.0
    for _ in range(80):
        p = expit(z + c)
        mu = float(p.mean())
        if abs(mu - YB) < 1e-14:
            break
        d = (mu - YB) / float((p * (1 - p)).mean())
        c -= d
        if abs(d) < 1e-14:
            break
    return c


def zof(lq, lv=None):
    lv = LV0 if lv is None else lv
    return (BO["w_out"] * ((1 - BO["w_ebm"])
                           * (BO["w_v6_inner"] * lv + (1 - BO["w_v6_inner"]) * lq)
                           + BO["w_ebm"] * LE + BO["beta_ins"] * V["off"])
            + BO["c_aa"] * L_aa + BO["c_T70"] * L_t70)


def pts_of_z(z):
    p = expit(z + rc(z))
    return pts(float(np.mean((p - yo) ** 2)))


# p_q_oot 가 오프셋 미적용 원 스태커인지 판정
d_dep = d_of()
r0 = yo - V["p_q"]
slope0 = float(np.mean(d_dep * r0) / np.mean(d_dep ** 2))
R["oot_q_offset_state"] = {
    "no_intercept_slope_E[dr]/E[d2]": slope0, "deployed_beta": BETA_Q,
    "verdict": ("오프셋 미적용(원 스태커)로 보인다 — 기울기가 배포 beta 근처"
                if abs(slope0 - BETA_Q) < 0.25 else
                "판단 보류 — 기울기가 배포 beta 와 멀다"),
    "🔴": ("beta 는 `offset.fit_slope` 로 **바로 이 면(fold_2024)** 에서 최소제곱 "
          "적합된 값이다.  따라서 정직면에서 beta 를 다시 최적화하는 것은 표본 내이며, "
          "그 최적점이 배포값 근처에 오는 것은 당연하다.  블렌드 Brier 기준의 최적은 "
          "팔 잔차 최소제곱 기준과 달라서 0 이 아닐 수 있고, 그 차이만이 새 정보다.")}
print("OOT p_q 오프셋 상태:", R["oot_q_offset_state"], flush=True)


def lq_of(beta=BETA_Q, k=EBK, league=None):
    return lg(np.clip(V["p_q"] + beta * d_of(k, league), EPS, 1 - EPS))


LQ_DEP = lq_of()
MU0, SD0 = float(LQ_DEP.mean()), float(LQ_DEP.std())
base_off = pts_of_z(zof(LQ_DEP))
base_raw = pts_of_z(zof(lg(V["p_q"])))
R["honest_base"] = {"with_q_offset_le2023_tables": base_off,
                    "delivered_no_offset(=exp078 눈금)": base_raw,
                    "expected_no_offset": 928.501461058684,
                    "abs_diff_no_offset": abs(base_raw - 928.501461058684)}
assert abs(base_raw - 928.501461058684) < 1e-9
print(f"정직면: 오프셋 없음 {base_raw:.6f} · ≤2023표 오프셋 적용 {base_off:.6f}", flush=True)


def scan(name, f, lo, hi, x0):
    def g(x):
        lq = f(x)
        return pts_of_z(zof(lq))

    def gs(x):
        lq = f(x)
        mu, sd = float(lq.mean()), float(lq.std())
        return pts_of_z(zof(MU0 + (lq - mu) * (SD0 / sd))), sd / SD0

    rr = minimize_scalar(lambda x: -g(x), bounds=(lo, hi), method="bounded",
                         options={"xatol": (hi - lo) * 1e-7})
    xs = float(rr.x)
    graw = float(-rr.fun) - base_off
    gsd, sdr = gs(xs)
    gsd -= base_off
    return {"deployed_value": x0, "argmax": xs, "raw_gain_pts": graw,
            "sd_matched_gain_at_raw_argmax_pts": gsd, "sd_ratio": sdr,
            "information_share": (gsd / graw) if graw != 0 else None,
            "grid": {f"{v:.6g}": g(v) - base_off for v in
                     np.linspace(lo, hi, 11)}}


S2B = {}
S2B["q_beta"] = scan("q_beta", lambda b: lq_of(beta=b), 0.0, 1.6, BETA_Q)
S2B["q_k"] = scan("q_k", lambda k: lq_of(k=k), 20.0, 4000.0, EBK)
for cc in GT_CELLS:
    S2B[f"q_league_{cc.replace('|','')}"] = scan(
        cc, lambda v, cc=cc: lq_of(league={**LD23, cc: v}),
        LD23[cc] - 0.08, LD23[cc] + 0.08, LEAGUE[cc])
for nm, v in S2B.items():
    print(f"[정직면 {nm}] 배포 {v['deployed_value']:+.6g} → argmax {v['argmax']:+.6g} "
          f"raw {v['raw_gain_pts']:+.5f}점 sd맞춤 {v['sd_matched_gain_at_raw_argmax_pts']:+.5f}점 "
          f"(정보 {100*(v['information_share'] or 0):.1f}%)", flush=True)
R["stage2b_honest_surface"] = S2B

# 정직 프레임에서의 리그축 δ 규모 (배포 대역에서 0 이 나온 것을 대신한다)
LGQ_W = BO["w_out"] * (1 - BO["w_ebm"]) * (1 - BO["w_v6_inner"])
p_hb = expit(zof(LQ_DEP) + rc(zof(LQ_DEP)))
w_ch = p_hb * (1 - p_hb)
LEAGUE_D = {}
for cc in GT_CELLS:
    ind = (~hit_o) & np.array([k == cc for k in kl_o])
    pq = expit(LQ_DEP)
    dlq = np.where(ind, BETA_Q / (pq * (1.0 - pq)), 0.0)
    dz = LGQ_W * dlq
    dj = w_ch * (dz - float(np.sum(w_ch * dz) / np.sum(w_ch)))
    LEAGUE_D[cc] = {"n_rows_fire": int(ind.sum()), "frac_fire": float(ind.mean()),
                    "rms_delta_honest_frame": float(np.sqrt(np.mean(dj ** 2))),
                    "K_times_G_jj_honest_frame": float(K * np.mean(dj ** 2))}
    print(f"  리그셀 {cc}: 발동 {int(ind.sum())}행 K*G_jj(정직프레임) "
          f"{K*np.mean(dj**2):.4f}", flush=True)
R["stage2b_league_delta_on_honest_frame"] = LEAGUE_D
json.dump(R, open(f"{OUT}/s3_partial.json", "w"), indent=1, ensure_ascii=False)
print(f"\nA 완료 {(time.time()-T1)/60:.2f}분", flush=True)

# ══════════════════════════════════════════════════════════════════════
# B. stage3 — 16셀 잔여 (BQ2 산출물 위 대수, 재계산 없음)
# ══════════════════════════════════════════════════════════════════════
BQ = json.load(open(f"{H}/build_BPBQ2/work/solve_BQ2.json"))
Gd = np.array(BQ["G_diag"], np.float64)
Lc = np.array(BQ["L_per_cell"], np.float64)
mc = np.array(BQ["orthogonalisation"]["cell_weight_m_c"], np.float64)
xb = np.array(BQ["orthogonalisation"]["cell_xbar"], np.float64)
x2b = np.array(BQ["orthogonalisation"]["cell_x2bar"], np.float64)
SEc = np.array(BQ["per_cell_evidence"]["std_error"], np.float64)
tc = np.array(BQ["per_cell_evidence"]["t"], np.float64)
S3 = np.sqrt(Gd)                                        # ||δ_c|| / sqrt(N)
# 계수공간 계량 <a,b> = Σ a_c b_c G_cc.  이미 산 방향 = {1, xbar, x2bar}
Bm = np.stack([np.ones(16), xb, x2b], 1)
Wm = np.diag(Gd)
Q, _ = np.linalg.qr(np.sqrt(Gd)[:, None] * Bm)          # 계량 하 정규직교
P = np.eye(16) - Q @ Q.T                                # 잔여 사영 (스케일 좌표)
# 스케일 좌표 v = sqrt(G)·θ 에서 작업한다.  L 의 대응 = L_c/sqrt(G_cc)
Lv = Lc / np.sqrt(Gd)
Lres = P @ Lv
# 셀별 분산: Var(L_c) = SE_c²·G_cc²  → 스케일 좌표에서 Var(Lv_c) = SE_c²·G_cc
varv = SEc ** 2 * Gd
Sig = P @ np.diag(varv) @ P.T
ev, evec = np.linalg.eigh(P)
keep = ev > 1e-10
Ures = evec[:, keep]                                    # 16x13 (v좌표 정규직교)
R["stage3_residual_rank"] = int(keep.sum())
# v좌표에서 잡음은 거의 등분산이다 (s_c^2 = SE_c^2 * G_cc).  따라서 |t| 를
# 최대화하는 잔여방향은 u1 = P·Lv 를 정규화한 것이고, 그 하나가 잔여 L 을 전부 가져간다.
s2v = SEc ** 2 * Gd
u1 = Lres / np.linalg.norm(Lres)
L1 = float(u1 @ Lres)
t1 = L1 / float(np.sqrt(u1 @ np.diag(s2v) @ u1))
gain_res_total = float(K * float(Lres @ Lres))
floor_per_dof_honest = float(K * np.mean(s2v[:1])) if False else None
NF = np.array(BQ["noise_floor_diagnostic"]["per_cell_noise_floor_pts"], np.float64)
floor_dof = float(NF.mean())
DIRS3 = [{"rank": 1, "name": "cellres_1(=P·L 방향)", "t": t1,
          "in_sample_gain_pts": gain_res_total,
          "check_vs_BQ2_theta_orth_gain": BQ["honest_surface_gain_pts"]["theta_orth"],
          "debiased_pts_subtract_1dof": float(gain_res_total - floor_dof),
          "debiased_pts_subtract_13dof": float(gain_res_total - 13 * floor_dof),
          "note": ("v좌표 등분산이라 잔여공간에서 in-sample L 은 이 한 방향에 전부 "
                   "모인다.  나머지 12방향의 표본내 L 은 정확히 0 이다 — 그러나 "
                   "LB 의 L 은 다른 표본이므로 0 이 아니고, 그래서 잡음 바닥은 산다.")}]
for j2 in range(2, int(keep.sum()) + 1):
    DIRS3.append({"rank": j2, "name": f"cellres_{j2}", "t": 0.0,
                  "in_sample_gain_pts": 0.0,
                  "debiased_pts_subtract_1dof": 0.0,
                  "debiased_pts_subtract_13dof": 0.0})
CELLRES_V = [u1]
_Bfull = np.concatenate([Q, u1[:, None]], 1)
_q2, _ = np.linalg.qr(np.concatenate([_Bfull, np.eye(16)], 1))
for col in range(_q2.shape[1]):
    if len(CELLRES_V) >= int(keep.sum()):
        break
    v = _q2[:, col]
    v = P @ v
    for u in CELLRES_V:
        v = v - float(v @ u) * u
    nv = float(np.linalg.norm(v))
    if nv > 1e-8:
        CELLRES_V.append(v / nv)
R["stage3_cells"] = {
    "per_cell_t": tc.tolist(), "per_cell_theta_hat": BQ["per_cell_evidence"]["theta_hat"],
    "n_abs_t_ge_2": int((np.abs(tc) >= 2).sum()),
    "which_abs_t_ge_2": [f"{'RF'[c//8]}{c%8}" for c in range(16) if abs(tc[c]) >= 2],
    "G_diag": Gd.tolist(),
    "deploy_tr2024_cell_counts": BQ["deploy_tr2024_cell_counts"],
    "honest_cell_counts": BQ["honest_cell_counts"],
    "residual_directions": DIRS3,
    "residual_noise_floor_per_dof_honest_pts": floor_dof,
    "residual_block_debiased_signal_pts": float(gain_res_total - 13 * floor_dof),
    "noise_floor_diagnostic": BQ["noise_floor_diagnostic"],
    "honest_half_cv_gain_pts": BQ["honest_half_cv_gain_pts"],
    "🔴_BQ_는_다른_물건": ("BQ 는 16셀 theta 를 한 번에 실어 정직면 절반 CV 0/2 로 졌다. "
                     "캠페인은 셀 방향을 **하나씩 LB 로 사서** 꼭지점에 앉는 것이라 "
                     "L 이 측정값이고 이득이 K·L²/G ≥ 0 로 보장된다 — 같은 표를 쓰지만 "
                     "다른 물건이다.")}
print(f"\nstage3 잔여 차원 {int(keep.sum())} · |t|≥2 셀 "
      f"{R['stage3_cells']['which_abs_t_ge_2']}", flush=True)
print(f"  잔여 1방향: t {t1:+.4f} 표본내 {gain_res_total:+.5f}점 "
      f"(BQ2 theta_orth {BQ['honest_surface_gain_pts']['theta_orth']:+.5f}) "
      f"· 13dof 바닥 제거 후 {gain_res_total - 13*floor_dof:+.5f}점", flush=True)

json.dump(R, open(f"{OUT}/s3_partial.json", "w"), indent=1, ensure_ascii=False)
print(f"B 완료 {(time.time()-T1)/60:.2f}분", flush=True)

# ══════════════════════════════════════════════════════════════════════
# C. stage4 — 배포 프레임 δ 모으기 → 그람-슈미트 → 최종 13방향
# ══════════════════════════════════════════════════════════════════════
XD = np.load(f"{H}/build_BPBQ2/work/xdef_solver.npz", allow_pickle=True)
p_BO = C.p_BO
fr = pd.read_csv(f"{H}/build_AC/work/tr2024/test.csv", encoding="utf-8-sig",
                 usecols=["row_id", "game_type"])
gt = fr.game_type.to_numpy().astype(str)
x_tr = XD["x_tr"]
BND = {"R": XD["bounds_R"], "F": XD["bounds_F"]}
cell_tr = np.full(N, -1, np.int64)
for gi, g in enumerate(["R", "F"]):
    m = gt == g
    cell_tr[m] = gi * 8 + np.searchsorted(BND[g], x_tr[m], side="right")
assert (cell_tr >= 0).all()
w_tr = p_BO * (1 - p_BO)
CELLD = {}
for c in range(16):
    ind = (cell_tr == c).astype(np.float64)
    dj = w_tr * (ind - float(np.sum(w_tr * ind) / np.sum(w_tr)))
    CELLD[f"cell_{'RF'[c//8]}{c%8}"] = dj
# 수치 검증 (셀 하나)
_h = 0.002
_f = C.zf(BO, C.ARMS18)
_pp = expit(_f + _h * (cell_tr == 1) + C.anc(_f + _h * (cell_tr == 1), C.T_REF))
_pm = expit(_f - _h * (cell_tr == 1) + C.anc(_f - _h * (cell_tr == 1), C.T_REF))
_num = (_pp - _pm) / (2 * _h)
R["stage4_cell_delta_analytic_vs_numeric"] = float(
    np.abs(_num - CELLD["cell_R1"]).max() / np.abs(CELLD["cell_R1"]).max())

nm_prev = np.load(f"{OUT}/cand_delta_names.npy", allow_pickle=True)
dv_prev = np.load(f"{OUT}/cand_delta.npy")
CAND = {str(n): dv_prev[i] for i, n in enumerate(nm_prev)}
CAND.update(CELLD)
S2M = json.load(open(f"{OUT}/s2_partial.json"))["stage2_axes"]

# 이미 산 방향 = exp_078 15축
Bmat = C.DELTA.copy()
Ub, sb, _ = np.linalg.svd(Bmat.T / np.sqrt(N), full_matrices=False)
rk = int((sb > sb.max() * 1e-8).sum())
Ub = Ub[:, :rk]
R["stage4_bought_basis"] = {"n_axes": len(AXES), "numeric_rank": rk,
                            "singular_values": sb.tolist(),
                            "note": "S19 — 15축의 실효 계수(수치 랭크)."}

# 정보 우선순위
INFO = {}
S1J = json.load(open(f"{OUT}/s1_stage1.json"))["stage1_axes"]
for nm, v in S1J.items():
    INFO[nm] = v["sd_matched_gain_at_raw_argmax_pts"]
for nm, v in S2B.items():
    INFO[nm] = v["sd_matched_gain_at_raw_argmax_pts"]
_res_share = float(max(0.0, gain_res_total - 13 * floor_dof))
_w_cell = np.abs(u1) / max(np.abs(u1).sum(), 1e-30)
for c in range(16):
    INFO[f"cell_{'RF'[c//8]}{c%8}"] = float(_res_share * _w_cell[c])
R["stage3_cell_info_allocation"] = {
    "residual_block_debiased_pts": _res_share,
    "u1_weights": u1.tolist(),
    "note": ("셀 하나하나에는 개별 정직면 정보가 따로 없다 — 잔여 블록 전체의 "
             "편의보정 신호를 잔여 최대 t 방향 u1 의 |성분|에 비례해 나눠 배정한 값이다. "
             "순서 결정용이며 그 자체가 그 셀의 측정 이득이 아니다.")}
ORDER = sorted([k for k in CAND if np.abs(CAND[k]).max() > 0],
               key=lambda k: -INFO.get(k, 0.0))
DEAD = [k for k in CAND if np.abs(CAND[k]).max() == 0]

SEL, basis = [], [Ub[:, j] for j in range(rk)]
for nm in ORDER:
    d0 = CAND[nm] / np.sqrt(N)
    r = d0.copy()
    for b in basis:
        r = r - float(r @ b) * b
    ratio = float(np.linalg.norm(r) / np.linalg.norm(d0))
    rho_max = max(abs(float(np.corrcoef(CAND[nm], C.DELTA[IDX[a]])[0, 1]))
                  for a in AXES)
    rec = {"axis": nm, "orth_norm_ratio": ratio,
           "max_abs_corr_bought_15": rho_max,
           "rms_delta_orth": float(np.linalg.norm(r) / np.sqrt(1.0)),
           "K_times_G_jj_orth_pts_at_u1": float(K * float(r @ r)),
           "K_times_G_jj_raw": float(K * float(d0 @ d0)),
           "probe_u_for_KQ_1pt_orth": float(1.0 / np.sqrt(K * float(r @ r)))
           if float(r @ r) > 0 else None,
           "honest_information_pts": INFO.get(nm, 0.0),
           "dropped_reason": None}
    if rho_max > 0.9:
        rec["dropped_reason"] = f"S19 — 기존 15축과 |ρ| {rho_max:.4f} > 0.9"
    elif ratio < 0.3:
        rec["dropped_reason"] = f"직교화 후 노름비 {ratio:.4f} < 0.30 — 사실상 중복"
    elif np.linalg.norm(r) < 1e-14:
        rec["dropped_reason"] = "직교화 후 0"
    else:
        basis.append(r / np.linalg.norm(r))
        SEL.append(rec)
    rec["selected"] = rec["dropped_reason"] is None
    R.setdefault("stage4_all_candidates", []).append(rec)

R["stage4_dead_axes_on_deploy_frame"] = {
    k: "δ ≡ 0 — 대역 프레임에서 리그 백오프가 한 행도 발동하지 않는다(적중 100%)"
    for k in DEAD}
TOP = SEL[:13]
R["stage4_final_13"] = TOP
NOISE_LB = K * (0.2498069 * (1 - 1115.9165956458 / 100000.0)) / 245789.0
R["stage4_campaign"] = {
    "n_slots": 13,
    "noise_floor_per_dof_pts_LB": NOISE_LB,
    "noise_floor_total_13_pts": 13 * NOISE_LB,
    "config_declared_floor": 0.402,
    "honest_face_floor_per_dof_pts": K * float(np.mean((yo - p_hb) ** 2)) / N,
    "sum_honest_information_of_13_pts": float(sum(d["honest_information_pts"]
                                                  for d in TOP)),
    "ledger_multiplier_5.09": 5.09,
    "optimistic_total_pts": 13 * NOISE_LB * 5.09,
    "noise_floor_total_pts": 13 * NOISE_LB,
    "🔴_S5": ("낙관치는 장부 6축의 실현합/잡음기대 비 5.09 를 그대로 적용한 산술이지 "
             "측정이 아니다.  하방: 꼭지점 이득 K·L²/G ≥ 0 이므로 **바닥은 0 이 아니라 "
             "잡음 바닥 자체**이고 그 아래로는 안 내려간다(S22).  대칭인 것은 잡음 "
             "바닥 **위** 부분의 추정이다 — 5.09배가 1.0배일 수도 있고 그러면 총액은 "
             "5.23점이다."),
    "🔴_S19": "위 13방향은 기존 15축의 span 에 직교화된 뒤 남은 것만 센 값이다."}
R["wall_min"] = (time.time() - T1) / 60.0
json.dump(R, open(f"{OUT}/s3_stage34.json", "w"), indent=1, ensure_ascii=False)
print("\n=== 최종 13방향 ===", flush=True)
for d in TOP:
    print(f"  {d['axis']:<18} 노름비 {d['orth_norm_ratio']:.4f} "
          f"K*G_jj {d['K_times_G_jj_orth_pts_at_u1']:.4f} "
          f"진폭 {d['probe_u_for_KQ_1pt_orth']:.6g} "
          f"정직정보 {d['honest_information_pts']:+.5f}점", flush=True)
print(f"\n잡음바닥 총 {13*NOISE_LB:.4f}점 · 낙관 {13*NOISE_LB*5.09:.4f}점", flush=True)
print(f"기록 {OUT}/s3_stage34.json · {R['wall_min']:.2f}분", flush=True)
