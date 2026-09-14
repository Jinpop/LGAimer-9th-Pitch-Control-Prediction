#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp_084 stage2·4·5 — 곡률 G 를 미적중률의 함수로, 반대칭 축, 교차항.  학습 0건.

기준면: build_BR/work/dump_le23_bo_dump (≤2023 표 → 2024 행, 미적중 21.4988%).
       build_q 는 build_BR/src/diag_qleague.py 와 같은 경로다 (자체증명 3종으로 못박는다).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "6"
import io, json, sys, time                                          # noqa: E402
import numpy as np                                                  # noqa: E402
import pandas as pd                                                 # noqa: E402
from scipy.special import expit                                     # noqa: E402

sys.path.insert(0, "/home/<user>/SungjinPark/exp078/src")
import core078 as C                                                 # noqa: E402

H = "/home/<user>/SungjinPark"
OUT = f"{H}/exp084/out"
FRAME = f"{H}/build_AC/work/tr2024"
EPS = 1e-6
K, N, T_REF = C.K, C.N_ROWS, C.T_REF
T0 = time.time()
CELLS = ["1|1", "1|2", "2|1", "2|2"]
R = {"exp_id": "exp_084", "stage": "2/4/5", "date": time.strftime("%Y-%m-%d %H:%M:%S"),
     "K_pts_per_brier": K, "frame": "build_AC/work/tr2024 + ≤2023 Q 표 (diag_qleague 와 동일)"}

MQ = json.load(open(f"{H}/build_BR/parent_from_zip/mate/model_q/meta.json", encoding="utf-8"))
S = json.load(open(f"{H}/build_BR/work/solve_BR.json", encoding="utf-8"))
BETA = float(MQ["offset"]["beta"])
LD24 = MQ["backoff"]["league_dev"]
NB24 = MQ["backoff"]["n_by_cell"]
LD23 = S["le2023_tables"]["league_dev"]
R["league_dev_le2024_shipped"] = LD24
R["league_dev_le2023_rebuilt"] = LD23
R["n_by_cell_le2024_shipped"] = NB24

DD = f"{H}/build_BR/work/dump_le23_bo_dump"
p_pre = np.load(f"{DD}/p_pre.npy")
d_p23 = np.load(f"{DD}/d_p.npy")
hit23 = np.load(f"{DD}/hit.npy")
hasL23 = np.load(f"{DD}/has_L.npy")
d_m23 = np.load(f"{DD}/d_m.npy")
p_off23 = np.load(f"{DD}/p_off.npy")

fr = pd.read_csv(f"{FRAME}/test.csv", encoding="utf-8-sig",
                 usecols=["row_id", "game_type", "pitcher_id", "batter_hand", "pitcher_hand"])
assert len(fr) == N


def _key(a, b):
    ka = pd.to_numeric(a, errors="coerce"); kb = pd.to_numeric(b, errors="coerce")
    ok = (ka.notna() & kb.notna()).to_numpy()
    out = np.array([""] * len(a), dtype=object)
    out[ok] = [f"{int(x)}|{int(y)}" for x, y in zip(ka[ok].astype(int), kb[ok].astype(int))]
    return out


kl = _key(fr.pitcher_hand, fr.batter_hand)
CM = {c: np.array([k == c for k in kl]) for c in CELLS}


def build_q(lgq, th=None, boff=None):
    """diag_qleague.build_q 와 같은 경로.  boff 는 실패유형 축 오프셋(외곽면)."""
    th = dict(C.BO if th is None else th)
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
    a = (th["alpha"] if th["seg_spread_BH"] == 0.0
         else th["alpha"] + th["seg_spread_BH"] * C.CON)
    f = a * zp + th["beta_skew"] * zp * zp
    if boff is not None:
        f = f + boff
    return expit(f + C.anc(f, T_REF) + th["d_mean"])


def dof(fire, league):
    """d = d_s(≤2023) + (fire ? L : 0).  fire 는 개인 셀 미적중 마스크."""
    Lr = np.array([league.get(kk, 0.0) for kk in kl], np.float64)
    return d_p23 + np.where(fire, Lr, 0.0)


def lgq_of(fire, league):
    raw = p_pre + BETA * dof(fire, league)
    return C.lg(np.clip(raw, EPS, 1 - EPS))


MISS = ~hit23
# ── 자체증명 3종 (diag_qleague 와 같은 경로임을 기계로 못박는다) ───────
_chk = build_q(C.LGQ)
R["G0a_build_q_reproduces_p_BO"] = float(np.abs(_chk - C.p_BO).max())
assert R["G0a_build_q_reproduces_p_BO"] < 1e-12
_dm = dof(MISS, LD23)
R["G0b_d_m_bitwise"] = bool(np.array_equal(_dm, d_m23))
_lgq_base = lgq_of(MISS, LD23)
R["G0c_lgq_bitwise"] = bool(np.array_equal(_lgq_base, C.lg(p_off23)))
assert R["G0b_d_m_bitwise"] and R["G0c_lgq_bitwise"]
print(f"[G0] build_q==p_BO {R['G0a_build_q_reproduces_p_BO']:.2e} · d_m 비트동일 "
      f"{R['G0b_d_m_bitwise']} · lgq 비트동일 {R['G0c_lgq_bitwise']}", flush=True)

HL = 0.002


def measure(fire, tag):
    """이 발동 마스크 아래에서 4셀 δ·G 와 반대칭 축을 잰다."""
    base = build_q(lgq_of(fire, LD23))
    d = {}
    for c in CELLS:
        pp = build_q(lgq_of(fire, {**LD23, c: LD23[c] + HL}))
        pm = build_q(lgq_of(fire, {**LD23, c: LD23[c] - HL}))
        d[c] = (pp - pm) / (2 * HL)
    ent = {"tag": tag, "n_fire": int(fire.sum()), "miss_frac": float(fire.mean()),
           "per_cell": {}}
    for c in CELLS:
        m = CM[c] & fire
        g = float(np.mean(d[c] ** 2))
        ent["per_cell"][c] = {
            "n_fire_in_cell": int(m.sum()),
            "fire_frac_of_frame": float(m.mean()),
            "G_jj": g, "K_G_jj": K * g,
            "K_E_delta2_given_fire": (K * float(np.mean(d[c][m] ** 2))
                                      if m.sum() else None),
            "rms_delta": float(np.sqrt(g)),
            "max_abs_delta": float(np.abs(d[c]).max()),
            "n_nonzero": int(np.count_nonzero(d[c]))}
    # 반대칭 축 (ph=2 안에서 표본수 가중 합 0 유지): ΔL22 = +1, ΔL21 = −n22/n21
    w = NB24["2|2"] / NB24["2|1"]
    da = d["2|2"] - w * d["2|1"]
    ent["antisym_ph2"] = {
        "weight_ratio_n22_over_n21": w,
        "G_jj": float(np.mean(da ** 2)), "K_G_jj": K * float(np.mean(da ** 2)),
        "rho_d22_d21": float(np.corrcoef(d["2|2"], d["2|1"])[0, 1]),
        "check_sum_G": (K * float(np.mean(d["2|2"] ** 2))
                        + w * w * K * float(np.mean(d["2|1"] ** 2))
                        - 2 * w * K * float(np.mean(d["2|2"] * d["2|1"])))}
    return ent, d, base


ENT0, D0, P_BASE = measure(MISS, "actual_le2023_21.4988%")
R["frame_le2023_actual"] = ENT0
print(f"[실측 ≤2023] 미적중 {ENT0['miss_frac']*100:.4f}% · "
      f"K·G(2|2) {ENT0['per_cell']['2|2']['K_G_jj']:.5f} · "
      f"K·G(2|1) {ENT0['per_cell']['2|1']['K_G_jj']:.5f} · "
      f"반대칭 K·G {ENT0['antisym_ph2']['K_G_jj']:.5f}", flush=True)

# 대조 — diag_qleague 가 기록한 K·G_jj 와 맞는가
DQ = json.load(open(f"{H}/build_BR/work/diag_qleague.json", encoding="utf-8"))
R["control_vs_diag_qleague"] = {
    nm: {"diag": DQ["axes"][f"q_league_{nm.replace('|','')}"]["K_times_G_jj_pts_at_u1"],
         "here": ENT0["per_cell"][nm]["K_G_jj"],
         "rel_diff": abs(DQ["axes"][f"q_league_{nm.replace('|','')}"]["K_times_G_jj_pts_at_u1"]
                         - ENT0["per_cell"][nm]["K_G_jj"])
         / DQ["axes"][f"q_league_{nm.replace('|','')}"]["K_times_G_jj_pts_at_u1"]}
    for nm in CELLS}
print("[대조 diag_qleague]", json.dumps(R["control_vs_diag_qleague"], ensure_ascii=False), flush=True)

# ── 합성 프레임: 미적중률을 바꿔가며 G 를 잰다 ─────────────────────────
rng = np.random.default_rng(42)
perm_hit = rng.permutation(np.flatnonzero(hit23))
perm_miss = rng.permutation(np.flatnonzero(MISS))
GRID = [0.0, 0.05, 0.10, 0.1614, 0.21498814628392904, 0.30, 0.43, 0.60, 0.80, 1.00]
SYN = []
for m in GRID:
    want = int(round(m * N))
    if want >= MISS.sum():
        fire = MISS.copy()
        extra = want - int(MISS.sum())
        fire[perm_hit[:extra]] = True
    else:
        fire = np.zeros(N, bool)
        fire[perm_miss[:want]] = True
    ent, _, _ = measure(fire, f"synth_m={m}")
    ent["target_m"] = m
    SYN.append(ent)
    print(f"[합성 m={m:.4f}] 실현 {ent['miss_frac']*100:7.4f}% · 2|2 발동 "
          f"{ent['per_cell']['2|2']['n_fire_in_cell']:6d} "
          f"({ent['per_cell']['2|2']['fire_frac_of_frame']*100:6.4f}%) · K·G(2|2) "
          f"{ent['per_cell']['2|2']['K_G_jj']:9.5f} · K·G(반대칭) "
          f"{ent['antisym_ph2']['K_G_jj']:9.5f}", flush=True)
R["synthetic_frames"] = SYN

# ── G ∝ 발동비중 이 성립하는가 — 원점 통과 1계수 적합 ─────────────────
def prop_fit(xs, ys):
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    c = float(xs @ ys / (xs @ xs))
    pred = c * xs
    ss = float(((ys - ys.mean()) ** 2).sum())
    return {"coef_K_E_delta2_given_fire": c,
            "r2_through_origin": float(1 - ((ys - pred) ** 2).sum() / ss) if ss > 0 else None,
            "max_rel_resid": float(np.max(np.abs(ys - pred) / np.maximum(ys, 1e-30))),
            "points_x_fire_frac": xs.tolist(), "points_y_K_G": ys.tolist(),
            "pred": pred.tolist()}


FITS = {}
for c in CELLS:
    xs = [e["per_cell"][c]["fire_frac_of_frame"] for e in SYN]
    ys = [e["per_cell"][c]["K_G_jj"] for e in SYN]
    FITS[c] = prop_fit(xs, ys)
FITS["antisym_ph2"] = prop_fit(
    [e["per_cell"]["2|2"]["fire_frac_of_frame"] + (NB24["2|2"] / NB24["2|1"]) ** 2
     * e["per_cell"]["2|1"]["fire_frac_of_frame"] for e in SYN],
    [e["antisym_ph2"]["K_G_jj"] for e in SYN])
R["proportionality_fit"] = FITS
for k, v in FITS.items():
    print(f"[비례 {k}] 계수 {v['coef_K_E_delta2_given_fire']:.4f} · "
          f"R²(원점) {v['r2_through_origin']:.6f} · 최대상대잔차 {v['max_rel_resid']:.4f}",
          flush=True)

# ── stage5: 팀원 실패유형 축 b 와의 δ 상관 (같은 ≤2023 기준면) ─────────
MU_MR, MU_WO = -0.45376277509804325, -2.0056142032773767
ft = pd.read_csv(f"{H}/exp082/runs/ft_matecal.csv", encoding="utf-8-sig",
                 float_precision="round_trip")
assert list(ft.columns) == ["row_id", "lg_mr", "lg_wo"] and len(ft) == N
assert ft.row_id.astype(str).tolist() == fr.row_id.astype(str).tolist()
LGMR = ft.lg_mr.to_numpy(np.float64); LGWO = ft.lg_wo.to_numpy(np.float64)
HB = 0.002
db = (build_q(_lgq_base, boff=HB * (LGMR - MU_MR))
      - build_q(_lgq_base, boff=-HB * (LGMR - MU_MR))) / (2 * HB)
dc = (build_q(_lgq_base, boff=HB * (LGWO - MU_WO))
      - build_q(_lgq_base, boff=-HB * (LGWO - MU_WO))) / (2 * HB)
d22, d21 = D0["2|2"], D0["2|1"]
w = NB24["2|2"] / NB24["2|1"]
danti = d22 - w * d21


def rho(a, b):
    return float(np.corrcoef(a, b)[0, 1])


R["stage5_cross"] = {
    "note": ("b·c 는 팀원 실패유형 축(exp_082) — 외곽면에 b·(lg_mr−μ) 를 더하고 d_adj 재고정. "
             "여기서는 ≤2023 표 기준면 위에서 다시 쟀다 (exp_082 는 ≤2024 표 면이라 "
             "q_league δ 가 정확히 0 이어서 상관을 잴 수 없다 — S32)."),
    "rho_b_vs_q22": rho(db, d22), "rho_b_vs_q21": rho(db, d21),
    "rho_b_vs_antisym": rho(db, danti),
    "rho_c_vs_q22": rho(dc, d22), "rho_c_vs_antisym": rho(dc, danti),
    "rho_q22_vs_q21": rho(d22, d21),
    "K_G_b": K * float(np.mean(db ** 2)), "K_G_c": K * float(np.mean(dc ** 2)),
    "K_cross_b_q22": K * float(np.mean(db * d22)),
    "K_cross_b_antisym": K * float(np.mean(db * danti)),
    "threshold": "|ρ| > 0.3 이면 단순 합산 위험 (S16)"}
print("[stage5]", json.dumps({k: v for k, v in R["stage5_cross"].items()
                              if k.startswith(("rho", "K_"))}, ensure_ascii=False), flush=True)

# ── real5 실측: 출하 ≤2024 표에서 5행의 개인 셀 적중 여부 ──────────────
try:
    r5 = pd.read_csv(f"{H}/build_BT/frames/real5/test.csv", encoding="utf-8-sig",
                     usecols=["row_id", "season", "game_type", "pitcher_id",
                              "batter_hand", "pitcher_hand"])
    LK = MQ["offset"]["lookup_d"]
    k5 = _key(r5.pitcher_id, r5.batter_hand)
    c5 = _key(r5.pitcher_hand, r5.batter_hand)
    R["real5_direct"] = {
        "n": int(len(r5)),
        "rows": [{"row_id": str(a), "season": int(b), "game_type": str(g),
                  "personal_key": str(kk), "hit": bool(kk in LK), "cell": str(cc)}
                 for a, b, g, kk, cc in zip(r5.row_id, r5.season, r5.game_type, k5, c5)],
        "n_miss": int(sum(1 for kk in k5 if kk not in LK)),
        "miss_frac": float(sum(1 for kk in k5 if kk not in LK) / len(r5))}
    print("[real5 실측]", json.dumps(R["real5_direct"], ensure_ascii=False), flush=True)
except Exception as e:                                              # noqa: BLE001
    R["real5_direct"] = {"error": repr(e)}
    print("[real5] 실패", e, flush=True)

R["wall_sec"] = time.time() - T0
json.dump(R, open(f"{OUT}/s2_curv.json", "w"), indent=1, ensure_ascii=False)
np.save(f"{OUT}/delta_q22_le2023.npy", d22)
np.save(f"{OUT}/delta_q21_le2023.npy", d21)
np.save(f"{OUT}/delta_b_le2023.npy", db)
print(f"\n기록 {OUT}/s2_curv.json · {R['wall_sec']:.1f}s\nS2_OK")
