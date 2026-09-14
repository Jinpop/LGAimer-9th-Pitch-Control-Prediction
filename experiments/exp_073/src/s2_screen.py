"""exp_073 stage 2/3 — BN 잔차 위 부호 스크린 + 🔴 잔차화 검사.

프로토콜 (exp_059 형식 승계, 사전 등록)
    방향 v -> 유한값에서 중심화·1sd(모집단) 정규화 u, 비유한 행은 0 = 무이동
    프로브 f = z_BN + 0.02*u,  앵커는 mean(p_BN) 유지 (brentq)
    delta = expit(f + anc) - p_BN,  M = mean(delta * r0),  r0 = y - p_BN
    t = M / se,  se = 투수 블록 부트스트랩 1000회 (seed 42, 전 축 공통 IDX)
    통과선 |t| >= 2.  🔴 결과를 보고 바꾸지 않는다.  크기는 쓰지 않는다.

🔴 잔차화 검사가 본론이다.  통과 축을 ① ctrl_asof_psr ② 서로 에 대해
   잔차화해 고유 성분의 t 를 다시 낸다.  exp_059 에서 sp_pit_bhand_all 이
   t -2.03 -> 잔차화 후 +0.075 로 붕괴한 전례가 있다.

🔴 부호는 프레임에서 확정하지 않는다 (S15 — 피처 방향 오프셋 계열의 프레임
   부호 신탁은 1/3).  후보를 만들면 LB 로 사서 읽는다.
"""
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit

t0 = time.time()
H = "/home/<user>/SungjinPark"
OUT = f"{H}/exp073_tm"
FRAME = f"{H}/build_AC/work/tr2024"
PKG = f"{H}/build_BN/pkg_BN"
EPS = 1e-6
EPS_PROBE = 0.02
N_ROWS = 253507
THRESH = 2.0
TM_AXES = ["tm_rel_sd", "tm_rel_sd_wp", "tm_speed_trend", "tm_speed_sd",
           "tm_break_sd", "tm_ext_mean", "tm_spin_mean"]

lg = lambda p: logit(np.clip(np.asarray(p, np.float64), EPS, 1 - EPS))


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


PIN = {
    f"{H}/build_AY/work/pcal_aa.npy":
        "3e7159abd25fbfb7441f69bf3f4022b9aac3dbdaf73af8da2765f9c1b645d5bd",
    f"{H}/build_AY/work/pcal_S2.npy":
        "f99a6240c08c53e3e40fa293a0f9efb0911d067236a8cf359d1855f11fb58b99",
    f"{H}/build_AY/work/pcal_S1.npy":
        "776cb0accc0236823b6b2d71612de92fb3711908254037b564d64c0e1c8c0905",
    f"{H}/build_AY/work/pcal_T70.npy":
        "1bac0abde366df280c28f421527066d8c132eb55a1182efc513c36ba5652ac5c",
    f"{FRAME}/test.csv":
        "4db38113d5355546339c4391fbbf8b2e152a7c248ad9261caa1071f6bb81f5d0",
    f"{H}/exp059_screen/directions.npz":
        "5aea557dff22a18b1943419b7890e74220753cf687821d3513f8a0d0341effe6",
}
print("=== 재사용 벡터 sha 대조 ===")
SHA = {}
for p, want in PIN.items():
    g = sha_file(p)
    assert g == want, f"sha 불일치 {p}: {g}"
    SHA[os.path.relpath(p, H)] = g
    print(f"  🟢 {os.path.relpath(p, H):<44} {g[:16]}")
P_V19 = f"{H}/build_BN/work/tr2024_p_v19_BN_wv6_070.npy"
SHA[os.path.relpath(P_V19, H)] = sha_file(P_V19)
SHA["build_BN/pkg_BN/blend.json"] = sha_file(f"{PKG}/blend.json")

# ── BN 부모 재구성 (assemble_BN 과 같은 결합 순서) ──────────────────────
bl = json.load(open(f"{PKG}/blend.json", encoding="utf-8"))
A = float(bl["alpha"]); B = float(bl["beta_skew"])
D_BN = float(bl["d_adj"]); T = float(bl["target_rate"])
W_OUT = float(bl["w_v6"]); C_AA = float(bl["c_aa"])
SW = bl["smooth_weights"]
AXL = bl["axis_offsets"]
assert [a["name"] for a in AXL] == ["beta_ins_off", "ctrl_asof_psr"]

# allow_pickle: row_id 가 object 배열이다.  파일 sha 는 위에서 못박았다.
D59 = np.load(f"{H}/exp059_screen/directions.npz", allow_pickle=True)
ids_frame = pd.read_csv(f"{FRAME}/test.csv", encoding="utf-8-sig",
                        usecols=["row_id", "pitcher_id", "season"])
assert len(ids_frame) == N_ROWS
assert list(ids_frame["row_id"]) == list(D59["row_id"]), "프레임 행 순서 불일치"
assert set(ids_frame["season"].unique()) == {2024}, "프레임이 season 2024 단독이 아니다"
pid_row = ids_frame["pitcher_id"].to_numpy(np.int64)
assert np.array_equal(pid_row, D59["pitcher_id"].astype(np.int64))
y = D59["y"].astype(np.float64)

p_v19 = np.load(P_V19)
p_aa = np.load(f"{H}/build_AY/work/pcal_aa.npy")
p_S2 = np.load(f"{H}/build_AY/work/pcal_S2.npy")
p_S1 = np.load(f"{H}/build_AY/work/pcal_S1.npy")
p_T70 = np.load(f"{H}/build_AY/work/pcal_T70.npy")


def unit(v):
    v = np.asarray(v, np.float64)
    fin = np.isfinite(v)
    mu, sd = float(v[fin].mean()), float(v[fin].std())
    u = np.zeros(len(v))
    u[fin] = (v[fin] - mu) / sd
    return u, mu, sd, int((~fin).sum())


RAW_AX = {"beta_ins_off": np.asarray(D59["beta_ins_off"], np.float64),
          "ctrl_asof_psr": np.asarray(D59["ctrl_asof_psr"], np.float64)}
DEL = {}
for a in AXL:
    nm = a["name"]
    DEL[nm], mu, sd, nnf = unit(RAW_AX[nm])
    assert mu == a["center"] and sd == a["scale"] and nnf == \
        a["n_nonfinite_train2024"], (nm, mu, sd, nnf)

z = (W_OUT * lg(p_v19) + C_AA * lg(p_aa) + float(SW["S2"]) * lg(p_S2)
     + float(SW["S1"]) * lg(p_S1) + float(SW["T70"]) * lg(p_T70))
zp = z + float(AXL[0]["u"]) * DEL["beta_ins_off"] \
     + float(AXL[1]["u"]) * DEL["ctrl_asof_psr"]
f_bn = A * zp + B * zp * zp
z_BN = f_bn + D_BN
p_BN = expit(z_BN)
r0 = y - p_BN
mean_gap = abs(float(p_BN.mean()) - T)
print(f"\n=== BN 부모 재구성 ===")
print(f"  mean(p_BN) {p_BN.mean():.17f}  target {T!r}  |차| {mean_gap:.3e}")
print(f"  표본내 Brier {np.mean(r0**2):.8f}  (🔴 2024 는 다섯 팔 전부의 학습 구간)")
assert mean_gap < 1e-12, "🔴 BN 부모 재구성 실패 — 평균이 target 에 안 맞는다"
print("  🟢 평균이 출하 target 과 1e-12 안에서 일치 — 부모 재구성 검증")

# ── TrackMan 축을 행에 붙인다 ─────────────────────────────────────────
tbl = pd.read_csv(f"{OUT}/tm_table_Y2024.csv", index_col="pitcher_id")
print(f"\n=== 축 부착 (Y=2024 표: season<=2023 · 투수 {len(tbl)}명) ===")
V = {}
COVER = {}
for c in TM_AXES + ["tm_n"]:
    d = {int(k): float(v) for k, v in tbl[c].items()}
    v = np.array([d.get(int(p), np.nan) for p in pid_row], dtype=np.float64)
    V[c] = v
    COVER[c] = {"row_finite": float(np.isfinite(v).mean()),
                "row_missing": int((~np.isfinite(v)).sum())}
    print(f"  {c:<16} 유한 행 {COVER[c]['row_finite']:.6f}"
          f"  결측(무이동) {COVER[c]['row_missing']:,}행")
V["log_tm_n"] = np.log(np.where(np.isfinite(V["tm_n"]), V["tm_n"], np.nan))

# 🔴 as-of 재확인 — 표는 season<=2023 으로 만들어졌고 프레임은 season 2024
rep_tbl = json.load(open(f"{OUT}/tables_report.json"))
assert rep_tbl["table_Y2024"]["season_max_used"] <= 2023
print(f"  🟢 as-of: 표 season {rep_tbl['table_Y2024']['season_min_used']}"
      f"~{rep_tbl['table_Y2024']['season_max_used']} < 프레임 season 2024")

# 대조군
CTRL = {"ctrl_asof_psr": RAW_AX["ctrl_asof_psr"]}
rng = np.random.default_rng(123)
pids_sorted = np.array(sorted(tbl.index.astype(int)))
perm_vals = rng.permutation(tbl["tm_rel_sd"].to_numpy(np.float64))
dperm = {int(k): float(v) for k, v in zip(pids_sorted, perm_vals)}
CTRL["perm_shuffle"] = np.array([dperm.get(int(p), np.nan) for p in pid_row],
                                dtype=np.float64)

# ── 스크린 ────────────────────────────────────────────────────────────
blk = np.unique(pid_row, return_inverse=True)[1]
nb = int(blk.max()) + 1
IDX = np.random.default_rng(42).integers(0, nb, size=(1000, nb))
print(f"\n부트스트랩 블록 = 투수 {nb}명 · 1000 재표본 (seed 42)")


def screen(v):
    u, _, _, nnf = unit(v)
    fq = z_BN + EPS_PROBE * u
    anc = brentq(lambda d: expit(fq + d).mean() - float(p_BN.mean()), -5, 5,
                 xtol=1e-15)
    delta = expit(fq + anc) - p_BN
    g = delta * r0
    M = float(g.mean())
    bs = np.bincount(blk, weights=g, minlength=nb)
    se = float(bs[IDX].sum(axis=1).std(ddof=1)) / N_ROWS
    t = M / se if se else 0.0
    return {"M": M, "t": round(float(t), 3), "sign": "+" if M > 0 else "-",
            "n_nonfinite": nnf, "pass_t2": bool(abs(t) >= THRESH)}


print(f"\n{'방향':<20}{'M':>13}{'t':>8}{'부호':>6}{'무이동행':>10}")
RES = {}
for nm in TM_AXES:
    RES[nm] = screen(V[nm])
    print(f"{nm:<20}{RES[nm]['M']:>+13.3e}{RES[nm]['t']:>8.2f}"
          f"{RES[nm]['sign']:>6}{RES[nm]['n_nonfinite']:>10,}"
          f"{'  ◀ |t|>=2' if RES[nm]['pass_t2'] else ''}")
print("  ── 대조군 ──")
for nm, v in CTRL.items():
    RES[nm] = screen(v)
    print(f"{nm:<20}{RES[nm]['M']:>+13.3e}{RES[nm]['t']:>8.2f}"
          f"{RES[nm]['sign']:>6}{RES[nm]['n_nonfinite']:>10,}")
RES["supp_log_tm_n"] = screen(V["log_tm_n"])
print(f"{'supp_log_tm_n':<20}{RES['supp_log_tm_n']['M']:>+13.3e}"
      f"{RES['supp_log_tm_n']['t']:>8.2f}{RES['supp_log_tm_n']['sign']:>6}"
      f"{RES['supp_log_tm_n']['n_nonfinite']:>10,}   (보조 진단 — 등록 축 아님)")

PASSED = [n for n in TM_AXES if RES[n]["pass_t2"]]
print(f"\n통과 축: {PASSED if PASSED else '없음'}")

# ── 쌍상관 (유한 교집합에서) ──────────────────────────────────────────
ALL = {n: V[n] for n in TM_AXES} | {"ctrl_asof_psr": CTRL["ctrl_asof_psr"],
                                    "log_tm_n": V["log_tm_n"]}
names = list(ALL)
PC = {}
for i, a in enumerate(names):
    for b in names[i + 1:]:
        m = np.isfinite(ALL[a]) & np.isfinite(ALL[b])
        PC[f"{a}|{b}"] = round(float(np.corrcoef(ALL[a][m], ALL[b][m])[0, 1]), 4)
print("\n=== 쌍상관 |r| >= 0.30 만 표시 (전체는 metrics.json) ===")
for k, v in sorted(PC.items(), key=lambda x: -abs(x[1])):
    if abs(v) >= 0.30:
        print(f"  {k:<40} r {v:+.4f}")

# ── 🔴 잔차화 검사 ────────────────────────────────────────────────────
def resid_on(target, controls):
    """target 을 controls 에 최소제곱 잔차화.  전 행 유한 교집합에서 적합하고
    유한 행에만 잔차를 남긴다 (비유한 행은 NaN 유지 = 무이동)."""
    m = np.isfinite(target)
    for c in controls:
        m &= np.isfinite(c)
    X = np.column_stack([np.ones(m.sum())] + [c[m] for c in controls])
    beta, *_ = np.linalg.lstsq(X, target[m], rcond=None)
    out = np.full(len(target), np.nan)
    out[m] = target[m] - X @ beta
    return out


RESID = {}
if PASSED:
    print("\n=== 🔴 잔차화 검사 (본론) ===")
    print(f"{'축':<20}{'corr(ctrl)':>11}{'t_원':>8}{'t_|ctrl':>9}"
          f"{'t_|ctrl+타축':>13}")
    for nm in PASSED:
        others = [V[o] for o in PASSED if o != nm]
        rc = resid_on(V[nm], [CTRL["ctrl_asof_psr"]])
        ra = resid_on(V[nm], [CTRL["ctrl_asof_psr"]] + others)
        sc, sa = screen(rc), screen(ra)
        m = np.isfinite(V[nm]) & np.isfinite(CTRL["ctrl_asof_psr"])
        r_ctrl = float(np.corrcoef(V[nm][m], CTRL["ctrl_asof_psr"][m])[0, 1])
        RESID[nm] = {"corr_with_ctrl": round(r_ctrl, 4),
                     "t_raw": RES[nm]["t"], "t_resid_ctrl": sc["t"],
                     "M_resid_ctrl": sc["M"],
                     "t_resid_ctrl_and_others": sa["t"],
                     "M_resid_ctrl_and_others": sa["M"],
                     "sign_raw": RES[nm]["sign"],
                     "sign_resid_ctrl": sc["sign"],
                     "sign_resid_all": sa["sign"],
                     "survives_ctrl": bool(abs(sc["t"]) >= THRESH),
                     "survives_all": bool(abs(sa["t"]) >= THRESH)}
        print(f"{nm:<20}{r_ctrl:>+11.4f}{RES[nm]['t']:>8.2f}{sc['t']:>9.2f}"
              f"{sa['t']:>13.2f}"
              f"{'  ✅ 생존' if abs(sa['t']) >= THRESH else '  ❌ 붕괴'}")
else:
    print("\n=== 잔차화 검사 — 통과 축이 없어 수행하지 않음 ===")

# ── metrics ───────────────────────────────────────────────────────────
M = {
    "exp_id": "exp_073", "date": "2026-08-21",
    "protocol": {
        "frame": "tr2024 보정 프레임 253,507행 (season 2024 단독)",
        "parent": "BN (LB 1115.9165956458) = expit(alpha*z'+beta_skew*z'^2+d_adj)",
        "parent_consts": {"alpha": A, "beta_skew": B, "d_adj": D_BN,
                          "target_rate": T, "w_v6": W_OUT, "c_aa": C_AA,
                          "smooth_weights": SW,
                          "axis_u": {a["name"]: a["u"] for a in AXL}},
        "parent_mean": float(p_BN.mean()), "parent_mean_gap_to_target": mean_gap,
        "parent_in_sample_brier": float(np.mean(r0 ** 2)),
        "probe": "z_BN + eps*u, u=(v-mean)/sd on finite, nonfinite->0",
        "eps": EPS_PROBE, "anchor": "mean(p_BN) preserved (brentq)",
        "stat": "M = mean(delta*r0)",
        "bootstrap": "pitcher-block, 1000 reps, rng seed 42, shared IDX",
        "n_blocks": nb, "threshold_registered": "|t| >= 2 (사전 등록)",
        "size_use": "none — 부호와 t 만 읽는다 (S1)"},
    "screen": RES, "passed_raw": PASSED, "residualization": RESID,
    "pairwise_corr": PC, "row_coverage": COVER,
    "tables": {"Y2024": rep_tbl["table_Y2024"], "Y2025": rep_tbl["table_Y2025"],
               "map571_sha256": rep_tbl["map571_sha256"],
               "row_cov_571_of_trackman": rep_tbl["row_cov_571_of_trackman"],
               "row_cov_685_of_trackman": rep_tbl["row_cov_685_of_trackman"],
               "missing_rate_raw": rep_tbl["missing_rate_raw"]},
    "sha256": SHA,
    "wall_s_s2": round(time.time() - t0, 1),
}
json.dump(M, open(f"{OUT}/screen.json", "w"), indent=1, ensure_ascii=False)
print(f"\nscreen.json 저장 · s2 {time.time()-t0:.0f}s")
