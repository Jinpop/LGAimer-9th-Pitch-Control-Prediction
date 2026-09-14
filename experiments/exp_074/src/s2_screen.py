"""exp_074 stage 2/3 — BN 잔차 위 부호 스크린 + 🔴 잔차화 4종 (본론).

프로토콜 (exp_073 = exp_059 형식 승계, 사전 등록 · 한 글자도 안 바꿨다)
    방향 v -> 유한값에서 중심화·1sd(모집단) 정규화 u, 비유한 행은 0 = 무이동
    프로브 f = z_BN + 0.02*u,  앵커는 mean(p_BN) 유지 (brentq)
    delta = expit(f + anc) - p_BN,  M = mean(delta * r0),  r0 = y - p_BN
    t = M / se,  se = 투수 블록 부트스트랩 1000회 (seed 42, 전 축 공통 IDX)

🔴 사전 등록 문턱 (S8 — 결과를 보고 바꾸지 않는다)
    원 |t| >= 2  **그리고**  잔차화 3종 전부 통과 후에도 |t| >= 2.

🔴 잔차화가 이 실험의 본론이다.  네 종류 전부:
    R1  ctrl_prev3_suc  (주최측이 이미 준 최근경기 폼) — 핵심 비교 상대.
                         여기서 죽으면 '같은 정보의 다른 포장' 이다.
    R2  ctrl_asof_psr   (커리어) — 통과가 정상인 교정용
    R3  ins_p_success · ins_p_success_dev (시즌 내 폼) — exp_048 이 이미 산 정보와
                         겹치면 AN 꼴이 난다 (팔 +41.13 -> 배포 -1.23)
    R4  R1+R2+R3 + 통과한 다른 축 전부 (서로에 대해)

🔴 S1  tr2024 표본내 잔차 스크린이다.  정직면이 아니다.  점수로 환산하지 않는다.
🔴 S15 부호를 프레임에서 확정하지 않는다.  방향은 LB 로 산다.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit

t0 = time.time()
H = "/home/<user>/SungjinPark"
OUT = f"{H}/exp074_fm"
FRAME = f"{H}/build_AC/work/tr2024"
PKG = f"{H}/build_BN/pkg_BN"
ANCHOR_LT2024 = f"{H}/build_AI/src/ins/anchor_lt2024.npz"
EPS, EPS_PROBE, N_ROWS, THRESH = 1e-6, 0.02, 253507, 2.0
AXES = ["fm_rev_g5", "fm_ball_g5", "fm_strike_g5", "fm_suc_sd5",
        "fm_suc_sd10", "fm_suc_trend"]
SUPP_AXES = ["supp_fm_suc_g5", "supp_fm_mid_g5", "fm_ngames", "fm_npitch"]

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
    ANCHOR_LT2024:
        "0e4b586da41cd6e78c5e19d4dd656bc9df088413625945e2a88ae85d100e5eb7",
    f"{PKG}/inseason.py":
        "1b4a407d2b53f2cbdbf0d2c837d0cab7c3db5e1e01bfc2e6b639813723303086",
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
SHA["exp074_fm/fm_table_Y2024.csv"] = sha_file(f"{OUT}/fm_table_Y2024.csv")

sys.path.insert(0, PKG)
import inseason as INS       # noqa: E402  (원본 그대로)

# ── BN 부모 재구성 (assemble_BN 과 같은 결합 순서) ──────────────────────
bl = json.load(open(f"{PKG}/blend.json", encoding="utf-8"))
A = float(bl["alpha"]); B = float(bl["beta_skew"])
D_BN = float(bl["d_adj"]); T = float(bl["target_rate"])
W_OUT = float(bl["w_v6"]); C_AA = float(bl["c_aa"])
SW, AXL = bl["smooth_weights"], bl["axis_offsets"]
assert [a["name"] for a in AXL] == ["beta_ins_off", "ctrl_asof_psr"]

D59 = np.load(f"{H}/exp059_screen/directions.npz", allow_pickle=True)
fr = pd.read_csv(f"{FRAME}/test.csv", encoding="utf-8-sig")
assert len(fr) == N_ROWS
assert list(fr["row_id"]) == list(D59["row_id"]), "프레임 행 순서 불일치"
assert set(fr["season"].unique()) == {2024}, "프레임이 season 2024 단독이 아니다"
pid_row = fr["pitcher_id"].to_numpy(np.int64)
assert np.array_equal(pid_row, D59["pitcher_id"].astype(np.int64))
y = D59["y"].astype(np.float64)


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

z = (W_OUT * lg(np.load(P_V19)) + C_AA * lg(np.load(f"{H}/build_AY/work/pcal_aa.npy"))
     + float(SW["S2"]) * lg(np.load(f"{H}/build_AY/work/pcal_S2.npy"))
     + float(SW["S1"]) * lg(np.load(f"{H}/build_AY/work/pcal_S1.npy"))
     + float(SW["T70"]) * lg(np.load(f"{H}/build_AY/work/pcal_T70.npy")))
zp = z + float(AXL[0]["u"]) * DEL["beta_ins_off"] \
     + float(AXL[1]["u"]) * DEL["ctrl_asof_psr"]
z_BN = A * zp + B * zp * zp + D_BN
p_BN = expit(z_BN)
r0 = y - p_BN
mean_gap = abs(float(p_BN.mean()) - T)
print(f"\n=== BN 부모 재구성 ===")
print(f"  mean(p_BN) {p_BN.mean():.17f}  target {T!r}  |차| {mean_gap:.3e}")
print(f"  표본내 Brier {np.mean(r0**2):.8f}  (🔴 2024 는 다섯 팔 전부의 학습 구간)")
assert mean_gap < 1e-12, "🔴 BN 부모 재구성 실패"
print("  🟢 평균이 출하 target 과 1e-12 안에서 일치 — 부모 재구성 검증")

# ── 축 부착 (투수당 스칼라 · Y=2024 표 = season<=2023) ─────────────────
tbl = pd.read_csv(f"{OUT}/fm_table_Y2024.csv", index_col="pitcher_id")
rep_tbl = json.load(open(f"{OUT}/tables_report.json", encoding="utf-8"))
assert rep_tbl["table_Y2024"]["season_max_used"] <= 2023, "🔴 as-of 위반"
print(f"\n=== 축 부착 (Y=2024 표: season "
      f"{rep_tbl['table_Y2024']['season_min_used']}~"
      f"{rep_tbl['table_Y2024']['season_max_used']} < 프레임 season 2024, "
      f"투수 {len(tbl)}명) ===")
V, COVER = {}, {}
for c in AXES + SUPP_AXES:
    d = {int(k): float(v) for k, v in tbl[c].items()}
    v = np.array([d.get(int(p), np.nan) for p in pid_row], dtype=np.float64)
    V[c] = v
    COVER[c] = {"row_finite": float(np.isfinite(v).mean()),
                "row_missing": int((~np.isfinite(v)).sum())}
    print(f"  {c:<16} 유한 행 {COVER[c]['row_finite']:.6f}"
          f"  결측(무이동) {COVER[c]['row_missing']:,}행")
V["log_fm_npitch"] = np.log(np.where(np.isfinite(V["fm_npitch"]),
                                     V["fm_npitch"], np.nan))

# ── 대조군 ────────────────────────────────────────────────────────────
ctrl_prev3 = fr["asof_pitcher_prev3_game_success_rate"].to_numpy(np.float64)
CTRL_PSR_IS_FRAME_COL = bool(np.allclose(
    fr["asof_pitcher_success_rate"].to_numpy(np.float64),
    RAW_AX["ctrl_asof_psr"], equal_nan=True))
print(f"  ctrl_asof_psr == 프레임 asof_pitcher_success_rate: "
      f"{CTRL_PSR_IS_FRAME_COL}")
INSF = INS.apply(fr, INS.load(ANCHOR_LT2024))
ins_suc = INSF["ins_p_success"].to_numpy(np.float64)
ins_dev = INSF["ins_p_success_dev"].to_numpy(np.float64)
print(f"\n=== 대조군 ===")
print(f"  ctrl_prev3_suc  유한 {np.isfinite(ctrl_prev3).mean():.6f}"
      f"  (주최측 asof_pitcher_prev3_game_success_rate · 행 수준)")
print(f"  ctrl_asof_psr   유한 {np.isfinite(RAW_AX['ctrl_asof_psr']).mean():.6f}")
print(f"  ins_p_success   유한 {np.isfinite(ins_suc).mean():.6f}  "
      f"(앵커 season<2024, 원본 inseason.py)")

rng = np.random.default_rng(123)
pids_sorted = np.array(sorted(tbl.index.astype(int)))
perm_vals = rng.permutation(tbl["fm_rev_g5"].to_numpy(np.float64))
dperm = {int(k): float(v) for k, v in zip(pids_sorted, perm_vals)}
CTRL = {"ctrl_prev3_suc": ctrl_prev3,
        "ctrl_asof_psr": RAW_AX["ctrl_asof_psr"],
        "perm_shuffle": np.array([dperm.get(int(p), np.nan) for p in pid_row],
                                 dtype=np.float64)}

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
    return {"M": M, "se": se, "t": round(float(t), 3),
            "sign": "+" if M > 0 else "-", "n_nonfinite": nnf,
            "pass_t2": bool(abs(t) >= THRESH)}


print(f"\n{'방향':<20}{'M':>13}{'t':>8}{'부호':>6}{'무이동행':>10}")
RES = {}
for nm in AXES:
    RES[nm] = screen(V[nm])
    print(f"{nm:<20}{RES[nm]['M']:>+13.3e}{RES[nm]['t']:>8.2f}"
          f"{RES[nm]['sign']:>6}{RES[nm]['n_nonfinite']:>10,}"
          f"{'  ◀ |t|>=2' if RES[nm]['pass_t2'] else ''}")
print("  ── 대조군 ──")
for nm, v in CTRL.items():
    RES[nm] = screen(v)
    print(f"{nm:<20}{RES[nm]['M']:>+13.3e}{RES[nm]['t']:>8.2f}"
          f"{RES[nm]['sign']:>6}{RES[nm]['n_nonfinite']:>10,}")
print("  ── 보조 진단 (등록 축 아님) ──")
SUPP = {}
for nm in SUPP_AXES + ["log_fm_npitch"]:
    SUPP[nm] = screen(V[nm])
    print(f"{nm:<20}{SUPP[nm]['M']:>+13.3e}{SUPP[nm]['t']:>8.2f}"
          f"{SUPP[nm]['sign']:>6}{SUPP[nm]['n_nonfinite']:>10,}")
SUPP["ins_p_success"] = screen(ins_suc)
SUPP["ins_p_success_dev"] = screen(ins_dev)

PASSED = [n for n in AXES if RES[n]["pass_t2"]]
print(f"\n원 t 통과 축: {PASSED if PASSED else '없음'}")

# ── 쌍상관 (유한 교집합에서) ──────────────────────────────────────────
ALL = ({n: V[n] for n in AXES + SUPP_AXES}
       | {"ctrl_prev3_suc": CTRL["ctrl_prev3_suc"],
          "ctrl_asof_psr": CTRL["ctrl_asof_psr"],
          "ins_p_success": ins_suc, "ins_p_success_dev": ins_dev})
names = list(ALL)
PC = {}
for i, a in enumerate(names):
    for b in names[i + 1:]:
        m = np.isfinite(ALL[a]) & np.isfinite(ALL[b])
        PC[f"{a}|{b}"] = (round(float(np.corrcoef(ALL[a][m], ALL[b][m])[0, 1]), 4)
                          if m.sum() > 2 else None)
print("\n=== 쌍상관 |r| >= 0.30 만 표시 (전체는 metrics.json) ===")
for k, v in sorted(PC.items(), key=lambda x: -abs(x[1] or 0)):
    if v is not None and abs(v) >= 0.30:
        print(f"  {k:<46} r {v:+.4f}")


# ── 🔴 잔차화 검사 (본론) ────────────────────────────────────────────
def resid_on(target, controls):
    """target 을 controls 에 최소제곱 잔차화.  전 열 유한 교집합에서 적합하고
    유한 행에만 잔차를 남긴다 (비유한 행은 NaN 유지 = 무이동)."""
    m = np.isfinite(target)
    for c in controls:
        m &= np.isfinite(c)
    X = np.column_stack([np.ones(m.sum())] + [c[m] for c in controls])
    beta, *_ = np.linalg.lstsq(X, target[m], rcond=None)
    out = np.full(len(target), np.nan)
    out[m] = target[m] - X @ beta
    return out


C1 = [CTRL["ctrl_prev3_suc"]]
C2 = [CTRL["ctrl_asof_psr"]]
C3 = [ins_suc, ins_dev]
RESID = {}
# 🔴 통과 축이 없어도 등록 축 전부에 대해 잔차화를 낸다 (기록용).
#    판정은 PASSED 에만 적용한다.
print("\n=== 🔴 잔차화 검사 (본론) ===")
print(f"{'축':<16}{'t_원':>8}{'t|prev3':>9}{'t|psr':>8}{'t|ins':>8}{'t|전부':>9}")
for nm in AXES:
    others = [V[o] for o in PASSED if o != nm] if nm in PASSED else []
    s1_ = screen(resid_on(V[nm], C1))
    s2_ = screen(resid_on(V[nm], C2))
    s3_ = screen(resid_on(V[nm], C3))
    s4_ = screen(resid_on(V[nm], C1 + C2 + C3 + others))
    surv = all(abs(s["t"]) >= THRESH for s in (s1_, s2_, s3_, s4_))
    RESID[nm] = {
        "t_raw": RES[nm]["t"], "sign_raw": RES[nm]["sign"],
        "R1_vs_ctrl_prev3_suc": {"t": s1_["t"], "M": s1_["M"], "sign": s1_["sign"]},
        "R2_vs_ctrl_asof_psr": {"t": s2_["t"], "M": s2_["M"], "sign": s2_["sign"]},
        "R3_vs_ins_success": {"t": s3_["t"], "M": s3_["M"], "sign": s3_["sign"]},
        "R4_vs_all_plus_passed_others": {"t": s4_["t"], "M": s4_["M"],
                                         "sign": s4_["sign"],
                                         "n_other_axes": len(others)},
        "raw_passed_t2": RES[nm]["pass_t2"],
        "survives_all_residualizations": bool(surv),
        "REGISTERED_VERDICT": bool(RES[nm]["pass_t2"] and surv)}
    mark = ("  ✅ 생존" if RESID[nm]["REGISTERED_VERDICT"]
            else ("  ❌ 잔차화 붕괴" if RES[nm]["pass_t2"] else "  ❌ 원 t 미달"))
    print(f"{nm:<16}{RES[nm]['t']:>8.2f}{s1_['t']:>9.2f}{s2_['t']:>8.2f}"
          f"{s3_['t']:>8.2f}{s4_['t']:>9.2f}{mark}")

SURVIVE = [n for n in AXES if RESID[n]["REGISTERED_VERDICT"]]
print(f"\n🔴 사전 등록 문턱 최종 생존: {SURVIVE if SURVIVE else '없음 (0/%d)' % len(AXES)}")

M = {
    "exp_id": "exp_074", "date": "2026-08-21",
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
        "n_blocks": nb,
        "threshold_registered": ("원 |t| >= 2 AND 잔차화 R1~R4 전부 |t| >= 2 "
                                 "(사전 등록 · S8)"),
        "size_use": "none — 부호와 t 만 읽는다 (S1)"},
    "screen": RES, "supplementary_screen": SUPP,
    "passed_raw": PASSED, "residualization": RESID, "survivors": SURVIVE,
    "pairwise_corr": PC, "row_coverage": COVER,
    "ctrl_asof_psr_equals_frame_column": CTRL_PSR_IS_FRAME_COL,
    "sha256": SHA,
    "wall_s_s2": round(time.time() - t0, 1)}
json.dump(M, open(f"{OUT}/screen.json", "w"), indent=1, ensure_ascii=False)
print(f"\nscreen.json 저장 · s2 {time.time()-t0:.0f}s")
