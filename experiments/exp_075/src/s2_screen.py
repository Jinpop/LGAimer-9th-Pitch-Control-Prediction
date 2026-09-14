"""exp_075 stage 2/3 — BN 잔차 위 부호 스크린 + 🔴 잔차화 4종 (본론).

프로토콜 (exp_074 = exp_073 = exp_059 형식 승계)
    방향 v -> 유한값에서 중심화·1sd(모집단) 정규화 u, 비유한 행은 0 = 무이동
    프로브 f = z_BN + 0.02*u,  앵커는 mean(p_BN) 유지 (brentq)
    delta = expit(f + anc) - p_BN,  M = mean(delta * r0),  r0 = y - p_BN
    t = M / se,  se = 블록 부트스트랩 1000회 (seed 42, 전 축 공통 IDX)

🔴 exp_074 대비 **의도적으로 바꾼 변인 하나** (기록 정확성 규약)
    부트스트랩 블록 = **batter_id** (exp_074 는 pitcher_id).
    근거: 이번 축의 집계 단위가 타자다.  타자 안에서 상수인 축을 투수 블록으로
    재표본하면 타자 군집이 쪼개져 se 가 작아지고 t 가 부풀려진다.
    투수 블록 t 도 **함께** 보고해 두 값을 비교할 수 있게 남긴다.
    등록 판정은 타자 블록 쪽이다 (사전 등록).

🔴 사전 등록 문턱 (S8 — 결과를 보고 바꾸지 않는다)
    원 |t| >= 2  **그리고**  잔차화 R1~R4 전부 |t| >= 2.

🔴 잔차화가 이 실험의 본론이다.
    R1  ctrl_asof_bsr  = 주최측 asof_batter_success_rate — **핵심 비교 상대**
    R2  ctrl_asof_psr  = 주최측 asof_pitcher_success_rate (BJ 가 이미 산 축)
    R3  ctrl_asof_bmr  = 주최측 asof_batter_middle_rate — 준 타자 열 **둘 다**
    R4  R1+R2+R3 + 통과한 다른 축 전부 (서로에 대해)

🔴 사전 등록된 해석 단서 — `bt_d_psr` 은 이 스크린으로 시험할 수 없다
    bt_d_psr = ctrl_asof_psr - ctrl_asof_bsr 는 R1·R2 대조군의 **선형 결합**이다.
    잔차화는 최소제곱이므로 R4(그리고 R1+R2 동시) 잔차가 **해석적으로 0** 이고
    t 도 0 이 된다.  이건 축이 죽었다는 뜻이 아니라 **선형 스크린이 그 축을
    측정할 수 없다**는 뜻이다 (축의 주장은 'CatBoost 의 축 평행 분기가 이 모양을
    못 만든다' 인데, 선형 프로브에게는 그 모양이 새롭지 않다).
    → 결과를 보기 전에 여기 적어 둔다.  R^2 를 함께 재서 1.0 임을 기계로 보인다.
    곱인 bt_x_psr 은 선형 span 밖이므로 이 스크린으로 시험 가능하다.

🔴 S1  tr2024 표본내 잔차 스크린이다.  정직면이 아니다.  점수로 환산하지 않는다.
🔴 S15 부호를 프레임에서 확정하지 않는다.  방향은 LB 로 산다.
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
OUT = f"{H}/exp075_bt"
FRAME = f"{H}/build_AC/work/tr2024"
PKG = f"{H}/build_BN/pkg_BN"
EPS, EPS_PROBE, N_ROWS, THRESH = 1e-6, 0.02, 253507, 2.0
N_PERM = 200

AXES_TABLE = ["bt_rev", "bt_ball", "bt_strike", "bt_suc_sd", "bt_hand_dev"]
AXES_ROW = ["bt_x_psr", "bt_d_psr"]
AXES = AXES_TABLE + AXES_ROW
SUPP_TABLE = ["supp_bt_suc", "supp_bt_mid", "bt_n_rows", "bt_n_blocks"]

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
for t in ("Y2024", "Y2025", "Y2024strict"):
    SHA[f"exp075_bt/bt_table_{t}.csv"] = sha_file(f"{OUT}/bt_table_{t}.csv")

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
bid_row = fr["batter_id"].to_numpy(np.int64)
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
print("\n=== BN 부모 재구성 ===")
print(f"  mean(p_BN) {p_BN.mean():.17f}  target {T!r}  |차| {mean_gap:.3e}")
print(f"  표본내 Brier {np.mean(r0**2):.8f}  (🔴 2024 는 다섯 팔 전부의 학습 구간)")
assert mean_gap < 1e-12, "🔴 BN 부모 재구성 실패"
print("  🟢 평균이 출하 target 과 1e-12 안에서 일치 — 부모 재구성 검증")

# ── 축 부착 ───────────────────────────────────────────────────────────
tbl = pd.read_csv(f"{OUT}/bt_table_Y2024.csv", index_col="batter_id")
tbs = pd.read_csv(f"{OUT}/bt_table_Y2024strict.csv", index_col="batter_id")
rep_tbl = json.load(open(f"{OUT}/tables_report.json", encoding="utf-8"))
assert rep_tbl["table_Y2024"]["season_max_used"] <= 2023, "🔴 as-of 위반"
print(f"\n=== 축 부착 (Y=2024 표: season "
      f"{rep_tbl['table_Y2024']['season_min_used']}~"
      f"{rep_tbl['table_Y2024']['season_max_used']} < 프레임 season 2024, "
      f"타자 {len(tbl)}명) ===")


def attach(tb, c):
    d = {int(k): float(v) for k, v in tb[c].items()}
    return np.array([d.get(int(b), np.nan) for b in bid_row], dtype=np.float64)


ctrl_bsr = fr["asof_batter_success_rate"].to_numpy(np.float64)
ctrl_bmr = fr["asof_batter_middle_rate"].to_numpy(np.float64)
ctrl_psr = fr["asof_pitcher_success_rate"].to_numpy(np.float64)
CTRL_PSR_IS_FRAME_COL = bool(np.allclose(ctrl_psr, RAW_AX["ctrl_asof_psr"],
                                         equal_nan=True))

V, COVER = {}, {}
for c in AXES_TABLE + SUPP_TABLE:
    V[c] = attach(tbl, c)
V["bt_x_psr"] = ctrl_psr * ctrl_bsr
V["bt_d_psr"] = ctrl_psr - ctrl_bsr
V["log_bt_n_rows"] = np.log(np.where(np.isfinite(V["bt_n_rows"]),
                                     V["bt_n_rows"], np.nan))
for c in AXES + SUPP_TABLE:
    v = V[c]
    COVER[c] = {"row_finite": float(np.isfinite(v).mean()),
                "row_missing": int((~np.isfinite(v)).sum())}
    print(f"  {c:<14} 유한 행 {COVER[c]['row_finite']:.6f}"
          f"  결측(무이동) {COVER[c]['row_missing']:,}행")

print(f"\n=== 대조군 ===")
print(f"  ctrl_asof_bsr 유한 {np.isfinite(ctrl_bsr).mean():.6f}  (주최측 타자 성공률)")
print(f"  ctrl_asof_bmr 유한 {np.isfinite(ctrl_bmr).mean():.6f}  (주최측 타자 middle)")
print(f"  ctrl_asof_psr 유한 {np.isfinite(ctrl_psr).mean():.6f}  "
      f"· D59 벡터와 동일 {CTRL_PSR_IS_FRAME_COL}")

# 🔴 결측 보존 순열 (exp_074 에서 NaN 을 섞어 대조군이 오작동했다)
rngp = np.random.default_rng(123)
pv = tbl["bt_rev"].to_numpy(np.float64).copy()
finp = np.isfinite(pv)
pv[finp] = rngp.permutation(pv[finp])
tperm = pd.DataFrame({"bt_rev": pv}, index=tbl.index)
CTRL = {"ctrl_asof_bsr": ctrl_bsr, "ctrl_asof_bmr": ctrl_bmr,
        "ctrl_asof_psr": ctrl_psr,
        "perm_shuffle_missing_preserved": attach(tperm, "bt_rev")}

# ── 블록 부트스트랩 (타자 = 등록, 투수 = 비교) ────────────────────────
blkB = np.unique(bid_row, return_inverse=True)[1]
nbB = int(blkB.max()) + 1
IDXB = np.random.default_rng(42).integers(0, nbB, size=(1000, nbB))
blkP = np.unique(pid_row, return_inverse=True)[1]
nbP = int(blkP.max()) + 1
IDXP = np.random.default_rng(42).integers(0, nbP, size=(1000, nbP))
print(f"\n부트스트랩 블록: 타자 {nbB}명(등록) · 투수 {nbP}명(비교) · 1000회 seed 42")


def screen(v, blk=blkB, nb=nbB, IDX=IDXB, zf=z_BN, p0=p_BN, rr=r0, n=N_ROWS):
    u, _, _, nnf = unit(v)
    fq = zf + EPS_PROBE * u
    anc = brentq(lambda d: expit(fq + d).mean() - float(p0.mean()), -5, 5,
                 xtol=1e-15)
    delta = expit(fq + anc) - p0
    g = delta * rr
    M = float(g.mean())
    bs = np.bincount(blk, weights=g, minlength=nb)
    se = float(bs[IDX].sum(axis=1).std(ddof=1)) / n
    t = M / se if se else 0.0
    return {"M": M, "se": se, "t": round(float(t), 3),
            "sign": "+" if M > 0 else "-", "n_nonfinite": nnf,
            "pass_t2": bool(abs(t) >= THRESH)}


print(f"\n{'방향':<24}{'M':>13}{'t_타자':>9}{'t_투수':>9}{'부호':>6}{'무이동행':>10}")
RES, RES_P = {}, {}
for nm in AXES:
    RES[nm] = screen(V[nm])
    RES_P[nm] = screen(V[nm], blkP, nbP, IDXP)
    print(f"{nm:<24}{RES[nm]['M']:>+13.3e}{RES[nm]['t']:>9.2f}"
          f"{RES_P[nm]['t']:>9.2f}{RES[nm]['sign']:>6}"
          f"{RES[nm]['n_nonfinite']:>10,}"
          f"{'  ◀ |t|>=2' if RES[nm]['pass_t2'] else ''}")
print("  ── 대조군 ──")
for nm, v in CTRL.items():
    RES[nm] = screen(v)
    RES_P[nm] = screen(v, blkP, nbP, IDXP)
    print(f"{nm:<24}{RES[nm]['M']:>+13.3e}{RES[nm]['t']:>9.2f}"
          f"{RES_P[nm]['t']:>9.2f}{RES[nm]['sign']:>6}"
          f"{RES[nm]['n_nonfinite']:>10,}")
print("  ── 보조 진단 (등록 축 아님) ──")
SUPP = {}
for nm in SUPP_TABLE + ["log_bt_n_rows"]:
    SUPP[nm] = screen(V[nm])
    SUPP[nm]["t_pitcher_block"] = screen(V[nm], blkP, nbP, IDXP)["t"]
    print(f"{nm:<24}{SUPP[nm]['M']:>+13.3e}{SUPP[nm]['t']:>9.2f}"
          f"{SUPP[nm]['t_pitcher_block']:>9.2f}{SUPP[nm]['sign']:>6}"
          f"{SUPP[nm]['n_nonfinite']:>10,}")

PASSED = [n for n in AXES if RES[n]["pass_t2"]]
print(f"\n원 t 통과 축(타자 블록): {PASSED if PASSED else '없음'}")

# ── 🔴 선형 span 진단 (사전 등록 단서의 기계 증명) ────────────────────
def r2_on(target, controls):
    m = np.isfinite(target)
    for c in controls:
        m &= np.isfinite(c)
    X = np.column_stack([np.ones(m.sum())] + [c[m] for c in controls])
    beta, *_ = np.linalg.lstsq(X, target[m], rcond=None)
    res = target[m] - X @ beta
    ss = float(((target[m] - target[m].mean()) ** 2).sum())
    return {"r2": round(1.0 - float((res ** 2).sum()) / ss, 10) if ss else None,
            "max_abs_resid": float(np.abs(res).max()), "n": int(m.sum())}


SPAN = {}
for nm in AXES:
    SPAN[nm] = r2_on(V[nm], [ctrl_psr, ctrl_bsr])
print("\n=== 🔴 선형 span 진단: 축 ~ {ctrl_asof_psr, ctrl_asof_bsr} R^2 ===")
for nm in AXES:
    print(f"  {nm:<14} R^2 {SPAN[nm]['r2']}  |잔차|max {SPAN[nm]['max_abs_resid']:.3e}")

# ── 쌍상관 ────────────────────────────────────────────────────────────
ALL = ({n: V[n] for n in AXES + SUPP_TABLE}
       | {"ctrl_asof_bsr": ctrl_bsr, "ctrl_asof_bmr": ctrl_bmr,
          "ctrl_asof_psr": ctrl_psr})
names = list(ALL)
PC = {}
for i, a in enumerate(names):
    for b in names[i + 1:]:
        m = np.isfinite(ALL[a]) & np.isfinite(ALL[b])
        PC[f"{a}|{b}"] = (round(float(np.corrcoef(ALL[a][m], ALL[b][m])[0, 1]), 4)
                          if m.sum() > 2 else None)
print("\n=== 쌍상관 |r| >= 0.30 만 표시 ===")
for k, v in sorted(PC.items(), key=lambda x: -abs(x[1] or 0)):
    if v is not None and abs(v) >= 0.30:
        print(f"  {k:<50} r {v:+.4f}")


# ── 🔴 잔차화 검사 (본론) ────────────────────────────────────────────
def resid_on(target, controls):
    m = np.isfinite(target)
    for c in controls:
        m &= np.isfinite(c)
    X = np.column_stack([np.ones(m.sum())] + [c[m] for c in controls])
    beta, *_ = np.linalg.lstsq(X, target[m], rcond=None)
    out = np.full(len(target), np.nan)
    out[m] = target[m] - X @ beta
    return out


C1, C2, C3 = [ctrl_bsr], [ctrl_psr], [ctrl_bmr]
RESID = {}
print("\n=== 🔴 잔차화 검사 (본론 · 타자 블록 t) ===")
print(f"{'축':<14}{'t_원':>8}{'t|bsr':>8}{'t|psr':>8}{'t|bmr':>8}{'t|전부':>9}")
for nm in AXES:
    others = [V[o] for o in PASSED if o != nm] if nm in PASSED else []
    s1_ = screen(resid_on(V[nm], C1))
    s2_ = screen(resid_on(V[nm], C2))
    s3_ = screen(resid_on(V[nm], C3))
    s4_ = screen(resid_on(V[nm], C1 + C2 + C3 + others))
    surv = all(abs(s["t"]) >= THRESH for s in (s1_, s2_, s3_, s4_))
    RESID[nm] = {
        "t_raw": RES[nm]["t"], "sign_raw": RES[nm]["sign"],
        "t_raw_pitcher_block": RES_P[nm]["t"],
        "R1_vs_ctrl_asof_bsr": {"t": s1_["t"], "M": s1_["M"], "sign": s1_["sign"]},
        "R2_vs_ctrl_asof_psr": {"t": s2_["t"], "M": s2_["M"], "sign": s2_["sign"]},
        "R3_vs_ctrl_asof_bmr": {"t": s3_["t"], "M": s3_["M"], "sign": s3_["sign"]},
        "R4_vs_all_plus_passed_others": {"t": s4_["t"], "M": s4_["M"],
                                         "sign": s4_["sign"],
                                         "n_other_axes": len(others)},
        "linear_span_r2_on_psr_bsr": SPAN[nm]["r2"],
        "testable_by_linear_screen": bool((SPAN[nm]["r2"] or 0) < 0.999999),
        "raw_passed_t2": RES[nm]["pass_t2"],
        "survives_all_residualizations": bool(surv),
        "REGISTERED_VERDICT": bool(RES[nm]["pass_t2"] and surv)}
    mark = ("  ✅ 생존" if RESID[nm]["REGISTERED_VERDICT"]
            else ("  ❌ 잔차화 붕괴" if RES[nm]["pass_t2"] else "  ❌ 원 t 미달"))
    if not RESID[nm]["testable_by_linear_screen"]:
        mark += "  🔴 선형 span 안 — 시험 불가(사전 등록 단서)"
    print(f"{nm:<14}{RES[nm]['t']:>8.2f}{s1_['t']:>8.2f}{s2_['t']:>8.2f}"
          f"{s3_['t']:>8.2f}{s4_['t']:>9.2f}{mark}")

SURVIVE = [n for n in AXES if RESID[n]["REGISTERED_VERDICT"]]
print(f"\n🔴 사전 등록 문턱 최종 생존: {SURVIVE if SURVIVE else '없음 (0/%d)' % len(AXES)}")

# ── 🔴 결측 보존 순열 200회 — 경험적 귀무분포 ─────────────────────────
print(f"\n=== 🔴 경험적 귀무분포 — 결측 보존 순열 {N_PERM}회 (타자 블록) ===")
NULL = {}
rngn = np.random.default_rng(2024)
bid_tab = tbl.index.to_numpy(np.int64)
for c in ["bt_rev", "bt_hand_dev", "supp_bt_suc"]:
    vals = tbl[c].to_numpy(np.float64)
    fin = np.isfinite(vals)
    ts = []
    for _ in range(N_PERM):
        q = vals.copy()
        q[fin] = rngn.permutation(vals[fin])
        d = {int(k): float(x) for k, x in zip(bid_tab, q)}
        vv = np.array([d.get(int(b), np.nan) for b in bid_row], np.float64)
        ts.append(screen(vv)["t"])
    ts = np.array(ts, np.float64)
    NULL[c] = {"n_perm": N_PERM, "t_mean": float(ts.mean()),
               "t_sd": float(ts.std(ddof=1)),
               "abs_t_q50": float(np.quantile(np.abs(ts), 0.50)),
               "abs_t_q95": float(np.quantile(np.abs(ts), 0.95)),
               "frac_abs_t_ge_2": float((np.abs(ts) >= 2).mean()),
               "t_min": float(ts.min()), "t_max": float(ts.max()),
               "observed_t": RES.get(c, SUPP.get(c, {})).get("t")}
    print(f"  {c:<14} 귀무 t 평균 {ts.mean():+.3f} sd {ts.std(ddof=1):.3f}  "
          f"|t|95분위 {np.quantile(np.abs(ts),0.95):.2f}  "
          f"P(|t|>=2) {(np.abs(ts)>=2).mean():.3f}  관측 t "
          f"{NULL[c]['observed_t']}")

# ── 보조 진단 ─────────────────────────────────────────────────────────
print("\n=== 보조 진단 ===")
D = {}
D["power_M_needed_for_t2"] = {c: round(2 * RES[c]["se"], 12) for c in AXES}
D["covered_only"] = {}
for c in AXES:
    m = np.isfinite(V[c])
    if m.sum() < 1000:
        continue
    b2 = np.unique(bid_row[m], return_inverse=True)[1]
    nb2 = int(b2.max()) + 1
    I2 = np.random.default_rng(42).integers(0, nb2, size=(1000, nb2))
    s = screen(V[c][m], b2, nb2, I2, z_BN[m], p_BN[m], r0[m], int(m.sum()))
    D["covered_only"][c] = {"n_rows": int(m.sum()), "n_blocks": nb2,
                            "M": s["M"], "t": s["t"]}
    print(f"  커버행만 {c:<14} 행 {int(m.sum()):,} 타자 {nb2}  t {s['t']:+.2f}")
obs = pd.DataFrame({"bid": bid_row, "y": y}).groupby("bid")["y"].mean()
D["model_free"] = {}
for c in AXES_TABLE + SUPP_TABLE:
    d = tbl[c]
    common = obs.index.intersection(d.index)
    a_ = d.loc[common].to_numpy(np.float64)
    b_ = obs.loc[common].to_numpy(np.float64)
    mm = np.isfinite(a_) & np.isfinite(b_)
    u_, _, _, _ = unit(V[c])
    fin = np.isfinite(V[c])
    D["model_free"][c] = {
        "n_batters": int(mm.sum()),
        "corr_batter_level_vs_obs_success":
            round(float(np.corrcoef(a_[mm], b_[mm])[0, 1]), 4) if mm.sum() > 2 else None,
        "corr_row_level_u_vs_y":
            round(float(np.corrcoef(u_[fin], y[fin])[0, 1]), 4)}
D["parent_coupling"] = {}
for c in AXES:
    fin = np.isfinite(V[c])
    u_, _, _, _ = unit(V[c])
    m3 = fin & np.isfinite(ctrl_bsr)
    D["parent_coupling"][c] = {
        "corr_u_vs_logit_pBN": round(float(np.corrcoef(u_[fin], z_BN[fin])[0, 1]), 4),
        "corr_raw_vs_ctrl_asof_bsr":
            round(float(np.corrcoef(V[c][m3], ctrl_bsr[m3])[0, 1]), 4)}

# ── 🔴 경계 보수판 재스크린 ───────────────────────────────────────────
print("\n=== 🔴 시즌 경계 보수판 재스크린 ===")
STRICT = {}
print(f"{'축':<14}{'t_원판':>9}{'t_보수판':>11}{'표최대차':>12}")
ci = tbl.index.intersection(tbs.index)
for c in AXES_TABLE:
    vs_ = attach(tbs, c)
    so = screen(V[c]); ss = screen(vs_)
    x = tbl.loc[ci, c].to_numpy(np.float64)
    w = tbs.loc[ci, c].to_numpy(np.float64)
    m = np.isfinite(x) & np.isfinite(w)
    STRICT[c] = {"t_original": so["t"], "t_strict": ss["t"],
                 "M_original": so["M"], "M_strict": ss["M"],
                 "max_abs_table_diff": (float(np.abs(x[m] - w[m]).max())
                                        if m.sum() else None),
                 "n_batters_compared": int(m.sum()),
                 "verdict_same": bool((abs(so["t"]) >= THRESH) ==
                                      (abs(ss["t"]) >= THRESH))}
    print(f"{c:<14}{so['t']:>9.2f}{ss['t']:>11.2f}"
          f"{STRICT[c]['max_abs_table_diff']:>12.2e}")
STRICT_SAME = all(v["verdict_same"] for v in STRICT.values())
print(f"  원판/보수판 판정 동일: {STRICT_SAME}")

M = {
    "exp_id": "exp_075", "date": "2026-08-21",
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
        "bootstrap_registered": "batter-block, 1000 reps, rng seed 42, shared IDX",
        "bootstrap_secondary": "pitcher-block (exp_074 와 같은 단위 · 비교용)",
        "n_blocks_batter": nbB, "n_blocks_pitcher": nbP,
        "changed_variable_vs_exp074": (
            "부트스트랩 블록 단위 pitcher_id -> batter_id (축의 집계 단위가 타자). "
            "투수 블록 t 도 함께 보고한다."),
        "threshold_registered": ("원 |t| >= 2 AND 잔차화 R1~R4 전부 |t| >= 2 "
                                 "(사전 등록 · S8)"),
        "size_use": "none — 부호와 t 만 읽는다 (S1)"},
    "screen_batter_block": RES, "screen_pitcher_block": RES_P,
    "supplementary_screen": SUPP,
    "passed_raw": PASSED, "residualization": RESID, "survivors": SURVIVE,
    "linear_span_r2_on_psr_bsr": SPAN,
    "empirical_null_missing_preserved": NULL,
    "diagnostics": D,
    "strict_boundary_rescreen": STRICT,
    "strict_boundary_verdict_unchanged": bool(STRICT_SAME),
    "pairwise_corr": PC, "row_coverage": COVER,
    "ctrl_asof_psr_equals_frame_column": CTRL_PSR_IS_FRAME_COL,
    "sha256": SHA,
    "wall_s_s2": round(time.time() - t0, 1)}
json.dump(M, open(f"{OUT}/screen.json", "w"), indent=1, ensure_ascii=False)
print(f"\nscreen.json 저장 · s2 {time.time()-t0:.0f}s")
