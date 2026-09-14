"""exp_076 stage 2/3 — BN 잔차 위 부호 스크린 + 🔴 잔차화 (본론).

프로토콜 (exp_074 = exp_073 = exp_059 형식 승계 — 한 글자도 안 바꿨다)
    방향 v -> 유한값에서 중심화·1sd(모집단) 정규화 u, 비유한 행은 0 = 무이동
    프로브 f = z_BN + 0.02*u,  앵커는 mean(p_BN) 유지 (brentq)
    delta = expit(f + anc) - p_BN,  M = mean(delta * r0),  r0 = y - p_BN
    t = M / se,  se = **투수 블록** 부트스트랩 1000회 (seed 42, 전 축 공통 IDX)

🔴 exp_075 대비 의도적으로 되돌린 변인 하나 (기록 정확성 규약)
    부트스트랩 블록 = pitcher_id.  exp_075 는 축의 집계 단위가 타자라 batter_id
    였다.  이번 축은 전부 투수 키라 exp_074 와 같은 pitcher_id 로 돌아간다.
    비교용으로 타자 블록 t 도 함께 낸다.

🔴 사전 등록 문턱 (S8 — 결과를 보고 바꾸지 않는다)
    원 |t| >= 2  **그리고**  R1~R4 전부 통과 후에도 |t| >= 2.

🔴 잔차화 (config 지정)
    R1  ctrl_prev3_suc  (주최측 원본 · exp_074 에서 t -3.95 로 **살아 있는 열**)
    R2  ctrl_asof_psr   (주최측 커리어)
    R3  ins_p_success · ins_p_success_dev · ins_p_middle · ins_p_ball ·
        ins_p_strike · ins_p_reverse  — exp_048 이 이미 산 정보.  **1차 관문**
    R4  R1+R2+R3 + 통과한 다른 축 전부

보조 잔차화 (등록 아님 · 더 엄한 쪽으로만 추가한다 — 문턱을 낮추지 않는다)
    R5_supp  주최측 prev 6열 전부
    R6_supp  이 축이 쓴 **동결 앵커 스칼라 자체**
    R7_supp  ins_p_* 투수 10열 전부 + R1 + R2

════════════════════════════════════════════════════════════════════════
🔴 사전 등록된 해석 단서 — 결과를 보기 전에 적는다
════════════════════════════════════════════════════════════════════════
`inseason.py` 는 prior 를 이렇게 쓴다:
      ins_p_r      = (k + K*prior) / (ins_n + K)
      ins_p_r_dev  = ins_p_r - prior
따라서 **prior(= 동결 커리어 앵커) = ins_p_r - ins_p_r_dev 로 정확히 복원된다.**
등록 대조군에 ins_p_success 와 ins_p_success_dev 가 **둘 다** 들어 있으므로:

  (a) rg_prev3_suc_dev = prev3 - anc_suc 는 {ctrl_prev3_suc, ins_p_success,
      ins_p_success_dev} 의 **선형결합**이다 (카운트 반올림 <=0.5/n 과 신인
      대체값만 예외).  → R4 가 해석적으로 죽인다.  이건 '축이 죽었다' 가
      아니라 **선형 스크린이 이 축을 측정할 수 없다**는 뜻이다 (exp_075 의
      bt_d_psr 과 같은 상황).  R^2 로 기계 증명한다.
  (b) rg_prev1_suc_dev · rg_prev5_suc_dev 는 앵커 쪽만 span 안이고 prev1·prev5
      는 등록 대조군에 없다 → 시험 가능.
  (c) rg_prev3_mid_dev 의 앵커 anc_mid = ins_p_middle - ins_p_middle_dev 인데
      **ins_p_middle_dev 는 등록 R3 에 없다**(config 목록 6개에 빠져 있다)
      → 앵커 쪽도 부분적으로만 span 안. 시험 가능.  R7_supp 가 그 구멍을 막는다.
  (d) rg_trend = prev1 - prev5 는 등록 대조군 어디에도 없다 → 시험 가능.
      다만 **R5_supp(주최측 prev 6열) 에는 해석적으로 span 안**이라 R^2=1 이
      나올 것이다.  그건 '팀원 팔이 raw 열을 이미 갖고 있다' 와 같은 말이고,
      rg_trend 의 주장(축 평행 분기가 못 만드는 **형태**)은 선형 스크린으로
      반증되지 않는다 — 그 주장은 스태커 열 실험에서만 시험된다.
  (e) 🔴 **구종믹스 4축(rg_fb_dev·rg_br_dev·rg_off_dev·rg_mix_ent) 만이
      등록 대조군 전체의 선형 span 밖에 온전히 있다.**  ins_* 는 success·
      reverse·middle·ball·strike 5종뿐이고 구종 비율을 담지 않는다.
      → 이 실험에서 **선형으로 온전히 시험 가능한 층은 B층**이다.

🔴 검출력 점검 (config 지시)  exp_075 의 0/7 은 검출력 부재였다.
   이번에는 교정 대조군 ctrl_prev3_suc 가 t -3.95 로 켜져 있어야 한다.
   켜졌는지 기계로 확인하고 metrics 에 power_gate 로 남긴다.
   **안 켜지면 0/N 을 '없다' 로 적지 않는다.**

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
OUT = f"{H}/exp076_rg"
FRAME = f"{H}/build_AC/work/tr2024"
PKG = f"{H}/build_BN/pkg_BN"
ANCHOR_LT2024 = f"{H}/build_AI/src/ins/anchor_lt2024.npz"
EPS, EPS_PROBE, N_ROWS, THRESH = 1e-6, 0.02, 253507, 2.0
N_PERM = 200

AX_A = ["rg_prev1_suc_dev", "rg_prev3_suc_dev", "rg_prev5_suc_dev",
        "rg_prev3_mid_dev", "rg_trend"]
AX_B = ["rg_fb_dev", "rg_br_dev", "rg_off_dev", "rg_mix_ent"]
AXES = AX_A + AX_B

PREV_S = [f"asof_pitcher_prev{k}_game_success_rate" for k in (1, 3, 5)]
PREV_M = [f"asof_pitcher_prev{k}_game_middle_rate" for k in (1, 3, 5)]
MIXC = ["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate"]
INS_R3 = ["ins_p_success", "ins_p_success_dev", "ins_p_middle", "ins_p_ball",
          "ins_p_strike", "ins_p_reverse"]
INS_ALL = ["ins_p_n", "ins_p_success", "ins_p_success_dev", "ins_p_reverse",
           "ins_p_reverse_dev", "ins_p_middle", "ins_p_middle_dev",
           "ins_p_ball", "ins_p_ball_dev", "ins_p_strike", "ins_p_strike_dev"]

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
for t in ("Y2024", "Y2025", "Y2024robust"):
    SHA[f"exp076_rg/anchor_{t}.csv"] = sha_file(f"{OUT}/anchor_{t}.csv")

sys.path.insert(0, PKG)
import inseason as INS       # noqa: E402  (원본 그대로 · sha 고정)

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

# ── 앵커 부착 · 축 제작 ───────────────────────────────────────────────
anc = pd.read_csv(f"{OUT}/anchor_Y2024.csv", index_col="pitcher_id")
ancR = pd.read_csv(f"{OUT}/anchor_Y2024robust.csv", index_col="pitcher_id")
rep_a = json.load(open(f"{OUT}/anchors_report.json", encoding="utf-8"))
assert rep_a["anchor_Y2024"]["season_max_used"] <= 2023, "🔴 as-of 위반"
assert rep_a["anchor_Y2024"]["anchor_row_season_max"] <= 2023, "🔴 앵커 행 as-of 위반"
print(f"\n=== 앵커 부착 (season "
      f"{rep_a['anchor_Y2024']['season_min_used']}~"
      f"{rep_a['anchor_Y2024']['season_max_used']} < 프레임 season 2024, "
      f"투수 {rep_a['anchor_Y2024']['n_pitchers']}명) ===")


def attach(tb, c):
    d = {int(k): float(v) for k, v in tb[c].items()}
    return np.array([d.get(int(p), np.nan) for p in pid_row], dtype=np.float64)


COL = {c: fr[c].to_numpy(np.float64)
       for c in PREV_S + PREV_M + MIXC + ["asof_pitcher_success_rate",
                                          "asof_pitcher_pitchmix_n"]}


def entropy3(a, b, c):
    P = np.stack([a, b, c], 1)
    bad = ~np.isfinite(P).all(1)
    Q = np.where(np.isfinite(P) & (P > 0), P, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(np.isfinite(Q), -Q * np.log(Q), 0.0)
    out = term.sum(1)
    out[bad] = np.nan
    return out


def make_axes(anchor_tbl):
    a_suc = attach(anchor_tbl, "anc_suc")
    a_mid = attach(anchor_tbl, "anc_mid")
    a_fb = attach(anchor_tbl, "anc_fb")
    a_br = attach(anchor_tbl, "anc_br")
    a_off = attach(anchor_tbl, "anc_off")
    V = {
        "rg_prev1_suc_dev": COL[PREV_S[0]] - a_suc,
        "rg_prev3_suc_dev": COL[PREV_S[1]] - a_suc,
        "rg_prev5_suc_dev": COL[PREV_S[2]] - a_suc,
        "rg_prev3_mid_dev": COL[PREV_M[1]] - a_mid,
        "rg_trend": COL[PREV_S[0]] - COL[PREV_S[2]],
        "rg_fb_dev": COL[MIXC[0]] - a_fb,
        "rg_br_dev": COL[MIXC[1]] - a_br,
        "rg_off_dev": COL[MIXC[2]] - a_off,
        "rg_mix_ent": entropy3(COL[MIXC[0]], COL[MIXC[1]], COL[MIXC[2]]),
    }
    ANC = {"anc_suc": a_suc, "anc_mid": a_mid, "anc_fb": a_fb,
           "anc_br": a_br, "anc_off": a_off}
    return V, ANC


V, ANC = make_axes(anc)
VR, _ = make_axes(ancR)

# 축이 쓴 동결 앵커 (R6_supp 용)
AXIS_ANCHOR = {"rg_prev1_suc_dev": ["anc_suc"], "rg_prev3_suc_dev": ["anc_suc"],
               "rg_prev5_suc_dev": ["anc_suc"], "rg_prev3_mid_dev": ["anc_mid"],
               "rg_trend": [], "rg_fb_dev": ["anc_fb"], "rg_br_dev": ["anc_br"],
               "rg_off_dev": ["anc_off"], "rg_mix_ent": []}

COVER = {}
for c in AXES:
    v = V[c]
    COVER[c] = {"row_finite": float(np.isfinite(v).mean()),
                "row_missing": int((~np.isfinite(v)).sum())}
    print(f"  {c:<18} 유한 행 {COVER[c]['row_finite']:.6f}"
          f"  결측(무이동) {COVER[c]['row_missing']:,}행")

# ── 대조군 ────────────────────────────────────────────────────────────
ctrl_prev3 = COL[PREV_S[1]]
ctrl_psr = COL["asof_pitcher_success_rate"]
CTRL_PSR_IS_FRAME_COL = bool(np.allclose(ctrl_psr, RAW_AX["ctrl_asof_psr"],
                                         equal_nan=True))
INSF = INS.apply(fr, INS.load(ANCHOR_LT2024))
INSV = {c: INSF[c].to_numpy(np.float64) for c in INS_ALL}
print(f"\n=== 대조군 ===")
print(f"  ctrl_prev3_suc 유한 {np.isfinite(ctrl_prev3).mean():.6f} "
      f"(주최측 원본 · exp_074 t -3.948)")
print(f"  ctrl_asof_psr  유한 {np.isfinite(ctrl_psr).mean():.6f} · "
      f"D59 벡터와 동일 {CTRL_PSR_IS_FRAME_COL}")
print(f"  ins_p_success  유한 {np.isfinite(INSV['ins_p_success']).mean():.6f} "
      f"(앵커 season<2024 · 원본 inseason.py)")

# 🔴 prior 복원 항등 — 사전 등록 단서 (a) 의 기계 증명
prior_rec = INSV["ins_p_success"] - INSV["ins_p_success_dev"]
m_pr = np.isfinite(prior_rec) & np.isfinite(ANC["anc_suc"])
PRIOR_ID = {
    "n_compared": int(m_pr.sum()),
    "max_abs_diff_vs_our_anchor": float(
        np.abs(prior_rec[m_pr] - ANC["anc_suc"][m_pr]).max()),
    "corr": round(float(np.corrcoef(prior_rec[m_pr],
                                    ANC["anc_suc"][m_pr])[0, 1]), 6),
    "note": ("prior = ins_p_success - ins_p_success_dev 로 동결 커리어 앵커가 "
             "정확 복원된다 (inseason.py 정의).  float32 저장·카운트 반올림만큼 "
             "차이가 남는다.  이것이 사전 등록 단서 (a) 의 근거다.")}
print(f"  🔴 prior 복원 항등: 최대차 {PRIOR_ID['max_abs_diff_vs_our_anchor']:.3e} "
      f"· r {PRIOR_ID['corr']}")

# 🔴 결측 보존 순열 (exp_074 에서 NaN 섞은 순열이 +1.96 을 냈다)
rngp = np.random.default_rng(123)
pv = anc["anc_suc"].to_numpy(np.float64).copy()
finp = np.isfinite(pv)
pv[finp] = rngp.permutation(pv[finp])
tperm = pd.DataFrame({"anc_suc": pv}, index=anc.index)
CTRL = {"ctrl_prev3_suc": ctrl_prev3, "ctrl_asof_psr": ctrl_psr,
        "perm_shuffle_missing_preserved":
            COL[PREV_S[1]] - attach(tperm, "anc_suc")}

# ── 블록 부트스트랩 (투수 = 등록, 타자 = 비교) ────────────────────────
blkP = np.unique(pid_row, return_inverse=True)[1]
nbP = int(blkP.max()) + 1
IDXP = np.random.default_rng(42).integers(0, nbP, size=(1000, nbP))
blkB = np.unique(bid_row, return_inverse=True)[1]
nbB = int(blkB.max()) + 1
IDXB = np.random.default_rng(42).integers(0, nbB, size=(1000, nbB))
print(f"\n부트스트랩 블록: 투수 {nbP}명(등록) · 타자 {nbB}명(비교) · 1000회 seed 42")


def screen(v, blk=blkP, nb=nbP, IDX=IDXP, zf=z_BN, p0=p_BN, rr=r0, n=N_ROWS):
    u, _, _, nnf = unit(v)
    fq = zf + EPS_PROBE * u
    a_ = brentq(lambda d: expit(fq + d).mean() - float(p0.mean()), -5, 5,
                xtol=1e-15)
    delta = expit(fq + a_) - p0
    g = delta * rr
    M = float(g.mean())
    bs = np.bincount(blk, weights=g, minlength=nb)
    se = float(bs[IDX].sum(axis=1).std(ddof=1)) / n
    t = M / se if se else 0.0
    return {"M": M, "se": se, "t": round(float(t), 3),
            "sign": "+" if M > 0 else "-", "n_nonfinite": nnf,
            "pass_t2": bool(abs(t) >= THRESH)}


print(f"\n{'방향':<20}{'M':>13}{'t_투수':>9}{'t_타자':>9}{'부호':>6}{'무이동행':>11}")
RES, RES_B = {}, {}
for nm in AXES:
    RES[nm] = screen(V[nm])
    RES_B[nm] = screen(V[nm], blkB, nbB, IDXB)
    print(f"{nm:<20}{RES[nm]['M']:>+13.3e}{RES[nm]['t']:>9.2f}"
          f"{RES_B[nm]['t']:>9.2f}{RES[nm]['sign']:>6}"
          f"{RES[nm]['n_nonfinite']:>11,}"
          f"{'  ◀ |t|>=2' if RES[nm]['pass_t2'] else ''}")
print("  ── 대조군 ──")
for nm, v in CTRL.items():
    RES[nm] = screen(v)
    RES_B[nm] = screen(v, blkB, nbB, IDXB)
    print(f"{nm:<20}{RES[nm]['M']:>+13.3e}{RES[nm]['t']:>9.2f}"
          f"{RES_B[nm]['t']:>9.2f}{RES[nm]['sign']:>6}"
          f"{RES[nm]['n_nonfinite']:>11,}")

# 🔴 검출력 게이트 — config 지시
POWER = {
    "ctrl_prev3_suc_t_here": RES["ctrl_prev3_suc"]["t"],
    "ctrl_prev3_suc_t_exp074": -3.948,
    "ctrl_asof_psr_t_here": RES["ctrl_asof_psr"]["t"],
    "ctrl_asof_psr_t_exp074": -5.45,
    "calibration_control_lit": bool(abs(RES["ctrl_prev3_suc"]["t"]) >= THRESH),
    "verdict": None}
POWER["verdict"] = ("검출력 있음 — 교정 대조군이 |t|>=2 로 켜졌다"
                    if POWER["calibration_control_lit"] else
                    "🔴 검출력 없음 — 0/N 을 '신호 없음' 으로 읽으면 안 된다")
print(f"\n🔴 검출력 게이트: ctrl_prev3_suc t {RES['ctrl_prev3_suc']['t']:+.3f} "
      f"(exp_074 -3.948) → {POWER['verdict']}")

# ── 보조 진단 스크린 (등록 축 아님) ──────────────────────────────────
SUPP_V = {"supp_anc_suc": ANC["anc_suc"], "supp_anc_mid": ANC["anc_mid"],
          "supp_anc_fb": ANC["anc_fb"], "supp_anc_br": ANC["anc_br"],
          "supp_anc_off": ANC["anc_off"],
          "supp_prev1_suc": COL[PREV_S[0]], "supp_prev5_suc": COL[PREV_S[2]],
          "supp_prev1_mid": COL[PREV_M[0]], "supp_prev3_mid": COL[PREV_M[1]],
          "supp_prev5_mid": COL[PREV_M[2]],
          "supp_fastball": COL[MIXC[0]], "supp_breaking": COL[MIXC[1]],
          "supp_offspeed": COL[MIXC[2]],
          "supp_log_pitchmix_n": np.log1p(COL["asof_pitcher_pitchmix_n"]),
          "supp_ins_p_success": INSV["ins_p_success"],
          "supp_ins_p_success_dev": INSV["ins_p_success_dev"]}
print("\n  ── 보조 진단 (등록 축 아님 — 🔴 supp_anc_* 는 '순수 동결 스칼라') ──")
SUPP = {}
for nm, v in SUPP_V.items():
    SUPP[nm] = screen(v)
    SUPP[nm]["t_batter_block"] = screen(v, blkB, nbB, IDXB)["t"]
    print(f"{nm:<24}{SUPP[nm]['M']:>+13.3e}{SUPP[nm]['t']:>9.2f}"
          f"{SUPP[nm]['t_batter_block']:>9.2f}{SUPP[nm]['sign']:>6}"
          f"{SUPP[nm]['n_nonfinite']:>11,}")

PASSED = [n for n in AXES if RES[n]["pass_t2"]]
print(f"\n원 t 통과 축(투수 블록): {PASSED if PASSED else '없음'}")


# ── 선형 span 진단 ────────────────────────────────────────────────────
def _fitmask(target, controls):
    m = np.isfinite(target)
    for c in controls:
        m &= np.isfinite(c)
    return m


def r2_on(target, controls):
    m = _fitmask(target, controls)
    if m.sum() < 10:
        return {"r2": None, "max_abs_resid": None, "n": int(m.sum())}
    X = np.column_stack([np.ones(m.sum())] + [c[m] for c in controls])
    beta, *_ = np.linalg.lstsq(X, target[m], rcond=None)
    res = target[m] - X @ beta
    ss = float(((target[m] - target[m].mean()) ** 2).sum())
    return {"r2": round(1.0 - float((res ** 2).sum()) / ss, 10) if ss else None,
            "max_abs_resid": float(np.abs(res).max()), "n": int(m.sum())}


def resid_on(target, controls):
    m = _fitmask(target, controls)
    out = np.full(len(target), np.nan)
    if m.sum() < 10:
        return out
    X = np.column_stack([np.ones(m.sum())] + [c[m] for c in controls])
    beta, *_ = np.linalg.lstsq(X, target[m], rcond=None)
    out[m] = target[m] - X @ beta
    return out


C1 = [ctrl_prev3]
C2 = [ctrl_psr]
C3 = [INSV[c] for c in INS_R3]
C5 = [COL[c] for c in PREV_S + PREV_M]
C7 = [INSV[c] for c in INS_ALL] + C1 + C2
REG_SET = C1 + C2 + C3

SPAN = {}
for nm in AXES:
    SPAN[nm] = {"on_registered_R4_set": r2_on(V[nm], REG_SET),
                "on_prev6": r2_on(V[nm], C5),
                "on_full_ins_plus_prev3_psr": r2_on(V[nm], C7)}
print("\n=== 🔴 선형 span 진단 (사전 등록 단서의 기계 증명) ===")
print(f"{'축':<20}{'R2|R4집합':>12}{'R2|prev6':>11}{'R2|ins전체':>12}")
for nm in AXES:
    s = SPAN[nm]
    print(f"{nm:<20}{str(s['on_registered_R4_set']['r2']):>12}"
          f"{str(s['on_prev6']['r2']):>11}"
          f"{str(s['on_full_ins_plus_prev3_psr']['r2']):>12}")

# ── 🔴 잔차화 검사 (본론) ────────────────────────────────────────────
RESID = {}
print("\n=== 🔴 잔차화 (본론 · 투수 블록 t) ===")
print(f"{'축':<20}{'t_원':>8}{'R1':>8}{'R2':>8}{'R3':>8}{'R4':>8}"
      f"{'R5s':>8}{'R6s':>8}{'R7s':>8}")
for nm in AXES:
    others = [V[o] for o in PASSED if o != nm]
    s1_ = screen(resid_on(V[nm], C1))
    s2_ = screen(resid_on(V[nm], C2))
    s3_ = screen(resid_on(V[nm], C3))
    s4_ = screen(resid_on(V[nm], C1 + C2 + C3 + others))
    s5_ = screen(resid_on(V[nm], C5))
    ancs = [ANC[a] for a in AXIS_ANCHOR[nm]]
    s6_ = screen(resid_on(V[nm], ancs)) if ancs else None
    s7_ = screen(resid_on(V[nm], C7))
    surv = all(abs(s["t"]) >= THRESH for s in (s1_, s2_, s3_, s4_))
    testable = bool((SPAN[nm]["on_registered_R4_set"]["r2"] or 0) < 0.999999)
    RESID[nm] = {
        "t_raw": RES[nm]["t"], "sign_raw": RES[nm]["sign"],
        "t_raw_batter_block": RES_B[nm]["t"],
        "R1_vs_ctrl_prev3_suc": {"t": s1_["t"], "M": s1_["M"], "sign": s1_["sign"]},
        "R2_vs_ctrl_asof_psr": {"t": s2_["t"], "M": s2_["M"], "sign": s2_["sign"]},
        "R3_vs_ins_6cols": {"t": s3_["t"], "M": s3_["M"], "sign": s3_["sign"],
                            "controls": INS_R3},
        "R4_vs_all_plus_passed_others": {"t": s4_["t"], "M": s4_["M"],
                                         "sign": s4_["sign"],
                                         "n_other_axes": len(others)},
        "R5_supp_vs_prev6": {"t": s5_["t"], "M": s5_["M"]},
        "R6_supp_vs_own_frozen_anchor": ({"t": s6_["t"], "M": s6_["M"],
                                          "anchors": AXIS_ANCHOR[nm]}
                                         if s6_ else
                                         {"t": None, "anchors": [],
                                          "note": "앵커를 안 쓰는 축"}),
        "R7_supp_vs_full_ins_prev3_psr": {"t": s7_["t"], "M": s7_["M"]},
        "linear_span_r2": SPAN[nm],
        "testable_by_linear_screen": testable,
        "raw_passed_t2": RES[nm]["pass_t2"],
        "survives_registered_residualizations": bool(surv),
        "REGISTERED_VERDICT": bool(RES[nm]["pass_t2"] and surv),
        "supp_also_survives_R5_R7": bool(abs(s5_["t"]) >= THRESH
                                         and abs(s7_["t"]) >= THRESH)}
    mark = ("  ✅ 생존" if RESID[nm]["REGISTERED_VERDICT"]
            else ("  ❌ 잔차화 붕괴" if RES[nm]["pass_t2"] else "  ❌ 원 t 미달"))
    if not testable:
        mark += "  🔴 선형 span 안 — 시험 불가"
    print(f"{nm:<20}{RES[nm]['t']:>8.2f}{s1_['t']:>8.2f}{s2_['t']:>8.2f}"
          f"{s3_['t']:>8.2f}{s4_['t']:>8.2f}{s5_['t']:>8.2f}"
          f"{(s6_['t'] if s6_ else float('nan')):>8.2f}{s7_['t']:>8.2f}{mark}")

SURVIVE = [n for n in AXES if RESID[n]["REGISTERED_VERDICT"]]
print(f"\n🔴 사전 등록 문턱 최종 생존: "
      f"{SURVIVE if SURVIVE else '없음 (0/%d)' % len(AXES)}")

# ── 🔴 결측 보존 순열 200회 — 경험적 귀무 눈금 ────────────────────────
print(f"\n=== 🔴 경험적 귀무분포 — 결측 보존 순열 {N_PERM}회 (투수 블록) ===")
NULL = {}
rngn = np.random.default_rng(2024)
pid_tab = anc.index.to_numpy(np.int64)


def perm_anchor(col, row_col):
    """앵커 열을 투수 사이에서 섞고(결측 보존) 축을 다시 만든다."""
    vals = anc[col].to_numpy(np.float64)
    fin = np.isfinite(vals)
    q = vals.copy()
    q[fin] = rngn.permutation(vals[fin])
    d = {int(k): float(x) for k, x in zip(pid_tab, q)}
    return row_col - np.array([d.get(int(p), np.nan) for p in pid_row],
                              np.float64)


def perm_pitcher_mean(v):
    """투수 평균을 투수 사이에서 섞고 투수내 편차는 행에 남긴다 (군집 보존)."""
    fin = np.isfinite(v)
    s = pd.Series(v[fin]).groupby(blkP[fin]).mean()
    mean_full = np.full(nbP, np.nan)
    mean_full[s.index.to_numpy()] = s.to_numpy(np.float64)
    fm = np.isfinite(mean_full)
    q = mean_full.copy()
    q[fm] = rngn.permutation(mean_full[fm])
    out = np.full(len(v), np.nan)
    out[fin] = v[fin] - mean_full[blkP[fin]] + q[blkP[fin]]
    return out


def perm_row(v):
    """행 수준 섞기 (군집 파괴 — 귀무 폭의 하한)."""
    fin = np.isfinite(v)
    q = v.copy()
    q[fin] = rngn.permutation(v[fin])
    return q


NULL_CFG = [
    ("rg_prev3_suc_dev", "perm_anchor_level",
     lambda: perm_anchor("anc_suc", COL[PREV_S[1]])),
    ("rg_fb_dev", "perm_anchor_level",
     lambda: perm_anchor("anc_fb", COL[MIXC[0]])),
    ("rg_trend", "perm_pitcher_mean", lambda: perm_pitcher_mean(V["rg_trend"])),
    ("rg_mix_ent", "perm_pitcher_mean",
     lambda: perm_pitcher_mean(V["rg_mix_ent"])),
    ("rg_trend", "perm_row_level", lambda: perm_row(V["rg_trend"])),
]
for axname, kind, fn in NULL_CFG:
    ts = np.array([screen(fn())["t"] for _ in range(N_PERM)], np.float64)
    NULL[f"{axname}|{kind}"] = {
        "n_perm": N_PERM, "kind": kind, "axis": axname,
        "t_mean": float(ts.mean()), "t_sd": float(ts.std(ddof=1)),
        "abs_t_q50": float(np.quantile(np.abs(ts), 0.50)),
        "abs_t_q95": float(np.quantile(np.abs(ts), 0.95)),
        "frac_abs_t_ge_2": float((np.abs(ts) >= 2).mean()),
        "t_min": float(ts.min()), "t_max": float(ts.max()),
        "observed_t": RES[axname]["t"]}
    print(f"  {axname:<18} {kind:<20} 귀무 t 평균 {ts.mean():+.3f} "
          f"sd {ts.std(ddof=1):.3f} |t|95분위 "
          f"{np.quantile(np.abs(ts),0.95):.2f} P(|t|>=2) "
          f"{(np.abs(ts)>=2).mean():.3f}  관측 t {RES[axname]['t']:+.3f}")

# ── 쌍상관 ────────────────────────────────────────────────────────────
ALL = ({n: V[n] for n in AXES} | {n: v for n, v in SUPP_V.items()}
       | {"ctrl_prev3_suc": ctrl_prev3, "ctrl_asof_psr": ctrl_psr})
names = list(ALL)
PC = {}
for i, a_ in enumerate(names):
    for b_ in names[i + 1:]:
        m = np.isfinite(ALL[a_]) & np.isfinite(ALL[b_])
        PC[f"{a_}|{b_}"] = (round(float(np.corrcoef(ALL[a_][m],
                                                    ALL[b_][m])[0, 1]), 4)
                            if m.sum() > 2 else None)
print("\n=== 쌍상관 |r| >= 0.50 만 표시 ===")
for k, v in sorted(PC.items(), key=lambda x: -abs(x[1] or 0)):
    if v is not None and abs(v) >= 0.50:
        print(f"  {k:<56} r {v:+.4f}")

# ── 보조 진단 ─────────────────────────────────────────────────────────
print("\n=== 보조 진단 ===")
D = {"power_M_needed_for_t2": {c: round(2 * RES[c]["se"], 12) for c in AXES},
     "covered_only": {}, "model_free": {}, "parent_coupling": {}}
for c in AXES:
    m = np.isfinite(V[c])
    if m.sum() < 1000:
        continue
    p2 = np.unique(pid_row[m], return_inverse=True)[1]
    nb2 = int(p2.max()) + 1
    I2 = np.random.default_rng(42).integers(0, nb2, size=(1000, nb2))
    s = screen(V[c][m], p2, nb2, I2, z_BN[m], p_BN[m], r0[m], int(m.sum()))
    D["covered_only"][c] = {"n_rows": int(m.sum()), "n_blocks": nb2,
                            "M": s["M"], "t": s["t"]}
    print(f"  커버행만 {c:<18} 행 {int(m.sum()):,} 투수 {nb2}  t {s['t']:+.2f}")
for c in AXES:
    fin = np.isfinite(V[c])
    u_, _, _, _ = unit(V[c])
    D["model_free"][c] = {
        "n_finite_rows": int(fin.sum()),
        "corr_row_level_u_vs_y": round(float(np.corrcoef(u_[fin],
                                                         y[fin])[0, 1]), 4)}
lp_v6 = lg(np.load(P_V19))
lp_aa = lg(np.load(f"{H}/build_AY/work/pcal_aa.npy"))
for c in AXES:
    fin = np.isfinite(V[c])
    D["parent_coupling"][c] = {
        "corr_vs_logit_pBN": round(float(np.corrcoef(V[c][fin],
                                                     z_BN[fin])[0, 1]), 4),
        "corr_vs_logit_v19_mate": round(float(np.corrcoef(V[c][fin],
                                                          lp_v6[fin])[0, 1]), 4),
        "corr_vs_logit_aa": round(float(np.corrcoef(V[c][fin],
                                                    lp_aa[fin])[0, 1]), 4)}

# ── 🔴 앵커 신선도 강건성 재스크린 ───────────────────────────────────
print("\n=== 🔴 앵커 신선도 강건성 재스크린 (앵커 행이 season==2023 인 투수만) ===")
ROB = {}
print(f"{'축':<20}{'t_원판':>9}{'t_강건판':>10}{'무이동행차':>12}")
for c in AXES:
    so, ss = screen(V[c]), screen(VR[c])
    ROB[c] = {"t_original": so["t"], "t_robust": ss["t"],
              "n_nonfinite_original": so["n_nonfinite"],
              "n_nonfinite_robust": ss["n_nonfinite"],
              "verdict_same": bool((abs(so["t"]) >= THRESH) ==
                                   (abs(ss["t"]) >= THRESH))}
    print(f"{c:<20}{so['t']:>9.2f}{ss['t']:>10.2f}"
          f"{ss['n_nonfinite']-so['n_nonfinite']:>12,}")
ROB_SAME = all(v["verdict_same"] for v in ROB.values())
print(f"  원판/강건판 판정 동일: {ROB_SAME}")

M = {
    "exp_id": "exp_076", "date": "2026-08-21",
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
        "bootstrap_registered": "pitcher-block, 1000 reps, rng seed 42, shared IDX",
        "bootstrap_secondary": "batter-block (비교용)",
        "n_blocks_pitcher": nbP, "n_blocks_batter": nbB,
        "changed_variables_vs_exp075": [
            "부트스트랩 블록 batter_id -> pitcher_id (축이 전부 투수 키 · exp_074 와 동일)",
            "라벨 복원(labels.py) 미사용 — 앵커를 주최측 asof_* 열에서 읽는다",
            "축 계열: 타자 커리어 -> 최근경기(prev) · 구종믹스",
            "잔차화 대조군: {bsr,psr,bmr} -> {prev3, psr, ins 6열}",
            "귀무 순열: 표 1종 -> 앵커순열·투수평균순열·행순열 3종"],
        "threshold_registered": ("원 |t| >= 2 AND R1~R4 전부 |t| >= 2 "
                                 "(사전 등록 · S8 · 낮추지 않았다)"),
        "size_use": "none — 부호와 t 만 읽는다 (S1)"},
    "power_gate": POWER,
    "screen_pitcher_block": RES, "screen_batter_block": RES_B,
    "supplementary_screen": SUPP,
    "passed_raw": PASSED, "residualization": RESID, "survivors": SURVIVE,
    "linear_span": SPAN,
    "prior_reconstruction_identity": PRIOR_ID,
    "empirical_null_missing_preserved": NULL,
    "diagnostics": D,
    "anchor_freshness_robustness": ROB,
    "anchor_freshness_verdict_unchanged": bool(ROB_SAME),
    "pairwise_corr": PC, "row_coverage": COVER,
    "ctrl_asof_psr_equals_frame_column": CTRL_PSR_IS_FRAME_COL,
    "sha256": SHA,
    "wall_s_s2": round(time.time() - t0, 1)}
json.dump(M, open(f"{OUT}/screen.json", "w"), indent=1, ensure_ascii=False)
print(f"\nscreen.json 저장 · s2 {time.time()-t0:.0f}s")
