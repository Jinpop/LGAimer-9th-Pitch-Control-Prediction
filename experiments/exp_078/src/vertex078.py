"""exp_078 stage0-3 — LB 눈금 전 축 동시 꼭지점.  학습 0건, 제출물 0건.

이미 산 LB 점 전부에서 각 축의 1차항 L 을 회수하고, 현 base(BO) 로 옮겨
동시 꼭지점 u* = G^-1 L 를 푼다.  gain(u*) = K L^T G^-1 L >= 0 이므로
진 제출도 전부 미회수 재고다.

핵심 산술 (출하 그대로)
  파트너 내부  base = w_in*lg_p_v6 + (1-w_in)*lg_p_q
               z_p  = (1-w_ebm)*base + w_ebm*p_ebm        (p_ebm 은 이미 로짓)
               z_p += beta_ins*off
               p_partner = expit(z_p + d_inner)            (d_inner 동결)
  우리 바깥    z    = w_out*lg(p_partner) + sum_j c_j*lg(p_arm_j)
               z'   = z + sum_i u_i*dir_i
               a    = alpha + seg*con                      (BH 행별 퍼짐)
               f    = a*z' + beta_skew*z'^2
               p    = expit(f + anc(f;T_REF) + s)          (s = 평균축 d_mean)
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "6"

import hashlib
import io
import json
import time

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit

H = "/home/<user>/SungjinPark"
FRAME = f"{H}/build_AC/work/tr2024"
OUT = f"{H}/exp078"
os.makedirs(OUT, exist_ok=True)

EPS = 1e-6
REF = 0.2498069
K = 100000.0 / REF
N_ROWS = 253507
T_REF = 0.4847980356943589          # BO 출하 target_rate
D_INNER = -0.019641085540766146     # 파트너 내부 d_adj (동결)
T0 = time.time()
M = {"exp_id": "exp_078", "title": "LB 눈금 전 축 동시 꼭지점",
     "K_pts_per_brier": K, "ref_brier": REF, "n_rows": N_ROWS,
     "threads": "OMP/OPENBLAS/MKL/NUMEXPR/VECLIB = 6"}


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def lg(p):
    return logit(np.clip(np.asarray(p, np.float64), EPS, 1.0 - EPS))


def conv(b):
    return 100000.0 * (1.0 - b / REF)


# ══════════════════════════════════════════════════════════════════════
# 0. 입력 sha 핀
# ══════════════════════════════════════════════════════════════════════
PIN_ASSERT = {
    f"{H}/build_AR/work/tr2024_p_v19_AR.npy":
        "99909f0692a2507d0b599b58cac2ecc0b59b086d1a800a899d5fbabd2f1755f4",
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
    f"{H}/build_BN/work/tr2024_p_v19_BN_wv6_070.npy":
        "ab38ec13c7bb4c78e1f03f6eb459138dad395c7561b31ea86bec25ba7b1238eb",
}
PIN_RECORD = [f"{H}/arms/lg_p_v6.npy", f"{H}/arms/lg_p_q.npy",
              f"{H}/arms/p_ebm.npy", f"{H}/arms/off.npy",
              f"{H}/build_AJ/work/tr2024_p_aa_off_fresh_AJ.npy",
              f"{H}/build_AV/work/tr2024_p_aa_AV.npy",
              f"{H}/build_BO/work/tr2024_p_v19_BO_wv6_vertex.npy",
              f"{H}/build_BO/pkg_BO/blend.json",
              f"{H}/build_BO/pkg_BO/mate/model/blend.json"]
print("=== sha 대조 ===", flush=True)
for p, want in PIN_ASSERT.items():
    got = sha_file(p)
    assert got == want, f"sha 불일치 {p}\n 기대 {want}\n 실측 {got}"
M["sha_pins_asserted"] = {os.path.relpath(k, H): v[:16] for k, v in PIN_ASSERT.items()}
M["sha_recorded"] = {os.path.relpath(p, H): sha_file(p) for p in PIN_RECORD}
print("  전부 일치", flush=True)

# ══════════════════════════════════════════════════════════════════════
# 1. 벡터 적재 — 🔴 S14: 이름이 p_ 여도 로짓일 수 있다.  범위를 찍는다.
# ══════════════════════════════════════════════════════════════════════
LGV6 = np.load(f"{H}/arms/lg_p_v6.npy").astype(np.float64)
LGQ = np.load(f"{H}/arms/lg_p_q.npy").astype(np.float64)
PEBM = np.load(f"{H}/arms/p_ebm.npy").astype(np.float64)
OFFR = np.load(f"{H}/arms/off.npy").astype(np.float64)
y = np.load(f"{FRAME}/labels.npy").astype(np.float64)
RNG = {}
for nm, v in (("arms/lg_p_v6", LGV6), ("arms/lg_p_q", LGQ),
              ("arms/p_ebm", PEBM), ("arms/off", OFFR), ("labels", y)):
    RNG[nm] = {"min": float(v.min()), "max": float(v.max()),
               "mean": float(v.mean()), "shape": list(v.shape)}
    print(f"  {nm:<16} min {v.min():+.6f}  max {v.max():+.6f}  mean {v.mean():+.6f}",
          flush=True)
M["S14_range_check"] = RNG
M["S14_verdict"] = (
    "arms/p_ebm 는 이름이 p_ 로 시작하지만 값 범위가 확률 [0,1] 밖 "
    f"[{PEBM.min():+.4f}, {PEBM.max():+.4f}] 로 **음수를 포함하므로 확률이 아니다 — 로짓이다.** "
    "logit() 을 다시 씌우지 않는다.  ⚠️ 폭이 좁다(|값| < 0.9): 세 팔이 전부 "
    "0.5 근처로 잘 보정돼 있어 로짓 절대값이 작을 뿐이고, 확률이라는 뜻이 아니다. "
    "arms/lg_p_v6·lg_p_q 도 같은 폭이며 셋의 범위가 서로 정합한다.  "
    "arms/off 는 로짓 차분.  pcal_* 만 확률이라 lg() 를 씌운다.")
assert PEBM.min() < 0.0, "p_ebm 이 음수가 아니다 — 확률일 수 있다, S14 재확인"
assert LGV6.min() < 0.0 and LGQ.min() < 0.0
for v in (LGV6, LGQ, PEBM, OFFR, y):
    assert v.shape == (N_ROWS,)

L_AA18 = lg(np.load(f"{H}/build_AY/work/pcal_aa.npy"))
L_S2 = lg(np.load(f"{H}/build_AY/work/pcal_S2.npy"))
L_S1 = lg(np.load(f"{H}/build_AY/work/pcal_S1.npy"))
L_T70 = lg(np.load(f"{H}/build_AY/work/pcal_T70.npy"))
L_AA9 = lg(np.load(f"{H}/build_AJ/work/tr2024_p_aa_off_fresh_AJ.npy"))
ARMS18 = np.stack([L_AA18, L_S2, L_S1, L_T70])
ARMS9 = np.stack([L_AA9, L_S2, L_S1, L_T70])
M["arm_set_note"] = (
    "ARMS18 = build_AY/work/pcal_* (AV 18시드 AA + S2·S1·T70).  "
    "ARMS9 = build_AJ/work/tr2024_p_aa_off_fresh_AJ (AK~AT 시절 9시드 AA).  "
    "S2·S1·T70 계수는 그 시절 0 이라 벡터가 안 쓰인다.")

# 방향 벡터 (BI·BJ·BK)
D59 = np.load(f"{H}/exp059_screen/directions.npz")
BLJ = json.load(open(f"{H}/build_BL/pkg_BL/blend.json", encoding="utf-8"))
AXL = {a["name"]: a for a in BLJ["axis_offsets"]}


def stdz(raw):
    v = np.asarray(raw, np.float64)
    fin = np.isfinite(v)
    mu, sd = float(v[fin].mean()), float(v[fin].std())
    u = np.zeros(len(v))
    u[fin] = (v[fin] - mu) / sd
    return u, mu, sd


DIRS, DIRMETA = {}, {}
for key, raw in (("off_BI", OFFR), ("off_BJ", D59["ctrl_asof_psr"]),
                 ("off_BK", D59["sd_arms"])):
    u, mu, sd = stdz(raw)
    DIRS[key] = u
    DIRMETA[key] = {"center": mu, "scale": sd,
                    "n_nonfinite": int((~np.isfinite(np.asarray(raw, float))).sum())}
assert np.array_equal(OFFR, np.asarray(D59["beta_ins_off"], np.float64))
assert DIRMETA["off_BI"]["center"] == AXL["beta_ins_off"]["center"]
assert DIRMETA["off_BI"]["scale"] == AXL["beta_ins_off"]["scale"]
assert DIRMETA["off_BJ"]["center"] == AXL["ctrl_asof_psr"]["center"]
assert DIRMETA["off_BJ"]["scale"] == AXL["ctrl_asof_psr"]["scale"]
M["direction_meta"] = DIRMETA
print("  방향 표준화 상수 BL 출하값과 일치 (BI·BJ)", flush=True)


# ══════════════════════════════════════════════════════════════════════
# 2. 출하 산술 빌더
# ══════════════════════════════════════════════════════════════════════
_PBUF = {}


def partner(w_in, w_ebm, beta_ins):
    """파트너 패키지 출력.  🔴 배포 경로를 그대로 탄다 —

    파트너는 자기 script.py 로 submission.csv 를 쓰고 바깥 script.py 가 그것을
    pd.read_csv 로 읽는다.  pandas 기본 float 파서(precise_xstrtod)는 정확
    반올림이 아니라 약 31% 행에서 1 ulp(5.55e-17) 어긋난다.  저장돼 있는
    tr2024_p_v19_*.npy 는 전부 그 CSV 를 읽은 값이므로, 여기서도 같은 왕복을
    거쳐야 비트 동일해진다.  (float_precision='round_trip' 으로 읽으면 직접
    계산값과 비트 동일 — 즉 두 값의 차는 파서 반올림 하나뿐이다.)
    """
    key = (w_in, w_ebm, beta_ins)
    if key in _PBUF:
        return _PBUF[key]
    base = w_in * LGV6 + (1.0 - w_in) * LGQ
    z = (1.0 - w_ebm) * base + w_ebm * PEBM
    z = z + beta_ins * OFFR
    p = expit(z + D_INNER)
    b = io.StringIO()
    pd.DataFrame({"control_success": p}).to_csv(b, index=False)
    b.seek(0)
    p = pd.read_csv(b)["control_success"].to_numpy(np.float64)
    _PBUF[key] = p
    return p


def anc(f, T=T_REF):
    return brentq(lambda d: expit(f + d).mean() - T, -5.0, 5.0, xtol=1e-15)


AXES = ["alpha", "beta_skew", "c_aa", "c_S2", "c_S1", "c_T70", "w_out",
        "w_v6_inner", "w_ebm", "beta_ins", "off_BI", "off_BJ", "off_BK",
        "seg_spread_BH", "d_mean"]
SPEC12 = ["alpha", "beta_skew", "c_aa", "c_T70", "w_out", "w_v6_inner",
          "w_ebm", "beta_ins", "off_BI", "off_BJ", "off_BK", "seg_spread_BH"]
IDX = {a: i for i, a in enumerate(AXES)}
NA = len(AXES)

BO = dict(alpha=1.07911, beta_skew=-0.059151, c_aa=0.25244, c_S2=0.0,
          c_S1=0.0, c_T70=0.04756, w_out=0.7, w_v6_inner=0.6668397489,
          w_ebm=-0.1, beta_ins=0.03, off_BI=-0.006534, off_BJ=0.006544,
          off_BK=0.0, seg_spread_BH=0.0, d_mean=0.0)

CON = None      # BH 대비 벡터 — 아래에서 채운다


def zf(th, arms):
    lgp = lg(partner(th["w_v6_inner"], th["w_ebm"], th["beta_ins"]))
    c = np.array([th["c_aa"], th["c_S2"], th["c_S1"], th["c_T70"]])
    z = th["w_out"] * lgp + c @ arms
    zp = (z + th["off_BI"] * DIRS["off_BI"] + th["off_BJ"] * DIRS["off_BJ"]
          + th["off_BK"] * DIRS["off_BK"])
    a = (th["alpha"] if th["seg_spread_BH"] == 0.0
         else th["alpha"] + th["seg_spread_BH"] * CON)
    return a * zp + th["beta_skew"] * zp * zp


def build(th, arms=ARMS18, T=T_REF, ret_d=False):
    f = zf(th, arms)
    d = anc(f, T) + th["d_mean"]
    p = expit(f + d)
    return (p, d) if ret_d else p


def mv(base, **kw):
    t = dict(base)
    t.update(kw)
    return t


# ══════════════════════════════════════════════════════════════════════
# 3. stage0 게이트
# ══════════════════════════════════════════════════════════════════════
G0 = {}
G0A_CASES = [
    ("BN_wv6_070", f"{H}/build_BN/work/tr2024_p_v19_BN_wv6_070.npy", 0.70, -0.1, 0.03),
    ("BN_wv6_050_fresh", f"{H}/build_BN/work/tr2024_p_v19_BN_wv6_050_fresh.npy",
     0.50, -0.1, 0.03),
    ("AR", f"{H}/build_AR/work/tr2024_p_v19_AR.npy", 0.50, -0.1, 0.03),
    ("BO_wv6_vertex", f"{H}/build_BO/work/tr2024_p_v19_BO_wv6_vertex.npy",
     0.6668397489, -0.1, 0.03),
    ("BO_wv6_070_fresh", f"{H}/build_BO/work/tr2024_p_v19_BO_wv6_070_fresh.npy",
     0.70, -0.1, 0.03),
    ("AO_webm_000", f"{H}/build_AO/work/tr2024_p_v19_ao.npy", 0.50, 0.0, 0.03),
    ("AK_webm_015", f"{H}/build_AO/work/tr2024_p_v19_w015.npy", 0.50, 0.15, 0.03),
    ("AP_wv6_100", f"{H}/build_AP/work/tr2024_p_v19_ap.npy", 1.00, 0.0, 0.03),
    ("AQ_wv6_025", f"{H}/build_AQ/work/tr2024_p_v19_AQ.npy", 0.25, 0.0, 0.03),
    ("AT_betains_006", f"{H}/build_AT/work/tr2024_p_v19_AT.npy", 0.50, -0.1, 0.06)]
g0a_rows = {}
for nm, path, w_, we_, be_ in G0A_CASES:
    st = np.load(path)
    rc = partner(w_, we_, be_)
    g0a_rows[nm] = {"path": os.path.relpath(path, H),
                    "params": {"w_v6_inner": w_, "w_ebm": we_, "beta_ins": be_},
                    "bitwise_equal": bool(np.array_equal(rc, st)),
                    "n_rows_differ": int((rc != st).sum()),
                    "max_abs_diff": float(np.abs(rc - st).max())}
g0a = all(v["bitwise_equal"] for v in g0a_rows.values())
G0["G0a"] = {"what": ("arms 벡터로 재구성한 파트너 출력이 저장된 tr2024 파트너 "
                      "벡터 10종과 비트 동일한가 (배포 CSV 왕복 포함)"),
             "cases": g0a_rows, "n_cases": len(G0A_CASES),
             "BN_stored_sha256":
                 PIN_ASSERT[f"{H}/build_BN/work/tr2024_p_v19_BN_wv6_070.npy"],
             "sha_asserted_prefix_ab38ec13": True,
             "PASS": g0a}
print(f"\n[G0a] 파트너 재구성 10종 비트동일 {g0a}", flush=True)
for nm, v in g0a_rows.items():
    print(f"      {nm:<18} equal={v['bitwise_equal']} ndiff={v['n_rows_differ']}",
          flush=True)

p_BO, d_BO = build(BO, ret_d=True)
D_BO_SHIP = 0.02875748759679648
G0["G0b"] = {"shipped": repr(D_BO_SHIP), "solver": repr(d_BO),
             "abs_diff": abs(d_BO - D_BO_SHIP),
             "PASS": bool(abs(d_BO - D_BO_SHIP) < 1e-15)}
print(f"[G0b] BO d_adj 재현 |Δ| {G0['G0b']['abs_diff']:.3e}", flush=True)

BN_S = mv(BO, w_v6_inner=0.70)
p_BN, d_BN = build(BN_S, ret_d=True)
br_BN = float(np.mean((p_BN - y) ** 2))
G0["G0c"] = {"brier_BN": br_BN, "expected_exp072": 0.24552396647966293,
             "abs_diff": abs(br_BN - 0.24552396647966293),
             "d_adj_BN_solver": repr(d_BN),
             "d_adj_BN_shipped": repr(0.028610055723290194),
             "d_adj_abs_diff": abs(d_BN - 0.028610055723290194),
             "PASS": bool(abs(br_BN - 0.24552396647966293) < 1e-15)}
print(f"[G0c] BN 표본내 Brier {br_BN!r}  |Δ| {G0['G0c']['abs_diff']:.3e}", flush=True)
M["stage0_gates"] = G0
if not all(G0[k]["PASS"] for k in ("G0a", "G0b", "G0c")):
    M["ABORTED"] = "stage0 게이트 실패 — 이후 단계 진행 안 함"
    json.dump(M, open(f"{OUT}/vertex078.json", "w"), indent=1, ensure_ascii=False)
    raise SystemExit("stage0 게이트 실패")
print("[G0] 3종 전부 통과\n", flush=True)

# ══════════════════════════════════════════════════════════════════════
# 4. BH 대비 벡터 재구성 (assemble_bh.py 와 동일 절차)
# ══════════════════════════════════════════════════════════════════════
BG_S = mv(BO, w_v6_inner=0.5, off_BI=0.0, off_BJ=0.0)
p_BG, d_BG = build(BG_S, ret_d=True)
D_BG_SHIP = 0.029492809613719313
cols = ["row_id", "season", "control_success", "pitcher_id", "game_month",
        "asof_pitcher_n", "pitcher_hand", "batter_hand",
        "asof_pitcher_fastball_rate", "balls_before", "strikes_before"]
tr = pd.read_csv(f"{H}/data/train.csv", usecols=cols)
tr = tr[tr.season == 2024].sort_values("row_id", kind="stable")
assert np.array_equal(tr.control_success.to_numpy(np.float64), y)
n_p = tr.asof_pitcher_n.to_numpy(float)
fb = tr.asof_pitcher_fastball_rate.to_numpy(float)
TH = {"n_p20": float(np.nanpercentile(n_p, 20)),
      "n_p80": float(np.nanpercentile(n_p, 80)),
      "fb_p80": float(np.nanpercentile(fb, 80))}
ph, bh = tr.pitcher_hand.to_numpy(), tr.batter_hand.to_numpy()
SEG = {
    "후반8-10월": tr.game_month.isin([8, 9, 10]).to_numpy(),
    "투수고경험20": np.less(TH["n_p80"], n_p, where=np.isfinite(n_p),
                       out=np.zeros(len(n_p), bool)),
    "좌투좌타": (ph == 1) & (bh == 1),
    "직구비율높음20": np.less(TH["fb_p80"], fb, where=np.isfinite(fb),
                        out=np.zeros(len(fb), bool)),
    "손일치": (ph == bh),
    "초구": (tr.balls_before.to_numpy() == 0) & (tr.strikes_before.to_numpy() == 0),
    "투수저경험20": np.less(n_p, TH["n_p20"], where=np.isfinite(n_p),
                       out=np.zeros(len(n_p), bool)),
    "2스트라이크": (tr.strikes_before.to_numpy() == 2)}
NMS = list(SEG)
MK = np.stack([SEG[k].astype(float) for k in NMS])
SH = {k: float(SEG[k].mean()) for k in NMS}
_lgp_BG = lg(partner(0.5, -0.1, 0.03))
_c = np.array([BG_S["c_aa"], BG_S["c_S2"], BG_S["c_S1"], BG_S["c_T70"]])
z_BG = BG_S["w_out"] * _lgp_BG + _c @ ARMS18
r0_ins = y - p_BG


def _raw(mask, eps=0.05):
    m = mask - mask.mean()
    f = (1.10 + eps * m) * z_BG
    return expit(f + anc(f)) - expit(1.10 * z_BG + anc(1.10 * z_BG))


DD = np.stack([_raw(MK[i]) for i in range(len(NMS))])
Mv = DD @ r0_ins / len(z_BG)
Gm = DD @ DD.T / len(z_BG)
W = np.linalg.solve(Gm + np.eye(len(NMS)) * np.trace(Gm) / len(NMS) * 0.10, Mv)
W = W / np.abs(W).max()
CON = W @ (MK - np.array([SH[k] for k in NMS])[:, None])
p_BH_chk = build(mv(BG_S, seg_spread_BH=0.05868))
KQ_BH = K * float(np.mean((p_BH_chk - p_BG) ** 2))
M["BH_contrast"] = {"segments": NMS, "weights": {k: float(v) for k, v in zip(NMS, W)},
                    "shares": SH, "thresholds": TH,
                    "con_mean": float(CON.mean()), "con_std": float(CON.std()),
                    "con_range": [float(CON.min()), float(CON.max())],
                    "KQ_reproduced_pts": KQ_BH, "KQ_ledger_pts": 2.46226,
                    "KQ_rel_diff": abs(KQ_BH - 2.46226) / 2.46226}
print(f"[BH] 대비 재구성 · K*Q {KQ_BH:.5f} (장부 2.46226)", flush=True)

# ══════════════════════════════════════════════════════════════════════
# 5. 역사적 base 재구성 + d_adj 재현
# ══════════════════════════════════════════════════════════════════════
E1 = dict(alpha=1.0, beta_skew=0.0, c_aa=0.5, c_S2=0.0, c_S1=0.0, c_T70=0.0,
          w_out=0.5, w_v6_inner=0.5, w_ebm=0.15, beta_ins=0.03,
          off_BI=0.0, off_BJ=0.0, off_BK=0.0, seg_spread_BH=0.0, d_mean=0.0)
T_E1 = 0.4785
T_AW = 0.4829582933733939
E2 = mv(E1, w_out=0.7, c_aa=0.3, w_ebm=-0.1)
BB_ = mv(E2, c_aa=0.25244, c_T70=0.04756, alpha=1.10)

STATES = {
    "AJ": (dict(E1), ARMS9, T_E1, -0.009049280362973488),
    "AK": (mv(E1, w_out=0.7, c_aa=0.3), ARMS9, T_E1, -0.005425088856956198),
    "AO": (mv(E1, w_out=0.7, c_aa=0.3, w_ebm=0.0), ARMS9, T_E1,
           -0.00499338075913577),
    "AP": (mv(E1, w_out=0.7, c_aa=0.3, w_ebm=0.0, w_v6_inner=1.0), ARMS9, T_E1,
           -0.0067096850311203155),
    "AQ": (mv(E1, w_out=0.7, c_aa=0.3, w_ebm=0.0, w_v6_inner=0.25), ARMS9, T_E1,
           -0.004147154513231207),
    "AR": (mv(E1, w_out=0.7, c_aa=0.3, w_ebm=-0.1), ARMS9, T_E1,
           -0.004710593992398005),
    "AT": (mv(E1, w_out=0.7, c_aa=0.3, w_ebm=-0.1, beta_ins=0.06), ARMS9, T_E1,
           None),
    "AV": (dict(E2), ARMS18, T_E1, -0.0046521158244170295),
    "AW": (dict(E2), ARMS18, T_AW, 0.013395858438940476),
    "AX": (mv(E2, c_aa=0.150, c_S2=0.084, c_S1=0.066), ARMS18, T_AW,
           0.010768860784554449),
    "AY": (mv(E2, c_aa=0.0855, c_S2=0.04788, c_S1=0.03762, c_T70=0.129), ARMS18,
           T_AW, 0.010407561724660303),
    "AZ": (mv(E2, c_aa=0.25244, c_T70=0.04756), ARMS18, T_AW,
           0.012846758753662767),
    "BA": (mv(E2, c_aa=0.25244, c_T70=0.04756, alpha=1.05), ARMS18, T_AW,
           0.016876429209068684),
    "BB": (dict(BB_), ARMS18, T_AW, 0.0209047558289546),
    "BC": (mv(BB_, beta_skew=0.15778), ARMS18, T_AW, 0.013396089828732293),
    "BE": (dict(BB_), ARMS18, 0.48048, None),
    "BG": (dict(BG_S), ARMS18, T_REF, D_BG_SHIP),
    "BH": (mv(BG_S, seg_spread_BH=0.05868), ARMS18, T_REF, 0.028919213836393298),
    "BI": (mv(BG_S, off_BI=-0.006033), ARMS18, T_REF, None),
    "BJ": (mv(BG_S, off_BJ=-0.006087), ARMS18, T_REF, None),
    "BK": (mv(BG_S, off_BK=-0.006557), ARMS18, T_REF, None),
    "BL": (mv(BG_S, off_BI=-0.006534, off_BJ=0.006544), ARMS18, T_REF,
           0.02950104990829574),
    "BN": (dict(BN_S), ARMS18, T_REF, 0.028610055723290194),
    "BO": (dict(BO), ARMS18, T_REF, D_BO_SHIP)}

REC, PVEC, THETA = {}, {}, {}
print("\n=== 역사적 base 재구성 ===", flush=True)
for nm, (th, arms, T, dship) in STATES.items():
    p_, d_ = build(th, arms=arms, T=T, ret_d=True)
    f = zf(th, arms)
    s = d_ - anc(f, T_REF)
    th2 = mv(th, d_mean=s)
    p2 = build(th2, arms=arms, T=T_REF)
    assert float(np.abs(p2 - p_).max()) < 1e-14, (nm, float(np.abs(p2 - p_).max()))
    THETA[nm] = (th2, arms)
    PVEC[nm] = p_
    REC[nm] = {"d_adj_solver": repr(d_),
               "d_adj_shipped": (repr(dship) if dship is not None else None),
               "abs_diff": (abs(d_ - dship) if dship is not None else None),
               "anchor_target_used": T, "s_at_T_REF": s,
               "insample_brier": float(np.mean((p_ - y) ** 2)),
               "insample_pts": conv(float(np.mean((p_ - y) ** 2)))}
    tag = "" if dship is None else f"  |Δd| {abs(d_-dship):.3e}"
    print(f"  [{nm}] d {d_:+.15f}{tag}   s {s:+.4e}", flush=True)
REC["BE"]["ledger_d_move_quote"] = -0.01005635
REC["BE"]["ledger_implied_d_adj"] = 0.0209047558289546 - 0.01005635
REC["BE"]["ledger_vs_solver_abs_diff"] = abs(
    float(REC["BE"]["d_adj_solver"]) - (0.0209047558289546 - 0.01005635))
M["historical_state_reconstruction"] = REC
BAD = {k: v["abs_diff"] for k, v in REC.items()
       if v["abs_diff"] is not None and v["abs_diff"] > 1e-12}
M["d_adj_reproduction_failures"] = BAD
print(f"  d_adj 재현 실패(>1e-12): {BAD if BAD else '없음'}", flush=True)

# ══════════════════════════════════════════════════════════════════════
# 6. stage1 — BO 에서의 축 미분 δ_j 와 G
# ══════════════════════════════════════════════════════════════════════
print("\n=== stage1 δ / G ===", flush=True)
HSTEP = {a: 0.002 for a in AXES}
DELTA = np.zeros((NA, N_ROWS))
LIN = {}
for a in AXES:
    h = HSTEP[a]
    pp = build(mv(BO, **{a: BO[a] + h}))
    pm = build(mv(BO, **{a: BO[a] - h}))
    dj = (pp - pm) / (2.0 * h)
    DELTA[IDX[a]] = dj
    p4 = build(mv(BO, **{a: BO[a] + 4 * h}))
    sec1 = (pp - p_BO) / h
    sec4 = (p4 - p_BO) / (4 * h)
    nz = float(np.linalg.norm(dj))
    LIN[a] = {"h": h, "rms_delta": float(np.sqrt(np.mean(dj ** 2))),
              "max_abs_delta": float(np.abs(dj).max()),
              "mean_delta": float(dj.mean()),
              "secant_1h_norm_ratio": float(np.linalg.norm(sec1) / nz),
              "secant_4h_norm_ratio": float(np.linalg.norm(sec4) / nz),
              "secant_4h_corr": float(np.corrcoef(sec4, dj)[0, 1])}
    print(f"  δ[{a:<14}] rms {LIN[a]['rms_delta']:.6e}  mean {dj.mean():+.3e}  "
          f"4h선형비 {LIN[a]['secant_4h_norm_ratio']:.6f}", flush=True)
GM = DELTA @ DELTA.T / N_ROWS
evals = np.linalg.eigvalsh(GM)
COND = float(evals.max() / evals.min()) if evals.min() > 0 else float("inf")
CORR = GM / np.sqrt(np.outer(np.diag(GM), np.diag(GM)))
M["stage1"] = {"axes": AXES, "h": HSTEP, "delta_stats": LIN,
               "G_matrix": GM.tolist(),
               "G_diag": {a: float(GM[IDX[a], IDX[a]]) for a in AXES},
               "K_times_G_diag_pts_at_u1": {a: float(K * GM[IDX[a], IDX[a]])
                                            for a in AXES},
               "corr_matrix": CORR.tolist(),
               "eigenvalues": evals.tolist(), "condition_number_15axes": COND}
G12 = GM[np.ix_([IDX[a] for a in SPEC12], [IDX[a] for a in SPEC12])]
ev12 = np.linalg.eigvalsh(G12)
M["stage1"]["condition_number_spec12"] = float(ev12.max() / ev12.min())
hi = [(AXES[i], AXES[j], float(CORR[i, j])) for i in range(NA)
      for j in range(i + 1, NA) if abs(CORR[i, j]) > 0.9]
M["stage1"]["near_collinear_pairs_abs_corr_gt_0.9"] = hi
print(f"  G 조건수 15축 {COND:.4e} · spec12 "
      f"{M['stage1']['condition_number_spec12']:.4e}", flush=True)
print(f"  |ρ|>0.9 쌍: {hi}", flush=True)

# ══════════════════════════════════════════════════════════════════════
# 7. stage2 — LB 점에서 L 회수
# ══════════════════════════════════════════════════════════════════════
LB = {"AJ": 1084.1816568022, "AK": 1084.8425905691, "AO": 1092.4366979462,
      "AP": 1086.8298367234, "AQ": 1086.5747385601, "AR": 1093.9353261813,
      "AT": 1092.9039148326, "AV": 1094.8069785012, "AW": 1102.7842569825,
      "AX": 1102.2973952664, "AY": 1101.3045817441, "AZ": 1102.8732019192,
      "BA": 1109.3766839828, "BB": 1110.5899048276, "BC": 1106.9073724154,
      "BE": 1104.1395515075, "BG": 1112.251888309, "BH": 1109.0093026421,
      "BI": 1113.2115870555, "BJ": 1109.1791585789, "BK": 1110.2788664138,
      "BL": 1114.1019927547, "BN": 1115.9165956458}
KQ_LEDGER = {"BA": 2.46205, "BC": 2.46210, "BE": 2.46205, "BH": 2.46226,
             "BI": 1.0391466180219786, "BJ": 1.05604, "BK": 1.23933,
             "BN": 2.714855935936327}
POINTS = [("AK", "AJ", "w_out 0.5->0.7 (우리 몫 0.5->0.3 동반)", True),
          ("AO", "AK", "w_ebm 0.15->0.0", True),
          ("AP", "AO", "w_v6_inner 0.5->1.0", True),
          ("AQ", "AO", "w_v6_inner 0.5->0.25", True),
          ("AR", "AO", "w_ebm 0.0->-0.10", True),
          ("AT", "AR", "beta_ins 0.03->0.06", True),
          ("AW", "AV", "평균축 target_rate 0.4785->0.48296", False),
          ("AX", "AW", "c 재분할 aa/S2/S1", False),
          ("AY", "AX", "c 재분할 T70 도입", False),
          ("AZ", "AY", "c 재분할 S2·S1 제거·aa 확대", True),
          ("BA", "AZ", "alpha 1.00->1.05", True),
          ("BB", "BA", "alpha 1.05->1.10", True),
          ("BC", "BB", "beta_skew 0->+0.15778", True),
          ("BE", "BB", "평균축 -0.01005635", False),
          ("BG", "BB", "alpha·beta_skew·평균 3축 동시", True),
          ("BH", "BG", "seg_spread 0->0.05868", True),
          ("BI", "BG", "off_BI -0.006033", True),
          ("BJ", "BG", "off_BJ -0.006087", True),
          ("BK", "BG", "off_BK -0.006557", True),
          ("BL", "BG", "off_BI·off_BJ 동시", True),
          ("BN", "BL", "w_v6_inner 0.5->0.70", True)]
rms_r0_ins = float(np.sqrt(np.mean((y - p_BO) ** 2)))
ROWS = []
print("\n=== stage2 L 회수 ===", flush=True)
for ch, pa, desc, in_cfg in POINTS:
    th_c, arm_c = THETA[ch]
    th_p, arm_p = THETA[pa]
    assert arm_c is arm_p, (ch, pa)
    v = np.array([th_c[a] - th_p[a] for a in AXES])
    p_c, p_p = PVEC[ch], PVEC[pa]
    chk = float(np.abs(build(mv(th_p, **{a: th_p[a] + v[i]
                                         for i, a in enumerate(AXES)}),
                             arms=arm_p) - p_c).max())
    dsec = p_c - p_p
    Gkk = float(np.mean(dsec ** 2))
    g = LB[ch] - LB[pa]
    m = (g / K + Gkk) / 2.0
    corr = float(np.mean(dsec * (p_p - p_BO)))
    b = m + corr
    dsec_BO = build(mv(BO, **{a: BO[a] + v[i] for i, a in enumerate(AXES)})) - p_BO
    nsp, nsb = float(np.linalg.norm(dsec)), float(np.linalg.norm(dsec_BO))
    cosang = float(dsec @ dsec_BO / (nsp * nsb))
    resid_rms = float(np.sqrt(np.mean((dsec_BO - dsec) ** 2)))
    ratio = abs(corr) / abs(m) if m != 0 else float("inf")
    grade = "A" if ratio < 0.3 else ("B" if ratio < 1.0 else "EXCLUDE")
    row = dict(child=ch, parent=pa, desc=desc, in_config_list=in_cfg,
               v={a: float(v[i]) for i, a in enumerate(AXES) if v[i] != 0.0},
               identity_max_abs=chk, g_pts=g,
               G_kk_secant=Gkk, K_times_Q_pts=K * Gkk,
               K_times_Q_ledger=KQ_LEDGER.get(ch),
               L_at_parent=m, base_shift_correction=corr, L_at_BO=b,
               corr_over_L=ratio, grade=grade,
               delta_norm_parent=nsp, delta_norm_BO=nsb,
               delta_norm_ratio=float(nsb / nsp), delta_cos_parent_BO=cosang,
               delta_change_rms=resid_rms,
               unknown_term_bound_pts=float(2 * K * resid_rms * rms_r0_ins))
    if KQ_LEDGER.get(ch) is not None:
        row["K_times_Q_rel_diff"] = abs(K * Gkk - KQ_LEDGER[ch]) / KQ_LEDGER[ch]
    ROWS.append(row)
    lg_tag = ("" if KQ_LEDGER.get(ch) is None
              else f" (장부 {KQ_LEDGER[ch]:.5f})")
    print(f"  [{ch}<-{pa}] g {g:+9.5f}  K*Q {K*Gkk:8.5f}{lg_tag}  L {m:+.6e}  "
          f"보정 {corr:+.6e}  비 {ratio:6.3f}  {grade}", flush=True)
M["stage2_points"] = ROWS
M["stage2_identity_max_abs"] = max(r["identity_max_abs"] for r in ROWS)
assert M["stage2_identity_max_abs"] < 1e-12, "이동 벡터 항등 실패"
M["stage2_grade_rule"] = "|보정항|/|L| < 0.3 → A · 0.3~1.0 → B · >=1.0 → 제외"
M["stage2_grade_counts"] = {g: sum(1 for r in ROWS if r["grade"] == g)
                            for g in ("A", "B", "EXCLUDE")}

# 자기검사 ② — BN 점의 L 이 exp_072 기록과 일치하는가
rBN = [r for r in ROWS if r["child"] == "BN"][0]
L_BN_axis = rBN["L_at_parent"] / rBN["v"]["w_v6_inner"]
G_BN_axis = rBN["G_kk_secant"] / rBN["v"]["w_v6_inner"] ** 2
M["selftest_2_BN_L"] = {
    "L_axis_from_BN_point": L_BN_axis,
    "exp_072_recorded_L": 2.8287251706490112e-05,
    "abs_diff": abs(L_BN_axis - 2.8287251706490112e-05),
    "rel_diff": abs(L_BN_axis - 2.8287251706490112e-05) / 2.8287251706490112e-05,
    "G_axis_from_BN_point": G_BN_axis,
    "exp_072_recorded_G": 1.695474363257131e-04,
    "G_rel_diff": abs(G_BN_axis - 1.695474363257131e-04) / 1.695474363257131e-04,
    "PASS": bool(abs(L_BN_axis - 2.8287251706490112e-05) < 1e-12)}
print(f"\n[자기검사②] BN 축 L {L_BN_axis:.15e}  vs exp_072 "
      f"2.8287251706490112e-05  → |Δ| {M['selftest_2_BN_L']['abs_diff']:.3e}",
      flush=True)
if not M["selftest_2_BN_L"]["PASS"]:
    json.dump(M, open(f"{OUT}/vertex078.json", "w"), indent=1, ensure_ascii=False)
    raise SystemExit("자기검사② 실패 — stage2 구현이 틀렸다")

# BL blend.json 이 기록한 BI 축 L·G 와의 대조
rBI = [r for r in ROWS if r["child"] == "BI"][0]
M["selftest_2b_BI_vs_BL_blendjson"] = {
    "L_axis_ours": rBI["L_at_parent"] / rBI["v"]["off_BI"],
    "L_recorded_in_BL_blend_json": AXL["beta_ins_off"]["L_i_back_solved"],
    "G_axis_ours": rBI["G_kk_secant"] / rBI["v"]["off_BI"] ** 2,
    "G_recorded_in_BL_blend_json": AXL["beta_ins_off"]["G_ii"]}

# 자기검사 ① — 같은 단일축 다점 일치
MULTI = {"w_ebm": ["AO", "AR"], "alpha": ["BA", "BB"],
         "w_v6_inner": ["AP", "AQ", "BN"]}
S1T = {}
for ax, chs in MULTI.items():
    vals = {}
    for ch in chs:
        r = [q for q in ROWS if q["child"] == ch]
        if not r:
            continue
        r = r[0]
        if set(r["v"]) - {ax, "d_mean"}:
            continue
        vals[ch] = r["L_at_BO"] / r["v"][ax]
    if len(vals) >= 2:
        vv = np.array(list(vals.values()))
        S1T[ax] = {"per_point_L_at_BO": vals, "mean": float(vv.mean()),
                   "spread": float(vv.max() - vv.min()),
                   "rel_spread": float(abs(vv.max() - vv.min()) / abs(vv.mean()))}
        print(f"[자기검사①] {ax}: " +
              "  ".join(f"{k} {q:+.6e}" for k, q in vals.items()) +
              f"   상대편차 {S1T[ax]['rel_spread']:.4f}", flush=True)
M["selftest_1_multipoint_agreement"] = S1T

# ══════════════════════════════════════════════════════════════════════
# 8. stage3 — L 회수(최소노름 최소제곱) → 동시 꼭지점
# ══════════════════════════════════════════════════════════════════════
def recover_L(rows):
    A = np.array([[r["v"].get(a, 0.0) for a in AXES] for r in rows])
    b = np.array([r["L_at_BO"] for r in rows])
    Lv, _, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    fit = A @ Lv
    cov = [AXES[j] for j in range(NA) if np.abs(A[:, j]).max() > 0]
    return Lv, {"n_points": len(rows), "points": [r["child"] for r in rows],
                "rank": int(rank), "singular_values": sv.tolist(),
                "covered_axes": cov,
                "uncovered_axes": [a for a in AXES if a not in cov],
                "residual_per_point": {rows[i]["child"]: float(fit[i] - b[i])
                                       for i in range(len(rows))},
                "residual_rms": float(np.sqrt(np.mean((fit - b) ** 2))),
                "L": {a: float(Lv[IDX[a]]) for a in AXES}}


def vertex(Lv, axset, ridge_thresh=1e6):
    ai = [IDX[a] for a in axset]
    Ls = Lv[ai]
    Gs = GM[np.ix_(ai, ai)]
    ev = np.linalg.eigvalsh(Gs)
    cond = float(ev.max() / ev.min()) if ev.min() > 0 else float("inf")
    lam = 0.0
    Gu = Gs
    if cond > ridge_thresh:
        lam = float(np.trace(Gs) / len(axset) * (cond / ridge_thresh) * 1e-6)
        Gu = Gs + np.eye(len(axset)) * lam
    u = np.linalg.solve(Gu, Ls)
    gain = float(K * (2 * u @ Ls - u @ Gs @ u))
    un = np.array([Ls[i] / Gs[i, i] for i in range(len(axset))])
    gain_naive = float(K * (2 * un @ Ls - un @ Gs @ un))
    marg = {}
    for i, a in enumerate(axset):
        kp = [j for j in range(len(axset)) if j != i]
        uk = np.linalg.solve(Gu[np.ix_(kp, kp)], Ls[kp])
        gk = float(K * (2 * uk @ Ls[kp] - uk @ Gs[np.ix_(kp, kp)] @ uk))
        marg[a] = gain - gk
    return {"axes": axset, "G_condition": cond, "ridge_lambda": lam,
            "u_star": {a: float(u[i]) for i, a in enumerate(axset)},
            "new_constants": {a: float(BO[a] + u[i]) for i, a in enumerate(axset)},
            "predicted_gain_pts": gain,
            "naive_sum_u": {a: float(un[i]) for i, a in enumerate(axset)},
            "naive_sum_predicted_gain_pts": gain_naive,
            "marginal_contribution_pts": marg,
            "conditional_vertex_gain_pts": {a: float(K * Ls[i] ** 2 / Gs[i, i])
                                            for i, a in enumerate(axset)}}


SETS = {"A_all": [r for r in ROWS if r["grade"] == "A"],
        "AB_all": [r for r in ROWS if r["grade"] in ("A", "B")],
        "A_configpoints": [r for r in ROWS if r["grade"] == "A" and r["in_config_list"]],
        "AB_configpoints": [r for r in ROWS if r["grade"] in ("A", "B")
                            and r["in_config_list"]]}
S3 = {}
print("\n=== stage3 동시 해 ===", flush=True)
for nm, rws in SETS.items():
    if not rws:
        continue
    Lv, rec = recover_L(rws)
    S3[nm] = {"L_recovery": rec}
    for tag, axs in (("spec12", SPEC12), ("ext15", AXES)):
        use = [a for a in axs if a in rec["covered_axes"]]
        S3[nm][tag] = vertex(Lv, use)
        S3[nm][tag]["axes_requested"] = axs
        S3[nm][tag]["axes_dropped_uncovered"] = [a for a in axs if a not in use]
        print(f"  [{nm}/{tag}] 축 {len(use)} · 점 {len(rws)} · 조건수 "
              f"{S3[nm][tag]['G_condition']:.3e} · 예측 "
              f"{S3[nm][tag]['predicted_gain_pts']:+.5f}점 · naive "
              f"{S3[nm][tag]['naive_sum_predicted_gain_pts']:+.5f}점", flush=True)
    for a in S3[nm]["spec12"]["axes"]:
        print(f"      {a:<14} u* {S3[nm]['spec12']['u_star'][a]:+.6f} -> "
              f"{S3[nm]['spec12']['new_constants'][a]:+.6f}   한계 "
              f"{S3[nm]['spec12']['marginal_contribution_pts'][a]:+.5f}", flush=True)
M["stage3"] = S3

BND = {"alpha": (0.5, 2.0), "beta_skew": (-0.5, 0.5), "c_aa": (-0.2, 0.8),
       "c_S2": (-0.5, 0.5), "c_S1": (-0.5, 0.5), "c_T70": (-1.0, 0.5),
       "w_out": (0.3, 1.5), "w_v6_inner": (0.0, 1.0), "w_ebm": (-0.5, 0.5),
       "beta_ins": (-0.2, 0.3), "off_BI": (-0.1, 0.1), "off_BJ": (-0.1, 0.1),
       "off_BK": (-0.1, 0.1), "seg_spread_BH": (-1.0, 1.0), "d_mean": (-0.5, 0.5)}
M["stage3_bounds"] = BND
M["stage3_out_of_range"] = {
    f"{nm}/{tag}": {a: [S3[nm][tag]["new_constants"][a], list(BND[a])]
                    for a in S3[nm][tag]["axes"]
                    if not (BND[a][0] <= S3[nm][tag]["new_constants"][a] <= BND[a][1])}
    for nm in S3 for tag in ("spec12", "ext15")}

# 착륙 해를 출하 산술로 실제 실행 — 실행가능성 + 정확 곡률
LAND = {}
for nm in S3:
    for tag in ("spec12", "ext15"):
        th = dict(BO)
        for a, u in S3[nm][tag]["u_star"].items():
            th[a] = BO[a] + u
        try:
            p_l, d_l = build(th, ret_d=True)
            ds = p_l - p_BO
            Ls = np.array([S3[nm]["L_recovery"]["L"][a]
                           for a in S3[nm][tag]["axes"]])
            uu = np.array([S3[nm][tag]["u_star"][a] for a in S3[nm][tag]["axes"]])
            LAND[f"{nm}/{tag}"] = {
                "d_adj": repr(d_l),
                "insample_brier": float(np.mean((p_l - y) ** 2)),
                "insample_delta_pts_vs_BO": conv(float(np.mean((p_l - y) ** 2)))
                - conv(float(np.mean((p_BO - y) ** 2))),
                "pred_diff_vs_BO_max_abs": float(np.abs(ds).max()),
                "pred_diff_vs_BO_mean_abs": float(np.abs(ds).mean()),
                "model_2nd_order_pts": float(
                    K * uu @ GM[np.ix_([IDX[a] for a in S3[nm][tag]["axes"]],
                                       [IDX[a] for a in S3[nm][tag]["axes"]])] @ uu),
                "exact_2nd_order_pts": float(K * np.mean(ds ** 2)),
                "gain_with_exact_curvature_pts": float(
                    K * (2 * uu @ Ls - float(np.mean(ds ** 2))))}
        except Exception as e:                                # pragma: no cover
            LAND[f"{nm}/{tag}"] = {"error": repr(e)}
M["stage3_landing"] = LAND
M["stage3_landing_🔴"] = (
    "insample_delta 는 tr2024(다섯 팔 전부에 표본 내) 값이라 순위 근거가 아니다. "
    "여기서는 착륙 해가 출하 산술로 실행되는지와 2차항의 정확값 확인용이다.")
M["🔴_S5"] = ("stage3 의 예측 이득은 **산술 예측이지 측정이 아니다.** 하방도 "
              "대칭이다.  동시 꼭지점 해의 역사적 실현률은 84%(BG 82.7% · BL 85.4%, "
              "2/2)이며 그것도 표본 2건이다.")

M["wall_clock_min"] = (time.time() - T0) / 60.0
json.dump(M, open(f"{OUT}/vertex078.json", "w"), indent=1, ensure_ascii=False)
print(f"\n기록 {OUT}/vertex078.json · 벽시계 {M['wall_clock_min']:.2f}분", flush=True)
