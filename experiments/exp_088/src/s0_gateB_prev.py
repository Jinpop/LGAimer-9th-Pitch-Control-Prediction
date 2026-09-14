"""exp_088 stage 0d — 관문 B 예비(진단). 학습 0건.

물음: **팀원 딥 팔이 이미 들어간 면** 위에서 우리 팔(exp_085/086, ≤2023 학습 4시드)이
      아직 얹을 것이 있는가. 관문 B 의 통계량(최적 가중 · 이득 · 투수 블록 부트스트랩
      1000회 t)을 그대로 쓰되, 팔은 아직 배포 학습량이 아니다 — 그래서 **예비**다.

🔴 면의 한계 (config 가 미리 적어 둔 것 + 내가 덧붙이는 것)
  · 딥 팔·mate 팔들은 ≤2024 학습이라 이 2024 면에서 **표본 내**다 → 기준선이 실제보다 강하다
    → 우리 팔의 증분은 **과소평가**된다(우리에게 보수적인 방향).
  · 우리 팔은 ≤2023 학습 out-of-time. 즉 두 쪽의 일반화 오차 등급이 다르다.
  · 이득의 **크기**를 예측·외삽하지 않는다(S78). 부호와 t 만 읽는다.
"""
import hashlib, json, time
import numpy as np
import pandas as pd
from scipy.special import expit, logit

H = "/home/<user>/SungjinPark"
B = f"{H}/exp088"
REF = 0.2498069
K = 100000.0 / REF
EPS = 1e-6
rng = np.random.default_rng(42)
t0 = time.time()


def lg(p):
    return logit(np.clip(np.asarray(p, np.float64), EPS, 1.0 - EPS))


def conv(b):
    return 100000.0 * (1.0 - b / REF)


tr = pd.read_csv(f"{H}/data/train.csv",
                 usecols=["row_id", "season", "game_type", "control_success", "pitcher_id"])
v = tr[tr.season == 2024]
N = len(v)
assert N == 253507
y = v.control_success.to_numpy(np.float64)
blk = pd.factorize(v.pitcher_id)[0]
nb = int(blk.max()) + 1
ids = np.asarray(np.load(f"{H}/arms/ids.npy", allow_pickle=True)).astype(str)
assert np.array_equal(ids, v.row_id.to_numpy().astype(str))
BOOT = rng.integers(0, nb, size=(1000, nb))


def brier(p):
    return float(np.mean((np.asarray(p, np.float64) - y) ** 2))


def se_of(d):
    bs = np.bincount(blk, weights=d, minlength=nb)
    return float(bs[BOOT].sum(axis=1).std(ddof=1)) * K / N


X = np.load(f"{H}/arms/X_arms.npy")
Z = {
    "NN_avg4": lg(np.load(f"{H}/exp064/out/preds/A99ship_seedavg.npy")),
    "AA67": lg(np.load(f"{H}/experiments/exp_048/pred/pred_A1_u2024_seed42_at1400.npy")),
    "mate_v6": X[0], "mate_Q": X[1], "p_EBM": X[2], "AA_dep": X[3],
    "off_ins": np.load(f"{H}/arms/off.npy"),
    "DEEP5": np.load(f"{B}/out/preds/deep_route1_logit_seed5.npy"),
    "F1_FM": lg(np.load(f"{H}/exp086/out/preds/F1_FM_seedavg4.npy")),
    "F2_EMB": lg(np.load(f"{H}/exp086/out/preds/F2_EMB_seedavg4.npy")),
}
for k, z in Z.items():
    assert z.shape == (N,), (k, z.shape)

GRID = np.round(np.arange(0.0, 0.8001, 0.005), 4)


def add_one(zbase, zs):
    bb = brier(expit(zbase))
    p0 = expit(zbase)
    g = np.array([K * (bb - brier(expit((1 - w) * zbase + w * zs))) for w in GRID])
    i = int(np.argmax(g))
    w = float(GRID[i])
    pm = expit((1 - w) * zbase + w * zs)
    se = se_of((p0 - y) ** 2 - (pm - y) ** 2)
    return {"w_star": w, "gain_pts": float(g[i]), "se_pts": se,
            "t": float(g[i] / se) if se else 0.0,
            "base_score": conv(bb), "blend_score": conv(brier(pm)),
            "w_at_cap": bool(w >= 0.8 - 1e-9)}


M = {"exp_id": "exp_088", "stage": "s0d_gateB_preview",
     "surface": {"fold": "fold_2024 valid_all", "n": N},
     "🔴_limits": ["DEEP·mate 팔은 <=2024 학습 → 이 면에서 표본 내(기준선 과대) → 우리 증분 과소평가",
                   "우리 팔은 <=2023 out-of-time — 일반화 오차 등급이 다르다",
                   "S78: 이득의 크기를 외삽하지 않는다. 부호와 t 만 읽는다"]}

# ── 기준선 A: 우리 최강 정직 기준선 (exp_086 등록 기준선) ────────────────
zA = 0.75 * Z["NN_avg4"] + 0.25 * Z["AA67"]
M["baseline_A_ours_only"] = {"weights": {"NN_avg4": 0.75, "AA67": 0.25},
                             "score": conv(brier(expit(zA))),
                             "exp086_recorded": 907.5995596662945}

# 그 위에 딥 팔을 얹으면?
addA = add_one(zA, Z["DEEP5"])
M["deep_added_to_A"] = addA
zA_deep = (1 - addA["w_star"]) * zA + addA["w_star"] * Z["DEEP5"]

M["GATE_B_preview_on_A_plus_deep"] = {a: add_one(zA_deep, Z[a]) for a in ("F1_FM", "F2_EMB")}
M["GATE_B_ref_on_A_only"] = {a: add_one(zA, Z[a]) for a in ("F1_FM", "F2_EMB")}

# ── 기준선 B: 배포형 재구성 (파트너 내부 + 딥 20% + 바깥 AA) ─────────────
bj = json.load(open(f"{H}/audit_deep/pkg/DEEP/mate/model/blend.json", encoding="utf-8"))
w_v6, w_ebm, beta_ins, w_deep = (float(bj["w_v6"]), float(bj["w_ebm"]),
                                 float(bj["beta_ins"]), float(bj["w_deep"]))
z_mate = ((1 - w_ebm) * (w_v6 * Z["mate_v6"] + (1 - w_v6) * Z["mate_Q"])
          + w_ebm * Z["p_EBM"] + beta_ins * Z["off_ins"])
z_mate_deep = (1 - w_deep) * z_mate + w_deep * Z["DEEP5"]
W_OUT = 0.7
zB_nodeep = W_OUT * z_mate + (1 - W_OUT) * Z["AA_dep"]
zB_deep = W_OUT * z_mate_deep + (1 - W_OUT) * Z["AA_dep"]
M["baseline_B_deployed_recon"] = {
    "coeffs": {"w_v6": w_v6, "w_ebm": w_ebm, "beta_ins": beta_ins,
               "w_deep": w_deep, "w_outer_mate": W_OUT},
    "score_no_deep": conv(brier(expit(zB_nodeep))),
    "score_with_deep": conv(brier(expit(zB_deep))),
    "deep_delta_pts_on_this_surface": conv(brier(expit(zB_deep))) - conv(brier(expit(zB_nodeep))),
    "🔴": ("전 구성원이 이 면에서 표본 내라 절대점수는 우리 팔과 비교 불가. "
           "d_adj·alpha·beta_skew·failtype 상수는 뺐다(재구성 근사)"),
}
M["GATE_B_preview_on_B_deployed"] = {a: add_one(zB_deep, Z[a]) for a in ("F1_FM", "F2_EMB")}
M["GATE_B_ref_on_B_nodeep"] = {a: add_one(zB_nodeep, Z[a]) for a in ("F1_FM", "F2_EMB")}

M["wall_sec"] = round(time.time() - t0, 2)
json.dump(M, open(f"{B}/out/s0d_gateB_preview.json", "w"), ensure_ascii=False, indent=1)

def pr(tag, d):
    print(f"  {tag:34s} w* {d['w_star']:.3f}  이득 {d['gain_pts']:+10.5f}점  "
          f"SE {d['se_pts']:8.5f}  t {d['t']:7.3f}  기준 {d['base_score']:10.5f} → {d['blend_score']:10.5f}")

print("=== 기준선 A (우리 정직 최강, %.5f) ===" % M["baseline_A_ours_only"]["score"])
pr("+DEEP5", addA)
print("=== A+DEEP 위에 우리 팔 (관문 B 예비) ===")
for a in ("F1_FM", "F2_EMB"):
    pr(a, M["GATE_B_preview_on_A_plus_deep"][a])
print("=== (대조) A 위에 우리 팔 — exp_086 재현 ===")
for a in ("F1_FM", "F2_EMB"):
    pr(a, M["GATE_B_ref_on_A_only"][a])
print("=== 배포형 재구성 B (표본 내, 절대점수 비교 불가) ===")
b = M["baseline_B_deployed_recon"]
print(f"  딥 없음 {b['score_no_deep']:.5f} → 딥 20% {b['score_with_deep']:.5f} "
      f"(Δ {b['deep_delta_pts_on_this_surface']:+.5f})")
for a in ("F1_FM", "F2_EMB"):
    pr("B+deep → " + a, M["GATE_B_preview_on_B_deployed"][a])
for a in ("F1_FM", "F2_EMB"):
    pr("B(딥없음) → " + a, M["GATE_B_ref_on_B_nodeep"][a])
