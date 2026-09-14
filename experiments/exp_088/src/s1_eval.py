"""exp_088 stage 1b — 배포 학습량(≤2024 전량) F1_FM 평가. 학습 0건.

무엇을 재나
  (1) 🔴 배포 학습량에서의 관문 A — rho(F1_FM ≤2024 5시드, DEEP 5시드).
      이 값이 중요한 이유: stage0 의 관문 A 는 **out-of-time 우리 팔 ↔ 표본 내 딥 팔**을
      비교했다. 학습 구간이 다르면 우리 쪽 일반화 오차가 상관을 **낮추는 쪽으로** 편향한다.
      ≤2024 로 맞추면 그 비대칭이 사라진다. 상관은 라벨이 필요 없으므로 이 측정은 정직하다.
  (2) 감쇠 보정 계수 — 같은 팔의 rho 가 ≤2023 → ≤2024 로 얼마나 움직이는가(Fisher z).
      F2_EMB 를 학습하지 않고도 그 팔의 배포 학습량 rho 를 **추정**하는 데 쓴다(추정이지 측정 아님).
  (3) 관문 B — 딥 팔이 들어간 면 위 증분. ⚠️ ≤2024 팔은 이 2024 면에서 표본 내라
      **판정에 쓸 수 없다**. 기록만 한다.
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
SEEDS = [42, 7, 2024, 101, 777]
rng = np.random.default_rng(42)
t0 = time.time()


def lg(p):
    return logit(np.clip(np.asarray(p, np.float64), EPS, 1.0 - EPS))


def conv(b):
    return 100000.0 * (1.0 - b / REF)


def shaf(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


tr = pd.read_csv(f"{H}/data/train.csv",
                 usecols=["row_id", "season", "game_type", "control_success", "pitcher_id"])
v = tr[tr.season == 2024]
N = len(v)
y = v.control_success.to_numpy(np.float64)
blk = pd.factorize(v.pitcher_id)[0]
nb = int(blk.max()) + 1
BOOT = rng.integers(0, nb, size=(1000, nb))
ids = np.asarray(np.load(f"{H}/arms/ids.npy", allow_pickle=True)).astype(str)
assert np.array_equal(ids, v.row_id.to_numpy().astype(str))


def brier(p):
    return float(np.mean((np.asarray(p, np.float64) - y) ** 2))


def se_of(d):
    bs = np.bincount(blk, weights=d, minlength=nb)
    return float(bs[BOOT].sum(axis=1).std(ddof=1)) * K / N


M = {"exp_id": "exp_088", "stage": "s1b_eval_le2024",
     "surface": {"rows": "train season==2024, 253,507행 (arms/ids.npy 정렬)"},
     "🔴_limits": [
         "≤2024 학습 팔은 이 2024 행에서 **표본 내**다. Brier·이득은 판정에 쓸 수 없다",
         "상관(rho)은 라벨을 안 쓰므로 표본 내 여부와 무관하게 읽을 수 있다 — 이것이 이 단계의 산출물이다",
         "PIN_EP 는 ≤2023 면에서 고른 에폭이다. held-out valid 가 없는 상태의 유일하게 정의된 규칙",
     ]}

# ── 시드별 적재 ──────────────────────────────────────────────────────────
def load_arm(tag):
    zs, rows = [], {}
    for s in SEEDS:
        if tag == "le2023" and s != 777:
            d = f"{H}/exp085/out" if s == 42 else f"{H}/exp086/out"
            pf = f"{d}/preds/F1_FM_s{s}_r2_best.npy"
            jf = f"{d}/F1_FM_s{s}_r2.json"
        else:
            pf = f"{B}/out/preds/F1_FM_s{s}_{tag}_best.npy"
            jf = f"{B}/out/F1_FM_s{s}_{tag}.json"
        a = np.load(pf).astype(np.float64)
        assert a.shape == (N,), (tag, s, a.shape)
        j = json.load(open(jf))
        rows[str(s)] = {"brier": brier(a), "score": conv(brier(a)),
                        "best_epoch": j["best_epoch"], "take": j["take"],
                        "n_train": j["n_train"], "fit_wall_min": j["fit_wall_min"],
                        "row_independence": j["row_independence_rule_2_4"],
                        "pred_sha256": shaf(pf), "path": pf.replace(H, "~")}
        zs.append(lg(a))
    z = np.mean(zs, axis=0)
    np.save(f"{B}/out/preds/F1_FM_{tag}_seedavg5.npy", expit(z))
    return z, rows


z23, rows23 = load_arm("le2023")
z24, rows24 = load_arm("le2024")
M["per_seed"] = {"le2023": rows23, "le2024": rows24}
M["seedavg5"] = {
    "le2023": {"brier": brier(expit(z23)), "score": conv(brier(expit(z23))),
               "status": "out-of-time (정직)"},
    "le2024": {"brier": brier(expit(z24)), "score": conv(brier(expit(z24))),
               "status": "🔴 표본 내 — 비교 금지"},
}

# ── 상관 ────────────────────────────────────────────────────────────────
X = np.load(f"{H}/arms/X_arms.npy")
OTH = {
    "DEEP_5seed": np.load(f"{B}/out/preds/deep_route1_logit_seed5.npy"),
    "DEEP_4seed": np.load(f"{B}/out/preds/deep_route1_logit_seed4.npy"),
    "mate_v6": X[0], "mate_Q": X[1], "p_EBM": X[2], "AA_dep": X[3],
    "NN_ours_4seed": lg(np.load(f"{H}/exp064/out/preds/A99ship_seedavg.npy")),
    "AA67": lg(np.load(f"{H}/experiments/exp_048/pred/pred_A1_u2024_seed42_at1400.npy")),
    "T70": lg(np.load(f"{H}/experiments/exp_055/pred/pred_T70_u2024_seed42_at1400.npy")),
    "F2_EMB_le2023_4seed": lg(np.load(f"{H}/exp086/out/preds/F2_EMB_seedavg4.npy")),
}
def rho(a, b):
    return float(np.corrcoef(a, b)[0, 1])


M["GATE_A_deployment_scale"] = {
    "F1_FM_le2024_5seed_vs_DEEP_5seed": rho(z24, OTH["DEEP_5seed"]),
    "F1_FM_le2023_5seed_vs_DEEP_5seed": rho(z23, OTH["DEEP_5seed"]),
    "F1_FM_le2023_4seed_vs_DEEP_4seed_stage0": 0.8382196692790216,
    "thresholds": {"pass": "<0.90", "gray": "0.90~0.95", "reject": ">=0.95"},
}
g = M["GATE_A_deployment_scale"]
g["verdict_le2024"] = ("PASS(<0.90)" if abs(g["F1_FM_le2024_5seed_vs_DEEP_5seed"]) < 0.90
                       else "REJECT(>=0.95)" if abs(g["F1_FM_le2024_5seed_vs_DEEP_5seed"]) >= 0.95
                       else "GRAY(0.90~0.95)")

M["rho_full"] = {
    "F1_FM_le2023_5seed": {k: rho(z23, z) for k, z in OTH.items()},
    "F1_FM_le2024_5seed": {k: rho(z24, z) for k, z in OTH.items()},
    "F1_FM_le2023_vs_le2024": rho(z23, z24),
}

# ── 감쇠 보정 (추정, 측정 아님) ──────────────────────────────────────────
def fz(r):
    return 0.5 * np.log((1 + r) / (1 - r))


r23, r24 = g["F1_FM_le2023_5seed_vs_DEEP_5seed"], g["F1_FM_le2024_5seed_vs_DEEP_5seed"]
dz = fz(r24) - fz(r23)
r_f2 = 0.9171898171767191          # stage0: F2_EMB le2023 4seed vs DEEP 5seed
est = float(np.tanh(fz(r_f2) + dz))
M["attenuation_transfer_ESTIMATE"] = {
    "method": "Fisher z 이동을 같은 크기로 옮긴다",
    "F1_FM_rho_le2023": r23, "F1_FM_rho_le2024": r24,
    "delta_fisher_z": float(dz),
    "F2_EMB_rho_le2023_measured": r_f2,
    "F2_EMB_rho_le2024_ESTIMATED": est,
    "🔴": ("이것은 **추정**이다 — F2_EMB 는 학습하지 않았다(관문 A 회색 → 정지 규칙). "
           "두 팔의 감쇠가 같다는 보장이 없다. 판정 근거로 쓰지 말고 "
           "'그 팔을 학습할 값이 있는가'의 사전 확률로만 쓸 것"),
}

# ── 다른 팔 전부에 대한 직교 성분 (라벨 무관) ────────────────────────────
def resid_share(zt, basis):
    Adm = np.column_stack([np.ones(N)] + [OTH[b] for b in basis])
    coef, *_ = np.linalg.lstsq(Adm, zt, rcond=None)
    r = zt - Adm @ coef
    return {"resid_sd": float(r.std()), "arm_sd": float(zt.std()),
            "resid_share": float(r.std() / zt.std())}


BASIS = ["DEEP_5seed", "mate_v6", "mate_Q", "p_EBM", "AA_dep", "NN_ours_4seed", "T70"]
M["orthogonal_component_vs_all_arms"] = {
    "basis": BASIS,
    "F1_FM_le2023_5seed": resid_share(z23, BASIS),
    "F1_FM_le2024_5seed": resid_share(z24, BASIS),
    "F2_EMB_le2023_4seed": resid_share(OTH["F2_EMB_le2023_4seed"], BASIS),
    "F1_FM_le2024_vs_DEEP_only": resid_share(z24, ["DEEP_5seed"]),
    "F2_EMB_vs_DEEP_only": resid_share(OTH["F2_EMB_le2023_4seed"], ["DEEP_5seed"]),
}

# ── 관문 B (기록만 — 이 면은 판정 불가) ──────────────────────────────────
GRID = np.round(np.arange(0.0, 0.8001, 0.005), 4)


def add_one(zbase, zs):
    bb = brier(expit(zbase))
    p0 = expit(zbase)
    gg = np.array([K * (bb - brier(expit((1 - w) * zbase + w * zs))) for w in GRID])
    i = int(np.argmax(gg))
    w = float(GRID[i])
    pm = expit((1 - w) * zbase + w * zs)
    se = se_of((p0 - y) ** 2 - (pm - y) ** 2)
    return {"w_star": w, "gain_pts": float(gg[i]), "se_pts": se,
            "t": float(gg[i] / se) if se else 0.0,
            "base_score": conv(bb), "blend_score": conv(brier(pm)),
            "w_at_cap": bool(w >= 0.8 - 1e-9)}


zA = 0.75 * OTH["NN_ours_4seed"] + 0.25 * OTH["AA67"]
addA = add_one(zA, OTH["DEEP_5seed"])
zA_deep = (1 - addA["w_star"]) * zA + addA["w_star"] * OTH["DEEP_5seed"]
M["GATE_B_record_only"] = {
    "deep_on_our_baseline": addA,
    "F1_FM_le2023_5seed_on_A_plus_deep": add_one(zA_deep, z23),
    "F1_FM_le2024_5seed_on_A_plus_deep": add_one(zA_deep, z24),
    "F1_FM_le2023_5seed_on_A_only": add_one(zA, z23),
    "F1_FM_le2024_5seed_on_A_only": add_one(zA, z24),
    "🔴_왜_판정_불가": (
        "① 딥 팔이 이 면에서 표본 내라 기준선이 실제보다 훨씬 강하다 — w*(deep) 가 격자 상한 0.8 에 닿는다. "
        "② ≤2024 팔은 평가 행을 학습에서 봤다 — 이득이 기억이다. "
        "두 오염이 반대 방향이라 상쇄 여부도 알 수 없다. 부호·t 둘 다 못 읽는다"),
}

M["wall_sec"] = round(time.time() - t0, 2)
json.dump(M, open(f"{B}/out/s1b_eval.json", "w"), ensure_ascii=False, indent=1)

print("=== 🔴 배포 학습량 관문 A (F1_FM) ===")
print(json.dumps(M["GATE_A_deployment_scale"], ensure_ascii=False, indent=1))
print("=== rho 전문 ===")
for tag in ("F1_FM_le2023_5seed", "F1_FM_le2024_5seed"):
    print(" ", tag)
    for k, r in M["rho_full"][tag].items():
        print(f"    {k:24s} {r:+.6f}")
print("  le2023 ↔ le2024 자기 상관", M["rho_full"]["F1_FM_le2023_vs_le2024"])
print("=== 감쇠 이전 추정 ===")
print(json.dumps(M["attenuation_transfer_ESTIMATE"], ensure_ascii=False, indent=1))
print("=== 직교 성분 ===")
print(json.dumps(M["orthogonal_component_vs_all_arms"], ensure_ascii=False, indent=1))
print("=== 단독 점수 ===")
print(json.dumps(M["seedavg5"], ensure_ascii=False, indent=1))
print("=== 관문 B (기록만) ===")
for k, d in M["GATE_B_record_only"].items():
    if isinstance(d, dict):
        print(f"  {k:38s} w* {d['w_star']:.3f} 이득 {d['gain_pts']:+11.5f} t {d['t']:7.3f}")
