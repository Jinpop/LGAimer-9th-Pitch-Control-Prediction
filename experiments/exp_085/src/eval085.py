"""exp_085 stage 3 — 사전 등록 관문 둘 + 부가 측정. 학습 0건.

면: fold_2024 valid_all 253,507행 (train <=2023 → valid 2024).
기존 팔 예측은 **재학습하지 않고** 저장 벡터를 읽어 sha 를 기록한다.

⚠️ 눈금 두 종류
  · out-of-time (정직) — exp_048/049/053/055 저장 예측, exp_064 NN. 새 팔과 같은 눈금.
  · tr2024 표본 내 — `arms/X_arms.npy` (팀원 배포본 계측). mate v6·Q 는 **이 판본밖에
    없다**(그쪽은 <=2024 학습 배포본뿐). 상관은 읽을 수 있으나 그 팔의 점수는 정직하지
    않다. 관문 G0b 는 config 가 지정한 6팔이 이 두 눈금에 걸쳐 있어 둘 다 싣는다.
"""
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.special import expit, logit

H = "/home/<user>/SungjinPark"
B = f"{H}/exp085"
OUT = f"{B}/out"
REF = 0.2498069
K = 100000.0 / REF
EPS = 1e-6
SEED = 42
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
                 usecols=["row_id", "season", "game_type", "control_success",
                          "pitcher_id"])
v = tr[tr.season == 2024]
N = len(v)
assert N == 253507
y = v.control_success.to_numpy(np.float64)
seg = v.game_type.to_numpy().astype(str)
blk = pd.factorize(v.pitcher_id)[0]
nb = int(blk.max()) + 1
ids = np.asarray(np.load(f"{H}/arms/ids.npy", allow_pickle=True)).astype(str)
assert np.array_equal(ids, v.row_id.to_numpy().astype(str))


def brier(p):
    return float(np.mean((np.asarray(p, np.float64) - y) ** 2))


def se_of(d):
    bs = np.bincount(blk, weights=d, minlength=nb)
    tot = bs[rng.integers(0, nb, size=(1000, nb))].sum(axis=1)
    return float(tot.std(ddof=1)) * K / N


M = {"exp_id": "exp_085",
     "surface": {"fold": "fold_2024", "train_upto": 2023, "valid_season": 2024,
                 "n_valid_all": N, "y_mean": float(y.mean())},
     "score_conversion": {"formula": "100000*(1 - Brier/0.2498069)",
                          "sensitivity": "Brier 1e-6 = 0.4003점 (S3)"}}

# ── 기존 팔 (저장 벡터 재사용, 재학습 0) ──────────────────────────────────
P = {
    "AA67_exp048_A1_s42@1400": f"{H}/experiments/exp_048/pred/pred_A1_u2024_seed42_at1400.npy",
    "AA81_exp049_A1_s42@1400": f"{H}/experiments/exp_049/pred/pred_A1_u2024_seed42_at1400.npy",
    "S1_exp053_s42@1400": f"{H}/experiments/exp_053/pred/pred_S1_u2024_seed42_at1400.npy",
    "S2_exp053_s42@1400": f"{H}/experiments/exp_053/pred/pred_S2_u2024_seed42_at1400.npy",
    "CE0_exp053_s42@1400": f"{H}/experiments/exp_053/pred/pred_CE0_u2024_seed42_at1400.npy",
    "T70_exp055_s42@1400": f"{H}/experiments/exp_055/pred/pred_T70_u2024_seed42_at1400.npy",
    "NN_exp064_A99ship_s42": f"{H}/exp064/out/preds/A99ship_s42.npy",
    "NN_exp064_A99ship_seedavg": f"{H}/exp064/out/preds/A99ship_seedavg.npy",
}
honest, prov = {}, {}
for k, p in P.items():
    a = np.load(p).astype(np.float64)
    assert a.shape == (N,), (k, a.shape)
    honest[k] = a
    prov[k] = {"path": p.replace(H, "~"), "sha256": shaf(p), "reused": True}

X = np.load(f"{H}/arms/X_arms.npy")
assert X.shape == (5, N)
band = {"lg_p_v6(tr2024 표본내)": X[0], "lg_p_Q(tr2024 표본내)": X[1],
        "p_EBM(tr2024 표본내, 이미 로짓 — S14)": X[2],
        "lg_p_AA_deployed(tr2024 표본내)": X[3], "off(arms)": X[4]}
prov["arms/X_arms.npy"] = {"path": "~/arms/X_arms.npy",
                           "sha256": shaf(f"{H}/arms/X_arms.npy"), "reused": True}
M["reused_vectors"] = prov

# ── 새 팔 ────────────────────────────────────────────────────────────────
# 등록 측정값 = r2 (exp_064 프로토콜: epochs 25 고정, 조기 종료 없음, take=best).
# r1 은 schedule 을 60에폭으로 잡고 patience 5 로 warm-up 에서 잘린 **구현 오류** 실행이라
# 판정에 쓰지 않고 기록만 한다 (산출물은 지우지 않았다).
new, fits = {}, {}
for arm in ("F1_FM", "F2_EMB"):
    j = f"{OUT}/{arm}_s{SEED}_r2.json"
    assert os.path.exists(j), j
    fits[arm] = json.load(open(j))
    for tag in ("best", "last"):
        a = np.load(f"{OUT}/preds/{arm}_s{SEED}_r2_{tag}.npy").astype(np.float64)
        assert a.shape == (N,)
        new[f"{arm}_{tag}"] = a
M["fits"] = {k: {kk: vv for kk, vv in val.items() if kk != "curve"}
             for k, val in fits.items()}
M["rule_2_4_row_independence"] = {
    k: val["row_independence_rule_2_4"] for k, val in fits.items()}

r1 = {}
for arm in ("F1_FM", "F2_EMB"):
    j = f"{OUT}/{arm}_s{SEED}.json"
    if os.path.exists(j):
        d = json.load(open(j))
        p = np.load(f"{OUT}/preds/{arm}_s{SEED}_best.npy").astype(np.float64)
        b = brier(p)
        r1[arm] = {"best_epoch": d["best_epoch"], "epochs_run": d["epochs_run"],
                   "brier": b, "score": conv(b),
                   "note": "구현 오류 실행 — OneCycle 을 60에폭으로 잡고 patience 5 로 "
                           "warm-up 에서 잘렸다. 판정에 쓰지 않는다"}
M["run_r1_not_registered"] = r1

# ── 단독 성능 ────────────────────────────────────────────────────────────
def scoreblock(p):
    b = brier(p)
    r = {"brier": b, "score": conv(b)}
    for s in ("R", "F"):
        m = seg == s
        bb = float(np.mean((p[m] - y[m]) ** 2))
        r[f"brier_{s}"] = bb
        r[f"n_{s}"] = int(m.sum())
    return r


solo = {k: scoreblock(p) for k, p in honest.items()}
solo.update({k: scoreblock(p) for k, p in new.items()})
# 밴드 프레임 팔은 로짓으로 저장돼 있어 확률로 되돌려 점수만 참고로 적는다
band_solo = {k: scoreblock(expit(z)) for k, z in band.items()
             if k.startswith("lg_") or k.startswith("p_EBM")}
M["standalone_out_of_time"] = solo
M["standalone_tr2024_insample_NOT_COMPARABLE"] = band_solo

# ── G0a ──────────────────────────────────────────────────────────────────
G0A_SET = ["AA67_exp048_A1_s42@1400", "AA81_exp049_A1_s42@1400",
           "S1_exp053_s42@1400", "S2_exp053_s42@1400",
           "T70_exp055_s42@1400", "NN_exp064_A99ship_s42"]
weak_name = min(G0A_SET, key=lambda k: solo[k]["score"])
weak = solo[weak_name]["score"]
g0a = {"threshold_pts": -500.0,
       "reference_set": {k: solo[k]["score"] for k in G0A_SET},
       "weakest_existing_arm": weak_name, "weakest_score": weak,
       "floor_score": weak - 500.0, "arms": {}}
for k in new:
    d = solo[k]["score"] - weak
    g0a["arms"][k] = {"score": solo[k]["score"], "delta_vs_weakest_pts": d,
                      "pass": bool(d >= -500.0)}
M["G0a_standalone"] = g0a

# ── G0b ──────────────────────────────────────────────────────────────────
series = {k: lg(p) for k, p in honest.items()}
series.update({k: lg(p) for k, p in new.items()})
series.update({k: v for k, v in band.items() if k != "off(arms)"})
names = list(series)
C = np.corrcoef(np.vstack([series[n] for n in names]))
idx = {n: i for i, n in enumerate(names)}
M["corr_matrix_logit"] = {a: {b: float(C[i, j]) for j, b in enumerate(names)}
                          for i, a in enumerate(names)}

# config 가 지정한 기존 6팔 = AA · mate v6 · Q · T70 · S1 · S2
GATE_SET = ["AA67_exp048_A1_s42@1400", "AA81_exp049_A1_s42@1400",
            "lg_p_v6(tr2024 표본내)", "lg_p_Q(tr2024 표본내)",
            "T70_exp055_s42@1400", "S1_exp053_s42@1400", "S2_exp053_s42@1400"]
EXTRA = ["NN_exp064_A99ship_s42", "NN_exp064_A99ship_seedavg",
         "CE0_exp053_s42@1400", "p_EBM(tr2024 표본내, 이미 로짓 — S14)",
         "lg_p_AA_deployed(tr2024 표본내)"]
g0b = {"threshold_max_abs_rho": 0.93, "space": "로짓", "arms": {}}
for k in new:
    r = {n: float(C[idx[k], idx[n]]) for n in GATE_SET}
    e = {n: float(C[idx[k], idx[n]]) for n in EXTRA}
    mx = max(r, key=lambda n: abs(r[n]))
    g0b["arms"][k] = {"rho_vs_gate_set": r, "rho_vs_reference_only": e,
                      "max_abs_rho": abs(r[mx]), "argmax": mx,
                      "pass": bool(abs(r[mx]) <= 0.93)}
M["G0b_diversity"] = g0b

# ── 판정 ────────────────────────────────────────────────────────────────
M["verdict"] = {k: {"G0a": g0a["arms"][k]["pass"], "G0b": g0b["arms"][k]["pass"],
                    "both": bool(g0a["arms"][k]["pass"] and g0b["arms"][k]["pass"])}
                for k in new}
M["verdict"]["any_arm_passes_both"] = any(
    v["both"] for k, v in M["verdict"].items() if isinstance(v, dict))

# ── 부가 1: 기존 3원 블렌드 위 최적 가중 ─────────────────────────────────
zb = (0.50 * lg(honest["AA67_exp048_A1_s42@1400"])
      + 0.28 * lg(honest["S2_exp053_s42@1400"])
      + 0.22 * lg(honest["S1_exp053_s42@1400"]))
b_base = brier(expit(zb))
zn = 0.75 * lg(honest["NN_exp064_A99ship_seedavg"]) + 0.25 * lg(
    honest["AA67_exp048_A1_s42@1400"])
b_nn = brier(expit(zn))
add = {"base_3way": {"weights": {"AA67": 0.50, "S2": 0.28, "S1": 0.22},
                     "src": "exp_053 → 후보 AX 안쪽 3원", "brier": b_base,
                     "score": conv(b_base)},
       "base_NN_AA": {"weights": {"NN_seedavg": 0.75, "AA67": 0.25},
                      "src": "exp_065/066 w*=0.75", "brier": b_nn,
                      "score": conv(b_nn)},
       "on_3way": {}, "on_NN_AA": {}}
grid = np.round(np.arange(0.0, 0.601, 0.005), 4)
for base_key, zbase, bb in (("on_3way", zb, b_base), ("on_NN_AA", zn, b_nn)):
    for k, p in new.items():
        zs = lg(p)
        gains = np.array([K * (bb - brier(expit((1 - w) * zbase + w * zs)))
                          for w in grid])
        i = int(np.argmax(gains))
        w = float(grid[i])
        pm = expit((1 - w) * zbase + w * zs)
        d = (expit(zbase) - y) ** 2 - (pm - y) ** 2
        se = se_of(d)
        add[base_key][k] = {"w_star": w, "gain_pts": float(gains[i]),
                            "se_pts": se, "t": float(gains[i] / se) if se else 0.0,
                            "blend_score": conv(brier(pm)),
                            "grid": "0..0.6 step 0.005",
                            "w_at_cap": bool(w >= 0.6 - 1e-9)}
M["addendum_blend"] = add

# ── 부가 2: ID 미등장률 (prep 에서 실측한 값을 옮겨 적는다) ───────────────
prep = json.load(open(f"{OUT}/prep.json"))
M["addendum_unseen_ids"] = {k: prep["categorical"]["valid_unknown"][k]
                            for k in ("pitcher_id", "batter_id", "dom6",
                                      "pitcher_team_id")}
M["addendum_prep"] = {"n_train": prep["alignment"]["n_train"],
                      "dense_width": prep["dense"]["width"],
                      "n_missing_flags": prep["dense"]["n_missing_flags"],
                      "cache_sha256": prep["cache_sha256"]}
M["wall_sec"] = round(time.time() - t0, 2)
json.dump(M, open(f"{OUT}/metrics.json", "w"), ensure_ascii=False, indent=1)

print("=== G0a 단독성능 (문턱 −500점) ===")
print(f"기존 최약 팔 {weak_name}  {weak:.5f}점  →  바닥 {weak-500:.5f}")
for k, r in g0a["arms"].items():
    print(f"  {k:20s} {r['score']:12.5f}  Δ {r['delta_vs_weakest_pts']:+11.5f}  "
          f"{'PASS' if r['pass'] else 'FAIL'}")
print("=== G0b 다양성 (문턱 |ρ| ≤ 0.93) ===")
for k, r in g0b["arms"].items():
    print(f"  {k:20s} max|ρ| {r['max_abs_rho']:.6f} vs {r['argmax']}  "
          f"{'PASS' if r['pass'] else 'FAIL'}")
    for n, val in sorted(r["rho_vs_gate_set"].items(), key=lambda x: -abs(x[1])):
        print(f"      {n:42s} {val:+.6f}")
print("=== 부가: 블렌드 ===")
for bk in ("on_3way", "on_NN_AA"):
    for k, r in add[bk].items():
        print(f"  {bk:10s} {k:20s} w* {r['w_star']:.3f}  "
              f"gain {r['gain_pts']:+10.5f}  t {r['t']:+6.2f}")
