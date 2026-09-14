"""exp_086 stage 3 — 4시드 평균 새 팔을 **최강 기준선** 위에서 잰다. 학습 0건.

답하는 질문 하나: exp_085 가 최강 기준선 위에서 본 +0.42 / +1.90 은 **모형의 한계**인가
**시드 평균을 아직 안 산 탓**인가. 기준선의 NN 은 4시드 평균인데 새 팔은 1시드였다.
여기서는 새 팔도 4시드 평균(로짓 평균, exp_064 `logit_avg` 와 같은 방식)으로 올려
같은 조건에서 다시 잰다.

🔴 사전 등록 판정(결과를 보고 바꾸지 않는다 — S8):
   기준선 = 0.75·logit(NN 4시드평균) + 0.25·logit(AA67) → 907.59956점, **이것 하나로만 판정**
   통과 = 어느 한 팔이라도 t >= 2.0. 둘 다 미달이면 이 계열 종료.
   3원 기준선 등 다른 기준선 위 숫자는 **부가**이고 판정에 쓰지 않는다.

면: fold_2024 valid_all 253,507행. SE 는 투수 블록 부트스트랩 1000회(exp_085 와 동일 절차).
"""
import hashlib
import json
import time

import numpy as np
import pandas as pd
from scipy.special import expit, logit

H = "/home/<user>/SungjinPark"
O85 = f"{H}/exp085/out"
O86 = f"{H}/exp086/out"
REF = 0.2498069
K = 100000.0 / REF
EPS = 1e-6
SEEDS = [42, 7, 2024, 101]          # 42 는 exp_085 r2 재사용
NN_SEEDS = [42, 777, 2024, 101]     # exp_064 NN 이 쓴 시드(참고)
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
assert N == 253507, N
y = v.control_success.to_numpy(np.float64)
seg = v.game_type.to_numpy().astype(str)
blk = pd.factorize(v.pitcher_id)[0]
nb = int(blk.max()) + 1
ids = np.asarray(np.load(f"{H}/arms/ids.npy", allow_pickle=True)).astype(str)
assert np.array_equal(ids, v.row_id.to_numpy().astype(str))
BOOT = rng.integers(0, nb, size=(1000, nb))     # 전 비교에 같은 재표집을 쓴다


def brier(p):
    return float(np.mean((np.asarray(p, np.float64) - y) ** 2))


def se_of(d):
    bs = np.bincount(blk, weights=d, minlength=nb)
    return float(bs[BOOT].sum(axis=1).std(ddof=1)) * K / N


M = {"exp_id": "exp_086",
     "surface": {"fold": "fold_2024", "train_upto": 2023, "valid_season": 2024,
                 "n_valid_all": N, "y_mean": float(y.mean())},
     "score_conversion": {"formula": "100000*(1 - Brier/0.2498069)",
                          "sensitivity": "Brier 1e-6 = 0.4003점 (S3)"},
     "seed_avg_rule": "로짓 평균 후 시그모이드 — exp_064 run064.logit_avg 와 동일"}

# ── 재사용 벡터 ──────────────────────────────────────────────────────────
EXIST = {
    "AA67": f"{H}/experiments/exp_048/pred/pred_A1_u2024_seed42_at1400.npy",
    "AA81": f"{H}/experiments/exp_049/pred/pred_A1_u2024_seed42_at1400.npy",
    "S1": f"{H}/experiments/exp_053/pred/pred_S1_u2024_seed42_at1400.npy",
    "S2": f"{H}/experiments/exp_053/pred/pred_S2_u2024_seed42_at1400.npy",
    "T70": f"{H}/experiments/exp_055/pred/pred_T70_u2024_seed42_at1400.npy",
    "NN_s42": f"{H}/exp064/out/preds/A99ship_s42.npy",
    "NN_avg4": f"{H}/exp064/out/preds/A99ship_seedavg.npy",
}
A, prov = {}, {}
for k, p in EXIST.items():
    a = np.load(p).astype(np.float64)
    assert a.shape == (N,), (k, a.shape)
    A[k] = a
    prov[k] = {"path": p.replace(H, "~"), "sha256": shaf(p), "reused": True,
               "retrained": False}

X = np.load(f"{H}/arms/X_arms.npy")
assert X.shape == (5, N)
BAND = {"mate_v6(tr2024 표본내)": X[0], "mate_Q(tr2024 표본내)": X[1],
        "p_EBM(tr2024 표본내, 이미 로짓 — S14)": X[2],
        "AA_deployed(tr2024 표본내)": X[3]}
prov["arms/X_arms.npy"] = {"path": "~/arms/X_arms.npy",
                           "sha256": shaf(f"{H}/arms/X_arms.npy"),
                           "reused": True, "retrained": False}

# ── 새 팔 시드별 + 4시드 평균 ────────────────────────────────────────────
ARMS = ["F1_FM", "F2_EMB"]
per_seed, Zavg, seed_block = {}, {}, {}
for arm in ARMS:
    zs, rows = [], {}
    for s in SEEDS:
        src = O85 if s == 42 else O86
        pf = f"{src}/preds/{arm}_s{s}_r2_best.npy"
        jf = f"{src}/{arm}_s{s}_r2.json"
        a = np.load(pf).astype(np.float64)
        assert a.shape == (N,), (arm, s, a.shape)
        j = json.load(open(jf))
        assert j["seed"] == s and j["epochs"] == 25 and j["early_stop"] is False
        assert j["take"] == "best"
        b = brier(a)
        rows[s] = {"brier": b, "score": conv(b), "best_epoch": j["best_epoch"],
                   "last_valid_score": j["last_valid_score"],
                   "fit_wall_min": j["fit_wall_min"],
                   "pred_sha256": hashlib.sha256(
                       np.ascontiguousarray(a).tobytes()).hexdigest(),
                   "source": "exp_085 r2 재사용" if s == 42 else "exp_086 신규",
                   "row_independence": j["row_independence_rule_2_4"],
                   "n_params": j["n_params"], "device": j["device"]}
        zs.append(lg(a))
        prov[f"{arm}_s{s}"] = {"path": pf.replace(H, "~"), "sha256": shaf(pf),
                               "reused": s == 42, "retrained": s != 42}
    z = np.mean(zs, axis=0)
    Zavg[arm] = z
    pav = expit(z)
    bav = brier(pav)
    sc = [rows[s]["score"] for s in SEEDS]
    seed_block[arm] = {
        "seeds": SEEDS, "per_seed": rows,
        "per_seed_score_mean": float(np.mean(sc)),
        "per_seed_score_sd": float(np.std(sc, ddof=1)),
        "per_seed_score_min": float(np.min(sc)), "per_seed_score_max": float(np.max(sc)),
        "per_seed_brier_mean": float(np.mean([rows[s]["brier"] for s in SEEDS])),
        "seedavg_brier": bav, "seedavg_score": conv(bav),
        "S_seed_pts": conv(bav) - float(np.mean(sc)),
        "best_epochs": [rows[s]["best_epoch"] for s in SEEDS]}
    np.save(f"{O86}/preds/{arm}_seedavg4.npy", pav)
M["reused_vectors"] = prov
M["seed_expansion"] = seed_block

# ── 기준선 ───────────────────────────────────────────────────────────────
Z = {k: lg(a) for k, a in A.items()}
Z.update({k: np.asarray(z, np.float64) for k, z in BAND.items()})
z_main = 0.75 * Z["NN_avg4"] + 0.25 * Z["AA67"]
b_main = brier(expit(z_main))
z_3way = 0.50 * Z["AA67"] + 0.28 * Z["S2"] + 0.22 * Z["S1"]
b_3way = brier(expit(z_3way))
z_nn1 = Z["NN_s42"]
b_nn1 = brier(expit(z_nn1))
M["baselines"] = {
    "PRIMARY_NNavg4_AA67": {"weights": {"NN_avg4": 0.75, "AA67": 0.25},
                            "brier": b_main, "score": conv(b_main),
                            "exp_085_recorded_score": 907.5995596662945,
                            "reproduced": abs(conv(b_main) - 907.5995596662945) < 1e-6,
                            "role": "🔴 사전 등록 판정 기준선 — 이것 하나로만 판정"},
    "aux_3way_AX": {"weights": {"AA67": 0.50, "S2": 0.28, "S1": 0.22},
                    "brier": b_3way, "score": conv(b_3way), "role": "부가"},
    "aux_NN_s42_alone": {"brier": b_nn1, "score": conv(b_nn1), "role": "부가"}}

GRID = np.round(np.arange(0.0, 0.8001, 0.005), 4)


def add_one(zbase, bb, zs):
    p0 = expit(zbase)
    g = np.array([K * (bb - brier(expit((1 - w) * zbase + w * zs))) for w in GRID])
    i = int(np.argmax(g))
    w = float(GRID[i])
    pm = expit((1 - w) * zbase + w * zs)
    se = se_of((p0 - y) ** 2 - (pm - y) ** 2)
    return {"w_star": w, "gain_pts": float(g[i]), "se_pts": se,
            "t": float(g[i] / se) if se else 0.0,
            "base_score": conv(bb), "blend_score": conv(brier(pm)),
            "grid": "0..0.8 step 0.005", "w_at_cap": bool(w >= 0.8 - 1e-9)}


# ── 🔴 사전 등록 판정 ────────────────────────────────────────────────────
verdict = {"threshold_t": 2.0, "baseline": "PRIMARY_NNavg4_AA67", "arms": {}}
for arm in ARMS:
    r = add_one(z_main, b_main, Zavg[arm])
    r["pass"] = bool(r["t"] >= 2.0)
    verdict["arms"][arm] = r
verdict["any_arm_passes"] = any(verdict["arms"][a]["pass"] for a in ARMS)
verdict["decision"] = ("통과 — 본 배포 경로 설계 착수 대상"
                       if verdict["any_arm_passes"] else
                       "기각 — 이 계열 종료. 파라미터 재탐색 금지(exp_054 판례)")
M["VERDICT_primary"] = verdict

# 같은 기준선 위 exp_085 1시드 값과 나란히
M["seed1_vs_seed4_on_primary"] = {
    "F1_FM": {"seed1_exp085": {"w_star": 0.045, "gain_pts": 0.42433427955570907,
                               "t": 0.387140851898784},
              "seed4_exp086": {k: verdict["arms"]["F1_FM"][k]
                               for k in ("w_star", "gain_pts", "t")}},
    "F2_EMB": {"seed1_exp085": {"w_star": 0.11, "gain_pts": 1.9015395460683866,
                                "t": 0.7971968709615096},
               "seed4_exp086": {k: verdict["arms"]["F2_EMB"][k]
                                for k in ("w_star", "gain_pts", "t")}}}

# ── 부가 1: 다른 기준선 위 (판정 아님) ───────────────────────────────────
M["aux_other_baselines"] = {
    "on_3way": {a: add_one(z_3way, b_3way, Zavg[a]) for a in ARMS},
    "on_NN_s42_alone": {a: add_one(z_nn1, b_nn1, Zavg[a]) for a in ARMS},
    "note": "🔴 판정에 쓰지 않는다. 기준선이 약할수록 이득이 커진다는 것을 보이는 용도"}

# ── 부가 2: 두 팔 동시 최적 가중 (S16 — naive 합산 금지, 2차원 동시 해) ──
def joint(zbase, bb, tag):
    p0 = expit(zbase)
    coarse = np.round(np.arange(0.0, 0.8001, 0.01), 4)
    best = {"gain_pts": -1e18}
    for wf in coarse:
        for we in coarse:
            if wf + we > 0.8 + 1e-9:
                continue
            g = K * (bb - brier(expit((1 - wf - we) * zbase
                                      + wf * Zavg["F1_FM"] + we * Zavg["F2_EMB"])))
            if g > best["gain_pts"]:
                best = {"w_FM": float(wf), "w_EMB": float(we), "gain_pts": float(g)}
    fine = lambda c: np.round(np.arange(max(0.0, c - 0.015), c + 0.0151, 0.0025), 5)
    for wf in fine(best["w_FM"]):
        for we in fine(best["w_EMB"]):
            if wf + we > 0.8 + 1e-9:
                continue
            g = K * (bb - brier(expit((1 - wf - we) * zbase
                                      + wf * Zavg["F1_FM"] + we * Zavg["F2_EMB"])))
            if g > best["gain_pts"]:
                best = {"w_FM": float(wf), "w_EMB": float(we), "gain_pts": float(g)}
    pm = expit((1 - best["w_FM"] - best["w_EMB"]) * zbase
               + best["w_FM"] * Zavg["F1_FM"] + best["w_EMB"] * Zavg["F2_EMB"])
    best["se_pts"] = se_of((p0 - y) ** 2 - (pm - y) ** 2)
    best["t"] = best["gain_pts"] / best["se_pts"] if best["se_pts"] else 0.0
    best["base_score"] = conv(bb)
    best["blend_score"] = conv(brier(pm))
    best["grid"] = "0..0.8 step 0.01 → argmax 주변 ±0.015 step 0.0025"
    sing = max(verdict["arms"]["F1_FM"]["gain_pts"],
               verdict["arms"]["F2_EMB"]["gain_pts"]) if tag == "primary" else None
    if sing is not None:
        best["gain_over_best_single_pts"] = best["gain_pts"] - sing
        best["naive_sum_of_singles_pts"] = (verdict["arms"]["F1_FM"]["gain_pts"]
                                            + verdict["arms"]["F2_EMB"]["gain_pts"])
        best["joint_minus_naive_sum_pts"] = (best["gain_pts"]
                                             - best["naive_sum_of_singles_pts"])
    return best


M["aux_joint_two_arms"] = {"on_primary": joint(z_main, b_main, "primary"),
                           "on_3way": joint(z_3way, b_3way, "3way")}

# ── 부가 3: 4시드 평균 후 상관 ───────────────────────────────────────────
series = dict(Z)
for arm in ARMS:
    series[f"{arm}_seedavg4"] = Zavg[arm]
    series[f"{arm}_s42_only"] = lg(np.load(f"{O85}/preds/{arm}_s42_r2_best.npy"))
names = list(series)
C = np.corrcoef(np.vstack([series[n] for n in names]))
ix = {n: i for i, n in enumerate(names)}
GATE_SET = ["AA67", "AA81", "S1", "S2", "T70",
            "mate_v6(tr2024 표본내)", "mate_Q(tr2024 표본내)"]
EXTRA = ["NN_s42", "NN_avg4", "p_EBM(tr2024 표본내, 이미 로짓 — S14)",
         "AA_deployed(tr2024 표본내)"]
corr = {"space": "로짓", "exp_085_1seed_max_abs_rho": {"F1_FM": 0.866801,
                                                      "F2_EMB": 0.885631}}
for arm in ARMS:
    for tag in ("seedavg4", "s42_only"):
        k = f"{arm}_{tag}"
        rg = {n: float(C[ix[k], ix[n]]) for n in GATE_SET}
        ex = {n: float(C[ix[k], ix[n]]) for n in EXTRA}
        mx = max(rg, key=lambda n: abs(rg[n]))
        corr[k] = {"rho_vs_gate_set": rg, "rho_vs_reference_only": ex,
                   "max_abs_rho": abs(rg[mx]), "argmax": mx,
                   "still_under_0.93": bool(abs(rg[mx]) <= 0.93)}
corr["rho_FM_seedavg4_vs_EMB_seedavg4"] = float(
    C[ix["F1_FM_seedavg4"], ix["F2_EMB_seedavg4"]])
corr["rho_FM_s42_vs_EMB_s42"] = float(C[ix["F1_FM_s42_only"], ix["F2_EMB_s42_only"]])
M["aux_corr_after_seedavg"] = corr

# ── 부가 4: 단독 성능 · 세그먼트 ─────────────────────────────────────────
def scoreblock(p):
    b = brier(p)
    r = {"brier": b, "score": conv(b)}
    for s in ("R", "F"):
        m = seg == s
        r[f"brier_{s}"] = float(np.mean((p[m] - y[m]) ** 2))
        r[f"n_{s}"] = int(m.sum())
    return r


M["standalone_out_of_time"] = {k: scoreblock(a) for k, a in A.items()}
for arm in ARMS:
    M["standalone_out_of_time"][f"{arm}_seedavg4"] = scoreblock(expit(Zavg[arm]))
    M["standalone_out_of_time"][f"{arm}_s42_only"] = scoreblock(
        np.load(f"{O85}/preds/{arm}_s42_r2_best.npy"))

M["wall_sec"] = round(time.time() - t0, 2)
json.dump(M, open(f"{O86}/metrics_core.json", "w"), ensure_ascii=False, indent=1)

print("=== 🔴 사전 등록 판정 (기준선 %.5f, 문턱 t >= 2.0) ===" % conv(b_main))
for arm in ARMS:
    r = verdict["arms"][arm]
    print(f"  {arm:8s} w* {r['w_star']:.3f}  이득 {r['gain_pts']:+10.5f}점  "
          f"SE {r['se_pts']:8.5f}  t {r['t']:6.3f}  "
          f"{'PASS' if r['pass'] else 'FAIL'}   블렌드 {r['blend_score']:.5f}")
print("  →", verdict["decision"])
print("=== 시드 1개 → 4개 (같은 기준선) ===")
for arm in ARMS:
    d = M["seed1_vs_seed4_on_primary"][arm]
    print(f"  {arm:8s} 1시드 {d['seed1_exp085']['gain_pts']:+8.5f}(t "
          f"{d['seed1_exp085']['t']:.3f}) → 4시드 "
          f"{d['seed4_exp086']['gain_pts']:+8.5f}(t {d['seed4_exp086']['t']:.3f})")
print("=== 팔 단독 점수 (4시드 평균) ===")
for arm in ARMS:
    b = seed_block[arm]
    print(f"  {arm:8s} 시드별 {['%.2f' % s for s in [b['per_seed'][s]['score'] for s in SEEDS]]} "
          f"평균 {b['per_seed_score_mean']:.5f} sd {b['per_seed_score_sd']:.5f} "
          f"→ 로짓평균 {b['seedavg_score']:.5f} (S_seed {b['S_seed_pts']:+.5f})")
print("=== 두 팔 동시 (기준선 위) ===")
j = M["aux_joint_two_arms"]["on_primary"]
print(f"  w_FM {j['w_FM']} w_EMB {j['w_EMB']} 이득 {j['gain_pts']:+.5f} t {j['t']:.3f} "
      f"| 단독합 {j['naive_sum_of_singles_pts']:+.5f} 차 {j['joint_minus_naive_sum_pts']:+.5f}")
print("=== 상관 max|ρ| ===")
for arm in ARMS:
    print(f"  {arm:8s} 1시드 {corr[arm + '_s42_only']['max_abs_rho']:.6f} "
          f"→ 4시드 {corr[arm + '_seedavg4']['max_abs_rho']:.6f} "
          f"(vs {corr[arm + '_seedavg4']['argmax']})")
print("wall", M["wall_sec"], "s")
