"""exp_088 stage 0c — 관문 A. 우리 팔 로짓 ↔ 팀원 딥 팔 로짓의 |rho|.

면: fold_2024 valid_all 253,507행 (arms/ids.npy 정렬). 두 팔 다 **같은 행·같은 99열**을 본다.
시드 등급: 우리 4시드(42/7/2024/101) ↔ 딥 4시드(같은 시드 라벨) 를 1차로, 딥 5시드를 병기.

🔴 사전 등록 판정 (S8 — 결과를 보고 바꾸지 않는다)
   |rho| < 0.90 → stage1 진행 / >= 0.95 → 그 팔 종료 / 0.90~0.95 → 정지, 직교 성분 보고
"""
import hashlib, json, os
import numpy as np

H = "/home/<user>/SungjinPark"
B = f"{H}/exp088"
REF = 0.2498069


def lg(p):
    p = np.clip(np.asarray(p, np.float64), 1e-12, 1 - 1e-12)
    return np.log(p) - np.log1p(-p)


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def shaf(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


R = {"exp_id": "exp_088", "stage": "s0c_gateA"}
y = np.load(f"{H}/exp085/work/y.npy")[np.load(f"{H}/exp085/work/valid_mask.npy")]
y = y.astype(np.float64)
n = len(y)

V = {}
SRC = {}
def add(name, arr, path, kind):
    V[name] = np.asarray(arr, np.float64)
    SRC[name] = {"path": path.replace(H, "~") if path else None, "kind": kind,
                 "sha256": shaf(path) if path else sha_arr(arr)}

# 우리 팔
add("F1_FM_4seed", lg(np.load(f"{H}/exp086/out/preds/F1_FM_seedavg4.npy")),
    f"{H}/exp086/out/preds/F1_FM_seedavg4.npy", "prob->logit")
add("F2_EMB_4seed", lg(np.load(f"{H}/exp086/out/preds/F2_EMB_seedavg4.npy")),
    f"{H}/exp086/out/preds/F2_EMB_seedavg4.npy", "prob->logit")
add("F1_FM_1seed42", lg(np.load(f"{H}/exp085/out/preds/F1_FM_s42_r2_best.npy")),
    f"{H}/exp085/out/preds/F1_FM_s42_r2_best.npy", "prob->logit")
add("F2_EMB_1seed42", lg(np.load(f"{H}/exp085/out/preds/F2_EMB_s42_r2_best.npy")),
    f"{H}/exp085/out/preds/F2_EMB_s42_r2_best.npy", "prob->logit")
# 팀원 딥 팔 (route1 — 우리 면, 정직 2024 입력)
add("DEEP_5seed", np.load(f"{B}/out/preds/deep_route1_logit_seed5.npy"),
    f"{B}/out/preds/deep_route1_logit_seed5.npy", "logit")
add("DEEP_4seed", np.load(f"{B}/out/preds/deep_route1_logit_seed4.npy"),
    f"{B}/out/preds/deep_route1_logit_seed4.npy", "logit")
add("DEEP_1seed42", np.load(f"{B}/out/preds/deep_route1_s42_logit.npy"),
    f"{B}/out/preds/deep_route1_s42_logit.npy", "logit")
# 기존 팔 (문맥)
add("mate_v6", np.load(f"{H}/arms/lg_p_v6.npy"), f"{H}/arms/lg_p_v6.npy", "logit")
add("mate_Q", np.load(f"{H}/arms/lg_p_q.npy"), f"{H}/arms/lg_p_q.npy", "logit")
add("p_EBM", np.load(f"{H}/arms/p_ebm.npy"), f"{H}/arms/p_ebm.npy", "logit(S14)")
add("NN_ours_4seed", lg(np.load(f"{H}/exp064/out/preds/A99ship_seedavg.npy")),
    f"{H}/exp064/out/preds/A99ship_seedavg.npy", "prob->logit")
add("AA67", lg(np.load(f"{H}/experiments/exp_048/pred/pred_A1_u2024_seed42_at1400.npy")),
    f"{H}/experiments/exp_048/pred/pred_A1_u2024_seed42_at1400.npy", "prob->logit")
add("T70", lg(np.load(f"{H}/experiments/exp_055/pred/pred_T70_u2024_seed42_at1400.npy")),
    f"{H}/experiments/exp_055/pred/pred_T70_u2024_seed42_at1400.npy", "prob->logit")
for k, v in V.items():
    assert v.shape == (n,), (k, v.shape)
R["vectors"] = SRC
R["n_rows"] = n

names = list(V)
M = np.vstack([V[k] for k in names])
C = np.corrcoef(M)
R["corr_matrix"] = {"names": names,
                    "rho": {a: {b: float(C[i, j]) for j, b in enumerate(names)}
                            for i, a in enumerate(names)}}

# 관문 A
OURS = ["F1_FM_4seed", "F2_EMB_4seed"]
G = {}
for a in OURS:
    o = {}
    for d in ("DEEP_4seed", "DEEP_5seed"):
        r = float(np.corrcoef(V[a], V[d])[0, 1])
        # 직교 성분: 우리 팔 로짓을 딥 로짓에 회귀했을 때 남는 잔차의 sd
        b = np.polyfit(V[d], V[a], 1)
        res = V[a] - (b[0] * V[d] + b[1])
        o[d] = {"rho": r, "abs_rho": abs(r),
                "resid_sd": float(res.std()),
                "our_sd": float(V[a].std()),
                "resid_share_of_our_sd": float(res.std() / V[a].std()),
                "verdict": ("PASS(<0.90)" if abs(r) < 0.90
                            else "REJECT(>=0.95)" if abs(r) >= 0.95 else "GRAY(0.90~0.95)")}
    o["1seed_vs_DEEP_1seed42"] = float(
        np.corrcoef(V[a.replace("_4seed", "_1seed42")], V["DEEP_1seed42"])[0, 1])
    G[a] = o
R["GATE_A"] = G

# 표본 내 경고용: 각 팔의 단독 성능 (딥은 2024 표본 내 — 비교 불가, 기록만)
def sc(z):
    p = 1.0 / (1.0 + np.exp(-z))
    br = float(np.mean((p - y) ** 2))
    return {"brier": br, "score": 100000.0 * (1.0 - br / REF)}
R["standalone_on_this_surface"] = {k: sc(V[k]) for k in names}
R["🔴_standalone_warning"] = ("DEEP_* 는 <=2024 학습이라 이 면(2024)에서 **표본 내**다. "
                              "우리 팔·AA·T70·NN 은 <=2023 학습 out-of-time. "
                              "단독 점수를 서로 비교하지 말 것 — 상관 판정용 문맥일 뿐이다")
R["reproduce_check_exp086"] = {
    "F2_EMB_4seed_vs_mate_v6": float(np.corrcoef(V["F2_EMB_4seed"], V["mate_v6"])[0, 1]),
    "exp086_recorded": 0.920703,
    "F1_FM_4seed_vs_mate_v6": float(np.corrcoef(V["F1_FM_4seed"], V["mate_v6"])[0, 1]),
    "exp086_recorded_F1": 0.866074,
    "F2_EMB_4seed_vs_NN_ours_4seed": float(
        np.corrcoef(V["F2_EMB_4seed"], V["NN_ours_4seed"])[0, 1]),
    "exp086_recorded_NN": 0.952551,
}
json.dump(R, open(f"{B}/out/s0c_gateA.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: R[k] for k in ("n_rows", "GATE_A", "reproduce_check_exp086")},
                 ensure_ascii=False, indent=1))
print("--- rho vs DEEP_5seed ---")
for k in names:
    print(f"  {k:16s} {C[names.index(k), names.index('DEEP_5seed')]:+.6f}")
print("--- standalone score (표본 내 경고) ---")
for k in names:
    print(f"  {k:16s} {R['standalone_on_this_surface'][k]['score']:10.5f}")
