"""exp_089 stage 1 — probe 집계 + 복제 검증.

두 축으로 본다.
  ① **접힌 지문**: 우리 fit 의 A·c0·T{0..4} sd 가 출하 5시드 범위 안에 드는가.
     이건 예측을 안 보고 **구조·최적화기 자체**를 재는 검사다.
  ② **예측 상관**: 팀원 5시드 로짓평균과의 rho (게이트 지표).

🔴 게이트 천장: 팀원 팔의 시드 간 rho r 이 유한하므로, 구조가 **완벽해도**
   5시드평균끼리의 rho 는 5·r/(1+4·r) 를 못 넘는다. 그 값을 함께 낸다.
"""
import glob, itertools, json, os, sys
import numpy as np

H = "/home/<user>/SungjinPark"
B = f"{H}/exp089"
E88 = f"{H}/exp088/out/preds"
REF = 0.2498069
SEEDS = [42, 777, 2024, 101, 7]
PAT = sys.argv[1] if len(sys.argv) > 1 else "P_*"

y = np.load(f"{B}/work/y.npy")[np.load(f"{B}/work/valid_mask.npy")].astype(np.float64)
Zm = {s: np.load(f"{E88}/deep_route1_s{s}_logit.npy") for s in SEEDS}
zm5 = np.mean([Zm[s] for s in SEEDS], axis=0)
rw = float(np.mean([np.corrcoef(Zm[a], Zm[b])[0, 1]
                    for a, b in itertools.combinations(SEEDS, 2)]))

G = json.load(open(f"{B}/out/s0_gate.json"))
RANGE = G["shipped_fingerprint_range"]
FPK = ["A", "c0", "T0", "T1", "T2", "T3", "T4"]


def br(z):
    p = 1.0 / (1.0 + np.exp(-z))
    return float(np.mean((p - y) ** 2))


def pts(b):
    return 100000.0 * (1.0 - b / REF)


R = {"exp_id": "exp_089", "stage": "s1_probe",
     "teammate_within_seed_rho": rw,
     "ceiling_rho_5v5_if_perfect": 5 * rw / (1 + 4 * rw),
     "teammate_deep_alone": {"brier": br(zm5), "score": pts(br(zm5)),
                             "z_sd": float(zm5.std()), "z_mean": float(zm5.mean())},
     "y_mean": float(y.mean())}

rows = []
for f in sorted(glob.glob(f"{B}/out/fit_{PAT}.json")):
    J = json.load(open(f))
    tag = J["tag"]
    z = np.load(f"{B}/out/preds/z_{tag}.npy")
    fp = J["folded_fingerprint_sd"]
    inrange = {k: bool(RANGE[k]["min"] <= fp[k] <= RANGE[k]["max"]) for k in FPK}
    ratio = {k: fp[k] / np.sqrt(RANGE[k]["min"] * RANGE[k]["max"]) for k in FPK}
    logd = float(np.mean([abs(np.log(ratio[k])) for k in FPK]))
    rows.append({
        "tag": tag, "init": J["init"], "drop": J["drop"], "opt": J["opt"],
        "split": J["split"], "seed": J["seed"],
        "rho_vs_mate5": float(np.corrcoef(z, zm5)[0, 1]),
        "rho_vs_mate_s42": float(np.corrcoef(z, Zm[42])[0, 1]),
        "mean_rho_vs_each_mate_seed": float(np.mean(
            [np.corrcoef(z, Zm[s])[0, 1] for s in SEEDS])),
        "z_sd": J["z_sd"], "z_mean": J["z_mean"],
        "brier": J["valid_brier_raw"], "score": J["valid_score_raw"],
        "train_logloss_ep2": J["curve"][-1]["train_logloss"],
        "fp_ratio_to_shipped": {k: round(ratio[k], 4) for k in FPK},
        "fp_in_shipped_range": inrange,
        "fp_n_in_range": int(sum(inrange.values())),
        "fp_mean_abs_log_ratio": logd,
        "fit_wall_sec": J["fit_wall_sec"],
    })
rows.sort(key=lambda r: (-r["fp_n_in_range"], r["fp_mean_abs_log_ratio"]))
R["fits"] = rows
json.dump(R, open(f"{B}/out/s1_probe.json", "w"), ensure_ascii=False, indent=1)
print(f"teammate within-seed rho {rw:.6f} → 5v5 천장 "
      f"{5*rw/(1+4*rw):.6f}   deep 단독 score {pts(br(zm5)):.3f} z_sd {zm5.std():.4f}")
print(f"{'tag':30s} {'rho5':>8s} {'rho_s42':>8s} {'z_sd':>7s} {'score':>9s} "
      f"{'fpIN':>5s} {'fpD':>7s}  A/c0/T0/T4 ratio")
for r in rows:
    fr = r["fp_ratio_to_shipped"]
    print(f"{r['tag']:30s} {r['rho_vs_mate5']:8.5f} {r['rho_vs_mate_s42']:8.5f} "
          f"{r['z_sd']:7.4f} {r['score']:9.2f} {r['fp_n_in_range']:5d} "
          f"{r['fp_mean_abs_log_ratio']:7.3f}  "
          f"{fr['A']:.3f}/{fr['c0']:.3f}/{fr['T0']:.3f}/{fr['T4']:.3f}")
