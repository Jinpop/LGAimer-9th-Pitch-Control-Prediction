"""exp_104 축A 후속 — `D5h3_rand47` 의 +7.41 이 되풀이되는가, 그리고 이 계열의 잡음 분해.

무작위 47열 마스크 3개(h1·h2·h3) × 서로소 5시드 묶음 3개(총 9점, 45 fit)를 한 표로 놓고
분산을 둘로 가른다:
  · **묶음-안**(같은 마스크, 시드 묶음만 다름)  = 이 계열의 적합-간 sd
  · **마스크-사이**(어느 47열을 골랐나)         = 귀무 가족의 진짜 퍼짐
그래야 "무작위 마스크 하나가 +7.4 를 냈다" 가 신호인지 뽑기인지 말할 수 있다.
🔴 사전 등록 문턱은 바꾸지 않는다(S8). 이건 문턱을 다시 재는 자다.
"""
import json, os, sys, time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np

H = "/home/<user>/SungjinPark"
B = f"{H}/exp104"
sys.path.insert(0, f"{B}/src")
import surface104 as S                                          # noqa: E402

t0 = time.time()
GRID = [round(x, 4) for x in np.arange(0.0, 0.6001, 0.01)]
GROUPS = {"G1_S5": [42, 777, 2024, 101, 7],
          "G2": [11, 13, 17, 19, 23], "G3": [29, 31, 37, 41, 43]}
NAMES = ["D5h1_rand47", "D5h2_rand47", "D5h3_rand47"]


def best(dz):
    cur = [(w, S.sc(S.root((1 - w) * S.Z_MATE + w * dz), "abd") - S.BASE_ABD)
           for w in GRID]
    w, g = max(cur, key=lambda c: c[1])
    return {"w": w, "gain": g}


R = {"exp_id": "exp_104", "stage": "sC_replicate", "steps": 1632,
     "groups": GROUPS, "표": {}}
M = np.zeros((3, 3))
for i, nm in enumerate(NAMES):
    R["표"][nm] = {}
    for j, (gn, gs) in enumerate(GROUPS.items()):
        dz = np.mean([np.load(f"{B}/out/preds/z_{nm}__k1632_s{s}.npy") for s in gs], 0)
        b = best(dz)
        M[i, j] = b["gain"]
        R["표"][nm][gn] = {"argmax": b, "alone_recentred": S.sc(dz, "rc"),
                           "rho_vs_z_mate": float(np.corrcoef(dz, S.Z_MATE)[0, 1])}
        print(f"{nm:14s} {gn:6s} w {b['w']:.2f} gain {b['gain']:+9.4f}", flush=True)

within = float(np.mean(np.var(M, axis=1, ddof=1)))
grand = float(M.mean())
mask_means = M.mean(axis=1)
between_raw = float(np.var(mask_means, ddof=1))
between = max(0.0, between_raw - within / 3.0)
R["분산분해"] = {
    "표_3마스크x3묶음": M.tolist(),
    "grand_mean": grand,
    "within_mask_sd_적합간": float(np.sqrt(within)),
    "between_mask_sd_귀무가족": float(np.sqrt(between)),
    "between_raw_sd_of_mask_means": float(np.sqrt(between_raw)),
    "🔴": "묶음-안 sd 는 같은 마스크를 다른 5시드로 다시 구웠을 때의 흔들림이고, "
          "마스크-사이 sd 는 '어느 47열을 골랐나' 가 만드는 진짜 퍼짐이다. "
          "후보를 기준선과 견줄 때 써야 할 자는 앞의 것이다."}
h3 = M[2]
R["D5h3_재현_판정"] = {
    "gains": h3.tolist(), "mean": float(h3.mean()),
    "sd": float(h3.std(ddof=1)),
    "delta_vs_26.2698_per_group": (h3 - S.PREREG_BASELINE).tolist(),
    "mean_delta": float(h3.mean() - S.PREREG_BASELINE),
    "t_of_mean_delta_vs_within_sd": float(
        (h3.mean() - S.PREREG_BASELINE) / (np.sqrt(within) / np.sqrt(3))),
    "🔴": "첫 묶음의 +7.41 이 되풀이되면 이 계열에 신호가 있다는 뜻이고, "
          "되풀이 안 되면 그 값은 시드 뽑기다."}
R["wall_sec"] = round(time.time() - t0, 1)
json.dump(R, open(f"{B}/out/sC_replicate.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({"분산분해": {k: v for k, v in R["분산분해"].items() if k != "표_3마스크x3묶음"},
                  "D5h3_재현_판정": R["D5h3_재현_판정"]}, ensure_ascii=False, indent=1))
