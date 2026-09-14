"""정직면 OOT-2024 Q 팔 대조: exp069 CPU seed42 vs 우리 GPU seed42/seed7."""
import json, numpy as np, pandas as pd
H = "/home/<user>/SungjinPark"
REF = 0.2498069
tr = pd.read_csv(f"{H}/data/train.csv", encoding="utf-8-sig",
                 usecols=["row_id", "season", "control_success"])
m24 = tr.season.to_numpy() == 2024
file_order = tr.loc[m24, "row_id"].to_numpy().astype(str)          # feat3 = train.csv 순서
sorted_idx = np.argsort(file_order, kind="stable")
rid69 = np.asarray(np.load(f"{H}/exp069/oot/row_id_oot2024.npy", allow_pickle=True)).astype(str)
same = bool(np.array_equal(file_order, rid69))
assert np.array_equal(file_order[sorted_idx], rid69), "row_id 정렬로도 안 맞는다"
y = tr.loc[m24, "control_success"].to_numpy(np.float64)            # feat3 순서
def pts(b): return 100000.0 * (1.0 - b / REF)
def sc(p, yy): return pts(float(np.mean((p - yy) ** 2)))

P = {}
p69 = np.load(f"{H}/exp069/oot/p_q_oot2024.npy").astype(np.float64)  # row_id 정렬 순서
inv = np.empty_like(sorted_idx); inv[sorted_idx] = np.arange(len(sorted_idx))
P["cpu42_exp069"] = p69[inv]                                        # feat3 순서로 되돌림
for tag, f in (("gpu42", "gpu_s42"), ("gpu7", "gpu_s7")):
    P[tag] = np.load(f"{H}/qseed/verify/oot/p_q_oot2024_{f}.npy").astype(np.float64)
R = {"row_order_identical_without_sort": same, "n": int(m24.sum()),
     "exp089_Q_reference": 859.79, "arms": {}, "pairs": {}}
for k, v in P.items():
    R["arms"][k] = {"score_oot2024": sc(v, y), "brier": float(np.mean((v - y) ** 2)),
                    "p_mean": float(v.mean()), "p_sd": float(v.std())}
ks = list(P)
for i in range(len(ks)):
    for j in range(i + 1, len(ks)):
        a, b = ks[i], ks[j]
        d = np.abs(P[a] - P[b])
        R["pairs"][f"{a}__vs__{b}"] = {
            "max_abs_dp": float(d.max()), "mean_abs_dp": float(d.mean()),
            "rho": float(np.corrcoef(P[a], P[b])[0, 1]),
            "score_diff": R["arms"][a]["score_oot2024"] - R["arms"][b]["score_oot2024"]}
print(json.dumps(R, ensure_ascii=False, indent=1))
json.dump(R, open(f"{H}/qseed/verify/oot/cmp_oot.json", "w"), ensure_ascii=False, indent=1)
