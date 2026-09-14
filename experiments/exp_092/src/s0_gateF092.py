"""exp_092 축 F 실현성 조사 — **기억이 아니라 기계로** 판정한다.

F 가 요구하는 것: 학습 행(oot23 = 1,116,277행, season<=2023)에 대해 **그 행보다 앞선
데이터만으로 만든**(out-of-time) 팔 예측. 그것을 로짓 오프셋으로 두고 딥 팔이 잔차를 학습한다.
🔴 표본 내(in-sample) 오프셋을 쓰면 잔차가 인위적으로 작아져 측정이 전부 무의미해진다.

이 스크립트는 서버에 있는 모든 예측 산출물의 **길이**를 훑어 그런 벡터가 있는지 본다.
길이가 253,507(=valid 2024)뿐이면 전부 "≤2023 학습 → 2024 예측"이라 학습 행을 안 덮는다.
"""
import glob, json, os
import numpy as np

H = "/home/<user>/SungjinPark"
N_ALL, N_OOT23, N_VALID = 1475092, 1116277, 253507
ROOTS = [f"{H}/experiments/exp_049/pred", f"{H}/experiments/exp_055/pred",
         f"{H}/exp069/oot", f"{H}/exp070/oot3", f"{H}/exp071/out",
         f"{H}/oof_new", f"{H}/experiments/exp_051/oof",
         f"{H}/build_AH/localref/oof", f"{H}/arms"]
R = {"exp_id": "exp_092", "stage": "s0_gateF",
     "question": ("학습 행(oot23 1,116,277행)에 대한 out-of-time 팔 예측이 있는가 — "
                  "있으면 축 F(잔차 타깃)를 돌리고, 없으면 미착수로 적는다"),
     "n_all_rows": N_ALL, "n_oot23_train_rows": N_OOT23, "n_valid_2024": N_VALID,
     "scanned_roots": ROOTS}
lens, hits = {}, []
for r in ROOTS:
    if not os.path.isdir(r):
        lens[r] = "MISSING_DIR"
        continue
    per = {}
    for p in sorted(glob.glob(f"{r}/*.npy")) + sorted(glob.glob(f"{r}/*.npz")):
        try:
            if p.endswith(".npz"):
                with np.load(p, allow_pickle=True) as z:
                    for k in z.files:
                        a = z[k]
                        n = int(a.shape[0]) if a.ndim >= 1 else 0
                        per[f"{os.path.basename(p)}::{k}"] = n
                        if n in (N_ALL, N_OOT23):
                            hits.append({"path": p, "key": k, "n": n})
            else:
                a = np.load(p, mmap_mode="r", allow_pickle=True)
                n = int(np.asarray(a).shape[0]) if np.asarray(a).ndim >= 1 else 0
                per[os.path.basename(p)] = n
                if n in (N_ALL, N_OOT23):
                    hits.append({"path": p, "key": None, "n": n})
        except Exception as e:                       # noqa: BLE001
            per[os.path.basename(p)] = f"ERR {type(e).__name__}"
    cnt = {}
    for v in per.values():
        cnt[str(v)] = cnt.get(str(v), 0) + 1
    lens[r] = {"n_files_scanned": len(per), "length_histogram": cnt,
               "sample": dict(list(per.items())[:6])}
R["by_root"] = lens
R["candidate_train_row_vectors"] = hits
R["FEASIBLE"] = bool(hits)
R["verdict"] = ("가능 — 위 벡터로 오프셋을 만든다" if hits else
                "🔴 불가 — 학습 행을 덮는 팔 예측이 한 개도 없다. 모든 납품 벡터가 "
                "'≤2023 적합 → 2024 예측'(길이 253,507)이라 학습 행에 대해서는 표본 내이거나 "
                "아예 존재하지 않는다. 축 F 는 **미착수**로 적고 건너뛴다 (config 지시).")
os.makedirs(f"{H}/exp092/out", exist_ok=True)
json.dump(R, open(f"{H}/exp092/out/s0_gateF.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in R.items() if k != "by_root"},
                 ensure_ascii=False, indent=1))
for k, v in lens.items():
    print(k, json.dumps(v, ensure_ascii=False)[:300])
