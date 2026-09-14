"""exp_114 stage1 — 재현·체크포인트 게이트. **fit 결과를 판정에 쓰기 전에** 통과해야 한다."""
import hashlib
import json
import os

import numpy as np

H = "/home/<user>/SungjinPark"
B = f"{H}/exp114"
REF = f"{H}/exp097/out/preds"
R = {"exp_id": "exp_114", "stage": "s1_gates"}

gate = json.load(open(f"{B}/out/fits/GATE.json"))
ck = {c["steps"]: c["z_sha256"] for c in gate["ckpts"]}
rows = []
for k in (544, 720, 816, 1088, 1360, 1632, 2176, 2720, 3264, 4080):
    p = f"{REF}/z_F{k}_f64o_s42.npy"
    if not os.path.exists(p):
        continue
    r = hashlib.sha256(np.ascontiguousarray(np.load(p)).tobytes()).hexdigest()
    rows.append({"steps": k, "mine": ck.get(k, "")[:16], "exp097": r[:16],
                 "MATCH": bool(ck.get(k) == r)})
R["G_A_reproduce_exp097"] = {
    "설명": "EXTRA 없는 94열 seed42 적합의 체크포인트가 exp_097 독립 적합과 바이트 동일한가. "
            "1632 의 기대 sha16 = b1da6808c7f16dea",
    "rows": rows, "n": len(rows), "n_match": sum(r["MATCH"] for r in rows),
    "PASS": bool(rows and all(r["MATCH"] for r in rows))}
R["G_B_s98_checkpoint"] = {
    "설명": "S98 — 상수 lr 이라 긴 적합의 앞부분이 짧은 적합과 비트 동일. G_A 가 그것을 "
            "10지점에서 증명한다(exp_097 은 각 스텝을 따로 구웠다).",
    "PASS": R["G_A_reproduce_exp097"]["PASS"]}

m42 = json.load(open(f"{B}/out/fits/M_s42.json"))
c42 = {c["steps"]: c["z_sha256"] for c in m42["ckpts"]}
R["G_C_determinism"] = {
    "설명": "같은 인자를 다른 GPU 레인에서 다시 적합 (GATE vs M_s42)",
    "n_ckpts": len(ck), "identical": bool(ck == c42),
    "wall": [gate["fit_wall_sec"], m42["fit_wall_sec"]],
    "PASS": bool(ck == c42)}
R["G_D_deployed_fold"] = {
    "설명": "일반화한 fold 가 배포 deep_infer._fold 와 일치 (n_num==94)",
    "max_over_fits": None, "PASS": None}
mx = 0.0
for p in sorted(os.listdir(f"{B}/out/fits")):
    d = json.load(open(f"{B}/out/fits/{p}"))
    mx = max(mx, max(d["deployed_fold_gate"].values()))
R["G_D_deployed_fold"]["max_over_fits"] = mx
R["G_D_deployed_fold"]["PASS"] = bool(mx == 0.0)
R["ALL_PASS"] = bool(all(R[k]["PASS"] for k in R if k.startswith("G_")))
json.dump(R, open(f"{B}/out/gates114.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1))
