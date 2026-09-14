"""exp_113 — fit 작업 목록 생성. 4레인(GPU 2장 × 2) × 1스레드.

🔴 시드 묶음은 **서로소**다 (S187·S194 — 단일 뽑기는 평균보다 +0.78 위다).
   BUNDLE0 이 주 비교, BUNDLE1·2 는 이 화면의 1σ 를 직접 재는 데 쓴다.
"""
import json
import os

H = "/home/<user>/SungjinPark"
B = f"{H}/exp113"
BLK = f"{B}/work/blk"
PY = f"{H}/venv/bin/python"
SRC = f"{B}/src/train113.py"

BUNDLE = {0: [42, 777, 2024, 101, 7], 1: [11, 13, 17, 19, 23], 2: [29, 31, 37, 41, 43]}
MAIN = ["P", "R", "PR", "N1_P", "N1_R", "N1_PR"]
NULLS = ["N2_P", "N2_R", "N2_PR", "N2b_PR"]
SIGMA = ["PR", "N1_PR"]

jobs = []
def add(tag, seed, steps, extra):
    jobs.append((f"{tag}__k{steps}_s{seed}", seed, steps, extra))

add("GATE_base94", 42, 1632, "")                       # 94열 재현 게이트
for t in MAIN + NULLS:
    for s in BUNDLE[0]:
        for k in (1632, 2176):
            add(t, s, k, f"{BLK}/{t}.npy")
for t in MAIN:
    for s in BUNDLE[0]:
        add(t, s, 2720, f"{BLK}/{t}.npy")
for t in SIGMA:
    for b in (1, 2):
        for s in BUNDLE[b]:
            add(t, s, 1632, f"{BLK}/{t}.npy")

os.makedirs(f"{B}/logs", exist_ok=True)
lines = []
for i, (tag, seed, steps, extra) in enumerate(jobs):
    dev = f"cuda:{i % 2}"
    lines.append(f"{PY} {SRC} {seed} {dev} {steps} {tag} {extra} "
                 f"> {B}/logs/fit_{tag}.log 2>&1")
open(f"{B}/work/jobs.txt", "w").write("\n".join(lines) + "\n")
json.dump({"n_jobs": len(jobs), "bundles": BUNDLE, "main": MAIN, "nulls": NULLS,
           "sigma_extra_bundles": SIGMA,
           "steps_main": [1632, 2176, 2720], "steps_nulls": [1632, 2176]},
          open(f"{B}/out/jobs.json", "w"), ensure_ascii=False, indent=1)
print(len(jobs), "jobs")
