"""exp_114 — fit 목록. 4레인(GPU 2장 x 2) x 1스레드(S102).
주 격자: 서로소 5시드 묶음 5개 = 25 fit, 각 4352스텝 1회 + 23 체크포인트(S98).
축척 시험: frac 0.50 / 0.75 x 묶음0 = 10 fit. 배포는 OOT 대비 1.3214배 데이터라
  '최적 스텝이 n 에 비례하는가(에폭 법칙)' 대 '고정인가(스텝 법칙)' 를 이걸로 가른다.
"""
import json
import os

H = "/home/<user>/SungjinPark"
B = f"{H}/exp114"
PY = f"{H}/venv/bin/python"
SRC = f"{B}/src/train114.py"
BUNDLE = {0: [42, 777, 2024, 101, 7], 1: [11, 13, 17, 19, 23], 2: [29, 31, 37, 41, 43],
          3: [47, 53, 59, 61, 67], 4: [71, 73, 79, 83, 89]}
TOTAL = 4352

jobs = []
for b in range(5):
    for s in BUNDLE[b]:
        jobs.append((f"M_s{s}", s, TOTAL, 1.0))
for fr, nm in ((0.50, "f50"), (0.75, "f75")):
    for s in BUNDLE[0]:
        jobs.append((f"{nm}_s{s}", s, TOTAL, fr))

lines = [f"env EXP113_W=/dev/shm/e113 {PY} {SRC} {seed} cuda:{i % 2} {steps} {tag} {fr} "
         f"> {B}/logs/fit_{tag}.log 2>&1"
         for i, (tag, seed, steps, fr) in enumerate(jobs)]
os.makedirs(f"{B}/work", exist_ok=True)
open(f"{B}/work/jobs.txt", "w").write("\n".join(lines) + "\n")
json.dump({"n_jobs": len(jobs), "bundles": BUNDLE, "total_steps": TOTAL,
           "fracs": [1.0, 0.75, 0.50]},
          open(f"{B}/out/jobs.json", "w"), ensure_ascii=False, indent=1)
print(len(jobs), "jobs")
