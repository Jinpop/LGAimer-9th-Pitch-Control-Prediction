"""exp_108 작업 목록. 형식: TAG|INIT|STEP|SEED

🔴 각 지점을 {816, 1632, 2720} 세 스텝에서 굽는다 — 자기 argmax 로 비교하기 위해서다(S183·S94).
🔴 5시드 고정 42/777/2024/101/7 (다양성·ρ 는 같은 시드 수에서만 비교, S72).
🔴 사다리 맨 아래는 `N3_identity`(INIT=ftt, 무변형 항등 복제본) 다 — S135.
"""
import json, os, sys

import numpy as np

B = "/home/<user>/SungjinPark/exp108"
S5 = [42, 777, 2024, 101, 7]
STEPS = [1632, 816, 2720]          # 1632 먼저 — 중간 판독용
D = 96
BD = 1.0 / np.sqrt(D)              # ftt 의 균등분포 경계 = D^-0.5 = 0.102062...

PHASE = sys.argv[1] if len(sys.argv) > 1 else "1"

POINTS = {}
if PHASE == "1":
    # ── 사다리 맨 아래: 무변형 (S135) ────────────────────────────────────
    POINTS["N3_identity"] = ("ftt", "N3_항등복제본",
                             f"현행 배포 재현 초기화 U(±{BD:.8f}) — 무변형")
    # ── 축 A: 스케일 ────────────────────────────────────────────────────
    #  (a) 균등 분포를 현행 대비 배율로 (config 의 '현행×{0.5, 2}')
    for m, nm in ((0.25, "A_u0p25"), (0.5, "A_u0p5"), (2.0, "A_u2"), (4.0, "A_u4")):
        POINTS[nm] = (f"u{m:g}", "A_스케일_균등",
                      f"w,b ~ U(±{m}/√D) = 현행 ftt × {m}")
    #  (b) 정규 분포 — 팀원 문구 'N(0, D^-0.5)' 의 두 읽기를 **둘 다** 넣는다
    for e, nm, note in ((0.50, "A_nD05",  "D^-0.5 를 **표준편차**로 읽음 (문자 그대로)"),
                        (0.25, "A_nD025", "D^-0.5 를 **분산**으로 읽음 (= sd D^-0.25)"),
                        (0.75, "A_nD075", "더 작은 쪽 브래킷 (sd D^-0.75)")):
        sd = float(D ** (-e))
        POINTS[nm] = (f"n{sd:.10f}", "A_스케일_정규",
                      f"w,b ~ N(0, sd={sd:.8f}) — {note}")
    #  (c) 위쪽 브래킷 — exp_089 probe 의 `randn` 지점(그때 단독 818.98)
    POINTS["A_randn"] = ("randn", "A_스케일_정규",
                         "w,b ~ N(0,1) = 현행의 약 17배. exp_089 probe 지점")
else:                                # phase 2 — A 판독 뒤 확정
    for n in json.load(open(f"{B}/out/phase2_points.json")):
        POINTS[n[0]] = tuple(n[1:])

lines, meta = [], {}
for nm, (init, axis, desc) in POINTS.items():
    meta[nm] = {"axis": axis, "desc": desc, "init": init}
    for k in STEPS:
        for s in S5:
            lines.append(f"{nm}__k{k}_s{s}|{init}|{k}|{s}")

os.makedirs(f"{B}/logs", exist_ok=True)
open(f"{B}/logs/jobs{PHASE}.txt", "w").write("\n".join(lines) + "\n")
json.dump({"points": meta, "seeds": S5, "steps": STEPS, "phase": PHASE,
           "D": D, "ftt_bound": BD, "ftt_sd": BD / np.sqrt(3.0)},
          open(f"{B}/out/jobs{PHASE}_meta.json", "w"), ensure_ascii=False, indent=1)
print(f"phase {PHASE}: {len(POINTS)} points, {len(lines)} fits")
for k, v in meta.items():
    print(f"   {k:16s} init={v['init']:16s} {v['desc']}")
