"""exp_108 phase 2 작업 목록. 형식: TAG|INIT|STEP|SEED

🔴 왜 phase 2 인가 (문턱 변경이 아니다 — S8 은 그대로다):
   phase 1 의 상위 세 지점(`A_u0p25`·`A_u0p5`·`A_nD075`)이 **자기 argmax 를 스텝 격자의
   최소값 816 에서** 잡았고, `A_randn`·`A_u4`·`A_nD025` 는 **최대값 2720 에서** 잡았다.
   격자 경계에 닿은 argmax 는 무효라는 것이 우리 규약(라운드 cap 규칙)이므로 격자를 넓힌다.
   사전 등록 문턱(+9.31 · t ≥ 2.0 · 1σ 4.654)은 **한 글자도 안 바꾼다.**

🔴 phase 2 가 답할 결정적 질문:
   초기화 스케일이 **학습량 축의 재매개변수화일 뿐인가**, 아니면 그 너머를 주는가.
   · 재매개변수화라면 → 스텝을 넉넉히 훑었을 때 **스케일별 포락선(각자 최고점)이 평평**해야 한다.
     그러면 이 축은 새 축이 아니라 이미 LB 로 산 학습량 축(S111, CE +4.828)이다.
   · 포락선이 스케일이 작아질수록 계속 오르면 → 새 축이다.
"""
import json, os, sys

import numpy as np

B = "/home/<user>/SungjinPark/exp108"
S5 = [42, 777, 2024, 101, 7]
D = 96
BD = 1.0 / np.sqrt(D)

DONE = [816, 1632, 2720]          # phase 1 에서 이미 구운 스텝
NEW = [272, 408, 544, 1088]       # phase 2 가 더하는 스텝

# (태그, INIT, 스텝 목록, 축, 설명)
POINTS = [
    # ── 이미 3스텝이 있는 지점: 새 스텝만 채운다 ──────────────────────────
    ("N3_identity", "ftt", NEW, "N3_항등복제본",
     "무변형 — 사다리 맨 아래(S135). 새 스텝에서도 기준선이어야 한다"),
    ("A_u0p25", "u0.25", NEW, "A_스케일_균등", "현행 ftt × 0.25"),
    ("A_u0p5", "u0.5", NEW, "A_스케일_균등", "현행 ftt × 0.5"),
    ("A_nD075", "n0.0326059288", NEW, "A_스케일_정규", "N(0, sd=D^-0.75)"),
    # ── 아래쪽 연장: 스케일을 더 줄이면 계속 오르는가 ─────────────────────
    ("A_u0p125", "u0.125", [272, 408, 544, 816, 1088, 1632], "A_스케일_균등",
     "현행 ftt × 0.125"),
    ("A_u0p06", "u0.0625", [272, 408, 544, 816, 1088, 1632], "A_스케일_균등",
     "현행 ftt × 0.0625"),
    # ── 귀무 N4: 같은 크기의 스케일을 *이웃 블록* ce.* 에 준다 ────────────
    ("N4_A_u0p25", "ftt@ce0.25", [272, 408, 544, 816, 1088, 1632], "N4_형태맞춤_귀무",
     "토크나이저는 ftt 그대로 두고 범주 임베딩 ce.* 만 × 0.25 — "
     "'초기화를 흔들면 아무거나 이만큼 달라진다' 를 배제한다"),
    # ── 축 C: 층별 (w 만 / b 만) — exp_089 의 c0 지문이 b 쪽을 가리켰다 ──
    ("C_wonly_0p25", "uv0.25", [408, 544, 816, 1632], "C_층별",
     "w 만 × 0.25, b 는 ftt 그대로"),
    ("C_bonly_0p25", "ub0.25", [408, 544, 816, 1632], "C_층별",
     "b 만 × 0.25, w 는 ftt 그대로"),
]

lines, meta = [], {}
if os.path.exists(f"{B}/out/jobs1_meta.json"):
    meta.update(json.load(open(f"{B}/out/jobs1_meta.json"))["points"])
for nm, init, steps, axis, desc in POINTS:
    prev = meta.get(nm, {}).get("steps", DONE if nm in meta else [])
    meta[nm] = {"axis": axis, "desc": desc, "init": init,
                "steps": sorted(set(list(prev) + list(steps)))}
    for k in steps:
        for s in S5:
            lines.append(f"{nm}__k{k}_s{s}|{init}|{k}|{s}")

open(f"{B}/logs/jobs2.txt", "w").write("\n".join(lines) + "\n")
json.dump({"points": meta, "seeds": S5, "phase": "2", "D": D,
           "steps_phase1": DONE, "steps_added": NEW, "ftt_bound": BD},
          open(f"{B}/out/jobs2_meta.json", "w"), ensure_ascii=False, indent=1)
print(f"phase 2: {len(POINTS)} point-jobs, {len(lines)} fits")
for nm, init, steps, axis, desc in POINTS:
    print(f"   {nm:16s} init={init:16s} steps={steps}")
