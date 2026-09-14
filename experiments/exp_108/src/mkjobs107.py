"""exp_107 작업 목록. 형식: TAG|TGT_PATH|STEP|SEED   (빈 TGT = 무변형 기본 경로)

🔴 각 지점을 {816, 1632, 2720} 세 스텝에서 굽는다 — 자기 argmax 로 비교하기 위해서다(S183·S94).
🔴 5시드 고정 42/777/2024/101/7 (다양성·ρ 는 같은 시드 수에서만 비교, S72).
"""
import json, os, sys

B = "/home/<user>/SungjinPark/exp107"
S5 = [42, 777, 2024, 101, 7]
STEPS = [1632, 816, 2720]          # 1632 먼저 — 중간 판독용

PHASE = sys.argv[1] if len(sys.argv) > 1 else "1"
prep = json.load(open(f"{B}/out/s0_prep107.json"))
EA, EB = prep["grids"]["A_eps"], prep["grids"]["B_eps"]

POINTS = {}
if PHASE == "1":
    POINTS["N3_identity"] = ("N3_identity", "N3_항등복제본", "무변형 — 사다리 맨 아래(S135)")
    for e in EA:
        n = f"A{e:g}".replace(".", "p")
        POINTS[n] = (n, "A_라벨스무딩", f"y -> (1-e)y + e*0.5, e={e}")
    for e in EB:
        n = f"B{e:g}".replace(".", "p")
        POINTS[n] = (n, "B_자기증류", f"y -> (1-e)y + e*p_self, e={e} (커버 행만)")
    for e in EB:
        n = f"N2_B{e:g}".replace(".", "p")
        POINTS[n] = (n, "N2_귀무_p_self치환", f"p_self 무작위 치환, e={e}")
    POINTS["N1a_perm_e0"] = ("N1a_perm_e0", "N1_바닥대조",
                             "(season,pitcher) 안 라벨 순열, 스무딩 없음")
else:                                # phase 2 — eA*, eB* 확정 뒤
    for n in json.load(open(f"{B}/out/phase2_points.json")):
        POINTS[n[0]] = tuple(n)

lines, meta = [], {}
for nm, (tgt, axis, desc) in POINTS.items():
    p = "" if tgt is None else f"{B}/work/tgt_{tgt}.npy"
    if p:
        assert os.path.exists(p), p
    meta[nm] = {"axis": axis, "desc": desc, "target_file": p or None}
    for k in STEPS:
        for s in S5:
            lines.append(f"{nm}__k{k}_s{s}|{p}|{k}|{s}")

os.makedirs(f"{B}/logs", exist_ok=True)
open(f"{B}/logs/jobs{PHASE}.txt", "w").write("\n".join(lines) + "\n")
json.dump({"points": meta, "seeds": S5, "steps": STEPS, "phase": PHASE},
          open(f"{B}/out/jobs{PHASE}_meta.json", "w"), ensure_ascii=False, indent=1)
print(f"phase {PHASE}: {len(POINTS)} points, {len(lines)} fits")
for k in POINTS:
    print("  ", k)
