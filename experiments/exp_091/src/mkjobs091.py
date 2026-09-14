"""exp_091 — fit 작업 목록 생성. 축 셋을 한 파일로 펴서 6개 일꾼이 나눠 집는다.

고정: 총 스텝 1632(exp_090 고원 argmax) · oot23 · init ftt · lr 1e-3 상수 · batch 4096 · D 96.
축 A(시드 수)  : 새 시드 15개 (기존 5시드는 exp_090 산출물을 sha 대조 후 재사용 — 여기서 안 돈다)
축 B(구조)     : 폭 x{0.5,1,2} x 깊이 {-1, 0, +1}. (x1, 0) 은 기준선이라 안 돈다
축 C(정규화)   : dropout {0, 0.1, 0.2} x weight decay {0, 0.01, 0.1}. (0, 0.01) 은 기준선

⚠️ config 의 "dropout {0, 현행, 2배}" 는 **현행이 0 이라 그대로면 축퇴한다**(0의 2배도 0).
   그래서 0(=현행) 과 양의 두 수준 0.1 · 0.2 로 읽었다. 이 해석은 report 에 명시한다.
"""
import json, sys

S5 = [42, 777, 2024, 101, 7]                     # exp_089/090 시드 집합
S15 = [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67]  # 새 15개(소수)
STEPS = 1632
BASE_HID = (256, 128, 64)


def widen(hid, f):
    return tuple(int(h * f) for h in hid)


def depth(hid, d):
    if d == 0:
        return hid
    if d == -1:
        return hid[:-1]
    if d == 1:
        return hid + (hid[-1] // 2,)
    raise ValueError(d)


JOBS = []          # (tag, hid_str, drop, opt, seed)
POINTS = {}        # 지점 이름 → {"tags": [...], "meta": {...}}

# 축 A — 시드 수. 새 시드 15개만 돈다.
for s in S15:
    JOBS.append((f"A_s{s}", ",".join(map(str, BASE_HID)), 0.0, "adamw0.01", s))

# 축 B — 구조
for wf, wn in ((0.5, "w05"), (1.0, "w1"), (2.0, "w2")):
    for dd, dn in ((-1, "dm1"), (0, "d0"), (1, "dp1")):
        hid = depth(widen(BASE_HID, wf), dd)
        name = f"B_{wn}_{dn}"
        if wn == "w1" and dn == "d0":
            POINTS[name] = {"kind": "B", "hid": list(hid), "drop": 0.0,
                            "opt": "adamw0.01", "seeds": S5, "BASELINE": True}
            continue
        POINTS[name] = {"kind": "B", "hid": list(hid), "drop": 0.0,
                        "opt": "adamw0.01", "seeds": S5, "BASELINE": False}
        for s in S5:
            JOBS.append((f"{name}_s{s}", ",".join(map(str, hid)), 0.0,
                         "adamw0.01", s))

# 축 C — 정규화
for dr, drn in ((0.0, "dr0"), (0.1, "dr01"), (0.2, "dr02")):
    for wd, wdn in ((0.0, "wd0"), (0.01, "wd001"), (0.1, "wd01")):
        name = f"C_{drn}_{wdn}"
        base = (dr == 0.0 and wd == 0.01)
        POINTS[name] = {"kind": "C", "hid": list(BASE_HID), "drop": dr,
                        "opt": f"adamw{wd}", "seeds": S5, "BASELINE": base}
        if base:
            continue
        for s in S5:
            JOBS.append((f"{name}_s{s}", ",".join(map(str, BASE_HID)), dr,
                         f"adamw{wd}", s))

if __name__ == "__main__":
    with open(sys.argv[1], "w") as f:
        for tag, hid, dr, opt, seed in JOBS:
            f.write(f"{tag}\t{hid}\t{dr}\t{opt}\t{seed}\n")
    json.dump({"n_jobs": len(JOBS), "S5": S5, "S15": S15, "steps": STEPS,
               "points": POINTS},
              open(sys.argv[2], "w"), ensure_ascii=False, indent=1)
    print(f"{len(JOBS)} jobs")
