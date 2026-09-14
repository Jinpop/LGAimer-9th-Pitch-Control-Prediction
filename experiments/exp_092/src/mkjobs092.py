"""exp_092 — fit 작업 목록 생성. 축 D(피처 부분집합)·E(손실) 를 한 파일로 편다.

고정: 총 스텝 1632 · oot23 · init ftt · lr 1e-3 상수 · batch 4096 · D 96 ·
      폭 256/깊이 3 · dropout 0 · wd 0.01 · 5시드 [42,777,2024,101,7].
D0(전체 99열 · BCE) = 기준선이며 **exp_090 의 K1632_oot23_s* 를 sha 대조 후 재사용**한다.

축 D — 열을 빼는 대신 **표준화 공간의 0 으로 덮는다**(S101, 99토큰 형상 유지).
  D1 sp_*        D2 ins_*        D3 i3_anom_* + p6_*        D4 범주5 + 상황11열만 남긴다
  🔴 D5 = 귀무 대조. config 가 요구한 **무작위 절반 ×3** 에 더해,
     **D1~D4 와 마스크 개수를 정확히 맞춘 무작위 대조 4개**를 추가한다.
     이유: 무작위 절반(47열)은 D1(6열)·D3(13열)·D2(16열)보다 훨씬 큰 개입이라
     그것만으로는 "계열 때문인가 개수 때문인가"를 못 가른다. 개수를 맞춰야 가른다.

축 E — 손실. BCE(기준선) · focal(gamma 1,2) · label smoothing(0.02, 0.05).
  ⚠️ 딸려 오는 변인: 손실이 바뀌면 스텝당 기울기 크기가 달라져 1632스텝의 **유효 학습량**이
     함께 바뀐다. 이 실험은 그것을 분리하지 않는다 — "이 조합의 효과"로만 읽는다.

축 F — 잔차 타깃. config 가 지목한 AA·S1·S2·T70 의 ≤2023 OOF 는 **없다**(s0_gateF.json).
  그러나 조사가 `oof_new/stack_fold2024_seed*.npz` 의 **시즌 전진 연쇄 OOF** 를 찾았고,
  그것이 학습 행마다 out-of-time 이라 오프셋으로 쓸 수 있다 (s0_offF092.py 가 게이트 넷을 건다).
    F0_rows_only : 커버된 904,650행으로만 학습, 오프셋 없음  ← **행 제한 대조**
    F1_resid_off : 같은 행 + 학습 시 로짓 오프셋(잔차 학습)
  🔴 F0 가 없으면 "오프셋 효과"와 "학습 행 19% 감소"가 겹쳐서 못 읽는다.
"""
import json, sys
import numpy as np

PREP = "/home/<user>/SungjinPark/audit_deep/pkg/DEEP/mate/model/deep/prep.json"
S5 = [42, 777, 2024, 101, 7]
STEPS = 1632
HID = "256,128,64"
DROP, OPT = 0.0, "adamw0.01"
SIT = ["season", "game_month", "game_dayofweek", "inning", "balls_before",
       "strikes_before", "run_top_before", "run_total_before",
       "score_diff_pitcher_team", "home_win_expectancy", "li"]

P = json.load(open(PREP, encoding="utf-8"))
NUM = P["NUM"]
assert len(NUM) == 94 and len(P["CAT"]) == 5
IDX = {c: i for i, c in enumerate(NUM)}
ALL = np.arange(94)


def by_prefix(*prefs):
    return sorted(i for i, c in enumerate(NUM) if any(c.startswith(p) for p in prefs))


D1 = by_prefix("sp_")
D2 = by_prefix("ins_")
D3 = by_prefix("i3_anom_", "p6_")
keep4 = sorted(IDX[c] for c in SIT)
D4 = sorted(set(ALL.tolist()) - set(keep4))
assert len(D1) == 6 and len(D2) == 16, (len(D1), len(D2))


def rmask(k, seed):
    return sorted(np.random.default_rng(seed).choice(94, k, replace=False).tolist())


POINTS = {}                    # 이름 → meta
POINTS["D0_all99_bce"] = {"axis": "D", "mask": [], "loss": "bce",
                          "desc": "전체 99열 · BCE (= 기준선, exp_090 재사용)",
                          "BASELINE": True}
for nm, m, ds in (
        ("D1_no_sp", D1, "sp_* 6열 마스킹"),
        ("D2_no_ins", D2, "ins_* 16열 마스킹"),
        ("D3_no_i3anom_p6", D3, "i3_anom_* + p6_* 13열 마스킹"),
        ("D4_sit_cat_only", D4, "범주 5열 + 상황 11열만 남기고 83열 마스킹")):
    POINTS[nm] = {"axis": "D", "mask": m, "loss": "bce", "desc": ds,
                  "BASELINE": False}
for j, sd in enumerate((9201, 9202, 9203), 1):
    POINTS[f"D5h{j}_rand47"] = {"axis": "D5_null", "mask": rmask(47, sd),
                                "loss": "bce",
                                "desc": f"무작위 절반 47열 마스킹 (rng {sd})",
                                "BASELINE": False}
for k, sd, ref in ((len(D1), 9211, "D1"), (len(D2), 9212, "D2"),
                   (len(D3), 9213, "D3"), (len(D4), 9214, "D4")):
    POINTS[f"D5m{k:02d}_rand"] = {"axis": "D5_null", "mask": rmask(k, sd),
                                  "loss": "bce",
                                  "desc": f"무작위 {k}열 마스킹 — {ref} 개수 맞춤 대조 (rng {sd})",
                                  "BASELINE": False}
for nm, ls, ds in (("E_focal_g1", "focal1", "focal loss gamma=1 (alpha 없음)"),
                   ("E_focal_g2", "focal2", "focal loss gamma=2 (alpha 없음)"),
                   ("E_ls002", "ls0.02", "label smoothing eps=0.02"),
                   ("E_ls005", "ls0.05", "label smoothing eps=0.05")):
    POINTS[nm] = {"axis": "E", "mask": [], "loss": ls, "desc": ds,
                  "BASELINE": False}
FW = "/home/<user>/SungjinPark/exp092/work"
POINTS["F0_rows_only"] = {"axis": "F", "mask": [], "loss": "bce",
                          "rows": f"{FW}/rowsF_all.npy", "off": "",
                          "desc": "OOF 커버 904,650행만 학습 · 오프셋 없음 (행 제한 대조)",
                          "BASELINE": False}
POINTS["F1_resid_off"] = {"axis": "F", "mask": [], "loss": "bce",
                          "rows": f"{FW}/rowsF_all.npy", "off": f"{FW}/offF_all.npy",
                          "desc": "같은 행 + 학습 시 out-of-time 로짓 오프셋 (잔차 학습)",
                          "BASELINE": False}

JOBS = []
for nm, m in POINTS.items():
    if m["BASELINE"]:
        continue
    ms = ",".join(map(str, m["mask"]))
    for s in S5:
        JOBS.append((f"{nm}_s{s}", ms, m["loss"], m.get("rows", ""),
                     m.get("off", ""), s))

if __name__ == "__main__":
    with open(sys.argv[1], "w") as f:
        for tag, ms, ls, rw, of, seed in JOBS:
            # 🔴 구분자는 '|' 다. 탭은 bash IFS 공백문자라 **연속된 빈 필드가 하나로 뭉개져**
            #    ROWS·OFF 가 빈 줄에서 SEED 가 밀려 85 fit 이 전부 즉사했다(1차 발사).
            f.write(f"{tag}|{ms}|{ls}|{rw}|{of}|{seed}\n")
    meta = {"n_jobs": len(JOBS), "S5": S5, "steps": STEPS, "hid": HID,
            "drop": DROP, "opt": OPT,
            "NUM_names": NUM, "CAT_names": P["CAT"],
            "families": {"sp_": D1, "ins_": D2, "i3_anom_+p6_": D3,
                         "kept_situational": keep4},
            "F_axis": "착수 — s0_gateF.json(팔 OOF 부재) + s0_offF.json(전진연쇄 OOF 로 대체)",
            "points": {k: dict(v, n_masked=len(v["mask"])) for k, v in POINTS.items()}}
    json.dump(meta, open(sys.argv[2], "w"), ensure_ascii=False, indent=1)
    print(f"{len(JOBS)} jobs / {len(POINTS)} points")
    for k, v in POINTS.items():
        print(f"  {k:20s} n_mask {len(v['mask']):3d}  loss {v['loss']:8s} {v['desc']}")
