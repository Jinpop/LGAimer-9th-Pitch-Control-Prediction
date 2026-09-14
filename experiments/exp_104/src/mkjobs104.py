"""exp_104 축A — exp_091·092·093 이 **결함면**에서 기각한 지점을 고친 면에서 다시 굽기 위한 작업 목록.

🔴 변형의 정의는 새로 짓지 않는다. exp_091 `mkjobs091.py` · exp_092 `mkjobs092.py` 의
   생성기를 **글자 그대로 옮겨** 같은 열 목록·같은 폭·같은 손실이 나오게 한다
   (D5 무작위 마스크는 `np.random.default_rng(seed).choice` 라 재현된다 — 이 파일이
    exp_092 metrics 의 목록과 일치하는지 self-check 로 확인한다).

바뀐 것은 셋뿐이고 전부 화면 쪽이다:
  ① 입력 X — 결함면 `X_oot23.npy` → **고친 f64 입력** `X_f64_oot23.npy` (split 이름 f64o, S134)
  ② 스텝 — 1632 고정 → **{816, 1632, 2720} 세 지점** (S183·S94: 고정 스텝 비교는 용량 확장을
     그 자체로 −4점쯤 벌한다. 각 변형을 자기 argmax 스텝에서 비교해야 한다)
  ③ 이미 고친 면에서 읽힌 지점은 다시 굽지 않는다 — exp_102 축B 5지점 + 시드수 축 + 기준선.

🔴 사전 등록(발사 전 확정, S8): 혼합 이득 − 26.2698 ≥ +2.5 **그리고** 짝지은 투수 블록
   부트스트랩 t ≥ 2.0. ρ 로 판정하지 않는다(S83) — ρ·단독점수는 진단으로만 적는다.
"""
import json
import sys

import numpy as np

PREP = "/home/<user>/SungjinPark/audit_deep/pkg/DEEP/mate/model/deep/prep.json"
FW092 = "/home/<user>/SungjinPark/exp092/work"
S5 = [42, 777, 2024, 101, 7]
STEPS = [1632, 816, 2720]          # 🔴 1632 를 먼저 — 중간 판독을 위해
BASE_HID = (256, 128, 64)
SIT = ["season", "game_month", "game_dayofweek", "inning", "balls_before",
       "strikes_before", "run_top_before", "run_total_before",
       "score_diff_pitcher_team", "home_win_expectancy", "li"]

P = json.load(open(PREP, encoding="utf-8"))
NUM = P["NUM"]
assert len(NUM) == 94 and len(P["CAT"]) == 5
IDX = {c: i for i, c in enumerate(NUM)}
ALL = np.arange(94)


# ── exp_091 mkjobs091.py 의 생성기 (글자 그대로) ─────────────────────────
def widen(hid, f):
    return tuple(int(h * f) for h in hid)


def depth(hid, d):
    if d == 0:
        return hid
    if d == -1:
        return hid[:-1]
    return hid + (hid[-1] // 2,)


# ── exp_092 mkjobs092.py 의 생성기 (글자 그대로) ─────────────────────────
def by_prefix(*prefs):
    return sorted(i for i, c in enumerate(NUM) if any(c.startswith(p) for p in prefs))


def rmask(k, seed):
    return sorted(np.random.default_rng(seed).choice(94, k, replace=False).tolist())


D1 = by_prefix("sp_")
D2 = by_prefix("ins_")
D3 = by_prefix("i3_anom_", "p6_")
keep4 = sorted(IDX[c] for c in SIT)
D4 = sorted(set(ALL.tolist()) - set(keep4))
assert len(D1) == 6 and len(D2) == 16 and len(D3) == 13 and len(D4) == 83

# 🔴 exp_092 metrics.json 이 기록한 목록과 대조 (생성기가 같은지 기계 확인)
SELFCHECK = {"sp_": [72, 73, 74, 75, 76, 77],
             "ins_": [78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93],
             "i3_anom_+p6_": [29, 30, 31, 32, 33, 34, 42, 43, 44, 45, 46, 47, 48],
             "kept_situational": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
assert D1 == SELFCHECK["sp_"] and D2 == SELFCHECK["ins_"]
assert D3 == SELFCHECK["i3_anom_+p6_"] and keep4 == SELFCHECK["kept_situational"]

# ── 이미 고친 면에서 읽힌 것 — 다시 굽지 않는다 ─────────────────────────
DONE_ON_FIXED = {
    "A_n05_seeds5":  "exp_097/102 기준선 자체 (26.2698). z_F1632_f64o_s*",
    "B_w1_d0":       "= 기준선",
    "C_dr0_wd001":   "= 기준선",
    "D0_all99_bce":  "= 기준선",
    "A_n10_seeds10": "exp_102 N0A_1632x10seeds",
    "A_n15_seeds15": "exp_102 N0A_1632x15seeds",
    "A_n20_seeds20": "exp_102 N0A_1632x20seeds",
    "B_w1_dm1":      "exp_102 축B Bw1dm1",
    "B_w2_dm1":      "exp_102 축B Bw2dm1",
    "C_dr02_wd001":  "exp_102 축B Bdr02 (drop 0.2 · adamw0.01)",
    "D5m16_rand":    "exp_102 축B Bm16",
    "E_ls005":       "exp_102 축B Bls005",
}

POINTS = {}     # name -> dict(runner, hid, drop, opt, mask, loss, rows, off, target)


def add(name, axis, desc, **kw):
    assert name not in POINTS
    m = {"axis": axis, "desc": desc, "runner": "train104",
         "hid": list(BASE_HID), "drop": 0.0, "opt": "adamw0.01",
         "mask": [], "loss": "bce", "rows": "", "off": "", "target": ""}
    m.update(kw)
    POINTS[name] = m


# 축 B — 폭·깊이 (exp_091)
for wf, wn in ((0.5, "w05"), (1.0, "w1"), (2.0, "w2")):
    for dd, dn in ((-1, "dm1"), (0, "d0"), (1, "dp1")):
        nm = f"B_{wn}_{dn}"
        if nm in DONE_ON_FIXED:
            continue
        add(nm, "B_widthdepth", f"폭 x{wf} · 깊이 {dd:+d}",
            hid=list(depth(widen(BASE_HID, wf), dd)))

# 축 C — dropout x weight decay (exp_091)
for dr, drn in ((0.0, "dr0"), (0.1, "dr01"), (0.2, "dr02")):
    for wd, wdn in ((0.0, "wd0"), (0.01, "wd001"), (0.1, "wd01")):
        nm = f"C_{drn}_{wdn}"
        if nm in DONE_ON_FIXED:
            continue
        add(nm, "C_regularization", f"dropout {dr} · weight decay {wd}",
            drop=dr, opt=f"adamw{wd}")

# 축 D — 이름 붙인 계열 마스킹 (exp_092)
for nm, m, ds in (("D1_no_sp", D1, "sp_* 6열 마스킹"),
                  ("D2_no_ins", D2, "ins_* 16열 마스킹"),
                  ("D3_no_i3anom_p6", D3, "i3_anom_* + p6_* 13열 마스킹"),
                  ("D4_sit_cat_only", D4, "범주 5열 + 상황 11열만 남기고 83열 마스킹")):
    add(nm, "D_featuremask", ds, mask=m)

# 축 D5 — 개수 맞춤 무작위 귀무 (exp_092). S104 가 요구한 대조다.
for j, sd in enumerate((9201, 9202, 9203), 1):
    add(f"D5h{j}_rand47", "D5_null", f"무작위 절반 47열 마스킹 (rng {sd})",
        mask=rmask(47, sd))
for k, sd, ref in ((len(D1), 9211, "D1"), (len(D2), 9212, "D2"),
                   (len(D3), 9213, "D3"), (len(D4), 9214, "D4")):
    nm = f"D5m{k:02d}_rand"
    if nm in DONE_ON_FIXED:
        continue
    add(nm, "D5_null", f"무작위 {k}열 마스킹 — {ref} 개수 맞춤 대조 (rng {sd})",
        mask=rmask(k, sd))

# 축 E — 손실 (exp_092)
for nm, ls, ds in (("E_focal_g1", "focal1", "focal loss gamma=1 (alpha 없음)"),
                   ("E_focal_g2", "focal2", "focal loss gamma=2 (alpha 없음)"),
                   ("E_ls002", "ls0.02", "label smoothing eps=0.02")):
    add(nm, "E_loss", ds, loss=ls)

# 축 F — 잔차 학습 (exp_092). 🔑 F0 가 config 의 1순위다.
add("F0_rows_only", "F_residual",
    "OOF 커버 904,650행만 학습 · 오프셋 없음 (행 제한 대조)",
    rows=f"{FW092}/rowsF_all.npy")
add("F1_resid_off", "F_residual",
    "같은 행 + 학습 시 out-of-time 로짓 오프셋 (잔차 학습). "
    "평가에서 z 와 z+off 두 읽기를 낸다",
    rows=f"{FW092}/rowsF_all.npy", off=f"{FW092}/offF_all.npy")

# 축 G — 4셀 멀티헤드 (exp_093). 🔴 exp_093 은 ρ 하나로 기각했다 — S83 이 금지하는 판정이다.
add("G_multihead5", "G_multihead", "n_out 5 · 실패 4셀 소프트 타깃 교차엔트로피",
    runner="train104m", target="cells5")
add("G_collapsed5", "G_multihead",
    "n_out 5 인데 실패 4셀 균등 0.25 — 유형 정보만 없앤 귀무 대조(S104·S116)",
    runner="train104m", target="collapsed5")

JOBS = []
for nm, m in POINTS.items():
    for st in STEPS:
        for s in S5:
            JOBS.append((f"{nm}__k{st}_s{s}", nm, st, s, m))

if __name__ == "__main__":
    with open(sys.argv[1], "w") as f:
        for tag, nm, st, s, m in JOBS:
            f.write("|".join([tag, m["runner"], ",".join(map(str, m["hid"])),
                              str(m["drop"]), m["opt"],
                              ",".join(map(str, m["mask"])), m["loss"],
                              m["rows"], m["off"], m["target"],
                              str(st), str(s)]) + "\n")
    meta = {"exp_id": "exp_104", "stage": "mkjobs", "n_jobs": len(JOBS),
            "n_points": len(POINTS), "S5": S5, "steps": STEPS,
            "split": "f64o (= exp096/work/X_f64_oot23.npy, 고친 f64 입력)",
            "done_on_fixed_surface_not_refit": DONE_ON_FIXED,
            "generators": "exp_091 mkjobs091.py + exp_092 mkjobs092.py 생성기 그대로",
            "selfcheck_family_indices_match_exp092_metrics": True,
            "points": {k: dict(v, n_masked=len(v["mask"])) for k, v in POINTS.items()}}
    json.dump(meta, open(sys.argv[2], "w"), ensure_ascii=False, indent=1)
    print(f"{len(JOBS)} jobs / {len(POINTS)} points / steps {STEPS}")
    for k, v in POINTS.items():
        print(f"  {k:20s} {v['runner']:10s} hid {str(v['hid']):20s} "
              f"drop {v['drop']} {v['opt']:9s} nmask {len(v['mask']):3d} "
              f"loss {v['loss']:8s} tgt {v['target'] or '-'}")
