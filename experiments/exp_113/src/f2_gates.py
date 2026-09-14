"""exp_113 — 🔴 배포 게이트 둘. 통과 못 하면 실험 중단.

A. 행 독립성 (규칙 2-4)
   G_replace  KEEP 행은 원문·원위치 그대로, 나머지 전 행을 이 프레임에 없던 값으로 전량 교체
              → KEEP 행 피처가 **정확히 0.0** 움직여야 한다.
   G_degen    나머지 전 행을 극단 두 행의 반복으로 채운다 (S28 계열).
   G_position KEEP 행의 인덱스만 바꾼다 (프레임 통째로 뒤집기).
   G_single   한 행만 담은 프레임으로 돌려도 같은가 (배포 극단).
   🔴 KEEP 행은 **바꾼 계산을 실제로 타는 행**이어야 한다 (S61) — 동결표 적중·미적중을
      둘 다 넣고 각 열의 발동 행 수가 0 이 아님을 확인한다.

B. 시간축 (S171 의 변형 — S197)
   이 파이프라인은 라벨을 아예 안 쓰므로 라벨 난수화의 양성 대조가 원리적으로 없다.
   그래서 **시즌 대역 난수화**로 대체한다:
     T_label  `control_success` 를 전량 난수화 → 피처 비트 동일 (라벨 경로가 0 임을 증명)
     T_neg    season >= S 행을 통째로 난수화 → 동결표가 **비트 동일** (표가 미래를 안 본다)
     T_pos    season <  S 행을 난수화       → 2024 피처가 **움직여야 한다** (양성 대조)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from f1_feats import (BASE, COLS, ROW_COLS, TRAIN, apply_table, build_table)  # noqa

S = 2024
rep = {"exp_id": "exp_113", "stage": "f2_gates", "target_season": S}
df = pd.read_csv(TRAIN, usecols=["season", "control_success"] + ROW_COLS,
                 encoding="utf-8-sig")
season = df["season"].to_numpy(np.int32)
src = df[season < S]
tgt = df.loc[season == S, ROW_COLS].reset_index(drop=True)
T = build_table(src)
F0 = apply_table(tgt, T)
n = len(tgt)
rs = np.random.RandomState(113)

miss_i = COLS.index("insp_miss")
hit = np.where(F0[:, miss_i] == 0)[0]
mis = np.where(F0[:, miss_i] == 1)[0]
KEEP = np.unique(np.concatenate([[0, 1, n - 2, n - 1],
                                 rs.choice(n, 200, replace=False),
                                 hit[:40], mis[:40] if len(mis) else []]).astype(int))
rep["n_target_rows"] = int(n)
rep["n_keep"] = int(len(KEEP))
rep["keep_table_hit"] = int((F0[KEEP, miss_i] == 0).sum())
rep["keep_table_miss"] = int((F0[KEEP, miss_i] == 1).sum())
rep["activation_rows_per_col"] = {c: int(np.isfinite(F0[:, i]).sum())
                                  for i, c in enumerate(COLS)}
rep["keep_activation_per_col"] = {c: int(np.isfinite(F0[KEEP, i]).sum())
                                  for i, c in enumerate(COLS)}
rep["anchor_row_same_for_n_and_pm"] = bool(T["_anchor_row_same_for_n_and_pm"])


def diff(a, b):
    bn = np.isnan(a) & np.isnan(b)
    return (np.where(bn, 0.0, np.abs(np.nan_to_num(a) - np.nan_to_num(b))),
            bool((np.isnan(a) == np.isnan(b)).all()))


def corrupt(base, mode):
    d = base.copy()
    other = np.setdiff1d(np.arange(n), KEEP)
    if mode == "replace":
        pool = df.loc[season == 2019, ROW_COLS].reset_index(drop=True)
        take = pool.iloc[rs.randint(0, len(pool), len(other))].reset_index(drop=True)
        for c in ROW_COLS:
            d.loc[other, c] = take[c].to_numpy()
    else:                                                    # degen
        i_hi = int(np.argmax(base["asof_pitcher_n"].to_numpy()))
        i_lo = int(np.argmin(base["asof_pitcher_n"].to_numpy()))
        for c in ROW_COLS:
            v = np.array([base[c].iloc[i_hi], base[c].iloc[i_lo]])
            d.loc[other, c] = np.resize(v, len(other))
    return d


for mode in ("replace", "degen"):
    Fc = apply_table(corrupt(tgt, mode), T)
    dd, nanok = diff(F0[KEEP], Fc[KEEP])
    rep[f"G_{mode}"] = {"max_abs_diff": float(dd.max()),
                        "n_keep_rows_moved": int((dd > 0).any(1).sum()),
                        "nan_pattern_identical": nanok,
                        "PASS": bool(dd.max() == 0.0 and nanok)}

Fr = apply_table(tgt.iloc[::-1].reset_index(drop=True), T)[::-1]
dd, nanok = diff(F0, Fr)
rep["G_position"] = {"max_abs_diff": float(dd.max()),
                     "n_rows_moved": int((dd > 0).any(1).sum()),
                     "nan_pattern_identical": nanok,
                     "PASS": bool(dd.max() == 0.0 and nanok)}

d4 = 0.0
single = [int(x) for x in KEEP[:40]]
for i in single:
    f1 = apply_table(tgt.iloc[[i]].reset_index(drop=True), T)[0]
    dd, _ = diff(F0[i], f1)
    d4 = max(d4, float(dd.max()))
rep["G_single_row"] = {"n_tested": len(single), "max_abs_diff": d4,
                       "PASS": bool(d4 == 0.0)}


# ── B. 시간축 ────────────────────────────────────────────────────
def tbl_sig(t):
    return {k: float(np.abs(np.asarray(v, np.float64)).sum())
            for k, v in t.items() if isinstance(v, np.ndarray) and v.dtype.kind == "f"}


def tbl_identical(a, b):
    ka = [k for k in a if isinstance(a[k], np.ndarray)]
    if sorted(ka) != sorted([k for k in b if isinstance(b[k], np.ndarray)]):
        return False
    for k in ka:
        if a[k].shape != b[k].shape or not np.array_equal(a[k], b[k]):
            return False
    return True


NUMC = [c for c in ROW_COLS if c != "pitcher_id"]
dfl = df.copy()
dfl["control_success"] = rs.randint(0, 2, len(dfl))
Tl = build_table(dfl[season < S])
Fl = apply_table(dfl.loc[season == S, ROW_COLS].reset_index(drop=True), Tl)
dd, nanok = diff(F0, Fl)
rep["T_label_randomize_all"] = {
    "무엇": "control_success 를 전량 난수화",
    "table_identical": tbl_identical(T, Tl),
    "feature_max_abs_diff": float(dd.max()), "nan_pattern_identical": nanok,
    "PASS": bool(tbl_identical(T, Tl) and dd.max() == 0.0),
    "해석": "라벨 경로가 0 이다 — 이 열들은 asof_* 와 동결표만의 함수"}

dfn = df.copy()
m_future = season >= S
for c in NUMC:
    v = dfn[c].to_numpy(np.float64).copy()
    v[m_future] = rs.permutation(v[m_future]) * (1.0 + rs.rand(int(m_future.sum())))
    dfn[c] = v
dfn.loc[m_future, "pitcher_id"] = rs.randint(10 ** 7, 10 ** 8, int(m_future.sum()))
Tn = build_table(dfn[season < S])
rep["T_neg_randomize_future"] = {
    "무엇": f"season >= {S} 행을 통째로 난수화 → 동결표 재구축",
    "n_rows_randomized": int(m_future.sum()),
    "table_bitwise_identical": tbl_identical(T, Tn),
    "PASS": bool(tbl_identical(T, Tn))}

dfp = df.copy()
m_past = season < S
for c in NUMC:
    v = dfp[c].to_numpy(np.float64).copy()
    v[m_past] = rs.permutation(v[m_past])
    dfp[c] = v
Tp = build_table(dfp[season < S])
Fp = apply_table(tgt, Tp)
dd, _ = diff(F0, Fp)
rep["T_pos_randomize_past"] = {
    "무엇": f"season < {S} 행을 난수화 → 동결표가 달라져야 한다",
    "table_bitwise_identical": tbl_identical(T, Tp),
    "feature_max_abs_diff": float(dd.max()),
    "share_rows_moved": float((dd > 0).any(1).mean()),
    "PASS": bool((not tbl_identical(T, Tp)) and (dd > 0).any(1).mean() > 0.5)}

keys = ["G_replace", "G_degen", "G_position", "G_single_row",
        "T_label_randomize_all", "T_neg_randomize_future", "T_pos_randomize_past"]
rep["ALL_PASS"] = bool(all(rep[k]["PASS"] for k in keys))
json.dump(rep, open(f"{BASE}/out/f2_gates.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(rep, ensure_ascii=False, indent=1))
sys.exit(0 if rep["ALL_PASS"] else 3)
