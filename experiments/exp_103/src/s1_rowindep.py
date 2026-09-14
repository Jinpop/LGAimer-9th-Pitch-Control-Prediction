"""exp_103 — 🔴 행 독립성 게이트 (규칙 2-4). 통과 못 하면 실험 중단.

세 가지를 기계로 판정한다:
  G1 내용 분리 : KEEP 행은 원문·원위치 그대로 두고 **나머지 전 행을 다른 값으로 전량 교체**.
                 KEEP 행의 피처가 **정확히 0.0** 만큼 움직여야 한다.
  G2 축퇴 프레임: 나머지 전 행을 **극단 두 행의 반복**으로 채운다(S28 계열).
  G3 위치 분리 : KEEP 행의 인덱스만 바꿔 (프레임을 뒤집어) 같은 값이 나오는가.
🔴 KEEP 행은 **바꾼 계산을 실제로 타는 행**이어야 한다(S61) — 동결표 적중 행과 미적중 행을
   둘 다 넣고, 각 열의 발동 행 수를 세서 0 이 아님을 확인한다.
"""
import json, os, sys
import numpy as np
import pandas as pd

H = os.environ.get("EXP103_HOME", "/home/<user>/SungjinPark")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_feats2 import build_table, apply_table, COLS, ROW_COLS, TRAIN, BASE  # noqa

df = pd.read_csv(TRAIN, usecols=["season", "control_success"] + ROW_COLS,
                 encoding="utf-8-sig")
season = df["season"].to_numpy(np.int32)
src = df[season < 2023 + 1 - 1]                     # season <= 2023
src = df[season <= 2023]
tgt = df[season == 2024][ROW_COLS].reset_index(drop=True)
T = build_table(src)
assert (src["season"].to_numpy() <= 2023).all()

F0 = apply_table(tgt, T)
n = len(tgt)
rs = np.random.RandomState(103)
KEEP = np.unique(np.concatenate([[0, 1, n - 2, n - 1],
                                 rs.choice(n, 200, replace=False)]))
# KEEP 이 동결표 적중/미적중을 모두 포함하게 보강 (S61)
miss_i = COLS.index("bat_form_miss")
hit = np.where(F0[:, miss_i] == 0)[0]
mis = np.where(F0[:, miss_i] == 1)[0]
KEEP = np.unique(np.concatenate([KEEP, hit[:30], mis[:30]]))
rep = {"n_target_rows": int(n), "n_keep": int(len(KEEP)),
       "keep_hit": int((F0[KEEP, miss_i] == 0).sum()),
       "keep_miss": int((F0[KEEP, miss_i] == 1).sum())}
rep["activation_rows_per_col"] = {c: int(np.isfinite(F0[:, i]).sum())
                                  for i, c in enumerate(COLS)}
rep["keep_activation_per_col"] = {c: int(np.isfinite(F0[KEEP, i]).sum())
                                  for i, c in enumerate(COLS)}


def corrupt(base, mode):
    d = base.copy()
    other = np.setdiff1d(np.arange(n), KEEP)
    if mode == "replace":
        # 다른 시즌(2019) 행으로 전량 교체 — 이 프레임 어디에도 없던 값
        pool = df[season == 2019][ROW_COLS].reset_index(drop=True)
        take = pool.iloc[rs.randint(0, len(pool), len(other))].reset_index(drop=True)
        for c in ROW_COLS:
            d.loc[other, c] = take[c].to_numpy()
    elif mode == "degen":
        # 극단 두 행의 반복
        ex = pd.DataFrame({c: [base[c].iloc[int(np.argmax(base["asof_batter_n"]))],
                               base[c].iloc[int(np.argmin(base["asof_batter_n"]))]]
                           for c in ROW_COLS})
        for c in ROW_COLS:
            d.loc[other, c] = np.resize(ex[c].to_numpy(), len(other))
    return d


for mode in ("replace", "degen"):
    Fc = apply_table(corrupt(tgt, mode), T)
    a, b = F0[KEEP], Fc[KEEP]
    both_nan = np.isnan(a) & np.isnan(b)
    diff = np.where(both_nan, 0.0, np.abs(np.nan_to_num(a) - np.nan_to_num(b)))
    rep[f"G_{mode}"] = {"max_abs_diff": float(diff.max()),
                        "n_rows_moved": int((diff > 0).any(1).sum()),
                        "nan_pattern_identical": bool(
                            (np.isnan(a) == np.isnan(b)).all()),
                        "PASS": bool(diff.max() == 0.0
                                     and (np.isnan(a) == np.isnan(b)).all())}

# G3 — 위치만 바꾼다 (프레임을 통째로 뒤집는다)
rev = tgt.iloc[::-1].reset_index(drop=True)
Fr = apply_table(rev, T)[::-1]
both_nan = np.isnan(F0) & np.isnan(Fr)
d3 = np.where(both_nan, 0.0, np.abs(np.nan_to_num(F0) - np.nan_to_num(Fr)))
rep["G_position"] = {"max_abs_diff": float(d3.max()),
                     "n_rows_moved": int((d3 > 0).any(1).sum()),
                     "PASS": bool(d3.max() == 0.0)}

# G4 — 단행 추론: 한 행만 담은 프레임으로 돌려도 같은가 (배포 극단 검사)
single = [int(x) for x in KEEP[:40]]
d4 = 0.0
for i in single:
    f1 = apply_table(tgt.iloc[[i]].reset_index(drop=True), T)[0]
    a, b = F0[i], f1
    bn = np.isnan(a) & np.isnan(b)
    d4 = max(d4, float(np.where(bn, 0.0, np.abs(np.nan_to_num(a) - np.nan_to_num(b))).max()))
rep["G_single_row"] = {"n_tested": len(single), "max_abs_diff": d4,
                       "PASS": bool(d4 == 0.0)}

rep["ALL_PASS"] = bool(all(rep[k]["PASS"] for k in
                           ("G_replace", "G_degen", "G_position", "G_single_row")))
json.dump(rep, open(f"{BASE}/out/s1_rowindep.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(rep, ensure_ascii=False, indent=1))
sys.exit(0 if rep["ALL_PASS"] else 3)
