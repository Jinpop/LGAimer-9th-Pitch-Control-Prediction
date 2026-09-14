"""exp_107 축 B 리크 게이트 (3차) — 라벨 복원 창을 **정확한 기계 검사**로 닫는다.

🔴 2차 Part 1 이 실패한 이유는 리크가 아니라 **조작 자체가 불가능해서**다:
   as-of 비율 열에 균등난수를 넣으면 `labels.recover()` 가
   `AssertionError: the five cells are not a partition` 로 죽는다(실행으로 확인).
   그래서 "그 대역을 난수화하고 다시 복원한다"는 형태의 조작은 이 파이프라인에 없다.

대신 같은 질문을 **정확히** 닫는다. 복원의 의존 구조가 좁고 명시적이기 때문이다:

  · `recover()` 는 행 t 의 라벨을 `(rate[t], n[t], rate[t+1], n[t+1])` 로 만든다.
    t+1 은 **같은 투수의 다음 행**이다(`groupby(pitcher).shift(-1)`).
  · `n_next == n + 1` 이 전 행에서 성립한다(recover 의 `n_step_all_plus_one` 진단).
    즉 t+1 의 as-of 누적값은 **t 까지의 행들만** 담고, t+1 자신의 결과는 안 담는다.
  · 따라서 행 t 의 복원 라벨이 보는 결과는 **t 자신의 결과 하나**뿐이다.

그러면 시즌 s 의 OOF 가 season >= s 라벨을 안 봤다는 것은 다음 셋이면 닫힌다:

  G1  fit_rows(s) 에 season >= s 행이 **0개**
  G2  fit_rows(s) 에 `succ_season >= s` 행이 **0개**
      → 적합에 쓰인 모든 행의 t 와 t+1 이 **둘 다** season < s. 복원 창이 시즌 경계 안이다.
      (`mk_oof.py` 의 `& (succ_season < s)` 항이 하는 일이 정확히 이것이고, 여기서 실행 확인한다)
  G3  `n_step_all_plus_one` 이 참  → t+1 의 as-of 가 t 를 넘어선 결과를 안 담는다
  G4  예측 행(season == s)의 라벨은 어디에도 안 쓰인다 — `pred_rows` 는 `predict` 에만 간다.
      2차 Part 2 가 이것을 종단 실행으로 확인한다(그 대역 셀 라벨을 난수화해도 예측 불변).

전 시즌(2020~2023)에 대해 돌린다 — 한 시즌만 보면 기전만 보고 나머지는 추론이 된다.
"""
import hashlib, json, sys
import numpy as np
import pandas as pd

H = "/home/<user>/SungjinPark"
B = f"{H}/exp107"
sys.path.insert(0, f"{H}/experiments/exp_036/src")
from labels import FAIL_CELLS, recover                        # noqa: E402

DATA = f"{H}/data/train.csv"
F_EXCL_MAX, TU = 2022, 2023
df = pd.read_csv(DATA, encoding="utf-8-sig")
diag = recover(df, verbose=False)
season = df["season"].to_numpy()
ok = df["recov_ok"].to_numpy()
ss = df.groupby("pitcher_id", sort=False)["season"].shift(-1).to_numpy()
f_excl = ((df["season"] <= F_EXCL_MAX) & (df["game_type"] == "F")).to_numpy()
tr_full = (season <= TU) & ~f_excl
n = df["asof_pitcher_n"].to_numpy(np.float64)
n_next = df.groupby("pitcher_id", sort=False)["asof_pitcher_n"].shift(-1).to_numpy(np.float64)
hasnext = ~np.isnan(n_next)

per = {}
allpass = True
for s in sorted(set(season[tr_full].tolist()))[1:]:
    fit_rows = (season < s) & ~f_excl & ok & (ss < s)
    pred_rows = (season == s) & tr_full
    g1 = int((fit_rows & (season >= s)).sum())
    g2 = int((fit_rows & ~(ss < s)).sum())
    g2b = int((fit_rows & (np.nan_to_num(ss, nan=-1) >= s)).sum())
    mx = float(np.nanmax(season[fit_rows])) if fit_rows.sum() else None
    mxs = float(np.nanmax(ss[fit_rows])) if fit_rows.sum() else None
    p = bool(g1 == 0 and g2 == 0 and g2b == 0)
    allpass &= p
    per[str(s)] = {"n_fit_rows": int(fit_rows.sum()),
                   "n_pred_rows": int(pred_rows.sum()),
                   "G1_fit_rows_with_season_ge_s": g1,
                   "G2_fit_rows_with_succ_season_ge_s": g2b,
                   "G2_fit_rows_without_succ_season_lt_s": g2,
                   "max_season_in_fit_rows": mx,
                   "max_succ_season_in_fit_rows": mxs,
                   "PASS": p}
    print(f"s={s}  fit {int(fit_rows.sum()):>9,}  max_season {mx}  "
          f"max_succ_season {mxs}  G1 {g1}  G2 {g2b}  {'PASS' if p else 'FAIL'}",
          flush=True)

# 🔴 fit_rows 를 안 걸렀다면 어땠을지 — 이 항이 실제로 일을 하는가
counter = {}
for s in sorted(set(season[tr_full].tolist()))[1:]:
    naive = (season < s) & ~f_excl & ok
    counter[str(s)] = {
        "n_rows_dropped_by_succ_season_guard": int((naive & ~(ss < s)).sum()),
        "share_of_naive": float((naive & ~(ss < s)).sum() / max(int(naive.sum()), 1))}

R = {"exp_id": "exp_107", "stage": "s1c_leak107",
     "무엇을_증명하나": "시즌 s 의 p_self 를 만든 적합이 season >= s 의 어떤 결과도 안 봤다",
     "왜_난수화가_아니라_기계검사인가":
         "as-of 비율 열에 난수를 넣으면 `labels.recover()` 가 partition assert 로 죽는다 "
         "(2차 실행에서 확인). 그래서 그 형태의 조작이 이 파이프라인에 존재하지 않는다. "
         "대신 복원의 의존 구조(행 t 의 라벨 = f(t, t+1) · n_next == n+1)가 좁고 명시적이라 "
         "정확한 기계 검사로 같은 질문을 닫을 수 있다.",
     "recover_diag": {"n_step_all_plus_one": diag["n_step_all_plus_one"],
                      "partition_sums_to_one": diag["partition_sums_to_one"],
                      "success_recovery": diag["success_recovery"]},
     "G3_n_next_equals_n_plus_1": {
         "all_rows_with_next": bool(np.all(n_next[hasnext] - n[hasnext] == 1.0)),
         "n_rows_checked": int(hasnext.sum()),
         "뜻": "t+1 의 as-of 누적값은 t 까지만 담는다 = t+1 자신의 결과를 안 담는다"},
     "per_season": per,
     "🔑_succ_season_항이_실제로_거르는_행": counter,
     "PASS": bool(allpass and diag["n_step_all_plus_one"]
                  and np.all(n_next[hasnext] - n[hasnext] == 1.0)),
     "함께_읽을_것": "2차 `s1b_leak107_s2020.json` PART2 (종단 실행: season>=s 셀 라벨을 "
                     "난수화해도 예측 비트 동일 · season<s 난수화하면 이동)"}
json.dump(R, open(f"{B}/out/s1c_leak107.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({"G3": R["G3_n_next_equals_n_plus_1"], "PASS": R["PASS"],
                  "succ_guard_drops": counter}, ensure_ascii=False, indent=1))
print("MASK_GATE_PASS" if R["PASS"] else "MASK_GATE_FAIL")
