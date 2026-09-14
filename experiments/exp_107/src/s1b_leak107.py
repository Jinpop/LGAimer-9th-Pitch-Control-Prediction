"""exp_107 축 B 리크 게이트 (2차) — **라벨이 실제로 오는 통로**를 친다.

🔴 1차(`s1_leak107.py`)가 양성 대조에서 실패했고 그 실패가 사실 하나를 가르쳤다:
   **OOF 의 라벨은 `control_success` 가 아니다.** `labels.recover()` 는 라벨을
   `asof_pitcher_{success,reverse,middle,ball,strike}_rate` + `asof_pitcher_n` 에서
   **같은 투수의 다음 행**을 써서 복원하고, `control_success` 는 그 복원을 **검산하는 데만**
   쓴다(`labels.py:51`). 그래서 `control_success` 를 시즌 대역째 난수화해도 4개 실패셀이
   한 비트도 안 움직였다(양성 대조가 0.0). 1차 실행의 값어치는 그 확인이다.

이제 두 부분으로 친다.

  Part 1 — 라벨 복원 통로 (CatBoost 없음, 수 초):
    `asof_*_rate` 5열을 **season >= S 에서 전량 난수화** → `recover()` 재실행 →
    `fit_rows`(= (season<S) & ~f_excl & ok & (succ_season<S)) 와 그 위의 4셀 라벨이
    **비트 동일**해야 한다.
    🔑 이게 비어 있지 않은 이유: `ok` 는 **다음 행**의 as-of 값에 의존하므로, 다음 행이
       season >= S 인 season<S 행은 원리적으로 오염될 수 있다. `succ_season < S` 가
       그 행들을 빼는지를 이 게이트가 **실행으로** 확인한다.
    양성 대조: `season < S` 에서 난수화 → fit 라벨이 **움직여야** 한다.

  Part 2 — 종단 (CatBoost 3판):
    4개 실패셀 값을 대역째 난수화하고 `mk_oof.py` 의 시즌 S 고리를 그대로 돌린다.
    N(season>=S 난수화) → 예측 비트 동일 · P(season<S 난수화) → 예측 이동.
    결정성 대조는 1차에서 이미 통과했다(A == D, sha 1b67d93f88a252cb).
"""
import gc, hashlib, importlib.util, json, os, sys, time
import numpy as np
import pandas as pd

H = "/home/<user>/SungjinPark"
B = f"{H}/exp107"
sys.path.insert(0, f"{H}/experiments/exp_036/src")
import exp016_ref as ref                                     # noqa: E402
from labels import FAIL_CELLS, RATE_COLS, recover            # noqa: E402

DATA = f"{H}/data/train.csv"
F_EXCL_MAX, TU = 2022, 2023
CAT = ["top_bottom", "game_type", "base_state"]
NT = 2
S = int(os.environ.get("SEASON107", "2020"))
SEED = 42
REFNPZ = f"{H}/build_AH/localref/oof/stack_fold2024_seed42.npz"
sha = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]

spec = importlib.util.spec_from_file_location(
    "e27", f"{H}/experiments/exp_027/src/train.py")
e27 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e27)
bfn = e27.build_frame_native

base = pd.read_csv(DATA, encoding="utf-8-sig")
base_feats = [c for c in base.columns if c not in ("row_id", "control_success")]
cat_native = [c for c in base_feats if c in CAT]
season = base["season"].to_numpy()
HI, LO = season >= S, season < S
z0 = np.load(REFNPZ, allow_pickle=False)
cell_round = {c: int(v) for c, v in zip(FAIL_CELLS, z0["cell_round"].tolist())}
RATE = list(RATE_COLS.values())
rng = np.random.default_rng(20260826)
RND = {b: {c: rng.random(int(m.sum())) for c in RATE}
       for b, m in (("HI", HI), ("LO", LO))}


def masks_and_labels(band):
    """band None | 'HI' | 'LO' — 그 대역의 as-of 비율 5열을 난수화하고 recover 를 다시 돌린다."""
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    if band:
        m = HI if band == "HI" else LO
        for c in RATE:
            v = df[c].to_numpy(np.float64).copy()
            v[m] = RND[band][c]
            df[c] = v
    try:
        diag = recover(df, verbose=False)
    except AssertionError as e:
        return None, {"recover_assert": repr(e)[:200]}
    ok = df["recov_ok"].to_numpy()
    ss = df.groupby("pitcher_id", sort=False)["season"].shift(-1).to_numpy()
    f_excl = ((df["season"] <= F_EXCL_MAX) & (df["game_type"] == "F")).to_numpy()
    fit_rows = (season < S) & ~f_excl & ok & (ss < S)
    lab = {c: df.loc[fit_rows, c].to_numpy(np.float64) for c in FAIL_CELLS}
    return (df, fit_rows, lab), {
        "agree_pct": diag["success_recovery"]["agree_pct"],
        "n_fit_rows": int(fit_rows.sum()), "fit_rows_sha16": sha(fit_rows),
        "fit_labels_sha16": {c: sha(lab[c]) for c in FAIL_CELLS},
        "n_recov_ok": int(ok.sum())}


t0 = time.time()
P1 = {}
ORIG, P1["A"] = masks_and_labels(None)
_, P1["N_rand_asof_season_ge_S"] = masks_and_labels("HI")
_, P1["P_rand_asof_season_lt_S"] = masks_and_labels("LO")
p1_neg = bool(P1["N_rand_asof_season_ge_S"].get("fit_rows_sha16") == P1["A"]["fit_rows_sha16"]
              and P1["N_rand_asof_season_ge_S"].get("fit_labels_sha16")
              == P1["A"]["fit_labels_sha16"])
p1_pos = bool(P1["P_rand_asof_season_lt_S"].get("fit_labels_sha16")
              != P1["A"]["fit_labels_sha16"])
print("PART1 neg(동일)", p1_neg, " pos(이동)", p1_pos, f"({time.time()-t0:.0f}s)", flush=True)

# ── Part 2 — 4셀 라벨을 대역째 난수화하고 종단 실행 ─────────────────────────
df0, fit_rows, lab0 = ORIG
tr_full = (season <= TU) & ~(((base["season"] <= F_EXCL_MAX)
                              & (base["game_type"] == "F")).to_numpy())
pred_rows = (season == S) & tr_full
Ff = bfn(df0.loc[fit_rows], base_feats, cat_native)
Fp = bfn(df0.loc[pred_rows], base_feats, cat_native)
ci = [base_feats.index(c) for c in cat_native]
CELLM = df0[FAIL_CELLS].to_numpy(np.float64)
rng2 = np.random.default_rng(20260827)


def labels_for(band):
    if band is None:
        return {c: lab0[c] for c in FAIL_CELLS}
    m = HI if band == "HI" else LO
    M = CELLM.copy()
    r = rng2.integers(0, 2, size=(int(m.sum()), len(FAIL_CELLS))).astype(np.float64)
    M[m] = r
    return {c: M[fit_rows, j] for j, c in enumerate(FAIL_CELLS)}


def run(band):
    import catboost as cb
    L = labels_for(band)
    tot = 0.0
    for c in FAIL_CELLS:
        m = cb.CatBoostClassifier(**dict(e27.CB_BASE, random_seed=SEED,
                                         iterations=cell_round[c], thread_count=NT))
        m.fit(cb.Pool(Ff, label=L[c], cat_features=ci))
        tot = tot + ref.sigmoid(np.asarray(
            m.predict(cb.Pool(Fp, cat_features=ci),
                      prediction_type="RawFormulaVal", thread_count=NT), np.float64))
        del m; gc.collect()
    return tot, {c: sha(L[c]) for c in FAIL_CELLS}


PS, LS = {}, {}
for band, tag in ((None, "A"), ("HI", "N"), ("LO", "P")):
    PS[tag], LS[tag] = run(band)
    print(f"[{tag}] pred sha16 {sha(PS[tag])} ({time.time()-t0:.0f}s)", flush=True)

p2_neg = bool(np.array_equal(PS["A"], PS["N"]))
p2_pos = bool(not np.array_equal(PS["A"], PS["P"]))
R = {"exp_id": "exp_107", "stage": "s1b_leak107", "season_tested": S,
     "🔴1차_실패가_가르친_것":
         "OOF 라벨은 `control_success` 가 아니라 `labels.recover()` 가 asof_*_rate 에서 "
         "복원한 5셀이다. 1차는 control_success 를 난수화해 양성 대조가 0.0 이었고, "
         "그 자체가 이 사실의 실행 증거다(`out/s1_leak107_s2020.json`).",
     "cell_round": cell_round, "catboost_thread_count": NT, "seed": SEED,
     "PART1_라벨복원_통로": {
         "무엇": f"asof_*_rate 5열을 대역째 난수화 → recover 재실행 → fit_rows·fit라벨 대조",
         "왜_비지_않았나": "`recov_ok` 는 **같은 투수의 다음 행**에 의존하므로 다음 행이 "
                          f"season>={S} 인 season<{S} 행이 오염될 수 있다. "
                          f"`succ_season < {S}` 가 그 행을 빼는지를 실행으로 확인한다.",
         "runs": P1,
         "NEG_identical": p1_neg, "POS_moves": p1_pos},
     "PART2_종단": {
         "무엇": "4셀 라벨을 대역째 난수화하고 mk_oof 의 시즌 고리를 그대로 돌린다",
         "n_fit_rows": int(fit_rows.sum()), "n_pred_rows": int(pred_rows.sum()),
         "fit_label_sha16": LS, "pred_sha16": {k: sha(v) for k, v in PS.items()},
         "NEG_identical": p2_neg, "POS_moves": p2_pos,
         "P_max_abs_move": float(np.abs(PS["P"] - PS["A"]).max()),
         "P_share_rows_moved": float((PS["P"] != PS["A"]).mean()),
         "D_determinism": "1차에서 통과 (A == D, sha 1b67d93f88a252cb)"},
     "PASS": bool(p1_neg and p1_pos and p2_neg and p2_pos),
     "wall_sec": round(time.time() - t0, 1)}
json.dump(R, open(f"{B}/out/s1b_leak107_s{S}.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({"PART1": {k: R["PART1_라벨복원_통로"][k]
                            for k in ("NEG_identical", "POS_moves")},
                  "PART2": {k: R["PART2_종단"][k]
                            for k in ("NEG_identical", "POS_moves",
                                      "P_max_abs_move", "P_share_rows_moved")},
                  "PASS": R["PASS"]}, ensure_ascii=False, indent=1))
print("LEAK_GATE_PASS" if R["PASS"] else "LEAK_GATE_FAIL")
