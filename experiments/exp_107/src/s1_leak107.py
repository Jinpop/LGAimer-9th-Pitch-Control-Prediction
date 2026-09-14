"""exp_107 축 B 리크 게이트 — **실행으로** 증명한다(S171 방법).

주장: `p_self[season == s]` 는 **season >= s 의 라벨을 한 톨도 안 봤다**.
     (그래야 축 B 의 타깃 `(1-e)y + e*p_self` 가 그 행에 대해 out-of-time 이다)

`mk_oof.py` 의 한 시즌 고리(기본 s = 2020, 연쇄의 첫 예측 시즌이라 가장 싸다)를
**파이프라인 입구인 `control_success` 부터** 네 번 통째로 돌린다:

  A  원본           원 라벨
  D  결정성 대조    원 라벨로 한 번 더                     → p_D == p_A 여야 한다.
                    🔴 이게 없으면 "다르다"를 리크와 비결정성으로 못 가른다.
  N  음성(리크 시험) `control_success` 를 season >= s 에서 전량 난수화 →
                    recover() 재실행 → 셀 라벨·마스크·프레임 전부 재유도
                    → p_N == p_A 여야 한다 (그 시즌 라벨이 예측에 안 닿는다)
  P  양성 대조      `control_success` 를 season <  s 에서 전량 난수화
                    → p_P != p_A 여야 한다 (하니스가 살아 있다)

🔴 N 에서 마스크(`recov_ok` 유래 `fit_rows`)와 피처 프레임이 **비트 동일한지도 따로 적는다** —
   예측이 같아도 그 이유가 "마스크가 우연히 같아서"인지 구별해야 하기 때문이다.
"""
import gc, hashlib, importlib.util, json, os, sys, time
import numpy as np
import pandas as pd

H = "/home/<user>/SungjinPark"
B = f"{H}/exp107"
sys.path.insert(0, f"{H}/experiments/exp_036/src")
import exp016_ref as ref                                     # noqa: E402
from labels import FAIL_CELLS, recover                       # noqa: E402

DATA = f"{H}/data/train.csv"
F_EXCL_MAX, TU = 2022, 2023
CAT = ["top_bottom", "game_type", "base_state"]
NT = 2                                     # 🔴 CPU 2코어 (6코어 상한 안)
S = int(os.environ.get("SEASON107", "2020"))
SEED = 42
REFNPZ = f"{H}/build_AH/localref/oof/stack_fold2024_seed42.npz"

spec = importlib.util.spec_from_file_location(
    "e27", f"{H}/experiments/exp_027/src/train.py")
e27 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e27)
bfn = e27.build_frame_native

raw_df = pd.read_csv(DATA, encoding="utf-8-sig")
base_feats = [c for c in raw_df.columns if c not in ("row_id", "control_success")]
assert len(base_feats) == 47
cat_native = [c for c in base_feats if c in CAT]
season = raw_df["season"].to_numpy()
band_hi = season >= S                      # 예측 시즌 **포함** — 자기 라벨도 여기 들어간다
band_lo = season < S
z0 = np.load(REFNPZ, allow_pickle=False)
cell_round = {c: int(v) for c, v in zip(FAIL_CELLS, z0["cell_round"].tolist())}
sha = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]
rng = np.random.default_rng(20260826)
RND = {"N": rng.integers(0, 2, size=int(band_hi.sum())),
       "P": rng.integers(0, 2, size=int(band_lo.sum()))}


def pipeline(tag):
    """`control_success` 를 손댄 뒤 mk_oof.py 의 시즌 s 고리를 그대로 재현한다."""
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    if tag == "N":
        cs = df["control_success"].to_numpy().copy(); cs[band_hi] = RND["N"]
        df["control_success"] = cs
    elif tag == "P":
        cs = df["control_success"].to_numpy().copy(); cs[band_lo] = RND["P"]
        df["control_success"] = cs
    rec = recover(df, verbose=False)
    ok = df["recov_ok"].to_numpy()
    ss = df.groupby("pitcher_id", sort=False)["season"].shift(-1).to_numpy()
    f_excl = ((df["season"] <= F_EXCL_MAX) & (df["game_type"] == "F")).to_numpy()
    tr_full = (season <= TU) & ~f_excl
    fit_rows = (season < S) & ~f_excl & ok & (ss < S)
    pred_rows = (season == S) & tr_full
    Ff = bfn(df.loc[fit_rows], base_feats, cat_native)
    Fp = bfn(df.loc[pred_rows], base_feats, cat_native)
    ci = [base_feats.index(c) for c in cat_native]
    import catboost as cb
    tot = 0.0
    for c in FAIL_CELLS:
        par = dict(e27.CB_BASE, random_seed=SEED,
                   iterations=cell_round[c], thread_count=NT)
        m = cb.CatBoostClassifier(**par)
        m.fit(cb.Pool(Ff, label=df.loc[fit_rows, c].to_numpy(np.float64),
                      cat_features=ci))
        tot = tot + ref.sigmoid(np.asarray(
            m.predict(cb.Pool(Fp, cat_features=ci),
                      prediction_type="RawFormulaVal", thread_count=NT), np.float64))
        del m; gc.collect()
    info = {"agree_pct": float(rec["success_recovery"]["agree_pct"]),
            "n_fit_rows": int(fit_rows.sum()), "n_pred_rows": int(pred_rows.sum()),
            "fit_rows_sha16": sha(fit_rows), "pred_rows_sha16": sha(pred_rows),
            "fit_labels_sha16": {c: sha(df.loc[fit_rows, c].to_numpy(np.float64))
                                 for c in FAIL_CELLS},
            "Ff_sha16": sha(np.ascontiguousarray(
                Ff.select_dtypes("number").to_numpy(np.float64))),
            "Fp_sha16": sha(np.ascontiguousarray(
                Fp.select_dtypes("number").to_numpy(np.float64)))}
    del Ff, Fp; gc.collect()
    return tot, info


t0 = time.time()
PS, INFO = {}, {}
for tag in ("A", "D", "N", "P"):
    PS[tag], INFO[tag] = pipeline(tag)
    print(f"[{tag}] p_fail_sum sha16 {sha(PS[tag])} n_pred {len(PS[tag])} "
          f"({time.time()-t0:.0f}s)", flush=True)

det = bool(np.array_equal(PS["A"], PS["D"]))
neg = bool(np.array_equal(PS["A"], PS["N"]))
pos = bool(not np.array_equal(PS["A"], PS["P"]))
same_mask_N = bool(INFO["N"]["fit_rows_sha16"] == INFO["A"]["fit_rows_sha16"]
                   and INFO["N"]["pred_rows_sha16"] == INFO["A"]["pred_rows_sha16"])
same_lab_N = bool(INFO["N"]["fit_labels_sha16"] == INFO["A"]["fit_labels_sha16"])
same_frame_N = bool(INFO["N"]["Ff_sha16"] == INFO["A"]["Ff_sha16"]
                    and INFO["N"]["Fp_sha16"] == INFO["A"]["Fp_sha16"])

R = {"exp_id": "exp_107", "stage": "s1_leak107", "season_tested": S,
     "무엇을_증명하나": f"시즌 {S} 행의 p_self 가 season >= {S} 의 라벨을 안 봤다 — "
                       "control_success 를 그 대역에서 전량 난수화하고 recover 부터 다시 돌렸다",
     "cell_round": cell_round, "catboost_thread_count": NT, "seed": SEED,
     "per_run": INFO,
     "sha16_pred": {k: sha(v) for k, v in PS.items()},
     "GATES": {
         "D_determinism_A_equals_A2": det,
         "N_negative_randomize_season_ge_S__pred_identical": neg,
         "N_masks_identical": same_mask_N,
         "N_fit_labels_identical": same_lab_N,
         "N_feature_frames_identical": same_frame_N,
         "P_positive_randomize_season_lt_S__pred_moves": pos,
         "P_max_abs_move": float(np.abs(PS["P"] - PS["A"]).max()),
         "P_share_rows_moved": float((PS["P"] != PS["A"]).mean()),
         "P_n_fit_rows": INFO["P"]["n_fit_rows"]},
     "읽는_법": "N 이 통과하는 기전은 셋이 함께다 — ① 마스크가 안 변하고 ② 적합 라벨이 안 변하고 "
                "③ 피처 프레임이 안 변한다. 셋 다 sha 로 적었다.",
     "PASS": bool(det and neg and pos),
     "wall_sec": round(time.time() - t0, 1)}
json.dump(R, open(f"{B}/out/s1_leak107_s{S}.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(R["GATES"], ensure_ascii=False, indent=1))
print("LEAK_GATE_PASS" if R["PASS"] else "LEAK_GATE_FAIL")
