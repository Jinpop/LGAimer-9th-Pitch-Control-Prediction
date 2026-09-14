"""exp_090 stage 2 보강 — 규칙 2-4 내용/위치 분리 + 배포 팔 대비 이동량 + 추론 시간.

왜: 게이트 G5(전체 순열 왕복)에서 9,169행이 **2.2e-16**(1 ulp) 움직였다. 100행·1행 단독
검사만으로는 위치와 내용을 못 가른다(CLAUDE.md §규칙 2-4 판정). 그래서 **내용 분리**를 한다:
  KEEP 110행은 값도 인덱스도 그대로 두고, **나머지 253,397행을 두 극단 행의 반복으로 전량 교체**
  → KEEP 행의 예측이 **정확히 0.0** 만큼 움직이면 다른 행의 *내용*은 통로가 아니다.
남는 것은 인덱스(청크 안 위치)뿐이고, 인덱스는 다른 행의 정보가 아니다.
"""
import json, os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "6"
import numpy as np
import pandas as pd

H = "/home/<user>/SungjinPark"
B90 = f"{H}/exp090"
PKG = f"{H}/audit_deep/pkg/DEEP/mate"
E60 = f"{H}/exp060_nn"
sys.path.insert(0, PKG)
import deep_infer as DI                                    # noqa: E402

S5 = [42, 777, 2024, 101, 7]
TAGS = sys.argv[1:]
P = json.load(open(f"{PKG}/model/deep/prep.json", encoding="utf-8"))
NUM, CAT = P["NUM"], P["CAT"]
df = pd.read_parquet(f"{E60}/cache/frame99.parquet",
                     columns=list(dict.fromkeys(NUM + CAT + ["row_id", "season"])))
tr = pd.read_csv(f"{H}/data/train.csv", usecols=["row_id", "season"])
assert np.array_equal(df.row_id.to_numpy().astype(str), tr.row_id.to_numpy().astype(str))
dfv = df.loc[tr.season.to_numpy() == 2024].reset_index(drop=True)
n = len(dfv)
assert n == 253507
MATE = [f"{PKG}/model/deep/deepf_s{s}.npz" for s in S5]

rng = np.random.default_rng(90090)
KEEP = np.sort(rng.choice(n, 110, replace=False))
# 두 극단 행 (KEEP 에 없는 행에서 고른다) 을 반복해 나머지를 전량 교체
pool = np.setdiff1d(np.arange(n), KEEP)
lo, hi = int(pool[0]), int(pool[-1])
degen = dfv.copy()
fill = dfv.iloc[[lo if i % 2 == 0 else hi for i in range(n)]].reset_index(drop=True)
mask = np.ones(n, bool); mask[KEEP] = False
for c in dfv.columns:
    degen.loc[mask, c] = fill.loc[mask, c].to_numpy()
assert (degen.iloc[KEEP].reset_index(drop=True)
        .equals(dfv.iloc[KEEP].reset_index(drop=True)))

R = {"exp_id": "exp_090", "stage": "s3_rule24", "n_rows": n,
     "n_keep": int(len(KEEP)),
     "method": ("KEEP 110행은 원문·원위치, 나머지 253,397행을 두 극단 행 반복으로 전량 교체 "
                "(S28 계열). KEEP 행 delta 가 정확히 0.0 이면 다른 행의 내용은 통로가 아니다.")}
for T in TAGS:
    D = f"{B90}/out/deploy_{T}"
    NPZ = [f"{D}/deepf_s{s}.npz" for s in S5]
    t0 = time.time()
    p_full = DI.predict_deep(dfv, f"{D}/prep.json", NPZ)
    t_inf = time.time() - t0
    p_deg = DI.predict_deep(degen, f"{D}/prep.json", NPZ)
    d = np.abs(p_deg[KEEP] - p_full[KEEP])
    pm = DI.predict_deep(dfv, f"{PKG}/model/deep/prep.json", MATE)
    zl = lambda q: np.log(q / (1 - q))
    R[T] = {
        "content_separation_max_abs_delta_on_KEEP": float(d.max()),
        "content_separation_n_moving": int((d > 0).sum()),
        "PASS_content_independent": bool(d.max() == 0.0),
        "inference_sec_253507rows_5seeds_6threads": round(t_inf, 2),
        "vs_shipped_deep_arm": {
            "rho_logit": float(np.corrcoef(zl(p_full), zl(pm))[0, 1]),
            "sd_logit_ours": float(zl(p_full).std()),
            "sd_logit_shipped": float(zl(pm).std()),
            "max_abs_prob_diff": float(np.abs(p_full - pm).max()),
            "mean_abs_prob_diff": float(np.abs(p_full - pm).mean())},
    }
    print(T, json.dumps(R[T], ensure_ascii=False), flush=True)
json.dump(R, open(f"{B90}/out/s3_rule24.json", "w"), ensure_ascii=False, indent=1)
