"""exp_104 축B2 — TabPFN 을 **전 행이 아니라 팔들이 가장 엇갈리는 행에만** 태운다.

행 고르는 규칙: `|z_deep − z_mate| >= tau`.
🔴 분위(상위 5%)가 아니라 **고정 상수 tau** 를 쓴다 — 분위는 test 프레임 전체를 봐야 정해지므로
   규칙 2-4 위반이다(S26·S28 계열). 상수는 행마다 독립으로 판정된다.
   tau 는 2024 면에서 목표 비중이 나오게 고른 값이고, 그 사실을 그대로 적는다
   (배포에서는 이 상수를 동결해 싣는다. 발동 비중은 test 에서 달라질 수 있다).

시간: 질의 행 수에 선형이므로 5% 면 1/20 이다. A5000 실측 -> L4 보수 2.94배(S164) 로 환산.
"""
import argparse, json, os, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--n_ctx", type=int, default=20000)
ap.add_argument("--n_est", type=int, default=1)
ap.add_argument("--bags", type=int, default=1)
ap.add_argument("--chunk", type=int, default=40000)
ap.add_argument("--ckpt", default="ckpt/tabpfn-v2-classifier-v2_default.ckpt")
ap.add_argument("--fracs", default="0.05,0.10")
ap.add_argument("--topk", type=int, default=0, help="0 = 94열 전부")
a = ap.parse_args()
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS"):
    os.environ[v] = str(a.threads)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"
import numpy as np
import torch

torch.set_num_threads(a.threads)
H = "/home/<user>/SungjinPark"
B101, B = f"{H}/exp101", f"{H}/exp104"
sys.path.insert(0, f"{B}/src")
import surface104 as S                                        # noqa: E402
from tabpfn import TabPFNClassifier                           # noqa: E402

X = np.load(f"{H}/exp096/work/X_f64_oot23.npy", mmap_mode="r")
y = np.load(f"{H}/exp096/workb/y.npy")
trm = np.load(f"{H}/exp089/work/trm_oot23.npy")
vam = np.load(f"{H}/exp089/work/valid_mask.npy")
tr, va = np.flatnonzero(trm), np.flatnonzero(vam)
assert len(va) == len(S.y)
NUM = json.load(open(f"{H}/audit_deep/pkg/DEEP/mate/model/deep/prep.json",
                     encoding="utf-8"))["NUM"]
cols = list(range(94))
if a.topk:
    RK = json.load(open(f"{B}/work/colrank_le2023.json"))
    cols = sorted(RK["order_idx"][:a.topk])

disc = np.abs(S.Z_DEEP - S.Z_MATE)
rng = np.random.default_rng(2026)
perm = rng.permutation(tr)
REP = {"exp_id": "exp_104", "stage": "sB2_rows", "n_ctx": a.n_ctx,
       "n_est": a.n_est, "bags": a.bags, "n_cols": len(cols), "topk": a.topk,
       "row_rule": "|z_deep - z_mate| >= tau  (고정 상수. 분위 아님 — 규칙 2-4)",
       "disagreement_stats": {"mean": float(disc.mean()), "sd": float(disc.std()),
                              "p50": float(np.percentile(disc, 50)),
                              "p90": float(np.percentile(disc, 90)),
                              "p95": float(np.percentile(disc, 95))},
       "L4_factor": 2.94, "L4_budget_sec": 484.0, "rows": []}
for fr in [float(x) for x in a.fracs.split(",")]:
    tau = float(np.quantile(disc, 1.0 - fr))
    sel = disc >= tau
    qi = va[sel]
    Xq = np.ascontiguousarray(X[np.ix_(qi, cols)]).astype(np.float32)
    Z = np.zeros(len(qi))
    t_fit = t_pred = 0.0
    for b in range(a.bags):
        ci = np.sort(perm[b * a.n_ctx:(b + 1) * a.n_ctx])
        assert len(np.intersect1d(ci, qi)) == 0
        clf = TabPFNClassifier(device=a.device, model_path=f"{B101}/{a.ckpt}",
                               n_estimators=a.n_est,
                               ignore_pretraining_limits=True, random_state=42 + b)
        t0 = time.time()
        clf.fit(np.ascontiguousarray(X[np.ix_(ci, cols)]).astype(np.float32),
                y[ci].astype(np.int64))
        t_fit += time.time() - t0
        t1 = time.time()
        p = np.concatenate([clf.predict_proba(Xq[s:s + a.chunk])[:, 1]
                            for s in range(0, len(Xq), a.chunk)])
        t_pred += time.time() - t1
        Z += S.lg(p)
    Z /= a.bags
    tag = f"f{int(round(fr*100)):02d}_k{a.topk or 94}_b{a.bags}"
    np.save(f"{B}/work/z_tabB2_{tag}.npy", Z)
    np.save(f"{B}/work/sel_tabB2_{tag}.npy", sel)
    tot = t_fit + t_pred
    sc245 = tot * 245789.0 / len(S.y)
    yq = y[qi].astype(np.float64)
    pz = 1.0 / (1.0 + np.exp(-Z))
    r = {"tag": tag, "target_frac": fr, "tau": tau, "n_query": int(sel.sum()),
         "share_of_2024": float(sel.mean()),
         "brier_on_selected_rows": float(np.mean((pz - yq) ** 2)),
         "t_fit_sec": round(t_fit, 2), "t_pred_sec": round(t_pred, 2),
         "t_total_sec": round(tot, 2),
         "a5000_sec_245789frame": round(sc245, 2),
         "L4_conservative_sec": round(sc245 * 2.94, 1),
         "fits_L4_budget_484s": bool(sc245 * 2.94 <= 484.0)}
    REP["rows"].append(r)
    print(json.dumps(r, ensure_ascii=False), flush=True)
json.dump(REP, open(f"{B}/out/sB2_rows_k{a.topk or 94}_b{a.bags}.json", "w"),
          ensure_ascii=False, indent=1)
print("SB2_DONE")
