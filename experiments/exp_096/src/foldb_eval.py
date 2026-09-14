"""exp_096 진단 — 독립 fold(<=2022 -> 2023) 결과 판정.

혼합 이득은 이 fold 에서 못 잰다(z_mate 가 없다).  재는 것은 팔 단독 대조 하나:
  rank 5시드 평균 vs 표준화 5시드 평균, 같은 구조·같은 예산.
짝지은 투수 블록 부트스트랩으로 t 를 낸다 (fold_2024 와 같은 추출 규약, 시드만 같은 상수).
"""
import itertools, json, os, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np
from scipy.special import expit

H = "/home/<user>/SungjinPark"
B, W = f"{H}/exp096", f"{H}/exp096/workb"
REF = 0.2498069
K = 100000.0 / REF
S5 = [42, 777, 2024, 101, 7]
y = np.load(f"{W}/y.npy")[np.load(f"{W}/vam.npy")].astype(np.float64)
pit = np.load(f"{W}/pit_valid.npy", allow_pickle=True)
YB = float(y.mean())


def pin(z):
    c = 0.0
    for _ in range(200):
        p = expit(z + c)
        g = float(p.mean()) - YB
        if abs(g) < 1e-15:
            break
        c -= g / float((p * (1 - p)).mean())
    return c


sc = lambda z: 100000.0 * (1.0 - float(np.mean((expit(z + pin(z)) - y) ** 2)) / REF)
rng = np.random.default_rng(20260824)
uq, inv = np.unique(pit, return_inverse=True)
nb = len(uq)
order = np.argsort(inv, kind="stable")
st = np.searchsorted(inv[order], np.arange(nb))
en = np.append(st[1:], len(order))
BSIZE = (en - st).astype(np.float64)
PICKS = np.stack([rng.integers(0, nb, nb) for _ in range(1000)])
IDX = [order[st[j]:en[j]] for j in range(nb)]


def boot(d):
    bs_ = np.array([d[IDX[j]].sum() for j in range(nb)])
    bs = bs_[PICKS].sum(1) / BSIZE[PICKS].sum(1)
    se = float(bs.std(ddof=1))
    obs = float(d.mean())
    return {"gain_pts": -obs * K, "t": float(-obs / se) if se > 0 else None,
            "p_share_rank_worse": float((bs > 0).mean()),
            "ci95_pts": [float(-np.percentile(bs, 97.5) * K),
                         float(-np.percentile(bs, 2.5) * K)]}


Z = {}
for i in ("std", "rank"):
    zs = [np.load(f"{B}/out/predsb/z_{i}_le22_s{s}.npy") for s in S5]
    Z[i] = {"per_seed": [sc(z) for z in zs], "mean_z": np.mean(zs, 0),
            "seed_rho": [float(np.corrcoef(zs[a], zs[b])[0, 1])
                         for a, b in itertools.combinations(range(5), 2)]}
    Z[i]["seedmean_score"] = sc(Z[i]["mean_z"])
Pr = expit(Z["rank"]["mean_z"] + pin(Z["rank"]["mean_z"]))
Ps = expit(Z["std"]["mean_z"] + pin(Z["std"]["mean_z"]))
R = {"exp_id": "exp_096", "stage": "foldb_eval",
     "fold": "train <= 2022  ->  valid season 2023",
     "n_valid": int(len(y)), "y_mean": YB,
     "🔴_what_this_can_and_cannot_say": (
         "이 fold 에는 z_mate·z_aa·z_T70 이 없어 **혼합 이득은 못 잰다**. "
         "여기서 나오는 것은 '같은 구조·같은 예산에서 순위변환 입력이 팔을 낫게 하는가' 하나다."),
     "std": {k: v for k, v in Z["std"].items() if k != "mean_z"},
     "rank": {k: v for k, v in Z["rank"].items() if k != "mean_z"},
     "rho_rank_vs_std": float(np.corrcoef(Z["rank"]["mean_z"], Z["std"]["mean_z"])[0, 1]),
     "delta_seedmean_score": Z["rank"]["seedmean_score"] - Z["std"]["seedmean_score"],
     "paired_boot_rank_minus_std": boot((Pr - y) ** 2 - (Ps - y) ** 2)}
json.dump(R, open(f"{B}/out/foldb_eval.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1))
