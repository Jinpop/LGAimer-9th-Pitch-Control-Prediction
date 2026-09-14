"""exp_103 stage1 — 값싼 신호 선별. GPU CatBoost 3판본.

① base   : 기존 99열 (수치 94 + 범주 5) — exp097/work 의 OOT 프레임 그대로
② form   : ① + 타자 폼 19열
③ null   : ① + **같은 19열을 시즌 안에서 행 단위 순열** (주변 분포·열간 상관 보존, 행 연결만 절단)
   🔴 ③ 없이는 판정이 성립하지 않는다 (S104).
   보조 ③b : 스펙 문자 그대로 batter_id 안 순열 — 동결 열에는 무효라 감도가 낮다. 둘 다 낸다.

사전 등록 문턱 (발사 전 확정, S8): ② > ① **그리고** ② > ③, 둘 다 짝지은 투수 블록 부트스트랩 t >= 2.0
"""
import hashlib, json, os, sys, time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

H = "/home/<user>/SungjinPark"
B = f"{H}/exp103"
W97 = f"{H}/exp097/work"
OUT = f"{B}/out"
REF = 0.2498069
SEEDS = (42, 7, 2024)
ITERS = int(os.environ.get("ITERS103", "3000"))
DEPTH = 6
LR = 0.05
NBOOT = 2000

X = np.load(f"{W97}/X_oot23.npy")
C = np.load(f"{W97}/codes_oot23.npy")
y = np.load(f"{W97}/y.npy").astype(np.float64)
trm = np.load(f"{W97}/trm_oot23.npy")
vam = np.load(f"{W97}/valid_mask.npy")
F = np.load(f"{B}/work/batform.npy")
FCOLS = json.load(open(f"{B}/work/batform_cols.json"))["cols"]

meta = pd.read_csv(f"{H}/data/train.csv",
                   usecols=["season", "pitcher_id", "batter_id"], encoding="utf-8-sig")
season = meta["season"].values.astype(np.int32)
pit = meta["pitcher_id"].values.astype(np.int64)
bat = meta["batter_id"].values.astype(np.int64)
N = len(y)
assert X.shape[0] == N == F.shape[0]

rep = {"n_rows": int(N), "iters": ITERS, "depth": DEPTH, "lr": LR, "seeds": list(SEEDS),
       "n_new_cols": len(FCOLS), "new_cols": FCOLS,
       "prereg": "② > ① AND ② > ③, 둘 다 짝지은 투수 블록 부트스트랩 t >= 2.0 (발사 전 확정)"}
rep["mask_season_check"] = {
    "train_seasons": {str(k): int(v) for k, v in
                      pd.Series(season[trm]).value_counts().sort_index().items()},
    "valid_seasons": {str(k): int(v) for k, v in
                      pd.Series(season[vam]).value_counts().sort_index().items()},
    "n_train": int(trm.sum()), "n_valid": int(vam.sum())}
# 🔴 리크 게이트: 학습 마스크에 목표 시즌이 없어야 한다
assert (season[trm] <= 2023).all()
assert (season[vam] == 2024).all()

# ── 귀무 판본 만들기 ────────────────────────────────────────────
rs = np.random.RandomState(20260826)
perm_a = np.arange(N)
for s in np.unique(season):                      # 시즌 안 행 단위 순열
    idx = np.where(season == s)[0]
    perm_a[idx] = rs.permutation(idx)
Fa = F[perm_a]
perm_b = np.arange(N)                            # batter_id 안 순열 (보조)
o = np.lexsort((np.arange(N), bat))
bs = bat[o]
bnd = np.concatenate([[0], np.where(np.diff(bs) != 0)[0] + 1, [N]])
for i in range(len(bnd) - 1):
    seg = o[bnd[i]:bnd[i + 1]]
    perm_b[seg] = rs.permutation(seg)
Fb = F[perm_b]

VARIANTS = {"base": None, "form": F, "null_row": Fa, "null_bat": Fb}
CATIDX = list(range(94, 99))


def make_df(extra):
    """범주 5열은 int 로 둔다 — CatBoost 는 전부 float 인 배열에 cat_features 를 못 받는다."""
    d = {f"n{j}": X[:, j] for j in range(94)}
    for j in range(5):
        d[f"c{j}"] = C[:, j].astype(np.int32)
    if extra is not None:
        for j, cn in enumerate(FCOLS):
            d[cn] = extra[:, j]
    return pd.DataFrame(d)


CATNAMES = [f"c{j}" for j in range(5)]


def fit_arm(tag, extra):
    D = make_df(extra)
    ci = CATNAMES
    ptr = Pool(D[trm], y[trm], cat_features=ci)
    pva = Pool(D[vam], cat_features=ci)
    ps, times = [], []
    for sd in SEEDS:
        t0 = time.time()
        m = CatBoostClassifier(iterations=ITERS, depth=DEPTH, learning_rate=LR,
                               loss_function="Logloss", task_type="GPU", devices="0",
                               random_seed=sd, verbose=0, thread_count=4,
                               l2_leaf_reg=3.0, bootstrap_type="Bernoulli", subsample=0.8)
        m.fit(ptr)
        ps.append(m.predict_proba(pva)[:, 1].astype(np.float64))
        times.append(round(time.time() - t0, 1))
        print(f"{tag} seed {sd} {times[-1]}s", flush=True)
    del D, ptr, pva
    P = np.mean(ps, axis=0)
    np.save(f"{B}/work/p_{tag}.npy", P)
    per_seed = [float(100000.0 * (1 - np.mean((q - y[vam]) ** 2) / REF)) for q in ps]
    return P, {"per_seed_score": per_seed, "fit_sec": times}


yv = y[vam]
pv = pit[vam]
preds, info = {}, {}
for tag, extra in VARIANTS.items():
    preds[tag], info[tag] = fit_arm(tag, extra)
    br = float(np.mean((preds[tag] - yv) ** 2))
    info[tag].update({"brier": br, "score": 100000.0 * (1 - br / REF)})
    print(tag, info[tag]["score"], flush=True)
rep["arms"] = info

# ── 짝지은 투수 블록 부트스트랩 ─────────────────────────────────
upit, inv = np.unique(pv, return_inverse=True)
G = len(upit)
ordi = np.argsort(inv, kind="stable")
inv_s = inv[ordi]
bnd2 = np.concatenate([[0], np.where(np.diff(inv_s) != 0)[0] + 1, [len(inv_s)]])
blocks = [ordi[bnd2[i]:bnd2[i + 1]] for i in range(len(bnd2) - 1)]
sq = {t: (preds[t] - yv) ** 2 for t in preds}
blk_sum = {t: np.array([sq[t][b].sum() for b in blocks]) for t in preds}
blk_n = np.array([len(b) for b in blocks], dtype=np.float64)
rb = np.random.RandomState(777)
draws = rb.randint(0, G, size=(NBOOT, G))


def boot_t(a, c):
    d = blk_sum[a] - blk_sum[c]
    obs = -(d.sum() / blk_n.sum()) / REF * 100000.0
    num = d[draws].sum(1)
    den = blk_n[draws].sum(1)
    vals = -(num / den) / REF * 100000.0
    return {"delta_score": float(obs), "boot_sd": float(vals.std(ddof=1)),
            "t": float(obs / vals.std(ddof=1)), "boot_mean": float(vals.mean()),
            "frac_positive": float((vals > 0).mean())}


rep["contrasts"] = {
    "form_minus_base": boot_t("form", "base"),
    "form_minus_null_row": boot_t("form", "null_row"),
    "form_minus_null_bat": boot_t("form", "null_bat"),
    "null_row_minus_base": boot_t("null_row", "base"),
    "null_bat_minus_base": boot_t("null_bat", "base"),
}
c1, c2 = rep["contrasts"]["form_minus_base"], rep["contrasts"]["form_minus_null_row"]
rep["VERDICT"] = {
    "pass": bool(c1["delta_score"] > 0 and c1["t"] >= 2.0
                 and c2["delta_score"] > 0 and c2["t"] >= 2.0),
    "form_minus_base": [c1["delta_score"], c1["t"]],
    "form_minus_null_row": [c2["delta_score"], c2["t"]]}

# ── 중복 진단: 기존 열과의 상관 ─────────────────────────────────
Xv = X[vam]
corr = {}
for j, cn in enumerate(FCOLS):
    v = F[vam][:, j].astype(np.float64)
    ok = np.isfinite(v)
    if ok.sum() < 1000 or np.nanstd(v[ok]) == 0:
        corr[cn] = {"max_abs_rho": None, "note": "degenerate"}
        continue
    best, bj = 0.0, -1
    for k in range(94):
        w = Xv[ok, k].astype(np.float64)
        if w.std() == 0:
            continue
        r = abs(np.corrcoef(v[ok], w)[0, 1])
        if np.isfinite(r) and r > best:
            best, bj = r, k
    corr[cn] = {"max_abs_rho": float(best), "arg_col_index": int(bj),
                "cov_frac": float(ok.mean())}
rep["corr_vs_existing94"] = corr

json.dump(rep, open(f"{OUT}/s1_screen.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({"arms": {k: v["score"] for k, v in info.items()},
                  "contrasts": rep["contrasts"], "VERDICT": rep["VERDICT"]},
                 ensure_ascii=False, indent=1))
