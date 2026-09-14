"""exp_103b stage1 — exp_103 의 17열 중 **3열만** 다시 재는 판본.

왜: exp_103 은 17열을 한꺼번에 얹어 기각됐다(② − ① = −28.942 t −2.029).
    독립 감사가 원인을 좁혔다 — 17열 중 14열이 이미 배포된 `ins_b_*`·`sp_bat_phand`·
    `asof_batter_*` 의 준중복이다(`bat_cs_n` 은 `ins_b_n − 1` 로 대수적으로 동일, ρ 1.0000).
    기존 94열과 진짜로 안 겹치는 것은 셋뿐이다:
        bat_base_dev  max|ρ| 0.048 · bat_n_growth 0.170 · bat_count_dev 0.269
    그 셋은 단독으로 시험된 적이 없다. 기각이 3열 탓인지 14열의 분기 후보 잠식 탓인지 가른다.

🔴 새 피처를 만들지 않는다. `exp103/work/batform2.npy`(17열)에서 **3열만 슬라이스**한다.
   따라서 규칙 2-4 행 독립성 게이트(exp_103 out/s1_rowindep.json, 4종 전부 max|Δ| = 0.0)가
   그대로 승계된다 — 열 부분집합은 새 계산 경로가 아니다.

판본 넷 (① 은 exp_103 산출물 재사용):
  ① base      : 기존 99열 (수치 94 + 범주 5). **exp103/work/p2_base.npy 재사용**
                — extra=None 이라 입력이 exp_103 과 한 비트도 다르지 않다. sha 기록.
  ② form3     : ① + 3열
  ③ null_bat3 : ① + **3열을 batter_id 안에서 순열** (사전 등록 귀무, 코디네이터 지정)
  ③b null_row3: ① + 3열을 시즌 안 행 단위 순열 (보조 — exp_103 의 주 귀무와 같은 방식)
  ④ rand3     : ① + **기존 94열에서 뽑은 무작위 3열을 복제·순열** (개수 효과 자체의 대조)

🔴 사전 등록 문턱 (exp_103 과 동일, 낮추지 않는다 — S8):
   ② > ① **그리고** ② > ③, 둘 다 짝지은 투수 블록 부트스트랩 t >= 2.0

⚠️ 이 화면의 검출한계: exp_103 에서 부트스트랩 1σ 가 14~18점 → t≥2.0 은 약 30점 효과를 요구한다.
   3열로 30점이 나올 가능성은 낮다. **문턱 통과 여부와 별개로 ②−① · ②−③ 의 점추정·부호·t 를 전부 낸다.**

하니스·시드·하이퍼는 exp_103 s1_screen2.py 와 완전히 같다(RandomState 소비 순서까지).
"""
import hashlib, json, os, time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

H = "/home/<user>/SungjinPark"
B103 = f"{H}/exp103"
B = f"{H}/exp103b"
W97 = f"{H}/exp097/work"
OUT = f"{B}/out"
os.makedirs(f"{B}/work", exist_ok=True)
os.makedirs(OUT, exist_ok=True)
REF = 0.2498069
SEEDS = (42, 7, 2024)
ITERS = int(os.environ.get("ITERS103", "3000"))
DEPTH = 6
LR = 0.05
NBOOT = 2000

# ── 3열 선택 ────────────────────────────────────────────────────
SEL_NAMES = ["bat_n_growth", "bat_count_dev", "bat_base_dev"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()[:16]


X = np.load(f"{W97}/X_oot23.npy")
C = np.load(f"{W97}/codes_oot23.npy")
y = np.load(f"{W97}/y.npy").astype(np.float64)
trm = np.load(f"{W97}/trm_oot23.npy")
vam = np.load(f"{W97}/valid_mask.npy")
F17 = np.load(f"{B103}/work/batform2.npy")
COLS17 = json.load(open(f"{B103}/work/batform2_cols.json"))["cols"]
NUM94 = json.load(open(f"{H}/audit_deep/pkg/DEEP/mate/model/deep/prep.json",
                      encoding="utf-8"))["NUM"]
SEL = [COLS17.index(c) for c in SEL_NAMES]
F = F17[:, SEL].copy()

meta = pd.read_csv(f"{H}/data/train.csv",
                   usecols=["season", "pitcher_id", "batter_id"], encoding="utf-8-sig")
season = meta["season"].values.astype(np.int32)
pit = meta["pitcher_id"].values.astype(np.int64)
bat = meta["batter_id"].values.astype(np.int64)
N = len(y)
assert X.shape[0] == N == F17.shape[0]

rep = {"exp_id": "exp_103b",
       "n_rows": int(N), "iters": ITERS, "depth": DEPTH, "lr": LR, "seeds": list(SEEDS),
       "sel_cols": SEL_NAMES, "sel_index_in_batform2": SEL,
       "source_matrix": f"{B103}/work/batform2.npy",
       "source_matrix_sha16": sha(f"{B103}/work/batform2.npy"),
       "prereg": "② > ① AND ② > ③(null_bat3), 둘 다 짝지은 투수 블록 부트스트랩 t >= 2.0 "
                 "— exp_103 과 동일, 결과 보고 안 바꿨다 (S8)"}
rep["mask_season_check"] = {
    "train_seasons": {str(k): int(v) for k, v in
                      pd.Series(season[trm]).value_counts().sort_index().items()},
    "valid_seasons": {str(k): int(v) for k, v in
                      pd.Series(season[vam]).value_counts().sort_index().items()},
    "n_train": int(trm.sum()), "n_valid": int(vam.sum())}
assert (season[trm] <= 2023).all()
assert (season[vam] == 2024).all()

# ── 귀무 순열 (exp_103 s1_screen2.py 와 RandomState 소비 순서까지 동일) ──
rs = np.random.RandomState(20260826)
perm_a = np.arange(N)
for s in np.unique(season):
    idx = np.where(season == s)[0]
    perm_a[idx] = rs.permutation(idx)
perm_b = np.arange(N)
o = np.lexsort((np.arange(N), bat))
bs = bat[o]
bnd = np.concatenate([[0], np.where(np.diff(bs) != 0)[0] + 1, [N]])
for i in range(len(bnd) - 1):
    seg = o[bnd[i]:bnd[i + 1]]
    perm_b[seg] = rs.permutation(seg)

Fa = F[perm_a]           # 시즌 안 행 단위
Fb = F[perm_b]           # batter_id 안  ← 사전 등록 귀무

# ── ④ 무작위 3열: 기존 94열에서 뽑아 복제·순열 ────────────────────
rr = np.random.RandomState(31337)
sd94 = X.std(axis=0)
cand = np.where(np.isfinite(sd94) & (sd94 > 0))[0]
rand_idx = np.sort(rr.choice(cand, size=3, replace=False))
Frand = X[:, rand_idx][perm_b].astype(np.float32)
rep["rand3_cols"] = {"index": [int(i) for i in rand_idx],
                     "name": [NUM94[int(i)] for i in rand_idx],
                     "note": "기존 94열에서 무작위 3열을 복제해 batter_id 안 순열 — "
                             "주변 분포는 실제 열, 행 연결은 절단. 개수 효과 자체의 대조."}

VARIANTS = {"form3": F, "null_bat3": Fb, "null_row3": Fa, "rand3": Frand}
CATNAMES = [f"c{j}" for j in range(5)]
EXTRA_NAMES = {"form3": SEL_NAMES, "null_bat3": SEL_NAMES,
               "null_row3": SEL_NAMES, "rand3": ["r0", "r1", "r2"]}


def make_df(extra, names):
    d = {f"n{j}": X[:, j] for j in range(94)}
    for j in range(5):
        d[f"c{j}"] = C[:, j].astype(np.int32)
    if extra is not None:
        for j, cn in enumerate(names):
            d[cn] = extra[:, j]
    return pd.DataFrame(d)


def fit_arm(tag, extra):
    names = EXTRA_NAMES[tag]
    D = make_df(extra, names)
    ptr = Pool(D[trm], y[trm], cat_features=CATNAMES)
    pva = Pool(D[vam], cat_features=CATNAMES)
    ps, times, imps = [], [], None
    for sd in SEEDS:
        t0 = time.time()
        m = CatBoostClassifier(iterations=ITERS, depth=DEPTH, learning_rate=LR,
                               loss_function="Logloss", task_type="GPU", devices="1",
                               random_seed=sd, verbose=0, thread_count=3,
                               l2_leaf_reg=3.0, bootstrap_type="Bernoulli", subsample=0.8)
        m.fit(ptr)
        ps.append(m.predict_proba(pva)[:, 1].astype(np.float64))
        times.append(round(time.time() - t0, 1))
        if sd == SEEDS[0]:
            fi = m.get_feature_importance(type="PredictionValuesChange")
            fn = list(D.columns)
            imps = {fn[k]: float(fi[k]) for k in range(len(fn))}
            imps = {"new_cols": {c: imps[c] for c in names},
                    "rank_of_new_in_all": {c: int(sorted(imps.values(), reverse=True)
                                                  .index(imps[c]) + 1) for c in names},
                    "top10": dict(sorted(imps.items(), key=lambda kv: -kv[1])[:10])}
        print(f"{tag} seed {sd} {times[-1]}s", flush=True)
    del D, ptr, pva
    P = np.mean(ps, axis=0)
    np.save(f"{B}/work/p3_{tag}.npy", P)
    per_seed = [float(100000.0 * (1 - np.mean((q - y[vam]) ** 2) / REF)) for q in ps]
    return P, {"per_seed_score": per_seed, "fit_sec": times, "importance_seed42": imps}


yv = y[vam]
pv = pit[vam]
preds, info = {}, {}

# ① base — exp_103 산출물 재사용 (입력이 비트 동일, S 규약 §stage-1 재사용)
pb = f"{B103}/work/p2_base.npy"
preds["base"] = np.load(pb)
prev = json.load(open(f"{B103}/out/s1_screen2.json", encoding="utf-8"))
info["base"] = {"per_seed_score": prev["arms"]["base"]["per_seed_score"],
                "fit_sec": prev["arms"]["base"]["fit_sec"],
                "reused_from": pb, "reused_sha16": sha(pb),
                "reuse_justification": "extra=None 이라 X·C·y·trm·vam·시드·하이퍼가 exp_103 과 "
                                       "완전히 같다. 재적합해도 같은 계산이므로 GPU 시간을 쓰지 않았다.",
                "importance_seed42": None}
assert len(preds["base"]) == int(vam.sum())

for tag, extra in VARIANTS.items():
    preds[tag], info[tag] = fit_arm(tag, extra)
for tag in preds:
    br = float(np.mean((preds[tag] - yv) ** 2))
    info[tag].update({"brier": br, "score": 100000.0 * (1 - br / REF)})
    print(tag, info[tag]["score"], flush=True)
rep["arms"] = info

# ── 짝지은 투수 블록 부트스트랩 (exp_103 과 동일 시드·동일 draws) ──
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
    vals = -(d[draws].sum(1) / blk_n[draws].sum(1)) / REF * 100000.0
    return {"delta_score": float(obs), "boot_sd": float(vals.std(ddof=1)),
            "t": float(obs / vals.std(ddof=1)), "boot_mean": float(vals.mean()),
            "frac_positive": float((vals > 0).mean())}


rep["contrasts"] = {
    "form3_minus_base": boot_t("form3", "base"),
    "form3_minus_null_bat3": boot_t("form3", "null_bat3"),
    "form3_minus_null_row3": boot_t("form3", "null_row3"),
    "form3_minus_rand3": boot_t("form3", "rand3"),
    "null_bat3_minus_base": boot_t("null_bat3", "base"),
    "null_row3_minus_base": boot_t("null_row3", "base"),
    "rand3_minus_base": boot_t("rand3", "base"),
}
c1 = rep["contrasts"]["form3_minus_base"]
c2 = rep["contrasts"]["form3_minus_null_bat3"]
rep["VERDICT"] = {
    "pass": bool(c1["delta_score"] > 0 and c1["t"] >= 2.0
                 and c2["delta_score"] > 0 and c2["t"] >= 2.0),
    "form3_minus_base": [c1["delta_score"], c1["t"]],
    "form3_minus_null_bat3": [c2["delta_score"], c2["t"]],
    "rule": "② > ① AND ② > ③, 둘 다 t >= 2.0 (exp_103 과 동일 문턱)"}
rep["detection_limit"] = {
    "boot_sd_pts": {k: v["boot_sd"] for k, v in rep["contrasts"].items()},
    "effect_needed_for_t2": {k: 2.0 * v["boot_sd"] for k, v in rep["contrasts"].items()},
    "note": "t>=2.0 이 요구하는 효과 크기. 3열로 이만한 이득이 나올 가능성은 낮다 — "
            "그래서 부호와 점추정을 문턱과 별개로 함께 읽는다."}

# ── 중복 진단: 3열 vs 기존 94열, 그리고 3열 서로 ────────────────
Xv = X[vam]
Fv = F[vam].astype(np.float64)
corr = {}
for j, cn in enumerate(SEL_NAMES):
    v = Fv[:, j]
    ok = np.isfinite(v)
    best, bj = 0.0, -1
    for k in range(94):
        w = Xv[ok, k].astype(np.float64)
        if w.std() == 0:
            continue
        r = abs(np.corrcoef(v[ok], w)[0, 1])
        if np.isfinite(r) and r > best:
            best, bj = r, k
    corr[cn] = {"max_abs_rho": float(best), "arg_col_name": NUM94[bj] if bj >= 0 else None,
                "cov_frac": float(ok.mean()),
                "sd_2024": float(np.nanstd(v)),
                "nan_frac_2024": float((~ok).mean())}
rep["corr_vs_existing94"] = corr
ok3 = np.isfinite(Fv).all(axis=1)
rep["corr_among_the_3"] = {
    f"{SEL_NAMES[i]}|{SEL_NAMES[j]}": float(np.corrcoef(Fv[ok3, i], Fv[ok3, j])[0, 1])
    for i in range(3) for j in range(i + 1, 3)}

json.dump(rep, open(f"{OUT}/s1b_screen3.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({"arms": {k: v["score"] for k, v in info.items()},
                  "contrasts": rep["contrasts"], "VERDICT": rep["VERDICT"]},
                 ensure_ascii=False, indent=1))
