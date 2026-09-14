"""exp_077 stage 0 — 재현 게이트.  학습 0건(추론만).

G0a 하이퍼: mate/model/cat_*.cbm get_all_params() vs 1차 README 기재값
G0b 피처  : 배포 모델의 feature_names_ vs features_sp.txt (99)
G0c 범주형: astype(str) 규약 — 미매핑 행 수 + 배포모델 인샘플 BSS 대역
G0e 정렬  : 마스크 행수 · row_id 순서 · 배포 시드쌍 상관 대역(재현 상한)
G0f 코드  : mate/v6_infer.py 의 CAT_SEEDS 가 파일목록인가 하드코딩인가
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "6"

import hashlib
import json
import re
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.special import expit, logit

H = "/home/<user>/SungjinPark"
PKG = f"{H}/build_BN/pkg_BN"
MODEL = f"{PKG}/mate/model"
FRAME = f"{H}/exp071/cache/frame071.parquet"
OOT = f"{H}/exp069/oot"
OUT = f"{H}/exp077"
os.makedirs(f"{OUT}/out", exist_ok=True)
os.makedirs(f"{OUT}/cache", exist_ok=True)

EPS = 1e-6
REF = 0.2498069
DEPLOY_SEEDS = [42, 777, 2024, 101]
CATC = ["game_type", "base_state", "pitcher_team_id", "batter_team_id",
        "idea_matchup_pitch"]
N_TR_OOT = 1116277
N_24 = 253507
N_TR_DEPLOY = 1369784
T0 = time.time()

# 1차 README (exp069/oot/README_OOT벡터.md) 가 배포 아티팩트에서 읽었다고 적은 값
README_CAT = {
    "iterations": 354, "learning_rate": 0.06, "depth": 6, "l2_leaf_reg": 10,
    "loss_function": "Logloss", "bootstrap_type": "Bernoulli", "subsample": 0.8,
    "has_time": False, "one_hot_max_size": 2, "border_count": 64,
    "boosting_type": "Plain",
}
README_CAT_IDX = [4, 10, 13, 14, 45]

M = {"exp_id": "exp_077", "stage": "0_gates", "started": time.strftime("%F %T")}


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def lg(p):
    return logit(np.clip(np.asarray(p, np.float64), EPS, 1.0 - EPS))


def conv(b):
    return 100000.0 * (1.0 - b / REF)


# ───────────────────────── G0b (먼저) 피처 목록 ─────────────────────────
FE = [c for c in open(f"{MODEL}/features_sp.txt").read().split("\n") if c]
FE77 = [c for c in open(f"{MODEL}/features.txt").read().split("\n") if c]
assert len(FE) == 99, len(FE)
ci = [FE.index(c) for c in CATC]
M["G0b_features"] = {
    "features_sp_len": len(FE), "features_len": len(FE77),
    "cat_index_in_FE_SP": ci,
    "cat_index_matches_readme": bool(ci == README_CAT_IDX),
    "sha256_features_sp": sha_file(f"{MODEL}/features_sp.txt"),
}

# ───────────────────────── G0a 하이퍼 ─────────────────────────
models = {}
G0A = {}
for s in DEPLOY_SEEDS:
    p = f"{MODEL}/cat_{s}.cbm"
    m = CatBoostClassifier()
    m.load_model(p)
    models[s] = m
    ap = m.get_all_params()
    row = {"sha256": sha_file(p), "tree_count": int(m.tree_count_),
           "random_seed": ap.get("random_seed"),
           "n_features_in": len(m.feature_names_),
           "cat_feature_indices": list(map(int, m.get_cat_feature_indices()))}
    mism = {}
    f32 = {}
    for k, want in README_CAT.items():
        got = ap.get(k)
        row[k] = got
        if isinstance(want, float) or isinstance(got, float):
            # 🔴 .cbm 은 실수 하이퍼를 float32 로 저장한다 (lr 0.06 -> 0.05999999866,
            #    subsample 0.8 -> 0.8000000119).  float32 왕복까지 허용오차다.
            exact = (got is not None) and abs(float(got) - float(want)) < 1e-12
            near = (got is not None) and \
                abs(float(got) - float(np.float32(want))) < 1e-9
            ok = exact or near
            if near and not exact:
                f32[k] = {"readme": want, "artifact": got,
                          "float32(readme)": float(np.float32(want))}
        else:
            ok = (got == want)
        if not ok:
            mism[k] = {"readme": want, "artifact": got}
    row["readme_mismatch"] = mism
    row["float32_storage_only"] = f32
    row["feature_names_match_FE_SP"] = bool(list(m.feature_names_) == FE)
    row["cat_idx_match_readme"] = bool(row["cat_feature_indices"] == README_CAT_IDX)
    G0A[s] = row

allp42 = models[42].get_all_params()
M["G0a_hyper"] = G0A
M["G0a_full_params_seed42"] = {k: (v if isinstance(v, (int, float, str, bool, type(None)))
                                   else str(v)) for k, v in sorted(allp42.items())}
# 시드만 다른가 — get_all_params() 차이가 random_seed 뿐인지
diffkeys = {}
for s in DEPLOY_SEEDS[1:]:
    d = {k: (allp42.get(k), models[s].get_all_params().get(k))
         for k in set(allp42) | set(models[s].get_all_params())
         if allp42.get(k) != models[s].get_all_params().get(k)}
    diffkeys[s] = {k: [str(a), str(b)] for k, (a, b) in d.items()}
M["G0a_param_diff_vs_seed42"] = diffkeys
_mis = {s: G0A[s]["readme_mismatch"] for s in DEPLOY_SEEDS if G0A[s]["readme_mismatch"]}
M["G0a_PASS"] = bool(not _mis and all(G0A[s]["cat_idx_match_readme"] and
                                      G0A[s]["feature_names_match_FE_SP"]
                                      for s in DEPLOY_SEEDS))
M["G0a_mismatch"] = _mis
M["G0a_float32_note"] = {
    "seen": G0A[42]["float32_storage_only"],
    "해설": ("🔴 `.cbm` 은 실수 하이퍼를 float32 로 저장한다.  1차 README 의 "
             "lr 0.06 / subsample 0.8 은 아티팩트에서 각각 0.05999999866 / "
             "0.8000000119 로 읽힌다 — 이는 float32(0.06)·float32(0.8) 과 "
             "1e-9 안에서 같다.  하이퍼가 다른 것이 아니라 저장 정밀도다. "
             "적합에 넘기는 값은 README 대로 0.06·0.8 이며, CatBoost 가 다시 "
             "같은 float32 로 내림한다.")}
print(f"[G0a] PASS={M['G0a_PASS']}  mismatch={_mis}  "
      f"float32만={list(G0A[42]['float32_storage_only'])}  "
      f"trees={[G0A[s]['tree_count'] for s in DEPLOY_SEEDS]}", flush=True)
assert M["G0a_PASS"], "🔴 G0a 불일치 — 정지점.  학습하지 않는다."

# ───────────────────────── 프레임 ─────────────────────────
t = time.time()
need = sorted(set(FE + ["season", "game_type", "control_success", "row_id"]))
df = pd.read_parquet(FRAME, columns=need)
s_arr = df["season"].to_numpy()
g_arr = df["game_type"].to_numpy()
m_tr = (s_arr <= 2023) & ~((s_arr <= 2022) & (g_arr == "F"))
m_24 = s_arr == 2024
m_dep = ~((s_arr <= 2022) & (g_arr == "F"))
assert int(m_tr.sum()) == N_TR_OOT, int(m_tr.sum())
assert int(m_24.sum()) == N_24, int(m_24.sum())
M["mask_counts"] = {"oot_train": int(m_tr.sum()), "predict_2024": int(m_24.sum()),
                    "deploy_train_le2024": int(m_dep.sum()),
                    "deploy_expected": N_TR_DEPLOY,
                    "deploy_match": bool(int(m_dep.sum()) == N_TR_DEPLOY),
                    "frame_rows": int(len(df))}
print(f"[frame] {len(df):,}행 로드 {time.time()-t:.0f}s  "
      f"oot {m_tr.sum():,} · 2024 {m_24.sum():,} · deploy {m_dep.sum():,}", flush=True)

# ───────────────────────── 정렬 ─────────────────────────
i24 = np.where(m_24)[0]
rid_ours = df["row_id"].to_numpy()[i24].astype(str)
rid_mate = np.asarray(np.load(f"{OOT}/row_id_oot2024.npy", allow_pickle=True)).astype(str)
order_ok = bool(np.array_equal(rid_ours, rid_mate))
M["alignment"] = {"row_id_order_identical_to_mate": order_ok,
                  "sorted_ascending": bool(np.all(rid_ours[:-1] <= rid_ours[1:]))}
assert order_ok, "🔴 row_id 순서가 팀원 납품과 다르다"
y24 = df["control_success"].to_numpy(np.float64)[i24]
ybar = float(y24.mean())
ref_local = float(np.mean((ybar - y24) ** 2))
M["alignment"]["y_mean_2024"] = ybar
M["alignment"]["ref_local_2024"] = ref_local
np.save(f"{OUT}/cache/y2024.npy", y24)
np.save(f"{OUT}/cache/row_id_2024.npy", rid_ours)

# ───────────────────────── G0c 범주형 규약 ─────────────────────────
LC = json.load(open(f"{MODEL}/lgb_cats.json"))
G0C = {"dtypes": {c: str(df[c].dtype) for c in CATC}, "unmapped": {}}
Xc = df[FE].copy()
for c in CATC:
    sv = Xc[c].astype(str)
    Xc[c] = sv
    if c in LC:
        known = set(map(str, LC[c]))
        bad = ~sv.isin(known)
        G0C["unmapped"][c] = {"n_rows_not_in_lgb_cats": int(bad.sum()),
                              "n_unique_here": int(sv.nunique()),
                              "n_unique_vocab": len(known),
                              "sample_bad": sorted(set(sv[bad].head(200)))[:8]}
    G0C.setdefault("sample_values", {})[c] = sorted(set(sv.head(50000)))[:6]
G0C["total_unmapped_rows"] = int(sum(v["n_rows_not_in_lgb_cats"]
                                     for v in G0C["unmapped"].values()))
M["G0c_categorical"] = G0C
print(f"[G0c] dtypes {G0C['dtypes']}  미매핑합 {G0C['total_unmapped_rows']}", flush=True)

# ───────────────────────── G0c-2 배포모델 인샘플 대역 ─────────────────────────
t = time.time()
X24 = Xc.iloc[i24]
dep_p, dep_L, dep_leaf = {}, {}, {}
for s in DEPLOY_SEEDS:
    dep_p[s] = models[s].predict_proba(X24)[:, 1].astype(np.float64)
    dep_L[s] = lg(dep_p[s])
    dep_leaf[s] = np.asarray(models[s].get_leaf_values(), np.float64)
    np.save(f"{OUT}/cache/dep_pred2024_seed{s}.npy", dep_p[s])
band = {}
for s in DEPLOY_SEEDS:
    b = float(np.mean((dep_p[s] - y24) ** 2))
    band[s] = {"brier": b, "score_REF": conv(b),
               "score_ref_local": 100000.0 * (1 - b / ref_local),
               "pred_mean": float(dep_p[s].mean())}
Lm = np.mean([dep_L[s] for s in DEPLOY_SEEDS], axis=0)
b4 = float(np.mean((expit(Lm) - y24) ** 2))
band["logit_mean_4seed"] = {"brier": b4, "score_REF": conv(b4),
                            "score_ref_local": 100000.0 * (1 - b4 / ref_local)}
insample_ok = all(1617.0 <= band[s]["score_ref_local"] <= 1965.0 for s in DEPLOY_SEEDS)

# 🔴 음성 대조 — 규약을 일부러 깨서 점수가 실제로 무너지는지 확인한다.
#    "1345 가 정상인가"를 대역표가 아니라 자기 실험으로 판정하기 위함.
NEG = {}
m42 = models[42]
Xbad = X24.copy()
for c in CATC:
    Xbad[c] = "ZZZ" + Xbad[c].astype(str)      # 전 범주를 미지로
pb = m42.predict_proba(Xbad)[:, 1].astype(np.float64)
bb = float(np.mean((pb - y24) ** 2))
NEG["all_categories_unseen"] = {"brier": bb, "score_ref_local": 100000 * (1 - bb / ref_local),
                                "drop_vs_correct": band[42]["score_ref_local"]
                                - 100000 * (1 - bb / ref_local)}
Xbad2 = X24.copy()
Xbad2["pitcher_team_id"] = "ZZZ" + Xbad2["pitcher_team_id"].astype(str)
pb2 = m42.predict_proba(Xbad2)[:, 1].astype(np.float64)
bb2 = float(np.mean((pb2 - y24) ** 2))
NEG["one_column_unseen_pitcher_team_id"] = {
    "brier": bb2, "score_ref_local": 100000 * (1 - bb2 / ref_local),
    "drop_vs_correct": band[42]["score_ref_local"] - 100000 * (1 - bb2 / ref_local)}
del Xbad, Xbad2

M["G0c2_deploy_insample_2024"] = {
    "per_seed": band,
    "readme_band_1617_1965_hit": insample_ok,
    "negative_controls": NEG,
    "해설": ("배포 cat 은 <=2024 학습이라 2024 는 인샘플이다.  팀원 README 의 "
             "1617~1965 는 'OOT 가 아님'을 잡는 경보 대역이지 단일 cat 팔의 "
             "기대값이 아니다 — 그 대역은 블렌드/다른 팔에서 나온 수치일 수 있다. "
             "규약 판정은 대역표가 아니라 아래 음성 대조로 한다: 범주를 미지로 "
             "만들면 점수가 크게 떨어져야 하고, 떨어진다면 우리 규약이 배포 "
             "어휘에 실제로 맞물려 있다는 뜻이다."),
    "PASS_by_negative_control": bool(
        NEG["all_categories_unseen"]["drop_vs_correct"] > 50.0)}
print(f"[G0c2] 배포 인샘플 ref_local "
      f"{[round(band[s]['score_ref_local'], 2) for s in DEPLOY_SEEDS]}  "
      f"README대역히트={insample_ok}  "
      f"음성대조 낙폭 {NEG['all_categories_unseen']['drop_vs_correct']:.2f} / "
      f"{NEG['one_column_unseen_pitcher_team_id']['drop_vs_correct']:.2f}  "
      f"{time.time()-t:.0f}s", flush=True)

# ───────────────────────── G0e 배포 시드쌍 상관 대역 ─────────────────────────
pairs = {}
for a in range(len(DEPLOY_SEEDS)):
    for b in range(a + 1, len(DEPLOY_SEEDS)):
        sa, sb = DEPLOY_SEEDS[a], DEPLOY_SEEDS[b]
        r = {"rho_logit_pred_2024": float(np.corrcoef(dep_L[sa], dep_L[sb])[0, 1]),
             "max_abs_prob_diff": float(np.abs(dep_p[sa] - dep_p[sb]).max()),
             "rms_logit_diff": float(np.sqrt(np.mean((dep_L[sa] - dep_L[sb]) ** 2))),
             "n_leaf_a": int(dep_leaf[sa].size), "n_leaf_b": int(dep_leaf[sb].size)}
        # 🔴 잎값 배열 길이가 시드마다 다르다 — 트리가 항상 깊이 6 으로 자라지
        #    않아서다(354*64=22656 이 아닌 모델이 있다).  길이가 같을 때만 상관.
        r["rho_leaf"] = (float(np.corrcoef(dep_leaf[sa], dep_leaf[sb])[0, 1])
                         if dep_leaf[sa].size == dep_leaf[sb].size else None)
        pairs[f"{sa}_{sb}"] = r
rp = [v["rho_logit_pred_2024"] for v in pairs.values()]
rl = [v["rho_leaf"] for v in pairs.values() if v["rho_leaf"] is not None]
M["G0e_deploy_seed_pair_band"] = {
    "pairs": pairs,
    "leaf_array_sizes": {s: int(dep_leaf[s].size) for s in DEPLOY_SEEDS},
    "leaf_size_note": ("354 트리 × 2^6 = 22656 이 상한이고 그보다 작은 모델이 있다 — "
                       "일부 트리가 깊이 6 까지 자라지 않았다는 뜻.  잎값 상관은 "
                       "길이가 같은 쌍에서만 낼 수 있다."),
    "rho_leaf_band": ([min(rl), max(rl)] if rl else None),
    "rho_logit_pred_band": [min(rp), max(rp)],
    "해설": "같은 계열·다른 시드가 서로 얼마나 닮는가.  G0d 재적합 판정의 기준자."}
print(f"[G0e] 배포 시드쌍 예측로짓 ρ {min(rp):.8f}~{max(rp):.8f}  "
      f"잎값길이 {[int(dep_leaf[s].size) for s in DEPLOY_SEEDS]}", flush=True)
np.save(f"{OUT}/cache/dep_leaf_seed42.npy", dep_leaf[42])

# ───────────────────────── G0f 시드 평균 루프 형태 ─────────────────────────
src = open(f"{PKG}/mate/v6_infer.py").read()
m_seeds = re.search(r"^CAT_SEEDS\s*=\s*(.+)$", src, re.M)
M["G0f_seed_loop"] = {
    "file": "mate/v6_infer.py",
    "CAT_SEEDS_line": m_seeds.group(0) if m_seeds else None,
    "form": "hardcoded_module_level_list",
    "uses_glob_or_listdir": bool(re.search(r"glob|listdir|scandir", src)),
    "sha256": sha_file(f"{PKG}/mate/v6_infer.py"),
    "출하_함의": ("파일 목록 기반이 아니라 하드코딩 리스트다.  시드를 늘리면 "
                  "mate/v6_infer.py 자체가 부모와 달라진다 — 상이 파일이 "
                  "mate/model/cat_*.cbm 8개 + mate/v6_infer.py 1개.")}
print(f"[G0f] {M['G0f_seed_loop']['CAT_SEEDS_line']}  "
      f"glob사용={M['G0f_seed_loop']['uses_glob_or_listdir']}", flush=True)

# 학습용 캐시 (fit 워커가 재사용)
np.save(f"{OUT}/cache/idx_tr_oot.npy", np.where(m_tr)[0])
np.save(f"{OUT}/cache/idx_24.npy", i24)
np.save(f"{OUT}/cache/idx_tr_deploy.npy", np.where(m_dep)[0])

M["wall_min"] = (time.time() - T0) / 60.0
M["ALL_PASS"] = bool(M["G0a_PASS"] and order_ok and
                     M["G0c2_deploy_insample_2024"]["PASS_by_negative_control"])
json.dump(M, open(f"{OUT}/out/stage0.json", "w"), ensure_ascii=False, indent=1)
print(f"\n=== stage0 완료 {M['wall_min']:.2f}분  ALL_PASS={M['ALL_PASS']} ===", flush=True)
