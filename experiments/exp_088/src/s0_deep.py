"""exp_088 stage 0b — 팀원 딥 팔의 로짓을 우리 면(fold_2024 valid 253,507행)에서 뽑는다.

⭐ 발견: 딥 팔의 입력 99열(NUM 94 + CAT 5)은 `sp_def_recv/features_sp.txt` 와 **완전히
   같은 집합**이고, 그 99열이 전부 `exp060_nn/cache/frame99.parquet` 안에 있다.
   → 파트너 파이프라인을 다시 돌릴 필요 없이 frame99 행에 `predict_deep` 을 직접 태울 수 있다.
   이 경로(route1)는 **우리 팔이 사는 바로 그 면**에서 딥 로짓을 준다.

route2(대조): 감사 세션이 이미 계산한 cov 프레임 딥 확률(`audit_deep/runs/dbg2__cov.deepdbg.csv`)을
   row_id 로 매핑. cov 는 2024 행 245,789개의 season 을 2025 로 덮은 대역 프레임이라
   시즌-내 폼 계열이 축퇴한다(S32). 두 경로의 차이가 그 축퇴의 크기다.

🔴 audit_deep 은 **읽기만** 한다. 쓰기는 ~/SungjinPark/exp088/ 아래에만.
"""
import hashlib, json, os, sys, time
import numpy as np
import pandas as pd

H = "/home/<user>/SungjinPark"
B = f"{H}/exp088"
PKG = f"{H}/audit_deep/pkg/DEEP/mate"
sys.path.insert(0, PKG)
import deep_infer as DI            # noqa: E402  (읽기 전용 import)

PREP = f"{PKG}/model/deep/prep.json"
SEEDS_DEEP = [42, 777, 2024, 101, 7]        # 배포 5시드
SEEDS_OURS = [42, 7, 2024, 101]             # exp_086 4시드
t0 = time.time()
os.makedirs(f"{B}/out/preds", exist_ok=True)


def shaf(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


R = {"exp_id": "exp_088", "stage": "s0b_deep_logits"}
P = json.load(open(PREP, encoding="utf-8"))
NUM, CAT = P["NUM"], P["CAT"]
R["deep_input_spec"] = {"n_num": len(NUM), "n_cat": len(CAT),
                        "prep_sha256": shaf(PREP),
                        "npz_sha256": {str(s): shaf(f"{PKG}/model/deep/deepf_s{s}.npz")
                                       for s in SEEDS_DEEP}}

# ── 우리 면 ────────────────────────────────────────────────────────────
valid_mask = np.load(f"{H}/exp085/work/valid_mask.npy")
df = pd.read_parquet(f"{H}/exp060_nn/cache/frame99.parquet",
                     columns=list(dict.fromkeys(NUM + CAT + ["row_id", "season"])))
assert len(df) == len(valid_mask)
dv = df.loc[valid_mask].reset_index(drop=True)
n = len(dv)
assert n == 253507, n
ids_arms = np.asarray(np.load(f"{H}/arms/ids.npy", allow_pickle=True)).astype(str)
assert np.array_equal(dv.row_id.to_numpy().astype(str), ids_arms), "면 정렬 불일치"
R["surface"] = {"name": "fold_2024 valid_all (train<=2023 -> valid 2024)",
                "n_rows": n, "aligned_to": "arms/ids.npy",
                "frame99_sha256": shaf(f"{H}/exp060_nn/cache/frame99.parquet")}

# ── 딥 팔 전방 통과 (시드별 로짓) ────────────────────────────────────────
mu = np.array(P["mu"], np.float32)
sd = np.array(P["sd"], np.float32)
codes = np.zeros((n, len(CAT)), np.int64)
unk = {}
for i, c in enumerate(CAT):
    lut = {v: j + 1 for j, v in enumerate(P["levels"][c])}
    s = dv[c].astype(str)
    codes[:, i] = s.map(lut).fillna(0).astype(np.int64).values
    unk[c] = {"n_levels": len(lut), "unk_rows": int((codes[:, i] == 0).sum()),
              "unk_share": float((codes[:, i] == 0).mean())}
R["deep_cat_unknown_on_our_surface"] = unk

X = dv[NUM].astype("float32").values
with np.errstate(invalid="ignore"):
    X = (X - mu) / sd
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

Z = {}
for s in SEEDS_DEEP:
    z = DI._one(np.load(f"{PKG}/model/deep/deepf_s{s}.npz"), codes, X)
    Z[s] = z
    np.save(f"{B}/out/preds/deep_route1_s{s}_logit.npy", z)
z_deep5 = np.mean([Z[s] for s in SEEDS_DEEP], axis=0)
z_deep4 = np.mean([Z[s] for s in SEEDS_OURS], axis=0)
np.save(f"{B}/out/preds/deep_route1_logit_seed5.npy", z_deep5)
np.save(f"{B}/out/preds/deep_route1_logit_seed4.npy", z_deep4)
R["deep_route1"] = {
    "seeds5": SEEDS_DEEP, "seeds4_matched_to_ours": SEEDS_OURS,
    "logit_stats_seed5": {"min": float(z_deep5.min()), "max": float(z_deep5.max()),
                          "mean": float(z_deep5.mean()), "sd": float(z_deep5.std())},
    "per_seed_sd": {str(s): float(Z[s].std()) for s in SEEDS_DEEP},
    "sha256_seed5": sha_arr(z_deep5),
}

# ── route2 대조: cov 프레임 (감사 세션 산출물, 읽기만) ─────────────────────
DBG = f"{H}/audit_deep/runs/dbg2__cov.deepdbg.csv"
dbg = pd.read_csv(DBG, usecols=["row_id", "p_deep"])
pos = pd.Index(ids_arms).get_indexer(dbg.row_id.to_numpy().astype(str))
assert (pos >= 0).all()
p_cov = dbg.p_deep.to_numpy(np.float64)
z_cov = np.log(p_cov) - np.log1p(-p_cov)
z_r1_sub = z_deep5[pos]
d = z_cov - z_r1_sub
R["route2_cov_crosscheck"] = {
    "n_matched": int(len(pos)),
    "deepdbg_sha256": shaf(DBG),
    "pearson_r_route1_vs_route2": float(np.corrcoef(z_r1_sub, z_cov)[0, 1]),
    "max_abs_logit_diff": float(np.abs(d).max()),
    "mean_abs_logit_diff": float(np.abs(d).mean()),
    "sd_route1_sub": float(z_r1_sub.std()), "sd_route2": float(z_cov.std()),
    "note": ("route2 는 cov 대역 프레임(season 2025 덮어쓰기)에서 배포 파이프라인이 "
             "조립한 입력으로 계산됐고, route1 은 frame99 의 정직한 2024 값이다. "
             "차이는 그 조립 차이(시즌-내 폼 축퇴 포함)의 크기다"),
}
np.save(f"{B}/work/cov_pos_in_arms.npy", pos.astype(np.int64))
np.save(f"{B}/out/preds/deep_route2_cov_logit.npy", z_cov)

# ── 시즌-내 폼 축퇴 진단 ────────────────────────────────────────────────
cov_raw = pd.read_csv(f"{H}/build_CA/frames/cov/test.csv",
                      usecols=["row_id"], dtype=str)
assert np.array_equal(cov_raw.row_id.to_numpy().astype(str),
                      dbg.row_id.to_numpy().astype(str))
INS = [c for c in NUM if c.startswith("ins_")]
ours_ins = dv.loc[pos, INS]
R["inseason_on_our_surface"] = {
    c: {"n_zero": int((ours_ins[c].fillna(-1) == 0).sum()),
        "n_nan": int(ours_ins[c].isna().sum()),
        "sd": float(np.nanstd(ours_ins[c].to_numpy(np.float64)))}
    for c in INS[:4]}
R["inseason_cols"] = INS

R["wall_sec"] = round(time.time() - t0, 2)
json.dump(R, open(f"{B}/out/s0b_deep.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1))
