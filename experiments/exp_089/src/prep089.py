"""exp_089 stage 0/2 — 딥 팔 입력 행렬 조립 (두 학습 구간).

면·프레임
  · 피처 프레임은 exp_060 `cache/frame99.parquet` (exp_048/049 stage0 리크 감사 통과 경로).
    exp_088 이 딥 팔의 99열(NUM 94 + CAT 5)이 전부 이 프레임 안에 있음을 확인했다.
  · 예측면 = season == 2024 전체 253,507행, row_id 오름차순(= train.csv 자연 순서
    = `arms/ids.npy` = OOT 납품 `row_id_oot2024.npy`). 세 정렬을 전부 assert 한다.

두 학습 구간 (🔴 변경 변인)
  A `le2024`  : 전 행 1,475,092  — 팀원 딥 팔의 배포 학습량. 복제 검증용.
                mu/sd/levels 를 **팀원 prep.json 것 그대로** 쓴다. 그래야 남는 차이가
                구조·최적화기뿐이 된다.
  B `oot23`   : (season<=2023) & ~((season<=2022) & game_type=="F") = 1,116,277
                — 파트너 OOT 납품 벡터와 **같은 학습 규칙**. mu/sd/levels 는 이 구간에서
                새로 만든다(어휘·표준화 통계가 함께 바뀌는 것이 이 구간의 변인이다).

리크 규약
  · 현재 행 TrackMan 물리량 없음. 모든 시계열 집계는 shift(1)/as-of (frame99 승계).
  · 어휘·표준화 통계는 **각 학습 구간 안에서만** 만든다. 미등장 레벨은 학습된 UNK(0).
  · 결측 수치를 sentinel 로 안 채운다 — 표준화 공간의 0(=학습 구간 평균)으로 대입한다.
    ⚠️ 팀원 딥 팔이 결측 플래그 열을 **안 쓴다**(NUM 94열이 전부). 그 규약을 그대로 따른다.
"""
import hashlib, json, os, sys, time
import numpy as np
import pandas as pd

H = "/home/<user>/SungjinPark"
B = f"{H}/exp089"
E60 = f"{H}/exp060_nn"
PKG = f"{H}/audit_deep/pkg/DEEP/mate"
W = f"{B}/work"
os.makedirs(W, exist_ok=True)
os.makedirs(f"{B}/out", exist_ok=True)
t0 = time.time()


def shaf(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


M = {"exp_id": "exp_089", "stage": "prep"}
P = json.load(open(f"{PKG}/model/deep/prep.json", encoding="utf-8"))
NUM, CAT = P["NUM"], P["CAT"]
assert len(NUM) == 94 and len(CAT) == 5

df = pd.read_parquet(f"{E60}/cache/frame99.parquet",
                     columns=list(dict.fromkeys(NUM + CAT + ["row_id", "season"])))
tr = pd.read_csv(f"{H}/data/train.csv",
                 usecols=["row_id", "season", "game_type", "control_success"])
N = len(df)
assert len(tr) == N
assert np.array_equal(df.row_id.to_numpy().astype(str),
                      tr.row_id.to_numpy().astype(str)), "frame99 ↔ train.csv 순서 불일치"

season = tr.season.to_numpy(np.int64)
gt = tr.game_type.to_numpy().astype(str)
y = tr.control_success.to_numpy(np.float32)
valid_mask = season == 2024
assert int(valid_mask.sum()) == 253507

ids_arms = np.asarray(np.load(f"{H}/arms/ids.npy", allow_pickle=True)).astype(str)
ids_oot = np.asarray(np.load(f"{H}/exp069/oot/row_id_oot2024.npy",
                             allow_pickle=True)).astype(str)
vid = tr.loc[valid_mask, "row_id"].to_numpy().astype(str)
assert np.array_equal(vid, ids_arms), "valid 순서 != arms/ids.npy"
assert np.array_equal(vid, ids_oot), "valid 순서 != OOT row_id"
M["alignment"] = {"n_rows": int(N), "n_valid": 253507,
                  "valid_eq_arms_ids": True, "valid_eq_oot_row_id": True,
                  "frame99_sha256": shaf(f"{E60}/cache/frame99.parquet"),
                  "train_csv_sha256": shaf(f"{H}/data/train.csv"),
                  "deep_prep_sha256": shaf(f"{PKG}/model/deep/prep.json"),
                  "y_valid_mean": float(y[valid_mask].mean())}

MASKS = {
    "le2024": np.ones(N, bool),
    "oot23": (season <= 2023) & ~((season <= 2022) & (gt == "F")),
}
M["splits"] = {k: {"n_train": int(v.sum())} for k, v in MASKS.items()}
assert MASKS["le2024"].sum() == 1475092, MASKS["le2024"].sum()
assert MASKS["oot23"].sum() == 1116277, MASKS["oot23"].sum()

Xraw = df[NUM].astype("float32").to_numpy()

# ── 프레임 동일성 진단: frame99 의 ≤2024 mu/sd 가 팀원 prep.json 과 같은가 ────────
mu_t = np.array(P["mu"], np.float64)
sd_t = np.array(P["sd"], np.float64)
mu_o = np.nanmean(Xraw, axis=0).astype(np.float64)
sd_o = np.nanstd(Xraw, axis=0).astype(np.float64)
rel_mu = np.abs(mu_o - mu_t) / np.maximum(np.abs(sd_t), 1e-9)
rel_sd = np.abs(sd_o - sd_t) / np.maximum(np.abs(sd_t), 1e-9)
ordm = np.argsort(-rel_mu)[:8]
M["frame_identity_vs_teammate_prep"] = {
    "n_cols": 94,
    "max_rel_mu_diff_in_sd_units": float(rel_mu.max()),
    "max_rel_sd_diff": float(rel_sd.max()),
    "median_rel_mu_diff": float(np.median(rel_mu)),
    "worst_cols": [{"col": NUM[int(i)], "rel_mu": float(rel_mu[i]),
                    "rel_sd": float(rel_sd[i]), "mu_ours": float(mu_o[i]),
                    "mu_mate": float(mu_t[i]), "sd_ours": float(sd_o[i]),
                    "sd_mate": float(sd_t[i])} for i in ordm],
    "means_nan_policy": "nanmean/nanstd (팀원 규약 추정 — 이 대조가 그 추정의 검사다)",
}

# ── 범주 어휘 동일성 진단 ──────────────────────────────────────────────
CS = {c: df[c].astype(str) for c in CAT}


def levels_of(s, mask):
    return sorted(map(str, pd.unique(s[mask])))


lv_ours_2024 = {c: levels_of(CS[c], MASKS["le2024"]) for c in CAT}
M["vocab_identity_vs_teammate_prep"] = {
    c: {"n_ours": len(lv_ours_2024[c]), "n_mate": len(P["levels"][c]),
        "identical": lv_ours_2024[c] == list(P["levels"][c])} for c in CAT}

# ── 두 구간 각각 조립 ──────────────────────────────────────────────────
for tag, trm in MASKS.items():
    if tag == "le2024":
        mu, sd = mu_t.astype(np.float32), sd_t.astype(np.float32)
        levels = {c: list(P["levels"][c]) for c in CAT}
        src = "teammate prep.json (mu/sd/levels 그대로)"
    else:
        mu = np.nanmean(Xraw[trm], axis=0).astype(np.float32)
        sd = np.nanstd(Xraw[trm], axis=0).astype(np.float32)
        sd = np.where(sd < 1e-6, np.float32(1.0), sd)
        levels = {c: levels_of(CS[c], trm) for c in CAT}
        src = "이 학습 구간에서 새로 계산 (nanmean/nanstd, sorted(str) 어휘)"
    with np.errstate(invalid="ignore"):
        X = (Xraw - mu) / sd
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    codes = np.zeros((N, len(CAT)), np.int32)
    unk = {}
    for j, c in enumerate(CAT):
        lut = {v: i + 1 for i, v in enumerate(levels[c])}
        codes[:, j] = CS[c].map(lut).fillna(0).astype(np.int64).to_numpy()
        u = codes[valid_mask, j] == 0
        unk[c] = {"n_levels": len(lut), "valid_unk_rows": int(u.sum()),
                  "valid_unk_share": float(u.mean())}
    np.save(f"{W}/X_{tag}.npy", X)
    np.save(f"{W}/codes_{tag}.npy", codes)
    json.dump({"cards": [len(levels[c]) + 1 for c in CAT], "CAT": CAT,
               "levels_n": {c: len(levels[c]) for c in CAT}},
              open(f"{W}/schema_{tag}.json", "w"), ensure_ascii=False, indent=1)
    M["splits"][tag].update({
        "mu_sd_levels_source": src,
        "cards_incl_unk": [len(levels[c]) + 1 for c in CAT],
        "valid_unknown": unk,
        "X_sha256": sha_arr(X), "codes_sha256": sha_arr(codes),
        "n_nan_filled": int(np.isnan(Xraw).sum()),
    })
    del X, codes

np.save(f"{W}/y.npy", y)
np.save(f"{W}/valid_mask.npy", valid_mask)
for tag, trm in MASKS.items():
    np.save(f"{W}/trm_{tag}.npy", trm)
M["wall_sec"] = round(time.time() - t0, 2)
json.dump(M, open(f"{B}/out/prep089.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(M, ensure_ascii=False, indent=1))
