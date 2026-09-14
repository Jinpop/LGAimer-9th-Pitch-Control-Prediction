"""exp_085 stage 1 — 학습 행렬 조립. 학습 없음, 캐시만 만든다.

면: fold_2024 (train season<=2023, F 마스크는 exp_060 규약 승계 / valid season==2024 전체).

🔴 리크 규약
  · 피처 프레임은 exp_060 `cache/frame99.parquet` 를 그대로 쓴다 (모든 시계열 집계
    shift(1)·as-of, 시즌별 <=s-1 표 재적합. exp_048/exp_049 stage 0 감사 통과 경로).
  · 현재 행 TrackMan 물리량 없음 — `p6_*` 는 투구 **전** 정보(이닝·카운트·양손·투수 id)만
    보는 LightGBM 의 구종 예측 확률이다. 원 구종 라벨은 어디에도 안 들어간다.
  · 어휘(레벨)는 **train 분할에서만** 만든다. valid/test 미등장 ID 는 학습된 <UNK> 행(0번).
  · 표준화 mu/sd 는 train 분할 nan 통계.

⚠️ 결측: 수치 sentinel 로 채우지 않는다. NaN 이 있는 열마다 **결측 플래그 열을 별도로**
   만들고(62열), 그 뒤 표준화 공간의 0(=학습 분할 평균)으로 대입한다.
   ⚠️ exp_060 은 `sp_*` 6열을 표준화 **전에** 원값 0.0 으로 채웠다 — 이 실험은 그러지
   않고 다른 열과 똑같이 플래그+평균으로 다룬다. 변인으로 기록한다.
"""
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

H = "/home/<user>/SungjinPark"
B = f"{H}/exp085"
E60 = f"{H}/exp060_nn"
SPD = f"{H}/sp_def_recv"
W = f"{B}/work"
os.makedirs(W, exist_ok=True)
t0 = time.time()

CATS5 = ["base_state", "game_type", "pitcher_team_id", "batter_team_id", "cnt_state"]
P6 = [f"p6_{t}" for t in ("curve", "cutter", "fastball", "offspeed", "sinker", "slider")]


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


M = {"exp_id": "exp_085", "stage": "prep"}

FSP = [l.strip() for l in open(f"{SPD}/features_sp.txt", encoding="utf-8") if l.strip()]
DENSE = [c for c in FSP if c not in CATS5]
assert len(DENSE) == 95, len(DENSE)

df = pd.read_parquet(f"{E60}/cache/frame99.parquet")
assert all(c in df.columns for c in CATS5 + DENSE + P6)
tr = pd.read_csv(f"{H}/data/train.csv",
                 usecols=["row_id", "season", "game_type", "pitcher_id",
                          "batter_id", "pitcher_hand", "batter_hand",
                          "control_success"])
N = len(df)
assert len(tr) == N

# ── 정렬 ───────────────────────────────────────────────────────────────
assert np.array_equal(df.row_id.to_numpy().astype(str),
                      tr.row_id.to_numpy().astype(str)), "frame99 ↔ train.csv 행 순서 불일치"
assert np.array_equal(df.control_success.to_numpy(np.int64),
                      tr.control_success.to_numpy(np.int64)), "라벨 불일치"

season = df.season.to_numpy(np.int64)
gt = df.game_type.to_numpy().astype(str)
train_mask = (season <= 2023) & ~((season <= 2022) & (gt == "F"))
valid_mask = season == 2024
n_tr, n_va = int(train_mask.sum()), int(valid_mask.sum())
assert n_va == 253507, n_va

vid = tr.loc[valid_mask, "row_id"].to_numpy().astype(str)
arms_ids = np.asarray(np.load(f"{H}/arms/ids.npy", allow_pickle=True)).astype(str)
assert np.array_equal(vid, arms_ids), "valid 행 순서가 arms/ids.npy 와 다르다"

y = tr.control_success.to_numpy(np.float32)
M["alignment"] = {
    "n_rows": int(N), "n_train": n_tr, "n_valid": n_va,
    "n_masked_out": int(N - n_tr - n_va),
    "train_mask_rule": "season<=2023 & ~((season<=2022)&(game_type=='F'))  (exp_060 승계)",
    "valid_order_eq_arms_ids": True,
    "valid_order_eq_train_csv_natural_2024": True,
    "y_valid_sha256": sha_arr(y[valid_mask].astype(np.float64)),
    "y_valid_mean": float(y[valid_mask].mean()),
    "train_csv_sha256": sha_file(f"{H}/data/train.csv"),
    "frame99_sha256": sha_file(f"{E60}/cache/frame99.parquet"),
    "features_sp_sha256": sha_file(f"{SPD}/features_sp.txt"),
}

# ── dense + 결측 플래그 ────────────────────────────────────────────────
X = df[DENSE].astype("float32").to_numpy()
nanmask = np.isnan(X)
flag_cols = [c for j, c in enumerate(DENSE) if nanmask[:, j].any()]
flag_idx = [j for j, c in enumerate(DENSE) if nanmask[:, j].any()]
F = nanmask[:, flag_idx].astype(np.float32)

mu = np.nanmean(X[train_mask], axis=0).astype(np.float32)
sd = np.nanstd(X[train_mask], axis=0).astype(np.float32)
sd[sd < 1e-6] = 1.0
X = (X - mu) / sd
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
Xd = np.concatenate([X, F], axis=1)
del X, F, nanmask

M["dense"] = {"n_v6_cols": len(DENSE), "n_missing_flags": len(flag_cols),
              "width": int(Xd.shape[1]),
              "missing_flag_cols": flag_cols,
              "standardize": "train 분할 nanmean/nanstd, sd<1e-6→1.0, 이후 nan_to_num(0)",
              "note": "sentinel 대입 아님 — 결측 정보는 플래그 62열이 따로 싣는다"}

# ── 범주 코드 (레벨은 train 분할에서만, 미등장=0=<UNK>) ──────────────────
CATSPEC = []          # (이름, 시리즈)
for c in CATS5:
    CATSPEC.append((c, df[c].astype(str)))
for c in ["pitcher_id", "batter_id", "pitcher_hand", "batter_hand"]:
    CATSPEC.append((c, tr[c].astype(str)))

p6 = df[P6].to_numpy(np.float64)
p6_allnan = np.isnan(p6).all(axis=1)
dom = np.where(p6_allnan, -1, np.nanargmax(np.nan_to_num(p6, nan=-1.0), axis=1))
CATSPEC.append(("dom6", pd.Series(np.where(dom < 0, "NA", dom.astype(str)), index=df.index)))

codes = np.zeros((N, len(CATSPEC)), dtype=np.int64)
levels, nlev, unseen = {}, [], {}
for i, (name, s) in enumerate(CATSPEC):
    try:
        lv = sorted(pd.unique(s[train_mask]), key=lambda x: int(x))
    except (ValueError, TypeError):
        lv = sorted(map(str, pd.unique(s[train_mask])))
    lv = [str(v) for v in lv]
    lut = {v: j + 1 for j, v in enumerate(lv)}
    codes[:, i] = s.map(lut).fillna(0).astype(np.int64).to_numpy()
    levels[name] = lv
    nlev.append(len(lv))
    u = codes[valid_mask, i] == 0
    unseen[name] = {"n_levels_train": len(lv),
                    "valid_unknown_rows": int(u.sum()),
                    "valid_unknown_share": float(u.mean()),
                    "valid_unknown_distinct": int(pd.unique(s[valid_mask][u]).size)}

M["categorical"] = {"fields": [n for n, _ in CATSPEC], "n_levels": nlev,
                    "unknown_index": 0,
                    "unknown_policy": "train 분할 미등장 레벨 → 학습된 <UNK> 행 0번",
                    "valid_unknown": unseen,
                    "dom6_def": "argmax(p6_*), 동점은 열 순서, 전부 NaN 이면 'NA' 레벨. "
                                "p6 는 투구 전 정보만 보는 구종 예측 — 현재 행 TrackMan 아님",
                    "p6_allnan_rows": {"total": int(p6_allnan.sum()),
                                       "train": int(p6_allnan[train_mask].sum()),
                                       "valid": int(p6_allnan[valid_mask].sum())}}

np.save(f"{W}/Xd.npy", Xd)
np.save(f"{W}/codes.npy", codes.astype(np.int32))
np.save(f"{W}/y.npy", y)
np.save(f"{W}/train_mask.npy", train_mask)
np.save(f"{W}/valid_mask.npy", valid_mask)
np.save(f"{W}/gt.npy", gt)
json.dump({"levels_n": {k: len(v) for k, v in levels.items()},
           "fields": [n for n, _ in CATSPEC],
           "dense_cols": DENSE, "flag_cols": flag_cols},
          open(f"{W}/schema.json", "w"), ensure_ascii=False, indent=1)

M["cache_sha256"] = {"Xd": sha_arr(Xd), "codes": sha_arr(codes.astype(np.int32)),
                     "y": sha_arr(y)}
M["wall_sec"] = round(time.time() - t0, 2)
json.dump(M, open(f"{B}/out/prep.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in M.items() if k != "dense"},
                 ensure_ascii=False, indent=1))
print("dense width", Xd.shape, "flags", len(flag_cols), "wall", M["wall_sec"])
