"""exp_093 stage 0 — AA 라벨 분해 4셀을 5분류 소프트 타깃으로 조립.

입력  : `build_CG/kit/qlab.parquet` (1,475,092행 · `valid`·`is_mid_only`·`is_rev_only`·
        `is_both`·`is_wayoff_true`) — 🔴 **학습 전용 자산**
출력  : `exp093/work/T5_oot23.npy` (N, 5) float32
        클래스 순서 [is_mid_only, is_rev_only, is_both, is_wayoff_true, success]
        🔴 **valid(2024) 행은 NaN 으로 채운다** — 학습 코드가 실수로 읽으면 즉시 터지게.

라벨 규약 (사전 등록 · 결과 보고 안 바꾼다)
  · y==1                      → 클래스 4 원핫
  · y==0 이고 셀 합 == 1      → 그 셀 원핫
  · y==0 이고 셀 합 == 0      → 4실패셀에 **균등 0.25** (실측 329행 = 학습구간의 0.02%)
  🔑 그래서 **모든 행에서 주변 성공확률 타깃이 정확히 y 다** — BCE 와 같은 정보를 주고
     실패의 *유형* 만 추가로 준다. 이게 이 실험의 단일 변인이 되게 하는 규약이다.
  ⚠️ `valid` 열은 **타깃 조립에 안 쓴다**(행을 안 버린다). 버리면 n_train 이 달라져
     기준선과의 짝짓기가 깨진다. valid==0 인 792행의 내역은 게이트에 적는다.

리크
  · 셀 라벨은 그 행 자신의 결과에서 나온 값이다 = `control_success` 와 같은 성격.
    학습 행(season<=2023)에만 쓰고 예측면(2024)에는 쓰지 않는다 — NaN 채움이 그 기계 증거.
"""
import hashlib, json, os
import numpy as np
import pandas as pd

H = "/home/<user>/SungjinPark"
B, W89 = f"{H}/exp093", f"{H}/exp089/work"
W, OUT = f"{B}/work", f"{B}/out"
os.makedirs(W, exist_ok=True); os.makedirs(OUT, exist_ok=True)
QP = f"{H}/build_CG/kit/qlab.parquet"
CELLS = ["is_mid_only", "is_rev_only", "is_both", "is_wayoff_true"]


def shaf(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


tr = pd.read_csv(f"{H}/data/train.csv",
                 usecols=["row_id", "season", "game_type", "control_success"])
q = pd.read_parquet(QP)
N = len(tr)
assert len(q) == N, (len(q), N)
y = tr.control_success.to_numpy(np.int64)
C = q[CELLS].to_numpy(np.int64)
vflag = q["valid"].to_numpy(np.int64)
s = C.sum(1)

# ── 정렬·구조 게이트 ────────────────────────────────────────────────────
G = {
    "qlab_sha256": shaf(QP), "train_csv_sha256": shaf(f"{H}/data/train.csv"),
    "n_rows": int(N),
    "cells_mutually_exclusive": bool(s.max() <= 1),
    "cellsum1_all_failure": bool(((s == 1) & (y == 1)).sum() == 0),
    "cellsum0_success_rows": int(((s == 0) & (y == 1)).sum()),
    "cellsum0_failure_rows_UNRESOLVED": int(((s == 0) & (y == 0)).sum()),
    "n_failures": int((y == 0).sum()), "n_success": int((y == 1).sum()),
    "cell_counts": {c: int(C[:, j].sum()) for j, c in enumerate(CELLS)},
    "qlab_valid_flag": {"n_valid0": int((vflag == 0).sum()),
                        "valid0_and_y1": int(((vflag == 0) & (y == 1)).sum()),
                        "valid0_and_y0": int(((vflag == 0) & (y == 0)).sum()),
                        "policy": "🔴 행을 안 버린다 — n_train 을 기준선과 같게 유지"},
    "alignment_evidence": ("셀 합==1 인 702,160행이 **전부** y==0 이고 y==1 행에는 셀이 "
                           "하나도 없다 → qlab 의 행 순서가 train.csv 와 같다는 기계 증거"),
}
assert G["cells_mutually_exclusive"] and G["cellsum1_all_failure"]
assert G["cellsum0_success_rows"] == int((y == 1).sum())

# ── 5분류 소프트 타깃 ──────────────────────────────────────────────────
T = np.zeros((N, 5), np.float32)
T[y == 1, 4] = 1.0
resolved = (y == 0) & (s == 1)
T[resolved, :4] = C[resolved].astype(np.float32)
unres = (y == 0) & (s == 0)
T[unres, :4] = 0.25
assert np.allclose(T.sum(1), 1.0, atol=1e-6)
assert np.allclose(T[:, 4], (y == 1).astype(np.float32))     # 주변 성공확률 == y

season = tr.season.to_numpy(np.int64)
gt = tr.game_type.to_numpy().astype(str)
trm = (season <= 2023) & ~((season <= 2022) & (gt == "F"))
vam = season == 2024
assert int(trm.sum()) == 1116277 and int(vam.sum()) == 253507
assert not (trm & vam).any()
# 기존 자산과 대조 (같은 마스크·같은 y 인지)
assert np.array_equal(trm, np.load(f"{W89}/trm_oot23.npy"))
assert np.array_equal(vam, np.load(f"{W89}/valid_mask.npy"))
assert np.array_equal(y.astype(np.float32), np.load(f"{W89}/y.npy"))

G["train_split"] = {
    "n_train": int(trm.sum()),
    "unresolved_failures_in_train": int((unres & trm).sum()),
    "unresolved_share_in_train": float((unres & trm).mean()),
    "class_prior_in_train": [float(T[trm, j].sum() / trm.sum()) for j in range(5)],
    "y_mean_in_train": float(y[trm].mean()),
}

T[vam] = np.nan                        # 🔴 예측면 라벨을 물리적으로 못 쓰게 만든다
assert np.isfinite(T[trm]).all() and np.isnan(T[vam]).all()
np.save(f"{W}/T5_oot23.npy", T)
G["T5_path"] = f"{W}/T5_oot23.npy"
G["T5_sha256"] = hashlib.sha256(np.ascontiguousarray(T).tobytes()).hexdigest()
G["T5_valid_rows_are_nan"] = True
G["class_order"] = CELLS + ["success"]
json.dump(G, open(f"{OUT}/prep_lab093.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(G, ensure_ascii=False, indent=1))
