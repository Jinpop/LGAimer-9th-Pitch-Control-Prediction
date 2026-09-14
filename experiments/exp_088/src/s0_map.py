"""exp_088 stage 0a — 면 정합 확인. 학습·추론 0건, 읽기만.

물음: 팀원 딥 팔이 이미 계산된 **cov 프레임 245,789행**과, 우리 팔(exp_085/086)이
      예측을 가진 **fold_2024 valid 253,507행**이 같은 물리적 투구를 가리키는가.

cov 는 `build_AL/frames/cov/test.csv` 이고 row_id 가 `TRAIN_*` 이다 — train.csv 행의
season 을 2025 로 덮은 대역 프레임(S32 계열). 그러므로 row_id 로 직접 대조할 수 있다.
"""
import hashlib, json, os
import numpy as np
import pandas as pd

H = "/home/<user>/SungjinPark"
B = f"{H}/exp088"
COV = f"{H}/build_CA/frames/cov/test.csv"
DBG = f"{H}/audit_deep/runs/dbg2__cov.deepdbg.csv"
os.makedirs(f"{B}/out", exist_ok=True)


def shaf(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


R = {"exp_id": "exp_088", "stage": "s0a_map"}

cov = pd.read_csv(COV, usecols=["row_id", "season", "game_type", "pitcher_id",
                                "batter_id"], dtype=str)
dbg = pd.read_csv(DBG, usecols=["row_id", "p_deep", "F_season", "F_game_type"])
ids_arms = np.asarray(np.load(f"{H}/arms/ids.npy", allow_pickle=True)).astype(str)

R["files"] = {
    "cov_test_csv": {"path": COV.replace(H, "~"), "sha256": shaf(COV),
                     "n_rows": int(len(cov))},
    "deepdbg_csv": {"path": DBG.replace(H, "~"), "sha256": shaf(DBG),
                    "n_rows": int(len(dbg))},
    "arms_ids_npy": {"path": "~/arms/ids.npy", "sha256": shaf(f"{H}/arms/ids.npy"),
                     "n_rows": int(ids_arms.shape[0])},
}

# 1) deepdbg 의 행 순서가 cov 와 같은가
R["deepdbg_order_eq_cov"] = bool(
    np.array_equal(dbg.row_id.to_numpy().astype(str), cov.row_id.to_numpy().astype(str)))
R["deepdbg_season_all_2025"] = bool((dbg.F_season.astype(int) == 2025).all())
R["cov_season_all_2025"] = bool((cov.season.astype(int) == 2025).all())

# 2) cov row_id 가 arms/ids (fold_2024 valid 253,507) 의 부분집합인가
cid = cov.row_id.to_numpy().astype(str)
pos = pd.Index(ids_arms).get_indexer(cid)
R["cov_rows_found_in_arms_ids"] = int((pos >= 0).sum())
R["cov_rows_missing"] = int((pos < 0).sum())
R["cov_is_prefix_of_arms_ids"] = bool(
    (pos >= 0).all() and np.array_equal(pos, np.arange(len(cid))))
R["cov_positions_monotonic"] = bool((pos >= 0).all() and (np.diff(pos) > 0).all())
R["cov_position_min_max"] = [int(pos.min()), int(pos.max())]

# 3) train.csv 상의 실제 season 은 무엇인가 (cov 가 덮기 전)
tr = pd.read_csv(f"{H}/data/train.csv", usecols=["row_id", "season", "game_type"],
                 dtype={"row_id": str})
tpos = pd.Index(tr.row_id.to_numpy().astype(str)).get_indexer(cid)
assert (tpos >= 0).all()
R["cov_true_season_counts"] = {str(k): int(v) for k, v in
                               pd.Series(tr.season.to_numpy()[tpos]).astype(str).value_counts().items()}
R["cov_train_row_index_min_max"] = [int(tpos.min()), int(tpos.max())]
R["cov_train_index_contiguous"] = bool(np.array_equal(
    np.sort(tpos), np.arange(tpos.min(), tpos.min() + len(tpos))))
R["cov_game_type_counts"] = {str(k): int(v) for k, v in
                             cov.game_type.value_counts().items()}
# 실제 2024 행 전체 대비 cov 가 덮는 비율
n2024 = int((tr.season.to_numpy() == 2024).sum())
R["n_train_rows_season2024"] = n2024
R["cov_share_of_2024"] = round(len(cid) / n2024, 6)

np.save(f"{B}/work/cov_pos_in_arms.npy", pos.astype(np.int64))
json.dump(R, open(f"{B}/out/s0a_map.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1))
