"""exp_064 stage 1 마무리 — frame99 의 14열만 배포 p6 산출로 갈아끼운다.

🔴 불변 게이트: 14열을 뺀 **나머지 88열(피처 85 + cnt_state·control_success·row_id)**
   이 원본 frame99 와 **바이트 동일**해야 한다. 하나라도 다르면 SystemExit.
   비교는 `values.tobytes()` — NaN 비트 패턴까지 본다.

원본 `exp060_nn/cache/frame99.parquet` 는 읽기 전용. 결과는 exp064/cache 에 쓴다.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

H = os.path.expanduser("~")
B = f"{H}/SungjinPark/exp064"
EXP60 = f"{H}/SungjinPark/exp060_nn"
SPD = f"{H}/SungjinPark/sp_def_recv"
YEARS = [2019, 2020, 2021, 2022, 2023, 2024]
COLS14 = ["p6_curve", "p6_cutter", "p6_fastball", "p6_offspeed", "p6_sinker",
          "p6_slider", "i3_anom_curve", "i3_anom_cutter", "i3_anom_fastball",
          "i3_anom_offspeed", "i3_anom_sinker", "i3_anom_slider", "i3_anom_l1",
          "idea_matchup_pitch"]
t0 = time.time()


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


f99 = pd.read_parquet(f"{EXP60}/cache/frame99.parquet")
prov = json.load(open(f"{EXP60}/cache/provenance.json"))
assert len(f99) == 1475092 == prov["n_rows"]
FSP = [l.strip() for l in open(f"{SPD}/features_sp.txt", encoding="utf-8")
       if l.strip()]
assert len(FSP) == 99
assert all(c in f99.columns for c in COLS14)

new = pd.concat([pd.read_parquet(f"{B}/work/p6_ship_{Y}.parquet")
                 for Y in YEARS]).sort_index()
assert len(new) == len(f99), (len(new), len(f99))
assert new.index.equals(f99.index), "인덱스 불일치"
assert np.array_equal(new.row_id.values, f99.row_id.values), "row_id 불일치"

# ── 교체 전 상태 기록 ────────────────────────────────────────────────────────
assert "season" in f99.columns, "frame99 에 season 열이 없다"
s = f99["season"].to_numpy()
assert sorted(np.unique(s).tolist()) == YEARS, np.unique(s).tolist()

before = {c: {"n_nan": int(f99[c].isna().sum()),
              "n_nan_2019": int(f99.loc[s == 2019, c].isna().sum()),
              "mean": float(np.nanmean(f99[c].values)),
              "std": float(np.nanstd(f99[c].values))} for c in COLS14}

out = f99.copy()
for c in COLS14:
    assert str(f99[c].dtype) == "float32", (c, f99[c].dtype)
    out[c] = new[c].values.astype("float32")

after = {c: {"n_nan": int(out[c].isna().sum()),
             "n_nan_2019": int(out.loc[s == 2019, c].isna().sum()),
             "mean": float(np.nanmean(out[c].values)),
             "std": float(np.nanstd(out[c].values))} for c in COLS14}

# ── 🔴 불변 게이트 ───────────────────────────────────────────────────────────
others = [c for c in f99.columns if c not in COLS14]
bad, obj_cols = [], []
for c in others:
    a, b = f99[c].values, out[c].values
    if a.dtype != b.dtype:
        bad.append(c)
        continue
    if a.dtype == object:
        # object 열의 tobytes() 는 포인터를 비교한다 — 내용으로 비교한다
        obj_cols.append(c)
        eq = bool(np.array_equal(a.astype(str), b.astype(str)))
    else:
        eq = (a.tobytes() == b.tobytes())      # NaN 비트 패턴까지 본다
    if not eq:
        bad.append(c)
gate = {"n_cols_total": int(f99.shape[1]), "n_cols_replaced": len(COLS14),
        "n_cols_checked": len(others), "byte_identical": len(bad) == 0,
        "mismatched": bad,
        "object_dtype_cols_compared_by_value": obj_cols,
        "dtypes_checked": {c: str(f99[c].dtype) for c in others},
        "col_order_preserved": list(out.columns) == list(f99.columns)}
if bad or not gate["col_order_preserved"]:
    json.dump({"invariance_gate": gate}, open(f"{B}/out/stage1_assemble.json", "w"),
              indent=1, ensure_ascii=False)
    raise SystemExit(f"🔴 불변 게이트 실패 — 바뀐 열 {bad}")

# ── 14열이 실제로 얼마나 움직였나 (참고 계측) ────────────────────────────────
moved = {}
for c in COLS14:
    a = f99[c].values.astype("float64")
    b = out[c].values.astype("float64")
    na, nb = np.isnan(a), np.isnan(b)
    both = ~na & ~nb
    d = np.abs(a[both] - b[both])
    moved[c] = {"max_abs": float(d.max()) if d.size else 0.0,
                "mean_abs": float(d.mean()) if d.size else 0.0,
                "n_both_valid": int(both.sum()),
                "n_nan_before_only": int((na & ~nb).sum()),
                "n_nan_after_only": int((~na & nb).sum())}

os.makedirs(f"{B}/cache", exist_ok=True)
out.to_parquet(f"{B}/cache/frame99_ship.parquet")

census = {str(Y): json.load(open(f"{B}/out/census_{Y}.json")) for Y in YEARS}
tot_rows = sum(census[str(Y)]["n_rows"] for Y in YEARS)
tot_unreg = sum(census[str(Y)]["pitcher_id"]["n_rows_unregistered"] for Y in YEARS)
res = {
    "invariance_gate": gate,
    "n_rows": int(len(out)),
    "frame99_src_sha256": sha(f"{EXP60}/cache/frame99.parquet"),
    "frame99_ship_sha256": sha(f"{B}/cache/frame99_ship.parquet"),
    "src_provenance": prov,
    "cols14": COLS14,
    "before_our_p6": before, "after_ship_p6": after, "moved": moved,
    "unregistered_census": {
        "by_season": {Y: census[str(Y)]["pitcher_id"] for Y in YEARS},
        "all_rows": tot_rows, "all_unregistered": tot_unreg,
        "all_frac": tot_unreg / tot_rows,
        "note": ("배포 pmcat6 의 pitcher_id 는 571 레벨. 미등재 투수 행은 "
                 "pd.Categorical 이 NaN 을 주고 LightGBM 이 결측으로 처리한다 — "
                 "출하 동작 그대로이며 임의 값으로 채우지 않았다. "
                 "p6 출력 자체는 NaN 이 아니라 '결측 분기'를 탄 예측값이다.")},
    "wall_min": (time.time() - t0) / 60.0}
json.dump(res, open(f"{B}/out/stage1_assemble.json", "w"), indent=1,
          ensure_ascii=False)
print(json.dumps({"invariance_gate": gate,
                  "unregistered_census": res["unregistered_census"],
                  "wall_min": res["wall_min"]}, indent=1, ensure_ascii=False))
print("STAGE1_OK", flush=True)
