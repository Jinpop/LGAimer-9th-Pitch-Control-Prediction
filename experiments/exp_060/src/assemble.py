"""exp_060 stage2 — 99열 + cnt_state + 라벨을 한 프레임으로 조립 → cache/frame99.parquet"""
import hashlib
import json
import os
import time

import pandas as pd

B = os.path.expanduser("~/SungjinPark/exp060_nn")
SPD = os.path.expanduser("~/SungjinPark/sp_def_recv")
t0 = time.time()

tr = pd.read_csv(os.path.expanduser("~/SungjinPark/data/train.csv"),
                 encoding="utf-8-sig")

v6 = pd.concat([pd.read_parquet(f"{B}/work/v6_{Y}.parquet")
                for Y in sorted(tr.season.unique())]).sort_index()
assert len(v6) == len(tr) and v6.index.equals(tr.index)
ins = pd.read_parquet(f"{B}/work/ins16.parquet")
sp = pd.read_parquet(f"{B}/work/sp6.parquet")
assert ins.index.equals(tr.index) and sp.index.equals(tr.index)

df = pd.concat([tr, v6, ins, sp], axis=1)
df["cnt_state"] = df.balls_before.astype(int) * 3 + df.strikes_before.astype(int)

FSP = [l.strip() for l in open(f"{SPD}/features_sp.txt", encoding="utf-8")
       if l.strip()]
assert len(FSP) == 99, f"features_sp {len(FSP)} != 99"
missing = [c for c in FSP if c not in df.columns]
assert not missing, f"missing cols: {missing}"

keep = FSP + ["cnt_state", "control_success", "row_id"]
os.makedirs(f"{B}/cache", exist_ok=True)
df[keep].to_parquet(f"{B}/cache/frame99.parquet")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


prov = {"train_csv_sha256": sha(os.path.expanduser("~/SungjinPark/data/train.csv")),
        "features_sp_sha256": sha(f"{SPD}/features_sp.txt"),
        "pipeline_py_sha256": sha(f"{B}/pipeline.py"),
        "inseason_py_sha256": sha(f"{SPD}/inseason.py"),
        "splits_py_sha256": sha(f"{SPD}/splits.py"),
        "n_rows": int(len(df)),
        "nan_share_by_block": {
            "v6_table_cols_2019_expected": "전부 NaN(선행 시즌 없음)",
            "sp_2019_rows": int(df.loc[df.season == 2019, "sp_pit_bhand"].isna().sum()),
            "ins_nan_total": int(df[[c for c in FSP if c.startswith("ins_")]]
                                 .isna().sum().sum())}}
json.dump(prov, open(f"{B}/cache/provenance.json", "w"), indent=1)
print(json.dumps(prov, indent=1))
print(f"assemble done {df[keep].shape} elapsed {(time.time()-t0)/60:.2f}min",
      flush=True)
