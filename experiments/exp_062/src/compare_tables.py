"""exp_062 step2b — 표 자체를 대조해 '어긋남의 출처'를 가른다.

출하 표(≤2024)와 우리 표(≤2024)는 **같은 코드**(pipeline.py sha a3181a10…)로 만들어졌다.
따라서 어긋난다면 원인은 둘 중 하나다:
  (i) hist 행 선택이 다르다      -> cnt_te / te48 / sit_* / pitcher_overall / meta.base 가 어긋난다
  (ii) trackman 매핑이 다르다     -> pitchmodel6 / pmcat6 (그리고 그것에 딸린 anom_cnt) 만 어긋난다
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd

SHIP = os.path.expanduser("~/SungjinPark/exp062_gate/ship/mate/model/tables")
OURS = os.path.expanduser("~/SungjinPark/exp062_gate/work/tables_up2024")
OUT = os.path.expanduser("~/SungjinPark/exp062_gate/out/compare_tables.json")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


r = {"sha": {}, "detail": {}}
for f in ["meta.json", "cnt_te.csv", "te48.csv", "pitcher_overall.csv",
          "sit_cnt.csv", "sit_run.csv", "sit_sc.csv", "anom_cnt.csv",
          "pitchmodel6.txt", "pmcat6.json"]:
    r["sha"][f] = {"ship": sha(f"{SHIP}/{f}"), "ours": sha(f"{OURS}/{f}"),
                   "equal": sha(f"{SHIP}/{f}") == sha(f"{OURS}/{f}")}

r["detail"]["meta_base"] = {"ship": json.load(open(f"{SHIP}/meta.json"))["base"],
                            "ours": json.load(open(f"{OURS}/meta.json"))["base"]}

for f, idx in [("cnt_te.csv", 0), ("te48.csv", 0)]:
    a = pd.read_csv(f"{SHIP}/{f}", index_col=0)["v"]
    b = pd.read_csv(f"{OURS}/{f}", index_col=0)["v"]
    j = a.align(b, join="outer")
    r["detail"][f] = {"n_ship": int(len(a)), "n_ours": int(len(b)),
                      "key_set_equal": bool(set(a.index) == set(b.index)),
                      "max_abs": float(np.nanmax(np.abs(j[0] - j[1]))),
                      "mean_abs": float(np.nanmean(np.abs(j[0] - j[1])))}

for f, keys in [("sit_cnt.csv", ["pitcher_id", "balls_before", "strikes_before"]),
                ("sit_run.csv", ["pitcher_id", "runner"]),
                ("sit_sc.csv", ["pitcher_id", "scoring"])]:
    a = pd.read_csv(f"{SHIP}/{f}").set_index(keys)
    b = pd.read_csv(f"{OURS}/{f}").set_index(keys)
    common = a.index.intersection(b.index)
    d = (a.loc[common] - b.loc[common]).abs()
    r["detail"][f] = {"n_ship": int(len(a)), "n_ours": int(len(b)),
                      "n_common": int(len(common)),
                      "key_set_equal": bool(len(a) == len(b) == len(common)),
                      "max_abs": float(d.to_numpy().max()),
                      "mean_abs": float(d.to_numpy().mean())}

a = pd.read_csv(f"{SHIP}/pitcher_overall.csv").set_index("pitcher_id")
b = pd.read_csv(f"{OURS}/pitcher_overall.csv").set_index("pitcher_id")
common = a.index.intersection(b.index)
r["detail"]["pitcher_overall.csv"] = {
    "n_ship": int(len(a)), "n_ours": int(len(b)), "n_common": int(len(common)),
    "key_set_equal": bool(len(a) == len(b) == len(common)),
    "max_abs_p_all": float((a.loc[common, "p_all"] - b.loc[common, "p_all"]).abs().max()),
    "max_abs_p_n": float((a.loc[common, "p_n"] - b.loc[common, "p_n"]).abs().max())}

a = pd.read_csv(f"{SHIP}/anom_cnt.csv", index_col=0)
b = pd.read_csv(f"{OURS}/anom_cnt.csv", index_col=0)
r["detail"]["anom_cnt.csv"] = {"key_set_equal": bool(set(a.index) == set(b.index)),
                               "max_abs": float((a - b.reindex(a.index)).abs().to_numpy().max()),
                               "mean_abs": float((a - b.reindex(a.index)).abs().to_numpy().mean())}

sa = json.load(open(f"{SHIP}/pmcat6.json"))
sb = json.load(open(f"{OURS}/pmcat6.json"))
r["detail"]["pmcat6.json"] = {
    k: {"n_ship": len(sa[k]), "n_ours": len(sb[k]),
        "equal": sa[k] == sb[k],
        "only_ship": len(set(map(str, sa[k])) - set(map(str, sb[k]))),
        "only_ours": len(set(map(str, sb[k])) - set(map(str, sa[k])))}
    for k in sa}

json.dump(r, open(OUT, "w"), indent=1)
print(json.dumps(r, indent=1))
