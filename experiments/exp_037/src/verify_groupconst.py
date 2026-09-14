"""exp_037 / A4 — group-constancy.

Every one of the 22 columns is, by construction, a pure function of (season,
lookup key).  If a column varies inside a (season, key) cell then something
row-specific leaked into it — the row's own outcome being the obvious
candidate.  This file states each column's declared key and checks it.

A cell is constant iff its non-NaN values are all equal AND its NaN pattern is
all-or-nothing.  `nunique()` alone would pass a cell that is half NaN and half
a single value, so the NaN count is checked separately.

Keys
----
  p6_* , i3_anom_* , idea_matchup_pitch
        the pitch model's eight inputs (they are that model's output, and its
        argmax / its ratio against a count-wise mean)
  idea_cnt_te        (balls_before, strikes_before)
  i3_te48            (pitcher_hand, batter_hand, balls_before, strikes_before)
  sit_cnt , sit_cnt_dev   (pitcher_id, balls_before, strikes_before)
  sit_run , sit_run_dev   (pitcher_id, num_runners_on > 0)
  sit_sc  , sit_sc_dev    (pitcher_id, num_runners_on >= 2)

Writes verify_groupconst.json; exits non-zero on any violation.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

DATA = os.path.expanduser("~/SungjinPark/data")
TYPES = ["curve", "cutter", "fastball", "offspeed", "sinker", "slider"]
PM_FEATS = ["inning", "top_bottom", "balls_before", "strikes_before",
            "outs_before", "batter_hand", "pitcher_hand", "pitcher_id"]
t0 = time.time()


def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)


def main():
    out = os.path.abspath(sys.argv[sys.argv.index("--out") + 1])
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "main"
    prov = json.load(open(os.path.join(out, f"featgen_{tag}.json")))
    feat = pd.read_parquet(os.path.join(out, prov["feat_file"]))

    tr = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                     usecols=["row_id", "season"] + PM_FEATS
                     + ["num_runners_on"])
    assert (tr["row_id"].to_numpy() == feat["row_id"].to_numpy()).all(), \
        "feature file is not aligned with train file order"
    df = pd.concat([tr.drop(columns=["row_id"]).reset_index(drop=True),
                    feat.drop(columns=["row_id"]).reset_index(drop=True)],
                   axis=1)
    df["runner"] = (df["num_runners_on"] > 0).astype(int)
    df["scoring"] = (df["num_runners_on"] >= 2).astype(int)
    # season 2019 is all-NaN by design; constancy there is vacuous but true.

    SPECS = [
        (["p6_" + t for t in TYPES] + ["i3_anom_" + t for t in TYPES]
         + ["i3_anom_l1", "idea_matchup_pitch"], PM_FEATS,
         "pitch-model inputs"),
        (["idea_cnt_te"], ["balls_before", "strikes_before"], "count"),
        (["i3_te48"], ["pitcher_hand", "batter_hand", "balls_before",
                       "strikes_before"], "hand x hand x count"),
        (["sit_cnt", "sit_cnt_dev"], ["pitcher_id", "balls_before",
                                      "strikes_before"], "pitcher x count"),
        (["sit_run", "sit_run_dev"], ["pitcher_id", "runner"],
         "pitcher x runner-on"),
        (["sit_sc", "sit_sc_dev"], ["pitcher_id", "scoring"],
         "pitcher x scoring-position"),
    ]

    report = {"tag": tag, "PASS": True, "columns": {}}
    for cols, keys, why in SPECS:
        gk = ["season"] + keys
        log(f"grouping by {gk} for {len(cols)} columns ({why}) ...")
        g = df.groupby(gk, sort=False, observed=True, dropna=False)
        size = g.size()
        for c in cols:
            # a cell is constant iff its non-NaN values agree AND its NaN
            # pattern is all-or-nothing; nunique() alone would pass a cell that
            # is half NaN and half one value.
            nun = g[c].nunique(dropna=True)
            nna = size - g[c].count()
            bad_val = int((nun > 1).sum())
            bad_na = int(((nna > 0) & (nna < size)).sum())
            ok = bad_val == 0 and bad_na == 0
            report["columns"][c] = {
                "key": gk, "why": why, "n_groups": int(len(size)),
                "groups_with_multiple_values": bad_val,
                "groups_with_partial_nan": bad_na,
                "PASS": ok,
            }
            report["PASS"] = report["PASS"] and ok
            log(f"  {c}: groups {len(size):,} multi-value {bad_val} "
                f"partial-nan {bad_na} {'PASS' if ok else 'FAIL'}")

    report["minutes"] = round((time.time() - t0) / 60, 2)
    with open(os.path.join(out, "verify_groupconst.json"), "w") as f:
        json.dump(report, f, indent=1)
    log("VERIFY_GROUPCONST", "PASS" if report["PASS"] else "FAIL")
    sys.exit(0 if report["PASS"] else 1)


if __name__ == "__main__":
    main()
