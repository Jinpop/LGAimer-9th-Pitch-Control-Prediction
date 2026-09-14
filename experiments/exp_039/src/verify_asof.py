"""exp_037 / A3 — the as-of check, from a SECOND INDEPENDENT READ of the data.

The project rule is explicit: a run-time assert inside the generator is not
evidence, because the generator is the thing under suspicion.  So this file
does not import featgen and does not trust featgen.json's declared maxima.  It
re-reads train.csv and trackman_history.csv from disk, builds its own
row_id -> season and trackman_id -> season dictionaries, and looks up EVERY id
that featgen recorded as having contributed to a table.

Three claims are checked per target season s (s > 2019):

  1. every contributing train row_id exists and has season <= s-1
  2. the contributing set is EXACTLY {row_id : season <= s-1} — neither a
     subset (which would be safe but not what was declared) nor a superset
  3. every contributing trackman_id exists and has season <= s-1

plus, for s = 2019, that the whole 22-column block is NaN and that no sentinel
(0 / -1 / column mean) was substituted.

Writes verify_asof.json; exits non-zero on any failure.
"""
import gzip
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

DATA = os.path.expanduser("~/SungjinPark/data")
t0 = time.time()


def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)


def sha_list(vals):
    h = hashlib.sha256()
    for v in vals:
        h.update(str(v).encode())
        h.update(b"\n")
    return h.hexdigest()


def main():
    out = os.path.abspath(sys.argv[sys.argv.index("--out") + 1])
    tag = "main"
    if "--tag" in sys.argv:
        tag = sys.argv[sys.argv.index("--tag") + 1]
    prov = json.load(open(os.path.join(out, f"featgen_{tag}.json")))

    log("SECOND READ of train.csv (row_id, season only) ...")
    tr = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                     usecols=["row_id", "season"])
    row_season = dict(zip(tr["row_id"].tolist(), tr["season"].tolist()))
    by_season = {}
    for s, g in tr.groupby("season"):
        by_season[int(s)] = set(g["row_id"].tolist())
    log("train rows:", len(tr))

    log("SECOND READ of trackman_history.csv (trackman_id, season only) ...")
    tm = pd.read_csv(os.path.join(DATA, "trackman_history.csv"),
                     encoding="utf-8-sig", usecols=["trackman_id", "season"])
    tm_season = dict(zip(tm["trackman_id"].tolist(), tm["season"].tolist()))
    log("trackman rows:", len(tm))

    feat = pd.read_parquet(os.path.join(out, prov["feat_file"]))
    feat_season = feat["row_id"].map(row_season)
    assert feat_season.notna().all(), "feature file has row_ids not in train"
    cols = [c for c in feat.columns if c != "row_id"]
    assert len(cols) == 22, len(cols)

    report = {"tag": tag, "checks": {}, "PASS": True}

    for s_str, p in sorted(prov["per_season"].items()):
        s = int(s_str)
        c = {"season": s, "n_rows_declared": p["n_rows"]}
        blk = feat.loc[(feat_season == s).to_numpy(), cols]
        c["n_rows_observed"] = int(len(blk))
        c["rows_match"] = bool(len(blk) == p["n_rows"])

        if s == 2019:
            allnan = bool(blk.isna().all().all())
            c["all_nan"] = allnan
            c["distinct_non_nan"] = int(blk.notna().sum().sum())
            c["PASS"] = bool(allnan and c["rows_match"])
        else:
            with gzip.open(os.path.join(out, "prov",
                                        f"hist_rowids_s{s}.txt.gz"), "rt") as f:
                hid = f.read().split("\n")
            hid = [h for h in hid if h]
            with gzip.open(os.path.join(out, "prov",
                                        f"tm_ids_s{s}.txt.gz"), "rt") as f:
                tid = [int(x) for x in f.read().split("\n") if x]

            unknown = [h for h in hid if h not in row_season]
            seas = np.array([row_season[h] for h in hid if h in row_season])
            c["hist_n"] = len(hid)
            c["hist_unknown_ids"] = len(unknown)
            c["hist_season_max_RELOOKED_UP"] = int(seas.max()) if len(seas) else None
            c["hist_season_min_RELOOKED_UP"] = int(seas.min()) if len(seas) else None
            c["hist_violations_gt_s_minus_1"] = int((seas > s - 1).sum())
            expect = set().union(*[by_season[k] for k in by_season if k <= s - 1])
            got = set(hid)
            c["hist_set_equals_seasons_le_s_minus_1"] = bool(got == expect)
            c["hist_missing_from_declared"] = len(expect - got)
            c["hist_extra_beyond_declared"] = len(got - expect)
            c["hist_sha_matches_featgen"] = bool(
                sha_list(hid) == p["hist_rowid_sha256"])

            tunknown = [x for x in tid if x not in tm_season]
            tseas = np.array([tm_season[x] for x in tid if x in tm_season])
            c["tm_n"] = len(tid)
            c["tm_unknown_ids"] = len(tunknown)
            c["tm_season_max_RELOOKED_UP"] = int(tseas.max()) if len(tseas) else None
            c["tm_violations_gt_s_minus_1"] = int((tseas > s - 1).sum())
            c["tm_sha_matches_featgen"] = bool(sha_list(tid) == p["tm_id_sha256"])

            c["PASS"] = bool(
                c["rows_match"] and c["hist_unknown_ids"] == 0
                and c["hist_violations_gt_s_minus_1"] == 0
                and c["hist_set_equals_seasons_le_s_minus_1"]
                and c["hist_sha_matches_featgen"]
                and c["tm_unknown_ids"] == 0
                and c["tm_violations_gt_s_minus_1"] == 0
                and c["tm_sha_matches_featgen"])
        report["checks"][s_str] = c
        report["PASS"] = report["PASS"] and c["PASS"]
        log(f"season {s}: {'PASS' if c['PASS'] else 'FAIL'}  {c}")

    # ------------------------------------------------ no sentinel smuggling
    # A banned fill would show up as an implausible mass at exactly 0, -1 or the
    # column mean.  Report the mass; do not fail on it (0.0 is a legitimate
    # value for a ratio feature), but make it visible to the auditor.
    sent = {}
    nz = feat.loc[(feat_season != 2019).to_numpy(), cols]
    for c2 in cols:
        v = nz[c2].to_numpy()
        m = np.nanmean(v)
        sent[c2] = {
            "na_rate": round(float(np.isnan(v).mean()), 6),
            "frac_exact_zero": round(float((v == 0).mean()), 6),
            "frac_exact_minus1": round(float((v == -1).mean()), 6),
            "frac_exact_mean": round(float((v == m).mean()), 8),
        }
    report["sentinel_scan_seasons_2020_2024"] = sent

    report["minutes"] = round((time.time() - t0) / 60, 2)
    with open(os.path.join(out, "verify_asof.json"), "w") as f:
        json.dump(report, f, indent=1)
    log("VERIFY_ASOF", "PASS" if report["PASS"] else "FAIL")
    sys.exit(0 if report["PASS"] else 1)


if __name__ == "__main__":
    main()
