"""exp_037 / A1 — AS-OF trackman<->train pitcher id maps.

WHY THIS FILE EXISTS
--------------------
The team-mate's `pipeline.py` trains a 6-class pitch-type model whose feature
list includes `pitcher_id`, i.e. the MAIN-DATA pitcher id.  trackman_history
carries `pitcher_trackman_id`, a different anonymisation.  Something has to
bridge them.  exp_002 built that bridge from full-sequence game fingerprints
(inning / top-bottom / B-S-O / pitcher hand arrays, md5) and a chain of
1:1 verification gates; this file is exp_002/src/01_build_mapping.py's control
flow with ONE change:

    the id dictionary is emitted ONCE PER CUTOFF SEASON c, aggregated only
    from games whose season is <= c.

exp_002 aggregated over every season at once.  A map built that way would let a
2023 row's pitch-type model know an identity that was only established by a
2024 game.  That is not a TARGET leak (no label touches this file) but it is
still information from the future, and exp_037's leak boundary says every table
a season-s row consumes is built from <= s-1.  So the map obeys it too.

WHY PER-CUTOFF RE-AGGREGATION IS SOUND AND CHEAP
------------------------------------------------
Steps 1-4 of exp_002 (block reconstruction, fingerprinting, matching, within-
game verification) are per-game and, because `season` is the first component of
every match key, per-season independent: restricting to seasons <= c cannot
create or destroy a match among the remaining games.  Only step 5, the global
1:1 conflict resolution, is cross-season.  So this file runs 1-4 once and 5-6
once per cutoff.

NO LABEL IS READ.  `control_success` is not in the usecols of either read.

Outputs (all under --out):
    map/pitcher_map_upto_{c}.csv    c in 2019..2023   (tr, tm, n_games)
    map/blocks_accepted.csv         block, season, game, stage, n_rows
    tm_lab.parquet                  trackman rows with normalised PM inputs
                                    and the 6-class pitch label `lab`
    mapgen.json                     provenance + coverage
"""
import gzip
import hashlib
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd

DATA = os.path.expanduser("~/SungjinPark/data")
t0 = time.time()


def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)


# --------------------------------------------------------------------------- #
# the 6-class pitch label.
#
# THIS IS A RECONSTRUCTION, NOT AN INHERITANCE.  `pipeline.py` receives `lab`
# already computed and never says how; the six class names (curve / cutter /
# fastball / offspeed / sinker / slider) are all we have of the team-mate's
# recipe.  The mapping below is OURS and is therefore a listed variable of
# exp_037.  It is built from `tagged_pitch_type` (0 NaN over 1,793,078 rows;
# `auto_pitch_type` has 72 NaN and splits Fastball/Four-Seam differently), and
# rows whose tag carries no pitch identity are DROPPED, not bucketed.
# --------------------------------------------------------------------------- #
LAB_MAP = {
    "Fastball": "fastball", "Four-Seam": "fastball",
    "Sinker": "sinker", "SInker": "sinker",
    "Cutter": "cutter",
    "Slider": "slider", "Sweeper": "slider", "Slurve": "slider",
    "Curveball": "curve", "Knuckleball": "curve",
    "ChangeUp": "offspeed", "Changeup": "offspeed", "Splitter": "offspeed",
}
LAB_DROP = {"Undefined", "Undefined#", "Undefind", "Other"}
TYPES = ["curve", "cutter", "fastball", "offspeed", "sinker", "slider"]
CUTOFFS = [2019, 2020, 2021, 2022, 2023]


def sha_list(vals):
    h = hashlib.sha256()
    for v in vals:
        h.update(str(v).encode())
        h.update(b"\n")
    return h.hexdigest()


def fingerprint(inning, tb01, balls, strikes, outs, phand):
    arr = np.column_stack([inning, tb01, balls, strikes, outs,
                           phand]).astype(np.int16)
    return hashlib.md5(arr.tobytes()).hexdigest()


def main():
    out = os.path.abspath(sys.argv[sys.argv.index("--out") + 1])
    os.makedirs(os.path.join(out, "map"), exist_ok=True)
    os.makedirs(os.path.join(out, "prov"), exist_ok=True)

    log("loading train (NO label column) ...")
    train = pd.read_csv(
        os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
        usecols=["row_id", "season", "game_month", "game_dayofweek",
                 "game_type", "inning", "top_bottom", "balls_before",
                 "strikes_before", "outs_before", "pitcher_id", "batter_id",
                 "pitcher_hand", "batter_hand", "pitcher_team_id",
                 "batter_team_id"])
    assert "control_success" not in train.columns
    log("train:", train.shape)

    log("loading trackman ...")
    tm = pd.read_csv(
        os.path.join(DATA, "trackman_history.csv"), encoding="utf-8-sig",
        usecols=["trackman_id", "season", "game_month", "game_dayofweek",
                 "trackman_game_id", "pitch_no", "inning", "top_bottom",
                 "balls_before", "strikes_before", "outs_before",
                 "pitcher_trackman_id", "batter_trackman_id",
                 "pitcher_hand", "batter_hand", "tagged_pitch_type"])
    log("trackman:", tm.shape)

    # -------------------------------------------------------------- normalise
    tm["tb"] = tm["top_bottom"].map({"Top": "T", "Bottom": "B",
                                     "T": "T", "B": "B"})
    assert tm["tb"].notna().all()
    HAND = {"Left": 1, "Right": 2, "L": 1, "R": 2}
    tm["p_hand"] = tm["pitcher_hand"].map(HAND).astype(np.int8)
    tm["b_hand"] = tm["batter_hand"].map(HAND).astype(np.int8)
    assert set(train["pitcher_hand"].unique()) <= {1, 2}
    assert set(train["top_bottom"].unique()) <= {"T", "B"}

    tag = tm["tagged_pitch_type"].astype(str).str.strip()
    unknown = sorted(set(tag.unique()) - set(LAB_MAP) - LAB_DROP)
    assert not unknown, ("unmapped tagged_pitch_type values", unknown)
    tm["lab"] = tag.map(LAB_MAP)
    lab_counts = tm["lab"].value_counts(dropna=False).to_dict()
    log("lab counts:", lab_counts)

    dup_mask = tm.duplicated(subset=["trackman_game_id", "pitch_no"],
                             keep=False)
    bad_tm_games = set(tm.loc[dup_mask, "trackman_game_id"].unique())
    log("tm games with duplicate pitch_no:", len(bad_tm_games))

    # ------------------------------------------- 1) train game block recovery
    log("reconstructing train game blocks ...")
    tp_min = np.minimum(train["pitcher_team_id"].values,
                        train["batter_team_id"].values)
    tp_max = np.maximum(train["pitcher_team_id"].values,
                        train["batter_team_id"].values)
    bk = (train["season"].astype(str) + "|" + train["game_month"].astype(str)
          + "|" + train["game_dayofweek"].astype(str) + "|" + train["game_type"]
          + "|" + pd.Series(tp_min).astype(str) + "_"
          + pd.Series(tp_max).astype(str))
    key_change = (bk != bk.shift(1)).values
    inn = train["inning"].values
    inning_drop = np.zeros(len(train), dtype=bool)
    inning_drop[1:] = inn[1:] < inn[:-1]
    new_block = key_change | (~key_change & inning_drop)
    new_block[0] = True
    train["block"] = np.cumsum(new_block)
    log("train blocks:", int(train["block"].iloc[-1]))

    # --------------------------------------------------------- 2) fingerprint
    TB01 = {"T": 0, "B": 1}
    train["tb01"] = train["top_bottom"].map(TB01).astype(np.int8)
    blk_rows = {b: idx for b, idx in train.groupby("block").indices.items()}
    blk_info = []
    for b, idx in blk_rows.items():
        s = train.iloc[idx]
        fp = fingerprint(s["inning"].values, s["tb01"].values,
                         s["balls_before"].values, s["strikes_before"].values,
                         s["outs_before"].values, s["pitcher_hand"].values)
        blk_info.append((b, s["season"].iat[0], s["game_month"].iat[0],
                         s["game_dayofweek"].iat[0], s["game_type"].iat[0],
                         len(s), fp))
    blocks = pd.DataFrame(blk_info, columns=["block", "season", "month", "dow",
                                             "game_type", "n", "fp"]
                          ).set_index("block")

    tm["tb01"] = tm["tb"].map(TB01).astype(np.int8)
    tm_sorted = tm.sort_values(["trackman_game_id", "pitch_no"],
                               kind="mergesort")
    tm_rows = {g: idx for g, idx in
               tm_sorted.groupby("trackman_game_id", sort=False).indices.items()}
    gm_info = []
    for g, idx in tm_rows.items():
        if g in bad_tm_games:
            continue
        s = tm_sorted.iloc[idx]
        fp = fingerprint(s["inning"].values, s["tb01"].values,
                         s["balls_before"].values, s["strikes_before"].values,
                         s["outs_before"].values, s["p_hand"].values)
        gm_info.append((g, s["season"].iat[0], s["game_month"].iat[0],
                        s["game_dayofweek"].iat[0], len(s), fp))
    games = pd.DataFrame(gm_info, columns=["game", "season", "month", "dow",
                                           "n", "fp"]).set_index("game")
    log("fingerprints:", len(blocks), "blocks /", len(games), "tm games")

    # ------------------------------------------------------------- 3) matching
    matches = {}

    def match_stage(bl, gm, keycols, stage):
        kb = bl[keycols].astype(str).agg("|".join, axis=1)
        kg = gm[keycols].astype(str).agg("|".join, axis=1)
        cb, cg = Counter(kb), Counter(kg)
        gmap = {k: g for g, k in kg.items() if cg[k] == 1}
        n_new = 0
        for b, k in kb.items():
            if cb[k] == 1 and k in gmap:
                matches[b] = (gmap[k], stage)
                n_new += 1
        log(f"stage {stage}: +{n_new} (cum {len(matches)})")

    match_stage(blocks, games, ["season", "month", "dow", "n", "fp"], "A")
    rb = blocks.loc[~blocks.index.isin(matches)]
    rg = games.loc[~games.index.isin({g for g, _ in matches.values()})]
    match_stage(rb, rg, ["season", "fp"], "B")

    # ------------------------------ 4) row-level pairs + within-game 1:1 gate
    game_pairs, fail_within = {}, []
    for b, (g, stage) in sorted(matches.items()):
        A = train.iloc[blk_rows[b]]
        B = tm_sorted.iloc[tm_rows[g]]
        assert len(A) == len(B)
        pp = pd.DataFrame({"tr": A["pitcher_id"].values,
                           "tm": B["pitcher_trackman_id"].values})
        bp = pd.DataFrame({"tr": A["batter_id"].values,
                           "tm": B["batter_trackman_id"].values})
        ok = (pp.groupby("tr")["tm"].nunique().eq(1).all()
              and pp.groupby("tm")["tr"].nunique().eq(1).all()
              and bp.groupby("tr")["tm"].nunique().eq(1).all()
              and bp.groupby("tm")["tr"].nunique().eq(1).all())
        if not ok or not (A["batter_hand"].values == B["b_hand"].values).all():
            fail_within.append(b)
            continue
        game_pairs[b] = dict(
            game=g, stage=stage, season=int(A["season"].iat[0]),
            pitch=set(map(tuple, pp.drop_duplicates().values)),
            bat=set(map(tuple, bp.drop_duplicates().values)),
            n_rows=len(A))
    log("within-game verification discarded:", len(fail_within))

    # ------------------ 5+6) PER CUTOFF: global conflict resolution + map
    def resolve_and_build(pairs):
        pairs = dict(pairs)
        discarded = 0
        for _ in range(10):
            edge_games = {}
            for b, d in pairs.items():
                for e in d["pitch"]:
                    edge_games.setdefault(("P", e), set()).add(b)
                for e in d["bat"]:
                    edge_games.setdefault(("B", e), set()).add(b)
            conflicts = set()
            for kind in ("P", "B"):
                by_tr, by_tm = {}, {}
                for (k, (tr, tmi)), gs in edge_games.items():
                    if k != kind:
                        continue
                    by_tr.setdefault(tr, set()).add((tr, tmi))
                    by_tm.setdefault(tmi, set()).add((tr, tmi))
                for node_edges in list(by_tr.values()) + list(by_tm.values()):
                    if len(node_edges) > 1:
                        supp = {e: len(edge_games[(kind, e)])
                                for e in node_edges}
                        mx = max(supp.values())
                        top = [e for e, c in supp.items() if c == mx]
                        losers = ([e for e in node_edges if e not in top]
                                  if len(top) == 1 else list(node_edges))
                        for e in losers:
                            conflicts |= edge_games[(kind, e)]
            if not conflicts:
                break
            discarded += len(conflicts)
            for b in conflicts:
                pairs.pop(b, None)
        rows = []
        for b, d in pairs.items():
            for tr, tmi in d["pitch"]:
                rows.append((tr, tmi, b))
        if not rows:
            return pd.DataFrame(columns=["tr", "tm", "n_games"]), 0, discarded
        dfm = pd.DataFrame(rows, columns=["tr", "tm", "block"])
        agg = (dfm.groupby(["tr", "tm"])
               .agg(n_games=("block", "nunique")).reset_index())
        assert agg["tr"].is_unique and agg["tm"].is_unique, "map not 1:1"
        return agg, len(pairs), discarded

    prov = {"cutoffs": {}}
    for c in CUTOFFS:
        sub = {b: d for b, d in game_pairs.items() if d["season"] <= c}
        agg, n_games_kept, n_disc = resolve_and_build(sub)
        path = os.path.join(out, "map", f"pitcher_map_upto_{c}.csv")
        agg.sort_values("tr").to_csv(path, index=False)
        seasons_in = sorted({d["season"] for d in sub.values()})
        assert not seasons_in or max(seasons_in) <= c
        prov["cutoffs"][str(c)] = {
            "n_games_considered": len(sub), "n_games_kept": n_games_kept,
            "n_games_discarded_conflict": n_disc,
            "n_pitchers_mapped": int(len(agg)),
            "seasons_present": seasons_in,
            "max_season_present": max(seasons_in) if seasons_in else None,
            "file": f"map/pitcher_map_upto_{c}.csv",
            "map_sha256": sha_list(
                [f"{a}|{b}" for a, b in
                 agg.sort_values("tr")[["tr", "tm"]].itertuples(index=False)]),
        }
        log(f"cutoff <= {c}: {len(agg)} pitchers from {n_games_kept} games "
            f"(discarded {n_disc})")

    pd.DataFrame([{"block": b, "season": d["season"], "game": d["game"],
                   "stage": d["stage"], "n_rows": d["n_rows"]}
                  for b, d in sorted(game_pairs.items())]
                 ).to_csv(os.path.join(out, "map", "blocks_accepted.csv"),
                          index=False)

    # ------------------------------------------------------- tm_lab.parquet
    keep = tm["lab"].notna()
    cols = ["trackman_id", "season", "pitcher_trackman_id", "inning", "tb",
            "balls_before", "strikes_before", "outs_before", "p_hand",
            "b_hand", "lab"]
    tml = tm.loc[keep, cols].copy()
    tml = tml.rename(columns={"tb": "top_bottom", "p_hand": "pitcher_hand",
                              "b_hand": "batter_hand"})
    tml.to_parquet(os.path.join(out, "tm_lab.parquet"), index=False)
    log("tm_lab rows:", len(tml), "of", len(tm))

    # trackman_id provenance per cutoff is written by featgen (it knows which
    # rows survive the id join); here we record the season census only.
    prov["lab_recipe"] = {"source_column": "tagged_pitch_type",
                          "map": LAB_MAP, "dropped_values": sorted(LAB_DROP),
                          "n_kept": int(keep.sum()),
                          "n_dropped": int((~keep).sum()),
                          "counts": {k: int(v) for k, v in lab_counts.items()
                                     if isinstance(k, str)}}
    prov["tm_rows_by_season"] = {str(k): int(v) for k, v in
                                 tml["season"].value_counts().sort_index()
                                 .items()}
    prov["n_blocks_matched"] = len(matches)
    prov["n_blocks_within_game_fail"] = len(fail_within)
    prov["exp002_reference"] = ("experiments/exp_002/src/01_build_mapping.py — "
                                "same fingerprint/verification logic, id "
                                "dictionary re-aggregated per cutoff season")
    prov["minutes"] = round((time.time() - t0) / 60, 2)
    with open(os.path.join(out, "mapgen.json"), "w") as f:
        json.dump(prov, f, indent=1, ensure_ascii=False)
    log("DONE", prov["minutes"], "min")


if __name__ == "__main__":
    main()
