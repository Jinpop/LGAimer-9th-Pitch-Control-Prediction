"""exp_029 — the SUCCESS-axis selection surface, re-drawn on THREE seeds.
NO TRAINING.  Every base vector is reused from exp_020 / exp_022 / exp_024 with
its sha256 asserted against the digest exp_024 recorded; a drift aborts the run.

WHY THIS FILE EXISTS
--------------------
exp_023 drew the `d_s` selection surface over k x {self, league} and picked
`k=400 x league`.  That surface was SEED 42 ALONE.  Candidates J (LB
899.2884396705) and K ship that cell.  exp_024 promoted THAT ONE CELL to three
seeds; it never re-drew the surface.  So the shipped choice rests on a
single-seed surface, and this file asks whether the surface itself survives
two more seeds.

Three gaps are closed in the same pass:
  1. the surface is drawn on seeds 42 / 7 / 2024 SEPARATELY, and each seed gets
     its own argmin, so "the surface is seed-stable" becomes a measurement
     rather than an assumption;
  2. k is extended to 1600 / 3200 / 6400 on BOTH backoff targets (exp_026
     extended `league` only), so the high-k shape of `self` stops being
     unmeasured and the two targets can be checked for a CROSSING;
  3. the block-surface argmin and the fold_2024-surface argmin are compared on
     BOTH targets, so exp_026's k-disagreement (block 400 vs fold 1600) can be
     told apart from a league-specific artefact.

WHAT IS AND IS NOT A CHOICE
---------------------------
The grid cell is CHOSEN, and it is chosen on the supply blocks 2022+2023 and
nowhere else.  fold_2024 is read only after every beta and every choice is
fixed; where the two surfaces disagree the gap is reported in points and the
BLOCK choice stands.  This experiment produces NO new deployment candidate: if
the three-seed surface points somewhere other than `k400 x league`, that is a
finding to report, not a swap to make.

BETA
----
Never an axis.  For every (cell, seed) beta is the closed-form no-intercept
slope `offset.fit_slope` — the deployed code — fitted on that seed's supply
blocks and on that cell's own column.  No beta ever sees 2024.

THE FORM OF d
-------------
Expanded, `d = (s - n*overall)/(n + k)`, with the league term regrouped as
`L * k/(n+k)`.  The literal subtraction `(s + k*o)/(n+k) - o` returns
`k*o/k - o`, which IEEE-754 does not guarantee to be 0 on a cell miss; exp_023's
first launch died on exactly that at k=200.  Miss rows carrying EXACTLY 0.0 is
load-bearing (the hard-backoff arm adds L only where d == 0), so it is asserted
on all ten k.

INHERITANCE
-----------
`dev_k.py`, `backoff.py`, `offset.py`, `features_platoon.py`, `labels.py`,
`exp016_ref.py`, `devgen.py` are BYTE-PRESERVED from exp_028 (which took them
from exp_026) and their sha256 are asserted before anything runs.  `devgen.py`
builds the REVERSE axis and is not used here; it is carried and hashed so the
inherited set stays one object.

GATES (all four expectations are READ from the records, not transcribed)
------------------------------------------------------------------------
 G1  `k400 x league` == exp_024 arm B on all three seeds, |diff| < 5e-6.
 G2  `k111 x self` at the CONTROL beta (plain-d_s slope) == exp_024 A_control
     == exp_022 BK1f, all three seeds.  Run at the control beta on purpose:
     the refitted-beta version also sits inside 5e-6, so gating on it would
     pass for the wrong reason (exp_023's note, preserved).
 G3  `k0_self` and `k0_league` are the same vector — at k=0 the league weight
     is 0 on hits and 1 on misses, which IS the hard backoff.
 G4  d == 0.0 exactly on cell-miss rows, for all ten k.
 G5  (free, and the strongest) the SEED-42 surface must reproduce exp_023's
     recorded 14 cells cell-by-cell, on the block surface AND on fold_2024.
"""

import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

import features_platoon as fp                                     # noqa: E402
import offset as off                                              # noqa: E402
import backoff as bk                                              # noqa: E402
import dev_k as dk                                                # noqa: E402
import exp016_ref as ref                                          # noqa: E402
import labels as lb                                               # noqa: E402
from labels import recover                                        # noqa: E402

DATA_PATH = os.path.expanduser("~/SungjinPark/data/train.csv")
OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_029")
EXP020_PRED = os.path.expanduser(
    "~/SungjinPark/experiments/exp_020/pred/pred_E0_B3_P4iter_seed42_fold_2024.npy")
EXP022_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_022")
EXP023_METRICS = os.path.expanduser(
    "~/SungjinPark/experiments/exp_023/metrics.json")
EXP024_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_024")
EXP024_METRICS = os.path.join(EXP024_DIR, "metrics.json")

SEEDS = [42, 7, 2024]
GATE_TOL = 5e-6

TARGET = "control_success"
DROP_COLS = ["row_id", TARGET]
F_EXCLUDE_MAX_SEASON = 2022
SUPPLY_BLOCKS = [2022, 2023]
VALID_SEASON = 2024
EPS = 1e-6
BSS_REF_BRIER = 0.2498069

K_REF = 111                          # the deployed success-axis k
K_DEPLOYED = 400                     # candidate J / K's k
K_GRID = [0, 30, 55, 111, 200, 400, 800, 1600, 3200, 6400]
K_EXP023 = [0, 30, 55, 111, 200, 400, 800]      # the cells exp_023 recorded
TARGETS = ["self", "league"]

EXPECTED = {"valid_all": 253507, "valid_R": 223497,
            "ph1": 71830, "ph2": 181677}
TRAIN_CSV_SHA = "d2081186b458b49f60b082be480c273135833e15ba59a76d033af28bcf8763ff"

SHA_INHERITED = {
    "offset.py": "68d6007db03ab850e2a5181383a4148fae8947c711558282c9846391f13e0488",
    "backoff.py": "4169d2dadd8a07dceef2098a88971d3b3e437276bfb28834295e7df523591039",
    "labels.py": "439cd504478b4bc8162e9dedc32be27bfadd5d1178472d6b384b60f46ed7a9f2",
    "features_platoon.py": "68d63063e8ce659cf2876fbca4f5adef7ee1e720620ae63bb78ea"
                           "496c8dadf09",
    "exp016_ref.py": "4f24deb71d9eef3d626900f34a71aecd2d6acbae646eb9c92c51189a38"
                     "a3ece1",
    "dev_k.py": "e1cd86dd2795a4d12f279f7ea600353f441cfc7b19eea12207fb87117207a0d9",
    "devgen.py": "ac9640b94c80566200c89b81642a4b2cfb23aac86b6bb2dbe81bcc595557833c",
}

DEPLOYED_CELL = f"S{K_DEPLOYED}_league"          # what candidates J / K ship
LEGACY_CELL = f"S{K_REF}_self"                   # what candidates H / I shipped


def sha256_file(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def score(brier):
    """LB-scale conversion of a FULL-PRECISION Brier."""
    return round(100000.0 * (1.0 - brier / BSS_REF_BRIER), 5)


def score_delta(dbrier):
    return round(-100000.0 * float(dbrier) / BSS_REF_BRIER, 5)


def seg(y, p):
    m = ref.seg_metrics(y, p)
    if len(y):
        b = float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))
        m["brier_full"] = b
        m["score_vs_fixed_ref"] = score(b)
    return m


def cell_name(k, tg):
    return f"S{k}_{tg}"


def argmin_of(vals):
    return min(vals, key=vals.get)


def enclosure(vals_by_k, ks):
    """Is the minimum INTERIOR to the k grid, i.e. bracketed on both sides?"""
    v = [vals_by_k[k] for k in ks]
    j = int(np.argmin(v))
    return {
        "argmin_k": ks[j],
        "argmin_index": j,
        "n_points": len(ks),
        "bracketed_below": bool(j > 0),
        "bracketed_above": bool(j < len(ks) - 1),
        "enclosed_both_sides": bool(0 < j < len(ks) - 1),
        "value_at_argmin": v[j],
        "value_at_lower_neighbour": (v[j - 1] if j > 0 else None),
        "value_at_upper_neighbour": (v[j + 1] if j < len(ks) - 1 else None),
        "score_drop_to_lower_neighbour": (score_delta(v[j - 1] - v[j])
                                          if j > 0 else None),
        "score_drop_to_upper_neighbour": (score_delta(v[j + 1] - v[j])
                                          if j < len(ks) - 1 else None),
        "verdict": ("interior minimum — the grid brackets it"
                    if 0 < j < len(ks) - 1 else
                    "EDGE minimum — the grid does not bracket it; conclusions "
                    "about this region are not valid until the grid is widened"),
    }


def crossing(self_by_k, league_by_k, ks):
    """Where, if anywhere, does `self` cross `league` along k?"""
    diff = {k: self_by_k[k] - league_by_k[k] for k in ks}
    # in Brier, LOWER is better, so self WINS where diff < 0
    wins = {k: ("self" if diff[k] < 0 else "league" if diff[k] > 0 else "tie")
            for k in ks}
    flips = []
    for a, b in zip(ks[:-1], ks[1:]):
        if wins[a] != wins[b] and "tie" not in (wins[a], wins[b]):
            flips.append({"between_k": [a, b],
                          "from_winner": wins[a], "to_winner": wins[b],
                          "diff_at_lower": diff[a], "diff_at_upper": diff[b]})
    return {
        "convention": "diff = self - league in Brier; diff < 0 means SELF wins",
        "diff_by_k": {str(k): diff[k] for k in ks},
        "diff_by_k_score_self_minus_league": {str(k): score_delta(diff[k])
                                              for k in ks},
        "winner_by_k": {str(k): wins[k] for k in ks},
        "n_k_where_self_wins": sum(1 for k in ks if wins[k] == "self"),
        "n_k_where_league_wins": sum(1 for k in ks if wins[k] == "league"),
        "crossings": flips,
        "any_crossing": bool(flips),
    }


# --------------------------------------------------------------------------- #
def main():
    t_start = time.time()
    os.makedirs(os.path.join(OUT_DIR, "curves"), exist_ok=True)

    csv_sha = sha256_file(DATA_PATH)
    assert csv_sha == TRAIN_CSV_SHA, ("train.csv SHA MISMATCH", csv_sha)
    src_sha = {f: sha256_file(os.path.join(SRC_DIR, f))
               for f in sorted(os.listdir(SRC_DIR)) if f.endswith(".py")}
    for f, want in SHA_INHERITED.items():
        assert src_sha[f] == want, ("INHERITED SOURCE DRIFTED", f, src_sha[f], want)
    print("inherited sha OK — dev_k.py "
          f"{src_sha['dev_k.py'][:16]}..., offset.py {src_sha['offset.py'][:16]}...",
          flush=True)

    # ---- the records exp_023 and exp_024 left.  Gate numbers are READ. ------- #
    exp024_metrics_sha = sha256_file(EXP024_METRICS)
    with open(EXP024_METRICS) as f:
        m24_rec = json.load(f)
    assert m24_rec["exp_id"] == "exp_024", m24_rec.get("exp_id")
    assert m24_rec["train_csv_sha256"] == csv_sha, "exp_024 read a different csv"
    BASE_SHA = {}
    for s_str, prov in m24_rec["base_vector_provenance"].items():
        s = int(s_str)
        BASE_SHA[s] = {}
        for key, v in prov.items():
            k = "fold_2024" if key == "fold_2024" else int(key.split("_")[1])
            BASE_SHA[s][k] = v["sha256"]
        assert set(BASE_SHA[s]) == {"fold_2024", 2022, 2023}, (s, BASE_SHA[s])
    assert set(BASE_SHA) == set(SEEDS), BASE_SHA.keys()

    GATE_DEPLOYED = {int(s): m24_rec["per_seed"][str(s)]["B_candidate"]["brier"]
                     for s in SEEDS}
    GATE_ANCHOR = {int(s): m24_rec["per_seed"][str(s)]["A_control"]["brier"]
                   for s in SEEDS}
    GATE_ANCHOR_BETA = {int(s): m24_rec["per_seed"][str(s)]["beta_control_plain_ds"]
                        for s in SEEDS}
    GATE_BASE = {int(s): m24_rec["per_seed"][str(s)]["base_no_offset"][
        "fold_2024_brier"] for s in SEEDS}

    exp023_metrics_sha = sha256_file(EXP023_METRICS)
    with open(EXP023_METRICS) as f:
        m23_rec = json.load(f)
    assert m23_rec["exp_id"] == "exp_023"
    assert m23_rec["k_grid"] == K_EXP023, m23_rec["k_grid"]
    assert m23_rec["targets"] == TARGETS
    # dev_k.py was NEW in exp_023, so it sits under source_sha256 there, not
    # under inherited_source_sha256.  Both files it inherited AND the file it
    # authored are checked, so "the same code" is a verified claim.
    for f in ("offset.py", "backoff.py", "labels.py", "features_platoon.py",
              "exp016_ref.py"):
        assert m23_rec["inherited_source_sha256"][f] == src_sha[f], \
            (f"{f} is not the file exp_023 ran", src_sha[f])
    assert m23_rec["source_sha256"]["dev_k.py"] == src_sha["dev_k.py"], \
        ("dev_k.py is not the file exp_023 ran", src_sha["dev_k.py"])
    E23_BLOCK = m23_rec["grid_blocks_SELECTION_SURFACE"]
    E23_FOLD = m23_rec["grid_fold_2024_DIAGNOSTIC_ONLY"]
    print(f"exp_024 record read (sha {exp024_metrics_sha[:16]}...), exp_023 record "
          f"read (sha {exp023_metrics_sha[:16]}...)", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] loading {DATA_PATH}", flush=True)
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    assert df["row_id"].is_unique
    assert df["season"].is_monotonic_increasing
    provided_cols = [c for c in df.columns if c not in DROP_COLS]
    assert len(provided_cols) == 47

    rec = recover(df, verbose=False)
    assert rec["success_recovery"]["agree_pct"] == 100.0

    # ---- recovered labels may never be an input ----------------------------- #
    lb.assert_no_recovered_in_features(provided_cols, where="provided 47 columns")
    recovered_present = [c for c in df.columns if c in lb.RECOVERED_COLS
                         or c.startswith(lb.RECOVERED_PREFIXES)]
    label_isolation = {
        "recovered_columns_present_in_frame": len(recovered_present),
        "recovered_columns_reaching_any_input_list": len(
            [c for c in provided_cols if c in recovered_present]),
        "provided_feature_columns": len(provided_cols),
        "check_a_no_recovered_in_provided_columns": True,
        "note": "no model is fitted here, so the design matrix is exp_024's; "
                "check (b) below is the operative one for this experiment",
    }
    assert label_isolation["recovered_columns_reaching_any_input_list"] == 0

    season = df["season"].to_numpy()
    ok = df["recov_ok"].to_numpy()
    succ_season = df.groupby("pitcher_id", sort=False)["season"].shift(-1).to_numpy()
    f_excl = ((df["season"] <= F_EXCLUDE_MAX_SEASON)
              & (df["game_type"] == "F")).to_numpy()
    y_all = df[TARGET].to_numpy(dtype=np.float64)
    is_r_all = (df["game_type"] == "R").to_numpy()
    ph_all = df[bk.PIT_HAND].to_numpy()
    row_id = df["row_id"].to_numpy()

    # ---- deviations.  SEED-INDEPENDENT (they come from labels, not the model) #
    print(f"[{time.strftime('%H:%M:%S')}] platoon lookups + k=111 reference",
          flush=True)
    fp.build_season_forward_chained(df, ok, succ_season, f_excl, verbose=False)
    dev_ref, hit_ref, _, _ = off.build_deviations(df, ok, succ_season, f_excl,
                                                  verbose=False)
    assert fp.RATE_SPEC[bk.SUCC_COL][1] == K_REF

    print(f"[{time.strftime('%H:%M:%S')}] success axis over k grid {K_GRID} via "
          f"BYTE-PRESERVED dev_k.py", flush=True)
    d_s, hit, n_cell0, _, ds_diag = dk.build_multi_k(df, ok, succ_season, f_excl,
                                                     K_GRID, verbose=False)
    assert np.array_equal(hit, hit_ref)
    L_s, has_Ls, _, _ = bk.build_league_dev(df, ok, succ_season, f_excl,
                                            verbose=False)
    Ls_eff = np.where(has_Ls, L_s, 0.0)

    # ---- GATE 4 + the expanded/literal equivalence at k=111 ------------------ #
    dd111 = d_s[K_REF] - dev_ref["d_s"]
    equiv = {
        "form_used": "(s_cell - n_cell*overall)/(n_cell + k)  [expanded, exp_023]",
        "form_inherited": "(s_cell + k*overall)/(n_cell + k) - overall  [literal]",
        "max_abs_diff_ds_k111_vs_inherited_literal_form": float(np.max(np.abs(dd111))),
        "max_abs_diff_k111_on_miss_rows": float(np.max(np.abs(dd111[~hit]))),
        "tolerance_used": 1e-12,
        "GATE4_miss_rows_exactly_zero_all_k": {
            str(k): bool(np.all(d_s[k][~hit] == 0.0)) for k in K_GRID},
    }
    assert equiv["max_abs_diff_ds_k111_vs_inherited_literal_form"] < 1e-12, equiv
    assert len(equiv["GATE4_miss_rows_exactly_zero_all_k"]) == len(K_GRID) == 10
    assert all(equiv["GATE4_miss_rows_exactly_zero_all_k"].values()), equiv
    print(f"GATE 4 OK — d == 0.0 exactly on miss rows for all {len(K_GRID)} k",
          flush=True)

    # ---- the 20 grid columns ------------------------------------------------- #
    TERMS = {}
    for k in K_GRID:
        for tg in TARGETS:
            TERMS[cell_name(k, tg)] = dk.build_arm(d_s[k], hit, Ls_eff, n_cell0,
                                                   k, tg)
    CELLS = list(TERMS)
    assert len(CELLS) == 20, len(CELLS)

    # ---- GATE 3 + structural asserts ----------------------------------------- #
    gate3 = bool(np.array_equal(TERMS["S0_self"], TERMS["S0_league"]))
    assert gate3, "GATE 3 FAILED: k=0 self and league must coincide by construction"
    for k in K_GRID:
        assert np.array_equal(TERMS[cell_name(k, "self")][~hit],
                              TERMS[cell_name(k, "league")][~hit]), \
            ("self and league must agree on miss rows", k)
        assert np.array_equal(TERMS[cell_name(k, "self")][hit], d_s[k][hit]), \
            ("self must be the bare deviation on hit rows", k)
    print("GATE 3 OK — k0_self and k0_league are the same vector", flush=True)

    # ---- label isolation (b): deviations constant inside the lookup group ---- #
    print(f"[{time.strftime('%H:%M:%S')}] label-isolation check (group constancy)",
          flush=True)
    gcodes = pd.DataFrame({"s": season, "p": df[fp.PITCHER].to_numpy(),
                           "h": df[fp.HAND].to_numpy(), "ph": ph_all}
                          ).groupby(["s", "p", "h", "ph"], sort=False,
                                    dropna=False).ngroup().to_numpy()
    const = {}
    for nm, v in list(TERMS.items()) + [(f"d_s_raw_k{k}", d_s[k]) for k in K_GRID]:
        first = pd.Series(v).groupby(gcodes, sort=False).transform("first").to_numpy()
        const[nm] = bool(np.array_equal(v, first))
    label_isolation["check_b_constant_within_season_pitcher_hand"] = const
    label_isolation["n_lookup_groups"] = int(gcodes.max() + 1)
    label_isolation["n_vectors_checked"] = len(const)
    assert all(const.values()), ("a deviation varies inside its own lookup group — "
                                 "a row-level label reached it", const)
    print(f"LABEL ISOLATION OK ({len(const)} vectors, "
          f"{label_isolation['n_lookup_groups']} groups)", flush=True)

    # ---- as-of, independently re-verified ------------------------------------ #
    print(f"[{time.strftime('%H:%M:%S')}] independent as-of re-verification",
          flush=True)
    indep = pd.read_csv(DATA_PATH, encoding="utf-8-sig", usecols=["row_id", "season"])
    assert indep["row_id"].is_unique
    s_map = pd.Series(indep["season"].to_numpy(), index=indep["row_id"].to_numpy())
    indep_season = s_map.reindex(pd.Index(row_id)).to_numpy()
    assert not np.isnan(indep_season).any()
    frame_matches_file = bool(np.array_equal(indep_season.astype(np.int64), season))
    assert frame_matches_file

    base_ok = ok & ~f_excl
    hand_ok = (~pd.isna(df[bk.PIT_HAND]).to_numpy()
               & ~pd.isna(df[bk.BAT_HAND]).to_numpy())
    asof_indep = {
        "method": "row_id -> season re-read from train.csv in a SECOND pass; the "
                  "max season of the CONTRIBUTING ROW IDS is looked up in that "
                  "independent map, not in the frame the tables were built from",
        "frame_season_matches_file": frame_matches_file,
        "tables_covered": ["personal_success", "league_success"],
        "per_season": {}}
    for s in sorted(set(season.tolist()))[1:]:
        m_pers = base_ok & (season <= s - 1) & (succ_season <= s - 1)
        m_leag = m_pers & hand_ok
        rec_s = {}
        for name, m in (("personal_success", m_pers), ("league_success", m_leag)):
            ids = row_id[m]
            mx = int(s_map.reindex(pd.Index(ids)).max())
            assert mx <= s - 1, ("AS-OF VIOLATION (independent)", name, s, mx)
            rec_s[name] = {"contributing_rows": int(m.sum()),
                           "max_season_of_contributors_independent": mx,
                           "allowed_max": int(s - 1)}
        asof_indep["per_season"][str(int(s))] = rec_s
    assert max(SUPPLY_BLOCKS) < VALID_SEASON
    print("AS-OF (independent) OK", flush=True)

    # ---- slices -------------------------------------------------------------- #
    y = {V: y_all[season == V] for V in SUPPLY_BLOCKS + [VALID_SEASON]}
    assert len(y[VALID_SEASON]) == EXPECTED["valid_all"]
    m24 = (season == VALID_SEASON)
    y24 = y[VALID_SEASON]
    hit24 = hit[m24]
    isr24 = is_r_all[m24]
    n_cell24 = n_cell0[m24]
    ph24 = ph_all[m24]
    assert int(isr24.sum()) == EXPECTED["valid_R"]
    assert int((ph24 == 1).sum()) == EXPECTED["ph1"]
    assert int((ph24 == 2).sum()) == EXPECTED["ph2"]

    T = {nm: {V: v[season == V] for V in SUPPLY_BLOCKS + [VALID_SEASON]}
         for nm, v in TERMS.items()}
    DS_RAW = {k: {V: d_s[k][season == V] for V in SUPPLY_BLOCKS} for k in K_GRID}

    hn = n_cell24[hit24]
    tert_lo, tert_hi = np.percentile(hn, [33.3333, 66.6667])
    bins24 = {
        "miss_n0": ~hit24,
        f"hit_low_n_le_{int(tert_lo)}": hit24 & (n_cell24 <= tert_lo),
        f"hit_mid_n_{int(tert_lo)}_{int(tert_hi)}": (hit24 & (n_cell24 > tert_lo)
                                                     & (n_cell24 <= tert_hi)),
        f"hit_high_n_gt_{int(tert_hi)}": hit24 & (n_cell24 > tert_hi),
        "fx_n_1_99": hit24 & (n_cell24 < 100),
        "fx_n_100_499": hit24 & (n_cell24 >= 100) & (n_cell24 < 500),
        "fx_n_500_1999": hit24 & (n_cell24 >= 500) & (n_cell24 < 2000),
        "fx_n_2000plus": hit24 & (n_cell24 >= 2000),
    }

    # ---- base vectors, all reused, all sha-asserted -------------------------- #
    def load_bases(s):
        prov = {}
        if s == 42:
            got = sha256_file(EXP020_PRED)
            assert got == BASE_SHA[42]["fold_2024"], ("BASE SHA MISMATCH", got)
            p24 = np.load(EXP020_PRED)
            prov["fold_2024"] = {"path": EXP020_PRED, "sha256": got,
                                 "reused_from": "exp_020 arm E0 (via exp_024)"}
            bl = {}
            for V in SUPPLY_BLOCKS:
                pth = os.path.join(EXP022_DIR, "pred",
                                   f"pred_B3_block_{V}_seed42.npy")
                with open(os.path.join(EXP022_DIR, f"block_{V}.json")) as f:
                    bm = json.load(f)
                assert bm["gate"]["passed"], bm["gate"]
                got = sha256_file(pth)
                assert bm["pred_sha256"] == got, ("block pred sha drift", V, got)
                assert got == BASE_SHA[42][V], ("block sha != exp_024's", V, got)
                assert bm["train_csv_sha256"] == csv_sha
                bl[V] = np.load(pth)
                prov[f"block_{V}"] = {"path": pth, "sha256": got,
                                      "reused_from": "exp_022 (via exp_024)"}
        else:
            bl = {}
            for V in SUPPLY_BLOCKS + [VALID_SEASON]:
                pth = os.path.join(EXP024_DIR, "pred",
                                   f"pred_B3_block_{V}_seed{s}.npy")
                with open(os.path.join(EXP024_DIR, "blocks",
                                       f"block_{V}_seed{s}.json")) as f:
                    bm = json.load(f)
                assert bm["train_csv_sha256"] == csv_sha
                got = sha256_file(pth)
                assert bm["pred_sha256"] == got, ("pred sha drift", V, s, got)
                key = "fold_2024" if V == VALID_SEASON else V
                assert got == BASE_SHA[s][key], ("sha != exp_024 metrics", V, s, got)
                arr = np.load(pth)
                if V == VALID_SEASON:
                    p24 = arr
                else:
                    bl[V] = arr
                prov[f"block_{V}" if V != VALID_SEASON else "fold_2024"] = {
                    "path": pth, "sha256": got, "reused_from": "exp_024 b3_block.py",
                    "iter": bm["stack_iter"], "cell_iter": bm["cell_iter"]}
        for V in SUPPLY_BLOCKS:
            assert bl[V].shape == y[V].shape, (s, V, bl[V].shape)
        assert p24.shape == y24.shape, (s, p24.shape)
        return p24, bl, prov

    ANCHOR = "ANCHOR_k111_self_control_beta"
    per_seed, provenance = {}, {}

    for s in SEEDS:
        print(f"[{time.strftime('%H:%M:%S')}] ===== seed {s} =====", flush=True)
        p24, blk, prov = load_bases(s)
        provenance[str(s)] = prov
        br_base24 = float(np.mean((p24 - y24) ** 2))
        y_sup = np.concatenate([y[V] for V in SUPPLY_BLOCKS])
        p_sup = np.concatenate([blk[V] for V in SUPPLY_BLOCKS])
        r_sup = y_sup - p_sup
        br_base_blocks = float(np.mean((p_sup - y_sup) ** 2))

        # the CONTROL beta of gate 2: the slope of the PLAIN d_s at k=111, with
        # no league term at all.  exp_024 recorded it per seed; recompute and
        # assert rather than transcribe.
        beta_plain = off.fit_slope(
            r_sup, np.concatenate([DS_RAW[K_REF][V] for V in SUPPLY_BLOCKS])
        )["beta"]
        assert abs(beta_plain - GATE_ANCHOR_BETA[s]) < 1e-9, \
            ("control beta drift", s, beta_plain, GATE_ANCHOR_BETA[s])

        # ---- coefficients, fitted on the supply blocks ONLY ------------------ #
        blockfit = {}
        for nm in CELLS:
            dsup = np.concatenate([T[nm][V] for V in SUPPLY_BLOCKS])
            f1 = off.fit_slope(r_sup, dsup)
            beta = f1["beta"]
            pb = off.apply_linear(p_sup, dsup, beta)
            fit = {"beta_fitted_on_blocks": beta, "beta_se": f1["se"],
                   "beta_t": f1["t"], "mean_d2": f1["mean_d2"],
                   "beta_unrounded": float(np.dot(dsup, r_sup) / np.dot(dsup, dsup)),
                   "fitter": "offset.fit_slope (deployed code, rounds to 6dp)",
                   "brier_blocks": float(np.mean((pb - y_sup) ** 2)),
                   "sd_d_blocks": float(np.std(dsup))}
            fit["delta_vs_base_blocks"] = fit["brier_blocks"] - br_base_blocks
            fit["score_blocks"] = score(fit["brier_blocks"])
            # honest cross-block surface: fit on one block, score the other
            cb = {}
            for Vf, Ve in ((SUPPLY_BLOCKS[0], SUPPLY_BLOCKS[1]),
                           (SUPPLY_BLOCKS[1], SUPPLY_BLOCKS[0])):
                c2 = off.fit_slope(y[Vf] - blk[Vf], T[nm][Vf])["beta"]
                pe = off.apply_linear(blk[Ve], T[nm][Ve], c2)
                cb[f"fit{Vf}_score{Ve}"] = {"beta": c2,
                                            "brier": float(np.mean((pe - y[Ve]) ** 2)),
                                            "n": int(len(y[Ve]))}
            tot_n = sum(v["n"] for v in cb.values())
            fit["crossblock_brier_pooled"] = float(
                sum(v["brier"] * v["n"] for v in cb.values()) / tot_n)
            fit["crossblock_score"] = score(fit["crossblock_brier_pooled"])
            fit["crossblock_detail"] = cb
            fit["beta_per_block"] = {
                str(V): off.fit_slope(y[V] - blk[V], T[nm][V])["beta"]
                for V in SUPPLY_BLOCKS}
            blockfit[nm] = fit

        seb = (p24 - y24) ** 2

        def evaluate(nm, beta, spec):
            c = T[nm][VALID_SEASON]
            p = off.apply_linear(p24, c, beta)
            raw = p24 + beta * c
            sh = p - p24
            se = (p - y24) ** 2
            br = float(np.mean(se))
            out = {
                "cell": nm, "spec": spec, "beta_applied": beta,
                "brier": br, "score": score(br),
                "delta_vs_base_no_offset": float(br - br_base24),
                "score_gain_vs_base": score_delta(br - br_base24),
                "valid_R": seg(y24[isr24], p[isr24]),
                "valid_F": seg(y24[~isr24], p[~isr24]),
                "seg_miss": seg(y24[~hit24], p[~hit24]),
                "seg_hit": seg(y24[hit24], p[hit24]),
                "correction_shift": {
                    "n_clip": int(((raw < EPS) | (raw > 1 - EPS)).sum()),
                    "raw_min": float(np.min(raw)), "raw_max": float(np.max(raw)),
                    "total_mean_abs": float(np.mean(np.abs(sh))),
                    "total_max_abs": float(np.max(np.abs(sh))),
                    "sd_d": float(np.std(c)),
                },
                "by_n_bin": {}, "by_pitcher_hand": {},
                "by_pitcher_hand_x_cellhit": {},
            }
            for bn, msk in bins24.items():
                if not msk.any():
                    continue
                out["by_n_bin"][bn] = {
                    "n": int(msk.sum()), "brier": float(np.mean(se[msk])),
                    "delta_vs_base_within_bin": float(np.mean(se[msk] - seb[msk])),
                }
            for h in (1, 2):
                mh = (ph24 == h)
                out["by_pitcher_hand"][str(h)] = {
                    "n": int(mh.sum()), "brier": float(np.mean(se[mh])),
                    "delta_vs_base": float(np.mean(se[mh] - seb[mh])),
                }
                for lab, mm in (("hit", mh & hit24), ("miss", mh & ~hit24)):
                    out["by_pitcher_hand_x_cellhit"][f"{h}_{lab}"] = {
                        "n": int(mm.sum()), "brier": float(np.mean(se[mm])),
                        "delta_vs_base": float(np.mean(se[mm] - seb[mm])),
                    }
            return out

        res = {nm: evaluate(nm, blockfit[nm]["beta_fitted_on_blocks"],
                            f"d_s at k={nm.split('_')[0][1:]} x backoff target "
                            f"{nm.split('_')[1]}, beta refitted on the blocks")
               for nm in CELLS}
        res[ANCHOR] = evaluate(
            LEGACY_CELL, beta_plain,
            "GATE 2: k=111 x self at the CONTROL beta (the plain-d_s slope) = "
            "exp_022 BK1f / G1f = exp_024 A_control = candidates H / I's d")
        res[ANCHOR]["cell"] = ANCHOR

        per_seed[str(s)] = {
            "base_no_offset": {"fold_2024_brier": br_base24,
                               "fold_2024_score": score(br_base24),
                               "blocks_brier": br_base_blocks},
            "control_beta_plain_ds_k111": beta_plain,
            "cells": res,
            "block_fit": blockfit,
        }
        # ---- per-seed surfaces and per-seed argmins -------------------------- #
        blk_surf = {nm: blockfit[nm]["brier_blocks"] for nm in CELLS}
        xbl_surf = {nm: blockfit[nm]["crossblock_brier_pooled"] for nm in CELLS}
        f24_surf = {nm: res[nm]["brier"] for nm in CELLS}
        per_seed[str(s)]["surfaces"] = {
            "block_brier": blk_surf, "block_score": {nm: score(v) for nm, v
                                                     in blk_surf.items()},
            "crossblock_brier": xbl_surf,
            "crossblock_score": {nm: score(v) for nm, v in xbl_surf.items()},
            "fold_2024_brier": f24_surf,
            "fold_2024_score": {nm: score(v) for nm, v in f24_surf.items()},
        }
        per_seed[str(s)]["argmin"] = {
            "block_all20": argmin_of(blk_surf),
            "block_self_only": argmin_of({n: v for n, v in blk_surf.items()
                                          if n.endswith("_self")}),
            "block_league_only": argmin_of({n: v for n, v in blk_surf.items()
                                            if n.endswith("_league")}),
            "crossblock_all20": argmin_of(xbl_surf),
            "fold_2024_all20_DIAGNOSTIC": argmin_of(f24_surf),
            "fold_2024_self_only_DIAGNOSTIC": argmin_of(
                {n: v for n, v in f24_surf.items() if n.endswith("_self")}),
            "fold_2024_league_only_DIAGNOSTIC": argmin_of(
                {n: v for n, v in f24_surf.items() if n.endswith("_league")}),
        }
        a = per_seed[str(s)]["argmin"]
        per_seed[str(s)]["argmin"]["block_choice_equals_deployed_cell"] = bool(
            a["block_all20"] == DEPLOYED_CELL)
        print(f"  seed {s}: block argmin {a['block_all20']}  "
              f"crossblock {a['crossblock_all20']}  "
              f"fold2024(diag) {a['fold_2024_all20_DIAGNOSTIC']}", flush=True)
        print(f"    {DEPLOYED_CELL} fold_2024 {res[DEPLOYED_CELL]['brier']:.14f} "
              f"({res[DEPLOYED_CELL]['score']:.5f})   anchor "
              f"{res[ANCHOR]['brier']:.14f}", flush=True)
        del p24, blk

    # ---- GATE 1 / GATE 2 ----------------------------------------------------- #
    gate = {"tolerance": GATE_TOL,
            "source": "exp_024/metrics.json per_seed[seed].B_candidate.brier and "
                      ".A_control.brier — READ from the record, not transcribed",
            "on_fail": "abort — the reused vectors or the inherited code are not "
                       "exp_024's and nothing downstream is readable",
            "checks": {}}
    allpass = True
    for s in SEEDS:
        for tag, nm, want in (("GATE1_" + DEPLOYED_CELL, DEPLOYED_CELL,
                               GATE_DEPLOYED[s]),
                              ("GATE2_anchor_k111_self", ANCHOR, GATE_ANCHOR[s]),
                              ("base_no_offset", None, GATE_BASE[s])):
            got_b = (per_seed[str(s)]["base_no_offset"]["fold_2024_brier"]
                     if nm is None else per_seed[str(s)]["cells"][nm]["brier"])
            diff = abs(got_b - want)
            ok_ = bool(diff < GATE_TOL)
            allpass = allpass and ok_
            gate["checks"][f"{tag}_seed{s}"] = {
                "expected_from_exp024": want, "got": got_b, "abs_diff": diff,
                "bit_identical": bool(got_b == want), "passed": ok_}
    gate["GATE3_k0_self_equals_k0_league"] = gate3
    gate["GATE4_miss_rows_zero_all_10_k"] = all(
        equiv["GATE4_miss_rows_exactly_zero_all_k"].values())
    gate["passed"] = bool(allpass and gate3 and gate["GATE4_miss_rows_zero_all_10_k"])
    print("REPRODUCTION GATE (1,2,3,4): " + json.dumps(
        {k: v for k, v in gate.items() if k != "checks"}), flush=True)
    for k, v in gate["checks"].items():
        print(f"   {k:<40} diff {v['abs_diff']:.3e}  bitid {v['bit_identical']}  "
              f"pass {v['passed']}", flush=True)
    assert gate["passed"], "GATE FAILED: " + json.dumps(gate)

    # ---- GATE 5: the seed-42 surface must reproduce exp_023 cell by cell ------ #
    g5 = {"what": "exp_023's recorded 14 cells (7 k x 2 targets), seed 42, on the "
                  "BLOCK surface and on fold_2024.  This is the strongest gate "
                  "available: it checks every beta and every Brier, not one arm",
          "tolerance": GATE_TOL, "cells": {}}
    g5_pass = True
    for k in K_EXP023:
        for tg in TARGETS:
            nm = cell_name(k, tg)
            old = f"k{k}_{tg}"
            wb = E23_BLOCK[old]["brier_blocks"]
            wf = E23_FOLD[old]["brier"]
            wbeta = E23_BLOCK[old]["beta_fitted_on_blocks"]
            gb = per_seed["42"]["block_fit"][nm]["brier_blocks"]
            gf = per_seed["42"]["cells"][nm]["brier"]
            gbeta = per_seed["42"]["block_fit"][nm]["beta_fitted_on_blocks"]
            row = {"exp023_cell": old,
                   "beta_exp023": wbeta, "beta_got": gbeta,
                   "beta_identical": bool(wbeta == gbeta),
                   "block_brier_exp023": wb, "block_brier_got": gb,
                   "block_abs_diff": abs(gb - wb),
                   "block_bit_identical": bool(gb == wb),
                   "fold2024_brier_exp023": wf, "fold2024_brier_got": gf,
                   "fold2024_abs_diff": abs(gf - wf),
                   "fold2024_bit_identical": bool(gf == wf)}
            row["passed"] = bool(row["block_abs_diff"] < GATE_TOL
                                 and row["fold2024_abs_diff"] < GATE_TOL
                                 and row["beta_identical"])
            g5_pass = g5_pass and row["passed"]
            g5["cells"][nm] = row
    g5["passed"] = bool(g5_pass)
    g5["n_cells"] = len(g5["cells"])
    g5["all_bit_identical"] = bool(all(
        c["block_bit_identical"] and c["fold2024_bit_identical"]
        for c in g5["cells"].values()))
    print(f"GATE 5 (seed-42 surface == exp_023's 14 cells): passed {g5['passed']}, "
          f"bit-identical {g5['all_bit_identical']}", flush=True)
    assert g5["passed"], "GATE 5 FAILED: " + json.dumps(g5)

    # ---- the three-seed surfaces --------------------------------------------- #
    def mean_surface(key):
        return {nm: float(np.mean([per_seed[str(s)]["surfaces"][key][nm]
                                   for s in SEEDS])) for nm in CELLS}

    mean_block = mean_surface("block_brier")
    mean_xblock = mean_surface("crossblock_brier")
    mean_fold = mean_surface("fold_2024_brier")

    def by_k(surface, tg):
        return {k: surface[cell_name(k, tg)] for k in K_GRID}

    surfaces = {
        "note": "block = Brier on the pooled supply blocks 2022+2023 at the "
                "block-fitted beta (exp_023's registered SELECTION surface).  "
                "crossblock = fit one block, score the other, pooled.  fold_2024 "
                "is DIAGNOSTIC and selects nothing.",
        "k_grid": K_GRID, "targets": TARGETS,
        "mean_over_seeds": {
            "block_brier": mean_block,
            "block_score": {nm: score(v) for nm, v in mean_block.items()},
            "crossblock_brier": mean_xblock,
            "crossblock_score": {nm: score(v) for nm, v in mean_xblock.items()},
            "fold_2024_brier": mean_fold,
            "fold_2024_score": {nm: score(v) for nm, v in mean_fold.items()},
        },
        "per_seed_block_brier": {str(s): per_seed[str(s)]["surfaces"]["block_brier"]
                                 for s in SEEDS},
        "per_seed_fold_2024_brier": {
            str(s): per_seed[str(s)]["surfaces"]["fold_2024_brier"] for s in SEEDS},
    }

    # ---- Q1: seed-by-seed argmin, then the mean surface ---------------------- #
    ans_q1 = {
        "ask": "does the three-seed block surface still point at "
               f"{DEPLOYED_CELL} (what candidates J and K ship)?",
        "per_seed_block_argmin": {str(s): per_seed[str(s)]["argmin"]["block_all20"]
                                  for s in SEEDS},
        "per_seed_crossblock_argmin": {
            str(s): per_seed[str(s)]["argmin"]["crossblock_all20"] for s in SEEDS},
        "mean_block_argmin": argmin_of(mean_block),
        "mean_crossblock_argmin": argmin_of(mean_xblock),
        "deployed_cell": DEPLOYED_CELL,
    }
    ans_q1["all_three_seeds_agree"] = bool(len(set(
        ans_q1["per_seed_block_argmin"].values())) == 1)
    ans_q1["every_seed_picks_deployed"] = bool(all(
        v == DEPLOYED_CELL for v in ans_q1["per_seed_block_argmin"].values()))
    ans_q1["mean_surface_picks_deployed"] = bool(ans_q1["mean_block_argmin"]
                                             == DEPLOYED_CELL)
    ans_q1["shipped_choice_reconfirmed_on_three_seeds"] = bool(
        ans_q1["every_seed_picks_deployed"] and ans_q1["mean_surface_picks_deployed"])
    # how much does a seed's own argmin differ from the deployed cell, in points?
    ans_q1["per_seed_gap_own_argmin_vs_deployed_on_block_surface"] = {
        str(s): {
            "own_argmin": per_seed[str(s)]["argmin"]["block_all20"],
            "block_brier_gap": (per_seed[str(s)]["surfaces"]["block_brier"][
                DEPLOYED_CELL]
                - per_seed[str(s)]["surfaces"]["block_brier"][
                    per_seed[str(s)]["argmin"]["block_all20"]]),
        } for s in SEEDS}
    for s in SEEDS:
        g = ans_q1["per_seed_gap_own_argmin_vs_deployed_on_block_surface"][str(s)]
        g["block_score_gap"] = score_delta(g["block_brier_gap"])
    ans_q1["seed_sensitivity_verdict"] = (
        "the block surface picks the same cell on all three seeds"
        if ans_q1["all_three_seeds_agree"] else
        "THE BLOCK SURFACE PICKS A DIFFERENT CELL DEPENDING ON THE SEED — the "
        "surface itself is seed-sensitive, which is a confidence problem for k "
        "selection generally, not just for this cell")

    # ---- Q2: the high-k shape of `self`, and any crossing -------------------- #
    ans_q2 = {
        "ask": "what does `self` do at high k, and does it cross `league`?",
        "block_mean": crossing(by_k(mean_block, "self"),
                               by_k(mean_block, "league"), K_GRID),
        "fold_2024_mean_DIAGNOSTIC": crossing(by_k(mean_fold, "self"),
                                              by_k(mean_fold, "league"), K_GRID),
        "per_seed_block": {str(s): crossing(
            by_k(per_seed[str(s)]["surfaces"]["block_brier"], "self"),
            by_k(per_seed[str(s)]["surfaces"]["block_brier"], "league"), K_GRID)
            for s in SEEDS},
        "high_k_cells_new_in_this_experiment": [cell_name(k, "self")
                                                for k in (1600, 3200, 6400)]
        + [cell_name(6400, "league")],
        "context": "exp_028 measured the REVERSE axis and found `self` beating "
                   "`league` at EVERY k it tried (k <= 3200).  On the success "
                   "axis exp_026 found the opposite for k <= 800.  A crossing "
                   "inside this grid would mean the preference is k-dependent, "
                   "not axis-dependent.",
    }

    # ---- Q3: block argmin vs fold_2024 argmin, per target -------------------- #
    q3 = {"ask": "does the block-surface argmin disagree with the fold_2024 "
                 "argmin on `self` too, or only on `league`?",
          "discipline": "fold_2024 is read AFTER every beta and every choice is "
                        "fixed.  It prices the discipline; it never picks.",
          "per_target": {}}
    for tg in TARGETS:
        cands = [cell_name(k, tg) for k in K_GRID]
        b_arm = argmin_of({n: mean_block[n] for n in cands})
        x_arm = argmin_of({n: mean_xblock[n] for n in cands})
        o_arm = argmin_of({n: mean_fold[n] for n in cands})
        gapb = mean_fold[b_arm] - mean_fold[o_arm]
        q3["per_target"][tg] = {
            "block_selected": b_arm, "crossblock_selected": x_arm,
            "fold_2024_oracle_DIAGNOSTIC": o_arm,
            "block_and_crossblock_agree": bool(b_arm == x_arm),
            "block_choice_matches_oracle": bool(b_arm == o_arm),
            "k_of_block_choice": int(b_arm.split("_")[0][1:]),
            "k_of_oracle": int(o_arm.split("_")[0][1:]),
            "fold_2024_brier_of_block_choice": mean_fold[b_arm],
            "fold_2024_brier_of_oracle": mean_fold[o_arm],
            "cost_of_honest_selection_brier": gapb,
            "cost_of_honest_selection_score": score_delta(gapb),
            "per_seed_block_argmin": {
                str(s): argmin_of({n: per_seed[str(s)]["surfaces"]["block_brier"][n]
                                   for n in cands}) for s in SEEDS},
            "per_seed_fold_argmin_DIAGNOSTIC": {
                str(s): argmin_of({n: per_seed[str(s)]["surfaces"][
                    "fold_2024_brier"][n] for n in cands}) for s in SEEDS},
        }
    # the whole 20-cell surface, not split by target
    b20 = argmin_of(mean_block)
    o20 = argmin_of(mean_fold)
    gap20 = mean_fold[b20] - mean_fold[o20]
    q3["all_20_cells"] = {
        "block_selected": b20, "crossblock_selected": argmin_of(mean_xblock),
        "fold_2024_oracle_DIAGNOSTIC": o20,
        "block_choice_matches_oracle": bool(b20 == o20),
        "cost_of_honest_selection_brier": gap20,
        "cost_of_honest_selection_score": score_delta(gap20),
        "prior_measurements_of_this_cost": {
            "exp_023_seed42_only": -2.66995, "exp_026_league_only": -1.58490,
            "exp_028_reverse_axis": 0.0},
    }

    # ---- Q4: is the minimum bracketed on both sides? ------------------------- #
    q4 = {"ask": "is the surface enclosed on both sides — is the minimum "
                 "interior to the k grid?",
          "why": "a minimum at the edge means the grid is too narrow and every "
                 "conclusion about that region is provisional",
          "block_mean": {tg: enclosure(by_k(mean_block, tg), K_GRID)
                         for tg in TARGETS},
          "crossblock_mean": {tg: enclosure(by_k(mean_xblock, tg), K_GRID)
                              for tg in TARGETS},
          "fold_2024_mean_DIAGNOSTIC": {tg: enclosure(by_k(mean_fold, tg), K_GRID)
                                        for tg in TARGETS},
          "per_seed_block": {str(s): {tg: enclosure(
              by_k(per_seed[str(s)]["surfaces"]["block_brier"], tg), K_GRID)
              for tg in TARGETS} for s in SEEDS}}
    q4["all_block_surfaces_enclosed"] = bool(
        all(q4["block_mean"][tg]["enclosed_both_sides"] for tg in TARGETS)
        and all(q4["per_seed_block"][str(s)][tg]["enclosed_both_sides"]
                for s in SEEDS for tg in TARGETS))
    q4["all_fold_surfaces_enclosed"] = bool(
        all(q4["fold_2024_mean_DIAGNOSTIC"][tg]["enclosed_both_sides"]
            for tg in TARGETS))

    # ---- selection, stated as the discipline requires ------------------------ #
    selection = {
        "rule": "argmin of the pooled-block Brier over all 20 cells, beta fitted "
                "on the same blocks.  fold_2024 is read only afterwards.",
        "selected_cell": b20,
        "selected_block_brier_mean": mean_block[b20],
        "selected_block_score_mean": score(mean_block[b20]),
        "selected_beta_by_seed": {
            str(s): per_seed[str(s)]["block_fit"][b20]["beta_fitted_on_blocks"]
            for s in SEEDS},
        "selected_fold_2024_brier_by_seed": {
            str(s): per_seed[str(s)]["cells"][b20]["brier"] for s in SEEDS},
        "selected_fold_2024_score_by_seed": {
            str(s): per_seed[str(s)]["cells"][b20]["score"] for s in SEEDS},
        "deployed_cell": DEPLOYED_CELL,
        "selection_equals_deployed": bool(b20 == DEPLOYED_CELL),
        "NO_NEW_DEPLOYMENT": "this experiment produces no candidate.  If the "
                             "surface had pointed elsewhere that would be a "
                             "finding to report, and any swap would go through "
                             "its own three-seed promotion, leak audit and "
                             "packager run.",
    }

    # ---- the deployed cell and the legacy cell, side by side ---------------- #
    headline = {}
    for nm in (DEPLOYED_CELL, LEGACY_CELL, ANCHOR, b20, o20):
        if nm in headline:
            continue
        br = [per_seed[str(s)]["cells"][nm]["brier"] for s in SEEDS]
        headline[nm] = {
            "fold_2024_brier_by_seed": {str(s): b for s, b in zip(SEEDS, br)},
            "fold_2024_score_by_seed": {str(s): score(b) for s, b in zip(SEEDS, br)},
            "mean_brier": float(np.mean(br)), "mean_score": score(float(np.mean(br))),
            "seed_std_ddof1": float(np.std(br, ddof=1)),
            "seed_std_score": score_delta(float(np.std(br, ddof=1))),
        }
    base_br = [per_seed[str(s)]["base_no_offset"]["fold_2024_brier"] for s in SEEDS]
    headline["base_no_offset"] = {
        "fold_2024_brier_by_seed": {str(s): b for s, b in zip(SEEDS, base_br)},
        "fold_2024_score_by_seed": {str(s): score(b) for s, b in zip(SEEDS, base_br)},
        "mean_brier": float(np.mean(base_br)),
        "mean_score": score(float(np.mean(base_br))),
        "seed_std_ddof1": float(np.std(base_br, ddof=1)),
        "seed_std_score": score_delta(float(np.std(base_br, ddof=1))),
    }

    # ---- curves on disk ------------------------------------------------------ #
    curve_files = []
    for s in SEEDS:
        pth = os.path.join(OUT_DIR, "curves", f"block_surface_seed{s}.csv")
        with open(pth, "w") as f:
            f.write("k,target,cell,beta_blocks,brier_blocks,score_blocks,"
                    "brier_crossblock,score_crossblock\n")
            for k in K_GRID:
                for tg in TARGETS:
                    nm = cell_name(k, tg)
                    bf = per_seed[str(s)]["block_fit"][nm]
                    f.write(f"{k},{tg},{nm},{bf['beta_fitted_on_blocks']!r},"
                            f"{bf['brier_blocks']!r},{bf['score_blocks']!r},"
                            f"{bf['crossblock_brier_pooled']!r},"
                            f"{bf['crossblock_score']!r}\n")
        curve_files.append(pth)
    pth = os.path.join(OUT_DIR, "curves", "fold_2024_surface.csv")
    with open(pth, "w") as f:
        f.write("k,target,cell," + ",".join(f"brier_seed{s}" for s in SEEDS)
                + ",brier_mean,score_mean\n")
        for k in K_GRID:
            for tg in TARGETS:
                nm = cell_name(k, tg)
                vs = [per_seed[str(s)]["cells"][nm]["brier"] for s in SEEDS]
                f.write(f"{k},{tg},{nm}," + ",".join(repr(v) for v in vs)
                        + f",{float(np.mean(vs))!r},{score(float(np.mean(vs)))!r}\n")
    curve_files.append(pth)
    pth = os.path.join(OUT_DIR, "curves", "block_surface_mean.csv")
    with open(pth, "w") as f:
        f.write("k,target,cell," + ",".join(f"brier_blocks_seed{s}" for s in SEEDS)
                + ",brier_blocks_mean,score_blocks_mean\n")
        for k in K_GRID:
            for tg in TARGETS:
                nm = cell_name(k, tg)
                vs = [per_seed[str(s)]["surfaces"]["block_brier"][nm] for s in SEEDS]
                f.write(f"{k},{tg},{nm}," + ",".join(repr(v) for v in vs)
                        + f",{float(np.mean(vs))!r},{score(float(np.mean(vs)))!r}\n")
    curve_files.append(pth)

    summary = {
        "cell_seed_table_fold2024_brier": {
            nm: {str(s): per_seed[str(s)]["cells"][nm]["brier"] for s in SEEDS}
            for nm in CELLS + [ANCHOR]},
        "cell_seed_table_fold2024_score": {
            nm: {str(s): per_seed[str(s)]["cells"][nm]["score"] for s in SEEDS}
            for nm in CELLS + [ANCHOR]},
        "cell_mean_fold2024_brier": mean_fold,
        "cell_mean_fold2024_score": {nm: score(v) for nm, v in mean_fold.items()},
        "cell_mean_block_brier": mean_block,
        "cell_mean_block_score": {nm: score(v) for nm, v in mean_block.items()},
        "cell_seed_std_ddof1_fold2024": {
            nm: float(np.std([per_seed[str(s)]["cells"][nm]["brier"]
                              for s in SEEDS], ddof=1)) for nm in CELLS},
        "beta_by_seed_by_cell": {
            str(s): {nm: per_seed[str(s)]["block_fit"][nm]["beta_fitted_on_blocks"]
                     for nm in CELLS} for s in SEEDS},
        "beta_per_block_by_cell_seed42": {
            nm: per_seed["42"]["block_fit"][nm]["beta_per_block"] for nm in CELLS},
    }

    metrics = {
        "exp_id": "exp_029",
        "title": "the success-axis k x backoff-target selection surface re-drawn "
                 "on three seeds and widened to k=6400 — post-hoc transforms of "
                 "stored predictions, no training",
        "no_training": True,
        "purpose": "the shipped cell (candidates J and K) was chosen on a "
                   "SINGLE-SEED surface (exp_023).  exp_024 promoted that one "
                   "cell to three seeds but never re-drew the surface.  This "
                   "re-draws it.",
        "k_grid": K_GRID, "targets": TARGETS, "n_cells": len(CELLS),
        "seeds": SEEDS,
        "deployed_cell": DEPLOYED_CELL, "legacy_cell": LEGACY_CELL,
        "changed_vs_exp_023_and_exp_026": [
            "the surface is drawn on THREE seeds (42/7/2024) with a SEPARATE "
            "argmin per seed; exp_023's surface was seed 42 alone",
            "the k grid goes 7 -> 10 values: 1600, 3200 and 6400 are added",
            "the high-k extension covers BOTH backoff targets; exp_026 extended "
            "`league` only, so k in {1600,3200,6400} x self and 6400 x league "
            "are measured here for the first time",
            "an explicit enclosure test (is the argmin interior to the grid) on "
            "every surface, per seed and on the mean",
            "an explicit self-vs-league crossing test along k",
            "the block-vs-fold_2024 argmin gap is computed for BOTH targets "
            "(exp_026 had it for league only)",
            "the seed-42 surface is gated cell-by-cell against exp_023's "
            "recorded 14 cells (GATE 5) — a check exp_023 could not run",
            "run with OMP_NUM_THREADS=2 because exp_027 is training on the same "
            "box (pure numpy vector work; GATE 5 shows it changes nothing)",
        ],
        "unchanged_vs_exp_023_and_exp_026": [
            "base vectors and their sha256, exp_004 F-mask, clip 1e-6, supply "
            "blocks 2022+2023, closed-form no-intercept slope, expanded d form, "
            "league term regrouped as L*k/(n+k), dev_k.py / backoff.py / "
            "offset.py / features_platoon.py / labels.py / exp016_ref.py / "
            "devgen.py byte-identical (sha asserted)",
        ],
        "inherited_source_sha256": SHA_INHERITED,
        "source_sha256": src_sha,
        "train_csv_sha256": csv_sha,
        "base_vector_provenance": provenance,
        "base_vector_expected_sha256_from_exp024_record": {
            str(k): {str(kk): vv for kk, vv in v.items()}
            for k, v in BASE_SHA.items()},
        "exp023_metrics_json_sha256": exp023_metrics_sha,
        "exp024_metrics_json_sha256": exp024_metrics_sha,
        "reproduction_gate": gate,
        "gate5_exp023_surface_reproduction": g5,
        "label_isolation_check": label_isolation,
        "k_equivalence_and_gate4": equiv,
        "asof_independent_recheck": asof_indep,
        "surfaces": surfaces,
        "Q1_seed_stability_of_the_argmin": ans_q1,
        "Q2_self_vs_league_and_crossing": ans_q2,
        "Q3_block_vs_fold2024_argmin_gap": q3,
        "Q4_enclosure": q4,
        "selection": selection,
        "headline_cells": headline,
        "summary": summary,
        "per_seed": per_seed,
        "success_axis_per_season": ds_diag["per_season"],
        "curve_files": curve_files,
        "conversion": {
            "formula": f"score = 100000 * (1 - brier / {BSS_REF_BRIER})",
            "sensitivity": "brier 1e-6 = score 0.4003.  Every Brier here is full "
                           "precision, so the converted numbers carry no rounding "
                           "band of their own",
            "caveat": "local measurement is at iter 650 (<=2023 training); the "
                      "deployed build is iter 7700 (<=2024).",
        },
        "leak_boundary": {
            "tables": "a season-s row reads tables built on seasons <= s-1 only; "
                      "asserted at runtime inside backoff.py / dev_k.py and again "
                      "here by an independent second read of train.csv",
            "recovered_labels": "the success numerator is a RECOVERED label of "
                                "row t+1.  It is used only as the numerator of a "
                                "frozen as-of table, never as a row-level input; "
                                "group-constancy of all 30 checked vectors inside "
                                "(season, pitcher, batter_hand, pitcher_hand) is "
                                "asserted, which a row-level leak would break",
            "coef_fit": "supply blocks 2022+2023 of the same seed, on that cell's "
                        "own column; no beta ever sees 2024",
            "cell_choice": "the grid is fixed by config.yaml before any 2024 "
                           "number is read, and the cell is chosen on the block "
                           "surface; fold_2024 is diagnostic only",
            "design_matrix": "unchanged — no deviation is ever a model feature, "
                             "and no training happens in this experiment",
            "missing": "no sentinel fill anywhere; L=0 and d=0 mean 'apply no "
                       "correction', the behaviour without the module",
        },
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__},
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "compute_time_min": round((time.time() - t_start) / 60, 2),
    }
    metrics["n_bin_definition"] = {
        "hit_tertile_boundaries_n_cell": [float(tert_lo), float(tert_hi)],
        "bin_sizes": {b: int(m.sum()) for b, m in bins24.items()},
    }
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(json.loads(json.dumps(metrics, default=str)), f, indent=2,
                  ensure_ascii=False)

    # ---- console summary ----------------------------------------------------- #
    print("\n--- BLOCK SURFACE (selection), mean over 3 seeds ---", flush=True)
    print(f"{'k':>6}{'self brier':>22}{'self score':>13}"
          f"{'league brier':>22}{'league score':>13}", flush=True)
    for k in K_GRID:
        a, b = mean_block[cell_name(k, "self")], mean_block[cell_name(k, "league")]
        print(f"{k:>6}{a:>22.14f}{score(a):>13.5f}{b:>22.14f}{score(b):>13.5f}",
              flush=True)
    print("\n--- fold_2024 SURFACE (diagnostic), mean over 3 seeds ---", flush=True)
    print(f"{'k':>6}{'self brier':>22}{'self score':>13}"
          f"{'league brier':>22}{'league score':>13}", flush=True)
    for k in K_GRID:
        a, b = mean_fold[cell_name(k, "self")], mean_fold[cell_name(k, "league")]
        print(f"{k:>6}{a:>22.14f}{score(a):>13.5f}{b:>22.14f}{score(b):>13.5f}",
              flush=True)
    print("\nQ1  per-seed block argmin: "
          + json.dumps(ans_q1["per_seed_block_argmin"])
          + f"   mean-surface argmin {ans_q1['mean_block_argmin']}"
          + f"   reconfirmed {ans_q1['shipped_choice_reconfirmed_on_three_seeds']}",
          flush=True)
    print("Q2  crossing (block mean): "
          + json.dumps({"any": ans_q2["block_mean"]["any_crossing"],
                        "crossings": ans_q2["block_mean"]["crossings"],
                        "winner_by_k": ans_q2["block_mean"]["winner_by_k"]}), flush=True)
    for tg in TARGETS:
        d = q3["per_target"][tg]
        print(f"Q3  {tg:>6}: block {d['block_selected']} vs oracle "
              f"{d['fold_2024_oracle_DIAGNOSTIC']}  cost "
              f"{d['cost_of_honest_selection_score']:+.5f}", flush=True)
    print(f"Q3  all20: block {q3['all_20_cells']['block_selected']} vs oracle "
          f"{q3['all_20_cells']['fold_2024_oracle_DIAGNOSTIC']}  cost "
          f"{q3['all_20_cells']['cost_of_honest_selection_score']:+.5f}", flush=True)
    for tg in TARGETS:
        e = q4["block_mean"][tg]
        print(f"Q4  block {tg:>6}: argmin k={e['argmin_k']}  enclosed "
              f"{e['enclosed_both_sides']}", flush=True)
        e = q4["fold_2024_mean_DIAGNOSTIC"][tg]
        print(f"Q4  fold  {tg:>6}: argmin k={e['argmin_k']}  enclosed "
              f"{e['enclosed_both_sides']}", flush=True)
    print(f"\n[{time.strftime('%H:%M:%S')}] DONE {metrics['compute_time_min']} min",
          flush=True)


if __name__ == "__main__":
    main()
