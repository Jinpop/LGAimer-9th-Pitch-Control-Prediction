"""exp_033 — every correction axis that was CLOSED on the old base (LGBM_B3),
re-measured on the NEW base (CatBoost C5_B3, exp_030 + exp_031).
NO TRAINING HAPPENS IN THIS FILE.

WHY IT EXISTS
-------------
exp_025/exp_026/exp_028 closed four axes on LGBM_B3's residual: the logit
coordinate system, the reverse-rate deviation `d_r` (solo and jointly with
`d_s`), the pitcher-hand-2 league ablation, and the backoff coverage variants.
Two facts already show that a verdict reached on one base does not automatically
carry to another:

  (a) exp_031 measured the BACKOFF TARGET preference flipping with the base --
      the old base picked `league`, the new base's block surface picks `self`;
  (b) the offset's whole share shrank from 74.45241 to 38.28802 score points
      when the base improved.

So every one of those axes is swept again here, from scratch, on the new
residual.  A negative result is a result: if an axis is dead on both bases, that
is worth knowing and is reported as such.

WHAT IS CHOSEN, AND WHERE
-------------------------
Only two things are chosen, and both are chosen on the SUPPLY BLOCKS
(2022 + 2023) and nowhere else:
  * DR1's (k, backoff target) cell, on the block Brier surface;
  * LG2's gamma, on the block Brier curve.
Everything else (R0, R1, LG1, DR2, AB1, AB2, BK1, BK2) is PRE-REGISTERED in
config.yaml -- those are not chosen, they are measured, so their fold_2024
numbers are reported as they fall.

THE CONTROL IS R1, NOT R0
-------------------------
The adoption rule is exp_024's, unrelaxed: three seeds agreeing in sign AND
|mean delta| >= 3 x the CONTROL's seed std.  Here the control is R1 (the
offset-corrected base), because every arm is a variation ON the offset.  R1's
seed std is LARGER than R0's, so the bar is HIGHER than exp_031's.  Both
multiples are reported and are labelled so they cannot be confused:
MULTIPLE_OF_THE_THRESHOLD vs MULTIPLE_OF_THE_CONTROL_STD differ by exactly 3x.

BLOCKS COME FROM exp_031, NOT exp_030
--------------------------------------
exp_030's supply-block predictions were built under a broken iteration policy
(they cost 102-126 score points on the 2022 block).  exp_031 regenerated them
under their own early stop; those four `_st3` vectors are the beta supply here,
and their sha256 are asserted against exp_031's own record.

GATES (every expectation is READ from a record, never transcribed)
------------------------------------------------------------------
 G0  uncorrected C5_B3 reproduces exp_030's fold_2024 Brier, all three seeds.
 G1  R1 reproduces exp_031's S400_self @ beta 0.59072, all three seeds, AND the
     regenerated prediction vectors are sha256-identical to exp_031's.
 G2  k0_self and k0_league are the same vector -- on BOTH axes.
 G3  `d == 0.0` exactly on cell-miss rows, on every k of both grids.
 G4  every deviation is constant inside its (season, pitcher, batter_hand,
     pitcher_hand) lookup group -- no row-level label reached a row's own d.
 G5  as-of, re-verified from a SECOND independent read of train.csv.
 G6  devgen's generalised builders, run at key="succ", are BIT-IDENTICAL to the
     byte-preserved dev_k / backoff builders.  That is what licenses using them
     at key="rev".
 G7  the two supply-block record files agree (the local exp_031 metrics.json
     copy uploaded here vs the remote metrics_grid.json the run wrote).

INHERITANCE
-----------
`dev_k.py`, `backoff.py`, `offset.py`, `features_platoon.py`, `labels.py`,
`exp016_ref.py`, `devgen.py` are BYTE-PRESERVED and their sha256 asserted before
anything runs.  `grid_c5.py` (exp_031) is carried and hashed as provenance; it
is not imported.
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
import devgen as dg                                               # noqa: E402
import exp016_ref as ref                                          # noqa: E402
import labels as lb                                               # noqa: E402
from labels import recover                                        # noqa: E402

ROOT = os.path.expanduser("~/SungjinPark")
DATA_PATH = os.path.join(ROOT, "data", "train.csv")
OUT_DIR = os.path.join(ROOT, "experiments", "exp_033")
EXP025_METRICS = os.path.join(ROOT, "experiments", "exp_025", "metrics.json")
EXP026_METRICS = os.path.join(ROOT, "experiments", "exp_026", "metrics.json")
EXP028_METRICS = os.path.join(ROOT, "experiments", "exp_028", "metrics.json")
EXP030_DIR = os.path.join(ROOT, "experiments", "exp_030")
EXP030_METRICS = os.path.join(EXP030_DIR, "metrics.json")
EXP031_DIR = os.path.join(ROOT, "experiments", "exp_031")
EXP031_REMOTE_METRICS = os.path.join(EXP031_DIR, "metrics_grid.json")
EXP031_LOCAL_COPY = os.path.join(OUT_DIR, "ref", "exp_031_metrics.json")

FOLD_SEEDS = [42, 7, 2024]
BLOCK_SEEDS = [42, 7]
GATE_TOL = 5e-6

TARGET = "control_success"
DROP_COLS = ["row_id", TARGET]
F_EXCLUDE_MAX_SEASON = 2022
SUPPLY_BLOCKS = [2022, 2023]
VALID_SEASON = 2024
EPS = 1e-6
BSS_REF_BRIER = 0.2498069

# the success axis: exp_031's declared grid, unchanged (only k=400 is used as a
# cell here, but the whole grid is built so GATE 2/3 cover it)
K_GRID_S = [0, 30, 55, 111, 200, 400, 800, 1600, 3200]
# the reverse axis: the UNION of exp_028's d_r grid (which has 65, the RATE_SPEC
# default for `rev`) and exp_031's d_s grid (which has 55).  Declared here, in
# advance, as the registered DR1 search space.
K_GRID_R = [0, 30, 55, 65, 111, 200, 400, 800, 1600, 3200]
K_RESERVE = [6400, 12800]      # enclosure only; not part of either declared grid
K_ALL_S = K_GRID_S + K_RESERVE
K_ALL_R = K_GRID_R + K_RESERVE
TARGETS = ["self", "league"]

R1_CELL = "S400_self"           # exp_031's block-selected cell = the control
BK1_CELL = "S400_self"          # hard backoff at the control's k
BK1B_CELL = "S111_self"         # the OLD candidate-H k, diagnostic only
BK2_CELL = "S400_league"        # exp_031's fold oracle cell
OLD_BASE_CELL = "S400_league"   # what LGBM_B3 selected
PH_ABLATE = 2                   # the unstable pitcher-hand cell

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
    "grid_c5.py": "03cbfbf1d1d9deba826ac06cb9b61b15d97f407c4670de5a723e6a27f8639324",
}


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
    return round(100000.0 * (1.0 - brier / BSS_REF_BRIER), 5)


def score_delta(dbrier):
    """A Brier CHANGE expressed in leaderboard points.  Sign convention: a
    NEGATIVE Brier change (an improvement) becomes a POSITIVE point gain."""
    return round(-100000.0 * float(dbrier) / BSS_REF_BRIER, 5)


def seg(y, p):
    m = ref.seg_metrics(y, p)
    if len(y):
        b = float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))
        m["brier_full"] = b
        m["score_vs_fixed_ref"] = score(b)
    return m


def cell_name(prefix, k, tg):
    return f"{prefix}{k}_{tg}"


def argmin_of(vals):
    return min(vals, key=vals.get)


def enclosure(vals_by_k, ks):
    v = [vals_by_k[k] for k in ks]
    j = int(np.argmin(v))
    return {
        "argmin_k": ks[j], "argmin_index": j, "n_points": len(ks),
        "enclosed_both_sides": bool(0 < j < len(ks) - 1),
        "value_at_argmin": v[j],
        "score_drop_to_lower_neighbour": (score_delta(v[j - 1] - v[j])
                                          if j > 0 else None),
        "score_drop_to_upper_neighbour": (score_delta(v[j + 1] - v[j])
                                          if j < len(ks) - 1 else None),
        "verdict": ("interior minimum — the grid brackets it"
                    if 0 < j < len(ks) - 1 else
                    "EDGE minimum — the grid does not bracket it; widen"),
    }


# --------------------------------------------------------------------------- #
def main():
    t_start = time.time()
    for d in ("curves", "pred", "ref"):
        os.makedirs(os.path.join(OUT_DIR, d), exist_ok=True)

    csv_sha = sha256_file(DATA_PATH)
    assert csv_sha == TRAIN_CSV_SHA, ("train.csv SHA MISMATCH", csv_sha)
    src_sha = {f: sha256_file(os.path.join(SRC_DIR, f))
               for f in sorted(os.listdir(SRC_DIR)) if f.endswith(".py")}
    for f, want in SHA_INHERITED.items():
        assert src_sha[f] == want, ("INHERITED SOURCE DRIFTED", f, src_sha[f], want)
    print("inherited sha OK", flush=True)

    # ---- the records.  Every gate number is READ, never transcribed. -------- #
    with open(EXP030_METRICS) as f:
        m30 = json.load(f)
    assert m30["exp_id"] == "exp_030"
    assert m30["reproducibility"]["train_csv_sha256"] == csv_sha
    GATE_R0 = {s: b for s, b in zip(m30["fold_2024"]["seeds"],
                                    m30["fold_2024"]["candidate_brier_by_seed"])}
    assert set(GATE_R0) == set(FOLD_SEEDS), GATE_R0
    GATE_R0_MEAN = m30["fold_2024"]["candidate_mean_brier"]
    FOLD_SHA = {int(k.split("seed")[1]): v["sha256"]
                for k, v in m30["predictions_recovered_locally"].items()
                if k.startswith("u2024_")}

    with open(EXP031_LOCAL_COPY) as f:
        m31 = json.load(f)
    with open(EXP031_REMOTE_METRICS) as f:
        m31r = json.load(f)
    assert m31["exp_id"] == "exp_031" and m31r["exp_id"] == "exp_031"
    # GATE 7 — the local copy uploaded here must be the file the run wrote
    gate7 = {"what": "the exp_031 metrics.json copied up from the laptop vs the "
                     "metrics_grid.json the exp_031 run left on the server.  The "
                     "laptop copy carries an extra report_derived block, so only "
                     "the sections this file READS are compared.",
             "sections_compared": ["adoption", "block_provenance",
                                   "fold_2024_base_provenance", "block_fit",
                                   "selection", "p_distribution_fold_2024",
                                   "candidate_predictions", "surfaces"],
             "local_sha256": sha256_file(EXP031_LOCAL_COPY),
             "remote_sha256": sha256_file(EXP031_REMOTE_METRICS)}
    for k in gate7["sections_compared"]:
        assert json.dumps(m31[k], sort_keys=True) == json.dumps(m31r[k],
                                                                sort_keys=True), k
    gate7["all_read_sections_identical"] = True
    print("GATE 7 OK — exp_031 record copies agree", flush=True)

    GATE_R1 = {s: b for s, b in zip(m31["adoption"]["seeds"],
                                    m31["adoption"]["candidate_brier_by_seed"])}
    GATE_R1_MEAN = m31["adoption"]["candidate_mean_brier"]
    GATE_R1_BETA = m31["adoption"]["beta_applied"]
    assert m31["adoption"]["selected_cell"] == R1_CELL, m31["adoption"]
    R1_PRED_SHA = {int(s): v["sha256"]
                   for s, v in m31["candidate_predictions"].items()}
    BLK_SHA = {(int(k.split("_")[1]), int(k.split("seed")[1])): v["sha256"]
               for k, v in m31["block_provenance"].items()}
    EXP031_BETA_CELL = {nm: v["beta_pooled"] for nm, v in m31["block_fit"].items()}
    EXP031_BLOCK_SURF = m31["surfaces"]["block_brier"]

    with open(EXP025_METRICS) as f:
        m25 = json.load(f)
    with open(EXP026_METRICS) as f:
        m26 = json.load(f)
    with open(EXP028_METRICS) as f:
        m28 = json.load(f)
    assert (m25["exp_id"], m26["exp_id"], m28["exp_id"]) == ("exp_025", "exp_026",
                                                             "exp_028")
    print("records read (exp_025/026/028/030/031)", flush=True)

    # ---- data --------------------------------------------------------------- #
    print(f"[{time.strftime('%H:%M:%S')}] loading {DATA_PATH}", flush=True)
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    assert df["row_id"].is_unique
    assert df["season"].is_monotonic_increasing
    provided_cols = [c for c in df.columns if c not in DROP_COLS]
    assert len(provided_cols) == 47

    rec = recover(df, verbose=False)
    assert rec["success_recovery"]["agree_pct"] == 100.0
    lb.assert_no_recovered_in_features(provided_cols, where="provided 47 columns")
    recovered_present = [c for c in df.columns if c in lb.RECOVERED_COLS
                         or c.startswith(lb.RECOVERED_PREFIXES)]
    label_isolation = {
        "recovered_columns_present_in_frame": len(recovered_present),
        "recovered_columns_reaching_any_input_list": len(
            [c for c in provided_cols if c in recovered_present]),
        "provided_feature_columns": len(provided_cols),
        "check_a_no_recovered_in_provided_columns": True,
        "note": "no model is fitted here.  The recovered labels enter ONLY as the "
                "numerators of frozen season-as-of tables (`succ` for d_s, `rev` "
                "for d_r); check (b) below proves no row-level label reached a "
                "row's own deviation.",
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

    # ---- deviations: both axes.  Seed-independent -- they come from labels. -- #
    print(f"[{time.strftime('%H:%M:%S')}] platoon lookups", flush=True)
    fp.build_season_forward_chained(df, ok, succ_season, f_excl, verbose=False)
    d_s, hit, n_cell0, _, ds_diag = dk.build_multi_k(df, ok, succ_season, f_excl,
                                                     K_ALL_S, verbose=False)
    L_s, has_Ls, _, _ = bk.build_league_dev(df, ok, succ_season, f_excl,
                                            verbose=False)
    Ls_eff = np.where(has_Ls, L_s, 0.0)

    # GATE 6 -- devgen at key="succ" must be BIT-IDENTICAL to the audited modules
    g_ds, g_hit, g_n, _, _ = dg.build_multi_k(df, ok, succ_season, f_excl,
                                              [0, 400], "succ", verbose=False)
    g_L, g_has, _, _ = dg.build_league_dev(df, ok, succ_season, f_excl, "succ",
                                           verbose=False)
    gate6 = {
        "what": "devgen.py is dev_k.build_multi_k / backoff.build_league_dev with "
                "the numerator key lifted to an argument.  Running it at "
                "key='succ' must reproduce the audited modules BIT for BIT; that "
                "is what licenses running it at key='rev'.",
        "d_s_k0_bit_identical": bool(np.array_equal(g_ds[0], d_s[0])),
        "d_s_k400_bit_identical": bool(np.array_equal(g_ds[400], d_s[400])),
        "hit_mask_bit_identical": bool(np.array_equal(g_hit, hit)),
        "n_cell_bit_identical": bool(np.array_equal(np.nan_to_num(g_n, nan=-1),
                                                    np.nan_to_num(n_cell0, nan=-1))),
        "league_L_bit_identical": bool(np.array_equal(g_L, L_s)),
        "league_has_bit_identical": bool(np.array_equal(g_has, has_Ls)),
    }
    assert all(v for k, v in gate6.items() if k != "what"), gate6
    print("GATE 6 OK — devgen@succ is bit-identical to the audited builders",
          flush=True)

    d_r, hit_r, n_cell0_r, _, dr_diag = dg.build_multi_k(df, ok, succ_season,
                                                         f_excl, K_ALL_R, "rev",
                                                         verbose=False)
    L_r, has_Lr, _, _ = dg.build_league_dev(df, ok, succ_season, f_excl, "rev",
                                            verbose=False)
    Lr_eff = np.where(has_Lr, L_r, 0.0)
    # the reverse axis shares the personal-cell geometry with the success axis
    assert np.array_equal(hit_r, hit)
    assert np.array_equal(np.nan_to_num(n_cell0_r, nan=-1),
                          np.nan_to_num(n_cell0, nan=-1))

    # ---- GATE 3 (both axes) -------------------------------------------------- #
    gate3 = {"d_s": {str(k): bool(np.all(d_s[k][~hit] == 0.0)) for k in K_ALL_S},
             "d_r": {str(k): bool(np.all(d_r[k][~hit] == 0.0)) for k in K_ALL_R}}
    assert all(gate3["d_s"].values()) and all(gate3["d_r"].values()), gate3
    gate3["n_k_checked"] = len(K_ALL_S) + len(K_ALL_R)
    gate3["n_miss_rows"] = int((~hit).sum())
    print(f"GATE 3 OK — d == 0.0 on miss rows for all {gate3['n_k_checked']} "
          f"(axis, k) pairs", flush=True)

    # ---- arm terms ----------------------------------------------------------- #
    S = {cell_name("S", k, tg): dk.build_arm(d_s[k], hit, Ls_eff, n_cell0, k, tg)
         for k in K_ALL_S for tg in TARGETS}
    R = {cell_name("R", k, tg): dk.build_arm(d_r[k], hit, Lr_eff, n_cell0, k, tg)
         for k in K_ALL_R for tg in TARGETS}
    CELLS_R = [cell_name("R", k, tg) for k in K_GRID_R for tg in TARGETS]
    CELLS_R_RES = [cell_name("R", k, tg) for k in K_RESERVE for tg in TARGETS]
    assert len(CELLS_R) == 20, len(CELLS_R)

    # ---- GATE 2 (both axes) + structural asserts ---------------------------- #
    gate2 = {"d_s": bool(np.array_equal(S["S0_self"], S["S0_league"])),
             "d_r": bool(np.array_equal(R["R0_self"], R["R0_league"]))}
    assert gate2["d_s"] and gate2["d_r"], gate2
    for k in K_ALL_S:
        assert np.array_equal(S[cell_name("S", k, "self")][~hit],
                              S[cell_name("S", k, "league")][~hit]), k
    for k in K_ALL_R:
        assert np.array_equal(R[cell_name("R", k, "self")][~hit],
                              R[cell_name("R", k, "league")][~hit]), k
    print("GATE 2 OK — k0_self == k0_league on both axes", flush=True)

    # ---- GATE 4: label isolation -------------------------------------------- #
    gcodes = pd.DataFrame({"s": season, "p": df[fp.PITCHER].to_numpy(),
                           "h": df[fp.HAND].to_numpy(), "ph": ph_all}
                          ).groupby(["s", "p", "h", "ph"], sort=False,
                                    dropna=False).ngroup().to_numpy()
    const = {}
    checked = (list(S.items()) + list(R.items())
               + [(f"d_s_raw_k{k}", d_s[k]) for k in K_ALL_S]
               + [(f"d_r_raw_k{k}", d_r[k]) for k in K_ALL_R]
               + [("L_success", Ls_eff), ("L_reverse", Lr_eff)])
    for nm, v in checked:
        first = pd.Series(v).groupby(gcodes, sort=False).transform("first").to_numpy()
        const[nm] = bool(np.array_equal(v, first))
    label_isolation["check_b_constant_within_season_pitcher_hand"] = const
    label_isolation["n_lookup_groups"] = int(gcodes.max() + 1)
    label_isolation["n_vectors_checked"] = len(const)
    assert all(const.values()), ("a deviation varies inside its lookup group",
                                 [k for k, v in const.items() if not v])
    print(f"GATE 4 OK ({len(const)} vectors, {label_isolation['n_lookup_groups']} "
          f"groups)", flush=True)

    # ---- GATE 5: as-of, independently re-verified --------------------------- #
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
    asof_indep = {"method": "row_id -> season re-read from train.csv in a SECOND "
                            "pass; the max season of the CONTRIBUTING ROW IDS is "
                            "looked up in that independent map.  This covers both "
                            "numerators (`succ` and `rev`) because both tables use "
                            "the identical row mask.",
                  "frame_season_matches_file": frame_matches_file,
                  "tables_covered": ["personal_succ", "personal_rev",
                                     "league_succ", "league_rev"],
                  "per_season": {}}
    for s in sorted(set(season.tolist()))[1:]:
        m_pers = base_ok & (season <= s - 1) & (succ_season <= s - 1)
        m_leag = m_pers & hand_ok
        rec_s = {}
        for name, m in (("personal", m_pers), ("league", m_leag)):
            mx = int(s_map.reindex(pd.Index(row_id[m])).max())
            assert mx <= s - 1, ("AS-OF VIOLATION (independent)", name, s, mx)
            rec_s[name] = {"contributing_rows": int(m.sum()),
                           "max_season_of_contributors_independent": mx,
                           "allowed_max": int(s - 1)}
        asof_indep["per_season"][str(int(s))] = rec_s
    assert max(SUPPLY_BLOCKS) < VALID_SEASON
    print("GATE 5 (as-of, independent) OK", flush=True)

    # ---- slices ------------------------------------------------------------- #
    ALL_V = SUPPLY_BLOCKS + [VALID_SEASON]
    y = {V: y_all[season == V] for V in ALL_V}
    m24 = (season == VALID_SEASON)
    y24 = y[VALID_SEASON]
    hit24, isr24, ph24 = hit[m24], is_r_all[m24], ph_all[m24]
    assert len(y24) == 253507 and int(isr24.sum()) == 223497

    TS = {nm: {V: v[season == V] for V in ALL_V} for nm, v in S.items()}
    TR = {nm: {V: v[season == V] for V in ALL_V} for nm, v in R.items()}

    # ---- base vectors ------------------------------------------------------- #
    blk, blk_prov = {}, {}
    for s in BLOCK_SEEDS:
        blk[s] = {}
        for V in SUPPLY_BLOCKS:
            stem = f"C5B3_u{V}_seed{s}_st3"
            pth = os.path.join(EXP031_DIR, "pred", f"pred_{stem}.npy")
            got = sha256_file(pth)
            assert got == BLK_SHA[(V, s)], ("block pred sha != exp_031 record",
                                            V, s, got)
            arr = np.load(pth)
            assert arr.shape == y[V].shape
            blk[s][V] = arr
            blk_prov[f"block_{V}_seed{s}"] = {
                "path": pth, "sha256": got,
                "source": "exp_031 block_self.py stage 3 (own early stop) — NOT "
                          "exp_030's stage-2 vectors, whose iteration policy was "
                          "broken",
                "brier": float(np.mean((arr - y[V]) ** 2))}

    p24, fold_prov = {}, {}
    for s in FOLD_SEEDS:
        pth = os.path.join(EXP030_DIR, "pred", f"pred_C5B3_u2024_seed{s}_st2.npy")
        got = sha256_file(pth)
        assert got == FOLD_SHA[s], ("fold_2024 base sha != exp_030 record", s, got)
        arr = np.load(pth)
        assert arr.shape == y24.shape
        p24[s] = arr
        fold_prov[str(s)] = {"path": pth, "sha256": got,
                             "reused_from": "exp_030 b3_cat.py stage 2 — NOT "
                                            "retrained"}

    # ---- GATE 0: R0 ---------------------------------------------------------- #
    base24 = {s: float(np.mean((p24[s] - y24) ** 2)) for s in FOLD_SEEDS}
    gate0 = {"tolerance": GATE_TOL,
             "source": "exp_030/metrics.json fold_2024.candidate_brier_by_seed — "
                       "READ, not transcribed", "checks": {}}
    for s in FOLD_SEEDS:
        d0 = abs(base24[s] - GATE_R0[s])
        gate0["checks"][str(s)] = {"expected_from_exp030": GATE_R0[s],
                                   "got": base24[s], "abs_diff": d0,
                                   "bit_identical": bool(base24[s] == GATE_R0[s]),
                                   "passed": bool(d0 < GATE_TOL)}
    r0_mean = float(np.mean([base24[s] for s in FOLD_SEEDS]))
    r0_std = float(np.std([base24[s] for s in FOLD_SEEDS], ddof=1))
    gate0["mean_got"] = r0_mean
    gate0["mean_expected_from_exp030"] = GATE_R0_MEAN
    gate0["mean_abs_diff"] = abs(r0_mean - GATE_R0_MEAN)
    gate0["mean_score_got"] = score(r0_mean)
    gate0["passed"] = bool(all(v["passed"] for v in gate0["checks"].values())
                           and gate0["mean_abs_diff"] < GATE_TOL)
    assert gate0["passed"], json.dumps(gate0)
    print(f"GATE 0 OK — R0 {r0_mean:.14f} ({score(r0_mean)})", flush=True)

    # ---- supply pooling ------------------------------------------------------ #
    sup_index = [(V, s) for V in SUPPLY_BLOCKS for s in BLOCK_SEEDS]
    y_sup = np.concatenate([y[V] for V, s in sup_index])
    p_sup = np.concatenate([blk[s][V] for V, s in sup_index])
    r_sup = y_sup - p_sup
    br_base_sup = float(np.mean((p_sup - y_sup) ** 2))

    def pool_S(nm):
        return np.concatenate([TS[nm][V] for V, s in sup_index])

    def pool_R(nm):
        return np.concatenate([TR[nm][V] for V, s in sup_index])

    def pool_vec(dct, nm):
        return np.concatenate([dct[nm][V] for V, s in sup_index])

    # ---- GATE 1: R1 ---------------------------------------------------------- #
    d_sel_sup = pool_S(R1_CELL)
    beta_r1 = off.fit_slope(r_sup, d_sel_sup)["beta"]
    gate1 = {"cell": R1_CELL,
             "beta_refitted_here": beta_r1,
             "beta_from_exp031_record": GATE_R1_BETA,
             "beta_bit_identical": bool(beta_r1 == GATE_R1_BETA),
             "tolerance": GATE_TOL, "checks": {}, "pred_sha256": {}}
    assert beta_r1 == GATE_R1_BETA, (beta_r1, GATE_R1_BETA)
    r1_pred = {}
    for s in FOLD_SEEDS:
        pv = off.apply_linear(p24[s], TS[R1_CELL][VALID_SEASON], beta_r1)
        r1_pred[s] = pv
        b = float(np.mean((pv - y24) ** 2))
        gate1["checks"][str(s)] = {"expected_from_exp031": GATE_R1[s], "got": b,
                                   "abs_diff": abs(b - GATE_R1[s]),
                                   "bit_identical": bool(b == GATE_R1[s]),
                                   "passed": bool(abs(b - GATE_R1[s]) < GATE_TOL)}
        pth = os.path.join(OUT_DIR, "pred",
                           f"regen_C5B3_offset_{R1_CELL}_fold2024_seed{s}.npy")
        np.save(pth, pv)
        got = sha256_file(pth)
        gate1["pred_sha256"][str(s)] = {
            "regenerated_here": got, "exp031_record": R1_PRED_SHA[s],
            "bit_identical": bool(got == R1_PRED_SHA[s])}
    r1_by_seed = {s: gate1["checks"][str(s)]["got"] for s in FOLD_SEEDS}
    r1_mean = float(np.mean([r1_by_seed[s] for s in FOLD_SEEDS]))
    r1_std = float(np.std([r1_by_seed[s] for s in FOLD_SEEDS], ddof=1))
    gate1["mean_got"] = r1_mean
    gate1["mean_expected_from_exp031"] = GATE_R1_MEAN
    gate1["mean_abs_diff"] = abs(r1_mean - GATE_R1_MEAN)
    gate1["mean_score_got"] = score(r1_mean)
    gate1["all_pred_vectors_sha_identical"] = bool(
        all(v["bit_identical"] for v in gate1["pred_sha256"].values()))
    gate1["passed"] = bool(all(v["passed"] for v in gate1["checks"].values())
                           and gate1["mean_abs_diff"] < GATE_TOL)
    assert gate1["passed"], json.dumps(gate1)
    print(f"GATE 1 OK — R1 {r1_mean:.14f} ({score(r1_mean)}), pred sha identical "
          f"{gate1['all_pred_vectors_sha_identical']}", flush=True)

    # ---- p distribution and the two coordinate systems ---------------------- #
    def pdesc(pv):
        v = pv * (1.0 - pv)
        return {"n": int(len(pv)), "mean": float(pv.mean()), "sd": float(pv.std()),
                "min": float(pv.min()), "max": float(pv.max()),
                "p01": float(np.percentile(pv, 1)),
                "p99": float(np.percentile(pv, 99)),
                "v_mean": float(v.mean()), "v_sd": float(v.std()),
                "v_cv": float(v.std() / v.mean()),
                "v_min": float(v.min()), "v_max": float(v.max()),
                "v_max_over_min": float(v.max() / v.min())}

    pdist = {"fold_2024_by_seed": {str(s): pdesc(p24[s]) for s in FOLD_SEEDS},
             "supply_blocks_pooled": pdesc(p_sup)}
    pdist["fold_2024_3seed_mean_of_sd"] = float(np.mean(
        [pdist["fold_2024_by_seed"][str(s)]["sd"] for s in FOLD_SEEDS]))
    pdist["fold_2024_3seed_mean_of_v_cv"] = float(np.mean(
        [pdist["fold_2024_by_seed"][str(s)]["v_cv"] for s in FOLD_SEEDS]))
    pdist["old_base_LGBM_B3_from_exp028_seed42"] = {
        "sd": m28["p_distribution"]["42"]["fold_2024"]["sd"],
        "min": m28["p_distribution"]["42"]["fold_2024"]["min"],
        "max": m28["p_distribution"]["42"]["fold_2024"]["max"],
        "v_cv": m28["p_distribution"]["42"]["fold_2024"]["v_cv"],
        "v_max_over_min": m28["p_distribution"]["42"]["fold_2024"]["v_max_over_min"],
        "source": "exp_028/metrics.json p_distribution.42.fold_2024 — READ"}
    # the ONLY thing the logit space can buy beyond a rescaling of beta
    dsel24 = TS[R1_CELL][VALID_SEASON]
    coord = {}
    for tag, pv, dv in (("fold_2024_seed42", p24[42], dsel24),
                        ("supply_blocks_pooled", p_sup, d_sel_sup)):
        vmean = float(np.mean(pv * (1.0 - pv)))
        g_eq = beta_r1 / vmean
        lin = off.apply_linear(pv, dv, beta_r1)
        lg = off.apply_logit(pv, dv, g_eq)
        raw = pv + beta_r1 * dv
        coord[tag] = {
            "beta_used": beta_r1, "mean_p1mp": vmean, "gamma_equivalent": g_eq,
            "mean_abs_correction_linear": float(np.mean(np.abs(lin - pv))),
            "max_abs_correction_linear": float(np.max(np.abs(lin - pv))),
            "mean_abs_divergence_linear_vs_logit": float(np.mean(np.abs(lin - lg))),
            "max_abs_divergence_linear_vs_logit": float(np.max(np.abs(lin - lg))),
            "n_rows_clipped_linear": int(((raw < EPS) | (raw > 1 - EPS)).sum()),
            "n_rows_p_clipped_before_logit": int(((pv < EPS) | (pv > 1 - EPS)).sum()),
            "raw_min": float(raw.min()), "raw_max": float(raw.max()),
        }
        coord[tag]["divergence_over_correction_mean"] = (
            coord[tag]["mean_abs_divergence_linear_vs_logit"]
            / coord[tag]["mean_abs_correction_linear"])
        coord[tag]["divergence_over_correction_max"] = (
            coord[tag]["max_abs_divergence_linear_vs_logit"]
            / coord[tag]["max_abs_correction_linear"])
    coord["note"] = ("v = p(1-p) is the local slope dp/dz of the logit map.  When "
                     "its coefficient of variation is small, sigmoid(logit(p)+g*d) "
                     "is p + g*v_mean*d to first order, i.e. the linear arm with "
                     "beta = g*v_mean -- the two coordinate systems then cannot "
                     "meaningfully disagree.  The clip is NOT the mechanism here: "
                     "n_rows_clipped_linear is reported and is 0.")
    coord["old_base_from_exp028_seed42"] = {
        k: m28["p_distribution"]["42"]["coordinate_divergence"][k]
        for k in ("beta_used", "gamma_equivalent", "mean_abs_correction_linear",
                  "max_abs_correction_linear", "mean_abs_divergence",
                  "max_abs_divergence", "divergence_over_correction_mean",
                  "divergence_over_correction_max")}
    coord["old_base_clip_probe_from_exp028_seed42"] = \
        m28["p_distribution"]["42"]["clip_probe_linear_A1"]

    # ---- d_s / d_r correlation — MEASURED HERE, not transcribed ------------- #
    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    hit_sup = np.concatenate([hit[season == V] for V, s in sup_index])
    corr_tbl = {"measured_on": "this run, on THIS base's deviation vectors.  The "
                               "deviations come from the labels only, so they do "
                               "NOT depend on the base -- which is exactly why the "
                               "old-base row below should agree, and is reported "
                               "next to it as a check rather than copied over.",
                "supply_blocks_2022_2023": {}, "fold_2024": {}}
    for tag, ds_v, dr_v, hv in (
            ("supply_blocks_2022_2023", pool_S(R1_CELL), pool_R("R400_league"),
             hit_sup),
            ("fold_2024", dsel24, TR["R400_league"][VALID_SEASON], hit24)):
        raw_s = np.concatenate([d_s[400][season == V] for V, s in sup_index]) \
            if tag.startswith("supply") else d_s[400][m24]
        raw_r = np.concatenate([d_r[400][season == V] for V, s in sup_index]) \
            if tag.startswith("supply") else d_r[400][m24]
        corr_tbl[tag] = {
            "corr_raw_d_s400_vs_raw_d_r400_hit_rows": corr(raw_s[hv], raw_r[hv]),
            "corr_raw_d_s400_vs_raw_d_r400_all_rows": corr(raw_s, raw_r),
            "corr_S400self_arm_vs_R400league_arm_all_rows": corr(ds_v, dr_v),
            "sd_raw_d_s400": float(np.std(raw_s)),
            "sd_raw_d_r400": float(np.std(raw_r)),
            "sd_S400self_arm": float(np.std(ds_v)),
            "cell_hit_pct": round(100.0 * float(hv.mean()), 3)}
    corr_tbl["old_base_exp026_exp028_READ"] = {
        "exp026_supply_corr_raw_d_s_vs_raw_d_r_k400_hit_rows":
            m26["d_s_d_r_correlation"]["supply_blocks_2022_2023"]
               ["corr_raw_d_s_vs_raw_d_r_k400_hit_rows"],
        "exp026_fold_corr_raw_d_s_vs_raw_d_r_k400_hit_rows":
            m26["d_s_d_r_correlation"]["fold_2024"]
               ["corr_raw_d_s_vs_raw_d_r_k400_hit_rows"],
        "exp028_supply_corr_A1col_vs_A2col_hit_rows":
            m28["d_s_d_r_correlation"]["supply_blocks_2022_2023"]
               ["corr_A1col_vs_A2col_hit_rows"],
        "note": "exp_026/exp_028 built the SAME tables from the SAME labels, so "
                "the raw-deviation correlations must agree with this run to "
                "float noise.  The arm-level correlations need not: the old base "
                "paired the d_s arm at k400 x LEAGUE, this run's control cell is "
                "k400 x SELF."}

    # ---- the arm machinery --------------------------------------------------- #
    def evaluate_pred(pv, s, tag):
        se = (pv - y24) ** 2
        br = float(np.mean(se))
        return {"seed": s, "brier": br, "score": score(br),
                "valid_R": seg(y24[isr24], pv[isr24]),
                "valid_F": seg(y24[~isr24], pv[~isr24]),
                "seg_hit_brier": float(np.mean(se[hit24])),
                "seg_miss_brier": float(np.mean(se[~hit24])),
                "ph1_brier": float(np.mean(se[ph24 == 1])),
                "ph2_brier": float(np.mean(se[ph24 == 2])),
                "shift_vs_R1_mean_abs": float(np.mean(np.abs(pv - r1_pred[s]))),
                "shift_vs_R1_max_abs": float(np.max(np.abs(pv - r1_pred[s]))),
                "shift_vs_base_mean_abs": float(np.mean(np.abs(pv - p24[s]))),
                "shift_vs_base_max_abs": float(np.max(np.abs(pv - p24[s]))),
                "arm": tag}

    ARMS = {}

    def register(tag, spec, preds, extra=None):
        per = {str(s): evaluate_pred(preds[s], s, tag) for s in FOLD_SEEDS}
        mb = float(np.mean([per[str(s)]["brier"] for s in FOLD_SEEDS]))
        rec = {"spec": spec, "per_seed": per, "mean_brier": mb,
               "mean_score": score(mb),
               "seed_std_ddof1": float(np.std(
                   [per[str(s)]["brier"] for s in FOLD_SEEDS], ddof=1)),
               "identical_to_R1_bitwise": bool(
                   all(np.array_equal(preds[s], r1_pred[s]) for s in FOLD_SEEDS))}
        if extra:
            rec.update(extra)
        ARMS[tag] = rec
        return rec

    # R0 ---------------------------------------------------------------------- #
    register("R0", "no correction — p as C5_B3 produced it",
             {s: p24[s] for s in FOLD_SEEDS})
    # R1 ---------------------------------------------------------------------- #
    register("R1", f"p + beta*d, cell {R1_CELL}, beta {beta_r1} (exp_031's "
                   f"block-selected configuration).  THE CONTROL.",
             r1_pred, {"beta": beta_r1, "cell": R1_CELL})

    # LG1 — the logit twin of R1, gamma from the bounded minimiser on the supply #
    fg = off.fit_gamma(p_sup, y_sup, d_sel_sup)
    gamma_hat = fg["gamma"]
    gamma_eq = coord["supply_blocks_pooled"]["gamma_equivalent"]
    register("LG1", f"sigmoid(logit(p) + gamma*d), cell {R1_CELL}, gamma "
                    f"{gamma_hat} from offset.fit_gamma on the supply blocks",
             {s: off.apply_logit(p24[s], dsel24, gamma_hat) for s in FOLD_SEEDS},
             {"gamma": gamma_hat, "cell": R1_CELL, "fit_gamma_record": fg,
              "gamma_equivalent_from_beta": gamma_eq,
              "gamma_hat_over_gamma_equivalent": gamma_hat / gamma_eq,
              "gamma_hat_minus_equivalent_pct": 100.0 * (gamma_hat / gamma_eq - 1.0)})

    # LG2 — gamma read off an EXPLICIT block grid (a supply-block selection) --- #
    g_hi = float(max(gamma_hat, gamma_eq))
    g_grid = sorted(set([float(round(float(x), 6))
                         for x in np.linspace(0.0, 2.0 * g_hi, 41)]
                        + [float(gamma_hat), float(round(gamma_eq, 6))]))
    gcurve = {}
    for g in g_grid:
        pb = off.apply_logit(p_sup, d_sel_sup, g)
        gcurve[str(g)] = {"block_brier": float(np.mean((pb - y_sup) ** 2))}
    g_star = float(min(g_grid, key=lambda g: gcurve[str(g)]["block_brier"]))
    for g in g_grid:                       # fold_2024 filled in AFTER the choice
        fb = [float(np.mean((off.apply_logit(p24[s], dsel24, g) - y24) ** 2))
              for s in FOLD_SEEDS]
        gcurve[str(g)]["fold_mean_brier"] = float(np.mean(fb))
        gcurve[str(g)]["fold_mean_score"] = score(float(np.mean(fb)))
    g_oracle = float(min(g_grid, key=lambda g: gcurve[str(g)]["fold_mean_brier"]))
    j_star = g_grid.index(g_star)
    register("LG2", f"LG1 with gamma re-searched on an explicit {len(g_grid)}-point "
                    f"block grid; block argmin gamma = {g_star}",
             {s: off.apply_logit(p24[s], dsel24, g_star) for s in FOLD_SEEDS},
             {"gamma": g_star, "cell": R1_CELL,
              "grid_n": len(g_grid), "grid_min": g_grid[0], "grid_max": g_grid[-1],
              "block_argmin_is_interior": bool(0 < j_star < len(g_grid) - 1),
              "gamma_hat_bounded_minimiser": gamma_hat,
              "gamma_equivalent_from_beta": gamma_eq,
              "fold_2024_oracle_gamma_DIAGNOSTIC": g_oracle,
              "honest_selection_cost_score_points": score_delta(
                  gcurve[str(g_star)]["fold_mean_brier"]
                  - gcurve[str(g_oracle)]["fold_mean_brier"])})

    # DR1 — the reverse axis, solo, with its OWN k x target block selection ---- #
    dr_fit = {}
    for nm in CELLS_R + CELLS_R_RES:
        dp = pool_R(nm)
        f1 = off.fit_slope(r_sup, dp)
        pb = off.apply_linear(p_sup, dp, f1["beta"])
        dr_fit[nm] = {"beta": f1["beta"], "se": f1["se"], "t": f1["t"],
                      "sd_d_blocks": float(np.std(dp)),
                      "block_brier": float(np.mean((pb - y_sup) ** 2))}
        dr_fit[nm]["block_score"] = score(dr_fit[nm]["block_brier"])
        dr_fit[nm]["block_delta_vs_base"] = (dr_fit[nm]["block_brier"]
                                             - br_base_sup)
    dr_blk = {nm: dr_fit[nm]["block_brier"] for nm in CELLS_R}
    DR_SEL = argmin_of(dr_blk)
    dr_fold = {}
    for nm in CELLS_R + CELLS_R_RES:
        fb = [float(np.mean((off.apply_linear(p24[s], TR[nm][VALID_SEASON],
                                              dr_fit[nm]["beta"]) - y24) ** 2))
              for s in FOLD_SEEDS]
        dr_fold[nm] = float(np.mean(fb))
    DR_ORACLE = argmin_of({nm: dr_fold[nm] for nm in CELLS_R})
    beta_dr = dr_fit[DR_SEL]["beta"]
    register("DR1", f"p + beta_r*d_r, cell {DR_SEL} chosen on the block surface "
                    f"({len(CELLS_R)} declared cells), beta_r {beta_dr}",
             {s: off.apply_linear(p24[s], TR[DR_SEL][VALID_SEASON], beta_dr)
              for s in FOLD_SEEDS},
             {"cell": DR_SEL, "beta_r": beta_dr,
              "fold_2024_oracle_cell_DIAGNOSTIC": DR_ORACLE,
              "honest_selection_cost_score_points": score_delta(
                  dr_fold[DR_SEL] - dr_fold[DR_ORACLE]),
              "block_argmin_self_only": argmin_of(
                  {n: v for n, v in dr_blk.items() if n.endswith("_self")}),
              "block_argmin_league_only": argmin_of(
                  {n: v for n, v in dr_blk.items() if n.endswith("_league")})})
    dr_enc = {tg: enclosure({k: dr_fit[cell_name("R", k, tg)]["block_brier"]
                             for k in K_GRID_R}, K_GRID_R) for tg in TARGETS}
    dr_enc_wide = {tg: enclosure({k: dr_fit[cell_name("R", k, tg)]["block_brier"]
                                  for k in K_ALL_R}, K_ALL_R) for tg in TARGETS}

    # DR2 — d_s and d_r fitted JOINTLY ---------------------------------------- #
    cols_sup = [d_sel_sup, pool_R(DR_SEL)]
    jf = dg.fit_joint(r_sup, cols_sup)
    bs_j, br_j = jf["beta"]
    pj_blocks = dg.apply_joint(p_sup, cols_sup, jf["beta"])
    register("DR2", f"p + beta_s*d_s({R1_CELL}) + beta_r*d_r({DR_SEL}), both "
                    f"fitted jointly (2x2 normal equations, no intercept)",
             {s: dg.apply_joint(p24[s], [TS[R1_CELL][VALID_SEASON],
                                         TR[DR_SEL][VALID_SEASON]], jf["beta"])
              for s in FOLD_SEEDS},
             {"beta_s_joint": bs_j, "beta_r_joint": br_j,
              "beta_s_solo": beta_r1, "beta_r_solo": beta_dr,
              "beta_s_inflation_pct": 100.0 * (bs_j / beta_r1 - 1.0),
              "beta_r_sign_flipped": bool(np.sign(br_j) != np.sign(beta_dr)),
              "beta_r_ratio_joint_over_solo": (br_j / beta_dr
                                               if beta_dr != 0 else None),
              "corr_of_the_two_terms_on_supply": jf["corr_terms"][0][1],
              "cond_gram": jf["cond_gram"], "se": jf["se"], "t": jf["t"],
              "block_brier": float(np.mean((pj_blocks - y_sup) ** 2)),
              "block_score": score(float(np.mean((pj_blocks - y_sup) ** 2))),
              "old_base_exp026_C5_READ": {
                  "beta_joint": m26["per_seed"]["42"]["block_fit"]["C5"]["beta"],
                  "beta_s_solo_s400_league":
                      m26["per_seed"]["42"]["single_term_betas"]
                         ["s400_league"]["beta"],
                  "beta_r_solo_r400_league":
                      m26["per_seed"]["42"]["single_term_betas"]
                         ["r400_league"]["beta"],
                  "corr_terms": m26["per_seed"]["42"]["block_fit"]["C5"]
                                   ["corr_terms"][0][1],
                  "note": "the old base paired d_s at k400 x LEAGUE with d_r at "
                          "k400 x LEAGUE; this run pairs the cells its own "
                          "surfaces chose.  Compare the PATTERN (sign flip, "
                          "inflation), not the levels."}})

    # AB1 / AB2 — the pitcher-hand-2 league ablation --------------------------- #
    ph_is2 = (ph_all == PH_ABLATE)
    Ls_ab1 = np.where(ph_is2, 0.0, Ls_eff)                    # every ph2 row
    Ls_ab2 = np.where(ph_is2 & ~hit, 0.0, Ls_eff)             # ph2 MISS rows only
    k_sel = int(R1_CELL[1:].split("_")[0])
    tg_sel = R1_CELL.split("_")[1]
    arm_ab1 = dk.build_arm(d_s[k_sel], hit, Ls_ab1, n_cell0, k_sel, tg_sel)
    arm_ab2 = dk.build_arm(d_s[k_sel], hit, Ls_ab2, n_cell0, k_sel, tg_sel)
    arm_r1 = S[R1_CELL]
    # the same two ablations on the OLD base's cell, as a DIAGNOSTIC, to make the
    # "the ablated row count differs" claim quantitative rather than asserted
    arm_lg = S[BK2_CELL]
    k_lg = int(BK2_CELL[1:].split("_")[0])
    arm_lg_ab1 = dk.build_arm(d_s[k_lg], hit, Ls_ab1, n_cell0, k_lg, "league")
    arm_lg_ab2 = dk.build_arm(d_s[k_lg], hit, Ls_ab2, n_cell0, k_lg, "league")

    def rows_changed(a, b, mask=None):
        d = (a != b)
        return int(d.sum()) if mask is None else int((d & mask).sum())

    ablation_scope = {
        "rule_AB1": f"L := 0 on EVERY pitcher_hand == {PH_ABLATE} row "
                    f"(exp_025 B_ph1only's rule)",
        "rule_AB2": f"L := 0 only where pitcher_hand == {PH_ABLATE} AND the "
                    f"personal cell is EMPTY (exp_026 C1's rule)",
        "STRUCTURAL_NOTE": f"the control cell is {R1_CELL} — target `self`.  Under "
                           f"`self` the league term is added ONLY on cell-miss "
                           f"rows, so zeroing L on ph{PH_ABLATE} HIT rows is a "
                           f"no-op and AB1 and AB2 collapse onto the same arm.  "
                           f"That is a property of the NEW base's selected cell; "
                           f"on the old base's `league` cell they differ, and the "
                           f"league_cell_DIAGNOSTIC rows below measure by how "
                           f"much.",
        "AB1_equals_AB2_on_the_self_cell": bool(np.array_equal(arm_ab1, arm_ab2)),
        "counts_all_rows": {
            "n_rows_total": int(len(df)),
            "n_ph2": int(ph_is2.sum()),
            "n_ph2_miss": int((ph_is2 & ~hit).sum()),
            "n_ph2_hit": int((ph_is2 & hit).sum()),
            "n_ph2_miss_with_nonzero_L": int((ph_is2 & ~hit & (Ls_eff != 0)).sum()),
        },
        "counts_fold_2024": {
            "n_rows": int(m24.sum()),
            "n_ph2": int((ph24 == PH_ABLATE).sum()),
            "n_ph2_miss": int(((ph24 == PH_ABLATE) & ~hit24).sum()),
            "n_ph2_hit": int(((ph24 == PH_ABLATE) & hit24).sum()),
            "pct_of_hit_rows_that_are_ph2": round(
                100.0 * float(((ph24 == PH_ABLATE) & hit24).sum()
                              / max(int(hit24.sum()), 1)), 3),
        },
        "rows_the_ablation_actually_changes": {
            "self_cell_AB1_vs_R1_all_rows": rows_changed(arm_ab1, arm_r1),
            "self_cell_AB2_vs_R1_all_rows": rows_changed(arm_ab2, arm_r1),
            "self_cell_AB1_vs_R1_fold_2024": rows_changed(arm_ab1, arm_r1, m24),
            "league_cell_DIAGNOSTIC_AB1_vs_control_all_rows":
                rows_changed(arm_lg_ab1, arm_lg),
            "league_cell_DIAGNOSTIC_AB2_vs_control_all_rows":
                rows_changed(arm_lg_ab2, arm_lg),
            "league_cell_DIAGNOSTIC_AB1_vs_control_fold_2024":
                rows_changed(arm_lg_ab1, arm_lg, m24),
            "ratio_league_over_self_ablated_rows_all": None,
        },
        "every_changed_row_is_ph2": bool(np.all(ph_is2[arm_ab1 != arm_r1])),
        "old_base_exp026_C1_scope_READ": m26["C1_scope_check"],
    }
    _rc = ablation_scope["rows_the_ablation_actually_changes"]
    _rc["ratio_league_over_self_ablated_rows_all"] = (
        _rc["league_cell_DIAGNOSTIC_AB1_vs_control_all_rows"]
        / max(_rc["self_cell_AB1_vs_R1_all_rows"], 1))
    assert ablation_scope["every_changed_row_is_ph2"]

    ab_terms = {V: {} for V in ALL_V}
    for nm, vec in (("AB1", arm_ab1), ("AB2", arm_ab2),
                    ("LGAB1", arm_lg_ab1), ("LGAB2", arm_lg_ab2)):
        for V in ALL_V:
            ab_terms[V][nm] = vec[season == V]

    def fit_and_register(tag, spec, vec_by_V, extra=None):
        dp = np.concatenate([vec_by_V[V] for V, s in sup_index])
        b = off.fit_slope(r_sup, dp)["beta"]
        pb = off.apply_linear(p_sup, dp, b)
        ex = {"beta_refitted": b,
              "block_brier": float(np.mean((pb - y_sup) ** 2))}
        ex["block_score"] = score(ex["block_brier"])
        # sensitivity: the control's beta held fixed instead of refitted
        held = {s: off.apply_linear(p24[s], vec_by_V[VALID_SEASON], beta_r1)
                for s in FOLD_SEEDS}
        hb = [float(np.mean((held[s] - y24) ** 2)) for s in FOLD_SEEDS]
        ex["beta_held_at_R1_mean_brier"] = float(np.mean(hb))
        ex["beta_held_at_R1_mean_score"] = score(float(np.mean(hb)))
        if extra:
            ex.update(extra)
        return register(tag, spec,
                        {s: off.apply_linear(p24[s], vec_by_V[VALID_SEASON], b)
                         for s in FOLD_SEEDS}, ex)

    fit_and_register("AB1", f"R1's cell {R1_CELL} with L := 0 on every "
                            f"pitcher_hand-{PH_ABLATE} row; beta refitted on the "
                            f"supply",
                     {V: ab_terms[V]["AB1"] for V in ALL_V})
    fit_and_register("AB2", f"R1's cell {R1_CELL} with L := 0 on "
                            f"pitcher_hand-{PH_ABLATE} CELL-MISS rows only; beta "
                            f"refitted on the supply",
                     {V: ab_terms[V]["AB2"] for V in ALL_V})

    # BK1 / BK2 — backoff coverage -------------------------------------------- #
    beta_bk1 = off.fit_slope(r_sup, pool_S(BK1_CELL))["beta"]
    register("BK1", f"hard backoff: league term on cell-MISS rows only, pure "
                    f"personal deviation on hit rows, k = {k_sel} "
                    f"(cell {BK1_CELL}).  On this base that IS the control's cell.",
             {s: off.apply_linear(p24[s], TS[BK1_CELL][VALID_SEASON], beta_bk1)
              for s in FOLD_SEEDS},
             {"cell": BK1_CELL, "beta": beta_bk1,
              "NOTE": "BK1 and R1 are the same configuration on this base — the "
                      "block surface selected `self`, which IS the hard backoff.  "
                      "The delta below is therefore exactly 0 by construction, "
                      "not by measurement."})
    beta_bk1b = off.fit_slope(r_sup, pool_S(BK1B_CELL))["beta"]
    register("BK1b_DIAGNOSTIC",
             f"hard backoff at the OLD candidate-H shrink constant "
             f"(cell {BK1B_CELL}).  Not a pre-registered arm; reported so the "
             f"'candidate H configuration' reading of BK1 is also on the record.",
             {s: off.apply_linear(p24[s], TS[BK1B_CELL][VALID_SEASON], beta_bk1b)
              for s in FOLD_SEEDS},
             {"cell": BK1B_CELL, "beta": beta_bk1b})
    beta_bk2 = off.fit_slope(r_sup, pool_S(BK2_CELL))["beta"]
    assert beta_bk2 == EXP031_BETA_CELL[BK2_CELL], (beta_bk2,
                                                    EXP031_BETA_CELL[BK2_CELL])
    register("BK2", f"soft shrinkage: the league term reaches HIT rows too with "
                    f"weight k/(n+k), k = {k_lg} (cell {BK2_CELL}).  exp_031's "
                    f"fold_2024 oracle cell and the OLD base's selected cell.",
             {s: off.apply_linear(p24[s], TS[BK2_CELL][VALID_SEASON], beta_bk2)
              for s in FOLD_SEEDS},
             {"cell": BK2_CELL, "beta": beta_bk2,
              "beta_matches_exp031_record": True,
              "block_brier_from_exp031_record": EXP031_BLOCK_SURF[BK2_CELL],
              "block_gap_self_minus_league_score_points": score_delta(
                  EXP031_BLOCK_SURF[R1_CELL] - EXP031_BLOCK_SURF[BK2_CELL]),
              "block_gap_note": "the block surface preferred `self` over `league` "
                                "at k=400 by this margin.  It is thin; exp_031 "
                                "measured per-cell seed std of 3-4 score points "
                                "on the old base's equivalent surface, so this "
                                "margin does not separate the two cells."})

    # ---- adoption: every arm against the CONTROL R1 -------------------------- #
    thr = 3.0 * r1_std
    adoption = {
        "rule": "3 seeds same sign AND |mean delta| >= 3 x CONTROL seed std.  "
                "NOT relaxed.",
        "control_arm": "R1", "control_cell": R1_CELL,
        "control_brier_by_seed": {str(s): r1_by_seed[s] for s in FOLD_SEEDS},
        "control_mean_brier": r1_mean, "control_mean_score": score(r1_mean),
        "control_seed_std_DENOMINATOR": r1_std,
        "control_seed_std_score_points": abs(score_delta(r1_std)),
        "threshold_abs_mean_delta_THIS_IS_THE_THRESHOLD": thr,
        "threshold_score_points": abs(score_delta(thr)),
        "label_warning": "MULTIPLE_OF_THE_THRESHOLD and MULTIPLE_OF_THE_CONTROL_"
                         "STD differ by exactly 3x.  They are NOT interchangeable "
                         "and must never be compared across experiments without "
                         "first checking which one the other experiment reported. "
                         " exp_031 reported a CONTROL-STD multiple of 2.464 and a "
                         "THRESHOLD multiple of 0.821 for the same measurement.",
        "control_choice_note": "exp_031's control was R0 (the uncorrected base), "
                               "whose seed std was smaller; every arm here varies "
                               "the offset, so the control is R1 and the bar is "
                               "correspondingly HIGHER.  The R0-referenced numbers "
                               "are given separately below and must not be mixed.",
        "vs_R1": {}, "vs_R0": {}}
    for tag, rec in ARMS.items():
        if tag == "R1":
            continue
        cand = [rec["per_seed"][str(s)]["brier"] for s in FOLD_SEEDS]
        dlt = [c - r1_by_seed[s] for c, s in zip(cand, FOLD_SEEDS)]
        md = float(np.mean(dlt))
        adoption["vs_R1"][tag] = {
            "candidate_brier_by_seed": {str(s): c for s, c in zip(FOLD_SEEDS, cand)},
            "candidate_mean_brier": rec["mean_brier"],
            "candidate_mean_score": rec["mean_score"],
            "paired_delta_by_seed": {str(s): d for s, d in zip(FOLD_SEEDS, dlt)},
            "paired_delta_score_by_seed": {str(s): score_delta(d)
                                           for s, d in zip(FOLD_SEEDS, dlt)},
            "mean_delta_brier": md, "mean_delta_score_points": score_delta(md),
            "paired_delta_seed_std_REFERENCE_ONLY": float(np.std(dlt, ddof=1)),
            "paired_delta_seed_std_score_REFERENCE_ONLY": abs(score_delta(
                float(np.std(dlt, ddof=1)))),
            "MULTIPLE_OF_THE_THRESHOLD": abs(md) / thr,
            "MULTIPLE_OF_THE_CONTROL_STD": abs(md) / r1_std,
            "signs_agree": bool(all(x < 0 for x in dlt) or all(x > 0 for x in dlt)),
            "improves": bool(md < 0),
            "passes_magnitude": bool(abs(md) >= thr),
            "ADOPTED": bool((all(x < 0 for x in dlt) or all(x > 0 for x in dlt))
                            and md < 0 and abs(md) >= thr),
            "direction": ("improves (lower Brier)" if md < 0 else
                          "worsens" if md > 0 else "no change"),
        }
        d0 = [c - base24[s] for c, s in zip(cand, FOLD_SEEDS)]
        adoption["vs_R0"][tag] = {
            "mean_delta_brier": float(np.mean(d0)),
            "mean_delta_score_points": score_delta(float(np.mean(d0))),
            "signs_agree": bool(all(x < 0 for x in d0) or all(x > 0 for x in d0))}
    d0_r1 = [r1_by_seed[s] - base24[s] for s in FOLD_SEEDS]
    adoption["vs_R0"]["R1"] = {
        "mean_delta_brier": float(np.mean(d0_r1)),
        "mean_delta_score_points": score_delta(float(np.mean(d0_r1))),
        "signs_agree": bool(all(x < 0 for x in d0_r1))}
    adoption["R0_referenced_threshold_for_contrast_only"] = {
        "R0_seed_std": r0_std, "threshold": 3.0 * r0_std,
        "threshold_score_points": abs(score_delta(3.0 * r0_std)),
        "exp031_reported_this_one": True}

    print("ADOPTION vs R1: " + json.dumps(
        {t: [round(v["mean_delta_score_points"], 5), v["ADOPTED"]]
         for t, v in adoption["vs_R1"].items()}), flush=True)

    # ---- old-base side by side ---------------------------------------------- #
    old = {
        "WARNING": "TWO DIFFERENT BASES.  The old rows are LGBM_B3 (exp_025/026/"
                   "028); the new rows are C5_B3 (exp_030/031).  Brier LEVELS are "
                   "not commensurable and must never be pooled.  Only the "
                   "DIRECTION and the ORDER OF MAGNITUDE of a delta transfer.",
        "logit_axis": {
            "old_B1_minus_A1_mean_score":
                m28["adoption"]["verdict"]["B1_minus_A1"]["mean_score"],
            "old_B1_minus_A1_signs_agree":
                m28["adoption"]["verdict"]["B1_minus_A1"]["signs_agree"],
            "old_B1_minus_A1_ADOPT":
                m28["adoption"]["verdict"]["B1_minus_A1"]["ADOPT"],
            "new_LG1_minus_R1_mean_score":
                adoption["vs_R1"]["LG1"]["mean_delta_score_points"],
            "new_LG1_ADOPTED": adoption["vs_R1"]["LG1"]["ADOPTED"]},
        "reverse_axis_solo": {
            "old_A2_minus_A1_mean_score":
                m28["adoption"]["verdict"]["A2_minus_A1"]["mean_score"],
            "old_A2_cell": m28["arm_terms"]["A2"]["term"],
            "old_A4_block_selected":
                m28["selection"]["groups"]["A4_dr_k_x_target"]["block_selected"],
            "new_DR1_cell": DR_SEL,
            "new_DR1_minus_R1_mean_score":
                adoption["vs_R1"]["DR1"]["mean_delta_score_points"],
            "new_DR1_ADOPTED": adoption["vs_R1"]["DR1"]["ADOPTED"]},
        "ablation": {
            "old_exp025_denominator": "gain of k400xleague OVER the DEPLOYED "
                                      "k111xself d, NOT over no-offset",
            "old_exp025_total_score":
                m25["summary"]["TOTAL_B_full_minus_A_deployed"]["mean_score"],
            "old_exp025_without_ph2_score":
                m25["summary"]["GAIN_WITHOUT_PH2_B_ph1only_minus_A_deployed"]
                   ["mean_score"],
            "old_exp025_retained_pct": 100.0 * (
                m25["summary"]["GAIN_WITHOUT_PH2_B_ph1only_minus_A_deployed"]
                   ["mean_score"]
                / m25["summary"]["TOTAL_B_full_minus_A_deployed"]["mean_score"]),
            "old_ph2_contribution_score":
                m25["summary"]["PH2_CONTRIBUTION_B_full_minus_B_ph1only"]
                   ["mean_score"]},
        "backoff": {
            "old_base_selected_cell": OLD_BASE_CELL,
            "new_base_selected_cell": R1_CELL,
            "new_block_brier_self": EXP031_BLOCK_SURF[R1_CELL],
            "new_block_brier_league": EXP031_BLOCK_SURF[BK2_CELL],
            "new_block_gap_score_points": score_delta(
                EXP031_BLOCK_SURF[R1_CELL] - EXP031_BLOCK_SURF[BK2_CELL]),
            "new_BK2_minus_R1_fold_score":
                adoption["vs_R1"]["BK2"]["mean_delta_score_points"]},
        "offset_share": {
            "old_offset_gain_score_exp028_A1_minus_A0":
                m28["adoption"]["verdict"]["A1_minus_A0"]["mean_score"],
            "new_offset_gain_score_R1_minus_R0":
                adoption["vs_R0"]["R1"]["mean_delta_score_points"]},
    }
    old["offset_share"]["new_over_old_pct"] = 100.0 * (
        old["offset_share"]["new_offset_gain_score_R1_minus_R0"]
        / old["offset_share"]["old_offset_gain_score_exp028_A1_minus_A0"])
    # the apples-to-apples old-base ablation ratio: retained share of the gain
    # measured OVER NO-OFFSET, which is the denominator this run uses
    old_A0 = m28["adoption"]["verdict"]["A1_minus_A0"]
    old_a1_brier = float(np.mean(list(
        m25["summary"]["arm_seed_table"]["B_full"].values())))
    old_abl_brier = float(np.mean(list(
        m25["summary"]["arm_seed_table"]["B_ph1only"].values())))
    old_a0_brier = old_a1_brier - old_A0["mean"]
    old["ablation"]["old_recomputed_on_the_no_offset_denominator"] = {
        "A0_brier_implied": old_a0_brier,
        "B_full_brier": old_a1_brier, "B_ph1only_brier": old_abl_brier,
        "gain_full_score": score_delta(old_a1_brier - old_a0_brier),
        "gain_ablated_score": score_delta(old_abl_brier - old_a0_brier),
        "retained_pct": 100.0 * ((old_abl_brier - old_a0_brier)
                                 / (old_a1_brier - old_a0_brier)),
        "why": "exp_025's headline 83.5% used the DEPLOYED k111xself arm as the "
               "denominator.  This run's denominator is no-offset, so the old "
               "ratio is recomputed on the same denominator here.  Quoting 83.5% "
               "next to this run's ratio without this line would compare two "
               "different quantities."}

    # ---- write out ----------------------------------------------------------- #
    out = {
        "exp_id": "exp_033",
        "title": "every correction axis closed on the OLD base (LGBM_B3), "
                 "re-measured on the NEW base (C5_B3) — logit coordinate, d_r "
                 "solo and joint, pitcher-hand-2 league ablation, backoff "
                 "coverage.  No training.",
        "primary_metric": "fold_2024 valid_all (R+F) Brier",
        "BASE_VALIDITY_WARNING":
            "every number here is measured on the exp_030 C5_B3 base as it stands "
            "(stage-2 iteration policy, fold_2024 stacker fixed at 481 rounds).  "
            "exp_032 is repairing that base's iteration policy.  If it succeeds, "
            "the LEVELS here go stale; the DIRECTIONS (which axis helps, which "
            "hurts) are expected to carry, and re-measuring the winner on a "
            "repaired base costs under a minute.",
        "conversion": {"formula": "score = 100000 * (1 - brier / 0.2498069)",
                       "sensitivity": "1e-06 Brier = 0.4003 score points",
                       "caveat": "every Brier here is full precision, so the "
                                 "+-0.20 rounding caveat does NOT apply"},
        "no_training": True,
        "arms_declared": {t: ARMS[t]["spec"] for t in ARMS},
        "fold_seeds": FOLD_SEEDS, "block_seeds": BLOCK_SEEDS,
        "k_grid_d_s": K_GRID_S, "k_grid_d_r": K_GRID_R, "k_reserve": K_RESERVE,
        "selection_discipline": {
            "chosen_on_the_supply_blocks_only": ["DR1 cell (k x target)",
                                                 "LG2 gamma"],
            "pre_registered_not_chosen": ["R0", "R1", "LG1", "DR2", "AB1", "AB2",
                                          "BK1", "BK2"],
            "diagnostics_not_arms": ["BK1b_DIAGNOSTIC",
                                     "league-cell ablation row counts",
                                     "fold_2024 oracle cells and gammas"]},
        "gate0_R0_reproduction": gate0,
        "gate1_R1_reproduction": gate1,
        "gate2_k0_self_equals_k0_league": gate2,
        "gate3_miss_rows_zero_all_k_both_axes": gate3,
        "gate4_label_isolation": label_isolation,
        "gate5_asof_independent": asof_indep,
        "gate6_devgen_equivalence": gate6,
        "gate7_exp031_record_copies_agree": gate7,
        "block_provenance": blk_prov,
        "fold_2024_base_provenance": fold_prov,
        "base_no_offset_blocks_pooled": {"brier": br_base_sup,
                                         "score": score(br_base_sup)},
        "p_distribution": pdist,
        "coordinate_divergence": coord,
        "ds_dr_correlation_MEASURED": corr_tbl,
        "gamma_curve": gcurve,
        "d_r_block_fit": dr_fit,
        "d_r_surfaces": {
            "block_brier": {nm: dr_fit[nm]["block_brier"] for nm in CELLS_R},
            "block_score": {nm: dr_fit[nm]["block_score"] for nm in CELLS_R},
            "fold_2024_brier_DIAGNOSTIC": {nm: dr_fold[nm] for nm in CELLS_R},
            "fold_2024_score_DIAGNOSTIC": {nm: score(dr_fold[nm])
                                           for nm in CELLS_R},
            "reserve_block_brier": {nm: dr_fit[nm]["block_brier"]
                                    for nm in CELLS_R_RES},
            "enclosure_declared": dr_enc, "enclosure_widened": dr_enc_wide},
        "ablation_scope": ablation_scope,
        "arms": ARMS,
        "adoption": adoption,
        "old_base_side_by_side": old,
        "ds_diagnostics_success_axis": ds_diag,
        "dr_diagnostics_reverse_axis": dr_diag,
        "source_sha256": src_sha,
        "train_csv_sha256": csv_sha,
        "reference_metrics_sha256": {
            "exp_025": sha256_file(EXP025_METRICS),
            "exp_026": sha256_file(EXP026_METRICS),
            "exp_028": sha256_file(EXP028_METRICS),
            "exp_030": sha256_file(EXP030_METRICS),
            "exp_031_local_copy": gate7["local_sha256"],
            "exp_031_remote": gate7["remote_sha256"]},
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__,
                     "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS")},
    }

    # ---- report_derived: EVERY number the report quotes, and a regeneration
    #      check that rebuilds it from the primary sections alone ------------- #
    def build_derived(o):
        ad, ar = o["adoption"], o["arms"]
        pd_ = o["p_distribution"]
        cd = o["coordinate_divergence"]
        rows = []
        for t in ["R0", "R1", "LG1", "LG2", "DR1", "DR2", "AB1", "AB2", "BK1",
                  "BK1b_DIAGNOSTIC", "BK2"]:
            v = ad["vs_R1"].get(t)
            rows.append({
                "arm": t,
                "brier_by_seed": [ar[t]["per_seed"][str(s)]["brier"]
                                  for s in o["fold_seeds"]],
                "mean_brier": ar[t]["mean_brier"],
                "mean_score": ar[t]["mean_score"],
                "delta_vs_R1_brier": (0.0 if v is None else v["mean_delta_brier"]),
                "delta_vs_R1_score": (0.0 if v is None
                                      else v["mean_delta_score_points"]),
                "signs_agree": (None if v is None else v["signs_agree"]),
                "mult_of_threshold": (None if v is None
                                      else v["MULTIPLE_OF_THE_THRESHOLD"]),
                "mult_of_control_std": (None if v is None
                                        else v["MULTIPLE_OF_THE_CONTROL_STD"]),
                "paired_delta_std_score": (None if v is None else
                                           v["paired_delta_seed_std_score_"
                                             "REFERENCE_ONLY"]),
                "ADOPTED": (None if v is None else v["ADOPTED"])})
        new_sd = pd_["fold_2024_3seed_mean_of_sd"]
        old_sd = pd_["old_base_LGBM_B3_from_exp028_seed42"]["sd"]
        abl = o["old_base_side_by_side"]["ablation"]
        gain_full = ad["vs_R0"]["R1"]["mean_delta_score_points"]
        gain_abl = ad["vs_R0"]["AB1"]["mean_delta_score_points"]
        return {
            "arm_table": rows,
            "p_dispersion": {
                "new_sd_3seed_mean": new_sd, "old_sd_seed42": old_sd,
                "sd_widening_pct": 100.0 * (new_sd / old_sd - 1.0),
                "new_v_cv_3seed_mean": pd_["fold_2024_3seed_mean_of_v_cv"],
                "old_v_cv_seed42":
                    pd_["old_base_LGBM_B3_from_exp028_seed42"]["v_cv"],
                "new_v_max_over_min_seed42":
                    pd_["fold_2024_by_seed"]["42"]["v_max_over_min"],
                "old_v_max_over_min_seed42":
                    pd_["old_base_LGBM_B3_from_exp028_seed42"]["v_max_over_min"]},
            "coordinate_gap": {
                "fold_mean_abs_divergence":
                    cd["fold_2024_seed42"]["mean_abs_divergence_linear_vs_logit"],
                "fold_max_abs_divergence":
                    cd["fold_2024_seed42"]["max_abs_divergence_linear_vs_logit"],
                "fold_mean_abs_correction":
                    cd["fold_2024_seed42"]["mean_abs_correction_linear"],
                "fold_max_abs_correction":
                    cd["fold_2024_seed42"]["max_abs_correction_linear"],
                "divergence_share_of_correction_mean":
                    cd["fold_2024_seed42"]["divergence_over_correction_mean"],
                "n_clipped_linear": cd["fold_2024_seed42"]["n_rows_clipped_linear"],
                "old_mean_abs_divergence":
                    cd["old_base_from_exp028_seed42"]["mean_abs_divergence"],
                "old_divergence_share":
                    cd["old_base_from_exp028_seed42"][
                        "divergence_over_correction_mean"]},
            "ablation_retention": {
                "gain_R1_over_R0_score": gain_full,
                "gain_AB1_over_R0_score": gain_abl,
                "retained_pct": 100.0 * gain_abl / gain_full,
                "ph2_contribution_score": gain_full - gain_abl,
                "old_base_retained_pct_same_denominator":
                    abl["old_recomputed_on_the_no_offset_denominator"]
                       ["retained_pct"],
                "old_base_retained_pct_exp025_headline":
                    abl["old_exp025_retained_pct"]},
            "thresholds": {
                "control_R1_seed_std": ad["control_seed_std_DENOMINATOR"],
                "control_R1_seed_std_score": ad["control_seed_std_score_points"],
                "threshold_brier":
                    ad["threshold_abs_mean_delta_THIS_IS_THE_THRESHOLD"],
                "threshold_score": ad["threshold_score_points"],
                "R0_referenced_threshold_score":
                    ad["R0_referenced_threshold_for_contrast_only"][
                        "threshold_score_points"]},
        }

    out["report_derived"] = build_derived(out)
    # regeneration check: rebuild from the SERIALISED primary sections alone
    rebuilt = build_derived(json.loads(json.dumps(
        {k: v for k, v in out.items() if k != "report_derived"})))
    same = (json.dumps(out["report_derived"], sort_keys=True)
            == json.dumps(rebuilt, sort_keys=True))
    out["report_derived_regeneration_check"] = {
        "method": "report_derived is rebuilt from a JSON round-trip of the primary "
                  "sections only (report_derived itself excluded) and compared "
                  "byte for byte.  Every number quoted in report.md comes from "
                  "report_derived, so this check is what makes the report "
                  "reproducible from metrics.json alone.",
        "identical": bool(same)}
    assert same, "report_derived could not be regenerated from the primary sections"

    out["compute_time_min"] = round((time.time() - t_start) / 60, 3)
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(out, f, indent=1)

    # ---- curves -------------------------------------------------------------- #
    with open(os.path.join(OUT_DIR, "curves", "dr_k_surface.csv"), "w") as f:
        f.write("k,target,beta_r,block_brier,block_score,fold_brier,fold_score,"
                "declared\n")
        for k in K_ALL_R:
            for tg in TARGETS:
                nm = cell_name("R", k, tg)
                f.write(f"{k},{tg},{dr_fit[nm]['beta']},"
                        f"{dr_fit[nm]['block_brier']:.14f},"
                        f"{dr_fit[nm]['block_score']},{dr_fold[nm]:.14f},"
                        f"{score(dr_fold[nm])},"
                        f"{'yes' if k in K_GRID_R else 'reserve'}\n")
    with open(os.path.join(OUT_DIR, "curves", "gamma_grid.csv"), "w") as f:
        f.write("gamma,block_brier,fold_mean_brier,fold_mean_score,is_block_argmin,"
                "is_bounded_minimiser,is_beta_equivalent\n")
        for g in g_grid:
            f.write(f"{g},{gcurve[str(g)]['block_brier']:.14f},"
                    f"{gcurve[str(g)]['fold_mean_brier']:.14f},"
                    f"{gcurve[str(g)]['fold_mean_score']},"
                    f"{'yes' if g == g_star else 'no'},"
                    f"{'yes' if g == gamma_hat else 'no'},"
                    f"{'yes' if g == round(gamma_eq, 6) else 'no'}\n")
    with open(os.path.join(OUT_DIR, "curves", "arm_table.csv"), "w") as f:
        f.write("arm,mean_brier,mean_score,delta_vs_R1_brier,delta_vs_R1_score,"
                "signs_agree,mult_of_threshold,mult_of_control_std,ADOPTED\n")
        for r in out["report_derived"]["arm_table"]:
            f.write(f"{r['arm']},{r['mean_brier']:.14f},{r['mean_score']},"
                    f"{r['delta_vs_R1_brier']:.6e},{r['delta_vs_R1_score']},"
                    f"{r['signs_agree']},{r['mult_of_threshold']},"
                    f"{r['mult_of_control_std']},{r['ADOPTED']}\n")

    print(f"DONE in {out['compute_time_min']} min — DR1 cell {DR_SEL}, LG2 gamma "
          f"{g_star}; adopted: "
          f"{[t for t, v in adoption['vs_R1'].items() if v['ADOPTED']]}",
          flush=True)


if __name__ == "__main__":
    main()
