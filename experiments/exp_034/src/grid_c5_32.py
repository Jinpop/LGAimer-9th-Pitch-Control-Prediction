"""exp_032 — the SUCCESS-axis offset surface, re-measured on a base whose
ITERATION COUNTS WERE CHOSEN BY exp_032's FITTED LAW instead of exp_030's
row-ratio rule.  NO TRAINING HAPPENS IN THIS FILE.

exp_032 DERIVATION  (this file is exp_031's grid_c5.py, which is carried
------------------   BYTE-IDENTICAL beside it and sha-asserted, with the
                     changes below and nothing else)

  1. THE fold_2024 BASE IS DIFFERENT ON PURPOSE.  exp_031 reused exp_030's
     stage-2 fold_2024 predictions byte for byte.  exp_032 rebuilds them
     (stage 4 of block_iter.py) at iteration counts that came out of
     fit_iter.py's law, fitted on the BLOCK units only.  Gate 0 is therefore
     INVERTED: the base must now DIFFER from exp_030's, and a match would mean
     the new iteration counts never reached the models.

  2. GATE 1 IS RE-RUN, NOT INHERITED.  Block 2023 seed 42 is regenerated here
     under exp_032's own copy of the code and its prediction vector must be
     BIT-IDENTICAL to exp_031's.  That is the only proof available that the
     three changes made to block_self.py did not move the arithmetic, since a
     reproduction gate on the fold_2024 base is impossible when the fold_2024
     base is the thing being changed.

  3. THE SUPPLY IS UNCHANGED.  beta is still fitted on blocks 2022 and 2023 at
     their own argmin, reused from exp_031 byte for byte and sha-checked.  So
     the k x target surface moves ONLY because the fold_2024 base moved — but
     note that this leaves a real asymmetry, recorded in the output: the supply
     blocks sit at their TRUE argmin while fold_2024 sits at an EXTRAPOLATED
     iteration, because a deployment build cannot early-stop either.

  4. THE HEADLINE MEASUREMENT IS THE SEED STANDARD DEVIATION OF THE
     UNCORRECTED BASE, not the offset.  exp_031 found the offset worth
     +38.28802 score points with all three seeds agreeing in sign, and rejected
     it anyway because the control's seed std was 3.881563e-05 and the
     adoption threshold three times that — 46.61476 points — is above every one
     of the eighteen cells.  If the new iteration counts shrink that std, the
     same offset can clear the same unrelaxed rule.  If they do not, the
     threshold is a property of this base and the offset cannot be adopted on
     it; that is a result, not a failure.

------------------------------------------------------------------------------
exp_031's original header follows, unchanged, because everything it describes
is still what this file does:

exp_031 — the SUCCESS-axis offset surface, re-drawn from scratch on the NEW
base (CatBoost C5_B3, exp_030).  NO TRAINING HAPPENS IN THIS FILE.

WHY IT EXISTS
-------------
`k = 400 x league` and `beta = 0.5704` were both chosen on the RESIDUALS OF
LGBM_B3.  An offset burns the base's residual, so when the base is replaced the
surface it is chosen on is a different surface.  exp_029 already proved the
backoff-target preference is AXIS-dependent (`d_s` wants `league`, `d_r` wants
`self`); nothing rules out that it is BASE-dependent too.  So the grid is swept
again, end to end, and the old cell is given no privileged status.

WHAT IS CHOSEN, AND WHERE
-------------------------
The cell is chosen on the SUPPLY BLOCKS (2022 + 2023) and nowhere else.
fold_2024 is read only after every beta and the cell are fixed, and where the
two surfaces disagree the gap is reported and the BLOCK choice stands.

BETA, AND WHY IT IS POOLED OVER THE BLOCK SEEDS
-----------------------------------------------
exp_030 built block predictions for seeds 42 and 7 only (fold_2024 has three).
`d` is SEED-INDEPENDENT -- it comes from the labels, not from the model -- so
concatenating (block 2022, block 2023) x (seed 42, seed 7) into one no-intercept
least squares is well defined, and because the two seeds contribute identical
`sum d^2`, the pooled slope is exactly the mean of the two per-seed slopes up to
`fit_slope`'s 6-dp rounding.  ONE beta per cell is therefore fitted on all four
supply vectors and applied to all three fold_2024 seeds.  The per-seed betas and
the seed-matched variant (42 -> beta42, 7 -> beta7, 2024 -> pooled) are computed
and reported as sensitivity; they select nothing.

THE FORM OF d  (inherited verbatim from exp_023 via exp_029's dev_k.py)
------------------------------------------------------------------------
Expanded, `d = (s - n*overall)/(n + k)`; the league term regrouped as
`L * k/(n+k)`.  The literal subtraction returns `k*o/k - o`, which IEEE-754 does
not guarantee to be 0 on a cell miss.  Miss rows carrying EXACTLY 0.0 is
load-bearing, and it is asserted on every k of the grid.

GATES (every expectation is READ from a record, never transcribed)
------------------------------------------------------------------
 G0  the UNCORRECTED C5_B3 must reproduce exp_030's fold_2024 Brier on all
     three seeds, |diff| < 5e-6.  Expected values are read out of
     exp_030/metrics.json["fold_2024"]["candidate_brier_by_seed"].
 G1  the regenerated block 2023 seed 42 (stage 3, own early stop) must be
     BIT-IDENTICAL to exp_030's stage-1 prediction for the same unit and seed.
     exp_030's stage 1 already stopped on its own curve, so stage 3 is the same
     arithmetic; this checks that block_self.py's derivation changed nothing
     else, and that running four jobs against a 20-CPU quota did not perturb
     CatBoost.
 G2  `k0_self` and `k0_league` are the same vector.
 G3  `d == 0.0` exactly on cell-miss rows, on every k of the grid.
 G4  the deviations are constant inside their own (season, pitcher, hand,
     pitcher_hand) lookup group -- no row-level label reached them.
 G5  as-of, re-verified from a SECOND independent read of train.csv.

INHERITANCE
-----------
`dev_k.py`, `backoff.py`, `offset.py`, `features_platoon.py`, `labels.py`,
`exp016_ref.py`, `devgen.py` are BYTE-PRESERVED from exp_029 and their sha256
are asserted before anything runs.  `grid_ds.py` (exp_029) and `b3_cat.py`
(exp_030) are carried and hashed as provenance; neither is imported here.
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
OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_032")
EXP029_METRICS = os.path.expanduser(
    "~/SungjinPark/experiments/exp_029/metrics.json")
EXP030_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_030")
EXP030_METRICS = os.path.join(EXP030_DIR, "metrics.json")
EXP031_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_031")
EXP031_METRICS = os.path.join(EXP031_DIR, "metrics.json")

FOLD_SEEDS = [42, 7, 2024]          # exp_030 fold_2024 predictions
BLOCK_SEEDS = [42, 7]               # exp_030 / exp_031 supply-block predictions
GATE_TOL = 5e-6

TARGET = "control_success"
DROP_COLS = ["row_id", TARGET]
F_EXCLUDE_MAX_SEASON = 2022
SUPPLY_BLOCKS = [2022, 2023]
VALID_SEASON = 2024
EPS = 1e-6
BSS_REF_BRIER = 0.2498069

K_GRID = [0, 30, 55, 111, 200, 400, 800, 1600, 3200]      # the declared grid
K_RESERVE = [6400, 12800]     # computed too, so an EDGE argmin can be widened
                              # without a second run; they are NOT part of the
                              # declared surface unless the declared argmin sits
                              # on the k=3200 edge, and the code says which.
K_ALL = K_GRID + K_RESERVE
TARGETS = ["self", "league"]

OLD_BASE_CELL = "S400_league"     # what the OLD base (LGBM_B3) selected
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
    "grid_ds.py": "62405e720849b68c384763f703029f6ab2200dd252d02dc7cdc5f59287b7e96d",
    "b3_cat.py": "0984c57e5af66600f9fca5532a984ddd6d4c22996c9e0105d3e798bc7bfa76ca",
    # exp_031's two drivers, carried byte-identical as the derivation source
    "block_self.py": "2146d0e78a487b3223be42a743e5d03ff5e56d35c12e502f812c0eb1f"
                     "3184406",
    "grid_c5.py": "03cbfbf1d1d9deba826ac06cb9b61b15d97f407c4670de5a723e6a27f863"
                  "9324",
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
    v = [vals_by_k[k] for k in ks]
    j = int(np.argmin(v))
    return {
        "argmin_k": ks[j], "argmin_index": j, "n_points": len(ks),
        "bracketed_below": bool(j > 0), "bracketed_above": bool(j < len(ks) - 1),
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
                    "EDGE minimum — the grid does not bracket it; widen"),
    }


def crossing(self_by_k, league_by_k, ks):
    diff = {k: self_by_k[k] - league_by_k[k] for k in ks}
    wins = {k: ("self" if diff[k] < 0 else "league" if diff[k] > 0 else "tie")
            for k in ks}
    flips = []
    for a, b in zip(ks[:-1], ks[1:]):
        if wins[a] != wins[b] and "tie" not in (wins[a], wins[b]):
            flips.append({"between_k": [a, b], "from_winner": wins[a],
                          "to_winner": wins[b], "diff_at_lower": diff[a],
                          "diff_at_upper": diff[b]})
    return {
        "convention": "diff = self - league in Brier; diff < 0 means SELF wins",
        "diff_by_k": {str(k): diff[k] for k in ks},
        "diff_by_k_score_self_minus_league": {str(k): score_delta(diff[k])
                                              for k in ks},
        "winner_by_k": {str(k): wins[k] for k in ks},
        "n_k_where_self_wins": sum(1 for k in ks if wins[k] == "self"),
        "n_k_where_league_wins": sum(1 for k in ks if wins[k] == "league"),
        "crossings": flips, "any_crossing": bool(flips),
    }


# --------------------------------------------------------------------------- #
def main():
    t_start = time.time()
    for d in ("curves", "pred"):
        os.makedirs(os.path.join(OUT_DIR, d), exist_ok=True)

    csv_sha = sha256_file(DATA_PATH)
    assert csv_sha == TRAIN_CSV_SHA, ("train.csv SHA MISMATCH", csv_sha)
    src_sha = {f: sha256_file(os.path.join(SRC_DIR, f))
               for f in sorted(os.listdir(SRC_DIR)) if f.endswith(".py")}
    for f, want in SHA_INHERITED.items():
        assert src_sha[f] == want, ("INHERITED SOURCE DRIFTED", f, src_sha[f], want)
    print("inherited sha OK", flush=True)

    # ---- the records.  Every gate number is READ. --------------------------- #
    with open(EXP030_METRICS) as f:
        m30 = json.load(f)
    assert m30["exp_id"] == "exp_030"
    assert m30["reproducibility"]["train_csv_sha256"] == csv_sha
    GATE_FOLD = {s: b for s, b in zip(m30["fold_2024"]["seeds"],
                                      m30["fold_2024"]["candidate_brier_by_seed"])}
    assert set(GATE_FOLD) == set(FOLD_SEEDS), GATE_FOLD
    FOLD_SHA = {int(k.split("seed")[1]): v["sha256"]
                for k, v in m30["predictions_recovered_locally"].items()
                if k.startswith("u2024_")}
    assert set(FOLD_SHA) == set(FOLD_SEEDS), FOLD_SHA
    EXP030_BLOCK_STAGE2 = {
        int(V): {int(s): d["brier"] for s, d in
                 m30["beta_supply_blocks"][f"block_{V}"]["C5_B3"].items()}
        for V in SUPPLY_BLOCKS}
    LGBM_BLOCK = {int(V): {int(s): b for s, b in
                           m30["beta_supply_blocks"][f"block_{V}"]
                           ["LGBM_B3_exp024_no_offset"].items()}
                  for V in SUPPLY_BLOCKS}
    EXP030_U2023_S42_SHA = m30["predictions_recovered_locally"]["u2023_seed42"]["sha256"]

    # exp_031's record: the base this experiment is trying to improve on, and
    # the block vectors that are reused unchanged.  Every number is READ.
    with open(EXP031_METRICS) as f:
        m31 = json.load(f)
    assert m31["exp_id"] == "exp_031"
    assert m31["train_csv_sha256"] == csv_sha
    OLD_BASE = m31["base_no_offset_fold_2024"]          # exp_030 iteration policy
    OLD_BASE_BY_SEED = {int(s): b for s, b in OLD_BASE["by_seed"].items()}
    OLD_BASE_STD = OLD_BASE["seed_std"]
    OLD_BASE_MEAN = OLD_BASE["mean_brier"]
    EXP031_BLOCK_PROV = m31["block_provenance"]
    EXP031_SELECTED = m31["selection"]["selected_cell"]
    EXP031_ADOPTION = m31["adoption"]
    EXP031_U2023_S42_SHA = EXP031_BLOCK_PROV["block_2023_seed42"]["sha256"]

    with open(EXP029_METRICS) as f:
        m29 = json.load(f)
    assert m29["exp_id"] == "exp_029"
    OLD_BLOCK = m29["surfaces"]["mean_over_seeds"]["block_brier"]
    OLD_FOLD = m29["surfaces"]["mean_over_seeds"]["fold_2024_brier"]
    print("exp_030 / exp_029 records read", flush=True)

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
        "note": "no model is fitted here; the design matrix is exp_030's and the "
                "recovered labels are TARGETS ONLY there (b3_cat.py asserts it "
                "before every frame).  check (b) below is this file's own.",
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

    # ---- deviations (seed-independent: they come from labels, not the model) - #
    print(f"[{time.strftime('%H:%M:%S')}] platoon lookups", flush=True)
    fp.build_season_forward_chained(df, ok, succ_season, f_excl, verbose=False)
    d_s, hit, n_cell0, _, ds_diag = dk.build_multi_k(df, ok, succ_season, f_excl,
                                                     K_ALL, verbose=False)
    L_s, has_Ls, _, _ = bk.build_league_dev(df, ok, succ_season, f_excl,
                                            verbose=False)
    Ls_eff = np.where(has_Ls, L_s, 0.0)

    # ---- GATE 3 ------------------------------------------------------------- #
    gate4 = {str(k): bool(np.all(d_s[k][~hit] == 0.0)) for k in K_ALL}
    assert all(gate4.values()), gate4
    assert len(gate4) == len(K_ALL)
    print(f"GATE 3 OK — d == 0.0 exactly on miss rows for all {len(K_ALL)} k",
          flush=True)

    TERMS = {}
    for k in K_ALL:
        for tg in TARGETS:
            TERMS[cell_name(k, tg)] = dk.build_arm(d_s[k], hit, Ls_eff, n_cell0,
                                                    k, tg)
    CELLS = [cell_name(k, tg) for k in K_GRID for tg in TARGETS]
    CELLS_RES = [cell_name(k, tg) for k in K_RESERVE for tg in TARGETS]
    assert len(CELLS) == 18, len(CELLS)

    # ---- GATE 2 + structural asserts ---------------------------------------- #
    gate2 = bool(np.array_equal(TERMS["S0_self"], TERMS["S0_league"]))
    assert gate2, "GATE 2 FAILED: k=0 self and league must coincide"
    for k in K_ALL:
        assert np.array_equal(TERMS[cell_name(k, "self")][~hit],
                              TERMS[cell_name(k, "league")][~hit]), k
        assert np.array_equal(TERMS[cell_name(k, "self")][hit], d_s[k][hit]), k
    print("GATE 2 OK — k0_self and k0_league are the same vector", flush=True)

    # ---- GATE 4: label isolation -------------------------------------------- #
    gcodes = pd.DataFrame({"s": season, "p": df[fp.PITCHER].to_numpy(),
                           "h": df[fp.HAND].to_numpy(), "ph": ph_all}
                          ).groupby(["s", "p", "h", "ph"], sort=False,
                                    dropna=False).ngroup().to_numpy()
    const = {}
    for nm, v in list(TERMS.items()) + [(f"d_s_raw_k{k}", d_s[k]) for k in K_ALL]:
        first = pd.Series(v).groupby(gcodes, sort=False).transform("first").to_numpy()
        const[nm] = bool(np.array_equal(v, first))
    label_isolation["check_b_constant_within_season_pitcher_hand"] = const
    label_isolation["n_lookup_groups"] = int(gcodes.max() + 1)
    label_isolation["n_vectors_checked"] = len(const)
    assert all(const.values()), ("a deviation varies inside its lookup group", const)
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
                            "looked up in that independent map",
                  "frame_season_matches_file": frame_matches_file,
                  "tables_covered": ["personal_success", "league_success"],
                  "per_season": {}}
    for s in sorted(set(season.tolist()))[1:]:
        m_pers = base_ok & (season <= s - 1) & (succ_season <= s - 1)
        m_leag = m_pers & hand_ok
        rec_s = {}
        for name, m in (("personal_success", m_pers), ("league_success", m_leag)):
            mx = int(s_map.reindex(pd.Index(row_id[m])).max())
            assert mx <= s - 1, ("AS-OF VIOLATION (independent)", name, s, mx)
            rec_s[name] = {"contributing_rows": int(m.sum()),
                           "max_season_of_contributors_independent": mx,
                           "allowed_max": int(s - 1)}
        asof_indep["per_season"][str(int(s))] = rec_s
    assert max(SUPPLY_BLOCKS) < VALID_SEASON
    print("GATE 5 (as-of, independent) OK", flush=True)

    # ---- slices ------------------------------------------------------------- #
    y = {V: y_all[season == V] for V in SUPPLY_BLOCKS + [VALID_SEASON]}
    m24 = (season == VALID_SEASON)
    y24 = y[VALID_SEASON]
    hit24, isr24, n_cell24, ph24 = hit[m24], is_r_all[m24], n_cell0[m24], ph_all[m24]
    assert len(y24) == 253507 and int(isr24.sum()) == 223497

    T = {nm: {V: v[season == V] for V in SUPPLY_BLOCKS + [VALID_SEASON]}
         for nm, v in TERMS.items()}

    hn = n_cell24[hit24]
    tert_lo, tert_hi = np.percentile(hn, [33.3333, 66.6667])
    bins24 = {
        "miss_n0": ~hit24,
        f"hit_low_n_le_{int(tert_lo)}": hit24 & (n_cell24 <= tert_lo),
        f"hit_mid_n_{int(tert_lo)}_{int(tert_hi)}": (hit24 & (n_cell24 > tert_lo)
                                                     & (n_cell24 <= tert_hi)),
        f"hit_high_n_gt_{int(tert_hi)}": hit24 & (n_cell24 > tert_hi),
    }

    # ---- base vectors: blocks from exp_031 (regenerated), fold from exp_030 -- #
    blk, blk_prov = {}, {}
    for s in BLOCK_SEEDS:
        blk[s] = {}
        for V in SUPPLY_BLOCKS:
            stem = f"C5B3_u{V}_seed{s}_st3"
            pth = os.path.join(EXP031_DIR, "pred", f"pred_{stem}.npy")
            with open(os.path.join(EXP031_DIR, "res", f"res_{stem}.json")) as f:
                rj = json.load(f)
            assert rj["exp_id"] == "exp_031" and rj["stage"] == 3
            # the supply is REUSED, not rebuilt: its sha must match exp_031's
            # own published record before a single beta is fitted on it
            assert sha256_file(pth) == EXP031_BLOCK_PROV[
                f"block_{V}_seed{s}"]["sha256"], ("supply block drifted", V, s)
            assert rj["self_early_stop"] is True
            assert rj["sha256"]["train.csv"] == csv_sha
            assert not rj["any_argmin_at_cap"], ("ARGMIN AT CAP — unit invalid",
                                                 V, s, rj["cap"])
            got = sha256_file(pth)
            assert got == rj["pred_sha256"], ("block pred sha drift", V, s)
            arr = np.load(pth)
            assert arr.shape == y[V].shape, (V, s, arr.shape)
            # the block's own record must agree with the vector on disk
            assert abs(float(np.mean((arr - y[V]) ** 2)) - rj["brier_full"]) < 1e-12,\
                ("block res json disagrees with its own prediction vector", V, s)
            blk[s][V] = arr
            blk_prov[f"block_{V}_seed{s}"] = {
                "path": pth, "sha256": got, "made_by": "exp_031 block_self.py "
                "stage 3 (own early stop)", "cap": rj["cap"],
                "stack_iter_used": rj["stacker"]["iter_used"],
                "stack_argmin": rj["stacker"]["argmin_iter"],
                "cell_iter_used": rj["cell_iter_used"],
                "cell_argmin": rj["cell_argmin_iter"],
                "brier": rj["brier_full"], "score": score(rj["brier_full"]),
                "minutes": rj["minutes"],
                "exp030_stage2_brier": EXP030_BLOCK_STAGE2[V][s],
                "exp030_stage2_score": score(EXP030_BLOCK_STAGE2[V][s]),
                "gain_vs_exp030_stage2_score": score_delta(
                    rj["brier_full"] - EXP030_BLOCK_STAGE2[V][s]),
            }
    # GATE 1: exp_032 re-ran block 2023 seed 42 stage 3 under its OWN copy of
    # the code.  It must be bit-identical to exp_031's vector (and therefore to
    # exp_030's stage-1 vector, which exp_031 already matched).  A reproduction
    # gate on the fold_2024 base is impossible here because that base is the
    # thing being changed, so THIS is the code-path identity proof.
    own_stem = "C5B3_u2023_seed42_st3"
    own_pth = os.path.join(OUT_DIR, "pred", f"pred_{own_stem}.npy")
    with open(os.path.join(OUT_DIR, "res", f"res_{own_stem}.json")) as f:
        own_rj = json.load(f)
    assert own_rj["exp_id"] == "exp_032" and own_rj["stage"] == 3
    assert own_rj["self_early_stop"] is True
    assert not own_rj["any_argmin_at_cap"], ("gate re-run hit the cap",
                                             own_rj["cap"])
    g1_sha = sha256_file(own_pth)
    gate1 = {
        "what": "exp_032's OWN re-run of block 2023 seed 42, stage 3, compared "
                "to exp_031's vector for the same unit and seed (which in turn "
                "already matched exp_030's stage-1 vector).  block_iter.py adds "
                "unit 2021, adds stage 4 and moves OUT_DIR; if any of that "
                "touched the arithmetic, this sha differs.",
        "exp032_sha256": g1_sha, "exp031_sha256": EXP031_U2023_S42_SHA,
        "exp030_sha256": EXP030_U2023_S42_SHA,
        "bit_identical_to_exp031": bool(g1_sha == EXP031_U2023_S42_SHA),
        "bit_identical_to_exp030": bool(g1_sha == EXP030_U2023_S42_SHA),
        "argmin_agreement": {
            "exp032_stack": own_rj["stack_argmin_iter"],
            "exp031_stack": EXP031_BLOCK_PROV["block_2023_seed42"]["stack_argmin"],
            "exp032_cell": own_rj["cell_argmin_iter"],
            "exp031_cell": EXP031_BLOCK_PROV["block_2023_seed42"]["cell_argmin"]},
        "brier_exp032": own_rj["brier_full"],
        "brier_exp031": EXP031_BLOCK_PROV["block_2023_seed42"]["brier"],
        "minutes": own_rj["minutes"]}
    assert gate1["bit_identical_to_exp031"], ("GATE 1 FAILED — exp_032's code "
                                              "path is not exp_031's", gate1)
    print(f"GATE 1 (block 2023 s42 bit-identity): "
          f"{gate1['bit_identical_to_exp031']}", flush=True)

    # ---- the NEW fold_2024 base: exp_032 stage 4, iterations from the law --- #
    p24, fold_prov = {}, {}
    for s in FOLD_SEEDS:
        stem = f"C5B3_u2024_seed{s}_st4"
        pth = os.path.join(OUT_DIR, "pred", f"pred_{stem}.npy")
        with open(os.path.join(OUT_DIR, "res", f"res_{stem}.json")) as f:
            rj = json.load(f)
        assert rj["exp_id"] == "exp_032" and rj["stage"] == 4, stem
        assert rj["self_early_stop"] is False, "stage 4 must NOT self-stop"
        assert rj["sha256"]["train.csv"] == csv_sha
        assert rj["iter_source"], "stage 4 with no recorded iteration source"
        got = sha256_file(pth)
        assert got == rj["pred_sha256"], ("fold pred sha drift", s)
        arr = np.load(pth)
        assert arr.shape == y24.shape, (s, arr.shape)
        assert abs(float(np.mean((arr - y24) ** 2)) - rj["brier_full"]) < 1e-12
        p24[s] = arr
        fold_prov[str(s)] = {
            "path": pth, "sha256": got,
            "made_by": "exp_032 block_iter.py stage 4 (iterations from "
                       "fit_iter.py's law, fitted on block units only)",
            "iter_source": rj["iter_source"],
            "cell_iter_used": rj["cell_iter_used"],
            "stack_iter_used": rj["stacker"]["iter_used"],
            "stack_argmin_DIAGNOSTIC_selects_nothing": rj["stack_argmin_iter"],
            "cell_argmin_DIAGNOSTIC_selects_nothing": rj["cell_argmin_iter"],
            "stacker_policy_cost_vs_own_curve_min":
                rj["stacker"]["policy_cost_vs_curve_min"],
            "brier": rj["brier_full"], "score": score(rj["brier_full"]),
            "minutes": rj["minutes"],
            "exp030_stage2_sha256": FOLD_SHA[s],
            "exp030_stage2_brier": GATE_FOLD[s],
            "exp030_stage2_score": score(GATE_FOLD[s]),
            "differs_from_exp030": bool(got != FOLD_SHA[s]),
            "gain_vs_exp030_score": score_delta(rj["brier_full"] - GATE_FOLD[s])}

    # ---- GATE 0, INVERTED: the base is CHANGED ON PURPOSE ------------------- #
    # exp_031 required |new - exp_030| < 5e-6 because it reused exp_030's
    # vectors.  exp_032 rebuilds them at different iteration counts, so the
    # SAME check with the SAME tolerance now has to FAIL: a base that still
    # reproduces exp_030 means the new iteration counts never reached the
    # models, and everything downstream would be a re-run of exp_031.
    base24 = {s: float(np.mean((p24[s] - y24) ** 2)) for s in FOLD_SEEDS}
    gate0 = {"direction": "INVERTED vs exp_031 — the base MUST differ",
             "why": "this experiment changes the iteration counts of the "
                    "fold_2024 base; identity with exp_030 would mean the "
                    "change never landed",
             "tolerance_that_must_NOT_be_met": GATE_TOL,
             "source": "exp_030/metrics.json fold_2024.candidate_brier_by_seed "
                       "— READ, not transcribed", "checks": {}}
    for s in FOLD_SEEDS:
        d0 = abs(base24[s] - GATE_FOLD[s])
        gate0["checks"][str(s)] = {
            "exp030_base": GATE_FOLD[s], "exp032_base": base24[s],
            "abs_diff": d0, "score_delta_vs_exp030": score_delta(
                base24[s] - GATE_FOLD[s]),
            "bit_identical": bool(base24[s] == GATE_FOLD[s]),
            "differs_as_required": bool(d0 > 0.0)}
    gate0["all_three_seeds_differ"] = all(v["differs_as_required"]
                                          for v in gate0["checks"].values())
    gate0["any_seed_still_within_5e-6_of_exp030"] = any(
        v["abs_diff"] < GATE_TOL for v in gate0["checks"].values())
    assert gate0["all_three_seeds_differ"], (
        "the new iteration counts produced a base identical to exp_030's",
        json.dumps(gate0))
    base_mean = float(np.mean([base24[s] for s in FOLD_SEEDS]))
    base_std = float(np.std([base24[s] for s in FOLD_SEEDS], ddof=1))
    print(f"GATE 0 (inverted) OK — NEW uncorrected C5_B3 mean "
          f"{base_mean:.14f} ({score(base_mean)}), seed std {base_std:.6e} "
          f"vs exp_030 policy {OLD_BASE_STD:.6e}", flush=True)

    # ---- THE HEADLINE MEASUREMENT ------------------------------------------- #
    # Did the new iteration policy shrink the seed spread of the base?  This is
    # a MEASUREMENT, not an adoption decision, and it is reported whatever it
    # says.  The adoption decision (did the new base BEAT the old one) is the
    # separate, unrelaxed 3-seed test below it.
    seed_std_measurement = {
        "what": "seed standard deviation (ddof=1) of the UNCORRECTED fold_2024 "
                "Brier over seeds 42 / 7 / 2024.  It is the denominator of "
                "every adoption threshold on this base, which is why exp_031's "
                "offset — worth +38.28802 points with 3/3 signs — was rejected.",
        "exp030_iteration_policy_seed_std": OLD_BASE_STD,
        "exp032_iteration_policy_seed_std": base_std,
        "ratio_new_over_old": float(base_std / OLD_BASE_STD),
        "shrank": bool(base_std < OLD_BASE_STD),
        "exp030_threshold_score_points": abs(score_delta(3.0 * OLD_BASE_STD)),
        "exp032_threshold_score_points": abs(score_delta(3.0 * base_std)),
        "exp031_measured_offset_gain_score_points":
            EXP031_ADOPTION["mean_delta_score_points"],
        "exp031_multiple_of_threshold": EXP031_ADOPTION[
            "MULTIPLE_OF_THE_THRESHOLD"],
        "naming_warning": "MULTIPLE_OF_THE_THRESHOLD and MULTIPLE_OF_THE_"
                          "CONTROL_STD differ by exactly 3x and must never be "
                          "quoted against each other.",
    }

    # ---- did the NEW BASE BEAT THE OLD BASE?  3 seeds, unrelaxed ------------ #
    bdelta = [base24[s] - OLD_BASE_BY_SEED[s] for s in FOLD_SEEDS]
    b_mean_d = float(np.mean(bdelta))
    b_thr = 3.0 * OLD_BASE_STD
    base_change_adoption = {
        "rule": "3 seeds same sign AND |mean delta| >= 3 x the CONTROL's seed "
                "std, where the CONTROL is exp_030's iteration policy.  NOT "
                "relaxed.",
        "control": "exp_030 row-ratio iteration policy (exp_031 base_no_offset)",
        "candidate": "exp_032 fitted-law iteration policy",
        "seeds": FOLD_SEEDS,
        "control_brier_by_seed": [OLD_BASE_BY_SEED[s] for s in FOLD_SEEDS],
        "candidate_brier_by_seed": [base24[s] for s in FOLD_SEEDS],
        "control_mean_brier": OLD_BASE_MEAN,
        "control_mean_score": score(OLD_BASE_MEAN),
        "candidate_mean_brier": base_mean, "candidate_mean_score": score(base_mean),
        "paired_delta_by_seed": bdelta,
        "paired_delta_score_by_seed": [score_delta(x) for x in bdelta],
        "mean_delta_brier": b_mean_d,
        "mean_delta_score_points": score_delta(b_mean_d),
        "control_seed_std_DENOMINATOR": OLD_BASE_STD,
        "candidate_seed_std": base_std,
        "threshold_abs_mean_delta_THIS_IS_THE_THRESHOLD": b_thr,
        "MULTIPLE_OF_THE_THRESHOLD": abs(b_mean_d) / b_thr,
        "MULTIPLE_OF_THE_CONTROL_STD": abs(b_mean_d) / OLD_BASE_STD,
        "signs_agree": bool(all(x < 0 for x in bdelta)
                            or all(x > 0 for x in bdelta)),
        "improves": bool(b_mean_d < 0),
        "passes_magnitude": bool(abs(b_mean_d) >= b_thr),
    }
    base_change_adoption["ADOPTED"] = bool(
        base_change_adoption["signs_agree"] and base_change_adoption["improves"]
        and base_change_adoption["passes_magnitude"])
    print("BASE CHANGE: " + json.dumps({
        k: base_change_adoption[k] for k in
        ("mean_delta_score_points", "signs_agree", "passes_magnitude",
         "MULTIPLE_OF_THE_THRESHOLD", "ADOPTED")}), flush=True)

    # ---- p distribution and p(1-p) ------------------------------------------ #
    pdist = {}
    for s in FOLD_SEEDS:
        pv = p24[s]
        v = pv * (1.0 - pv)
        pdist[str(s)] = {
            "n": int(len(pv)), "mean": float(pv.mean()), "sd": float(pv.std()),
            "min": float(pv.min()), "max": float(pv.max()),
            "p01": float(np.percentile(pv, 1)), "p99": float(np.percentile(pv, 99)),
            "var_p1mp_mean": float(v.mean()), "var_p1mp_sd": float(v.std()),
            "var_p1mp_cv": float(v.std() / v.mean()),
            "var_p1mp_min": float(v.min()), "var_p1mp_max": float(v.max()),
        }
    pdist["3seed_mean_of_sd"] = float(np.mean([pdist[str(s)]["sd"]
                                               for s in FOLD_SEEDS]))
    pdist["3seed_mean_of_var_cv"] = float(np.mean([pdist[str(s)]["var_p1mp_cv"]
                                                   for s in FOLD_SEEDS]))
    pdist["old_base_reference"] = {
        "sd": 0.038065, "range": [0.383, 0.635],
        "source": "exp_028 record of the LGBM_B3 base, quoted for contrast only"}

    # ---- beta: pooled over the supply blocks x block seeds ------------------ #
    y_sup = np.concatenate([y[V] for V in SUPPLY_BLOCKS for s in BLOCK_SEEDS])
    p_sup = np.concatenate([blk[s][V] for V in SUPPLY_BLOCKS for s in BLOCK_SEEDS])
    r_sup = y_sup - p_sup
    br_base_sup = float(np.mean((p_sup - y_sup) ** 2))
    sup_index = [(V, s) for V in SUPPLY_BLOCKS for s in BLOCK_SEEDS]

    def d_pooled(nm):
        return np.concatenate([T[nm][V] for V, s in sup_index])

    def d_seed(nm, s):
        return np.concatenate([T[nm][V] for V in SUPPLY_BLOCKS])

    per_seed_sup = {}
    for s in BLOCK_SEEDS:
        ys = np.concatenate([y[V] for V in SUPPLY_BLOCKS])
        ps = np.concatenate([blk[s][V] for V in SUPPLY_BLOCKS])
        per_seed_sup[s] = {"y": ys, "p": ps, "r": ys - ps,
                           "brier": float(np.mean((ps - ys) ** 2))}

    blockfit = {}
    for nm in CELLS + CELLS_RES:
        dp = d_pooled(nm)
        f1 = off.fit_slope(r_sup, dp)
        beta = f1["beta"]
        pb = off.apply_linear(p_sup, dp, beta)
        fit = {
            "beta_pooled": beta, "beta_se": f1["se"], "beta_t": f1["t"],
            "beta_unrounded": float(np.dot(dp, r_sup) / np.dot(dp, dp)),
            "mean_d2": f1["mean_d2"], "sd_d_blocks": float(np.std(dp)),
            "fitter": "offset.fit_slope (deployed code, rounds beta to 6dp)",
            "brier_blocks": float(np.mean((pb - y_sup) ** 2)),
            "n_clip_blocks": int((((p_sup + beta * dp) < EPS)
                                  | ((p_sup + beta * dp) > 1 - EPS)).sum()),
        }
        fit["score_blocks"] = score(fit["brier_blocks"])
        fit["delta_vs_base_blocks"] = fit["brier_blocks"] - br_base_sup
        # per block-seed beta and per-seed block surface
        ps_fit = {}
        for s in BLOCK_SEEDS:
            ds = d_seed(nm, s)
            b_s = off.fit_slope(per_seed_sup[s]["r"], ds)["beta"]
            pb_s = off.apply_linear(per_seed_sup[s]["p"], ds, b_s)
            ps_fit[str(s)] = {
                "beta": b_s,
                "brier_blocks": float(np.mean((pb_s - per_seed_sup[s]["y"]) ** 2))}
            ps_fit[str(s)]["score_blocks"] = score(ps_fit[str(s)]["brier_blocks"])
        fit["per_block_seed"] = ps_fit
        # per-block betas (2022 vs 2023), pooled over seeds
        fit["beta_per_block"] = {}
        for V in SUPPLY_BLOCKS:
            rr = np.concatenate([y[V] - blk[s][V] for s in BLOCK_SEEDS])
            dd = np.concatenate([T[nm][V] for s in BLOCK_SEEDS])
            fit["beta_per_block"][str(V)] = off.fit_slope(rr, dd)["beta"]
        # honest cross-block: fit on one block, score the other
        cb, num, den = {}, 0.0, 0
        for Vf, Ve in ((SUPPLY_BLOCKS[0], SUPPLY_BLOCKS[1]),
                       (SUPPLY_BLOCKS[1], SUPPLY_BLOCKS[0])):
            rr = np.concatenate([y[Vf] - blk[s][Vf] for s in BLOCK_SEEDS])
            dd = np.concatenate([T[nm][Vf] for s in BLOCK_SEEDS])
            c2 = off.fit_slope(rr, dd)["beta"]
            ye = np.concatenate([y[Ve] for s in BLOCK_SEEDS])
            pe = off.apply_linear(
                np.concatenate([blk[s][Ve] for s in BLOCK_SEEDS]),
                np.concatenate([T[nm][Ve] for s in BLOCK_SEEDS]), c2)
            b = float(np.mean((pe - ye) ** 2))
            cb[f"fit{Vf}_score{Ve}"] = {"beta": c2, "brier": b, "n": int(len(ye))}
            num += b * len(ye)
            den += len(ye)
        fit["crossblock_brier_pooled"] = float(num / den)
        fit["crossblock_score"] = score(fit["crossblock_brier_pooled"])
        fit["crossblock_detail"] = cb
        blockfit[nm] = fit
    print(f"[{time.strftime('%H:%M:%S')}] betas fitted on the supply "
          f"({len(CELLS)} declared + {len(CELLS_RES)} reserve cells)", flush=True)

    # ---- SELECTION — on the block surface, and nowhere else ----------------- #
    blk_surf = {nm: blockfit[nm]["brier_blocks"] for nm in CELLS}
    xbl_surf = {nm: blockfit[nm]["crossblock_brier_pooled"] for nm in CELLS}
    blk_surf_all = {nm: blockfit[nm]["brier_blocks"] for nm in CELLS + CELLS_RES}
    SELECTED = argmin_of(blk_surf)
    print(f"SELECTED (block surface, {len(CELLS)} declared cells): {SELECTED}",
          flush=True)

    enc = {}
    for tg in TARGETS:
        enc[f"declared_{tg}"] = enclosure(
            {k: blk_surf[cell_name(k, tg)] for k in K_GRID}, K_GRID)
        enc[f"widened_{tg}"] = enclosure(
            {k: blk_surf_all[cell_name(k, tg)] for k in K_ALL}, K_ALL)
    enc["reserve_note"] = (
        "K_RESERVE = [6400, 12800] was computed in the same pass so that an EDGE "
        "argmin on the declared grid could be widened without a second run.  The "
        "declared surface is the 18 cells of K_GRID; the widened rows are read "
        "ONLY to decide enclosure.")
    enc["declared_argmin_is_interior_both_targets"] = bool(
        enc["declared_self"]["enclosed_both_sides"]
        and enc["declared_league"]["enclosed_both_sides"])

    cross = {"block_surface": crossing(
        {k: blk_surf[cell_name(k, "self")] for k in K_GRID},
        {k: blk_surf[cell_name(k, "league")] for k in K_GRID}, K_GRID)}

    # ---- fold_2024 evaluation, AFTER the selection is fixed ----------------- #
    def evaluate(nm, beta, s):
        c = T[nm][VALID_SEASON]
        raw = p24[s] + beta * c
        p = off.apply_linear(p24[s], c, beta)
        se = (p - y24) ** 2
        seb = (p24[s] - y24) ** 2
        br = float(np.mean(se))
        out = {"cell": nm, "seed": s, "beta_applied": beta, "brier": br,
               "score": score(br), "delta_vs_base": float(br - base24[s]),
               "score_gain_vs_base": score_delta(br - base24[s]),
               "valid_R": seg(y24[isr24], p[isr24]),
               "valid_F": seg(y24[~isr24], p[~isr24]),
               "seg_hit": seg(y24[hit24], p[hit24]),
               "seg_miss": seg(y24[~hit24], p[~hit24]),
               "correction_shift": {
                   "n_clip": int(((raw < EPS) | (raw > 1 - EPS)).sum()),
                   "raw_min": float(raw.min()), "raw_max": float(raw.max()),
                   "mean_abs": float(np.mean(np.abs(p - p24[s]))),
                   "max_abs": float(np.max(np.abs(p - p24[s]))),
                   "sd_d": float(np.std(c))},
               "by_n_bin": {}, "by_pitcher_hand": {}}
        for bn, msk in bins24.items():
            if msk.any():
                out["by_n_bin"][bn] = {
                    "n": int(msk.sum()), "brier": float(np.mean(se[msk])),
                    "delta_vs_base_within_bin": float(np.mean(se[msk] - seb[msk]))}
        for h in (1, 2):
            mh = (ph24 == h)
            out["by_pitcher_hand"][str(h)] = {
                "n": int(mh.sum()), "brier": float(np.mean(se[mh])),
                "delta_vs_base": float(np.mean(se[mh] - seb[mh]))}
        return out

    fold_cells = {}
    for nm in CELLS + CELLS_RES:
        b = blockfit[nm]["beta_pooled"]
        per = {str(s): evaluate(nm, b, s) for s in FOLD_SEEDS}
        mb = float(np.mean([per[str(s)]["brier"] for s in FOLD_SEEDS]))
        fold_cells[nm] = {
            "beta_pooled_applied": b, "per_seed": per, "mean_brier": mb,
            "mean_score": score(mb),
            "mean_delta_vs_base": float(mb - base_mean),
            "mean_score_gain_vs_base": score_delta(mb - base_mean),
            "n_clip_total": int(sum(per[str(s)]["correction_shift"]["n_clip"]
                                    for s in FOLD_SEEDS))}
    f24_surf = {nm: fold_cells[nm]["mean_brier"] for nm in CELLS}
    ORACLE = argmin_of(f24_surf)

    # ---- adoption test on the SELECTED cell --------------------------------- #
    sel = fold_cells[SELECTED]
    cand = [sel["per_seed"][str(s)]["brier"] for s in FOLD_SEEDS]
    ctrl = [base24[s] for s in FOLD_SEEDS]
    dlt = [c - b for c, b in zip(cand, ctrl)]
    mean_d = float(np.mean(dlt))
    thr = 3.0 * base_std
    adoption = {
        "rule": "3 seeds same sign AND |mean delta| >= 3 x CONTROL seed std. "
                "NOT relaxed.",
        "selected_cell": SELECTED, "beta_applied": sel["beta_pooled_applied"],
        "seeds": FOLD_SEEDS,
        "control_brier_by_seed": ctrl, "candidate_brier_by_seed": cand,
        "control_mean_brier": base_mean, "control_mean_score": score(base_mean),
        "candidate_mean_brier": sel["mean_brier"],
        "candidate_mean_score": sel["mean_score"],
        "paired_delta_by_seed": dlt,
        "paired_delta_score_by_seed": [score_delta(x) for x in dlt],
        "mean_delta_brier": mean_d, "mean_delta_score_points": score_delta(mean_d),
        "control_seed_std_DENOMINATOR": base_std,
        "candidate_seed_std": float(np.std(cand, ddof=1)),
        "paired_delta_seed_std_REFERENCE_ONLY": float(np.std(dlt, ddof=1)),
        "threshold_abs_mean_delta_THIS_IS_THE_THRESHOLD": thr,
        "MULTIPLE_OF_THE_THRESHOLD": abs(mean_d) / thr,
        "MULTIPLE_OF_THE_CONTROL_STD": abs(mean_d) / base_std,
        "label_warning": "MULTIPLE_OF_THE_THRESHOLD and MULTIPLE_OF_THE_CONTROL_"
                         "STD differ by exactly 3x.  They are NOT interchangeable "
                         "and must never be compared across experiments without "
                         "checking which one the other experiment reported.",
        "signs_agree": bool(all(x < 0 for x in dlt) or all(x > 0 for x in dlt)),
        "improves": bool(mean_d < 0),
        "passes_magnitude": bool(abs(mean_d) >= thr),
    }
    adoption["ADOPTED"] = bool(adoption["signs_agree"] and adoption["improves"]
                               and adoption["passes_magnitude"])
    print("ADOPTION: " + json.dumps({k: adoption[k] for k in
          ("selected_cell", "mean_delta_score_points", "signs_agree",
           "passes_magnitude", "MULTIPLE_OF_THE_THRESHOLD", "ADOPTED")}),
          flush=True)

    # ---- sensitivity: seed-matched beta ------------------------------------- #
    seed_matched = {}
    for nm in (SELECTED, OLD_BASE_CELL):
        per = {}
        for s in FOLD_SEEDS:
            b = (blockfit[nm]["per_block_seed"][str(s)]["beta"]
                 if s in BLOCK_SEEDS else blockfit[nm]["beta_pooled"])
            per[str(s)] = {"beta": b, "brier": evaluate(nm, b, s)["brier"]}
        mb = float(np.mean([per[str(s)]["brier"] for s in FOLD_SEEDS]))
        seed_matched[nm] = {
            "policy": "seed 42 -> beta fitted on seed 42's blocks, seed 7 -> "
                      "seed 7's, seed 2024 -> pooled (no block prediction "
                      "exists for seed 2024)",
            "per_seed": per, "mean_brier": mb, "mean_score": score(mb),
            "vs_pooled_score": score_delta(mb - fold_cells[nm]["mean_brier"])}

    # ---- beta curve on the selected cell: is it exactly quadratic? ---------- #
    b_hat = blockfit[SELECTED]["beta_pooled"]
    dsel_sup = d_pooled(SELECTED)
    dsel_24 = T[SELECTED][VALID_SEASON]
    grid_b = [round(b_hat * m, 6) for m in
              (0.0, 0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0)]
    bcurve = {"beta_hat": b_hat, "grid": grid_b, "blocks": {}, "fold_2024": {}}
    for b in grid_b:
        pb = off.apply_linear(p_sup, dsel_sup, b)
        raw_b = p_sup + b * dsel_sup
        bcurve["blocks"][str(b)] = {
            "brier": float(np.mean((pb - y_sup) ** 2)),
            "n_clip": int(((raw_b < EPS) | (raw_b > 1 - EPS)).sum())}
        fb = []
        nclip = 0
        for s in FOLD_SEEDS:
            raw24 = p24[s] + b * dsel_24
            pf = off.apply_linear(p24[s], dsel_24, b)
            fb.append(float(np.mean((pf - y24) ** 2)))
            nclip += int(((raw24 < EPS) | (raw24 > 1 - EPS)).sum())
        bcurve["fold_2024"][str(b)] = {"mean_brier": float(np.mean(fb)),
                                       "mean_score": score(float(np.mean(fb))),
                                       "n_clip": nclip}
    # exact-quadratic test: fit a + b*x + c*x^2 through THREE points, then check
    # the remaining eight.  With no clipping the Brier in beta IS exactly
    # quadratic, so the residual must be at round-off level.
    quad = {}
    for lab, vals in (("blocks", [bcurve["blocks"][str(b)]["brier"] for b in grid_b]),
                      ("fold_2024", [bcurve["fold_2024"][str(b)]["mean_brier"]
                                     for b in grid_b])):
        x = np.array(grid_b, float)
        v = np.array(vals, float)
        idx = [0, len(x) // 2, len(x) - 1]
        A = np.vstack([np.ones(3), x[idx], x[idx] ** 2]).T
        coef = np.linalg.solve(A, v[idx])
        pred = coef[0] + coef[1] * x + coef[2] * x ** 2
        quad[lab] = {
            "coef_a_b_c_from_3_points": coef.tolist(),
            "max_abs_residual_on_other_points": float(np.max(np.abs(pred - v))),
            "max_rel_residual": float(np.max(np.abs(pred - v)) / float(np.mean(v))),
            "analytic_argmin_beta": float(-coef[1] / (2 * coef[2])),
            "is_exactly_quadratic_to_1e-12": bool(
                np.max(np.abs(pred - v)) < 1e-12),
            "total_clip_rows_over_curve": int(sum(
                bcurve[lab][str(b)]["n_clip"] for b in grid_b))}
    quad["note"] = ("Brier(p + beta*d) = E[(p-y)^2] - 2 beta E[d(y-p)] + beta^2 "
                    "E[d^2] is EXACTLY quadratic in beta whenever the clip never "
                    "fires.  A non-zero residual therefore means either the clip "
                    "fired or something else moved.")
    bcurve["quadratic_test"] = quad

    # ---- old-base surface, side by side (NOT mixed) -------------------------- #
    side = {"warning": "TWO DIFFERENT BASES.  exp_029's columns are LGBM_B3 "
                       "(3 seeds, blocks from exp_022/exp_024); exp_031's are "
                       "C5_B3 (2 block seeds, blocks regenerated under own early "
                       "stop).  Rows may be compared for SHAPE only; the Brier "
                       "levels are not commensurable and must never be pooled.",
            "cells": {}}
    for nm in CELLS:
        side["cells"][nm] = {
            "new_block_brier": blk_surf[nm], "new_block_score": score(blk_surf[nm]),
            "new_fold_brier": f24_surf[nm], "new_fold_score": score(f24_surf[nm]),
            "new_beta": blockfit[nm]["beta_pooled"],
            "old_block_brier": OLD_BLOCK.get(nm), "old_fold_brier": OLD_FOLD.get(nm),
            "old_block_score": (score(OLD_BLOCK[nm]) if nm in OLD_BLOCK else None),
            "old_fold_score": (score(OLD_FOLD[nm]) if nm in OLD_FOLD else None)}
    # rank correlation of the two block surfaces over the shared 18 cells
    shared = [nm for nm in CELLS if nm in OLD_BLOCK]
    rn = pd.Series([blk_surf[nm] for nm in shared]).rank().to_numpy()
    ro = pd.Series([OLD_BLOCK[nm] for nm in shared]).rank().to_numpy()
    side["n_shared_cells"] = len(shared)
    side["spearman_rank_corr_block_surfaces"] = float(np.corrcoef(rn, ro)[0, 1])
    side["old_base_block_argmin"] = min(shared, key=lambda n: OLD_BLOCK[n])
    side["old_base_fold_argmin"] = min(shared, key=lambda n: OLD_FOLD[n])
    side["new_base_block_argmin"] = SELECTED
    side["new_base_fold_argmin_DIAGNOSTIC"] = ORACLE
    side["argmin_moved"] = bool(SELECTED != side["old_base_block_argmin"])

    # ---- Stage A verdict: does C5_B3 still lose the blocks to LGBM_B3? ------ #
    stageA = {"question": "after regenerating the blocks under their OWN early "
                          "stop, does C5_B3 still lose to LGBM_B3 on the blocks?",
              "blocks": {}}
    for V in SUPPLY_BLOCKS:
        new = {s: float(np.mean((blk[s][V] - y[V]) ** 2)) for s in BLOCK_SEEDS}
        old = EXP030_BLOCK_STAGE2[V]
        lg = LGBM_BLOCK[V]
        nm_ = float(np.mean(list(new.values())))
        lm = float(np.mean([lg[s] for s in BLOCK_SEEDS]))
        stageA["blocks"][str(V)] = {
            "C5_B3_stage3_by_seed": {str(s): new[s] for s in BLOCK_SEEDS},
            "C5_B3_stage3_score_by_seed": {str(s): score(new[s]) for s in BLOCK_SEEDS},
            "C5_B3_stage2_exp030_by_seed": {str(s): old[s] for s in BLOCK_SEEDS},
            "repair_score_gain_by_seed": {str(s): score_delta(new[s] - old[s])
                                          for s in BLOCK_SEEDS},
            "LGBM_B3_exp024_by_seed": {str(s): lg[s] for s in lg},
            "C5_B3_stage3_mean": nm_, "C5_B3_stage3_mean_score": score(nm_),
            "LGBM_B3_mean_over_seeds_42_7": lm,
            "LGBM_B3_mean_score_over_seeds_42_7": score(lm),
            "C5_minus_LGBM_score_points": score_delta(nm_ - lm),
            "C5_wins_this_block": bool(nm_ < lm),
            "stack_iter_used": {str(s): blk_prov[f"block_{V}_seed{s}"]
                                ["stack_iter_used"] for s in BLOCK_SEEDS},
            "cell_iter_used": {str(s): blk_prov[f"block_{V}_seed{s}"]
                               ["cell_iter_used"] for s in BLOCK_SEEDS},
        }
    stageA["verdict"] = (
        "C5_B3 wins BOTH supply blocks after the repair"
        if all(v["C5_wins_this_block"] for v in stageA["blocks"].values()) else
        "C5_B3 still LOSES at least one supply block after the repair — the iter "
        "policy was not the (only) cause; C5 is weaker on small data, and that "
        "bears directly on how much to trust a beta fitted on this supply")
    print("STAGE A: " + stageA["verdict"], flush=True)

    # ---- write out ---------------------------------------------------------- #
    with open(os.path.join(OUT_DIR, "iter_policy.json")) as f:
        iter_policy = json.load(f)
    assert iter_policy["exp_id"] == "exp_032"
    for s in FOLD_SEEDS:
        assert (fold_prov[str(s)]["stack_iter_used"]
                == iter_policy["for_stage_C_fold_2024"]["stack_iter"]), s
        assert (fold_prov[str(s)]["cell_iter_used"]
                == iter_policy["for_stage_C_fold_2024"]["cell_iter"]), s

    out = {
        "exp_id": "exp_032",
        "title": "iteration-succession policy repaired — argmin fitted against "
                 "training rows on the block units, fold_2024 base rebuilt at "
                 "the fitted iterations, offset grid and beta re-measured on it",
        "primary_metric": "fold_2024 valid_all (R+F) Brier",
        "iteration_policy_applied": iter_policy,
        "SEED_STD_MEASUREMENT_HEADLINE": seed_std_measurement,
        "base_change_adoption": base_change_adoption,
        "supply_asymmetry_note":
            "the supply blocks (2022, 2023) sit at their OWN argmin, while the "
            "fold_2024 base sits at an EXTRAPOLATED iteration.  That asymmetry "
            "is deliberate — a deployment build cannot early-stop — but it "
            "means the beta fitted on the supply is fitted on residuals of a "
            "better-tuned model than the one it is applied to.",
        "conversion": {"formula": "score = 100000 * (1 - brier / 0.2498069)",
                       "sensitivity": "1e-06 Brier = 0.4003 score points",
                       "caveat": "every Brier here is full precision, so the "
                                 "+-0.20 rounding caveat does NOT apply"},
        "k_grid_declared": K_GRID, "k_reserve": K_RESERVE, "targets": TARGETS,
        "n_declared_cells": len(CELLS),
        "fold_seeds": FOLD_SEEDS, "block_seeds": BLOCK_SEEDS,
        "gate0_base_MUST_DIFFER": gate0,
        "gate1_block_bit_identity": gate1,
        "gate2_k0_self_equals_k0_league": gate2,
        "gate3_miss_rows_zero_all_k": gate4,
        "gate4_label_isolation": label_isolation,
        "gate5_asof_independent": asof_indep,
        "stage_A_block_repair": stageA,
        "block_provenance": blk_prov,
        "fold_2024_base_provenance": fold_prov,
        "base_no_offset_fold_2024": {
            "by_seed": {str(s): base24[s] for s in FOLD_SEEDS},
            "by_seed_score": {str(s): score(base24[s]) for s in FOLD_SEEDS},
            "mean_brier": base_mean, "mean_score": score(base_mean),
            "seed_std": base_std,
            "seed_std_score_points": abs(score_delta(base_std))},
        "base_no_offset_blocks_pooled": {"brier": br_base_sup,
                                         "score": score(br_base_sup)},
        "p_distribution_fold_2024": pdist,
        "block_fit": blockfit,
        "surfaces": {
            "note": "block = Brier on the pooled supply (2022+2023) x (seed 42, "
                    "seed 7) at the pooled beta.  THE ONLY SELECTION SURFACE.  "
                    "crossblock = fit one block, score the other.  fold_2024 is "
                    "DIAGNOSTIC and selects nothing.",
            "block_brier": blk_surf,
            "block_score": {nm: score(v) for nm, v in blk_surf.items()},
            "crossblock_brier": xbl_surf,
            "crossblock_score": {nm: score(v) for nm, v in xbl_surf.items()},
            "fold_2024_brier": f24_surf,
            "fold_2024_score": {nm: score(v) for nm, v in f24_surf.items()},
            "reserve_block_brier": {nm: blk_surf_all[nm] for nm in CELLS_RES},
            "reserve_fold_brier": {nm: fold_cells[nm]["mean_brier"]
                                   for nm in CELLS_RES}},
        "selection": {
            "selected_on": "block surface (supply blocks only)",
            "selected_cell": SELECTED,
            "block_argmin_self_only": argmin_of(
                {n: v for n, v in blk_surf.items() if n.endswith("_self")}),
            "block_argmin_league_only": argmin_of(
                {n: v for n, v in blk_surf.items() if n.endswith("_league")}),
            "crossblock_argmin": argmin_of(xbl_surf),
            "per_block_seed_argmin": {
                str(s): argmin_of({nm: blockfit[nm]["per_block_seed"][str(s)]
                                   ["brier_blocks"] for nm in CELLS})
                for s in BLOCK_SEEDS},
            "fold_2024_argmin_ORACLE_DIAGNOSTIC": ORACLE,
            "fold_2024_argmin_self_only": argmin_of(
                {n: v for n, v in f24_surf.items() if n.endswith("_self")}),
            "fold_2024_argmin_league_only": argmin_of(
                {n: v for n, v in f24_surf.items() if n.endswith("_league")}),
            "honest_selection_cost_score_points": score_delta(
                f24_surf[SELECTED] - f24_surf[ORACLE]),
            "old_base_selected_cell": OLD_BASE_CELL,
            "old_base_cell_on_new_fold_score": fold_cells[OLD_BASE_CELL]["mean_score"],
            "selected_vs_old_base_cell_score_points": score_delta(
                f24_surf[SELECTED] - f24_surf[OLD_BASE_CELL])},
        "enclosure": enc,
        "self_vs_league_crossing": cross,
        "fold_2024_cells": fold_cells,
        "adoption": adoption,
        "seed_matched_beta_sensitivity": seed_matched,
        "beta_curve": bcurve,
        "old_base_surface_side_by_side": side,
        "rulers": {
            "C5_B3_no_offset_exp032_NEW_ITER": {"brier": base_mean,
                                                "score": score(base_mean)},
            "C5_B3_no_offset_exp030_OLD_ITER": {"brier": OLD_BASE_MEAN,
                                                "score": score(OLD_BASE_MEAN)},
            "exp031_selected_cell": EXP031_SELECTED,
            "exp031_offset_gain_score_points": EXP031_ADOPTION[
                "mean_delta_score_points"],
            "old_base_full_offset_exp028_A1_score": 720.01460,
            "old_base_no_offset_exp028_A0_score": 645.56219,
            "offset_gain_on_OLD_base_score_points": 74.45241,
            "offset_gain_on_NEW_base_score_points": score_delta(
                sel["mean_brier"] - base_mean)},
        "ds_diagnostics": ds_diag,
        "source_sha256": src_sha,
        "train_csv_sha256": csv_sha,
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__},
        "compute_time_min": round((time.time() - t_start) / 60, 3),
    }

    # candidate prediction vectors for the selected cell (fold_2024)
    saved = {}
    for s in FOLD_SEEDS:
        p = off.apply_linear(p24[s], T[SELECTED][VALID_SEASON],
                             sel["beta_pooled_applied"])
        pth = os.path.join(OUT_DIR, "pred",
                           f"pred_C5B3_offset_{SELECTED}_fold2024_seed{s}.npy")
        np.save(pth, p)
        saved[str(s)] = {"file": os.path.basename(pth), "sha256": sha256_file(pth),
                         "brier": float(np.mean((p - y24) ** 2))}
    out["candidate_predictions"] = saved

    out["exp031_side_by_side"] = {
        "warning": "SAME model family, SAME features, SAME supply blocks; the "
                   "only difference is the fold_2024 base's iteration counts.  "
                   "Rows ARE commensurable, unlike the exp_029 comparison above.",
        "exp031_base_mean_score": score(OLD_BASE_MEAN),
        "exp032_base_mean_score": score(base_mean),
        "exp031_selected_cell": EXP031_SELECTED,
        "exp032_selected_cell": SELECTED,
        "exp031_fold_score_at_its_cell": score(
            m31["surfaces"]["fold_2024_brier"][EXP031_SELECTED]),
        "exp032_fold_score_at_its_cell": fold_cells[SELECTED]["mean_score"],
        "exp031_block_surface": m31["surfaces"]["block_brier"],
        "exp032_block_surface": blk_surf,
        "block_surface_is_identical": bool(
            all(abs(m31["surfaces"]["block_brier"][nm] - blk_surf[nm]) < 1e-15
                for nm in CELLS)),
        "block_surface_identity_note":
            "the supply is byte-identical to exp_031's, so the ENTIRE block "
            "surface and every beta must reproduce exp_031 exactly.  If this "
            "is False something reused was not actually reused.",
        "exp031_adoption": EXP031_ADOPTION,
    }
    assert out["exp031_side_by_side"]["block_surface_is_identical"], \
        "the reused supply did not reproduce exp_031's block surface"

    with open(os.path.join(OUT_DIR, "metrics_grid.json"), "w") as f:
        json.dump(out, f, indent=1)

    # curves/ — flat csv for the report
    with open(os.path.join(OUT_DIR, "curves", "k_surface.csv"), "w") as f:
        f.write("k,target,beta_pooled,block_brier,block_score,crossblock_brier,"
                "fold_brier,fold_score,old_block_brier,old_fold_brier,declared\n")
        for k in K_ALL:
            for tg in TARGETS:
                nm = cell_name(k, tg)
                dec = "yes" if k in K_GRID else "reserve"
                ob = OLD_BLOCK.get(nm, "")
                of_ = OLD_FOLD.get(nm, "")
                f.write(f"{k},{tg},{blockfit[nm]['beta_pooled']},"
                        f"{blk_surf_all[nm]:.14f},{score(blk_surf_all[nm])},"
                        f"{blockfit[nm]['crossblock_brier_pooled']:.14f},"
                        f"{fold_cells[nm]['mean_brier']:.14f},"
                        f"{fold_cells[nm]['mean_score']},{ob},{of_},{dec}\n")
    with open(os.path.join(OUT_DIR, "curves", "beta_curve.csv"), "w") as f:
        f.write("beta,block_brier,block_nclip,fold_mean_brier,fold_score,"
                "fold_nclip\n")
        for b in grid_b:
            f.write(f"{b},{bcurve['blocks'][str(b)]['brier']:.14f},"
                    f"{bcurve['blocks'][str(b)]['n_clip']},"
                    f"{bcurve['fold_2024'][str(b)]['mean_brier']:.14f},"
                    f"{bcurve['fold_2024'][str(b)]['mean_score']},"
                    f"{bcurve['fold_2024'][str(b)]['n_clip']}\n")

    print(f"DONE in {out['compute_time_min']} min — selected {SELECTED}, "
          f"beta {sel['beta_pooled_applied']}, fold mean {sel['mean_score']} "
          f"(base {score(base_mean)}), ADOPTED {adoption['ADOPTED']}", flush=True)


if __name__ == "__main__":
    main()
