"""exp_026 — four post-hoc transforms of the SAME stored predictions.

NO TRAINING.  Every base vector is reused from exp_020 / exp_022 / exp_024 with
its sha256 asserted against the digest exp_024 recorded; a drift aborts the run.

EIGHT ARMS, THREE SEEDS (42, 7, 2024), one base per seed.

  C0  k=400 x league                          = exp_024 arm B = candidate J's d.
                                                CONTROL and reproduction gate.
  C1  C0 with the league backoff REMOVED on pitcher-hand-2 CELL-MISS rows only.
      NOT exp_025's B_ph1only: that one also removed L from pitcher-hand-2 rows
      that HIT their personal cell.  exp_025 section 10-2 located the loss on the
      MISS rows (34,280 of them), so this arm removes exactly those and leaves
      every hit row of every pitcher hand bit-identical to C0.
  C2  k=400 x self (hard backoff).  Attribution: C0 - C2 is what moving the
      shrink target from the pitcher's own rate to the league buys AT FIXED k.
  C3  k=1600 x league     grid extension upward
  C4  k=3200 x league     grid extension upward
  C5  C0 + d_r:  p' = clip(p + beta_s*d_s + beta_r*d_r), d_r also k=400 x league.
      MAIN ARM.
  C6  as C5 but d_r at k=65 (exp_018's variance-decomposition prior for the
      reverse axis, which is not the prior for the success axis).
  C7  C1 + C5.  TWO variables change at once; its delta is attributable to the
      COMBINATION and to neither part.

WHAT `d_r` IS
-------------
`d_s` is the deviation of a pitcher's SUCCESS rate against a batter hand from
his own unconditional success rate.  `d_r` is the same object built on the
MARGINAL REVERSE rate (`is_rev_only + is_both`, the quantity the organisers'
`asof_pitcher_reverse_rate` measures) — not how often he succeeds, but which way
he misses.  exp_018's pooled variance decomposition put the reverse axis at
true SD 5.20pp / reliability 0.80 against the success axis's 4.75pp / 0.69, with
essentially zero correlation to the unconditional level.  That measurement is a
POOLED, all-seasons optimum; under the as-of discipline it can only get worse,
and exp_018 already saw exactly that happen to the batter-hand axis.  Nothing
here assumes it survives.

BETA
----
Never an axis.  For every (arm, seed) the betas are the closed-form no-intercept
least squares of the residual (y - p) on that arm's own deviation column(s),
fitted on supply blocks 2022 + 2023 OF THAT SEED.  One-term arms use
`offset.fit_slope` (the deployed code, which rounds to 6 decimals).  Two-term
arms solve the 2x2 normal equations JOINTLY — which is not the same as fitting
each slope alone whenever the two columns are correlated, and the correlation is
measured and reported.

SELECTION
---------
Anything that is CHOSEN is chosen on the supply blocks and on nothing else:
the k grid extension (C3 / C4) and the d_r shrink constant (C5 vs C6).  The
primary selection surface is the block Brier at the block-fitted beta, which is
the surface exp_023 registered; the honest cross-block surface (fit on 2022,
score 2023, and the reverse) is computed as well and reported whether or not it
agrees.  C1 and C2 are NOT selections — they are pre-registered contrasts whose
fold_2024 values are reported as measured.  fold_2024 is read once, after every
beta and every choice is fixed.
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

DATA_PATH = os.path.expanduser("~/SungjinPark/data/train.csv")
OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_026")
EXP020_PRED = os.path.expanduser(
    "~/SungjinPark/experiments/exp_020/pred/pred_E0_B3_P4iter_seed42_fold_2024.npy")
EXP022_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_022")
EXP024_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_024")
EXP024_METRICS = os.path.join(EXP024_DIR, "metrics.json")

SEEDS = [42, 7, 2024]
GATE_SEED = 42
GATE_TOL = 5e-6

TARGET = "control_success"
DROP_COLS = ["row_id", TARGET]
F_EXCLUDE_MAX_SEASON = 2022
SUPPLY_BLOCKS = [2022, 2023]
VALID_SEASON = 2024
EPS = 1e-6
BSS_REF_BRIER = 0.2498069

K_REF = 111                      # the deployed success-axis k (equivalence check)
K_S = 400                        # candidate J's success-axis k
KS_S_GRID = [1600, 3200]         # the upward extension
K_R_MAIN = 400
K_R_ALT = 65                     # exp_018's reverse-axis variance-decomposition k
PH2 = 2

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
}

# ARM -> ordered list of deviation-column names (see TERMS below).
ARMS = {
    "C0": ["s400_league"],
    "C1": ["s400_league_ph2missoff"],
    "C2": ["s400_self"],
    "C3": ["s1600_league"],
    "C4": ["s3200_league"],
    "C5": ["s400_league", "r400_league"],
    "C6": ["s400_league", "r65_league"],
    "C7": ["s400_league_ph2missoff", "r400_league"],
}
ARM_SPEC = {
    "C0": "k=400 x league  (= exp_024 arm B = the d of candidate J).  CONTROL.",
    "C1": "C0 with L forced to 0 on pitcher_hand==2 CELL-MISS rows only; every "
          "hit row and every pitcher-hand-1 row is bit-identical to C0",
    "C2": "k=400 x self (hard backoff): d = d_s(400) + (cell miss)*L",
    "C3": "k=1600 x league",
    "C4": "k=3200 x league",
    "C5": "C0 + d_r at k=400 x league, betas fitted JOINTLY (2x2, no intercept)",
    "C6": "as C5 but d_r at k=65",
    "C7": "C1 + d_r at k=400 x league (TWO variables vs C0 — not attributable)",
}
# contrasts reported with the adoption rule
CONTRASTS = [
    ("C5_minus_C0", "C5", "C0", "MAIN: does adding d_r beat one term"),
    ("C1_minus_C0", "C1", "C0", "pure loss removal (pre-registered)"),
    ("C0_minus_C2", "C0", "C2", "what the league shrink target buys at k=400"),
    ("C6_minus_C0", "C6", "C0", "d_r at k=65 instead of 400"),
    ("C5_minus_C6", "C5", "C6", "d_r shrink constant, 400 vs 65"),
    ("C7_minus_C0", "C7", "C0", "combination, two variables — not attributable"),
    ("C7_minus_C5", "C7", "C5", "loss removal on top of d_r"),
    ("C3_minus_C0", "C3", "C0", "k grid extension 1600"),
    ("C4_minus_C0", "C4", "C0", "k grid extension 3200"),
]
BLOCK_SELECTED = {"k_grid": ["C0", "C3", "C4"], "d_r_k": ["C5", "C6"]}


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
    """LB-scale conversion.  A 6-decimal Brier carries +-0.2002 here, so the
    trailing digits are a machine transform of a rounded input, not a
    measurement."""
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


# --------------------------------------------------------------------------- #
def main():
    t_start = time.time()
    os.makedirs(os.path.join(OUT_DIR, "pred"), exist_ok=True)

    csv_sha = sha256_file(DATA_PATH)
    assert csv_sha == TRAIN_CSV_SHA, ("train.csv SHA MISMATCH", csv_sha)
    src_sha = {f: sha256_file(os.path.join(SRC_DIR, f))
               for f in sorted(os.listdir(SRC_DIR)) if f.endswith(".py")}
    for f, want in SHA_INHERITED.items():
        assert src_sha[f] == want, ("INHERITED SOURCE DRIFTED", f, src_sha[f], want)
    print("inherited sha OK — backoff.py "
          f"{src_sha['backoff.py'][:16]}..., dev_k.py {src_sha['dev_k.py'][:16]}...",
          flush=True)

    # ---- the record exp_024 left.  Gate Briers are READ, never transcribed. -- #
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

    GATE_EXPECT = {int(s): m24_rec["per_seed"][str(s)]["B_candidate"]["brier"]
                   for s in SEEDS}
    A_DEPLOYED_REF = {int(s): m24_rec["per_seed"][str(s)]["A_control"]["brier"]
                      for s in SEEDS}
    print(f"exp_024 record read (metrics.json sha {exp024_metrics_sha[:16]}...); "
          f"gate expectations = {GATE_EXPECT}", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] loading {DATA_PATH}", flush=True)
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    assert df["row_id"].is_unique
    assert df["season"].is_monotonic_increasing
    provided_cols = [c for c in df.columns if c not in DROP_COLS]
    assert len(provided_cols) == 47

    rec = recover(df, verbose=False)
    assert rec["success_recovery"]["agree_pct"] == 100.0

    # ---- GATE 3: recovered labels may never be an input ---------------------- #
    # There is no training here, so the machine-checked form of the boundary is
    # (a) the 47 provided columns plus the platoon lookups carry no recovered
    # column, and (b) every deviation this file builds is CONSTANT within
    # (season, pitcher_id, batter_hand, pitcher_hand) — a row-level recovered
    # label reaching a row's own `d` would make it vary within that group.
    # (b) is asserted after the deviations exist, below.
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
    bh_all = df[bk.BAT_HAND].to_numpy()
    row_id = df["row_id"].to_numpy()

    # ---- deviations.  SEED-INDEPENDENT (they come from labels, not the model) #
    print(f"[{time.strftime('%H:%M:%S')}] platoon lookups + k=111 reference",
          flush=True)
    lu_diag, src_fp = fp.build_season_forward_chained(df, ok, succ_season, f_excl,
                                                      verbose=False)
    dev_ref, hit_ref, src_dev, dev_diag = off.build_deviations(
        df, ok, succ_season, f_excl, verbose=False)
    assert fp.RATE_SPEC[bk.SUCC_COL][1] == K_REF

    print(f"[{time.strftime('%H:%M:%S')}] success axis via BYTE-PRESERVED dev_k.py",
          flush=True)
    ks_s = [K_REF, K_S] + KS_S_GRID
    d_s, hit, n_cell0, src_k, dk_diag = dk.build_multi_k(
        df, ok, succ_season, f_excl, ks_s, verbose=False)
    assert np.array_equal(hit, hit_ref)

    print(f"[{time.strftime('%H:%M:%S')}] league table via BYTE-PRESERVED backoff.py",
          flush=True)
    L_s, has_Ls, src_Ls, Ls_diag = bk.build_league_dev(df, ok, succ_season, f_excl,
                                                       verbose=False)

    # ---- the generic module, PROVEN equal to the audited one on `succ` ------- #
    print(f"[{time.strftime('%H:%M:%S')}] devgen equivalence check on key='succ'",
          flush=True)
    g_s, g_hit, g_ncell, _, _ = dg.build_multi_k(df, ok, succ_season, f_excl,
                                                 ks_s, "succ", verbose=False)
    gL, g_hasL, _, _ = dg.build_league_dev(df, ok, succ_season, f_excl, "succ",
                                           verbose=False)
    devgen_equiv = {
        "what": "devgen at key='succ' vs the byte-preserved dev_k.py / backoff.py",
        "personal_bit_identical_by_k": {
            str(k): bool(np.array_equal(g_s[k], d_s[k])) for k in ks_s},
        "hit_mask_bit_identical": bool(np.array_equal(g_hit, hit)),
        "n_cell_bit_identical": bool(np.array_equal(
            np.nan_to_num(g_ncell, nan=-1.0), np.nan_to_num(n_cell0, nan=-1.0))),
        "league_bit_identical": bool(np.array_equal(gL, L_s)),
        "league_coverage_bit_identical": bool(np.array_equal(g_hasL, has_Ls)),
    }
    assert all(devgen_equiv["personal_bit_identical_by_k"].values()), devgen_equiv
    assert devgen_equiv["hit_mask_bit_identical"], devgen_equiv
    assert devgen_equiv["n_cell_bit_identical"], devgen_equiv
    assert devgen_equiv["league_bit_identical"], devgen_equiv
    assert devgen_equiv["league_coverage_bit_identical"], devgen_equiv
    print("DEVGEN EQUIVALENCE: " + json.dumps(devgen_equiv), flush=True)
    del g_s, gL

    # ---- the reverse axis ---------------------------------------------------- #
    print(f"[{time.strftime('%H:%M:%S')}] reverse axis (NEW): personal + league",
          flush=True)
    ks_r = [K_R_ALT, K_R_MAIN]
    d_r, hit_r, n_cell_r, src_r, dr_diag = dg.build_multi_k(
        df, ok, succ_season, f_excl, ks_r, "rev", verbose=True)
    L_r, has_Lr, src_Lr, Lr_diag = dg.build_league_dev(df, ok, succ_season, f_excl,
                                                       "rev", verbose=True)
    # the cell structure is the SAME table; only the numerator differs
    assert np.array_equal(hit_r, hit), "reverse axis disagrees on cell hits"
    assert np.array_equal(np.nan_to_num(n_cell_r, nan=-1.0),
                          np.nan_to_num(n_cell0, nan=-1.0))
    assert np.array_equal(has_Lr, has_Ls), "reverse league coverage differs"

    # ---- k=111 expanded-vs-literal equivalence (inherited check) ------------- #
    dd111 = d_s[K_REF] - dev_ref["d_s"]
    ddr = d_r[K_R_ALT] - dev_ref["d_r"]        # offset.py's d_r is at k=65
    equiv = {
        "form_used": "(s_cell - n_cell*overall)/(n_cell + k)  [expanded, exp_023]",
        "max_abs_diff_ds_k111_vs_inherited_literal_form": float(np.max(np.abs(dd111))),
        "max_abs_diff_dr_k65_vs_inherited_literal_form": float(np.max(np.abs(ddr))),
        "tolerance_used": 1e-12,
        "miss_rows_exactly_zero_ds": {str(k): bool(np.all(d_s[k][~hit] == 0.0))
                                      for k in ks_s},
        "miss_rows_exactly_zero_dr": {str(k): bool(np.all(d_r[k][~hit] == 0.0))
                                      for k in ks_r},
    }
    assert equiv["max_abs_diff_ds_k111_vs_inherited_literal_form"] < 1e-12, equiv
    assert equiv["max_abs_diff_dr_k65_vs_inherited_literal_form"] < 1e-12, equiv
    assert all(equiv["miss_rows_exactly_zero_ds"].values()), equiv
    assert all(equiv["miss_rows_exactly_zero_dr"].values()), equiv

    Ls_eff = np.where(has_Ls, L_s, 0.0)
    Lr_eff = np.where(has_Lr, L_r, 0.0)

    # ---- C1's surgical removal: pitcher-hand-2 CELL-MISS rows only ----------- #
    is_ph2 = (ph_all == PH2)
    Ls_c1 = np.where(is_ph2 & ~hit, 0.0, Ls_eff)
    changed = (Ls_c1 != Ls_eff)
    c1_check = {
        "rule": "L := 0 where pitcher_hand == 2 AND the personal cell is EMPTY",
        "differs_from_exp025_B_ph1only": "exp_025 zeroed L on ALL pitcher-hand-2 "
                                         "rows, hit rows included; this zeroes "
                                         "only the miss rows",
        "rows_ph2": int(is_ph2.sum()),
        "rows_ph2_miss": int((is_ph2 & ~hit).sum()),
        "rows_changed": int(changed.sum()),
        "all_changed_rows_are_ph2_and_miss": bool(
            np.all(is_ph2[changed] & ~hit[changed])),
        "every_hit_row_untouched": bool(np.array_equal(Ls_c1[hit], Ls_eff[hit])),
        "every_ph1_row_untouched": bool(
            np.array_equal(Ls_c1[~is_ph2], Ls_eff[~is_ph2])),
        "backoff_py_unmodified_sha256": src_sha["backoff.py"],
    }
    assert c1_check["all_changed_rows_are_ph2_and_miss"], c1_check
    assert c1_check["every_hit_row_untouched"], c1_check
    assert c1_check["every_ph1_row_untouched"], c1_check
    print("C1 CHECK: " + json.dumps(c1_check, ensure_ascii=False), flush=True)

    # ---- the deviation columns ---------------------------------------------- #
    TERMS = {
        "s400_league": dk.build_arm(d_s[K_S], hit, Ls_eff, n_cell0, K_S, "league"),
        "s400_league_ph2missoff": dk.build_arm(d_s[K_S], hit, Ls_c1, n_cell0,
                                               K_S, "league"),
        "s400_self": dk.build_arm(d_s[K_S], hit, Ls_eff, n_cell0, K_S, "self"),
        "s1600_league": dk.build_arm(d_s[1600], hit, Ls_eff, n_cell0, 1600, "league"),
        "s3200_league": dk.build_arm(d_s[3200], hit, Ls_eff, n_cell0, 3200, "league"),
        "r400_league": dk.build_arm(d_r[K_R_MAIN], hit, Lr_eff, n_cell0,
                                    K_R_MAIN, "league"),
        "r65_league": dk.build_arm(d_r[K_R_ALT], hit, Lr_eff, n_cell0,
                                   K_R_ALT, "league"),
    }
    # C1 may only differ from C0 on pitcher-hand-2 miss rows
    diffc1 = TERMS["s400_league_ph2missoff"] != TERMS["s400_league"]
    assert np.all(is_ph2[diffc1] & ~hit[diffc1]), "C1 moved a row it must not"
    # `self` and `league` agree on miss rows by construction (k/(n+k) == 1 there)
    assert np.array_equal(TERMS["s400_self"][~hit], TERMS["s400_league"][~hit])
    assert np.array_equal(TERMS["s400_self"][hit], d_s[K_S][hit])

    # ---- GATE 3(b): every deviation is constant within its lookup group ------ #
    print(f"[{time.strftime('%H:%M:%S')}] label-isolation check (group constancy)",
          flush=True)
    gcodes = pd.DataFrame({"s": season, "p": df[fp.PITCHER].to_numpy(),
                           "h": df[fp.HAND].to_numpy(), "ph": ph_all}
                          ).groupby(["s", "p", "h", "ph"], sort=False,
                                    dropna=False).ngroup().to_numpy()
    const = {}
    for nm, v in list(TERMS.items()) + [("d_s_raw_k400", d_s[K_S]),
                                        ("d_r_raw_k400", d_r[K_R_MAIN])]:
        first = pd.Series(v).groupby(gcodes, sort=False).transform("first").to_numpy()
        const[nm] = bool(np.array_equal(v, first))
    label_isolation["check_b_constant_within_season_pitcher_hand"] = const
    label_isolation["n_lookup_groups"] = int(gcodes.max() + 1)
    assert all(const.values()), ("a deviation varies inside its own lookup group — "
                                 "a row-level label reached it", const)
    print("LABEL ISOLATION: " + json.dumps(label_isolation), flush=True)

    # ---- as-of, independently re-verified (both axes) ----------------------- #
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
        "tables_covered": ["personal_success", "personal_reverse (NEW)",
                           "league_success", "league_reverse (NEW)"],
        "note": "the reverse tables use the IDENTICAL row masks as the success "
                "tables (asserted below), so one check covers both numerators",
        "per_season": {}}
    for s in sorted(set(season.tolist()))[1:]:
        m_pers = base_ok & (season <= s - 1) & (succ_season <= s - 1)
        m_leag = m_pers & hand_ok
        rec_s = {}
        for name, m in (("personal_success_and_reverse", m_pers),
                        ("league_success_and_reverse", m_leag)):
            ids = row_id[m]
            mx = int(s_map.reindex(pd.Index(ids)).max())
            assert mx <= s - 1, ("AS-OF VIOLATION (independent)", name, s, mx)
            rec_s[name] = {"contributing_rows": int(m.sum()),
                           "max_season_of_contributors_independent": mx,
                           "allowed_max": int(s - 1)}
        asof_indep["per_season"][str(int(s))] = rec_s
    assert max(SUPPLY_BLOCKS) < VALID_SEASON
    print("AS-OF (independent, both axes) OK", flush=True)

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
    d_ctrl = {V: dev_ref["d_s"][season == V] for V in SUPPLY_BLOCKS + [VALID_SEASON]}

    # ---- d_s / d_r correlation, the risk this experiment was warned about ---- #
    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    sup_slice = np.concatenate([np.flatnonzero(season == V) for V in SUPPLY_BLOCKS])
    corr_report = {}
    for tag, idx in (("supply_blocks_2022_2023", sup_slice),
                     ("fold_2024", np.flatnonzero(m24))):
        h = hit[idx]
        corr_report[tag] = {
            "corr_d_s_arm_vs_d_r_arm_k400_all_rows": corr(
                TERMS["s400_league"][idx], TERMS["r400_league"][idx]),
            "corr_d_s_arm_vs_d_r_arm_k400_hit_rows": corr(
                TERMS["s400_league"][idx][h], TERMS["r400_league"][idx][h]),
            "corr_raw_d_s_vs_raw_d_r_k400_hit_rows": corr(
                d_s[K_S][idx][h], d_r[K_R_MAIN][idx][h]),
            "corr_d_s_arm_vs_d_r_arm_k65_all_rows": corr(
                TERMS["s400_league"][idx], TERMS["r65_league"][idx]),
            "sd_d_s_arm": float(np.std(TERMS["s400_league"][idx])),
            "sd_d_r_arm_k400": float(np.std(TERMS["r400_league"][idx])),
            "sd_d_r_arm_k65": float(np.std(TERMS["r65_league"][idx])),
            "cell_hit_pct": round(100.0 * float(h.mean()), 3),
        }
    print("d_s / d_r CORRELATION: " + json.dumps(corr_report), flush=True)

    hn = n_cell24[hit24]
    q1, q2 = np.percentile(hn, [33.3333, 66.6667])
    bins24 = {
        "miss_n0": ~hit24,
        f"hit_low_n_le_{int(q1)}": hit24 & (n_cell24 <= q1),
        f"hit_mid_n_{int(q1)}_{int(q2)}": (hit24 & (n_cell24 > q1)
                                           & (n_cell24 <= q2)),
        f"hit_high_n_gt_{int(q2)}": hit24 & (n_cell24 > q2),
        "fx_n_1_99": hit24 & (n_cell24 < 100),
        "fx_n_100_499": hit24 & (n_cell24 >= 100) & (n_cell24 < 500),
        "fx_n_500_1999": hit24 & (n_cell24 >= 500) & (n_cell24 < 2000),
        "fx_n_2000plus": hit24 & (n_cell24 >= 2000),
    }

    # ---- base vectors, all reused, all sha-asserted -------------------------- #
    def load_bases(s):
        prov = {}
        if s == GATE_SEED:
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

    ARM_LIST = list(ARMS)
    SAVE_ARMS = ["C0", "C1", "C5", "C6", "C7"]

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

        # --- single-term betas, for the "one term vs two" comparison ---------- #
        single = {}
        for nm in TERMS:
            dd = np.concatenate([T[nm][V] for V in SUPPLY_BLOCKS])
            single[nm] = off.fit_slope(r_sup, dd)
        beta_ctrl_plain_ds = off.fit_slope(
            r_sup, np.concatenate([d_ctrl[V] for V in SUPPLY_BLOCKS]))["beta"]

        # --- per-arm betas, fitted on the supply blocks ONLY ------------------ #
        blockfit = {}
        for a, terms in ARMS.items():
            cols = [np.concatenate([T[nm][V] for V in SUPPLY_BLOCKS])
                    for nm in terms]
            if len(cols) == 1:
                f1 = off.fit_slope(r_sup, cols[0])
                betas = [f1["beta"]]
                fit = {"n_terms": 1, "beta": betas, "se": [f1["se"]],
                       "t": [f1["t"]], "beta_unrounded": [
                           float(np.dot(cols[0], r_sup) / np.dot(cols[0], cols[0]))],
                       "fitter": "offset.fit_slope (deployed code, rounds to 6dp)"}
                # the generic solver must agree with the deployed one
                fj = dg.fit_joint(r_sup, cols)
                assert abs(fj["beta_unrounded"][0] - fit["beta_unrounded"][0]) < 1e-12
                fit["cond_gram"] = fj["cond_gram"]
            else:
                fj = dg.fit_joint(r_sup, cols)
                betas = fj["beta"]
                fit = dict(fj)
                fit["fitter"] = "devgen.fit_joint (2x2 normal equations, no " \
                                "intercept, rounded to 6dp)"
            pb = dg.apply_joint(p_sup, cols, betas)
            fit["brier_blocks"] = float(np.mean((pb - y_sup) ** 2))
            fit["delta_vs_base_blocks"] = fit["brier_blocks"] - br_base_blocks
            fit["score_blocks"] = score(fit["brier_blocks"])
            # honest cross-block surface: fit on one block, score the other
            cb = {}
            for Vf, Ve in ((SUPPLY_BLOCKS[0], SUPPLY_BLOCKS[1]),
                           (SUPPLY_BLOCKS[1], SUPPLY_BLOCKS[0])):
                cf = [T[nm][Vf] for nm in terms]
                ce = [T[nm][Ve] for nm in terms]
                rf = y[Vf] - blk[Vf]
                bb = (off.fit_slope(rf, cf[0])["beta"] if len(cf) == 1
                      else dg.fit_joint(rf, cf)["beta"])
                pe = dg.apply_joint(blk[Ve], ce, bb if isinstance(bb, list) else [bb])
                cb[f"fit{Vf}_score{Ve}"] = {
                    "beta": bb if isinstance(bb, list) else [bb],
                    "brier": float(np.mean((pe - y[Ve]) ** 2)),
                    "n": int(len(y[Ve]))}
            tot_n = sum(v["n"] for v in cb.values())
            fit["crossblock_brier_pooled"] = float(
                sum(v["brier"] * v["n"] for v in cb.values()) / tot_n)
            fit["crossblock_detail"] = cb
            fit["beta_per_block"] = {
                str(V): (off.fit_slope(y[V] - blk[V], [T[nm][V] for nm in terms][0]
                                       )["beta"] if len(terms) == 1
                         else dg.fit_joint(y[V] - blk[V],
                                           [T[nm][V] for nm in terms])["beta"])
                for V in SUPPLY_BLOCKS}
            blockfit[a] = fit

        seb = (p24 - y24) ** 2

        def evaluate(a):
            terms = ARMS[a]
            betas = blockfit[a]["beta"]
            cols = [T[nm][VALID_SEASON] for nm in terms]
            p = dg.apply_joint(p24, cols, betas)
            if len(cols) == 1:                     # bitwise same as deployed path
                assert np.array_equal(p, off.apply_linear(p24, cols[0], betas[0]))
            shift = np.zeros_like(p24)
            per_term = {}
            for nm, c, b in zip(terms, cols, betas):
                per_term[nm] = {"beta": b,
                                "mean_abs_beta_d": float(np.mean(np.abs(b * c))),
                                "max_abs_beta_d": float(np.max(np.abs(b * c))),
                                "sd_d": float(np.std(c))}
                shift = shift + b * c
            raw = p24 + shift
            se = (p - y24) ** 2
            br = float(np.mean(se))
            out = {
                "arm": a, "spec": ARM_SPEC[a], "terms": terms,
                "betas_applied": betas,
                "brier": br, "score": score(br),
                "delta_vs_base_no_offset": float(br - br_base24),
                "score_gain_vs_base": score_delta(br - br_base24),
                "valid_R": seg(y24[isr24], p[isr24]),
                "valid_F": seg(y24[~isr24], p[~isr24]),
                "seg_miss": seg(y24[~hit24], p[~hit24]),
                "seg_hit": seg(y24[hit24], p[hit24]),
                "correction_shift_beta_d": {
                    "per_term": per_term,
                    "total_mean_abs": float(np.mean(np.abs(shift))),
                    "total_max_abs": float(np.max(np.abs(shift))),
                    "n_clip": int(((raw < EPS) | (raw > 1 - EPS)).sum()),
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
            return out, p

        res, preds = {}, {}
        for a in ARM_LIST:
            res[a], preds[a] = evaluate(a)
        for a in SAVE_ARMS:
            np.save(os.path.join(OUT_DIR, "pred",
                                 f"pred_{a}_seed{s}_fold_2024.npy"), preds[a])

        # C1 changes the DEVIATION only on pitcher-hand-2 miss rows, but its beta
        # is refitted, so its PREDICTIONS move everywhere.  Both facts recorded;
        # the scope claim is asserted on the deviation, above, not here.
        ph2miss24 = (ph24 == PH2) & ~hit24
        c1_scope_seed = {
            "rows_ph2_miss_in_fold_2024": int(ph2miss24.sum()),
            "deviation_differs_outside_ph2_miss": int(np.sum(
                T["s400_league_ph2missoff"][VALID_SEASON][~ph2miss24]
                != T["s400_league"][VALID_SEASON][~ph2miss24])),
            "beta_C0": blockfit["C0"]["beta"], "beta_C1": blockfit["C1"]["beta"],
            "note": "predictions differ on every row because beta is refitted",
        }
        assert c1_scope_seed["deviation_differs_outside_ph2_miss"] == 0

        per_seed[str(s)] = {
            "C1_scope_this_seed": c1_scope_seed,
            "base_no_offset": {"fold_2024_brier": br_base24,
                               "fold_2024_score": score(br_base24),
                               "blocks_brier": br_base_blocks},
            "beta_control_plain_ds_k111": beta_ctrl_plain_ds,
            "single_term_betas": {nm: {"beta": v["beta"], "se": v["se"],
                                       "t": v["t"],
                                       "expected_brier_gain": v["expected_brier_gain"]}
                                  for nm, v in single.items()},
            "arms": res,
            "block_fit": blockfit,
            "contrasts": {name: {
                "brier_delta": res[a]["brier"] - res[b]["brier"],
                "score_delta": score_delta(res[a]["brier"] - res[b]["brier"]),
                "what": what}
                for name, a, b, what in CONTRASTS},
            "arm_pred_gap_vs_C0": {
                a: {"mean_abs": float(np.mean(np.abs(preds[a] - preds["C0"]))),
                    "max_abs": float(np.max(np.abs(preds[a] - preds["C0"])))}
                for a in ARM_LIST if a != "C0"},
        }
        print("  seed %d: " % s + "  ".join(
            f"{a} {res[a]['brier']:.12f}" for a in ARM_LIST), flush=True)
        print(f"    C5-C0 {res['C5']['brier'] - res['C0']['brier']:+.4e} "
              f"({score_delta(res['C5']['brier'] - res['C0']['brier']):+.5f})  "
              f"C1-C0 {res['C1']['brier'] - res['C0']['brier']:+.4e} "
              f"({score_delta(res['C1']['brier'] - res['C0']['brier']):+.5f})",
              flush=True)
        del p24, blk, preds

    # ---- GATE 1: C0 reproduces exp_024 arm B on all three seeds -------------- #
    gate = {"tolerance": GATE_TOL,
            "source": "exp_024/metrics.json per_seed[seed].B_candidate.brier — "
                      "READ from the record, not transcribed",
            "on_fail": "abort — the reused vectors or the inherited code are not "
                       "exp_024's and nothing downstream is readable",
            "checks": {}}
    allpass = True
    for s in SEEDS:
        got_b = per_seed[str(s)]["arms"]["C0"]["brier"]
        want = GATE_EXPECT[s]
        diff = abs(got_b - want)
        ok_ = bool(diff < GATE_TOL)
        allpass = allpass and ok_
        gate["checks"][f"C0_seed{s}"] = {
            "expected_exp024_arm_B": want, "got": got_b, "abs_diff": diff,
            "bit_identical": bool(got_b == want), "passed": ok_}
    gate["passed"] = bool(allpass)
    print("REPRODUCTION GATE: " + json.dumps(gate), flush=True)
    assert gate["passed"], "REPRODUCTION GATE FAILED: " + json.dumps(gate)

    # ---- adoption rule, applied exactly as registered ----------------------- #
    c0_seed_std = float(np.std([per_seed[str(s)]["arms"]["C0"]["brier"]
                                for s in SEEDS], ddof=1))
    threshold = 3.0 * c0_seed_std

    def judge(name):
        v = [per_seed[str(s)]["contrasts"][name]["brier_delta"] for s in SEEDS]
        mean = float(np.mean(v))
        signs = bool(all(x < 0 for x in v) or all(x > 0 for x in v))
        return {"by_seed": {str(s): x for s, x in zip(SEEDS, v)},
                "by_seed_score": {str(s): score_delta(x)
                                  for s, x in zip(SEEDS, v)},
                "signs_agree": signs,
                "mean": mean, "mean_score": score_delta(mean),
                "paired_delta_std_ddof1_REFERENCE": float(np.std(v, ddof=1)),
                "abs_mean": abs(mean),
                "threshold_3x_C0_seed_std": threshold,
                "passes_magnitude": bool(abs(mean) >= threshold),
                "ADOPT": bool(signs and abs(mean) >= threshold),
                "direction": ("improves (lower Brier)" if mean < 0
                              else "worsens" if mean > 0 else "flat")}

    adoption = {"rule": "3 seeds agree in sign AND |mean delta| >= 3 x C0 seed "
                        "std.  Not relaxed.  The paired-delta std is reported as "
                        "a reference number only and is not part of the rule.",
                "C0_seed_std_ddof1": c0_seed_std,
                "threshold": threshold,
                "threshold_score": score_delta(threshold),
                "verdict": {name: judge(name) for name, _, _, _ in CONTRASTS}}

    # ---- selection discipline ------------------------------------------------ #
    def pick(cands, surface):
        vals = {a: float(np.mean([per_seed[str(s)]["block_fit"][a][surface]
                                  for s in SEEDS])) for a in cands}
        return min(vals, key=vals.get), vals

    def pick24(cands):
        vals = {a: float(np.mean([per_seed[str(s)]["arms"][a]["brier"]
                                  for s in SEEDS])) for a in cands}
        return min(vals, key=vals.get), vals

    selection = {"discipline": "everything CHOSEN is chosen on the supply blocks; "
                               "fold_2024 is read afterwards and is reported as "
                               "the oracle counterfactual only",
                 "primary_surface": "block Brier at the block-fitted beta "
                                    "(exp_023's registered surface)",
                 "secondary_surface": "cross-block: fit 2022 -> score 2023 and the "
                                      "reverse, pooled",
                 "not_a_selection": ["C1", "C2"],
                 "groups": {}}
    for g, cands in BLOCK_SELECTED.items():
        b_arm, b_vals = pick(cands, "brier_blocks")
        x_arm, x_vals = pick(cands, "crossblock_brier_pooled")
        o_arm, o_vals = pick24(cands)
        gap = (float(np.mean([per_seed[str(s)]["arms"][b_arm]["brier"]
                              for s in SEEDS]))
               - float(np.mean([per_seed[str(s)]["arms"][o_arm]["brier"]
                                for s in SEEDS])))
        selection["groups"][g] = {
            "candidates": cands,
            "block_selected": b_arm, "block_brier_mean": b_vals,
            "crossblock_selected": x_arm, "crossblock_brier_mean": x_vals,
            "two_block_surfaces_agree": bool(b_arm == x_arm),
            "fold2024_oracle": o_arm, "fold2024_brier_mean": o_vals,
            "selection_matches_oracle": bool(b_arm == o_arm),
            "cost_of_honest_selection_brier": gap,
            "cost_of_honest_selection_score": score_delta(gap),
        }
    print("SELECTION: " + json.dumps(selection["groups"]), flush=True)

    arm_brier = {a: [per_seed[str(s)]["arms"][a]["brier"] for s in SEEDS]
                 for a in ARM_LIST}
    summary = {
        "arm_seed_table_brier": {a: {str(s): b for s, b in zip(SEEDS, v)}
                                 for a, v in arm_brier.items()},
        "arm_seed_table_score": {a: {str(s): score(b) for s, b in zip(SEEDS, v)}
                                 for a, v in arm_brier.items()},
        "arm_mean_brier": {a: float(np.mean(v)) for a, v in arm_brier.items()},
        "arm_mean_score": {a: score(float(np.mean(v)))
                           for a, v in arm_brier.items()},
        "arm_seed_std_ddof1": {a: float(np.std(v, ddof=1))
                               for a, v in arm_brier.items()},
        "exp024_A_deployed_reference_brier": A_DEPLOYED_REF,
        "beta_two_term_vs_one_term": {
            str(s): {
                "C5_joint": per_seed[str(s)]["block_fit"]["C5"]["beta"],
                "C6_joint": per_seed[str(s)]["block_fit"]["C6"]["beta"],
                "C7_joint": per_seed[str(s)]["block_fit"]["C7"]["beta"],
                "single_s400_league": per_seed[str(s)]["single_term_betas"][
                    "s400_league"]["beta"],
                "single_r400_league": per_seed[str(s)]["single_term_betas"][
                    "r400_league"]["beta"],
                "single_r65_league": per_seed[str(s)]["single_term_betas"][
                    "r65_league"]["beta"],
            } for s in SEEDS},
    }

    metrics = {
        "exp_id": "exp_026",
        "title": "d_r alongside d_s, the pitcher-hand-2 miss-row backoff removed, "
                 "k400 x self attribution, and the k grid extended — all post-hoc "
                 "transforms of stored predictions, no training",
        "no_training": True,
        "arms": ARM_SPEC,
        "arm_terms": ARMS,
        "seeds": SEEDS,
        "changed_vs_exp_025": [
            "NEW deviation axis d_r (marginal reverse rate) with its own personal "
            "and league tables, built by devgen.py",
            "NEW two-term joint beta fit (2x2 normal equations, no intercept)",
            "C1 removes the league backoff ONLY on pitcher-hand-2 CELL-MISS rows; "
            "exp_025's B_ph1only removed it on ALL pitcher-hand-2 rows",
            "k grid extended upward to 1600 and 3200",
            "k=400 x self added as the attribution control for the shrink target",
            "no training, no new base vectors — identical reuse to exp_025",
        ],
        "unchanged_vs_exp_025": [
            "base vectors and their sha256, exp_004 F-mask, clip 1e-6, supply "
            "blocks 2022+2023, closed-form no-intercept slope for one-term arms, "
            "backoff.py / dev_k.py / offset.py / features_platoon.py / labels.py / "
            "exp016_ref.py byte-identical (sha asserted)",
        ],
        "inherited_source_sha256": SHA_INHERITED,
        "source_sha256": src_sha,
        "train_csv_sha256": csv_sha,
        "base_vector_provenance": provenance,
        "base_vector_expected_sha256_from_exp024_record": {
            str(k): {str(kk): vv for kk, vv in v.items()}
            for k, v in BASE_SHA.items()},
        "exp024_metrics_json_sha256": exp024_metrics_sha,
        "reproduction_gate": gate,
        "devgen_equivalence_check": devgen_equiv,
        "C1_scope_check": c1_check,
        "label_isolation_check": label_isolation,
        "k_equivalence": equiv,
        "asof_independent_recheck": asof_indep,
        "d_s_d_r_correlation": corr_report,
        "adoption": adoption,
        "selection": selection,
        "summary": summary,
        "per_seed": per_seed,
        "n_bin_definition": {
            "hit_tertile_boundaries_n_cell": [float(q1), float(q2)],
            "bin_sizes": {b: int(m.sum()) for b, m in bins24.items()},
        },
        "L_success_by_cell_by_season": {s: v["L_by_cell"]
                                        for s, v in Ls_diag["per_season"].items()},
        "L_reverse_by_cell_by_season": {s: v["L_by_cell"]
                                        for s, v in Lr_diag["per_season"].items()},
        "reverse_axis_per_season_sd": dr_diag["per_season"],
        "conversion": {
            "formula": f"score = 100000 * (1 - brier / {BSS_REF_BRIER})",
            "sensitivity": "brier 1e-6 = score 0.4003; a 6-decimal Brier carries "
                           "+-0.2002 in the converted number",
            "caveat": "local measurement is at iter 650 (<=2023 training); the "
                      "deployed build is iter 7700 (<=2024).",
        },
        "leak_boundary": {
            "tables": "a season-s row reads tables built on seasons <= s-1 only, "
                      "for BOTH numerators; asserted at runtime inside "
                      "backoff.py / dev_k.py / devgen.py and again here by an "
                      "independent second read of train.csv",
            "recovered_labels": "`rev` = is_rev_only + is_both is a RECOVERED "
                                "label (row t+1).  It is used only as the "
                                "numerator of a frozen table, never as a row-level "
                                "input; group-constancy of every deviation inside "
                                "(season, pitcher, batter_hand, pitcher_hand) is "
                                "asserted, which a row-level leak would break",
            "beta_fit": "supply blocks 2022+2023 of the same seed, on that arm's "
                        "own columns; no arm's beta ever sees 2024",
            "arm_choice": "arms fixed by config.yaml before any 2024 number is "
                          "read; k and the d_r shrink constant are selected on the "
                          "blocks; C1 and C2 are pre-registered contrasts",
            "design_matrix": "unchanged — no deviation is ever a model feature, "
                             "and no training happens in this experiment",
            "missing": "no sentinel fill anywhere; L=0 and d=0 mean 'apply no "
                       "correction', the behaviour without the module",
        },
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__},
        "compute_time_min": round((time.time() - t_start) / 60, 2),
    }
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(json.loads(json.dumps(metrics, default=str)), f, indent=2,
                  ensure_ascii=False)

    print("\n--- ARM x SEED (fold_2024 valid_all Brier / score) ---", flush=True)
    print(f"{'arm':>5}" + "".join(f"{'seed ' + str(s):>32}" for s in SEEDS),
          flush=True)
    for a in ARM_LIST:
        print(f"{a:>5}" + "".join(
            f"{per_seed[str(s)]['arms'][a]['brier']:>20.12f}"
            f"{per_seed[str(s)]['arms'][a]['score']:>12.5f}" for s in SEEDS),
            flush=True)
    print("\n--- CONTRASTS (mean over 3 seeds) ---", flush=True)
    for name, _, _, what in CONTRASTS:
        v = adoption["verdict"][name]
        print(f"{name:>14}  mean {v['mean']:+.4e}  score {v['mean_score']:+9.5f}  "
              f"signs {str(v['signs_agree']):>5}  ADOPT {v['ADOPT']}   ({what})",
              flush=True)
    print(f"\n[{time.strftime('%H:%M:%S')}] DONE {metrics['compute_time_min']} min",
          flush=True)


if __name__ == "__main__":
    main()
