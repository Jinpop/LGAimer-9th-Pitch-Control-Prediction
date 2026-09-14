"""exp_025 — how much of `k=400 x league`'s gain sits on the unstable
pitcher-hand-2 league cells.

WHAT IS AND IS NOT MEASURED
---------------------------
No training.  Every base vector is REUSED from exp_020 / exp_022 / exp_024 with
its sha256 asserted; if any sha drifts the run aborts.

Three arms, three seeds (42, 7, 2024), all on the same base vectors:

  A_deployed   k = 111 x self at the CONTROL beta = exp_022 G1f = the `d` of the
               deployed candidates H/I.  Baseline.
  B_full       k = 400 x league, all four league cells = exp_024 arm B, the arm
               promoted at three seeds.  CONTROL for the ablation.
  B_ph1only    identical except L is FORCED TO 0 on every row whose
               pitcher_hand is 2.  ABLATION arm.

  B_full - B_ph1only  = the contribution of the pitcher-hand-2 cells
  B_ph1only - A_deployed = what is left once the unstable coefficient is gone

A fourth arm is computed and labelled SECONDARY because the literal ablation
above removes two different things at once from pitcher-hand-2 rows:

  B_ph1only_keephard   L = 0 on pitcher-hand-2 rows THAT HIT their personal
                       cell, but pitcher-hand-2 MISS rows keep L.  Miss-row L
                       is the hard backoff that arm A ALREADY applies, so this
                       arm isolates the part that is genuinely new to `league`
                       (league mass landing on cell-hit rows) from coverage the
                       deployed arm already had.

HOW THE ABLATION IS IMPLEMENTED — AND WHAT IS NOT TOUCHED
----------------------------------------------------------
`backoff.py` builds the league table and was independently recomputed by the
exp_022 leak audit.  It is BYTE-PRESERVED here (its sha is asserted against the
inherited value) and is NOT modified.  The ablation is a MASK APPLIED TO THE
TABLE'S OUTPUT at the point of use:

    L_ablate = where(pitcher_hand == 2, 0.0, L_eff)

L[i] is by construction the value of the (pitcher_hand[i], batter_hand[i]) cell,
so zeroing the rows with pitcher_hand == 2 is exactly zeroing the two cells
(2,1) and (2,2) of the table.  That equivalence is asserted, not assumed.

The mask is applied in EVERY season, so the supply blocks see the ablated `d`
too and each arm's beta is re-fitted for the arm it belongs to.

VINTAGE PROBE (diagnostic — it cannot change any arm)
------------------------------------------------------
fold_2024 normally uses the league table built on seasons <= 2023.  The probe
re-scores B_full with the <= 2022 vintage table swapped in for fold_2024 only.
This is NOT an as-of violation: <= 2022 is STRICTLY MORE PAST than <= 2023, so
it cannot leak.  It measures what one season of table staleness costs, which is
the only handle we have on how a <= 2024 table might behave in 2025.

The <= 2022 vintage is not recomputed.  It is READ OFF the L array on the
season-2023 rows, where `build_league_dev` has already written exactly that
table's cell values at full precision.  Cell-level agreement with the module's
own diagnostic dict is asserted.

BETA
----
Never an axis.  For every (arm, seed) it is the closed-form no-intercept slope
E[d r]/E[d^2] fitted on supply blocks 2022 + 2023 OF THAT SEED, on that arm's
own `d`.  Arm A keeps the CONTROL beta (fitted on the plain d_s), because that
is what makes it G1f, i.e. the thing actually deployed.  The vintage probe
keeps B_full's beta unchanged — only the fold_2024 table is swapped.

fold_2024 is read only after every beta is fixed.  Nothing — not the ablation,
not a beta, not an arm — is chosen by looking at 2024.
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
from labels import recover                                        # noqa: E402

DATA_PATH = os.path.expanduser("~/SungjinPark/data/train.csv")
OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_025")
EXP020_PRED = os.path.expanduser(
    "~/SungjinPark/experiments/exp_020/pred/pred_E0_B3_P4iter_seed42_fold_2024.npy")
EXP022_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_022")
EXP024_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_024")

SEEDS = [42, 7, 2024]
GATE_SEED = 42

TARGET = "control_success"
DROP_COLS = ["row_id", TARGET]
F_EXCLUDE_MAX_SEASON = 2022
SUPPLY_BLOCKS = [2022, 2023]
VALID_SEASON = 2024
PROBE_VINTAGE_SEASON = 2023      # rows of 2023 carry the <= 2022 vintage table
EPS = 1e-6
BSS_REF_BRIER = 0.2498069

K_A = 111
K_B = 400
ABLATE_HAND = 2

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

# The gate Briers below are transcribed from exp_025/config.yaml, which is the
# spec for this run.  Sixty-four-hex base-vector digests are NOT transcribed by
# hand — they are read out of exp_024/metrics.json, the record that produced
# them, and the loader asserts today's files hash to those recorded values
# (regulation #5).  A typo in a hand-copied digest would abort a correct run;
# reading the record removes that failure mode without weakening the check.
EXP024_METRICS = os.path.join(EXP024_DIR, "metrics.json")

GATE = {
    "tolerance": 5e-6,
    "source": "exp_025/config.yaml validation.gate, cross-checked against "
              "exp_024/metrics.json at runtime",
    "on_fail": "abort — the reused vectors or the inherited code are not "
               "exp_024's and nothing downstream is readable",
    "A_deployed": {42: 0.2480377446210693,
                   7: 0.24805404666887054,
                   2024: 0.2480433978958496},
    "B_full": {42: 0.24799968498605615,
               7: 0.24801856644746204,
               2024: 0.2480065100917605},
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
    print("inherited sha OK (backoff.py byte-preserved: "
          f"{src_sha['backoff.py'][:16]}...)", flush=True)

    # ---- the record exp_024 left: expected base digests and the gate Briers -- #
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
    # the config's gate numbers must be the ones exp_024 actually recorded
    gate_xcheck = {}
    for a, rec_key in (("A_deployed", "A_control"), ("B_full", "B_candidate")):
        for s in SEEDS:
            rec = m24_rec["per_seed"][str(s)][rec_key]["brier"]
            gate_xcheck[f"{a}_seed{s}"] = {"config": GATE[a][s], "exp024_json": rec,
                                           "identical": bool(rec == GATE[a][s])}
            assert rec == GATE[a][s], ("config gate != exp_024 record", a, s, rec)
    print(f"exp_024 record read (metrics.json sha {exp024_metrics_sha[:16]}...), "
          "gate values cross-checked", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] loading {DATA_PATH}", flush=True)
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    assert df["row_id"].is_unique
    assert df["season"].is_monotonic_increasing
    assert len([c for c in df.columns if c not in DROP_COLS]) == 47

    rec = recover(df, verbose=False)
    assert rec["success_recovery"]["agree_pct"] == 100.0

    season = df["season"].to_numpy()
    ok = df["recov_ok"].to_numpy()
    succ_season = df.groupby("pitcher_id", sort=False)["season"].shift(-1).to_numpy()
    f_excl = ((df["season"] <= F_EXCLUDE_MAX_SEASON)
              & (df["game_type"] == "F")).to_numpy()
    y_all = df[TARGET].to_numpy(dtype=np.float64)
    is_r_all = (df["game_type"] == "R").to_numpy()
    ph_all = df[bk.PIT_HAND].to_numpy()
    bh_all = df[bk.BAT_HAND].to_numpy()

    # ---- deviations.  SEED-INDEPENDENT (they come from labels, not the model) #
    print(f"[{time.strftime('%H:%M:%S')}] personal lookups + k=111 reference",
          flush=True)
    lu_diag, src_fp = fp.build_season_forward_chained(df, ok, succ_season, f_excl,
                                                      verbose=False)
    dev_ref, hit_ref, src_dev, dev_diag = off.build_deviations(
        df, ok, succ_season, f_excl, verbose=False)
    print(f"[{time.strftime('%H:%M:%S')}] league table (backoff.py, unmodified)",
          flush=True)
    L, has_L, src_L, L_diag = bk.build_league_dev(df, ok, succ_season, f_excl,
                                                  verbose=False)
    assert fp.RATE_SPEC[bk.SUCC_COL][1] == K_A

    print(f"[{time.strftime('%H:%M:%S')}] personal deviations at k=111,400",
          flush=True)
    d_k, hit, n_cell0, src_k, dk_diag = dk.build_multi_k(
        df, ok, succ_season, f_excl, [K_A, K_B], verbose=False)
    assert np.array_equal(hit, hit_ref)
    dd111 = d_k[K_A] - dev_ref["d_s"]
    equiv = {
        "form_used": "(s_cell - n_cell*overall)/(n_cell + k)  [expanded, exp_023]",
        "max_abs_diff_k111_vs_inherited_literal_form": float(np.max(np.abs(dd111))),
        "tolerance_used": 1e-12,
        "miss_rows_exactly_zero_all_k": {
            str(k): bool(np.all(d_k[k][~hit] == 0.0)) for k in (K_A, K_B)},
    }
    assert equiv["max_abs_diff_k111_vs_inherited_literal_form"] < 1e-12, equiv
    assert all(equiv["miss_rows_exactly_zero_all_k"].values()), equiv

    L_eff = np.where(has_L, L, 0.0)

    # ---- THE ABLATION: mask the table's OUTPUT, do not touch the table code -- #
    is_ph2 = (ph_all == ABLATE_HAND)
    L_ph1only = np.where(is_ph2, 0.0, L_eff)
    L_keephard = np.where(is_ph2 & hit, 0.0, L_eff)

    # the mask must be equivalent to zeroing the two (2, bh) CELLS of the table,
    # not merely to zeroing some rows.  Verify: every row zeroed has ph == 2,
    # every surviving non-zero row has ph != 2, and no (2, bh) cell value
    # survives anywhere.
    changed = (L_ph1only != L_eff)
    ablation_check = {
        "rule": "L := 0 wherever pitcher_hand == 2 (both batter-hand cells)",
        "rows_with_ph2": int(is_ph2.sum()),
        "rows_zeroed": int(changed.sum()),
        "all_zeroed_rows_are_ph2": bool(np.all(is_ph2[changed])),
        "no_ph2_row_keeps_a_nonzero_L": bool(np.all(L_ph1only[is_ph2] == 0.0)),
        "ph1_rows_bit_identical": bool(
            np.array_equal(L_ph1only[~is_ph2], L_eff[~is_ph2])),
        "rows_ph2_with_L_but_zero_valued": int(
            np.sum(is_ph2 & has_L & (L_eff == 0.0))),
        "backoff_py_unmodified_sha256": src_sha["backoff.py"],
    }
    assert ablation_check["all_zeroed_rows_are_ph2"], ablation_check
    assert ablation_check["no_ph2_row_keeps_a_nonzero_L"], ablation_check
    assert ablation_check["ph1_rows_bit_identical"], ablation_check
    print("ABLATION CHECK: " + json.dumps(ablation_check), flush=True)

    # ---- the <= 2022 vintage league table, read off the season-2023 rows ----- #
    def cell_map_from_rows(mask):
        rows = np.flatnonzero(mask & has_L)
        key = pd.MultiIndex.from_arrays([ph_all[rows], bh_all[rows]])
        ser = pd.Series(L_eff[rows], index=key)
        g = ser.groupby(level=[0, 1]).agg(["nunique", "first", "size"])
        assert bool((g["nunique"] == 1).all()), "a cell carries two values"
        return ({(int(a), int(b)): float(v) for (a, b), v in g["first"].items()},
                {f"{int(a)}x{int(b)}": int(v) for (a, b), v in g["size"].items()})

    vint2022, vint2022_n = cell_map_from_rows(season == PROBE_VINTAGE_SEASON)
    vint2023, vint2023_n = cell_map_from_rows(season == VALID_SEASON)
    # cross-check against backoff.py's own diagnostic (rounded to 6 dp there)
    for tag, m in ((str(PROBE_VINTAGE_SEASON), vint2022), (str(VALID_SEASON),
                                                           vint2023)):
        ref_cells = L_diag["per_season"][tag]["L_by_cell"]
        for (a, b), v in m.items():
            k = f"{int(a)}x{int(b)}"
            assert k in ref_cells, (tag, k)
            assert abs(round(v, 6) - ref_cells[k]) < 5e-7, (tag, k, v, ref_cells[k])

    L_probe = L_eff.copy()
    m24 = (season == VALID_SEASON)
    rows24 = np.flatnonzero(m24 & has_L)
    got = np.array([vint2022.get((int(a), int(b)), np.nan)
                    for a, b in zip(ph_all[rows24], bh_all[rows24])])
    assert np.isfinite(got).all(), "a fold_2024 cell is absent from the <=2022 table"
    L_probe[rows24] = got

    # ---- multi-k arms -------------------------------------------------------- #
    d_A = dk.build_arm(d_k[K_A], hit, L_eff, n_cell0, K_A, "self")
    d_Bfull = dk.build_arm(d_k[K_B], hit, L_eff, n_cell0, K_B, "league")
    d_Bph1 = dk.build_arm(d_k[K_B], hit, L_ph1only, n_cell0, K_B, "league")
    d_Bkeep = dk.build_arm(d_k[K_B], hit, L_keephard, n_cell0, K_B, "league")
    d_probe = dk.build_arm(d_k[K_B], hit, L_probe, n_cell0, K_B, "league")
    # the probe must differ from B_full on fold_2024 rows only
    assert np.array_equal(d_probe[~m24], d_Bfull[~m24]), "probe leaked out of 2024"
    # the ablation must not move a single pitcher-hand-1 row
    assert np.array_equal(d_Bph1[~is_ph2], d_Bfull[~is_ph2]), "ablation hit ph1"
    # hit rows of arm A are the plain personal deviation (hard-backoff integrity)
    assert np.array_equal(d_A[hit], d_k[K_A][hit])

    d_grid = {"A_deployed": d_A, "B_full": d_Bfull, "B_ph1only": d_Bph1,
              "B_ph1only_keephard_SECONDARY": d_Bkeep,
              "B_full_vintage2022_PROBE": d_probe}

    # ---- as-of, independently re-verified ----------------------------------- #
    print(f"[{time.strftime('%H:%M:%S')}] independent as-of re-verification",
          flush=True)
    indep = pd.read_csv(DATA_PATH, encoding="utf-8-sig", usecols=["row_id", "season"])
    assert indep["row_id"].is_unique
    s_map = pd.Series(indep["season"].to_numpy(), index=indep["row_id"].to_numpy())
    row_id = df["row_id"].to_numpy()
    indep_season = s_map.reindex(pd.Index(row_id)).to_numpy()
    assert not np.isnan(indep_season).any()
    frame_matches_file = bool(np.array_equal(indep_season.astype(np.int64), season))
    assert frame_matches_file

    base_ok = ok & ~f_excl
    hand_ok = (~pd.isna(df[bk.PIT_HAND]).to_numpy()
               & ~pd.isna(df[bk.BAT_HAND]).to_numpy())
    asof_indep = {"method": "row_id -> season re-read from train.csv in a second "
                            "pass; max season of the CONTRIBUTING ROW IDS looked "
                            "up in that independent map",
                  "frame_season_matches_file": frame_matches_file,
                  "per_season": {}}
    for s in sorted(set(season.tolist()))[1:]:
        m_pers = base_ok & (season <= s - 1) & (succ_season <= s - 1)
        m_leag = m_pers & hand_ok
        rec_s = {}
        for name, m in (("personal", m_pers), ("league", m_leag)):
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
    y24 = y[VALID_SEASON]
    hit24 = hit[m24]
    isr24 = is_r_all[m24]
    n_cell24 = n_cell0[m24]
    ph24 = ph_all[m24]
    assert int(isr24.sum()) == EXPECTED["valid_R"]
    assert int((ph24 == 1).sum()) == EXPECTED["ph1"], int((ph24 == 1).sum())
    assert int((ph24 == 2).sum()) == EXPECTED["ph2"], int((ph24 == 2).sum())

    d_by_season = {a: {V: d_grid[a][season == V]
                       for V in SUPPLY_BLOCKS + [VALID_SEASON]} for a in d_grid}
    d_ctrl = {V: dev_ref["d_s"][season == V] for V in SUPPLY_BLOCKS + [VALID_SEASON]}

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

    ARMS_MAIN = ["A_deployed", "B_full", "B_ph1only"]
    ARMS_EXTRA = ["B_ph1only_keephard_SECONDARY", "B_full_vintage2022_PROBE"]

    per_seed, provenance = {}, {}
    for s in SEEDS:
        print(f"[{time.strftime('%H:%M:%S')}] ===== seed {s} =====", flush=True)
        p24, blk, prov = load_bases(s)
        provenance[str(s)] = prov
        br_base24 = float(np.mean((p24 - y24) ** 2))
        y_supply = np.concatenate([y[V] for V in SUPPLY_BLOCKS])
        p_supply = np.concatenate([blk[V] for V in SUPPLY_BLOCKS])
        r_supply = y_supply - p_supply
        br_base_blocks = float(np.mean((p_supply - y_supply) ** 2))

        # --- betas, every one fitted on THIS SEED's supply blocks ------------- #
        beta_ctrl = off.fit_slope(
            r_supply, np.concatenate([d_ctrl[V] for V in SUPPLY_BLOCKS]))["beta"]

        blockfit = {}
        for a in d_grid:
            dd = np.concatenate([d_by_season[a][V] for V in SUPPLY_BLOCKS])
            fit = off.fit_slope(r_supply, dd)
            b = fit["beta"]
            pb = off.apply_linear(p_supply, dd, b)
            blockfit[a] = {
                "beta_fitted_on_blocks": b, "beta_se": fit["se"], "beta_t": fit["t"],
                "brier_blocks": float(np.mean((pb - y_supply) ** 2)),
                "delta_vs_base_blocks": float(np.mean((pb - y_supply) ** 2)
                                              - br_base_blocks),
                "beta_per_block": {str(V): off.fit_slope(
                    y[V] - blk[V], d_by_season[a][V])["beta"] for V in SUPPLY_BLOCKS},
                "sd_d_blocks": float(np.std(dd)),
            }

        seb = (p24 - y24) ** 2

        def evaluate(a, beta):
            dv = d_by_season[a][VALID_SEASON]
            p = off.apply_linear(p24, dv, beta)
            raw = p24 + beta * dv
            se = (p - y24) ** 2
            br = float(np.mean(se))
            out = {
                "arm": a, "beta_applied": beta, "brier": br, "score": score(br),
                "delta_vs_base_no_offset": float(br - br_base24),
                "score_gain_vs_base": score_delta(br - br_base24),
                "valid_R": seg(y24[isr24], p[isr24]),
                "valid_F": seg(y24[~isr24], p[~isr24]),
                "seg_miss": seg(y24[~hit24], p[~hit24]),
                "seg_hit": seg(y24[hit24], p[hit24]),
                "correction_shift_beta_d": {
                    "mean_abs": float(np.mean(np.abs(beta * dv))),
                    "max_abs": float(np.max(np.abs(beta * dv))),
                    "sd_d": float(np.std(dv)),
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
                m = (ph24 == h)
                out["by_pitcher_hand"][str(h)] = {
                    "n": int(m.sum()), "brier": float(np.mean(se[m])),
                    "delta_vs_base": float(np.mean(se[m] - seb[m])),
                }
                for lab, mm in (("hit", m & hit24), ("miss", m & ~hit24)):
                    out["by_pitcher_hand_x_cellhit"][f"{h}_{lab}"] = {
                        "n": int(mm.sum()), "brier": float(np.mean(se[mm])),
                        "delta_vs_base": float(np.mean(se[mm] - seb[mm])),
                    }
            return out, p

        res, preds = {}, {}
        for a in ARMS_MAIN + ARMS_EXTRA:
            beta = (beta_ctrl if a == "A_deployed"
                    else blockfit["B_full"]["beta_fitted_on_blocks"]
                    if a == "B_full_vintage2022_PROBE"
                    else blockfit[a]["beta_fitted_on_blocks"])
            res[a], preds[a] = evaluate(a, beta)
        for a in ARMS_MAIN:
            np.save(os.path.join(OUT_DIR, "pred",
                                 f"pred_{a}_seed{s}_fold_2024.npy"), preds[a])

        contrib_ph2 = res["B_full"]["brier"] - res["B_ph1only"]["brier"]
        left_after = res["B_ph1only"]["brier"] - res["A_deployed"]["brier"]
        total = res["B_full"]["brier"] - res["A_deployed"]["brier"]
        probe_loss = (res["B_full_vintage2022_PROBE"]["brier"]
                      - res["B_full"]["brier"])
        keep_contrib = (res["B_full"]["brier"]
                        - res["B_ph1only_keephard_SECONDARY"]["brier"])

        per_seed[str(s)] = {
            "base_no_offset": {"fold_2024_brier": br_base24,
                               "fold_2024_score": score(br_base24),
                               "blocks_brier": br_base_blocks},
            "beta_control_plain_ds": beta_ctrl,
            "arms": res,
            "block_fit": blockfit,
            "decomposition": {
                "B_full_minus_B_ph1only_PH2_CONTRIBUTION": contrib_ph2,
                "B_full_minus_B_ph1only_score": score_delta(contrib_ph2),
                "B_ph1only_minus_A_deployed_GAIN_WITHOUT_PH2": left_after,
                "B_ph1only_minus_A_deployed_score": score_delta(left_after),
                "B_full_minus_A_deployed_TOTAL": total,
                "B_full_minus_A_deployed_score": score_delta(total),
                "additivity_residual": float(total - (contrib_ph2 + left_after)),
                "B_full_minus_keephard_SECONDARY": keep_contrib,
                "B_full_minus_keephard_score": score_delta(keep_contrib),
            },
            "vintage_probe": {
                "brier_with_2022_vintage_table": res[
                    "B_full_vintage2022_PROBE"]["brier"],
                "brier_B_full_2023_vintage": res["B_full"]["brier"],
                "loss_from_one_stale_season": probe_loss,
                "loss_score": score_delta(probe_loss),
                "beta_source": "B_full's beta, unchanged — only the fold_2024 "
                               "table is swapped",
                "still_better_than_A_deployed": bool(
                    res["B_full_vintage2022_PROBE"]["brier"]
                    < res["A_deployed"]["brier"]),
                "vs_A_deployed": float(res["B_full_vintage2022_PROBE"]["brier"]
                                       - res["A_deployed"]["brier"]),
                "vs_A_deployed_score": score_delta(
                    res["B_full_vintage2022_PROBE"]["brier"]
                    - res["A_deployed"]["brier"]),
            },
            "arm_pred_gap": {
                "B_full_vs_B_ph1only": {
                    "mean_abs": float(np.mean(np.abs(preds["B_full"]
                                                     - preds["B_ph1only"]))),
                    "max_abs": float(np.max(np.abs(preds["B_full"]
                                                   - preds["B_ph1only"])))},
                "B_ph1only_vs_A_deployed": {
                    "mean_abs": float(np.mean(np.abs(preds["B_ph1only"]
                                                     - preds["A_deployed"]))),
                    "max_abs": float(np.max(np.abs(preds["B_ph1only"]
                                                   - preds["A_deployed"])))},
            },
        }
        print(f"  seed {s}: A {res['A_deployed']['brier']:.12f}  "
              f"Bfull {res['B_full']['brier']:.12f}  "
              f"Bph1 {res['B_ph1only']['brier']:.12f}  | "
              f"ph2 contrib {contrib_ph2:+.4e} ({score_delta(contrib_ph2):+.5f})  "
              f"left {left_after:+.4e} ({score_delta(left_after):+.5f})  | "
              f"probe {probe_loss:+.4e} ({score_delta(probe_loss):+.5f})",
              flush=True)
        del p24, blk, preds

    # ---- GATE: both reproducible arms, all three seeds ---------------------- #
    gate = {"tolerance": GATE["tolerance"], "source": GATE["source"],
            "on_fail": GATE["on_fail"], "checks": {}}
    allpass = True
    for a in ("A_deployed", "B_full"):
        for s in SEEDS:
            got_b = per_seed[str(s)]["arms"][a]["brier"]
            want = GATE[a][s]
            diff = abs(got_b - want)
            ok_ = bool(diff < GATE["tolerance"])
            allpass = allpass and ok_
            gate["checks"][f"{a}_seed{s}"] = {
                "expected_exp024": want, "got": got_b, "abs_diff": diff,
                "bit_identical": bool(got_b == want), "passed": ok_}
    gate["passed"] = bool(allpass)
    print("REPRODUCTION GATE: " + json.dumps(gate), flush=True)
    assert gate["passed"], "REPRODUCTION GATE FAILED: " + json.dumps(gate)

    # ---- across-seed summary of the two differences ------------------------- #
    def summarize(key):
        v = [per_seed[str(s)]["decomposition"][key] for s in SEEDS]
        mean = float(np.mean(v))
        return {"by_seed": {str(s): x for s, x in zip(SEEDS, v)},
                "by_seed_score": {str(s): score_delta(x)
                                  for s, x in zip(SEEDS, v)},
                "signs_agree": bool(all(x < 0 for x in v) or all(x > 0 for x in v)),
                "mean": mean, "mean_score": score_delta(mean),
                "seed_std_ddof1": float(np.std(v, ddof=1))}

    arm_brier = {a: [per_seed[str(s)]["arms"][a]["brier"] for s in SEEDS]
                 for a in ARMS_MAIN + ARMS_EXTRA}
    summary = {
        "arm_seed_table": {a: {str(s): b for s, b in zip(SEEDS, v)}
                           for a, v in arm_brier.items()},
        "arm_seed_table_score": {a: {str(s): score(b) for s, b in zip(SEEDS, v)}
                                 for a, v in arm_brier.items()},
        "arm_seed_std_ddof1": {a: float(np.std(v, ddof=1))
                               for a, v in arm_brier.items()},
        "PH2_CONTRIBUTION_B_full_minus_B_ph1only": summarize(
            "B_full_minus_B_ph1only_PH2_CONTRIBUTION"),
        "GAIN_WITHOUT_PH2_B_ph1only_minus_A_deployed": summarize(
            "B_ph1only_minus_A_deployed_GAIN_WITHOUT_PH2"),
        "TOTAL_B_full_minus_A_deployed": summarize(
            "B_full_minus_A_deployed_TOTAL"),
        "SECONDARY_B_full_minus_keephard": summarize(
            "B_full_minus_keephard_SECONDARY"),
        "control_arm_seed_std_ddof1_for_reference": float(
            np.std(arm_brier["A_deployed"], ddof=1)),
    }
    tot = summary["TOTAL_B_full_minus_A_deployed"]["mean"]
    summary["share_of_total_gain_from_ph2_cells"] = (
        float(summary["PH2_CONTRIBUTION_B_full_minus_B_ph1only"]["mean"] / tot)
        if tot != 0 else None)
    summary["share_of_total_gain_surviving_ablation"] = (
        float(summary["GAIN_WITHOUT_PH2_B_ph1only_minus_A_deployed"]["mean"] / tot)
        if tot != 0 else None)

    probe = {
        "by_seed": {str(s): per_seed[str(s)]["vintage_probe"] for s in SEEDS},
        "mean_loss": float(np.mean([per_seed[str(s)]["vintage_probe"]
                                    ["loss_from_one_stale_season"] for s in SEEDS])),
        "cells_2023_vintage_applied_to_2024_NORMAL": {
            f"{int(a)}x{int(b)}": v for (a, b), v in vint2023.items()},
        "cells_2022_vintage_swapped_in_PROBE": {
            f"{int(a)}x{int(b)}": v for (a, b), v in vint2022.items()},
        "cell_row_counts_2024": vint2023_n,
        "discipline": "DIAGNOSTIC.  <=2022 is strictly more past than <=2023 so "
                      "this cannot leak; no arm is chosen or changed by it.",
    }
    probe["mean_loss_score"] = score_delta(probe["mean_loss"])

    print("SUMMARY: " + json.dumps(
        {"ph2_contribution_mean_score": summary[
            "PH2_CONTRIBUTION_B_full_minus_B_ph1only"]["mean_score"],
         "gain_without_ph2_mean_score": summary[
             "GAIN_WITHOUT_PH2_B_ph1only_minus_A_deployed"]["mean_score"],
         "total_mean_score": summary["TOTAL_B_full_minus_A_deployed"]["mean_score"],
         "vintage_probe_mean_loss_score": probe["mean_loss_score"]}), flush=True)

    metrics = {
        "exp_id": "exp_025",
        "title": "how much of k=400 x league's gain sits on the unstable "
                 "pitcher-hand-2 league cells; plus a table-vintage probe",
        "no_training": True,
        "arms": {
            "A_deployed": "k=111 x self at the CONTROL beta (exp_022 G1f) — the "
                          "deployed d",
            "B_full": "k=400 x league, all four cells = exp_024 arm B",
            "B_ph1only": "k=400 x league with L forced to 0 on every "
                         "pitcher_hand==2 row (both (2,bh) cells)",
            "B_ph1only_keephard_SECONDARY": "as B_ph1only but pitcher-hand-2 "
                                            "CELL-MISS rows keep L, so only the "
                                            "part new to `league` is removed",
            "B_full_vintage2022_PROBE": "B_full with the <=2022 league table "
                                        "swapped into fold_2024 only",
        },
        "seeds": SEEDS,
        "changed_vs_exp_024": [
            "no training at all — every base vector is reused with its sha256 "
            "asserted against the value exp_024 recorded",
            "two new arms: B_ph1only (the ablation) and its SECONDARY variant",
            "one new diagnostic: the <=2022 league-table vintage probe",
            "backoff.py, dev_k.py, offset.py, features_platoon.py, labels.py and "
            "exp016_ref.py are byte-identical to exp_024's (sha asserted)",
            "arm A is renamed A_deployed and arm B B_full; the quantities are the "
            "same and reproduce exp_024 exactly (see reproduction_gate)",
            "the per-seed block reselection diagnostic of exp_024 is NOT re-run",
        ],
        "unchanged_vs_exp_024": [
            "base vectors, d construction, league table, exp_004 F-mask, clip "
            "1e-6, closed-form no-intercept slope, supply blocks 2022+2023",
        ],
        "inherited_source_sha256": SHA_INHERITED,
        "source_sha256": src_sha,
        "train_csv_sha256": csv_sha,
        "base_vector_provenance": provenance,
        "base_vector_expected_sha256_from_exp024_record": {
            str(k): {str(kk): vv for kk, vv in v.items()}
            for k, v in BASE_SHA.items()},
        "exp024_metrics_json_sha256": exp024_metrics_sha,
        "gate_value_crosscheck_config_vs_exp024_record": gate_xcheck,
        "reproduction_gate": gate,
        "ablation_implementation_check": ablation_check,
        "k111_equivalence": equiv,
        "asof_independent_recheck": asof_indep,
        "summary": summary,
        "vintage_probe": probe,
        "per_seed": per_seed,
        "n_bin_definition": {
            "hit_tertile_boundaries_n_cell": [float(q1), float(q2)],
            "bin_sizes": {b: int(m.sum()) for b, m in bins24.items()},
        },
        "L_by_cell_by_season": {s: v["L_by_cell"]
                                for s, v in L_diag["per_season"].items()},
        "conversion": {
            "formula": f"score = 100000 * (1 - brier / {BSS_REF_BRIER})",
            "sensitivity": "brier 1e-6 = score 0.4003; a 6-decimal Brier carries "
                           "+-0.2002 in the converted number",
            "caveat": "local measurement is at iter 650 (<=2023 training); the "
                      "deployed build is iter 7700 (<=2024).",
        },
        "leak_boundary": {
            "tables": "row of season s uses seasons <= s-1 only; verified at "
                      "runtime inside backoff.py/dev_k.py and again here by an "
                      "independent re-read of train.csv",
            "beta_fit": "supply blocks 2022+2023 of the same seed, on that arm's "
                        "own d; the probe reuses B_full's beta unchanged",
            "arm_choice": "the three arms are fixed by config.yaml before any "
                          "2024 number is read; the ablation is not chosen by "
                          "looking at 2024",
            "vintage_probe": "<=2022 is strictly MORE PAST than the <=2023 table "
                             "fold_2024 normally uses, so it cannot leak; it is a "
                             "diagnostic and changes no arm",
            "design_matrix": "unchanged — no deviation is ever a model feature",
            "missing": "no sentinel fill anywhere; L=0 means 'apply no "
                       "correction', which is the behaviour without the module",
        },
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__},
        "compute_time_min": round((time.time() - t_start) / 60, 2),
    }
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(json.loads(json.dumps(metrics, default=str)), f, indent=2,
                  ensure_ascii=False)

    print("\n--- ARM x SEED (fold_2024 valid_all) ---", flush=True)
    print(f"{'seed':>6}" + "".join(f"{a:>34}" for a in ARMS_MAIN), flush=True)
    for s in SEEDS:
        print(f"{s:>6}" + "".join(
            f"{per_seed[str(s)]['arms'][a]['brier']:>20.12f}"
            f"{per_seed[str(s)]['arms'][a]['score']:>14.5f}"
            for a in ARMS_MAIN), flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] DONE {metrics['compute_time_min']} min",
          flush=True)


if __name__ == "__main__":
    main()
