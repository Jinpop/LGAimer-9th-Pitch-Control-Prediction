"""exp_023 — the offset's `d`: shrink constant k x shrink target, a 7 x 2 grid.

NOTHING IS TRAINED HERE.  Three stage-1 prediction vectors are reused, sha
asserted before use:

  fold_2024   exp_020/pred/pred_E0_B3_P4iter_seed42_fold_2024.npy
  block 2022  exp_022/pred/pred_B3_block_2022_seed42.npy
  block 2023  exp_022/pred/pred_B3_block_2023_seed42.npy

Every arm is `p' = clip(p + beta * d)` on the SAME base vector, so the base
model's seed noise cancels in every difference reported below.

THE SELECTION RULE — THIS IS THE POINT OF THE EXPERIMENT
--------------------------------------------------------
Reading 14 arms off fold_2024 and keeping the best one IS fitting to the
validation season.  So:

  1. every arm's `d` is built and its beta fitted on the SUPPLY BLOCKS
     (2022 + 2023) only;
  2. the arm is CHOSEN by its Brier on those same blocks;
  3. fold_2024 is read once, for the chosen arm — that is the headline;
  4. the other 13 fold_2024 numbers are computed and reported as DIAGNOSTIC.
     They have exactly the standing of an oracle beta: they are what you would
     have got had you cheated, and they are not evidence for adoption.

If the block argmin and the fold_2024 argmin disagree, the gap between them is
itself a measurement — the price of choosing honestly — and it is reported.

BETA IS NOT A GRID AXIS.  It is a dependent output: for each `d`, beta is the
closed-form no-intercept slope E[d r]/E[d^2] on the supply blocks.  Changing k
changes the scale of d, so beta moves mechanically in the opposite direction;
only the net effect is readable.

REPRODUCTION GATE
-----------------
k = 111 x self, with beta held at the CONTROL's value (the beta fitted on the
plain d_s, no league term) is exp_022's G1f = the `d` of deployed candidates
H/I.  Its fold_2024 Brier must come back as 0.248037745 within 5e-6.  Holding
beta at the control value is what makes it G1f rather than G1 — G1 re-fits beta
and lands 1.2e-6 away, which would pass the tolerance for the wrong reason.
Both are reported.
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

SEED = 42
DATA_PATH = os.path.expanduser("~/SungjinPark/data/train.csv")
OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_023")
EXP020_PRED = os.path.expanduser(
    "~/SungjinPark/experiments/exp_020/pred/pred_E0_B3_P4iter_seed42_fold_2024.npy")
EXP022_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_022")

TARGET = "control_success"
DROP_COLS = ["row_id", TARGET]
F_EXCLUDE_MAX_SEASON = 2022
SUPPLY_BLOCKS = [2022, 2023]          # efficiency rule #6: latest two
VALID_SEASON = 2024
EPS = 1e-6
BSS_REF_BRIER = 0.2498069

K_GRID = [0, 30, 55, 111, 200, 400, 800]
TARGETS = ["self", "league"]
K_DEPLOYED = 111

EXPECTED = {"valid_all": 253507, "valid_R": 223497}
BASE_SHA = "1e4fba70fe2322d02c246209e4d2dce967f890044aa182e8de4b47525cfafa97"
TRAIN_CSV_SHA = "d2081186b458b49f60b082be480c273135833e15ba59a76d033af28bcf8763ff"

# byte-preserved from exp_022/src — asserted, not assumed
SHA_INHERITED = {
    "offset.py": "68d6007db03ab850e2a5181383a4148fae8947c711558282c9846391f13e0488",
    "backoff.py": "4169d2dadd8a07dceef2098a88971d3b3e437276bfb28834295e7df523591039",
    "labels.py": "439cd504478b4bc8162e9dedc32be27bfadd5d1178472d6b384b60f46ed7a9f2",
    "features_platoon.py": "68d63063e8ce659cf2876fbca4f5adef7ee1e720620ae63bb78ea"
                           "496c8dadf09",
    "exp016_ref.py": "4f24deb71d9eef3d626900f34a71aecd2d6acbae646eb9c92c51189a38"
                     "a3ece1",
}

GATE = {
    "arm": "k=111 x self at the CONTROL beta = exp_022 G1f (BK1f) = the d of "
           "deployed candidates H/I",
    "fold_2024_valid_all_brier": 0.248037745,
    "control_beta": 0.541136,
    "tolerance": 5e-6,
    "on_fail": "abort — the inherited code or the reused predictions are not "
               "exp_022's and nothing downstream is readable",
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
    trailing digits are a machine transform of a rounded input, not measurement."""
    return round(100000.0 * (1.0 - brier / BSS_REF_BRIER), 5)


def seg(y, p):
    m = ref.seg_metrics(y, p)
    if len(y):
        b = float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))
        m["brier_full"] = b
        m["score_vs_fixed_ref"] = score(b)
    return m


def arm_id(k, t):
    return f"k{k}_{t}"


# --------------------------------------------------------------------------- #
def main():
    t_start = time.time()
    np.random.seed(SEED)
    os.makedirs(os.path.join(OUT_DIR, "pred"), exist_ok=True)

    csv_sha = sha256_file(DATA_PATH)
    assert csv_sha == TRAIN_CSV_SHA, ("train.csv SHA MISMATCH", csv_sha)
    src_sha = {f: sha256_file(os.path.join(SRC_DIR, f))
               for f in sorted(os.listdir(SRC_DIR)) if f.endswith(".py")}
    for f, want in SHA_INHERITED.items():
        assert src_sha[f] == want, ("INHERITED SOURCE DRIFTED", f, src_sha[f], want)

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

    # ---- inherited machinery, untouched ------------------------------------- #
    print(f"[{time.strftime('%H:%M:%S')}] personal lookups + k=111 reference",
          flush=True)
    lu_diag, src_fp = fp.build_season_forward_chained(df, ok, succ_season, f_excl,
                                                      verbose=False)
    dev_ref, hit_ref, src_dev, dev_diag = off.build_deviations(
        df, ok, succ_season, f_excl, verbose=False)
    print(f"[{time.strftime('%H:%M:%S')}] league backoff table (byte-preserved)",
          flush=True)
    L, has_L, src_L, L_diag = bk.build_league_dev(df, ok, succ_season, f_excl,
                                                  verbose=False)
    eb_k_default = fp.RATE_SPEC[bk.SUCC_COL][1]
    assert eb_k_default == K_DEPLOYED
    d_arm022, integrity022 = bk.build_arm_deviations(
        df, dev_ref["d_s"], hit_ref, L, has_L, eb_k_default)

    # ---- the k sweep --------------------------------------------------------- #
    print(f"[{time.strftime('%H:%M:%S')}] multi-k personal deviations", flush=True)
    d_k, hit, n_cell0, src_k, dk_diag = dk.build_multi_k(
        df, ok, succ_season, f_excl, K_GRID, verbose=False)
    assert np.array_equal(hit, hit_ref)
    assert np.array_equal(np.isnan(src_fp), np.isnan(src_dev))
    # dev_k uses the algebraically expanded form (s - n*o)/(n + k); the inherited
    # code uses the literal subtraction (s + k*o)/(n + k) - o.  Identical in exact
    # arithmetic, ulp-apart in float — and only the expanded form returns exactly
    # 0 on a miss for EVERY k, which is the hard backoff's integrity premise.  The
    # difference is measured here rather than asserted away.
    dd111 = d_k[K_DEPLOYED] - dev_ref["d_s"]
    miss_all = ~hit
    equiv = {
        "form_used": "(s_cell - n_cell*overall)/(n_cell + k)  [expanded]",
        "form_inherited": "(s_cell + k*overall)/(n_cell + k) - overall  [literal]",
        "why": "the literal form's miss row is (k*o)/k - o, which IEEE-754 does "
               "not guarantee to be 0.  It held at k=111 by luck and broke at "
               "k=200 (first launch died on season 2021).  Same bug class exp_022 "
               "fixed in the league term by regrouping to L*(k/(n+k)).",
        "bit_identical_to_offset_build_deviations_at_k111": bool(
            np.array_equal(d_k[K_DEPLOYED], dev_ref["d_s"])),
        "max_abs_diff_k111": float(np.max(np.abs(dd111))),
        "max_abs_diff_k111_on_hit_rows": float(np.max(np.abs(dd111[hit]))),
        "max_abs_diff_k111_on_miss_rows": float(np.max(np.abs(dd111[miss_all]))),
        "n_rows_differing_k111": int(np.sum(dd111 != 0.0)),
        "max_rel_diff_k111": float(np.max(
            np.abs(dd111[hit]) / np.maximum(np.abs(dev_ref["d_s"][hit]), 1e-300))),
        "tolerance_used": 1e-12,
        "miss_rows_exactly_zero_all_k": {
            str(k): bool(np.all(d_k[k][miss_all] == 0.0)) for k in K_GRID},
    }
    assert equiv["max_abs_diff_k111"] < equiv["tolerance_used"], equiv
    assert all(equiv["miss_rows_exactly_zero_all_k"].values()), equiv
    print("K111 EQUIVALENCE: " + json.dumps(equiv), flush=True)

    L_eff = np.where(has_L, L, 0.0)
    d_grid = {arm_id(k, t): dk.build_arm(d_k[k], hit, L_eff, n_cell0, k, t)
              for k in K_GRID for t in TARGETS}

    # the two anchors must BE exp_022's G1 and G2 up to the ulp difference above
    anchor_check = {
        "k111_self_vs_exp022_G1_max_abs": float(np.max(np.abs(
            d_grid[arm_id(K_DEPLOYED, "self")] - d_arm022["G1"]))),
        "k111_league_vs_exp022_G2_max_abs": float(np.max(np.abs(
            d_grid[arm_id(K_DEPLOYED, "league")] - d_arm022["G2"]))),
        "k111_self_miss_rows_bit_identical_to_G1": bool(np.array_equal(
            d_grid[arm_id(K_DEPLOYED, "self")][miss_all], d_arm022["G1"][miss_all])),
        "tolerance_used": 1e-12,
        "k0_self_equals_k0_league": bool(np.array_equal(
            d_grid[arm_id(0, "self")], d_grid[arm_id(0, "league")])),
        "k0_degeneracy_note": "at k=0 there is no shrinkage, so the league term "
                              "gets weight 0 on hit rows and weight 1 (the limit "
                              "from k->0+) on miss rows — which is the hard "
                              "backoff.  The two targets coincide.  13 distinct "
                              "arms, not 14.",
    }
    assert anchor_check["k111_self_vs_exp022_G1_max_abs"] < 1e-12, anchor_check
    assert anchor_check["k111_league_vs_exp022_G2_max_abs"] < 1e-12, anchor_check
    assert anchor_check["k111_self_miss_rows_bit_identical_to_G1"], anchor_check
    print("ANCHOR CHECK: " + json.dumps(anchor_check), flush=True)

    # d must still be exactly 0 on miss rows for the `self` target at every k
    for k in K_GRID:
        dd = d_grid[arm_id(k, "self")]
        assert np.array_equal(dd[hit], d_k[k][hit]), ("self arm moved a hit row", k)

    # ---- AS-OF, verified through an INDEPENDENT re-read of the file ---------- #
    # the exp_022 audit's objection: the runtime assert reads `season` out of the
    # very mask that was built from `season`, so it cannot prove anything.  Here
    # the season of every row that fed a table is looked up again from a FRESH
    # read of train.csv, keyed by row_id, and compared against the frame in use.
    print(f"[{time.strftime('%H:%M:%S')}] independent as-of re-verification",
          flush=True)
    indep = pd.read_csv(DATA_PATH, encoding="utf-8-sig", usecols=["row_id", "season"])
    assert indep["row_id"].is_unique
    s_map = pd.Series(indep["season"].to_numpy(), index=indep["row_id"].to_numpy())
    row_id = df["row_id"].to_numpy()
    indep_season = s_map.reindex(pd.Index(row_id)).to_numpy()
    assert not np.isnan(indep_season).any(), "row_id present in df but not in file"
    frame_matches_file = bool(np.array_equal(indep_season.astype(np.int64), season))
    assert frame_matches_file, "the in-memory season column no longer matches the file"

    base_ok = ok & ~f_excl
    hand_ok = (~pd.isna(df[bk.PIT_HAND]).to_numpy()
               & ~pd.isna(df[bk.BAT_HAND]).to_numpy())
    asof_indep = {"method": "row_id -> season re-read from train.csv in a second "
                            "pass; the table masks are re-derived and the max "
                            "season of the CONTRIBUTING ROW IDS is looked up in "
                            "that independent map",
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
    # and the row-level source-season arrays, for the seasons this run reads
    asof_runtime = {}
    for V in SUPPLY_BLOCKS + [VALID_SEASON]:
        for name, arr in (("personal", src_fp), ("league", src_L), ("multi_k", src_k)):
            u = arr[season == V]
            mx = float(np.nanmax(u)) if np.any(~np.isnan(u)) else float("-inf")
            assert mx < V, f"AS-OF VIOLATION {name} season {V}: used {mx}"
            asof_runtime[f"{name}_{V}"] = mx
    assert max(SUPPLY_BLOCKS) < VALID_SEASON
    print("AS-OF (independent) OK: "
          + json.dumps({k: v["personal"]["max_season_of_contributors_independent"]
                        for k, v in asof_indep["per_season"].items()}), flush=True)

    # ---- reused predictions -------------------------------------------------- #
    got_sha = sha256_file(EXP020_PRED)
    assert got_sha == BASE_SHA, ("BASE PRED SHA MISMATCH", got_sha, BASE_SHA)
    p24 = np.load(EXP020_PRED)
    blk, blk_meta = {}, {}
    for V in SUPPLY_BLOCKS:
        pth = os.path.join(EXP022_DIR, "pred", f"pred_B3_block_{V}_seed42.npy")
        with open(os.path.join(EXP022_DIR, f"block_{V}.json")) as f:
            bm = json.load(f)
        assert bm["gate"]["passed"], bm["gate"]
        got = sha256_file(pth)
        assert bm["pred_sha256"] == got, ("block pred sha drift", V, got)
        assert bm["train_csv_sha256"] == csv_sha
        blk[V] = np.load(pth)
        blk_meta[str(V)] = {"pred_sha256": got, "gate_brier": bm["gate"]["brier"],
                            "iter": bm["gate"]["iter"], "n_valid": bm["n_valid"]}

    y = {V: y_all[season == V] for V in SUPPLY_BLOCKS + [VALID_SEASON]}
    for V in SUPPLY_BLOCKS:
        assert blk[V].shape == y[V].shape, (V, blk[V].shape, y[V].shape)
    assert p24.shape == y[VALID_SEASON].shape
    assert len(y[VALID_SEASON]) == EXPECTED["valid_all"]

    y24 = y[VALID_SEASON]
    hit24 = hit[season == VALID_SEASON]
    isr24 = is_r_all[season == VALID_SEASON]
    n_cell24 = n_cell0[season == VALID_SEASON]
    assert int(isr24.sum()) == EXPECTED["valid_R"]
    br_base24 = float(np.mean((p24 - y24) ** 2))

    d_by_season = {a: {V: d_grid[a][season == V]
                       for V in SUPPLY_BLOCKS + [VALID_SEASON]} for a in d_grid}

    # the control d (plain d_s at k=111, no league term) — only used for the gate
    d_ctrl = {V: dev_ref["d_s"][season == V] for V in SUPPLY_BLOCKS + [VALID_SEASON]}
    r_supply = np.concatenate([y[V] - blk[V] for V in SUPPLY_BLOCKS])
    beta_ctrl = off.fit_slope(
        r_supply, np.concatenate([d_ctrl[V] for V in SUPPLY_BLOCKS]))["beta"]
    assert abs(beta_ctrl - GATE["control_beta"]) < 5e-6, beta_ctrl

    y_supply = np.concatenate([y[V] for V in SUPPLY_BLOCKS])
    p_supply = np.concatenate([blk[V] for V in SUPPLY_BLOCKS])
    br_base_blocks = float(np.mean((p_supply - y_supply) ** 2))

    # ======================================================================== #
    #  STEP 1-2 — fit beta and SELECT on the supply blocks.  fold_2024 unread.  #
    # ======================================================================== #
    print(f"[{time.strftime('%H:%M:%S')}] block fit + selection (2022+2023)",
          flush=True)
    block = {}
    for a in d_grid:
        dd = np.concatenate([d_by_season[a][V] for V in SUPPLY_BLOCKS])
        fit = off.fit_slope(r_supply, dd)
        b = fit["beta"]
        pb = off.apply_linear(p_supply, dd, b)
        br = float(np.mean((pb - y_supply) ** 2))
        # honest-within-blocks cross-check: fit on one block, score the other
        cross = {}
        for V, W in ((SUPPLY_BLOCKS[0], SUPPLY_BLOCKS[1]),
                     (SUPPLY_BLOCKS[1], SUPPLY_BLOCKS[0])):
            bv = off.fit_slope(y[V] - blk[V], d_by_season[a][V])["beta"]
            pw = off.apply_linear(blk[W], d_by_season[a][W], bv)
            cross[f"fit{V}_score{W}"] = {
                "beta": bv,
                "brier": float(np.mean((pw - y[W]) ** 2)),
                "delta_vs_base": float(np.mean((pw - y[W]) ** 2)
                                       - np.mean((blk[W] - y[W]) ** 2)),
            }
        cross_delta = float(np.mean([v["delta_vs_base"] for v in cross.values()]))
        block[a] = {
            "beta_fitted_on_blocks": b,
            "beta_se": fit["se"], "beta_t": fit["t"],
            "mean_d2": fit["mean_d2"],
            "brier_blocks": br,
            "score_blocks": score(br),
            "delta_vs_base_blocks": float(br - br_base_blocks),
            "beta_per_block": {str(V): off.fit_slope(
                y[V] - blk[V], d_by_season[a][V])["beta"] for V in SUPPLY_BLOCKS},
            "cross_block": cross,
            "cross_block_mean_delta_vs_base": cross_delta,
            "sd_d_blocks": float(np.std(dd)),
            "mean_abs_correction_shift_blocks": float(np.mean(np.abs(b * dd))),
        }

    selected = min(block, key=lambda a: block[a]["brier_blocks"])
    selected_cross = min(block, key=lambda a: block[a]["cross_block_mean_delta_vs_base"])
    print(f"BLOCK ARGMIN: {selected}  (cross-block argmin: {selected_cross})",
          flush=True)

    # ======================================================================== #
    #  STEP 3-4 — fold_2024.  The selected arm is the headline; the rest are    #
    #  diagnostic and have the standing of an oracle beta.                      #
    # ======================================================================== #
    print(f"[{time.strftime('%H:%M:%S')}] fold_2024 evaluation", flush=True)

    def nbin_split(n_cell_arr, hit_arr):
        hn = n_cell_arr[hit_arr]
        q1, q2 = np.percentile(hn, [33.3333, 66.6667])
        bins = {
            "miss_n0": ~hit_arr,
            f"hit_low_n_le_{int(q1)}": hit_arr & (n_cell_arr <= q1),
            f"hit_mid_n_{int(q1)}_{int(q2)}": (hit_arr & (n_cell_arr > q1)
                                               & (n_cell_arr <= q2)),
            f"hit_high_n_gt_{int(q2)}": hit_arr & (n_cell_arr > q2),
        }
        fixed = {
            "fx_miss_n0": ~hit_arr,
            "fx_n_1_99": hit_arr & (n_cell_arr < 100),
            "fx_n_100_499": hit_arr & (n_cell_arr >= 100) & (n_cell_arr < 500),
            "fx_n_500_1999": hit_arr & (n_cell_arr >= 500) & (n_cell_arr < 2000),
            "fx_n_2000plus": hit_arr & (n_cell_arr >= 2000),
        }
        bins.update(fixed)
        return bins, float(q1), float(q2)

    bins24, q1, q2 = nbin_split(n_cell24, hit24)
    anchor_self = arm_id(K_DEPLOYED, "self")

    pred24, fold = {}, {}
    for a in d_grid:
        dv = d_by_season[a][VALID_SEASON]
        b = block[a]["beta_fitted_on_blocks"]
        p = off.apply_linear(p24, dv, b)
        pred24[a] = p
        raw = p24 + b * dv
        se = (p - y24) ** 2
        seb = (p24 - y24) ** 2
        br = float(np.mean(se))
        dv_or = float(np.dot(dv, y24 - p24) / np.dot(dv, dv)) if np.dot(dv, dv) else 0.0
        p_or = off.apply_linear(p24, dv, dv_or)
        fold[a] = {
            "beta_applied": b,
            "brier": br,
            "score": score(br),
            "delta_vs_base_no_offset": float(br - br_base24),
            "score_gain_vs_base": round(
                -100000.0 * float(br - br_base24) / BSS_REF_BRIER, 5),
            "valid_R": seg(y24[isr24], p[isr24]),
            "valid_F": seg(y24[~isr24], p[~isr24]),
            "seg_miss": seg(y24[~hit24], p[~hit24]),
            "seg_hit": seg(y24[hit24], p[hit24]),
            "correction_shift_beta_d": {
                "mean_abs": float(np.mean(np.abs(b * dv))),
                "max_abs": float(np.max(np.abs(b * dv))),
                "sd_d": float(np.std(dv)),
                "rows_moved_pct": round(100.0 * float((dv != 0).mean()), 3),
                "n_clip": int(((raw < EPS) | (raw > 1 - EPS)).sum()),
            },
            "arm_pred_gap_vs_k111_self": None,   # filled below
            "oracle_beta_DISPLAY_ONLY": round(dv_or, 6),
            "brier_at_oracle_beta_DISPLAY_ONLY": float(np.mean((p_or - y24) ** 2)),
            "transfer_efficiency_DISPLAY_ONLY": (
                round((br - br_base24) / (float(np.mean((p_or - y24) ** 2))
                                          - br_base24), 4)
                if float(np.mean((p_or - y24) ** 2)) != br_base24 else None),
            "by_n_bin": {},
        }
        for bn, msk in bins24.items():
            if not msk.any():
                continue
            fold[a]["by_n_bin"][bn] = {
                "n": int(msk.sum()),
                "brier": float(np.mean(se[msk])),
                "delta_vs_base_within_bin": float(np.mean(se[msk] - seb[msk])),
                "contribution_to_full_delta": float(
                    np.sum(se[msk] - seb[msk]) / len(y24)),
                "mean_abs_shift": float(np.mean(np.abs(b * dv[msk]))),
                "base_rate": round(float(np.mean(y24[msk])), 6),
            }
    for a in d_grid:
        fold[a]["arm_pred_gap_vs_k111_self"] = {
            "mean_abs": float(np.mean(np.abs(pred24[a] - pred24[anchor_self]))),
            "max_abs": float(np.max(np.abs(pred24[a] - pred24[anchor_self]))),
        }

    fold_argmin = min(fold, key=lambda a: fold[a]["brier"])
    sel_headline = fold[selected]

    # ---- the reproduction gate: G1f, i.e. the CONTROL beta ------------------- #
    dv111 = d_by_season[anchor_self][VALID_SEASON]
    p_g1f = off.apply_linear(p24, dv111, beta_ctrl)
    br_g1f = float(np.mean((p_g1f - y24) ** 2))
    # the same gate run through the INHERITED float path (exp_022's literal
    # subtraction), so the exact exp_022 number is on the record next to ours
    dv111_legacy = d_arm022["G1"][season == VALID_SEASON]
    br_g1f_legacy = float(np.mean(
        (off.apply_linear(p24, dv111_legacy, beta_ctrl) - y24) ** 2))
    gate = dict(GATE)
    gate.update({
        "beta_control_recovered": beta_ctrl,
        "brier_got": br_g1f,
        "brier_got_through_inherited_literal_form": br_g1f_legacy,
        "abs_diff_inherited_form": abs(
            br_g1f_legacy - GATE["fold_2024_valid_all_brier"]),
        "expanded_minus_inherited_brier": br_g1f - br_g1f_legacy,
        "abs_diff": abs(br_g1f - GATE["fold_2024_valid_all_brier"]),
        "passed": bool(abs(br_g1f - GATE["fold_2024_valid_all_brier"])
                       < GATE["tolerance"]),
        "same_arm_with_REFITTED_beta": {
            "beta": block[anchor_self]["beta_fitted_on_blocks"],
            "brier": fold[anchor_self]["brier"],
            "abs_diff_vs_G1f": abs(fold[anchor_self]["brier"]
                                   - GATE["fold_2024_valid_all_brier"]),
            "note": "this is exp_022 G1 (0.2480365266).  It also sits inside 5e-6 "
                    "of G1f, which is why the gate is run at the control beta — "
                    "otherwise it would pass for the wrong reason.",
        },
        "base_brier_no_offset": br_base24,
        "base_pred_sha256": got_sha,
    })
    print("REPRODUCTION GATE: " + json.dumps(gate), flush=True)
    np.save(os.path.join(OUT_DIR, "pred",
                         f"pred_SELECTED_{selected}_seed42_fold_2024.npy"),
            pred24[selected])
    np.save(os.path.join(OUT_DIR, "pred",
                         f"pred_ANCHOR_{anchor_self}_G1f_seed42_fold_2024.npy"),
            p_g1f)

    # ---- the honesty cost ---------------------------------------------------- #
    selection_cost = {
        "block_argmin": selected,
        "fold_2024_argmin_DIAGNOSTIC": fold_argmin,
        "agree": bool(selected == fold_argmin),
        "brier_of_block_choice_on_fold_2024": fold[selected]["brier"],
        "brier_of_fold_2024_argmin": fold[fold_argmin]["brier"],
        "cost_of_choosing_on_blocks_brier": float(fold[selected]["brier"]
                                                  - fold[fold_argmin]["brier"]),
        "cost_of_choosing_on_blocks_score": round(
            -100000.0 * float(fold[selected]["brier"] - fold[fold_argmin]["brier"])
            / BSS_REF_BRIER, 5),
        "cross_block_argmin": selected_cross,
        "cross_block_choice_fold_2024_brier": fold[selected_cross]["brier"],
        "note": "the fold_2024 argmin is what cheating would have bought.  It is "
                "reported to price the discipline, not to be adopted.",
    }
    print("SELECTION: " + json.dumps(selection_cost), flush=True)

    # ---- league-target risk, measured where the audit said to look ----------- #
    ph24 = df.loc[season == VALID_SEASON, bk.PIT_HAND].to_numpy()
    league_risk = {
        "warning": "the `league` target pushes L onto CELL-HIT rows too, so the "
                   "pitcher-hand-2 cells — whose L flips sign across vintages "
                   "(exp_022 audit) — reach rows that the hard backoff never "
                   "touched.  A local gain does not retire this risk.",
        "L_by_cell_by_season": {s: v["L_by_cell"]
                                for s, v in L_diag["per_season"].items()},
        "n_by_cell_2024": L_diag["per_season"]["2024"]["n_by_cell"],
        "fold_2024_rows_by_pitcher_hand": {
            str(h): int((ph24 == h).sum())
            for h in pd.unique(ph24[~pd.isna(ph24)])},
        "hit_rows_touched_by_L_under_league_k111": int(np.sum(
            (d_by_season[arm_id(K_DEPLOYED, 'league')][VALID_SEASON][hit24]
             != d_by_season[anchor_self][VALID_SEASON][hit24]))),
        "hit_rows_untouched_under_self": int(np.sum(
            d_by_season[anchor_self][VALID_SEASON][hit24]
            != d_k[K_DEPLOYED][season == VALID_SEASON][hit24])),
    }
    ph2 = (ph24 == 2) if np.any(ph24 == 2) else None
    if ph2 is not None:
        for a in [anchor_self, arm_id(K_DEPLOYED, "league"), selected]:
            se = (pred24[a] - y24) ** 2
            seb = (p24 - y24) ** 2
            league_risk.setdefault("by_pitcher_hand", {})[a] = {
                "ph2_n": int(ph2.sum()),
                "ph2_delta_vs_base": float(np.mean(se[ph2] - seb[ph2])),
                "ph1_delta_vs_base": float(np.mean(se[~ph2] - seb[~ph2])),
            }

    # ---- output --------------------------------------------------------------- #
    metrics = {
        "exp_id": "exp_023",
        "title": "offset d construction — shrink constant k x shrink target "
                 "(self / league), 7 x 2 grid, selected on prior-season blocks",
        "stage": "stage 1 only, seed 42.  No model is trained; three stage-1 "
                 "prediction vectors are reused byte-for-byte (sha asserted).",
        "seed": SEED,
        "k_grid": K_GRID,
        "targets": TARGETS,
        "n_arms": len(d_grid),
        "n_distinct_arms": len(d_grid) - 1,
        "changed_vs_exp_022": [
            "the EB shrink constant k of the personal (pitcher, batter_hand) "
            "estimate is swept over [0,30,55,111,200,400,800]; exp_022 held it "
            "at the exp_018 value 111 throughout",
            "the same k also sets the league term's weight k/(n+k) in the "
            "`league` target (exp_022 G2 used 111 there)",
            "the arm is CHOSEN on supply blocks 2022+2023; exp_022 reported all "
            "arms on fold_2024 with no selection step",
            "beta is re-fitted per arm on the same blocks (unchanged in kind, "
            "but its value moves with k — it is a dependent output, not an axis)",
        ],
        "unchanged_vs_exp_022": [
            "base prediction vectors (fold_2024 and both blocks), iter 650/170, "
            "seed 42, exp_004 F-mask, clip bounds 1e-6, closed-form no-intercept "
            "slope estimator, supply blocks 2022+2023, league table code "
            "(backoff.py byte-identical, sha asserted)",
        ],
        "inherited_source_sha256": SHA_INHERITED,
        "source_sha256": src_sha,
        "sha256_reused_inputs": {
            "train.csv": csv_sha,
            "base_pred_fold_2024": got_sha,
            "blocks": blk_meta,
        },
        "reproduction_gate": gate,
        "k111_equivalence": equiv,
        "anchor_check": anchor_check,
        "d_vector_integrity_exp022_G1": integrity022,
        "asof_independent_recheck": asof_indep,
        "asof_runtime_source_seasons": asof_runtime,
        "selection": {
            "rule": "argmin Brier on supply blocks 2022+2023, beta fitted on the "
                    "same blocks.  fold_2024 is read only after this line.",
            "selected_arm": selected,
            "selected_block_brier": block[selected]["brier_blocks"],
            "selected_block_score": block[selected]["score_blocks"],
            "selected_beta": block[selected]["beta_fitted_on_blocks"],
            "headline_fold_2024_brier": sel_headline["brier"],
            "headline_fold_2024_score": sel_headline["score"],
            "headline_delta_vs_base_no_offset": sel_headline[
                "delta_vs_base_no_offset"],
            "headline_score_gain_vs_base": sel_headline["score_gain_vs_base"],
            "vs_deployed_anchor_k111_self": {
                "anchor_brier_refit_beta": fold[anchor_self]["brier"],
                "anchor_brier_G1f_control_beta": br_g1f,
                "selected_minus_anchor_G1f_brier": float(sel_headline["brier"]
                                                         - br_g1f),
                "selected_minus_anchor_G1f_score": round(
                    -100000.0 * float(sel_headline["brier"] - br_g1f)
                    / BSS_REF_BRIER, 5),
            },
        },
        "selection_cost": selection_cost,
        "base_no_offset": {
            "fold_2024_brier": br_base24, "fold_2024_score": score(br_base24),
            "blocks_brier": br_base_blocks,
        },
        "grid_blocks_SELECTION_SURFACE": block,
        "grid_fold_2024_DIAGNOSTIC_ONLY": fold,
        "n_bin_definition": {
            "hit_tertile_boundaries_n_cell": [q1, q2],
            "note": "n_cell is the (pitcher, batter_hand) cell count that fed the "
                    "row's shrunk estimate.  miss_n0 rows have no personal cell "
                    "at all; `fx_*` are fixed cuts, the others are hit-row "
                    "tertiles.",
            "bin_sizes": {b: int(m.sum()) for b, m in bins24.items()},
        },
        "league_target_risk": league_risk,
        "conversion": {
            "formula": f"score = 100000 * (1 - brier / {BSS_REF_BRIER})",
            "sensitivity": "brier 1e-6 = score 0.4003; a 6-decimal Brier carries "
                           "+-0.2002 in the converted number",
            "caveat": "local measurement is at iter 650 (<=2023 training); the "
                      "deployed build is iter 7700 (<=2024).  Every number here "
                      "carries that caveat.",
        },
        "leak_boundary": {
            "tables": "row of season s uses seasons <= s-1 only; verified twice — "
                      "at runtime, and independently by re-reading train.csv and "
                      "looking the contributing row_ids' seasons up again",
            "beta_fit": "supply blocks 2022+2023 only",
            "arm_selection": "supply blocks 2022+2023 only",
            "design_matrix": "unchanged — no deviation is ever a model feature",
            "missing": "no sentinel fill anywhere; L=0 and d=0 are the "
                       "estimators' own no-op outputs",
        },
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__},
    }
    metrics["train_time_min"] = 0.0
    metrics["compute_time_min"] = round((time.time() - t_start) / 60, 2)
    metrics["inference_time_sec"] = None

    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(json.loads(json.dumps(metrics, default=str)), f, indent=2,
                  ensure_ascii=False)

    print("\n--- GRID: block (selection surface) vs fold_2024 (diagnostic) ---",
          flush=True)
    print(f"{'arm':<16}{'beta':>10}{'blk_brier':>13}{'blk_dlt':>12}"
          f"{'f24_brier':>13}{'f24_score':>12}{'f24_dlt':>12}", flush=True)
    for k in K_GRID:
        for t in TARGETS:
            a = arm_id(k, t)
            print(f"{a:<16}{block[a]['beta_fitted_on_blocks']:>10.6f}"
                  f"{block[a]['brier_blocks']:>13.9f}"
                  f"{block[a]['delta_vs_base_blocks']:>12.2e}"
                  f"{fold[a]['brier']:>13.9f}{fold[a]['score']:>12.3f}"
                  f"{fold[a]['delta_vs_base_no_offset']:>12.2e}", flush=True)
    print(f"\nBLOCK ARGMIN = {selected}   fold_2024 argmin = {fold_argmin}",
          flush=True)
    assert gate["passed"], "REPRODUCTION GATE FAILED: " + json.dumps(gate)
    print(f"[{time.strftime('%H:%M:%S')}] DONE "
          f"{metrics['compute_time_min']} min", flush=True)


if __name__ == "__main__":
    main()
