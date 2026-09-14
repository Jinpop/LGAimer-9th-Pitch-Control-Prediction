"""exp_022 stage 1 — how much is the offset's missing 21.5% coverage worth?

WHAT IS AND IS NOT RE-COMPUTED
------------------------------
Nothing is trained here.  The stage-1 predictions are reused:

  fold_2024   exp_020/pred/pred_E0_B3_P4iter_seed42_fold_2024.npy   (sha asserted)
  block 2022  exp_022/pred/pred_B3_block_2022_seed42.npy   (b3_block.py, gated
  block 2023  exp_022/pred/pred_B3_block_2023_seed42.npy    against exp_020's
                                                            recorded Brier)

The blocks had to be regenerated because exp_020 kept its block residuals in
memory and wrote only the fold_2024 vectors; `beta` is fitted on the block
residuals, so they are unavoidable.  `b3_block.py` explains and gates that.

ARMS (all on the SAME fold_2024 base vector — a paired comparison, the base
model's seed noise cancels in every difference below)
------------------------------------------------------------------------------
  G0   p' = clip(p + beta_G0 * d_G0)   control = exp_020 E1, and the gate
  G1   p' = clip(p + beta_G1 * d_G1)   hard backoff, beta re-fitted (PRIMARY)
  G1f  p' = clip(p + beta_G0 * d_G1)   hard backoff at the CONTROL's beta
  G2   p' = clip(p + beta_G2 * d_G2)   soft shrink, beta re-fitted

WHY G1f EXISTS (added, not substituted)
---------------------------------------
The brief asks for two things that cannot both hold of G1: (a) beta is re-fitted
per arm, and (b) G1's predictions on HIT rows are identical to G0's.  If beta
moves, every hit-row prediction moves with it, because the hit rows carry the
same `d` in both arms.  The two requests are reconciled by splitting them:

  * the integrity property is asserted where it actually lives — on the
    DEVIATION vector: `d_G1[hit] == d_G0[hit]` bit for bit (backoff.py).  That is
    what "the hard backoff touches only miss rows" means, and a violation there
    is a code bug.
  * G1f then realises the requested property on the PREDICTIONS: holding beta at
    the control's value, G1f's hit-row predictions ARE byte-identical to G0's,
    so `G1f - G0` is a pure coverage effect with nothing else moving.  It is
    asserted, and it is the cleanest attribution.
  * G1 stays the primary arm as specified.  `G1 - G1f` isolates the beta shift.

BETA
----
Closed form, no intercept, `beta = E[d r] / E[d^2]`, fitted ONLY on the supply
blocks 2022 + 2023 (efficiency rule #6: the latest two).  Valid 2024 never
enters any beta that is applied.  The oracle beta (fitted ON valid 2024) is
computed for the transfer-efficiency ratio and is DISPLAY ONLY.
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
import exp016_ref as ref                                          # noqa: E402
from labels import recover                                        # noqa: E402

SEED = 42
DATA_PATH = os.path.expanduser("~/SungjinPark/data/train.csv")
OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_022")
EXP020_PRED = os.path.expanduser(
    "~/SungjinPark/experiments/exp_020/pred/pred_E0_B3_P4iter_seed42_fold_2024.npy")

TARGET = "control_success"
DROP_COLS = ["row_id", TARGET]
F_EXCLUDE_MAX_SEASON = 2022
SUPPLY_BLOCKS = [2022, 2023]          # efficiency rule #6: latest two
VALID_SEASON = 2024
EPS = 1e-6
BSS_REF_BRIER = 0.2498069             # the brief's fixed conversion reference

EXPECTED = {"valid_all": 253507, "valid_R": 223497}

# --- everything below is transcribed from experiments/exp_020/metrics.json --- #
BASE_SHA = "1e4fba70fe2322d02c246209e4d2dce967f890044aa182e8de4b47525cfafa97"
TRAIN_CSV_SHA = "d2081186b458b49f60b082be480c273135833e15ba59a76d033af28bcf8763ff"
GATE = {
    "arm": "G0 = exp_020 E1 (B3 @ iter 650 + linear offset on d_s)",
    "beta": 0.541136,
    "fold_2024_valid_all_brier": 0.248061,
    "base_brier_no_offset": 0.248191,
    "oracle_beta": 0.652631,
    "tolerance": 5e-6,
    "on_fail": "abort — the reused prediction or the lookup table is not "
               "exp_020's, and nothing downstream is readable",
}
SHA_EXPECTED = {
    "offset.py": "68d6007db03ab850e2a5181383a4148fae8947c711558282c9846391f13e0488",
    "labels.py": "439cd504478b4bc8162e9dedc32be27bfadd5d1178472d6b384b60f46ed7a9f2",
    "features_platoon.py": "68d63063e8ce659cf2876fbca4f5adef7ee1e720620ae63bb78ea"
                           "496c8dadf09",
    "exp016_ref.py": "4f24deb71d9eef3d626900f34a71aecd2d6acbae646eb9c92c51189a38"
                     "a3ece1",
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


def seg(y, p, extra_ref=True):
    m = ref.seg_metrics(y, p)
    if extra_ref and len(y):
        b = float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))
        m["score_vs_fixed_ref"] = round(100000.0 * (1.0 - b / BSS_REF_BRIER), 3)
        m["brier_full"] = b
    return m


def clip_stats(p, d, beta):
    raw = p + beta * d
    lo = raw < EPS
    hi = raw > 1.0 - EPS
    return {"beta": beta,
            "n_clip_low": int(lo.sum()), "n_clip_high": int(hi.sum()),
            "n_clip_total": int(lo.sum() + hi.sum()),
            "clip_pct": round(100.0 * float((lo | hi).mean()), 6),
            "max_abs_shift": round(float(np.max(np.abs(beta * d))), 8),
            "mean_abs_shift": round(float(np.mean(np.abs(beta * d))), 8),
            "n_rows_moved": int((d != 0).sum()),
            "moved_pct": round(100.0 * float((d != 0).mean()), 3)}


def main():
    t_start = time.time()
    np.random.seed(SEED)
    os.makedirs(os.path.join(OUT_DIR, "pred"), exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] loading {DATA_PATH}", flush=True)
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    assert df["row_id"].is_unique
    assert df["season"].is_monotonic_increasing
    base_feats = [c for c in df.columns if c not in DROP_COLS]
    assert len(base_feats) == 47

    rec = recover(df, verbose=False)
    assert rec["success_recovery"]["agree_pct"] == 100.0

    season = df["season"].to_numpy()
    ok = df["recov_ok"].to_numpy()
    succ_season = df.groupby("pitcher_id", sort=False)["season"].shift(-1).to_numpy()
    f_excl = ((df["season"] <= F_EXCLUDE_MAX_SEASON)
              & (df["game_type"] == "F")).to_numpy()
    y_all = df[TARGET].to_numpy(dtype=np.float64)
    is_r_all = (df["game_type"] == "R").to_numpy()

    # ---- personal lookups + deviations (exp_019/exp_020 machinery, untouched) #
    print(f"[{time.strftime('%H:%M:%S')}] personal lookups + deviations", flush=True)
    lu_diag, src_fp = fp.build_season_forward_chained(df, ok, succ_season, f_excl,
                                                      verbose=False)
    dev, hit, src_dev, dev_diag = off.build_deviations(df, ok, succ_season, f_excl,
                                                       verbose=False)
    assert np.array_equal(np.isnan(src_fp), np.isnan(src_dev))

    # ---- NEW: league (pitcher_hand x batter_hand) deviations ---------------- #
    print(f"[{time.strftime('%H:%M:%S')}] league backoff table", flush=True)
    L, has_L, src_L, L_diag = bk.build_league_dev(df, ok, succ_season, f_excl)
    eb_k = fp.RATE_SPEC[bk.SUCC_COL][1]
    d_arm, integrity = bk.build_arm_deviations(df, dev["d_s"], hit, L, has_L, eb_k)
    print("G1 INTEGRITY: " + json.dumps(integrity), flush=True)

    # ---- as-of asserts over every season this run touches ------------------- #
    asof = {}
    for V in SUPPLY_BLOCKS + [VALID_SEASON]:
        for name, arr in (("personal", src_fp), ("league", src_L)):
            u = arr[season == V]
            mx = float(np.nanmax(u)) if np.any(~np.isnan(u)) else float("-inf")
            assert mx < V, f"AS-OF VIOLATION {name} season {V}: used {mx}"
            asof[f"{name}_{V}"] = mx
    asof["slope_blocks"] = SUPPLY_BLOCKS
    asof["valid_season"] = VALID_SEASON
    asof["max_slope_block_lt_valid"] = bool(max(SUPPLY_BLOCKS) < VALID_SEASON)
    asof["max_league_source_season_in_slope_blocks"] = float(
        np.nanmax(src_L[np.isin(season, SUPPLY_BLOCKS)]))
    assert asof["max_slope_block_lt_valid"]
    assert asof["max_league_source_season_in_slope_blocks"] < VALID_SEASON

    # ---- reused predictions ------------------------------------------------- #
    got_sha = sha256_file(EXP020_PRED)
    assert got_sha == BASE_SHA, ("BASE PRED SHA MISMATCH", got_sha, BASE_SHA)
    csv_sha = sha256_file(DATA_PATH)
    assert csv_sha == TRAIN_CSV_SHA, ("train.csv SHA MISMATCH", csv_sha)
    p24 = np.load(EXP020_PRED)
    blk = {}
    blk_meta = {}
    for V in SUPPLY_BLOCKS:
        pth = os.path.join(OUT_DIR, "pred", f"pred_B3_block_{V}_seed42.npy")
        blk[V] = np.load(pth)
        with open(os.path.join(OUT_DIR, f"block_{V}.json")) as f:
            bm = json.load(f)
        assert bm["gate"]["passed"], bm["gate"]
        assert bm["pred_sha256"] == sha256_file(pth), ("block pred sha drift", V)
        assert bm["train_csv_sha256"] == csv_sha
        blk_meta[str(V)] = bm

    y = {V: y_all[season == V] for V in SUPPLY_BLOCKS + [VALID_SEASON]}
    for V in SUPPLY_BLOCKS:
        assert blk[V].shape == y[V].shape, (V, blk[V].shape, y[V].shape)
    assert p24.shape == y[VALID_SEASON].shape
    assert len(y[VALID_SEASON]) == EXPECTED["valid_all"]

    d_blk = {a: {V: d_arm[a][season == V] for V in SUPPLY_BLOCKS + [VALID_SEASON]}
             for a in d_arm}
    hit24 = hit[season == VALID_SEASON]
    isr24 = is_r_all[season == VALID_SEASON]
    assert int(isr24.sum()) == EXPECTED["valid_R"]

    # ---- beta, fitted on the supply blocks only ----------------------------- #
    r_supply = np.concatenate([y[V] - blk[V] for V in SUPPLY_BLOCKS])
    beta_fit, beta_block = {}, {}
    for a in ["G0", "G1", "G2"]:
        dd = np.concatenate([d_blk[a][V] for V in SUPPLY_BLOCKS])
        beta_fit[a] = off.fit_slope(r_supply, dd)
        beta_block[a] = {str(V): off.fit_slope(y[V] - blk[V], d_blk[a][V])
                         for V in SUPPLY_BLOCKS}
    print("--- beta (supply blocks 2022+2023) ---\n"
          + json.dumps({a: beta_fit[a]["beta"] for a in beta_fit}, indent=2),
          flush=True)

    # ---- arms on fold_2024 --------------------------------------------------- #
    y24 = y[VALID_SEASON]
    b_G0 = beta_fit["G0"]["beta"]
    arm_spec = {
        "G0": ("G0", b_G0),
        "G1": ("G1", beta_fit["G1"]["beta"]),
        "G1f": ("G1", b_G0),
        "G2": ("G2", beta_fit["G2"]["beta"]),
    }
    pred, clips = {}, {}
    for a, (dname, b) in arm_spec.items():
        dv = d_blk[dname][VALID_SEASON]
        pred[a] = off.apply_linear(p24, dv, b)
        clips[a] = clip_stats(p24, dv, b)

    # ---- reproduction gate --------------------------------------------------- #
    br_G0 = float(np.mean((pred["G0"] - y24) ** 2))
    br_base = float(np.mean((p24 - y24) ** 2))
    d24_G0 = d_blk["G0"][VALID_SEASON]
    beta_or_G0 = float(np.dot(d24_G0, y24 - p24) / np.dot(d24_G0, d24_G0))
    gate = dict(GATE)
    gate.update({
        "beta_got": b_G0,
        "beta_abs_diff": round(abs(b_G0 - GATE["beta"]), 9),
        "brier_got": round(br_G0, 9),
        "brier_abs_diff": round(abs(br_G0 - GATE["fold_2024_valid_all_brier"]), 9),
        "base_brier_got": round(br_base, 9),
        "base_brier_abs_diff": round(
            abs(br_base - GATE["base_brier_no_offset"]), 9),
        "oracle_beta_got": round(beta_or_G0, 6),
        "oracle_beta_abs_diff": round(abs(beta_or_G0 - GATE["oracle_beta"]), 9),
        "base_pred_sha256": got_sha,
    })
    gate["passed"] = bool(gate["brier_abs_diff"] <= GATE["tolerance"]
                          and gate["base_brier_abs_diff"] <= GATE["tolerance"]
                          and gate["beta_abs_diff"] <= 5e-6)
    print("REPRODUCTION GATE: " + json.dumps(gate), flush=True)

    # ---- G1f prediction integrity (the brief's literal assert) --------------- #
    pred_integrity = {
        "G1f_hit_rows_byte_identical_to_G0": bool(
            np.array_equal(pred["G1f"][hit24], pred["G0"][hit24])),
        "n_hit_rows_differing_G1f_vs_G0": int(
            np.sum(pred["G1f"][hit24] != pred["G0"][hit24])),
        "n_miss_rows_differing_G1f_vs_G0": int(
            np.sum(pred["G1f"][~hit24] != pred["G0"][~hit24])),
        "n_hit_rows_differing_G1_vs_G0": int(
            np.sum(pred["G1"][hit24] != pred["G0"][hit24])),
        "G1_hit_rows_move_only_through_beta": bool(
            abs(beta_fit["G1"]["beta"] - b_G0) > 0),
        "note": "G1 re-fits beta, so its HIT rows move by (beta_G1 - beta_G0)*d_s. "
                "G1f holds beta at the control value and is byte-identical on hit "
                "rows, which is where the coverage-only attribution is read.",
    }
    assert pred_integrity["G1f_hit_rows_byte_identical_to_G0"], pred_integrity
    print("PRED INTEGRITY: " + json.dumps(pred_integrity), flush=True)

    # ---- metrics, with the miss / hit segment split -------------------------- #
    arms = {}
    for a in arm_spec:
        p = pred[a]
        se = (p - y24) ** 2
        se0 = (pred["G0"] - y24) ** 2
        seb = (p24 - y24) ** 2
        arms[a] = {
            "d_source": arm_spec[a][0], "beta_applied": arm_spec[a][1],
            "valid_all": seg(y24, p),
            "valid_R": seg(y24[isr24], p[isr24]),
            "valid_F": seg(y24[~isr24], p[~isr24]),
            "seg_miss": seg(y24[~hit24], p[~hit24]),
            "seg_hit": seg(y24[hit24], p[hit24]),
            "delta_vs_G0": round(float(se.mean() - se0.mean()), 12),
            "delta_vs_base_no_offset": round(float(se.mean() - seb.mean()), 12),
            "delta_vs_G0_within_miss_rows": round(
                float(se[~hit24].mean() - se0[~hit24].mean()), 12),
            "delta_vs_G0_within_hit_rows": round(
                float(se[hit24].mean() - se0[hit24].mean()), 12),
            "delta_vs_G0_contribution_miss_rows": round(
                float((se[~hit24] - se0[~hit24]).sum() / len(y24)), 12),
            "delta_vs_G0_contribution_hit_rows": round(
                float((se[hit24] - se0[hit24]).sum() / len(y24)), 12),
            "score_delta_vs_G0_fixed_ref": round(
                -100000.0 * float(se.mean() - se0.mean()) / BSS_REF_BRIER, 3),
            "clip": clips[a],
        }
        np.save(os.path.join(OUT_DIR, "pred", f"pred_{a}_seed42_fold_2024.npy"), p)

    # ---- oracle beta + transfer efficiency ----------------------------------- #
    oracle = {}
    for a in ["G0", "G1", "G2"]:
        dv = d_blk[a][VALID_SEASON]
        bo = float(np.dot(dv, y24 - p24) / np.dot(dv, dv))
        p_or = off.apply_linear(p24, dv, bo)
        g_or = float(np.mean((p_or - y24) ** 2)) - br_base
        g_fit = float(np.mean((pred[a] - y24) ** 2)) - br_base
        oracle[a] = {
            "oracle_beta_DISPLAY_ONLY": round(bo, 6),
            "fitted_beta": beta_fit[a]["beta"],
            "gain_at_fitted_beta": round(g_fit, 12),
            "gain_at_oracle_beta": round(g_or, 12),
            "transfer_efficiency": round(g_fit / g_or, 4) if g_or else None,
            "brier_at_oracle_beta": round(float(np.mean((p_or - y24) ** 2)), 9),
        }

    # ---- coverage, measured, and who the miss rows are ----------------------- #
    v = season == VALID_SEASON
    pit24 = df.loc[v, "pitcher_id"].to_numpy()
    seen24 = np.isfinite(df.loc[v, bk.SUCC_COL].to_numpy(dtype=np.float64))
    ncell24 = df.loc[v, fp.COUNT_COL].to_numpy(dtype=np.float64)
    apn = df.loc[v, "asof_pitcher_n"].to_numpy(dtype=np.float64)
    rows_per_pit = pd.Series(pit24).value_counts()

    def q(x):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if not len(x):
            return None
        return {"n": int(len(x)), "min": float(np.min(x)),
                "p10": round(float(np.percentile(x, 10)), 2),
                "median": round(float(np.median(x)), 2),
                "mean": round(float(np.mean(x)), 2),
                "p90": round(float(np.percentile(x, 90)), 2),
                "max": float(np.max(x))}

    miss_pits = set(pit24[~hit24].tolist())
    hit_pits = set(pit24[hit24].tolist())
    coverage = {
        "fold_2024_n_rows": int(v.sum()),
        "hit_rows": int(hit24.sum()), "miss_rows": int((~hit24).sum()),
        "hit_pct_measured": round(100.0 * float(hit24.mean()), 3),
        "hit_pct_claimed_exp_019_D4": 78.5,
        "rows_given_league_L_among_miss": int((d_blk["G1"][VALID_SEASON][~hit24]
                                               != 0).sum()),
        "miss_rows_left_at_zero": int((d_blk["G1"][VALID_SEASON][~hit24] == 0).sum()),
        "pitcher_seen_in_table_but_cell_empty": int((seen24 & ~hit24).sum()),
        "pitcher_never_seen_in_table": int((~seen24).sum()),
        "n_distinct_pitchers_all": int(len(set(pit24.tolist()))),
        "n_distinct_pitchers_with_any_miss_row": int(len(miss_pits)),
        "n_distinct_pitchers_with_any_hit_row": int(len(hit_pits)),
        "n_pitchers_entirely_miss": int(len(miss_pits - hit_pits)),
        "asof_pitcher_n_on_miss_rows": q(apn[~hit24]),
        "asof_pitcher_n_on_hit_rows": q(apn[hit24]),
        "fold2024_rows_per_pitcher_entirely_miss": q(
            [rows_per_pit[p] for p in (miss_pits - hit_pits)]),
        "fold2024_rows_per_pitcher_any_hit": q(
            [rows_per_pit[p] for p in hit_pits]),
        "personal_cell_n_on_hit_rows": q(ncell24[hit24]),
        "base_rate_miss_rows": round(float(y24[~hit24].mean()), 6),
        "base_rate_hit_rows": round(float(y24[hit24].mean()), 6),
        "F_share_miss_rows_pct": round(100.0 * float((~isr24[~hit24]).mean()), 3),
        "F_share_hit_rows_pct": round(100.0 * float((~isr24[hit24]).mean()), 3),
        "sd_d_G0_hit": round(float(np.std(d_blk["G0"][VALID_SEASON][hit24])), 6),
        "sd_d_G1_miss": round(float(np.std(d_blk["G1"][VALID_SEASON][~hit24])), 6),
        "abs_L_values_fold_2024": L_diag["per_season"]["2024"]["L_by_cell"],
    }

    metrics = {
        "exp_id": "exp_022",
        "title": "offset coverage backoff — league (pitcher_hand x batter_hand) "
                 "deviation on personal-lookup misses",
        "stage": "stage 1 only (seed 42).  stage 2 NOT run — separate approval.",
        "base_exp": "exp_020 E1 (exp_016 B3 @ iter 650 + linear offset), seed 42",
        "seed": SEED,
        "changed_vs_exp_020_E1": [
            "d_s is replaced by d_G1 / d_G2, which add a league "
            "(pitcher_hand x batter_hand) deviation where the personal cell is "
            "empty (G1) or as the shrink target (G2)",
            "beta is re-fitted per arm on the SAME supply blocks (2022+2023); "
            "beta is therefore a dependent output of each d, not a free variable",
            "the fold_2024 stage-1 prediction is REUSED byte-for-byte (sha "
            "asserted), so no model differs between arms",
            "blocks 2022/2023 stage-1 predictions were REGENERATED (exp_020 did "
            "not save them) and gated against exp_020's recorded block Brier",
        ],
        "unchanged_vs_exp_020_E1": [
            "base prediction vector, iter 650, seed 42, exp_004 F-mask, EB k, "
            "clip bounds 1e-6, closed-form no-intercept slope estimator, "
            "supply blocks 2022+2023",
        ],
        "eb_k_success": eb_k,
        "supply_blocks": SUPPLY_BLOCKS,
        "valid_season": VALID_SEASON,
        "reproduction_gate": gate,
        "d_vector_integrity_G1": integrity,
        "prediction_integrity": pred_integrity,
        "asof_asserts": asof,
        "coverage": coverage,
        "beta_fitted_supply_blocks": beta_fit,
        "beta_per_block": beta_block,
        "oracle_beta_and_transfer": oracle,
        "arms_fold_2024": arms,
        "headline_deltas": {
            "G1_minus_G0_brier": arms["G1"]["delta_vs_G0"],
            "G1_minus_G0_score": arms["G1"]["score_delta_vs_G0_fixed_ref"],
            "G1f_minus_G0_brier": arms["G1f"]["delta_vs_G0"],
            "G1f_minus_G0_score": arms["G1f"]["score_delta_vs_G0_fixed_ref"],
            "G2_minus_G0_brier": arms["G2"]["delta_vs_G0"],
            "G2_minus_G0_score": arms["G2"]["score_delta_vs_G0_fixed_ref"],
            "G2_minus_G1_brier": round(arms["G2"]["delta_vs_G0"]
                                       - arms["G1"]["delta_vs_G0"], 12),
            "G1_minus_G1f_brier": round(arms["G1"]["delta_vs_G0"]
                                        - arms["G1f"]["delta_vs_G0"], 12),
            "exp_019_D4_upper_bound_score": 11.0,
            "conversion": f"score = 100000 * (1 - brier / {BSS_REF_BRIER})",
        },
        "league_table_diag": L_diag,
        "personal_deviation_diag": dev_diag,
        "block_regeneration": blk_meta,
        "leak_boundary": {
            "league_table_inputs": "control_success of rows with season <= s-1, "
                                   "recov_ok, successor inside the window, "
                                   "exp_004 F-mask — identical mask to the "
                                   "personal table",
            "league_table_as_of": "row of season s uses seasons <= s-1; asserted "
                                  "at runtime per season (backoff.build_league_dev)",
            "beta_fit": "supply blocks 2022+2023 only; valid season is 2024",
            "design_matrix": "unchanged — no deviation column is ever a feature",
            "test_path": "the same structure frozen on <= 2024; no test row "
                         "consults another test row (rule 2-4)",
            "missing_hand": "L = 0, i.e. current behaviour, not a sentinel fill",
        },
        "sha256": {
            "train.csv": csv_sha,
            "base_pred_fold_2024": got_sha,
            **{f: sha256_file(os.path.join(SRC_DIR, f))
               for f in ["fit_offset.py", "backoff.py", "b3_block.py",
                         "offset.py", "labels.py", "features_platoon.py",
                         "exp016_ref.py"]},
        },
        "sha256_expected": SHA_EXPECTED,
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__},
    }
    for k, want in SHA_EXPECTED.items():
        assert metrics["sha256"][k] == want, (k, metrics["sha256"][k])
    metrics["stage1_offset_time_min"] = round((time.time() - t_start) / 60, 2)
    metrics["train_time_min"] = round(
        sum(blk_meta[str(V)]["minutes"] for V in SUPPLY_BLOCKS)
        + metrics["stage1_offset_time_min"], 2)
    metrics["inference_time_sec"] = None

    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(json.loads(json.dumps(metrics, default=str)), f, indent=2,
                  ensure_ascii=False)
    print("--- headline ---\n"
          + json.dumps(metrics["headline_deltas"], indent=2), flush=True)
    print("--- arms ---\n" + json.dumps(
        {a: {"brier": arms[a]["valid_all"]["brier_full"],
             "score": arms[a]["valid_all"]["score_vs_fixed_ref"],
             "d_vs_G0": arms[a]["delta_vs_G0"],
             "miss_brier": arms[a]["seg_miss"]["brier_full"],
             "hit_brier": arms[a]["seg_hit"]["brier_full"]} for a in arms},
        indent=2), flush=True)
    assert gate["passed"], "REPRODUCTION GATE FAILED: " + json.dumps(gate)
    print(f"[{time.strftime('%H:%M:%S')}] DONE "
          f"{metrics['stage1_offset_time_min']} min", flush=True)


if __name__ == "__main__":
    main()
