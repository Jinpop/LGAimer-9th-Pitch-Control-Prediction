"""exp_024 — three-seed promotion test of exp_023's block-selected arm.

WHAT IS AND IS NOT MEASURED HERE
--------------------------------
Two arms, three seeds.  Nothing is selected.

  A_control    k = 111 x self, at the CONTROL beta  = exp_022 G1f = the `d` of
               deployed candidates H/I
  B_candidate  k = 400 x league, beta re-fitted on the supply blocks
               = the arm exp_023's block-selection procedure chose

exp_023's 7 x 2 grid is NOT re-run on fold_2024.  The grid appears in this file
in exactly one place: the per-seed RESELECTION DIAGNOSTIC, which recomputes the
block (2022 + 2023) argmin at seeds 7 and 2024 to see whether the selection
procedure is stable across seeds.  That diagnostic never touches fold_2024, and
by pre-registration it cannot change the arm under test.

THE ADOPTION RULE — PRE-REGISTERED IN config.yaml, NOT RELAXABLE
----------------------------------------------------------------
  (1) the three paired deltas (B - A, same base vector, same seed) all share a
      sign, AND
  (2) |mean delta| >= 3 x the SEED STD OF THE CONTROL ARM (ddof=1).

Rule (2)'s denominator is the control's spread, as in exp_018 and exp_021.  The
std of the paired delta itself is also reported because the paired design
cancels part of the base noise and that number is the more natural one to look
at — but it is NOT the denominator of the registered test, and swapping it in
after seeing the data would be moving the goalposts.  Both are printed side by
side and labelled.

BETA
----
Not an axis.  For every (arm, seed) it is the closed-form no-intercept slope
E[d r] / E[d^2] fitted on supply blocks 2022 + 2023 of THAT SEED.  Arm A's beta
is the CONTROL beta — fitted on the plain d_s (no league term) — because that,
and not the re-fitted one, is what makes the arm G1f, i.e. the thing actually
deployed.  Arm A at its own re-fitted beta (exp_022's G1) is reported as a
secondary comparison so the choice is visible rather than buried.

fold_2024 is read only after every beta is fixed.  No k, target, or beta is ever
chosen by looking at 2024.
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
OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_024")
EXP020_PRED = os.path.expanduser(
    "~/SungjinPark/experiments/exp_020/pred/pred_E0_B3_P4iter_seed42_fold_2024.npy")
EXP022_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_022")

SEEDS = [42, 7, 2024]
GATE_SEED = 42

TARGET = "control_success"
DROP_COLS = ["row_id", TARGET]
F_EXCLUDE_MAX_SEASON = 2022
SUPPLY_BLOCKS = [2022, 2023]
VALID_SEASON = 2024
EPS = 1e-6
BSS_REF_BRIER = 0.2498069

K_GRID = [0, 30, 55, 111, 200, 400, 800]
TARGETS = ["self", "league"]
K_DEPLOYED = 111
ARM_A = ("k111_self", "CONTROL beta")
ARM_B = ("k400_league", "beta re-fitted on blocks")

EXPECTED = {"valid_all": 253507, "valid_R": 223497}
TRAIN_CSV_SHA = "d2081186b458b49f60b082be480c273135833e15ba59a76d033af28bcf8763ff"
BASE_SHA_42 = "1e4fba70fe2322d02c246209e4d2dce967f890044aa182e8de4b47525cfafa97"
BLOCK_SHA_42 = {
    2022: "9087d3dc078f7ac36dc569ae4366f97e61a2c28e506c40db8c02d0a0e6a56f45",
    2023: "467055bd65b2233a7fd54a9b71347100c42a5b33d7abdc840c0ab6eed9e1a411",
}

SHA_INHERITED = {
    "offset.py": "68d6007db03ab850e2a5181383a4148fae8947c711558282c9846391f13e0488",
    "backoff.py": "4169d2dadd8a07dceef2098a88971d3b3e437276bfb28834295e7df523591039",
    "labels.py": "439cd504478b4bc8162e9dedc32be27bfadd5d1178472d6b384b60f46ed7a9f2",
    "features_platoon.py": "68d63063e8ce659cf2876fbca4f5adef7ee1e720620ae63bb78ea"
                           "496c8dadf09",
    "exp016_ref.py": "4f24deb71d9eef3d626900f34a71aecd2d6acbae646eb9c92c51189a38"
                     "a3ece1",
    # exp_023's expanded-form (B) deviation module, byte-identical
    "dev_k.py": "e1cd86dd2795a4d12f279f7ea600353f441cfc7b19eea12207fb87117207a0d9",
}

# exp_023 seed-42 numbers this run must land on again
GATE = {
    "arm_A_k111_self_at_control_beta": 0.2480377446210693,
    "arm_B_k400_league": 0.24799968498605615,
    "tolerance": 5e-6,
    "source": "exp_023/metrics.json (reproduction_gate.brier_got and "
              "selection.headline_fold_2024_brier)",
    "on_fail": "abort — the reused vectors or the inherited code are not "
               "exp_023's and no seed comparison downstream is readable",
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


def arm_id(k, t):
    return f"k{k}_{t}"


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

    # ---- deviations.  SEED-INDEPENDENT: they come from the labels, not the
    # ---- model, so one build serves all three seeds -------------------------- #
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
    assert fp.RATE_SPEC[bk.SUCC_COL][1] == K_DEPLOYED

    print(f"[{time.strftime('%H:%M:%S')}] multi-k personal deviations", flush=True)
    d_k, hit, n_cell0, src_k, dk_diag = dk.build_multi_k(
        df, ok, succ_season, f_excl, K_GRID, verbose=False)
    assert np.array_equal(hit, hit_ref)
    dd111 = d_k[K_DEPLOYED] - dev_ref["d_s"]
    equiv = {
        "form_used": "(s_cell - n_cell*overall)/(n_cell + k)  [expanded, exp_023]",
        "max_abs_diff_k111_vs_inherited_literal_form": float(np.max(np.abs(dd111))),
        "tolerance_used": 1e-12,
        "miss_rows_exactly_zero_all_k": {
            str(k): bool(np.all(d_k[k][~hit] == 0.0)) for k in K_GRID},
    }
    assert equiv["max_abs_diff_k111_vs_inherited_literal_form"] < 1e-12, equiv
    assert all(equiv["miss_rows_exactly_zero_all_k"].values()), equiv

    L_eff = np.where(has_L, L, 0.0)
    d_grid = {arm_id(k, t): dk.build_arm(d_k[k], hit, L_eff, n_cell0, k, t)
              for k in K_GRID for t in TARGETS}
    assert np.array_equal(d_grid["k0_self"], d_grid["k0_league"])   # 13 distinct
    for k in K_GRID:
        assert np.array_equal(d_grid[arm_id(k, "self")][hit], d_k[k][hit])

    # ---- GATE 2: as-of verified through an INDEPENDENT re-read of the file --- #
    # the exp_022 audit's objection stands: a runtime assert reads `season` out
    # of the very mask built from `season` and so proves nothing.  Here the
    # season of every row that fed a table is looked up again from a FRESH read
    # of train.csv, keyed by row_id.
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

    # ---- per-season slices -------------------------------------------------- #
    y = {V: y_all[season == V] for V in SUPPLY_BLOCKS + [VALID_SEASON]}
    assert len(y[VALID_SEASON]) == EXPECTED["valid_all"]
    y24 = y[VALID_SEASON]
    hit24 = hit[season == VALID_SEASON]
    isr24 = is_r_all[season == VALID_SEASON]
    n_cell24 = n_cell0[season == VALID_SEASON]
    ph24 = df.loc[season == VALID_SEASON, bk.PIT_HAND].to_numpy()
    assert int(isr24.sum()) == EXPECTED["valid_R"]

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

    # ---- load the base vectors --------------------------------------------- #
    def load_bases(s):
        """(p_fold_2024, {block: p}, provenance).  Seed 42 REUSES the stored
        exp_020/exp_022 vectors with their sha asserted; 7 and 2024 read what
        b3_block.py produced in this experiment."""
        prov = {}
        if s == GATE_SEED:
            got = sha256_file(EXP020_PRED)
            assert got == BASE_SHA_42, ("BASE PRED SHA MISMATCH", got)
            p24 = np.load(EXP020_PRED)
            prov["fold_2024"] = {"path": EXP020_PRED, "sha256": got,
                                 "reused_from": "exp_020 arm E0"}
            bl = {}
            for V in SUPPLY_BLOCKS:
                pth = os.path.join(EXP022_DIR, "pred",
                                   f"pred_B3_block_{V}_seed42.npy")
                with open(os.path.join(EXP022_DIR, f"block_{V}.json")) as f:
                    bm = json.load(f)
                assert bm["gate"]["passed"], bm["gate"]
                got = sha256_file(pth)
                assert bm["pred_sha256"] == got, ("block pred sha drift", V, got)
                assert got == BLOCK_SHA_42[V], ("block sha != exp_023's", V, got)
                assert bm["train_csv_sha256"] == csv_sha
                bl[V] = np.load(pth)
                prov[f"block_{V}"] = {"path": pth, "sha256": got,
                                      "reused_from": "exp_022",
                                      "gate_brier": bm["gate"]["brier"],
                                      "iter": bm["gate"]["iter"]}
        else:
            bl = {}
            for V in SUPPLY_BLOCKS + [VALID_SEASON]:
                pth = os.path.join(OUT_DIR, "pred",
                                   f"pred_B3_block_{V}_seed{s}.npy")
                with open(os.path.join(OUT_DIR, "blocks",
                                       f"block_{V}_seed{s}.json")) as f:
                    bm = json.load(f)
                assert bm["train_csv_sha256"] == csv_sha
                got = sha256_file(pth)
                assert bm["pred_sha256"] == got, ("pred sha drift", V, s, got)
                arr = np.load(pth)
                if V == VALID_SEASON:
                    p24 = arr
                else:
                    bl[V] = arr
                prov[f"block_{V}" if V != VALID_SEASON else "fold_2024"] = {
                    "path": pth, "sha256": got, "produced_by": "exp_024 b3_block.py",
                    "brier_no_offset": bm["gate"]["brier_full"],
                    "iter": bm["stack_iter"], "cell_iter": bm["cell_iter"],
                    "minutes": bm["minutes"]}
        for V in SUPPLY_BLOCKS:
            assert bl[V].shape == y[V].shape, (s, V, bl[V].shape)
        assert p24.shape == y24.shape, (s, p24.shape)
        return p24, bl, prov

    # ---- CODE-PATH CHECK: exp_024's own fold_2024 unit at seed 42 ------------ #
    # exp_024's b3_block.py had never produced a fold_2024 unit before.  Without
    # this run, seeds 7 and 2024 would sit on that new code path while seed 42
    # sat on exp_020's, and any seed-to-seed difference would be confounded with
    # the code path.  The arm evaluation still READS exp_020's stored vector —
    # this unit is a control, not a replacement.
    codepath = {"purpose": "prove exp_024's fold_2024 code path is exp_020's, so "
                           "that the three seeds differ only by seed"}
    p42_own_path = os.path.join(OUT_DIR, "pred", "pred_B3_block_2024_seed42.npy")
    if os.path.exists(p42_own_path):
        p42_own = np.load(p42_own_path)
        p42_ref = np.load(EXP020_PRED)
        d42 = p42_own - p42_ref
        codepath.update({
            "exp024_own_seed42_sha256": sha256_file(p42_own_path),
            "exp020_stored_sha256": sha256_file(EXP020_PRED),
            "bit_identical": bool(np.array_equal(p42_own, p42_ref)),
            "max_abs_diff": float(np.max(np.abs(d42))),
            "n_rows_differing": int(np.sum(d42 != 0.0)),
            "brier_exp024_own": float(np.mean((p42_own - y24) ** 2)),
            "brier_exp020_stored": float(np.mean((p42_ref - y24) ** 2)),
            "brier_abs_diff": abs(float(np.mean((p42_own - y24) ** 2))
                                  - float(np.mean((p42_ref - y24) ** 2))),
            "used_for_arm_evaluation": "NO — exp_020's stored vector is used, "
                                       "sha asserted, exactly as the spec requires",
        })
        del p42_own, p42_ref, d42
    else:
        codepath["ran"] = False
    print("CODE-PATH CHECK: " + json.dumps(codepath), flush=True)

    # ---- per seed ----------------------------------------------------------- #
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

        # --- betas, ALL fitted on the supply blocks of this seed ------------- #
        beta_ctrl = off.fit_slope(
            r_supply, np.concatenate([d_ctrl[V] for V in SUPPLY_BLOCKS]))["beta"]

        block = {}
        for a in d_grid:
            dd = np.concatenate([d_by_season[a][V] for V in SUPPLY_BLOCKS])
            fit = off.fit_slope(r_supply, dd)
            b = fit["beta"]
            pb = off.apply_linear(p_supply, dd, b)
            br = float(np.mean((pb - y_supply) ** 2))
            block[a] = {
                "beta_fitted_on_blocks": b, "beta_se": fit["se"], "beta_t": fit["t"],
                "brier_blocks": br, "score_blocks": score(br),
                "delta_vs_base_blocks": float(br - br_base_blocks),
                "beta_per_block": {str(V): off.fit_slope(
                    y[V] - blk[V], d_by_season[a][V])["beta"] for V in SUPPLY_BLOCKS},
                "sd_d_blocks": float(np.std(dd)),
            }
        # DIAGNOSTIC ONLY — the arm under test is fixed by pre-registration
        block_argmin = min(block, key=lambda a: block[a]["brier_blocks"])

        # --- the two arms on fold_2024 -------------------------------------- #
        def evaluate(name, a, beta):
            dv = d_by_season[a][VALID_SEASON]
            p = off.apply_linear(p24, dv, beta)
            raw = p24 + beta * dv
            se = (p - y24) ** 2
            seb = (p24 - y24) ** 2
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
                "by_n_bin": {},
                "by_pitcher_hand": {},
            }
            for bn, msk in bins24.items():
                if not msk.any():
                    continue
                out["by_n_bin"][bn] = {
                    "n": int(msk.sum()), "brier": float(np.mean(se[msk])),
                    "delta_vs_base_within_bin": float(np.mean(se[msk] - seb[msk])),
                }
            for h in sorted(set(ph24[~pd.isna(ph24)].tolist())):
                m = (ph24 == h)
                out["by_pitcher_hand"][str(int(h))] = {
                    "n": int(m.sum()),
                    "delta_vs_base": float(np.mean(se[m] - seb[m])),
                }
            return out, p

        res_A, pA = evaluate("A_control", ARM_A[0], beta_ctrl)
        res_B, pB = evaluate("B_candidate", ARM_B[0],
                             block[ARM_B[0]]["beta_fitted_on_blocks"])
        # secondary: arm A at its OWN re-fitted beta (exp_022's G1 convention)
        res_A_refit, _ = evaluate("A_refit", ARM_A[0],
                                  block[ARM_A[0]]["beta_fitted_on_blocks"])

        np.save(os.path.join(OUT_DIR, "pred",
                             f"pred_ARM_A_k111_self_G1f_seed{s}_fold_2024.npy"), pA)
        np.save(os.path.join(OUT_DIR, "pred",
                             f"pred_ARM_B_k400_league_seed{s}_fold_2024.npy"), pB)

        delta = res_B["brier"] - res_A["brier"]
        per_seed[str(s)] = {
            "base_no_offset": {"fold_2024_brier": br_base24,
                               "fold_2024_score": score(br_base24),
                               "blocks_brier": br_base_blocks},
            "beta_control_plain_ds": beta_ctrl,
            "A_control": res_A,
            "B_candidate": res_B,
            "A_control_at_own_refitted_beta_SECONDARY": res_A_refit,
            "paired_delta_B_minus_A": delta,
            "paired_delta_score": score_delta(delta),
            "paired_delta_vs_A_refit_SECONDARY": float(res_B["brier"]
                                                       - res_A_refit["brier"]),
            "arm_pred_gap_B_vs_A": {
                "mean_abs": float(np.mean(np.abs(pB - pA))),
                "max_abs": float(np.max(np.abs(pB - pA)))},
            "reselection_diagnostic": {
                "block_argmin": block_argmin,
                "agrees_with_exp023_choice": bool(block_argmin == ARM_B[0]),
                "block_brier_of_argmin": block[block_argmin]["brier_blocks"],
                "block_brier_of_k400_league": block[ARM_B[0]]["brier_blocks"],
                "gap": float(block[ARM_B[0]]["brier_blocks"]
                             - block[block_argmin]["brier_blocks"]),
                "discipline": "DIAGNOSTIC.  The arm under test is fixed at "
                              "k400_league by pre-registration; this cannot "
                              "change it.",
                "block_surface": {a: {"beta": block[a]["beta_fitted_on_blocks"],
                                      "brier_blocks": block[a]["brier_blocks"]}
                                  for a in sorted(block)},
            },
            "beta_per_block": {
                "A_control_plain_ds": {str(V): off.fit_slope(
                    y[V] - blk[V], d_ctrl[V])["beta"] for V in SUPPLY_BLOCKS},
                "B_candidate": block[ARM_B[0]]["beta_per_block"],
            },
        }
        print(f"  seed {s}: A {res_A['brier']:.12f} ({res_A['score']:.5f})  "
              f"B {res_B['brier']:.12f} ({res_B['score']:.5f})  "
              f"delta {delta:.6e} ({score_delta(delta):+.5f})  "
              f"block argmin {block_argmin}", flush=True)
        del p24, blk

    # ---- GATE 1: seed 42 must land on exp_023's two numbers ------------------ #
    g42 = per_seed[str(GATE_SEED)]
    gate = dict(GATE)
    gate.update({
        "arm_A_got": g42["A_control"]["brier"],
        "arm_A_abs_diff": abs(g42["A_control"]["brier"]
                              - GATE["arm_A_k111_self_at_control_beta"]),
        "arm_B_got": g42["B_candidate"]["brier"],
        "arm_B_abs_diff": abs(g42["B_candidate"]["brier"]
                              - GATE["arm_B_k400_league"]),
        "beta_control_got": g42["beta_control_plain_ds"],
        "beta_B_got": g42["B_candidate"]["beta_applied"],
        "beta_control_exp023": 0.541136,
        "beta_B_exp023": 0.676950,
    })
    gate["passed"] = bool(gate["arm_A_abs_diff"] < GATE["tolerance"]
                          and gate["arm_B_abs_diff"] < GATE["tolerance"])
    print("REPRODUCTION GATE: " + json.dumps(gate), flush=True)
    assert gate["passed"], "REPRODUCTION GATE FAILED: " + json.dumps(gate)

    # ---- the pre-registered decision ---------------------------------------- #
    a_b = [per_seed[str(s)]["A_control"]["brier"] for s in SEEDS]
    b_b = [per_seed[str(s)]["B_candidate"]["brier"] for s in SEEDS]
    ds = [b - a for a, b in zip(a_b, b_b)]
    ctrl_std = float(np.std(a_b, ddof=1))
    cand_std = float(np.std(b_b, ddof=1))
    delta_std = float(np.std(ds, ddof=1))
    mean_d = float(np.mean(ds))
    same_sign = bool(all(d < 0 for d in ds) or all(d > 0 for d in ds))
    ratio = abs(mean_d) / ctrl_std if ctrl_std > 0 else None
    decision = {
        "rule": "PRE-REGISTERED (config.yaml adoption_criterion, same as exp_018 "
                "and exp_021): same sign on all 3 seeds AND |mean delta| >= "
                "3 x CONTROL-arm seed std (ddof=1).  Not relaxable after the fact.",
        "seeds": SEEDS,
        "A_control_brier_by_seed": {str(s): v for s, v in zip(SEEDS, a_b)},
        "B_candidate_brier_by_seed": {str(s): v for s, v in zip(SEEDS, b_b)},
        "paired_delta_by_seed": {str(s): v for s, v in zip(SEEDS, ds)},
        "paired_delta_score_by_seed": {str(s): score_delta(v)
                                       for s, v in zip(SEEDS, ds)},
        "signs_agree": same_sign,
        "mean_delta": mean_d,
        "mean_delta_score": score_delta(mean_d),
        "control_seed_std_DENOMINATOR_OF_THE_TEST": ctrl_std,
        "candidate_seed_std": cand_std,
        "paired_delta_seed_std_REFERENCE_ONLY": delta_std,
        "abs_mean_delta_over_control_std": round(ratio, 4) if ratio else None,
        "abs_mean_delta_over_paired_delta_std_REFERENCE_ONLY": (
            round(abs(mean_d) / delta_std, 4) if delta_std > 0 else None),
        "threshold_multiple": 3.0,
        "MEASURED": bool(same_sign and ctrl_std > 0
                         and abs(mean_d) >= 3.0 * ctrl_std),
        "note": "the paired-delta std is reported because the shared base makes "
                "it the natural spread of the quantity of interest, and it is "
                "expected to be smaller than the control's.  It is NOT the "
                "denominator of the registered test.",
        "reference_effect_seed42_exp023": -3.805964e-05,
    }
    print("DECISION: " + json.dumps(decision), flush=True)

    reselect = {str(s): per_seed[str(s)]["reselection_diagnostic"]["block_argmin"]
                for s in SEEDS}
    reselect_summary = {
        "block_argmin_by_seed": reselect,
        "all_seeds_agree": bool(len(set(reselect.values())) == 1),
        "agrees_with_exp023": bool(all(v == ARM_B[0] for v in reselect.values())),
        "meaning": "if the seeds disagree, the SELECTION PROCEDURE is seed-"
                   "sensitive — a separate finding from whether the selected arm "
                   "beats the control.",
    }
    print("RESELECTION: " + json.dumps(reselect_summary), flush=True)

    metrics = {
        "exp_id": "exp_024",
        "title": "three-seed promotion test of exp_023's block-selected arm "
                 "(k=400 x league) against the deployed d (k=111 x self)",
        "arms": {"A_control": "k=111 x self at the CONTROL beta = exp_022 G1f = "
                              "the d of deployed candidates H/I",
                 "B_candidate": "k=400 x league, beta re-fitted on supply blocks "
                                "= exp_023's block-selected arm"},
        "seeds": SEEDS,
        "changed_vs_exp_023": [
            "seeds 7 and 2024 are added; their three base B3 vectors (blocks "
            "2022, 2023 and fold_2024) are TRAINED here — exp_023 had none",
            "only two arms are evaluated on fold_2024; exp_023's other 11 "
            "distinct arms are not re-run there",
            "the block selection is re-run per seed as a DIAGNOSTIC",
            "b3_block.py gains a 2024 unit spec and a seed argument, and its "
            "reproduction gate fires only at seed 42 (see that file's docstring)",
        ],
        "unchanged_vs_exp_023": [
            "d construction (dev_k.py byte-identical, expanded form B), league "
            "table (backoff.py byte-identical), exp_004 F-mask, clip 1e-6, "
            "closed-form no-intercept slope, supply blocks 2022+2023, "
            "iter 170/650/650, cell iter 240/240/1150, num_threads 6",
        ],
        "inherited_source_sha256": SHA_INHERITED,
        "source_sha256": src_sha,
        "train_csv_sha256": csv_sha,
        "base_vector_provenance": provenance,
        "reproduction_gate": gate,
        "codepath_check_seed42_fold_2024": codepath,
        "k111_equivalence": equiv,
        "asof_independent_recheck": asof_indep,
        "asof_runtime_source_seasons": asof_runtime,
        "decision": decision,
        "reselection_diagnostic_summary": reselect_summary,
        "per_seed": per_seed,
        "n_bin_definition": {
            "hit_tertile_boundaries_n_cell": [float(q1), float(q2)],
            "bin_sizes": {b: int(m.sum()) for b, m in bins24.items()},
        },
        "league_target_risk_UNRESOLVED": {
            "warning": "the `league` target puts L on CELL-HIT rows as well, so "
                       "the pitcher-hand-2 cells reach rows the hard backoff "
                       "never touched.  Those cells' L flips sign across "
                       "vintages (-0.0077 in the 2020 table, +0.0024 in the "
                       "2024 one).  THIS EXPERIMENT DOES NOT ADDRESS THAT.  A "
                       "three-seed pass is about base-model noise, not about "
                       "whether the league cell is stable out of sample.",
            "L_by_cell_by_season": {s: v["L_by_cell"]
                                    for s, v in L_diag["per_season"].items()},
        },
        "conversion": {
            "formula": f"score = 100000 * (1 - brier / {BSS_REF_BRIER})",
            "sensitivity": "brier 1e-6 = score 0.4003; a 6-decimal Brier carries "
                           "+-0.2002 in the converted number",
            "caveat": "local measurement is at iter 650 (<=2023 training); the "
                      "deployed build is iter 7700 (<=2024).",
        },
        "leak_boundary": {
            "tables": "row of season s uses seasons <= s-1 only; verified twice — "
                      "at runtime, and independently by re-reading train.csv and "
                      "looking the contributing row_ids' seasons up again",
            "beta_fit": "supply blocks 2022+2023 of the same seed only",
            "arm_choice": "fixed by pre-registration before any 2024 number was "
                          "read; the per-seed block argmin is diagnostic",
            "design_matrix": "unchanged — no deviation is ever a model feature",
            "missing": "no sentinel fill anywhere",
        },
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__},
        "compute_time_min": round((time.time() - t_start) / 60, 2),
    }
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(json.loads(json.dumps(metrics, default=str)), f, indent=2,
                  ensure_ascii=False)

    print("\n--- ARM x SEED (fold_2024 valid_all) ---", flush=True)
    print(f"{'seed':>6}{'A_brier':>16}{'A_score':>11}{'B_brier':>16}"
          f"{'B_score':>11}{'delta':>13}{'d_score':>11}", flush=True)
    for s in SEEDS:
        v = per_seed[str(s)]
        print(f"{s:>6}{v['A_control']['brier']:>16.12f}"
              f"{v['A_control']['score']:>11.3f}"
              f"{v['B_candidate']['brier']:>16.12f}"
              f"{v['B_candidate']['score']:>11.3f}"
              f"{v['paired_delta_B_minus_A']:>13.4e}"
              f"{v['paired_delta_score']:>11.3f}", flush=True)
    print(f"\nmean delta {mean_d:.6e} | control seed std {ctrl_std:.6e} | "
          f"ratio {decision['abs_mean_delta_over_control_std']} | "
          f"paired-delta std {delta_std:.6e} (reference) | "
          f"MEASURED = {decision['MEASURED']}", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] DONE {metrics['compute_time_min']} min",
          flush=True)


if __name__ == "__main__":
    main()
