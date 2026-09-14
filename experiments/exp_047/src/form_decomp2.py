"""exp_047 — generalisation of exp_046's in-season form decomposition.

WHAT IS INHERITED, AND HOW
--------------------------
`experiments/exp_046/src/form_decomp.py` (sha256 1a3b3212…b4a6, audited in
exp_046/leak_audit.json → PASS_WITH_NOTES) is IMPORTED, not retyped, and its
sha is asserted before anything is read out of it.  Its two row-wise pieces —
`career_successes()` and `apply_decomposition()` — are reused VERBATIM for all
nine new axes.  Nothing in this file changes their behaviour.

WHAT IS NEW
-----------
  1. `build_pre_boundary_tables()` — a TRAIN-ONLY frozen-table builder for the
     nine cumulative rate columns that have NO per-row label.
  2. `build_traj_table()` / `apply_traj()` — the trajectory/deviation split of
     the success axis (config axis B).
  3. `map_ids_rowwise()` — id → table slot without `pd.factorize`, for the
     scoring path.

======================================================================
!!  WHY A *PRE*-BOUNDARY TABLE FOR THE NINE NEW AXES  !!
======================================================================
exp_046 could freeze the success table exactly AT the season boundary:

    n_end[p] = asof_pitcher_n(last <=S row of p) + 1
    S_end[p] = S_career(last <=S row)  +  control_success(last <=S row)

because `control_success` — the boundary row's OWN event — is a train label.

The nine new columns have no per-row label.  Their per-row event is only
recoverable by differencing with the entity's NEXT row (labels.py), and for the
LAST row of season S that successor lives in season S+1.  For the fold_2024
table that successor is a VALIDATION row, and for the deployment table
(boundary 2024) it is a TEST row — which does not exist and could not be read
if it did (rule 2-4).

So these axes freeze one row EARLIER instead:

    n_end[e] = n(last <=S row of e)          # no +1
    S_end[e] = round(n*rate)(last <=S row)   # no own event

Both halves shift together, so the decomposition stays EXACT arithmetic:
0 <= S_season <= n_season still holds identically (gate G1).  The only cost is
that "this season" starts one pitch early — the entity's final pitch of the
previous season is counted into the current season's window.  That is one pitch
out of a season that has hundreds (fold_2024 median measured in stage 1), it is
a definition rather than an approximation, and it involves NO imputation and NO
sentinel.  It is measured, not assumed: gate G2b runs the SUCCESS column
through this same pre-boundary path and checks it against the ground-truth
label cumulation shifted by exactly that one row.

======================================================================
!!  RULE 2-4 BOUNDARY  --  READ BEFORE TOUCHING ANYTHING HERE  !!
======================================================================
Two kinds of function live in this file and they must never merge:

  BUILDERS  `build_pre_boundary_tables`, `build_traj_table`
      may use groupby / sort / unique.  They are called ONLY on train.csv.
      Calling either on test/valid data is a rule 2-4 violation: differencing
      these same cumulative columns between adjacent rows of the scoring frame
      reconstructs the per-row label (measured on train at 99.846% of rows,
      100.000% accuracy — this is not hypothetical, and this experiment applies
      the same arithmetic to nine more columns).

  APPLIERS  `apply_traj`, `map_ids_rowwise`, and exp_046's
            `apply_decomposition` (re-exported unchanged)
      read ONLY the current row's own values plus frozen train-derived arrays.
      They contain NO groupby, NO sort, NO shift, NO cumsum, NO factorize, NO
      np.unique.  `src/ast_gate.py` proves that mechanically from the parsed
      syntax tree (not by regex — a regex matched a docstring and produced a
      false verdict in an exp_046 follow-up build).
======================================================================
"""

import hashlib
import importlib.util
import os

import numpy as np
import pandas as pd

EXP046_FORM_DECOMP = os.path.expanduser(
    "~/SungjinPark/experiments/exp_046/src/form_decomp.py")
EXP046_SHA = "1a3b321206a0939fecf0a3f0889acd8c2f6e33acfe4bc09b2ee99c2ddb4bb4a6"


def sha256_file(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _load_exp046():
    got = sha256_file(EXP046_FORM_DECOMP)
    assert got == EXP046_SHA, ("exp_046 form_decomp.py changed", got, EXP046_SHA)
    spec = importlib.util.spec_from_file_location("exp046_form_decomp",
                                                  EXP046_FORM_DECOMP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fd = _load_exp046()
career_successes = fd.career_successes          # verbatim, sha-gated
apply_decomposition = fd.apply_decomposition    # verbatim, sha-gated
build_boundary_tables = fd.build_boundary_tables
extend_table_for_unseen = fd.extend_table_for_unseen

# --------------------------------------------------------------------------- #
#  The nine new axes.  `related` is the PRE-REGISTERED set of existing base
#  columns used for the incremental-R^2 test (config stage1_diagnostic).
#  Rule: same-quantity career column + its prevN siblings if the organisers
#  provide any + the entity's own count column.
# --------------------------------------------------------------------------- #
PITCH_N = "asof_pitcher_n"
BAT_N = "asof_batter_n"
MIX3 = ["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate"]

AXES = {
    "rev":  {"entity": "pitcher_id", "n": PITCH_N,
             "rate": "asof_pitcher_reverse_rate", "group": "pitcher_outcome",
             "related": ["asof_pitcher_reverse_rate", PITCH_N]},
    "mid":  {"entity": "pitcher_id", "n": PITCH_N,
             "rate": "asof_pitcher_middle_rate", "group": "pitcher_outcome",
             "related": ["asof_pitcher_middle_rate",
                         "asof_pitcher_prev1_game_middle_rate",
                         "asof_pitcher_prev3_game_middle_rate",
                         "asof_pitcher_prev5_game_middle_rate", PITCH_N]},
    "ball": {"entity": "pitcher_id", "n": PITCH_N,
             "rate": "asof_pitcher_ball_rate", "group": "pitcher_outcome",
             "related": ["asof_pitcher_ball_rate", PITCH_N]},
    "strk": {"entity": "pitcher_id", "n": PITCH_N,
             "rate": "asof_pitcher_strike_rate", "group": "pitcher_outcome",
             "related": ["asof_pitcher_strike_rate", PITCH_N]},
    "fb":   {"entity": "pitcher_id", "n": PITCH_N,
             "rate": "asof_pitcher_fastball_rate", "group": "pitchmix",
             "related": MIX3 + [PITCH_N]},
    "brk":  {"entity": "pitcher_id", "n": PITCH_N,
             "rate": "asof_pitcher_breaking_rate", "group": "pitchmix",
             "related": MIX3 + [PITCH_N]},
    "ofs":  {"entity": "pitcher_id", "n": PITCH_N,
             "rate": "asof_pitcher_offspeed_rate", "group": "pitchmix",
             "related": MIX3 + [PITCH_N]},
    "bsuc": {"entity": "batter_id", "n": BAT_N,
             "rate": "asof_batter_success_rate", "group": "batter",
             "related": ["asof_batter_success_rate", BAT_N]},
    "bmid": {"entity": "batter_id", "n": BAT_N,
             "rate": "asof_batter_middle_rate", "group": "batter",
             "related": ["asof_batter_middle_rate", BAT_N]},
}

# feature-column name emitted per axis (exactly ONE column per axis: the shared
# n_season / has_form information is already carried by exp_046's four columns
# for the pitcher entity, and by `has_form_bat` for the batter entity)
def axis_col(tag):
    return f"d_{tag}_eb"


# =========================================================================== #
#  BUILDERS — TRAIN ONLY.  groupby / sort / unique are legal here and nowhere
#  else.  See the rule 2-4 block in the module docstring.
# =========================================================================== #
def build_pre_boundary_tables(season, ecode, n, rate, n_ent):
    """TRAIN ONLY.  Frozen per-entity state one row BEFORE each season boundary.

    Returns {boundary_season: (n_end_array, S_end_array)} indexed by `ecode`,
    carried forward across seasons the entity missed.  An entity never seen up
    to that boundary keeps (0, 0) — the correct arithmetic state of somebody
    with no recorded history, not an imputation.
    """
    n = np.asarray(n, dtype=np.float64)
    cnt, _, _ = career_successes(n, rate)
    last = (pd.DataFrame({"e": ecode, "s": season})
              .groupby(["e", "s"], sort=False).tail(1).index.to_numpy())
    cur_n = np.zeros(n_ent, dtype=np.float64)
    cur_s = np.zeros(n_ent, dtype=np.float64)
    tables = {}
    last_season = np.asarray(season)[last]
    for s in np.sort(np.unique(season)):
        idx = last[last_season == s]
        cur_n = cur_n.copy()
        cur_s = cur_s.copy()
        cur_n[ecode[idx]] = n[idx]        # PRE-boundary: no +1
        cur_s[ecode[idx]] = cnt[idx]      # PRE-boundary: no own event
        tables[int(s)] = (cur_n, cur_s)
    return tables


def build_traj_table(season, pcode, n_ent, n_end, s_end, n_season, s_season, y):
    """TRAIN ONLY.  Per-pitcher career trajectory of the SUCCESS d_form.

    `n_end`/`s_end`/`n_season`/`s_season` are exp_046's row-level decomposition
    arrays (the +1 / own-label convention), `y` is control_success.  For each
    (pitcher, season) the season's FINAL state is
        n_final = n_season(last row) + 1 ,  S_final = S_season(last row) + y
    which is exactly the quantity exp_046's stage-1 true-SD block used.

    traj[p] at boundary B = n_final-weighted mean of d_season over the pitcher's
    seasons <= B that have a d_season at all (i.e. the pitcher's 2nd season
    onwards).  Returns {boundary: (traj_raw, N_traj)}; N_traj == 0 means the
    pitcher has fewer than two seasons of history, which the `has_traj` flag
    carries.  No sentinel is written anywhere.
    """
    season = np.asarray(season)
    last = (pd.DataFrame({"p": pcode, "s": season})
              .groupby(["p", "s"], sort=False).tail(1).index.to_numpy())
    n_fin = n_season[last] + 1.0
    s_fin = s_season[last] + y[last]
    n_pri = n_end[last]
    s_pri = s_end[last]
    ok = (n_pri > 0) & (n_fin > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        d_season = np.where(ok, s_fin / np.where(n_fin > 0, n_fin, 1.0)
                            - s_pri / np.where(n_pri > 0, n_pri, 1.0), 0.0)
    w = np.where(ok, n_fin, 0.0)
    lp = pcode[last]
    ls = season[last]

    acc_w = np.zeros(n_ent, dtype=np.float64)
    acc_wd = np.zeros(n_ent, dtype=np.float64)
    tables = {}
    for s in np.sort(np.unique(season)):
        m = ls == s
        np.add.at(acc_w, lp[m], w[m])
        np.add.at(acc_wd, lp[m], w[m] * d_season[m])
        with np.errstate(invalid="ignore", divide="ignore"):
            traj = np.where(acc_w > 0, acc_wd / np.where(acc_w > 0, acc_w, 1.0),
                            np.nan)
        tables[int(s)] = (traj.copy(), acc_w.copy())
    return tables


def id_map(unique_ids):
    """TRAIN ONLY.  id -> table slot; slot len(unique_ids) is 'never seen'."""
    return {v: i for i, v in enumerate(unique_ids)}, len(unique_ids)


# =========================================================================== #
#  APPLIERS — ROW-WISE ONLY.  No groupby / sort / shift / cumsum / factorize /
#  np.unique.  Mechanically checked by src/ast_gate.py.
# =========================================================================== #
def map_ids_rowwise(ids, mapping, unseen_ix):
    """Row-wise id lookup.  Each element is resolved independently against a
    frozen train-built dict; an id absent from train gets the 'unseen' slot,
    whose frozen state is (0, 0) — the same state a train rookie already has.
    """
    return np.array([mapping.get(v, unseen_ix) for v in ids], dtype=np.int64)


def apply_traj(pcode, traj_tab, ntraj_tab, d_form_row, n_season_row,
               k_traj, k_dev):
    """ROW-WISE.  Trajectory / deviation split of the success axis.

    `d_form_row` and `n_season_row` are THIS row's own values from exp_046's
    `apply_decomposition`; `traj_tab`/`ntraj_tab` are frozen train arrays.
    Undefined entries stay NaN and `has_traj` carries the defined-ness.
    """
    traj_raw = traj_tab[pcode]
    n_tr = ntraj_tab[pcode]
    has_traj = n_tr > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        traj_eb = np.where(has_traj,
                           np.where(has_traj, traj_raw, 0.0) * n_tr
                           / (n_tr + float(k_traj)), np.nan)
        dev = np.where(has_traj, d_form_row - np.where(has_traj, traj_raw, 0.0),
                       np.nan)
        shr = np.asarray(n_season_row, dtype=np.float64)
        dev_eb = np.where(has_traj & np.isfinite(d_form_row),
                          dev * shr / (shr + float(k_dev)), np.nan)
    return {"traj_eb": traj_eb, "dev_eb": dev_eb,
            "has_traj": has_traj.astype(np.int8)}
