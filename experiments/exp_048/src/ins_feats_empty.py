"""exp_048 ROBUSTNESS VARIANT (A1e) — earliest season gets the EMPTY anchor.

Identical to ins_feats.py except that season 2019 is handed build_anchor() of an
empty frame, which is what the teammate module does on its own: every pitcher
is unknown, so anchor = 0 and the league prior evaluates to 0.0.  This arm
exists to measure what the NaN choice in ins_feats.py was worth.

"""

_ORIG_DOC = """exp_048 — the ONLY place the teammate's `ins/` pack is turned into columns.

Season policy (this is a VARIABLE of the experiment, recorded, not hidden)
-------------------------------------------------------------------------
`ins_*` is "career cumulative MINUS the pitcher's state at the end of the
previous season".  That subtraction needs a previous season.

  season 2024 (the fold's VALIDATION rows)
      -> `anchor_lt2024.npz`, i.e. build_anchor(train[season < 2024]).
         Using `deploy_anchor.npz` here would subtract the END-OF-2024 state
         from a 2024 row, making n - anchor negative and flattening it to 0.
  seasons 2020..2023 (training rows)
      -> build_anchor(train[season < Y]) for that Y, exactly the loop the
         pack's README prescribes.
  season 2019 (the EARLIEST season, 211,627 rows = 19% of the fold's train)
      -> all 16 columns are NaN.
         There is no prior season, so the anchor would be empty; the module
         then falls back to prior = league rate computed over an empty table,
         which evaluates to 0.0.  That is not "no information", it is a WRONG
         constant: the ten `_dev` columns would sit near +0.5 for 2019 and
         near 0.0 for every other season, i.e. a 19%-of-train scale break.
         Filling with a sentinel is forbidden (CLAUDE.md), so the honest
         encoding is missing.  CatBoost splits NaN natively and the 2024
         validation rows contain none.
"""

import os

import numpy as np
import pandas as pd

COLS16 = ["ins_p_n",
          "ins_p_success", "ins_p_success_dev",
          "ins_p_reverse", "ins_p_reverse_dev",
          "ins_p_middle", "ins_p_middle_dev",
          "ins_p_ball", "ins_p_ball_dev",
          "ins_p_strike", "ins_p_strike_dev",
          "ins_b_n",
          "ins_b_success", "ins_b_success_dev",
          "ins_b_middle", "ins_b_middle_dev"]

EARLIEST_SEASON_IS_NAN = False


def build_ins_matrix(df, ins, pack_dir, valid_season=2024, log=print):
    """Return (values (len(df),16) float64, cols, info).

    `ins` is the imported teammate module.  The frozen `anchor_lt2024.npz` is
    used for `valid_season` and is ASSERTED to equal build_anchor(train[season
    < valid_season]) rebuilt here from train.csv.
    """
    season = df["season"].to_numpy()
    seasons = sorted({int(s) for s in season.tolist()})
    first = seasons[0]
    V = np.full((len(df), 16), np.nan, dtype=np.float64)
    info = {"season_policy": {}, "cols": list(COLS16), "first_season": first,
            "valid_season": int(valid_season)}

    A_frozen = ins.load(os.path.join(pack_dir, "anchor_lt2024.npz"))
    A_reb = ins.build_anchor(df.loc[season < valid_season])
    same = all(np.array_equal(np.asarray(A_frozen[k]), np.asarray(A_reb[k]))
               for k in sorted(A_frozen))
    assert sorted(A_frozen) == sorted(A_reb), (sorted(A_frozen), sorted(A_reb))
    assert same, ("anchor_lt2024.npz is NOT build_anchor(train[season<%d])"
                  % valid_season)
    info["frozen_anchor_equals_rebuild"] = True
    info["frozen_anchor_n_pitchers"] = int(len(A_frozen["p_ids"]))

    for s in seasons:
        m = season == s
        A = A_frozen if s == valid_season else ins.build_anchor(
            df.loc[season < s])
        I = ins.apply(df.loc[m], A)
        assert list(I.columns) == COLS16, list(I.columns)
        V[m] = I.to_numpy(np.float64)
        info["season_policy"][str(s)] = {
            "anchor": ("anchor_lt2024.npz (frozen, verified)"
                       if s == valid_season
                       else f"build_anchor(train[season < {s}])"),
            "n_rows": int(m.sum()),
            "n_anchor_pitchers": int(len(A["p_ids"])),
            "ins_p_n_mean": round(float(np.mean(V[m, 0])), 3),
            "ins_p_n_zero_pct": round(100.0 * float((V[m, 0] == 0).mean()), 4),
        }
        log(f"    ins season {s}: {int(m.sum()):,} rows, anchor "
            f"{info['season_policy'][str(s)]['anchor']}, mean ins_p_n "
            f"{info['season_policy'][str(s)]['ins_p_n_mean']}")

    info["nan_row_pct_overall"] = round(
        100.0 * float(np.isnan(V[:, 0]).mean()), 4)
    return V, list(COLS16), info
