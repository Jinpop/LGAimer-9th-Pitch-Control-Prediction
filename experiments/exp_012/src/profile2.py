"""exp_012 — pitcher COMMAND profile from trackman_history (as-of aggregation).

Spec: experiments/exp_012/config.yaml (2026-08-05, campaign 4).

WHY THIS FILE EXISTS
--------------------
The label `control_success` is "actual location vs the location the catcher
asked for"; failure = (1) middle of the zone, (2) way off the zone, (3) the
opposite side from the target.  exp_003's TrackMan profile measured *stuff*
(velocity / spin / movement means) and earned only 1.5% of LGBM gain.  Nothing
we have built so far measures **command** — the pitcher's ability to put the
ball where he intends.

trackman_history.csv has NO plate-location column (30 columns, verified: no
plate_x / plate_z / zone / location of any kind).  So the config's literal ask
("plate location std, zone-middle rate, zone-miss rate") is NOT computable.
Two substitutes are used instead, in order of directness:

  1. DERIVED PITCH OUTCOME (ball vs strike).  trackman does carry
     balls_before / strikes_before / pitch_of_pa, so the outcome of every
     NON-TERMINAL pitch of a plate appearance is recoverable exactly from the
     count transition to the next pitch of the same PA:
         balls +1, strikes +0  -> BALL     (missed the zone)
         balls +0, strikes +1  -> STRIKE
         balls +0, strikes +0  -> foul with 2 strikes -> not a ball
     A ball IS "way off / out of the zone", i.e. failure mode (2) of the label,
     observed directly.  Ball rate conditioned on count / pitch type / batter
     hand, and its game-to-game dispersion, is the closest thing to a command
     meter this dataset allows.
     The terminal pitch of each PA has no next-pitch, so its outcome is
     unknown and it is excluded (~25% of pitches).

  2. PHYSICS DISPERSION, computed WITHIN pitch-type group.  exp_003 already had
     career_rel_height_std etc., but pooled over all pitch types, so those
     numbers mostly encode the pitch MIX (a guy with a big curve has huge
     pooled IVB std).  Here every std is computed per (pitcher, pitch_type_
     group) and then pitch-count-weighted, which isolates true repeatability:
     release-point repeatability and movement consistency.

LEAK RULES OBSERVED
-------------------
* Aggregation only from trackman rows with `season <= cutoff`; for a fold that
  validates year Y the cutoff is Y-1, so validation rows never contribute.
  (Same as exp_003; asserted at runtime.)
* No current-row TrackMan quantity: train/test rows carry no TrackMan columns
  at all, the join key is pitcher_id only.
* The static table shipped for inference uses cutoff = 2024 (trackman ends in
  2024, asserted), so the 2025 test set is reproducible from history alone.
* Nothing is read from test.csv and no cross-row test statistic is used
  (competition rule 2-4).
* Deterministic: aggregation + sorting only, no randomness.

MISSINGNESS
-----------
Every conditional rate has a minimum-denominator guard and becomes NaN below
it.  NaN is never filled here (project rule: no sentinel).  The consumer adds
an explicit `has_profile2` flag before any imputation.
"""

import numpy as np
import pandas as pd

# columns actually needed out of the 30 in trackman_history.csv
TM_USECOLS = [
    "season", "trackman_game_id", "pitch_no", "inning", "top_bottom",
    "balls_before", "strikes_before", "pitch_of_pa",
    "pitcher_trackman_id", "batter_trackman_id", "batter_hand",
    "pitch_type_group",
    "rel_speed", "spin_rate", "induced_vert_break", "horz_break",
    "extension", "rel_height", "rel_side",
]

PGROUPS = ("fastball", "breaking", "offspeed")

# minimum denominators (below -> NaN, never a sentinel)
MIN_N_OVERALL = 100     # a pitcher needs this many graded pitches at all
MIN_N_COND = 30         # conditional split (count state / hand / pitch type)
MIN_N_PERGAME = 10      # pitches in a game for that game's rate to count
MIN_GAMES = 10          # games for a game-to-game dispersion figure
MIN_N_STD = 25          # pitches in a (pitcher, pitch type) cell for a std

KEY = "pitcher_trackman_id"


# --------------------------------------------------------------------------- #
# step 1: recover per-pitch ball/strike outcome from the count transition
# --------------------------------------------------------------------------- #
def derive_outcomes(tm):
    """Add `is_ball` (float 0/1, NaN where ungradable) to a trackman frame.

    A pitch is gradable iff the immediately following row is the next pitch of
    the SAME plate appearance (same game, pitcher, batter, pitch_of_pa + 1).
    Returns (frame_with_is_ball, diagnostics_dict).
    """
    tm = tm.sort_values(["trackman_game_id", "pitch_no"], kind="mergesort").reset_index(drop=True)

    nxt_game = tm["trackman_game_id"].shift(-1)
    nxt_pit = tm[KEY].shift(-1)
    nxt_bat = tm["batter_trackman_id"].shift(-1)
    nxt_pop = tm["pitch_of_pa"].shift(-1)
    nxt_b = tm["balls_before"].shift(-1)
    nxt_s = tm["strikes_before"].shift(-1)

    # inning / top_bottom are part of the identity so a batter facing the same
    # pitcher twice in a game can never produce a false adjacency.
    same_pa = (
        (nxt_game == tm["trackman_game_id"])
        & (nxt_pit == tm[KEY])
        & (nxt_bat == tm["batter_trackman_id"])
        & (tm["inning"].shift(-1) == tm["inning"])
        & (tm["top_bottom"].shift(-1) == tm["top_bottom"])
        & (nxt_pop == tm["pitch_of_pa"] + 1)
    )
    db = nxt_b - tm["balls_before"]
    ds = nxt_s - tm["strikes_before"]

    is_ball_t = same_pa & (db == 1) & (ds == 0)
    is_strk_t = same_pa & (db == 0) & (ds == 1)
    # a frozen count is a foul, which is only legal at 2 strikes; without that
    # guard a corrupt transition would silently be graded as a strike.
    is_foul_t = same_pa & (db == 0) & (ds == 0) & (tm["strikes_before"] == 2)
    gradable = is_ball_t | is_strk_t | is_foul_t

    tm["is_ball"] = np.where(gradable, is_ball_t.astype(float), np.nan)

    anomalous = int((same_pa & ~gradable).sum())
    diag = {
        "n_rows": int(len(tm)),
        "n_same_pa_pairs": int(same_pa.sum()),
        "n_gradable": int(gradable.sum()),
        "gradable_share": round(float(gradable.mean()), 4),
        "n_ball": int(is_ball_t.sum()),
        "n_strike": int(is_strk_t.sum()),
        "n_foul_2strike": int(is_foul_t.sum()),
        "n_anomalous_transition_dropped": anomalous,
        "overall_ball_rate": round(float(is_ball_t.sum() / max(gradable.sum(), 1)), 6),
    }
    return tm, diag


# --------------------------------------------------------------------------- #
# small aggregation helpers (all min-count guarded -> NaN)
# --------------------------------------------------------------------------- #
def _rate(df, mask, min_n):
    """Mean of is_ball over `mask`, per pitcher, NaN below min_n."""
    sub = df.loc[mask, [KEY, "is_ball"]]
    g = sub.groupby(KEY)["is_ball"]
    r = g.mean()
    n = g.size()
    return r.where(n >= min_n)


def _wstd(df, col, min_n=MIN_N_STD):
    """Pitch-count-weighted mean of the WITHIN-pitch-type std of `col`.

    Removes pitch-mix contamination: a pitcher who throws 4 pitch types no
    longer looks 'inconsistent' just because his pitch types differ.
    """
    g = df.groupby([KEY, "pitch_type_group"])[col]
    s = g.std()
    n = g.size()
    ok = n >= min_n
    s, n = s[ok], n[ok]
    valid = s.notna()
    s, n = s[valid], n[valid]
    num = (s * n).groupby(level=0).sum()
    den = n.groupby(level=0).sum()
    return (num / den).replace([np.inf, -np.inf], np.nan)


def _grp_stat(df, col, pg, how, min_n=MIN_N_STD):
    """std / iqr of `col` for one pitch-type group, per pitcher."""
    sub = df[df["pitch_type_group"] == pg]
    g = sub.groupby(KEY)[col]
    n = g.size()
    if how == "std":
        v = g.std()
    elif how == "iqr":
        v = g.quantile(0.75) - g.quantile(0.25)
    else:
        raise ValueError(how)
    return v.where(n >= min_n)


def _cond_std(df, col, mask, min_n=MIN_N_COND):
    g = df.loc[mask, [KEY, col]].groupby(KEY)[col]
    return g.std().where(g.size() >= min_n)


# --------------------------------------------------------------------------- #
# step 2: the profile itself
# --------------------------------------------------------------------------- #
def build_profile2(tm_all, cutoff):
    """Per-pitcher command + consistency profile from seasons <= cutoff.

    `tm_all` must already carry `is_ball` (see derive_outcomes).
    Index: pitcher_trackman_id.  All values NaN-able, nothing imputed.
    """
    df = tm_all[tm_all["season"] <= cutoff]
    assert len(df) > 0 and int(df["season"].max()) <= cutoff, "as-of cutoff violated"

    gr = df[df["is_ball"].notna()]          # gradable pitches only
    idx = pd.Index(df[KEY].unique(), name=KEY).sort_values()
    out = pd.DataFrame(index=idx)

    b = gr["balls_before"]
    s = gr["strikes_before"]
    ahead = (s > b).to_numpy()
    behind = (b > s).to_numpy()
    even = (b == s).to_numpy()

    # ---------------- command: overall & conditional ball rate --------------
    gn = gr.groupby(KEY)["is_ball"]
    out["cmd_n"] = gn.size().reindex(idx)
    out["cmd_ball_rate"] = gn.mean().reindex(idx).where(out["cmd_n"] >= MIN_N_OVERALL)

    out["cmd_ball_rate_ahead"] = _rate(gr, ahead, MIN_N_COND).reindex(idx)
    out["cmd_ball_rate_behind"] = _rate(gr, behind, MIN_N_COND).reindex(idx)
    out["cmd_ball_rate_even"] = _rate(gr, even, MIN_N_COND).reindex(idx)
    # command collapse under pressure: behind in the count vs ahead
    out["cmd_pressure_delta"] = out["cmd_ball_rate_behind"] - out["cmd_ball_rate_ahead"]

    out["cmd_ball_rate_3b"] = _rate(gr, (b == 3).to_numpy(), MIN_N_COND).reindex(idx)
    out["cmd_ball_rate_2s"] = _rate(gr, (s == 2).to_numpy(), MIN_N_COND).reindex(idx)
    # first-pitch strike rate — the classic command/aggression marker
    fp = (gr["pitch_of_pa"] == 1).to_numpy()
    out["cmd_f1s_rate"] = 1.0 - _rate(gr, fp, MIN_N_COND).reindex(idx)

    # ---------------- command: platoon split --------------------------------
    out["cmd_ball_rate_vL"] = _rate(gr, (gr["batter_hand"] == "Left").to_numpy(), MIN_N_COND).reindex(idx)
    out["cmd_ball_rate_vR"] = _rate(gr, (gr["batter_hand"] == "Right").to_numpy(), MIN_N_COND).reindex(idx)
    out["cmd_platoon_delta"] = out["cmd_ball_rate_vL"] - out["cmd_ball_rate_vR"]

    # ---------------- command: per pitch type -------------------------------
    for pg in PGROUPS:
        out[f"cmd_ball_rate_{pg}"] = _rate(
            gr, (gr["pitch_type_group"] == pg).to_numpy(), MIN_N_COND).reindex(idx)
    pgcols = [f"cmd_ball_rate_{pg}" for pg in PGROUPS]
    # how uneven is his command across his arsenal
    out["cmd_ball_rate_spread_pg"] = out[pgcols].max(axis=1) - out[pgcols].min(axis=1)
    out["cmd_ball_rate_offspeed_gap"] = out["cmd_ball_rate_offspeed"] - out["cmd_ball_rate_fastball"]
    out["cmd_ball_rate_breaking_gap"] = out["cmd_ball_rate_breaking"] - out["cmd_ball_rate_fastball"]

    # ---------------- command: game-to-game volatility ----------------------
    # the "quantile width" the config asked for, at the game level: a pitcher
    # whose per-game ball rate swings wildly has volatile command.
    pg_ = gr.groupby([KEY, "trackman_game_id"])["is_ball"]
    grate = pg_.mean()
    gcnt = pg_.size()
    grate = grate[gcnt >= MIN_N_PERGAME]
    byp = grate.groupby(level=0)
    ngames = byp.size()
    keep = ngames >= MIN_GAMES
    out["cmd_n_games_graded"] = ngames.reindex(idx)
    out["cmd_ball_rate_game_std"] = byp.std().where(keep).reindex(idx)
    q10 = byp.quantile(0.10).where(keep).reindex(idx)
    q90 = byp.quantile(0.90).where(keep).reindex(idx)
    q25 = byp.quantile(0.25).where(keep).reindex(idx)
    q75 = byp.quantile(0.75).where(keep).reindex(idx)
    out["cmd_ball_rate_game_p10_p90"] = q90 - q10
    out["cmd_ball_rate_game_iqr"] = q75 - q25
    out["cmd_ball_rate_game_worst"] = q90            # his bad-day command level
    out["cmd_ball_rate_game_best"] = q10

    # ---------------- command: PA-level shape -------------------------------
    pa_keys = [KEY, "trackman_game_id", "inning", "top_bottom", "batter_trackman_id"]
    pa = df.groupby(pa_keys).agg(
        n_pitch=("pitch_of_pa", "size"),
        max_b=("balls_before", "max"),
        max_s=("strikes_before", "max"),
    )
    pab = pa.groupby(level=0)
    n_pa = pab.size()
    ok_pa = n_pa >= 30
    out["cmd_n_pa"] = n_pa.reindex(idx)
    out["cmd_pitches_per_pa"] = pab["n_pitch"].mean().where(ok_pa).reindex(idx)
    out["cmd_3ball_pa_rate"] = pa["max_b"].ge(3).groupby(level=0).mean().where(ok_pa).reindex(idx)
    out["cmd_2strike_pa_rate"] = pa["max_s"].ge(2).groupby(level=0).mean().where(ok_pa).reindex(idx)

    # ---------------- command: recency --------------------------------------
    rec = gr[gr["season"] == cutoff]
    rg = rec.groupby(KEY)["is_ball"]
    out["cmd_recent_n"] = rg.size().reindex(idx)
    out["cmd_recent_ball_rate"] = rg.mean().reindex(idx).where(out["cmd_recent_n"] >= MIN_N_COND)
    out["cmd_ball_rate_trend"] = out["cmd_recent_ball_rate"] - out["cmd_ball_rate"]

    # ---------------- consistency: within-pitch-type dispersion -------------
    for col, name in [
        ("induced_vert_break", "ivb"), ("horz_break", "hb"),
        ("rel_speed", "relspeed"), ("spin_rate", "spin"),
        ("rel_height", "relheight"), ("rel_side", "relside"),
        ("extension", "ext"),
    ]:
        out[f"cons_{name}_std_w"] = _wstd(df, col).reindex(idx)

    out["cons_release_disp_w"] = np.sqrt(
        out["cons_relheight_std_w"] ** 2 + out["cons_relside_std_w"] ** 2)
    out["cons_move_disp_w"] = np.sqrt(
        out["cons_ivb_std_w"] ** 2 + out["cons_hb_std_w"] ** 2)

    # per pitch-type detail (config: "구종별 무브먼트 일관성")
    for pg in PGROUPS:
        out[f"cons_{pg}_ivb_std"] = _grp_stat(df, "induced_vert_break", pg, "std").reindex(idx)
        out[f"cons_{pg}_hb_std"] = _grp_stat(df, "horz_break", pg, "std").reindex(idx)
        out[f"cons_{pg}_relspeed_std"] = _grp_stat(df, "rel_speed", pg, "std").reindex(idx)
    # robust (outlier-insensitive) versions on the fastball
    out["cons_fastball_ivb_iqr"] = _grp_stat(df, "induced_vert_break", "fastball", "iqr").reindex(idx)
    out["cons_fastball_hb_iqr"] = _grp_stat(df, "horz_break", "fastball", "iqr").reindex(idx)
    out["cons_fastball_relside_iqr"] = _grp_stat(df, "rel_side", "fastball", "iqr").reindex(idx)

    # ---------------- consistency under pressure / by platoon ---------------
    fb = df[df["pitch_type_group"] == "fastball"]
    fb_b = fb["balls_before"]
    fb_s = fb["strikes_before"]
    m_ahead = (fb_s > fb_b).to_numpy()
    m_behind = (fb_b > fb_s).to_numpy()
    for col, name in [("rel_speed", "relspeed"), ("rel_height", "relheight"),
                      ("rel_side", "relside")]:
        a = _cond_std(fb, col, m_ahead).reindex(idx)
        h = _cond_std(fb, col, m_behind).reindex(idx)
        out[f"cons_fb_{name}_std_pressure_delta"] = h - a

    mL = (fb["batter_hand"] == "Left").to_numpy()
    mR = (fb["batter_hand"] == "Right").to_numpy()
    relside_L = fb.loc[mL].groupby(KEY)["rel_side"].mean()
    relside_R = fb.loc[mR].groupby(KEY)["rel_side"].mean()
    out["cons_fb_relside_platoon_shift"] = (relside_L - relside_R).reindex(idx)
    out["cons_fb_relside_std_vL"] = _cond_std(fb, "rel_side", mL).reindex(idx)
    out["cons_fb_relside_std_vR"] = _cond_std(fb, "rel_side", mR).reindex(idx)

    # recency of consistency
    out["cons_recent_relspeed_std_w"] = _wstd(df[df["season"] == cutoff], "rel_speed").reindex(idx)
    out["cons_relspeed_std_trend"] = out["cons_recent_relspeed_std_w"] - out["cons_relspeed_std_w"]

    out.index.name = KEY
    return out


def to_pitcher_id(prof, pmap):
    """Re-key a profile from pitcher_trackman_id to the train-side pitcher_id."""
    p = prof.reset_index()
    p = pmap[["pitcher_id", KEY]].merge(p, on=KEY, how="inner")
    assert p["pitcher_id"].is_unique, "pitcher_id map is not 1:1"
    return p.drop(columns=[KEY]).set_index("pitcher_id")


# feature families, declared A PRIORI (before seeing any validation number)
def feature_families(cols):
    cmd = [c for c in cols if c.startswith("cmd_")]
    cons = [c for c in cols if c.startswith("cons_")]
    return {"command": cmd, "consistency": cons, "all": cmd + cons}


# Compact set, hand-picked from domain reasoning BEFORE the run, to test the
# feature-dilution hypothesis (RF's max_features='sqrt' samples fewer of the 47
# base columns per split once ~50 weak columns are appended).
COMPACT = [
    "cmd_ball_rate",
    "cmd_pressure_delta",
    "cmd_ball_rate_game_std",
    "cmd_f1s_rate",
    "cmd_platoon_delta",
    "cmd_pitches_per_pa",
    "cmd_3ball_pa_rate",
    "cmd_ball_rate_trend",
    "cmd_ball_rate_spread_pg",
    "cons_release_disp_w",
    "cons_move_disp_w",
    "cons_relspeed_std_w",
]
