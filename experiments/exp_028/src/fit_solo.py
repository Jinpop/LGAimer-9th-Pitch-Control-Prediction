"""exp_028 — closing two open axes: `d_r` as a SOLO term, and the logit
coordinate system.  NO TRAINING.  Every base vector is reused from
exp_020 / exp_022 / exp_024 with its sha256 asserted against the digest exp_024
recorded; a drift aborts the run.

CARD 1 — `d_r` ALONE
--------------------
exp_026 rejected `p + beta_s*d_s + beta_r*d_r`.  What it actually measured was
the TWO-TERM fit, in which beta_r flipped from -0.367 (alone) to +0.327 (joint)
because the two axes correlate at -0.83.  That pathology belongs to the joint
fit; a one-term arm cannot have it.  `p + beta_r*d_r` has never been measured.

  A0  no correction                       (floor; cross-checked against exp_024)
  A1  p + beta_s * d_s , k=400 x league   (CONTROL = exp_024 arm B = candidate J)
  A2  p + beta_r * d_r , k=400 x league   (MAIN — the SAME shrink constant and
                                           the SAME backoff target as A1, so
                                           A2 - A1 changes exactly one thing:
                                           which axis the deviation is on)
  A3  as A2 with the reverse axis at k=65 (exp_018's variance-decomposition
                                           prior for that axis).  A3 differs
                                           from A1 in TWO ways, so it is only
                                           ever compared against A2.
  A4  d_r solo over k in {0,30,65,111,200,400,800,1600,3200} x {self, league}.
      A SELECTION, therefore selected on the supply blocks and nowhere else.

CARD 2 — THE LOGIT COORDINATE SYSTEM
------------------------------------
  B1  sigmoid(logit(p) + gamma_s * d_s)   (A1's logit twin; B1 - A1 is the
                                           coordinate swap and nothing else)
  B2  sigmoid(logit(p) + gamma_r * d_r)   (A2's logit twin)
  B3  as B1 with gamma taken from an explicit block grid instead of the bounded
      1-D minimiser, so the curve — and the residual of a quadratic fit to it —
      is on the record.

The argument for the logit space is NOT numerical: this file measures the clip
and reports how often it fires (the pre-registered expectation is never).  The
argument is that probability-space addition moves every row by the same amount
while logit-space addition moves a row by `gamma*d*p(1-p)`, i.e. less at the
extremes.  Whether the two can differ AT ALL here is an empirical question
about the spread of `p`, and it is answered BEFORE the arms are scored: the
`p_distribution` block below reports sd, the 1/99 percentiles, and the
coefficient of variation of `p(1-p)`, plus a direct measurement of how far the
two transforms can pull apart at matched calibration (`coordinate_divergence`).
If that spread is narrow, `p(1-p)` is nearly constant, the logit map is nearly
an affine rescaling of the linear one, and the card is structurally void.  That
is a result, not a failure, and it is reported as measured.

BETA / GAMMA
------------
Never an axis.  For every (arm, seed) the coefficient is fitted on supply blocks
2022 + 2023 OF THAT SEED and on that arm's own column.  Linear arms use
`offset.fit_slope` (the deployed closed form, rounded to 6dp).  Logit arms use
`offset.fit_gamma` (bounded 1-D Brier minimisation) — both are code that already
existed in `offset.py`; neither is rewritten here.  No coefficient ever sees
2024.

SELECTION
---------
The A4 surface and B3's gamma are the only things CHOSEN, and both are chosen on
the supply blocks.  A1, A2, A3, B1 and B2 are pre-registered contrasts: they are
measured, not picked.  fold_2024 is read after every coefficient and every
choice is fixed, and where a block choice disagrees with the fold_2024 oracle
the gap is reported in points.

INHERITANCE
-----------
`devgen.py` (which builds `d_r`), `dev_k.py`, `backoff.py`, `offset.py`,
`features_platoon.py`, `labels.py`, `exp016_ref.py` are BYTE-PRESERVED from
exp_026 and their sha256 are asserted.  The rebuilt reverse tables are further
checked against the per-season standard deviations and league cell values
exp_026 recorded, so "inherited" is a verified claim rather than a filename.
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
OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_028")
EXP020_PRED = os.path.expanduser(
    "~/SungjinPark/experiments/exp_020/pred/pred_E0_B3_P4iter_seed42_fold_2024.npy")
EXP022_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_022")
EXP024_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_024")
EXP024_METRICS = os.path.join(EXP024_DIR, "metrics.json")
EXP026_METRICS = os.path.expanduser(
    "~/SungjinPark/experiments/exp_026/metrics.json")

SEEDS = [42, 7, 2024]
GATE_TOL = 5e-6

TARGET = "control_success"
DROP_COLS = ["row_id", TARGET]
F_EXCLUDE_MAX_SEASON = 2022
SUPPLY_BLOCKS = [2022, 2023]
VALID_SEASON = 2024
EPS = 1e-6
BSS_REF_BRIER = 0.2498069

K_REF = 111                      # the deployed success-axis k (equivalence check)
K_S = 400                        # candidate J's success-axis k = A1's k
K_R_MAIN = 400                   # A2
K_R_ALT = 65                     # A3
K_GRID = [0, 30, 65, 111, 200, 400, 800, 1600, 3200]     # A4
TARGETS = ["self", "league"]
GAMMA_GRID = np.round(np.arange(-2.0, 12.0001, 0.05), 4)  # B3
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
    "devgen.py": "ac9640b94c80566200c89b81642a4b2cfb23aac86b6bb2dbe81bcc595557833c",
}

# arm -> (deviation column name or None, coordinate space)
ARMS = {
    "A0": (None, "none"),
    "A1": ("s400_league", "linear"),
    "A2": ("r400_league", "linear"),
    "A3": ("r65_league", "linear"),
    "B1": ("s400_league", "logit"),
    "B2": ("r400_league", "logit"),
    "B3": ("s400_league", "logit_grid"),
}
ARM_SPEC = {
    "A0": "no correction — p as the model produced it",
    "A1": "p + beta_s*d_s, d_s at k=400 x league  (= exp_024 arm B = candidate "
          "J's d).  CONTROL and reproduction gate",
    "A2": "p + beta_r*d_r, d_r at k=400 x league.  MAIN: same k, same backoff "
          "target, same one-term structure as A1 — only the axis differs",
    "A3": "p + beta_r*d_r, d_r at k=65 x league.  TWO things differ from A1, so "
          "compared against A2 only",
    "B1": "sigmoid(logit(p) + gamma_s*d_s), d_s at k=400 x league.  A1's logit "
          "twin: B1 - A1 is the coordinate swap",
    "B2": "sigmoid(logit(p) + gamma_r*d_r), d_r at k=400 x league.  A2's twin",
    "B3": "as B1 but gamma read off an explicit block grid rather than the "
          "bounded minimiser; the whole curve is stored",
}
CONTRASTS = [
    ("A2_minus_A1", "A2", "A1", "MAIN card 1: the axis swap, one variable"),
    ("A1_minus_A0", "A1", "A0", "what the deployed d_s correction is worth"),
    ("A2_minus_A0", "A2", "A0", "what a d_r-only correction is worth"),
    ("A3_minus_A2", "A3", "A2", "the reverse axis's k, 65 vs 400"),
    ("B1_minus_A1", "B1", "A1", "MAIN card 2: the coordinate swap, one variable"),
    ("B2_minus_A2", "B2", "A2", "coordinate swap on the reverse axis"),
    ("B3_minus_B1", "B3", "B1", "grid gamma vs the bounded minimiser"),
    ("B2_minus_B1", "B2", "B1", "axis swap INSIDE the logit space"),
]


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
    """LB-scale conversion of a FULL-PRECISION Brier."""
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


def logit(p):
    pc = np.clip(np.asarray(p, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(pc / (1.0 - pc))


def describe_p(p, tag):
    p = np.asarray(p, dtype=np.float64)
    v = p * (1.0 - p)
    qs = [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]
    pq = np.percentile(p, qs)
    vq = np.percentile(v, qs)
    return {
        "tag": tag, "n": int(len(p)),
        "mean": float(p.mean()), "sd": float(p.std()),
        "min": float(p.min()), "max": float(p.max()),
        "percentiles": {str(q): float(x) for q, x in zip(qs, pq)},
        "p99_minus_p1": float(pq[-2] - pq[1]),
        "v_mean": float(v.mean()), "v_sd": float(v.std()),
        "v_cv": float(v.std() / v.mean()),
        "v_min": float(v.min()), "v_max": float(v.max()),
        "v_max_over_min": (float(v.max() / v.min()) if v.min() > 0 else None),
        "v_percentiles": {str(q): float(x) for q, x in zip(qs, vq)},
        "v_p99_over_p1": float(vq[-2] / vq[1]),
        "note": "v = p(1-p) is the local slope dp/dz of the logit map.  If its "
                "coefficient of variation is small, sigmoid(logit(p)+g*d) is "
                "approximately p + g*v_mean*d, i.e. the linear arm with "
                "beta = g*v_mean, and the two coordinate systems cannot "
                "meaningfully disagree",
    }


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
    print("inherited sha OK — devgen.py "
          f"{src_sha['devgen.py'][:16]}..., offset.py {src_sha['offset.py'][:16]}...",
          flush=True)

    # ---- the records exp_024 and exp_026 left.  Gate numbers are READ. ------- #
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

    GATE_A1 = {int(s): m24_rec["per_seed"][str(s)]["B_candidate"]["brier"]
               for s in SEEDS}
    GATE_A0 = {int(s): m24_rec["per_seed"][str(s)]["base_no_offset"][
        "fold_2024_brier"] for s in SEEDS}
    A_DEPLOYED_REF = {int(s): m24_rec["per_seed"][str(s)]["A_control"]["brier"]
                      for s in SEEDS}

    exp026_metrics_sha = sha256_file(EXP026_METRICS)
    with open(EXP026_METRICS) as f:
        m26_rec = json.load(f)
    assert m26_rec["exp_id"] == "exp_026"
    assert m26_rec["train_csv_sha256"] == csv_sha
    assert m26_rec["source_sha256"]["devgen.py"] == src_sha["devgen.py"], \
        "devgen.py is not the file exp_026 ran"
    print(f"exp_024 record read (sha {exp024_metrics_sha[:16]}...), exp_026 record "
          f"read (sha {exp026_metrics_sha[:16]}...)", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] loading {DATA_PATH}", flush=True)
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    assert df["row_id"].is_unique
    assert df["season"].is_monotonic_increasing
    provided_cols = [c for c in df.columns if c not in DROP_COLS]
    assert len(provided_cols) == 47

    rec = recover(df, verbose=False)
    assert rec["success_recovery"]["agree_pct"] == 100.0

    # ---- recovered labels may never be an input ----------------------------- #
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
    row_id = df["row_id"].to_numpy()

    # ---- deviations.  SEED-INDEPENDENT (they come from labels, not the model) #
    print(f"[{time.strftime('%H:%M:%S')}] platoon lookups + k=111 reference",
          flush=True)
    fp.build_season_forward_chained(df, ok, succ_season, f_excl, verbose=False)
    dev_ref, hit_ref, _, _ = off.build_deviations(df, ok, succ_season, f_excl,
                                                  verbose=False)
    assert fp.RATE_SPEC[bk.SUCC_COL][1] == K_REF

    print(f"[{time.strftime('%H:%M:%S')}] success axis via BYTE-PRESERVED dev_k.py",
          flush=True)
    d_s, hit, n_cell0, _, _ = dk.build_multi_k(df, ok, succ_season, f_excl,
                                               [K_REF, K_S], verbose=False)
    assert np.array_equal(hit, hit_ref)
    L_s, has_Ls, _, _ = bk.build_league_dev(df, ok, succ_season, f_excl,
                                            verbose=False)

    print(f"[{time.strftime('%H:%M:%S')}] reverse axis via INHERITED devgen.py "
          f"(k grid {K_GRID})", flush=True)
    d_r, hit_r, n_cell_r, _, dr_diag = dg.build_multi_k(
        df, ok, succ_season, f_excl, K_GRID, "rev", verbose=False)
    L_r, has_Lr, _, Lr_diag = dg.build_league_dev(df, ok, succ_season, f_excl,
                                                  "rev", verbose=False)
    assert np.array_equal(hit_r, hit), "reverse axis disagrees on cell hits"
    assert np.array_equal(np.nan_to_num(n_cell_r, nan=-1.0),
                          np.nan_to_num(n_cell0, nan=-1.0))
    assert np.array_equal(has_Lr, has_Ls), "reverse league coverage differs"

    # ---- INHERITANCE CHECK: the rebuilt reverse tables must equal exp_026's -- #
    inh = {"what": "the reverse tables are rebuilt by the byte-preserved devgen.py; "
                   "their per-season sd and league cell values are compared to the "
                   "numbers exp_026 recorded, so 'inherited' is verified, not "
                   "assumed",
           "per_season_sd": {}, "league_cells": {}}
    for s_str, want in m26_rec["reverse_axis_per_season_sd"].items():
        got = dr_diag["per_season"][s_str]
        for k in ("65", "400"):
            g = got["sd_d_by_k"][k]
            w = want["sd_d_by_k"][k]
            assert g == w, ("reverse axis sd drifted", s_str, k, g, w)
        assert got["cell_hit_pct"] == want["cell_hit_pct"]
        inh["per_season_sd"][s_str] = {"k65": got["sd_d_by_k"]["65"],
                                       "k400": got["sd_d_by_k"]["400"],
                                       "matches_exp026": True}
    for s_str, want in m26_rec["L_reverse_by_cell_by_season"].items():
        got = Lr_diag["per_season"][s_str]["L_by_cell"]
        assert got == want, ("reverse league table drifted", s_str, got, want)
        inh["league_cells"][s_str] = {"matches_exp026": True, "L_by_cell": got}
    print("INHERITANCE CHECK (reverse tables == exp_026's) OK", flush=True)

    # ---- k=111 / k=65 expanded-vs-literal equivalence ------------------------ #
    dd111 = d_s[K_REF] - dev_ref["d_s"]
    ddr = d_r[K_R_ALT] - dev_ref["d_r"]         # offset.py's d_r is at k=65
    equiv = {
        "form_used": "(s_cell - n_cell*overall)/(n_cell + k)  [expanded, exp_023]",
        "max_abs_diff_ds_k111_vs_inherited_literal_form": float(np.max(np.abs(dd111))),
        "max_abs_diff_dr_k65_vs_inherited_literal_form": float(np.max(np.abs(ddr))),
        "tolerance_used": 1e-12,
        "miss_rows_exactly_zero_dr": {str(k): bool(np.all(d_r[k][~hit] == 0.0))
                                      for k in K_GRID},
    }
    assert equiv["max_abs_diff_ds_k111_vs_inherited_literal_form"] < 1e-12, equiv
    assert equiv["max_abs_diff_dr_k65_vs_inherited_literal_form"] < 1e-12, equiv
    assert all(equiv["miss_rows_exactly_zero_dr"].values()), equiv

    Ls_eff = np.where(has_Ls, L_s, 0.0)
    Lr_eff = np.where(has_Lr, L_r, 0.0)

    # ---- the deviation columns ---------------------------------------------- #
    TERMS = {"s400_league": dk.build_arm(d_s[K_S], hit, Ls_eff, n_cell0,
                                         K_S, "league")}
    GRID_ARMS = {}
    for k in K_GRID:
        for tg in TARGETS:
            nm = f"r{k}_{tg}"
            TERMS[nm] = dk.build_arm(d_r[k], hit, Lr_eff, n_cell0, k, tg)
            GRID_ARMS[f"G_{nm}"] = nm
    # A2 / A3 are grid points; assert the columns are literally the same object
    assert np.array_equal(TERMS["r400_league"], TERMS[ARMS["A2"][0]])
    assert np.array_equal(TERMS["r65_league"], TERMS[ARMS["A3"][0]])
    # at k = 0 the league weight is 0 on hits and 1 on misses (dev_k's documented
    # limit), which is exactly the hard backoff — the two targets must coincide
    assert np.array_equal(TERMS["r0_self"], TERMS["r0_league"]), \
        "k=0 self and league must coincide by construction"
    # self and league agree on miss rows for every k>0 by construction
    for k in K_GRID:
        assert np.array_equal(TERMS[f"r{k}_self"][~hit], TERMS[f"r{k}_league"][~hit])
        assert np.array_equal(TERMS[f"r{k}_self"][hit], d_r[k][hit])

    # ---- label isolation (b): deviations constant inside the lookup group ---- #
    print(f"[{time.strftime('%H:%M:%S')}] label-isolation check (group constancy)",
          flush=True)
    gcodes = pd.DataFrame({"s": season, "p": df[fp.PITCHER].to_numpy(),
                           "h": df[fp.HAND].to_numpy(), "ph": ph_all}
                          ).groupby(["s", "p", "h", "ph"], sort=False,
                                    dropna=False).ngroup().to_numpy()
    const = {}
    for nm, v in list(TERMS.items()) + [("d_r_raw_k400", d_r[K_R_MAIN]),
                                        ("d_r_raw_k65", d_r[K_R_ALT]),
                                        ("d_s_raw_k400", d_s[K_S])]:
        first = pd.Series(v).groupby(gcodes, sort=False).transform("first").to_numpy()
        const[nm] = bool(np.array_equal(v, first))
    label_isolation["check_b_constant_within_season_pitcher_hand"] = const
    label_isolation["n_lookup_groups"] = int(gcodes.max() + 1)
    label_isolation["n_vectors_checked"] = len(const)
    assert all(const.values()), ("a deviation varies inside its own lookup group — "
                                 "a row-level label reached it", const)
    print(f"LABEL ISOLATION OK ({len(const)} vectors, "
          f"{label_isolation['n_lookup_groups']} groups)", flush=True)

    # ---- as-of, independently re-verified ------------------------------------ #
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
        "tables_covered": ["personal_success", "personal_reverse",
                           "league_success", "league_reverse"],
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

    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    sup_idx = np.concatenate([np.flatnonzero(season == V) for V in SUPPLY_BLOCKS])
    corr_report = {}
    for tag, idx in (("supply_blocks_2022_2023", sup_idx),
                     ("fold_2024", np.flatnonzero(m24))):
        h = hit[idx]
        corr_report[tag] = {
            "corr_A1col_vs_A2col_all_rows": corr(TERMS["s400_league"][idx],
                                                 TERMS["r400_league"][idx]),
            "corr_A1col_vs_A2col_hit_rows": corr(TERMS["s400_league"][idx][h],
                                                 TERMS["r400_league"][idx][h]),
            "sd_A1col": float(np.std(TERMS["s400_league"][idx])),
            "sd_A2col": float(np.std(TERMS["r400_league"][idx])),
            "sd_A3col": float(np.std(TERMS["r65_league"][idx])),
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
        if s == 42:
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

    ARM_LIST = list(ARMS) + list(GRID_ARMS)
    SAVE_ARMS = ["A0", "A1", "A2", "A3", "B1", "B2"]

    def block_brier(p, d, coef, space):
        if space == "none":
            q = np.asarray(p, dtype=np.float64)
        elif space == "linear":
            q = off.apply_linear(p, d, coef)
        else:
            q = off.apply_logit(p, d, coef)
        return q

    per_seed, provenance = {}, {}
    p_dist = {}
    for s in SEEDS:
        print(f"[{time.strftime('%H:%M:%S')}] ===== seed {s} =====", flush=True)
        p24, blk, prov = load_bases(s)
        provenance[str(s)] = prov
        br_base24 = float(np.mean((p24 - y24) ** 2))
        y_sup = np.concatenate([y[V] for V in SUPPLY_BLOCKS])
        p_sup = np.concatenate([blk[V] for V in SUPPLY_BLOCKS])
        r_sup = y_sup - p_sup
        br_base_blocks = float(np.mean((p_sup - y_sup) ** 2))

        # ---- CARD 2 PRECONDITION: the spread of p, measured before any arm --- #
        pd24 = describe_p(p24, "fold_2024")
        pdsup = describe_p(p_sup, "supply_blocks_2022_2023")
        # how far can the two coordinate systems pull apart at matched
        # calibration?  gamma_equiv = beta / mean(v) is the first-order match.
        beta_a1_probe = off.fit_slope(r_sup, np.concatenate(
            [T["s400_league"][V] for V in SUPPLY_BLOCKS]))["beta"]
        v24 = p24 * (1.0 - p24)
        g_equiv = beta_a1_probe / float(v24.mean())
        d24_a1 = T["s400_league"][VALID_SEASON]
        p_lin = off.apply_linear(p24, d24_a1, beta_a1_probe)
        p_log = off.apply_logit(p24, d24_a1, g_equiv)
        shift_lin = np.abs(p_lin - p24)
        div = np.abs(p_log - p_lin)
        p_dist[str(s)] = {
            "fold_2024": pd24, "supply_blocks": pdsup,
            "coordinate_divergence": {
                "what": "the linear arm at its fitted beta vs the logit arm at "
                        "gamma = beta/mean(p(1-p)), which is the first-order "
                        "match of the two maps.  What is left is the ONLY thing "
                        "the logit space can buy beyond a rescaling of beta",
                "beta_used": beta_a1_probe, "gamma_equivalent": g_equiv,
                "mean_abs_correction_linear": float(shift_lin.mean()),
                "max_abs_correction_linear": float(shift_lin.max()),
                "mean_abs_divergence": float(div.mean()),
                "max_abs_divergence": float(div.max()),
                "divergence_over_correction_mean": float(div.mean()
                                                         / shift_lin.mean()),
                "divergence_over_correction_max": float(div.max()
                                                        / shift_lin.max()),
            },
            "clip_probe_linear_A1": {
                "raw_min": float((p24 + beta_a1_probe * d24_a1).min()),
                "raw_max": float((p24 + beta_a1_probe * d24_a1).max()),
                "n_clipped": int(((p24 + beta_a1_probe * d24_a1 < EPS)
                                  | (p24 + beta_a1_probe * d24_a1 > 1 - EPS)).sum()),
                "headroom_low": float((p24 + beta_a1_probe * d24_a1).min() - EPS),
                "headroom_high": float(1 - EPS
                                       - (p24 + beta_a1_probe * d24_a1).max()),
            },
        }
        if s == SEEDS[0]:
            print("P DISTRIBUTION (fold_2024): " + json.dumps(
                {k: pd24[k] for k in ("mean", "sd", "min", "max", "v_mean",
                                      "v_sd", "v_cv", "v_max_over_min")}),
                  flush=True)
            print("COORD DIVERGENCE: " + json.dumps(
                p_dist[str(s)]["coordinate_divergence"]), flush=True)

        # ---- coefficients, fitted on the supply blocks ONLY ------------------ #
        blockfit = {}
        b3_curve = None
        for a in ARM_LIST:
            if a in ARMS:
                nm, space = ARMS[a]
            else:
                nm, space = GRID_ARMS[a], "linear"
            if space == "none":
                fit = {"space": "none", "coef": None,
                       "brier_blocks": br_base_blocks,
                       "delta_vs_base_blocks": 0.0,
                       "score_blocks": score(br_base_blocks),
                       "crossblock_brier_pooled": br_base_blocks,
                       "coef_per_block": {}}
                blockfit[a] = fit
                continue
            dsup = np.concatenate([T[nm][V] for V in SUPPLY_BLOCKS])
            if space == "linear":
                f1 = off.fit_slope(r_sup, dsup)
                coef = f1["beta"]
                fit = {"space": "linear", "coef": coef, "se": f1["se"],
                       "t": f1["t"],
                       "coef_unrounded": float(np.dot(dsup, r_sup)
                                               / np.dot(dsup, dsup)),
                       "fitter": "offset.fit_slope (deployed code, rounds to 6dp)"}
            elif space == "logit":
                fg = off.fit_gamma(p_sup, y_sup, dsup, bracket=(-20.0, 20.0))
                coef = fg["gamma"]
                assert fg["converged"], (a, s, fg)
                fit = {"space": "logit", "coef": coef,
                       "fit_detail": fg,
                       "fitter": "offset.fit_gamma (bounded 1-D Brier minimiser, "
                                 "rounds to 6dp) — pre-existing, never deployed"}
            else:                                            # logit_grid  (B3)
                zs = logit(p_sup)
                curve = []
                for g in GAMMA_GRID:
                    q = 1.0 / (1.0 + np.exp(-(zs + g * dsup)))
                    curve.append(float(np.mean((q - y_sup) ** 2)))
                curve = np.asarray(curve)
                j = int(np.argmin(curve))
                assert 0 < j < len(curve) - 1, ("B3 gamma argmin hit the grid edge "
                                                "— widen GAMMA_GRID", j, s)
                coef = float(GAMMA_GRID[j])
                # quadratic fit in a window around the minimum
                w = (np.abs(GAMMA_GRID - coef) <= 1.0)
                cq = np.polyfit(GAMMA_GRID[w], curve[w], 2)
                resid = curve[w] - np.polyval(cq, GAMMA_GRID[w])
                b3_curve = {
                    "gamma_grid_min": float(GAMMA_GRID[0]),
                    "gamma_grid_max": float(GAMMA_GRID[-1]),
                    "gamma_grid_step": float(GAMMA_GRID[1] - GAMMA_GRID[0]),
                    "n_points": int(len(GAMMA_GRID)),
                    "argmin_gamma_on_blocks": coef,
                    "brier_at_argmin": float(curve[j]),
                    "brier_at_gamma_0": float(curve[int(np.argmin(
                        np.abs(GAMMA_GRID - 0.0)))]),
                    "quadratic_window": 1.0,
                    "quadratic_coeffs_a_b_c": [float(x) for x in cq],
                    "quadratic_vertex_gamma": float(-cq[1] / (2 * cq[0])),
                    "quadratic_max_abs_residual": float(np.max(np.abs(resid))),
                    "quadratic_max_abs_residual_score": score_delta(
                        float(np.max(np.abs(resid)))),
                    "gamma_naive_beta_over_mean_v": g_equiv,
                    "gamma_naive_minus_argmin": float(g_equiv - coef),
                    "curve": {str(float(g)): float(c)
                              for g, c in zip(GAMMA_GRID, curve)},
                }
                fit = {"space": "logit_grid", "coef": coef,
                       "fitter": "explicit block grid argmin",
                       "grid": {k: v for k, v in b3_curve.items() if k != "curve"}}
            pb = block_brier(p_sup, dsup, coef, space if space != "logit_grid"
                             else "logit")
            fit["brier_blocks"] = float(np.mean((pb - y_sup) ** 2))
            fit["delta_vs_base_blocks"] = fit["brier_blocks"] - br_base_blocks
            fit["score_blocks"] = score(fit["brier_blocks"])
            # honest cross-block surface: fit on one block, score the other
            cb = {}
            for Vf, Ve in ((SUPPLY_BLOCKS[0], SUPPLY_BLOCKS[1]),
                           (SUPPLY_BLOCKS[1], SUPPLY_BLOCKS[0])):
                rf = y[Vf] - blk[Vf]
                if space == "linear":
                    c2 = off.fit_slope(rf, T[nm][Vf])["beta"]
                    sp = "linear"
                else:
                    c2 = off.fit_gamma(blk[Vf], y[Vf], T[nm][Vf],
                                       bracket=(-20.0, 20.0))["gamma"]
                    sp = "logit"
                pe = block_brier(blk[Ve], T[nm][Ve], c2, sp)
                cb[f"fit{Vf}_score{Ve}"] = {"coef": c2,
                                            "brier": float(np.mean((pe - y[Ve]) ** 2)),
                                            "n": int(len(y[Ve]))}
            tot_n = sum(v["n"] for v in cb.values())
            fit["crossblock_brier_pooled"] = float(
                sum(v["brier"] * v["n"] for v in cb.values()) / tot_n)
            fit["crossblock_detail"] = cb
            if space == "logit_grid":
                fit["crossblock_note"] = ("the cross-block and per-block gammas "
                                          "here come from the bounded minimiser, "
                                          "not the grid; only B3's APPLIED gamma "
                                          "is the grid argmin")
            fit["coef_per_block"] = {
                str(V): (off.fit_slope(y[V] - blk[V], T[nm][V])["beta"]
                         if space == "linear"
                         else off.fit_gamma(blk[V], y[V], T[nm][V],
                                            bracket=(-20.0, 20.0))["gamma"])
                for V in SUPPLY_BLOCKS}
            blockfit[a] = fit

        seb = (p24 - y24) ** 2

        def evaluate(a):
            if a in ARMS:
                nm, space = ARMS[a]
            else:
                nm, space = GRID_ARMS[a], "linear"
            coef = blockfit[a]["coef"]
            if space == "none":
                p = np.asarray(p24, dtype=np.float64).copy()
                c = np.zeros_like(p24)
                shift_info = {"n_clip": 0, "total_mean_abs": 0.0,
                              "total_max_abs": 0.0}
            else:
                c = T[nm][VALID_SEASON]
                if space == "linear":
                    p = off.apply_linear(p24, c, coef)
                    raw = p24 + coef * c
                    n_clip = int(((raw < EPS) | (raw > 1 - EPS)).sum())
                else:
                    p = off.apply_logit(p24, c, coef)
                    raw = p
                    n_clip = 0
                sh = p - p24
                shift_info = {
                    "n_clip": n_clip,
                    "raw_min": float(np.min(raw)), "raw_max": float(np.max(raw)),
                    "total_mean_abs": float(np.mean(np.abs(sh))),
                    "total_max_abs": float(np.max(np.abs(sh))),
                    "sd_d": float(np.std(c)),
                }
            se = (p - y24) ** 2
            br = float(np.mean(se))
            out = {
                "arm": a, "spec": ARM_SPEC.get(a, f"A4 grid point: d_r {nm}"),
                "term": nm, "space": space, "coef_applied": coef,
                "brier": br, "score": score(br),
                "delta_vs_base_no_offset": float(br - br_base24),
                "score_gain_vs_base": score_delta(br - br_base24),
                "valid_R": seg(y24[isr24], p[isr24]),
                "valid_F": seg(y24[~isr24], p[~isr24]),
                "seg_miss": seg(y24[~hit24], p[~hit24]),
                "seg_hit": seg(y24[hit24], p[hit24]),
                "correction_shift": shift_info,
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
        # the grid points that ARE A2 / A3 must be bit-identical to them
        assert res["G_r400_league"]["brier"] == res["A2"]["brier"]
        assert res["G_r65_league"]["brier"] == res["A3"]["brier"]
        assert np.array_equal(preds["A0"], p24)
        for a in SAVE_ARMS:
            np.save(os.path.join(OUT_DIR, "pred",
                                 f"pred_{a}_seed{s}_fold_2024.npy"), preds[a])

        per_seed[str(s)] = {
            "base_no_offset": {"fold_2024_brier": br_base24,
                               "fold_2024_score": score(br_base24),
                               "blocks_brier": br_base_blocks},
            "arms": res,
            "block_fit": blockfit,
            "B3_gamma_curve": b3_curve,
            "contrasts": {name: {
                "brier_delta": res[a]["brier"] - res[b]["brier"],
                "score_delta": score_delta(res[a]["brier"] - res[b]["brier"]),
                "what": what}
                for name, a, b, what in CONTRASTS},
            "arm_pred_gap_vs_A1": {
                a: {"mean_abs": float(np.mean(np.abs(preds[a] - preds["A1"]))),
                    "max_abs": float(np.max(np.abs(preds[a] - preds["A1"])))}
                for a in ("A0", "A2", "A3", "B1", "B2", "B3")},
        }
        print("  seed %d: " % s + "  ".join(
            f"{a} {res[a]['brier']:.12f}" for a in ARMS), flush=True)
        print(f"    A2-A1 {res['A2']['brier'] - res['A1']['brier']:+.4e} "
              f"({score_delta(res['A2']['brier'] - res['A1']['brier']):+.5f})  "
              f"B1-A1 {res['B1']['brier'] - res['A1']['brier']:+.4e} "
              f"({score_delta(res['B1']['brier'] - res['A1']['brier']):+.5f})",
              flush=True)
        del p24, blk, preds

    # ---- GATE 1 / 2: A1 and A0 reproduce exp_024 on all three seeds ---------- #
    gate = {"tolerance": GATE_TOL,
            "source": "exp_024/metrics.json per_seed[seed].B_candidate.brier and "
                      ".base_no_offset.fold_2024_brier — READ from the record, "
                      "not transcribed",
            "on_fail": "abort — the reused vectors or the inherited code are not "
                       "exp_024's and nothing downstream is readable",
            "checks": {}}
    allpass = True
    for s in SEEDS:
        for tag, arm, want in (("A1", "A1", GATE_A1[s]), ("A0", "A0", GATE_A0[s])):
            got_b = per_seed[str(s)]["arms"][arm]["brier"]
            diff = abs(got_b - want)
            ok_ = bool(diff < GATE_TOL)
            allpass = allpass and ok_
            gate["checks"][f"{tag}_seed{s}"] = {
                "expected_from_exp024": want, "got": got_b, "abs_diff": diff,
                "bit_identical": bool(got_b == want), "passed": ok_}
    gate["passed"] = bool(allpass)
    print("REPRODUCTION GATE: " + json.dumps(gate), flush=True)
    assert gate["passed"], "REPRODUCTION GATE FAILED: " + json.dumps(gate)

    # ---- adoption rule, applied exactly as registered ------------------------ #
    a1_seed_std = float(np.std([per_seed[str(s)]["arms"]["A1"]["brier"]
                                for s in SEEDS], ddof=1))
    threshold = 3.0 * a1_seed_std

    def judge(name):
        v = [per_seed[str(s)]["contrasts"][name]["brier_delta"] for s in SEEDS]
        mean = float(np.mean(v))
        signs = bool(all(x < 0 for x in v) or all(x > 0 for x in v))
        pstd = float(np.std(v, ddof=1))
        return {"by_seed": {str(s): x for s, x in zip(SEEDS, v)},
                "by_seed_score": {str(s): score_delta(x) for s, x in zip(SEEDS, v)},
                "signs_agree": signs,
                "mean": mean, "mean_score": score_delta(mean),
                "paired_delta_std_ddof1_REFERENCE": pstd,
                "paired_delta_std_score_REFERENCE": score_delta(pstd),
                "threshold_over_paired_std": (float(threshold / pstd)
                                              if pstd > 0 else None),
                "abs_mean": abs(mean),
                "threshold_3x_A1_seed_std": threshold,
                "passes_magnitude": bool(abs(mean) >= threshold),
                "ADOPT": bool(signs and abs(mean) >= threshold),
                "direction": ("improves (lower Brier)" if mean < 0
                              else "worsens" if mean > 0 else "flat")}

    adoption = {"rule": "3 seeds agree in sign AND |mean delta| >= 3 x A1 seed "
                        "std.  Not relaxed.  The paired-delta std is reported "
                        "alongside as a reference number and is NOT part of the "
                        "rule; changing the rule is a user decision and would "
                        "have to be pre-registered and applied forward only.",
                "A1_seed_std_ddof1": a1_seed_std,
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
                               "fold_2024 is read afterwards and reported as the "
                               "oracle counterfactual only",
                 "primary_surface": "block Brier at the block-fitted coefficient "
                                    "(exp_023's registered surface)",
                 "secondary_surface": "cross-block: fit 2022 -> score 2023 and the "
                                      "reverse, pooled",
                 "not_a_selection": ["A1", "A2", "A3", "B1", "B2"],
                 "groups": {}}
    grid_ids = list(GRID_ARMS)
    for g, cands in (("A4_dr_k_x_target", grid_ids),
                     ("A4_dr_k_league_only",
                      [f"G_r{k}_league" for k in K_GRID]),
                     ("A4_dr_k_self_only", [f"G_r{k}_self" for k in K_GRID])):
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
    selection["B3_gamma"] = {
        "surface": "block Brier over an explicit gamma grid, per seed",
        "argmin_by_seed": {str(s): per_seed[str(s)]["B3_gamma_curve"][
            "argmin_gamma_on_blocks"] for s in SEEDS},
        "bounded_minimiser_gamma_by_seed": {
            str(s): per_seed[str(s)]["block_fit"]["B1"]["coef"] for s in SEEDS},
        "naive_beta_over_mean_v_by_seed": {
            str(s): per_seed[str(s)]["B3_gamma_curve"][
                "gamma_naive_beta_over_mean_v"] for s in SEEDS},
        "quadratic_residual_by_seed": {
            str(s): per_seed[str(s)]["B3_gamma_curve"][
                "quadratic_max_abs_residual"] for s in SEEDS},
    }
    print("SELECTION: " + json.dumps(
        {g: {k: v for k, v in d.items() if k not in
             ("block_brier_mean", "crossblock_brier_mean", "fold2024_brier_mean",
              "candidates")}
         for g, d in selection["groups"].items()}), flush=True)

    arm_brier = {a: [per_seed[str(s)]["arms"][a]["brier"] for s in SEEDS]
                 for a in ARM_LIST}
    summary = {
        "arm_seed_table_brier": {a: {str(s): b for s, b in zip(SEEDS, v)}
                                 for a, v in arm_brier.items()},
        "arm_seed_table_score": {a: {str(s): score(b) for s, b in zip(SEEDS, v)}
                                 for a, v in arm_brier.items()},
        "arm_mean_brier": {a: float(np.mean(v)) for a, v in arm_brier.items()},
        "arm_mean_score": {a: score(float(np.mean(v))) for a, v in arm_brier.items()},
        "arm_seed_std_ddof1": {a: float(np.std(v, ddof=1))
                               for a, v in arm_brier.items()},
        "exp024_A_deployed_reference_brier": A_DEPLOYED_REF,
        "coefficients_by_seed": {
            str(s): {a: per_seed[str(s)]["block_fit"][a]["coef"]
                     for a in ARM_LIST} for s in SEEDS},
    }

    metrics = {
        "exp_id": "exp_028",
        "title": "d_r as a SOLO one-term correction, and the logit coordinate "
                 "system — post-hoc transforms of stored predictions, no training",
        "no_training": True,
        "arms": ARM_SPEC,
        "arm_terms": {a: {"term": v[0], "space": v[1]} for a, v in ARMS.items()},
        "a4_grid_arms": GRID_ARMS,
        "k_grid": K_GRID, "targets": TARGETS,
        "seeds": SEEDS,
        "changed_vs_exp_026": [
            "d_r enters as a ONE-TERM correction (p + beta_r*d_r); exp_026 only "
            "ever fitted it jointly with d_s",
            "the reverse-axis k grid is swept over 9 values x 2 backoff targets "
            "(exp_026 had two k values, league only)",
            "the logit coordinate system is applied for the first time "
            "(offset.apply_logit / fit_gamma existed but had never been run in "
            "an arm)",
            "an explicit gamma grid with a stored curve and a quadratic-fit "
            "residual (B3)",
            "the distribution of p and the linear/logit divergence at matched "
            "calibration are measured as a precondition, before any arm is scored",
            "A0 (no correction) is an explicit arm and a second gate",
        ],
        "unchanged_vs_exp_026": [
            "base vectors and their sha256, exp_004 F-mask, clip 1e-6, supply "
            "blocks 2022+2023, closed-form no-intercept slope for linear arms, "
            "backoff.py / dev_k.py / devgen.py / offset.py / features_platoon.py / "
            "labels.py / exp016_ref.py byte-identical (sha asserted)",
        ],
        "inherited_source_sha256": SHA_INHERITED,
        "source_sha256": src_sha,
        "train_csv_sha256": csv_sha,
        "base_vector_provenance": provenance,
        "base_vector_expected_sha256_from_exp024_record": {
            str(k): {str(kk): vv for kk, vv in v.items()} for k, v in BASE_SHA.items()},
        "exp024_metrics_json_sha256": exp024_metrics_sha,
        "exp026_metrics_json_sha256": exp026_metrics_sha,
        "reverse_table_inheritance_check": inh,
        "reproduction_gate": gate,
        "label_isolation_check": label_isolation,
        "k_equivalence": equiv,
        "asof_independent_recheck": asof_indep,
        "d_s_d_r_correlation": corr_report,
        "p_distribution": p_dist,
        "adoption": adoption,
        "selection": selection,
        "summary": summary,
        "per_seed": per_seed,
        "n_bin_definition": {
            "hit_tertile_boundaries_n_cell": [float(q1), float(q2)],
            "bin_sizes": {b: int(m.sum()) for b, m in bins24.items()},
        },
        "reverse_axis_per_season_sd": dr_diag["per_season"],
        "conversion": {
            "formula": f"score = 100000 * (1 - brier / {BSS_REF_BRIER})",
            "sensitivity": "brier 1e-6 = score 0.4003.  Every Brier here is "
                           "full precision, so the converted numbers carry no "
                           "rounding band of their own",
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
            "coef_fit": "supply blocks 2022+2023 of the same seed, on that arm's "
                        "own column; no arm's beta or gamma ever sees 2024",
            "arm_choice": "arms fixed by config.yaml before any 2024 number is "
                          "read; the A4 grid point and B3's gamma are selected on "
                          "the blocks; A1/A2/A3/B1/B2 are pre-registered contrasts",
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
    for a in ARM_LIST:
        print(f"{a:>16}" + "".join(
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
