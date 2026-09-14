"""exp_036 Q4 — is the S400_self offset still a gain at the UNCENSORED argmin?

WHAT IS BEING ASKED
-------------------
exp_034 adopted a linear offset  p' = clip(p + beta * d)  with cell S400_self
and beta pooled over the SUPPLY BLOCKS (2022, 2023) x block seeds (42, 7).
exp_036 moves the fold_2024 stacker from 481 rounds to its measured argmin.
The question is whether the offset still helps on that different base.

WHERE beta COMES FROM, AND WHY IT DOES NOT MOVE
------------------------------------------------
beta is a no-intercept least squares of (y - p) on d fitted ENTIRELY on the
supply blocks.  exp_036 changed exactly one thing — how far the fold_2024
stacker is trained — and the supply blocks are different units built by
different runs (exp_031 stage 3), untouched here.  So "refit beta on the supply
blocks" is, by construction, a reproduction rather than a new estimate.  This
file refits it anyway, from the supply blocks' own prediction files, and
asserts the result reproduces exp_034's recorded value read off disk.  That
turns "beta did not move" from an argument into a check.

fold_2024 IS NEVER USED TO CHOOSE ANYTHING HERE.  Neither the cell (S400_self,
inherited) nor beta (supply-fitted) looks at the 2024 validation rows.  The
2024 rows are only ever SCORED.

WHAT IS COMPARED
----------------
Two bases, the same offset applied to both:
  * base@481   — exp_036's own gate prediction, which is bit-checked against
                 exp_030's stage-2 vector.  This is the base under exp_034's
                 adoption, so applying the offset here must reproduce a known
                 number.
  * base@argmin — exp_036's prediction at the measured stacker argmin.
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

import backoff as bk                                                # noqa: E402
import dev_k as dk                                                  # noqa: E402
import features_platoon as fp                                       # noqa: E402
import offset as off                                                # noqa: E402
from labels import recover                                          # noqa: E402

DATA_PATH = os.path.expanduser("~/SungjinPark/data/train.csv")
OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_036")
EXP031 = os.path.expanduser("~/SungjinPark/experiments/exp_031")
EXP032 = os.path.expanduser("~/SungjinPark/experiments/exp_032")   # == exp_034
EXP030 = os.path.expanduser("~/SungjinPark/experiments/exp_030")

TARGET = "control_success"
F_EXCLUDE_MAX_SEASON = 2022
REF_BRIER = 0.2498069
K = 400
ARM = "self"
CELL = f"S{K}_{ARM}"
SUPPLY_BLOCKS = [2022, 2023]
BLOCK_SEEDS = [42, 7]
FOLD_SEEDS = [42, 7, 2024]
VALID_SEASON = 2024
EPS = off.EPS


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
    return float(f"{100000.0 * (1.0 - brier / REF_BRIER):.5f}")


def score_delta(dbrier):
    return float(f"{-100000.0 * dbrier / REF_BRIER:.5f}")


def main():
    t0 = time.time()
    out = {"exp_id": "exp_036", "what": "Q4 — offset on the uncensored base",
           "cell": CELL, "k": K, "arm": ARM,
           "supply_blocks": SUPPLY_BLOCKS, "block_seeds": BLOCK_SEEDS}

    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    rec = recover(df, verbose=False)
    assert rec["success_recovery"]["agree_pct"] == 100.0
    season = df["season"].to_numpy()
    ok = df["recov_ok"].to_numpy()
    succ_season = df.groupby("pitcher_id", sort=False)["season"].shift(-1).to_numpy()
    f_excl = ((df["season"] <= F_EXCLUDE_MAX_SEASON)
              & (df["game_type"] == "F")).to_numpy()
    y_all = df[TARGET].to_numpy(dtype=np.float64)

    print(f"[{time.strftime('%H:%M:%S')}] platoon lookups", flush=True)
    fp.build_season_forward_chained(df, ok, succ_season, f_excl, verbose=False)
    d_s, hit, n_cell0, _, _ = dk.build_multi_k(df, ok, succ_season, f_excl,
                                               [K], verbose=False)
    L_s, has_Ls, _, _ = bk.build_league_dev(df, ok, succ_season, f_excl,
                                            verbose=False)
    Ls_eff = np.where(has_Ls, L_s, 0.0)
    assert np.all(d_s[K][~hit] == 0.0), "d must be exactly 0 on cell misses"
    arm = dk.build_arm(d_s[K], hit, Ls_eff, n_cell0, K, ARM)
    out["d_zero_on_miss"] = True

    D = {V: arm[season == V] for V in SUPPLY_BLOCKS + [VALID_SEASON]}
    Y = {V: y_all[season == V] for V in SUPPLY_BLOCKS + [VALID_SEASON]}

    # ------------------------------------------------ beta, refitted on supply
    with open(os.path.join(EXP032, "curves", "iter_fit.json")) as f:
        ifit = json.load(f)
    prov = {}
    blk = {}
    for o in ifit["observations_raw"]:
        if o["unit"] in SUPPLY_BLOCKS and o["seed"] in BLOCK_SEEDS:
            p = os.path.join(EXP031, "pred",
                             f"pred_C5B3_u{o['unit']}_seed{o['seed']}_st3.npy")
            got = sha256_file(p)
            assert got == o["pred_sha256"], ("supply block pred drifted", p)
            blk[(o["unit"], o["seed"])] = np.load(p)
            prov[f"{o['unit']}_{o['seed']}"] = {
                "file": p, "sha256": got, "stack_argmin": o["stack_argmin"],
                "made_by": "exp_031 block_self.py stage 3 (own early stop)"}
    assert len(blk) == 4, sorted(blk)
    out["supply_provenance"] = prov

    y_sup = np.concatenate([Y[V] for V in SUPPLY_BLOCKS for s in BLOCK_SEEDS])
    p_sup = np.concatenate([blk[(V, s)] for V in SUPPLY_BLOCKS
                            for s in BLOCK_SEEDS])
    d_sup = np.concatenate([D[V] for V in SUPPLY_BLOCKS for s in BLOCK_SEEDS])
    fit = off.fit_slope(y_sup - p_sup, d_sup)
    beta = fit["beta"]
    out["beta_refit_on_supply"] = fit
    out["beta_applied"] = beta

    with open(os.path.join(EXP032, "metrics.json")) as f:
        m32 = json.load(f)
    prev = m32["report_derived"]["offset_adoption_on_new_base"]
    assert prev["selected_cell"] == CELL, (prev["selected_cell"], CELL)
    out["beta_recorded_by_exp034"] = prev["beta_applied"]
    out["beta_reproduces_exp034"] = bool(beta == prev["beta_applied"])
    assert out["beta_reproduces_exp034"], (beta, prev["beta_applied"])
    print(f"    beta refit = {beta} (exp_034 recorded "
          f"{prev['beta_applied']}) — reproduces", flush=True)

    # ------------------------------------------------------- the two bases
    y24, d24 = Y[VALID_SEASON], D[VALID_SEASON]
    out["d_2024"] = {"n": int(len(d24)), "sd": float(np.std(d24)),
                     "hit_pct": round(100.0 * float(hit[season == VALID_SEASON]
                                                    .mean()), 3),
                     "max_abs_beta_d": float(np.max(np.abs(beta * d24))),
                     "note": ("max_abs_beta_d is a PROBABILITY shift, not a "
                              "score; its sign per row is not known to be "
                              "helpful, so it does not convert to points")}

    bases = {}
    for label, patt in (("base_at_481_GATE",
                         "pred_C5B3_u2024_seed{s}_at481_GATE.npy"),
                        ("base_at_argmin", "pred_C5B3_u2024_seed{s}_st5.npy")):
        rows = {}
        for s in FOLD_SEEDS:
            p = np.load(os.path.join(OUT_DIR, "pred", patt.format(s=s)))
            assert len(p) == len(y24), (label, s, len(p), len(y24))
            po = off.apply_linear(p, d24, beta)
            b0 = float(np.mean((p - y24) ** 2))
            b1 = float(np.mean((po - y24) ** 2))
            rows[str(s)] = {
                "base_brier": b0, "base_score": score(b0),
                "offset_brier": b1, "offset_score": score(b1),
                "delta_brier": b1 - b0, "delta_score": score_delta(b1 - b0),
                "n_clipped": int((((p + beta * d24) < EPS)
                                  | ((p + beta * d24) > 1 - EPS)).sum()),
            }
        bb = [rows[str(s)]["base_brier"] for s in FOLD_SEEDS]
        oo = [rows[str(s)]["offset_brier"] for s in FOLD_SEEDS]
        dd = [rows[str(s)]["delta_brier"] for s in FOLD_SEEDS]
        bases[label] = {
            "by_seed": rows,
            "base_mean_brier": float(np.mean(bb)),
            "base_mean_score": score(float(np.mean(bb))),
            "base_seed_std": float(np.std(bb, ddof=0)),
            "offset_mean_brier": float(np.mean(oo)),
            "offset_mean_score": score(float(np.mean(oo))),
            "offset_seed_std": float(np.std(oo, ddof=0)),
            "mean_delta_brier": float(np.mean(dd)),
            "mean_delta_score": score_delta(float(np.mean(dd))),
            "signs_agree": bool(all(v < 0 for v in dd)
                                or all(v > 0 for v in dd)),
            "improves": bool(np.mean(dd) < 0),
        }
        # exp_034's adoption rule, restated on THIS base's own control std
        std = bases[label]["base_seed_std"]
        thr = 3.0 * std
        bases[label]["adoption_rule"] = {
            "rule": "3 seeds same sign AND |mean delta| >= 3 x CONTROL seed std",
            "CONTROL_seed_std": std,
            "THRESHOLD_abs_mean_delta": thr,
            "MULTIPLE_OF_THE_THRESHOLD": abs(float(np.mean(dd))) / thr
            if thr > 0 else None,
            "MULTIPLE_OF_THE_CONTROL_STD": abs(float(np.mean(dd))) / std
            if std > 0 else None,
            "naming_warning": ("MULTIPLE_OF_THE_THRESHOLD and MULTIPLE_OF_THE_"
                               "CONTROL_STD differ by exactly 3x and are not "
                               "interchangeable"),
            "signs_agree": bases[label]["signs_agree"],
            "passes_magnitude": bool(abs(float(np.mean(dd))) >= thr),
            "PASSES": bool(bases[label]["signs_agree"]
                           and np.mean(dd) < 0
                           and abs(float(np.mean(dd))) >= thr),
        }
    out["bases"] = bases

    # cross-check: the 481 base must equal exp_030's stage-2 Brier
    x = {}
    for s in FOLD_SEEDS:
        with open(os.path.join(EXP030, "res",
                               f"res_C5B3_u2024_seed{s}_st2.json")) as f:
            r = json.load(f)
        got = bases["base_at_481_GATE"]["by_seed"][str(s)]["base_brier"]
        x[str(s)] = {"exp030": r["brier_full"], "ours": got,
                     "abs_diff": abs(got - r["brier_full"])}
    out["cross_check_481_equals_exp030"] = x
    assert all(v["abs_diff"] < 5e-6 for v in x.values()), x

    out["compute_min"] = round((time.time() - t0) / 60, 2)
    p = os.path.join(OUT_DIR, "offset_q4.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print("WROTE " + p, flush=True)
    print(json.dumps({k: bases[k]["mean_delta_score"] for k in bases}), flush=True)


if __name__ == "__main__":
    main()
