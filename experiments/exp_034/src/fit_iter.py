"""exp_032 Stage B + Stage E — how does the honest argmin scale with training rows?

NO TRAINING HAPPENS IN THIS FILE.  It reads the stage-3 result files (every one
of which stopped on its OWN validation curve) and fits a law to
`argmin` vs `n_train`.  The law is then used to CHOOSE the iteration counts for
fold_2024 (Stage C) and to RECORD the iteration counts a deployment build on
`season <= 2024` would need (Stage E).

WHAT MAY BE READ, AND WHAT MAY NOT
----------------------------------
Only the BLOCK units (2021, 2022, 2023) are read.  fold_2024's and fold_2023's
validation curves are never opened here.  That is the whole point: an iteration
count chosen from fold_2024's own curve would make fold_2024 an oracle and every
downstream Brier uninterpretable.

WHY A LAW IS NEEDED AT ALL
--------------------------
exp_030 scaled stage-1's argmin by the ratio of training rows — i.e. it assumed
argmin is PROPORTIONAL to n_train.  exp_031 measured that assumption and found
it wrong in both directions at once: on block 2022 the stacker policy said 282
where the truth was 80 / 140, the cell `is_rev_only` policy said 788 where the
truth was 230 / 240, and the cell `is_wayoff_true` policy said 526 where the
truth was 625 / 725.  A rule that is too big on one cell and too small on
another is not miscalibrated, it is the wrong functional form.  So the form
itself is put up for selection.

THE FOUR CANDIDATE FORMS   (x = log n_train, y = log argmin)
------------------------------------------------------------
  loglog        y = a + b x                     2 params.  argmin ~ n^b.
  constant      y = a                           1 param.   argmin does not
                                                depend on n at all.
  proportional  y = a + 1.0 * x                 1 param.   THE OLD POLICY,
                                                carried so its error is
                                                measured on the same ruler.
  saturating    argmin = A * n / (B + n)        2 params.  argmin approaches a
                                                ceiling A; B is found on a log
                                                grid and A in closed form.

THE SELECTION RULE  (declared in this file before any stage-3 result for unit
--------------------  2021 existed; the file was written and uploaded while
                      those three jobs were still running)
Leave-one-SIZE-out.  For each of the three training sizes in turn: fit each
form on the OTHER two sizes' observations, predict log(argmin) at the held-out
size, and take the absolute error against the mean of the observations actually
seen there.  Sum the three absolute errors.  Lowest total wins; ties go to the
form with fewer parameters, and then to `constant`.

  ⚠️ This is a weak test and it is reported as one.  With three sizes, holding
  out an END size makes the prediction an EXTRAPOLATION from two points — which
  is exactly the operation that has to be trusted later — while holding out the
  MIDDLE size makes it an interpolation.  Both are reported per fold, so a form
  that only wins on the easy fold is visible.

  ⚠️ Seed spread is NOT small next to the effect being fitted.  On block 2022
  the stacker's two seeds are 80 and 140 — a factor of 1.75 — while the whole
  2022 -> 2023 movement is a factor of about 3.3.  Any claim about the slope
  carries that noise, and the residual table is the place to see it.

  ⚠️ The units are not pure rescalings of each other.  The B3 stacker's OOF
  columns are forward-chained by season and the earliest season keeps NaN, so
  the NaN share of the training pool falls monotonically with n_train (48.9% ->
  32.4% -> 24.3% -> 19.0% -> 15.4%).  A law in n_train alone absorbs that
  difference silently.  The share is carried into the output next to every
  observation so that it cannot be forgotten.

WHAT IS WRITTEN
---------------
  curves/iter_fit.json  every form, every fit, every residual, both targets
  curves/iter_fit.csv   the observation table, flat
  iter_policy.json      ONLY the chosen numbers, in the shape Stage C consumes
"""

import hashlib
import json
import os
import sys

import numpy as np

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

from labels import FAIL_CELLS                                     # noqa: E402

EXP031_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_031")
OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_032")
DATA_PATH = os.path.expanduser("~/SungjinPark/data/train.csv")
TRAIN_CSV_SHA = "d2081186b458b49f60b082be480c273135833e15ba59a76d033af28bcf8763ff"

BLOCK_UNITS = [2021, 2022, 2023]        # the ONLY units read
BLOCK_SEEDS = [42, 7]
UNIT_DIR = {2021: OUT_DIR, 2022: EXP031_DIR, 2023: EXP031_DIR}
UNIT_EXP = {2021: "exp_032", 2022: "exp_031", 2023: "exp_031"}

# row counts, measured on the server from train.csv (not transcribed from prose)
N_ROWS = {
    2021: {"full": 432501, "mode": 432317, "nan_stack_pct": 48.931},
    2022: {"full": 653728, "mode": 653545, "nan_stack_pct": 32.372},
    2023: {"full": 870752, "mode": 870542, "nan_stack_pct": 24.304},
    2024: {"full": 1116277, "mode": 1115798, "nan_stack_pct": 18.958},
}
DEPLOY = {"full": 1369784, "mode": 1369186, "nan_stack_pct": 15.450,
          "what": "train on EVERY row with season <= 2024 under the exp_004 "
                  "F-exclusion mask; this is what a submission build fits"}

# compute guard.  Not a modelling choice — a wall-clock one.  If a form's
# extrapolation exceeds these, the number actually used is the guard and BOTH
# numbers are recorded, loudly.
GUARD = {"cell": 2000, "stack": 2500}


def sha256_file(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
#  the four forms.  each returns (predict_fn, n_params, description)           #
# --------------------------------------------------------------------------- #
def fit_loglog(x, y):
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    return (lambda xq: a + b * np.asarray(xq, float)), 2, \
        {"a_intercept": a, "b_slope": b,
         "reading": f"argmin proportional to n_train ** {b:.4f}"}


def fit_constant(x, y):
    a = float(np.mean(y))
    return (lambda xq: a + 0.0 * np.asarray(xq, float)), 1, \
        {"a_intercept": a, "b_slope": 0.0,
         "reading": f"argmin does not move with n_train; exp(a) = "
                    f"{np.exp(a):.1f} rounds at every size"}


def fit_proportional(x, y):
    a = float(np.mean(y - x))
    return (lambda xq: a + np.asarray(xq, float)), 1, \
        {"a_intercept": a, "b_slope": 1.0,
         "reading": "THE OLD exp_030 POLICY: argmin scales with the row ratio"}


def fit_saturating(x, y):
    """argmin = A * n / (B + n), fitted in log space.

    For a fixed B the model is log A + log(n/(B+n)), whose only free parameter
    enters additively, so A is the mean residual.  B is searched on a log grid
    spanning a hundredth of the smallest observed n to a hundred times the
    largest, which brackets both the "already saturated" and the "still linear"
    ends of the family.
    """
    n = np.exp(np.asarray(x, float))
    best = None
    for logB in np.linspace(np.log(n.min()) - 4.6, np.log(n.max()) + 4.6, 4001):
        B = float(np.exp(logB))
        c = np.log(n / (B + n))
        logA = float(np.mean(y - c))
        sse = float(np.sum((logA + c - y) ** 2))
        if best is None or sse < best[0]:
            best = (sse, logA, B)
    _, logA, B = best
    return (lambda xq: logA + np.log(np.exp(np.asarray(xq, float))
                                     / (B + np.exp(np.asarray(xq, float))))), 2, \
        {"log_A_ceiling": logA, "A_ceiling_rounds": float(np.exp(logA)),
         "B_half_saturation_rows": B,
         "reading": f"argmin approaches a ceiling of {np.exp(logA):.0f} rounds; "
                    f"half of it is reached at {B:,.0f} training rows"}


FORMS = {"loglog": fit_loglog, "constant": fit_constant,
         "proportional": fit_proportional, "saturating": fit_saturating}


def fit_all(obs_x, obs_y, sizes_x, sizes_ymean, q_points):
    """obs_* are the per-seed observations; sizes_* are the per-size means."""
    out = {}
    for name, fn in FORMS.items():
        pred, npar, info = fn(np.asarray(obs_x, float), np.asarray(obs_y, float))
        res = np.asarray(obs_y, float) - pred(obs_x)
        sse = float(np.sum(res ** 2))
        tss = float(np.sum((np.asarray(obs_y, float)
                            - np.mean(obs_y)) ** 2))
        # leave-one-SIZE-out
        loso, tot = {}, 0.0
        for i, xs in enumerate(sizes_x):
            keep = [j for j in range(len(obs_x))
                    if abs(obs_x[j] - xs) > 1e-9]
            p2, _, _ = fn(np.asarray([obs_x[j] for j in keep], float),
                          np.asarray([obs_y[j] for j in keep], float))
            err = float(p2([xs])[0] - sizes_ymean[i])
            kind = ("interpolation (middle size held out)"
                    if 0 < i < len(sizes_x) - 1 else
                    "EXTRAPOLATION (end size held out)")
            loso[str(int(round(np.exp(xs))))] = {
                "predicted_log": float(p2([xs])[0]),
                "observed_mean_log": float(sizes_ymean[i]),
                "abs_err_log": abs(err),
                "ratio_pred_over_obs": float(np.exp(err)),
                "fold_kind": kind}
            tot += abs(err)
        out[name] = {
            "n_params": npar, "params": info,
            "residual_log_by_obs": [float(v) for v in res],
            "residual_ratio_by_obs": [float(np.exp(v)) for v in res],
            "max_abs_residual_log": float(np.max(np.abs(res))),
            "max_residual_ratio": float(np.exp(np.max(np.abs(res)))),
            "rmse_log": float(np.sqrt(sse / len(res))),
            "r2_in_log_space": (float(1.0 - sse / tss) if tss > 0 else None),
            "loso_by_held_out_size": loso,
            "LOSO_TOTAL_ABS_ERR_LOG_SELECTION_CRITERION": tot,
            "predictions": {str(k): {"log": float(pred([v])[0]),
                                     "rounds_unrounded": float(np.exp(pred([v])[0])),
                                     "rounds": int(round(float(np.exp(pred([v])[0])))) }
                            for k, v in q_points.items()},
        }
    return out


# --------------------------------------------------------------------------- #
def main():
    assert sha256_file(DATA_PATH) == TRAIN_CSV_SHA
    obs = []
    for V in BLOCK_UNITS:
        for s in BLOCK_SEEDS:
            p = os.path.join(UNIT_DIR[V], "res", f"res_C5B3_u{V}_seed{s}_st3.json")
            with open(p) as f:
                r = json.load(f)
            assert r["exp_id"] == UNIT_EXP[V], (p, r["exp_id"])
            assert r["stage"] == 3 and r["self_early_stop"] is True, p
            assert r["sha256"]["train.csv"] == TRAIN_CSV_SHA, p
            assert not r["any_argmin_at_cap"], \
                ("ARGMIN AT CAP — this unit is INVALID, re-run with a larger "
                 "cap before fitting anything to it", V, s, r["cap"])
            assert r["n_train_full"] == N_ROWS[V]["full"], (p, r["n_train_full"])
            assert r["n_train_mode"] == N_ROWS[V]["mode"], (p, r["n_train_mode"])
            obs.append({
                "unit": V, "seed": s, "cap": r["cap"],
                "n_full": r["n_train_full"], "n_mode": r["n_train_mode"],
                "nan_stack_pct": r["nan_stack_rows_in_train_pct"],
                "stack_argmin": int(r["stack_argmin_iter"]),
                "cell_argmin": {c: int(r["cell_argmin_iter"][c]) for c in FAIL_CELLS},
                "brier_full": r["brier_full"], "minutes": r["minutes"],
                "res_file": p, "pred_sha256": r["pred_sha256"]})
    assert len(obs) == 6, len(obs)

    q = {"fold_2024": None, "deploy_le2024": None}
    targets = {}
    for tgt in ["stacker"] + list(FAIL_CELLS):
        key = "full" if tgt == "stacker" else "mode"
        xs = [np.log(o["n_" + key]) for o in obs]
        ys = [np.log(o["stack_argmin"] if tgt == "stacker"
                     else o["cell_argmin"][tgt]) for o in obs]
        sizes = sorted({o["n_" + key] for o in obs})
        sx = [np.log(v) for v in sizes]
        sy = [float(np.mean([ys[i] for i in range(len(obs))
                             if obs[i]["n_" + key] == v])) for v in sizes]
        qp = {"fold_2024": np.log(N_ROWS[2024][key]),
              "deploy_le2024": np.log(DEPLOY[key])}
        fits = fit_all(xs, ys, sx, sy, qp)

        # is the seed-mean of the three sizes a straight line in log-log?
        # chord through the two END sizes, evaluated at the MIDDLE size.
        slope_chord = (sy[2] - sy[0]) / (sx[2] - sx[0])
        mid_pred = sy[0] + slope_chord * (sx[1] - sx[0])
        curvature = float(sy[1] - mid_pred)
        # how noisy is a single size, for scale?
        seed_spread = {str(int(v)): float(abs(np.diff(
            [ys[i] for i in range(len(obs)) if obs[i]["n_" + key] == v])[0]))
            for v in sizes}
        lev = {k: float((v - max(sx)) / (max(sx) - min(sx))) for k, v in qp.items()}

        best = min(FORMS,
                   key=lambda m: (fits[m]["LOSO_TOTAL_ABS_ERR_LOG_SELECTION_CRITERION"],
                                  fits[m]["n_params"], m != "constant"))
        raw = fits[best]["predictions"]["fold_2024"]["rounds"]
        guard = GUARD["stack" if tgt == "stacker" else "cell"]
        used = int(min(raw, guard))
        raw_dep = fits[best]["predictions"]["deploy_le2024"]["rounds"]

        targets[tgt] = {
            "x_variable": ("n_train_full (the stacker fits the full pool)"
                           if tgt == "stacker" else
                           "n_train_mode (cell models fit the mode pool)"),
            "observations": [{"unit": obs[i]["unit"], "seed": obs[i]["seed"],
                              "n": obs[i]["n_" + key],
                              "argmin": int(round(np.exp(ys[i]))),
                              "log_n": xs[i], "log_argmin": ys[i],
                              "nan_stack_pct": obs[i]["nan_stack_pct"]}
                             for i in range(len(obs))],
            "size_means_log_argmin": {str(int(sizes[i])): sy[i]
                                      for i in range(len(sizes))},
            "size_means_argmin": {str(int(sizes[i])): float(np.exp(sy[i]))
                                  for i in range(len(sizes))},
            "straightness_of_the_three_size_means": {
                "what": "chord through the two END sizes, evaluated at the "
                        "MIDDLE size, minus the middle size's own mean.  Zero "
                        "means the three points are collinear in log-log.",
                "chord_slope": float(slope_chord),
                "curvature_log": curvature,
                "curvature_as_ratio": float(np.exp(curvature)),
                "seed_spread_log_within_each_size": seed_spread,
                "max_seed_spread_log": float(max(seed_spread.values())),
                "curvature_over_max_seed_spread": (
                    abs(curvature) / max(seed_spread.values())
                    if max(seed_spread.values()) > 0 else None),
                "verdict": ("curvature is SMALLER than the seed noise — the "
                            "three points are consistent with a straight line, "
                            "but that is not evidence FOR one"
                            if abs(curvature) <= max(seed_spread.values()) else
                            "curvature EXCEEDS the seed noise — the three points "
                            "are NOT on a straight line in log-log"),
            },
            "extrapolation_leverage_in_log_x_units": {
                "what": "(log n_target - log n_largest_observed) / (log n_largest "
                        "- log n_smallest).  0 = at the edge of the data, 1 = as "
                        "far beyond the data as the data itself is wide.",
                **lev},
            "fits": fits,
            "SELECTED_FORM": best,
            "selected_by": "lowest leave-one-SIZE-out total absolute error in "
                           "log space; ties to fewer parameters, then constant",
            "loso_ranking": sorted(
                FORMS, key=lambda m:
                fits[m]["LOSO_TOTAL_ABS_ERR_LOG_SELECTION_CRITERION"]),
            "fold_2024_iter_unguarded": raw,
            "fold_2024_iter_USED": used,
            "compute_guard": guard,
            "guard_fired": bool(raw > guard),
            "deploy_le2024_iter_unguarded": raw_dep,
            "deploy_le2024_iter_RECORD_ONLY": int(min(raw_dep, guard)),
            "deploy_guard_fired": bool(raw_dep > guard),
        }

    policy = {
        "exp_id": "exp_032",
        "for_stage_C_fold_2024": {
            "cell_iter": {c: targets[c]["fold_2024_iter_USED"] for c in FAIL_CELLS},
            "stack_iter": targets["stacker"]["fold_2024_iter_USED"],
            "sum_cell_iter": int(sum(targets[c]["fold_2024_iter_USED"]
                                     for c in FAIL_CELLS)),
        },
        "for_stage_E_deploy_RECORD_ONLY": {
            "cell_iter": {c: targets[c]["deploy_le2024_iter_RECORD_ONLY"]
                          for c in FAIL_CELLS},
            "stack_iter": targets["stacker"]["deploy_le2024_iter_RECORD_ONLY"],
            "cell_iter_unguarded": {c: targets[c]["deploy_le2024_iter_unguarded"]
                                    for c in FAIL_CELLS},
            "stack_iter_unguarded": targets["stacker"]["deploy_le2024_iter_unguarded"],
            "warning": "NOT BUILT IN THIS EXPERIMENT.  Recorded so a submission "
                       "build has a written basis instead of a guess.",
        },
        "exp030_policy_for_contrast": {
            "what": "what the OLD row-ratio rule would have said for fold_2024",
            "cell_iter": {"is_mid_only": 577, "is_rev_only": 1346,
                          "is_both": 449, "is_wayoff_true": 897},
            "stack_iter": 481,
            "source": "exp_030 res_C5B3_u2024_seed42_st2.json cell_iter_used / "
                      "stacker.iter_used — read, not transcribed from prose"},
    }
    out = {
        "exp_id": "exp_032", "stage": "B (fit) + E (deploy record)",
        "what_was_read": "stage-3 result files of block units 2021, 2022, 2023 "
                         "x seeds 42, 7.  fold_2024 and fold_2023 validation "
                         "curves were NOT opened.",
        "block_units": BLOCK_UNITS, "block_seeds": BLOCK_SEEDS,
        "row_counts": N_ROWS, "deploy_row_counts": DEPLOY,
        "observations_raw": obs,
        "selection_rule": "leave-one-SIZE-out, lowest total absolute error in "
                          "log space.  Declared in fit_iter.py's docstring "
                          "before unit 2021 finished running.",
        "compute_guard": GUARD,
        "targets": targets,
        "policy": policy,
        "source_sha256": {f: sha256_file(os.path.join(SRC_DIR, f))
                          for f in sorted(os.listdir(SRC_DIR))
                          if f.endswith(".py")},
        "train_csv_sha256": TRAIN_CSV_SHA,
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__},
    }
    os.makedirs(os.path.join(OUT_DIR, "curves"), exist_ok=True)
    with open(os.path.join(OUT_DIR, "curves", "iter_fit.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(OUT_DIR, "iter_policy.json"), "w") as f:
        json.dump(policy, f, indent=1)
    with open(os.path.join(OUT_DIR, "curves", "iter_fit.csv"), "w") as f:
        f.write("target,unit,seed,n_train,argmin,log_n,log_argmin,nan_stack_pct\n")
        for tgt in targets:
            for o in targets[tgt]["observations"]:
                f.write(f"{tgt},{o['unit']},{o['seed']},{o['n']},{o['argmin']},"
                        f"{o['log_n']:.6f},{o['log_argmin']:.6f},"
                        f"{o['nan_stack_pct']}\n")

    for tgt in ["stacker"] + list(FAIL_CELLS):
        t = targets[tgt]
        print(f"\n=== {tgt} ({t['x_variable'].split(' ')[0]}) ===")
        print("  observed:", {o["n"]: o["argmin"] for o in t["observations"]}
              if False else [(o["unit"], o["seed"], o["argmin"])
                             for o in t["observations"]])
        print("  size means:", {k: round(v, 1)
                                for k, v in t["size_means_argmin"].items()})
        st = t["straightness_of_the_three_size_means"]
        print(f"  curvature {st['curvature_log']:+.4f} log "
              f"(ratio {st['curvature_as_ratio']:.3f}), max seed spread "
              f"{st['max_seed_spread_log']:.4f} -> {st['verdict'][:60]}")
        print("  LOSO:", {m: round(t["fits"][m]
                          ["LOSO_TOTAL_ABS_ERR_LOG_SELECTION_CRITERION"], 4)
                          for m in FORMS})
        print(f"  SELECTED {t['SELECTED_FORM']}  -> fold_2024 "
              f"{t['fold_2024_iter_unguarded']} (used {t['fold_2024_iter_USED']}"
              f"{', GUARD FIRED' if t['guard_fired'] else ''}), deploy "
              f"{t['deploy_le2024_iter_unguarded']}"
              f"{' GUARD FIRED' if t['deploy_guard_fired'] else ''}")
    print("\nSTAGE C COMMAND ARGS:")
    print("  --cell-iter '" + json.dumps(policy["for_stage_C_fold_2024"]
                                         ["cell_iter"]) + "'"
          f"  --stack-iter {policy['for_stage_C_fold_2024']['stack_iter']}")
    print("  sum cell iter:", policy["for_stage_C_fold_2024"]["sum_cell_iter"])


if __name__ == "__main__":
    main()
