"""exp_052 stage 0.5 — reachability probe.  P-A projection, P-B mini
distillation, P-C dilution.  Approved 2026-08-15.  STAGE 1 IS NOT OPENED:
no lambda grid, no full teacher OOF generation, fold_2024 never read.

WHY THE PROBE IS SHAPED THIS WAY (coordinator's correction, recorded)
---------------------------------------------------------------------
Distillation's optimum is E[t|x]; a calibrated teacher makes that the same
point as E[y|x].  So soft labels do not add information -- they cut the
TARGET'S NOISE.  hard-label noise is Var(y|x) = p(1-p) ~ 0.2498.  soft-label
noise is Var(t|x) = (1-R^2)*Var(t).  Because the teacher's probabilities live
in a narrow band, Var(t) is small, so a LOW R^2 does NOT imply a useless soft
label.  R^2 alone can close the axis wrongly.  What must be measured is the
ABSOLUTE residual variance and, decisively, whether distillation actually wins.

THE TEACHER
-----------
arm D (93 cols) fitted on season <= 2022, read at its stage-0 argmin of 700
rounds.  Stage 0 did not persist the prediction vector, so the model is refit
here at iterations=700 and GATED: the Brier of the recovered vector on all
245,525 season-2023 rows must equal the value stage 0's curve recorded at
round 700 to < 5e-6.  (A fixed-seed Plain booster's first 700 trees ARE the
700-round model -- exp_047 measured staged prefixes bit-identical to
independent fits; the gate is what makes that a checked claim here.)
The vector is saved with its sha256 so nothing downstream is transcribed.

THE PLANE
---------
Everything below happens INSIDE season 2023, split in TIME order (train.csv
row order is chronological -- stage-0 diagnostic D5 confirmed season and
row_id monotone).  Front 60% trains, back 40% evaluates.  No random KFold.
The teacher never saw 2023 at all (it was fitted on <= 2022), so its 2023
probabilities are out-of-sample for the teacher.

  ⚠️ ONE OPTIMISM, RECORDED: the teacher's 700-round reading point was chosen
  on ALL 245,525 season-2023 rows in stage 0, and the back 40% is a subset of
  those.  The teacher is therefore very slightly flattered on this plane.  It
  is not re-chosen here.

P-B ARMS -- one variable, the target
------------------------------------
  P0  target = hard label y                      (control)
  P1  target = teacher probability t             (pure soft)
  P2  target = 0.5*t + 0.5*y                     (mixed)
Input is the SAME student 67 columns in all three.  Loss is `CrossEntropy` in
all three -- for 0/1 targets CrossEntropy IS Logloss, so holding it fixed
keeps the loss function from becoming a second variable.  Same seed, same
params, same cap; each arm reads its own argmin on the back 40%'s REAL labels,
which is the same generous-to-all reading stage 0 used.

P-C -- rows without a soft label
--------------------------------
A row whose train<->trackman match failed has NO teacher soft label in stage 1.
This file reproduces that: for such rows P1 and P2 both fall back to the hard
label y.  Deltas are reported twice -- over all back-40% rows, and over the
has_tm subset alone -- so the dilution share is visible.

READING (pre-registered by the coordinator, not applied as a threshold)
-----------------------------------------------------------------------
P1 or P2 beating P0 is direct evidence distillation works on this scale.
Neither beating P0 means stage 1 is not opened.  Seed 42 alone: direction and
size only, never an adoption call.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

BASE = os.path.expanduser("~/SungjinPark/experiments/exp_052")
E027 = os.path.expanduser("~/SungjinPark/experiments/exp_027/src/train.py")
E027_SHA = "e4b308a5f6e4564ed43aed6715d05c0aa617b5e9d22e1bfd51e71dd127416f0f"
TARGET = "control_success"
BASE_CAT = ["top_bottom", "game_type", "base_state"]
TM_LAB = ["tagged_pitch_type", "auto_pitch_type", "pitch_type_group"]
REF_BRIER = 0.2498069
N_THREADS = 6
SEED = 42
TEACHER_ROUNDS = 700
TEACHER_BRIER_EXPECT = 0.2472790209422709      # stage 0 res/g0b_armD.json
GATE_TOL = 5e-6
FRONT_FRAC = 0.60
PROBE_CAP = 3000
PROJ_CAP = 2000
EVAL_PERIOD = 25
EPS = 1e-6


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def pts(brier_reduction):
    """Brier REDUCTION (positive = better) -> leaderboard score points."""
    return round(brier_reduction / REF_BRIER * 100000.0, 5)


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def load_cb_base():
    import importlib.util
    h = hashlib.sha256()
    with open(E027, "rb") as f:
        h.update(f.read())
    assert h.hexdigest() == E027_SHA, "exp_027 train.py changed"
    spec = importlib.util.spec_from_file_location("e027", E027)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.CB_BASE["has_time"] is True
    assert m.CB_BASE["thread_count"] == N_THREADS
    return dict(m.CB_BASE)


def frame(df, feats, cat_native):
    """exp_027 build_frame_native."""
    data = {}
    for c in feats:
        if c in cat_native:
            s = df[c]
            assert s.notna().all(), c
            data[c] = s.astype(str).to_numpy()
        else:
            data[c] = pd.to_numeric(df[c], errors="raise").to_numpy(np.float64)
    return pd.DataFrame(data, columns=feats)


def main():
    t0 = time.time()
    import catboost as cb
    CB = load_cb_base()
    cols = json.load(open(os.path.join(BASE, "work", "cols.json")))
    student67 = cols["base47"] + cols["oof4"] + cols["ins16"]
    teacher93 = student67 + cols["f14"] + cols["tm12"]
    assert len(student67) == 67 and len(teacher93) == 93

    df = pd.read_parquet(os.path.join(BASE, "work", "design_fold2023.parquet"))
    is_tr = df["_is_train"].to_numpy(bool)
    y_all = df[TARGET].to_numpy(np.float64)
    has_tm_all = df["has_tm"].to_numpy(np.float64) > 0.5
    out = {"exp_id": "exp_052", "stage": 0.5, "seed": SEED,
           "stage1_opened": False, "fold_2024_opened": False}

    # =====================================================================
    # teacher: refit arm D at its stage-0 argmin, gate, persist
    # =====================================================================
    pred_path = os.path.join(BASE, "work", "teacher_D_2023_at700.npy")
    if os.path.exists(pred_path):
        t_prob = np.load(pred_path)
        log("teacher prediction reused from disk")
    else:
        log("teacher refit @", TEACHER_ROUNDS)
        catD = [c for c in BASE_CAT + TM_LAB if c in teacher93]
        FD = frame(df, teacher93, catD)
        ci = [teacher93.index(c) for c in catD]
        ptr = cb.Pool(FD.loc[is_tr], label=y_all[is_tr], cat_features=ci)
        pva = cb.Pool(FD.loc[~is_tr], cat_features=ci)
        mD = cb.CatBoostClassifier(**dict(CB, random_seed=SEED,
                                          iterations=TEACHER_ROUNDS))
        mD.fit(ptr)
        assert mD.tree_count_ == TEACHER_ROUNDS
        t_prob = mD.predict_proba(pva)[:, 1].astype(np.float64)
        del FD, ptr, pva, mD
        np.save(pred_path, t_prob)
    y23 = y_all[~is_tr]
    got = float(np.mean((t_prob - y23) ** 2))
    out["teacher"] = {
        "arm": "D(93col)", "rounds": TEACHER_ROUNDS,
        "brier_recovered": got, "brier_stage0_recorded": TEACHER_BRIER_EXPECT,
        "abs_diff": abs(got - TEACHER_BRIER_EXPECT), "gate_tol": GATE_TOL,
        "gate_pass": bool(abs(got - TEACHER_BRIER_EXPECT) < GATE_TOL),
        "pred_sha256": sha256_bytes(t_prob.tobytes()),
        "pred_path": pred_path,
        "argmin_chosen_on": ("all 245,525 season-2023 rows in stage 0 -- the "
                             "back-40% eval set is a subset (recorded optimism)"),
    }
    assert out["teacher"]["gate_pass"], out["teacher"]
    log("teacher gate ok, brier", got)

    # =====================================================================
    # the 2023 plane, split in TIME order
    # =====================================================================
    d23 = df.loc[~is_tr].reset_index(drop=True)
    n23 = len(d23)
    assert n23 == 245525, n23
    n_front = int(round(n23 * FRONT_FRAC))
    front = np.zeros(n23, bool)
    front[:n_front] = True
    back = ~front
    has_tm = has_tm_all[~is_tr]
    out["plane"] = {
        "n_2023": n23, "n_front": int(front.sum()), "n_back": int(back.sum()),
        "split": "time order (train.csv row order is chronological)",
        "has_tm_pct_front": round(100 * float(has_tm[front].mean()), 4),
        "has_tm_pct_back": round(100 * float(has_tm[back].mean()), 4),
        "base_rate_front": round(float(y23[front].mean()), 6),
        "base_rate_back": round(float(y23[back].mean()), 6),
    }

    catS = [c for c in BASE_CAT if c in student67]
    FS = frame(d23, student67, catS)
    cis = [student67.index(c) for c in catS]
    pool_f = cb.Pool(FS.loc[front], cat_features=cis)
    pool_b = cb.Pool(FS.loc[back], cat_features=cis)

    # =====================================================================
    # P-A: can the student 67 columns express the teacher's probability?
    # =====================================================================
    log("P-A projection")
    t_b = t_prob[back]
    pa = {"note": ("holdout R^2 is the record; in-sample is reference only. "
                   "Var(t|x) is the holdout residual MSE, measured not derived.")}
    for space in ("prob", "logit"):
        if space == "prob":
            z = t_prob.copy()
        else:
            c = np.clip(t_prob, EPS, 1 - EPS)
            z = np.log(c / (1 - c))
        pool_ft = cb.Pool(FS.loc[front], label=z[front], cat_features=cis)
        reg = cb.CatBoostRegressor(**dict(CB, loss_function="RMSE",
                                          random_seed=SEED,
                                          iterations=PROJ_CAP))
        reg.fit(pool_ft)
        curve = {}
        for k, p in enumerate(reg.staged_predict(
                pool_b, ntree_start=0, ntree_end=PROJ_CAP,
                eval_period=EVAL_PERIOD, thread_count=N_THREADS)):
            curve[(k + 1) * EVAL_PERIOD] = float(
                np.mean((p.astype(np.float64) - z[back]) ** 2))
        best_n = min(curve, key=curve.get)
        mse = curve[best_n]
        var_b = float(np.var(z[back]))
        p_in = reg.predict(pool_f, ntree_end=best_n).astype(np.float64)
        mse_in = float(np.mean((p_in - z[front]) ** 2))
        # holdout MSE restricted to rows that would actually carry a soft label
        p_b = reg.predict(pool_b, ntree_end=best_n).astype(np.float64)
        m = has_tm[back]
        pa[space] = {
            "argmin_iter": int(best_n),
            "argmin_at_cap_INVALID": bool(best_n == PROJ_CAP),
            "Var_t_all2023": float(np.var(z)),
            "Var_t_back40": var_b,
            "Var_t_given_x_holdout_MSE": mse,
            "RMSE_holdout": float(np.sqrt(mse)),
            "R2_holdout": 1.0 - mse / var_b,
            "R2_in_sample_reference": 1.0 - mse_in / float(np.var(z[front])),
            "Var_t_given_x_on_has_tm_rows": float(
                np.mean((p_b[m] - z[back][m]) ** 2)),
        }
        if space == "prob":
            pa[space]["hard_label_noise_Var_y_given_x"] = REF_BRIER
            pa[space]["noise_ratio_soft_over_hard"] = mse / REF_BRIER
            pa[space]["teacher_prob_range"] = [float(t_prob.min()),
                                               float(t_prob.max())]
            pa[space]["teacher_prob_pctiles"] = {
                str(q): float(np.percentile(t_prob, q))
                for q in (1, 5, 25, 50, 75, 95, 99)}
        del reg, pool_ft
    out["P_A"] = pa
    log("P-A done R2_prob", round(pa["prob"]["R2_holdout"], 5),
        "noise ratio", round(pa["prob"]["noise_ratio_soft_over_hard"], 5))

    # =====================================================================
    # P-B: mini distillation.  ONE variable: the target.
    # =====================================================================
    log("P-B mini distillation")
    soft = np.where(has_tm, t_prob, y23)                  # P-C fallback
    targets = {"P0_hard": y23,
               "P1_soft": soft,
               "P2_mix50": np.where(has_tm, 0.5 * t_prob + 0.5 * y23, y23)}
    pb = {}
    for name, tgt in targets.items():
        pool_ft = cb.Pool(FS.loc[front], label=tgt[front], cat_features=cis)
        m = cb.CatBoostClassifier(**dict(CB, loss_function="CrossEntropy",
                                         random_seed=SEED,
                                         iterations=PROBE_CAP))
        m.fit(pool_ft)
        curve, curve_tm = {}, {}
        msk = has_tm[back]
        for k, p in enumerate(m.staged_predict(
                pool_b, prediction_type="Probability", ntree_start=0,
                ntree_end=PROBE_CAP, eval_period=EVAL_PERIOD,
                thread_count=N_THREADS)):
            v = p[:, 1].astype(np.float64)
            n = (k + 1) * EVAL_PERIOD
            curve[n] = float(np.mean((v - y23[back]) ** 2))
            curve_tm[n] = float(np.mean((v[msk] - y23[back][msk]) ** 2))
        best_n = min(curve, key=curve.get)
        pb[name] = {
            "argmin_iter": int(best_n),
            "argmin_at_cap_INVALID": bool(best_n == PROBE_CAP),
            "brier_back40_all": curve[best_n],
            "brier_back40_has_tm_at_same_iter": curve_tm[best_n],
            "target_mean": float(tgt[front].mean()),
            "target_var": float(np.var(tgt[front])),
        }
        log(" ", name, "argmin", best_n, "brier", f"{curve[best_n]:.8f}")
        del m, pool_ft

    b0 = pb["P0_hard"]["brier_back40_all"]
    b0t = pb["P0_hard"]["brier_back40_has_tm_at_same_iter"]
    for name in ("P1_soft", "P2_mix50"):
        pb[name]["delta_vs_P0_all"] = {
            "brier": b0 - pb[name]["brier_back40_all"],
            "score_points": pts(b0 - pb[name]["brier_back40_all"])}
        pb[name]["delta_vs_P0_has_tm_only"] = {
            "brier": b0t - pb[name]["brier_back40_has_tm_at_same_iter"],
            "score_points": pts(
                b0t - pb[name]["brier_back40_has_tm_at_same_iter"])}
    pb["reading"] = {
        "pre_registered": ("P1 or P2 beating P0 = distillation works on this "
                           "scale; neither beating P0 = stage 1 stays shut"),
        "P1_beats_P0": bool(pb["P1_soft"]["brier_back40_all"] < b0),
        "P2_beats_P0": bool(pb["P2_mix50"]["brier_back40_all"] < b0),
        "any_argmin_at_cap": any(pb[k]["argmin_at_cap_INVALID"]
                                 for k in targets),
        "caveat_scale": ("~147k training rows here vs ~1.37M at deployment. "
                         "Distillation gains SHRINK with more data, so a "
                         "positive here must not be carried across as a "
                         "deployment expectation; a negative here is the "
                         "stronger signal."),
        "caveat_seed": "seed 42 alone -- direction and size, not an adoption call",
    }
    out["P_B"] = pb

    # =====================================================================
    # P-C: dilution
    # =====================================================================
    out["P_C"] = {
        "rows_without_soft_label_pct_back40": round(
            100 * float((~has_tm[back]).mean()), 4),
        "rows_without_soft_label_pct_front60": round(
            100 * float((~has_tm[front]).mean()), 4),
        "rows_without_soft_label_pct_train_all_seasons": 13.174,
        "policy": "no soft label -> hard label y (what stage 1 would do)",
        "note": ("delta_vs_P0_has_tm_only isolates the undiluted effect; the "
                 "gap between it and delta_vs_P0_all IS the dilution share"),
    }

    out["elapsed_min"] = round((time.time() - t0) / 60, 2)
    with open(os.path.join(BASE, "res", "stage05_probe.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(json.dumps({"P_A_prob": out["P_A"]["prob"],
                      "P_B": {k: v for k, v in pb.items()}}, indent=1,
                     ensure_ascii=False)[:4000])
    log("done", out["elapsed_min"], "min")


if __name__ == "__main__":
    main()
