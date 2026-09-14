"""exp_044 — the hyper-parameter search for ONE model of the C5_B3 pipeline.

Two modes, one engine:
  --mode cell --cell is_mid_only     the four binary cell models
  --mode stacker                     the stacker that consumes their OOF columns

WHERE IT IS MEASURED
--------------------
fold_2023 only (train <= 2022 -> valid 2023).  fold_2024 is never opened by this
file; the promotion runs are a separate program.  The pools are the ones
b3_cat.py actually fits: celldata.py for a cell, stackdata.py for the stacker,
both sha-gated against the pipeline source.

OBJECTIVE
---------
cell     : valid-2023 logloss of that cell's own binary target.
stacker  : valid-2023 valid_all Brier of control_success — the stacker's output
           is the pipeline's prediction, so its own metric is the real one.
Both are read at the iteration the early-stopping detector picks, and
`iterations` is therefore NOT a searched dimension.

CENSORING
---------
A trial whose detector never fired before the cap is `censored`: a better
iteration could lie past the edge, so its number is not a measurement.  Those
are re-run at `--recap` and the re-run is what ranks them.  A trial still
censored at the recap is excluded from the pick and reported.

SEARCH SPACE
------------
Two Optuna studies, because one study cannot hold two different distributions
under one parameter name:
  narrow  the region exp_042's top 10 clustered in, with room around it
  wide    exp_042's original space, unchanged
asked according to a fixed index schedule (see prereg.json).  In --mode stacker
every trial is wide: the stacker's inputs are four probabilities plus the
features, and the cell region has no claim on it.

SEEDS
-----
Screen all trials at fit seed 42; re-fit the top `--n-confirm` at seeds 7 and
2024 and re-rank by the 3-seed mean.  Rank 1 of that re-ranking is the pick.

Usage:
    python hpo044.py --mode cell --cell is_wayoff_true --workers 9 --threads 2
    python hpo044.py --mode stacker --workers 9 --threads 2
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_044")

FIXED = {
    "loss_function": "Logloss", "task_type": "CPU",
    "boosting_type": "Plain", "bootstrap_type": "Bernoulli", "has_time": True,
    "allow_writing_files": False, "verbose": False,
}

# the exp_042 search winner = the settings the phase-3 control arm puts on all
# five models.  Enqueued as trial 0 of every search so the control is measured
# on the same ruler as the candidates.
EXP042_WINNER = {
    "depth": 8,
    "l2_leaf_reg": 37.829181614725194,
    "learning_rate": 0.015742633699761966,
    "subsample": 0.8420564481615168,
    "rsm": 0.7460509492068245,
    "random_strength": 0.5267169564355272,
    "min_data_in_leaf": 11,
    "one_hot_max_size": 3,
}

SCREEN_SEED = 42
CONFIRM_SEEDS = [7, 2024]

_G = {}


# --------------------------------------------------------------------- spaces
def suggest_narrow(t):
    """exp_042's distribution SHAPES with clipped bounds (prereg.json)."""
    return {
        "depth": t.suggest_int("depth", 7, 10),
        "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 10.0, 80.0, log=True),
        "learning_rate": t.suggest_float("learning_rate", 0.008, 0.030, log=True),
        "subsample": t.suggest_float("subsample", 0.70, 0.95),
        "rsm": t.suggest_float("rsm", 0.45, 0.95),
        "random_strength": t.suggest_float("random_strength", 0.3, 1.3, log=True),
        "min_data_in_leaf": t.suggest_int("min_data_in_leaf", 1, 150, log=True),
        "one_hot_max_size": t.suggest_int("one_hot_max_size", 2, 10, log=True),
    }


def suggest_wide(t):
    """exp_042's original space, byte-for-byte the same bounds."""
    return {
        "depth": t.suggest_int("depth", 4, 10),
        "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 1.0, 100.0, log=True),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.10, log=True),
        "subsample": t.suggest_float("subsample", 0.6, 1.0),
        "rsm": t.suggest_float("rsm", 0.5, 1.0),
        "random_strength": t.suggest_float("random_strength", 0.5, 10.0, log=True),
        "min_data_in_leaf": t.suggest_int("min_data_in_leaf", 1, 200, log=True),
        "one_hot_max_size": t.suggest_int("one_hot_max_size", 2, 64, log=True),
    }


SPACES = {"narrow": suggest_narrow, "wide": suggest_wide}


def wide_indices(n_trials, frac):
    """Fixed, data-free schedule.  See prereg.json for the identical wording."""
    k = int(round(n_trials * frac))
    if k <= 0:
        return []
    return sorted({int(round((j + 0.5) * n_trials / k)) for j in range(k)})


# ---------------------------------------------------------------- worker side
def _init(D, threads, eval_metric):
    import catboost as cb
    _G["cb"] = cb
    _G["threads"] = threads
    _G["eval_metric"] = eval_metric
    _G["y_va"] = D["y_va"]
    _G["pool_tr"] = cb.Pool(D["F_tr"], label=D["y_tr"], cat_features=D["cat_idx"])
    _G["pool_va"] = cb.Pool(D["F_va"], label=D["y_va"], cat_features=D["cat_idx"])


def _run(job):
    """One fit.  Returns a record; never raises into the parent."""
    rid, params, seed, cap, od_wait = job
    cb = _G["cb"]
    t0 = time.time()
    rec = {"rid": rid, "params": dict(params), "seed": int(seed), "cap": int(cap),
           "od_wait": int(od_wait), "threads": _G["threads"],
           "eval_metric": _G["eval_metric"]}
    try:
        m = cb.CatBoostClassifier(**FIXED, **params, eval_metric=_G["eval_metric"],
                                  random_seed=int(seed), iterations=int(cap),
                                  thread_count=_G["threads"], od_type="Iter",
                                  od_wait=int(od_wait), use_best_model=False)
        m.fit(_G["pool_tr"], eval_set=_G["pool_va"])
        bi = int(m.best_iteration_)
        built = int(m.tree_count_)
        raw = m.predict(_G["pool_va"], prediction_type="RawFormulaVal",
                        ntree_start=0, ntree_end=bi + 1,
                        thread_count=_G["threads"])
        p = 1.0 / (1.0 + np.exp(-np.asarray(raw, dtype=np.float64)))
        y = _G["y_va"]
        pc = np.clip(p, 1e-15, 1 - 1e-15)
        rec.update({
            "ok": True,
            "best_iteration": bi,
            "trees_built": built,
            "censored": bool(built >= cap and bi + 1 + od_wait > cap),
            "logloss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
            "brier": float(np.mean((p - y) ** 2)),
            "pred_mean": float(p.mean()),
            "fit_sec": round(time.time() - t0, 2),
        })
    except Exception as e:                                          # noqa: BLE001
        rec.update({"ok": False, "error": f"{type(e).__name__}: {e}",
                    "fit_sec": round(time.time() - t0, 2)})
    return rec


# ------------------------------------------------------------------ main body
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["cell", "stacker"])
    ap.add_argument("--cell", default="")
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--n-trials", type=int, default=55)
    ap.add_argument("--n-confirm", type=int, default=15)
    ap.add_argument("--wide-frac", type=float, default=0.25)
    ap.add_argument("--cap", type=int, default=3300)
    ap.add_argument("--recap", type=int, default=6600)
    ap.add_argument("--od-wait", type=int, default=250)
    ap.add_argument("--stack-npz", default="")
    ap.add_argument("--stage1-json", default="")
    a = ap.parse_args()
    assert a.workers * a.threads <= 18, ("18 vCPU budget (2 reserved)",
                                         a.workers * a.threads)
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    t_start = time.time()
    if a.mode == "cell":
        assert a.cell, "--cell is required in cell mode"
        import celldata
        D = celldata.build(a.cell)
        D["n_tr"], D["n_va"] = D["n_train_mode"], D["n_valid_ok"]
        name = a.cell
        objective_key = "logloss"
        eval_metric = "Logloss"
        wide_frac = a.wide_frac
    else:
        assert a.stack_npz and a.stage1_json, "--stack-npz and --stage1-json"
        import stackdata
        D = stackdata.build(a.stack_npz, a.stage1_json)
        name = "stacker"
        objective_key = "brier"
        eval_metric = "BrierScore"
        wide_frac = 1.0                    # prereg: the stacker is wide only

    os.makedirs(os.path.join(OUT_DIR, "res"), exist_ok=True)
    print(f"[{time.strftime('%H:%M:%S')}] {name}: n_tr {D['n_tr']:,} "
          f"n_va {D['n_va']:,} n_feats {len(D['feats'])} "
          f"pos_rate_tr {D['pos_rate_train']:.6f} "
          f"pos_rate_va {D['pos_rate_valid']:.6f} "
          f"objective={objective_key} eval_metric={eval_metric}", flush=True)

    widx = set(wide_indices(a.n_trials, wide_frac))
    schedule = ["wide" if i in widx else "narrow" for i in range(a.n_trials)]
    schedule[0] = "wide" if wide_frac >= 1.0 else "narrow"   # trial 0 = enqueued
    print(f"[{time.strftime('%H:%M:%S')}] schedule: "
          f"{schedule.count('narrow')} narrow / {schedule.count('wide')} wide; "
          f"wide indices {sorted(widx)}", flush=True)

    studies = {}
    for s in ("narrow", "wide"):
        if s in schedule:
            studies[s] = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=42))
    studies[schedule[0]].enqueue_trial(dict(EXP042_WINNER))

    trials_path = os.path.join(OUT_DIR, "res", f"trials_{name}.jsonl")
    tf = open(trials_path, "w")

    def flush(rec):
        tf.write(json.dumps(rec) + "\n")
        tf.flush()
        os.fsync(tf.fileno())

    ctx = mp.get_context("fork")
    pool = ctx.Pool(a.workers, initializer=_init,
                    initargs=(D, a.threads, eval_metric))

    # ------------------------------------------------------------ screen stage
    records, done = [], 0
    while done < a.n_trials:
        b = min(a.workers, a.n_trials - done)
        idxs = list(range(done, done + b))
        asked = [(i, schedule[i], studies[schedule[i]].ask()) for i in idxs]
        jobs = [(i, SPACES[sp](t), SCREEN_SEED, a.cap, a.od_wait)
                for i, sp, t in asked]
        outs = {r["rid"]: r for r in pool.map(_run, jobs)}
        for i, sp, t in asked:
            r = outs[i]
            assert r["rid"] == i, (r["rid"], i)
            if r["ok"]:
                studies[sp].tell(t, r[objective_key])
            else:
                studies[sp].tell(t, state=optuna.trial.TrialState.FAIL)
            r["index"] = i
            r["space"] = sp
            r["optuna_trial_number"] = int(t.number)
            r["is_exp042_winner"] = bool(i == 0)
            r["stage"] = "screen"
            records.append(r)
            flush(r)
        done += b
        el = (time.time() - t_start) / 60
        best = min((x[objective_key] for x in records if x.get("ok")), default=None)
        print(f"[{time.strftime('%H:%M:%S')}] {done}/{a.n_trials} trials, "
              f"{el:.1f} min, best {objective_key} "
              f"{best if best is None else round(best, 8)}", flush=True)
    screen_min = (time.time() - t_start) / 60

    # -------------------------------------------------- screen cap adaptation
    cens = [r for r in records if r.get("ok") and r["censored"]]
    print(f"[{time.strftime('%H:%M:%S')}] screen censored at cap {a.cap}: "
          f"{len(cens)} -> re-run at {a.recap}", flush=True)
    rer = {}
    if cens:
        jobs = [(r["index"], r["params"], SCREEN_SEED, a.recap, a.od_wait)
                for r in cens]
        for r in pool.map(_run, jobs):
            r["stage"] = "screen_recap"
            rer[r["rid"]] = r
            flush(r)

    screen = []
    for r in records:
        if r["index"] in rer and rer[r["index"]].get("ok"):
            e = dict(rer[r["index"]])
            e.update({"index": r["index"], "space": r["space"],
                      "is_exp042_winner": r["is_exp042_winner"],
                      "stage": "screen(recapped)",
                      "replaced_censored": {objective_key: r[objective_key],
                                            "best_iteration": r["best_iteration"],
                                            "cap": r["cap"]}})
            screen.append(e)
        else:
            screen.append(dict(r))
    usable = [r for r in screen
              if r.get("ok") and not r.get("censored")]
    usable.sort(key=lambda r: r[objective_key])
    for i, r in enumerate(usable):
        r["screen_rank"] = i + 1
    n_still_censored = sum(1 for r in screen if r.get("ok") and r.get("censored"))

    # ----------------------------------------------------------- confirm stage
    t_conf = time.time()
    cand = usable[:a.n_confirm]
    print(f"[{time.strftime('%H:%M:%S')}] confirm: top {len(cand)} x seeds "
          f"{CONFIRM_SEEDS}", flush=True)
    jobs = [(f"{r['index']}|{s}", r["params"], s, a.cap, a.od_wait)
            for r in cand for s in CONFIRM_SEEDS]
    conf = {}
    for r in pool.map(_run, jobs):
        r["stage"] = "confirm"
        conf[r["rid"]] = r
        flush(r)
    cens2 = [r for r in conf.values() if r.get("ok") and r["censored"]]
    if cens2:
        print(f"[{time.strftime('%H:%M:%S')}] confirm censored: {len(cens2)} "
              f"-> re-run at {a.recap}", flush=True)
        jobs = [(r["rid"], r["params"], r["seed"], a.recap, a.od_wait)
                for r in cens2]
        for r in pool.map(_run, jobs):
            r["stage"] = "confirm_recap"
            flush(r)
            if r.get("ok"):
                conf[r["rid"]] = r
    pool.close()
    pool.join()
    tf.close()

    confirm_rows = []
    for r in cand:
        per = {str(SCREEN_SEED): r[objective_key]}
        iters = {str(SCREEN_SEED): r["best_iteration"]}
        caps = {str(SCREEN_SEED): r["cap"]}
        censd = {str(SCREEN_SEED): bool(r["censored"])}
        bad = False
        for s in CONFIRM_SEEDS:
            c = conf.get(f"{r['index']}|{s}")
            if c is None or not c.get("ok") or c.get("censored"):
                bad = True
                per[str(s)] = None
                iters[str(s)] = None if c is None else c.get("best_iteration")
                caps[str(s)] = None if c is None else c.get("cap")
                censd[str(s)] = None if c is None else bool(c.get("censored"))
            else:
                per[str(s)] = c[objective_key]
                iters[str(s)] = c["best_iteration"]
                caps[str(s)] = c["cap"]
                censd[str(s)] = bool(c["censored"])
        row = {"index": r["index"], "space": r["space"],
               "is_exp042_winner": r["is_exp042_winner"],
               "screen_rank": r["screen_rank"], "params": r["params"],
               f"per_seed_{objective_key}": per,
               "best_iteration_per_seed": iters, "cap_per_seed": caps,
               "censored_per_seed": censd, "usable": not bad}
        if not bad:
            v = [per[str(s)] for s in [SCREEN_SEED] + CONFIRM_SEEDS]
            row[f"mean3_{objective_key}"] = float(np.mean(v))
            row[f"std3_{objective_key}"] = float(np.std(v, ddof=1))
        confirm_rows.append(row)
    ranked = [r for r in confirm_rows if r["usable"]]
    ranked.sort(key=lambda r: r[f"mean3_{objective_key}"])
    for i, r in enumerate(ranked):
        r["confirm_rank"] = i + 1
    pick = ranked[0] if ranked else None

    imp = {}
    for s, st in studies.items():
        try:
            imp[s] = {k: float(v) for k, v in
                      optuna.importance.get_param_importances(st).items()}
        except Exception as e:                                      # noqa: BLE001
            imp[s] = {"error": f"{type(e).__name__}: {e}"}

    w0 = [r for r in screen if r["is_exp042_winner"]]
    fits = [r["fit_sec"] / 60.0 for r in records]
    out = {
        "exp_id": "exp_044", "phase": "search", "mode": a.mode, "name": name,
        "fold": "fold_2023 (train <= 2022 -> valid 2023)",
        "objective": objective_key, "eval_metric": eval_metric,
        "n_trials": a.n_trials, "n_confirm": a.n_confirm,
        "cap": a.cap, "recap": a.recap, "od_wait": a.od_wait,
        "workers": a.workers, "threads_per_fit": a.threads,
        "screen_seed": SCREEN_SEED, "confirm_seeds": CONFIRM_SEEDS,
        "schedule_counts": {"narrow": schedule.count("narrow"),
                            "wide": schedule.count("wide")},
        "wide_indices": sorted(widx),
        "sampler": "optuna TPESampler(seed=42) per study, batched ask/tell",
        "optuna_version": optuna.__version__,
        "n_failed_screen": int(sum(1 for r in records if not r.get("ok"))),
        "n_censored_screen_at_cap": len(cens),
        "censored_screen_indices": [r["index"] for r in cens],
        "n_censored_screen_after_recap": int(n_still_censored),
        "n_censored_confirm_at_cap": len(cens2),
        "data": {k: D[k] for k in D if k.startswith(("n_", "pos_", "sha", "note"))},
        "exp042_winner_params": dict(EXP042_WINNER),
        "exp042_winner_screen": (w0[0] if w0 else None),
        "exp042_winner_screen_rank": (w0[0].get("screen_rank") if w0 else None),
        "exp042_winner_confirm": next(
            (r for r in confirm_rows if r["is_exp042_winner"]), None),
        "param_importances": imp,
        "pick": pick,
        "pick_space": (pick["space"] if pick else None),
        "confirm_table": confirm_rows,
        "confirm_ranked": ranked,
        "top10_screen": usable[:10],
        "all_screen_sorted": usable,
        "screen_min": round(screen_min, 2),
        "confirm_min": round((time.time() - t_conf) / 60, 2),
        "total_min": round((time.time() - t_start) / 60, 2),
        "fit_min_screen": {"median": round(float(np.median(fits)), 2),
                           "mean": round(float(np.mean(fits)), 2),
                           "max": round(float(np.max(fits)), 2),
                           "sum": round(float(np.sum(fits)), 1)},
        "trials_jsonl": trials_path,
    }
    p = os.path.join(OUT_DIR, "res", f"search_{name}.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    if pick:
        bp = os.path.join(OUT_DIR, "res", f"best_{name}.json")
        with open(bp, "w") as f:
            json.dump(pick["params"], f, indent=1)
        print("PICK: " + json.dumps({
            "index": pick["index"], "space": pick["space"],
            "screen_rank": pick["screen_rank"],
            f"mean3_{objective_key}": pick[f"mean3_{objective_key}"],
            "params": pick["params"]}), flush=True)
        print(f"PICK SPACE = {pick['space']}", flush=True)
    print(f"exp042-winner screen rank: {out['exp042_winner_screen_rank']} "
          f"/ {len(usable)}", flush=True)
    print("WROTE " + p, flush=True)


if __name__ == "__main__":
    main()
