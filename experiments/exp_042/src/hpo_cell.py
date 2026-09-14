"""exp_042 step 2 — the hyper-parameter search, on ONE cell of C5_B3.

WHAT IS SEARCHED, WHERE, AND WHY THAT PLACE
-------------------------------------------
Fold: unit 2023 (train <= 2022 -> valid 2023).  fold_2024 is never opened here.
Model: the binary cell model named on the command line, fitted on the same
`tr_mode` pool and scored on the same `va_ok` rows as
`experiments/exp_030/src/b3_cat.py` uses (see celldata.py, sha-gated).
Objective: valid-2023 **logloss** of that cell's own binary target, at the
iteration the early-stopping detector picks.  Brier is recorded alongside but
does not select.

ITERATIONS ARE NOT SEARCHED.  Every trial trains under a cap and reports at its
own best iteration.  A trial whose curve is CENSORED — the detector never fired
before the cap, so a better iteration could exist past the edge — is marked
`censored` and re-run at a larger cap; its censored value is kept in the record
but the re-run value is what ranks it.

REPRODUCIBILITY.  One TPE sampler, seed 42, in one process.  Trials are asked
in batches of `--workers`, evaluated in a fork pool, and told back in trial
order, so the sequence of (ask, tell) is a function of the batch size alone.
Trial 0 is the configuration in production today, enqueued so that the current
settings and the searched ones are measured on one ruler.

Usage:
    python hpo_cell.py --cell is_wayoff_true --workers 5 --threads 4 \
        --n-trials 150 --cap 1650
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
import celldata                                                    # noqa: E402

OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_042")
FIT_SEED = 42          # the seed b3_cat.py stage 1 fits the cells with

FIXED = {
    "loss_function": "Logloss", "eval_metric": "Logloss", "task_type": "CPU",
    "boosting_type": "Plain", "bootstrap_type": "Bernoulli", "has_time": True,
    "allow_writing_files": False, "verbose": False,
}
CURRENT = {
    "depth": 6, "l2_leaf_reg": 10.0, "learning_rate": 0.03, "subsample": 0.8,
    "rsm": 1.0, "random_strength": 1.0, "min_data_in_leaf": 1,
    "one_hot_max_size": 10,
}

_G = {}


def _init(D, threads):
    import catboost as cb
    _G["cb"] = cb
    _G["threads"] = threads
    _G["y_va"] = D["y_va"]
    _G["pool_tr"] = cb.Pool(D["F_tr"], label=D["y_tr"], cat_features=D["cat_idx"])
    _G["pool_va"] = cb.Pool(D["F_va"], label=D["y_va"], cat_features=D["cat_idx"])


def _run(job):
    """One fit.  Returns the trial record; never raises into the parent."""
    number, params, cap, od_wait = job
    cb = _G["cb"]
    t0 = time.time()
    rec = {"number": number, "params": dict(params), "cap": cap,
           "od_wait": od_wait, "threads": _G["threads"], "fit_seed": FIT_SEED}
    try:
        m = cb.CatBoostClassifier(**FIXED, **params, random_seed=FIT_SEED,
                                  iterations=cap, thread_count=_G["threads"],
                                  od_type="Iter", od_wait=od_wait,
                                  use_best_model=False)
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
            # the detector never fired: a better iteration could lie past the cap
            "censored": bool(built >= cap and bi + 1 + od_wait > cap),
            "logloss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
            "brier": float(np.mean((p - y) ** 2)),
            "pred_mean": float(p.mean()),
            "fit_sec": round(time.time() - t0, 2),
        })
    except Exception as e:                                         # noqa: BLE001
        rec.update({"ok": False, "error": f"{type(e).__name__}: {e}",
                    "fit_sec": round(time.time() - t0, 2)})
    return rec


def suggest(trial):
    return {
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 100.0, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.10, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "rsm": trial.suggest_float("rsm", 0.5, 1.0),
        "random_strength": trial.suggest_float("random_strength", 0.5, 10.0, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 200, log=True),
        "one_hot_max_size": trial.suggest_int("one_hot_max_size", 2, 64, log=True),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="is_wayoff_true")
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--threads", type=int, required=True)
    ap.add_argument("--n-trials", type=int, default=150)
    ap.add_argument("--cap", type=int, default=1650)
    ap.add_argument("--recap", type=int, default=3300)
    ap.add_argument("--od-wait", type=int, default=250)
    a = ap.parse_args()
    assert a.workers * a.threads <= 20, ("container CPU quota is 20",
                                         a.workers * a.threads)
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    t_start = time.time()
    D = celldata.build(a.cell)
    print(f"[{time.strftime('%H:%M:%S')}] cell {a.cell}: n_tr "
          f"{D['n_train_mode']:,} n_va_ok {D['n_valid_ok']:,} "
          f"pos_rate_tr {D['pos_rate_train']:.6f} "
          f"pos_rate_va {D['pos_rate_valid']:.6f}", flush=True)

    trials_path = os.path.join(OUT_DIR, "res", f"trials_{a.cell}.jsonl")
    tf = open(trials_path, "w")

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.enqueue_trial(dict(CURRENT))          # trial 0 = production settings

    ctx = mp.get_context("fork")
    pool = ctx.Pool(a.workers, initializer=_init, initargs=(D, a.threads))

    records, done = [], 0
    while done < a.n_trials:
        b = min(a.workers, a.n_trials - done)
        asked = [study.ask() for _ in range(b)]
        jobs = [(t.number, suggest(t), a.cap, a.od_wait) for t in asked]
        outs = pool.map(_run, jobs)
        outs.sort(key=lambda r: r["number"])
        for t, r in zip(asked, outs):
            assert t.number == r["number"], (t.number, r["number"])
            if r["ok"]:
                study.tell(t, r["logloss"])
            else:
                study.tell(t, state=optuna.trial.TrialState.FAIL)
            r["is_current_config"] = bool(r["number"] == 0)
            records.append(r)
            tf.write(json.dumps(r) + "\n")
        tf.flush()
        os.fsync(tf.fileno())
        done += b
        el = (time.time() - t_start) / 60
        best = min((x["logloss"] for x in records if x.get("ok")), default=None)
        print(f"[{time.strftime('%H:%M:%S')}] {done}/{a.n_trials} trials, "
              f"{el:.1f} min elapsed, best logloss "
              f"{best if best is None else round(best, 7)}", flush=True)

    search_min = (time.time() - t_start) / 60

    # ------------------------------------------------- cap adaptation re-runs
    cens = [r for r in records if r.get("ok") and r["censored"]]
    print(f"[{time.strftime('%H:%M:%S')}] censored trials: {len(cens)} "
          f"-> re-running at cap {a.recap}", flush=True)
    rer = {}
    if cens:
        jobs = [(r["number"], r["params"], a.recap, a.od_wait) for r in cens]
        outs = pool.map(_run, jobs)
        for r in outs:
            rer[r["number"]] = r
            tf.write(json.dumps(dict(r, rerun=True)) + "\n")
        tf.flush()
        os.fsync(tf.fileno())
    pool.close()
    pool.join()
    tf.close()

    # effective record: the re-run replaces the censored one where it exists
    eff = []
    for r in records:
        if r["number"] in rer and rer[r["number"]].get("ok"):
            e = dict(rer[r["number"]])
            e["is_current_config"] = r["is_current_config"]
            e["replaced_censored_value"] = {
                "logloss": r["logloss"], "best_iteration": r["best_iteration"],
                "cap": r["cap"]}
            eff.append(e)
        else:
            eff.append(dict(r))
    ok = [r for r in eff if r.get("ok")]
    ok.sort(key=lambda r: r["logloss"])
    for i, r in enumerate(ok):
        r["rank"] = i + 1

    imp = None
    try:
        imp = {k: float(v) for k, v in
               optuna.importance.get_param_importances(study).items()}
    except Exception as e:                                         # noqa: BLE001
        imp = {"error": f"{type(e).__name__}: {e}"}

    cur = [r for r in ok if r.get("is_current_config")]
    out = {
        "exp_id": "exp_042", "phase": "search", "cell": a.cell,
        "fold": "fold_2023 (train <= 2022 -> valid 2023)",
        "objective": "valid-2023 logloss of the cell's own binary target",
        "n_trials_declared": a.n_trials,
        "n_trials_run": len(records),
        "n_failed": int(sum(1 for r in records if not r.get("ok"))),
        "n_censored_at_first_cap": len(cens),
        "censored_trial_numbers": [r["number"] for r in cens],
        "n_censored_after_recap": int(sum(1 for r in eff
                                          if r.get("ok") and r.get("censored"))),
        "cap": a.cap, "recap": a.recap, "od_wait": a.od_wait,
        "workers": a.workers, "threads_per_fit": a.threads,
        "sampler": "optuna TPESampler(seed=42), batched ask/tell",
        "optuna_version": optuna.__version__,
        "fit_seed": FIT_SEED,
        "n_train_mode": D["n_train_mode"], "n_valid_ok": D["n_valid_ok"],
        "no_row_subsetting": ("full pool used: no [:N] cap anywhere; "
                              f"n_train={D['n_train_mode']} of "
                              f"{D['n_train_full']} tr_full rows, "
                              f"n_valid_scored={D['n_valid_ok']} of "
                              f"{D['n_valid_all']} season-2023 rows"),
        "train_csv_sha256": D["train_csv_sha256"],
        "search_min": round(search_min, 2),
        "total_min": round((time.time() - t_start) / 60, 2),
        "param_importances": imp,
        "current_config": dict(CURRENT),
        "current_config_rank": (cur[0]["rank"] if cur else None),
        "current_config_record": (cur[0] if cur else None),
        "best": ok[0] if ok else None,
        "top10": ok[:10],
        "all_trials_sorted": ok,
        "trials_jsonl": trials_path,
    }
    p = os.path.join(OUT_DIR, "res", f"search_{a.cell}.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print("BEST: " + json.dumps(out["best"]), flush=True)
    print(f"CURRENT CONFIG RANK: {out['current_config_rank']} / {len(ok)}",
          flush=True)
    print("WROTE " + p, flush=True)


if __name__ == "__main__":
    main()
