"""exp_018 — conditional (pitcher x batter-hand) tendency features.

Spec: experiments/exp_018/config.yaml.
Control / reproduction gate: exp_010 D (47 provided columns, strongly
regularised LightGBM), fold_2024 valid_all Brier 0.248261 under the P4 policy.

Question
--------
Every provided pitcher-tendency column is UNCONDITIONAL.  Does an explicit
(pitcher, batter_hand) conditional rate — EB-shrunk towards that pitcher's own
unconditional rate — buy anything that a num_leaves=15 model cannot build for
itself?  Prior evidence in config.yaml measured a between-pitcher true SD of
5.20pp on the L/R reverse split with r = 0.00 against the unconditional rate;
this run re-measures that INSIDE each fold's training window (prelim.json) and
then tests whether it converts into Brier.

Arms (features accumulate, nothing else changes between them):
  A0  47 provided columns                        <- control, reproduction gate
  A1  + pit_reverse_rate_vs_hand
  A2  + pit_success_rate_vs_hand
  A3  + pit_middle_rate_vs_hand + pit_n_vs_hand

Iteration policy: P10 (exp_017), i.e. an arm carries its OWN previous fold's
valid_all Brier argmin, scaled by c* = 2.75.  fold_2022 has no predecessor and
is reported at its own argmin (oracle, flagged, not comparable across
experiments).  A0 additionally reports the unscaled P4 shadow because the
reproduction gate is stated in P4 terms.  valid 2024 is never consulted to
choose anything.

CHANGED vs exp_010 D (both are variables, both are recorded):
  1. the platoon feature columns (A1/A2/A3 only; A0 is byte-identical in spec)
  2. the iteration policy P4 -> P10 (x2.75)

Environment: A5000, ~/SungjinPark/venv (Python 3.11.15 / numpy 1.26.4 /
pandas 2.0.3 / lightgbm 4.7.0).
"""

import gc
import hashlib
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features_platoon as fp                                    # noqa: E402
from labels import assert_no_recovered_in_features, recover      # noqa: E402

SEED = 42
SEED_REPEATS = [7, 2024]
N_THREADS = 6

DATA_PATH = os.path.expanduser("~/SungjinPark/data/train.csv")
OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_018")
SRC_DIR = os.path.dirname(os.path.abspath(__file__))

TARGET = "control_success"
DROP_COLS = ["row_id", TARGET]
CAT_COLS = ["top_bottom", "game_type", "base_state"]

FOLDS = [("fold_2022", 2021, 2022), ("fold_2023", 2022, 2023),
         ("fold_2024", 2023, 2024)]
F_EXCLUDE_MAX_SEASON = 2022
EXPECTED_2024 = {"valid_all": 253507, "valid_R": 223497}

D_PARAMS = {                       # exp_010 D verbatim
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.01,
    "num_leaves": 15,
    "min_data_in_leaf": 5000,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": SEED,
    "num_threads": N_THREADS,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}
N_ROUNDS = 3000
P10_C = 2.75

GATE = {
    "source": "exp_010 metrics.json -> summary_fold_2024_valid_ALL_PRIMARY.D_P4-all",
    "arm": "A0",
    "policy": "P4 (unscaled carry) — the policy the reference was measured under",
    "fold_2024_valid_all_brier": 0.248261,
    "fold_2024_iter": 750,
    "tolerance_brier": 5e-6,
}

SMOKE = os.environ.get("EXP018_SMOKE") == "1"
if SMOKE:
    N_ROUNDS = 40
    SEED_REPEATS = [7]
    OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_018_smoke")


# --------------------------------------------------------------------------- #
def sha256_file(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def lb_score(brier, r):
    d = r * (1.0 - r)
    return None if d <= 0 else float(max(0.0, 100000.0 * (1.0 - brier / d)))


def seg_metrics(y, p):
    n = int(len(y))
    if n == 0:
        return {"n": 0}
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    brier = float(np.mean((p - y) ** 2))
    r = float(np.mean(y))
    d = r * (1.0 - r)
    out = {"n": n, "brier": round(brier, 6),
           "bss": round(float(1.0 - brier / d), 6) if d > 0 else None,
           "score": round(lb_score(brier, r), 3) if d > 0 else None,
           "base_rate": round(r, 6), "pred_mean": round(float(np.mean(p)), 6)}
    if 0.0 < r < 1.0:
        out["auc"] = round(float(roc_auc_score(y, p)), 6)
        out["logloss"] = round(float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15))), 6)
    return out


def curve_grid(n_max):
    g = list(range(1, 21)) + list(range(25, 101, 5)) + list(range(110, 301, 10))
    g += list(range(325, min(n_max, 1000) + 1, 25))
    if n_max > 1000:
        g += list(range(1050, n_max + 1, 50))
    return sorted(set([k for k in g if k <= n_max] + [n_max]))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def nearest_grid_iter(grid, k):
    return min(grid, key=lambda g: (abs(g - k), g))


def raw_margin_curve(booster, X, grid):
    """{k: raw margin of the first k rounds} in one pass over the trees.

    exp_010 / exp_016 machinery, unchanged: for a fixed booster the first k
    rounds ARE the k-round model and raw margins add across round ranges.
    """
    sub = X.iloc[: min(2000, len(X))]
    a = booster.predict(sub, start_iteration=0, num_iteration=5, raw_score=True)
    b = booster.predict(sub, start_iteration=5, num_iteration=5, raw_score=True)
    full = booster.predict(sub, start_iteration=0, num_iteration=10, raw_score=True)
    resid = full - (a + b)
    const = float(np.median(resid))
    assert np.allclose(resid, const, atol=1e-8), (
        "raw margins are not additive across round ranges",
        float(np.abs(resid - const).max()))
    out, acc, prev = {}, None, 0
    for k in grid:
        chunk = booster.predict(X, start_iteration=prev, num_iteration=k - prev,
                                raw_score=True)
        acc = chunk if acc is None else acc + chunk + const
        prev = k
        out[k] = acc.copy()
    return out


def argmin_of(curve):
    k = min(curve, key=lambda x: curve[x])
    return int(k), float(curve[k])


# --------------------------------------------------------------------------- #
def main():
    t_start = time.time()
    np.random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] loading {DATA_PATH}", flush=True)
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    print(f"loaded {df.shape}", flush=True)
    assert df["row_id"].is_unique
    assert df["season"].is_monotonic_increasing, "file order assumption broken"

    # 47 provided columns, in the exact order exp_010 used
    base_feats = [c for c in df.columns if c not in DROP_COLS]
    assert len(base_feats) == 47, len(base_feats)
    assert_no_recovered_in_features(base_feats, "base_feats (before recovery)")

    # ---- recovered mode labels: AGGREGATION INPUT ONLY --------------------- #
    diag = recover(df)
    assert diag["success_recovery"]["agree_pct"] == 100.0, diag["success_recovery"]

    season = df["season"].to_numpy()
    ok = df["recov_ok"].to_numpy()
    succ_season = df.groupby("pitcher_id", sort=False)["season"].shift(-1).to_numpy()
    f_excl = ((df["season"] <= F_EXCLUDE_MAX_SEASON)
              & (df["game_type"] == "F")).to_numpy()
    y_all = df[TARGET].to_numpy(dtype=np.float64)

    # ---- platoon lookups, season-forward-chained --------------------------- #
    print(f"[{time.strftime('%H:%M:%S')}] building platoon lookups", flush=True)
    t_lu = time.time()
    lu_diag, src_max_season = fp.build_season_forward_chained(
        df, ok, succ_season, f_excl)
    lu_diag["build_seconds"] = round(time.time() - t_lu, 1)
    print(f"  platoon lookups built in {lu_diag['build_seconds']}s", flush=True)

    # ---- prior-evidence reproduction inside each fold's training window ---- #
    print(f"[{time.strftime('%H:%M:%S')}] prelim stats per fold", flush=True)
    prelim = {"note": "re-measured inside each fold's TRAINING window only; the "
                      "config values were pooled over all seasons and are "
                      "therefore optimistic by construction",
              "config_prior_evidence": {
                  "pit_reverse_rate_vs_hand": {"true_sd_pp": 5.20,
                                               "reliability": 0.80,
                                               "obs_over_noise": 2.23,
                                               "eb_k": 65},
                  "pit_success_rate_vs_hand": {"true_sd_pp": 4.75,
                                               "reliability": 0.69,
                                               "obs_over_noise": 1.81,
                                               "eb_k": 111},
                  "pit_middle_rate_vs_hand": {"true_sd_pp": 1.48,
                                              "reliability": 0.30,
                                              "obs_over_noise": 1.20,
                                              "eb_k": 583}},
              "folds": {}, "warnings": []}
    cfg_sd = {"pit_reverse_rate_vs_hand": 5.20, "pit_success_rate_vs_hand": 4.75,
              "pit_middle_rate_vs_hand": 1.48}
    for fold_name, tu, vy in FOLDS:
        st = fp.prelim_stats(df, ok, succ_season, f_excl, tu, next_season=vy)
        prelim["folds"][fold_name] = st
        for ax, v in st["axes"].items():
            if v.get("eb_k_dev_over_20pct"):
                prelim["warnings"].append(
                    f"{fold_name}/{ax}: recomputed eb_k {v['eb_k_recomputed']} "
                    f"deviates >20% from config {v['eb_k_config_used_for_features']} "
                    f"(features still use the config value — k is never chosen "
                    f"on validation data)")
            dev = abs(v["true_sd_pp"] - cfg_sd[ax]) / cfg_sd[ax]
            if dev > 0.20:
                prelim["warnings"].append(
                    f"{fold_name}/{ax}: true_sd {v['true_sd_pp']}pp deviates "
                    f"{dev:.0%} from the config value {cfg_sd[ax]}pp")
    with open(os.path.join(OUT_DIR, "prelim.json"), "w") as f:
        json.dump(prelim, f, indent=2, ensure_ascii=False)
    print("prelim warnings: " + json.dumps(prelim["warnings"], ensure_ascii=False),
          flush=True)
    for fn in prelim["folds"]:
        print(f"  {fn}: " + json.dumps(
            {a: {"true_sd_pp": v["true_sd_pp"], "reliability": v["reliability"],
                 "r_uncond": v["corr_with_unconditional"],
                 "eb_k": v["eb_k_recomputed"]}
             for a, v in prelim["folds"][fn]["axes"].items()}), flush=True)

    for c in CAT_COLS:
        df[c] = df[c].astype("category")

    grid = curve_grid(N_ROUNDS)
    cat = [c for c in CAT_COLS if c in base_feats]

    metrics = {
        "exp_id": "exp_018",
        "base_exp": "exp_010 D (47 provided columns, strongly regularised LGBM)",
        "hypothesis": "an explicit (pitcher x batter_hand) conditional tendency, "
                      "EB-shrunk towards the pitcher's own unconditional rate, "
                      "gives a num_leaves=15 model an interaction it cannot "
                      "build for itself",
        "changed_vs_exp_010_D": [
            "platoon feature columns (A1/A2/A3 only — A0 is spec-identical)",
            "iteration policy P4 -> P10 (own previous fold argmin x 2.75, exp_017)",
        ],
        "seed_primary": SEED, "seed_repeats": SEED_REPEATS,
        "num_threads": N_THREADS, "lgb_params": D_PARAMS,
        "num_boost_round_cap": N_ROUNDS, "p10_c": P10_C,
        "mask": "training rows with season<=2022 & game_type=='F' dropped "
                "(exp_004); validation untouched",
        "n_rows_masked": int(f_excl.sum()),
        "arms": {a: {"platoon_cols": v, "n_features": 47 + len(v)}
                 for a, v in fp.ARM_COLS.items()},
        "eb_k_used": {k: v[1] for k, v in fp.RATE_SPEC.items()},
        "recovery_diagnostics": diag,
        "platoon_lookup": lu_diag,
        "leak_boundary": {
            "recovered_labels_used_as": "LOOKUP-TABLE AGGREGATION INPUT ONLY",
            "feature_list_assert": "labels.assert_no_recovered_in_features() is "
                                   "called before every Dataset build",
            "as_of": "row of season s joins a table built from seasons <= s-1; "
                     "season 2019 keeps NaN",
            "successor_rule": "rows whose next same-pitcher row falls outside the "
                              "table window are excluded from the table (their "
                              "is_* could not be recovered without a later row)",
        },
        "sha256": {"train.csv": sha256_file(DATA_PATH),
                   "train.py": sha256_file(os.path.abspath(__file__)),
                   "labels.py": sha256_file(os.path.join(SRC_DIR, "labels.py")),
                   "features_platoon.py": sha256_file(
                       os.path.join(SRC_DIR, "features_platoon.py"))},
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__, "lightgbm": lgb.__version__},
        "gate_spec": GATE,
    }

    def save():
        with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
            json.dump(json.loads(json.dumps(metrics, default=str)), f,
                      indent=2, ensure_ascii=False)

    # ---------------------------------------------------------------- #
    def run_arm(arm, seed, folds_wanted):
        """Fit one arm across the requested folds at one seed."""
        params = dict(D_PARAMS, seed=seed)
        feats = base_feats + fp.ARM_COLS[arm]
        res = {"arm": arm, "seed": seed, "n_features": len(feats),
               "features_added": fp.ARM_COLS[arm]}
        curves, oracles = {}, {}
        bst24 = None

        for fold_name, tu, vy in FOLDS:
            if fold_name not in folds_wanted:
                continue
            tr = (season <= tu) & ~f_excl
            va = season == vy
            y_va = y_all[va]
            is_r = (df.loc[va, "game_type"] == "R").to_numpy()
            if vy == 2024:
                assert int(va.sum()) == EXPECTED_2024["valid_all"], int(va.sum())
                assert int(is_r.sum()) == EXPECTED_2024["valid_R"]

            # ---- runtime as-of assert: no lookup used season >= vy --------- #
            used = src_max_season[tr | va]
            mx = np.nanmax(used) if np.any(~np.isnan(used)) else -np.inf
            assert mx < vy, (
                f"AS-OF VIOLATION in {arm}/{fold_name}: a lookup value was built "
                f"from season {mx} but the validation season is {vy}")

            assert_no_recovered_in_features(feats, f"{arm}/{fold_name} design matrix")
            X_tr = df.loc[tr, feats]
            X_va = df.loc[va, feats]

            t0 = time.time()
            dtr = lgb.Dataset(X_tr, label=y_all[tr], categorical_feature=cat)
            bst = lgb.train(params, dtr, num_boost_round=N_ROUNDS)
            t1 = time.time()
            marg = raw_margin_curve(bst, X_va, grid)
            cv = {k: float(np.mean((sigmoid(marg[k]) - y_va) ** 2)) for k in grid}
            a_min, a_val = argmin_of(cv)
            print(f"    [{arm} s{seed} {fold_name}] fit {(t1-t0)/60:.2f} min, "
                  f"curve {time.time()-t1:.0f}s ({int(tr.sum()):,} x {len(feats)}) "
                  f"-> valid_all argmin {a_min} = {a_val:.6f}", flush=True)

            curves[fold_name] = cv
            oracles[fold_name] = (a_min, a_val)
            res.setdefault("_marg", {})[fold_name] = marg
            res.setdefault("_meta", {})[fold_name] = {
                "y_va": y_va, "is_r": is_r, "n_train": int(tr.sum()),
                "fit_min": round((t1 - t0) / 60, 2),
                "nan_platoon_pct_train": {
                    c: round(100.0 * float(df.loc[tr, c].isna().mean()), 3)
                    for c in fp.ARM_COLS[arm]},
            }
            res.setdefault("curves", {})[fold_name] = {
                str(k): round(v, 6) for k, v in cv.items()}
            if fold_name == "fold_2024":
                bst24 = bst
            else:
                del bst
            gc.collect()

        # ---- report each fold under P10 (and the P4 shadow) ---------------- #
        def report(fold_name, k, source, oracle_flag):
            meta = res["_meta"][fold_name]
            p = sigmoid(res["_marg"][fold_name][k])
            y_va, is_r = meta["y_va"], meta["is_r"]
            a_min, a_val = oracles[fold_name]
            return {"iter_used": int(k), "iter_source": source,
                    "oracle_flag": oracle_flag,
                    "n_train": meta["n_train"], "fit_min": meta["fit_min"],
                    "nan_platoon_pct_train": meta["nan_platoon_pct_train"],
                    "valid_all": seg_metrics(y_va, p),
                    "valid_R": seg_metrics(y_va[is_r], p[is_r]),
                    "valid_F": seg_metrics(y_va[~is_r], p[~is_r]),
                    "oracle_iter": a_min, "oracle_brier": round(a_val, 6),
                    "policy_cost_vs_oracle": round(
                        curves[fold_name][k] - a_val, 6)}, p

        order = [f for f, _, _ in FOLDS if f in curves]
        prev = None
        p24 = None
        for fold_name in order:
            if prev is None:
                k = oracles[fold_name][0]
                rep, p = report(fold_name, k,
                                "own argmin (no predecessor; ORACLE, feeder only)",
                                True)
            else:
                want = int(round(P10_C * oracles[prev][0]))
                k = nearest_grid_iter(grid, want)
                rep, p = report(fold_name, k,
                                f"P10: {oracles[prev][0]} (own {prev} valid_all "
                                f"argmin) x {P10_C}", False)
                rep["iter_requested"] = want
                rep["iter_capped_by_num_boost_round"] = bool(want > N_ROUNDS)
            res[fold_name] = rep
            if fold_name == "fold_2024":
                p24 = p
            prev = fold_name

        # P4 shadow (unscaled carry) — the gate is stated in P4 terms
        prev = None
        for fold_name in order:
            if prev is None:
                rep, _ = report(fold_name, oracles[fold_name][0],
                                "own argmin (ORACLE, feeder only)", True)
            else:
                rep, _ = report(fold_name,
                                nearest_grid_iter(grid, oracles[prev][0]),
                                f"P4: {oracles[prev][0]} (own {prev} argmin), "
                                f"no scaling", False)
            res[fold_name + "_P4"] = rep
            prev = fold_name

        res["oracle_iters"] = {f: oracles[f][0] for f in order}

        # gain importance at the reported fold_2024 iteration
        if "fold_2024" in res and bst24 is not None:
            imp = sorted(zip(feats, bst24.feature_importance(
                "gain", iteration=res["fold_2024"]["iter_used"])),
                key=lambda t: -t[1])
            tot = sum(g for _, g in imp) or 1.0
            res["fold_2024_gain_top20"] = [
                [f, round(float(g), 1), round(100.0 * g / tot, 2)] for f, g in imp[:20]]
            res["fold_2024_gain_share_platoon_pct"] = round(
                100.0 * sum(g for f, g in imp if f in fp.ALL_COLS) / tot, 3)
            res["fold_2024_gain_platoon_each_pct"] = {
                f: round(100.0 * g / tot, 3) for f, g in imp if f in fp.ALL_COLS}
            del bst24

        if p24 is not None:
            np.save(os.path.join(OUT_DIR, f"pred_{arm}_seed{seed}_fold_2024.npy"),
                    p24)
        del res["_marg"], res["_meta"]
        gc.collect()
        return res

    # ---------------- stage 1: 4 arms x 3 folds x seed 42 ------------------ #
    print(f"[{time.strftime('%H:%M:%S')}] ===== stage 1 (seed {SEED}) =====",
          flush=True)
    all_folds = {f for f, _, _ in FOLDS}
    stage1 = {}
    for arm in ["A0", "A1", "A2", "A3"]:
        stage1[arm] = run_arm(arm, SEED, all_folds)
        metrics[f"{arm}_seed{SEED}"] = {k: v for k, v in stage1[arm].items()
                                        if not k.startswith("_")}
        metrics["elapsed_min"] = round((time.time() - t_start) / 60, 2)
        save()

        if arm == "A0":
            got = stage1["A0"]["fold_2024_P4"]["valid_all"]["brier"]
            d = abs(got - GATE["fold_2024_valid_all_brier"])
            metrics["A0_reproduction_gate"] = {
                "expected": GATE["fold_2024_valid_all_brier"], "got": got,
                "abs_diff": round(d, 8),
                "expected_iter": GATE["fold_2024_iter"],
                "got_iter": stage1["A0"]["fold_2024_P4"]["iter_used"],
                "policy": GATE["policy"],
                "passed": bool(d <= GATE["tolerance_brier"])}
            print("A0 gate: " + json.dumps(metrics["A0_reproduction_gate"]),
                  flush=True)
            save()
            if not SMOKE:
                assert d <= GATE["tolerance_brier"], (
                    f"A0 failed to reproduce exp_010 D ({got} vs "
                    f"{GATE['fold_2024_valid_all_brier']}) — everything "
                    "downstream would be uninterpretable, aborting.")

    base24 = stage1["A0"]["fold_2024"]["valid_all"]["brier"]
    table = {}
    for arm in ["A0", "A1", "A2", "A3"]:
        v = stage1[arm]["fold_2024"]["valid_all"]
        table[arm] = {
            "iter": stage1[arm]["fold_2024"]["iter_used"],
            "n_features": stage1[arm]["n_features"],
            "valid_all_brier": v["brier"], "valid_all_bss": v["bss"],
            "valid_all_score": v["score"], "valid_all_auc": v.get("auc"),
            "valid_R_brier": stage1[arm]["fold_2024"]["valid_R"]["brier"],
            "valid_F_brier": stage1[arm]["fold_2024"]["valid_F"]["brier"],
            "delta_vs_A0": round(v["brier"] - base24, 6),
            "fold_2023_valid_all_brier":
                stage1[arm]["fold_2023"]["valid_all"]["brier"],
            "fold_2022_valid_all_brier_ORACLE":
                stage1[arm]["fold_2022"]["valid_all"]["brier"],
        }
    metrics["summary_stage1_seed42_fold_2024_valid_all_P10"] = table
    print("--- stage 1 summary (P10, fold_2024 valid_all) ---\n"
          + json.dumps(table, indent=2), flush=True)

    best = min([a for a in table if a != "A0"],
               key=lambda a: table[a]["valid_all_brier"])
    metrics["best_arm_seed42"] = best
    print(f"best arm at seed {SEED}: {best}", flush=True)
    metrics["elapsed_min"] = round((time.time() - t_start) / 60, 2)
    save()

    # ---------------- stage 2: A0 + best arm, seeds 7 / 2024 --------------- #
    print(f"[{time.strftime('%H:%M:%S')}] ===== stage 2: A0 + {best}, seeds "
          f"{SEED_REPEATS} =====", flush=True)
    seed_tab = {"A0": {str(SEED): base24},
                best: {str(SEED): table[best]["valid_all_brier"]}}
    for s in SEED_REPEATS:
        for arm in ["A0", best]:
            r = run_arm(arm, s, all_folds)
            metrics[f"{arm}_seed{s}"] = {k: v for k, v in r.items()
                                         if not k.startswith("_")}
            seed_tab[arm][str(s)] = r["fold_2024"]["valid_all"]["brier"]
            metrics["elapsed_min"] = round((time.time() - t_start) / 60, 2)
            save()

    seeds_all = [str(SEED)] + [str(s) for s in SEED_REPEATS]
    spread = {}
    for a, d_ in seed_tab.items():
        vals = [d_[s] for s in seeds_all if s in d_]
        spread[a] = {"per_seed": d_, "mean": round(float(np.mean(vals)), 6),
                     "std": round(float(np.std(vals, ddof=1)), 6),
                     "range": round(float(max(vals) - min(vals)), 6)}
    ds = [seed_tab[best][s] - seed_tab["A0"][s] for s in seeds_all]
    a0_std = spread["A0"]["std"]
    mean_d = float(np.mean(ds))
    verdict = {
        "arm": best,
        "per_seed_delta": {s: round(seed_tab[best][s] - seed_tab["A0"][s], 6)
                           for s in seeds_all},
        "mean_delta": round(mean_d, 6),
        "std_delta": round(float(np.std(ds, ddof=1)), 6),
        "all_seeds_same_sign": bool(all(x < 0 for x in ds) or all(x > 0 for x in ds)),
        "A0_seed_std": a0_std,
        "abs_mean_delta_over_A0_std": (round(abs(mean_d) / a0_std, 2)
                                       if a0_std > 0 else None),
        "criterion": "measured := same sign on all 3 seeds AND |mean delta| >= "
                     "3 x A0 seed std",
        "measured": bool((all(x < 0 for x in ds) or all(x > 0 for x in ds))
                         and a0_std > 0 and abs(mean_d) >= 3 * a0_std),
    }
    metrics["seed_summary"] = {"brier_by_seed": seed_tab, "spread": spread,
                               "paired_delta_vs_A0": verdict}
    print("--- seed summary ---\n"
          + json.dumps(metrics["seed_summary"], indent=2), flush=True)

    metrics["train_time_min"] = round((time.time() - t_start) / 60, 2)
    metrics["stage3_note"] = ("NOT RUN — stacking the same features on top of "
                              "exp_016 B3 requires separate approval")
    save()
    print(f"[{time.strftime('%H:%M:%S')}] DONE {metrics['train_time_min']} min",
          flush=True)


if __name__ == "__main__":
    main()
