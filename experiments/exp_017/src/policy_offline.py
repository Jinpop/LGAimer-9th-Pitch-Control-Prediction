"""exp_017 stage 1 — offline evaluation of iter-carry policies on stored curves.

No training.  exp_016 stored the full valid_all Brier-vs-iteration curve for
every arm / fold / seed, and any policy that only reads earlier folds' curves
can therefore be scored exactly, without refitting anything.

Usage:
    python policy_offline.py <metrics.json> [<extra_metrics.json> ...] > out.json

Extra metrics files (exp_017's own runs) are merged so that folds exp_016 never
ran (B3 fold_2022, B0 fold_2022 for the repeat seeds, the lr=0.005 arms) join
the same table.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from policies import (argmin_of, nearest_grid_iter, p4_argmin,  # noqa: E402
                      p5_flat, p5_upper, p6_plateau, p7_scale, p8_trend_geo,
                      p8_trend_lin, p10_mult)

FOLDS = ["fold_2022", "fold_2023", "fold_2024"]
MULTS = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 5.0]


# --------------------------------------------------------------------------- #
def collect(m):
    """-> {(arm, seed): {fold: {"curve": {...}, "n_train": int}}}"""
    out = {}

    def put(arm, seed, fold, curve, n_train):
        if not curve:
            return
        d = out.setdefault((arm, str(seed)), {})
        d[fold] = {"curve": {int(k): float(v) for k, v in curve.items()},
                   "n_train": n_train}

    # --- exp_016 layout ---------------------------------------------------- #
    for arm in ["B0", "B0b", "B2c", "B2n", "B2m", "B1"]:
        if arm in m and "curves" in m.get(arm, {}):
            for f, c in m[arm]["curves"].items():
                put(arm, 42, f, c, m[arm].get(f, {}).get("n_train"))
    for s in ["7", "2024"]:
        blk = m.get(f"seed{s}", {})
        for arm, d in blk.items():
            if isinstance(d, dict) and "curves" in d:
                for f, c in d["curves"].items():
                    put(arm, s, f, c, d.get(f, {}).get("n_train"))
    for key, seed in [("B3", 42), ("B3_seed7", 7), ("B3_seed2024", 2024)]:
        blk = m.get(key, {})
        for f in FOLDS:
            if f in blk and "curve" in blk[f]:
                put("B3", seed, f, blk[f]["curve"], blk[f].get("n_train"))

    # --- exp_017 layout: m["runs"] = [{arm, seed, fold, curve, n_train}, ...] #
    # Runs at the production spec (lr 0.01 / cap 3000) merge with exp_016's
    # curves; anything else is a different model and gets its own arm name so
    # the two are never averaged together.
    lr = m.get("learning_rate")
    cap = m.get("num_boost_round_cap")
    if lr is None:                     # exp_016 file: no "runs" section anyway
        return out
    suffix = "" if (lr, cap) == (0.01, 3000) else f"_lr{lr:g}_cap{cap}"
    for r in m.get("runs", []):
        put(r["arm"] + suffix, r["seed"], r["fold"], r["curve"],
            r.get("n_train"))
    return out


def merge(dicts):
    out = {}
    for d in dicts:
        for k, v in d.items():
            out.setdefault(k, {}).update(v)
    return out


# --------------------------------------------------------------------------- #
def policy_menu(prev, prevprev, n_prev, n_cur):
    """{name: raw iter proposal}.  prevprev may be None."""
    men = {"P4_argmin": p4_argmin(prev)}
    for eps, tag in [(1e-6, "1e-6"), (3e-6, "3e-6"), (1e-5, "1e-5")]:
        men[f"P5_flat_run_eps{tag}"] = p5_flat(prev, eps, contiguous=True)
        men[f"P5_flat_all_eps{tag}"] = p5_flat(prev, eps, contiguous=False)
        men[f"P5u_upper_eps{tag}"] = p5_upper(prev, eps)
    for c in MULTS:
        men[f"P10_mult_{c:g}x"] = p10_mult(prev, c)
    for K in (200, 500):
        for delta, tag in [(1e-6, "1e-6"), (3e-6, "3e-6")]:
            men[f"P6_plateau_K{K}_d{tag}"] = p6_plateau(prev, K, delta)
    if n_prev and n_cur:
        men["P7_scale_rows"] = p7_scale(prev, n_prev, n_cur)
    if prevprev is not None:
        men["P8_trend_lin"] = p8_trend_lin(prevprev, prev)
        men["P8_trend_geo"] = p8_trend_geo(prevprev, prev)
    return men


def evaluate(store):
    """One row per (arm, seed, transition, policy)."""
    rows = []
    for (arm, seed), folds in sorted(store.items()):
        for i in (1, 2):
            cur, prev = FOLDS[i], FOLDS[i - 1]
            if cur not in folds or prev not in folds:
                continue
            ccur = folds[cur]["curve"]
            cprev = folds[prev]["curve"]
            cpp = folds[FOLDS[i - 2]]["curve"] if i == 2 and FOLDS[0] in folds else None
            grid = sorted(ccur)
            o_it, o_br = argmin_of(ccur)
            men = policy_menu(cprev, cpp, folds[prev].get("n_train"),
                              folds[cur].get("n_train"))
            base = ccur[nearest_grid_iter(grid, men["P4_argmin"])]
            for name, raw in men.items():
                k = nearest_grid_iter(grid, raw)
                b = ccur[k]
                gap = base - o_br
                rows.append({
                    "arm": arm, "seed": seed,
                    "transition": f"{prev}->{cur}",
                    "policy": name,
                    "iter_raw": int(raw), "iter_used": int(k),
                    "brier": round(b, 6),
                    "oracle_iter": o_it, "oracle_brier": round(o_br, 6),
                    "iter_ratio_vs_oracle": round(k / o_it, 3) if o_it else None,
                    "cost_vs_oracle_e6": round(1e6 * (b - o_br), 2),
                    "delta_vs_P4_e6": round(1e6 * (b - base), 2),
                    "recovery_pct_of_P4_gap": (round(100.0 * (base - b) / gap, 1)
                                               if gap > 1e-12 else None),
                })
    return rows


def aggregate(rows):
    """Mean over seeds, per (arm, transition, policy) and per (arm, policy)."""
    def _agg(keyfn):
        acc = {}
        for r in rows:
            acc.setdefault(keyfn(r), []).append(r)
        out = {}
        for k, v in acc.items():
            n = len(v)
            out["|".join(k)] = {
                "n": n,
                "iters": sorted({x["iter_used"] for x in v}),
                "oracle_iters": sorted({x["oracle_iter"] for x in v}),
                "mean_brier": round(sum(x["brier"] for x in v) / n, 6),
                "mean_cost_vs_oracle_e6": round(
                    sum(x["cost_vs_oracle_e6"] for x in v) / n, 2),
                "mean_delta_vs_P4_e6": round(
                    sum(x["delta_vs_P4_e6"] for x in v) / n, 2),
                "recovery_pcts": [x["recovery_pct_of_P4_gap"] for x in v],
                "worst_delta_vs_P4_e6": max(x["delta_vs_P4_e6"] for x in v),
            }
        return out
    return {
        "by_arm_transition_policy": _agg(
            lambda r: (r["arm"], r["transition"], r["policy"])),
        "by_arm_policy": _agg(lambda r: (r["arm"], r["policy"])),
    }


def argmin_table(store):
    """The raw fact the whole experiment turns on: argmin per arm/seed/fold."""
    t = {}
    for (arm, seed), folds in sorted(store.items()):
        row = {}
        for f in FOLDS:
            if f in folds:
                it, br = argmin_of(folds[f]["curve"])
                row[f] = {"argmin": it, "brier": round(br, 6),
                          "n_train": folds[f].get("n_train")}
        # growth ratios of consecutive argmins
        av = [row[f]["argmin"] for f in FOLDS if f in row]
        nv = [row[f]["n_train"] for f in FOLDS if f in row and row[f]["n_train"]]
        row["argmin_growth_ratios"] = [round(av[i + 1] / av[i], 3)
                                       for i in range(len(av) - 1)] if len(av) > 1 else []
        row["n_train_growth_ratios"] = [round(nv[i + 1] / nv[i], 3)
                                        for i in range(len(nv) - 1)] if len(nv) > 1 else []
        t[f"{arm}|seed{seed}"] = row
    return t


def curve_shape(store, eps_list=(1e-6, 5e-6, 2e-5)):
    """How flat is each fold's curve near its optimum, in grid-iteration width."""
    out = {}
    for (arm, seed), folds in sorted(store.items()):
        for f, d in folds.items():
            c = d["curve"]
            m = min(c.values())
            row = {}
            for e in eps_list:
                ins = [k for k in sorted(c) if c[k] <= m + e]
                row[f"band_{e:g}"] = [min(ins), max(ins)] if ins else None
            # cost of being 2x below / above the argmin
            a = argmin_of(c)[0]
            g = sorted(c)
            row["argmin"] = a
            row["cost_at_half_argmin_e6"] = round(
                1e6 * (c[nearest_grid_iter(g, a / 2)] - m), 2)
            row["cost_at_double_argmin_e6"] = round(
                1e6 * (c[nearest_grid_iter(g, a * 2)] - m), 2)
            out[f"{arm}|seed{seed}|{f}"] = row
    return out


FINE = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 4.0, 4.5,
        5.0, 6.0]


def multiplier_scan(store):
    """Target-fold Brier as a function of 'previous argmin x c'.

    This is the decision surface behind P7/P8/P10: all three are just different
    ways of guessing one multiplier.  Reported per (arm, seed, transition) so
    that the multiplier chosen on the EARLIER transition can be checked against
    the later one without the later one ever entering the choice.
    """
    out = {}
    for (arm, seed), folds in sorted(store.items()):
        for i in (1, 2):
            cur, prev = FOLDS[i], FOLDS[i - 1]
            if cur not in folds or prev not in folds:
                continue
            ccur, cprev = folds[cur]["curve"], folds[prev]["curve"]
            grid = sorted(ccur)
            a = argmin_of(cprev)[0]
            o_it, o_br = argmin_of(ccur)
            row = {"prev_argmin": a, "oracle_iter": o_it,
                   "oracle_brier": round(o_br, 6),
                   "oracle_mult": round(o_it / a, 2),
                   "by_mult": {}}
            for c in FINE:
                k = nearest_grid_iter(grid, int(round(a * c)))
                row["by_mult"][f"{c:g}"] = {
                    "iter": k, "brier": round(ccur[k], 6),
                    "cost_vs_oracle_e6": round(1e6 * (ccur[k] - o_br), 1)}
            best = min(row["by_mult"], key=lambda c: row["by_mult"][c]["brier"])
            row["best_mult_in_hindsight"] = best
            out[f"{arm}|seed{seed}|{prev}->{cur}"] = row
    return out


def holdout_multiplier(store, arms=("B0", "B3")):
    """Honest test of the multiplier family.

    Pick c on the 2022->2023 transition only (mean rank over the arms/seeds
    that have it), then score that single c on 2023->2024.  The later fold
    never touches the choice.
    """
    scan = multiplier_scan(store)
    early = {k: v for k, v in scan.items() if k.endswith("fold_2022->fold_2023")}
    late = {k: v for k, v in scan.items() if k.endswith("fold_2023->fold_2024")}
    if not early:
        return {"note": "no fold_2022->fold_2023 transition available"}
    tot, used = {}, []
    for k, v in early.items():
        if k.split("|")[0] not in arms:
            continue
        used.append(k)
        for c, d in v["by_mult"].items():
            tot.setdefault(c, []).append(d["cost_vs_oracle_e6"])
    mean_cost = {c: round(sum(v) / len(v), 2) for c, v in tot.items()}
    if not mean_cost:
        return {"note": "no early transition for the requested arms"}
    c_star = min(mean_cost, key=lambda c: mean_cost[c])
    applied = {}
    for k, v in late.items():
        if k.split("|")[0] not in arms:
            continue
        d1 = v["by_mult"][c_star]
        d0 = v["by_mult"]["1"]
        applied[k] = {"iter": d1["iter"], "brier": d1["brier"],
                      "P4_iter": d0["iter"], "P4_brier": d0["brier"],
                      "delta_vs_P4_e6": round(1e6 * (d1["brier"] - d0["brier"]), 2),
                      "cost_vs_oracle_e6": d1["cost_vs_oracle_e6"],
                      "recovery_pct": (round(100.0 * (d0["brier"] - d1["brier"])
                                             / (d0["brier"] - v["oracle_brier"]), 1)
                                       if d0["brier"] > v["oracle_brier"] else None)}
    return {"chosen_on": sorted(used), "mean_cost_e6_on_early": mean_cost,
            "c_star": c_star, "applied_to_late": applied}


def main():
    store = merge([collect(json.load(open(p))) for p in sys.argv[1:]])
    rows = evaluate(store)
    print(json.dumps({
        "multiplier_scan": multiplier_scan(store),
        "holdout_multiplier": holdout_multiplier(store),
        "available": {f"{a}|seed{s}": sorted(v) for (a, s), v in sorted(store.items())},
        "argmin_table": argmin_table(store),
        "curve_shape": curve_shape(store),
        "rows": rows,
        "aggregate": aggregate(rows),
    }, indent=1))


if __name__ == "__main__":
    main()
