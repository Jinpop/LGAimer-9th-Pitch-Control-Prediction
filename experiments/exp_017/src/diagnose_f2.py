"""exp_017 F2 — is the B3 iteration shift a stacking-specific effect?

Two things are measured, both from the exp_017 runs:

 1. how much stack information a fold's TRAINING set actually carries:
    every training row either has NaN stack columns (earliest season, no
    predecessor) or carries an out-of-fold prediction whose AUC we measured on
    the very rows it was written to.  The index below is
        (1 - NaN share) * weighted mean (AUC - 0.5)
    i.e. the share of rows that have a stack value, times how informative that
    value is.  It is a descriptive index, not a calibrated quantity.

 2. the same argmin-growth statistic for B0, which has no stack columns at all.
    If B0 shows the same growth, the shift is not stacking-specific.

    python diagnose_f2.py <exp_017 metrics.json> [...]
"""
import json
import sys

CELL = "is_wayoff_true"          # the most predictive cell (exp_016 §4)


def main():
    runs, oof, seen_r, seen_b = [], [], set(), set()
    for p in sys.argv[1:]:
        m = json.load(open(p))
        for r in m["runs"]:
            r["_lr"] = m["learning_rate"]
            r["_cap"] = m["num_boost_round_cap"]
            k = (r["_lr"], r["_cap"], r["arm"], r["seed"], r["fold"])
            if k in seen_r:            # full01 and extend overlap on seed 42
                continue               # fold_2022; the values are identical
            seen_r.add(k)
            runs.append(r)
        for b in m["oof_diagnostics"]:
            b["_lr"] = m["learning_rate"]
            b["_cap"] = m["num_boost_round_cap"]
            k = (b["_lr"], b["_cap"], b["fold"], b["seed"], b["oof_season"])
            if k in seen_b:
                continue
            seen_b.add(k)
            oof.append(b)

    # ---- per-season OOF quality ------------------------------------------ #
    print("### OOF stack-column AUC by (lr, cap, fold, seed, predicted season)")
    print("| lr | cap | fold | seed | season | n_fit | cell iter | wayoff | rev | mid | both |")
    print("|" + "---|" * 11)
    for b in oof:
        a = b["auc"]
        print(f"| {b['_lr']} | {b['_cap']} | {b['fold']} | {b['seed']} | "
              f"{b['oof_season']} | {b['n_fit']:,} | {b['cell_iter']} | "
              f"{a['is_wayoff_true']} | {a['is_rev_only']} | {a['is_mid_only']} | "
              f"{a['is_both']} |")

    # ---- stack information index per fold -------------------------------- #
    print("\n### stack information index of each fold's TRAINING set")
    print("| lr | cap | fold | seed | n_train | NaN % | blocks | "
          "weighted mean wayoff AUC | index |")
    print("|" + "---|" * 9)
    for r in runs:
        if r["arm"] != "B3":
            continue
        blocks = [b for b in oof if b["fold"] == r["fold"]
                  and b["seed"] == r["seed"] and b["_lr"] == r["_lr"]
                  and b["_cap"] == r["_cap"]]
        if not blocks:
            continue
        n = sum(b["n_pred"] for b in blocks)
        wa = sum(b["n_pred"] * b["auc"][CELL] for b in blocks) / n
        share = 1.0 - r["nan_stack_rows_in_train_pct"] / 100.0
        print(f"| {r['_lr']} | {r['_cap']} | {r['fold']} | {r['seed']} | "
              f"{r['n_train']:,} | {r['nan_stack_rows_in_train_pct']} | "
              f"{len(blocks)} | {wa:.4f} | {share * (wa - 0.5):.4f} |")

    # ---- argmin growth, B0 vs B3 ----------------------------------------- #
    print("\n### argmin by (lr, cap, arm, seed, fold)")
    key = {}
    for r in runs:
        key.setdefault((r["_lr"], r["_cap"], r["arm"], r["seed"]), {})[r["fold"]] = \
            r["oracle_iter"]
    print("| lr | cap | arm | seed | f2022 | f2023 | f2024 | ratios |")
    print("|" + "---|" * 8)
    for k, v in sorted(key.items(), key=lambda x: (x[0][0], x[0][2], x[0][3])):
        fs = ["fold_2022", "fold_2023", "fold_2024"]
        vals = [v.get(f) for f in fs]
        got = [x for x in vals if x]
        rat = [round(got[i + 1] / got[i], 2) for i in range(len(got) - 1)]
        print(f"| {k[0]} | {k[1]} | {k[2]} | {k[3]} | "
              + " | ".join(str(x) if x else "—" for x in vals) + f" | {rat} |")


if __name__ == "__main__":
    main()
