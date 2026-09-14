"""exp_017 — render the report tables from the merged offline evaluation.

    python summarize.py <offline.json>
"""
import json
import sys

ARMS = ("B0", "B3")


def main():
    d = json.load(open(sys.argv[1]))

    print("### argmin per arm / seed / fold")
    print("| arm | seed | f2022 | f2023 | f2024 | growth ratios | n_train ratios |")
    print("|---|---|---|---|---|---|---|")
    for k, v in d["argmin_table"].items():
        arm = k.split("|")[0]
        if arm not in ARMS:
            continue
        cells = [str(v[f]["argmin"]) if f in v else "—"
                 for f in ("fold_2022", "fold_2023", "fold_2024")]
        print(f"| {arm} | {k.split('seed')[1]} | " + " | ".join(cells)
              + f" | {v['argmin_growth_ratios']} | {v['n_train_growth_ratios']} |")

    print("\n### policy table, per arm x transition (mean over seeds)")
    agg = d["aggregate"]["by_arm_transition_policy"]
    for arm in ARMS:
        for tr in ("fold_2022->fold_2023", "fold_2023->fold_2024"):
            keys = [k for k in agg if k.startswith(f"{arm}|{tr}|")]
            if not keys:
                continue
            print(f"\n**{arm} · {tr}**")
            print("| policy | n | iters | mean Brier | Δ vs P4 (e-6) | "
                  "cost vs oracle (e-6) | worst Δ | recovery % |")
            print("|---|---|---|---|---|---|---|---|")
            for k in sorted(keys, key=lambda x: agg[x]["mean_brier"]):
                a = agg[k]
                print(f"| {k.split('|')[2]} | {a['n']} | {a['iters']} | "
                      f"{a['mean_brier']:.6f} | {a['mean_delta_vs_P4_e6']:+.1f} | "
                      f"{a['mean_cost_vs_oracle_e6']:.1f} | "
                      f"{a['worst_delta_vs_P4_e6']:+.1f} | {a['recovery_pcts']} |")

    print("\n### multiplier scan (target-fold cost vs oracle, e-6)")
    ms = d["multiplier_scan"]
    cs = ["1", "1.5", "2", "2.5", "3", "3.5", "4", "5", "6"]
    print("| key | prev argmin | oracle iter | oracle mult | "
          + " | ".join(f"x{c}" for c in cs) + " |")
    print("|" + "---|" * (4 + len(cs)))
    for k, v in ms.items():
        if k.split("|")[0] not in ARMS:
            continue
        print(f"| {k} | {v['prev_argmin']} | {v['oracle_iter']} | "
              f"{v['oracle_mult']} | "
              + " | ".join(str(v["by_mult"][c]["cost_vs_oracle_e6"]) for c in cs)
              + " |")

    print("\n### honest hold-out of the multiplier family")
    print("```json")
    print(json.dumps(d["holdout_multiplier"], indent=1))
    print("```")


if __name__ == "__main__":
    main()
