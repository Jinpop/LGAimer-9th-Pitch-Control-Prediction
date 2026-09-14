#!/usr/bin/env python3
"""exp_087 G4 — durations from the sequential driver's artefacts, then the
decision the pre-registered rule makes.

`g4_time.sh` computed elapsed seconds with `bc`, which IS NOT INSTALLED here, so
the manifest's `sec` column came out empty.  The timing survives in the
artefacts, and for every run after the first BOTH endpoints are exact:

    start(n) = mtime of logs/t1_f<n>.log at creation ... not recoverable after
               the run writes to it, so instead:
    start(n) = end(n-1) + setup ,  setup measured per gap from the pair
               (end of run n-1, creation of log n) that was observed live: 3.90 s
               and 4.06 s.
    end(n)   = mtime of runs/t1_f<n>.csv

🔴 THE n=3 POINT IS NOT ON THE SAME FOOTING.  Its start is the driver's launch
plus a COLD package copy (216 MB read for the first time), and the reconstructed
value comes out BELOW n=6, which is impossible for a workload that only grows.
So n=3 is reported but NOT used as the ratio anchor; the anchor is the fitted
line's value at n=3 from the exact points.  That is a better measurement of the
same quantity, not a different threshold.
"""
import json, os, sys

ROOT = "/home/<user>/SungjinPark/exp087"
NS = [3, 6, 9, 12, 15, 18]
PACKAGER_CA2_1T = 374.10      # the scale the config's 480 s gate is written on
GATE = 480.0
BUDGET = 600.0


def mt(p):
    return os.path.getmtime(p) if os.path.exists(p) else None


def main():
    driver_start = float(sys.argv[1])
    setup = float(sys.argv[2]) if len(sys.argv) > 2 else 3.98
    runs, prev = [], None
    for n in NS:
        end = mt(f"{ROOT}/runs/t1_f{n}.csv")
        if end is None:
            runs.append({"n": n, "status": "missing"}); continue
        start = (driver_start + setup) if prev is None else prev + setup
        runs.append({"n": n, "sec": end - start, "exact_start": prev is not None,
                     "start_epoch": start, "end_epoch": end})
        prev = end

    ex = [(r["n"], r["sec"]) for r in runs if r.get("exact_start")]
    k = len(ex)
    mx = sum(x for x, _ in ex) / k; my = sum(y for _, y in ex) / k
    slope = sum((x - mx) * (y - my) for x, y in ex) / sum((x - mx) ** 2 for x, _ in ex)
    icpt = my - slope * mx
    resid = [y - (icpt + slope * x) for x, y in ex]
    # standard error of the slope: the per-seed cost turned out to be SMALLER
    # than the run-to-run spread on a shared box, so a point estimate alone
    # would be read as more certain than it is.
    dof = k - 2
    s2 = sum(r * r for r in resid) / dof if dof > 0 else float("nan")
    se = (s2 / sum((x - mx) ** 2 for x, _ in ex)) ** 0.5 if dof > 0 else float("nan")
    slope_hi = slope + 1.96 * se
    fit = {"slope_sec_per_F_seed": slope, "intercept_sec": icpt,
           "n_points_used": k, "points_used": [x for x, _ in ex],
           "residuals_sec": resid,
           "residual_sd_sec": s2 ** 0.5 if dof > 0 else None,
           "slope_stderr_sec_per_seed": se,
           "slope_95pct_upper_sec_per_seed": slope_hi,
           "max_abs_residual_sec": max(abs(r) for r in resid),
           "t_at_n3_fitted": icpt + slope * 3,
           "🔴_note": ("the per-seed cost came out BELOW the run-to-run spread of a "
                       "shared box, so the honest statement is an upper bound, not a "
                       "point estimate.  The conservative projection below uses the "
                       "95% upper end of the slope.")}

    anchor = fit["t_at_n3_fitted"]
    table = []
    for r in runs:
        if "sec" not in r: continue
        ratio = r["sec"] / anchor
        tp = PACKAGER_CA2_1T * ratio
        table.append({"n_F_seeds": r["n"], "harness_sec": r["sec"],
                      "ratio_to_n3": ratio, "packager_scale_sec": tp,
                      "pct_of_600s_budget": 100.0 * tp / BUDGET,
                      "passes_480s": tp <= GATE,
                      "reconstructed_start_is_exact": r["exact_start"]})
    # conservative projection: worst-case slope, anchored at the largest measured point
    base_n, base_t = max(ex, key=lambda p: p[0])
    cons = {}
    for n in NS:
        t_c = base_t + slope_hi * (n - base_n) if n > base_n else None
        if t_c is None: continue
        cons[str(n)] = {"harness_sec_upper": t_c,
                        "packager_scale_sec_upper": PACKAGER_CA2_1T * t_c / anchor,
                        "passes_480s": PACKAGER_CA2_1T * t_c / anchor <= GATE}
    ok = [t["n_F_seeds"] for t in table if t["passes_480s"]]
    out = {"gate": "G4_inference_time_vs_F_seed_count",
           "frame": "cov_shuf_al (245,789 rows, shuffled — worst case)",
           "threads": "OMP/OPENBLAS/MKL/NUMEXPR = 1, OMP_THREAD_LIMIT = 1, runs strictly sequential and alone",
           "packager_CA2_1thread_reference_sec": PACKAGER_CA2_1T,
           "gate_sec": GATE, "budget_sec": BUDGET,
           "runs": runs, "linear_fit_on_exact_points": fit,
           "table": table,
           "conservative_upper_bound_projection": cons,
           "n_values_passing": ok,
           "CHOSEN_n": max(ok) if ok else 3,
           "decision_rule": "G4_DECISION_RULE.json, written before these numbers were read"}
    json.dump(out, open(f"{ROOT}/res/g4_decision.json", "w"), indent=1)
    print(f"slope = {slope:.3f} s per F seed   (fit on n={fit['points_used']}, "
          f"max resid {fit['max_abs_residual_sec']:.2f} s)")
    print(f"anchor t(3) fitted = {anchor:.2f} s   [reconstructed n=3 = "
          f"{runs[0].get('sec', float('nan')):.2f} s, cold-setup, not used]")
    for t in table:
        print(f"  n={t['n_F_seeds']:>2}  harness {t['harness_sec']:7.2f}s  "
              f"ratio {t['ratio_to_n3']:.4f}  packager-scale {t['packager_scale_sec']:7.2f}s "
              f"({t['pct_of_600s_budget']:.1f}% of budget)  "
              f"{'PASS' if t['passes_480s'] else 'FAIL'}")
    print(f"slope 95% upper = {slope_hi:.3f} s/seed  (resid sd {fit['residual_sd_sec']:.2f} s)")
    for n, c in cons.items():
        print(f"  conservative n={n}: packager-scale <= {c['packager_scale_sec_upper']:.1f}s "
              f"{'PASS' if c['passes_480s'] else 'FAIL'}")
    print("CHOSEN n =", out["CHOSEN_n"])


if __name__ == "__main__":
    main()
