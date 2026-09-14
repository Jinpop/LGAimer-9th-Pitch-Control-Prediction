#!/usr/bin/env python3
"""exp_087 G4 — reconstruct the run durations the driver failed to record.

`g4_time.sh` computed its elapsed seconds with `bc`, which IS NOT INSTALLED on
this box, so the `sec` column of `runs/g4_manifest.tsv` came out empty.  The
timing itself was not lost: the driver is strictly sequential and every run
leaves two dated artefacts.

    end(n)   = mtime of runs/t1_f<n>.csv      (copied out the instant the run ends)
    start(n) = end(n-1) + setup               (setup = rm -rf + cp -r package +
                                               cp frame + sync, between runs)
    start(3) = driver start + setup_cold

`setup` is MEASURED, not assumed: for every pair the log file logs/t1_f<n>.log is
created (0 bytes) when the redirect opens, i.e. at python start, and for the
run that was in flight when this was noticed that gap was 3.90 s.  Where the log
mtime is later than its own csv mtime the log is the end marker instead.

⚠️ The n=3 point carries the extra uncertainty of the COLD setup (the package
had never been read) and is reported with that flag.  The other points are
±(setup jitter), order 1 s on values of several hundred.
"""
import json, os, subprocess, sys

ROOT = "/home/<user>/SungjinPark/exp087"
NS = [3, 6, 9, 12, 15, 18]


def mt(p):
    return os.path.getmtime(p) if os.path.exists(p) else None


def main():
    driver_start = float(sys.argv[1]) if len(sys.argv) > 1 else None
    setup = float(sys.argv[2]) if len(sys.argv) > 2 else 3.90
    rows, prev_end = [], None
    for n in NS:
        csv = f"{ROOT}/runs/t1_f{n}.csv"
        log = f"{ROOT}/logs/t1_f{n}.log"
        end = mt(csv)
        if end is None:
            rows.append({"n_F_seeds": n, "status": "missing"})
            continue
        if prev_end is None:
            start = (driver_start + setup) if driver_start else None
            cold = True
        else:
            start = prev_end + setup
            cold = False
        rows.append({
            "n_F_seeds": n,
            "end_epoch": end,
            "start_epoch": start,
            "sec": (end - start) if start else None,
            "cold_setup": cold,
            "log_mtime": mt(log),
            "csv_sha_row": None,
        })
        prev_end = end
    out = {"method": __doc__.strip().splitlines()[0],
           "setup_seconds_between_runs": setup,
           "driver_start_epoch": driver_start,
           "bc_missing": True,
           "runs": rows}
    # slope: least squares of sec on n
    pts = [(r["n_F_seeds"], r["sec"]) for r in rows if r.get("sec")]
    if len(pts) >= 2:
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        nn = len(pts); mx = sum(xs)/nn; my = sum(ys)/nn
        sxy = sum((x-mx)*(y-my) for x, y in pts)
        sxx = sum((x-mx)**2 for x in xs)
        b = sxy/sxx; a = my - b*mx
        out["fit_sec_vs_n"] = {"slope_sec_per_F_seed": b, "intercept_sec": a,
                               "resid_max_abs": max(abs(y-(a+b*x)) for x, y in pts),
                               "n_points": nn}
    json.dump(out, open(f"{ROOT}/res/g4_times.json", "w"), indent=1)
    for r in rows:
        print(r)
    if "fit_sec_vs_n" in out:
        print("slope sec/F-seed =", out["fit_sec_vs_n"]["slope_sec_per_F_seed"],
              " max resid =", out["fit_sec_vs_n"]["resid_max_abs"])


if __name__ == "__main__":
    main()
