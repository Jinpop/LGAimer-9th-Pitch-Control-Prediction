"""exp_038 — the inheritance gate.  Nothing trains until this passes.

exp_038 reruns NOTHING that exp_037 already ran.  It reuses two things:

  1. the feature parquet (205 MB of it), which exp_037's leak audit cleared, and
  2. the R0 control — fold_2023 seed 42, and fold_2024 seeds 42/7/2024 —
     which is the denominator of every delta this experiment reports.

Reuse is only legitimate if the artefacts on disk are provably the ones the
numbers came from, so this file refuses to take anyone's word for it:

  * the parquet's sha256 must equal the value exp_037's metrics.json recorded
    when it built the file (which is itself compared with the sha pinned in
    exp_038's config);
  * every inherited prediction vector's sha256 must equal the one written
    beside it at fit time;
  * and the Brier is RECOMPUTED from the .npy and train.csv and must land on
    exp_037's reported number.  A json that agrees with itself proves nothing;
    a json that agrees with a recomputation from the raw prediction vector and
    the raw labels proves the number is real.

Expected values are READ from exp_037/metrics.json.  None of them is typed
into this file, because a transcribed constant would make the gate a test of
the typist.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

DATA_PATH = os.path.expanduser("~/SungjinPark/data/train.csv")
CONFIG_PINNED_FEAT_SHA = (
    "312d0ccb0d293d0193256d2067404c4b11a93505a6d28d72d8a09b532475acd4")
R0_UNITS = [(2023, 42, 1), (2024, 42, 2), (2024, 7, 2), (2024, 2024, 2)]
TOL = 1e-12


def sha256_file(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-exp", required=True, help="exp_037 root on server")
    ap.add_argument("--out", required=True, help="exp_038 root")
    args = ap.parse_args()
    src = os.path.abspath(os.path.expanduser(args.src_exp))
    out = os.path.abspath(os.path.expanduser(args.out))

    G = {"what": "exp_038 inherits exp_037's feature parquet and R0 control",
         "src_exp_root": src, "checks": {}}
    ok = True

    with open(os.path.join(src, "metrics.json")) as f:
        M37 = json.load(f)

    # ------------------------------------------------------- 1. the parquet
    pq = os.path.join(src, "featdir", "feat", "feat_main.parquet")
    pq_sha = sha256_file(pq)
    exp_sha = M37["stage_A"]["feat_sha256"]
    c = {"path": pq, "observed_sha256": pq_sha,
         "expected_sha256_from_exp037_metrics": exp_sha,
         "expected_sha256_pinned_in_exp038_config": CONFIG_PINNED_FEAT_SHA,
         "bytes": os.path.getsize(pq),
         "PASS": bool(pq_sha == exp_sha == CONFIG_PINNED_FEAT_SHA)}
    G["checks"]["feat_parquet"] = c
    ok &= c["PASS"]

    # the leak gates exp_037 ran on this exact file, carried forward as the
    # reason no new audit is required here
    G["checks"]["exp037_leak_gates"] = {
        k: v.get("PASS") for k, v in M37["leak_gates"].items()}
    ok &= all(v.get("PASS") is True for v in M37["leak_gates"].values())

    # -------------------------------------------- 2. the inherited R0 control
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig",
                     usecols=["row_id", "season", "game_type",
                              "control_success"])
    assert df["row_id"].is_unique
    assert df["season"].is_monotonic_increasing
    season = df["season"].to_numpy()
    y_all = df["control_success"].to_numpy(dtype=np.float64)
    gt_all = df["game_type"].to_numpy()

    exp23 = None
    for r in M37["stage_B_ladder"]["rungs"]:
        if r["arm"] == "R0":
            exp23 = r["brier_fold2023"]
    exp24 = {s: M37["stage_D_promotion"]["arms"]["R0"]["seeds"][s]
             ["brier_fold2024"] for s in ("42", "7", "2024")}

    units = {}
    for (u, s, st) in R0_UNITS:
        stem = f"R0_u{u}_seed{s}_st{st}"
        rp = os.path.join(src, "res", f"res_{stem}.json")
        pp = os.path.join(src, "pred", f"pred_{stem}.npy")
        d = {"res_json": rp, "pred_npy": pp}
        try:
            with open(rp) as f:
                r = json.load(f)
            got_sha = sha256_file(pp)
            p = np.load(pp)
            va = season == u
            y = y_all[va]
            assert len(p) == int(va.sum()), (len(p), int(va.sum()))
            recomputed = float(np.mean((p - y) ** 2))
            expected = exp23 if u == 2023 else exp24[str(s)]
            is_r = gt_all[va] == "R"
            d.update({
                "pred_sha256_in_res_json": r["pred_sha256"],
                "pred_sha256_on_disk": got_sha,
                "sha_match": bool(got_sha == r["pred_sha256"]),
                "n_valid": int(va.sum()),
                "brier_in_res_json": r["brier_full"],
                "brier_recomputed_from_npy": recomputed,
                "brier_expected_from_exp037_metrics": expected,
                "abs_diff_recomputed_vs_expected": abs(recomputed - expected),
                "abs_diff_resjson_vs_expected": abs(r["brier_full"] - expected),
                "tol": TOL,
                "feat_sha256_in_res_json": r["sha256"].get("feat_parquet"),
                "train_csv_sha256_in_res_json": r["sha256"].get("train.csv"),
                "valid_R_brier_recomputed": float(
                    np.mean((p[is_r] - y[is_r]) ** 2)),
                "valid_F_brier_recomputed": float(
                    np.mean((p[~is_r] - y[~is_r]) ** 2)),
                "n_valid_R": int(is_r.sum()), "n_valid_F": int((~is_r).sum()),
            })
            d["PASS"] = bool(d["sha_match"]
                             and d["abs_diff_recomputed_vs_expected"] < TOL
                             and d["abs_diff_resjson_vs_expected"] < TOL)
        except Exception as e:                                       # noqa: BLE001
            d["PASS"] = False
            d["error"] = repr(e)
        units[stem] = d
        ok &= bool(d["PASS"])
    G["checks"]["R0_inherited_units"] = units

    # train.csv must also be the same file exp_037 trained on, or the inherited
    # control was fitted on different rows than this experiment's arms will be
    G["checks"]["train_csv"] = {
        "sha256_now": sha256_file(DATA_PATH),
        "sha256_in_exp037_res": units["R0_u2023_seed42_st1"].get(
            "train_csv_sha256_in_res_json"),
    }
    G["checks"]["train_csv"]["PASS"] = bool(
        G["checks"]["train_csv"]["sha256_now"]
        == G["checks"]["train_csv"]["sha256_in_exp037_res"])
    ok &= G["checks"]["train_csv"]["PASS"]

    G["PASS"] = bool(ok)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "inherit_gate.json"), "w") as f:
        json.dump(G, f, indent=1)
    print(json.dumps({"PASS": G["PASS"],
                      "feat_parquet": G["checks"]["feat_parquet"]["PASS"],
                      "units": {k: v["PASS"] for k, v in units.items()}},
                     indent=1), flush=True)
    sys.exit(0 if G["PASS"] else 1)


if __name__ == "__main__":
    main()
