#!/usr/bin/env python
"""exp_039 — the inheritance gate.

exp_039 re-measures the seed sigma of R0 / fold_2024 at n=9.  Six of the nine
seeds are new; three (42, 7, 2024) are INHERITED from exp_037 and are NOT
refitted.  Inheritance is only legitimate if the thing inherited is provably
the same object, so this gate runs BEFORE any fit and blocks the driver if
anything is off.

WHAT IT PROVES
  1. exp_037's src/ is byte-identical to exp_039's src/ for every file that
     sits on the model code path (b3_feat.py and everything it sha-asserts).
     So "new seeds ran the same code" is a fact about bytes, not a memory.
  2. The three inherited res/pred files still hash to what they hashed to,
     and their briers equal the values recorded in exp_037/metrics.json.
     The expected values are READ from that metrics.json, never hardcoded.
  3. The iteration counts the six new units will be handed are the same ones
     the three inherited units were handed.  They come from exp_037's
     stage-1 base json by the same scale_full / scale_mode arithmetic; this
     gate reads those scales out of an inherited res file (audit note N1 —
     no hardcoding) and records them.

ON FAILURE the gate exits non-zero and the driver refuses to fit.  The
instruction is explicit: a mismatch means STOP AND REPORT, not "re-run it".
"""
import hashlib
import json
import os
import sys

E37 = os.path.expanduser("~/SungjinPark/experiments/exp_037")
E39 = os.path.expanduser("~/SungjinPark/experiments/exp_039")
INHERITED_SEEDS = [42, 7, 2024]
NEW_SEEDS = [1, 13, 101, 777, 2025, 31337]      # pre-registered, frozen

# every file b3_feat.py either IS or sha-asserts, plus the arm registry it
# imports.  driver.sh / aggregate.py are excluded on purpose: they orchestrate,
# they do not enter a model.
CODE_PATH_FILES = ["b3_feat.py", "b3_cat.py", "exp016_ref.py", "labels.py",
                   "arms.py"]


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
    out = {"PASS": None, "checks": {}, "failures": []}

    def fail(msg):
        out["failures"].append(msg)
        print("FAIL: " + msg, flush=True)

    # ---------------------------------------------------- 1. code path bytes
    code = {}
    for fn in CODE_PATH_FILES:
        a = os.path.join(E37, "src", fn)
        b = os.path.join(E39, "src", fn)
        ha, hb = sha256_file(a), sha256_file(b)
        code[fn] = {"exp_037": ha, "exp_039": hb, "identical": ha == hb}
        if ha != hb:
            fail(f"src/{fn} differs between exp_037 and exp_039")
    out["checks"]["code_path_sha256"] = code

    # ------------------------------------------- 2. inherited results by sha
    with open(os.path.join(E37, "metrics.json")) as f:
        m37 = json.load(f)
    r0 = m37["stage_D_promotion"]["arms"]["R0"]
    exp_seeds = r0["seeds"]

    inh = {}
    for s in INHERITED_SEEDS:
        stem = f"R0_u2024_seed{s}_st2"
        rp = os.path.join(E37, "res", f"res_{stem}.json")
        pp = os.path.join(E37, "pred", f"pred_{stem}.npy")
        if not (os.path.exists(rp) and os.path.exists(pp)):
            fail(f"seed {s}: exp_037 res or pred missing")
            continue
        with open(rp) as f:
            r = json.load(f)
        want = exp_seeds[str(s)]["brier_fold2024"]        # READ, not hardcoded
        got = r["brier_full"]
        ok_b = (got == want)
        ok_p = (sha256_file(pp) == r["pred_sha256"])
        if not ok_b:
            fail(f"seed {s}: brier {got!r} != metrics.json {want!r}")
        if not ok_p:
            fail(f"seed {s}: pred npy sha does not match the sha recorded "
                 f"inside its own res json — the file changed on disk")
        inh[str(s)] = {
            "res_path": rp, "res_sha256": sha256_file(rp),
            "pred_path": pp, "pred_sha256_recomputed": sha256_file(pp),
            "pred_sha256_in_res": r["pred_sha256"], "pred_sha_match": ok_p,
            "brier_in_res": got, "brier_in_metrics": want,
            "brier_match": ok_b,
            "stack_round": r["stack_round"], "cell_round": r["cell_round"],
            "scale_full": r["scale_full"], "scale_mode": r["scale_mode"],
            "base_json": r["base_json"],
            "b3_feat_sha_at_runtime": r["sha256"]["b3_feat.py"],
            "train_csv_sha": r["sha256"]["train.csv"],
            "n_train_full": r["n_train_full"], "n_valid": r["n_valid"],
            "minutes": r["minutes"],
        }
    out["checks"]["inherited"] = inh

    # ------------------------------------- 3. iteration counts are one object
    # every inherited unit must agree on rounds/scales, otherwise "the iters
    # were fixed" is not true even of exp_037 and the premise collapses.
    if len(inh) == len(INHERITED_SEEDS):
        ref = inh[str(INHERITED_SEEDS[0])]
        for s in INHERITED_SEEDS[1:]:
            v = inh[str(s)]
            for k in ("stack_round", "cell_round", "scale_full", "scale_mode",
                      "base_json", "b3_feat_sha_at_runtime", "train_csv_sha"):
                if v[k] != ref[k]:
                    fail(f"seed {s}: {k} differs from seed "
                         f"{INHERITED_SEEDS[0]} — the three inherited units "
                         f"are not the same configuration")
        out["checks"]["iter_contract"] = {
            "stack_round": ref["stack_round"],
            "cell_round": ref["cell_round"],
            "scale_full": ref["scale_full"],
            "scale_mode": ref["scale_mode"],
            "base_json": ref["base_json"],
            "source": "read out of exp_037 res json — not hardcoded",
        }
        # the base json the new units will be pointed at must be the same file
        bj = ref["base_json"]
        if not os.path.exists(bj):
            fail(f"stage-1 base json missing: {bj}")
        else:
            out["checks"]["base_json_sha256"] = sha256_file(bj)
            with open(bj) as f:
                bjd = json.load(f)
            if bjd.get("any_argmin_at_cap"):
                fail("stage-1 base had an argmin on the cap — invalid base")
            out["checks"]["base_stage1"] = {
                "arm": bjd["arm"], "unit": bjd["unit"], "seed": bjd["seed"],
                "stage": bjd["stage"],
                "stack_argmin_iter": bjd["stack_argmin_iter"],
                "cell_argmin_iter": bjd["cell_argmin_iter"],
                "any_argmin_at_cap": bjd["any_argmin_at_cap"],
                "cap": bjd["cap"],
            }

    # ---------------------------------------- 4. n=3 sigma reproduces exactly
    if len(inh) == len(INHERITED_SEEDS):
        import statistics
        b3 = [inh[str(s)]["brier_in_res"] for s in INHERITED_SEEDS]
        s3 = statistics.stdev(b3)
        want3 = r0["std_brier_ddof1"]
        out["checks"]["n3_sigma"] = {
            "recomputed": s3, "in_metrics": want3,
            "abs_diff": abs(s3 - want3),
            "match_1e_18": abs(s3 - want3) < 1e-18,
        }
        if abs(s3 - want3) > 1e-18:
            fail(f"n=3 sigma does not reproduce: {s3!r} vs {want3!r}")

    out["pre_registration"] = {
        "seeds_inherited": INHERITED_SEEDS,
        "seeds_new": NEW_SEEDS,
        "n_total": len(INHERITED_SEEDS) + len(NEW_SEEDS),
        "note": "the six new seeds are frozen in config.yaml before any "
                "result was seen; a failed unit reduces n, it is never "
                "replaced by another seed",
    }
    out["PASS"] = (len(out["failures"]) == 0)

    with open(os.path.join(E39, "inherit_gate.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({"PASS": out["PASS"],
                      "n_failures": len(out["failures"])}), flush=True)
    sys.exit(0 if out["PASS"] else 1)


if __name__ == "__main__":
    main()
