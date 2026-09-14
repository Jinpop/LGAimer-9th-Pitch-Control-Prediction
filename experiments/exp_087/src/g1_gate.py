#!/usr/bin/env python3
"""exp_087 G1 — is the F-branch seed-42 refit MODEL-IDENTICAL to the shipped one?

The 15 new F seeds are only comparable to the 3 already in the package if the
recipe restored here is the recipe that made those three.  Nothing about that
inheritance is self-evident: candidate S's trainer reads its iteration base out
of exp_037's res files, its 16 extra columns out of a parquet, its parameter
dict out of exp_027's module.  Any of those could have moved since 8월.  So seed
42 is refitted through the restored command line and the result must be the
model the package ships.

WHAT "IDENTICAL" MEANS, AND WHY NOT BYTES
-----------------------------------------
🔴 `.cbm` bytes DO NOT reproduce (CLAUDE.md; candidate AG's measurement).  Every
fit writes a fresh `model_guid` and a wall-clock `train_finish_time`.  A byte
gate would test the clock.  Required instead, all of it:
  * every leaf value equal in float64, exactly
  * every leaf weight equal in float64, exactly (leaf weights encode which
    TRAINING ROWS fell where, so a moved mask cannot match)
  * identical per-tree split lists, leaf counts, tree count, feature order
  * identical `scale_and_bias`, recorded parameter block, `learn_best_error`
  * 🔴 and NO OTHER metadata key differing — without this the check is weak.

Structure and criteria are build_AI/src/03_gate_repro.py's, retargeted from one
stacker per seed to the five boosters of one F seed.
"""
import argparse, hashlib, json, os, sys, tempfile, zipfile
import numpy as np
from catboost import CatBoostClassifier

MODELS = ["cell_is_both.cbm", "cell_is_mid_only.cbm", "cell_is_rev_only.cbm",
          "cell_is_wayoff_true.cbm", "stacker.cbm"]
ALLOWED_META_DIFF = {"model_guid", "train_finish_time"}


def dump(model_path):
    m = CatBoostClassifier()
    m.load_model(model_path)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as t:
        p = t.name
    m.save_model(p, format="json")
    j = json.load(open(p))
    os.unlink(p)
    return m, j


def leaves(j):
    lv, lw, nl, splits = [], [], [], []
    for t in j.get("oblivious_trees", []):
        lv.extend(t.get("leaf_values", []))
        lw.extend(t.get("leaf_weights", []))
        nl.append(len(t.get("leaf_values", [])))
        splits.append([(s.get("float_feature_index"), s.get("border"),
                        s.get("split_type"), s.get("cat_feature_index"),
                        s.get("ctr_target_border_idx"))
                       for s in t.get("splits", [])])
    return (np.asarray(lv, dtype=np.float64), np.asarray(lw, dtype=np.float64),
            nl, splits)


def compare(parent_path, refit_path, member):
    mp, jp = dump(parent_path)
    mn, jn = dump(refit_path)
    lvp, lwp, nlp, spp = leaves(jp)
    lvn, lwn, nln, spn = leaves(jn)
    mi_p, mi_n = jp.get("model_info", {}), jn.get("model_info", {})
    mi_diff = sorted(k for k in (set(mi_p) | set(mi_n))
                     if mi_p.get(k) != mi_n.get(k))
    top_diff = sorted(k for k in (set(jp) | set(jn))
                      if k != "model_info" and jp.get(k) != jn.get(k))
    par_p = json.loads(mi_p.get("params", "{}")) if isinstance(
        mi_p.get("params"), str) else mi_p.get("params")
    par_n = json.loads(mi_n.get("params", "{}")) if isinstance(
        mi_n.get("params"), str) else mi_n.get("params")
    e = {
        "member": member,
        "parent_sha256": hashlib.sha256(open(parent_path, "rb").read()).hexdigest(),
        "refit_sha256": hashlib.sha256(open(refit_path, "rb").read()).hexdigest(),
        "file_bytes": [os.path.getsize(parent_path), os.path.getsize(refit_path)],
        "tree_count": [int(mp.tree_count_), int(mn.tree_count_)],
        "tree_count_equal": int(mp.tree_count_) == int(mn.tree_count_),
        "feature_names_equal": list(mp.feature_names_) == list(mn.feature_names_),
        "n_features": [len(mp.feature_names_), len(mn.feature_names_)],
        "n_leaf_values": [int(lvp.size), int(lvn.size)],
        "leaf_values_max_abs_diff": (float(np.abs(lvp - lvn).max())
                                     if lvp.shape == lvn.shape else None),
        "leaf_values_identical": bool(lvp.shape == lvn.shape and np.array_equal(lvp, lvn)),
        "leaf_weights_max_abs_diff": (float(np.abs(lwp - lwn).max())
                                      if lwp.shape == lwn.shape else None),
        "leaf_weights_identical": bool(lwp.shape == lwn.shape and np.array_equal(lwp, lwn)),
        "leaf_counts_per_tree_equal": nlp == nln,
        "n_trees_with_differing_splits": (sum(1 for a, b in zip(spp, spn) if a != b)
                                          if len(spp) == len(spn) else -1),
        "scale_and_bias_equal": jp.get("scale_and_bias") == jn.get("scale_and_bias"),
        "params_block_equal": par_p == par_n,
        "learn_best_error": [mi_p.get("catboost_learn_best_error"),
                             mi_n.get("catboost_learn_best_error")],
        "model_info_keys_differing": mi_diff,
        "model_info_unexpected_diff": [k for k in mi_diff if k not in ALLOWED_META_DIFF],
        "top_level_keys_differing_excluding_model_info": top_diff,
        "bytes_equal": (hashlib.sha256(open(parent_path, "rb").read()).hexdigest()
                        == hashlib.sha256(open(refit_path, "rb").read()).hexdigest()),
    }
    e["learn_best_error_equal"] = e["learn_best_error"][0] == e["learn_best_error"][1]
    e["PASS"] = bool(
        e["tree_count_equal"] and e["feature_names_equal"]
        and e["leaf_values_identical"] and e["leaf_weights_identical"]
        and e["leaf_counts_per_tree_equal"]
        and e["n_trees_with_differing_splits"] == 0
        and e["scale_and_bias_equal"] and e["params_block_equal"]
        and e["learn_best_error_equal"]
        and not e["model_info_unexpected_diff"]
        and not e["top_level_keys_differing_excluding_model_info"])
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-zip", required=True)
    ap.add_argument("--branch", default="F")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--refit-model-dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    z = zipfile.ZipFile(a.parent_zip)
    res = {"gate": "G1_F_seed42_refit_is_model_identical_to_shipped",
           "parent_zip": a.parent_zip,
           "parent_zip_sha256": hashlib.sha256(open(a.parent_zip, "rb").read()).hexdigest(),
           "refit_model_dir": a.refit_model_dir,
           "bytes_are_NOT_compared_because": (
               "CatBoost writes model_guid (fresh UUID) and train_finish_time "
               "(wall clock) into every .cbm; three identical fits give three "
               "different file hashes and identical leaf hashes (candidate AG). "
               "A byte gate here would test the clock, not the model."),
           "per_model": {}}
    tmpd = tempfile.mkdtemp()
    allok = True
    for name in MODELS:
        member = f"model/{a.branch}/seed_{a.seed}/{name}"
        pp = os.path.join(tmpd, f"parent_{name}")
        with open(pp, "wb") as f:
            f.write(z.read(member))
        e = compare(pp, os.path.join(a.refit_model_dir, name), member)
        allok &= e["PASS"]
        res["per_model"][name] = e
        print(f"{name:<26} trees {e['tree_count']}  leaves "
              f"{e['leaf_values_identical']}  weights {e['leaf_weights_identical']}"
              f"  splits {e['n_trees_with_differing_splits']}  meta "
              f"{e['model_info_keys_differing']}  -> "
              f"{'PASS' if e['PASS'] else 'FAIL'}", flush=True)
        os.unlink(pp)
    res["ALL_PASS"] = bool(allok)
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print("G1_ALL_PASS", allok, flush=True)
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
