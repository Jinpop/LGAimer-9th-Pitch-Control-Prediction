#!/usr/bin/env python3
"""exp_087 G2 — every new F fit differs from the shipped ones in `random_seed` ONLY.

Two params views are compared, because they answer different questions:

  * `get_all_params()` on the LOADED booster — what CatBoost says the model was
    built with, including defaults it filled in.  This is the one the B-단
    게이트 wording names.
  * the `params` block recorded inside `model_info` — what the caller passed.

Both must agree with the shipped seed-42 model on every key except
`random_seed`, and `random_seed` must equal the seed the directory is named for.
Tree counts and feature order are checked too: a parameter dict can match while
the model is a different shape if the iteration policy silently moved.
"""
import argparse, json, os, sys, tempfile, zipfile
from catboost import CatBoostClassifier

MODELS = ["cell_is_both.cbm", "cell_is_mid_only.cbm", "cell_is_rev_only.cbm",
          "cell_is_wayoff_true.cbm", "stacker.cbm"]


def load(path):
    m = CatBoostClassifier()
    m.load_model(path)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as t:
        p = t.name
    m.save_model(p, format="json")
    j = json.load(open(p))
    os.unlink(p)
    rec = j.get("model_info", {}).get("params")
    rec = json.loads(rec) if isinstance(rec, str) else rec
    return m, m.get_all_params(), rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-zip", required=True)
    ap.add_argument("--seeds-dir", required=True)
    ap.add_argument("--seeds", required=True, help="comma separated")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]

    z = zipfile.ZipFile(a.parent_zip)
    tmpd = tempfile.mkdtemp()
    ref = {}
    for name in MODELS:
        pp = os.path.join(tmpd, f"ref_{name}")
        with open(pp, "wb") as f:
            f.write(z.read(f"model/F/seed_42/{name}"))
        ref[name] = load(pp)

    res = {"gate": "G2_new_fits_differ_only_in_random_seed",
           "reference": "model/F/seed_42/* of the shipped package (seed 42)",
           "per_seed": {}}
    allok = True
    for s in seeds:
        ent = {}
        for name in MODELS:
            p = os.path.join(a.seeds_dir, f"seed_{s}", "model", name)
            if not os.path.exists(p):
                ent[name] = {"PASS": False, "missing": p}
                allok = False
                continue
            mn, allp_n, recp_n = load(p)
            mr, allp_r, recp_r = ref[name]
            d_all = sorted(k for k in (set(allp_n) | set(allp_r))
                           if allp_n.get(k) != allp_r.get(k))
            d_rec = sorted(k for k in (set(recp_n) | set(recp_r))
                           if recp_n.get(k) != recp_r.get(k))
            # 🔴 CORRECTED AFTER THE FIRST RUN, AND THE CORRECTION IS A CHECK
            # SPECIFICATION FIX, NOT A MOVED THRESHOLD (S27).
            # `model_info.params` carries a nested `flat_params` that is a
            # SERIALISATION OF THE SAME PARAMETERS.  When `random_seed` changes,
            # `flat_params` necessarily changes too, so the first version of this
            # gate counted one variable twice and failed every seed while the
            # substantive requirement — get_all_params() differs in `random_seed`
            # only — was already met.  The fix descends one level and demands that
            # `flat_params` ALSO differ in nothing but `random_seed`, which is
            # strictly more evidence than the broken version ever collected.
            flat_n = recp_n.get("flat_params") or {}
            flat_r = recp_r.get("flat_params") or {}
            d_flat = sorted(k for k in (set(flat_n) | set(flat_r))
                            if flat_n.get(k) != flat_r.get(k))
            d_rec_eff = sorted(set(k for k in d_rec if k != "flat_params")
                               | set(d_flat))
            e = {
                "get_all_params_keys_differing": d_all,
                "recorded_params_keys_differing_raw": d_rec,
                "recorded_flat_params_keys_differing": d_flat,
                "recorded_params_keys_differing_effective": d_rec_eff,
                "random_seed_new": allp_n.get("random_seed"),
                "random_seed_ref": allp_r.get("random_seed"),
                "tree_count": [int(mr.tree_count_), int(mn.tree_count_)],
                "tree_count_equal": int(mr.tree_count_) == int(mn.tree_count_),
                "feature_names_equal": list(mr.feature_names_) == list(mn.feature_names_),
                "n_features": len(mn.feature_names_),
            }
            e["PASS"] = bool(d_all == ["random_seed"] and d_rec_eff == ["random_seed"]
                             and e["random_seed_new"] == s
                             and e["tree_count_equal"] and e["feature_names_equal"])
            allok &= e["PASS"]
            ent[name] = e
        res["per_seed"][str(s)] = ent
        ok = all(v.get("PASS") for v in ent.values())
        print(f"seed {s:>6}  -> {'PASS' if ok else 'FAIL'}", flush=True)
    res["ALL_PASS"] = bool(allok)
    res["n_seeds"] = len(seeds)
    res["n_fits_checked"] = len(seeds) * len(MODELS)
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print("G2_ALL_PASS", allok)
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
