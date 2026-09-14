"""exp_087 — assemble a MEASUREMENT package whose F branch carries the real
seeds fitted by this experiment.  NOT a submission: it lives under exp087/pkg/
and exists so that the 18-seed and 3-seed final predictions can be compared
through the shipped inference path rather than through a reimplementation of it.

The only edits to the parent package are:
  * new directories model/F/seed_<s>/ holding the five .cbm files this
    experiment fitted (copied, never re-derived — S17: only copying is identity)
  * branches.F.seeds and branches.F.seed_dirs extended to list them
Everything else — R branch, mate, v6, tables, blend.json, script.py — is the
parent's bytes.
"""
import hashlib, json, os, shutil, subprocess, sys

H = "/home/<user>/SungjinPark"
BASE = f"{H}/exp087/pkg/f3"                     # parent package, unmodified copy
SEEDS_DIR = f"{H}/exp087/seeds"
MODELS = ["cell_is_both.cbm", "cell_is_mid_only.cbm", "cell_is_rev_only.cbm",
          "cell_is_wayoff_true.cbm", "stacker.cbm"]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def build(tag, new_seeds):
    dst = f"{H}/exp087/pkg/{tag}"
    if os.path.exists(dst):
        shutil.rmtree(dst)
    subprocess.run(["cp", "-r", BASE, dst], check=True)
    mp = f"{dst}/model/meta.json"
    m = json.load(open(mp))
    F = m["branches"]["F"]
    assert F["seeds"] == [42, 7, 2024], F["seeds"]
    shas = {}
    for s in new_seeds:
        src = f"{SEEDS_DIR}/seed_{s}/model"
        d = f"{dst}/model/F/seed_{s}"
        os.makedirs(d, exist_ok=True)
        for name in MODELS:
            shutil.copy2(f"{src}/{name}", f"{d}/{name}")
            shas[f"F/seed_{s}/{name}"] = sha(f"{d}/{name}")
        F["seeds"].append(int(s))
        F["seed_dirs"][str(s)] = f"F/seed_{s}"
    F["averaging"]["n_seeds"] = len(F["seeds"])
    F["averaging"]["form"] = ("p_bar = (" + " + ".join(f"p_{s}" for s in F["seeds"])
                              + f") / {len(F['seeds'])}, per row")
    F["exp_087"] = {
        "what": "F-branch seed expansion 3 -> %d" % len(F["seeds"]),
        "new_seed_shas": shas,
        "recipe": ("candidate S's train_final_c5_feat63.py, --cell-policy "
                   "stage1_scaled, unchanged; only --seed differs"),
        "MEASUREMENT_PACKAGE_NOT_A_SUBMISSION": True,
    }
    json.dump(m, open(mp, "w"), ensure_ascii=False, indent=1)
    print(f"{tag}: F seeds = {len(F['seeds'])} -> {dst}")
    return dst


if __name__ == "__main__":
    tag = sys.argv[1]
    build(tag, [int(x) for x in sys.argv[2:]])
