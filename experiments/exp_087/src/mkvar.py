"""exp_087 G4 harness — build inference-time probe packages with N F-branch seeds.

The extra seed directories are BYTE COPIES of the shipped F/seed_42.  That makes
the PREDICTIONS of these probe packages meaningless (a duplicated model adds no
information) but leaves the COMPUTE identical to a package with N distinct F
seeds: aa_infer loads one directory per entry of `seeds` and runs five boosters
over every test row for each.  Inference time is what is being measured and it
does not depend on the leaf values.

These packages are NEVER submitted.  They live under exp087/pkg/ and are used by
the timing runs only.
"""
import json, os, shutil, subprocess, sys

H = "/home/<user>/SungjinPark"
BASE = f"{H}/exp087/pkg/f3"
# 15 fake seed ids that cannot collide with the real ones
FAKE = [900001 + i for i in range(15)]

def build(n):
    dst = f"{H}/exp087/pkg/f{n}"
    if os.path.exists(dst):
        shutil.rmtree(dst)
    subprocess.run(["cp", "-r", BASE, dst], check=True)
    mp = f"{dst}/model/meta.json"
    m = json.load(open(mp))
    F = m["branches"]["F"]
    assert F["seeds"] == [42, 7, 2024], F["seeds"]
    add = n - 3
    assert 0 <= add <= 15
    for i in range(add):
        s = FAKE[i]
        sd = f"F/seed_{s}"
        subprocess.run(["cp", "-r", f"{dst}/model/F/seed_42", f"{dst}/model/{sd}"],
                       check=True)
        F["seeds"].append(s)
        F["seed_dirs"][str(s)] = sd
    F["averaging"]["n_seeds_probe"] = n
    F["PROBE_NOT_A_SUBMISSION"] = (
        "exp_087 G4 timing probe: seeds beyond the shipped three are BYTE COPIES "
        "of F/seed_42.  Predictions are meaningless; only wall time is read.")
    json.dump(m, open(mp, "w"), ensure_ascii=False, indent=1)
    print(f"f{n}: seeds={len(F['seeds'])} dirs={len(F['seed_dirs'])}")

for n in [int(x) for x in sys.argv[1:]]:
    build(n)
