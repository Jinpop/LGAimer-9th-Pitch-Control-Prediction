"""exp_044 — assemble a per-model parameter map from the search results.

    --which cellsA   the four phase-1 cell winners + the exp_042 winner in the
                     stacker slot.  This map exists only to generate the
                     forward-chaining OOF columns the stacker search needs; its
                     stacker slot is never promoted.
    --which final    the four cell winners + the phase-2 stacker winner.

Nothing here invents a number: every value is read out of a res/best_*.json
that a search wrote, and the map records which file each slot came from with
that file's sha256.
"""

import argparse
import hashlib
import json
import os
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)
from labels import FAIL_CELLS                                       # noqa: E402

OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_044")


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", required=True, choices=["cellsA", "final"])
    a = ap.parse_args()

    prov = {}
    cells = {}
    for c in FAIL_CELLS:
        p = os.path.join(OUT_DIR, "res", f"best_{c}.json")
        cells[c] = json.load(open(p))
        prov[c] = {"file": os.path.basename(p), "sha256": sha256_file(p)}

    if a.which == "cellsA":
        p = os.path.join(OUT_DIR, "maps", "map_ctrl.json")
        stk = json.load(open(p))["stacker"]
        prov["stacker"] = {"file": "maps/map_ctrl.json (exp_042 winner — "
                                   "placeholder, never promoted)",
                           "sha256": sha256_file(p)}
        out = os.path.join(OUT_DIR, "maps", "map_pcA.json")
    else:
        p = os.path.join(OUT_DIR, "res", "best_stacker.json")
        stk = json.load(open(p))
        prov["stacker"] = {"file": os.path.basename(p),
                           "sha256": sha256_file(p)}
        out = os.path.join(OUT_DIR, "maps", "map_pc.json")

    m = {"cells": cells, "stacker": stk}
    with open(out, "w") as f:
        json.dump(m, f, indent=1)
    with open(out.replace(".json", "_provenance.json"), "w") as f:
        json.dump({"which": a.which, "slots": prov,
                   "map_sha256": sha256_file(out)}, f, indent=1)
    print("WROTE " + out)
    print(json.dumps(prov, indent=1))


if __name__ == "__main__":
    main()
