"""exp_097 stageA: (a) float32 signature scan of every shipped JSON constant,
(b) independent float64 rebuild of the shipped frozen TABLES.

(a) A float64 value that survives a round trip through float32 unchanged almost
    never happens by accident (probability ~2^-29 per value).  A block of such
    values means the constant was COMPUTED or STORED in float32.
(b) inseason.build_anchor and splits.build are shipped, so they can be re-run
    on train.csv and compared to the shipped npz element by element.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

BASE = "/home/<user>/SungjinPark"
PKG = os.path.join(BASE, "exp097/pkg/CH")
OUT = os.path.join(BASE, "exp097/out/sA_tables.json")
res = {}

# ------------------------------------------------------------------ (a) scan
def walk(o, path, acc):
    if isinstance(o, dict):
        for k, v in o.items():
            walk(v, path + "/" + str(k), acc)
    elif isinstance(o, list):
        if o and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                     for x in o):
            acc.append((path, [float(x) for x in o]))
        else:
            for i, v in enumerate(o):
                walk(v, path + "[%d]" % i, acc)
    elif isinstance(o, float):
        acc.append((path, [o]))


scan = {}
for root, _dirs, files in os.walk(PKG):
    for fn in files:
        if not fn.endswith(".json"):
            continue
        p = os.path.join(root, fn)
        rel = os.path.relpath(p, PKG)
        try:
            o = json.load(open(p, encoding="utf-8"))
        except Exception as e:
            scan[rel] = {"error": str(e)}
            continue
        acc = []
        walk(o, "", acc)
        blocks = []
        n_tot = n_f32 = 0
        for path, vals in acc:
            v = np.array(vals, dtype=np.float64)
            v = v[np.isfinite(v) & (v != 0.0)]
            if v.size == 0:
                continue
            same = (v.astype(np.float32).astype(np.float64) == v)
            n_tot += v.size
            n_f32 += int(same.sum())
            if v.size >= 4 and same.mean() > 0.9:
                blocks.append({"path": path, "n": int(v.size),
                               "frac_float32_exact": float(same.mean())})
        if n_tot:
            scan[rel] = {"n_floats": n_tot, "n_float32_exact": n_f32,
                         "frac": round(n_f32 / n_tot, 6),
                         "float32_blocks": blocks[:20]}
res["json_float32_signature_scan"] = scan
print("=== (a) float32 signature per shipped JSON ===")
for k in sorted(scan):
    s = scan[k]
    if "frac" in s:
        flag = "  <== FLOAT32" if s["frac"] > 0.5 else ""
        print("  %-42s n=%-7d frac_f32_exact=%.4f%s"
              % (k, s["n_floats"], s["frac"], flag))
        for b in s["float32_blocks"][:6]:
            print("        block %s n=%d frac=%.3f"
                  % (b["path"][:70], b["n"], b["frac_float32_exact"]))

# ------------------------------------------------------------------ (b) rebuild
sys.path.insert(0, PKG)
import inseason as INS           # noqa: E402
sys.path.insert(0, os.path.join(PKG, "mate"))
import splits as SP              # noqa: E402

TRAIN = None
for cand in (os.path.join(BASE, "data/train.csv"),
             os.path.join(BASE, "exp064/data/train.csv"),
             os.path.join(BASE, "train.csv")):
    if os.path.exists(cand):
        TRAIN = cand
        break
res["train_csv"] = TRAIN
print("\n=== (b) rebuild of shipped frozen tables ===\ntrain.csv:", TRAIN)
if TRAIN:
    tr = pd.read_csv(TRAIN, encoding="utf-8-sig", low_memory=False)
    hist = tr[tr.season <= 2024]
    print("  train rows", len(tr), " <=2024 rows", len(hist))

    A_new = INS.build_anchor(hist)
    for shipped in ("mate/model/anchor.npz", "model/ins/deploy_anchor.npz"):
        z = np.load(os.path.join(PKG, shipped))
        d = {}
        for k in z.files:
            a = np.asarray(z[k], dtype=np.float64)
            b = np.asarray(A_new[k], dtype=np.float64)
            if a.shape != b.shape:
                d[k] = {"shape_shipped": list(a.shape),
                        "shape_rebuilt": list(b.shape)}
                continue
            d[k] = {"max_abs_diff": float(np.nanmax(np.abs(a - b))),
                    "n_diff": int((a != b).sum()), "n": int(a.size)}
        res["anchor__" + shipped.replace("/", "_")] = d
        print("  %s:" % shipped)
        for k, v in d.items():
            print("     ", k, v)

    T_new = SP.build(hist)
    z = np.load(os.path.join(PKG, "mate/model/deploy_splits.npz"))
    d = {"league": {"shipped": float(z["league"][0]),
                    "rebuilt": float(T_new["league"]),
                    "abs_diff": abs(float(z["league"][0]) - T_new["league"])}}
    for name, A in T_new["axes"].items():
        for k2 in ("ids", "D", "n", "cn", "p_all"):
            a = np.asarray(z["%s__%s" % (name, k2)], dtype=np.float64)
            b = np.asarray(A[k2], dtype=np.float64)
            if a.shape != b.shape:
                d["%s__%s" % (name, k2)] = {"shape_shipped": list(a.shape),
                                            "shape_rebuilt": list(b.shape)}
                continue
            d["%s__%s" % (name, k2)] = {"max_abs_diff":
                                        float(np.nanmax(np.abs(a - b))),
                                        "n_diff": int((a != b).sum()),
                                        "n": int(a.size)}
    res["deploy_splits"] = d
    print("  mate/model/deploy_splits.npz:")
    for k, v in d.items():
        print("     ", k, v)

json.dump(res, open(OUT, "w"), indent=1)
print("WROTE", OUT)
