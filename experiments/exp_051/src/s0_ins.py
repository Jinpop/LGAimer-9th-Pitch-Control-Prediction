"""exp_051 STAGE 0 — inherit-check + build the ins_* matrix ONCE.

G0a (matrix inheritance): exp_048 did NOT save the per-season ins matrix as a
file (checked: experiments/exp_048/**.npz contains only the two anchors), so
the config's fallback clause applies — REGENERATE with exp_048's generation
code and compare against exp_048's records.  Concretely:
  1. the four generation inputs (train.csv, ins_feats.py, inseason.py,
     anchor_lt2024.npz) are sha256-asserted equal to the digests exp_048's
     three res JSONs recorded (and the three JSONs are asserted to agree);
  2. the regenerated matrix's per-season stats (n_rows, n_anchor_pitchers,
     ins_p_n_mean, ins_p_n_zero_pct) and NaN percentages are asserted equal
     to the values exp_048 recorded in res_exp048_s1_seed42.json.
  The generation is deterministic vector arithmetic (no RNG, no threading),
  so byte-identical inputs give the identical matrix; the value comparison
  is the observable check on top of that argument.

G0b (anchor for fold_2024 valid rows): build_ins_matrix itself asserts
anchor_lt2024.npz == build_anchor(train[season < 2024]) at build time, and
the season-2024 policy line must say the frozen anchor was used.

G0c (missing policy): earliest season (2019) all-NaN, no sentinel — inherited
code, exp_048-audited; the NaN percentages are recorded here.

Also pre-checks the control artefacts this experiment will consume:
exp_048 A1 stored predictions (sha vs exp_048 res) and AH OOF files (sha vs
local_reference_AH.json).
"""

import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.join(SRC_DIR, "ins"))

import inseason as ins                    # noqa: E402
from ins_feats import build_ins_matrix, COLS16   # noqa: E402

DATA = os.path.expanduser("~/SungjinPark/data/train.csv")
OUT = os.path.expanduser("~/SungjinPark/experiments/exp_051")
E48 = os.path.expanduser("~/SungjinPark/experiments/exp_048")
AH_REF = os.path.expanduser("~/SungjinPark/build_AH/local_reference_AH.json")
AH_OOF = os.path.expanduser("~/SungjinPark/build_AH/localref/oof")
PACK = os.path.join(SRC_DIR, "ins")
SEEDS = (42, 7, 2024)
F_EXCLUDE_MAX_SEASON = 2022


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
    t0 = time.time()
    os.makedirs(os.path.join(OUT, "res"), exist_ok=True)
    out = {"exp_id": "exp_051", "stage": 0,
           "what": ("G0a matrix inheritance, G0b valid-row anchor, G0c "
                    "missing policy, control artefact availability")}

    # ---- G0a part 1: generation inputs byte-equal exp_048's records --------
    recs = {}
    for s in SEEDS:
        p = os.path.join(E48, "res", f"res_exp048_s1_seed{s}.json")
        with open(p) as f:
            recs[s] = {"path": p, "sha256": json.load(f)["sha256"]}
    assert recs[42]["sha256"] == recs[7]["sha256"] == recs[2024]["sha256"], \
        "exp_048's three res JSONs disagree on the input shas"
    rec = recs[42]["sha256"]
    obs = {"train.csv": sha256_file(DATA),
           "ins_feats.py": sha256_file(os.path.join(SRC_DIR, "ins_feats.py")),
           "inseason.py": sha256_file(os.path.join(PACK, "inseason.py")),
           "anchor_lt2024.npz": sha256_file(
               os.path.join(PACK, "anchor_lt2024.npz"))}
    g0a1 = {k: {"exp048_recorded": rec[k], "observed": obs[k],
                "match": bool(rec[k] == obs[k])} for k in obs}
    assert all(v["match"] for v in g0a1.values()), g0a1
    out["G0a_inputs_sha_vs_exp048"] = {
        "recorded_in": [recs[s]["path"] for s in SEEDS],
        "all_three_res_agree": True, "per_input": g0a1}
    out["G0a_no_saved_matrix_note"] = (
        "exp_048 saved no matrix file (its .npz files are the two anchors "
        "only) — the matrix is regenerated below with the byte-identical "
        "inputs asserted above; build_ins_matrix is deterministic "
        "(vector arithmetic, no RNG, no threading).")

    # ---- build --------------------------------------------------------------
    print(f"[{time.strftime('%H:%M:%S')}] stage0: reading train.csv",
          flush=True)
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    assert df["row_id"].is_unique
    assert df["season"].is_monotonic_increasing
    t_b = time.time()
    V, cols, info = build_ins_matrix(df, ins, PACK, valid_season=2024)
    assert cols == COLS16 and len(cols) == 16
    out["build_sec"] = round(time.time() - t_b, 1)

    # ---- G0a part 2: value comparison against exp_048's recorded stats -----
    with open(os.path.join(E48, "res", "res_exp048_s1_seed42.json")) as f:
        e48_ins = json.load(f)["ins_columns"]
    cmp_sp = {}
    for s_key, rec_sp in e48_ins["season_policy"].items():
        got = info["season_policy"][s_key]
        same = all(got.get(k) == rec_sp.get(k) for k in rec_sp)
        cmp_sp[s_key] = {"exp048": rec_sp, "observed": got,
                         "match": bool(same)}
    assert all(v["match"] for v in cmp_sp.values()), cmp_sp
    out["G0a_season_policy_vs_exp048"] = cmp_sp
    assert e48_ins["cols"] == cols

    season = df["season"].to_numpy()
    f_excl = ((df["season"] <= F_EXCLUDE_MAX_SEASON)
              & (df["game_type"] == "F")).to_numpy()
    tr_full = (season <= 2023) & ~f_excl
    va = season == 2024
    nan_tr = round(100.0 * float(np.isnan(V[tr_full, 0]).mean()), 4)
    nan_va = round(100.0 * float(np.isnan(V[va, 0]).mean()), 4)
    assert nan_tr == e48_ins["nan_pct_in_train_rows"], (
        nan_tr, e48_ins["nan_pct_in_train_rows"])
    assert nan_va == e48_ins["nan_pct_in_valid_rows"] == 0.0
    out["G0a_nan_pcts_vs_exp048"] = {
        "train_rows": {"exp048": e48_ins["nan_pct_in_train_rows"],
                       "observed": nan_tr},
        "valid_rows": {"exp048": e48_ins["nan_pct_in_valid_rows"],
                       "observed": nan_va}}
    out["G0a_PASS"] = True

    # ---- G0b: fold_2024 valid rows use the <=2023 anchor -------------------
    assert info["frozen_anchor_equals_rebuild"] is True
    pol_2024 = info["season_policy"]["2024"]
    assert pol_2024["anchor"] == "anchor_lt2024.npz (frozen, verified)"
    assert pol_2024["ins_p_n_zero_pct"] < 1.0, (
        "deploy-anchor degeneracy would put this at 100.0 (exp_048 trap)",
        pol_2024)
    out["G0b_valid_row_anchor"] = {
        "anchor_used_for_2024_rows": pol_2024["anchor"],
        "frozen_anchor_equals_rebuild_of_train_lt2024": True,
        "ins_p_n_zero_pct_2024": pol_2024["ins_p_n_zero_pct"],
        "degeneracy_reference": ("exp_048 G0d measured 100.0% zeros with the "
                                 "deploy anchor; this run's value above is "
                                 "the non-degenerate one"),
        "PASS": True}

    # ---- G0c: missing policy (no sentinel) ---------------------------------
    out["G0c_missing_policy"] = {
        "earliest_season_2019": info["season_policy"]["2019"]["anchor"],
        "nan_pct_train_rows": nan_tr, "nan_pct_valid_rows": nan_va,
        "note": ("inherited exp_048 code: 2019 rows carry NaN in all 16 "
                 "columns (no prior-season anchor exists); no sentinel fill "
                 "anywhere in ins_feats.py — audited in exp_048/AI, not "
                 "re-audited here per config stage0"),
        "PASS": True}

    # ---- save the matrix ----------------------------------------------------
    mpath = os.path.join(OUT, "ins_matrix_fold2024.npz")
    np.savez_compressed(mpath, values=V, cols=np.array(cols),
                        train_csv_sha256=np.array(obs["train.csv"]))
    out["ins_matrix"] = {"path": mpath, "sha256": sha256_file(mpath),
                         "shape": list(V.shape),
                         "bytes": os.path.getsize(mpath)}
    print(f"    matrix saved: {mpath} sha {out['ins_matrix']['sha256'][:16]}…",
          flush=True)

    # ---- control artefact availability -------------------------------------
    ctrl = {}
    for s in SEEDS:
        with open(os.path.join(E48, "res",
                               f"res_exp048_s1_seed{s}.json")) as f:
            r48 = json.load(f)
        a1 = r48["arms"]["A1"]["by_round"]["1400"]
        got_sha = sha256_file(a1["pred_file"])
        ctrl[str(s)] = {
            "pred_file": a1["pred_file"],
            "recorded_sha256": a1["pred_sha256"],
            "observed_sha256": got_sha,
            "match": bool(got_sha == a1["pred_sha256"]),
            "recorded_brier_1400": a1["valid_all_brier"]}
    assert all(v["match"] for v in ctrl.values()), ctrl
    out["control_exp048_A1_pred_sha"] = ctrl

    with open(AH_REF) as f:
        ahref = json.load(f)
    ahchk = {}
    for s in SEEDS:
        p = os.path.join(AH_OOF, f"stack_fold2024_seed{s}.npz")
        got_sha = sha256_file(p)
        want = ahref["per_seed"][str(s)]["run"]["oof_sha256"]
        ahchk[str(s)] = {
            "oof_file": p, "recorded_sha256": want,
            "observed_sha256": got_sha, "match": bool(got_sha == want),
            "cell_iter_used":
                ahref["per_seed"][str(s)]["run"]["cell_iter_used"]}
    assert all(v["match"] for v in ahchk.values()), ahchk
    ci_sets = [json.dumps(ahchk[str(s)]["cell_iter_used"], sort_keys=True)
               for s in SEEDS]
    assert len(set(ci_sets)) == 1, ("cell iters differ across seeds", ci_sets)
    out["ah_oof_sha_vs_reference_json"] = ahchk
    out["cell_iter_to_be_pinned"] = ahchk["42"]["cell_iter_used"]
    out["cell_iter_source"] = {
        "path": AH_REF, "key": "per_seed.<seed>.run.cell_iter_used"}

    out["ALL_PASS"] = True
    out["minutes"] = round((time.time() - t0) / 60, 2)
    with open(os.path.join(OUT, "res", "stage0_gates.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("STAGE0 RESULT: " + json.dumps({
        "ALL_PASS": True, "matrix_sha": out["ins_matrix"]["sha256"],
        "cell_iter": out["cell_iter_to_be_pinned"],
        "minutes": out["minutes"]}), flush=True)


if __name__ == "__main__":
    main()
