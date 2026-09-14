"""exp_051 — derive train51.py from candidate AH's local_ref_stack.py.

Convention (exp_046, recorded in orchestration/실행시간_실측.md): NEVER re-type a
parent pipeline.  Read it, assert its sha, apply REGISTERED string
substitutions (each asserted to match exactly once), write the child, and save
a unified diff.

Parent: ~/SungjinPark/build_AH/localref/src/local_ref_stack.py
        sha256 1993f63e3468133c904f1294fc3749057a7218cd8c55905a18f623a1ef486c85
        (the file that generated the control OOF + the AH local reference)

The one experimental change (config exp_051 `the_one`): the four CELL models'
input goes 47 -> 63 columns (+ the 16 ins_* columns), so the OOF must be
regenerated (same forward-chaining, same season blocks, same PINNED cell
iterations read from local_reference_AH.json) and the stacker refit (67
columns exactly as exp_048 A1, rounds fixed at 1400).

Everything else in the substitution list is bookkeeping forced by that change:
  * the exp_030 reproduction gate cannot hold (the stacker's design matrix
    changed by design) -> replaced by the exp_051 control gate: exp_048 A1
    stored predictions, sha-compared, Brier recomputed; plus AH-OOF mask
    cross-checks and the OOF movement measurement the config asks for.
  * fit_catboost -> fit_cb (exp_048 s1_stack.py's wrapper, verbatim: the same
    fit plus the model object, needed for the ins gain share measurement;
    exp_048's A0 gate proved the wrapper bit-identical to fit_catboost).
  * route-A refits at r < cap dropped (exp_047 measured staged prefixes
    bit-identical to independent fits; exp_048's 655/1000 numbers are staged).
"""

import difflib
import hashlib
import os
import sys

PARENT_SHA = "1993f63e3468133c904f1294fc3749057a7218cd8c55905a18f623a1ef486c85"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


# --------------------------------------------------------------------------
# the registered substitutions.  (old, new) — old must occur EXACTLY once.
# --------------------------------------------------------------------------
SUBS = []

# S1 — module docstring: prepend the child's identity, keep the parent's text
SUBS.append((
    '"""exp_036 — the UNCENSORED fold_2024 argmin of the B3 stacker.\n',
    '"""exp_051 — 63-column cells (47 + 16 ins_*): OOF regenerated, stacker '
    'refit at 1400.\n\nDerived from build_AH/localref/src/local_ref_stack.py '
    '(sha 1993f63e...86c85) by\nregistered substitutions '
    '(patch_train51.py); unified diff saved as train51.diff.\nThe parent '
    'docstring is kept below for lineage.\n\n--- parent docstring '
    '---\nexp_036 — the UNCENSORED fold_2024 argmin of the B3 stacker.\n'))

# S2 — identity + output directory
SUBS.append((
    'EXP_ID = "AH_local_reference"\n'
    'DATA_PATH = os.path.expanduser("~/SungjinPark/data/train.csv")\n'
    'OUT_DIR = os.path.expanduser("~/SungjinPark/build_AH/localref")\n',
    'EXP_ID = "exp_051"\n'
    'DATA_PATH = os.path.expanduser("~/SungjinPark/data/train.csv")\n'
    'OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_051")\n'))

# S3 — imports: the ins pack and the ins_feats builder (both inherited from
# exp_048 byte-identically; stage 0 asserts their shas against exp_048's res)
SUBS.append((
    'from labels import (FAIL_CELLS, assert_no_recovered_in_features,     '
    '# noqa: E402\n                    recover)\n',
    'from labels import (FAIL_CELLS, assert_no_recovered_in_features,     '
    '# noqa: E402\n                    recover)\n'
    'sys.path.insert(0, os.path.join(SRC_DIR, "ins"))\n'))

# S4 — new paths (control artefacts, AH reference, the stage-0 ins matrix)
SUBS.append((
    'EXP030_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_030")\n',
    'EXP030_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_030")\n'
    'EXP048_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_048")\n'
    'AH_REF_JSON = os.path.expanduser(\n'
    '    "~/SungjinPark/build_AH/local_reference_AH.json")\n'
    'AH_OOF_DIR = os.path.expanduser("~/SungjinPark/build_AH/localref/oof")\n'
    'AH_RES_DIR = os.path.expanduser("~/SungjinPark/build_AH/localref/res")\n'
    'INS_MATRIX = os.path.expanduser(\n'
    '    "~/SungjinPark/experiments/exp_051/ins_matrix_fold2024.npz")\n'))

# S5 — fit_cb: exp_048 s1_stack.py's wrapper VERBATIM (fit + model object)
SUBS.append((
    'def score_of(brier):\n',
    'def fit_cb(e27, spec, seed, F_tr, y_tr, F_va, feats, cat_native, '
    'n_round):\n'
    '    """exp_048 s1_stack.py `fit_cb` VERBATIM — exp_027\'s fit_catboost '
    'plus the\n'
    '    model object.  exp_048\'s A0 gate proved it bit-identical to '
    'fit_catboost."""\n'
    '    import catboost as cb\n'
    '    params = dict(e27.CB_BASE, random_seed=seed, iterations=n_round,\n'
    '                  **spec["params"])\n'
    '    cat_idx = [feats.index(c) for c in cat_native]\n'
    '    pool_tr = cb.Pool(F_tr, label=y_tr, cat_features=cat_idx)\n'
    '    pool_va = cb.Pool(F_va, cat_features=cat_idx)\n'
    '    model = cb.CatBoostClassifier(**params)\n'
    '    model.fit(pool_tr)\n'
    '    assert model.tree_count_ == n_round, (model.tree_count_, n_round)\n'
    '\n'
    '    def predict_range(a, b):\n'
    '        return model.predict(pool_va, prediction_type="RawFormulaVal",\n'
    '                             ntree_start=a, ntree_end=b,\n'
    '                             thread_count=N_THREADS).astype(np.float64)\n'
    '\n'
    '    return predict_range, params, model\n'
    '\n'
    '\n'
    'def score_of(brier):\n'))

# S6a — --cell-iter becomes optional (the values are READ from the artefact)
SUBS.append((
    '    ap.add_argument("--cell-iter", required=True,\n'
    '                    help="JSON dict {cell: iterations}; pinned from '
    'outside")\n',
    '    ap.add_argument("--cell-iter", default=None,\n'
    '                    help="optional cross-check; the values are READ from '
    'local_reference_AH.json")\n'))

# S6b — cell iterations read from the control's artefact, never transcribed
SUBS.append((
    '    ci = json.loads(args.cell_iter)\n',
    '    with open(AH_REF_JSON) as fah:\n'
    '        ahref = json.load(fah)\n'
    '    ah_seed = ahref["per_seed"][str(SEED)]\n'
    '    ci = dict(ah_seed["run"]["cell_iter_used"])\n'
    '    res["cell_iter_source"] = {\n'
    '        "path": AH_REF_JSON,\n'
    '        "key": f"per_seed.{SEED}.run.cell_iter_used",\n'
    '        "values_read": {k: int(v) for k, v in ci.items()}}\n'
    '    if args.cell_iter:\n'
    '        assert json.loads(args.cell_iter) == ci, (args.cell_iter, ci)\n'))

# S7 — load the stage-0 ins matrix, add the 16 columns, define cell_feats(63)
SUBS.append((
    '    cat_native = [c for c in base_feats if c in CAT_COLS]\n'
    '    assert len(cat_native) == 3, cat_native\n'
    '    assert not any(c in cat_native for c in ID_COLS), "C5 keeps IDs '
    'numeric"\n',
    '    cat_native = [c for c in base_feats if c in CAT_COLS]\n'
    '    assert len(cat_native) == 3, cat_native\n'
    '    assert not any(c in cat_native for c in ID_COLS), "C5 keeps IDs '
    'numeric"\n'
    '\n'
    '    # ---- exp_051: the 16 ins_* columns, built once by s0_ins.py -------\n'
    '    zi = np.load(INS_MATRIX, allow_pickle=False)\n'
    '    assert str(zi["train_csv_sha256"].item()) == sha256_file(DATA_PATH), \\\n'
    '        "the ins matrix was built on a different train.csv"\n'
    '    ins_cols = [str(c) for c in zi["cols"].tolist()]\n'
    '    assert len(ins_cols) == 16, ins_cols\n'
    '    V_ins = np.ascontiguousarray(zi["values"], dtype=np.float64)\n'
    '    assert V_ins.shape == (len(df), 16), V_ins.shape\n'
    '    assert not np.isnan(V_ins[va, 0]).any(), "valid rows must carry no NaN"\n'
    '    for j_ins, c_ins in enumerate(ins_cols):\n'
    '        df[c_ins] = V_ins[:, j_ins]\n'
    '    ins_matrix_sha = sha256_file(INS_MATRIX)\n'
    '    cell_feats = base_feats + ins_cols\n'
    '    assert len(cell_feats) == 63, len(cell_feats)\n'
    '    assert_no_recovered_in_features(cell_feats, "cell_feats (63)")\n'
    '    del V_ins\n'))

# S8 — record the new shas + column counts once `res` exists
SUBS.append((
    '    res["versions"]["catboost"] = cb.__version__\n',
    '    res["versions"]["catboost"] = cb.__version__\n'
    '    res["sha256"]["ins_matrix_fold2024.npz"] = ins_matrix_sha\n'
    '    res["sha256"]["ins_feats.py"] = sha256_file(\n'
    '        os.path.join(SRC_DIR, "ins_feats.py"))\n'
    '    res["sha256"]["inseason.py"] = sha256_file(\n'
    '        os.path.join(SRC_DIR, "ins", "inseason.py"))\n'
    '    res["sha256"]["anchor_lt2024.npz"] = sha256_file(\n'
    '        os.path.join(SRC_DIR, "ins", "anchor_lt2024.npz"))\n'
    '    res["cell_n_features"] = 63\n'
    '    res["stacker_n_features"] = 67\n'
    '    res["ins_cols"] = list(ins_cols)\n'))

# S9 — cell models see 63 columns
SUBS.append((
    '    assert_no_recovered_in_features(base_feats, "cell models")\n'
    '    F_mode = build_frame_native(df.loc[tr_mode], base_feats, cat_native)\n'
    '    F_va = build_frame_native(df.loc[va], base_feats, cat_native)\n',
    '    assert_no_recovered_in_features(cell_feats, "cell models (63)")\n'
    '    F_mode = build_frame_native(df.loc[tr_mode], cell_feats, cat_native)\n'
    '    F_va = build_frame_native(df.loc[va], cell_feats, cat_native)\n'))

# S10 — cell fit through fit_cb (same fit + the model, for the gain share)
SUBS.append((
    '        pr, params, note = fit_catboost(C5, SEED, F_mode, yc_tr, F_va,\n'
    '                                        base_feats, cat_native, k)\n',
    '        pr, params, model_c = fit_cb(e27, C5, SEED, F_mode, yc_tr, F_va,\n'
    '                                     cell_feats, cat_native, k)\n'))

# S11 — honest argmin-at-cap flag (parent hardcoded False)
SUBS.append((
    '            "argmin_at_cap": False,\n',
    '            "argmin_at_cap": bool(a_min == k),\n'))

# S12 — per-cell ins gain share (the measurement the config asks for)
SUBS.append((
    '        cell_curves[c] = {str(g): float(f"{v:.10g}") for g, v in '
    'cv.items()}\n',
    '        cell_curves[c] = {str(g): float(f"{v:.10g}") for g, v in '
    'cv.items()}\n'
    '        imp_c = model_c.get_feature_importance(\n'
    '            type="PredictionValuesChange")\n'
    '        order_c = np.argsort(-imp_c)\n'
    '        tot_gain = float(imp_c.sum())\n'
    '        cell_info[c]["ins_gain_share"] = float(\n'
    '            f"{sum(imp_c[cell_feats.index(cc)] for cc in ins_cols) / max(tot_gain, 1e-12):.6g}")\n'
    '        cell_info[c]["importance_top15"] = {\n'
    '            cell_feats[i]: float(f"{imp_c[i]:.6g}") for i in '
    'order_c[:15]}\n'
    '        cell_info[c]["ins_importance_rank"] = {\n'
    '            cc: int(np.where(order_c == cell_feats.index(cc))[0][0] + 1)\n'
    '            for cc in ins_cols}\n'
    '        del model_c\n'))

# S13 — OOF fits see 63 columns
SUBS.append((
    '        assert_no_recovered_in_features(base_feats, f"B3 OOF {s}")\n'
    '        Ff = build_frame_native(df.loc[fit_rows], base_feats, '
    'cat_native)\n'
    '        Fp = build_frame_native(df.loc[pred_rows], base_feats, '
    'cat_native)\n',
    '        assert_no_recovered_in_features(cell_feats, f"B3 OOF {s} (63)")\n'
    '        Ff = build_frame_native(df.loc[fit_rows], cell_feats, '
    'cat_native)\n'
    '        Fp = build_frame_native(df.loc[pred_rows], cell_feats, '
    'cat_native)\n'))

SUBS.append((
    '            pr, _, _ = fit_catboost(C5, SEED, Ff, yc, Fp, base_feats,\n'
    '                                    cat_native, k)\n'
    '            stack[f"p_{c}"][pred_rows] = ref.sigmoid(pr(0, k))\n'
    '            del pr\n',
    '            pr, _, model_o = fit_cb(e27, C5, SEED, Ff, yc, Fp, '
    'cell_feats,\n'
    '                                    cat_native, k)\n'
    '            stack[f"p_{c}"][pred_rows] = ref.sigmoid(pr(0, k))\n'
    '            del pr, model_o\n'))

# S14 — the OOF npz records the cell input width
SUBS.append((
    '        train_csv_sha256=np.array(sha256_file(DATA_PATH)),\n',
    '        train_csv_sha256=np.array(sha256_file(DATA_PATH)),\n'
    '        cell_n_features=np.int64(len(cell_feats)),\n'))

# S15 — the stacker sees 67 columns (base 47 + p_* 4 + ins 16, exp_048 A1's
# exact ORDER — CatBoost is order-sensitive)
SUBS.append((
    '    feats = base_feats + stack_cols\n'
    '    assert_no_recovered_in_features(feats, "B3 stacker")\n'
    '    assert len(feats) == 51, len(feats)\n',
    '    feats = base_feats + stack_cols + ins_cols\n'
    '    assert_no_recovered_in_features(feats, "B3 stacker (67)")\n'
    '    assert len(feats) == 67, len(feats)\n'))

# S16 — stacker fit through fit_cb (model kept for importance)
SUBS.append((
    '    pr, params, note = fit_catboost(C5, SEED, Fs_tr, y_all[tr_full], '
    'Fs_va,\n'
    '                                    feats, cat_native, K)\n',
    '    pr, params, model_s = fit_cb(e27, C5, SEED, Fs_tr, y_all[tr_full], '
    'Fs_va,\n'
    '                                 feats, cat_native, K)\n'))

# S17 — the exp_030 reproduction gate is REPLACED by the exp_051 control gate
# (the parent gate cannot hold: the stacker design matrix changed by design)
SUBS.append((
    '    # ---- GATE PASS: the INHERITED accumulation on the INHERITED grid '
    '------- #\n'
    '    # This is exp_030\'s arithmetic, unmodified, so the number it '
    'produces is\n'
    '    # directly comparable to exp_030\'s recorded artefacts.\n'
    '    GK = int(args.gate_k)\n'
    '    assert GK <= K\n'
    '    g_grid = ref.curve_grid(GK)\n'
    '    assert g_grid[-1] == GK, (g_grid[-3:], GK)\n'
    '    marg_g, const_g = chunked_margin_curve(pr, n_va, g_grid)\n'
    '    cv_g = {g: float(np.mean((ref.sigmoid(marg_g[g]) - y_va) ** 2))\n'
    '            for g in g_grid}\n'
    '    p_gate = ref.sigmoid(marg_g[GK])\n'
    '    brier_gate = float(np.mean((p_gate - y_va) ** 2))\n'
    '    gate_path = os.path.join(out_dir, "pred",\n'
    '                             f"pred_C5B3_u{V}_seed{SEED}_at{GK}_GATE.npy")\n'
    '    np.save(gate_path, p_gate)\n'
    '    del marg_g\n'
    '    gc.collect()\n'
    '\n'
    '    ref030 = os.path.join(EXP030_DIR, "res",\n'
    '                          f"res_C5B3_u{V}_seed{SEED}_st2.json")\n'
    '    with open(ref030) as f:\n'
    '        r030 = json.load(f)\n'
    '    assert r030["exp_id"] == "exp_030" and r030["stage"] == 2\n'
    '    assert r030["unit"] == V and r030["seed"] == SEED\n'
    '    assert r030["stack_round"] == GK, (r030["stack_round"], GK)\n'
    '    assert r030["cell_iter_used"] == cell_iter_used, (\n'
    '        "the pinned cell iterations are NOT exp_030\'s",\n'
    '        r030["cell_iter_used"], cell_iter_used)\n'
    '    exp030_brier = float(r030["brier_full"])\n'
    '    abs_diff = abs(brier_gate - exp030_brier)\n'
    '    gate = {\n'
    '        "what": (f"the stacker trained to {K} rounds, its margins '
    'accumulated "\n'
    '                 f"by the INHERITED chunked_margin_curve over the '
    'INHERITED "\n'
    '                 f"curve_grid({GK}), evaluated at round {GK} — must '
    'reproduce "\n'
    '                 f"exp_030\'s stage-2 fold_2024 prediction for this '
    'seed"),\n'
    '        "expected_source": ref030,\n'
    '        "expected_brier": exp030_brier,\n'
    '        "observed_brier": brier_gate,\n'
    '        "abs_diff": abs_diff,\n'
    '        "tol": GATE_TOL,\n'
    '        "PASS": bool(abs_diff < GATE_TOL),\n'
    '        "expected_pred_sha256": r030["pred_sha256"],\n'
    '        "observed_pred_sha256": sha256_file(gate_path),\n'
    '        "bit_identical": bool(sha256_file(gate_path) == '
    'r030["pred_sha256"]),\n'
    '        "exp030_cell_iter_used": r030["cell_iter_used"],\n'
    '    }\n'
    '    res["reproduction_gate"] = gate\n'
    '    print(f"    GATE @{GK}: ours {brier_gate:.17g} vs exp_030 "\n'
    '          f"{exp030_brier:.17g}, abs_diff {abs_diff:.3e}, "\n'
    '          f"{\'PASS\' if gate[\'PASS\'] else \'FAIL\'}, "\n'
    '          f"bit_identical={gate[\'bit_identical\']}", flush=True)\n'
    '    assert gate["PASS"], ("reproduction gate FAILED", gate)\n',
    '    # ---- exp_051 CONTROL GATE: exp_048 A1 stored predictions ---------\n'
    '    r48_path = os.path.join(EXP048_DIR, "res",\n'
    '                            f"res_exp048_s1_seed{SEED}.json")\n'
    '    with open(r48_path) as f:\n'
    '        r48 = json.load(f)\n'
    '    assert r48["exp_id"] == "exp_048" and r48["seed"] == SEED\n'
    '    a1_1400 = r48["arms"]["A1"]["by_round"]["1400"]\n'
    '    ctrl_pred_path = a1_1400["pred_file"]\n'
    '    ctrl_sha_obs = sha256_file(ctrl_pred_path)\n'
    '    p_ctrl = np.load(ctrl_pred_path)\n'
    '    assert p_ctrl.shape == y_va.shape, (p_ctrl.shape, y_va.shape)\n'
    '    b_ctrl_re = float(np.mean((p_ctrl - y_va) ** 2))\n'
    '    gate = {\n'
    '        "what": ("the CONTROL is exp_048 arm A1 (cells 47 cols, stacker "\n'
    '                 "67 cols @1400).  Nothing is refitted: the stored "\n'
    '                 "prediction vector is sha256-compared against the "\n'
    '                 "digest exp_048 recorded, and its Brier is recomputed "\n'
    '                 "here against this run\'s own y_va."),\n'
    '        "control_res_json": r48_path,\n'
    '        "control_pred_file": ctrl_pred_path,\n'
    '        "recorded_pred_sha256": a1_1400["pred_sha256"],\n'
    '        "observed_pred_sha256": ctrl_sha_obs,\n'
    '        "sha_match": bool(ctrl_sha_obs == a1_1400["pred_sha256"]),\n'
    '        "recorded_brier": a1_1400["valid_all_brier"],\n'
    '        "recomputed_brier": b_ctrl_re,\n'
    '        "abs_diff_brier": abs(b_ctrl_re - a1_1400["valid_all_brier"]),\n'
    '        "control_brier_by_round": {\n'
    '            rr: r48["arms"]["A1"]["by_round"][rr]["valid_all_brier"]\n'
    '            for rr in ("655", "1000", "1400")},\n'
    '    }\n'
    '    gate["PASS"] = bool(gate["sha_match"]\n'
    '                        and gate["abs_diff_brier"] < 1e-12)\n'
    '    res["A0_control_gate_vs_exp048_A1"] = gate\n'
    '    print(f"    A0 GATE: sha_match {gate[\'sha_match\']}, abs_diff "\n'
    '          f"{gate[\'abs_diff_brier\']:.3e}, PASS {gate[\'PASS\']}",\n'
    '          flush=True)\n'
    '    assert gate["PASS"], ("A0 control gate FAILED", gate)\n'
    '\n'
    '    # ---- AH-OOF cross-check + OOF movement (config: 추가로_남길_것) ----\n'
    '    ah_oof_path = os.path.join(AH_OOF_DIR,\n'
    '                               f"stack_fold2024_seed{SEED}.npz")\n'
    '    ah_oof_sha = sha256_file(ah_oof_path)\n'
    '    assert ah_oof_sha == ah_seed["run"]["oof_sha256"], (\n'
    '        "AH OOF sha mismatch", ah_oof_sha)\n'
    '    z_ah = np.load(ah_oof_path, allow_pickle=False)\n'
    '    assert np.array_equal(z_ah["tr_full"].astype(bool), tr_full)\n'
    '    assert np.array_equal(z_ah["va"].astype(bool), va)\n'
    '    assert np.array_equal(z_ah["season"], season)\n'
    '    assert {c: int(v) for c, v in zip(FAIL_CELLS,\n'
    '                                      z_ah["cell_round"].tolist())} == \\\n'
    '        cell_iter_used, "AH OOF cell rounds differ from the pinned ones"\n'
    '    oof_delta = {}\n'
    '    for c in stack_cols:\n'
    '        new_v, old_v = stack[c], z_ah[c]\n'
    '        oof_delta[c] = {}\n'
    '        for seg_name, m_seg in (("train_rows", tr_full),\n'
    '                                ("valid_rows", va)):\n'
    '            nv, ov = new_v[m_seg], old_v[m_seg]\n'
    '            assert np.array_equal(np.isnan(nv), np.isnan(ov)), (\n'
    '                "NaN pattern moved", c, seg_name)\n'
    '            keep = ~np.isnan(nv)\n'
    '            d_ = nv[keep] - ov[keep]\n'
    '            ad = np.abs(d_)\n'
    '            q = np.quantile(ad, [0.5, 0.9, 0.99])\n'
    '            oof_delta[c][seg_name] = {\n'
    '                "n": int(d_.size),\n'
    '                "mean_abs": float(f"{ad.mean():.6g}"),\n'
    '                "p50_abs": float(f"{q[0]:.6g}"),\n'
    '                "p90_abs": float(f"{q[1]:.6g}"),\n'
    '                "p99_abs": float(f"{q[2]:.6g}"),\n'
    '                "max_abs": float(f"{ad.max():.6g}"),\n'
    '                "corr_new_old": float(\n'
    '                    f"{np.corrcoef(nv[keep], ov[keep])[0, 1]:.6g}"),\n'
    '            }\n'
    '    res["oof_delta_vs_AH"] = {\n'
    '        "ah_oof_file": ah_oof_path, "ah_oof_sha256": ah_oof_sha,\n'
    '        "per_column": oof_delta}\n'
    '    del z_ah\n'
    '\n'
    '    # ---- 47-column cell self-metrics, read from AH\'s res for compare --\n'
    '    ahres_path = os.path.join(AH_RES_DIR,\n'
    '                              f"res_C5B3_u{V}_seed{SEED}_localref.json")\n'
    '    with open(ahres_path) as f:\n'
    '        ah_res = json.load(f)\n'
    '    res["cells_vs_AH_47col"] = {\n'
    '        "source": ahres_path,\n'
    '        "per_cell": {\n'
    '            c: {"ah_47col_own": ah_res["cells"][c]["own_mode_at_used"],\n'
    '                "exp051_63col_own": cell_info[c]["own_mode_at_used"],\n'
    '                "ah_argmin_iter": ah_res["cells"][c]["argmin_iter"],\n'
    '                "exp051_argmin_iter": cell_info[c]["argmin_iter"],\n'
    '                "iter_used_pinned": cell_info[c]["iter_used"]}\n'
    '            for c in FAIL_CELLS}}\n'))

# S18 — no independent refits at r < cap (staged only, like exp_048)
SUBS.append((
    '        t0 = time.time()\n'
    '        pr_r, params_r, _note_r = fit_catboost(C5, SEED, Fs_tr, '
    'y_all[tr_full],\n'
    '                                               Fs_va, feats, cat_native, '
    'r)\n'
    '        fm = (time.time() - t0) / 60\n'
    '        p_r = ref.sigmoid(pr_r(0, r))\n'
    '        probs_A[r] = p_r\n'
    '        routeA[str(r)] = evaluate(p_r, "")\n'
    '        routeA[str(r)]["source"] = "an independent fit at exactly this '
    'many rounds"\n'
    '        fits[str(r)] = {"fit_min": round(fm, 3), "refitted": True,\n'
    '                        "iterations_param": params_r.get("iterations"),\n'
    '                        "random_seed_param": '
    'params_r.get("random_seed")}\n'
    '        print(f"    [fit    @{r}] {fm:.2f} min  valid_all brier "\n'
    '              f"{routeA[str(r)][\'valid_all_brier\']:.12f}  score "\n'
    '              f"{routeA[str(r)][\'valid_all_score_converted\']}", '
    'flush=True)\n'
    '        del pr_r\n'
    '        gc.collect()\n',
    '        # exp_051: no refit at r < cap.  exp_047 measured the staged\n'
    '        # prefix bit-identical to an independent fit, and exp_048\'s\n'
    '        # 655/1000 numbers are staged, so this stays comparable.\n'
    '        routeA[str(r)] = dict(routeB[str(r)])\n'
    '        routeA[str(r)]["source"] = ("staged from the cap fit -- no '
    'refit; "\n'
    '                                    "prefix identity measured by '
    'exp_047")\n'
    '        probs_A[r] = probs_B[r]\n'
    '        fits[str(r)] = {"fit_min": 0.0, "refitted": False}\n'))

# S19 — drop the gate_k fields (that gate no longer exists)
SUBS.append((
    '        "fits": fits,\n'
    '        "gate_k": GK,\n'
    '        "brier_at_gate_k": brier_gate,\n'
    '    }\n',
    '        "fits": fits,\n'
    '    }\n'))

# S20 — stacker importance instead of the family note; drop const_g
SUBS.append((
    '    res["stacker_family_note"] = note\n'
    '    res["margin_const_gate_path"] = float(const_g)\n',
    '    imp_s = model_s.get_feature_importance('
    'type="PredictionValuesChange")\n'
    '    order_s = np.argsort(-imp_s)\n'
    '    res["stacker_importance_PredictionValuesChange"] = {\n'
    '        feats[i]: float(f"{imp_s[i]:.6g}") for i in order_s}\n'
    '    res["stacker_importance_rank"] = {feats[i]: int(k2 + 1)\n'
    '                                      for k2, i in enumerate(order_s)}\n'
    '    del model_s\n'))

# S21 — output stem
SUBS.append((
    '    stem = f"C5B3_u{V}_seed{SEED}_localref"\n',
    '    stem = f"exp051_u{V}_seed{SEED}"\n'))

# S22 — the final RESULT print referenced the removed gate key.
# (added AFTER the first run: the three seeds crashed HERE, on the line after
# json.dump(res) — every artefact had already been persisted, so the runs were
# NOT relaunched.  The version that ran is sha
# 91e1454546d3d72b61547a523afcd349f38a961d8b3d74d28b0de86498e548b7, recorded
# in each res JSON under sha256.self_stack_argmin.py; this fix changes only
# the post-persist summary line.)
SUBS.append((
    '        "gate_pass": gate["PASS"], "gate_bit_identical": '
    'gate["bit_identical"],\n',
    '        "gate_pass": gate["PASS"], "gate_sha_match": '
    'gate["sha_match"],\n'))


def main():
    parent = sys.argv[1]
    out = sys.argv[2]
    got = sha256_file(parent)
    assert got == PARENT_SHA, ("parent local_ref_stack.py drifted", got)
    with open(parent) as f:
        src = f.read()
    orig = src
    for i, (old, new) in enumerate(SUBS):
        n = src.count(old)
        assert n == 1, (f"substitution S{i}: anchor occurs {n} times, "
                        f"need exactly 1", old[:90])
        src = src.replace(old, new)
    with open(out, "w") as f:
        f.write(src)
    diff = difflib.unified_diff(
        orig.splitlines(keepends=True), src.splitlines(keepends=True),
        fromfile="local_ref_stack.py (parent, sha 1993f63e...)",
        tofile="train51.py")
    diff_path = os.path.join(os.path.dirname(out), "train51.diff")
    with open(diff_path, "w") as f:
        f.writelines(diff)
    import py_compile
    py_compile.compile(out, doraise=True)
    print(f"OK: {len(SUBS)} substitutions, wrote {out}")
    print(f"    child sha256 {sha256_file(out)}")
    print(f"    diff saved   {diff_path}")


if __name__ == "__main__":
    main()
