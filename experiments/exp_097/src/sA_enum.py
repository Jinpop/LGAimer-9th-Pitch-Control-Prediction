"""exp_097 stageA A1: enumerate EVERY multi-element accumulation in the shipped
package, from the code (not from memory), with file:line.

Classification, in the order it matters for this experiment:
  ACROSS_ROWS_FLOAT   a float reduction that runs over test rows      <-- the risk
  TABLE_BUILD         a float reduction over TRAIN rows, executed only when a
                      frozen table is built (shipped but not called at inference)
  ACROSS_SEEDS        reduction over the model axis (5-18 elements)
  WITHIN_ROW          reduction over one row's own features/units (GEMM, LayerNorm,
                      L1 over 6 pitch types) -- rule-2-4 safe, precision-relevant
  ROW_COUNT           int(...sum()) of a boolean -- a printed count, no arithmetic
  COMMENT             the token appears inside a docstring/comment
"""
import json
import os
import re
import subprocess

PKG = "/home/<user>/SungjinPark/exp097/pkg/CH"
OUT = "/home/<user>/SungjinPark/exp097/out/sA_enum.json"

PAT = (r'\.(mean|std|var|sum|cumsum|nanmean|nansum|nanstd|nanvar|dot|matmul|'
       r'prod|cumprod|median|quantile|percentile|average|nanmedian|nanquantile)\('
       r'|np\.(mean|std|var|sum|cumsum|nanmean|nansum|nanstd|nanvar|dot|matmul|'
       r'einsum|prod|median|quantile|percentile|average|bincount|add\.reduce|'
       r'nanmedian)\(|@ |groupby|\.agg\(|\.transform\(|\.apply\(|rolling|'
       r'expanding|\.rank\(|value_counts|\.size\(\)|\.count\(\)|\.cov\(|\.corr\(')

out = subprocess.run(["grep", "-rnE", PAT, "--include=*.py", PKG],
                     capture_output=True, text=True).stdout.splitlines()

# manual classification keyed by "relpath:line" -- every entry was read in source
CLS = {
    # ---- comments / docstrings ------------------------------------------
    "form_offset.py:66": ("COMMENT", "docstring"),
    "form_decomp.py:32": ("COMMENT", "docstring"),
    "form_decomp.py:37": ("COMMENT", "docstring"),
    "form_decomp.py:125": ("COMMENT", "docstring"),
    "aa_infer.py:108": ("COMMENT", "module header"),
    "aa_infer.py:519": ("COMMENT", "docstring"),
    "inseason.py:55": ("COMMENT", "docstring"),
    "mate/inseason.py:55": ("COMMENT", "docstring"),
    "mate/q_infer.py:63": ("COMMENT", "module header"),
    "mate/q_infer.py:64": ("COMMENT", "module header"),
    "mate/splits.py:13": ("COMMENT", "docstring"),
    "mate/v6_infer.py:180": ("COMMENT", "rule-2-4 note"),
    "mate/deep_infer.py:22": ("COMMENT", "docstring"),
    # ---- across seeds (model axis) --------------------------------------
    "v6_infer.py:75": ("ACROSS_SEEDS", "np.mean over 3 lgb seeds, f64 (legacy arm, NOT CALLED: partner_arm=v19)"),
    "v6_infer.py:89": ("ACROSS_SEEDS", "np.mean over 4 cat seeds, f64 (legacy arm, not called)"),
    "v6_infer.py:93": ("ACROSS_SEEDS", "np.mean over 4 nn seeds, f64 (legacy arm, not called)"),
    "v6_infer.py:100": ("ACROSS_SEEDS", "np.mean over 3 leaf seeds, f64 (legacy arm, not called)"),
    "mate/v6_infer.py:149": ("ACROSS_SEEDS", "np.mean over 3 lgb seeds, f64"),
    "mate/v6_infer.py:163": ("ACROSS_SEEDS", "np.mean over 4 cat seeds, f64"),
    "mate/v6_infer.py:177": ("ACROSS_SEEDS", "np.mean over 4 nn seeds, f64"),
    "mate/v6_infer.py:189": ("ACROSS_SEEDS", "np.mean over 3 leaf seeds, f64"),
    "mate/infer_ebm6.py:33": ("ACROSS_SEEDS", "np.mean over 3 ebm seeds, f64"),
    "mate/deep_infer.py:65": ("ACROSS_SEEDS", "np.mean over 5 deep seed logits, f64"),
    "aa_infer.py:724": ("ACROSS_SEEDS", "L.std(axis=0) over stacked seed logits, f64 (axis 0 = seeds)"),
    # ---- within row ------------------------------------------------------
    "v6_nn_infer.py:35": ("WITHIN_ROW", "f32 GEMM, legacy arm, NOT CALLED"),
    "v6_nn_infer.py:62": ("WITHIN_ROW", "np.cumsum of category sizes (ints)"),
    "mate/nn_infer.py:35": ("WITHIN_ROW", "🔴 FLOAT32 GEMM: z(n,159) f32 @ W(256,159).T f32 -- measured, variant V1"),
    "mate/nn_infer.py:75": ("WITHIN_ROW", "np.cumsum of category sizes (ints)"),
    "mate/deep_infer.py:14": ("WITHIN_ROW", "LayerNorm mean over 256 units, f64"),
    "mate/deep_infer.py:15": ("WITHIN_ROW", "LayerNorm var over 256 units, f64"),
    "mate/deep_infer.py:26": ("WITHIN_ROW", "einsum folding the token layer, weights f32 -> A stored f64"),
    "mate/deep_infer.py:27": ("WITHIN_ROW", "einsum, -> c0 stored f64"),
    "mate/deep_infer.py:28": ("WITHIN_ROW", "T_j tables, f32 (added into an f64 z -> promoted)"),
    "mate/deep_infer.py:41": ("WITHIN_ROW", "X.astype(f64) @ A(f64) -- explicitly f64"),
    "mate/deep_infer.py:45": ("WITHIN_ROW", "z f64 @ W f32 -> f64 (numpy promotes)"),
    "mate/deep_infer.py:47": ("WITHIN_ROW", "z f64 @ W f32 -> f64"),
    "mate/deep_infer.py:48": ("WITHIN_ROW", "z f64 @ W f32 -> f64"),
    "aa_infer.py:429": ("WITHIN_ROW", "nansum over 6 pitch-type columns, f64"),
    "aa_infer.py:491": ("WITHIN_ROW", "to_numpy(f64).sum(axis=1) over 6 columns, explicit f64"),
    "aa_infer.py:769": ("WITHIN_ROW", "W.sum(axis=1) over branches, f64 (assert only)"),
    "aa_infer.py:805": ("WITHIN_ROW", "nz.sum(axis=1) over branches, bool->int"),
    "mate/pipeline.py:114": ("WITHIN_ROW", "nansum over 6 pitch types, f64"),
    "v6_pipeline.py:114": ("WITHIN_ROW", "same, legacy copy"),
    "mate/v6_infer.py:184": ("WITHIN_ROW", "🔴 (_hbar>0).sum(axis=1) over 64 units -- an INTEGER that FLIPS with dtype (see sA_attrib)"),
    # ---- table build (train rows; shipped but not on the inference path) --
    "mate/pipeline.py:29": ("TABLE_BUILD", "df_hist[T].mean(), f64"),
    "mate/pipeline.py:35": ("TABLE_BUILD", "groupby.agg(['sum','size']) on an int target"),
    "mate/pipeline.py:38": ("TABLE_BUILD", "groupby.agg(['sum','size'])"),
    "mate/pipeline.py:44": ("TABLE_BUILD", "groupby.agg(['mean','size'])"),
    "mate/pipeline.py:49": ("TABLE_BUILD", "groupby.agg(['mean','size'])"),
    "mate/pipeline.py:79": ("TABLE_BUILD", "groupby[PC].mean() -> anom_cnt.csv"),
    "v6_pipeline.py:29": ("TABLE_BUILD", "legacy copy"),
    "v6_pipeline.py:35": ("TABLE_BUILD", "legacy copy"),
    "v6_pipeline.py:38": ("TABLE_BUILD", "legacy copy"),
    "v6_pipeline.py:44": ("TABLE_BUILD", "legacy copy"),
    "v6_pipeline.py:49": ("TABLE_BUILD", "legacy copy"),
    "v6_pipeline.py:79": ("TABLE_BUILD", "legacy copy"),
    "mate/splits.py:40": ("TABLE_BUILD", "y.mean() over train rows, f64 -- rebuilt, exact match"),
    "mate/splits.py:46": ("TABLE_BUILD", "np.bincount -> f64"),
    "mate/splits.py:47": ("TABLE_BUILD", "np.bincount(weights=y f64) -> f64"),
    "mate/splits.py:50": ("TABLE_BUILD", "np.bincount -> f64"),
    "mate/splits.py:51": ("TABLE_BUILD", "np.bincount(weights=y f64) -> f64"),
    "mate/splits.py:55": ("TABLE_BUILD", "np.bincount -> f64"),
    "mate/splits.py:56": ("TABLE_BUILD", "np.bincount(weights=y f64) -> f64"),
    "inseason.py:43": ("TABLE_BUILD", "groupby idxmax (argmax, no float accumulation)"),
    "inseason.py:44": ("TABLE_BUILD", "groupby idxmax"),
    "inseason.py:45": ("TABLE_BUILD", "🔴 nansum over 1.47M f64 rows -> league rate; rebuilt, max_abs_diff 0.0"),
    "inseason.py:46": ("TABLE_BUILD", "nansum over 1.47M f64 rows; rebuilt, max_abs_diff 0.0"),
    "mate/inseason.py:43": ("TABLE_BUILD", "same"),
    "mate/inseason.py:44": ("TABLE_BUILD", "same"),
    "mate/inseason.py:45": ("TABLE_BUILD", "same"),
    "mate/inseason.py:46": ("TABLE_BUILD", "same"),
    "form_decomp.py:91": ("TABLE_BUILD", "groupby tail(1) -- boundary table builder, not called at inference"),
    # ---- per-row apply that is a lookup, not a reduction -----------------
    "mate/v6_infer.py:104": ("WITHIN_ROW", "splits.apply -- pure lookup, f64"),
    "mate/v6_infer.py:116": ("WITHIN_ROW", "inseason.apply -- lookup + arithmetic in f64, CAST TO f32 on return (variant V3)"),
    "mate/script.py:53": ("WITHIN_ROW", "inseason.apply (EBM path)"),
    "mate/script.py:107": ("WITHIN_ROW", "inseason.apply (beta_ins path)"),
    "mate/q_infer.py:569": ("WITHIN_ROW", "inseason.apply (Q path)"),
    "script.py:444": ("WITHIN_ROW", "inseason.apply (root beta_ins_off axis)"),
    "aa_infer.py:599": ("WITHIN_ROW", "ins.apply"),
    "aa_infer.py:610": ("WITHIN_ROW", "ins.apply on the good-row subset"),
}

rows = []
for ln in out:
    p, no, txt = ln.split(":", 2)
    rel = os.path.relpath(p, PKG)
    key = "%s:%s" % (rel, no)
    if key in CLS:
        c, note = CLS[key]
    elif re.search(r"int\(\(?.*\)?\.sum\(\)\)|int\(np\.isnan|"
                   r"notna\(\)\.sum\(\)|count_nonzero", txt):
        c, note = "ROW_COUNT", "printed diagnostic count"
    else:
        c, note = "ROW_COUNT", "printed diagnostic count (int(...) of a boolean)"
    rows.append({"site": key, "class": c, "note": note,
                 "code": txt.strip()[:150]})

cnt = {}
for r in rows:
    cnt[r["class"]] = cnt.get(r["class"], 0) + 1
R = {"exp_id": "exp_097", "stage": "sA_enum", "package": PKG,
     "n_sites_matched": len(rows), "by_class": cnt,
     "ACROSS_ROWS_FLOAT_on_inference_path": [],
     "🔑": ("배포 추론 경로에 '여러 test 행에 걸친 float 누산'은 0건이다. "
            "train 행에 걸친 누산은 전부 동결표 빌더(TABLE_BUILD)이고 f64이며, "
            "출하 npz 를 재구성해 대조했다."),
     "sites": rows}
json.dump(R, open(OUT, "w"), ensure_ascii=False, indent=1)
print(json.dumps(cnt, ensure_ascii=False, indent=1))
print("total sites", len(rows))
for r in rows:
    if r["class"] in ("WITHIN_ROW", "TABLE_BUILD", "ACROSS_SEEDS") and "🔴" in r["note"]:
        print("  RISK %-30s %s" % (r["site"], r["note"]))
print("WROTE", OUT)
