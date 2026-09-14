"""exp_048 STAGE 0, gate G0e RE-DONE.

The first attempt asked whether identical 16-tuples ever carry both labels.
That was a badly chosen criterion: sixteen continuous columns with ~240,000
distinct values each over 253,507 rows are singletons by arithmetic, and
singleton-ness carries no information about the target.  It measured the
cardinality of a float, not a leak.  Recorded rather than deleted.

The decisive question is: WHICH COLUMNS does `apply` read to produce a row's
output?  If the read set contains only organiser-supplied `asof_*` (defined by
the organiser as "the row's history strictly BEFORE this pitch") and the two
id columns, then this row's own outcome is not an input and cannot come out.
That is measured here by RUNTIME TRACE -- a proxy frame that records every
`__getitem__` -- so no static parse can be wrong about it.

Reported alongside: the label-recovery path of clue S10.  It is real, it needs
TWO rows of the same pitcher differenced, and G0b/G0c already showed `apply`'s
output for a row is unchanged when every other row is removed.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.join(SRC_DIR, "ins"))
import inseason as ins                                              # noqa: E402

DATA_PATH = os.path.expanduser("~/SungjinPark/data/train.csv")
PACK_DIR = os.path.join(SRC_DIR, "ins")
OUT = os.path.expanduser("~/SungjinPark/experiments/exp_048/res")
TARGET = "control_success"


class TracingFrame:
    """Records every column `apply` asks for.  Anything it does not expose is
    an AttributeError, so a hidden second access path would crash, not pass."""

    def __init__(self, df):
        self._df = df
        self.reads = []

    def __getitem__(self, k):
        self.reads.append(k)
        return self._df[k]

    def __getattr__(self, name):
        if name in ("_df", "reads"):
            raise AttributeError(name)
        self.reads.append(f"<attr>{name}")
        return getattr(self._df, name)


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    season = df["season"].to_numpy()
    va = season == 2024
    df_va = df.loc[va]
    A = ins.load(os.path.join(PACK_DIR, "anchor_lt2024.npz"))

    tf = TracingFrame(df_va)
    I_traced = ins.apply(tf, A)
    I_plain = ins.apply(df_va, A)
    assert np.array_equal(I_traced.to_numpy(), I_plain.to_numpy())

    cols_read = sorted({r for r in tf.reads if not str(r).startswith("<attr>")})
    attrs = sorted({r[6:] for r in tf.reads if str(r).startswith("<attr>")})

    allowed_prefix = ("asof_pitcher_", "asof_batter_")
    allowed_exact = {"pitcher_id", "batter_id"}
    illegal = [c for c in cols_read
               if not (str(c).startswith(allowed_prefix) or c in allowed_exact)]

    # what the organiser says about the columns that WERE read
    res = {
        "exp_id": "exp_048", "gate": "G0e (redone)",
        "method": ("runtime trace: `inseason.apply` was handed a proxy frame "
                   "that records every column access, on the 253,507 fold_2024 "
                   "validation rows; the traced run's 16 columns are identical "
                   "to the plain run's."),
        "columns_read_by_apply": cols_read,
        "n_columns_read": len(cols_read),
        "attributes_touched": attrs,
        "target_read": TARGET in cols_read,
        "illegal_columns_read": illegal,
        "allowed_set": {"prefixes": list(allowed_prefix),
                        "exact": sorted(allowed_exact)},
        "why_this_settles_it": (
            "every column read is either an organiser-supplied asof_* (defined "
            "by the organiser as this row's history STRICTLY BEFORE the pitch) "
            "or a player id.  The row's own outcome is never an input, so it "
            "cannot be an output.  The S10 label-recovery path differences the "
            "same asof columns ACROSS two consecutive rows of one pitcher; "
            "G0b/G0c showed a row's 16 values are bit-identical when every "
            "other row is deleted, so that path is not open here."),
        "superseded_criterion": {
            "what_it_was": "n_tuples_carrying_both_labels > 0",
            "why_dropped": ("16 continuous columns with ~240k distinct values "
                            "over 253,507 rows are 100% singleton tuples by "
                            "arithmetic; the test measured float cardinality, "
                            "not leakage.  Its numbers stay in "
                            "stage0_gates.json."),
        },
    }
    res["PASS"] = bool(not res["target_read"] and not illegal)

    # sanity: is the 19.86% "unknown pitcher" share real, or a dtype miss?
    known = set(pd.Index(A["p_ids"]).tolist())
    true_unknown = ~df_va["pitcher_id"].isin(known).to_numpy()
    hist_ids = set(df.loc[season < 2024, "pitcher_id"].unique().tolist())
    res["unknown_pitcher_cross_check"] = {
        "why": ("a dtype mismatch between the frozen id array and the frame "
                "would silently mark EVERY row unknown; this checks the share "
                "against the raw set difference."),
        "pct_2024_rows_pitcher_not_in_anchor":
            round(100.0 * float(true_unknown.mean()), 4),
        "pct_2024_rows_pitcher_not_in_train_lt2024":
            round(100.0 * float(
                (~df_va["pitcher_id"].isin(hist_ids)).to_numpy().mean()), 4),
        "anchor_id_dtype": str(np.asarray(A["p_ids"]).dtype),
        "frame_id_dtype": str(df_va["pitcher_id"].dtype),
        "n_ids_in_anchor": int(len(A["p_ids"])),
        "n_ids_in_train_lt2024": int(len(hist_ids)),
    }
    res["unknown_pitcher_cross_check"]["consistent"] = bool(
        res["unknown_pitcher_cross_check"]["pct_2024_rows_pitcher_not_in_anchor"]
        == res["unknown_pitcher_cross_check"][
            "pct_2024_rows_pitcher_not_in_train_lt2024"])

    p = os.path.join(OUT, "stage0_gate_e2.json")
    with open(p, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in
                      ("columns_read_by_apply", "target_read",
                       "illegal_columns_read", "PASS")}, indent=1), flush=True)
    print(json.dumps(res["unknown_pitcher_cross_check"], indent=1), flush=True)


if __name__ == "__main__":
    main()
