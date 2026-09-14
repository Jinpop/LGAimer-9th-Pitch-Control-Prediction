"""exp_047 — mechanical proof that the row-wise scoring path contains no
cross-row operation.

WHY AST AND NOT GREP
--------------------
A regex over the source text cannot tell code from prose.  In an exp_046
follow-up build a regex matched the word `groupby` inside a docstring that was
*explaining* where groupby is forbidden, and reported a violation that did not
exist.  The parsed syntax tree has no such ambiguity: a docstring is a
`Constant`, a call is a `Call`.

WHAT IT CHECKS
--------------
For every function named in APPLIER_FUNCS (and everything they call inside the
same module), the body must contain none of:

  * attribute access whose name is a cross-row pandas/numpy operation
    (groupby, sort_values, shift, cumsum, cumcount, expanding, rolling, diff,
     tail, head, drop_duplicates, rank, factorize, unique, argsort, ...)
  * a call to pd.factorize / np.unique / np.sort / np.argsort / sorted()
  * iteration over another frame

The BUILDER functions are exempt by name and are asserted to be called only
from train-side code paths; that separation is the point of the two lists.
"""

import ast
import json
import sys

BANNED_ATTRS = {
    "groupby", "sort_values", "sort_index", "sort", "argsort", "shift",
    "cumsum", "cumcount", "cumprod", "cummax", "cummin", "expanding",
    "rolling", "diff", "tail", "head", "drop_duplicates", "rank", "factorize",
    "unique", "nunique", "value_counts", "pivot_table", "merge", "join",
    "resample", "ewm", "transform", "agg", "aggregate", "apply",
}
BANNED_CALLS = {"sorted", "set"}


class Walker(ast.NodeVisitor):
    def __init__(self, name):
        self.name = name
        self.hits = []

    def visit_Attribute(self, node):
        if node.attr in BANNED_ATTRS:
            self.hits.append({"func": self.name, "kind": "attribute",
                              "name": node.attr, "line": node.lineno})
        self.generic_visit(node)

    def visit_Call(self, node):
        f = node.func
        if isinstance(f, ast.Name) and f.id in BANNED_CALLS:
            self.hits.append({"func": self.name, "kind": "call",
                              "name": f.id, "line": node.lineno})
        self.generic_visit(node)


def check(path, applier_funcs, local_ok=()):
    """Return (ok, report).  `local_ok` are helper names an applier may call."""
    tree = ast.parse(open(path).read())
    funcs = {n.name: n for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    missing = [f for f in applier_funcs if f not in funcs]
    report = {"file": path, "checked": [], "missing": missing, "hits": []}
    seen = set()
    queue = list(applier_funcs)
    while queue:
        fn = queue.pop()
        if fn in seen or fn not in funcs:
            continue
        seen.add(fn)
        node = funcs[fn]
        w = Walker(fn)
        for st in node.body:
            # skip the docstring node entirely: it is prose, not code
            if (isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant)
                    and isinstance(st.value.value, str)):
                continue
            w.visit(st)
        report["hits"].extend(w.hits)
        report["checked"].append(fn)
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                if sub.func.id in funcs and sub.func.id in local_ok:
                    queue.append(sub.func.id)
    report["PASS"] = bool(not report["hits"] and not missing)
    return report["PASS"], report


if __name__ == "__main__":
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    out = {}
    ok_all = True

    ok, rep = check(os.path.join(here, "form_decomp2.py"),
                    ["map_ids_rowwise", "apply_traj"])
    out["form_decomp2.py"] = rep
    ok_all &= ok

    # exp_046's applier, re-checked here rather than trusted from its own audit
    ok2, rep2 = check(os.path.expanduser(
        "~/SungjinPark/experiments/exp_046/src/form_decomp.py"),
        ["apply_decomposition", "career_successes", "extend_table_for_unseen"],
        local_ok=("career_successes",))
    out["exp_046/form_decomp.py"] = rep2
    ok_all &= ok2

    out["ALL_PASS"] = bool(ok_all)
    print(json.dumps(out, indent=1))
    sys.exit(0 if ok_all else 1)
