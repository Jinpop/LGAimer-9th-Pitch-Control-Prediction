"""exp_097 stageA — 출하 패키지 .py 를 AST 로 계측한다.

여러 행에 걸친 누산의 *입력 객체*를 `_pt.p("파일:행:이름", <객체>)` 로 감싼다.
`_pt.p` 는 off 모드에서 인자를 그대로 돌려주므로 항등이고, 그 항등성은
run 스크립트가 부모 submission.csv sha 로 기계 검증한다.
"""
import ast
import os
import shutil
import sys

METHODS = {
    "mean", "std", "var", "sum", "cumsum", "cumprod", "prod", "median",
    "quantile", "dot", "matmul", "einsum", "average", "corrcoef", "cov",
    "percentile", "nansum", "nanmean", "nanstd", "nanvar",
    "rolling", "expanding", "groupby", "transform", "agg", "aggregate",
    "apply", "idxmax", "idxmin", "value_counts", "bincount", "reduce",
    "min", "max", "argmin", "argmax", "any", "all", "count", "nunique",
}
NPFUNCS = {
    "mean", "std", "var", "sum", "nanmean", "nansum", "nanstd", "nanvar",
    "cumsum", "einsum", "dot", "matmul", "average", "percentile", "quantile",
    "median", "bincount", "inner", "tensordot", "nanargmax", "nanargmin",
    "min", "max", "prod", "corrcoef", "cov",
}
# 승격 대상이 아니라 '기록만' 하는 부류 (양자화 지점)
ASTYPE_NOTE = set()


class T(ast.NodeTransformer):
    def __init__(self, fname):
        self.f = fname
        self.sites = []

    def _wrap(self, node, tag):
        self.sites.append(tag)
        return ast.Call(
            func=ast.Attribute(value=ast.Name(id="_pt", ctx=ast.Load()),
                               attr="p", ctx=ast.Load()),
            args=[ast.Constant(value=tag), node], keywords=[])

    def visit_Call(self, node):
        self.generic_visit(node)
        fn = node.func
        # a.b.mean(...) / s.groupby(...)  -> 수신자를 감싼다
        if isinstance(fn, ast.Attribute) and fn.attr in METHODS:
            # np.xxx(...) 형태는 아래 분기가 처리하므로 여기서 제외
            recv = fn.value
            is_np_mod = (isinstance(recv, ast.Name) and recv.id in ("np", "numpy"))
            if not is_np_mod:
                tag = "%s:%d:.%s" % (self.f, node.lineno, fn.attr)
                fn.value = self._wrap(recv, tag)
                return node
        # np.mean(a, ...) -> 첫 인자를 감싼다
        if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) \
                and fn.value.id in ("np", "numpy") and fn.attr in NPFUNCS \
                and node.args:
            tag = "%s:%d:np.%s" % (self.f, node.lineno, fn.attr)
            node.args[0] = self._wrap(node.args[0], tag)
            return node
        # .astype(np.float32) / .astype('float32') 기록 (승격 안 함)
        if isinstance(fn, ast.Attribute) and fn.attr == "astype" and node.args:
            a = node.args[0]
            lit = None
            if isinstance(a, ast.Constant):
                lit = str(a.value)
            elif isinstance(a, ast.Attribute):
                lit = a.attr
            if lit in ("float32", "float16"):
                ASTYPE_NOTE.add("%s:%d:.astype(%s)" % (self.f, node.lineno, lit))
        return node

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.MatMult):
            tag = "%s:%d:@L" % (self.f, node.lineno)
            node.left = self._wrap(node.left, tag)
            tag = "%s:%d:@R" % (self.f, node.lineno)
            node.right = self._wrap(node.right, tag)
        return node


def transform(src_path, rel):
    src = open(src_path, encoding="utf-8").read()
    tree = ast.parse(src)
    t = T(rel)
    tree = t.visit(tree)
    ast.fix_missing_locations(tree)
    out = ast.unparse(tree)
    # `import _pt` 를 맨 앞에 넣는다. sys.path 는 건드리지 않는다
    # (script.py:20 이 경고하는 해석 순서 위험을 만들지 않기 위해).
    out = "import _pt\n" + out + "\n"
    return out, t.sites


def main(pkg_in, pkg_out, probe):
    if os.path.exists(pkg_out):
        raise SystemExit("출력 경로가 이미 있다: " + pkg_out)
    shutil.copytree(pkg_in, pkg_out)
    sites = []
    pyfiles = []
    for root, _, files in os.walk(pkg_out):
        for fn in sorted(files):
            if fn.endswith(".py"):
                pyfiles.append(os.path.join(root, fn))
    dirs = set()
    for pth in sorted(pyfiles):
        rel = os.path.relpath(pth, pkg_out)
        code, s = transform(pth, rel)
        open(pth, "w", encoding="utf-8").write(code)
        sites += s
        dirs.add(os.path.dirname(pth))
    for d in sorted(dirs):
        shutil.copy(probe, os.path.join(d, "_pt.py"))
    import json
    json.dump({"n_files": len(pyfiles), "n_sites": len(sites),
               "sites": sorted(set(sites)),
               "astype_f32": sorted(ASTYPE_NOTE)},
              open(os.path.join(pkg_out, "_pt_sites.json"), "w"), indent=1)
    print("files=%d sites=%d uniq=%d astype_f32=%d"
          % (len(pyfiles), len(sites), len(set(sites)), len(ASTYPE_NOTE)))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
