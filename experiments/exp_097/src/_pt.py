"""exp_097 stageA — 배포 경로 부동소수 정밀도 계측/승격 프로브.

MODE (env PT_MODE):
  off      : 완전 항등. AST 변환이 결과를 바꾸지 않음을 증명하는 대조군.
  trace    : 각 누산 지점의 입력 dtype·원소 수를 기록만 하고 객체는 그대로 반환.
  promote  : trace + float32(및 float16) 입력을 float64 로 승격해서 누산시킨다.
             ⚠️ 원본 배열은 안 바꾼다 — 누산에 들어가는 임시 사본만 바꾼다.

PT_ONLY  : promote 모드에서 이 태그(쉼표 구분 접두)만 승격한다. 지점별 귀속용.
PT_SKIP  : promote 모드에서 이 태그(쉼표 구분 접두)는 승격하지 않는다.
PT_OUT   : 요약 json 출력 디렉터리.
"""
import atexit
import json
import os
import sys

import numpy as np

MODE = os.environ.get("PT_MODE", "off")
_OUT = os.environ.get("PT_OUT", "")
_ONLY = [s for s in os.environ.get("PT_ONLY", "").split(",") if s]
_SKIP = [s for s in os.environ.get("PT_SKIP", "").split(",") if s]

LEDGER = {}
PROMOTED = {}

_F32 = (np.float32, np.float16)


def _kind(o):
    t = type(o)
    return t.__module__ + "." + t.__name__


def _describe(o):
    """실행 중 실제 dtype·원소 수. 정적 추정 금지 (config A2)."""
    try:
        import pandas as pd
    except Exception:
        pd = None
    if isinstance(o, np.ndarray):
        return str(o.dtype), int(o.size), "ndarray"
    if pd is not None and isinstance(o, pd.Series):
        return str(o.dtype), int(o.size), "Series"
    if pd is not None and isinstance(o, pd.DataFrame):
        dts = sorted({str(d) for d in o.dtypes})
        return "|".join(dts), int(o.size), "DataFrame"
    if isinstance(o, (list, tuple)):
        inner = set()
        n = 0
        for x in o:
            if isinstance(x, np.ndarray):
                inner.add(str(x.dtype)); n += x.size
            else:
                inner.add(_kind(x)); n += 1
        return "|".join(sorted(inner)), int(n), "list"
    if isinstance(o, (int, float, np.generic)):
        return _kind(o), 1, "scalar"
    return _kind(o), 0, "other"


def _record(tag, o):
    dt, n, kind = _describe(o)
    e = LEDGER.get(tag)
    if e is None:
        e = LEDGER[tag] = {"calls": 0, "kinds": {}, "dtypes": {}, "max_n": 0,
                           "sum_n": 0}
    e["calls"] += 1
    e["kinds"][kind] = e["kinds"].get(kind, 0) + 1
    e["dtypes"][dt] = e["dtypes"].get(dt, 0) + 1
    if n > e["max_n"]:
        e["max_n"] = n
    e["sum_n"] += n
    return dt, n


def _is_f32_dtype(d):
    try:
        return d.kind == "f" and d.itemsize < 8
    except Exception:
        return False


def _promote(tag, o):
    try:
        import pandas as pd
    except Exception:
        pd = None
    if isinstance(o, np.ndarray):
        if _is_f32_dtype(o.dtype):
            PROMOTED[tag] = PROMOTED.get(tag, 0) + 1
            return o.astype(np.float64)
        return o
    if pd is not None and isinstance(o, pd.Series):
        if _is_f32_dtype(o.dtype):
            PROMOTED[tag] = PROMOTED.get(tag, 0) + 1
            return o.astype(np.float64)
        return o
    if pd is not None and isinstance(o, pd.DataFrame):
        cols = [c for c in o.columns if _is_f32_dtype(o.dtypes[c])]
        if cols:
            PROMOTED[tag] = PROMOTED.get(tag, 0) + 1
            o2 = o.copy()
            for c in cols:
                o2[c] = o2[c].astype(np.float64)
            return o2
        return o
    if isinstance(o, list) and o and all(isinstance(x, np.ndarray) for x in o):
        if any(_is_f32_dtype(x.dtype) for x in o):
            PROMOTED[tag] = PROMOTED.get(tag, 0) + 1
            return [x.astype(np.float64) if _is_f32_dtype(x.dtype) else x
                    for x in o]
        return o
    return o


def _selected(tag):
    if _ONLY and not any(tag.startswith(p) for p in _ONLY):
        return False
    if _SKIP and any(tag.startswith(p) for p in _SKIP):
        return False
    return True


def p(tag, o):
    if MODE == "off":
        return o
    _record(tag, o)
    if MODE == "promote" and _selected(tag):
        return _promote(tag, o)
    return o


def _dump():
    if MODE == "off" or not _OUT:
        return
    os.makedirs(_OUT, exist_ok=True)
    who = os.path.basename(os.getcwd())
    path = os.path.join(_OUT, "ledger_%s_%d.json" % (who, os.getpid()))
    with open(path, "w") as f:
        json.dump({"mode": MODE, "cwd": os.getcwd(), "argv": sys.argv[:3],
                   "only": _ONLY, "skip": _SKIP,
                   "ledger": LEDGER, "promoted": PROMOTED}, f, indent=1)


atexit.register(_dump)
