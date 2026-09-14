#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S77 — 게이트를 쏘기 전에 산문을 얼린다.  99_pack 이 이 sha 를 다시 대조한다."""
import hashlib
import os

H = "/home/<user>/SungjinPark"
B = f"{H}/build_DN"
REL = ["pkg/DN/blend.json", "pkg/DN/script.py",
       "pkg/ANC_DN/blend.json", "pkg/ANC_DN/script.py",
       "pkg/DNP/blend.json", "pkg/DNP/script.py"]


def sh(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


lines = []
for r in REL:
    assert os.path.exists(f"{B}/{r}"), r
    lines.append(f"{sh(f'{B}/{r}')}  {r}")
open(f"{B}/work/frozen_prose_sha.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print("FREEZE_OK")
