"""exp_097: aa_infer.py:546 asserts sha256(inseason.py) against
model/meta.json branches.*.stacker_extra_features.module_sha256.
Any edit to inseason.py -- even a print -- stops the run.  Re-stamp the record
so the variant can run.  The field is used ONLY by that assert; no arithmetic
reads it.
"""
import hashlib
import json
import os
import sys

D = sys.argv[1]
p_ins = os.path.join(D, "inseason.py")
h = hashlib.sha256(open(p_ins, "rb").read()).hexdigest()
p_meta = os.path.join(D, "model/meta.json")
s = open(p_meta, encoding="utf-8").read()
M = json.loads(s)
old = set()
for br in M["branches"].values():
    sx = br.get("stacker_extra_features")
    if sx and "module_sha256" in sx:
        old.add(sx["module_sha256"])
assert len(old) == 1, old
o = old.pop()
assert s.count(o) >= 1
s2 = s.replace(o, h)
open(p_meta, "w", encoding="utf-8").write(s2)
print("re-stamped", p_meta, o[:12], "->", h[:12],
      "occurrences", s.count(o))
