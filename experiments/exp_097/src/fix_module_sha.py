"""계측 사본 전용 — aa_infer.py 가 assert 로 잠그는 `module_sha256` 를 갱신한다.

⚠️ 측정용 사본에만 쓴다. 출하 후보에는 쓰지 않는다.
"""
import hashlib
import json
import os
import sys

PKG = sys.argv[1]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


META = os.path.join(PKG, "model/meta.json")
m = json.load(open(META))
changed = []


def walk(o, path=""):
    if isinstance(o, dict):
        if "module" in o and "module_sha256" in o:
            p = os.path.join(PKG, o["module"])
            if os.path.exists(p):
                new = sha(p)
                if new != o["module_sha256"]:
                    changed.append((path + ".module_sha256", o["module_sha256"], new))
                    o["module_sha256"] = new
        if "anchor_file" in o and "anchor_sha256" in o:
            p = os.path.join(PKG, "model", o["anchor_file"])
            if os.path.exists(p):
                new = sha(p)
                if new != o["anchor_sha256"]:
                    changed.append((path + ".anchor_sha256", o["anchor_sha256"], new))
                    o["anchor_sha256"] = new
        for k, v in o.items():
            walk(v, path + "." + str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            walk(v, path + "[%d]" % i)


walk(m)
json.dump(m, open(META, "w"), indent=1, ensure_ascii=False)
print("updated %d fields" % len(changed))
for c in changed:
    print("  ", c[0], c[1][:16], "->", c[2][:16])
