"""exp_097 stageA: turn the deployed package's float32 sites into float64.

Each substitution is asserted to occur EXACTLY ONCE so a silent no-op is
impossible (the sed version of this failed silently once).
"""
import os
import sys

NN = [
    ("mu = np.array(M['mu'], dtype=np.float32)",
     "mu = np.array(M['mu'], dtype=np.float64)  # EXP097"),
    ("sd = np.array(M['sd'], dtype=np.float32)",
     "sd = np.array(M['sd'], dtype=np.float64)  # EXP097"),
    ("X = df[DENSE].astype('float32').values",
     "X = df[DENSE].astype('float64').values  # EXP097"),
    ("X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)",
     "X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)  # EXP097"),
    ("embs = [W[f'emb{i}'] for i in range(len(CATS))]",
     "embs = [W[f'emb{i}'].astype(np.float64) for i in range(len(CATS))]  # EXP097"),
]
DEEP = [
    ("mu = np.array(P['mu'], np.float32); sd = np.array(P['sd'], np.float32)",
     "mu = np.array(P['mu'], np.float64); sd = np.array(P['sd'], np.float64)  # EXP097"),
    ("X = df[NUM].astype('float32').values",
     "X = df[NUM].astype('float64').values  # EXP097"),
    ("X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)",
     "X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)  # EXP097"),
]
INS = [("return out.astype('float32')",
        "return out.astype('float64')  # EXP097")]


def do(path, subs):
    s = open(path, encoding="utf-8").read()
    for a, b in subs:
        assert s.count(a) == 1, (path, a, s.count(a))
        s = s.replace(a, b)
    open(path, "w", encoding="utf-8").write(s)
    print("  patched %s (%d sites)" % (path, len(subs)))


D = sys.argv[1]
which = sys.argv[2:]
if "nn" in which:
    do(os.path.join(D, "mate/nn_infer.py"), NN)
if "deep" in which:
    do(os.path.join(D, "mate/deep_infer.py"), DEEP)
if "ins" in which:
    do(os.path.join(D, "inseason.py"), INS)
    do(os.path.join(D, "mate/inseason.py"), INS)
print("PATCH_OK", D, which)
