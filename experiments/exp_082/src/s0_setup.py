"""exp_082 stage0 준비 — 패키지 3종 구성 + 파일 sha 대조.  파일 삭제 없음(새로 만드는 것만)."""
import os, json, shutil, hashlib, zipfile, sys

H = "/home/<user>/SungjinPark"
OUT = f"{H}/exp082"
EXBO = f"{H}/audit_BT/ex_BO"
BOZIP = f"{H}/build_BO/submit_BO_inner_vertex.zip"
NEW = f"{OUT}/newfiles"
D_BO = 0.02875748759679648
D_NEW = 0.028693042338963914
R = {}

def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()

def tree_sha(root):
    out = {}
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in ("__pycache__", "data", "output")]
        for f in fn:
            if f in ("_module_origins.json", "_module_origins_root.json"):
                continue
            p = os.path.join(dp, f)
            if os.path.islink(p):
                continue
            out[os.path.relpath(p, root)] = sha_file(p)
    return out

# 1. ex_BO 가 BO zip 의 충실한 전개인가
z = zipfile.ZipFile(BOZIP)
zsha = {}
for i in z.infolist():
    if i.is_dir():
        continue
    h = hashlib.sha256()
    with z.open(i) as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    zsha[i.filename] = h.hexdigest()
exbo = tree_sha(EXBO)
mism = {k: (zsha.get(k), exbo.get(k)) for k in set(zsha) | set(exbo)
        if zsha.get(k) != exbo.get(k)}
R["G0_prep_exBO_matches_BO_zip"] = {"n_members": len(zsha), "n_mismatch": len(mism),
                                    "mismatch": {k: v for k, v in list(mism.items())[:10]},
                                    "PASS": len(mism) == 0}
print("[prep] ex_BO vs BO zip 멤버", len(zsha), "불일치", len(mism), flush=True)
assert not mism, mism

want = {k: v[0] for k, v in json.load(open(f"{H}/bonly_members.json")).items()}

# 2. pkg_bonly = ex_BO + 4파일 교체
def build(dst, edits=None):
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    shutil.copytree(EXBO, dst, symlinks=False,
                    ignore=shutil.ignore_patterns("__pycache__", "data", "output",
                                                  "_module_origins.json",
                                                  "_module_origins_root.json"))
    for rel in ("blend.json", "script.py", "mate/script.py", "mate/q_infer.py"):
        shutil.copyfile(f"{NEW}/{rel}", f"{dst}/{rel}")
    if edits:
        bp = f"{dst}/blend.json"
        bl = json.load(open(bp, encoding="utf-8"))
        edits(bl)
        json.dump(bl, open(bp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    return tree_sha(dst)

PK = f"{OUT}/pkg_bonly"
got = build(PK)
mism2 = {k: (want.get(k), got.get(k)) for k in set(want) | set(got) if want.get(k) != got.get(k)}
R["G0_prep_pkg_bonly_reconstructed"] = {"n_files": len(got), "n_mismatch": len(mism2),
                                        "mismatch": {k: v for k, v in list(mism2.items())[:10]},
                                        "PASS": len(mism2) == 0}
print("[prep] pkg_bonly 재구성 파일", len(got), "불일치", len(mism2), flush=True)
assert not mism2, mism2

# 3. pkg_ident = b=0, c=0, d_adj=d_BO  (항등 앵커의 계산판)
def _ed(bl):
    bl["failtype_offset"]["b"] = 0.0
    bl["failtype_offset"]["c"] = 0.0
    bl["d_adj"] = D_BO
gi = build(f"{OUT}/pkg_ident", _ed)
non_blend_ok = all(gi[k] == want[k] for k in gi if k != "blend.json")
R["G0_prep_pkg_ident"] = {"non_blend_files_identical_to_bonly": bool(non_blend_ok),
                          "n_files": len(gi)}
print("[prep] pkg_ident 구성 (blend.json 외 전부 bonly 와 동일:", non_blend_ok, ")", flush=True)
assert non_blend_ok

# 4. pkg_BOref = ex_BO 그대로 (같은 세션 앵커용)
BOREF = f"{OUT}/pkg_BOref"
if os.path.isdir(BOREF):
    shutil.rmtree(BOREF)
shutil.copytree(EXBO, BOREF, symlinks=False,
                ignore=shutil.ignore_patterns("__pycache__", "data", "output",
                                              "_module_origins.json",
                                              "_module_origins_root.json"))

# 5. mate_cal — tr2024 보정 규약 (lt2024 앵커 스왑)
PACK = f"{H}/build_AI/pack/ins"
FRAME = f"{H}/build_AC/work/tr2024"
MC = f"{OUT}/work/mate_cal"
if os.path.isdir(MC):
    shutil.rmtree(MC)
shutil.copytree(f"{PK}/mate", MC, symlinks=False)
shipped_anchor = sha_file(f"{MC}/model/anchor.npz")
shutil.copyfile(f"{PACK}/anchor_lt2024.npz", f"{MC}/model/anchor.npz")
cal_anchor = sha_file(f"{MC}/model/anchor.npz")
mb = json.load(open(f"{MC}/model/blend.json", encoding="utf-8"))
os.makedirs(f"{MC}/data", exist_ok=True)
for f in ("test.csv", "sample_submission.csv"):
    shutil.copyfile(f"{FRAME}/{f}", f"{MC}/data/{f}")
R["mate_cal"] = {"shipped_anchor_sha256": shipped_anchor,
                 "calibration_anchor_sha256": cal_anchor,
                 "expect_cal_anchor": "0e4b586da41cd6e78c5e19d4dd656bc9df088413625945e2a88ae85d100e5eb7",
                 "inner_w_v6": mb["w_v6"], "inner_w_ebm": mb["w_ebm"],
                 "inner_beta_ins": mb["beta_ins"], "inner_d_adj": mb["d_adj"],
                 "frame": FRAME,
                 "frame_test_sha256": sha_file(f"{FRAME}/test.csv")}
assert cal_anchor == "0e4b586da41cd6e78c5e19d4dd656bc9df088413625945e2a88ae85d100e5eb7"
assert mb["w_v6"] == 0.6668397489 and mb["w_ebm"] == -0.1 and mb["beta_ins"] == 0.03
print("[prep] mate_cal 준비 · 앵커 스왑 확인 · inner w_v6", mb["w_v6"], flush=True)

bl = json.load(open(f"{PK}/blend.json", encoding="utf-8"))
ft = bl["failtype_offset"]
R["shipped_constants"] = {"b": ft["b"], "c": ft["c"], "mu_mr": ft["mu_mr"],
                          "mu_wo": ft["mu_wo"], "d_adj_new": bl["d_adj"],
                          "d_adj_BO": D_BO, "target_rate": bl["target_rate"],
                          "alpha": bl["alpha"], "beta_skew": bl["beta_skew"],
                          "beta_f": bl["beta_f"], "w_v6_outer": bl["w_v6"]}
assert bl["d_adj"] == D_NEW and ft["b"] == -0.05 and ft["c"] == 0.0
json.dump(R, open(f"{OUT}/out/s0_prep.json", "w"), indent=1, ensure_ascii=False)
print("[prep] 완료 ->", f"{OUT}/out/s0_prep.json", flush=True)
