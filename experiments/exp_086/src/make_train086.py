"""exp_086 stage 0 — 학습 스크립트를 exp_085 `train085.py` 에서 **기계로** 파생한다.

왜 손으로 안 쓰나: config 가 "바꾸는 것은 시드뿐"을 요구한다. 손으로 옮겨 쓰면
레시피가 조용히 흔들려도 아무도 못 본다. 여기서는 exp_085 원본에 **명시된 치환만**
적용하고 각 치환의 발생 횟수를 assert 한 뒤 unified diff 를 남긴다.
diff 에 레시피 줄(에폭·LR·스케줄·batch·손실·take)이 나오면 그 자체가 실패다.
"""
import difflib
import hashlib
import json
import os

H = "/home/<user>/SungjinPark"
SRC = f"{H}/exp085/src/train085.py"
DST = f"{H}/exp086/src/train086.py"
os.makedirs(os.path.dirname(DST), exist_ok=True)

orig = open(SRC).read()
src_sha = hashlib.sha256(orig.encode()).hexdigest()
assert src_sha == "29c1973a19173a50fa5e4cc716acf3afdfa23e33006cefca8e8f16ad1b89b033", src_sha

SUBS = [
    # (1) 시드를 인자로 — 이것이 이 실험의 유일한 의도적 변경
    ('SEED = 42\n',
     'SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 42\n', 1),
    # (2) 출력 뿌리를 exp086 으로, 조립 캐시는 exp085 를 읽기 전용으로
    ('B = f"{H}/exp085"\n',
     'B = f"{H}/exp086"\nWIN = f"{H}/exp085/work"   # 읽기 전용 — exp_085 조립 캐시\n', 1),
    ('os.makedirs(f"{OUT}/preds", exist_ok=True)\n',
     'os.makedirs(f"{OUT}/preds", exist_ok=True)\nos.makedirs(W, exist_ok=True)\n', 1),
    # (3) 읽기 경로 6개 → WIN
    ('open(f"{W}/schema.json")', 'open(f"{WIN}/schema.json")', 1),
    ('np.load(f"{W}/Xd.npy")', 'np.load(f"{WIN}/Xd.npy")', 1),
    ('np.load(f"{W}/codes.npy")', 'np.load(f"{WIN}/codes.npy")', 1),
    ('np.load(f"{W}/y.npy")', 'np.load(f"{WIN}/y.npy")', 1),
    ('np.load(f"{W}/train_mask.npy")', 'np.load(f"{WIN}/train_mask.npy")', 1),
    ('np.load(f"{W}/valid_mask.npy")', 'np.load(f"{WIN}/valid_mask.npy")', 1),
    # (4) 머리말 한 줄 — 파생 사실을 파일 안에 남긴다
    ('"""exp_085 stage 2 — 팔 1개 학습 (F1_FM 또는 F2_EMB), fold_2024, 시드 42 단독.\n',
     '"""exp_086 stage 2 — 팔 1개 학습 (F1_FM 또는 F2_EMB), fold_2024, 시드 인자.\n'
     '\n'
     '🔴 이 파일은 exp_085 `train085.py` 에서 `make_train086.py` 가 기계로 파생했다.\n'
     '   의도적 변경은 **시드를 인자로 받는 것 하나**이고 나머지는 경로뿐이다.\n'
     '   조립 캐시(Xd/codes/y/mask/schema)는 exp_085 work 를 **읽기만** 한다.\n', 1),
]

out = orig
log = []
for a, b, n in SUBS:
    c = out.count(a)
    assert c == n, (a[:60], c, n)
    out = out.replace(a, b)
    log.append({"from": a, "to": b, "count": c})

open(DST, "w").write(out)
dst_sha = hashlib.sha256(out.encode()).hexdigest()
diff = list(difflib.unified_diff(orig.splitlines(True), out.splitlines(True),
                                 "exp085/src/train085.py", "exp086/src/train086.py"))

# 레시피 줄이 diff 에 나타나면 승계 실패
RECIPE_TOKENS = ["EPOCHS =", "BATCH =", "MAX_LR =", "WD =", "PCT_START =",
                 "OneCycleLR", "BCEWithLogits", "AdamW", "best[", "take"]
touched = [l for l in diff if l[:1] in "+-" and l[:3] not in ("---", "+++")
           and any(t in l for t in RECIPE_TOKENS)]
res = {"src": SRC.replace(H, "~"), "dst": DST.replace(H, "~"),
       "src_sha256": src_sha, "dst_sha256": dst_sha,
       "n_substitutions": len(SUBS),
       "diff_added_lines": sum(1 for l in diff if l.startswith("+") and l[:3] != "+++"),
       "diff_removed_lines": sum(1 for l in diff if l.startswith("-") and l[:3] != "---"),
       "recipe_lines_touched": touched,
       "recipe_intact": len(touched) == 0,
       "unified_diff": "".join(diff)}
assert res["recipe_intact"], touched
json.dump(res, open(f"{H}/exp086/out/derive.json", "w"), ensure_ascii=False, indent=1)
print("".join(diff))
print(json.dumps({k: v for k, v in res.items() if k != "unified_diff"},
                 ensure_ascii=False, indent=1))
