"""exp_088 stage 1a — 조립·학습 스크립트를 exp_085/086 원본에서 **기계로** 파생한다.

config: "exp_085 r2 레시피 그대로. 바꾸는 것은 **학습 구간뿐**(≤2023 → ≤2024 전량)".
손으로 옮겨 쓰면 레시피가 조용히 흔들려도 아무도 못 본다. 여기서는 원본에 **명시된
치환만** 적용하고 각 치환의 발생 횟수를 assert 한 뒤 unified diff 를 남긴다.

🔴 선언된 예외 하나: ≤2024 전량으로 학습하면 **held-out valid 가 사라진다**.
   exp_085 r2 의 상태 선택 규칙(valid 2024 Brier 최소 에폭)은 그 면에서 정의되지 않는다.
   그래서 상태 선택만 "≤2023 면에서 고른 에폭에 못박기"로 바꾼다. 이 변경은 아래
   DECLARED 목록에 정확히 열거하고, 그 밖의 레시피 토큰이 diff 에 나오면 죽는다.
"""
import difflib, hashlib, json, os

H = "/home/<user>/SungjinPark"
OUT = f"{H}/exp088/out"
os.makedirs(OUT, exist_ok=True)
RES = {}

HARD_TOKENS = ["EPOCHS =", "BATCH =", "MAX_LR =", "WD =", "PCT_START =",
               "OneCycleLR", "BCEWithLogits", "AdamW"]
SOFT_TOKENS = ["best[", "take", "train_mask", "season <="]


def derive(tag, src, dst, subs, declared):
    orig = open(src).read()
    out = orig
    log = []
    for a, b, n in subs:
        c = out.count(a)
        assert c == n, (tag, a[:70], c, n)
        out = out.replace(a, b)
        log.append({"from": a, "to": b, "count": c})
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    open(dst, "w").write(out)
    diff = list(difflib.unified_diff(orig.splitlines(True), out.splitlines(True),
                                     src.replace(H, "~"), dst.replace(H, "~")))
    chg = [l for l in diff if l[:1] in "+-" and l[:3] not in ("---", "+++")]
    hard = [l for l in chg if any(t in l for t in HARD_TOKENS)]
    soft = [l for l in chg if any(t in l for t in SOFT_TOKENS)]
    r = {"src": src.replace(H, "~"), "dst": dst.replace(H, "~"),
         "src_sha256": hashlib.sha256(orig.encode()).hexdigest(),
         "dst_sha256": hashlib.sha256(out.encode()).hexdigest(),
         "n_substitutions": len(subs), "substitutions": log,
         "diff_added_lines": sum(1 for l in diff if l.startswith("+") and l[:3] != "+++"),
         "diff_removed_lines": sum(1 for l in diff if l.startswith("-") and l[:3] != "---"),
         "hard_recipe_lines_touched": hard,
         "recipe_intact": len(hard) == 0,
         "soft_lines_touched": soft,
         "declared_change": declared,
         "unified_diff": "".join(diff)}
    assert r["recipe_intact"], (tag, hard)
    RES[tag] = r
    print(f"--- {tag}: +{r['diff_added_lines']}/-{r['diff_removed_lines']} "
          f"recipe_intact={r['recipe_intact']} soft={len(soft)}")
    return r


# ═══ (1) prep088.py — 조립. 바꾸는 것은 학습 구간뿐 ═══════════════════════
derive(
    "prep088",
    f"{H}/exp085/src/prep085.py", f"{H}/exp088/src/prep088.py",
    [
        ('"""exp_085 stage 1 — 학습 행렬 조립. 학습 없음, 캐시만 만든다.\n',
         '"""exp_088 stage 1 — 학습 행렬 조립 (**배포 학습량 ≤2024 전량**).\n'
         '\n'
         '🔴 이 파일은 exp_085 `prep085.py` 에서 `make088.py` 가 기계로 파생했다.\n'
         '   의도적 변경은 학습 분할 하나 — 표준화 통계·어휘가 전 행에서 만들어진다.\n'
         '   나머지(열 목록·결측 플래그·정렬 assert·<UNK> 규약)는 바이트 그대로다.\n', 1),
        ('B = f"{H}/exp085"\n', 'B = f"{H}/exp088"\n', 1),
        ('M = {"exp_id": "exp_085", "stage": "prep"}\n',
         'M = {"exp_id": "exp_088", "stage": "prep_le2024_full"}\n', 1),
        ('train_mask = (season <= 2023) & ~((season <= 2022) & (gt == "F"))\n',
         'train_mask = np.ones(len(season), dtype=bool)   # ≤2024 전량 (배포 학습량)\n', 1),
        ('    "train_mask_rule": "season<=2023 & ~((season<=2022)&(game_type==\'F\'))  (exp_060 승계)",\n',
         '    "train_mask_rule": "전 행 (season<=2024 전량, F 필터 없음) — 팀원 딥 팔과 같은 1,475,092행",\n', 1),
    ],
    declared="train_mask: (season<=2023 & F필터) → 전 행. 그 결과 mu/sd·어휘도 전 행에서 나온다",
)

# ═══ (2) train088.py — 학습. 상태 선택만 선언된 예외 ═══════════════════════
derive(
    "train088",
    f"{H}/exp086/src/train086.py", f"{H}/exp088/src/train088.py",
    [
        ('"""exp_086 stage 2 — 팔 1개 학습 (F1_FM 또는 F2_EMB), fold_2024, 시드 인자.\n',
         '"""exp_088 stage 2 — 팔 1개 학습, 조립 캐시·에폭 못박기를 인자로.\n'
         '\n'
         '🔴 이 파일은 exp_086 `train086.py` 에서 `make088.py` 가 기계로 파생했다.\n'
         '   사용: train088.py <ARM> <DEV> <RUNTAG> <SEED> <CACHE_DIR> <PIN_EP>\n'
         '   PIN_EP=0 이면 exp_086 과 동일하게 valid 2024 최소 Brier 에폭의 상태를 쓴다.\n'
         '   PIN_EP>0 이면 그 에폭의 상태를 쓴다 — ≤2024 전량 학습에는 held-out valid 가\n'
         '   없으므로 ≤2023 면에서 고른 에폭을 못박는 것이 유일하게 정의된 규칙이다.\n'
         '   스케줄·에폭 수·LR·batch·손실·최적화기는 한 글자도 안 바뀐다.\n', 1),
        ('B = f"{H}/exp086"\nWIN = f"{H}/exp085/work"   # 읽기 전용 — exp_085 조립 캐시\n',
         'B = f"{H}/exp088"\nWIN = sys.argv[5] if len(sys.argv) > 5 else f"{H}/exp088/work"\n'
         'PIN_EP = int(sys.argv[6]) if len(sys.argv) > 6 else 0\n', 1),
        ('    if br < best["brier"]:\n',
         '    if (ep == PIN_EP) if PIN_EP else (br < best["brier"]):\n', 1),
        ('    "take": "best", "best_epoch": best["epoch"],\n',
         '    "take": ("pinned_epoch" if PIN_EP else "best"), "pin_ep": PIN_EP,\n'
         '    "cache_dir": WIN.replace(H, "~"), "best_epoch": best["epoch"],\n', 1),
    ],
    declared=("상태 선택 1줄: PIN_EP>0 이면 valid-argmin 대신 그 에폭의 상태. "
              "≤2024 전량 학습에는 held-out valid 가 없다. 그 밖의 레시피는 불변"),
)

json.dump(RES, open(f"{OUT}/derive088.json", "w"), ensure_ascii=False, indent=1)
for t in RES:
    print(f"\n===== {t} unified diff =====")
    print(RES[t]["unified_diff"])
