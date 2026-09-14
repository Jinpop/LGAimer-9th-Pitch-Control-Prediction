"""exp_064 stage 0 — 배포본 sha 대조 + 계산용 표 dir 구성.

만드는 것: work/tables_shipp6/
  · p6 3종 (pitchmodel6.txt · pmcat6.json · anom_cnt.csv) = **출하 배포본 복사**
  · 나머지 7종 = exp_062 가 만든 우리 ≤2024 재구축본 (exp062_gate/work/tables_up2024)

exp_062 는 그 7종이 출하본과 **바이트 동일**함을 이미 쟀다. 여기서 다시 확인한다 —
7종이 동일하고 3종을 복사했으면 이 dir 은 출하 표 dir 과 바이트 동일해지고,
그때 44열 대조는 **구조상 0** 이 된다. 그것이 stage 2 게이트가 확인할 명제다.

🔴 실행 전 sha 대조. 안 맞으면 SystemExit.
"""
import hashlib
import json
import os
import shutil
import sys

H = os.path.expanduser("~")
B = f"{H}/SungjinPark/exp064"
SHIP_T = f"{H}/SungjinPark/exp062_gate/ship/mate/model/tables"
SHIP_PIPE = f"{H}/SungjinPark/exp062_gate/ship/mate/pipeline.py"
OURS_T = f"{H}/SungjinPark/exp062_gate/work/tables_up2024"
CFG_T = f"{H}/SungjinPark/build_AI/pkg/model_v6/tables"
DST = f"{B}/work/tables_shipp6"

# config.yaml 이 못 박은 기대값 (결과를 보고 바꾸지 않는다)
EXPECT = {
    "pitchmodel6.txt": "b743ac3d2cd14bdc31670db8457f1806bdf1f989d947fd012905c81e2a1b2a70",
    "pmcat6.json": "4b73ebb9f8787cb2753c7d9c15e4c41c8d6e4539335297709759e022ea58ead6",
    "anom_cnt.csv": "4266ee9a274d122cf18abadc5612c1f01b147b13160fb8130deb9abadd9d18fe",
}
PIPE_EXPECT = "a3181a10a1c621524460f14bc06b3c7f188fe23955814bb5e5b890e0df3016c9"
P6_FILES = list(EXPECT)
OTHER7 = ["meta.json", "cnt_te.csv", "te48.csv", "pitcher_overall.csv",
          "sit_cnt.csv", "sit_run.csv", "sit_sc.csv"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


res = {"expect_from_config": EXPECT, "pipeline_expect": PIPE_EXPECT}
fail = []

# ── ① 배포본 p6 3종 sha ──────────────────────────────────────────────────────
res["ship_p6_sha"] = {}
for f in P6_FILES:
    a, b = sha(f"{SHIP_T}/{f}"), sha(f"{CFG_T}/{f}")
    ok = (a == EXPECT[f]) and (b == EXPECT[f])
    res["ship_p6_sha"][f] = {
        "ship_mate_model_tables": a, "build_AI_pkg_model_v6_tables": b,
        "expected": EXPECT[f], "pass": ok}
    if not ok:
        fail.append(f"p6 sha 불일치: {f}")

# ── ② 출하 pipeline.py sha ───────────────────────────────────────────────────
ps = sha(SHIP_PIPE)
res["ship_pipeline_sha"] = {"sha": ps, "pass": ps == PIPE_EXPECT}
if ps != PIPE_EXPECT:
    fail.append("pipeline.py sha 불일치")

# ── ③ 나머지 7종: 우리 ≤2024 재구축본 vs 출하본 ───────────────────────────────
res["other7_ours_vs_ship"] = {}
for f in OTHER7:
    a, b = sha(f"{OURS_T}/{f}"), sha(f"{SHIP_T}/{f}")
    res["other7_ours_vs_ship"][f] = {"ours_up2024": a, "ship": b, "equal": a == b}
    if a != b:
        fail.append(f"7종 중 {f} 가 출하본과 다르다 (exp_062 관측과 어긋남)")

if fail:
    json.dump(res, open(f"{B}/out/stage0_tables.json", "w"), indent=1,
              ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))
    raise SystemExit("🔴 STAGE0 FAIL: " + " | ".join(fail))

# ── ④ 표 dir 구성 ────────────────────────────────────────────────────────────
os.makedirs(DST, exist_ok=True)
for f in OTHER7:
    shutil.copy2(f"{OURS_T}/{f}", f"{DST}/{f}")
for f in P6_FILES:
    shutil.copy2(f"{SHIP_T}/{f}", f"{DST}/{f}")

res["tables_shipp6"] = {f: {"sha": sha(f"{DST}/{f}"),
                            "equals_ship": sha(f"{DST}/{f}") == sha(f"{SHIP_T}/{f}"),
                            "src": "ship" if f in EXPECT else "ours_up2024"}
                        for f in OTHER7 + P6_FILES}
res["tables_shipp6_all_equal_ship"] = all(
    v["equals_ship"] for v in res["tables_shipp6"].values())

# ── ⑤ pitcher_id 어휘 대조 (배포 571 vs 우리 685) ────────────────────────────
lv_ship = json.load(open(f"{SHIP_T}/pmcat6.json"))
lv_ours = json.load(open(f"{OURS_T}/pmcat6.json"))
ss, so = set(lv_ship["pitcher_id"]), set(lv_ours["pitcher_id"])
res["pitcher_id_levels"] = {
    "ship": len(ss), "ours_up2024": len(so), "common": len(ss & so),
    "ship_only": len(ss - so), "ours_only": len(so - ss),
    "other_cats_ship": {c: len(v) for c, v in lv_ship.items() if c != "pitcher_id"},
    "other_cats_ours": {c: len(v) for c, v in lv_ours.items() if c != "pitcher_id"}}

json.dump(res, open(f"{B}/out/stage0_tables.json", "w"), indent=1,
          ensure_ascii=False)
print(json.dumps({k: v for k, v in res.items()
                  if k not in ("expect_from_config",)}, indent=1,
                 ensure_ascii=False))
print("STAGE0_OK", flush=True)
