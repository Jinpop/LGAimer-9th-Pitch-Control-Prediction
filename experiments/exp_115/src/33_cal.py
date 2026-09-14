#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""캘프레임 실행 — 팀원 관례(calBX/calCG)를 그대로 재구성한다.  부모 CT 와 같은 하네스.

관례 출처: 부모 zip blend.json 키 `CR_d_adj_provenance` / `CQ_d_adj_provenance` (원문):
  ① ins 앵커 두 곳을 anchor_lt2024(sha256 0e4b586d…, 투수 711 / 타자 742)로 스왑
  ② offset.lookup_d / lookup_n 키를 ≤2023 집합 1,202 로 제한 — mate/model_q/meta.json 과
     model/meta.json 양쪽
  프레임 = train season==2024 전량 253,507행 (build_AC/work/tr2024)
  조준점(G-AIM) = 명목 target_rate 가 아니라 ★같은 프레임에서 부모가 내는 평균★

🔴 CZ 의 축은 ★표 조회를 탄다★ — 부모가 이미 싣고 있는 ./model/deploy_splits.npz 의
   네 번째 셀(bat_phand)이다.  하네스는 조상들과 한 글자도 다르지 않게 유지한다: 조준점
   (G-AIM)이 "부모가 이 프레임에서 내는 평균" 이라서, 하네스를 바꾸면 조준점이 바뀐다.
   ⚠️ 부모 CT 가 새로 넣은 ./model/deploy_splits.npz 는 ≤2024 동결표라 이 프레임(2024 행)을
   본 표다.  그래도 하네스에서 건드리지 않는다 — 부모와 후보가 같은 표를 쓰므로 그 기여가
   조준점과 후보 양쪽에 똑같이 들어가 d_adj 해에서 상쇄된다.

사용: 33_cal.py PAR CZ_0 ZI ZP ZS
"""
import hashlib, json, os, shutil, subprocess, sys, time
import numpy as np
import pandas as pd

H = "/home/<user>/SungjinPark"
B = f"{H}/build_pb3_probe"
PKG, WORK, LOGS = f"{B}/pkg", f"{B}/work", f"{B}/logs"
PY = f"{H}/venv/bin/python"
FRAME = f"{H}/build_AC/work/tr2024"
ANCHOR_LT2024 = f"{H}/build_AI/pack/ins/anchor_lt2024.npz"
FRAME_SHA = "4db38113d5355546339c4391fbbf8b2e152a7c248ad9261caa1071f6bb81f5d0"
ANCHOR_SHA = "0e4b586da41cd6e78c5e19d4dd656bc9df088413625945e2a88ae85d100e5eb7"
SHIPPED_ANCHOR_SHA = "7b96e3de0baeac54bf8cb1727b8913a075efd487465adda1b762aef1f2d1b020"
N_ROWS = 253507
DECLARED_AIM = 0.48589279261815      # 부모 CS3 blend.json `target_rate`
TAGS = sys.argv[1:] or ["PAR"]
SUF = os.environ.get("CALSUF", "")


def sh(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


assert sh(f"{FRAME}/test.csv") == FRAME_SHA
assert sh(ANCHOR_LT2024) == ANCHOR_SHA
os.makedirs(LOGS, exist_ok=True)

tr = pd.read_csv(f"{H}/data/train.csv",
                 usecols=["row_id", "season", "game_type", "control_success",
                          "pitcher_id", "batter_hand"])


def keyset(maxseason):
    m = (tr.season <= maxseason) & ~((tr.season <= 2022) & (tr.game_type == "F"))
    d = tr[m & tr.control_success.notna()]
    pid = d.pitcher_id.astype("Int64")
    g = pd.DataFrame({"p": pid, "b": d.batter_hand}).dropna().drop_duplicates()
    return {f"{int(a)}|{int(b)}" for a, b in zip(g.p, g.b)}, int(len(d))


K23, NFIT23 = keyset(2023)
assert len(K23) == 1202, len(K23)
print(f"≤2023 키 {len(K23)} · 적합행 {NFIT23}")
INFO = {"keyset_le2023": len(K23), "n_fit_rows_le2023": NFIT23,
        "declared_aim_in_parent_target_rate": DECLARED_AIM, "suffix": SUF}


def mk_cal(tag):
    d = f"{WORK}/cal_{tag}{SUF}"
    if os.path.isdir(d):
        shutil.rmtree(d)
    shutil.copytree(f"{PKG}/{tag}", d, symlinks=False,
                    ignore=shutil.ignore_patterns("__pycache__", "data", "output",
                                                  "_module_origins.json",
                                                  "_module_origins_root.json"))
    swapped = []
    for rel in ("mate/model/anchor.npz", "model/ins/deploy_anchor.npz"):
        p = f"{d}/{rel}"
        assert sh(p) == SHIPPED_ANCHOR_SHA, (rel, sh(p))
        shutil.copyfile(ANCHOR_LT2024, p)
        swapped.append(rel)
    rec = {}
    mp = f"{d}/model/meta.json"
    mm = json.load(open(mp, encoding="utf-8"))
    for bn, br in mm["branches"].items():
        sx = br.get("stacker_extra_features")
        if not sx or "anchor_sha256" not in sx:
            continue
        rec[bn] = [sx["anchor_sha256"], sx.get("anchor_n_pitchers"),
                   sx.get("anchor_n_batters")]
        sx["anchor_sha256"] = ANCHOR_SHA
        sx["anchor_n_pitchers"] = 711
        sx["anchor_n_batters"] = 742
        sx["CALIBRATION_ONLY_SUBSTITUTION"] = "AV/AX/AY/CE/CG/CN/CQ/CR/CT/CS3 와 같은 규약"
    open(mp, "w", encoding="utf-8", newline="").write(
        json.dumps(mm, ensure_ascii=False, indent=1))
    restricted = {}
    for rel in ("mate/model_q/meta.json", "model/meta.json"):
        p = f"{d}/{rel}"
        m = json.load(open(p, encoding="utf-8"))
        n0 = len(m["offset"]["lookup_d"])
        m["offset"]["lookup_d"] = {k: v for k, v in m["offset"]["lookup_d"].items()
                                   if k in K23}
        m["offset"]["lookup_n"] = {k: v for k, v in m["offset"]["lookup_n"].items()
                                   if k in K23}
        n1 = len(m["offset"]["lookup_d"])
        open(p, "w", encoding="utf-8", newline="").write(
            json.dumps(m, ensure_ascii=False, indent=1))
        restricted[rel] = [n0, n1]
        assert n1 == 1202, (rel, n1)
    os.symlink(FRAME, f"{d}/data")
    return d, {"anchors_swapped": swapped, "lookup_restricted": restricted,
               "anchor_record_fields_before": rec,
               "league_dev_in_this_run":
                   json.load(open(f"{d}/mate/model_q/meta.json",
                                  encoding="utf-8"))["backoff"]["league_dev"],
               "d_adj_in_this_run":
                   repr(json.load(open(f"{d}/blend.json",
                                       encoding="utf-8"))["d_adj"])}


env = dict(os.environ, OMP_NUM_THREADS="6", OPENBLAS_NUM_THREADS="6",
           MKL_NUM_THREADS="6", NUMEXPR_NUM_THREADS="6")
procs = {}
t0 = time.time()
for tag in TAGS:
    d, meta = mk_cal(tag)
    INFO[f"cal_{tag}"] = meta
    lf = open(f"{LOGS}/cal_{tag}{SUF}.log", "w")
    procs[tag] = (subprocess.Popen([PY, "script.py"], cwd=d, stdout=lf,
                                   stderr=subprocess.STDOUT, env=env), d, lf)
    print(f"  발사 cal_{tag}{SUF}", flush=True)

ids = pd.read_csv(f"{FRAME}/test.csv", encoding="utf-8-sig",
                  usecols=["row_id"])["row_id"].tolist()
assert len(ids) == N_ROWS

for tag in TAGS:
    pr, d, lf = procs[tag]
    rc = pr.wait()
    lf.close()
    assert rc == 0, f"cal_{tag} rc={rc} — {LOGS}/cal_{tag}{SUF}.log"
    out = f"{d}/output/submission.csv"
    df = pd.read_csv(out, encoding="utf-8-sig", float_precision="round_trip")
    assert list(df.columns) == ["row_id", "control_success"]
    m = dict(zip(df["row_id"].tolist(),
                 df["control_success"].to_numpy(np.float64).tolist()))
    v = np.asarray([m[r] for r in ids], np.float64)
    assert np.isfinite(v).all() and len(v) == N_ROWS
    dfd = pd.read_csv(out, encoding="utf-8-sig")
    md = dict(zip(dfd["row_id"].tolist(),
                  dfd["control_success"].to_numpy(np.float64).tolist()))
    vd = np.asarray([md[r] for r in ids], np.float64)
    INFO[f"cal_{tag}"]["csv_default_vs_roundtrip_max_abs"] = float(np.abs(v - vd).max())
    INFO[f"cal_{tag}"]["out_sha256"] = sh(out)
    INFO[f"cal_{tag}"]["mean"] = float(v.mean())
    INFO[f"cal_{tag}"]["max_prediction"] = float(v.max())
    INFO[f"cal_{tag}"]["min_prediction"] = float(v.min())
    np.save(f"{WORK}/calp_{tag}{SUF}.npy", v)
    ft = pd.read_csv(f"{d}/mate/output/failtype.csv", encoding="utf-8-sig",
                     float_precision="round_trip")
    assert list(ft.columns) == ["row_id", "lg_mr", "lg_wo"], list(ft.columns)
    ft = ft.set_index("row_id").reindex(ids)
    assert ft.notna().all().all()
    lm = ft["lg_mr"].to_numpy(np.float64)
    lw = ft["lg_wo"].to_numpy(np.float64)
    np.save(f"{WORK}/ftmr_{tag}{SUF}.npy", lm)
    np.save(f"{WORK}/ftwo_{tag}{SUF}.npy", lw)
    INFO[f"cal_{tag}"]["mu_mr_measured"] = repr(float(lm.mean()))
    INFO[f"cal_{tag}"]["mu_wo_measured"] = repr(float(lw.mean()))
    print(f"  cal_{tag}{SUF} mean={v.mean()!r}  max={v.max()!r}  "
          f"mu_mr={lm.mean()!r}  mu_wo={lw.mean()!r}", flush=True)
print(f"{len(TAGS)} 캘 실행 완료 벽시계 {time.time() - t0:.1f}초", flush=True)

p = f"{WORK}/cal{SUF}.json"
try:
    _prev = json.load(open(p, encoding="utf-8"))
except Exception:
    _prev = {}
_prev.update(INFO)
json.dump(_prev, open(p, "w"), ensure_ascii=False, indent=1)
print("\nCAL_OK", p)
