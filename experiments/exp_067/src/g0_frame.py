"""exp_067 stage 0 — 출하 실현 가능성 게이트.

Q1 NN 99열이 **추론 시점 test 행에서** 실제로 만들어지는가.
Q2 빠진 열이 있는가. 있으면 몇 개이고 어느 모듈 소관인가.
Q4 추론 시간(프레임 조립 + NN 4시드 전방향 전파).

방법: 출하 `mate/v6_infer.py::run` 이 df 를 만드는 절차를 **그대로** 재현한다
(같은 모듈·같은 인자·같은 순서). 새 코드를 쓰지 않고 출하 모듈만 호출한다.
build_BL/pkg_BL3 는 읽기 전용 — 아무것도 쓰지 않는다.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

PKG = os.path.expanduser("~/SungjinPark/build_BL/pkg_BL3")
MATE = f"{PKG}/mate"
MDIR = f"{MATE}/model"
TDIR = f"{MDIR}/tables"
DDIR = f"{MATE}/data"
OUT = os.path.expanduser("~/SungjinPark/exp067/out")
sys.path.insert(0, MATE)

import pipeline as P        # noqa: E402
import splits               # noqa: E402
import inseason             # noqa: E402

CATS = ["base_state", "game_type", "pitcher_team_id", "batter_team_id",
        "cnt_state"]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    R = {"stage": "g0", "t": {}}
    t_all = time.time()

    # ── 소스 파일 sha (출하본 == 우리가 학습에 쓴 것) ────────────────
    R["sha"] = {p: sha(f"{MATE}/{p}") for p in
                ("pipeline.py", "splits.py", "inseason.py", "nn_infer.py")}
    R["sha"]["model/features_sp.txt"] = sha(f"{MDIR}/features_sp.txt")
    R["sha"]["model/anchor.npz"] = sha(f"{MDIR}/anchor.npz")
    R["sha"]["model/deploy_splits.npz"] = sha(f"{MDIR}/deploy_splits.npz")
    for t in ("pitchmodel6.txt", "pmcat6.json", "anom_cnt.csv"):
        R["sha"][f"model/tables/{t}"] = sha(f"{TDIR}/{t}")

    FSP = [c for c in open(f"{MDIR}/features_sp.txt").read().split("\n") if c]
    assert len(FSP) == 99, len(FSP)
    R["n_features_sp"] = len(FSP)

    # ── 출하 v6_infer.run 의 프레임 조립 절차를 그대로 ────────────────
    t0 = time.time()
    test = pd.read_csv(f"{DDIR}/test.csv", encoding="utf-8-sig")
    R["t"]["read_test_s"] = round(time.time() - t0, 2)
    R["n_test"] = len(test)
    raw_cols = list(test.columns)

    t0 = time.time()
    feats = P.apply_features(test, TDIR)
    R["t"]["apply_features_s"] = round(time.time() - t0, 2)
    f_cols = list(feats.columns)

    t0 = time.time()
    ratios = P.apply_ratio_features(test, feats)
    R["t"]["apply_ratio_features_s"] = round(time.time() - t0, 2)
    r_cols = list(ratios.columns)

    df = pd.concat([test.reset_index(drop=True), feats.reset_index(drop=True),
                    ratios.reset_index(drop=True)], axis=1)
    df["cnt_state"] = df.balls_before.astype(int) * 3 + df.strikes_before.astype(int)

    t0 = time.time()
    S = splits.apply(df, splits.load(f"{MDIR}/deploy_splits.npz"))
    for c in S.columns:
        df[c] = S[c].values
    R["t"]["splits_s"] = round(time.time() - t0, 2)
    s_cols = list(S.columns)

    t0 = time.time()
    I = inseason.apply(df, inseason.load(f"{MDIR}/anchor.npz"))
    for c in I.columns:
        df[c] = I[c].values
    R["t"]["inseason_s"] = round(time.time() - t0, 2)
    i_cols = list(I.columns)

    # ── Q1/Q2 — 99열 전량 대조 + 산출 모듈 귀속 ─────────────────────
    def src_of(c):
        hits = []
        if c in i_cols:
            hits.append("inseason.apply")
        if c in s_cols:
            hits.append("splits.apply")
        if c == "cnt_state":
            hits.append("v6_infer(cnt_state)")
        if c in r_cols:
            hits.append("pipeline.apply_ratio_features")
        if c in f_cols:
            hits.append("pipeline.apply_features")
        if c in raw_cols:
            hits.append("test.csv(raw)")
        return hits

    attrib, missing, ambiguous = {}, [], []
    for c in FSP:
        h = src_of(c)
        attrib[c] = h
        if not h:
            missing.append(c)
        if len(h) > 1:
            ambiguous.append({c: h})
    R["Q1_all_99_present"] = (len(missing) == 0)
    R["Q2_missing"] = missing
    R["Q2_n_missing"] = len(missing)
    R["attrib_ambiguous"] = ambiguous
    tally = {}
    for c, h in attrib.items():
        k = h[0] if h else "MISSING"
        tally[k] = tally.get(k, 0) + 1
    R["attrib_tally"] = tally
    R["attrib"] = attrib

    # ── 결측·축퇴 계측 (대역 프레임 사고 방지: 실제 test 행이다) ──────
    sub = df[FSP]
    nan = sub.isna().sum()
    R["nan_counts_nonzero"] = {k: int(v) for k, v in nan.items() if v > 0}
    R["nan_total_cols"] = int((nan > 0).sum())
    R["coverage"] = {
        "ins_p_n_zero_rows": int((df["ins_p_n"] == 0).sum()),
        "ins_p_n_mean": float(df["ins_p_n"].mean()),
        "ins_p_n_max": float(df["ins_p_n"].max()),
        "ins_p_success_dev_absmean": float(df["ins_p_success_dev"].abs().mean()),
        "sp_pit_bhand_nan": int(df["sp_pit_bhand"].isna().sum()),
        "sp_pit_bhand_nonzero": int((df["sp_pit_bhand"].fillna(0) != 0).sum()),
        "sp_pit_bhand_std": float(df["sp_pit_bhand"].std()),
        "p6_fastball_mean": float(df["p6_fastball"].mean()),
        "p6_fastball_std": float(df["p6_fastball"].std()),
        "i3_anom_l1_mean": float(df["i3_anom_l1"].mean()),
        "season_unique": sorted(df["season"].unique().tolist()),
        "game_type_counts": {str(k): int(v) for k, v in
                             df["game_type"].value_counts().items()},
    }
    # 범주 5열의 test 값이 학습 레벨에 있는가 (미등재는 index 0 = 미지)
    R["cat_nunique_test"] = {c: int(df[c].astype(str).nunique()) for c in CATS}

    # ── Q4 준비: dtype (표준화 입력이 float 로 떨어지는가) ────────────
    DENSE = [c for c in FSP if c not in CATS]
    bad = {c: str(df[c].dtype) for c in DENSE
           if not np.issubdtype(df[c].dtype, np.number)}
    R["nonnumeric_dense"] = bad
    R["n_dense"] = len(DENSE)

    # ── 프레임을 캐시에 남겨 stage 3 추론시간 측정에 재사용 ───────────
    keep = FSP + ["row_id"]
    t0 = time.time()
    df[keep].to_parquet(os.path.expanduser(
        "~/SungjinPark/exp067/cache/test_frame99.parquet"), index=False)
    R["t"]["cache_write_s"] = round(time.time() - t0, 2)
    R["t"]["total_s"] = round(time.time() - t_all, 2)

    os.makedirs(OUT, exist_ok=True)
    json.dump(R, open(f"{OUT}/g0_frame.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in R.items()
                      if k not in ("attrib", "sha")}, indent=1,
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
