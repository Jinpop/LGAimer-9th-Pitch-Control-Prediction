"""exp_067 stage 0b — **실제 대회 표본 행**으로 같은 게이트를 다시 돈다.

🔴 g0_frame.py 는 `mate/data/test.csv` 를 읽었는데 그것은 대역(스탠드인) 프레임이다
(row_id 가 TRAIN_1221586.. 로 시작 = train 2024 행을 2025 로 덮은 것).
CLAUDE.md 가 명시한 사고 그대로 — 대역에서는 시즌-내 폼 계열이 전부 축퇴한다.
따라서 **커버리지 판정은 대역으로 하지 않는다.** 여기서는 `data/test.csv` (대회가
공개한 실제 표본 5행) 로 ① 99열 생성 ② ins_*/sp_* 발동을 확인한다.
"""
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
OUT = os.path.expanduser("~/SungjinPark/exp067/out")
sys.path.insert(0, MATE)

import pipeline as P        # noqa: E402
import splits               # noqa: E402
import inseason             # noqa: E402

CATS = ["base_state", "game_type", "pitcher_team_id", "batter_team_id",
        "cnt_state"]


def build(test):
    feats = P.apply_features(test, TDIR)
    ratios = P.apply_ratio_features(test, feats)
    df = pd.concat([test.reset_index(drop=True), feats.reset_index(drop=True),
                    ratios.reset_index(drop=True)], axis=1)
    df["cnt_state"] = df.balls_before.astype(int) * 3 + df.strikes_before.astype(int)
    S = splits.apply(df, splits.load(f"{MDIR}/deploy_splits.npz"))
    for c in S.columns:
        df[c] = S[c].values
    I = inseason.apply(df, inseason.load(f"{MDIR}/anchor.npz"))
    for c in I.columns:
        df[c] = I[c].values
    return df


def main():
    R = {"stage": "g0b_real5"}
    FSP = [c for c in open(f"{MDIR}/features_sp.txt").read().split("\n") if c]
    real = pd.read_csv(os.path.expanduser("~/SungjinPark/data/test.csv"),
                       encoding="utf-8-sig")
    R["n_real_rows"] = len(real)
    R["real_row_ids"] = real["row_id"].tolist()
    t0 = time.time()
    df = build(real)
    R["build_s"] = round(time.time() - t0, 3)

    miss = [c for c in FSP if c not in df.columns]
    R["Q1_all_99_present_on_real_rows"] = (len(miss) == 0)
    R["Q2_missing"] = miss
    R["cnt_state_present"] = "cnt_state" in df.columns
    R["cnt_state_in_FSP"] = "cnt_state" in FSP
    R["n_model_inputs"] = len(set(FSP) | set(CATS))

    ins = [c for c in df.columns if c.startswith("ins_")]
    sp = [c for c in df.columns if c.startswith("sp_")]
    R["ins_cols"] = len(ins)
    R["sp_cols"] = len(sp)
    R["ins_p_n"] = df["ins_p_n"].tolist()
    R["has_ins_form"] = int((df["ins_p_n"] > 0).sum())
    R["ins_b_n"] = df["ins_b_n"].tolist()
    R["has_ins_bform"] = int((df["ins_b_n"] > 0).sum())
    R["ins_p_success_dev"] = df["ins_p_success_dev"].tolist()
    R["sp_pit_bhand"] = df["sp_pit_bhand"].tolist()
    R["sp_nonzero_cells"] = int((df[sp].fillna(0.0).abs() > 0).sum().sum())
    R["p6_fastball"] = df["p6_fastball"].tolist()
    R["nan_cols_on_real"] = {c: int(df[c].isna().sum())
                             for c in FSP if df[c].isna().any()}
    R["asof_pitcher_n"] = real["asof_pitcher_n"].tolist()

    # 같은 절차를 대역 프레임에서 돌린 결과와 나란히 (축퇴 대조)
    band = json.load(open(f"{OUT}/g0_frame.json"))
    R["band_contrast"] = {
        "band_ins_p_n_zero_rows": band["coverage"]["ins_p_n_zero_rows"],
        "band_n_rows": band["n_test"],
        "band_ins_p_n_mean": band["coverage"]["ins_p_n_mean"],
        "note": "대역 프레임은 앵커(<=2024)보다 앞선 행이라 ins_* 가 100% 축퇴한다. "
                "실제 2025 행에서는 발동한다 — 커버리지 판정은 실제 표본으로만."}

    json.dump(R, open(f"{OUT}/g0_real5.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(R, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
