"""exp_113 stage1 — 투수 in-season 분해 확장 열 14개.

왜 투수 쪽인가 (S176): `ins_*` 계열의 값은 타자가 아니라 투수 쪽에 실려 있다.
`ins_p_success` 가 중요도 1·2·6위인데 `ins_b_middle` 은 12위로 한 자릿수 차이다.
exp_103 은 타자 쪽을 파서 0 을 얻었다.

🔴 배포 가능성이 1번 제약 (S170) — 추론에 쓸 수 있는 것은 둘뿐이다:
  ① 그 행 자신의 열 (운영진이 test 에도 주는 `asof_*` 포함)
  ② train 으로 만든 **동결표를 키로 조회한 값**
이 파일은 출하 `mate/inseason.py` 의 구조를 **글자 그대로** 승계한다 —
build_anchor(season < S 행) → apply(행별 lookup + 산술).
정렬·groupby·cumsum·rolling·분위수 연산이 apply 안에 한 줄도 없다.

🔴 라벨을 한 번도 안 쓴다. `control_success` 가 이 파일에 등장하지 않는다
   (exp_103 의 `bat_*_dev` 3열은 라벨 EB 수축 타깃 인코딩이라 별도 논증이 필요했다 — S177).
   따라서 시간축 게이트의 음성 대조는 라벨 난수화가 아니라 **시즌 대역 난수화**로 한다(S197).

🔑 무엇이 새것인가 — 출하 94열이 투수 쪽에서 이미 내놓는 것:
     ins_p_n · ins_p_{success,reverse,middle,ball,strike}(+_dev)   = 11열
   출하되지 않는 것 둘:
     ① **구종 믹스 in-season 분해** — `asof_pitcher_pitchmix_n` 이 NUM 94열에 아예 없고
        (실측: 시즌 경계에서 안 끊기는 커리어 누적, 전이 1,468건 전부 diff==1,
         round(pm·rate) 정수 복원 잔차 max 0.0076) 그 분모로만 만들 수 있다.
     ② **비선형 재조합** — 딥 팔의 첫 은닉층은 수치 입력에 대해 **정확히 선형**이다
        (arch097.TokenMLP: t = x·w + b → concat → Linear). 그래서 기존 열의
        선형결합은 표현력을 한 톨도 안 늘린다. 곱·나눗셈·제곱근만이 새것이다.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

H = os.environ.get("EXP113_HOME", "/home/<user>/SungjinPark")
BASE = f"{H}/exp113"
TRAIN = f"{H}/data/train.csv"
K_EB = 50.0                       # 🔴 출하 inseason.py 와 같은 값

PR5 = ["success", "reverse", "middle", "ball", "strike"]     # 분모 asof_pitcher_n
MIX = ["fastball", "breaking", "offspeed"]                   # 분모 asof_pitcher_pitchmix_n

COLS = [
    # ── 그룹 P : 구종 믹스 in-season 분해 (미출하 성분) ──
    "insp_fb_dev",      # 당시즌 속구 비율(EB) − 커리어 속구 비율
    "insp_br_dev",
    "insp_os_dev",
    "insp_mix_l1",      # 0.5·Σ|dev| — 레퍼토리를 얼마나 바꿨나 (비선형 집계)
    "insp_fb_lvl",      # 당시즌 속구 비율(EB) 자체
    # ── 그룹 R : 출하 in-season 블록의 비선형 재조합 ──
    "insp_w",           # ins_n/(ins_n+50)  — ins_p_* 중 데이터 몫 (사전 몫의 반대)
    "insp_share",       # ins_n/asof_pitcher_n — 커리어 중 당시즌 비중
    "insp_suc_z",       # ins_p_success_dev · sqrt(ins_n) — 신뢰도 가중 편차
    "insp_ball_z",
    "insp_mid_z",
    "insp_rev_z",
    "insp_suc_raw",     # 수축 안 한 당시즌 성공률 k/ins_n
    "insp_suc_cnt",     # 당시즌 성공 개수 k (n·rate 곱 → 비선형)
    "insp_miss",        # 동결표 미적중(신인) 플래그
]
GROUP_P = ["insp_fb_dev", "insp_br_dev", "insp_os_dev", "insp_mix_l1", "insp_fb_lvl"]
GROUP_R = [c for c in COLS if c not in GROUP_P]

# apply_table 이 읽어도 되는 행-로컬 열 (이 목록 밖은 쓰지 않는다)
ROW_COLS = (["pitcher_id", "asof_pitcher_n", "asof_pitcher_pitchmix_n"]
            + [f"asof_pitcher_{r}_rate" for r in PR5]
            + [f"asof_pitcher_{r}_rate" for r in MIX])


def _counts(n, rate):
    """정수 누적 카운트 복원 — 출하 inseason.py::_counts 와 같은 식."""
    return np.round(np.asarray(n, np.float64) * np.nan_to_num(np.asarray(rate, np.float64)))


def build_table(src):
    """🔴 src = season < S 행만. 반환은 키→값 배열 묶음(동결표)."""
    n = src["asof_pitcher_n"].to_numpy(np.float64)
    pm = src["asof_pitcher_pitchmix_n"].to_numpy(np.float64)
    t = pd.DataFrame({"pitcher_id": src["pitcher_id"].to_numpy(), "n": n, "pm": pm})
    for r in PR5:
        t["c_" + r] = _counts(n, src["asof_pitcher_" + r + "_rate"])
    for r in MIX:
        t["m_" + r] = _counts(pm, src["asof_pitcher_" + r + "_rate"])
    li = t.groupby("pitcher_id")["n"].idxmax()
    p = t.loc[li].set_index("pitcher_id")
    # 두 누적 카운터가 같은 행 순서를 타는지 확인 (같은 행이 둘 다의 최대여야 한다)
    li2 = t.groupby("pitcher_id")["pm"].idxmax()
    same_row = bool((li.to_numpy() == li2.to_numpy()).all())
    lg = {r: float(np.nansum(t["c_" + r]) / max(np.nansum(t["n"]), 1)) for r in PR5}
    lgm = {r: float(np.nansum(t["m_" + r]) / max(np.nansum(t["pm"]), 1)) for r in MIX}
    return {"p_ids": p.index.to_numpy(), "p_n": p["n"].to_numpy(),
            "p_pm": p["pm"].to_numpy(),
            "p_c": np.stack([p["c_" + r].to_numpy() for r in PR5], 1),
            "p_m": np.stack([p["m_" + r].to_numpy() for r in MIX], 1),
            "lg": np.array([lg[r] for r in PR5]),
            "lgm": np.array([lgm[r] for r in MIX]),
            "_anchor_row_same_for_n_and_pm": same_row}


def apply_table(rows, T):
    """🔴 행마다 자기 열 + 표 조회만. rows 안의 다른 행을 보는 연산 0개."""
    n = rows["asof_pitcher_n"].to_numpy(np.float64)
    pm = rows["asof_pitcher_pitchmix_n"].to_numpy(np.float64)
    pos = pd.Index(T["p_ids"]).get_indexer(rows["pitcher_id"].to_numpy())
    hit = pos >= 0
    safe = np.where(hit, pos, 0)
    an = np.where(hit, T["p_n"][safe], 0.0)          # 미지 투수 = 신인 = 앵커 0
    apm = np.where(hit, T["p_pm"][safe], 0.0)
    ins_n = np.maximum(n - an, 0.0)
    ins_pm = np.maximum(pm - apm, 0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        # ── 그룹 P : 구종 믹스 ──
        dev_mix, lvl_mix = [], []
        for i, r in enumerate(MIX):
            ac = np.where(hit, T["p_m"][safe, i], 0.0)
            prior = np.where(apm > 0, ac / np.maximum(apm, 1.0), T["lgm"][i])
            k = np.maximum(_counts(pm, rows["asof_pitcher_" + r + "_rate"]) - ac, 0.0)
            lvl = (k + K_EB * prior) / (ins_pm + K_EB)
            lvl_mix.append(lvl)
            dev_mix.append(lvl - prior)
        mix_l1 = 0.5 * sum(np.abs(d) for d in dev_mix)

        # ── 그룹 R : 결과 5분해의 비선형 재조합 ──
        dev_out, k_out = {}, {}
        for i, r in enumerate(PR5):
            ac = np.where(hit, T["p_c"][safe, i], 0.0)
            prior = np.where(an > 0, ac / np.maximum(an, 1.0), T["lg"][i])
            k = np.maximum(_counts(n, rows["asof_pitcher_" + r + "_rate"]) - ac, 0.0)
            dev_out[r] = (k + K_EB * prior) / (ins_n + K_EB) - prior
            k_out[r] = k
        rt = np.sqrt(ins_n)
        raw_suc = np.where(ins_n >= 1, k_out["success"] / np.maximum(ins_n, 1.0), np.nan)

    F = np.column_stack([
        dev_mix[0], dev_mix[1], dev_mix[2], mix_l1, lvl_mix[0],
        ins_n / (ins_n + K_EB),
        ins_n / np.maximum(n, 1.0),
        dev_out["success"] * rt, dev_out["ball"] * rt,
        dev_out["middle"] * rt, dev_out["reverse"] * rt,
        raw_suc, k_out["success"],
        (~hit).astype(np.float64)])
    assert F.shape[1] == len(COLS)
    return F


def build_all(df):
    """시즌마다 season < S 행으로 표를 짓고 season == S 행에 적용."""
    season = df["season"].to_numpy(np.int32)
    F = np.full((len(df), len(COLS)), np.nan, dtype=np.float64)
    info = {}
    for S in sorted(set(season.tolist())):
        tgt, src = season == S, season < S
        if not src.any():
            F[tgt, COLS.index("insp_miss")] = 1.0
            info[int(S)] = {"src_rows": 0, "note": "앞선 시즌 없음 — 전 열 NaN"}
            continue
        T = build_table(df[src])
        assert int(df["season"].to_numpy()[src].max()) < S      # 기계 검사 (S197)
        F[tgt] = apply_table(df.loc[tgt, ROW_COLS], T)
        info[int(S)] = {"src_rows": int(src.sum()), "tgt_rows": int(tgt.sum()),
                        "n_pitchers_in_table": int(len(T["p_ids"])),
                        "miss_frac": float(F[tgt, COLS.index("insp_miss")].mean()),
                        "anchor_row_same_for_n_and_pm": T["_anchor_row_same_for_n_and_pm"]}
    return F, info


if __name__ == "__main__":
    os.makedirs(f"{BASE}/work", exist_ok=True)
    os.makedirs(f"{BASE}/out", exist_ok=True)
    df = pd.read_csv(TRAIN, usecols=["season"] + ROW_COLS, encoding="utf-8-sig")
    F, info = build_all(df)
    season = df["season"].to_numpy(np.int32)
    np.save(f"{BASE}/work/pfeat.npy", F.astype(np.float32))
    json.dump({"cols": COLS, "group_P": GROUP_P, "group_R": GROUP_R},
              open(f"{BASE}/work/pfeat_cols.json", "w"))
    rep = {"exp_id": "exp_113", "stage": "f1_feats", "n_rows": int(len(df)),
           "cols": COLS, "group_P": GROUP_P, "group_R": GROUP_R, "eb_k": K_EB,
           "라벨_사용": "없음 — control_success 를 읽지 않는다",
           "per_season": info,
           "nan_frac_2024": {c: float(np.isnan(F[season == 2024, i]).mean())
                             for i, c in enumerate(COLS)},
           "sd_2024": {c: float(np.nanstd(F[season == 2024, i]))
                       for i, c in enumerate(COLS)},
           "mean_2024": {c: float(np.nanmean(F[season == 2024, i]))
                         for i, c in enumerate(COLS)}}
    json.dump(rep, open(f"{BASE}/out/f1_feats.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False, indent=1))
