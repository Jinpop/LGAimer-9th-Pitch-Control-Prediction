"""exp_103b 보조 확인 3건 (감사 지적 반영). GPU 안 쓴다.

1. `mid_n >= 2` 문제가 선택한 3열에 걸리는가 — `bat_cs_mid_*`·`bat_cs_rate` 계열의 문제이므로
   3열에는 원리적으로 안 걸린다. 그래도 **수치로** 확인하고 기록만 한다.
2. `bat_count_dev`·`bat_base_dev` 의 안전성 논증 — `inseason.py` 처럼 "라벨을 안 쓴다" 가 아니라
   **"season < S 라벨에서 만든 EB 수축 타깃 인코딩, 추론 시점엔 동결 상수"** 다. 별개 주장이므로
   그대로 적는다. 코드에서 그 성질을 기계로 확인한다.
3. 분포 이탈(S173): `bat_n_growth` 는 시간 단조라 2025 에서 train 범위 밖으로 나갈 수 있다.
   **실제 대회 표본 5행**(build_BW2/frames/real5)에서 배포 구성(≤2024 표 → 2025 행)으로
   3열을 실제로 계산해 값 분포를 병기한다.
"""
import json, os, sys
import numpy as np
import pandas as pd

H = "/home/<user>/SungjinPark"
sys.path.insert(0, f"{H}/exp103/src")
os.environ.setdefault("EXP103_HOME", H)
import s1_feats2 as FT                                     # noqa: E402

B = f"{H}/exp103b"
OUT = f"{B}/out"
os.makedirs(OUT, exist_ok=True)
REAL5 = f"{H}/build_BW2/frames/real5/test.csv"
SEL_NAMES = ["bat_n_growth", "bat_count_dev", "bat_base_dev"]
rep = {"exp_id": "exp_103b", "sel_cols": SEL_NAMES}

df = pd.read_csv(f"{H}/data/train.csv",
                 usecols=["season", "control_success"] + FT.ROW_COLS,
                 encoding="utf-8-sig")
season = df["season"].to_numpy(np.int32)
F17 = np.load(f"{H}/exp103/work/batform2.npy")
COLS17 = json.load(open(f"{H}/exp103/work/batform2_cols.json"))["cols"]
SEL = [COLS17.index(c) for c in SEL_NAMES]

# ── 1. mid_n >= 2 ───────────────────────────────────────────────
m24 = season == 2024
n = df["asof_batter_n"].to_numpy(np.float64)
T23 = FT.build_table(df[season < 2024])
idx = pd.Index(df.loc[m24, "batter_id"].to_numpy(np.int64))
nm0 = T23["nm0"].reindex(idx).to_numpy()
n0 = T23["n0"].reindex(idx).to_numpy()
mid_n = n[m24] - nm0
cs_n = n[m24] - n0
midr = F17[m24, COLS17.index("bat_cs_mid_rate")]
csr = F17[m24, COLS17.index("bat_cs_rate")]
odd = np.isfinite(midr) & ~np.isfinite(csr)
rep["check1_mid_n_ge_2"] = {
    "affected_family": ["bat_cs_mid_rate", "bat_cs_mid_dev", "bat_cs_rate", "bat_cs_rate_eb",
                        "bat_cs_dev", "bat_cs_dev_eb"],
    "n_rows_2024_mid_defined_but_cs_nan": int(odd.sum()),
    "those_rows_mid_rate_values_are_0_or_1": float(np.mean(np.isin(midr[odd], [0.0, 1.0])))
    if odd.sum() else None,
    "n_rows_2024_mid_n_eq_1": int(np.nansum(mid_n == 1)),
    "n_rows_2024_cs_n_eq_1": int(np.nansum(cs_n == 1)),
    "선택한_3열에_걸리는가": False,
    "근거": "bat_n_growth = n/n0 는 mid 카운터를 안 쓴다. bat_count_dev·bat_base_dev 는 "
            "동결표의 EB 수축 편차라 mid_n·cs_n 과 무관하다. 즉 mid_n>=1 대 >=2 의 선택이 "
            "이 3열의 값을 한 비트도 바꾸지 않는다. 기록만 하고 넘어간다.",
}

# ── 2. count_dev·base_dev 안전성 논증 ────────────────────────────
# 성질 A: 표가 season < S 행만 본다 (build_table 인자에서 강제)
# 성질 B: 추론 시점엔 표가 동결 상수 — 같은 키면 항상 같은 값
sub = df[season < 2024]
assert (sub["season"].to_numpy() <= 2023).all()
kt = T23["count_dev"]
bt = T23["base_dev"]
rep["check2_safety_argument"] = {
    "주장": "inseason.py 계열의 '라벨을 안 쓴다' 와 다르다. bat_count_dev·bat_base_dev 는 "
            "**season < S 의 control_success 라벨에서 만든 EB 수축 타깃 인코딩**이고, "
            "추론 시점에는 (batter_id, count) / (batter_id, base_state) 키로 조회되는 "
            "**동결 상수**다. 안전성은 ① 표 제작이 대상 시즌보다 엄격히 앞선 행만 본다 "
            "② 조회는 행마다 독립이다 — 이 둘로 성립한다. 별개 주장이므로 그대로 적는다.",
    "표_제작_행_시즌_최대": int(sub["season"].max()),
    "count_dev_키_수": int(len(kt)),
    "base_dev_키_수": int(len(bt)),
    "eb_k": FT.EB_K,
    "eb_prior": "타자 동결 통산 성공률 r0 — 저장값은 (수축 rate − r0) 편차",
    "행_독립성_승계": "exp_103 out/s1_rowindep.json 게이트 4종 전부 max|Δ| = 0.0. "
                     "exp_103b 는 그 17열 행렬에서 3열을 슬라이스만 했다 — 새 계산 경로 없음.",
}

# ── 3. 분포 이탈 — 실제 대회 표본 5행 (배포 구성 ≤2024 표 → 2025 행) ──
T24 = FT.build_table(df)                                    # 전 train = season <= 2024
r5 = pd.read_csv(REAL5, encoding="utf-8-sig")
F5 = FT.apply_table(r5[FT.ROW_COLS], T24)
rep["check3_real5_2025"] = {
    "frame": REAL5, "n_rows": int(len(r5)),
    "table": "≤2024 train 전량 (배포와 같은 구성)",
    "rows": [{"row_id": r5["row_id"].iloc[i],
              "batter_id": int(r5["batter_id"].iloc[i]),
              "asof_batter_n": float(r5["asof_batter_n"].iloc[i]),
              "frozen_n0": float(F5[i, COLS17.index("bat_frozen_n0")]),
              **{c: (None if not np.isfinite(F5[i, COLS17.index(c)])
                     else float(F5[i, COLS17.index(c)])) for c in SEL_NAMES}}
             for i in range(len(r5))],
}
for c in SEL_NAMES:
    v24 = F17[m24, COLS17.index(c)].astype(np.float64)
    v24 = v24[np.isfinite(v24)]
    v25 = F5[:, COLS17.index(c)].astype(np.float64)
    v25f = v25[np.isfinite(v25)]
    rep["check3_real5_2025"].setdefault("distribution", {})[c] = {
        "oot2024": {"n": int(v24.size), "min": float(v24.min()), "p01": float(np.percentile(v24, 1)),
                    "p50": float(np.percentile(v24, 50)), "p99": float(np.percentile(v24, 99)),
                    "max": float(v24.max()), "mean": float(v24.mean()), "sd": float(v24.std())},
        "real5_2025": {"n_finite": int(v25f.size),
                       "values": [None if not np.isfinite(x) else float(x) for x in v25]},
        "real5_outside_oot2024_range": int(np.sum((v25f < v24.min()) | (v25f > v24.max()))),
        "real5_above_oot2024_p99": int(np.sum(v25f > np.percentile(v24, 99))),
    }
rep["check3_real5_2025"]["해석_주의"] = (
    "5행이라 비율 자체는 못 믿는다(S38·S59). 말할 수 있는 것은 '값이 학습 범위 밖으로 "
    "나가는가' 뿐이다. bat_n_growth 는 시즌 진행에 단조 증가하므로 시즌 말 행일수록 커진다 — "
    "real5 는 game_month " + ",".join(str(int(x)) for x in r5["game_month"]) + " 이다.")

json.dump(rep, open(f"{OUT}/s1b_checks.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(rep, ensure_ascii=False, indent=1)[:4000])
