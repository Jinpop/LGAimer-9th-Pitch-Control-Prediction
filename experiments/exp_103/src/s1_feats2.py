"""exp_103 stage1 (정정판) — 타자 폼 열 17개.

🔴🔴 규칙 2-4 정정 반영 (코디네이터 지시 2026-08-26)
  추론 시점에 쓸 수 있는 것은 **둘뿐**이다:
    ① 그 행 자신의 열 (운영진이 test 에도 주는 asof_batter_* 포함)
    ② train 으로 만든 **동결표를 키로 조회한 값**
  다른 test 행을 훑는 계산은 어떤 형태로도 없다.
  → `bat_prevK_*`(직전 K개 투구 창)·`bat_prevseason_*` 계열은 **폐기**했다.

이 파일은 계산을 두 함수로 **구조적으로 분리**한다:
  build_table(src)  : season < S 행만 본다. 결과는 **키→값 사전** 다섯 개뿐.
  apply_table(rows) : 행마다 자기 열 8개 + 사전 조회만 쓴다.
      🔴 rows 안의 다른 행을 참조하는 연산이 한 줄도 없다 (groupby·cumsum·rolling·정렬 0개).
      이 성질은 s1_rowindep.py 의 게이트가 기계로 증명한다.

🔑 차분 기전 — 이 설계의 핵심
  stage0 이 증명했다: asof_batter_n·asof_batter_success_rate 는 **통산 shift(1)** 이고,
  Sc = round(n*r) 로 통산 누적 성공 개수가 정확히 복원된다(잔차 max 0.0069).
  동결 시점의 (n0, Sc0) 을 표에서 읽으면
        (Sc − Sc0) / (n − n0)  =  **동결 이후 그 타자가 상대한 투구들의 성공률**
  이 값은 test 안에서 **행마다 갱신된다** — 운영진 열이 갱신되기 때문이다.
  동결 스냅샷을 그대로 붙이는 것과 달리 낡지 않는다. 그러면서 다른 test 행은 안 본다.

🔴 결측은 NaN. sentinel(0/-1/평균) 로 채우지 않는다. 플래그 열 2개 별도.
"""
import json, os, sys
import numpy as np
import pandas as pd

H = os.environ.get("EXP103_HOME", "/home/<user>/SungjinPark")
BASE = f"{H}/exp103"
TRAIN = f"{H}/data/train.csv"
OUT = f"{BASE}/work"
os.makedirs(OUT, exist_ok=True)
EB_K = 200.0

COLS = ["bat_cs_n", "bat_n_growth", "bat_cs_rate", "bat_cs_rate_eb",
        "bat_cs_dev", "bat_cs_dev_eb", "bat_form_dev",
        "bat_cs_mid_rate", "bat_cs_mid_dev", "bat_form_dev_mid",
        "bat_frozen_n0", "bat_frozen_rate",
        "bat_phand_dev", "bat_count_dev", "bat_base_dev",
        "bat_form_miss", "bat_cs_short"]

# apply_table 이 읽어도 되는 행-로컬 열 (이 목록 밖은 쓰지 않는다)
ROW_COLS = ["batter_id", "pitcher_hand", "balls_before", "strikes_before",
            "base_state", "asof_batter_n", "asof_batter_success_rate",
            "asof_batter_middle_rate"]

BASE_STATES = ["___", "1__", "_2_", "__3", "12_", "1_3", "_23", "123"]
BS_CODE = {s: i for i, s in enumerate(BASE_STATES)}


def _recover(n, r, m):
    """운영진 통산 rate 에서 통산 누적 개수를 복원한다 (행마다 독립)."""
    Sc = np.round(n * np.nan_to_num(r, nan=0.0))
    Mc = np.round(n * np.nan_to_num(m, nan=0.0))
    return Sc, Mc


def build_table(src):
    """🔴 src = season < S 행만. 반환은 전부 **키 → 값** 사전이다."""
    b = src["batter_id"].to_numpy(np.int64)
    y = src["control_success"].to_numpy(np.float64)
    n = src["asof_batter_n"].to_numpy(np.float64)
    r = src["asof_batter_success_rate"].to_numpy(np.float64)
    m = src["asof_batter_middle_rate"].to_numpy(np.float64)
    Sc, Mc = _recover(n, r, m)

    # 타자별 **마지막** pre-S 행 = 동결 지점. (파일 순서 = 시간순, stage0 증명)
    last = pd.DataFrame({"b": b, "i": np.arange(len(b))}).groupby("b")["i"].max()
    li = last.to_numpy()
    keys = last.index.to_numpy()
    n0 = n[li] + 1.0                 # 그 마지막 투구까지 포함한 통산 개수
    Sc0 = Sc[li] + y[li]
    nm0 = n[li]                      # middle 은 라벨이 없어 한 투구 뒤진다(문서화된 근사)
    Mc0 = Mc[li]
    r0 = Sc0 / np.maximum(n0, 1.0)
    m0 = np.where(nm0 > 0, Mc0 / np.maximum(nm0, 1.0), np.nan)

    T = {
        "n0": pd.Series(n0, index=keys), "Sc0": pd.Series(Sc0, index=keys),
        "nm0": pd.Series(nm0, index=keys), "Mc0": pd.Series(Mc0, index=keys),
        "r0": pd.Series(r0, index=keys), "m0": pd.Series(m0, index=keys),
    }
    r0map = T["r0"]

    def eb_sub(keycols, name):
        g = pd.DataFrame(dict(b=b, k=keycols, y=y)).groupby(["b", "k"])["y"].agg(["sum", "size"])
        pri = r0map.reindex(g.index.get_level_values(0)).to_numpy()
        val = (g["sum"].to_numpy() + EB_K * pri) / (g["size"].to_numpy() + EB_K)
        T[name] = pd.Series(val - pri, index=g.index)      # 동결 통산 대비 편차로 저장

    eb_sub(src["pitcher_hand"].to_numpy(np.int64), "phand_dev")
    eb_sub(src["balls_before"].to_numpy(np.int64) * 10
           + src["strikes_before"].to_numpy(np.int64), "count_dev")
    eb_sub(src["base_state"].map(BS_CODE).fillna(-1).to_numpy(np.int64), "base_dev")
    return T


def apply_table(rows, T):
    """🔴 행마다 자기 열 + 사전 조회만. rows 안의 다른 행을 보는 연산 0개."""
    b = rows["batter_id"].to_numpy(np.int64)
    n = rows["asof_batter_n"].to_numpy(np.float64)
    r = rows["asof_batter_success_rate"].to_numpy(np.float64)
    m = rows["asof_batter_middle_rate"].to_numpy(np.float64)
    Sc, Mc = _recover(n, r, m)
    idx = pd.Index(b)
    n0 = T["n0"].reindex(idx).to_numpy()
    Sc0 = T["Sc0"].reindex(idx).to_numpy()
    nm0 = T["nm0"].reindex(idx).to_numpy()
    Mc0 = T["Mc0"].reindex(idx).to_numpy()
    r0 = T["r0"].reindex(idx).to_numpy()
    m0 = T["m0"].reindex(idx).to_numpy()
    miss = ~np.isfinite(n0)

    with np.errstate(invalid="ignore", divide="ignore"):
        cs_n = n - n0
        ok = np.isfinite(cs_n) & (cs_n >= 1)
        cs_rate = np.where(ok, (Sc - Sc0) / np.where(ok, cs_n, 1.0), np.nan)
        cs_rate_eb = np.where(np.isfinite(cs_n),
                              (Sc - Sc0 + EB_K * r0) / (np.nan_to_num(cs_n) + EB_K),
                              np.nan)
        mid_n = n - nm0
        okm = np.isfinite(mid_n) & (mid_n >= 1)
        cs_mid = np.where(okm, (Mc - Mc0) / np.where(okm, mid_n, 1.0), np.nan)
        growth = np.where(np.isfinite(n0) & (n0 > 0), n / n0, np.nan)

    mi = pd.MultiIndex.from_arrays([b, rows["pitcher_hand"].to_numpy(np.int64)])
    phd = T["phand_dev"].reindex(mi).to_numpy()
    mi = pd.MultiIndex.from_arrays([b, rows["balls_before"].to_numpy(np.int64) * 10
                                    + rows["strikes_before"].to_numpy(np.int64)])
    cnd = T["count_dev"].reindex(mi).to_numpy()
    mi = pd.MultiIndex.from_arrays([b, rows["base_state"].map(BS_CODE)
                                    .fillna(-1).to_numpy(np.int64)])
    bsd = T["base_dev"].reindex(mi).to_numpy()

    F = np.column_stack([
        cs_n, growth, cs_rate, cs_rate_eb,
        cs_rate - r0, cs_rate_eb - r0, r - r0,
        cs_mid, cs_mid - m0, m - m0,
        n0, r0, phd, cnd, bsd,
        miss.astype(np.float64),
        np.where(np.isfinite(cs_n), (cs_n < 20).astype(np.float64), 1.0)])
    assert F.shape[1] == len(COLS)
    return F


if __name__ == "__main__":
    df = pd.read_csv(TRAIN, usecols=["season", "control_success"] + ROW_COLS,
                     encoding="utf-8-sig")
    N = len(df)
    season = df["season"].to_numpy(np.int32)
    F = np.full((N, len(COLS)), np.nan, dtype=np.float64)
    rep = {"n_rows": int(N), "cols": COLS, "eb_k": EB_K,
           "폐기": "bat_prevK_* 창 계열·bat_prevseason_* — 배포에서 다른 test 행을 훑지 않고는 못 만든다"}
    for S in sorted(set(season.tolist())):
        tgt = season == S
        src = season < S
        if not src.any():
            F[tgt, COLS.index("bat_form_miss")] = 1.0
            F[tgt, COLS.index("bat_cs_short")] = 1.0
            continue
        T = build_table(df[src])
        F[tgt] = apply_table(df[tgt][ROW_COLS], T)
    np.save(f"{OUT}/batform2.npy", F.astype(np.float32))
    json.dump({"cols": COLS}, open(f"{OUT}/batform2_cols.json", "w"))
    rep["nan_frac_2024"] = {c: float(np.isnan(F[season == 2024, i]).mean())
                            for i, c in enumerate(COLS)}
    rep["sd_2024"] = {c: float(np.nanstd(F[season == 2024, i]))
                      for i, c in enumerate(COLS)}
    json.dump(rep, open(f"{BASE}/out/s1_feats2.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False, indent=1))
