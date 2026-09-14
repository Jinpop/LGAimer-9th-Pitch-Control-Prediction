"""exp_103 stage1 — 타자 폼 열 19개 생성.

🔴 리크 설계 (stage0 에서 증명된 사실 위에 세운다)
  · train.csv 행 순서 = 전역 시간순 (asof 통산 카운터가 792/830 그룹 전부에서 정확히 +1)
  · asof_batter_n            = **통산** shift(1) 투구 수 (100.00% 재현)
  · asof_batter_success_rate = **통산** shift(1) control_success 평균 (max|Δ| 5.0e-07 = 6자리 반올림 한계)
  → 따라서 행의 (n, r, m) 에서 **통산 누적 개수**를 정확히 복원할 수 있다:
        Sc = round(n*r)   Mc = round(n*m)
     이 둘은 **현재 행의 라벨을 포함하지 않는다**(운영진이 shift(1) 로 준 값이므로).

  동결표: 목표 시즌 S 에 대해 **season < S 행만** 으로 만든다.
     · 배포는 S=2025, 표는 ≤2024 → test 행에 batter_id 키로 붙인다 (lookup_d 와 같은 기전)
     · 측정은 S=2024, 표는 ≤2023 → 2024 행에 붙인다
     · 학습 행(season s ≤ 2023)에도 **각자의 시즌 시작 동결표**를 붙인다 —
       배포(2025 행에 2024말 표)와 같은 관계를 학습 분포에서도 유지하기 위해서다.
       행별 실시간 창을 학습에 쓰면 배포에서 낡은 값이 들어와 분포가 어긋난다.

  🔑 차분 기전: 동결 (n0, Sc0) 과 행의 실시간 (n, Sc) 의 **차**가
     "이 시즌 안에서 그 타자가 상대한 투구들의 성공률" 이다.
     운영진이 투수에게만 준 prev1/3/5경기 계열의 타자판이고, **test 에서 행마다 갱신된다**
     (test 다른 행을 안 본다 — 운영진 제공 asof 열과 우리 동결표만 쓴다. 규칙 2-4 준수).

  🔴 결측은 NaN 으로 둔다. sentinel(0/-1/평균) 로 채우지 않는다. 별도 플래그 열 2개.
"""
import json, os, sys
import numpy as np
import pandas as pd

H = os.environ.get("EXP103_HOME", "/home/<user>/SungjinPark")
BASE = f"{H}/exp103"
TRAIN = f"{H}/data/train.csv"
OUT = f"{BASE}/work"
os.makedirs(OUT, exist_ok=True)

KS = (50, 200, 1000)
EB_K = 200.0

COLS = ["bat_cs_n", "bat_cs_rate", "bat_cs_dev", "bat_cs_mid_rate", "bat_cs_mid_dev",
        "bat_prev50_rate", "bat_prev200_rate", "bat_prev1000_rate",
        "bat_prev50_dev", "bat_prev200_dev", "bat_prev1000_dev",
        "bat_prevseason_rate", "bat_prevseason_n", "bat_prevseason_dev",
        "bat_frozen_n0", "bat_phand_eb", "bat_cnt_eb",
        "bat_form_miss", "bat_cs_short"]

USE = ["row_id", "season", "pitcher_hand", "balls_before", "strikes_before",
       "batter_id", "asof_batter_n", "asof_batter_success_rate",
       "asof_batter_middle_rate", "control_success"]

df = pd.read_csv(TRAIN, usecols=USE, encoding="utf-8-sig")[USE]
N = len(df)
rep = {"n_rows": int(N), "cols": COLS, "Ks": list(KS), "eb_k": EB_K}

season = df["season"].values.astype(np.int32)
bid = df["batter_id"].values.astype(np.int64)
phand = df["pitcher_hand"].values.astype(np.int64)
balls = df["balls_before"].values.astype(np.int64)
strikes = df["strikes_before"].values.astype(np.int64)
y = df["control_success"].values.astype(np.float64)
n_as = df["asof_batter_n"].values.astype(np.float64)
r_as = df["asof_batter_success_rate"].values.astype(np.float64)
m_as = df["asof_batter_middle_rate"].values.astype(np.float64)

# ── 통산 누적 개수 복원 (반올림이 정확히 되는지 assert) ────────────
prod_s = n_as * np.nan_to_num(r_as, nan=0.0)
prod_m = n_as * np.nan_to_num(m_as, nan=0.0)
Sc = np.round(prod_s)
Mc = np.round(prod_m)
rep["recover_max_resid_success"] = float(np.nanmax(np.abs(prod_s - Sc)))
rep["recover_max_resid_middle"] = float(np.nanmax(np.abs(prod_m - Mc)))
assert rep["recover_max_resid_success"] < 0.05, rep["recover_max_resid_success"]
assert rep["recover_max_resid_middle"] < 0.05, rep["recover_max_resid_middle"]
# 독립 확인: Sc 가 실제 shift(1) 누적 성공 합과 같은가
g = df.groupby("batter_id", sort=False)
Sc_direct = g["control_success"].cumsum().values - y
rep["Sc_exact_match_frac"] = float((Sc == Sc_direct).mean())
assert rep["Sc_exact_match_frac"] == 1.0

SEASONS = sorted(set(season.tolist()))
rep["seasons"] = [int(s) for s in SEASONS]

F = np.full((N, len(COLS)), np.nan, dtype=np.float64)
IDX = {c: i for i, c in enumerate(COLS)}
order = np.arange(N)

for S in SEASONS:
    tgt = season == S
    if not tgt.any():
        continue
    src = season < S                      # 🔴 동결표는 이 시즌 이전 행만 본다
    rows_t = order[tgt]
    if not src.any():
        F[rows_t, IDX["bat_form_miss"]] = 1.0
        F[rows_t, IDX["bat_cs_short"]] = 1.0
        continue

    si = order[src]
    sb = bid[si]
    # 타자별 마지막 pre-S 행 (파일 순서 = 시간순이므로 그 그룹의 마지막 원소)
    lastpos = {}
    for pos, b in zip(si, sb):
        lastpos[b] = pos
    lb = np.array(list(lastpos.keys()), dtype=np.int64)
    lp = np.array([lastpos[b] for b in lb], dtype=np.int64)
    n0 = n_as[lp] + 1.0                    # 그 마지막 투구까지 포함한 통산 개수
    Sc0 = Sc[lp] + y[lp]
    Mc0 = Mc[lp]                           # middle 라벨이 없어 한 투구 뒤짐(문서화된 근사)
    nm0 = n_as[lp]
    r0 = Sc0 / np.maximum(n0, 1.0)

    # 직전 K개 창 · 직전 시즌 · EB 표 — 전부 season < S 행만 쓴다
    ysrc = y[si]
    ord_b = np.argsort(sb, kind="stable")
    sb_s, si_s, y_s = sb[ord_b], si[ord_b], ysrc[ord_b]
    starts = np.searchsorted(sb_s, lb, side="left")
    ends = np.searchsorted(sb_s, lb, side="right")
    csum = np.concatenate([[0.0], np.cumsum(y_s)])
    prevK = {}
    for K in KS:
        st = np.maximum(starts, ends - K)
        cnt = (ends - st).astype(np.float64)
        rate = np.where(cnt > 0, (csum[ends] - csum[st]) / np.maximum(cnt, 1.0), np.nan)
        prevK[K] = rate

    # 직전 시즌
    pm = src & (season == S - 1)
    if pm.any():
        pdf = pd.DataFrame({"b": bid[pm], "y": y[pm]}).groupby("b")["y"].agg(["mean", "size"])
        ps_rate = pdf["mean"].reindex(lb).values
        ps_n = pdf["size"].reindex(lb).values.astype(np.float64)
    else:
        ps_rate = np.full(len(lb), np.nan)
        ps_n = np.full(len(lb), np.nan)

    # EB 표: 타자×투수손 / 타자×(볼,스트라이크)
    r0map = pd.Series(r0, index=lb)
    ph_g = pd.DataFrame({"b": bid[src], "h": phand[src], "y": y[src]}).groupby(["b", "h"])["y"].agg(["sum", "size"])
    ph_prior = r0map.reindex(ph_g.index.get_level_values(0)).values
    ph_eb = (ph_g["sum"].values + EB_K * ph_prior) / (ph_g["size"].values + EB_K)
    ph_tab = pd.Series(ph_eb, index=ph_g.index)

    cs_key = balls[src] * 10 + strikes[src]
    ct_g = pd.DataFrame({"b": bid[src], "c": cs_key, "y": y[src]}).groupby(["b", "c"])["y"].agg(["sum", "size"])
    ct_prior = r0map.reindex(ct_g.index.get_level_values(0)).values
    ct_eb = (ct_g["sum"].values + EB_K * ct_prior) / (ct_g["size"].values + EB_K)
    ct_tab = pd.Series(ct_eb, index=ct_g.index)

    # ── 목표 시즌 행에 붙인다 ──────────────────────────────────
    tb = bid[tgt]
    loc = pd.Index(lb).get_indexer(tb)
    miss = loc < 0
    ok = ~miss
    li = loc[ok]
    rt = rows_t[ok]

    n0r, Sc0r, Mc0r, nm0r, r0r = n0[li], Sc0[li], Mc0[li], nm0[li], r0[li]
    cs_n = n_as[rt] - n0r
    with np.errstate(invalid="ignore", divide="ignore"):
        cs_rate = np.where(cs_n >= 1, (Sc[rt] - Sc0r) / np.maximum(cs_n, 1.0), np.nan)
        cs_mid_n = n_as[rt] - nm0r
        cs_mid = np.where(cs_mid_n >= 1, (Mc[rt] - Mc0r) / np.maximum(cs_mid_n, 1.0), np.nan)

    F[rt, IDX["bat_cs_n"]] = cs_n
    F[rt, IDX["bat_cs_rate"]] = cs_rate
    F[rt, IDX["bat_cs_dev"]] = cs_rate - r_as[rt]
    F[rt, IDX["bat_cs_mid_rate"]] = cs_mid
    F[rt, IDX["bat_cs_mid_dev"]] = cs_mid - m_as[rt]
    for K in KS:
        F[rt, IDX[f"bat_prev{K}_rate"]] = prevK[K][li]
        F[rt, IDX[f"bat_prev{K}_dev"]] = prevK[K][li] - r0r
    F[rt, IDX["bat_prevseason_rate"]] = ps_rate[li]
    F[rt, IDX["bat_prevseason_n"]] = ps_n[li]
    F[rt, IDX["bat_prevseason_dev"]] = ps_rate[li] - r0r
    F[rt, IDX["bat_frozen_n0"]] = n0r
    F[rt, IDX["bat_phand_eb"]] = ph_tab.reindex(
        pd.MultiIndex.from_arrays([tb[ok], phand[rt]])).values
    F[rt, IDX["bat_cnt_eb"]] = ct_tab.reindex(
        pd.MultiIndex.from_arrays([tb[ok], balls[rt] * 10 + strikes[rt]])).values
    F[rows_t, IDX["bat_form_miss"]] = miss.astype(np.float64)
    short = np.ones(len(rows_t))
    short[ok] = (cs_n < 20).astype(np.float64)
    F[rows_t, IDX["bat_cs_short"]] = short

np.save(f"{OUT}/batform.npy", F.astype(np.float32))
json.dump({"cols": COLS}, open(f"{OUT}/batform_cols.json", "w"))

rep["nan_frac_by_col"] = {c: float(np.isnan(F[:, i]).mean()) for c, i in IDX.items()}
rep["nan_frac_2024_by_col"] = {c: float(np.isnan(F[season == 2024, i]).mean())
                               for c, i in IDX.items()}
rep["describe_2024"] = {c: {"mean": float(np.nanmean(F[season == 2024, i])),
                            "sd": float(np.nanstd(F[season == 2024, i]))}
                        for c, i in IDX.items()}
json.dump(rep, open(f"{BASE}/out/s1_feats.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in rep.items() if k != "describe_2024"},
                 ensure_ascii=False, indent=1))
