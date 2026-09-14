"""Q 팔 단독 점수 — 배포 경로(최종 셀 4개 -> 스태커) 그대로.

q_infer.py 의 경로와 같다: seed 디렉터리의 cell_*.cbm 4개를 행에 적용해 p_* 를 만들고,
그 4열을 FEAT 뒤에 붙여 stacker.cbm 을 태운다. OOF 캐시를 안 쓴다(배포는 못 쓴다).

프레임 둘:
  F2024  season==2024 전체 253,507행 — <=2024 학습 팔에게는 **표본 안**이다.
         절대 점수는 부풀지만 세 팔(CPU42·CPU7·GPU42)에 **똑같이** 부풀므로 차이 비교는 성립한다.
  FHELD  (season<=2022) & game_type=='F' — 학습 마스크가 뺀 행. 셀·스태커 **어느 쪽도 안 본 행**.
         2019 행은 stack 열이 아니라 셀 직접 적용이라 문제없다. 진짜 표본 밖.
env: ARMS="이름=경로;이름=경로;..."   OUTJ=결과 json 경로
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import catboost as cb
import feats as FT

REF_BRIER = 0.2498069
P6 = ["p6_curve","p6_cutter","p6_fastball","p6_offspeed","p6_sinker","p6_slider",
      "idea_matchup_pitch","i3_anom_curve","i3_anom_cutter","i3_anom_fastball",
      "i3_anom_offspeed","i3_anom_sinker","i3_anom_slider","i3_anom_l1"]
CATF = ["top_bottom", "game_type", "base_state"]
CELLS = ["is_mid_only", "is_rev_only", "is_both", "is_wayoff_true"]
ARMS = dict(kv.split("=", 1) for kv in os.environ["ARMS"].split(";") if kv)
OUTJ = os.environ.get("OUTJ", "verify/arms.json")
PT = int(os.environ.get("PT", "4"))   # 예측 스레드 — 다른 세션 CPU 를 비켜 잡는다

t0 = time.time()
df = pd.read_parquet("feat3.parquet")
base = [c for c in df.columns if c != FT.TGT and not c.startswith("ins_")
        and c not in P6 and not any(c.startswith(v) for v in FT.FAMILIES.values())]
ins = [c for c in df.columns if c.startswith("ins_")]
FEAT = base + ins + P6
for c in CATF:
    df[c] = df[c].astype(str)
y = df[FT.TGT].to_numpy(np.float64)
ssn = df[FT.SEASON].to_numpy()
gt = df["game_type"].to_numpy()
F2024 = ssn == 2024
FHELD = (ssn <= 2022) & (gt == "F")
X = df[FEAT]
print(f"[load] F2024 {int(F2024.sum()):,} FHELD {int(FHELD.sum()):,} {time.time()-t0:.0f}s", flush=True)

def score(b):
    return 100000.0 * (1.0 - b / REF_BRIER)

P = {}
R = {"ref_brier": REF_BRIER, "n_F2024": int(F2024.sum()), "n_FHELD": int(FHELD.sum()),
     "arms": {}}
for name, d in ARMS.items():
    S = X.copy()
    for c in CELLS:
        m = cb.CatBoostClassifier(); m.load_model(os.path.join(d, f"cell_{c}.cbm"))
        S[f"p_{c}"] = m.predict_proba(X, thread_count=PT)[:, 1]
        del m
    st = cb.CatBoostClassifier(); st.load_model(os.path.join(d, "stacker.cbm"))
    p = st.predict_proba(S, thread_count=PT)[:, 1]
    del st, S
    P[name] = p
    ent = {"dir": d}
    for fn, msk in (("F2024", F2024), ("FHELD", FHELD)):
        b = float(np.mean((p[msk] - y[msk]) ** 2))
        ent[fn] = {"brier": b, "score": score(b)}
    R["arms"][name] = ent
    print(f"[arm] {name} {json.dumps(ent)} {time.time()-t0:.0f}s", flush=True)

names = list(P)
R["pairs"] = {}
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = names[i], names[j]
        e = {}
        for fn, msk in (("F2024", F2024), ("ALL", np.ones(len(y), bool))):
            d = np.abs(P[a][msk] - P[b][msk])
            e[fn] = {"max_abs_dp": float(d.max()), "mean_abs_dp": float(d.mean()),
                     "rho": float(np.corrcoef(P[a][msk], P[b][msk])[0, 1])}
        e["score_diff_F2024"] = R["arms"][a]["F2024"]["score"] - R["arms"][b]["F2024"]["score"]
        e["score_diff_FHELD"] = R["arms"][a]["FHELD"]["score"] - R["arms"][b]["FHELD"]["score"]
        R["pairs"][f"{a}__vs__{b}"] = e
        print(f"[pair] {a} vs {b} {json.dumps(e)}", flush=True)

os.makedirs(os.path.dirname(OUTJ) or ".", exist_ok=True)
json.dump(R, open(OUTJ, "w"), ensure_ascii=False, indent=1)
print("[done]", OUTJ, flush=True)
