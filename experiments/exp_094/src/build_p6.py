"""feat3.parquet = feat2 + p6 계열 (타겟 무관 · 트랙맨 유래)

포함: p6_* (6) · idea_matchup_pitch (1) · i3_anom_* (6) · i3_anom_l1 (1)  = 14개

왜 배포 모델을 전 연도에 그대로 쓰는가
  p6 는 **타겟 유래가 아니다** (트랙맨 구종 라벨로 학습). anom_cnt 도 카운트별 평균
  구종확률이라 타겟 무관이다. 팀 학습 규율은 *"연도 Y 의 **타겟 유래** 테이블은
  season < Y"* 이므로 p6 계열은 per-year 재생성 대상이 아니다.
  (제8계열 A/B 도 배포 p6 를 재사용했다.)

제외: idea_cnt_te · i3_te48 · sit_* — 전부 타겟 유래라 per-year 표가 필요하다.
"""
import time
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.abspath("v16"))
import pipeline as P

t0 = time.time()
TD = "v16/model/tables"
df = pd.read_parquet("feat2.parquet")
print(f"[load] {df.shape} {time.time()-t0:.0f}s", flush=True)

full = P.apply_features(df, TD)
KEEP = P.PC + ["idea_matchup_pitch"] + [f"i3_anom_{t}" for t in P.TYPES] + ["i3_anom_l1"]
add = full[KEEP].astype(np.float32)
print(f"[p6] {add.shape} {list(add.columns)} {time.time()-t0:.0f}s")
print("[nan]", {c: int(add[c].isna().sum()) for c in add.columns if add[c].isna().any()})

out = pd.concat([df, add], axis=1)
out.to_parquet("feat3.parquet", index=False)
print(f"[save] feat3.parquet {out.shape} {time.time()-t0:.0f}s")
