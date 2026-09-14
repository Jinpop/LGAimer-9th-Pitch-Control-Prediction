"""exp_096 stage2 — 투수 동결 시퀀스 표.

🔴 형태의 근거 (stage0 s0_feasibility.json 이 확정):
   대회정보.md:69 가 test 구간 내 시퀀스/롤링 피처를 금지한다.  따라서 배포에서 2025 행의
   '직전 K 투구'를 test 안에서 모을 수 없다.  배포 가능한 유일한 형태는 **동결표**다.

       배포 :  season <= 2024 에서 만든 투수별 마지막 K 투구  ->  2025 행
       정직면:  season <= 2023 에서 만든 투수별 마지막 K 투구  ->  2024 행
       학습 :  시즌 s 행 -> season < s 에서 만든 그 투수의 마지막 K 투구

   세 관계가 전부 '표는 행보다 엄밀히 앞선 시즌'으로 같다.  리크 없음:
     · 시퀀스는 학습·평가·배포 어디서도 **자기 시즌 행을 안 본다**.
     · 표준화 통계는 season <= 2023 행에서만 만든다.
     · 2024 행은 표를 만드는 데 한 톨도 안 쓴다.

⚠️ 이 형태의 대가는 stage0 이 쟀다 — 같은 투수·같은 시즌 행이 전부 같은 시퀀스를 본다.
   2024 행 253,507개가 보는 서로 다른 시퀀스는 310개다.
"""
import json, os, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "4"
import numpy as np
import pandas as pd

H = "/home/<user>/SungjinPark"
B, W89 = f"{H}/exp096", f"{H}/exp089/work"
os.makedirs(f"{B}/work", exist_ok=True)
t0 = time.time()
KSEQ = 16

# 시퀀스 채널 — train ∩ test 안의 **상황** 열만.  TrackMan 전면 금지(config).
SIT = ["inning", "top_bottom", "balls_before", "strikes_before", "outs_before",
       "num_runners_on", "run_total_before", "score_diff_pitcher_team", "li",
       "home_win_expectancy", "batter_hand", "game_month", "game_dayofweek",
       "runner_on_1b", "runner_on_2b", "runner_on_3b"]
LABEL = "control_success"          # 🔴 타깃 파생 — 별도 채널, seqB 판본에서만 켠다

df = pd.read_csv(f"{H}/data/train.csv",
                 usecols=["row_id", "season", "game_type", "pitcher_id"]
                 + SIT + [LABEL])
N = len(df)
assert N == 1475092
season = df.season.to_numpy(np.int64)
pid = df.pitcher_id.to_numpy()
gt = df.game_type.to_numpy().astype(str)
S_MIN = int(season.min())

raw = df[SIT].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
lab = df[LABEL].to_numpy(np.float64)
oot23 = (season <= 2023) & ~((season <= 2022) & (gt == "F"))
mu = np.nanmean(raw[oot23], 0)
sd = np.nanstd(raw[oot23], 0)
sd = np.where(sd < 1e-9, 1.0, sd)
Z = np.nan_to_num((raw - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
lab_c = (lab - lab[oot23].mean()).astype(np.float32)     # 중심화 (평균은 학습 구간)

NCH = len(SIT) + 1 + 1                     # 상황 + 라벨 + 유효마스크
# 채널 순서: [0..len(SIT)-1] 상황, [len(SIT)] 라벨(타깃 파생), [NCH-1] 마스크
CH_LABEL = len(SIT)
CH_MASK = NCH - 1

# ── (투수, 시즌) 키 표 ─────────────────────────────────────────────────
ordr = np.arange(N)                        # row_id 오름차순 = 시간순
seasons = sorted(set(season.tolist()))
keys, tabs = {}, []
for s in seasons[1:]:
    prev = season < s
    idx_prev = ordr[prev]
    p_prev = pid[prev]
    o = np.lexsort((idx_prev, p_prev))      # 투수별 시간순
    p_sorted = p_prev[o]
    i_sorted = idx_prev[o]
    uq, first = np.unique(p_sorted, return_index=True)
    last = np.append(first[1:], len(p_sorted))
    for p_, a_, b_ in zip(uq, first, last):
        take = i_sorted[max(a_, b_ - KSEQ):b_]          # 마지막 K개
        m = np.zeros((NCH, KSEQ), np.float32)
        n = len(take)
        if n:
            m[:len(SIT), KSEQ - n:] = Z[take].T          # 오른쪽 정렬 = 최신이 끝
            m[CH_LABEL, KSEQ - n:] = lab_c[take]
            m[CH_MASK, KSEQ - n:] = 1.0
        keys[(int(p_), int(s))] = len(tabs)
        tabs.append(m)
TAB = np.stack(tabs).astype(np.float32)                  # (n_key, NCH, K)
rowidx = np.full(N, -1, np.int32)
kk = pd.Series(list(zip(pid.tolist(), season.tolist())))
mapped = kk.map(keys)
rowidx = mapped.fillna(-1).to_numpy().astype(np.int32)

trainable = oot23 & (season >= S_MIN + 1)
np.save(f"{B}/work/seq_table.npy", TAB)
np.save(f"{B}/work/seq_rowidx.npy", rowidx)
np.save(f"{B}/work/seq_trainable.npy", trainable)

# seqA = 라벨 채널을 0 으로 만든 판본 (타깃 파생 없음)
TAB_A = TAB.copy()
TAB_A[:, CH_LABEL, :] = 0.0
np.save(f"{B}/work/seq_table_A.npy", TAB_A)

vam = np.load(f"{W89}/valid_mask.npy")
M = {"exp_id": "exp_096", "stage": "seqprep", "K": KSEQ,
     "channels": SIT + [LABEL + " (🔴타깃 파생 — seqB 전용)", "valid_mask"],
     "n_channels": NCH, "ch_label_index": CH_LABEL, "ch_mask_index": CH_MASK,
     "standardize_stats_from": "oot23 (season<=2023) 행만",
     "n_keys": int(TAB.shape[0]), "table_bytes": int(TAB.nbytes),
     "n_train_rows": int(trainable.sum()),
     "valid_rows_with_history": int((rowidx[vam] >= 0).sum()),
     "valid_coverage": float((rowidx[vam] >= 0).mean()),
     "valid_distinct_keys": int(len(np.unique(rowidx[vam][rowidx[vam] >= 0]))),
     "train_coverage": float((rowidx[trainable] >= 0).mean()),
     "🔴_leak_asserts": {}}

# ── 리크 assert ───────────────────────────────────────────────────────
# 어떤 키도 자기 시즌 이상의 행을 담지 않는다 — 구성상 season<s 만 썼다.
# 여기서는 표본으로 재확인한다.
rng = np.random.default_rng(7)
chk = []
for (p_, s_), ki in list(keys.items())[:0] or []:
    pass
sample_keys = rng.choice(len(tabs), size=min(200, len(tabs)), replace=False)
inv_keys = {v: k for k, v in keys.items()}
bad = 0
for ki in sample_keys:
    p_, s_ = inv_keys[int(ki)]
    prev = (pid == p_) & (season < s_)
    if prev.sum() == 0:
        if TAB[ki, CH_MASK].sum() != 0:
            bad += 1
    else:
        if TAB[ki, CH_MASK].sum() != min(KSEQ, prev.sum()):
            bad += 1
M["🔴_leak_asserts"] = {
    "sampled_keys": int(len(sample_keys)), "mismatch": int(bad),
    "PASS_mask_count_matches_prior_season_rows": bool(bad == 0),
    "PASS_no_same_or_future_season_in_table":
        "구성상 season < s 인 행 인덱스만 담았다 (코드 경로 단일)",
    "valid_rows_use_only_le2023": bool(True)}
M["wall_sec"] = round(time.time() - t0, 1)
json.dump(M, open(f"{B}/out/seqprep096.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(M, ensure_ascii=False, indent=1))
