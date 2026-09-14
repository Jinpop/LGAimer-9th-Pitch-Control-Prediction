# exp_002 / 01: trackman <-> train 매핑 사전 구축
# 방법: exp_000_eda 04/05 검증 로직의 확장.
#   1) train 경기 블록 복원: (season,month,dow,game_type,team_pair) 변화점 + 블록 내 inning 역행 분할
#   2) 경기 풀-시퀀스 지문(이닝·초말·B/S/O·투수손 전체 배열 해시)으로 train 블록 <-> tm 경기 매칭
#      - Stage A: (season,month,dow,n,지문) 유일-유일
#      - Stage B: (season,지문) 유일-유일 (month/dow 불일치 구제, 여전히 풀시퀀스 일치 필수)
#   3) 매칭 경기에서 행 단위 (train_id <-> trackman_id) 대응 추출, 경기 내/전역 1:1 위반 시 경기 매칭 폐기
#   4) 투수 손 일치 검증 실패 쌍 폐기 (오매핑 > 무매핑)
# 결정론적: 해시·정렬 기반, 난수 없음.
import pandas as pd
import numpy as np
import hashlib
import json
import os
import sys
import time

DATA = os.path.expanduser("~/SungjinPark/data")
OUT = os.path.expanduser("~/SungjinPark/experiments/exp_002/out")
os.makedirs(OUT, exist_ok=True)
t0 = time.time()


def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)


# ---------- load ----------
log("loading train ...")
train = pd.read_csv(
    os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
    usecols=["row_id", "season", "game_month", "game_dayofweek", "game_type",
             "inning", "top_bottom", "balls_before", "strikes_before", "outs_before",
             "pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
             "pitcher_team_id", "batter_team_id"])
log("train:", train.shape)

log("loading trackman ...")
tm = pd.read_csv(
    os.path.join(DATA, "trackman_history.csv"), encoding="utf-8-sig",
    usecols=["trackman_id", "season", "game_date", "game_month", "game_dayofweek",
             "trackman_game_id", "pitch_no", "inning", "top_bottom",
             "balls_before", "strikes_before", "outs_before",
             "pitcher_trackman_id", "batter_trackman_id",
             "pitcher_hand", "batter_hand", "pitcher_team", "batter_team"])
log("trackman:", tm.shape)

# ---------- normalize ----------
# train: top_bottom 'T'/'B', hand 1(Left)/2(Right)  [exp_000_eda/05 검증]
# tm:    top_bottom 'Top'/'Bottom', hand 'Left'/'Right'
tm["tb"] = tm["top_bottom"].map({"Top": "T", "Bottom": "B", "T": "T", "B": "B"})
assert tm["tb"].notna().all()
HAND = {"Left": 1, "Right": 2, "L": 1, "R": 2}
tm["p_hand"] = tm["pitcher_hand"].map(HAND).astype(np.int8)
tm["b_hand"] = tm["batter_hand"].map(HAND).astype(np.int8)
assert set(train["pitcher_hand"].unique()) <= {1, 2}

# tm 경기 내 pitch_no 중복 확인 (있으면 해당 경기 매칭 대상 제외)
dup_mask = tm.duplicated(subset=["trackman_game_id", "pitch_no"], keep=False)
bad_tm_games = set(tm.loc[dup_mask, "trackman_game_id"].unique())
log("tm games with duplicate pitch_no:", len(bad_tm_games))

# ---------- 1) train 경기 블록 복원 ----------
log("reconstructing train game blocks ...")
tp_min = np.minimum(train["pitcher_team_id"].values, train["batter_team_id"].values)
tp_max = np.maximum(train["pitcher_team_id"].values, train["batter_team_id"].values)
bk = (train["season"].astype(str) + "|" + train["game_month"].astype(str) + "|" +
      train["game_dayofweek"].astype(str) + "|" + train["game_type"] + "|" +
      pd.Series(tp_min).astype(str) + "_" + pd.Series(tp_max).astype(str))
key_change = (bk != bk.shift(1)).values
inn = train["inning"].values
inning_drop = np.zeros(len(train), dtype=bool)
inning_drop[1:] = inn[1:] < inn[:-1]          # 경기 내 이닝은 단조 → 역행 = 새 경기
new_block = key_change | (~key_change & inning_drop)
new_block[0] = True
train["block"] = np.cumsum(new_block)
n_blocks = train["block"].iloc[-1]
log("train blocks:", n_blocks, "(exp_000_eda 기준 4,821 — 분할 규칙 추가로 달라질 수 있음)")

# ---------- 2) 지문 계산 ----------
TB01 = {"T": 0, "B": 1}


def fingerprint(inning, tb01, balls, strikes, outs, phand):
    arr = np.column_stack([inning, tb01, balls, strikes, outs, phand]).astype(np.int16)
    return hashlib.md5(arr.tobytes()).hexdigest()


log("fingerprinting train blocks ...")
train["tb01"] = train["top_bottom"].map(TB01).astype(np.int8)
blk_rows = {}   # block -> index array (파일 순서 = 투구 순서)
for b, idx in train.groupby("block").indices.items():
    blk_rows[b] = idx
blk_info = []
for b, idx in blk_rows.items():
    s = train.iloc[idx]
    fp = fingerprint(s["inning"].values, s["tb01"].values, s["balls_before"].values,
                     s["strikes_before"].values, s["outs_before"].values,
                     s["pitcher_hand"].values)
    blk_info.append((b, s["season"].iat[0], s["game_month"].iat[0],
                     s["game_dayofweek"].iat[0], s["game_type"].iat[0],
                     len(s), fp, s["row_id"].iat[0], s["row_id"].iat[-1]))
blocks = pd.DataFrame(blk_info, columns=["block", "season", "month", "dow", "game_type",
                                         "n", "fp", "row_id_start", "row_id_end"]).set_index("block")
log("train fingerprints done:", len(blocks))

log("fingerprinting tm games ...")
tm["tb01"] = tm["tb"].map(TB01).astype(np.int8)
tm_sorted = tm.sort_values(["trackman_game_id", "pitch_no"], kind="mergesort")
tm_rows = {g: idx for g, idx in tm_sorted.groupby("trackman_game_id", sort=False).indices.items()}
gm_info = []
for g, idx in tm_rows.items():
    if g in bad_tm_games:
        continue
    s = tm_sorted.iloc[idx]
    fp = fingerprint(s["inning"].values, s["tb01"].values, s["balls_before"].values,
                     s["strikes_before"].values, s["outs_before"].values,
                     s["p_hand"].values)
    gm_info.append((g, s["season"].iat[0], s["game_month"].iat[0],
                    s["game_dayofweek"].iat[0], len(s), fp, s["game_date"].iat[0]))
games = pd.DataFrame(gm_info, columns=["game", "season", "month", "dow", "n", "fp",
                                       "game_date"]).set_index("game")
log("tm fingerprints done:", len(games))

# ---------- 3) 매칭 ----------
from collections import Counter

matches = {}      # block -> (game, stage)


def match_stage(bl, gm, keycols, stage):
    kb = bl[keycols].astype(str).agg("|".join, axis=1)
    kg = gm[keycols].astype(str).agg("|".join, axis=1)
    cb, cg = Counter(kb), Counter(kg)
    gmap = {}
    for g, k in kg.items():
        if cg[k] == 1:
            gmap[k] = g
    n_new = 0
    for b, k in kb.items():
        if cb[k] == 1 and k in gmap:
            matches[b] = (gmap[k], stage)
            n_new += 1
    log(f"stage {stage}: +{n_new} matches (cum {len(matches)})")


match_stage(blocks, games, ["season", "month", "dow", "n", "fp"], "A")
rb = blocks.loc[~blocks.index.isin(matches)]
rg = games.loc[~games.index.isin({g for g, _ in matches.values()})]
match_stage(rb, rg, ["season", "fp"], "B")

# ---------- 4) 행 대응 추출 + 경기 내 1:1 검증 ----------
log("extracting row-level correspondences ...")
game_pairs = {}    # block -> dict(pitch_pairs=set, bat_pairs=set, team_pairs=set)
fail_within = []
for b, (g, stage) in sorted(matches.items()):
    A = train.iloc[blk_rows[b]]
    B = tm_sorted.iloc[tm_rows[g]]
    assert len(A) == len(B)
    pp = pd.DataFrame({"tr": A["pitcher_id"].values, "tm": B["pitcher_trackman_id"].values})
    bp = pd.DataFrame({"tr": A["batter_id"].values, "tm": B["batter_trackman_id"].values})
    ok = (pp.groupby("tr")["tm"].nunique().eq(1).all() and
          pp.groupby("tm")["tr"].nunique().eq(1).all() and
          bp.groupby("tr")["tm"].nunique().eq(1).all() and
          bp.groupby("tm")["tr"].nunique().eq(1).all())
    if not ok:
        fail_within.append(b)
        continue
    # 손 검증 (지문에 투수손 포함이지만 타자손은 별도 확인)
    if not (A["batter_hand"].values == B["b_hand"].values).all():
        fail_within.append(b)
        continue
    tmp = pd.DataFrame({"s": A["season"].values,
                        "tr_t": A["pitcher_team_id"].values, "tm_t": B["pitcher_team"].values})
    game_pairs[b] = dict(
        game=g, stage=stage,
        pitch=set(map(tuple, pp.drop_duplicates().values)),
        bat=set(map(tuple, bp.drop_duplicates().values)),
        team=set(map(tuple, tmp.drop_duplicates().values)),
        n_rows=len(A))
log("within-game verification failed (discarded):", len(fail_within), fail_within[:10])

# ---------- 5) 전역 1:1 충돌 제거 (오매핑 > 무매핑: 충돌 경기 통째 폐기) ----------
discarded_conflict = []
for it in range(10):
    edge_games = {}
    for b, d in game_pairs.items():
        for e in d["pitch"]:
            edge_games.setdefault(("P", e), set()).add(b)
        for e in d["bat"]:
            edge_games.setdefault(("B", e), set()).add(b)
    # 노드별 엣지 수집
    conflicts = set()
    for kind in ["P", "B"]:
        by_tr, by_tm = {}, {}
        for (k, (tr, tmi)), gs in edge_games.items():
            if k != kind:
                continue
            by_tr.setdefault(tr, set()).add((tr, tmi))
            by_tm.setdefault(tmi, set()).add((tr, tmi))
        for node_edges in list(by_tr.values()) + list(by_tm.values()):
            if len(node_edges) > 1:
                # 지지 경기수 최대 엣지만 생존, 동률이면 전부 폐기
                supp = {e: len(edge_games[(kind, e)]) for e in node_edges}
                mx = max(supp.values())
                top = [e for e, c in supp.items() if c == mx]
                losers = [e for e in node_edges if e not in top] if len(top) == 1 else list(node_edges)
                for e in losers:
                    conflicts |= edge_games[(kind, e)]
    if not conflicts:
        log(f"global conflict resolution converged at iter {it}")
        break
    log(f"iter {it}: discarding {len(conflicts)} conflicting game matches")
    for b in conflicts:
        discarded_conflict.append(b)
        game_pairs.pop(b, None)

# ---------- 6) 매핑 사전 집계 ----------
def build_map(kind):
    rows = []
    for b, d in game_pairs.items():
        for tr, tmi in d[kind]:
            rows.append((tr, tmi, b))
    df = pd.DataFrame(rows, columns=["tr", "tm", "block"])
    agg = df.groupby(["tr", "tm"]).agg(n_games=("block", "nunique")).reset_index()
    assert agg["tr"].is_unique and agg["tm"].is_unique, f"{kind} map not 1:1"
    return agg


pmap = build_map("pitch")
bmap = build_map("bat")
log("pitcher map:", len(pmap), " batter map:", len(bmap))

# 손 일치 검증 (전체 데이터 기준 최빈값 비교)
tr_ph = train.groupby("pitcher_id")["pitcher_hand"].agg(lambda s: s.mode().iat[0])
tm_ph = tm.groupby("pitcher_trackman_id")["p_hand"].agg(lambda s: s.mode().iat[0])
pmap["hand_train"] = pmap["tr"].map(tr_ph)
pmap["hand_tm"] = pmap["tm"].map(tm_ph)
pmap["hand_match"] = pmap["hand_train"] == pmap["hand_tm"]
n_handfail = int((~pmap["hand_match"]).sum())
log("pitcher hand mismatch pairs (discarding):", n_handfail)
pmap_bad = pmap[~pmap["hand_match"]]
pmap = pmap[pmap["hand_match"]].copy()

tr_bh = train.groupby("batter_id")["batter_hand"].agg(lambda s: s.mode().iat[0])
tm_bh = tm.groupby("batter_trackman_id")["b_hand"].agg(lambda s: s.mode().iat[0])
bmap["hand_train"] = bmap["tr"].map(tr_bh)
bmap["hand_tm"] = bmap["tm"].map(tm_bh)
bmap["hand_match"] = bmap["hand_train"] == bmap["hand_tm"]
n_bhandfail = int((~bmap["hand_match"]).sum())
log("batter hand mismatch pairs (discarding):", n_bhandfail)
bmap = bmap[bmap["hand_match"]].copy()

# 투구수 근거 (매칭 경기 내 해당 선수 투구 행수)
mapped_blocks = set(game_pairs)
tr_m = train[train["block"].isin(mapped_blocks)]
pmap["n_pitches_evidence"] = pmap["tr"].map(tr_m.groupby("pitcher_id").size()).fillna(0).astype(int)
bmap["n_pitches_evidence"] = bmap["tr"].map(tr_m.groupby("batter_id").size()).fillna(0).astype(int)
pmap["confidence"] = np.where(pmap["n_games"] >= 3, "A", np.where(pmap["n_games"] == 2, "B", "C"))
bmap["confidence"] = np.where(bmap["n_games"] >= 3, "A", np.where(bmap["n_games"] == 2, "B", "C"))

pmap = pmap.rename(columns={"tr": "pitcher_id", "tm": "pitcher_trackman_id"})
bmap = bmap.rename(columns={"tr": "batter_id", "tm": "batter_trackman_id"})
pmap.sort_values("pitcher_id").to_csv(os.path.join(OUT, "pitcher_id_map.csv"), index=False)
bmap.sort_values("batter_id").to_csv(os.path.join(OUT, "batter_id_map.csv"), index=False)

# ---------- 7) game_date_map (분석 전용 — 본번 피처 금지) ----------
gd = []
for b, d in sorted(game_pairs.items()):
    g = d["game"]
    date_str = games.loc[g, "game_date"]
    parts = g.split("-")
    stadium = parts[1] if len(parts) >= 3 else ""
    gd.append((b, blocks.loc[b, "row_id_start"], blocks.loc[b, "row_id_end"],
               blocks.loc[b, "season"], blocks.loc[b, "month"], blocks.loc[b, "dow"],
               blocks.loc[b, "game_type"], blocks.loc[b, "n"],
               g, pd.to_datetime(date_str, format="mixed").date().isoformat(),
               stadium, d["stage"]))
gmap_df = pd.DataFrame(gd, columns=["train_block", "row_id_start", "row_id_end", "season",
                                    "month", "dow", "game_type", "n_pitches",
                                    "trackman_game_id", "real_date_ANALYSIS_ONLY",
                                    "stadium_ANALYSIS_ONLY", "match_stage"])
gmap_df.to_csv(os.path.join(OUT, "game_date_map.csv"), index=False)

# ---------- 8) 커버리지 ----------
cov = {}
cov["blocks_total"] = int(n_blocks)
cov["blocks_matched"] = len(game_pairs)
cov["blocks_matched_pct"] = round(100 * len(game_pairs) / n_blocks, 2)
cov["rows_total"] = len(train)
cov["rows_in_matched_games"] = int(tr_m.shape[0])
cov["rows_in_matched_games_pct"] = round(100 * tr_m.shape[0] / len(train), 2)
cov["pitchers_total"] = int(train["pitcher_id"].nunique())
cov["pitchers_mapped"] = len(pmap)
cov["pitchers_mapped_pct"] = round(100 * len(pmap) / train["pitcher_id"].nunique(), 2)
mp = set(pmap["pitcher_id"])
cov["rows_with_mapped_pitcher_pct"] = round(100 * train["pitcher_id"].isin(mp).mean(), 2)
tr24 = train[train["season"] == 2024]
cov["rows2024_with_mapped_pitcher_pct"] = round(100 * tr24["pitcher_id"].isin(mp).mean(), 2)
trR = train[train["game_type"] == "R"]
cov["rowsR_with_mapped_pitcher_pct"] = round(100 * trR["pitcher_id"].isin(mp).mean(), 2)
cov["batters_total"] = int(train["batter_id"].nunique())
cov["batters_mapped"] = len(bmap)
cov["batters_mapped_pct"] = round(100 * len(bmap) / train["batter_id"].nunique(), 2)
mb = set(bmap["batter_id"])
cov["rows_with_mapped_batter_pct"] = round(100 * train["batter_id"].isin(mb).mean(), 2)
cov["fail_within_game_verification"] = len(fail_within)
cov["fail_global_conflict"] = len(set(discarded_conflict))
cov["fail_pitcher_hand"] = n_handfail
cov["fail_batter_hand"] = n_bhandfail
cov["stageA_matches"] = sum(1 for d in game_pairs.values() if d["stage"] == "A")
cov["stageB_matches"] = sum(1 for d in game_pairs.values() if d["stage"] == "B")
cov["tm_games_total"] = len(games)
with open(os.path.join(OUT, "coverage.json"), "w") as f:
    json.dump(cov, f, indent=2, ensure_ascii=False)
log(json.dumps(cov, indent=2, ensure_ascii=False))

# 블록 테이블 저장 (검증 스크립트 재사용)
blocks.reset_index().to_csv(os.path.join(OUT, "_train_blocks.csv"), index=False)
log("DONE 01_build_mapping")
