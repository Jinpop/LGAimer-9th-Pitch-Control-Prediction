"""exp_071 stage 1(프레임) + stage 0(순전파 구현 검증 게이트).

순서: 프레임을 먼저 만들고 그 위에서 게이트를 돌린다. 게이트는 학습 전에 끝난다.
게이트 3종:
  G0a  numpy 독립 순전파(사양서 verify_spec 동형) vs 배포 nn_infer.predict_nn
  G0b  torch DeployNN(배포 가중치 로드) vs 배포 nn_infer.predict_nn   ← 우리 학습 모듈이 같은 구조인가
  G0c  export_npz 왕복 — DeployNN → npz → nn_infer 로 다시 읽어 동일한가
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "6"
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd

H = "/home/<user>/SungjinPark"
E = f"{H}/exp071"
FR = f"{H}/exp064/cache/frame99_ship.parquet"
MATE = f"{E}/mate"
OUT = f"{E}/out"
CACHE = f"{E}/cache"
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)
T0 = time.time()
R = {}

META = json.load(open(f"{MATE}/model/nn_a42_meta.json"))
CATS, DENSE = META["CATS"], META["DENSE"]

# meta 4개가 동일한지 (mu/sd/levels 공유 확인)
msha = {s: hashlib.sha256(open(f"{MATE}/model/nn_a{s}_meta.json", "rb").read()).hexdigest()
        for s in (42, 777, 101, 2024)}
R["meta_sha256"] = msha
R["meta_4seed_identical"] = len(set(msha.values())) == 1
R["npz_sha256"] = {s: hashlib.sha256(open(f"{MATE}/model/nn_a{s}.npz", "rb").read()).hexdigest()
                   for s in (42, 777, 101, 2024)}
R["frame_sha256"] = hashlib.sha256(open(FR, "rb").read()).hexdigest()

# ── stage 1 : 프레임 ──────────────────────────────────────────────
df = pd.read_parquet(FR)
ids = pd.read_csv(f"{H}/data/train.csv", usecols=["row_id", "pitcher_id", "batter_id"])
n0 = len(df)
df = df.merge(ids, on="row_id", how="left", validate="one_to_one")
assert len(df) == n0, "조인이 행수를 바꿨다"
assert df.pitcher_id.notna().all() and df.batter_id.notna().all(), "조인 결측"
df["cnt_state"] = df.balls_before.astype(int) * 3 + df.strikes_before.astype(int)

# 🔴 밀집 열 순서 = meta['DENSE'] 와 정확히 동일해야 한다 (순서가 곧 입력 순서)
DENSE_USED = list(DENSE)
assert DENSE_USED == META["DENSE"], "DENSE 순서 불일치"
assert list(df[DENSE_USED].columns) == META["DENSE"], "선택 후 열 순서 불일치"
R["dense_order_assert"] = "PASS (73열, meta['DENSE'] 와 원소·순서 동일)"
R["cats_order_assert"] = list(CATS) == META["CATS"]

s = df.season.to_numpy()
g = df.game_type.astype(str).to_numpy()
m_tr = (s <= 2023) & ~((s <= 2022) & (g == "F"))
m_24 = (s == 2024)
R["n_train_rows"] = int(m_tr.sum())
R["n_2024_rows"] = int(m_24.sum())
assert int(m_tr.sum()) == 1116277, f"학습행 {m_tr.sum()} != 1116277"
assert int(m_24.sum()) == 253507, f"2024행 {m_24.sum()} != 253507"
R["row_count_assert"] = "PASS (학습 1,116,277 · 2024 253,507)"

# row_id 오름차순 정렬 확인 (OOT 벡터 정렬 규약)
rid24 = df.loc[m_24, "row_id"].to_numpy()
R["rowid_2024_sorted_in_frame"] = bool((rid24[:-1] <= rid24[1:]).all())
ref_rid = np.load(f"{H}/exp069/oot/row_id_oot2024.npy", allow_pickle=True)
R["rowid_matches_oot_vector"] = bool(len(ref_rid) == len(rid24) and
                                     (np.asarray(ref_rid).astype(str) ==
                                      np.sort(rid24).astype(str)).all())

df.to_parquet(f"{CACHE}/frame071.parquet", index=False)
R["stage1_min"] = (time.time() - T0) / 60.0
print(f"[stage1] 프레임 완료 {R['stage1_min']:.2f}분  학습 {R['n_train_rows']}  "
      f"2024 {R['n_2024_rows']}  rowid정렬 {R['rowid_2024_sorted_in_frame']} "
      f"OOT일치 {R['rowid_matches_oot_vector']}", flush=True)

# ── stage 0 : 게이트 ─────────────────────────────────────────────
t = time.time()
sys.path.insert(0, MATE)
import nn_infer  # noqa: E402

sub = df.head(20000).reset_index(drop=True)


def make_inputs(d, meta):
    mu = np.array(meta["mu"], np.float32)
    sd = np.array(meta["sd"], np.float32)
    codes = np.zeros((len(d), len(meta["CATS"])), np.int64)
    for i, c in enumerate(meta["CATS"]):
        lut = {v: j + 1 for j, v in enumerate(meta["levels"][c])}   # ★함정① +1, 0=미지
        codes[:, i] = d[c].astype(str).map(lut).fillna(0).astype(np.int64).values
    X = (d[meta["DENSE"]].astype("float32").values - mu) / sd       # ★함정③ 동결 mu/sd
    X = np.nan_to_num(X, nan=0., posinf=0., neginf=0.).astype(np.float32)  # ★함정④ 표준화 후
    return codes, X


codes, X = make_inputs(sub, META)
R["unknown_code0_counts"] = {c: int((codes[:, i] == 0).sum()) for i, c in enumerate(CATS)}
R["code_max_vs_embrows"] = {c: [int(codes[:, i].max()), len(META["levels"][c]) + 1]
                            for i, c in enumerate(CATS)}

W = np.load(f"{MATE}/model/nn_a42.npz")
lin = lambda z, i: z @ W[f"mlp_{i}_weight"].T + W[f"mlp_{i}_bias"]
bn = lambda z, i: ((z - W[f"mlp_{i}_running_mean"]) /
                   np.sqrt(W[f"mlp_{i}_running_var"] + 1e-5) *
                   W[f"mlp_{i}_weight"] + W[f"mlp_{i}_bias"])
rl = lambda z: np.maximum(z, 0)
z = np.concatenate([W[f"emb{i}"][codes[:, i]] for i in range(7)] + [X], axis=1)
assert z.shape[1] == 159, z.shape
h = rl(bn(lin(z, 0), 1))
h = rl(bn(lin(h, 4), 5))
h = rl(lin(h, 8))          # ★함정② 3번째 블록엔 BatchNorm 없음
h = lin(h, 10)
mine = 1 / (1 + np.exp(-h.ravel()))
ref = nn_infer.predict_nn(sub, f"{MATE}/model/nn_a42.npz", f"{MATE}/model/nn_a42_meta.json")
g0a = float(np.max(np.abs(mine - ref)))
R["G0a_numpy_vs_deploy_maxabs"] = g0a
R["G0a_verdict"] = "PASS" if g0a < 1e-12 else "FAIL"
print(f"[G0a] numpy 독립 순전파 max|Δ| = {g0a:.3e}  {R['G0a_verdict']}", flush=True)

# G0b : torch DeployNN
import torch  # noqa: E402
import torch.nn as tnn  # noqa: E402
torch.set_num_threads(6)


class DeployNN(tnn.Module):
    def __init__(self, n_levels, edims, n_dense, H1=256, p_drop=0.1):
        super().__init__()
        self.emb = tnn.ModuleList([tnn.Embedding(n + 1, d) for n, d in zip(n_levels, edims)])
        d_in = sum(edims) + n_dense
        H2, H3 = H1 // 2, H1 // 4
        self.mlp = tnn.Sequential(
            tnn.Linear(d_in, H1), tnn.BatchNorm1d(H1), tnn.ReLU(), tnn.Dropout(p_drop),
            tnn.Linear(H1, H2), tnn.BatchNorm1d(H2), tnn.ReLU(), tnn.Dropout(p_drop),
            tnn.Linear(H2, H3), tnn.ReLU(),
            tnn.Linear(H3, 1))

    def forward(self, codes, x_dense):
        z = torch.cat([e(codes[:, i]) for i, e in enumerate(self.emb)] + [x_dense], dim=1)
        return self.mlp(z).squeeze(1)


def export_npz(model, path):
    sd_ = model.state_dict()
    out = {}
    for i, e in enumerate(model.emb):
        out[f"emb{i}"] = sd_[f"emb.{i}.weight"].cpu().numpy().astype(np.float32)
    for k, v in sd_.items():
        if k.startswith("mlp."):
            out["mlp_" + k[4:].replace(".", "_")] = v.cpu().numpy()
    np.savez(path, **out)


nlv = [len(META["levels"][c]) for c in CATS]
mdl = DeployNN(nlv, META["edims"], len(DENSE), H1=META["H1"])
sd_new = {}
for i in range(7):
    sd_new[f"emb.{i}.weight"] = torch.from_numpy(W[f"emb{i}"].copy())
for i in (0, 1, 4, 5, 8, 10):
    for suf, key in (("weight", "weight"), ("bias", "bias"),
                     ("running_mean", "running_mean"), ("running_var", "running_var")):
        k = f"mlp_{i}_{key}"
        if k in W.files:
            sd_new[f"mlp.{i}.{suf}"] = torch.from_numpy(W[k].copy())
for i in (1, 5):
    sd_new[f"mlp.{i}.num_batches_tracked"] = torch.tensor(0, dtype=torch.long)
missing, unexpected = mdl.load_state_dict(sd_new, strict=False)
R["torch_load_missing"] = [str(x) for x in missing]
R["torch_load_unexpected"] = [str(x) for x in unexpected]
mdl.eval()
with torch.no_grad():
    lo = mdl(torch.from_numpy(codes), torch.from_numpy(X)).numpy().astype(np.float64)
tp = 1 / (1 + np.exp(-lo))
g0b = float(np.max(np.abs(tp - ref)))
R["G0b_torch_vs_deploy_maxabs"] = g0b
R["G0b_verdict"] = "PASS" if g0b < 1e-6 else "FAIL"
print(f"[G0b] torch DeployNN max|Δ| = {g0b:.3e}  {R['G0b_verdict']}", flush=True)

# G0c : export_npz 왕복
export_npz(mdl, f"{CACHE}/_roundtrip.npz")
ref2 = nn_infer.predict_nn(sub, f"{CACHE}/_roundtrip.npz", f"{MATE}/model/nn_a42_meta.json")
g0c = float(np.max(np.abs(ref2 - ref)))
R["G0c_export_roundtrip_maxabs"] = g0c
R["G0c_verdict"] = "PASS" if g0c < 1e-12 else "FAIL"
print(f"[G0c] export_npz 왕복 max|Δ| = {g0c:.3e}  {R['G0c_verdict']}", flush=True)

# 함정 넷 기록
R["함정_확인"] = {
    "①_임베딩행=레벨+1_index0=미지": {
        "emb_rows": {c: int(W[f'emb{i}'].shape[0]) for i, c in enumerate(CATS)},
        "levels+1": {c: len(META["levels"][c]) + 1 for c in CATS},
        "일치": all(int(W[f'emb{i}'].shape[0]) == len(META["levels"][c]) + 1
                    for i, c in enumerate(CATS)),
        "코드_0_행수_20000표본": R["unknown_code0_counts"]},
    "②_BatchNorm은_앞_두_블록만": {
        "npz에_있는_BN키": sorted([k for k in W.files if "running" in k]),
        "mlp8_BN없음": "mlp_9_running_mean" not in W.files and "mlp_8_running_mean" not in W.files},
    "③_mu/sd_동결값_재계산금지": {
        "meta_mu[0]": float(META["mu"][0]), "meta_sd[0]": float(META["sd"][0]),
        "프레임_실제_mu[0]_참고": float(df[DENSE[0]].astype("float32").mean()),
        "주석": "meta 값을 그대로 쓴다. 프레임 실제값과 다른 것이 정상(학습행 통계 동결)"},
    "④_nan_to_num은_표준화_후": {
        "구현": "X=(v-mu)/sd → nan_to_num(0)",
        "확인": "표준화 전에 0으로 채우면 (0-mu)/sd 가 되어 0이 아니다"}}
# ④ 실증
_v = df[DENSE].astype("float32").values[:20000]
_mu = np.array(META["mu"], np.float32); _sd = np.array(META["sd"], np.float32)
_after = np.nan_to_num((_v - _mu) / _sd, nan=0., posinf=0., neginf=0.)
_before = (np.nan_to_num(_v, nan=0., posinf=0., neginf=0.) - _mu) / _sd
R["함정_확인"]["④_nan_to_num은_표준화_후"]["두_순서_max|Δ|"] = float(np.max(np.abs(_after - _before)))
R["함정_확인"]["④_nan_to_num은_표준화_후"]["결측_있는_열수"] = int(np.isnan(_v).any(0).sum())

R["stage0_min"] = (time.time() - t) / 60.0
R["wall_min"] = (time.time() - T0) / 60.0
json.dump(R, open(f"{OUT}/stage01.json", "w"), indent=1, ensure_ascii=False)
ok = R["G0a_verdict"] == "PASS" and R["G0b_verdict"] == "PASS" and R["G0c_verdict"] == "PASS"
print("GATE_ALL_" + ("PASS" if ok else "FAIL"), flush=True)
if not ok:
    sys.exit(2)
print("DONE_STAGE01", flush=True)
