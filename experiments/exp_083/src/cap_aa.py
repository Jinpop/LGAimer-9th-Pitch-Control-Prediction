"""exp_083 캡처 1 — 우리 AA 팔의 per-row 중간값 포획.  학습 0건.  산술 변경 0.

방법: aa_infer.py 를 한 글자도 고치지 않고 monkeypatch 로 감싼다.
  - load_booster  : 적재한 부스터 경로를 순서대로 기록
  - predict_prob  : 그 부스터의 출력 벡터를 같은 순서로 기록
  - lookup_table_values : (값, 적중마스크) 를 기록 (F 갈래)
감싼 함수는 전부 원본을 그대로 호출하고 반환값을 손대지 않는다 —
따라서 submission.csv 는 부모와 비트 동일해야 하고, 그것을 게이트로 확인한다.

🔴 규칙 2-4: 여기서 꺼내는 것은 전부 '행 i 의 값'이다.  행 방향 집계 없음.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "6"
import sys, json, time                                              # noqa: E402
import numpy as np                                                  # noqa: E402

WORK = sys.argv[1]          # 캡처 작업 디렉터리 (data/, model/ 심볼릭 링크가 있는 곳)
OUTNPZ = sys.argv[2]        # 저장할 npz
os.chdir(WORK)
sys.path.insert(0, WORK)
T0 = time.time()

import aa_infer as A                                                # noqa: E402

LOADS, PREDS = [], []
_o_load, _o_pred = A.load_booster, A.predict_prob
_o_lut = A.lookup_table_values


def load_booster(path, feats, n_trees_expected, tag):
    m = _o_load(path, feats, n_trees_expected, tag)
    LOADS.append((os.path.relpath(path, "model"), tag))
    return m


def predict_prob(model, frame, cat_idx):
    out = _o_pred(model, frame, cat_idx)
    PREDS.append(np.asarray(out, np.float64))
    return out


LUT = []


def lookup_table_values(frame, table, key_cols):
    v, has = _o_lut(frame, table, key_cols)
    LUT.append((tuple(key_cols), np.asarray(v, np.float64),
                np.asarray(has, bool)))
    return v, has


A.load_booster = load_booster
A.predict_prob = predict_prob
A.lookup_table_values = lookup_table_values

ids, sub, p_off, preds, smooth_out = A.run(TEST_DIR="./data", MODEL_DIR="./model")
os.makedirs("./output", exist_ok=True)
A.save_submission("./output/submission.csv", sub)

assert len(LOADS) == len(PREDS), (len(LOADS), len(PREDS))
n = len(preds)
store = {"row_id": np.asarray(ids).astype(str),
         "p_off": np.asarray(p_off, np.float64),
         "p_final": np.asarray(preds, np.float64)}
names = []
for i, (rel, tag) in enumerate(LOADS):
    key = f"{tag.split()[0].strip('[]')}|{rel.replace('/', '_')}"
    # tag 형태: "R 시드 42 셀 모델 'is_mid_only'" 등 → rel 이 seed_dir/파일명 이라 유일
    names.append(key)
    store[f"v{i:03d}"] = PREDS[i].astype(np.float64)
store["names"] = np.asarray(names, dtype=object)
for j, (kc, v, has) in enumerate(LUT):
    store[f"lut{j}_keys"] = np.asarray([",".join(kc)], dtype=object)
    store[f"lut{j}_val"] = v
    store[f"lut{j}_has"] = has
for a, v in (smooth_out or {}).items():
    store[f"smooth_{a}"] = np.asarray(v, np.float64)
np.savez_compressed(OUTNPZ, **store)
print(f"[cap_aa] 부스터 호출 {len(PREDS)}건 · lut {len(LUT)}건 · rows={n} · "
      f"wall={(time.time()-T0)/60:.2f}min -> {OUTNPZ}", flush=True)
