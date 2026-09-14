"""exp_086 stage 0b — 재사용 자산 sha 대조. 학습·쓰기 0건(검사 결과 json 만 쓴다).

파이프라인 효율 규약: 저장된 stage-1 산출물을 재학습 없이 쓰려면 **sha 대조가 필수**다.
여기서 확인하는 것 셋:
  (1) exp_085 조립 캐시(Xd/codes/y) sha == exp_085 metrics 에 기록된 값
  (2) 시드 42 새 팔 예측 2개의 **배열 바이트** sha == exp_085 가 적은 pred_sha256_best
  (3) 기존 팔(AA67·NN 4시드평균·NN s42) 파일 sha == exp_085 가 적은 값
하나라도 어긋나면 재사용을 포기하고 재학습해야 한다.
"""
import hashlib
import json

import numpy as np

H = "/home/<user>/SungjinPark"
W85 = f"{H}/exp085/work"
O85 = f"{H}/exp085/out"

EXPECT_CACHE = {
    "Xd": "78640bf3e12c42e8dc8bf02ef4edbe5ab0db7f1fb4ef7beceb17bc1e76ceb03b",
    "codes": "94b62aa82006b6bfe6c11f721bce8ee542f4a4052eade99edc56356fbd2e8bc7",
    "y": "5d507684a1e12fc7666d52bd5bba1289dc32ebc35260497a4ee3a6404c5feccd",
}
EXPECT_PRED42 = {
    "F1_FM": "94c74952d99b29f55309cfa27c3aaac79e3afad809606d0e8be9eb5929eb0ad5",
    "F2_EMB": "cdba762e249bf952ba2eeb725d1ecd28ccd2111f3301bb988c70eec716e87c47",
}
EXPECT_FILES = {
    "AA67": (f"{H}/experiments/exp_048/pred/pred_A1_u2024_seed42_at1400.npy",
             "0141db5678de394f56da78804ef6087cc20b30233c323f7e466f23f301f74b7f"),
    "NN_s42": (f"{H}/exp064/out/preds/A99ship_s42.npy",
               "51b9d4bcc4e9388dfe843531597df9e471db9b07155918947b8db35157c986ff"),
    "NN_avg4": (f"{H}/exp064/out/preds/A99ship_seedavg.npy",
                "74cc52d8932ab4242ddc1ee25a048dc7bab2871fdf9be40c4e784ae12058667e"),
}


def shaf(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


R = {"cache": {}, "pred_seed42": {}, "existing_arms": {}}
# ⚠️ prep085 은 **파일**이 아니라 **배열 바이트** sha 를 적었다(`sha_arr`), 그리고
#    codes 는 int32 로 캐스팅한 뒤 쟀다. 같은 방식으로 재야 대조가 성립한다.
for k, e in EXPECT_CACHE.items():
    a = np.load(f"{W85}/{k}.npy", mmap_mode="r")
    a = np.ascontiguousarray(a)
    if k == "codes":
        a = a.astype(np.int32)
    g = hashlib.sha256(a.tobytes()).hexdigest()
    R["cache"][k] = {"expected_array_sha256": e, "got": g, "match": g == e,
                     "file_sha256": shaf(f"{W85}/{k}.npy"),
                     "dtype": str(a.dtype), "shape": list(a.shape)}
    del a
for arm, e in EXPECT_PRED42.items():
    a = np.load(f"{O85}/preds/{arm}_s42_r2_best.npy")
    g = hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
    R["pred_seed42"][arm] = {"expected_array_sha256": e, "got": g, "match": g == e,
                             "file_sha256": shaf(f"{O85}/preds/{arm}_s42_r2_best.npy"),
                             "dtype": str(a.dtype), "n": int(a.shape[0])}
for k, (p, e) in EXPECT_FILES.items():
    g = shaf(p)
    R["existing_arms"][k] = {"path": p.replace(H, "~"), "expected": e, "got": g,
                             "match": g == e}
R["all_match"] = all(v["match"] for grp in ("cache", "pred_seed42", "existing_arms")
                     for v in R[grp].values())
json.dump(R, open(f"{H}/exp086/out/reuse_check.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1))
assert R["all_match"], "재사용 sha 불일치 — 재학습 필요"
