"""exp_107 s0b — 1단계 결과가 정한 eA*, eB* 로 **축 C**와 **귀무 N1b** 타깃을 만든다.

사용: s0b_prep107.py <eA> <eB>
  C     t = (1-eA-eB)·y + eA·0.5 + eB·p_self   (커버 행)  /  (1-eA)·y + eA·0.5  (미커버)
  N1b   t = (1-eA)·y_perm + eA·0.5             — N1a 와 **같은 순열 라벨**, 스무딩만 더한다
🔴 두 벡터 모두 s0_prep107.py 가 저장한 `_yperm.npy`·`_pself.npy`·`_cov.npy` 를 그대로 쓴다
   (같은 난수 실현 — 순열을 다시 뽑지 않는다).
"""
import hashlib, json, sys
import numpy as np

H = "/home/<user>/SungjinPark"
B = f"{H}/exp107"
W = f"{H}/exp097/work"
eA, eB = float(sys.argv[1]), float(sys.argv[2])
assert 0.0 <= eA + eB <= 1.0

y = np.load(f"{W}/y.npy")
trm = np.load(f"{W}/trm_f64o.npy")
yd = y.astype(np.float64)
yp = np.load(f"{B}/work/_yperm.npy").astype(np.float64)
ps = np.load(f"{B}/work/_pself.npy")
cov = np.load(f"{B}/work/_cov.npy")
COVTR = cov & trm
sha = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]
REP = json.load(open(f"{B}/out/s0_prep107.json"))


def emit(name, t, note):
    t32 = np.asarray(t, np.float64).astype(np.float32)
    assert np.isfinite(t32).all() and t32.min() >= 0.0 and t32.max() <= 1.0
    ch = (t32 != y) & trm
    np.save(f"{B}/work/tgt_{name}.npy", t32)
    REP["targets"][name] = {
        "note": note, "n_train_rows_changed": int(ch.sum()),
        "share_train_rows_changed": float(ch.sum() / trm.sum()),
        "mean_abs_shift_train": float(np.abs(t32[trm].astype(np.float64) - yd[trm]).mean()),
        "target_mean_train": float(t32[trm].astype(np.float64).mean()),
        "n_distinct_values_train": int(len(np.unique(t32[trm]))),
        "sha16_train_slice": sha(t32[trm])}
    print(f"{name:14s} changed {ch.sum():>9,} ({ch.sum()/trm.sum():6.2%})  "
          f"mean|Δ| {REP['targets'][name]['mean_abs_shift_train']:.6f}", flush=True)


tC = (1 - eA) * yd + eA * 0.5
tC[COVTR] = (1 - eA - eB) * yd[COVTR] + eA * 0.5 + eB * ps[COVTR]
emit("C_mix", tC, f"축 C — 라벨 스무딩 eA={eA} × 자기 증류 eB={eB} (커버 행), 미커버는 A 만")
emit("N1b_perm_eA", (1 - eA) * yp + eA * 0.5,
     f"귀무 N1b — N1a 와 같은 순열 라벨 + eA={eA} 스무딩. 쌍 (N1b − N1a) 이 스무딩만의 몫이다")

REP["phase2_eps"] = {"eA_star": eA, "eB_star": eB}
json.dump(REP, open(f"{B}/out/s0_prep107.json", "w"), ensure_ascii=False, indent=1)
json.dump([["C_mix", "C_혼합", f"eA={eA} x eB={eB}"],
           ["N1b_perm_eA", "N1_바닥대조", f"순열 라벨 + eA={eA} 스무딩"]],
          open(f"{B}/out/phase2_points.json", "w"), ensure_ascii=False, indent=1)
print("PREP2_DONE")
