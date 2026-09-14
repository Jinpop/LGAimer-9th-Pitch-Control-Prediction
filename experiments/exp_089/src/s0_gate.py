"""exp_089 stage 0 게이트 — 복원 구조가 출하 npz 와 **키별로 같은 형상**을 내는가.

두 겹으로 검사한다.
  ① 형상: 우리 모델의 원본 파라미터를 `deep_infer._fold` 에 태워 나온
     A · c0 · T{0..4} · m.{1,4,5,8,10} 의 형상이 출하 npz 와 **키 집합·형상 모두** 일치.
     하나라도 어긋나면 rc=2 로 정지한다.
  ② 순전파: 같은 가중치를 우리 torch forward 와 `deep_infer._one`(접힌 경로) 에 태워
     같은 입력에서 로짓이 일치하는지. 형상만 맞고 결선이 다르면 여기서 걸린다.
🔴 audit_deep 은 읽기만 한다.
"""
import json, os, sys
import numpy as np
import torch

H = "/home/<user>/SungjinPark"
B = f"{H}/exp089"
PKG = f"{H}/audit_deep/pkg/DEEP/mate"
sys.path.insert(0, PKG)
sys.path.insert(0, f"{B}/src")
import deep_infer as DI                      # noqa: E402  (읽기 전용)
from arch089 import TokenMLP, raw_npz        # noqa: E402

P = json.load(open(f"{PKG}/model/deep/prep.json", encoding="utf-8"))
CARDS = [len(P["levels"][c]) + 1 for c in P["CAT"]]
R = {"exp_id": "exp_089", "stage": "s0_gate", "cards_from_prep_levels_plus_unk": CARDS}

torch.manual_seed(0)
model = TokenMLP(len(P["NUM"]), CARDS, p_drop=0.1, init="randn").eval()
R["n_params"] = int(sum(p.numel() for p in model.parameters()))
raw = raw_npz(model)
R["restored_raw_keys"] = {k: list(np.shape(v)) for k, v in raw.items()}

A, c0, T = DI._fold(raw)
ours = {"A": A.shape, "c0": c0.shape}
ours.update({f"T{j}": T[j].shape for j in range(5)})
for k in ("m.1.weight", "m.1.bias", "m.4.weight", "m.4.bias", "m.5.weight",
          "m.5.bias", "m.8.weight", "m.8.bias", "m.10.weight", "m.10.bias"):
    ours[k] = raw[k].shape

ship = np.load(f"{PKG}/model/deep/deepf_s42.npz")
theirs = {k: ship[k].shape for k in ship.files}
same_keys = set(ours) == set(theirs)
mismatch = {k: {"ours": list(map(int, ours[k])), "theirs": list(map(int, theirs[k]))}
            for k in sorted(set(ours) & set(theirs))
            if tuple(ours[k]) != tuple(theirs[k])}
R["fold_shape_check"] = {
    "key_sets_identical": bool(same_keys),
    "ours_only": sorted(set(ours) - set(theirs)),
    "theirs_only": sorted(set(theirs) - set(ours)),
    "shape_mismatches": mismatch,
    "n_keys": len(ours),
    "shapes": {k: list(map(int, v)) for k, v in sorted(ours.items())},
}
GATE0 = same_keys and not mismatch
R["fold_shape_check"]["PASS"] = bool(GATE0)

# ── ② 순전파 동치 ──────────────────────────────────────────────────────
rng = np.random.default_rng(7)
n = 4096
Xp = rng.standard_normal((n, len(P["NUM"]))).astype(np.float32)
Cp = np.stack([rng.integers(0, c, n) for c in CARDS], 1).astype(np.int64)
with torch.no_grad():
    z_torch = model(torch.from_numpy(Cp), torch.from_numpy(Xp)).double().numpy()
z_np = DI._one(raw, Cp, Xp)
d = np.abs(z_torch - z_np)
R["forward_equivalence"] = {
    "n_probe_rows": n,
    "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
    "sd_z_torch": float(z_torch.std()), "sd_z_numpy": float(z_np.std()),
    "note": "torch float32 vs deep_infer float64 접기 — 1e-4 이하면 같은 결선으로 본다",
}
GATE1 = float(d.max()) < 1e-4
R["forward_equivalence"]["PASS"] = bool(GATE1)

# ── 출하 npz 지문 (probe 대조용 기준) ────────────────────────────────────
FP = {}
for s in (42, 777, 2024, 101, 7):
    Wz = np.load(f"{PKG}/model/deep/deepf_s{s}.npz")
    FP[str(s)] = {k: float(np.asarray(Wz[k]).std()) for k in Wz.files}
    FP[str(s)]["m.10.bias_value"] = float(np.asarray(Wz["m.10.bias"]).ravel()[0])
R["shipped_fingerprint_sd"] = FP
keys = list(FP["42"].keys())
R["shipped_fingerprint_range"] = {
    k: {"min": min(FP[s][k] for s in FP), "max": max(FP[s][k] for s in FP)}
    for k in keys}

R["VERDICT"] = "PASS" if (GATE0 and GATE1) else "FAIL"
os.makedirs(f"{B}/out", exist_ok=True)
json.dump(R, open(f"{B}/out/s0_gate.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in R.items()
                  if k not in ("shipped_fingerprint_sd", "restored_raw_keys")},
                 ensure_ascii=False, indent=1))
sys.exit(0 if (GATE0 and GATE1) else 2)
