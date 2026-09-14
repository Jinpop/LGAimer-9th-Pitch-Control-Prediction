"""항등 앵커가 5행 1 ulp 어긋난 원인 특정 — 팀원 빌더의 결합 순서를 찾는다.

🔴 값을 맞추려고 결과를 고르는 것이 아니다. 팀원이 실제로 쓴 산술 형태를 찾아
   우리 4팔 식이 그 위에 정확히 얹히게 하는 것이 목적이다.
"""
import hashlib
import json
import itertools

import numpy as np
from scipy.special import expit, logit

H = "/home/<user>/SungjinPark"
OOT = f"{H}/exp069/oot"
EPS = 1e-6


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a, np.float64).tobytes()).hexdigest()


p_lgb = np.load(f"{OOT}/p_lgb_oot2024.npy").astype(np.float64)
p_cat = np.load(f"{OOT}/p_cat_oot2024.npy").astype(np.float64)
p_leaf = np.load(f"{OOT}/p_leaf_oot2024.npy").astype(np.float64)
p_rawW = np.load(f"{OOT}/p_v6_3arm_rawW_oot2024.npy").astype(np.float64)
p_renorm = np.load(f"{OOT}/p_v6_3arm_renorm_oot2024.npy").astype(np.float64)
TARGET_RAW, TARGET_REN = sha(p_rawW), sha(p_renorm)

# ── logit 구현 후보 ─────────────────────────────────────────────────────
LGS = {
    "scipy_logit_clip1e-6": lambda p: logit(np.clip(p, EPS, 1 - EPS)),
    "scipy_logit_noclip": lambda p: logit(p),
    "log_ratio_clip": lambda p: np.log(np.clip(p, EPS, 1 - EPS)
                                       / (1 - np.clip(p, EPS, 1 - EPS))),
    "log_minus_log1p_clip": lambda p: (np.log(np.clip(p, EPS, 1 - EPS))
                                       - np.log1p(-np.clip(p, EPS, 1 - EPS))),
}
W = {"lgb": 0.05, "cat": 0.68, "leaf": 0.12}
P = {"lgb": p_lgb, "cat": p_cat, "leaf": p_leaf}

res = []
for lname, lf in LGS.items():
    L = {k: lf(v) for k, v in P.items()}
    # (a) 좌→우 합, 모든 항 순서
    for order in itertools.permutations(("lgb", "cat", "leaf")):
        z = W[order[0]] * L[order[0]]
        for k in order[1:]:
            z = z + W[k] * L[k]
        res.append((f"{lname} | seq {'+'.join(order)}", sha(expit(z)),
                    sha(expit(z / 0.85)), float(np.max(np.abs(expit(z) - p_rawW)))))
    # (b) 스택 dot / einsum / sum(axis=0)  — SIMD 축약 순서가 다르다
    S = np.stack([L["lgb"], L["cat"], L["leaf"]])
    wv = np.array([0.05, 0.68, 0.12])
    for tag, z in (("dot(w,S)", wv @ S),
                   ("S.T@w", S.T @ wv),
                   ("einsum", np.einsum("i,ij->j", wv, S)),
                   ("sum(axis0)", (wv[:, None] * S).sum(axis=0)),
                   ("np.sum_list", np.sum([W[k] * L[k] for k in ("lgb", "cat", "leaf")],
                                          axis=0)),
                   ("add.reduce", np.add.reduce([W[k] * L[k]
                                                 for k in ("lgb", "cat", "leaf")]))):
        z = np.asarray(z, np.float64)
        res.append((f"{lname} | {tag}", sha(expit(z)), sha(expit(z / 0.85)),
                    float(np.max(np.abs(expit(z) - p_rawW)))))
    # (c) 4행 스택(NN 자리를 0.5 = 로짓 0 으로 채운 판)
    S4 = np.stack([L["lgb"], L["cat"], np.zeros_like(L["lgb"]), L["leaf"]])
    w4 = np.array([0.05, 0.68, 0.15, 0.12])
    for tag, z in (("dot4_zeroNN", w4 @ S4), ("sum4_zeroNN", (w4[:, None] * S4).sum(axis=0))):
        z = np.asarray(z, np.float64)
        res.append((f"{lname} | {tag}", sha(expit(z)), sha(expit(z / 0.85)),
                    float(np.max(np.abs(expit(z) - p_rawW)))))

out = {"target_rawW_sha": TARGET_RAW, "target_renorm_sha": TARGET_REN,
       "matches_rawW": [], "matches_renorm": [], "all": []}
for tag, s_raw, s_ren, mx in res:
    hit_r = s_raw == TARGET_RAW
    hit_n = s_ren == TARGET_REN
    out["all"].append({"form": tag, "byte_eq_rawW": hit_r,
                       "byte_eq_renorm_from_same_z": hit_n, "max_abs_diff_rawW": mx})
    if hit_r:
        out["matches_rawW"].append(tag)
    if hit_n:
        out["matches_renorm"].append(tag)

# renorm 이 z/0.85 가 아니라 가중치 재정규화판일 수도 있다
Lb = LGS["scipy_logit_clip1e-6"]
L = {k: Lb(v) for k, v in P.items()}
wn = {k: W[k] / 0.85 for k in W}
zr1 = wn["lgb"] * L["lgb"] + wn["cat"] * L["cat"] + wn["leaf"] * L["leaf"]
zr2 = (W["lgb"] * L["lgb"] + W["cat"] * L["cat"] + W["leaf"] * L["leaf"]) / 0.85
Sr = np.stack([L["lgb"], L["cat"], L["leaf"]])
zr3 = np.array([wn["lgb"], wn["cat"], wn["leaf"]]) @ Sr
out["renorm_forms"] = {
    "weights_prenormalized_seq": {"byte_eq": sha(expit(zr1)) == TARGET_REN,
                                  "max_abs_diff": float(np.max(np.abs(expit(zr1) - p_renorm)))},
    "z_divided_after_seq": {"byte_eq": sha(expit(zr2)) == TARGET_REN,
                            "max_abs_diff": float(np.max(np.abs(expit(zr2) - p_renorm)))},
    "weights_prenormalized_dot": {"byte_eq": sha(expit(zr3)) == TARGET_REN,
                                  "max_abs_diff": float(np.max(np.abs(expit(zr3) - p_renorm)))}}

# 어긋나는 행이 어디인가 (S13/규칙2-4 계보: '마지막 행'이라고 적지 말 것)
zseq = W["lgb"] * L["lgb"] + W["cat"] * L["cat"] + W["leaf"] * L["leaf"]
d = expit(zseq) - p_rawW
idx = np.nonzero(d)[0]
out["differing_rows_seq_form"] = {
    "n": int(idx.size), "indices": idx.tolist()[:40],
    "n_rows_total": int(d.size),
    "is_last_row_among_them": bool(d.size - 1 in idx.tolist()),
    "abs_diff_values": np.abs(d[idx]).tolist()[:40],
    "ulp_at_0.5": 2.0 ** -53,
    "p_values_there": p_rawW[idx].tolist()[:40]}
print(json.dumps(out, ensure_ascii=False, indent=1))
