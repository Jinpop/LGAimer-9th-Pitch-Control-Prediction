"""exp_108 metrics.json 조립 — s2_mix108.json + 재현 게이트 + 초기화 출처 기록."""
import glob, json, os

import numpy as np

H = "/home/<user>/SungjinPark"
B = f"{H}/exp108"
R = json.load(open(f"{B}/out/s2_mix108.json"))
G = json.load(open(f"{B}/out/gates_repro.json"))
S3 = json.load(open(f"{B}/out/s3_pairs108.json"))
import numpy as _np, hashlib as _hl
_ref = _np.load(f"{H}/exp097/out/preds/z_F1632_f64o_s42.npy")
_sh = lambda a: _hl.sha256(_np.ascontiguousarray(a).tobytes()).hexdigest()
for _t in ("GATE_uv1_s42", "GATE_ub1_s42"):
    _p = f"{B}/out/preds/z_{_t}.npy"
    if os.path.exists(_p):
        _z = _np.load(_p)
        G[_t] = {"sha_match_vs_exp097_z_F1632_f64o_s42": bool(_sh(_z) == _sh(_ref)),
                 "max_abs_diff": float(_np.abs(_z - _ref).max())}
G["ALL_PASS"] = all(v["sha_match_vs_exp097_z_F1632_f64o_s42"]
                    for k, v in G.items() if isinstance(v, dict))
TAB = R["표"]

# ── 초기화 출처 (🔴 "코드에서 읽어라" 에 대한 정직한 답) ────────────────────
PREP = f"{H}/audit_deep/pkg/DEEP/mate/model/deep/prep.json"
prep_keys = sorted(json.load(open(PREP)).keys())
INIT_PROV = {
 "🔴읽은_것": {
  "우리_OOT_복제본의_초기화": {
   "코드": "exp108/src/arch108.py TokenMLP.reset_tokenizer, spec 'ftt'"
           " (arch102b.py·arch089.py 와 본문 동일)",
   "정의": "w, b ~ U(-1/√D, +1/√D),  D = 96  →  경계 0.10206207261596575,"
           " 이론 sd 0.05892556509887896",
   "닿는_곳": "수치 토크나이저 파라미터 w(94×96)·b(94×96) 뿐",
   "안_닿는_곳": "nn.Linear(m.*) 는 파이토치 기본 U(±1/√fan_in),"
                 " nn.Embedding(ce.*) 는 파이토치 기본 N(0,1) — 실측 ce_sd ≈ 1.00",
   "호출_순서": "__init__ 이 w·b 생성 → ce 임베딩 → Sequential 구성 → reset_tokenizer."
                " 이 순서가 난수 소비 순서를 정하므로 재현 게이트가 이것까지 검증한다"},
  "실측_init_stats": None},
 "🔴읽을_수_없는_것": {
  "무엇": "**팀원 배포 딥 팔이 실제로 쓴 초기화**",
  "왜": "배포 패치팩 `token-MLP_deep_patch.zip` 은 추론 아티팩트만 담는다"
        "(deepf_s*.npz 5개 · prep.json · deep_infer.py · script.py · v6_infer.py)."
        " **학습 코드가 없다.**",
  "prep.json_키_전수": prep_keys,
  "init_필드_존재": False,
  "그럼_ftt_는_어디서_왔나":
    "exp_089 의 probe 17 fit 이 randn / randn_w / ones0 / ftt 넷을 굽고"
    " ① 접힌 지문(A·c0·T)의 배율 ② 팀원 5시드평균과의 ρ 로 골랐다."
    " 즉 **측정으로 고른 재구성이지 기록에서 읽은 값이 아니다.**",
  "exp_089_probe_실측": {
    "rho_vs_mate5": {"ftt": 0.96586, "randn": 0.92139, "randn_w": 0.92376,
                     "ones0": 0.91636},
    "천장(1시드 vs 5시드평균)": 0.966148,
    "접힌_지문_배율_A(출하=1.0)": {"ftt": 1.197, "randn": 7.909,
                                   "randn_w": 8.017, "ones0": 9.970},
    "접힌_지문_배율_c0(출하=1.0)": {"ftt": 6.081, "randn": 108.600,
                                    "randn_w": 2.435, "ones0": 2.408},
    "🔴읽히는_것": "w 쪽 배율 A 는 ftt 가 압도적으로 가깝다(1.20 vs 7.9~10.0)."
                   " 반면 b 쪽 배율 c0 는 ftt 가 6.08 로 **b=0 계열(2.41~2.44)보다 멀다**"
                   " — 배포 팔의 b 초기화가 ftt 보다 작았을 가능성이 남아 있고,"
                   " 그래서 이 실험이 `nw<sd>`(b=0) 갈래를 함께 훑는다."}},
 "🔴함의": "이 축의 '현행 기준'은 **기록이 아니라 재구성**이다."
           " 따라서 스케일 축의 argmax 가 1.0 에서 벗어나도 그것이 곧"
           " '배포 팔이 잘못 초기화됐다' 는 뜻이 아니다 — 우리 복제본 기준의 이동이다.",
}
gp = f"{B}/out/fit_GATE_ftt_s42.json"
if os.path.exists(gp):
    J = json.load(open(gp))
    INIT_PROV["🔴읽은_것"]["실측_init_stats"] = J["init_stats"]

# ── 팀원 관측 재현 여부 ────────────────────────────────────────────────────
def alone_of(nm):
    return TAB[nm]["per_step"].get("1632", {}).get("alone_recentred") if nm in TAB else None


BASE_ALONE = R["baseline_arm"]["per_step"]["1632"]["alone_recentred"]
MATE = {
 "팀원_문구": "제143이 이 방에서 직접 재빌드한 것(초기화 N(0, D^-0.5) → 833.70 → 872.39)",
 "우리가_읽은_두_가지": {
  "표준편차로": "N(0, sd = 96^-0.5 = 0.10206207)  → 지점 A_nD05",
  "분산으로": "N(0, sd = 96^-0.25 = 0.31947155)   → 지점 A_nD025"},
 "우리_면의_기준선_단독_1632": BASE_ALONE,
 "우리_면의_실측": {nm: {"alone_1632": alone_of(nm),
                        "delta_vs_N3_1632": (None if alone_of(nm) is None
                                             or alone_of("N3_identity") is None
                                             else alone_of(nm) - alone_of("N3_identity"))}
                   for nm in ("N3_identity", "A_nD05", "A_nD025", "A_nD075",
                              "A_u0p25", "A_u0p5", "A_u2", "A_u4", "A_randn")
                   if nm in TAB},
 "재현_판정": None,
}

M = {
 "exp_id": "exp_108",
 "title": "딥 팔 초기화 스케일 — exp_091~107 이 한 번도 안 훑은 축",
 "면": "exp_104 sA_mix104.py 면 그대로 (고친 f64 OOT 입력, S134)",
 "🔴결론": None,
 "🔴초기화_출처": INIT_PROV,
 "🔴팀원_관측_재현": MATE,
 "🔴판정": {"prereg": R["prereg"], "VERDICT": R["VERDICT"],
            "검출한계": R["검출한계"]},
 "🔴기준선_재추첨": R["🔴기준선_재추첨"],
 "게이트": {"재현_바이트동일": G,
            "면_재현": {"baseline_w0_rc": R["baseline_w0"]["rc"],
                        "expected": R["baseline_w0"]["expected"],
                        "gate_1632_reproduces_26.2698":
                            R["baseline_arm"]["gate_1632_reproduces_26.2698"]}},
 "출하_가능성": R["출하_가능성"],
 "지점표": TAB,
 "귀무_대조": R["귀무_대조"],
 "🔴포락선_재매개변수화_검사": R["🔴포락선_재매개변수화_검사"],
 "보충_짝지은_비교": S3,
 "baseline_arm": R["baseline_arm"],
 "missing_artifacts": R["missing_artifacts"],
}
json.dump(M, open(f"{B}/out/metrics.json", "w"), ensure_ascii=False, indent=1)
print("wrote", f"{B}/out/metrics.json")
