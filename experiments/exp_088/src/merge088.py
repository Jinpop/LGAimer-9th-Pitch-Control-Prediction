"""exp_088 산출물 조립 — 서버 out/*.json 을 experiments/exp_088/metrics.json 으로 합친다."""
import json, os

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
O = f"{D}/out"
L = lambda n: json.load(open(f"{O}/{n}", encoding="utf-8"))

m0a, m0b, m0c, m0d = L("s0a_map.json"), L("s0b_deep.json"), L("s0c_gateA.json"), L("s0d_gateB_preview.json")
m1b, m1c, der, prep = L("s1b_eval.json"), L("s1c_extra.json"), L("derive088.json"), L("prep.json")

REF = 0.2498069
GA = m0c["GATE_A"]
M = {
 "exp_id": "exp_088",
 "title": "두 번째 딥 팔 — 배포 학습량(≤2024)으로 다시 재고, 팀원 딥 팔과의 중복도부터 판정",
 "🔴_결론": {
   "관문_A_stage0": {
     "F1_FM_4시드_vs_DEEP_4시드": GA["F1_FM_4seed"]["DEEP_4seed"]["rho"],
     "F1_FM_판정": GA["F1_FM_4seed"]["DEEP_4seed"]["verdict"],
     "F2_EMB_4시드_vs_DEEP_4시드": GA["F2_EMB_4seed"]["DEEP_4seed"]["rho"],
     "F2_EMB_판정": GA["F2_EMB_4seed"]["DEEP_4seed"]["verdict"],
     "면": "fold_2024 valid_all 253,507행, 로짓 공간, 시드 등급 일치(우리 4 ↔ 딥 4, 같은 시드 라벨)"},
   "관문_A_배포학습량_stage1": {
     "F1_FM_le2024_5시드_vs_DEEP_5시드": m1b["GATE_A_deployment_scale"]["F1_FM_le2024_5seed_vs_DEEP_5seed"],
     "판정": m1b["GATE_A_deployment_scale"]["verdict_le2024"],
     "F1_FM_le2023_5시드_대조": m1b["GATE_A_deployment_scale"]["F1_FM_le2023_5seed_vs_DEEP_5seed"]},
   "관문_B": "🔴 이 면에서 **측정 불가** — 판정하지 않았다. 아래 gate_B_why_unmeasurable 참조",
   "행동": ("F1_FM = 관문 A 통과 → stage 1 수행(배포 학습량 5시드 완료). "
            "F2_EMB = 회색(0.90~0.95) → 사전 등록 규칙대로 **학습하지 않고 정지**, 지휘자 판단 대기"),
 },
 "score_conversion": {"formula": "100000*(1 - Brier/0.2498069)", "sensitivity": "Brier 1e-6 = 0.4003점 (S3)"},

 "stage0_surface_alignment": m0a,
 "stage0_deep_logits": m0b,
 "stage0_GATE_A": m0c,
 "stage0_gateB_preview": m0d,
 "stage1_derivation": der,
 "stage1_prep": prep,
 "stage1_eval": m1b,
 "stage1_extra": m1c,

 "⭐_문턱_눈금_반례": {
   "무엇": "LB +5.0813180433 을 실제로 산 팀원 딥 팔 자신의 중복도",
   "rho_vs_partner_internal_mix": m1c["deep_arm_own_overlap"]["vs_partner_internal_mix_z_mate"],
   "rho_vs_mate_v6": m1c["deep_arm_own_overlap"]["vs_mate_v6"],
   "rho_vs_mate_Q": m1c["deep_arm_own_overlap"]["vs_mate_Q"],
   "우리_관문A_기각선": 0.95,
   "함의": ("그 팔은 우리 사전 등록 기각선 0.95 를 넘거나 붙는다. 그런데 팀 신기록을 샀다. "
            "→ 상관으로 잰 '새로움'은 LB 이득을 예측하지 못한다. 관문 A 는 무엇을 거르는 자인지 "
            "다시 정의되어야 한다"),
 },
 "gate_B_why_unmeasurable": m1b["GATE_B_record_only"]["🔴_왜_판정_불가"],

 "fold_2022": None,
 "fold_2023": None,
 "fold_2024": {
   "note": "이 실험의 유일한 면. train<=2023 -> valid 2024 valid_all 253,507행",
   "n_valid_all": 253507,
   "우리_팔_out_of_time": {
     "F1_FM_le2023_5seed": {"brier": m1b["seedavg5"]["le2023"]["brier"],
                            "score": m1b["seedavg5"]["le2023"]["score"],
                            "bss_vs_ref_0.2498069": 1.0 - m1b["seedavg5"]["le2023"]["brier"] / REF},
     "F1_FM_le2023_4seed_exp086": {"score": 727.12053, "출처": "exp_086 재사용"},
     "F2_EMB_le2023_4seed_exp086": {"score": 866.18765, "출처": "exp_086 재사용"}},
   "🔴_표본_내_비교_금지": {
     "F1_FM_le2024_5seed": {"brier": m1b["seedavg5"]["le2024"]["brier"],
                            "score": m1b["seedavg5"]["le2024"]["score"],
                            "왜": "이 253,507행이 학습에 들어 있다"},
     "DEEP_5seed": {"score": m0c["standalone_on_this_surface"]["DEEP_5seed"]["score"],
                    "왜": "팀원 딥 팔은 <=2024 전량 학습"}},
 },
 "train_time_min": {
   "prep088_le2024_assembly_sec": prep["wall_sec"],
   "le2023_s777_1fit_min": m1b["per_seed"]["le2023"]["777"]["fit_wall_min"],
   "le2024_5fit_min": [m1b["per_seed"]["le2024"][str(s)]["fit_wall_min"] for s in (42,7,2024,101,777)],
   "le2024_5fit_sum_min": round(sum(m1b["per_seed"]["le2024"][str(s)]["fit_wall_min"]
                                    for s in (42,7,2024,101,777)), 3),
   "note": ("le2024 fit 은 25에폭 전량을 돌리고 못박은 에폭의 상태를 취한다. "
            "동시 3개(wave1) + 2개(wave2), GPU 2장. 학습행 1,116,277 -> 1,475,092 (1.321배)"),
 },
 "inference_time_sec": m1c["inference_time_sec"],
 "artifacts_server": {
   "root": "~/SungjinPark/exp088/",
   "preds": ["out/preds/deep_route1_logit_seed{4,5}.npy — 팀원 딥 팔 로짓, 우리 면",
             "out/preds/deep_route1_s{42,777,2024,101,7}_logit.npy",
             "out/preds/deep_route2_cov_logit.npy — cov 대역 프레임(대조)",
             "out/preds/F1_FM_s{42,7,2024,101,777}_le2024_best.npy",
             "out/preds/F1_FM_s777_le2023_best.npy",
             "out/preds/F1_FM_le202{3,4}_seedavg5.npy"],
   "weights": "work/F1_FM_s*_le2024_best.pt (로컬 미복사)",
   "cache": "work/{Xd,codes,y,train_mask,valid_mask,gt}.npy · work/schema.json (≤2024 전량 조립)",
   "🔴": "기존 디렉터리(audit_deep·exp085·exp086·build_*)는 읽기만. 삭제·덮어쓰기 0건",
 },
}
json.dump(M, open(f"{D}/metrics.json", "w"), ensure_ascii=False, indent=1)
print("wrote", f"{D}/metrics.json", os.path.getsize(f"{D}/metrics.json"), "bytes")
