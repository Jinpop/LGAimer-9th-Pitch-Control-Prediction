"""exp_063 — metrics_partial + corr 를 표준 metrics.json 으로 합치고
**사전 등록된 판정 규칙을 기계적으로 적용**한다. 규칙은 config.yaml 에서 왔고
결과를 보고 바꾸지 않는다(S8).

  S_14 >= -5      -> 🟢 85열 노선 채택 권고
  -15 < S_14 < -5 -> 🟡 상관 변화로 가른다 (99열판 corr(NN,AA67)=0.9014 기준.
                        내려가면 채택, 올라가면 팀원 매핑 대기)
  S_14 <= -15     -> 🔴 팀원 trackman 매핑 대기
  ⚠️ 분할 A·B 부호가 갈리면 위 판정을 적용하지 않고 그 사실만 보고한다.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nnlib as L  # noqa: E402

OUT = f"{L.B}/out"
M = json.load(open(f"{OUT}/metrics_partial.json"))
cp = f"{OUT}/corr.json"
if os.path.exists(cp):
    M["correlations"] = json.load(open(cp))

SA = M["stage_A"]["S_14_A"]
SB = M["stage_B"]["S_14_B"]
TH = M["prereg"]["thresholds"]
GATE = M["prereg"]["gate_loss_exp062"]

sign_agree = bool(np.sign(SA) == np.sign(SB))
corr_now = None
corr_ref = 0.9013795597911097          # exp_061 실측 corr(NN99, AA67)
if "correlations" in M:
    corr_now = M["correlations"]["corr_shift_85_minus_99"]["AA67"]["nn85"]

v = {"S_14_A": SA, "S_14_B": SB,
     "sign_agree_A_B": sign_agree,
     "S_14_A_plus_deploy_gap": SA - GATE,
     "note_실질비교": ("85열은 배포 불일치가 구조상 0 이므로 exp_062 가 잰 "
                   f"{GATE}점 손실이 사라진다. 실질 비교는 S_14 - ({GATE}) "
                   f"= {SA - GATE:.5f} 이다"),
     "corr_NN_vs_AA67": {"nn85": corr_now, "nn99": corr_ref,
                         "delta": (None if corr_now is None
                                   else corr_now - corr_ref)}}

if not sign_agree:
    v["verdict"] = "판정 미적용 — 분할 A·B 부호 불일치"
    v["verdict_code"] = "SPLIT_SIGN_DISAGREE"
    v["action"] = ("config 지시대로 판정을 적용하지 않고 부호가 갈렸다는 사실을 "
                   "보고한다(exp_061 stage 3 교훈)")
elif SA >= TH["adopt"]:
    v["verdict"] = "🟢 85열 노선 채택 권고"
    v["verdict_code"] = "ADOPT_85"
    v["action"] = "팀원 대기 없이 배포 재적합으로 진행(별건, 이번 실험 범위 밖)"
elif SA <= TH["reject"]:
    v["verdict"] = "🔴 팀원 trackman 매핑 대기 — 99열로 재학습"
    v["verdict_code"] = "WAIT_MAPPING"
    v["action"] = "그때까지 NN 후보를 만들지 않는다"
else:
    lower = (corr_now is not None and corr_now < corr_ref)
    v["verdict"] = ("🟡 애매 구간 — 상관으로 가름: "
                    + ("상관이 내려갔다 -> 채택" if lower
                       else "상관이 올라갔다 -> 팀원 매핑 대기"))
    v["verdict_code"] = "AMBIGUOUS_ADOPT" if lower else "AMBIGUOUS_WAIT"
    v["action"] = ("corr(NN,AA67) 이 0.9014 아래로 내려가면 채택, 올라가면 대기 "
                   "— config 사전 등록 규칙")
M["🔴_판정"] = v

# ── 요약 블록 (report 가 그대로 읽는다) ──────────────────────────────────────
a, b = M["stage_A"]["nn85"], M["stage_B"]["nn85"]
M["headline"] = {
    "주지표": "fold_2024 valid_all (train<=2023 -> valid 2024 전체 253,507행) Brier",
    "환산": ("점수 = 100000*(1 - Brier/0.2498069). 6자리 Brier 환산은 ±0.20 불확실 "
           "— 뒤 네 자리는 측정값이 아니다"),
    "exp063_85col_4seed_logitavg": {
        "brier": a["seedavg_logit_brier"], "score": a["seedavg_logit_score"]},
    "exp061_99col_4seed_logitavg(대조군)": {
        "brier": M["stage_A"]["nn99_control_recomputed"]["seedavg_logit_brier"],
        "score": M["stage_A"]["nn99_control_recomputed"]["seedavg_logit_score"]},
    "S_14_A": SA,
    "split_B": {"85col": b["seedavg_logit_score"],
                "99col": M["stage_B"]["nn99_control_recomputed"]["seedavg_logit_score"],
                "S_14_B": SB,
                "⚠️": "분할 B 는 환산 ref 가 2024 눈금 — 분할 안의 차만 읽는다"},
}
M["fold_2024"] = {
    "brier": a["seedavg_logit_brier"],
    "score": a["seedavg_logit_score"],
    "auc": a["metrics_seedavg"]["auc"],
    "logloss": a["metrics_seedavg"]["logloss"],
    "valid_R": a["metrics_seedavg"].get("valid_R"),
    "valid_F": a["metrics_seedavg"].get("valid_F"),
    "⚠️_S7": "F 쪽 변경을 valid_F 로 판정하지 않는다 — 보조 세그먼트일 뿐"}
M["변경_변인_전량_열거"] = {
    "대조군": "exp_061 out2 파일럿 팔 (99열, 256-128-64/do0.30/lr3e-3/25ep/4시드 로짓평균)",
    "의도한_변인": ["입력 열 14개 제거 (99 -> 85, DENSE 95 -> 81)"],
    "딸려_바뀐_것": [
        f"n_params {M['repro_gate']['fit']['n_params']} -> "
        f"{a['fits'][0]['n_params']} (첫 Linear 의 입력폭이 14 줄어든 결과 — "
        "열 제거의 직접 결과이지 별개 변인이 아니다)",
        "가중치 초기화 난수 소비량이 층 크기에 비례하므로 초기값 자체가 달라진다 "
        "(같은 시드라도 동일 초기값이 아님). 배치 순서 난수(gcpu)는 동일",
        "표준화 mu/sd 는 남은 81열에 대해서만 계산 — 각 열의 값은 99열판과 동일"],
    "고정": ["frame99 캐시(sha 4종 대조 통과)", "범주 5개·edim 6", "4층 구조",
           "batch 4096 drop_last", "AdamW wd 1e-4", "BCEWithLogits",
           "OneCycle pct_start 0.3 / 25에폭 완주", "시드 42/777/2024/101",
           "전처리 산술(float32 표준화 후 nan_to_num 0)", "결정론 설정",
           "분할 A·B 마스크"],
    "코드_동일성": ("nnlib.py 는 exp_061 것과 편집 2곳뿐(작업 dir, DROP14 부록) — "
                "fit/Split/Net/load_frame 은 바이트 동일. 재현 게이트가 이를 실측 증명")}
M["caveats"] = [
    "🔴 재현 게이트는 seed 42 단독(규약). 코드 경로 동일성 증명에는 충분하나 "
    "채택 판정의 4시드는 별개다",
    "🔴 시드 산포가 크다(exp_061 실측 sd 8.2~35.8점). 앙상블 차 S_14 는 시드 4개의 "
    "짝지은 차(paired_by_seed)와 함께 읽어야 한다",
    "분할 B 는 game_type R 단독이라 valid_all 규약과 세그먼트 구성이 다르다. "
    "분할 간 점수 수준 비교 금지",
    "arms/* 5팔은 tr2024 표본 내 계측 — 상관은 읽되 그 팔들의 '점수'는 out-of-time 이 아니다",
    "이 실험은 제출물이 아니다(감사 불요). 배포 후보로 갈 때 B단 감사 필요",
]
M["server_paths"] = {
    "workdir": "~/SungjinPark/exp063_nn85/",
    "out": "~/SungjinPark/exp063_nn85/out/",
    "frame_캐시": "~/SungjinPark/exp060_nn/cache/frame99.parquet (읽기 전용 재사용)",
    "대조군_예측": "~/SungjinPark/exp061_nn/out2/preds/ (읽기 전용)"}
json.dump(M, open(f"{OUT}/metrics.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({"headline": M["headline"], "판정": M["🔴_판정"],
                  "paired": {k: {kk: vv for kk, vv in d.items() if kk != "per_seed"}
                             for k, d in M["paired_by_seed"].items()
                             if isinstance(d, dict)}},
                 indent=1, ensure_ascii=False))
