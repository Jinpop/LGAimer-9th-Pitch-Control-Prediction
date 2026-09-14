"""로컬: out/s1b_screen3.json + out/s1b_checks.json → metrics.json 조립."""
import json, os, sys

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = json.load(open(f"{D}/out/s1b_screen3.json", encoding="utf-8"))
K = json.load(open(f"{D}/out/s1b_checks.json", encoding="utf-8"))
E = json.load(open(f"{D}/out/s1b_extra.json", encoding="utf-8"))
P = json.load(open(f"{D}/../exp_103/out/s1_screen2.json", encoding="utf-8"))

c = S["contrasts"]
v = S["VERDICT"]
arms = S["arms"]

m = {
    "exp_id": "exp_103b",
    "title": "exp_103 의 3열 재선별 — 준중복 14열을 걷어내고 안 겹치는 3열만 단독 측정",
    "date": "2026-08-26",
    "parent": "exp_103",
    "결론": None,          # 아래에서 채운다
    "prereg_threshold": {
        "rule": "② > ① AND ② > ③(null_bat3, batter_id 안 순열), 둘 다 짝지은 투수 블록 부트스트랩 t >= 2.0",
        "source": "experiments/exp_103b/config.yaml — exp_103 과 동일 문턱, 결과 보고 안 바꿨다 (S8)",
        "pass": v["pass"],
    },
    "면": {
        "harness": "GPU CatBoost, ≤2023(1,116,277행) 학습 → 2024(253,507행) 평가",
        "hyper": {k: S[k] for k in ("iters", "depth", "lr", "seeds")},
        "identical_to": "exp_103 src/s1_screen2.py (RandomState 소비 순서까지 동일)",
        "mask_season_check": S["mask_season_check"],
        "ref_brier": 0.2498069,
    },
    "선택한_3열": {
        "cols": S["sel_cols"],
        "왜_이_셋": "기존 94열과의 max|ρ| 가 가장 낮은 셋. 나머지 14열은 ins_b_*·sp_bat_phand·"
                   "asof_batter_* 의 준중복(0.80~1.00)이라 뺐다.",
        "source_matrix": S["source_matrix"],
        "source_matrix_sha16": S["source_matrix_sha16"],
        "새_피처_제작": "없음 — exp_103 batform2.npy 에서 3열 슬라이스만 했다",
        "corr_vs_existing94": S["corr_vs_existing94"],
        "corr_among_the_3": S["corr_among_the_3"],
    },
    "arms_score": {k: arms[k]["score"] for k in arms},
    "arms_brier": {k: arms[k]["brier"] for k in arms},
    "arms_per_seed": {k: arms[k]["per_seed_score"] for k in arms},
    "arms_fit_sec": {k: arms[k]["fit_sec"] for k in arms},
    "base_재사용": {k: arms["base"][k] for k in
                  ("reused_from", "reused_sha16", "reuse_justification")},
    "rand3_cols": S["rand3_cols"],
    "contrasts": c,
    "VERDICT": v,
    "⚠️_이_화면의_검출한계_보고된_부트스트랩": S["detection_limit"],
    "🔴_실제_검출한계": E["🔴_실제_검출한계"],
    "3열_vs_17열_직접대조": E["contrast_3col_vs_17col"],
    "전체_판본_점수": E["scores"],
    "importance_seed42_form3": arms["form3"]["importance_seed42"],
    "보조확인": K,
    "exp_103_대조": {
        "17열_판본": {
            "form_minus_base": [P["contrasts"]["form_minus_base"]["delta_score"],
                                P["contrasts"]["form_minus_base"]["t"]],
            "form_minus_null_row": [P["contrasts"]["form_minus_null_row"]["delta_score"],
                                    P["contrasts"]["form_minus_null_row"]["t"]],
            "form_minus_null_bat": [P["contrasts"]["form_minus_null_bat"]["delta_score"],
                                    P["contrasts"]["form_minus_null_bat"]["t"]],
        },
        "arms_score": {k: P["arms"][k]["score"] for k in P["arms"]},
        "base_동일": "exp_103b ① 은 exp_103 ① 의 예측 벡터를 그대로 썼다 — 두 실험의 기준선은 같은 수다.",
    },
}
m["결론"] = (
    "🔴 **기각 — 사전 등록 문턱 미달.** ② − ① = −7.820 (t −0.623) 로 **여전히 음수**이고, "
    "② − ③(batter_id 안 순열 귀무) = +9.532 (t +0.665) 는 부호가 양수로 뒤집혔으나 t 가 문턱의 "
    "3분의 1이다. 두 조건 모두 t >= 2.0 을 못 넘었다. 문턱을 낮추지 않았다 (S8). "
    "🟢 **다만 감사의 진단은 방향으로 지지된다**: 같은 ① base 위에서 3열 판본이 17열 판본보다 "
    "+21.122 (t +1.674, 부트스트랩 96.0% 가 양수) 낫고, 귀무 대비 부호가 −30.314 → +9.532 로 "
    "뒤집혔다. 14열 준중복이 손해의 주원인이라는 진단과 부합한다. "
    "🔴🔴 **그러나 이 실험의 진짜 산출은 따로 있다 — 이 화면은 3열 크기의 효과를 측정할 능력이 없다.** "
    "신호가 없도록 만든 개수 맞춤 귀무 판본 5개가 기준선 대비 −18.63 ~ +11.44 로 흩어진다"
    "(최대-최소 30.07, sd 12.91). 짝지은 투수 블록 부트스트랩은 시드를 고정한 채 행만 재표집하므로 "
    "이 적합-간 잡음을 한 톨도 못 잡는다 — 보고된 boot_sd 13.36 은 아래로 치우친 값이고, "
    "합치면 1σ ≈ 18.58, t>=2.0 이 요구하는 효과는 약 **37점**이다. "
    "**따라서 '3열은 값이 없다'가 아니라 '이 화면으로는 못 본다'가 옳은 판독이다.**")
m["다음_제안"] = {
    "🔴_이_화면을_그대로_또_쓰지_마라": "1σ 18.6점 화면에서 3열급 효과를 판정하려는 시도는 "
        "몇 번을 반복해도 같은 결과를 낸다. 다음에 이 계열을 재려면 **시드 수를 늘려 적합-간 "
        "잡음부터 줄여야** 한다 — 판본당 3시드에서는 못 본다.",
    "값싼_대안": "같은 GPU 예산(12 fit)을 판본 4개 × 3시드가 아니라 **판본 2개(① base · ② form3) "
        "× 6시드**에 쓰면 적합-간 잡음이 √2 로 줄고, 남는 fit 으로 귀무 1개를 6시드로 붙일 수 있다. "
        "다만 그래도 1σ 가 13점대라 여전히 26점을 요구한다 — **이 계열은 로컬 화면으로 못 사고 "
        "LB 로 사야 하는 크기일 가능성이 높다.**",
    "⚠️_기대치_정직하게": "팀 원장의 팔 손질 전이율 10~16% 를 적용하면, 설령 3열이 로컬 +20 을 "
        "낸다 해도 LB 는 +2~3 이다. 검출한계 37점과 겹쳐 놓으면 **이 축은 로컬에서 값을 "
        "증명할 수 없는 크기**다. 슬롯을 쓸지는 그 전제 위에서 판단해야 한다.",
    "🟢_남는_사실": "3열은 기존 94열과 실제로 안 겹친다(max|ρ| 0.048 / 0.170 / 0.269, 서로도 "
        "|ρ| <= 0.031). 중요도 순위도 99열 중 31·33·48위로 바닥이 아니다. 구조적으로는 "
        "새 방향이 맞다 — 못 잰 것이지 겹친 것이 아니다.",
}
json.dump(m, open(f"{D}/metrics.json", "w"), ensure_ascii=False, indent=1)
print("wrote metrics.json")
print(json.dumps({"arms": m["arms_score"], "contrasts": c, "VERDICT": v},
                 ensure_ascii=False, indent=1))
