"""exp_106 — 서버 산출물을 로컬 metrics.json 으로 합친다. 로컬에서 실행."""
import json
import os

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(D, "out")


def L(name):
    p = os.path.join(OUT, name)
    return json.load(open(p)) if os.path.exists(p) else None


m = L("metrics106.json")
s7, s8, s9, s10 = L("s7_crossfit.json"), L("s8_attrib.json"), \
    L("s9_multistart.json"), L("s10_shipvalid.json")


def A(w_ebm, w_in, w_deep, w_out, alpha):
    """mate 내부 사후 상수의 진폭 배율."""
    return (1 - w_ebm) * (1 - w_in) * (1 - w_deep) * w_out * alpha


th0 = m["s0_deploy_point"]["theta_shipped"]
A_dep = A(th0["w_ebm"], th0["w_v6_inner"], th0["w_deep"], th0["w_out"], th0["alpha"])

R = {
    "exp_id": "exp_106",
    "title": "현 팔 구성에서 앙상블 가중 7축 전면 재적합 — 고친 정직면",
    "date": "2026-08-26",
    "type": "read-only refit (재학습 0 · GPU 0 · 파일 삭제 0)",
    "fold_note": ("fold_2024 (train ≤2023 → valid 2024, 253,507행 valid_all). "
                  "fold_2022·fold_2023 은 이 실험의 측정 대상이 아니다 — "
                  "재료 벡터가 fold_2024 out-of-time 예측만 존재한다(exp_069 승계)."),
    "환산식": "점수 = 100000 × (1 − Brier / 0.2498069)",
}
if m:
    R["s0_배포점_재현"] = {
        "게이트": m["s0_gates"],
        "배포점": m["s0_deploy_point"],
    }
    R["s1_1자유도_조건부_스윕"] = {
        a: {k: v for k, v in e.items() if k != "curve"}
        for a, e in m["s1_1dof"].items()}
    R["s2_동시적합"] = {
        "2차_국소해": {k: v for k, v in m["s2_quadratic"].items()
                       if k not in ("G", "L")},
        "직접_7축": m["s2_direct"],
        "직접_출하형(🔴무효)": dict(
            m["s2_direct_shippable"],
            **{"🔴무효_사유": "w_out(=루트 blend.json w_v6) 1.176 → script.py:188 "
                              "assert 0.0<=w_v6<=1.0 위반. s10 에서 다시 풀었다."}),
        "직접_8축_v_out포함": m["s2_direct_plus_v_out"],
        "부트스트랩_vs_배포점": m["s2_boot_vs_deploy"]}
    R["s3_척도_분해"] = m["s3_scale"]
    R["s4_귀무"] = m["s4_null"]
    R["s4b_적합간_sd"] = {k: v for k, v in m["s4b_fit_to_fit"].items() if k != "판본"}
    R["s5_재판독_exp095"] = m["s5_reread_exp095"]
    R["s6_직교성"] = {k: v for k, v in m["s6_orthogonality"].items()
                      if k != "rho_matrix"}
    R["판정_사전등록"] = m["판정"]
if s7:
    R["s7_교차적합_7축(출하불가족)"] = {"설계": s7["설계"], "요약": s7["요약"]}
if s8:
    R["s8_귀속"] = {"subsets": {k: {kk: vv for kk, vv in v.items()}
                                for k, v in s8.get("subsets", {}).items()},
                    "leave_one_out": {k: {"gain_pts": v["gain_pts"],
                                          "gain_sd_matched_pts": v["gain_sd_matched_pts"]}
                                      for k, v in s8.get("leave_one_out", {}).items()}}
if s9:
    R["s9_다중출발"] = {"runs": {k: {"gain_pts": v["gain_pts"]}
                                 for k, v in s9.get("runs", {}).items()},
                        "best": s9.get("best"),
                        "partial_steps": s9.get("partial_steps")}
if s10:
    R["s10_출하가능족"] = {k: v for k, v in s10.items()
                           if k not in ("crossfit_ship_valid",)}
    sv = s10["fits"]["ship_valid"]["theta"]
    A_new = A(sv["w_ebm"], sv["w_v6_inner"], sv["w_deep"], sv["w_out"], sv["alpha"])
    R["s11_이미_산_상수의_off_vertex_누수"] = {
        "설명": ("가중을 옮기면 mate 내부 사후 상수의 진폭 단위가 s 배로 바뀌고 "
                 "(S138·S141), 꼭지점은 1/s 로 옮겨간다. 상수를 그대로 두면 "
                 "u/u* = s 자리에 서고 최대이득의 (s−1)² 를 흘린다."),
        "진폭식": "A = (1−w_ebm)(1−w_v6_inner)(1−w_deep) · w_out · alpha",
        "검산_S138": {"예측_1−w_deep(0→0.40)": 0.600, "S138_실측": 0.60036},
        "A_배포": A_dep, "A_ship_valid": A_new, "s": A_new / A_dep,
        "누수분율_(s−1)^2": (A_new / A_dep - 1.0) ** 2,
        "대상_상수": "CJ+CN(league_dev 1|1) · CQ+CR(1|2) — 전부 mate/model_q/meta.json",
        "그_축들의_최대이득_LB_pts": {"1|1": 2.1728, "1|2": 1.4920, "합": 3.6648},
        "부분_걸음_f_에서의_누수": None}
    steps = R["s10_출하가능족"].get("partial_steps_ship_valid") or []
    rows = []
    for st in steps:
        f = st["frac"]
        th = {a: th0[a] + f * (sv[a] - th0[a]) for a in
              ("alpha", "w_out", "w_v6_inner", "w_ebm", "w_deep")}
        sf = A(th["w_ebm"], th["w_v6_inner"], th["w_deep"], th["w_out"],
               th["alpha"]) / A_dep
        leak = (sf - 1.0) ** 2 * 3.6648
        rows.append({"frac": f, "local_gain_pts": st["gain_pts"],
                     "s": sf, "누수_LB_pts": leak})
    R["s11_이미_산_상수의_off_vertex_누수"]["부분_걸음_f_에서의_누수"] = rows
json.dump(R, open(os.path.join(D, "metrics.json"), "w"),
          ensure_ascii=False, indent=1)
print("wrote", os.path.join(D, "metrics.json"),
      "stages:", [k for k in R if k[0] == "s" or k.startswith("판정")])
