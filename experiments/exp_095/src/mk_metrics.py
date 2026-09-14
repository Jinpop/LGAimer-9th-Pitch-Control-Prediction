"""exp_095 — out/*.json 셋에서 metrics.json 을 조립한다 (로컬 실행)."""
import json, os

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
S1 = json.load(open(f"{D}/out/s1_wv.json", encoding="utf-8"))
S1B = json.load(open(f"{D}/out/s1b_decomp.json", encoding="utf-8"))
S2 = json.load(open(f"{D}/out/s2_leftover.json", encoding="utf-8"))
G = {(g["w"], g["v"]): g for g in S1["grid"]}
WG, VG = S1["grid_spec"]["w"], S1["grid_spec"]["v"]

M = {
    "exp_id": "exp_095",
    "title": "딥 팔 가중 기하 2자유도(w_inner, v_outer) + 배포 경로 중간값 전수 조사",
    "date": "2026-08-25",
    "학습_건수": 0,
    "제출물": 0,
    "면": ("정직면 OOT-2024 — 모든 팔 season<=2023 학습, season 2024 253,507행 채점. "
          "exp_090 s1_curve.py 면을 글자 그대로 승계."),
    "🔴_LB_환산_금지": (
        "정직면 크기를 LB 점수로 환산하지 않았다. S54(실패유형 계열 19.4배)·S78(정직면 유도 "
        "축 0.0184 = 54배). 이 계열에서 눈금이 잡힌 유일한 전이비는 S111 의 0.4779 인데 "
        "그것은 '같은 w·같은 학습량에서 학습량만 옮긴' 축의 값이고 여기서 재는 것은 "
        "★w 축★ 이라 다른 축이다. 순위와 부호만 쓴다."),

    "사전_등록_판정": {
        "stage1": "기각",
        "stage2": "없음",
        "🔴": "두 문턱 모두 config.yaml 사전 등록값이고 결과를 보고 바꾸지 않았다 (S8).",
    },

    # ───────────────────────── 재현 게이트 ─────────────────────────
    "재현_게이트": {
        "G1_기준점_rc": {
            "얻은값": S1["G1_baseline"]["rc"],
            "기대값": S1["G1_baseline"]["rc_expected"],
            "abs_diff": S1["G1_baseline"]["abs_diff"],
            "PASS": S1["G1_baseline"]["PASS"]},
        "G2_딥예측_sha": {"5시드 전부 exp_090 fit_*.json 기록값과 일치": True},
        "G3_exp090_곡선_재현": S1["G3_curve_repro"],
        "stage1b·stage2 도 같은 기준점 게이트를 각자 통과": {
            "s1b_abs_diff": S1B["baseline_gate"]["abs_diff"],
            "s2_abs_diff": S2["baseline_gate"]["abs_diff"]},
        "판정": "세 게이트 전부 abs_diff 정확히 0.0 — stage1 유효",
    },

    # ───────────────────────── stage 1 ─────────────────────────
    "stage1_딥_가중_2자유도": {
        "면_식": ("z(w,v) = w_out*[(1-w)*z_mate + w*z_deep] + c_aa*z_aa + c_T70*z_T70 "
                 "+ v*z_deep   →  deep 계수 = w_out*w + v,  mate 계수 = w_out*(1-w)"),
        "격자": {"w": [min(WG), max(WG), 0.02], "v": [min(VG), max(VG), 0.02],
                "n_points": len(G)},
        "출하점_0.20_0": {
            "gain_fixed_alpha": G[(0.20, 0.0)]["gain_fixed_alpha"],
            "gain_alpha_free": G[(0.20, 0.0)]["gain_alpha_free"],
            "gain_sd_matched": G[(0.20, 0.0)]["gain_sd_matched"],
            "alpha_star": G[(0.20, 0.0)]["alpha_star"],
            "vs_baseline_boot_t": S1["paired_boot"]["ship_vs_baseline"]["t"]},
        "argmax_2자유도_고정alpha": {
            "w": S1["argmax_2dof"]["fixed_alpha"]["w"],
            "v": S1["argmax_2dof"]["fixed_alpha"]["v"],
            "gain": S1["argmax_2dof"]["fixed_alpha"]["gain_fixed_alpha"],
            "gain_alpha_free": S1["argmax_2dof"]["fixed_alpha"]["gain_alpha_free"],
            "gain_sd_matched": S1["argmax_2dof"]["fixed_alpha"]["gain_sd_matched"]},
        "🔑_2자유도_argmax_는_1자유도_v0_argmax_와_같다": {
            "차": S1["residual_vs_ship"]["2dof_minus_best_1dof_v0_fixed_alpha"],
            "뜻": ("배포 관례(alpha 고정)에서 바깥 슬롯 v 는 최적점에서 정확히 0 이다. "
                  "딥 팔에 바깥 자리를 따로 주는 것은 안쪽 w 를 올리는 것 이상을 사지 못한다.")},
        "v_는_작은_w_에서만_도움이_된다": {
            "v*>0 인 w 행": sum(1 for w in WG
                               if max(VG, key=lambda v: G[(w, v)]["gain_fixed_alpha"]) > 0),
            "전체 w 행": len(WG),
            "경계": "w <= 0.22 에서만 v* > 0 이고 w >= 0.24 부터 v* = 0"},
        "w0.20_고정_v_곡선_argmax": {
            "v": S1["cut_1dof"]["w020_v_curve_argmax"]["v"],
            "gain": S1["cut_1dof"]["w020_v_curve_argmax"]["gain_fixed_alpha"],
            "vs_ship_boot_t": S1["paired_boot"]["best_1dof_w020"]["vs_ship"]["t"]},
        "잔여_argmax_minus_출하점": S1["residual_vs_ship"],
        "짝지은_투수_블록_부트스트랩": {
            "argmax(0.36,0)_vs_출하점": S1["paired_boot"]["argmax_fixed_alpha"]["vs_ship"],
            "출하점_vs_기준점": S1["paired_boot"]["ship_vs_baseline"],
            "🔴_선택편의": S1["paired_boot"]["🔴_selection_note"]},
        "딥_5시드_잭나이프": {
            k: v for k, v in S1["jackknife_over_deep_seeds"].items() if k != "per_loo"},
        "잭나이프_해석": ("시드 잡음 se 0.8666 로 재면 잔여 2.664 는 3.07σ 다. "
                       "그러나 사전 등록 문턱이 쓰는 것은 ★투수 블록 부트스트랩★ 이고 "
                       "그 t 는 0.9466 이다. 두 se 는 다른 것을 잰다 — 전자는 '이 딥 팔로 "
                       "다시 구우면 같은 w 를 고를까', 후자는 '다른 투수 표본에서도 "
                       "w=0.36 이 w=0.20 보다 나을까'."),
        "w_곡선_v0": [{"w": w, "gain_fixed_alpha": G[(w, 0.0)]["gain_fixed_alpha"],
                      "gain_alpha_free": G[(w, 0.0)]["gain_alpha_free"],
                      "gain_sd_matched": G[(w, 0.0)]["gain_sd_matched"]} for w in WG],
        "v_곡선_w0.20": [{"v": v, "deep_coef": G[(0.20, v)]["deep_coef"],
                         "gain_fixed_alpha": G[(0.20, v)]["gain_fixed_alpha"],
                         "gain_alpha_free": G[(0.20, v)]["gain_alpha_free"],
                         "alpha_star": G[(0.20, v)]["alpha_star"]} for v in VG],
        "PREREGISTERED_VERDICT": S1["PREREGISTERED_VERDICT"],
    },

    # ───────────────────────── stage 1b ─────────────────────────
    "stage1b_alpha자유_v이득_귀속": {
        "물음": ("alpha 를 자유로 두면 argmax 가 (w 0.06, v 0.30) 으로 옮겨가고 v=0 최선보다 "
                "+2.4888 크다. 그 +2.4888 은 바깥 슬롯의 기하인가, 다른 것인가?"),
        "방법": ("z 에 r 을 넣어 우리 두 팔만 따로 줄인다: "
                "z(w,v,r) = w_out*[...] + r*(c_aa*z_aa + c_T70*z_T70) + v*z_deep"),
        "alpha자유_2자유도_최대(r=1)": S1B["ATTRIBUTION"]["alpha_free_2dof_max_r1"],
        "alpha자유_v0_최대(r_자유)": S1B["ATTRIBUTION"]["alpha_free_v0_max_r_free"],
        "v_축이_팔_재가중_너머_더_주는_것": S1B["ATTRIBUTION"]["v_axis_adds_beyond_arm_reweight"],
        "최선_w_r_v0": S1B["best_w_r_v0"],
        "실효_계수_대조": S1B["anchors"],
        "🔑_판정": ("음수(−2.1636)다. v 축의 alpha 자유 이득은 ★우리 두 팔(AA·T70)을 줄이는 것★ "
                  "이 이미 사는 것이고, 바깥 슬롯 자체의 기여가 아니다. "
                  "실효 계수가 그것을 그대로 보여준다 — 출하점 aa/mate 0.4508, "
                  "고정alpha argmax 0.5635, alpha자유 argmax 0.3836."),
        "🔴_이것은_출하_제안이_아니다": (
            "r 격자의 최선이 격자 하단 r=0.40 에 닿았다(G-EDGE). 그리고 정직면의 AA 대역은 "
            "배포보다 훨씬 약하다 — exp_083 실측: 같은 자리에 배포 18시드 AA 를 넣으면 "
            "+403.43, 정직면이 실제로 쓰는 exp_049 A1 3시드는 +3.14 (차 +281.98). "
            "그래서 이 면은 AA 몫을 줄이는 방향이면 무엇이든 크게 벌어 보인다. "
            "c_aa·c_T70 재적합 통로는 exp_078 이 잔여 +0.2546 으로 이미 닫았다(S19·S33)."),
        "v_0.60까지_연장_격자끝_검사": S1B["long_v_argmax_per_w"],
    },

    # ───────────────────────── stage 2 ─────────────────────────
    "stage2_배포경로_중간값_전수조사": {
        "축_만드는_법": S2["axis_recipe"],
        "기준점": "출하점 z_ship = root(w=0.20, v=0), abd 관례",
        "귀무_A_사전등록_iid정규_200개": S2["null_A_iid_normal"],
        "귀무_B_보조_행순열_200개": S2["null_B_row_permutation"],
        "귀무_이론값_대조": ("S22/S41 환율 K*Brier/N = 0.39074 vs 귀무 A 실측 평균 0.36657 — "
                        "하니스가 눈금 안에 있다."),
        "양성_대조가_0_을_낸다": {
            "C1_ins_off(이미 산 축)": S2["candidates"]["C1_ins_off_BOUGHT"]["vertex_gain_pts"],
            "L12c_수축편차(beta 축)": S2["candidates"]["L12c_shrunk_dev_s400"]["vertex_gain_pts"],
            "뜻": ("이미 꼭지점에 착륙한 축은 잔여 기울기가 없어야 한다(S20). 둘 다 +0.03 "
                  "아래다 — 하니스가 '산 축'과 '안 산 축'을 구분한다는 증거.")},
        "내부_교차검증": {
            "stage2_L10s_deep_vs_mate_signed_꼭지점":
                S2["candidates"]["L10s_deep_vs_mate_signed"]["vertex_gain_pts"],
            "stage1_잔여_argmax_minus_ship": S1["residual_vs_ship"]["fixed_alpha"],
            "차": abs(S2["candidates"]["L10s_deep_vs_mate_signed"]["vertex_gain_pts"]
                     - S1["residual_vs_ship"]["fixed_alpha"]),
            "뜻": ("같은 양을 두 하니스가 다르게 계산했다 — stage1 은 w 격자 탐색, "
                  "stage2 는 표준화 방향의 2차 적합. 차 0.0011 (0.04%) 로 서로를 검증한다.")},
        "후보표": {k: {kk: vv for kk, vv in v.items() if kk != "rho_vs_bought_axes"}
                 for k, v in S2["candidates"].items()},
        "PREREGISTERED_VERDICT": S2["PREREGISTERED_VERDICT"],
        "측정_불가로_남긴_것": {
            "Q셀_3분할(p_is_mid_only·p_is_rev_only·p_is_both 를 따로)": {
                "코드": "mate/q_infer.py:366,672-683 — 세 셀이 _mr 하나로 합쳐진다",
                "왜 못 쟀나": ("정직면용 Q 셀 확률 벡터가 이 서버에 없다. exp_069 qoot.py 는 "
                            "셀을 season<=2023 로 적합해 2024 를 예측하므로 원리적으로는 "
                            "out-of-time 이 맞으나, 셀 체크포인트는 팀원 환경 "
                            "(/home/claude/oot/qck)에 있었고 a5000 에는 남아 있지 않다. "
                            "다시 만들려면 stage-1 셀 부스터 4개(551·708·1652·1101 트리)를 "
                            "재적합해야 한다 — 이 실험의 '학습 없음' 범위 밖.")},
            "딥_팔_은닉층_활성": {
                "코드": "mate/deep_infer.py:44-48 — 은닉 세 블록이 계산되고 버려진다",
                "왜 못 쟀나": ("exp_090 의 OOT 딥 적합은 z 만 저장하고 가중치를 안 남겼다"
                            "(fit_K1632_oot23_s42.json 의 npz_path = None). 5회 재적합이 "
                            "필요하다(각 20.4초) — 학습이므로 범위 밖. "
                            "🔑 이것이 exp_083 에서 유일하게 살아남은 축(v6 NN nact)의 "
                            "딥 팔 판이라 ★가장 값이 있어 보이는 미측정 후보★ 다."),
                "단서": ("다만 같은 자리의 다른 축약(시드 산포·시드 편차)이 전부 귀무 아래였다 — "
                        "L1 +0.6603 · L2 +1.3143 · L2b +0.5854 (귀무 A p95 = 1.3349).")},
            "d_form_eb(beta_f=0)·p_smooth S1/S2(가중 0)·lg_wo(ft_c=0)·p_aa_gamma": {
                "성격": "계산되고 ★정확히 0★ 이 곱해지는 벡터들 — 산 뒤 0 으로 착륙한 축이다",
                "근거": ("blend.json beta_f=0.0 · smooth_weights S1=S2=0.0 · "
                        "failtype_offset.c=0.0 · anchor_mode=null. lg_wo 는 exp_082 가 "
                        "이미 c 축으로 측정해 0 에 착륙시켰다(S54).")},
            "C2_asof_psr": ("이미 산 축(axis_offsets[1] ctrl_asof_psr)이지만 2024 행 81개"
                          "(0.032%)가 NaN 이라 유한성 필터에서 빠졌다. 배포 코드는 "
                          "script.py:477 _fin 마스크로 같은 행을 무이동 처리한다."),
        },
        "exp_083_과의_관계": (
            "exp_083 이 갈래 A(AA 셀)·B(EBM 항)·C(leaf 경로)·D(v6 NN 은닉)·E(시드 산포)·"
            "F(클립·백오프 마스크) 를 이미 닫았다. 이번에 새로 연 것은 ① 딥 팔 계열 전체"
            "(exp_083 당시 존재하지 않았다) ② 갈래 F 를 ★≤2023 표★ 로 다시 잰 것"
            "(exp_083 은 tr2024 프레임의 개인 셀 적중률 100% 때문에 못 쟀다 — S32/S47). "
            "둘 다 귀무 아래다."),
    },

    "🔴_계보_정정": {
        "관측": ("~/Downloads/submit_CF_3axis.zip 은 후보 CF 가 아니라 ★후보 CH★ 다. "
                "blend.json 의 candidate = 'CH' · changed_vs_CG_aa76 가 3축을 적고 있고, "
                "mate/model/blend.json 의 w_deep = ★0.4★ · AA R 분기 base 76열 6시드 · "
                "aux_offsets[1] aa_seed_sd18 u = 0.0 다. 파일 이름이 낡았다."),
        "왜_중요한가": ("config 가 전제한 '출하 w_deep = 0.20' 은 ★CF 기준★ 이다. "
                    "그 뒤 후보 CH 가 이미 0.40 으로 옮겨 놨고, 근거는 팀원 제126계열의 "
                    "f24 완전코어 w-곡선이다."),
        "우리_정직면과의_대조": {
            "팀원 f24 완전코어(2에폭 딥 팔)": {"0.20": 8.86, "0.30": 11.24, "0.40": 12.24,
                                        "0.45": 12.24, "0.50": 11.89, "0.60": 10.16},
            "우리 정직면 abd(6에폭 딥 팔, v=0)": {
                str(w): G[(w, 0.0)]["gain_fixed_alpha"]
                for w in (0.20, 0.30, 0.36, 0.40, 0.50)},
            "🔑": ("면이 다르고 팔이 다른데(2에폭 vs 6에폭) ★내부 정점이 0.36~0.45 로 겹친다★. "
                  "곡선 모양도 같다 — 0.20 에서 정점까지 +2.7~3.4, 정점 넘어 0.50 에서 "
                  "−2.2~−0.4. 두 면이 독립으로 같은 자리를 가리킨다."),
            "🔴_그래도": ("우리 사전 등록 문턱은 미달이다(t 0.9466). 정직면은 w=0.20 과 "
                        "w=0.36 을 투수 표본 재추출 수준에서 구별하지 못한다. "
                        "'위치는 두 면이 합의하나 크기는 우리 면이 확증하지 못한다'가 정확한 문장."),
        },
    },

    "산출물": {
        "s1_wv.json": "2자유도 격자 416점 · 부트스트랩 · 잭나이프 (서버 ~/SungjinPark/exp095/out/)",
        "s1b_decomp.json": "alpha 자유 v 이득 귀속 (w,r) 격자 · v 0.60 연장 스캔",
        "s2_leftover.json": "후보 21개 · 귀무 A/B 각 200개",
        "서버_경로": "a5000:~/SungjinPark/exp095/{src,out,logs}",
    },
    "실행시간": {
        "s1_wv_sec": S1["wall_sec"], "s1b_decomp_sec": S1B["wall_sec"],
        "s2_leftover_sec": S2["wall_sec"],
        "비고": ("학습 0건. 전부 기존 산출물 위의 읽기 전용 계산. "
               "동시 레인 1개로 돌렸다 — 같은 서버에서 build_CJKL 게이트가 돌고 있었다(load 12)."),
    },
}
json.dump(M, open(f"{D}/metrics.json", "w"), ensure_ascii=False, indent=1)
print("wrote metrics.json",
      os.path.getsize(f"{D}/metrics.json"), "bytes")
print(json.dumps(M["사전_등록_판정"], ensure_ascii=False))
