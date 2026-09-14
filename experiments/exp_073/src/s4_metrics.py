"""exp_073 s4 — 산출물 조립 (로컬).  서버 JSON 4종 + 커버리지를 metrics.json 으로."""
import json
import os

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
O = os.path.join(D, "out")
L = lambda n: json.load(open(os.path.join(O, n), encoding="utf-8"))

s0, tb, sc, sp, cd = (L("stage0.json"), L("tables_report.json"),
                      L("screen.json"), L("supp.json"),
                      L("coverage_deploy.json"))

M = {
    "exp_id": "exp_073",
    "date": "2026-08-21",
    "type": "부호 스크린. 학습 0건 · 추론 0회 · 제출물 없음.",
    "base": "BN (LB 1115.9165956458) · tr2024 보정 프레임 253,507행 (season 2024 단독)",
    "verdict": "🔴 전 축 기각 (0/7). 후보 없음. 축은 출하 가능하나 실을 것이 없다.",

    "stage0_shipping": s0,
    "stage1_tables": tb,
    "stage2_screen": sc,
    "stage3_supplementary": sp,
    "deployment_coverage": cd,

    "wall_clock": {
        "compute_s": {"s0_ship": s0["wall_s"], "s1_tables": tb["wall_s"],
                      "s2_screen": sc["wall_s_s2"], "s3_supp": sp["wall_s_s3"],
                      "total": round(s0["wall_s"] + tb["wall_s"]
                                     + sc["wall_s_s2"] + sp["wall_s_s3"], 1)},
        "session_min_approx": 30,
        "eta_min": "45~60",
        "note": "ETA 하회. 계산은 19초, 벽시계 지배항은 원료 위치 확인·코드 작성·검증."},

    "report": {
        "가설": ("rel_height·rel_side 의 산포 = 릴리스 일관성 = 커맨드. "
                 "우리 피처는 전부 결과 비율이고 원인 쪽 물리량이 하나도 없다."),
        "변경점": ("data/trackman_history.csv 의 물리 계측 8종을 처음으로 사용. "
                   "571쌍 매핑(p6_verify/src/pitcher_id_map.csv, sha 90b8d376…)으로 "
                   "pitcher_trackman_id → pitcher_id 를 붙이고 season<Y 로 시즌별 "
                   "as-of 표를 만들어 7축을 BN 잔차 위에서 스크린."),

        "stage0_결론": {
            "판정": "출하 가능. 단 C단(새 피처 소스)이다 — A단이 아니다.",
            "Q1": ("가능. 이 패키지는 이미 pitcher_id 로 키를 잡는 동결 표를 싣고 "
                   "있다(model_form/pmap_pitcher_ids.npy + 값 2개, form_offset.py 가 "
                   "행별 .get(pid, unseen_ix) 로 조회). 같은 형태로 model_tm/ 표를 "
                   "추가하고 blend.json 의 axis_offsets 에 항목 하나, script.py 에 "
                   "raw 블록 하나를 넣으면 된다. **상이·추가 파일 3개** "
                   "(추가 1 = 표, 변경 2 = blend.json · script.py). "
                   "script.py 의 `ax_names == [...]`·`len(_axl) == 2` assert 를 "
                   "함께 고쳐야 한다 — 안 고치면 채점 서버에서 크래시."),
            "Q2": ("충분하다. pitcher_id 는 대회 test.csv 의 공식 입력 열이고 "
                   "cov·real5 프레임 양쪽에 존재한다. 트랙맨 쪽 키 "
                   "pitcher_trackman_id 는 표를 만들 때만 쓰이고 추론에 등장하지 않는다."),
            "Q3": ("매핑 커버리지(571판)는 tr2024·cov 프레임 행 기준 **96.49%**, "
                   "트랙맨 행 기준 92.33%. 그러나 **표 값이 실제로 유한한 행은 "
                   "78.98%** 다(Y=2024 표 = season≤2023) — 매핑에는 있어도 그 "
                   "투수의 이전 시즌 트랙맨 행이 없는 경우가 17.5%p. 이 78.98% 가 "
                   "'한 시즌 앞' 의 정직한 대리값이다. 배포 쪽(Y=2025 표, 571명)을 "
                   "cov 대역 프레임에 붙이면 96.49% 지만 그 프레임은 2024 행을 2025 로 "
                   "덮은 것이라 **2025 신인이 없다 — 상한값**이다. 실제 2025 표본 "
                   "real5 는 5행 중 1행(20%)만 덮였다(n=5, 통계적 정보 없음·경고로만). "
                   "결측 행은 오프셋 정확히 0, sentinel 충전 없음."),
            "Q4": ("무시할 수 있다. 축 블록 하나 = test.csv 2열 재독 + 행별 dict 조회 "
                   "+ 정규화 = **0.46초**(245,789행, 단독 실행 5회 중앙값). "
                   "현행 146초 → 약 146.5초.")},

        "stage1_as_of": ("Y=2024 표는 season 2019~2023 · 1,357,879행 · 투수 531명, "
                         "Y=2025 표는 2019~2024 · 1,655,472행 · 571명. 두 표의 "
                         "season 경계를 assert 로 막았고 두 표가 실제로 다름도 "
                         "assert 했다(공통 투수 531명). 물리량 8종 원자료 결측률 "
                         "0.42~0.70%."),

        "stage2_판정": {
            "통과_축": [],
            "요약": ("🔴 **0/7 통과.** 사전 등록 문턱 |t| ≥ 2 를 넘은 축이 하나도 없다. "
                     "최대는 tm_rel_sd 의 t −1.34 (본 가설 축)."),
            "t표": {k: sc["screen"][k]["t"] for k in
                    ["tm_rel_sd", "tm_rel_sd_wp", "tm_speed_trend", "tm_speed_sd",
                     "tm_break_sd", "tm_ext_mean", "tm_spin_mean",
                     "ctrl_asof_psr", "perm_shuffle", "supp_log_tm_n"]},
            "대조군_판독": ("ctrl_asof_psr **−5.45** — 스크린은 잡을 힘이 있다"
                            "(BJ 가 이미 LB 로 산 축이라 통과가 정상, '흡수 안 됨'"
                            "으로 읽지 않는다). perm_shuffle **−0.79** — 귀무 눈금 정상. "
                            "즉 이 0/7 은 스크린이 죽어서 나온 0 이 아니다."),
            "잔차화": ("통과 축이 0 이라 등록 절차상 수행 대상이 없다. 그래도 "
                       "tm_rel_sd·tm_rel_sd_wp 는 보조로 ctrl 잔차화를 돌렸고 "
                       "t −1.14 / −1.22 — 원 t 와 거의 같다. ctrl 과의 상관이 "
                       "+0.034 / +0.018 로 애초에 겹치는 게 없다."),
            "쌍상관": ("tm_rel_sd ↔ tm_rel_sd_wp **+0.9877** (사실상 같은 축 — 구종 "
                       "믹스를 빼도 릴리스 산포는 그대로다). tm_speed_sd ↔ tm_break_sd "
                       "+0.5547. 나머지는 전부 |r| < 0.23. 트랙맨 축들끼리도, "
                       "ctrl 과도 거의 직교다.")},

        "왜_닫혔는가": {
            "결론": ("(A) 신호가 데이터에 없다 — (B) 블렌드가 이미 흡수했다, 가 **아니다.**"),
            "근거_D2_원시결합": ("모델을 거치지 않은 날것에서도 0 이다. 투수 273명 "
                                 "수준에서 corr(tm_rel_sd, 2024 관측 성공률) = "
                                 "**+0.0017**(비가중) · −0.0625(투구수 가중). "
                                 "행 수준 corr(u, y) = −0.0058. 7축 전부 |r| < 0.10. "
                                 "즉 릴리스 산포가 큰 투수가 제구에 실패한다는 관계가 "
                                 "**이 데이터의 라벨과는 애초에 성립하지 않는다.**"),
            "근거_D3_부모_무관": ("부모가 이 축들을 보고 있는 것도 아니다 — "
                                  "corr(u, logit p_BN) 최대 +0.093, "
                                  "corr(raw, asof_pitcher_success_rate) 최대 +0.157. "
                                  "흡수 가설이면 이 상관이 커야 한다. 안 크다."),
            "근거_D4_희석아님": ("커버 행(200,223행)만으로 다시 재도 t −1.30 — "
                                 "전체 프레임 −1.34 와 사실상 같다. 21% 무이동 행의 "
                                 "희석이 원인이 아니다."),
            "근거_D5_검출력": ("se 3.6e-06~4.2e-06 이고 ctrl 은 같은 눈금에서 "
                               "M −2.42e-05 를 내 t −5.45 를 받았다. 트랙맨 축이 "
                               "ctrl 의 **1/5 크기**만 됐어도 통과했을 것이다."),
            "가설_잔존분": ("반증된 것은 '**투수 커리어 평균 릴리스 산포**가 제구 성공을 "
                            "가른다' 이다. 반증되지 않은 것: 같은 물리량을 ① 투수-경기/"
                            "투수-시즌 수준의 **변동**으로 쓰거나 ② 팔의 **피처**로 넣어 "
                            "다른 열과 곱하게 두는 길. 이 실험은 **전역 스칼라 오프셋 축**"
                            "으로서의 자격만 재고, 그 자격은 없다고 답한다.")},

        "결론": ("**기각.** TrackMan 물리 계측 7축 전부 사전 등록 문턱 미달(최대 |t| 1.34). "
                 "출하 경로는 실재하고 비용도 싸지만(C단·상이 파일 3개·추론 +0.46초) "
                 "실을 것이 없다. 원인은 흡수가 아니라 **부재** — 원시 상관에서도 0 이다."),

        "다음_제안": ("① 이 계열을 오프셋 축으로 다시 여는 것은 권하지 않는다"
                      "(원시 상관이 0 이므로 다른 표현으로 바꿔도 같은 곳에 닿는다). "
                      "② 굳이 트랙맨을 살리려면 커리어 평균이 아니라 **투수-경기 수준 "
                      "변동**(그날 릴리스가 평소와 얼마나 다른가)이어야 하고, 그건 "
                      "오프셋 축이 아니라 **팔의 피처**로 들어가야 한다 — 학습 있는 "
                      "실험이고 비용이 한 자릿수 배 크다. ③ +37.75 격차에 대한 기여로 "
                      "보면 이 축은 닫혔다."),

        "부호_주의": ("🔴 통과 축이 없으므로 부호를 논하지 않는다. tm_rel_sd 의 t 부호가 "
                      "−(가설과 정합)이지만 **문턱 미달이고, 피처 방향 오프셋 계열의 "
                      "프레임 부호 신탁은 1/3 이다(S15). 방향이 확정됐다고 읽지 마라.**"),
        "S1_주의": ("이 t 는 tr2024 표본내 잔차 스크린이다 — 정직면이 아니다. "
                    "점수로 환산하지 않았고 크기도 쓰지 않았다. 부호와 t 만 읽는다."),

        "스펙_이탈_명기": [
            "1. 잔차화 검사는 통과 축이 0 이라 등록 절차상 대상이 없었다. 대신 "
            "본 가설 축 2개를 보조로 ctrl 잔차화해 수치를 남겼다(D5).",
            "2. 보조 진단 4종(D1~D5)은 config 에 없는 추가 산출이다. 등록 통계가 "
            "아니며 판정에 쓰지 않았다 — '왜 닫혔는가'를 (A)/(B) 로 가르기 위한 것.",
            "3. 사전 등록한 최소 표본은 s1_tables.py 머리말에 결과를 보기 전에 "
            "적어 뒀다(sd 계열 n≥2 · 구종 내 sd 는 구종당 n≥5 · speed_trend 는 "
            "양쪽 n≥10). 그 밖의 하한(예: 200구)은 두지 않았다.",
            "4. 685쌍 매핑은 쓰지 않고 커버리지 차이만 기록했다"
            "(트랙맨 행 92.33% vs 96.63%, 프레임 행 96.49% vs 99.36%).",
            "5. real5 프레임 커버리지 20%(5행 중 1행)는 n 이 작아 통계가 아니다. "
            "그러나 **2025 진짜 행의 커버리지가 대역 프레임의 96.49% 보다 낮을 수 "
            "있다는 경고**로는 남긴다(🔴 대역 프레임 함정과 같은 계열)."],

        "게이트": [
            "G1 재사용 벡터 sha 6종 못박기 통과 (pcal_aa/S2/S1/T70 · tr2024 test.csv "
            "· exp059 directions.npz)",
            "G2 BN 부모 재구성 검증: mean(p_BN) 이 출하 target_rate 와 "
            "**|차| 0.000e+00** — 부모를 옳게 세웠다는 기계 증명",
            "G3 as-of 경계 assert: 표 season_max ≤ Y−1 · (season < Y).all() · "
            "프레임 season == {2024} 단독 · Y2024 표 ≠ Y2025 표",
            "G4 프레임 행 순서 대조: frame test.csv 의 row_id·pitcher_id 가 "
            "exp059 directions.npz 와 일치",
            "G5 귀무 순열(perm_shuffle) t −0.79 — 부트스트랩 눈금 정상",
            "G6 스레드 5종 전부 6 고정 · 워처 0개(계산 19초로 발사 불필요)"],

        "산출물": [
            "experiments/exp_073/metrics.json — 이 파일",
            "experiments/exp_073/src/ — s0_ship.py · s1_tables.py · s2_screen.py "
            "· s3_supp.py · s4_metrics.py (서버 ~/SungjinPark/exp073_tm/src/ 동일본)",
            "experiments/exp_073/out/ — stage0.json · tables_report.json · "
            "screen.json · supp.json · coverage_deploy.json · "
            "tm_table_Y2024.csv · tm_table_Y2025.csv",
            "서버 작업 디렉터리 ~/SungjinPark/exp073_tm/ (삭제하지 않음)"],
    },
}
json.dump(M, open(os.path.join(D, "metrics.json"), "w"), indent=1,
          ensure_ascii=False)
print("metrics.json 저장", os.path.join(D, "metrics.json"))
