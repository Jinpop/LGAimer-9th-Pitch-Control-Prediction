"""exp_097 (097b 독립 판본) — metrics.json 조립."""
import json
import os
import sys

D = sys.argv[1]          # local out/ dir
OUT = sys.argv[2]

def L(name):
    p = os.path.join(D, name)
    return json.load(open(p)) if os.path.exists(p) else None

musd = L("sA_musd097b.json")
delta = L("sA_delta097b.json")
attr = L("sA_delta097b_attr.json")
sites = L("sA_sites.json")
cols = L("sB0_cols.json")
sb = L("sB_indep.json")
ledger = L("sA_ledger097b.json")
prior = L("sA_delta.json")

M = {
 "exp_id": "exp_097",
 "판본": "097b — 이 세션이 독립적으로 다시 잰 판본. 서버 ~/SungjinPark/exp097 의 앞선 판과 방법이 다르다(전 패키지 AST 계측 + f64 승격 종단 대조 vs 지점별 소스 변형).",
 "title": "부동소수 정밀도 감사 — 배포 경로 전수(stageA) + 정직면 재판독(stageB)",
 "date": "2026-08-25",
 "stageA_결론": (
   "배포 경로에 exp_096 급 float32 누산 결함은 없다. ① 여러 test 행에 걸친 float 누산이 실측 0건 "
   "② 양변이 f32 인 누산은 mate/nn_infer.py:35 GEMM 하나뿐이고 그것도 행 안(은닉·피처 축) 축약이다 "
   "③ 패키지 전체 182 지점의 f32 누산 입력을 f64 로 승격해 종단 재실행해도 245,789행 중 |Δp|>1e-6 인 행이 2개, "
   "점수 이동 상한이 0.101157점 — LB 검출한계 1.4 의 7.2%, 사전등록 문턱 +2.0 의 5.1%. 고침 후보 승격 0건."),
 "stageB_결론": (
   "고친 면에서 배포점 (OOT 1632스텝, w_deep 0.40) 이 이미 argmax 근처다. 전 격자 최고점 (1632, 0.43) 이 "
   "배포점 대비 +0.1184점, 짝지은 투수블록 부트스트랩 t 0.191. 고친 면에서 새로 유도한 전이비 0.34818 로 환산하면 "
   "LB +0.0412 — 검출한계 1.4 의 2.9%. 사전 등록 통과 0건. 🔴 다만 방법론 정정이 크다: 결함면은 딥 팔의 혼합 이득을 "
   "1632스텝에서 13.71 로 보고했는데 고친 면에서는 26.27 이다(1.92배). exp_089~095 의 딥 관련 크기 판정은 전부 그만큼 낮게 읽혔다."),
 "prereg_threshold": {
   "rule": "출하 후보 승격 = 정직면 이득 >= 배포점 + 2.0 AND 짝지은 투수블록 부트스트랩 t >= 2.0",
   "source": "experiments/exp_097/config.yaml — 발사 전 확정, 결과 보고 안 바꿨다 (S8)"},
 "parent_package": {
   "zip": "~/Downloads/submit_CF_3axis.zip",
   "sha256": "31adc7f513796d9da4763d9e2c9744116e5a0ab85d61dea6b4488bb2cfad035b",
   "candidate_key": "CH", "LB": 1140.3283619813,
   "server_copy": "a5000:~/SungjinPark/build_CJKL/pkg/PAR",
   "server_copy_검증": "script.py/aa_infer.py/mate/q_infer.py/mate/deep_infer.py/blend.json/mate/model/blend.json 6개 파일 sha256 로컬 압축해제본과 일치",
   "w_deep": 0.40},
 "measurement_frame": {"path": "a5000:~/SungjinPark/audit_deep/frames/cov", "n_rows": 245789},
 "stageA": {},
 "stageB": {},
 "artifacts_server": "a5000:~/SungjinPark/exp097b/",
}
if sites:
    M["stageA"]["A1_enumeration"] = {
      "방법": "출하 .py 18개를 AST 로 파싱해 여러 원소에 걸친 축약(reduce·GEMM·groupby·rolling·transform·agg)의 *입력 객체* 를 전부 감쌌다. 기억이 아니라 코드에서 뽑았다(G-LEDGER).",
      "n_py_files": sites["n_files"], "n_wrap_sites": sites["n_sites"],
      "n_unique_sites": len(sites["sites"]),
      "n_astype_float32_sites": len(sites["astype_f32"]),
      "astype_float32_sites": sites["astype_f32"],
      "artifact": "out/sA_sites.json"}
if ledger:
    M["stageA"]["A2_runtime_dtype"] = ledger
if delta:
    M["stageA"]["A3_end_to_end"] = {
      "설계": "같은 프레임·같은 패키지. off(항등)·trace(기록만)·promote(모든 f32 누산 입력을 f64 로) 세 판을 종단 실행하고 submission.csv 를 float_precision='round_trip' 로 읽어 비교(S21).",
      "identity_gate": "off·trace 의 submission.csv 가 무계측 baseline 과 바이트 동일해야 한다",
      "baseline_sha256": "93f67d5fae6b13ef1ac429268366d51aa1edff8c9e5b8b3aa049bc6a301a0f13",
      "K": 400309.19882517256,
      "variants": delta["variants"]}
if attr:
    M["stageA"]["A4_attribution_and_fix_candidates"] = {
      "방법": "promote 모드를 PT_ONLY 로 한 파일씩만 켜서 종단 재실행",
      "variants": attr["variants"],
      "귀속": ("전량이 mate/nn_infer.py:35 다 — promNN 이 전 패키지 promote 와 자리 수까지 같다"
              "(max|Δp| 8.889306e-05, 상한 0.101157점, >1e-6 행 2개). "
              "mate/deep_infer.py 만 켠 promDP 는 baseline 과 바이트 동일 — 딥 팔의 f32 가중치는 "
              "왼쪽 z 가 f64 라 numpy 가 이미 f64 로 누산하고 있었다. 기여 정확히 0."),
      "max|Δp|>1e-6 인 지점": ["mate/nn_infer.py:35 (해당 행 2개 / 245,789)"],
      "고침_후보로_승격된_지점": 0,
      "① 한 줄 고침 가능한가": ("가능하다 — mate/nn_infer.py:35 의 z 를 f64 로 캐스트하면 된다. "
                              "가중치 npz·동결표는 안 건드린다."),
      "② 동결표 재생성 필요한가": "아니다. 이 지점은 표를 안 만든다.",
      "③ 정직면 이득과 부호": ("측정 불가이자 측정 불필요다. cov 프레임에는 라벨이 없고, 코시-슈바르츠 상한 "
                             "|점수 이동| <= K*(m2 + 2*sqrt(m2*REF)) 가 0.101157점이라 어떤 라벨에서도 "
                             "부호와 무관하게 문턱 +2.0 에 20배 못 미친다. 정직면 부트스트랩 se 가 0.6~1.4점이라 "
                             "이 크기는 면의 분해능 아래다."),
      "🔴 고쳐도 좋다는 뜻이 아니다": ("그 f32 는 학습 때도 같은 코드로 돌았고, 하류 상수(d_adj·alpha·블렌드 가중)가 "
                                  "전부 그 출력 위에서 적합됐다. 고치면 그 상수들의 꼭지점이 옮겨간다(S97·S123). "
                                  "이득 상한 0.101점으로는 그 위험을 살 이유가 없다.")}
if musd:
    M["stageA"]["A5_frozen_constants"] = {
      "질문": "배포 딥 팔 표준화 상수(mu/sd)가 exp_096 식 f32 누산 결함을 갖는가?",
      "방법": "배포가 선언한 학습 프레임(train season<=2024, 1,475,092행)에서 f64 로 다시 계산해 94열 전수 대조. f32 누산 대조군도 함께 계산.",
      "frame": musd["frame"], "n_rows_used": musd["n_rows_used"], "n_cols": musd["n_cols"],
      "n_cols_sd_off_gt_1pct": musd["n_cols_sd_off_gt_1pct"],
      "n_cols_mu_off_gt_0.01sd": musd["n_cols_mu_off_gt_0.01sd"],
      "recipe_fit": musd["recipe_fit"],
      "season_row": [r for r in musd["rows"] if r["col"] == "season"][0],
      "주의": "f32 누산 대조군도 season sd 비가 1.0000009 로 나온다 — numpy nanmean/nanstd 는 pairwise 합이라 '배열이 f32' 만으로는 exp_096 의 13.8배가 안 생긴다. 따라서 이 검사는 레시피를 구별하지 못하고, 출하값이 f64 정답과 5.54e-05 안에서 맞는다는 것만 말한다.",
      "artifact": "out/sA_musd097b.json"}
if cols:
    M["stageB"]["B0_결함_열수"] = {
      "방법": "프레임을 다시 만들지 않고 표준화 *출력* 에서 직접 읽었다. 옳게 표준화됐다면 학습행 열 sd 가 1 이어야 한다(nan->0 대체분만큼만 내려간다).",
      "def_array": "a5000:~/SungjinPark/exp089/work/X_oot23.npy",
      "fix_array": "a5000:~/SungjinPark/exp096/work/X_f64_oot23.npy",
      "n_train_rows": cols["n_train_rows"], "n_cols": cols["n_cols"],
      "n_cols_off_gt_1pct": 2,
      "cols_off": [{"col": r["col"], "def_sd": r["def_sd"], "fix_sd": r["fix_sd"],
                    "fix_over_def": r["sd_ratio_fix_over_def"]}
                   for r in cols["rows"]
                   if abs(r["sd_ratio_fix_over_def"] - 1.0) > 0.01],
      "결론": "결함은 사실상 season 한 열이다. 표준화 뒤 sd 가 0.0726 (있어야 할 1.0 의 1/13.78) 이라 그 열이 사실상 죽어 있었다. idea_cnt_te 가 1.11% 로 두 번째이고 나머지 92열은 0.3% 안이다.",
      "artifact": "out/sB0_cols.json"}
if sb:
    M["stageB"]["B1_학습량_argmax_재판독"] = {
      "면": "exp_090 s1_curve.py 정의를 그대로 쓰되 코드는 새로 썼다(독립 구현).",
      "기준점_게이트": {"rc": sb["baseline_rc"], "expected": sb["baseline_rc_expected"],
                        "abs_diff": sb["baseline_gate_abs_diff"], "PASS": True},
      "결함면_1632_재현": {"argmax_w": sb["surfaces"]["def"]["1632"]["argmax_w"]
                           if "1632" in sb["surfaces"]["def"] else
                           sb["surfaces"]["def"][1632]["argmax_w"],
                           "argmax_gain": sb["surfaces"]["def"]["1632"]["argmax_gain"]
                           if "1632" in sb["surfaces"]["def"] else
                           sb["surfaces"]["def"][1632]["argmax_gain"],
                           "exp_090_092_기록값": 13.714309769419287},
      "deployed_point_fix": sb["deployed_point_fix"],
      "best_point_fix": sb["best_point_fix"],
      "CE_anchor": sb["CE_anchor"],
      "LB_prediction_for_best_move": sb["LB_prediction_for_best_move"],
      "PREREG": sb["PREREG"],
      "곡선": {s: {k: {kk: vv for kk, vv in v.items() if kk != "curve"}
                   for k, v in sb["surfaces"][s].items()} for s in ("def", "fix")},
      "artifact": "out/sB_indep.json"}
M["🔴 발견 (장부 후보)"] = {
 "1_행축_누산_0건": ("배포 추론 경로에서 여러 test 행을 하나로 합치는 float 누산이 실측 0건이다. "
   "245,789 이상 원소를 받는 지점은 전부 bool 행 카운트(출력용)·int8·정확한 max 이거나, "
   "float 이면 행 안 축약(GEMM 은닉축·LayerNorm 마지막축·6구종 L1) 또는 시드 축 평균/표준편차다."),
 "2_출하_패키지에_사문_파일_3개": ("root 의 v6_infer.py · v6_nn_infer.py · v6_pipeline.py 는 한 번도 import 되지 않는다. "
   "패키지 자신이 쓰는 _module_origins_root.json 이 근거다. root 의 NN 은 f32 GEMM 이지만 "
   "실행되지 않으므로 점수와 무관하다 — 정적 grep 만 하면 여기서 틀린다."),
 "3_inseason.py_는_sha_잠금이다": ("aa_infer.py:530-551 이 sha256(inseason.py) 를 "
   "model/meta.json branches.R.stacker_extra_features.module_sha256 과 assert 로 대조한다. "
   "계측 사본이 실제로 AssertionError 로 죽어서 확인했다(읽어서가 아니라 실행해서). "
   "같은 블록 578-583 이 anchor.npz sha 와 투수/타자 수도 대조한다. "
   "aa_infer.py:1118·1119·1141·1142·1148 에 weights_record·beta_record·gamma_record 대조도 있다 "
   "— S35 의 행 번호(1079-1080)는 이 판본에서 낡았다."),
 "4_결함면이_딥_팔을_1.92배_축소해_보고했다": ("1632스텝 혼합 이득 결함면 13.7143 vs 고친 면 26.2698. "
   "ρ(z_mate) 는 0.9142 -> 0.8930 으로 내려간다. exp_089~095 의 딥 관련 *크기* 판정은 전부 낮게 읽힌 값이다. "
   "🔴 그러나 이것은 측정면 정정이지 점수 기회가 아니다 — 배포 팔은 이미 깨끗한 prep.json 을 쓴다."),
 "5_S97_은_살아남되_w_곡선이_통째로_올라간다": ("스텝별 최적 w: 결함면 816:0.42 1088:0.39 1360:0.34 1632:0.36 "
   "2176:0.25 2720:0.20 → 고친 면 0.51 / 0.50 / 0.40 / 0.43 / 0.31 / 0.23. 모양(학습량↑ → 최적 w↓)은 같고 "
   "0.07~0.10 위로 옮겨간다. 배포 w=0.40 은 결함면에서는 최적(0.36) *위*로 보였는데 고친 면에서는 최적(0.43) *아래*다."),
 "6_exp_092_의_우승_무작위_16열_마스크에_결함_열이_없다": ("D5m16_rand 마스크 = NUM 인덱스 "
   "[1,6,12,17,20,51,52,56,61,62,71,75,78,82,87,89] 인데 season(0)·idea_cnt_te(41) 둘 다 없다. "
   "season 을 포함하는 마스크는 D5m83_rand·D5h2_rand47 뿐이다. → 그 +14.892 는 결함 제거가 아니었고 S104 의 해석이 유지된다."),
 "7_배열이_f32_인_것만으로는_exp_096_의_13.8배가_안_생긴다": ("numpy nanmean/nanstd 는 pairwise 합이라 "
   "1,475,092행 f32 배열에서도 season sd 비가 1.0000009 다(내 대조군 실측). exp_096 결함의 기전은 "
   "'배열이 f32' 보다 특정적이다 — 그 기전을 아직 아무도 특정하지 않았다."),
}
M["앞선_판본과의_대조"] = {
 "위치": "a5000:~/SungjinPark/exp097/ (metrics.json · runs/V0..V4,VD · out/)",
 "방법_차이": "그쪽은 지점별 소스 변형 6판(V0/VD/V1/V2/V3/V4), 이쪽은 전 패키지 AST 계측 + 선택적 f64 승격.",
 "일치": {"V4_max|dp|": 8.982219325959218e-05, "이번_prom_max|dp|": 8.889305883824639e-05,
          "V4_상한_pts": 0.1034, "이번_상한_pts": 0.101157,
          "V4_>1e-6행": 24, "이번_>1e-6행": 2,
          "해설": ("결론이 같고 크기도 같은 자릿수다. 차이는 승격 범위다 — 그쪽 V3 는 inseason 의 "
                  "반환 .astype('float32') 까지 f64 로 바꿨는데(양자화 지점), 이쪽은 *누산 입력*만 승격해 "
                  "그 양자화는 건드리지 않았다. 그래서 >1e-6 행이 22개 적다.")},
 "재확인한_것": {"기준점_게이트": "928.501461058684 를 abs_diff 0.0 으로 독립 재현",
                "결함면_1632_argmax": "w=0.36 gain 13.7143 (exp_090/092 기록값과 일치)",
                "CE_앵커_Δ": "720->1632 w=0.20 에서 10.10205 (S111 기록 10.102 재현)",
                "f64_입력_sha": "exp096/work/X_f64_oot23.npy == exp097/work/X_f64o.npy = f33de9c4db71cc33fa782c8287f4ba4241eb28b942d9c21cd1e80d43f93ba1ff",
                "결함_입력_sha": "exp089/work/X_oot23.npy == exp097/work/X_oot23.npy = aaf5cdb2f3833e61536b0fe1d8818c939e66c2653cd73e5fc8b609904e8856fa"},
 "재측정_안_한_것": "exp_095 의 바깥 딥 슬롯 v 축. 앞선 판본이 두 면 모두 argmax v=0 을 보고한다 — 내가 잰 값이 아니다.",
}
json.dump(M, open(OUT, "w"), indent=1, ensure_ascii=False)
print("written", OUT)
