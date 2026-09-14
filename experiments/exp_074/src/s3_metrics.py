"""exp_074 s3 — 산출물 조립 (로컬).  서버 JSON 3종 -> metrics.json.

report 본문(결론 문장)은 결과를 본 뒤 REPORT 딕셔너리에 채운다.
숫자는 전부 서버 JSON 에서 그대로 옮기고 여기서 다시 계산하지 않는다.
"""
import json
import os
import sys

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
O = os.path.join(D, "out")
L = lambda n: json.load(open(os.path.join(O, n), encoding="utf-8"))

s0, tb, sc = L("stage0.json"), L("tables_report.json"), L("screen.json")
sp, st = L("supp.json"), L("strict.json")
REPORT = json.load(open(os.path.join(O, "report_text.json"), encoding="utf-8"))

M = {
    "exp_id": "exp_074",
    "date": "2026-08-21",
    "type": "부호 스크린. 학습 0건 · 추론 0회 · 제출물 없음.",
    "base": ("BN (LB 1115.9165956458) · tr2024 보정 프레임 253,507행 "
             "(season 2024 단독)"),
    "verdict": REPORT["verdict"],

    "stage0_shipping": s0,
    "stage1_tables": tb,
    "stage2_3_screen": sc,
    "stage4_supplementary": sp,
    "stage5_strict_boundary_rerun": st,

    "wall_clock": {
        "compute_s": {"s1_tables": tb["wall_s"], "s0_ship": s0["wall_s"],
                      "s2_screen": sc["wall_s_s2"],
                      "s4_supp": sp["wall_s_s4"], "s5_strict": st["wall_s_s5"],
                      "total": round(tb["wall_s"] + s0["wall_s"] + sc["wall_s_s2"]
                                     + sp["wall_s_s4"] + st["wall_s_s5"], 1)},
        "session_min_approx": REPORT["session_min_approx"],
        "eta_min": "45~60",
        "note": REPORT["wall_note"]},

    "report": REPORT["body"],
}
p = os.path.join(D, "metrics.json")
json.dump(M, open(p, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print(f"wrote {p}  ({os.path.getsize(p):,} bytes)")
