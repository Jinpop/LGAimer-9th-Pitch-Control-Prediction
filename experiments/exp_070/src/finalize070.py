"""exp_070 산출물 조립 — 서버 metrics_core.json + followup.json → 로컬 metrics.json.

report 는 별도 JSON(report_body.json)에서 읽어 합친다 (이 환경은 .md 작성 차단).
"""
import json
import sys

CORE, FUP, RPT, OUTP = sys.argv[1:5]
m = json.load(open(CORE))
f = json.load(open(FUP))
r = json.load(open(RPT))

m["followup_지휘자지시_3건"] = f
m["report"] = r
# 표준 지표 블록을 맨 앞으로 오게 재구성
order = ["exp_id", "type", "report", "fold_2024", "fold_2023", "fold_2022",
         "train_time_min", "inference_time_sec", "wall_clock_min"]
out = {k: m[k] for k in order if k in m}
out.update({k: v for k, v in m.items() if k not in out})
json.dump(out, open(OUTP, "w"), indent=1, ensure_ascii=False)
print("wrote", OUTP, len(json.dumps(out)), "bytes")
