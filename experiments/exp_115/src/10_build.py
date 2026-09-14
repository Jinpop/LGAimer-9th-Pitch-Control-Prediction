#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp_115 1단 — sp_pit_ball3 측정용 패키지 조립.  후보 CZ 와 ★똑같은 코드 경로★.
   🔴 이것은 측정이다.  제출 후보가 아니다."""
import hashlib, json, os, shutil, sys
import numpy as np, pandas as pd

H = "/home/<user>/SungjinPark"; B = f"{H}/build_pb3_probe"
PKG, WORK = f"{B}/pkg", f"{B}/work"
FRAME = f"{H}/build_AC/work/tr2024"
sys.path.insert(0, f"{PKG}/PAR")
import splits as SPL

AXIS_NAME = "sp_pit_ball3"; COL = "sp_pit_ball3"
U = -0.007                      # 형제 pit_bhand 출하 걸음 0.014 의 절반 — CZ 와 같은 규약

cal = pd.read_csv(f"{FRAME}/test.csv", encoding="utf-8-sig",
                  usecols=["row_id", "pitcher_id", "batter_id", "pitcher_hand",
                           "batter_hand", "balls_before", "strikes_before", "game_type"])
T = SPL.load(f"{PKG}/PAR/model/deploy_splits.npz")
S = SPL.apply(cal, T)
v = S[COL].to_numpy(np.float64)
CENTER = float(v.mean()); SCALE = float(v.std())
print(f"{COL}: center={CENTER!r} scale={SCALE!r} zero={int((v==0).sum())}/{len(v)}")

SRC = open(f"{PKG}/PAR/script.py", encoding="utf-8").read()
REP = [
 (chr(39)*0 + "    assert len(_axl) == 3, \"CN은 축 오프셋을 정확히 3개 쓴다 (제154계열: sp_pit_split 추가)\"\n"
  "    assert ax_names == [\"beta_ins_off\", \"ctrl_asof_psr\", \"sp_pit_split\"], \\\n",
  "    assert len(_axl) == 4, \"PB3 측정: 축 오프셋 4개\"\n"
  "    assert ax_names == [\"beta_ins_off\", \"ctrl_asof_psr\", \"sp_pit_split\",\n"
  "                        \"" + AXIS_NAME + "\"], \\\n"),
 ("        # --- apply all three, row by row ----------------------------------\n"
  "        _rawmap = {\"beta_ins_off\": _raw_ins, \"ctrl_asof_psr\": _raw_psr,\n"
  "                   \"sp_pit_split\": _raw_sp}\n",
  "        # --- axis 4/4 `" + AXIS_NAME + "` — exp_115 측정 ------------------\n"
  "        _spvb = _spA[\"" + COL + "\"].to_numpy(np.float64)\n"
  "        _raw_spb = np.where(_pos >= 0, _spvb[np.maximum(_pos, 0)], np.nan)\n"
  "        print(f\"  sp 4축 표: 미지 주체(표값 0) 행=\"\n"
  "              f\"{int(np.count_nonzero(_spvb == 0.0))}/{len(_tdf)}\"\n"
  "              f\"  행 대응 실패={int((_pos < 0).sum())}행\")\n"
  "        # --- apply all four, row by row -----------------------------------\n"
  "        _rawmap = {\"beta_ins_off\": _raw_ins, \"ctrl_asof_psr\": _raw_psr,\n"
  "                   \"sp_pit_split\": _raw_sp, \"" + AXIS_NAME + "\": _raw_spb}\n"),
]
NEW = SRC
for i, (a, b) in enumerate(REP):
    assert NEW.count(a) == 1, (i, NEW.count(a))
    NEW = NEW.replace(a, b)
assert NEW != SRC

d = f"{PKG}/PB3"
if os.path.isdir(d): shutil.rmtree(d)
shutil.copytree(f"{PKG}/PAR", d, symlinks=False,
                ignore=shutil.ignore_patterns("__pycache__", "data", "output",
                                              "_module_origins.json",
                                              "_module_origins_root.json"))
open(f"{d}/script.py", "w", encoding="utf-8", newline="").write(NEW)
bl = json.load(open(f"{d}/blend.json", encoding="utf-8"))
bl["axis_offsets"] = list(bl["axis_offsets"]) + [{
  "name": AXIS_NAME, "u": U, "center": CENTER, "scale": SCALE,
  "fitted_at_inference": False,
  "frozen_from": "TRAIN 2024 calibration frame — exp_115 MEASUREMENT ONLY, not a submission",
  "raw_definition": "./model/deploy_splits.npz axis bat/pit cell " + COL}]
bl["candidate"] = "PB3_MEASUREMENT_NOT_A_SUBMISSION"
open(f"{d}/blend.json", "w", encoding="utf-8", newline="").write(
    json.dumps(bl, ensure_ascii=False, indent=1))
json.dump({"axis": AXIS_NAME, "col": COL, "u": U, "center": repr(CENTER),
           "scale": repr(SCALE), "n_zero_cal": int((v == 0).sum()),
           "n_cal": int(len(v)), "raw_absmax": float(np.abs(v).max())},
          open(f"{WORK}/build10.json", "w"), ensure_ascii=False, indent=1)
print("BUILD_OK pkg/PB3")
