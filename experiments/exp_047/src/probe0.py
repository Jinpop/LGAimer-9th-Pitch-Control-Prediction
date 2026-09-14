"""exp_047 probe 0 - structural facts needed to DESIGN stage 1 correctly.
Measurement only, train.csv, zero fits."""
import numpy as np, pandas as pd, json, os
D = os.path.expanduser("~/SungjinPark/data/train.csv")
use = ['season','game_type','pitcher_id','batter_id','asof_pitcher_n',
       'asof_pitcher_success_rate','asof_pitcher_reverse_rate','asof_pitcher_middle_rate',
       'asof_pitcher_ball_rate','asof_pitcher_strike_rate','asof_batter_n',
       'asof_batter_success_rate','asof_batter_middle_rate','asof_pitcher_pitchmix_n',
       'asof_pitcher_fastball_rate','asof_pitcher_breaking_rate','asof_pitcher_offspeed_rate',
       'control_success']
df = pd.read_csv(D, usecols=use, encoding='utf-8-sig')
out = {"shape": list(df.shape), "season_monotonic": bool(df.season.is_monotonic_increasing)}
for ent, ncol in (('pitcher_id','asof_pitcher_n'),('pitcher_id','asof_pitcher_pitchmix_n'),
                  ('batter_id','asof_batter_n')):
    g = df.groupby(ent, sort=False)[ncol]
    st = (g.shift(-1) - df[ncol]).dropna()
    out[f"step_{ent}_{ncol}"] = {str(k): int(v) for k, v in st.value_counts().head(6).items()}
    out[f"zero_pct_{ncol}"] = round(100*float((df[ncol]==0).mean()), 5)
    out[f"max_{ncol}"] = float(df[ncol].max())
mix = ['asof_pitcher_fastball_rate','asof_pitcher_breaking_rate','asof_pitcher_offspeed_rate']
s = df[mix].sum(axis=1)
out["pitchmix_sum"] = {"min": float(s.min()), "max": float(s.max()),
                       "mean": float(s.mean()),
                       "pct_within_1e-6_of_1": round(100*float((abs(s-1) < 1e-6).mean()), 5),
                       "n_nan": int(s.isna().sum())}
out["nan_pct"] = {c: round(100*float(df[c].isna().mean()), 5) for c in use if c.endswith('rate')}
out["pitchmix_n_eq_pitcher_n_pct"] = round(100*float((df.asof_pitcher_pitchmix_n==df.asof_pitcher_n).mean()), 5)
# denominator check: is round(pitchmix_n*rate) integral?
for c in mix:
    raw = df[c].to_numpy(np.float64) * df.asof_pitcher_pitchmix_n.to_numpy(np.float64)
    raw = np.where(df.asof_pitcher_pitchmix_n.to_numpy()==0, 0.0, raw)
    out[f"intgap_pitchmixn_{c}"] = float(np.nanmax(np.abs(raw-np.round(raw))))
    raw2 = df[c].to_numpy(np.float64) * df.asof_pitcher_n.to_numpy(np.float64)
    raw2 = np.where(df.asof_pitcher_n.to_numpy()==0, 0.0, raw2)
    out[f"intgap_pitchern_{c}"] = float(np.nanmax(np.abs(raw2-np.round(raw2))))
for c in ['asof_batter_success_rate','asof_batter_middle_rate']:
    raw = df[c].to_numpy(np.float64) * df.asof_batter_n.to_numpy(np.float64)
    raw = np.where(df.asof_batter_n.to_numpy()==0, 0.0, raw)
    out[f"intgap_battern_{c}"] = float(np.nanmax(np.abs(raw-np.round(raw))))
for c in ['asof_pitcher_reverse_rate','asof_pitcher_middle_rate','asof_pitcher_ball_rate','asof_pitcher_strike_rate']:
    raw = df[c].to_numpy(np.float64) * df.asof_pitcher_n.to_numpy(np.float64)
    raw = np.where(df.asof_pitcher_n.to_numpy()==0, 0.0, raw)
    out[f"intgap_pitchern_{c}"] = float(np.nanmax(np.abs(raw-np.round(raw))))
# how many seasons per pitcher (for traj coverage)
ps = df.groupby('pitcher_id')['season'].nunique()
out["pitcher_season_counts"] = {str(k): int(v) for k, v in ps.value_counts().sort_index().items()}
out["n_pitchers"] = int(df.pitcher_id.nunique()); out["n_batters"] = int(df.batter_id.nunique())
out["rows_by_season"] = {str(k): int(v) for k, v in df.season.value_counts().sort_index().items()}
print(json.dumps(out, indent=1))
