#!/usr/bin/env python3
"""
Chivalry 2 Ranking — Empirical Midpoints & Steepness (with KPM)  [NO PENALTIES]
Now also writes a JSON sidecar with the learned parameters and weights.

What’s included:
- Components: KD, KPM, W/L, EXP (hours), VERS (weapons >=1000), REV/min, BUILD/min
- Empirical fitting:
    * KD, KPM, EXP, VERS, REV/min, BUILD/min → midpoint = dataset median; steepness s maps p10→0.1, p90→0.9
    * W/L → midpoint fixed at 0.5; steepness s from p10→0.1, p90→0.9
- Weights default (sum to 1.0): KD=0.20, WR=0.20, KPM=0.20, EXP=0.20, VERS=0.10, REV=0.07, BUILD=0.03
- Outputs:
    * results.csv (scores + breakdown)
    * params.json (midpoints, steepness, percentiles, weights, provenance)

Usage:
  python chiv_ranker_empirical_json.py --input players.csv --output results.csv
  # optional: --params-out params.json
"""

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from typing import Dict, Any
import numpy as np
import pandas as pd

def logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def logit(y: float) -> float:
    return math.log(y / (1.0 - y))

def finite_series(x: pd.Series) -> pd.Series:
    return x.replace([np.inf, -np.inf], np.nan)

def pct_vals(x: pd.Series, pcts=(10, 50, 90)):
    x = finite_series(x)
    return np.nanpercentile(x.to_numpy(dtype=float), pcts)

def solve_steepness_from_p10_p90(p10: float, p90: float) -> float:
    if not np.isfinite(p10) or not np.isfinite(p90) or p10 == p90:
        return 1.0  # fallback
    return (logit(0.9) - logit(0.1)) / (p90 - p10)  # ≈ 4.394449/(spread)

def fit_logistic_params(values: pd.Series, midpoint_mode: str = "median", fixed_midpoint: float = None):
    v = finite_series(values)
    p10, p50, p90 = pct_vals(v, (10, 50, 90))
    if midpoint_mode == "fixed":
        m = float(fixed_midpoint)
    else:
        m = float(p50)
    s = solve_steepness_from_p10_p90(p10, p90)
    return {"m": m, "s": float(s), "p10": float(p10), "p50": float(p50), "p90": float(p90)}

def logistic_unit(x: float, m: float, s: float) -> float:
    if not (isinstance(x, (int, float)) or np.isscalar(x)):
        return 0.5
    if not np.isfinite(x):
        return 0.5
    return logistic((x - m) * s)

DEFAULT_WEIGHTS = {
    "kd": 0.20,
    "wr": 0.20,
    "kpm": 0.20,
    "exp": 0.20,
    "vers": 0.10,
    "rev": 0.07,
    "build": 0.03,
}

def compute_components(df: pd.DataFrame, params: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    D_safe = df["D"].replace(0, np.nan)
    minutes = (df["hours"] * 60.0).replace(0, np.nan)

    out["raw_kd"] = df["K"] / D_safe
    out["wr"] = df["Wins"] / (df["Wins"] + df["Losses"]).replace(0, np.nan)
    out["kpm"] = df["K"] / minutes
    out["hours"] = df["hours"]
    out["weapons_ge_1000"] = df["weapons_ge_1000"]
    out["rev_per_min"] = df["revives"] / minutes
    out["build_per_min"] = df["items"] / minutes

    out["kd_unit"]    = out["raw_kd"].apply(lambda x: logistic_unit(x, params["kd"]["m"], params["kd"]["s"]))
    out["wr_unit"]    = out["wr"].apply(lambda x: logistic_unit(x, params["wr"]["m"], params["wr"]["s"]))
    out["kpm_unit"]   = out["kpm"].apply(lambda x: logistic_unit(x, params["kpm"]["m"], params["kpm"]["s"]))
    out["exp_unit"]   = out["hours"].apply(lambda x: logistic_unit(x, params["exp"]["m"], params["exp"]["s"]))
    out["vers_unit"]  = out["weapons_ge_1000"].apply(lambda x: logistic_unit(x, params["vers"]["m"], params["vers"]["s"]))
    out["rev_unit"]   = out["rev_per_min"].apply(lambda x: logistic_unit(x, params["rev"]["m"], params["rev"]["s"]))
    out["build_unit"] = out["build_per_min"].apply(lambda x: logistic_unit(x, params["build"]["m"], params["build"]["s"]))

    return out

def combine_score(components: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    keys = ["kd","wr","kpm","exp","vers","rev","build"]
    s = sum(weights[k] for k in keys)
    if s <= 0:
        raise ValueError("Weights must sum to > 0")
    w = {k: weights[k]/s for k in keys}
    blended = (
        w["kd"]   * components["kd_unit"] +
        w["wr"]   * components["wr_unit"] +
        w["kpm"]  * components["kpm_unit"] +
        w["exp"]  * components["exp_unit"] +
        w["vers"] * components["vers_unit"] +
        w["rev"]  * components["rev_unit"] +
        w["build"]* components["build_unit"]
    )
    return (100.0 * blended).round(2)

def main():
    ap = argparse.ArgumentParser(description="Chivalry 2 empirical scorer with KPM and params.json output.")
    ap.add_argument("--input", required=True, help="Input players.csv")
    ap.add_argument("--output", required=True, help="Output results.csv")
    ap.add_argument("--params-out", default=None, help="Optional JSON path for learned parameters; default is OUTPUT with .json extension")

    ap.add_argument("--w-kd", type=float, default=DEFAULT_WEIGHTS["kd"])
    ap.add_argument("--w-wr", type=float, default=DEFAULT_WEIGHTS["wr"])
    ap.add_argument("--w-kpm", type=float, default=DEFAULT_WEIGHTS["kpm"])
    ap.add_argument("--w-exp", type=float, default=DEFAULT_WEIGHTS["exp"])
    ap.add_argument("--w-vers", type=float, default=DEFAULT_WEIGHTS["vers"])
    ap.add_argument("--w-rev", type=float, default=DEFAULT_WEIGHTS["rev"])
    ap.add_argument("--w-build", type=float, default=DEFAULT_WEIGHTS["build"])

    args = ap.parse_args()
    weights = {"kd":args.w_kd, "wr":args.w_wr, "kpm":args.w_kpm, "exp":args.w_exp,
               "vers":args.w_vers, "rev":args.w_rev, "build":args.w_build}

    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"Failed to read {args.input}: {e}", file=sys.stderr)
        sys.exit(1)

    required = ["player_name","K","D","Wins","Losses","hours","weapons_ge_1000","revives","items"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    # Fit parameters
    minutes = (df["hours"] * 60.0).replace(0, np.nan)

    kd_series   = df["K"] / df["D"].replace(0, np.nan)
    wr_series   = df["Wins"] / (df["Wins"] + df["Losses"]).replace(0, np.nan)
    kpm_series  = df["K"] / minutes
    exp_series  = df["hours"]
    vers_series = df["weapons_ge_1000"]
    rev_series  = df["revives"] / minutes
    bld_series  = df["items"] / minutes

    from_percentiles = {
        "kd":   {"series": kd_series,   "mid_mode":"median", "fixed":None},
        "wr":   {"series": wr_series,   "mid_mode":"fixed",  "fixed":0.5},
        "kpm":  {"series": kpm_series,  "mid_mode":"median", "fixed":None},
        "exp":  {"series": exp_series,  "mid_mode":"median", "fixed":None},
        "vers": {"series": vers_series, "mid_mode":"median", "fixed":None},
        "rev":  {"series": rev_series,  "mid_mode":"median", "fixed":None},
        "build":{"series": bld_series,  "mid_mode":"median", "fixed":None},
    }

    params = {}
    for key, spec in from_percentiles.items():
        params[key] = fit_logistic_params(spec["series"], midpoint_mode=spec["mid_mode"], fixed_midpoint=spec["fixed"])

    # Compute components and score
    comps = compute_components(df, params)
    score = combine_score(comps, weights)

    out = pd.DataFrame({
        "player_name": df["player_name"],
        "raw_kd": comps["raw_kd"].round(3),
        "kd_unit": comps["kd_unit"].round(4),
        "wr": comps["wr"].round(4),
        "wr_unit": comps["wr_unit"].round(4),
        "kpm": comps["kpm"].round(4),
        "kpm_unit": comps["kpm_unit"].round(4),
        "hours": comps["hours"].round(2),
        "exp_unit": comps["exp_unit"].round(4),
        "weapons_ge_1000": comps["weapons_ge_1000"],
        "vers_unit": comps["vers_unit"].round(4),
        "rev_per_min": comps["rev_per_min"].round(4),
        "rev_unit": comps["rev_unit"].round(4),
        "build_per_min": comps["build_per_min"].round(4),
        "build_unit": comps["build_unit"].round(4),
        "score": score
    })

    try:
        out.to_csv(args.output, index=False)
    except Exception as e:
        print(f"Failed to write {args.output}: {e}", file=sys.stderr)
        sys.exit(1)

    # Write params JSON
    params_out = args.params_out
    if not params_out:
        if args.output.lower().endswith(".csv"):
            params_out = args.output[:-4] + "json"
        else:
            params_out = args.output + ".json"

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": args.input,
        "output_csv": args.output,
        "weights": weights,
        "params": params,
        "notes": "Midpoints/steepness learned from dataset percentiles; WR midpoint forced to 0.5."
    }
    try:
        with open(params_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Failed to write {params_out}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Wrote {len(out)} rows to {args.output}")
    print(f"Wrote parameter JSON to {params_out}")
    print("Done.")

if __name__ == "__main__":
    main()
