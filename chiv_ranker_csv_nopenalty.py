#!/usr/bin/env python3
"""
Chivalry 2 Ranking (CSV edition, NO PENALTIES)

- Input: CSV with columns
    player_name,K,D,Wins,Losses,hours,weapons_ge_1000,revives,items
  Optional columns:
    M (matches)

  If M is missing, it will be computed as Wins + Losses.

- Output: CSV with per-player breakdown and final score (NO penalty term).
- Defaults:
    Weights: KD=35%, WR=25%, EXP=20%, VERS=10%, REV=7%, BUILD=3%
    KD mapping: logistic (midpoint=1.0, steepness=1.0)
    Experience: linear to 100h (0..0.6), then exponential toward 1.0 (tau=250)

Usage:
  python chiv_ranker_csv_nopenalty.py --input players.csv --output results.csv
"""

import csv
import math
import argparse
from typing import Dict, List

WEIGHT_PRESETS: Dict[str, Dict[str, float]] = {
    'default': {'kd':0.35,'wr':0.25,'exp':0.20,'vers':0.10,'rev':0.07,'build':0.03},
    'kd25_wr35': {'kd':0.25,'wr':0.35,'exp':0.10,'vers':0.15,'rev':0.07,'build':0.03},
    'kd30_wr25_exp25': {'kd':0.30,'wr':0.25,'exp':0.25,'vers':0.10,'rev':0.07,'build':0.03},
}

def logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def kd_logistic_unit(kd: float, mid: float, steep: float) -> float:
    return 1.0 / (1.0 + math.exp(-(kd - mid) * steep))

def exp_component_with_break(hours: float, break_h: float, v_break: float, tau: float) -> float:
    h = max(0.0, float(hours))
    if h <= break_h:
        return (h / break_h) * v_break
    return v_break + (1.0 - v_break) * (1.0 - math.exp(-(h - break_h) / tau))

def vers_from_count(count_ge_1000: int, mid_weapons: float, steep: float) -> float:
    return logistic((float(count_ge_1000) - mid_weapons) * steep)

def chiv_components(row: Dict[str, str], params: Dict[str, float]) -> Dict[str, float]:
    def fget(key, default=0.0):
        v = row.get(key, "")
        try:
            return float(v)
        except Exception:
            return float(default)

    K = fget('K', 0); D = fget('D', 1)
    W = fget('Wins', 0); L = fget('Losses', 0)
    M = fget('M', W + L)
    hours = fget('hours', 0)
    weapons_ge_1000 = int(fget('weapons_ge_1000', 0))
    revives = fget('revives', 0); items = fget('items', 0)

    minutes = max(hours * 60.0, 1.0)
    raw_kd = K / max(D, 1e-9)

    wr_post = (W + 10.0) / (M + 10.0 + 10.0 + 1e-9)
    wr_unit = logistic((wr_post - 0.5) * 10.0)

    exp_unit = exp_component_with_break(hours, params['exp_break_h'], params['exp_v_break'], params['exp_tau'])
    vers_unit = vers_from_count(weapons_ge_1000, params['vers_mid'], params['vers_steep'])
    rev_unit = logistic(((revives / minutes) - 0.01) * 80.0)
    build_unit = logistic(((items / minutes) - 0.01) * 80.0)

    kd_unit = kd_logistic_unit(raw_kd, params['kd_mid'], params['kd_steep'])

    return {
        'raw_kd': raw_kd,
        'kd_unit': kd_unit,
        'wr_unit': wr_unit,
        'exp_unit': exp_unit,
        'vers_unit': vers_unit,
        'rev_unit': rev_unit,
        'build_unit': build_unit
    }

def combine_score(components: Dict[str, float], weights: Dict[str, float]) -> float:
    pos = ['kd','wr','exp','vers','rev','build']
    s = sum(weights[k] for k in pos)
    norm = {k: weights[k]/s for k in pos}
    score = 100.0 * (
        norm['kd']*components['kd_unit'] +
        norm['wr']*components['wr_unit'] +
        norm['exp']*components['exp_unit'] +
        norm['vers']*components['vers_unit'] +
        norm['rev']*components['rev_unit'] +
        norm['build']*components['build_unit']
    )
    return round(score, 2)

def main():
    ap = argparse.ArgumentParser(description="Chivalry 2 CSV ranker (no penalties), logistic KD and weight presets.")
    ap.add_argument('--input', required=True, help='Input CSV path with player stats.')
    ap.add_argument('--output', required=True, help='Output CSV path for results (full breakdown).')
    ap.add_argument('--weights', default='default', choices=list(WEIGHT_PRESETS.keys()), help='Weight preset.')
    ap.add_argument('--kd-mid', type=float, default=1.0, help='KD logistic midpoint (KD where 0.5).')
    ap.add_argument('--kd-steep', type=float, default=1.0, help='KD logistic steepness.')
    ap.add_argument('--exp-break', type=float, default=100.0, help='Experience breakpoint (hours).')
    ap.add_argument('--exp-vbreak', type=float, default=0.6, help='EXP value at breakpoint.')
    ap.add_argument('--exp-tau', type=float, default=250.0, help='EXP exponential tau after breakpoint.')
    ap.add_argument('--vers-mid', type=float, default=2.0, help='Versatility midpoint (count).')
    ap.add_argument('--vers-steep', type=float, default=1.8, help='Versatility steepness.')

    args = ap.parse_args()

    weights = WEIGHT_PRESETS[args.weights].copy()
    params = {
        'kd_mid': args.kd_mid, 'kd_steep': args.kd_steep,
        'exp_break_h': args.exp_break, 'exp_v_break': args.exp_vbreak, 'exp_tau': args.exp_tau,
        'vers_mid': args.vers_mid, 'vers_steep': args.vers_steep
    }

    # Read input CSV
    with open(args.input, 'r', newline='', encoding='utf-8') as fin:
        reader = csv.DictReader(fin)
        rows_in = list(reader)

    # Compute
    results: List[Dict[str, str]] = []
    for row in rows_in:
        name = row.get('player_name', 'Player')
        comps = chiv_components(row, params)
        score = combine_score(comps, weights)
        out = {
            'player_name': name,
            'raw_kd': f"{comps['raw_kd']:.3f}",
            'kd_unit': f"{comps['kd_unit']:.4f}",
            'wr_unit': f"{comps['wr_unit']:.4f}",
            'exp_unit': f"{comps['exp_unit']:.4f}",
            'vers_unit': f"{comps['vers_unit']:.4f}",
            'rev_unit': f"{comps['rev_unit']:.4f}",
            'build_unit': f"{comps['build_unit']:.4f}",
            'score': f"{score:.2f}"
        }
        results.append(out)

    # Write output CSV
    fieldnames = ['player_name','raw_kd','kd_unit','wr_unit','exp_unit','vers_unit','rev_unit','build_unit','score']
    with open(args.output, 'w', newline='', encoding='utf-8') as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Wrote {len(results)} rows to {args.output}")

if __name__ == '__main__':
    main()
