#!/usr/bin/env python3
"""sr_levels.py — multi-method support/resistance engine for Delphi Nox Dark.

Identifies the next 5 SUPPORTS below and 5 RESISTANCES above the current price, each with a
STRENGTH score (0–100%) from multi-method confluence:
  P  swing pivots (fractal highs/lows) on 1m and 1h, clustered;
  V  volume-profile nodes (high-volume price bins from 1m history);
  R  round-number levels ($0.5 / $1 grid);
  D  prior-day classic pivots (P / R1 / S1 / H / L);
  M  long MA confluence (EMA 50/100/200 on 1h — dynamic levels).
Strength = method-agreement × touch-count × volume weight × recency (normalised → %).

SHADOW/DISPLAY ONLY — consumed by the dashboard ladder + logged per trade for the validator gate.
Pure pandas/numpy; no engine imports (testable standalone, reusable by backtests).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

CLUSTER_PCT = 0.0015      # merge raw levels within 0.15% of each other
PIVOT_K = 3               # fractal half-width (bars each side)
N_OUT = 5                 # levels per side


def _fractal_pivots(df: pd.DataFrame, k: int = PIVOT_K):
    """(price, kind, idx) for local highs/lows with k bars on each side."""
    h, l = df['high'].values, df['low'].values
    out = []
    for i in range(k, len(df) - k):
        if h[i] == max(h[i - k:i + k + 1]):
            out.append((float(h[i]), 'res', i))
        if l[i] == min(l[i - k:i + k + 1]):
            out.append((float(l[i]), 'supp', i))
    return out


def _volume_nodes(df: pd.DataFrame, bins: int = 60, top: int = 8):
    """High-volume price bins (local maxima of the volume profile)."""
    lo, hi = float(df['low'].min()), float(df['high'].max())
    if hi <= lo: return []
    edges = np.linspace(lo, hi, bins + 1)
    mid = (df['high'] + df['low']) / 2.0
    vol, _ = np.histogram(mid, bins=edges, weights=df['volume'].values)
    centers = (edges[:-1] + edges[1:]) / 2.0
    idx = [i for i in range(1, bins - 1) if vol[i] >= vol[i-1] and vol[i] >= vol[i+1] and vol[i] > 0]
    idx.sort(key=lambda i: -vol[i])
    vmax = vol[idx[0]] if idx else 1.0
    return [(float(centers[i]), float(vol[i] / vmax)) for i in idx[:top]]


def _round_levels(price: float, n: int = 8):
    """$0.5-grid levels near price (psychological)."""
    base = np.floor(price * 2) / 2.0
    return [round(base + 0.5 * d, 2) for d in range(-n, n + 1)]


def _day_pivots(df_1d: pd.DataFrame):
    if df_1d is None or len(df_1d) < 2: return {}
    p = df_1d.iloc[-2]
    H, L, C = float(p['high']), float(p['low']), float(p['close'])
    piv = (H + L + C) / 3.0
    return {'P': piv, 'R1': 2 * piv - L, 'S1': 2 * piv - H, 'pdH': H, 'pdL': L}


def compute_sr_levels(df_1m: pd.DataFrame, df_1h: pd.DataFrame, df_1d: pd.DataFrame,
                      price: float) -> dict:
    """Returns {'supports': [...], 'resistances': [...]} — N_OUT each, nearest-first,
    each {'price','strength_pct','dist_pct','methods'}."""
    if not price or df_1m is None or len(df_1m) < 50:
        return {'supports': [], 'resistances': []}
    cands: list = []   # (price, weight, method, recency)
    n1 = len(df_1m)
    for px, kind, i in _fractal_pivots(df_1m):
        cands.append((px, 1.0, 'P', i / n1))
    if df_1h is not None and len(df_1h) > 10:
        nh = len(df_1h)
        for px, kind, i in _fractal_pivots(df_1h):
            cands.append((px, 2.5, 'P', i / nh))          # 1h pivots count more
        for nma in (50, 100, 200):
            ma = df_1h['close'].ewm(span=nma, adjust=False, min_periods=nma).mean()
            v = float(ma.iloc[-1]) if len(ma.dropna()) else None
            if v: cands.append((v, 1.5, 'M', 1.0))
    for px, w in _volume_nodes(df_1m):
        cands.append((px, 1.0 + 2.0 * w, 'V', 0.8))
    for px in _round_levels(price):
        cands.append((px, 0.6, 'R', 1.0))
    for name, px in _day_pivots(df_1d).items():
        cands.append((float(px), 1.8, 'D', 1.0))

    # cluster within CLUSTER_PCT
    cands.sort(key=lambda c: c[0])
    clusters: list = []
    for px, w, m, rec in cands:
        if clusters and abs(px - clusters[-1]['px']) / max(clusters[-1]['px'], 1e-9) <= CLUSTER_PCT:
            c = clusters[-1]
            c['wsum'] += w * (0.5 + 0.5 * rec)
            c['n'] += 1
            c['methods'].add(m)
            c['px'] = (c['px'] * (c['n'] - 1) + px) / c['n']   # running mean
        else:
            clusters.append({'px': px, 'wsum': w * (0.5 + 0.5 * rec), 'n': 1, 'methods': {m}})
    if not clusters:
        return {'supports': [], 'resistances': []}
    # score: weight-sum × method-agreement bonus; normalise to the max cluster
    for c in clusters:
        c['score'] = c['wsum'] * (1.0 + 0.35 * (len(c['methods']) - 1))
    smax = max(c['score'] for c in clusters) or 1.0
    # a lone round number is NOT a level — psychological levels only count when another method
    # (pivot/volume/daily/MA) confirms them (merged cluster → multi-method). Without this, the
    # dense $0.5 grid crowds the nearest-5 slots and buries the real walls.
    clusters = [c for c in clusters if c['methods'] != {'R'}]
    if not clusters:
        return {'supports': [], 'resistances': []}
    sup, res = [], []
    for c in sorted(clusters, key=lambda c: c['px']):
        entry = {'price': round(c['px'], 3),
                 'strength_pct': round(100.0 * c['score'] / smax, 1),
                 'dist_pct': round((c['px'] - price) / price * 100.0, 3),
                 'methods': ''.join(sorted(c['methods']))}
        if c['px'] < price: sup.append(entry)
        elif c['px'] > price: res.append(entry)
    sup = sorted(sup, key=lambda e: -e['price'])[:N_OUT]     # nearest support first
    res = sorted(res, key=lambda e: e['price'])[:N_OUT]      # nearest resistance first
    return {'supports': sup, 'resistances': res}


def sr_features(levels: dict, price: float) -> dict:
    """Shadow features for the validator: distances, strengths, room ratio."""
    sup, res = levels.get('supports') or [], levels.get('resistances') or []
    f = {'sr_dist_supp': None, 'sr_dist_res': None, 'sr_supp_strength': None,
         'sr_res_strength': None, 'sr_room_ratio': None}
    if sup:
        f['sr_dist_supp'] = round(abs(sup[0]['dist_pct']), 4)
        f['sr_supp_strength'] = round(sup[0]['strength_pct'] / 100.0, 4)
    if res:
        f['sr_dist_res'] = round(abs(res[0]['dist_pct']), 4)
        f['sr_res_strength'] = round(res[0]['strength_pct'] / 100.0, 4)
    if sup and res and f['sr_dist_supp']:
        f['sr_room_ratio'] = round(f['sr_dist_res'] / max(f['sr_dist_supp'], 1e-6), 4)
    return f
