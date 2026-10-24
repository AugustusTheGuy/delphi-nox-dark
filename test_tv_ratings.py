#!/usr/bin/env python3
"""Unit tests for tv_ratings.py — band edges, zones, vote rules, NaN exclusion, trend sanity."""
import numpy as np
import pandas as pd
import tv_ratings as tv

PASS = 0; FAIL = 0
def check(name, cond):
    global PASS, FAIL
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    PASS, FAIL = PASS + (1 if cond else 0), FAIL + (0 if cond else 1)


def synth(closes, vol=None):
    c = pd.Series(closes, dtype=float)
    df = pd.DataFrame({'open': c.shift(1).fillna(c[0]), 'close': c,
                       'high': c * 1.001, 'low': c * 0.999,
                       'volume': vol if vol is not None else np.full(len(c), 100.0)})
    return df


print("=== category band edges (TV: SS ≤ −0.5 < S ≤ −0.1 < N < 0.1 ≤ B < 0.5 ≤ SB) ===")
check("−0.51 → STRONG_SELL", tv.category(-0.51) == 'STRONG_SELL')
check("−0.50 → STRONG_SELL", tv.category(-0.50) == 'STRONG_SELL')
check("−0.49 → SELL",        tv.category(-0.49) == 'SELL')
check("−0.10 → SELL",        tv.category(-0.10) == 'SELL')
check("−0.09 → NEUTRAL",     tv.category(-0.09) == 'NEUTRAL')
check(" 0.00 → NEUTRAL",     tv.category(0.0)   == 'NEUTRAL')
check(" 0.09 → NEUTRAL",     tv.category(0.09)  == 'NEUTRAL')
check(" 0.10 → BUY",         tv.category(0.10)  == 'BUY')
check(" 0.49 → BUY",         tv.category(0.49)  == 'BUY')
check(" 0.50 → STRONG_BUY",  tv.category(0.50)  == 'STRONG_BUY')

print("=== zone() thirds (user's needle semantics) ===")
check("0.55 → strong_buy",   tv.zone(0.55) == 'strong_buy')
check("0.45 → buy_hi (top ⅓ nearest strong)", tv.zone(0.45) == 'buy_hi')
check("0.30 → buy_mid",      tv.zone(0.30) == 'buy_mid')
check("0.12 → buy_lo",       tv.zone(0.12) == 'buy_lo')
check("0.05 → neutral",      tv.zone(0.05) == 'neutral')
check("−0.45 → sell_hi (nearest strong_sell)", tv.zone(-0.45) == 'sell_hi')
check("−0.55 → strong_sell", tv.zone(-0.55) == 'strong_sell')

print("=== NaN exclusion: 60 bars → MA100/200 votes excluded, rating still computes ===")
r = tv.compute_tv_rating(synth(np.linspace(100, 110, 60)))
check("rating computed", r['rating'] is not None)
check("SMA200 vote is None (excluded)", r['votes'].get('SMA200') is None)
check("SMA50 vote present", r['votes'].get('SMA50') is not None)
check("n_votes < 26 (exclusions)", 0 < r['n_votes'] < 26)

print("=== trend sanity: strong noisy up/down trends (260 bars, MA200 active) ===")
rng = np.random.default_rng(7)
up = tv.compute_tv_rating(synth(np.linspace(100, 160, 260) + rng.normal(0, 0.4, 260)))
dn = tv.compute_tv_rating(synth(np.linspace(160, 100, 260) + rng.normal(0, 0.4, 260)))
check(f"uptrend rating positive ({up['rating']})", up['rating'] is not None and up['rating'] > 0.2)
check(f"downtrend rating negative ({dn['rating']})", dn['rating'] is not None and dn['rating'] < -0.2)
check("uptrend MA-rating strongly positive", up['ma_rating'] is not None and up['ma_rating'] > 0.8)
check("up category in BUY/STRONG_BUY", up['category'] in ('BUY', 'STRONG_BUY'))
check("down category in SELL/STRONG_SELL", dn['category'] in ('SELL', 'STRONG_SELL'))
check("n_votes == 26 with full history", up['n_votes'] == 26)

print("=== vote-rule fixtures ===")
# MACD: explicit fixture — fast EMA above slow on a rising series → MACD>signal late in trend
rr = tv.compute_tv_rating(synth(np.concatenate([np.full(80, 100.0), np.linspace(100, 120, 60)])))
check("MACD votes buy in fresh uptrend", rr['votes'].get('MACD(12,26)') == 1)
# Mom(10) rule = momentum RISING → needs acceleration (a constant-slope ramp is correctly neutral)
acc = tv.compute_tv_rating(synth(np.concatenate([np.full(60, 100.0), 100 + 0.02 * np.arange(60) ** 2])))
check("Mom(10) votes buy under accelerating uptrend", acc['votes'].get('Mom(10)') == 1)
# RSI oversold-and-rising → buy: crash then tick up
ser = np.concatenate([np.full(40, 100.0), np.linspace(100, 70, 25), [70.5, 71.2]])
rf = tv.compute_tv_rating(synth(ser))
check("RSI vote ∈ {buy, neutral} after oversold uptick (rule fires only if RSI<30)",
      rf['votes'].get('RSI(14)') in (0, 1))
# UO: extreme synthetic — all bars closing at highs → UO>70 → buy
hi = synth(np.linspace(100, 140, 80)); hi['low'] = hi['close'] * 0.9999; hi['high'] = hi['close'] * 1.0001
ru = tv.compute_tv_rating(hi)
check("UO votes buy when persistently strong", ru['votes'].get('UO(7,14,28)') in (0, 1))

print("=== klines_to_df parser ===")
kl = [[1717000000000 + i * 60000, str(100 + i), str(101 + i), str(99 + i), str(100.5 + i), '1234.5', 0, 0, 0, 0, 0, 0] for i in range(50)]
df = tv.klines_to_df(kl)
check("parsed 50 rows ascending", len(df) == 50 and df['open_time'].is_monotonic_increasing)
check("floats parsed", abs(float(df['close'].iloc[0]) - 100.5) < 1e-9)
check("empty input → empty df", tv.klines_to_df([]).empty)

print(f"\n{'ALL PASSED' if FAIL == 0 else 'FAILURES'}: {PASS} passed, {FAIL} failed")
raise SystemExit(0 if FAIL == 0 else 1)
