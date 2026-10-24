#!/usr/bin/env python3
"""tv_ratings.py — exact local replication of TradingView's "Technical Rating" gauge.

TradingView's rating is a PUBLIC, DETERMINISTIC formula (they published the Pine source of the
"Technical Ratings" indicator): ~26 indicators each vote buy(+1)/neutral(0)/sell(−1); the gauge
value is the MEAN of the available votes, in [−1, +1], displayed with bands:
    StrongSell ≤ −0.5 < Sell ≤ −0.1 < Neutral < +0.1 ≤ Buy < +0.5 ≤ StrongBuy
Replicating it locally from venue candles is strictly better than scraping the TV page for the
live engine: zero staleness, no rate limits/fragility, and we get the CONTINUOUS rating value
(the "needle position"), which makes zone hypotheses ("top ⅓ of buy", etc.) exactly testable.
`tv_crosscheck.py` cross-validates this replica against the live TV page (exact-or-adjacent).

Pure numpy/pandas; no engine imports — shared by the live engine, the offline hypothesis
backtest (tv_backtest.py), and the unit tests. SHADOW/ANALYTICS ONLY — never trades.

Vote rules (faithful to TV's published Pine `calcRating*` functions):
  MAs (15): SMA/EMA 10,20,30,50,100,200 + Ichimoku Base(26) + VWMA(20) + HMA(9):
            buy if close > MA, sell if close < MA.
  RSI(14):  buy if RSI<30 and rising; sell if RSI>70 and falling.
  Stoch(14,3,3): buy if K<20 and D<20 and K crosses above D; sell if K>80,D>80,K crosses below D.
  CCI(20):  buy if CCI<−100 and rising; sell if CCI>+100 and falling.
  ADX(14):  buy if ADX>20 and +DI crossed above −DI this bar; sell on the mirror cross.
  AO:       buy on zero-line crossover up, or "saucer" (AO>0, dipped then turned up); sell mirror.
  Mom(10):  buy if rising, sell if falling.
  MACD(12,26,9): buy if MACD line > signal line; sell if below.
  StochRSI(14,14,3,3): same cross rules as Stoch, applied to stochastic of RSI.
  W%R(14):  buy if <−80 and rising; sell if >−20 and falling.
  BBPower(13): buy if uptrend (close>EMA13) and BearPower<0 and rising;
               sell if downtrend and BullPower>0 and falling.
  UO(7,14,28): buy if >70; sell if <30.
Votes with insufficient history (NaN) are EXCLUDED from the mean (TV shows '—' for them).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

BANDS = [(-1.01, -0.5, 'STRONG_SELL'), (-0.5, -0.1, 'SELL'), (-0.1, 0.1, 'NEUTRAL'),
         (0.1, 0.5, 'BUY'), (0.5, 1.01, 'STRONG_BUY')]

OSC_NAMES = ['RSI(14)', 'Stoch(14,3,3)', 'CCI(20)', 'ADX(14)', 'AO', 'Mom(10)',
             'MACD(12,26)', 'StochRSI', 'W%R(14)', 'BBPower', 'UO(7,14,28)']
MA_NAMES = ['SMA10', 'EMA10', 'SMA20', 'EMA20', 'SMA30', 'EMA30', 'SMA50', 'EMA50',
            'SMA100', 'EMA100', 'SMA200', 'EMA200', 'Ichimoku', 'VWMA20', 'HMA9']
ALL_NAMES = OSC_NAMES + MA_NAMES


# ---------- indicator primitives (vectorised; Wilder smoothing where TV uses it) ----------
def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()

def _sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()

def _rma(s: pd.Series, n: int) -> pd.Series:          # Wilder's smoothing (TV ta.rma)
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()

def _wma(s: pd.Series, n: int) -> pd.Series:
    w = np.arange(1, n + 1, dtype=float)
    return s.rolling(n, min_periods=n).apply(lambda x: float(np.dot(x, w) / w.sum()), raw=True)

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up, dn = d.clip(lower=0), (-d).clip(lower=0)
    rs = _rma(up, n) / _rma(dn, n)
    return 100 - 100 / (1 + rs)

def stoch_kd(h, l, c, n=14, k_smooth=3, d_smooth=3):
    ll, hh = l.rolling(n, min_periods=n).min(), h.rolling(n, min_periods=n).max()
    raw = 100 * (c - ll) / (hh - ll)
    k = _sma(raw, k_smooth)
    return k, _sma(k, d_smooth)

def cci(h, l, c, n=20):
    tp = (h + l + c) / 3.0
    ma = _sma(tp, n)
    md = (tp - ma).abs().rolling(n, min_periods=n).mean()
    return (tp - ma) / (0.015 * md)

def adx_di(h, l, c, n=14):
    up, dn = h.diff(), -l.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=h.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=h.index)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = _rma(tr, n)
    pdi = 100 * _rma(plus_dm, n) / atr
    mdi = 100 * _rma(minus_dm, n) / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi)
    return _rma(dx, n), pdi, mdi

def awesome_osc(h, l):
    med = (h + l) / 2.0
    return _sma(med, 5) - _sma(med, 34)

def macd_line(c, fast=12, slow=26, sig=9):
    m = _ema(c, fast) - _ema(c, slow)
    return m, _ema(m, sig)

def stochrsi_kd(c, rsi_n=14, stoch_n=14, k_smooth=3, d_smooth=3):
    r = rsi(c, rsi_n)
    ll, hh = r.rolling(stoch_n, min_periods=stoch_n).min(), r.rolling(stoch_n, min_periods=stoch_n).max()
    raw = 100 * (r - ll) / (hh - ll)
    k = _sma(raw, k_smooth)
    return k, _sma(k, d_smooth)

def williams_r(h, l, c, n=14):
    hh, ll = h.rolling(n, min_periods=n).max(), l.rolling(n, min_periods=n).min()
    return -100 * (hh - c) / (hh - ll)

def ultimate_osc(h, l, c, n1=7, n2=14, n3=28):
    pc = c.shift()
    bp = c - pd.concat([l, pc], axis=1).min(axis=1)
    tr = pd.concat([h, pc], axis=1).max(axis=1) - pd.concat([l, pc], axis=1).min(axis=1)
    def avg(n): return bp.rolling(n, min_periods=n).sum() / tr.rolling(n, min_periods=n).sum()
    return 100 * (4 * avg(n1) + 2 * avg(n2) + avg(n3)) / 7.0

def hull_ma(c, n=9):
    return _wma(2 * _wma(c, n // 2) - _wma(c, n), int(np.sqrt(n)))

def vwma(c, v, n=20):
    return (c * v).rolling(n, min_periods=n).sum() / v.rolling(n, min_periods=n).sum()

def ichimoku_base(h, l, n=26):
    return (h.rolling(n, min_periods=n).max() + l.rolling(n, min_periods=n).min()) / 2.0


# ---------- vote helpers ----------
def _v(cond_buy, cond_sell):
    """vote from two bools (NaN-safe upstream)."""
    return 1 if cond_buy else (-1 if cond_sell else 0)

def _f(s: pd.Series, i: int = -1):
    try:
        x = float(s.iloc[i]); return x if np.isfinite(x) else None
    except Exception:
        return None


def compute_tv_rating(df: pd.DataFrame) -> dict:
    """df: columns open/high/low/close/volume (ascending time). Returns the TV-style rating.
    Votes with insufficient history are excluded (None) — same as TV's '—'."""
    out = {'rating': None, 'category': 'NA', 'osc_rating': None, 'ma_rating': None,
           'votes': {}, 'n_votes': 0, 'bars': int(len(df))}
    if df is None or len(df) < 40:
        return out
    h, l, c = df['high'].astype(float), df['low'].astype(float), df['close'].astype(float)
    v = df['volume'].astype(float) if 'volume' in df else pd.Series(1.0, index=c.index)
    close = _f(c)
    votes: dict = {}

    # --- oscillators ---
    r = rsi(c); r0, r1 = _f(r), _f(r, -2)
    votes['RSI(14)'] = None if r0 is None or r1 is None else _v(r0 < 30 and r0 > r1, r0 > 70 and r0 < r1)
    k, d = stoch_kd(h, l, c); k0, d0, k1, d1 = _f(k), _f(d), _f(k, -2), _f(d, -2)
    votes['Stoch(14,3,3)'] = None if None in (k0, d0, k1, d1) else _v(
        k0 < 20 and d0 < 20 and k0 > d0 and k1 < d1, k0 > 80 and d0 > 80 and k0 < d0 and k1 > d1)
    cc = cci(h, l, c); c0, c1 = _f(cc), _f(cc, -2)
    votes['CCI(20)'] = None if c0 is None or c1 is None else _v(c0 < -100 and c0 > c1, c0 > 100 and c0 < c1)
    ax, pdi, mdi = adx_di(h, l, c)
    a0, p0, m0, p1, m1 = _f(ax), _f(pdi), _f(mdi), _f(pdi, -2), _f(mdi, -2)
    votes['ADX(14)'] = None if None in (a0, p0, m0, p1, m1) else _v(
        a0 > 20 and p1 < m1 and p0 > m0, a0 > 20 and p1 > m1 and p0 < m0)
    ao = awesome_osc(h, l); ao0, ao1, ao2 = _f(ao), _f(ao, -2), _f(ao, -3)
    votes['AO'] = None if None in (ao0, ao1, ao2) else _v(
        (ao1 <= 0 < ao0) or (ao0 > 0 and ao1 > 0 and ao0 > ao1 and ao2 > ao1),
        (ao1 >= 0 > ao0) or (ao0 < 0 and ao1 < 0 and ao0 < ao1 and ao2 < ao1))
    mom = c - c.shift(10); mo0, mo1 = _f(mom), _f(mom, -2)
    votes['Mom(10)'] = None if mo0 is None or mo1 is None else _v(mo0 > mo1, mo0 < mo1)
    mc, ms = macd_line(c); mc0, ms0 = _f(mc), _f(ms)
    votes['MACD(12,26)'] = None if mc0 is None or ms0 is None else _v(mc0 > ms0, mc0 < ms0)
    sk, sd = stochrsi_kd(c); sk0, sd0, sk1, sd1 = _f(sk), _f(sd), _f(sk, -2), _f(sd, -2)
    votes['StochRSI'] = None if None in (sk0, sd0, sk1, sd1) else _v(
        sk0 < 20 and sd0 < 20 and sk0 > sd0 and sk1 < sd1, sk0 > 80 and sd0 > 80 and sk0 < sd0 and sk1 > sd1)
    wr = williams_r(h, l, c); w0, w1 = _f(wr), _f(wr, -2)
    votes['W%R(14)'] = None if w0 is None or w1 is None else _v(w0 < -80 and w0 > w1, w0 > -20 and w0 < w1)
    e13 = _ema(c, 13); bull = h - e13; bear = l - e13
    e0, bu0, be0, bu1, be1 = _f(e13), _f(bull), _f(bear), _f(bull, -2), _f(bear, -2)
    votes['BBPower'] = None if None in (e0, bu0, be0, bu1, be1) else _v(
        close > e0 and be0 < 0 and be0 > be1, close < e0 and bu0 > 0 and bu0 < bu1)
    uo = ultimate_osc(h, l, c); u0 = _f(uo)
    votes['UO(7,14,28)'] = None if u0 is None else _v(u0 > 70, u0 < 30)

    # --- moving averages (close vs MA) ---
    ma_pairs = []
    for n in (10, 20, 30, 50, 100, 200):
        ma_pairs.append((f'SMA{n}', _f(_sma(c, n))))
        ma_pairs.append((f'EMA{n}', _f(_ema(c, n))))
    ma_pairs.append(('Ichimoku', _f(ichimoku_base(h, l))))
    ma_pairs.append(('VWMA20', _f(vwma(c, v))))
    ma_pairs.append(('HMA9', _f(hull_ma(c))))
    for name, ma in ma_pairs:
        votes[name] = None if (ma is None or close is None) else _v(close > ma, close < ma)

    osc = [votes[n] for n in OSC_NAMES if votes.get(n) is not None]
    mas = [votes[n] for n in MA_NAMES if votes.get(n) is not None]
    avail = osc + mas
    if not avail:
        return out
    rating = float(np.mean(avail))
    out.update({'rating': round(rating, 4),
                'category': category(rating),
                'osc_rating': round(float(np.mean(osc)), 4) if osc else None,
                'ma_rating': round(float(np.mean(mas)), 4) if mas else None,
                'votes': votes, 'n_votes': len(avail)})
    return out


def category(rating: float) -> str:
    """TV bands: SS: r ≤ −0.5 | S: −0.5 < r ≤ −0.1 | N: −0.1 < r < 0.1 | B: 0.1 ≤ r < 0.5 | SB: r ≥ 0.5."""
    if rating <= -0.5: return 'STRONG_SELL'
    if rating <= -0.1: return 'SELL'
    if rating < 0.1:   return 'NEUTRAL'
    if rating < 0.5:   return 'BUY'
    return 'STRONG_BUY'


def zone(rating: float) -> str:
    """Fine zone incl. the user's thirds-of-band semantics (the 'needle position')."""
    if rating is None: return 'na'
    a = abs(rating); side = 'buy' if rating > 0 else 'sell'
    if a >= 0.5: return f'strong_{side}'
    if a >= 0.1:
        third = int(min((a - 0.1) / (0.4 / 3.0), 2.999))   # 0=lo,1=mid,2=hi (nearest strong)
        return f'{side}_{["lo", "mid", "hi"][third]}'
    return 'neutral'


# ---------- extra indicator shadow features (TV-scripts research top picks) ----------
def supertrend_state(df: pd.DataFrame, n: int = 10, mult: float = 3.0):
    """SuperTrend(10,3) direction: +1 above (uptrend) / −1 below. Top recurring TV-scripts approach."""
    if df is None or len(df) < n + 5: return None
    h, l, c = df['high'].astype(float), df['low'].astype(float), df['close'].astype(float)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = _rma(tr, n)
    mid = (h + l) / 2.0
    ub, lb = mid + mult * atr, mid - mult * atr
    fub, flb = ub.copy(), lb.copy()
    for i in range(1, len(c)):
        fub.iloc[i] = ub.iloc[i] if (ub.iloc[i] < fub.iloc[i-1] or c.iloc[i-1] > fub.iloc[i-1]) else fub.iloc[i-1]
        flb.iloc[i] = lb.iloc[i] if (lb.iloc[i] > flb.iloc[i-1] or c.iloc[i-1] < flb.iloc[i-1]) else flb.iloc[i-1]
    trend = 1
    for i in range(1, len(c)):
        trend = 1 if c.iloc[i] > fub.iloc[i-1] else (-1 if c.iloc[i] < flb.iloc[i-1] else trend)
    return int(trend)

def vwap_deviation(df: pd.DataFrame, bars: int = 240) -> float | None:
    """Rolling-session VWAP deviation: (close − VWAP) / VWAP over the last `bars` 1m candles."""
    if df is None or len(df) < 30: return None
    d = df.tail(bars)
    tp = (d['high'].astype(float) + d['low'].astype(float) + d['close'].astype(float)) / 3.0
    vol = d['volume'].astype(float)
    den = float(vol.sum())
    if den <= 0: return None
    vw = float((tp * vol).sum() / den)
    return round((float(d['close'].iloc[-1]) - vw) / vw, 6)

def mr_zscore(df: pd.DataFrame, n_ma: int = 50, n_atr: int = 14):
    """Mean-reversion stretch: (close − SMA(n_ma)) / ATR(n_atr). Scalping-classes study
    (2026-06-11): strongest stable candle signal yet — AUC(−z→up15m)=0.531, monotonic bins
    (z>+2 → −3.6bps fade). Third independent confirmation of short-horizon mean reversion."""
    if df is None or len(df) < n_ma + 5: return None
    h, l, c = df['high'].astype(float), df['low'].astype(float), df['close'].astype(float)
    sma = c.rolling(n_ma, min_periods=n_ma).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / n_atr, adjust=False, min_periods=n_atr).mean()
    v = (float(c.iloc[-1]) - float(sma.iloc[-1])) / max(float(atr.iloc[-1]), 1e-9)
    return round(v, 4) if np.isfinite(v) else None


def ema_cross_age(df: pd.DataFrame, fast: int = 20, slow: int = 50, cap: int = 120):
    """Signed bars-since-cross of EMA(fast) vs SMA(slow): +age after an up-cross, −age after a
    down-cross (capped). Backtest finding (2026-06-10): the persistent state mean-reverts, but
    FRESH 1m crosses carry brief momentum (+12.7bps spread) — freshness is the signal."""
    if df is None or len(df) < slow + 5: return None
    c = df['close'].astype(float)
    f = c.ewm(span=fast, adjust=False, min_periods=fast).mean()
    s = c.rolling(slow, min_periods=slow).mean()
    st = (f > s).astype(int)
    flips = st.diff().fillna(0) != 0
    idx = np.where(flips.values)[0]
    if len(idx) == 0: return None
    age = min(len(c) - 1 - int(idx[-1]), cap)
    return int(age if st.iloc[-1] == 1 else -age)


def pivot_distance(df_1d: pd.DataFrame, price: float) -> float | None:
    """Distance of price from the classic daily pivot P=(H+L+C)/3 of the PRIOR day, as a fraction."""
    if df_1d is None or len(df_1d) < 2 or not price: return None
    prev = df_1d.iloc[-2]
    p = (float(prev['high']) + float(prev['low']) + float(prev['close'])) / 3.0
    return round((price - p) / p, 6) if p > 0 else None


def klines_to_df(kl: list) -> pd.DataFrame:
    """Binance/Aster-style kline arrays → OHLCV DataFrame (ascending time)."""
    if not kl: return pd.DataFrame()
    cols = ['open_time', 'open', 'high', 'low', 'close', 'volume']
    rows = [[k[0]] + [float(k[i]) for i in range(1, 6)] for k in kl if isinstance(k, (list, tuple)) and len(k) >= 6]
    df = pd.DataFrame(rows, columns=cols).sort_values('open_time').reset_index(drop=True)
    return df
