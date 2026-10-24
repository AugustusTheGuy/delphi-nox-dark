# ============================================================
# DELPHI NOVA 4.00
# Venues: Aster (primary) · ApeX Omni (secondary) · Paper (default)
# Max Leverage: 40x (risk-bounded band 10–40x, default 20x)
# Execution: modular exchange_adapter.ExchangeRouter (replaces DriftTrader)
# ============================================================
# Section order (optimal execution path):
#  1. Imports & Config          6. Data Fetching
#  2. Logging & State           7. Technical Analysis
#  3. Deep Learning Models      8. ML Ensemble + DL Manager
#  4. RL Optimizer              9. Signal Generation
#  5. Exchange Router          10. Execution & Monitoring
#                              11. FastAPI Dashboard
#                              12. Main Loop
# ============================================================

import os
# Must set BEFORE importing torch / xgboost / sklearn — prevents the
# duplicate-OpenMP segfault on macOS arm64 when PyTorch's libomp and
# XGBoost's libomp/libgomp both register the same symbols.
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import asyncio
import json
import hashlib
import hmac
import secrets
import logging
import math
import pickle
import threading
import time
import warnings
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any

import aiohttp
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from pyrate_limiter import Duration, Limiter, Rate

# PyTorch must load BEFORE Solana/Rust extensions to avoid segfault
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import talib

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import TimeSeriesSplit
from sklearn.base import clone
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import RobustScaler, StandardScaler

from hmmlearn.hmm import GaussianHMM
from xgboost import XGBClassifier

try:
    from fastdtw import fastdtw
    from scipy.spatial.distance import cosine as cosine_dist
    from scipy.stats import wasserstein_distance
    HAS_DTW = True
except ImportError:
    HAS_DTW = False

try:
    import praw
    HAS_PRAW = True
except ImportError:
    HAS_PRAW = False

try:
    from pytrends.request import TrendReq
    HAS_PYTRENDS = True
except ImportError:
    HAS_PYTRENDS = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from stable_baselines3 import PPO
    try:
        import gymnasium as gym
        from gymnasium import spaces
    except ImportError:
        import gym
        from gym import spaces
    HAS_RL = True
except ImportError:
    HAS_RL = False

# v4.00: Drift/Solana execution SDK (anchorpy/solders/solana/driftpy) removed —
# execution migrated to exchange_adapter.ExchangeRouter (Aster→ApeX→Paper). The
# only remaining Solana touchpoint is fetch_solana_on_chain(), which reads the
# wallet SOL balance over raw JSON-RPC (no SDK needed).
from fastapi import FastAPI, WebSocket, HTTPException, Request, Form
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

warnings.filterwarnings('ignore')
load_dotenv()

# ============================================================
# LOGGING
# ============================================================
class _FlushingFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record); self.flush()
class _FlushingStream(logging.StreamHandler):
    def emit(self, record):
        super().emit(record); self.flush()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        _FlushingFileHandler('delphi_nova_1_0.log'),
        _FlushingStream()
    ]
)
logger = logging.getLogger('DelphiNova1')

# ============================================================
# SECTION 1: CONFIGURATION
# ============================================================
CONFIG = {
    'VERSION': '4.00',
    # NAME is env-overridable so the self-contained "Delphi Nox Dark" Desktop
    # build (same engine, max-aggressive nox_dark profile) gets its own identity
    # in logs + dashboard without forking the code.
    'NAME': os.getenv('DELPHI_NAME', 'Delphi Nova 4.00'),

    # Wallet (Solana key retained for legacy price fallbacks + .env compatibility)
    'PUBLIC_KEY': os.getenv('SOLANA_PUBLIC_KEY', ''),
    'PRIVATE_KEY_ENV': 'SOLANA_PRIVATE_KEY',
    'SOLANA_RPC_URL': os.getenv('SOLANA_RPC_URL', 'https://api.mainnet-beta.solana.com'),

    # --- v4.00 Multi-venue execution -----------------------------------------
    # ENGINE_MODE: 'paper' (default, simulated fills on REAL venue prices —
    #   needs no API keys), 'testnet', or 'live'. Live fills require venue creds
    #   in .env AND an explicit operator go-live. Set via DELPHI_MODE env var.
    'ENGINE_MODE':      os.getenv('DELPHI_MODE', 'paper'),
    'PRIMARY_VENUE':    os.getenv('DELPHI_PRIMARY_VENUE', 'aster'),  # aster | apex
    'SECONDARY_VENUE':  'apex',
    'APEX_PROMO_MAKER': True,    # use ApeX 0%-maker promo fee schedule when making
    'SYMBOL':           os.getenv('DELPHI_SYMBOL', 'SOL'),
    'ACTIVE_MARKET':    0,       # logical market id (engine speaks "SOL")
    'SOL_MARKET_INDEX': 0,
    'BTC_MARKET_INDEX': 1,

    # Trading Parameters
    'STARTING_BALANCE': float(os.getenv('STARTING_BALANCE', '100.0')),
    # Leverage band (risk-bounded). MAX_LEVERAGE = hard ceiling (user-set 40x);
    # LEVERAGE_CAP = operating cap the sizing/governor clamps to (configurable
    # 10–40x); effective per-trade leverage is chosen by vol-target + SL-implied
    # logic and may be far lower (the fee-drag governor pushes high-freq → low-lev).
    'MAX_LEVERAGE':    40,
    'LEVERAGE_CAP':    int(os.getenv('DELPHI_LEVERAGE_CAP', '20')),
    'LEVERAGE_CAP_MIN': 10,      # lower bound of the configurable cap band
    'DEFAULT_LEVERAGE': 20,
    # Vol-target reference: the annualised realised vol at which the engine runs
    # its DEFAULT_LEVERAGE. SOL's normal 1m-realised vol annualises to ~150%, so
    # a classic 15% equities target would pin leverage to the 1x floor forever
    # (degenerate). Anchoring on a crypto-realistic reference keeps the 10–40x
    # band usable while preserving inverse vol-scaling (de-lever in high vol).
    'VOL_REF_ANNUAL': float(os.getenv('DELPHI_VOL_REF_ANNUAL', '1.5')),  # 150%
    'RISK_PER_TRADE': 0.02,
    'MAX_DAILY_DRAWDOWN': 0.15,
    'MAX_POSITION_PCT': 0.50,
    'MIN_POSITION_SIZE': 0.01,
    'SLIPPAGE_TOLERANCE': 0.005,

    # Signal Thresholds — relaxed lower bound to action probabilistic edge.
    # ML+DL composite + EV ratio still enforce quality downstream.
    'CONFIDENCE_THRESHOLD_MIN': 0.53,
    'CONFIDENCE_THRESHOLD_MAX': 0.58,
    'MIN_EV_RATIO': 1.35,

    # RL Parameters
    'RL_LEARNING_RATE': 0.001,
    'RL_GAMMA': 0.99,
    'RL_TAU': 0.005,
    'RL_BUFFER_SIZE': 5000,
    'RL_BATCH_SIZE': 64,
    'RL_RETRAIN_INTERVAL': 1000,

    # Data
    'CANDLE_LIMIT': 500,
    'FEATURE_WINDOW': 50,
    'RETRAIN_INTERVAL': 60,
    # Uncertainty threshold for DL Bayesian head — relaxed during warmup so
    # signals can fire before models are well-calibrated.
    'UNCERTAINTY_THRESHOLD': 0.65,
    'UNCERTAINTY_THRESHOLD_WARMUP': 0.95,

    # API Keys
    'HELIUS_API_KEY': os.getenv('HELIUS_API_KEY', ''),
    'LUNARCRUSH_API_KEY': os.getenv('LUNARCRUSH_API_KEY', ''),
    'POLYMARKET_API_KEY': os.getenv('POLYMARKET_API_KEY', ''),
    'BIRDEYE_API_KEY': os.getenv('BIRDEYE_API_KEY', ''),
    'COINGECKO_API_KEY': os.getenv('COINGECKO_API_KEY', ''),
    'FRED_API_KEY': os.getenv('FRED_API_KEY', ''),
    'DUNE_API_KEY': os.getenv('DUNE_API_KEY', ''),
    'X_BEARER_TOKEN': os.getenv('X_BEARER_TOKEN', ''),
    'REDDIT_CLIENT_ID': os.getenv('REDDIT_CLIENT_ID', ''),
    'REDDIT_CLIENT_SECRET': os.getenv('REDDIT_CLIENT_SECRET', ''),
    'REDDIT_USER_AGENT': os.getenv('REDDIT_USER_AGENT', 'DelphiNova1.0/1.0'),

    # Dashboard
    'DASHBOARD_PORT': int(os.getenv('DASHBOARD_PORT', '8080')),
    # 2026-06-10 security hardening: bind LOOPBACK by default; external access goes through the
    # cloudflared TLS tunnel (HTTPS) + the session-auth gate. Set DASHBOARD_HOST=0.0.0.0 for LAN.
    'DASHBOARD_HOST': os.getenv('DASHBOARD_HOST', '127.0.0.1'),
    # Dashboard auth gate (server-side content blocking — login page is the ONLY pre-auth content,
    # so the real dashboard can never be read/de-blurred pre-login). Creds live in .env as
    # DELPHI_DASH_USER + DELPHI_DASH_PWHASH (scrypt salt$hash) — NEVER plaintext. Auth activates
    # only when both are set (safe local-dev default = open on loopback).
    'DASH_AUTH_ENABLE':      True,
    'DASH_SESSION_TTL_SEC':  43200,   # 12h sliding session
    'DASH_LOGIN_MAX_FAILS':  5,       # per-IP fails before lockout
    'DASH_LOGIN_LOCK_SEC':   900,     # 15-min lockout
    'DASH_COOKIE_SECURE':    os.getenv('DASH_COOKIE_SECURE', '0') == '1',

    # ATR multipliers for RL SL/TP optimizer
    'SL_ATR_MULT': [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5],
    'TP_ATR_MULT': [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0],

    # Macro circuit breakers
    'DXY_MAX': 125.0,
    'VIX_MAX': 30.0,
    # Quant overlay
    'MAX_TOTAL_DRAWDOWN': 0.25,     # hard halt at 25% peak-to-trough
    'BAYES_MIN_FILTERS': 2,         # sanity floor: at least 2 filters must pass
    'USE_CVAR_SIZING': True,
    # Position-sizing policy selector (A/B-tested in validate_v4.py):
    #   'pessimistic'         — bootstrap 5th-pct win-rate → half-Kelly (SAFE default)
    #   'dynamic_half_kelly'  — Grok's Dynamic Half-Kelly on EMPIRICAL p/b, 25% cap
    #   'dynamic_pessimistic' — pessimistic base × bounded two-sided regime multiplier
    # Default STAYS 'pessimistic'; only switch if the A/B proves another strictly
    # better on RISK-ADJUSTED terms (Sharpe/Sortino/ruin prob), not raw terminal
    # wealth. Grok's 0.75-of-balance cap is rejected outright — all modes clip 0.25.
    'SIZING_MODE': os.getenv('DELPHI_SIZING_MODE', 'pessimistic'),

    # Trade-quality safeguards (Phase-4 ultra-review)
    'MIN_SL_PCT_OF_PRICE': 0.004,   # 0.4% — wider than 1m noise floor (3-5x ATR)
    'MIN_TP_RR_RATIO':     2.2,     # TP distance ≥ 2.2x SL distance (was 1.5 — payoff redesign)
    'PAPER_LEVERAGE_CAP':  20,      # paper mirrors the live operating cap (validate real regime)
    'LOSS_COOLDOWN_SEC':   120,     # wait 2 min after 2nd consecutive loss
    'LOSS_COOLDOWN_AFTER': 2,       # number of consecutive losses to trigger cooldown
    'ADAPTIVE_CONF_ENABLE': True,   # auto-tighten threshold after losses
    # --- Losing-streak / drawdown size de-risk (anti-martingale throttle) ------
    # Between the 2-loss cooldown (a time PAUSE) and the 25% hard halt there was
    # NO graduated SIZE reduction — yet kelly × vol_scale × reg_mult all scale UP
    # into a low-vol whipsaw mislabelled "trending" (the empirical failure mode:
    # 25% win-rate while dyn_kelly pins at the 1.5 mult ceiling and vol_target_lev
    # climbs to ~37x). This bounded factor (≤1.0, NEVER raises size) shrinks risk
    # per trade as the loss streak / drawdown grows, then SELF-HEALS on the next
    # win (consecutive_losses resets to 0). Floor keeps continuous trades flowing.
    'STREAK_DERISK_ENABLE':   True,
    'STREAK_DERISK_PER_LOSS': 0.15,  # shave 15% of size per loss beyond the 1st
    'STREAK_DERISK_DD_COEF':  2.0,   # +1 drawdown-frac → -2.0 (5% dd ⇒ -0.10)
    'STREAK_DERISK_FLOOR':    0.35,  # never below 35% of nominal size
    # --- Overlay-fallback model-abstention veto ------------------------------
    # The overlay-fallback path fires a directional trade when the trained ML+DL
    # ensembles BOTH predict 'hold'. Empirically (this paper cycle) it fired
    # 36/36 SHORT at |overlay_score|>=0.25 while ml_hold=0.96-0.99 & dl_hold=1.00
    # — i.e. heuristic overlays overriding models that are *confidently* flat in a
    # chop, producing 12 SL vs 2 TP. Principled guard: when BOTH models are
    # near-unanimously flat (min(ml_hold,dl_hold) >= CONVICTION), demand a STRONGER
    # overlay score before overriding them; when EITHER model has directional
    # conviction (low hold prob, e.g. dl_hold=0.00) keep the normal BASE bar.
    # Reversible: set ENABLE=False to restore the exact prior 0.25-flat behavior.
    'FALLBACK_MODEL_VETO_ENABLE': True,
    'FALLBACK_BASE_SCORE':        0.25,  # normal override bar (models not unanimous)
    'FALLBACK_FLAT_CONVICTION':   0.85,  # min(ml_hold,dl_hold) ⇒ "both confidently flat"
    'FALLBACK_STRONG_SCORE':      0.45,  # required |overlay_score| vs confidently-flat models
    # --- Statistical-alpha layer: OOF meta-stacking + honest eval (Phase 0) ----
    # BUG-FIX: the meta-LogReg used to train on base predict_proba of the SAME rows
    # the bases were just .fit on (in-sample) → RF/GB memorize → meta collapses to
    # ml_hold≈0.98 on live bars (and reported a phantom accuracy=1.0). De-leak by
    # training the meta on OUT-OF-FOLD base proba (TimeSeriesSplit, no shuffle).
    # Reversible: META_OOF_ENABLE=False restores the exact (leaky) legacy fit.
    'META_OOF_ENABLE':        True,   # train meta on out-of-fold base proba (de-leak)
    'META_OOF_SPLITS':        3,      # TimeSeriesSplit folds for cross_val_predict
    'META_OOF_MIN_ROWS':      120,    # below this, OOF unstable → legacy in-sample fit
    'EVAL_PURGED_ENABLE':     True,   # honest purged+embargoed walk-forward scoring
    'EVAL_EMBARGO_BARS':      5,      # = 5-bar fwd-return label horizon in build_features
    # Meta-label gate adapts to the de-leak (concept drift): fade stale (regime,action)
    # evidence each retrain so the OLD leaky model's losing buckets don't permanently
    # block the de-leaked model. MIN_SAMPLES gates how much evidence is needed to block.
    'METALABEL_MIN_SAMPLES':  5,      # min (regime,action) samples before the WR gate can block
    'METALABEL_DECAY':        0.90,   # per-retrain decay of meta-label evidence (1.0 = off);
                                      # fades the pre-de-leak model's polluted buckets over a
                                      # few retrains so the de-leaked model can re-prove a setup
    # --- Overlay-fallback component scales (were hard-coded in the signal path) ---
    # Defaults reproduce the prior overlay_score EXACTLY (byte-for-byte rollback).
    'OVERLAY_W_MTF':          0.30,   # multi-timeframe alignment
    'OVERLAY_W_FUNDZ':        0.20,   # funding-z extreme (±1.5)
    'OVERLAY_W_CROWD':        0.15,   # crowded-trade sign (subtracted)
    'OVERLAY_W_OMS':          0.10,   # oracle-mark spread sign
    'OVERLAY_W_DOPPEL':       0.15,   # doppelganger tanh
    'OVERLAY_W_COINTEG':      0.10,   # cointegration tanh (subtracted)
    # --- Local on-box AI analyst (Ollama; second-opinion veto + bounded fusion) ----
    # delphi-trader-nox-dark (→ qwen2.5 fallback) is an INDEPENDENT directional
    # confirmer/veto on CANDIDATE trades + a tiny downgrade-biased fusion member.
    # It can ONLY veto/haircut (≤-0.08) or nudge up ≤+0.03 — never raises leverage,
    # lowers MIN_EV_RATIO, or sizes. Zero cloud tokens. LOCAL_AI_ENABLE=False ⇒ exact
    # prior signal path. Invoked only on candidates (rare) with timeout+breaker+cache.
    'LOCAL_AI_ENABLE':            os.getenv('DELPHI_LOCAL_AI', '1') == '1',
    'LOCAL_AI_BASE_URL':          os.getenv('OLLAMA_HOST', 'http://localhost:11434'),
    'LOCAL_AI_MODEL':             'delphi-trader-nox-dark:latest',
    'LOCAL_AI_FALLBACK_MODEL':    'qwen2.5:7b-instruct',
    'LOCAL_AI_TIMEOUT_SEC':       1.2,
    'LOCAL_AI_MIN_INTERVAL_SEC':  2.0,
    'LOCAL_AI_CACHE_TTL_SEC':     8.0,
    'LOCAL_AI_MAX_TOKENS':        96,
    'LOCAL_AI_TEMPERATURE':       0.2,
    'LOCAL_AI_NUM_CTX':           4096,
    'LOCAL_AI_KEEP_ALIVE':        '30m',
    'LOCAL_AI_FAIL_OPEN_N':       3,
    'LOCAL_AI_COOLDOWN_SEC':      60,
    'LOCAL_AI_VETO_ENABLE':       True,
    'LOCAL_AI_VETO_MIN_DISAGREE': 0.62,
    'LOCAL_AI_POSTTRADE_LOG':     True,
    'LOCAL_AI_FUSE_MAX_UP':       0.03,
    'LOCAL_AI_FUSE_MAX_DOWN':     0.08,
    'LOCAL_AI_FUSE_GAIN':         0.10,
    'LOCAL_AI_JOURNAL_PATH':      'local_ai_journal.jsonl',
    # --- v6 AI data-augmentation (free/local Ollama; background loop, OFF the trading path) ---
    # The local model judges the CURRENT multi-source DATA's quality + directional lean every
    # INTERVAL sec; the result feeds the dashboard + a DOWN-ONLY, bounded confidence haircut when
    # the data looks low-quality/inconsistent (risk-reducing only). ENABLE=False ⇒ exact prior path.
    'LOCAL_AI_AUGMENT_ENABLE':        os.getenv('DELPHI_AI_AUGMENT', '1') == '1',
    'LOCAL_AI_AUGMENT_INTERVAL_SEC':  30,     # background refresh cadence (not the trade path)
    'LOCAL_AI_AUGMENT_TTL_SEC':       45.0,   # in-analyst cache TTL
    'LOCAL_AI_AUGMENT_TIMEOUT_SEC':   8.0,    # generous: off the trade path; measured warm ≈3.0s
    'LOCAL_AI_AUGMENT_MAX_TOKENS':    110,
    'LOCAL_AI_AUGMENT_APPLY_HAIRCUT': True,   # SAFE: down-only haircut on low data-quality
    'LOCAL_AI_AUGMENT_MAX_HAIRCUT':   0.06,   # max confidence reduction from a fully-low-quality verdict
    # --- Q2 AI news/event-sentiment layer (free RSS → local Ollama; background; DISPLAY-FIRST) ---
    # Reads recent crypto headlines into a structured SOL near-term sentiment + event-risk verdict.
    # Replaces the 3 dead social feeds (X/Reddit/Google). NOT wired into the signal yet (display-first).
    'LOCAL_AI_NEWS_ENABLE':       os.getenv('DELPHI_AI_NEWS', '1') == '1',
    'LOCAL_AI_NEWS_MODEL':        'qwen2.5:7b-instruct',  # general-purpose reads news better than the fine-tune
    'LOCAL_AI_NEWS_INTERVAL_SEC': 600,        # 10-min background refresh (news changes slowly)
    'LOCAL_AI_NEWS_TTL_SEC':      600.0,      # in-analyst cache TTL
    'LOCAL_AI_NEWS_TIMEOUT_SEC':  9.0,        # generous (off the trade path; measured ≈6.4s)
    'LOCAL_AI_NEWS_MAX_TOKENS':   160,
    'LOCAL_AI_NEWS_APPLY':        False,      # DISPLAY-FIRST: not consumed by the signal until validated
    # v7 Phase 2 (news event-risk de-risk) — ACTIVE ONLY when LOCAL_AI_NEWS_APPLY=True (enable after a
    # session of validated news reliability). Risk-reducing ONLY: cut leverage during a crash/hack/
    # regulatory shock; pause NEW entries on extreme risk (open positions untouched). Gated-off = no effect.
    'NEWS_DERISK_RISK':           0.60,       # event_risk >= this ⇒ cut leverage
    'NEWS_DERISK_LEV_FACTOR':     0.50,       # leverage multiplier when de-risking
    'NEWS_DERISK_PAUSE_RISK':     0.85,       # event_risk >= this ⇒ skip NEW entries entirely
    # v8 Phase D (anti-staleness) — event-driven cache refresh: on a high news event_risk, invalidate
    # the TTL data caches so the next decision uses FRESH data (never trade on pre-event stale data).
    # ALWAYS-ON (non-outcome: only freshens data; does not change trade logic).
    'NEWS_EVENT_REFRESH_RISK':    0.60,
    # v9 Phase D — real-time Aster mark-price + funding via WebSocket (push vs poll). DISPLAY-FIRST:
    # populates state.ws_* for the dashboard/freshness; ASTER_WS_FEED gates feeding it into the
    # decision path (kept False until validated → non-outcome, no fingerprint change). REST poll
    # stays the hot backstop so a WS drop never stalls trading.
    'ASTER_WS_ENABLE':            os.getenv('DELPHI_ASTER_WS', '1') == '1',
    'ASTER_WS_URL':               'wss://fstream.asterdex.com',
    'ASTER_WS_SYMBOL':            'solusdt',
    'ASTER_WS_FEED':              False,    # wire WS price/funding into decisions (outcome) — later
    'ASTER_WS_STALE_SEC':         5.0,      # WS value older than this ⇒ treated as stale (fall back to REST)
    'ADAPTIVE_TTL_ENABLE':        True,     # v9 Phase D: scale every cache TTL by vol+event_risk (fresher when it matters)
    # v10 Track 2b — order-book depth SHADOW features (LOG-ONLY): real bid/ask depth dynamics the
    # engine lacks (its "OFI" is a candle proxy). Logged per trade for validate_winprob.py; NEVER
    # used in decisions until it provably beats the 0.52 coin-flip OOS. Non-outcome.
    'OB_FEATURES_ENABLE':         True,
    'OB_POLL_SEC':                3.0,
    'OB_DEPTH':                   20,
    # v10 Track 2 — on-chain SHADOW features (a DIFFERENT signal class: network activity, not price
    # microstructure). Solana network TPS + trend via the public RPC (free, no key). LOG-ONLY; the
    # gate (validate_winprob.py) decides if it predicts. (Strongest on-chain signal = exchange net-flow,
    # needs a paid labeled-address feed — this is the free first-pass.) Non-outcome.
    'ONCHAIN_FEATURES_ENABLE':    True,
    'ONCHAIN_POLL_SEC':           60.0,
    'NEWS_RSS_FEEDS': [
        'https://www.coindesk.com/arc/outboundfeeds/rss/',
        'https://cointelegraph.com/rss',
        'https://decrypt.co/feed',
        'https://www.theblock.co/rss.xml',
    ],
    'NEWS_MAX_HEADLINES':         18,   # recency-sorted across feeds; the freshest 18 reach the model
    'TTL_NEWS_SEC':               540,        # RSS fetch cache (rate-limit guard)
    # --- Split-conformal prediction gate (statistical firing quality, Phase 2) -----
    # Calibrated per-class credibility: act on a DIRECTIONAL class only when its
    # conformal p-value clears alpha. 'soft' haircuts low-credibility calls (won't
    # re-starve firing); 'hard' abstains. Tuned for nox_dark (alpha 0.15, soft).
    'CONFORMAL_ENABLE':       True,
    'CONFORMAL_ALPHA':        0.15,
    'CONFORMAL_CAL_FRAC':     0.20,
    'CONFORMAL_MIN_CAL':      40,
    'CONFORMAL_CLASSES':      [0, 2],     # short=0, long=2 (never gate hold=1)
    'CONFORMAL_MODE':         'soft',     # 'soft' haircut | 'hard' abstain
    'CONFORMAL_SOFT_HAIRCUT': 0.06,
    # --- Triple-barrier labeling (López de Prado) — CONFIG-gated A/B vs quantile ---
    'LABEL_MODE':             'quantile', # 'quantile' (default) | 'triple_barrier'
    'TB_PT_ATR_MULT':         2.0,        # profit-take barrier = entry + 2.0*ATR
    'TB_SL_ATR_MULT':         1.5,        # stop-loss barrier  = entry - 1.5*ATR
    'TB_MAX_HOLD_BARS':       15,         # vertical (time) barrier in bars
    # --- Upgraded intelligence layer: Aster MLX MCP bridge (in-process) -----------
    # Bridges the live engine to /Projects/Delphi_Trading's pure-Python MCP tools:
    # pessimistic_dynamic_half_kelly_with_venue_routing (sizing), the 616-state 58-dim
    # vector DB (regime priors), and real on-device MLX inference. All bounded +
    # gracefully-degrading: if the upgraded root is absent, the engine uses internals.
    'ASTER_MCP_ENABLE':        os.getenv('DELPHI_ASTER_MCP', '1') == '1',
    'VECTORDB_OVERLAY_ENABLE': True,      # regime/outcome prior from the vector DB
    'VECTORDB_OVERLAY_MAX':    0.03,      # max |confidence nudge| from historic neighbours
    'VECTORDB_K':              8,         # k nearest historical states
    'MLX_OVERLAY_ENABLE':      True,      # fast on-device MLX directional nudge
    'MLX_OVERLAY_MAX':         0.015,     # max |confidence nudge| from MLX (small; linear model)
    'STATE_PERSIST_FILE': 'delphi_state.json',
    'STATE_PERSIST_EVERY': 5,       # persist after every N trades

    # --- Dynamic stop-loss engine (reversible; DSL_ENABLE=False = legacy SL/trail)
    # Two behaviours the user asked for, both bounded so the liquidation-distance
    # guard stays valid and a single noisy 1m bar can't shake us out:
    #  (1) Dynamic INITIAL stop — the ATR multiplier adapts to regime persistence
    #      (Hurst) and elevated volatility. Trending/volatile ⇒ WIDER stop (room to
    #      turn around when a trade opens slightly negative); calm ⇒ tighter (cut
    #      losers fast). Still floored at MIN_SL_PCT_OF_PRICE and capped so leverage
    #      stays liq-safe.
    #  (2) Dynamic TRAILING stop — anchored to the peak/trough (max favourable
    #      excursion), trailing at an ATR distance that TIGHTENS as profit grows so
    #      big winners give back little, with a hard volatility floor so short,
    #      unsustained reversals don't close us. Ratchets favourably only.
    'DSL_ENABLE':              True,
    # --- Asymmetric R:R payoff redesign (incident DN-PAYOFF-INV) -------------------
    # The live payoff ratio was INVERTED (0.79: avg loss > avg win) because every
    # green-then-red trade paid the FULL stop (no breakeven), winners gave back ~35%
    # of peak (loose 2.2×ATR trail), and stops were ATR-wide at high leverage. This
    # block tightens geometry + adds move-to-breakeven + a chandelier trail to push
    # realized payoff ≥1.5. Master flag below; legacy values noted in comments.
    'DSL_RR_REDESIGN':         True,  # gates breakeven + chandelier-noise-floor + $-cap (False=legacy)
    'DSL_INIT_ATR_MULT_MIN':   0.85,  # was 1.1 — tighter calm stop (cut losers fast)
    'DSL_INIT_ATR_MULT_MAX':   1.80,  # was 2.4 — cap loser size
    'DSL_INIT_HURST_GAIN':     0.90,  # was 1.2 — less aggressive trend-widening
    'DSL_INIT_VOL_GAIN':       0.6,   # elevated ewma_vol → widen factor (one-sided)
    'DSL_INIT_VOL_TARGET':     0.02,  # ewma_vol reference for the vol term
    'DSL_MAX_INIT_SL_PCT':     0.0060,# NEW hard ceiling: initial stop ≤ 0.60% of price ($-risk cap)
    'DSL_BREAKEVEN_ENABLE':    True,  # move-to-breakeven once modestly green → losers become scratches
    'DSL_BE_TRIGGER_R':        0.50,  # arm breakeven at +0.5R favorable excursion
    'DSL_BE_OFFSET_FEES_K':    1.5,   # lock 1.5× round-trip fee beyond entry (net-positive scratch)
    'DSL_BE_MIN_OFFSET_PCT':   0.0006,# floor on the BE offset (0.06% of price)
    'DSL_TRAIL_ACTIVATE_R':    1.0,   # start trailing once profit ≥ 1.0× initial-SL (R)
    'DSL_TRAIL_ATR_MULT':      1.40,  # was 2.2 — chandelier base (cuts the 35% give-back)
    'DSL_TRAIL_MIN_MULT':      0.60,  # was 0.8 — floor on big winners
    'DSL_TRAIL_TIGHTEN_PER_R': 0.12,  # was 0.18 — smoother taper
    'DSL_TRAIL_NOISE_ATR_MULT':0.90,  # NEW anti-shakeout: trail never tighter than 0.9×ATR
    'DSL_TRAIL_FLOOR_PCT':     0.0020,# was 0.0035 — absolute backstop (ATR-noise floor now binds)
    'DSL_TP_FEE_MULT':         2.5,   # TP ≥ 2.5× round-trip fee + slippage (no-op on 0% maker)
    # Partial-TP ladder: bank DSL_TP1_FRACTION at +DSL_TP1_R. DISABLED by data — replay_rr.py
    # over the real ledger showed EVERY ladder variant LOWERS payoff (50%@1.5R: 0.91→0.71;
    # 33%@2.5R: →0.85) because banking early CAPS the right tail, and breakeven already saves
    # the reversal cases. Kept CONFIG-gated/tested for future use; the no-ladder redesign wins.
    'DSL_TP_LADDER_ENABLE':    False,
    'DSL_TP1_R':               1.5,   # favorable excursion (in R) that triggers the partial
    'DSL_TP1_FRACTION':        0.5,   # fraction of the position banked at +TP1_R
    # Data-hygiene: when the outcome-affecting CONFIG (stop/TP/sizing/leverage/threshold)
    # changes, quarantine the trade-OUTCOME learners (meta-label buckets, adaptive-conf
    # offset, RL Q-tables) so a NEW rule-set never learns from trades made under a DIFFERENT
    # one. ML/DL + the 616-state vector DB train on candle forward-returns → NOT poisoned.
    'LEARN_FINGERPRINT_GATE':  True,
    # Confidence-scaled leverage (Phase 2): scale the operating leverage cap with signal
    # confidence — CONF_LEV_FLOOR_FRAC of the cap at the conf floor, full cap at FULL_AT.
    # Shrinks $-risk on the low-conviction tail (the 25x-on-losers problem); 40x hard cap
    # + SL-implied lev + liq/CVaR guards all remain downstream. =False restores prior.
    'CONF_SCALED_LEV_ENABLE':  True,
    'CONF_LEV_FLOOR_FRAC':     0.35,  # fraction of the cap used at floor-confidence
    'CONF_LEV_FULL_AT':        0.72,  # confidence at which the full cap is allowed
    # v7 1a — calibrate leverage by conformal credibility: scale the ABOVE-FLOOR leverage by
    # the conformal p-value so an OVERCONFIDENT-but-low-credibility trade is sized DOWN (the
    # exact mechanism behind the high-conf left-tail losses). =1.0 credibility ⇒ legacy sizing.
    'CONF_LEV_USE_CALIBRATION': True,
    'CONF_LEV_CALIB_MIN':       0.25,  # floor on the credibility multiplier (never below this)
    # v7 1b — hard per-trade $-loss cap (deterministic left-tail backstop, tighter than the
    # 2.5% CVaR/ES cap): size down so worst-case stop loss <= this fraction of balance. 0=off.
    'MAX_TRADE_LOSS_PCT':       0.005,  # cap at ~typical avg-loss so it clips only the tail (the
                                        # -$0.82/-$0.88 large-notional trades), not the body. Replay:
                                        # payoff 0.298->0.345, maxLoss -$0.876->-$0.512 on the cohort.
    'LOSS_CAP_LEVERAGED':       True,   # v10 FIX: include leverage in the loss-cap (was inert/~lev× loose)
    'LOSS_CAP_BACKSTOP':        True,   # 2026-06-10 B6 FIX: re-enforce the cap in execute_trade from the
                                        # FINAL size/lev/entry/SL. The in-sizing cap used an sl_pct that can
                                        # differ from the final sl_price → a few trades opened 5-6x over cap
                                        # (worst −$2.88 = ~70% of the cohort's loss). This deterministic
                                        # downstream clamp guarantees worst-case loss ≤ MAX_TRADE_LOSS_PCT.
    # v9 Phase A — Platt/logistic calibration of the FINAL confidence. The model is UNDER-confident
    # (ECE 0.30; conf compressed 0.55-0.66 vs true win-rates 50-100%) → it UNDER-sizes true winners.
    # Fit P(win)=sigmoid(a*conf+b) on current-rule-set (conf,win) pairs, then SIZE leverage by the
    # calibrated win-prob so the true-80% trades size up and the 0.55 coin-flips size down. The
    # 0.5% loss-cap bounds the now-larger leverage. Refits every RECAL_EVERY closes.
    # NOTE: built + tested, but kept DORMANT (=False) by data: the logistic fit on 79 real trades has a
    # near-ZERO slope → the FINAL confidence is non-predictive of win/loss (single-feature calibration
    # only flattens to the base rate ⇒ modest uniform-sizing, not worth a cohort reset). The real lever
    # is a MULTIVARIATE win-prob learner (Phase C). Flip True only when confidence becomes predictive.
    'CALIB_ENABLE':             False,
    'CALIB_MIN_TRADES':         40,     # need a real cohort before fitting (gate met at 41)
    'CALIB_RECAL_EVERY':        20,     # refit every N closes as data grows
    'CALIB_LO':                 0.50,   # calibrated win-prob mapped to floor leverage (coin-flip)
    'CALIB_HI':                 0.85,   # calibrated win-prob mapped to full leverage
    'CALIB_FIT_WINDOW':         120,    # fit on the last N post-fingerprint closes (sizing-invariant)
    # Adaptive confidence-floor optimizer (Phase 2): from CLOSED trades under the CURRENT
    # fingerprint, pick the floor maximizing net PnL (keeping ≥MIN_KEEP of volume); refit
    # every RECAL_EVERY closes. Only the floor RISES via this — it never lowers selectivity.
    'CONF_OPT_ENABLE':         True,
    'CONF_OPT_MIN_TRADES':     40,
    'CONF_OPT_RECAL_EVERY':    20,
    'CONF_OPT_GRID':           [0.51, 0.55, 0.58, 0.62, 0.66, 0.70],
    'CONF_OPT_MIN_KEEP':       0.30,
    'CONF_OPT_BLEND':          0.5,

    # --- Unidirectional-velocity scalp overlay (reversible) ---------------------
    # v_uni = (net_return / n) × EfficiencyRatio; ER=|net|/Σ|per-bar move| ∈[0,1]
    # discounts choppy moves. Verified on the SOL 1m substrate: ER-weighted
    # velocity gives a +6.7pp directional edge (0.567 vs 0.500) on strong-|v_norm|
    # bars at n=13/h=5, and BEATS raw velocity there — but the raw edge (~3.5bps/5
    # bars) only clears MAKER fees, so velocity is used as a bounded TIMING overlay
    # (#12 in the confidence stack, inside the ±0.12 soft cap) + a momentum-death
    # EXIT, never as a standalone fee-ignoring trigger.
    'VELO_NUDGE_ENABLE': True,
    'VELO_N_FAST':       13,     # fast efficiency-velocity window (bars)
    'VELO_N_SLOW':       26,     # slow window for acceleration (v_fast - v_slow)
    'VELO_LOOKBACK':     20,     # per-bar-vol lookback for v_norm
    'VELO_NORM_CAP':     0.40,   # |v_norm| that saturates the nudge to full strength
    'VELO_CONF_MAX':     0.025,  # max confidence bump for aligned clean velocity
    'VELO_ACCEL_BONUS':  0.005,  # extra bump when acceleration aligns with the side
    'VELO_EXIT_ENABLE':  True,   # momentum-death scalp exit (tighten stop in profit)
    'VELO_EXIT_FLIP_Z':  0.50,   # v_norm turned this far AGAINST an in-profit pos ⇒ snap stop
    # --- Hard "don't catch a falling knife" ENTRY veto (#16) -------------------
    # The entry-side mirror of the #13 momentum-death exit. The overlay-#12 nudge
    # only SOFT-haircuts an entry that fights opposite velocity (and that haircut
    # shares the ±0.12 overlay budget, so a confident model can still open INTO a
    # clean adverse leg). This veto HARD-rejects the clearest such cases: opening
    # directly into a STRONG (|v_norm|≥Z) and CLEAN/efficient (ER≥thr) move running
    # against the intended side. Bars are set high so it only kills falling-knife
    # entries — it will NOT starve normal entries in chop (low ER) or mild drift.
    'VELO_ENTRY_VETO_ENABLE': True,
    'VELO_ENTRY_VETO_ZNORM':  1.00,  # |v_norm| (vol units) of the ADVERSE move to veto
    'VELO_ENTRY_VETO_ER':     0.45,  # min efficiency ratio — only veto CLEAN adverse legs

    # --- Empirical historic-matching weights (#14) ----------------------------
    # The historic-similarity matchers (doppelganger k-NN + DTW/Wasserstein) used to
    # weight every feature EQUALLY, so proven-noise columns (demand_zone=0,
    # cdl_hammer, returns_5) diluted the distance as much as proven-signal columns
    # (obv, bb_lower, inst_vol_trend, poly_reg_diff). These weights re-rank each
    # feature by demonstrated predictive value (0.5·rank|Spearman(feat,fwd)| +
    # 0.5·rank|logit-coef| on the SOL substrate). DYNAMIC mode additionally blends a
    # live |Spearman(feat, fwd_return)| computed over the doppelganger's own rolling
    # window (so the weighting is "demonstrated value over a dynamic time window").
    # Mean-normalized to 1.0 so uniform weights reproduce the legacy distance exactly.
    'HISTORIC_WEIGHT_ENABLE':  True,
    'HISTORIC_WEIGHT_DYNAMIC': True,    # blend live rolling-window IC with static prior
    'HISTORIC_WEIGHT_PRIOR':   0.5,     # blend weight on the static substrate prior [0,1]
    'HISTORIC_WEIGHTS_PATH':   'models/feature_weights.json',

    # --- v4.00 Fee-drag governor + min-edge gate + liquidation guard ----------
    # Honest fee math: daily_fee_drag = RT_fee × leverage × trades/day on notional.
    # The governor caps cumulative fee spend per UTC day to a % of equity, and the
    # min-edge gate skips trades whose expected move can't clear fees + funding.
    'FEE_DRAG_GOVERNOR':      True,
    'DAILY_FEE_BUDGET_PCT':   float(os.getenv('DELPHI_FEE_BUDGET', '0.12')),  # 12% equity/day
    'MIN_EDGE_FEE_K':         1.3,   # require expected_move ≥ k×RT_fee + |funding|
    'GOVERNOR_HIFREQ_LEV':    3,     # ≤ this lev = "maker/hi-freq" regime (cheap)
    'GOVERNOR_SOFT_PCT':      0.75,  # at ≥75% of budget, require stronger edge
    'MAKER_BIAS':             True,  # prefer maker fee assumption when estimating edge
    # Liquidation-distance guard: reject entries whose SL/stop sits too close to
    # the venue liquidation price, and keep the hard stop well inside liq distance.
    'LIQ_GUARD_ENABLE':       True,
    'LIQ_STOP_FRACTION':      0.60,  # hard stop must be ≤ 60% of the liq distance
    'MIN_LIQ_DISTANCE_PCT':   0.015, # reject if liq < 1.5% away at chosen leverage

    # --- v4.00 Grok-derived SAFE additions (mined from the Grok conversation;
    # only the genuinely sound, non-ruinous ideas not already in the engine) ----
    # (1) Dynamic trade-limit governor — a regime/vol-aware CAP on opens-per-cycle
    # complementing the fee-drag governor. Bounds OVERTRADING in choppy/high-vol
    # regimes (serves max-profit-per-cycle + win-rate). NEVER blocks early trades.
    'DYNAMIC_TRADE_LIMIT':  True,
    'TRADE_LIMIT_BASE':     24,      # neutral-regime daily cap
    'TRADE_LIMIT_MIN':      12,      # floor (choppy/high-vol)
    'TRADE_LIMIT_MAX':      36,      # ceiling (clean trending/low-vol)
    # (2) Dynamic slippage model — volume/vol/size-aware sqrt market-impact, used
    # to mark PAPER fills realistically (live fills use the venue's real fill).
    'DYNAMIC_SLIPPAGE':     True,
    'SLIPPAGE_BASE_BPS':    1.0,     # baseline half-spread (bps) in calm conditions
    'SLIPPAGE_MAX_BPS':     20.0,    # hard cap on modeled slippage
    'SLIPPAGE_REF_DEPTH_USD': 250000.0,  # reference top-of-book depth for impact scaling
    # (3) Data-confidence circuit breaker — skip OPENS when cross-venue mark
    # divergence says the price feed is unreliable (robustness; complements the
    # macro breaker). Generous threshold => only fires on genuine dislocations.
    'DATA_CONFIDENCE_BREAKER': True,
    'MAX_FEED_DIVERGENCE':  0.010,   # 1.0% cross-venue basis => feed untrustworthy
    # data hygiene 2026-06-10: dead social scrapers off (X 401 / Reddit 403 / Trends 429 —
    # replaced by news_rss+Ollama); Birdeye quota circuit-breaker (skip 1h after usage-limit error,
    # candles come from Aster klines → Kraken meanwhile).
    'SOCIAL_SCRAPERS_ENABLE': False,
    'BIRDEYE_COOLDOWN_SEC':   3600,

    # (4) NoxSSD data-drive safety pause-switch — block NEW opens when the designated
    # data drive is not mounted / not R(/W)-accessible (debounced). Outcome-NEUTRAL:
    # these keys are deliberately NOT in _config_fingerprint(), so the guard never
    # resets the trade-outcome cohort. Mirrors DATA_CONFIDENCE_BREAKER (skip opens on
    # unavailable/untrustworthy data; open positions keep being managed/exited).
    'NOXSSD_GUARD_ENABLE':   True,
    'NOXSSD_MOUNT_PATH':     os.getenv('NOXSSD_MOUNT_PATH', '/Volumes/NoxSSD'),
    'NOXSSD_DATA_PATH':      os.getenv('NOXSSD_DATA_PATH', ''),   # '' => validate the mount itself
    'NOXSSD_REQUIRE_WRITE':  True,    # also require W_OK (the drive is normally read-write)
    'NOXSSD_GUARD_POLL_SEC': 20.0,    # responsive without spinning the disk
    'NOXSSD_FAIL_DEBOUNCE':  2,       # consecutive failed checks before blocking (anti-flap)
    'NOXSSD_RECOVER_DEBOUNCE': 2,     # consecutive OK checks before auto-resume (hysteresis)

    # (5) TradingView Technical-Rating replica — SHADOW/log-only multi-TF gauge (tv_ratings.py).
    # NOT in _config_fingerprint (no trading effect); the validator gates any future wiring.
    'TV_RATINGS_ENABLE':  True,
    'TV_TFS':             ['1m', '5m', '15m', '30m', '1h', '4h', '1d'],
    'TV_KLINES_LIMIT':    300,        # ≥250 so MA200 votes are live
    'TV_POLL_SEC':        20.0,
    'TV_TTLS':            {'1m': 30, '5m': 60, '15m': 120, '30m': 240,
                            '1h': 300, '4h': 900, '1d': 1800},
    # hold-matched TF weights from the MEASURED holding distribution (median 14.2 min, p25 6,
    # p75 23): weight mass on 5m–30m, tails on 1m/1h; 4h/1d logged unweighted (regime tilt only).
    'TV_HOLD_WEIGHTS':    {'1m': 0.10, '5m': 0.30, '15m': 0.30, '30m': 0.20, '1h': 0.10},

    # --- v4.00 Profiles (override base CONFIG keys; applied just below) --------
    # aggressive_hifreq = restored pre-checkpoint-1 Drift methodology: low
    # confidence floor, continuous trading, governor-bounded frequency. nox_dark
    # = max-aggressive (40x cap, looser fee budget) for the Desktop alt build.
    'ACTIVE_PROFILE': os.getenv('DELPHI_PROFILE', 'aggressive_hifreq'),
    'PROFILES': {
        'aggressive_hifreq': {
            # High-frequency probabilistic-edge profile: the confidence band is
            # intentionally narrow/low because RISK is governed downstream by the
            # EV gate (≥1.25), the fee-drag governor, pessimistic-Kelly + CVaR
            # sizing, and the liquidation-distance guard — NOT by a high conf bar.
            # MAX kept at 0.56 so the elite-overlay fallback path (caps ~0.567)
            # is actually reachable; a 0.58 ceiling made that path vestigial.
            'CONFIDENCE_THRESHOLD_MIN': 0.52,
            'CONFIDENCE_THRESHOLD_MAX': 0.56,
            'MIN_EV_RATIO':            1.25,
            'LEVERAGE_CAP':            20,
            'PAPER_LEVERAGE_CAP':      20,
            'DAILY_FEE_BUDGET_PCT':    0.12,
            'RISK_PER_TRADE':          0.02,
            'LOOP_SLEEP_SEC':          1,
        },
        'balanced': {
            'CONFIDENCE_THRESHOLD_MIN': 0.55,
            'CONFIDENCE_THRESHOLD_MAX': 0.62,
            'MIN_EV_RATIO':            1.5,
            'LEVERAGE_CAP':            15,
            'PAPER_LEVERAGE_CAP':      15,
            'DAILY_FEE_BUDGET_PCT':    0.08,
            'RISK_PER_TRADE':          0.015,
            'LOOP_SLEEP_SEC':          2,
        },
        'nox_dark': {   # max-aggressive (Delphi Nox Dark Desktop build)
            # Band shifted uniformly below aggressive_hifreq's [0.52,0.56] so the
            # max-aggressive build fires at least as readily; MAX=0.55 also keeps
            # the overlay-fallback path (caps ~0.567) reachable here too.
            'CONFIDENCE_THRESHOLD_MIN': 0.51,
            'CONFIDENCE_THRESHOLD_MAX': 0.55,
            'MIN_EV_RATIO':            1.20,
            'LEVERAGE_CAP':            40,
            'PAPER_LEVERAGE_CAP':      40,
            'DAILY_FEE_BUDGET_PCT':    0.20,
            'RISK_PER_TRADE':          0.025,
            'LOOP_SLEEP_SEC':          1,
        },
    },
    'LOOP_SLEEP_SEC': 1,

    # BOCPD knobs (was hardcoded constants in signal-gen)
    'BOCPD_HARD_P':       0.95,
    'BOCPD_SPIKE_P':      0.85,
    'BOCPD_SPIKES_REQ':   2,

    # 24-hour trading cycle
    'CYCLE_DURATION_HOURS':    24,
    'CYCLE_WINDDOWN_MINUTES':  30,   # last N min: no new opens, let existing close
    'CYCLE_SUMMARY_RETAIN':    30,   # keep last N cycle summaries in memory

    # TTL caches for hot-path fetches (seconds). Mitigates per-tick API spam
    # without sacrificing freshness on data that doesn't change frequently.
    'TTL_FRED_SEC':       3600,
    'TTL_SENTIMENT_SEC':  60,
    'TTL_POLYMARKET_SEC': 60,
    'TTL_NFT_SEC':        60,
    'TTL_ONCHAIN_SEC':    30,
    'TTL_FUNDING_SEC':    15,
    'TTL_LIQ_SEC':        30,
    'TTL_COLLATERAL_SEC': 2,
    # v6 data-freshness verification + /api/status hardening (display/robustness only)
    'DATA_STALE_SEC':          180,   # a configured feed with no ping in this long ⇒ flagged 'stale'
    'STATUS_UPSTREAM_TIMEOUT': 2.0,   # per-poll bound on the collateral/on-chain refresh (was hard 3.0)
    'STOP_LOSS_PCT': 0.035,
    'TAKE_PROFIT_BASE': 0.041,
}

# Apply the active profile (overrides base CONFIG keys). The profile is the
# single switch between the restored aggressive build, a balanced build, and
# the max-aggressive Nox Dark build.
def _apply_profile(cfg: dict) -> None:
    prof_name = cfg.get('ACTIVE_PROFILE', 'aggressive_hifreq')
    prof = cfg.get('PROFILES', {}).get(prof_name)
    if not prof:
        return
    for k, v in prof.items():
        cfg[k] = v
    # Keep the operating cap inside the risk-bounded band [10, MAX_LEVERAGE].
    cfg['LEVERAGE_CAP'] = int(min(max(cfg.get('LEVERAGE_CAP', 20),
                                      cfg.get('LEVERAGE_CAP_MIN', 10)),
                                  cfg.get('MAX_LEVERAGE', 40)))
    cfg['PAPER_LEVERAGE_CAP'] = int(min(cfg.get('PAPER_LEVERAGE_CAP', cfg['LEVERAGE_CAP']),
                                        cfg['MAX_LEVERAGE']))
_apply_profile(CONFIG)

# ============================================================
# SECTION 2: RATE LIMITERS
# ============================================================
helius_limiter    = Limiter(Rate(9,  Duration.SECOND))
lunarcrush_limiter= Limiter(Rate(4,  Duration.SECOND))
birdeye_limiter   = Limiter(Rate(10, Duration.SECOND))
polymarket_limiter= Limiter(Rate(5,  Duration.SECOND))

# ============================================================
# SECTION 3: GLOBAL STATE
# ============================================================
class TradingState:
    def __init__(self):
        self.balance              = CONFIG['STARTING_BALANCE']
        self.start_balance_today  = CONFIG['STARTING_BALANCE']
        self.initial_investment   = CONFIG['STARTING_BALANCE']
        self.halt_trading         = False
        self.trade_count          = 0
        self.trade_log: List[Dict]           = []
        self.open_positions: List[Dict]      = []
        self.daily_pnl: Dict[str, float]     = {}
        self.best_day             = {'date': None, 'pnl': 0.0, 'pct': 0.0}
        self.start_time           = datetime.now(timezone.utc)
        self.ws_data              = deque(maxlen=1000)
        self.model_performance    = {'accuracy': 0.5, 'precision': 0.5, 'recall': 0.5, 'f1': 0.5}
        self.current_regime       = 'unknown'
        self.sol_price            = 0.0
        self.onchain_balance      = 0.0
        self.available_collateral = 0.0
        self.unrealized_pnl       = 0.0
        self.funding_rate         = 0.0
        self.days_running         = 1
        self.last_trade_time: Optional[datetime] = None
        self.benchmarks: Dict     = {
            'total_signal_checks': 0,
            'signals_generated': 0,
            'filters_pass_rate': 0.0,
            'avg_confidence': 0.0,
            'ml_retrain_count': 0,
            'rl_update_count': 0,
            'circuit_breaker_count': 0,
            'uptime_seconds': 0,
        }
        self.macro_data: Dict     = {'dxy': 0.0, 'vix': 0.0, 'treasury_10y': 0.0}
        self.sentiment_score      = 0.0
        self.ai_augmentation: Dict = {}         # v6: latest AI data-augmentation verdict (display + haircut)
        self.news_sentiment: Dict = {}          # Q2: latest AI news/event-sentiment verdict (display-first)
        self.news_feeds_ok: int = 0             # v7 P3: RSS feeds that responded last fetch
        self._news_prev_price: float = 0.0      # v7 P3: SOL price at last news fetch (divergence calc)
        self._event_refreshed: bool = False     # v8 P-D: latched once caches were refreshed for an event
        self.ws_mark_price: float = 0.0         # v9 P-D: real-time Aster WS mark price
        self.ws_funding_rate: float = 0.0       # v9 P-D: real-time Aster WS funding rate
        self.ws_ts: float = 0.0                 # v9 P-D: epoch of last WS update (freshness)
        self.ob_features: Dict = {}             # v10 Track 2b: order-book depth shadow features (log-only)
        self.xv_features: Dict = {}             # research: cross-venue lead-lag shadow features (log-only)
        self.tv_ratings: Dict = {}              # TV-rating replica per TF (shadow/display only)
        self.tv_composites: Dict = {}           # TV alignment composites (shadow/display only)
        self.ti_features: Dict = {}             # extra indicator shadows: supertrend/vwap_dev/pivot_dist
        self.sr_levels: Dict = {}               # support/resistance ladder (display + shadow)
        self.sr_feat: Dict = {}                 # S/R shadow features (dist/strength/room)
        self.onchain_features: Dict = {}        # v10 Track 2: on-chain network-activity shadow features (log-only)
        self.historic_states: deque = deque(maxlen=500)
        self.lock                 = asyncio.Lock()

        # API health tracking — populated at startup
        self.api_health: Dict[str, Dict] = {}

        # Loss-cluster cooldown + adaptive threshold state
        self.consecutive_losses: int = 0
        self.consecutive_wins:   int = 0
        self.last_loss_time: Optional[datetime] = None
        self.adaptive_conf_offset: float = 0.0  # added to dynamic CONF threshold
        self.config_fingerprint: str = ''       # hash of outcome-affecting CONFIG (data-hygiene)
        self.opt_conf_floor: float = 0.0         # adaptive confidence floor (0 = use profile MIN)
        self.calib_a: Optional[float] = None     # v9 Phase A: Platt calibration slope  P(win)=sigmoid(a*conf+b)
        self.calib_b: Optional[float] = None     # v9 Phase A: Platt calibration intercept
        self.calib_n: int = 0                    # v9 Phase A: trades the calibrator was fit on
        self.last_risk_throttle: float = 1.0    # streak/drawdown size de-risk (≤1.0)

        # 24-hour cycle coordinator
        self.cycle_index:       int = 1
        self.cycle_start_time: datetime = datetime.now(timezone.utc)
        self.cycle_phase:       str = 'active'   # active | winding_down | summarizing
        self.cycle_trades:      List[Dict] = []  # trades opened *this* cycle
        self.cycle_summaries:   List[Dict] = []  # historical summaries

        # v4.00 fee-drag governor — fees spent so far this UTC day (reset daily)
        self.fees_today:        float = 0.0
        self.fee_day:           str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        self.governor_throttles: int = 0   # count of governor-blocked opens (this cycle)

        # v4.00 Grok-derived safe additions — telemetry/counters
        self.trade_limit_skips: int = 0    # opens blocked by the dynamic trade-limit governor
        self.last_trade_limit:  int = 0    # last computed dynamic daily trade cap
        self.data_breaker_skips: int = 0   # opens skipped by the data-confidence breaker
        self.last_slippage_bps: float = 0.0  # last modeled paper-fill slippage (bps)

        # NoxSSD data-drive safety guard (outcome-neutral; NOT fingerprinted) — opens are
        # blocked when the designated data drive is not mounted/accessible (debounced).
        self.noxssd_ok:          bool = True
        self.noxssd_block_opens: bool = False
        self.noxssd_skips:       int = 0      # opens skipped by the NoxSSD guard
        self.noxssd_last_check:  str = ''

        # Dashboard observability (DISPLAY-ONLY; never affects trading) ---------------
        self.display_epoch:         str = ''     # manual Reset/Checkpoint scope marker
        self.display_epoch_history: List = []    # undo stack for Checkpoint
        self.per_model_metrics:     Dict = {}    # per-base-model telemetry (MLEnsemble.fit)
        self.news_headlines:        List = []    # top recent headlines (news loop)
        self.last_quarantine:       Dict = {}    # last data-hygiene quarantine event (banner)
        self.live_model_proba:      Dict = {}    # live directional model conf (adj-conf source)

        # v4.00 venue / router telemetry (populated by the router each cycle)
        self.venue_health:      Dict = {}
        self.active_venue:      str = 'paper'
        self.cross_venue_basis: float = 0.0
        self.open_interest:     float = 0.0
        self.long_short_ratio:  float = 1.0
        self.liq_long:          float = 0.0
        self.liq_short:         float = 0.0

        # v4.00 capital-tier monitor — milestone notifications on equity growth
        self.capital_tier:      int = 0     # index into CAPITAL_TIERS that's been hit
        self.peak_equity:       float = CONFIG['STARTING_BALANCE']

state = TradingState()

# Capital milestones (USD equity). capital_tier_monitor() fires a dashboard
# banner + macOS notification + log line the first time each is crossed.
CAPITAL_TIERS = [150, 250, 500, 1_000, 2_500, 5_000, 10_000, 25_000, 50_000, 100_000]

# ============================================================
# SECTION 3a: STATE PERSISTENCE (survives restarts)
# ============================================================
def _config_fingerprint() -> str:
    """Hash the CONFIG keys that govern trade OUTCOMES (stop/TP/sizing/leverage/threshold).
    A change invalidates the trade-outcome learners — they must not carry stats from one
    rule-set into a different one (the data-poisoning the user flagged)."""
    keys = [
        'NAME', 'SIZING_MODE', 'LEVERAGE_CAP', 'PAPER_LEVERAGE_CAP', 'RISK_PER_TRADE',
        'MIN_EV_RATIO', 'CONFIDENCE_THRESHOLD_MIN', 'CONFIDENCE_THRESHOLD_MAX',
        'DSL_RR_REDESIGN', 'DSL_BREAKEVEN_ENABLE', 'DSL_BE_TRIGGER_R', 'DSL_MAX_INIT_SL_PCT',
        'DSL_INIT_ATR_MULT_MIN', 'DSL_INIT_ATR_MULT_MAX', 'DSL_TRAIL_ATR_MULT',
        'DSL_TRAIL_NOISE_ATR_MULT', 'DSL_TRAIL_MIN_MULT', 'DSL_TP_LADDER_ENABLE',
        'DSL_TP1_R', 'DSL_TP1_FRACTION', 'MIN_TP_RR_RATIO', 'MIN_SL_PCT_OF_PRICE',
        'CONF_SCALED_LEV_ENABLE', 'CONF_OPT_ENABLE',
        # v7 left-tail/calibration levers (outcome-affecting → own clean cohort)
        'CONF_LEV_USE_CALIBRATION', 'CONF_LEV_CALIB_MIN', 'MAX_TRADE_LOSS_PCT', 'LOSS_CAP_LEVERAGED',
        'LOSS_CAP_BACKSTOP',
        # v9 Phase A keys are intentionally NOT fingerprinted while CALIB_ENABLE=False (dormant);
        # re-add 'CALIB_ENABLE','CALIB_LO','CALIB_HI' here when activating it (then it owns a cohort).
    ]
    blob = json.dumps({k: CONFIG.get(k) for k in keys}, sort_keys=True, default=str)
    return hashlib.blake2b(blob.encode(), digest_size=8).hexdigest()


def calibrate_confidence(conf: float) -> float:
    """v9 Phase A — map raw confidence → calibrated win-probability via the fitted Platt model.
    Returns raw conf unchanged until the calibrator is fit (safe default)."""
    a, b = getattr(state, 'calib_a', None), getattr(state, 'calib_b', None)
    if a is None or b is None:
        return float(conf)
    try:
        return 1.0 / (1.0 + math.exp(-(a * float(conf) + b)))
    except Exception:
        return float(conf)

def fit_confidence_calibration():
    """v9 Phase A — fit P(win)=sigmoid(a*conf+b) on the CURRENT rule-set's closed trades so the
    compressed/under-confident raw confidence is mapped to TRUE win-probability (then used to size
    leverage). Robust 2-param logistic; no-op until CALIB_MIN_TRADES accrue; refit on the recal hook."""
    if not CONFIG.get('CALIB_ENABLE', True):
        return
    # confidence→win is SIZING-INVARIANT (sizing changes $ PnL, not win/loss), so fit on the last
    # CALIB_FIT_WINDOW closed trades that carry a cfg_fp (= post-payoff-fix, same geometry; excludes
    # the pre-fingerprint inverted-geometry cohort). This lets the calibrator activate IMMEDIATELY on
    # accrued history even right after a fingerprint bump, then refit as the new cohort grows.
    rows = [t for t in list(state.trade_log)
            if isinstance(t, dict) and t.get('status') == 'closed'
            and t.get('confidence') is not None and t.get('cfg_fp')][-int(CONFIG.get('CALIB_FIT_WINDOW', 120)):]
    if len(rows) < int(CONFIG.get('CALIB_MIN_TRADES', 40)):
        return
    X = np.array([[float(t['confidence'])] for t in rows], dtype=float)
    y = np.array([1 if (float(t.get('pnl_usd', 0) or 0) - float(t.get('fee_usd', 0) or 0)) > 0 else 0
                  for t in rows], dtype=int)   # NET-of-fees win label
    if len(np.unique(y)) < 2:
        return
    try:
        m = LogisticRegression(C=1.0, max_iter=300).fit(X, y)
        state.calib_a = float(m.coef_[0][0]); state.calib_b = float(m.intercept_[0])
        state.calib_n = len(rows)
        logger.info(f"  [calib] fit on {len(rows)} trades: P(win)=sigmoid({state.calib_a:.2f}*conf"
                    f"{state.calib_b:+.2f}) | calib(0.55)={calibrate_confidence(0.55):.2f} "
                    f"calib(0.60)={calibrate_confidence(0.60):.2f}")
    except Exception as e:
        logger.debug(f"calib fit: {e}")

def optimize_confidence_floor():
    """From CLOSED trades made under the CURRENT config fingerprint, pick the confidence
    floor maximizing total net PnL while keeping >=CONF_OPT_MIN_KEEP of volume; set
    state.opt_conf_floor (blended with the profile floor). No-op until MIN_TRADES accrue."""
    if not CONFIG.get('CONF_OPT_ENABLE', True):
        return
    fp = getattr(state, 'config_fingerprint', '')
    rows = [t for t in list(state.trade_log)
            if isinstance(t, dict) and t.get('status') == 'closed'
            and t.get('cfg_fp') == fp and t.get('confidence') is not None]
    if len(rows) < int(CONFIG.get('CONF_OPT_MIN_TRADES', 40)):
        return
    confs = np.array([float(t['confidence']) for t in rows], dtype=float)
    pnls = np.array([float(t.get('pnl_usd', 0) or 0) - float(t.get('fee_usd', 0) or 0)
                     for t in rows], dtype=float)
    n = len(rows); best_f, best_net = None, -1e18
    for f in CONFIG.get('CONF_OPT_GRID', [0.51, 0.55, 0.58, 0.62, 0.66, 0.70]):
        keep = confs >= float(f)
        if keep.mean() < float(CONFIG.get('CONF_OPT_MIN_KEEP', 0.30)):
            continue
        net = float(pnls[keep].sum())
        if net > best_net:
            best_net, best_f = net, float(f)
    if best_f is not None:
        blend = float(CONFIG.get('CONF_OPT_BLEND', 0.5))
        prof_min = float(CONFIG['CONFIDENCE_THRESHOLD_MIN'])
        state.opt_conf_floor = blend * best_f + (1.0 - blend) * prof_min
        logger.info(f"  [conf-opt] floor->{state.opt_conf_floor:.3f} "
                    f"(grid {best_f:.2f}, net=${best_net:+.2f}, n={n})")


def persist_state():
    """Atomically write minimal state to disk so a restart doesn't lose
    trade history, balance, drawdown peak, or RL learnings."""
    try:
        snap = {
            'version': CONFIG['VERSION'],
            'saved_at': datetime.now(timezone.utc).isoformat(),
            'config_fingerprint': getattr(state, 'config_fingerprint', '') or _config_fingerprint(),
            'opt_conf_floor': getattr(state, 'opt_conf_floor', 0.0),
            'balance': state.balance,
            'initial_investment': state.initial_investment,
            'start_balance_today': state.start_balance_today,
            'trade_count': state.trade_count,
            'days_running': state.days_running,
            'best_day': state.best_day,
            'consecutive_losses': state.consecutive_losses,
            'consecutive_wins':   state.consecutive_wins,
            'adaptive_conf_offset': state.adaptive_conf_offset,
            'fees_today': state.fees_today,
            'fee_day': state.fee_day,
            'capital_tier': state.capital_tier,
            'peak_equity': state.peak_equity,
            'benchmarks': state.benchmarks,
            'model_performance': state.model_performance,
            'display_epoch': state.display_epoch,                       # display-only view scope
            'display_epoch_history': state.display_epoch_history[-20:], # display-only undo stack
            'trade_log_tail': list(state.trade_log[-200:]),  # bounded
            'quant_overlay': {
                'peak_balance': getattr(quant_overlay, 'peak_balance', state.balance),
                'max_drawdown': getattr(quant_overlay, 'max_drawdown', 0.0),
                'metalabel_buckets': {
                    f"{k[0]}|{k[1]}": v for k,v in
                    getattr(quant_overlay, 'metalabel_buckets', {}).items()
                },
            },
        }
        tmp = CONFIG['STATE_PERSIST_FILE'] + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(snap, f, indent=2, default=str)
        os.replace(tmp, CONFIG['STATE_PERSIST_FILE'])
    except Exception as e:
        logger.debug(f"persist_state error: {e}")

def restore_state():
    """Load prior state if present. Safe no-op on first launch."""
    path = CONFIG.get('STATE_PERSIST_FILE')
    if not path or not os.path.exists(path):
        return
    try:
        with open(path, 'r') as f:
            snap = json.load(f)
        # --- data-hygiene fingerprint quarantine -------------------------------
        cur_fp = _config_fingerprint(); state.config_fingerprint = cur_fp
        old_fp = str(snap.get('config_fingerprint', '') or '')
        # Quarantine on ANY mismatch — INCLUDING a missing fp (a pre-fingerprint state whose
        # learners are a poisoned mix of older rule-sets, which is exactly the case to clean).
        quarantine = bool(CONFIG.get('LEARN_FINGERPRINT_GATE', True)) and old_fp != cur_fp
        state.balance              = float(snap.get('balance', state.balance))
        state.initial_investment   = float(snap.get('initial_investment', state.initial_investment))
        state.start_balance_today  = float(snap.get('start_balance_today', state.start_balance_today))
        state.trade_count          = int(snap.get('trade_count', 0))
        state.days_running         = int(snap.get('days_running', 1))
        state.best_day             = snap.get('best_day', state.best_day)
        state.consecutive_losses   = int(snap.get('consecutive_losses', 0))
        state.consecutive_wins     = int(snap.get('consecutive_wins', 0))
        state.adaptive_conf_offset = 0.0 if quarantine else float(snap.get('adaptive_conf_offset', 0.0))
        state.opt_conf_floor       = 0.0 if quarantine else float(snap.get('opt_conf_floor', 0.0))
        # Restore fee-governor + capital-tier so neither resets/re-fires on restart.
        _today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        if snap.get('fee_day') == _today:
            state.fees_today = float(snap.get('fees_today', 0.0))
            state.fee_day = snap['fee_day']
        state.capital_tier = int(snap.get('capital_tier', 0))
        state.peak_equity  = float(snap.get('peak_equity', state.balance))
        if isinstance(snap.get('benchmarks'), dict):
            state.benchmarks.update(snap['benchmarks'])
        if isinstance(snap.get('model_performance'), dict):
            state.model_performance.update(snap['model_performance'])
        # display-epoch (manual Reset/Checkpoint) — display-only, survives restart, fp-independent
        state.display_epoch = str(snap.get('display_epoch', '') or '')
        if isinstance(snap.get('display_epoch_history'), list):
            state.display_epoch_history = list(snap.get('display_epoch_history'))
        tail = snap.get('trade_log_tail', [])
        if isinstance(tail, list):
            # Orphan reconciliation (2026-06-10): open_positions are NOT persisted, so any
            # restored row still status='open' is definitionally a dead remnant of a prior run
            # (its position no longer exists). Mark it 'orphaned' — learners read 'closed' only
            # (unaffected); the dashboard stops showing phantom OPEN rows forever.
            _orph = 0
            for _t in tail:
                if isinstance(_t, dict) and _t.get('status') == 'open':
                    _t['status'] = 'orphaned'; _orph += 1
            if _orph:
                logger.info(f"  [ledger] reconciled {_orph} orphaned open row(s) from prior runs")
            state.trade_log.extend(tail)
        qov = snap.get('quant_overlay', {})
        if qov.get('peak_balance'):
            quant_overlay.peak_balance = float(qov['peak_balance'])
        if qov.get('max_drawdown') is not None:
            quant_overlay.max_drawdown = float(qov['max_drawdown'])
        if not quarantine and isinstance(qov.get('metalabel_buckets'), dict):
            for k, v in qov['metalabel_buckets'].items():
                if '|' in k and isinstance(v, list) and len(v) == 2:
                    parts = k.split('|', 1)
                    quant_overlay.metalabel_buckets[(parts[0], parts[1])] = list(v)
        if quarantine:
            # archive the RL pkl so the (now-after) rl_stop_optimizer.load() cold-starts;
            # trade_pnls (Kelly source) isn't persisted → already cold-starts.
            try:
                _rlp = 'rl_optimizer.pkl'
                if os.path.exists(_rlp):
                    os.replace(_rlp, _rlp + f".pre_fp_{int(time.time())}.bak")
            except Exception:
                pass
            quant_overlay.metalabel_buckets.clear()
            state.last_quarantine = {'old_fp': old_fp[:8], 'new_fp': cur_fp[:8],
                                     'at': datetime.now(timezone.utc).isoformat(),
                                     'reset': ['trade_pnls (Kelly)', 'meta_label_buckets',
                                               'adaptive_conf_offset', 'opt_conf_floor', 'rl_optimizer'],
                                     'preserved': ['ML/DL ensemble', 'vector_db (candle-trained)']}
            logger.warning(f"LEARN-QUARANTINE: outcome-config changed ({old_fp}->{cur_fp}); "
                           f"reset trade-outcome learners (meta-label/adaptive-offset/RL). "
                           f"ML/DL + vector DB unaffected (candle-trained).")
        logger.info(f"Restored state: balance=${state.balance:.4f} "
                    f"trades={state.trade_count} "
                    f"adaptive_offset={state.adaptive_conf_offset:+.3f}")
    except Exception as e:
        logger.warning(f"restore_state error (continuing fresh): {e}")

# ============================================================
# SECTION 3b: API HEALTH REGISTRY
# ============================================================
API_REGISTRY = {
    # category, display name, required_key flag
    'aster':          ('Venues',       'Aster (primary)',   False),
    'apex':           ('Venues',       'ApeX Omni (failover)', False),
    'solana_rpc':     ('Market Data',  'Solana RPC',     False),
    'helius':         ('Sentiment',    'Helius RPC',     True),
    'drift':          ('Legacy',       'Drift (retired)', False),
    'birdeye':        ('Market Data',  'Birdeye',        True),
    'kraken':         ('Market Data',  'Kraken OHLCV',   False),
    'coingecko':      ('Market Data',  'CoinGecko',      False),
    'cmc':            ('Market Data',  'CoinMarketCap',  True),
    'messari':        ('Market Data',  'Messari',        True),
    'lunarcrush':     ('Sentiment',    'LunarCrush',     True),
    'reddit':         ('Sentiment',    'Reddit (cookieless)', False),
    'twitter':        ('Sentiment',    'X / Twitter',    True),
    'pytrends':       ('Sentiment',    'Google Trends',  False),
    'gnews':          ('Sentiment',    'GNews',          True),
    'news_rss':       ('Sentiment',    'Crypto News (RSS)', False),
    'fred':           ('Macro',        'FRED',           True),
    'dune':           ('Analytics',    'Dune Analytics', True),
    'polymarket':     ('Prediction',   'Polymarket',     False),
}

def _init_api_health():
    key_map = {
        'helius': 'HELIUS_API_KEY', 'birdeye': 'BIRDEYE_API_KEY',
        'cmc': 'CMC_API_KEY', 'messari': 'MESSARI_API_KEY',
        'lunarcrush': 'LUNARCRUSH_API_KEY', 'twitter': 'X_BEARER_TOKEN',
        'gnews': 'GNEWS_API_KEY', 'fred': 'FRED_API_KEY',
        'dune': 'DUNE_API_KEY',
    }
    for key, (cat, name, needs_key) in API_REGISTRY.items():
        configured = True
        if needs_key:
            env_var = key_map.get(key, '')
            configured = bool(os.getenv(env_var, '').strip()) if env_var else False
        state.api_health[key] = {
            'category': cat, 'name': name, 'status': 'idle',
            'configured': configured, 'last_ping': None,
            'last_latency_ms': None, 'last_error': None,
            'success_count': 0, 'fail_count': 0,
        }

def record_api(key: str, ok: bool, latency_ms: float = 0.0, error: str = ''):
    h = state.api_health.get(key)
    if not h: return
    h['last_ping'] = datetime.now(timezone.utc).isoformat()
    h['last_latency_ms'] = round(latency_ms, 1)
    h['status'] = 'ok' if ok else 'fail'
    if ok:
        h['success_count'] += 1
        h['last_error'] = None
    else:
        h['fail_count'] += 1
        h['last_error'] = (error[:120] if error else 'request failed')

_init_api_health()

# ============================================================
# SECTION 3a-util: SHARED HTTP SESSION + TTL CACHE + SAFE WRAPPER
# ============================================================
_http_session: Optional[aiohttp.ClientSession] = None

async def get_http() -> aiohttp.ClientSession:
    """Single process-wide aiohttp session with a sane connector pool.
    Avoids fresh TCP+TLS handshake per fetch (~30-80ms saved each call)."""
    global _http_session
    if _http_session is None or _http_session.closed:
        _http_session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=20, ttl_dns_cache=300),
            timeout=aiohttp.ClientTimeout(total=12),
        )
    return _http_session

_ttl_cache: Dict[str, Tuple[float, Any]] = {}

def _ttl_scale() -> float:
    """v9 Phase D adaptive-TTL multiplier: shorten every cache when vol/event-risk is high (fresher
    data when it matters), lengthen slightly when calm. Bounded [0.4, 1.2] so it never spams the
    APIs or over-stales. ~1.0 in calm conditions (no change); only deviates during vol/events."""
    if not CONFIG.get('ADAPTIVE_TTL_ENABLE', True):
        return 1.0
    try:
        ns = getattr(state, 'news_sentiment', {}) or {}
        er = float(ns.get('event_risk', 0.0) or 0.0) if ns.get('ok') else 0.0
        vol = float(quant_overlay.garch_vol() or 0.02)
        vol_f = min(1.0, vol / 0.04)            # 0.04 ≈ elevated 1m vol
        return max(0.4, min(1.2, 1.0 - 0.5 * er - 0.3 * vol_f))
    except Exception:
        return 1.0

def ttl_get(key: str, max_age_sec: float):
    item = _ttl_cache.get(key)
    if item is None: return None
    ts, val = item
    return val if (time.time() - ts) <= (max_age_sec * _ttl_scale()) else None

def ttl_put(key: str, value: Any):
    _ttl_cache[key] = (time.time(), value)

def _safe(fn, *args, **kwargs):
    """Run fn(); swallow + debug-log on exception. Centralises the
    repeated `try: x(); except Exception: pass` pattern across the script."""
    try: return fn(*args, **kwargs)
    except Exception as e:
        logger.debug(f"_safe({getattr(fn,'__name__','fn')}): {e}")
        return None


# ============================================================
# SECTION 3c: QUANT OVERLAY (elite-grade signal & risk engine)
# ============================================================
# Layered over the base signal generator. Implements:
#   • Bayesian evidence aggregation       (replaces brittle N/9 filter count)
#   • GARCH(1,1) conditional volatility    (forecast σ for sizing)
#   • Triple-barrier labels                (López de Prado labeling for ML)
#   • Deflated / Probabilistic Sharpe      (Bailey & López de Prado)
#   • CVaR-95 position sizing              (Expected Shortfall budget)
#   • Engle-Granger cointegration (SOL/BTC) residual z-score
#   • Dynamic drawdown circuit breaker     (rolling max-DD halt)
#   • Regime-conditional Bayesian priors
# ============================================================

# Per-filter informational content as likelihood ratios.
# LR_pass = P(filter passes | edge) / P(filter passes | no edge)
# LR_fail = P(filter fails  | edge) / P(filter fails  | no edge)
# Rare-but-strong filters carry high LR_pass and LR_fail close to 1
# (their absence conveys little). Sanity filters are weak in both directions.
FILTER_LR: Dict[str, Tuple[float, float]] = {
    # Calibrated for low-vol crypto: sparse filters carry LR_fail near 1.0
    # (their absence conveys little). Medium filters carry meaningful both-ways.
    'fib_confluence': (3.0, 0.99),   # very rare; absence not informative
    'vpin':           (2.3, 0.96),   # informed-trading signal
    'volume_spike':   (2.0, 0.94),
    'adx_trend':      (1.9, 0.85),   # trend-regime confirmation (active)
    'order_flow':     (1.7, 0.90),
    'boll_keltner':   (1.6, 0.90),
    'inst_volume':    (1.4, 0.90),
    'volume_profile': (1.35, 0.92),
    'atr_range':      (1.2, 0.88),   # liquidity/vol sanity
}

# Regime-conditional posterior gates. Lowered after calibration showed
# the original 0.55-0.62 band rejected nearly all bars in low-vol regimes.
# Quality is preserved by the downstream ML+DL confidence + EV ratio gates.
REGIME_POSTERIOR_GATE = {
    'trending':       0.50,
    'mean_reverting': 0.54,
    'choppy':         0.58,
    'unknown':        0.58,
}

class QuantOverlay:
    def __init__(self):
        self.returns       = deque(maxlen=1000)
        self.trade_pnls    = deque(maxlen=500)
        self.sol_log_px    = deque(maxlen=500)
        self.btc_log_px    = deque(maxlen=500)
        self.sentiments    = deque(maxlen=500)
        # GARCH(1,1) state — calibrated for SOL 1m realised vol (~0.20–0.28%/bar,
        # ≈150–200% annualised). The old (ω=1e-6, σ²₀=4e-4) implied an
        # unconditional vol of ~0.7%/bar (~510% annualised) and a 2%/bar
        # (~1450% annualised) cold-start — a ~3–7× over-estimate that suppressed
        # vol-target leverage to the 1x floor and over-conservatived CVaR sizing.
        # ω=1.25e-7 with α+β=0.98 ⇒ unconditional vol √(ω/(1-α-β))≈0.25%/bar
        # (~181% annual); σ²₀ cold-start ≈0.30%/bar (~218% annual). The 50-sample
        # auto-calibration (update_return) still refines σ² to the live sample var.
        self.garch_omega   = 1.25e-7
        self.garch_alpha   = 0.10
        self.garch_beta    = 0.88
        self.garch_sigma2  = 9e-6
        # Drawdown tracking
        self.peak_balance  = CONFIG.get('STARTING_BALANCE', 100.0)
        self.max_drawdown  = 0.0
        # Information Coefficient & meta-label tracking
        self.ic_forecasts  = deque(maxlen=300)
        self.ic_realized   = deque(maxlen=300)
        # (regime, action) → [wins, losses]
        self.metalabel_buckets: Dict[Tuple[str,str], List[int]] = {}
        # Cached metrics for dashboard
        self.last_posterior      = 0.5
        self.last_log_odds       = 0.0
        self.last_garch_vol      = 0.003
        self.last_deflated_sr    = 0.0
        self.last_cointeg_z      = 0.0
        self.last_bocpd_p        = 0.0
        self.last_crowded_score  = 0.0
        self.last_reflexivity    = 0.0
        self.last_vol_target_lev = 5
        self.last_ic             = 0.0
        self.last_doppel_mean    = 0.0
        self.last_doppel_std     = 0.0
        self.last_metalabel_wr   = 0.5
        self.bayes_passes        = 0
        self.bayes_blocks        = 0
        self.bocpd_halts         = 0
        # Round-3 elite caches
        self.funding_history          = deque(maxlen=300)
        self.last_cf_cvar             = 0.0
        self.last_funding_z           = 0.0
        self.last_oracle_mark_spread  = 0.0
        self.last_cusum_event         = False
        self.last_pess_kelly          = 0.10
        self.last_ofi_persistence     = 0.0
        self.last_mtf_alignment       = 0.0
        # Regime caches for the blended dynamic-pessimistic sizing multiplier
        # (stashed by the signal path each tick; read by dynamic_pessimistic_kelly).
        self.last_hurst               = 0.5
        self.last_ewma_vol            = 0.02
        self.last_dyn_kelly           = 0.0   # last dynamic size emitted (dashboard)
        self.cusum_blocks             = 0
        self.oracle_arb_signals       = 0
        # BOCPD confirmation window — require two-of-five spikes before halt
        self.bocpd_history            = deque(maxlen=5)
        # Memo keys for expensive recomputes (rev'd when underlying deque grows)
        self._cf_cvar_rev    = -1   ; self._cf_cvar_cached    = 0.0
        self._pess_kelly_rev = -1   ; self._pess_kelly_cached = 0.05
        self._mtf_last_index = None ; self._mtf_cached        = (0.0, {})
        self._cointeg_rev    = -1   ; self._cointeg_cached    = 0.0

    # ---- Bayesian evidence aggregation ----
    def bayesian_posterior(self, passed: List[str], failed: List[str],
                           prior: float = 0.5) -> Tuple[float, float]:
        """Naive-Bayes log-odds update across filter evidences.
        LRs are clamped to [0.05, 20] to prevent any single mis-tuned entry from
        steamrolling the posterior; log_odds clamped to [-25, +25]."""
        if not (0 < prior < 1):
            prior = 0.5
        log_odds = math.log(prior / (1 - prior))
        for name in passed:
            lr_pass, _ = FILTER_LR.get(name, (1.2, 0.88))
            lr_pass = float(np.clip(lr_pass, 0.05, 20.0))
            log_odds += math.log(lr_pass)
        for name in failed:
            _, lr_fail = FILTER_LR.get(name, (1.2, 0.88))
            lr_fail = float(np.clip(lr_fail, 0.05, 20.0))
            log_odds += math.log(lr_fail)
        log_odds = max(-25.0, min(25.0, log_odds))
        try:
            posterior = 1.0 / (1.0 + math.exp(-log_odds))
        except OverflowError:
            posterior = 1.0 if log_odds > 0 else 0.0
        self.last_posterior = float(posterior)
        self.last_log_odds  = float(log_odds)
        return posterior, log_odds

    # ---- GARCH(1,1) volatility forecast ----
    def update_return(self, r: float):
        if r is None or not np.isfinite(r): return
        self.returns.append(float(r))
        # Auto-calibrate σ² on first reach to 50 samples (mitigates cold-start
        # mis-specification of the default ω/α/β triple).
        if len(self.returns) == 50:
            sample_var = float(np.var(list(self.returns), ddof=1))
            if sample_var > 0:
                self.garch_sigma2 = sample_var
        self.garch_sigma2 = (self.garch_omega
                             + self.garch_alpha * (r * r)
                             + self.garch_beta  * self.garch_sigma2)
        self.last_garch_vol = float(np.sqrt(max(self.garch_sigma2, 1e-12)))

    def garch_vol(self) -> float:
        return self.last_garch_vol

    # ---- Triple-barrier ML labels (offline use) ----
    @staticmethod
    def triple_barrier(prices: np.ndarray, atr: np.ndarray,
                       tp_mult: float = 2.0, sl_mult: float = 1.0,
                       max_hold: int = 20) -> np.ndarray:
        n = len(prices)
        labels = np.zeros(n, dtype=int)
        for i in range(n - max_hold):
            entry = prices[i]; a = atr[i]
            if a <= 0: continue
            tp = entry + tp_mult * a
            sl = entry - sl_mult * a
            win = prices[i+1:i+1+max_hold]
            tp_idx = np.argmax(win >= tp) if (win >= tp).any() else max_hold + 1
            sl_idx = np.argmax(win <= sl) if (win <= sl).any() else max_hold + 1
            if tp_idx < sl_idx:   labels[i] = 1
            elif sl_idx < tp_idx: labels[i] = -1
        return labels

    # ---- Deflated / Probabilistic Sharpe ----
    def deflated_sharpe(self) -> float:
        if len(self.trade_pnls) < 30: return 0.0
        r = np.array(self.trade_pnls, dtype=float)
        mu, sd = r.mean(), r.std(ddof=1)
        if sd <= 1e-12: return 0.0
        sharpe = (mu / sd) * np.sqrt(252.0)
        n = len(r)
        skew = float(((r - mu)**3).mean() / (sd**3))
        kurt = float(((r - mu)**4).mean() / (sd**4))
        denom = max(1.0 - skew*sharpe + (kurt - 1.0)/4.0 * (sharpe**2), 1e-6)
        psr = sharpe * np.sqrt(max((n - 1)/denom, 0.0))
        self.last_deflated_sr = float(psr)
        return self.last_deflated_sr

    # ---- CVaR-95 position sizing ----
    def cvar_position_notional(self, balance: float, edge_conf: float,
                               leverage: int, max_loss_pct: float = None) -> float:
        """Cap Expected Shortfall (95%) at `max_loss_pct` of balance,
        using Cornish-Fisher fat-tail-adjusted CVaR when enough history
        exists. Falls back to Gaussian 2.062σ otherwise."""
        if max_loss_pct is None:
            max_loss_pct = CONFIG.get('RISK_PER_TRADE', 0.02)
        # Prefer CF-CVaR (fat-tail aware) over Gaussian when we have history
        cf = self.cornish_fisher_cvar()
        sigma = max(self.garch_vol(), 1e-4)
        es_per_unit = max(cf, 2.062 * sigma) * leverage   # take the WORSE estimate
        if es_per_unit <= 0: return 0.0
        budget = balance * max_loss_pct
        notional = budget / es_per_unit
        scale = max(0.0, min(2.0 * edge_conf - 1.0, 1.0))
        return float(max(0.0, notional * scale))

    # ---- Engle-Granger cointegration residual ----
    def update_cointeg(self, sol_price: float, btc_price: float):
        if sol_price > 0 and btc_price > 0:
            self.sol_log_px.append(math.log(sol_price))
            self.btc_log_px.append(math.log(btc_price))

    def cointeg_zscore(self) -> float:
        if len(self.sol_log_px) < 100 or len(self.btc_log_px) < 100:
            return 0.0
        s = np.array(self.sol_log_px); b = np.array(self.btc_log_px)
        X = np.column_stack([np.ones_like(b), b])
        try:
            beta, *_ = np.linalg.lstsq(X, s, rcond=None)
            resid = s - X @ beta
            z = (resid[-1] - resid.mean()) / (resid.std(ddof=1) + 1e-9)
            self.last_cointeg_z = float(z)
            return self.last_cointeg_z
        except Exception:
            return 0.0

    # ---- Drawdown circuit breaker ----
    def update_drawdown(self, balance: float) -> Tuple[float, bool]:
        self.peak_balance = max(self.peak_balance, balance)
        dd = (balance - self.peak_balance) / self.peak_balance
        self.max_drawdown = min(self.max_drawdown, dd)
        halt = dd <= -CONFIG.get('MAX_TOTAL_DRAWDOWN', 0.25)
        return dd, halt

    def record_trade_pnl(self, pnl_usd: float, capital: float):
        if capital > 0:
            self.trade_pnls.append(pnl_usd / capital)
        self.deflated_sharpe()

    # ---- Volatility-targeted leverage (constant-vol portfolio) ----
    def vol_target_leverage(self, target_annual_vol: float = None,
                            max_lev: int = None) -> int:
        """Inverse-vol (risk-parity) leverage, anchored on a crypto-realistic
        reference so the user's leverage band is actually usable.

        leverage = DEFAULT_LEVERAGE × (VOL_REF_ANNUAL / annualised_realised_vol)

        At the reference vol the engine runs its DEFAULT leverage; when realised
        vol rises above the reference it de-levers (and vice-versa), all clipped
        to [1, max_lev]. A fixed 15% annual target (AQR/Bridgewater equities
        style) is degenerate on SOL — its ~150% annualised vol pins leverage to
        the 1x floor permanently, neutering the 10–40x band. The operating cap
        (LEVERAGE_CAP) + SL-implied lev + liq-distance guard + fee-drag governor
        remain the real downstream constraints."""
        if max_lev is None:
            max_lev = CONFIG.get('MAX_LEVERAGE', 40)
        sigma = max(self.garch_vol(), 1e-5)
        # 1m bars: 60*24*365 ≈ 525k periods/yr; sqrt(525k) ≈ 725
        annualized = sigma * 725.0
        if target_annual_vol is None:
            # DEFAULT_LEVERAGE worth of vol at the reference => lev≈DEFAULT there.
            target_annual_vol = (CONFIG.get('DEFAULT_LEVERAGE', 20)
                                 * CONFIG.get('VOL_REF_ANNUAL', 1.5))
        lev = target_annual_vol / max(annualized, 1e-5)
        lev = int(np.clip(lev, 1, max_lev))
        self.last_vol_target_lev = lev
        return lev

    # ---- Bayesian Online Changepoint Detection (simplified) ----
    def bocpd_proba(self, window: int = 50) -> float:
        """P(structural break in last `window` bars).
        Welch-t test between first/second half of GARCH-standardised returns,
        converted to a [0,1] changepoint probability.
        Warm-up grace: returns 0.0 until we have 2*window samples so GARCH
        self-calibration doesn't spuriously trigger a 'changepoint'."""
        if len(self.returns) < window * 2:
            self.last_bocpd_p = 0.0
            self.bocpd_history.append(0.0)
            return 0.0
        r = np.array(list(self.returns)[-window:], dtype=float)
        n = len(r); h = n // 2
        a, b = r[:h], r[h:]
        mu_a, mu_b = a.mean(), b.mean()
        var_a, var_b = a.var(ddof=1), b.var(ddof=1)
        se = np.sqrt(var_a/max(len(a),1) + var_b/max(len(b),1) + 1e-12)
        t = abs(mu_a - mu_b) / max(se, 1e-12)
        # Variance-shift component (Levene-style) — heavily dampened to avoid
        # over-firing on routine vol clustering.
        f_ratio = (max(var_a, var_b) + 1e-12) / (min(var_a, var_b) + 1e-12)
        f_signal = math.log(max(f_ratio, 1.0))   # only positive shifts; ratio>=1
        # Combine — variance coefficient dropped from 0.4 to 0.05
        score = 0.5 * t * t + 0.05 * f_signal * f_signal
        try:
            p = 1.0 - math.exp(-0.4 * score)
        except OverflowError:
            p = 1.0
        p = float(np.clip(p, 0.0, 1.0))
        self.last_bocpd_p = p
        self.bocpd_history.append(p)
        return p

    def bocpd_confirmed_halt(self, hard_threshold: float = 0.95,
                              spike_threshold: float = 0.85,
                              required_spikes: int = 2) -> bool:
        """Only halt when (a) latest BOCPD ≥ hard_threshold, OR
        (b) ≥ required_spikes recent bars all > spike_threshold.
        Eliminates single-bar false alarms in low-vol drift regimes."""
        if not self.bocpd_history:
            return False
        latest = self.bocpd_history[-1]
        if latest >= hard_threshold:
            return True
        spikes = sum(1 for p in self.bocpd_history if p > spike_threshold)
        return spikes >= required_spikes

    # ---- Crowded-trade detector ----
    def crowded_trade_score(self, funding_rate: float, sentiment: float) -> float:
        """Same-side extremity of funding & sentiment ⇒ counter-trend setup.
        Returns score in [-1, +1]:
          +1 = extremely crowded long (contrarian short setup)
          -1 = extremely crowded short (contrarian long setup / squeeze)"""
        f = float(np.clip(funding_rate * 1000.0, -1.0, 1.0))  # funding scaled
        s = float(np.clip(sentiment, -1.0, 1.0))
        score = 0.0
        if f * s > 0:  # same sign
            score = float(np.sign(f) * min(abs(f), abs(s)))
        self.last_crowded_score = score
        return score

    # ---- Reflexivity score (Soros feedback loop) ----
    def update_sentiment(self, sentiment: float):
        if sentiment is not None and np.isfinite(sentiment):
            self.sentiments.append(float(sentiment))

    def reflexivity_score(self, window: int = 30) -> float:
        """Rolling Pearson corr between Δsentiment and Δprice.
        |corr| → 1 ⇒ strong self-reinforcing feedback (Soros reflexivity).
        Sign indicates direction of loop."""
        if len(self.returns) < window or len(self.sentiments) < window:
            self.last_reflexivity = 0.0
            return 0.0
        n = min(len(self.returns), len(self.sentiments), window)
        r = np.array(list(self.returns)[-n:])
        s = np.array(list(self.sentiments)[-n:])
        ds = np.diff(s)
        if len(ds) < 5 or r[-len(ds):].std() == 0 or ds.std() == 0:
            self.last_reflexivity = 0.0
            return 0.0
        try:
            c = float(np.corrcoef(r[-len(ds):], ds)[0, 1])
            if not np.isfinite(c):
                c = 0.0
        except Exception:
            c = 0.0
        self.last_reflexivity = c
        return c

    # ---- Information Coefficient tracking ----
    def record_forecast_realized(self, forecast: float, realized: float):
        if np.isfinite(forecast) and np.isfinite(realized):
            self.ic_forecasts.append(float(forecast))
            self.ic_realized.append(float(realized))

    def information_coefficient(self) -> float:
        if len(self.ic_forecasts) < 20:
            self.last_ic = 0.0
            return 0.0
        f = np.array(self.ic_forecasts); r = np.array(self.ic_realized)
        if f.std() == 0 or r.std() == 0:
            self.last_ic = 0.0
            return 0.0
        try:
            ic = float(np.corrcoef(f, r)[0, 1])
            if not np.isfinite(ic):
                ic = 0.0
        except Exception:
            ic = 0.0
        self.last_ic = ic
        return ic

    # ---- Doppelganger / K-Nearest Pattern Forward Returns ----
    def doppelganger_forecast(self, current_feat: np.ndarray,
                              hist_feats: np.ndarray, hist_returns: np.ndarray,
                              k: int = 7, weights: np.ndarray = None) -> Tuple[float, float]:
        """Find K nearest historical feature vectors; return their forward-
        return distribution stats. Non-parametric "this looks like X" alpha.

        `weights` (optional, aligned to the feature columns, mean≈1) scales each
        z-scored dimension in the distance so demonstrably-predictive features
        dominate the "this looks like X" match and proven-noise features barely
        count. None ⇒ unweighted Euclidean (legacy)."""
        if (hist_feats is None or hist_returns is None
                or len(hist_feats) < k * 3 or len(hist_returns) != len(hist_feats)):
            return 0.0, 0.0
        try:
            cur = current_feat.reshape(1, -1)
            # Z-normalize feature columns for fair distance
            mu = hist_feats.mean(axis=0); sd = hist_feats.std(axis=0) + 1e-9
            hf = (hist_feats - mu) / sd
            cf = (cur - mu) / sd
            sq = (hf - cf) ** 2
            if weights is not None and len(weights) == sq.shape[1]:
                sq = sq * np.asarray(weights, dtype=float)
            d  = np.sqrt(sq.sum(axis=1))
            idx = np.argpartition(d, k)[:k]
            fwd = hist_returns[idx]
            m, s = float(fwd.mean()), float(fwd.std())
            self.last_doppel_mean, self.last_doppel_std = m, s
            return m, s
        except Exception:
            return 0.0, 0.0

    # ========================================================
    # ROUND-3 ELITE METHODS (Renaissance/Thorp/Drift-native)
    # ========================================================

    # ---- Cornish-Fisher Expected Shortfall (fat-tail CVaR) ----
    def cornish_fisher_cvar(self, alpha: float = 0.95) -> float:
        """Skewness + excess-kurtosis adjusted ES. Crypto returns have
        fat left tails (skew < 0, kurt > 3) — Gaussian CVaR systematically
        UNDER-estimates ES by 25-40%. CF expansion fixes this analytically.
        Memoized — recomputes only when the returns deque grows."""
        rev = len(self.returns)
        if rev == self._cf_cvar_rev and self._cf_cvar_cached:
            return self._cf_cvar_cached
        if rev < 50:
            self._cf_cvar_cached = 2.062 * self.garch_vol()
            self._cf_cvar_rev = rev
            return self._cf_cvar_cached
        r = np.array(list(self.returns))
        mu, sd = float(r.mean()), float(r.std(ddof=1))
        if sd <= 1e-12:
            self.last_cf_cvar = 0.0
            return 0.0
        try:
            from scipy.stats import norm
            z_a = float(norm.ppf(1 - alpha))            # negative
            sk  = float(((r - mu)**3).mean() / sd**3)
            kt  = float(((r - mu)**4).mean() / sd**4) - 3.0
            # Cornish-Fisher quantile expansion
            z_cf = (z_a
                    + (z_a**2 - 1) * sk / 6.0
                    + (z_a**3 - 3*z_a) * kt / 24.0
                    - (2*z_a**3 - 5*z_a) * sk**2 / 36.0)
            # ES approximation w/ CF quantile
            phi  = math.exp(-0.5 * z_cf * z_cf) / math.sqrt(2 * math.pi)
            es   = -mu + sd * phi / max(1 - alpha, 1e-6)
            self.last_cf_cvar = float(abs(es))
        except Exception:
            self.last_cf_cvar = 2.062 * sd
        self._cf_cvar_cached = self.last_cf_cvar
        self._cf_cvar_rev = rev
        return self.last_cf_cvar

    # ---- Funding-rate z-score (crowded carry reversal) ----
    def update_funding(self, funding_rate: float):
        if np.isfinite(funding_rate):
            self.funding_history.append(float(funding_rate))

    def funding_zscore(self) -> float:
        if len(self.funding_history) < 20:
            self.last_funding_z = 0.0
            return 0.0
        f = np.array(self.funding_history, dtype=float)
        sd = f.std(ddof=1)
        if sd <= 1e-12:
            self.last_funding_z = 0.0
            return 0.0
        z = float((f[-1] - f.mean()) / sd)
        if not np.isfinite(z): z = 0.0
        self.last_funding_z = z
        return z

    # ---- Oracle-Mark Spread (DRIFT-NATIVE EDGE) ----
    def oracle_mark_spread(self, oracle_price: float, mark_price: float) -> float:
        """Drift settles trades at the oracle price; the vAMM rebalances back
        toward oracle when mark deviates. |spread| > 0.30% is statistical-arb
        territory unique to this venue."""
        if oracle_price <= 0 or mark_price <= 0:
            return 0.0
        spread = (mark_price - oracle_price) / oracle_price
        if not np.isfinite(spread): spread = 0.0
        self.last_oracle_mark_spread = float(spread)
        if abs(spread) > 0.003:
            self.oracle_arb_signals += 1
        return self.last_oracle_mark_spread

    # ---- CUSUM Event Filter (López de Prado) ----
    def cusum_event(self, threshold: float = None, window: int = 50) -> bool:
        """Cumulative-sum filter — flag bars where price has moved enough
        to warrant a fresh independent observation. Avoids overtrading the
        same micro-structure feature repeatedly."""
        if len(self.returns) < 5:
            self.last_cusum_event = False
            return False
        thr = threshold if threshold is not None else 2.5 * self.garch_vol()
        r = np.array(list(self.returns)[-window:])
        s_pos = 0.0; s_neg = 0.0
        for ret in r:
            s_pos = max(0.0, s_pos + ret)
            s_neg = min(0.0, s_neg + ret)
            if s_pos > thr or s_neg < -thr:
                self.last_cusum_event = True
                return True
        self.last_cusum_event = False
        return False

    # ---- Pessimistic Kelly (bootstrap CI lower bound) ----
    def pessimistic_kelly(self, n_bootstrap: int = 200, alpha_floor: float = 0.05) -> float:
        """Bootstrap a CI on win-rate, plug the LOWER bound into Kelly.
        Thorp's "underbet" rule: regret-free under sampling error.
        Cold-start (< 20 closed trades) returns a conservative 5% Kelly.
        Memoized — recomputes only when trade_pnls grows (after a close)."""
        rev = len(self.trade_pnls)
        if rev == self._pess_kelly_rev:
            return self._pess_kelly_cached
        if rev < 20:
            self.last_pess_kelly = 0.05
            self._pess_kelly_cached = 0.05
            self._pess_kelly_rev = rev
            return 0.05
        pnls = np.array(list(self.trade_pnls), dtype=float)
        wins_mask = pnls > 0
        n = len(pnls)
        try:
            rng = np.random.default_rng(42)
            wr_boots = np.array([
                wins_mask[rng.integers(0, n, n)].mean() for _ in range(n_bootstrap)
            ])
            p_lo = float(np.percentile(wr_boots, alpha_floor * 100))
            avg_w = pnls[wins_mask].mean() if wins_mask.any() else 0.01
            avg_l = abs(pnls[~wins_mask].mean()) if (~wins_mask).any() else 0.01
            b = max(avg_w / max(avg_l, 1e-6), 1e-6)
            kelly = (b * p_lo - (1 - p_lo)) / b
            half = float(np.clip(kelly * 0.5, 0.02, 0.25))
            self.last_pess_kelly = half
            self._pess_kelly_cached = half
            self._pess_kelly_rev = rev
            return half
        except Exception:
            self.last_pess_kelly = 0.10
            self._pess_kelly_cached = 0.10
            self._pess_kelly_rev = rev
            return 0.10

    # ---- Dynamic Half-Kelly (Grok formula on EMPIRICAL p/b — honest A/B arm) ----
    def dynamic_half_kelly(self, confidence: float, atr: float = None,
                           atr_avg: float = None, velocity: float = None) -> float:
        """Grok's Dynamic Half-Kelly, but fed the EMPIRICAL win-rate/odds (NOT a
        fabricated 0.998 confidence) and HARD-CAPPED at the engine's 25% per-trade
        ceiling — Grok's min(0.75, f) is ruinous at leverage. This is the honest
        A/B counterpart to pessimistic_kelly(): it sizes off the bootstrap POINT
        estimate of win-rate (not the 5th-percentile lower bound), so it bets
        bigger when the realised edge is genuine — never recklessly.

            f = (b*p - q)/b ; f_half = f/2 ; bounded regime nudge ; clip[0.02,0.25]
        """
        n = len(self.trade_pnls)
        if n < 20:
            return 0.05    # same cold-start floor as pessimistic_kelly
        try:
            pnls = np.array(list(self.trade_pnls), dtype=float)
            wins = pnls > 0
            p_emp = float(wins.mean())
            # Blend a little live-signal confidence in (empirical base dominates),
            # then cap p well below 1.0 so the formula can never approach all-in.
            p = float(np.clip(0.75 * p_emp + 0.25 * float(confidence), 0.0, 0.90))
            q = 1.0 - p
            avg_w = pnls[wins].mean() if wins.any() else 0.01
            avg_l = abs(pnls[~wins].mean()) if (~wins).any() else 0.01
            b = max(avg_w / max(avg_l, 1e-6), 1e-6)
            f = (b * p - q) / b
            f_half = f * 0.5
            # Grok's ATR/velocity nudge — STRUCTURE preserved, thresholds made
            # realistic (his p>0.9985 gate never fires on real win-rates) and the
            # uplift bounded so a nudge can add at most ~15%.
            if atr is not None and atr_avg and atr_avg > 0:
                if atr < 1.5 * atr_avg and p > 0.55:
                    f_half *= (1.0 + min(0.10, 0.10 * (p - 0.55) / 0.45))
                elif atr > 2.0 * atr_avg and velocity is not None and abs(velocity) > 0.005:
                    f_half *= (1.0 + min(0.15, 20.0 * (abs(velocity) - 0.005)))
            out = float(np.clip(f_half, 0.02, 0.25))
            self.last_dyn_kelly = out
            return out
        except Exception:
            return 0.05

    # ---- Blended Dynamic-Pessimistic Kelly (pess base × bounded regime mult) ----
    def dynamic_pessimistic_kelly(self, confidence: float, regime: str = None) -> float:
        """The synergistic blend: keep the pessimistic-Kelly bootstrap base (the
        engine's safe, well-founded default) and modulate it by a BOUNDED,
        TWO-SIDED regime multiplier. Unlike Grok's Dynamic Half-Kelly (which only
        scales UP), this also scales DOWN in choppy / high-vol / high-uncertainty
        regimes. Preserves the floor, the 25% cap, and (downstream) the CVaR guard.

            dyn  = clip(1 + a*(conf-conf_mid) + b*(hurst-0.5)
                          - c*(ewma_vol/target - 1) + regime_tilt, 0.5, 1.5)
            size = clip(pess_k * dyn, 0.02, 0.25)
        """
        base = self.pessimistic_kelly()
        try:
            conf_mid   = 0.55
            target_vol = 0.02
            hurst = float(getattr(self, 'last_hurst', 0.5))
            ewma  = float(getattr(self, 'last_ewma_vol', 0.02))
            a, b_coef, c = 1.5, 0.8, 0.5
            dyn = (1.0
                   + a * (float(confidence) - conf_mid)
                   + b_coef * (hurst - 0.5)
                   - c * ((ewma / max(target_vol, 1e-6)) - 1.0))
            reg_adj = {'trending': 0.10, 'choppy': -0.10,
                       'mean_reverting': 0.0}.get(regime, 0.0)
            dyn = float(np.clip(dyn + reg_adj, 0.5, 1.5))
            out = float(np.clip(base * dyn, 0.02, 0.25))
            self.last_dyn_kelly = out
            return out
        except Exception:
            return base

    # ---- OFI Lag-1 Autocorrelation (microstructure persistence) ----
    def ofi_persistence(self, ofi_series: np.ndarray) -> float:
        """ρ(OFI_t, OFI_{t-1}) — positive => momentum regime,
        negative => mean-reversion regime."""
        if ofi_series is None or len(ofi_series) < 20:
            return 0.0
        o = np.asarray(ofi_series, dtype=float)
        o = o[np.isfinite(o)][-50:]
        if len(o) < 10 or o.std() == 0:
            return 0.0
        try:
            rho = float(np.corrcoef(o[:-1], o[1:])[0, 1])
            if not np.isfinite(rho): rho = 0.0
            self.last_ofi_persistence = rho
            return rho
        except Exception:
            return 0.0

    # ---- Multi-Timeframe Confluence ----
    @staticmethod
    def multi_timeframe_alignment(df_1m: pd.DataFrame) -> Tuple[float, Dict[str, str]]:
        """Trend direction on 1m / 5m / 15m. Returns (alignment_score, by_tf).
        alignment_score in [-1,+1]: +1 = all long, -1 = all short."""
        out = {'1m': 'neutral', '5m': 'neutral', '15m': 'neutral'}
        if df_1m is None or len(df_1m) < 60:
            return 0.0, out
        try:
            c = df_1m['close']
            ma20 = c.rolling(20).mean()
            out['1m']  = 'long' if c.iloc[-1] > ma20.iloc[-1] else 'short'
            for rule, key in (('5min','5m'), ('15min','15m')):
                rs = c.resample(rule).last().dropna()
                if len(rs) < 20: continue
                ma = rs.rolling(20).mean().iloc[-1]
                out[key] = 'long' if rs.iloc[-1] > ma else 'short'
            score = sum(1 if v == 'long' else (-1 if v == 'short' else 0) for v in out.values()) / 3.0
            return float(score), out
        except Exception:
            return 0.0, out

    # ---- Meta-labeling: (regime, action) win-rate bucket ----
    def metalabel_winrate(self, regime: str, action: str) -> float:
        wl = self.metalabel_buckets.get((regime, action))
        if not wl or sum(wl) < CONFIG.get('METALABEL_MIN_SAMPLES', 5):
            return 0.5  # neutral prior
        return wl[0] / max(sum(wl), 1)

    def update_metalabel(self, regime: str, action: str, won: bool):
        key = (regime, action)
        wl = self.metalabel_buckets.setdefault(key, [0, 0])
        if won: wl[0] += 1
        else:   wl[1] += 1
        # Cache last looked-up winrate for snapshot
        self.last_metalabel_wr = self.metalabel_winrate(regime, action)

    def decay_metalabels(self, factor: float):
        """Exponentially fade (regime,action) win/loss evidence once per retrain so
        the gate adapts to non-stationarity / a changed model (e.g. the OOF de-leak):
        the pre-fix model's polluted buckets shrink below MIN_SAMPLES and revert to
        the neutral 0.5 prior, letting the CURRENT model re-prove a setup. factor>=1.0
        disables. Win-rate RATIO is preserved (only the sample count decays)."""
        if factor >= 1.0:
            return
        for k, wl in list(self.metalabel_buckets.items()):
            self.metalabel_buckets[k] = [wl[0] * factor, wl[1] * factor]

    def snapshot(self) -> Dict:
        return {
            'posterior_last':    round(self.last_posterior, 4),
            'log_odds_last':     round(self.last_log_odds, 4),
            'bayes_passes':      self.bayes_passes,
            'bayes_blocks':      self.bayes_blocks,
            'garch_vol':         round(self.last_garch_vol, 6),
            'deflated_sharpe':   round(self.last_deflated_sr, 4),
            'cointeg_z':         round(self.last_cointeg_z, 4),
            'peak_balance':      round(self.peak_balance, 4),
            'max_drawdown':      round(self.max_drawdown, 4),
            'returns_window':    len(self.returns),
            'pnls_window':       len(self.trade_pnls),
            # Round-2 elite metrics
            'bocpd_p':           round(self.last_bocpd_p, 4),
            'bocpd_halts':       self.bocpd_halts,
            'crowded_score':     round(self.last_crowded_score, 4),
            'reflexivity':       round(self.last_reflexivity, 4),
            'vol_target_lev':    self.last_vol_target_lev,
            'info_coefficient':  round(self.last_ic, 4),
            'doppel_fwd_mean':   round(self.last_doppel_mean, 6),
            'doppel_fwd_std':    round(self.last_doppel_std, 6),
            'metalabel_winrate': round(self.last_metalabel_wr, 4),
            'metalabel_buckets': {f"{k[0]}|{k[1]}": v for k,v in self.metalabel_buckets.items()},
            # Round-3 elite metrics
            'cf_cvar':            round(self.last_cf_cvar, 6),
            'funding_z':          round(self.last_funding_z, 4),
            'oracle_mark_spread': round(self.last_oracle_mark_spread, 6),
            'cusum_event':        bool(self.last_cusum_event),
            'cusum_blocks':       self.cusum_blocks,
            'pess_kelly':         round(self.last_pess_kelly, 4),
            'sizing_mode':        CONFIG.get('SIZING_MODE', 'pessimistic'),
            'dyn_kelly':          round(self.last_dyn_kelly, 4),
            'ofi_persistence':    round(self.last_ofi_persistence, 4),
            'mtf_alignment':      round(self.last_mtf_alignment, 4),
            'oracle_arb_signals': self.oracle_arb_signals,
        }

quant_overlay = QuantOverlay()

# ============================================================
# SECTION 4: DEEP LEARNING MODELS
# ============================================================
class LSTMPriceModel(nn.Module):
    def __init__(self, input_size=64, hidden_size=128, num_layers=3, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            dropout=dropout, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_size * 2, num_heads=8, batch_first=True)
        self.fc1       = nn.Linear(hidden_size * 2, 64)
        self.fc2       = nn.Linear(64, 3)
        self.dropout   = nn.Dropout(dropout)
        self.norm      = nn.LayerNorm(hidden_size * 2)

    def forward(self, x):
        out, _   = self.lstm(x)
        attn, _  = self.attention(out, out, out)
        h        = self.norm(attn[:, -1, :])
        return torch.softmax(self.fc2(torch.relu(self.fc1(self.dropout(h)))), dim=-1)


class TransformerSignalModel(nn.Module):
    def __init__(self, input_size=64, d_model=128, nhead=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(input_size, d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, 512, dropout, batch_first=True)
        self.tf   = nn.TransformerEncoder(enc, num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Linear(d_model, 3)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.proj(x)
        x = self.tf(x).transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        return torch.softmax(self.fc(self.drop(x)), dim=-1)


class RegimeClassifierNet(nn.Module):
    def __init__(self, input_size=32, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden), nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 3)
        )
    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)


class BayesianUncertaintyHead(nn.Module):
    def __init__(self, input_size=3, mc_samples=20):
        super().__init__()
        self.mc_samples = mc_samples
        self.dropout    = nn.Dropout(p=0.3)
        self.fc         = nn.Linear(input_size, 3)

    def forward(self, x, training=True):
        n = self.mc_samples if training else 1
        samples = torch.stack([torch.softmax(self.fc(self.dropout(x)), dim=-1) for _ in range(n)])
        return samples.mean(0), samples.std(0).mean(-1)

# ============================================================
# SECTION 5: RL STOP/TP OPTIMIZER (Double Q-Learning)
# ============================================================
class RLStopTPOptimizer:
    def __init__(self):
        self.q1: Dict[tuple, np.ndarray] = {}
        self.q2: Dict[tuple, np.ndarray] = {}
        self.target_q1: Dict[tuple, np.ndarray] = {}
        self.target_q2: Dict[tuple, np.ndarray] = {}
        self.replay_buffer = deque(maxlen=CONFIG['RL_BUFFER_SIZE'])
        self.regime_buffers = {k: deque(maxlen=2000) for k in ('trending','choppy','mean_reverting')}
        self.tau          = CONFIG['RL_TAU']
        self.lr           = CONFIG['RL_LEARNING_RATE']
        self.gamma        = CONFIG['RL_GAMMA']
        self.n_actions    = 10
        self.epsilon      = 0.15
        self.epsilon_min  = 0.01
        self.epsilon_decay= 0.995
        self.update_count = 0

    def _init(self, s):
        if s not in self.q1:
            for d in (self.q1, self.q2, self.target_q1, self.target_q2):
                d[s] = np.zeros(self.n_actions)

    def _disc(self, hurst, vol, regime, atr_pct, momentum):
        r_map = {'trending': 0, 'choppy': 1, 'mean_reverting': 2, 'unknown': 1}
        return (min(int(hurst*5),4), min(int(vol/0.01),4),
                min(int((momentum+1)*2.5),4), min(int(atr_pct/0.005),4),
                r_map.get(regime,1))

    def get_sl_tp_multipliers(self, hurst, vol, regime, atr_pct, momentum):
        s = self._disc(hurst, vol, regime, atr_pct, momentum)
        self._init(s)
        if np.random.random() < self.epsilon:
            a = np.random.randint(self.n_actions)
        else:
            a = int(np.argmax((self.q1[s]+self.q2[s])/2))
        return CONFIG['SL_ATR_MULT'][a], CONFIG['TP_ATR_MULT'][a], s, a

    def update(self, s, a, r, ns, regime, done):
        self._init(s); self._init(ns)
        exp = (s, a, r, ns, regime, done)
        self.replay_buffer.append(exp)
        self.regime_buffers.get(regime, self.regime_buffers['choppy']).append(exp)
        if len(self.replay_buffer) < CONFIG['RL_BATCH_SIZE']:
            return
        buf = list(self.replay_buffer)
        pri = np.array([abs(e[2])+1e-5 for e in buf])
        idx = np.random.choice(len(buf), CONFIG['RL_BATCH_SIZE'], p=pri/pri.sum())
        for i in idx:
            _s,_a,_r,_ns,_reg,_d = buf[i]
            self._init(_s); self._init(_ns)
            q1t = _r if _d else _r + self.gamma * self.q2[_ns][np.argmax(self.q1[_ns])]
            q2t = _r if _d else _r + self.gamma * self.q1[_ns][np.argmax(self.q2[_ns])]
            self.q1[_s][_a] += self.lr*(q1t-self.q1[_s][_a])
            self.q2[_s][_a] += self.lr*(q2t-self.q2[_s][_a])
        for k in list(self.q1):
            tq1 = self.target_q1.get(k, self.q1[k].copy())
            tq2 = self.target_q2.get(k, self.q2[k].copy())
            self.target_q1[k] = self.tau*self.q1[k] + (1-self.tau)*tq1
            self.target_q2[k] = self.tau*self.q2[k] + (1-self.tau)*tq2
        self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)
        self.update_count += 1
        state.benchmarks['rl_update_count'] = self.update_count
        if self.update_count % CONFIG['RL_RETRAIN_INTERVAL'] == 0:
            self._offline_retrain()

    def _offline_retrain(self):
        for regime, buf in self.regime_buffers.items():
            if len(buf) < 20: continue
            for s,a,r,ns,reg,d in list(buf)[-200:]:
                self._init(s); self._init(ns)
                tgt = r if d else r + self.gamma * min(
                    self.q2[ns][np.argmax(self.q1[ns])],
                    self.q1[ns][np.argmax(self.q2[ns])])
                self.q1[s][a] += self.lr*0.5*(tgt-self.q1[s][a])
                self.q2[s][a] += self.lr*0.5*(tgt-self.q2[s][a])

    def save(self, path='rl_optimizer.pkl'):
        with open(path,'wb') as f:
            pickle.dump({'q1':self.q1,'q2':self.q2,'target_q1':self.target_q1,
                         'target_q2':self.target_q2,'epsilon':self.epsilon}, f)

    def load(self, path='rl_optimizer.pkl'):
        if os.path.exists(path):
            with open(path,'rb') as f:
                d = pickle.load(f)
            self.q1=d['q1']; self.q2=d['q2']
            self.target_q1=d['target_q1']; self.target_q2=d['target_q2']
            self.epsilon=d['epsilon']
            logger.info(f"RL optimizer loaded from {path}")

# ============================================================
# SECTION 6: RL REGIME CLASSIFIER (Online Fine-Tuning)
# ============================================================
class RLRegimeClassifier:
    def __init__(self):
        self.model   = RegimeClassifierNet(input_size=32, hidden=64)
        self.opt     = optim.Adam(self.model.parameters(), lr=0.0003)
        self.loss_fn = nn.CrossEntropyLoss()
        self.labels  = ['trending','choppy','mean_reverting']
        self.scaler  = RobustScaler()
        self.fitted  = False

    def predict(self, features: np.ndarray) -> str:
        if not self.fitted: return 'choppy'
        try:
            x = torch.tensor(self.scaler.transform(features.reshape(1,-1)), dtype=torch.float32)
            with torch.no_grad():
                return self.labels[int(self.model(x).argmax())]
        except Exception:
            return 'choppy'

    def online_update(self, features: np.ndarray, reward: float):
        if not self.fitted: return
        try:
            x = torch.tensor(self.scaler.transform(features.reshape(1,-1)), dtype=torch.float32)
            probs = self.model(x)
            idx   = int(probs.argmax())
            if reward > 0:
                loss = self.loss_fn(probs, torch.tensor([idx]))
                self.opt.zero_grad(); loss.backward(); self.opt.step()
        except Exception:
            pass

    def fit(self, X: np.ndarray, y: np.ndarray):
        if len(X) < 10: return
        self.scaler.fit(X)
        self.fitted = True
        Xt = torch.tensor(self.scaler.transform(X), dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.long)
        for _ in range(20):
            self.opt.zero_grad()
            loss = self.loss_fn(self.model(Xt), yt)
            loss.backward(); self.opt.step()

rl_stop_optimizer    = RLStopTPOptimizer()
rl_regime_classifier = RLRegimeClassifier()

# ============================================================
# SECTION 7: HMM + GMM REGIME DETECTOR (4-state)
# ============================================================
class HMMRegimeDetector:
    """4-state HMM+GMM ensemble (Trending-Bull, Trending-Bear,
       High-Vol-MeanReversion, Accumulation-LowVol)"""
    def __init__(self):
        self.hmm = GaussianHMM(n_components=4, covariance_type='full',
                               n_iter=1200, tol=1e-4, init_params='kmeans',
                               random_state=42)
        self.gmm    = GaussianMixture(n_components=4, covariance_type='full', random_state=42)
        self.scaler = StandardScaler()
        self.fitted = False
        self.states = ['Trending-Bull','Trending-Bear','High-Vol-MeanReversion','Accumulation-LowVol']

    def fit(self, features: np.ndarray):
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        self.n_features = features.shape[1]
        X = self.scaler.fit_transform(features)
        self.hmm.n_components = min(4, max(2, len(X) // 50))
        try:
            self.hmm.fit(X); self.gmm.fit(X)
            self.fitted = True
        except Exception as e:
            logger.debug(f"HMM fit error: {e}")

    def predict(self, features: np.ndarray) -> str:
        if not self.fitted:
            return self.states[2]
        try:
            f = np.array(features)
            if hasattr(self, 'n_features'):
                if f.ndim == 1 and len(f) != self.n_features:
                    f = f[:self.n_features] if len(f) > self.n_features else np.pad(f, (0, self.n_features-len(f)))
            X = self.scaler.transform(f.reshape(1, -1))
            return self.states[max(int(self.hmm.predict(X)[0]),
                                   int(self.gmm.predict(X)[0])) % len(self.states)]
        except Exception:
            return self.states[2]

hmm_detector = HMMRegimeDetector()

# ============================================================
# SECTION 8: PPO MDP ENVIRONMENT (optional, requires stable-baselines3)
# ============================================================
if HAS_RL:
    class DelphiTradingEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
            self.action_space      = spaces.Discrete(3)
            self.current_step      = 0

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.current_step=0
            return np.zeros(12, dtype=np.float32), {}
        def step(self, a):
            self.current_step += 1
            obs = np.zeros(12, dtype=np.float32)
            terminated = False; truncated = self.current_step >= 256
            return obs, 0.0, terminated, truncated, {}

    ppo_env   = DelphiTradingEnv()
    ppo_agent = PPO("MlpPolicy", ppo_env, verbose=0, learning_rate=3e-4, clip_range=0.2)

# ============================================================
# SECTION 9: KALMAN FILTER
# ============================================================
class KalmanPriceFilter:
    def __init__(self, q=1e-5, r=1e-2):
        self.q=q; self.r=r; self.p=1.0; self.x=0.0; self.initialized=False

    def update(self, m):
        if not self.initialized: self.x=m; self.initialized=True; return self.x
        self.p += self.q
        k = self.p/(self.p+self.r)
        self.x = self.x + k*(m-self.x)
        self.p = (1-k)*self.p
        return self.x

    def smooth_series(self, prices: np.ndarray) -> np.ndarray:
        s=np.zeros_like(prices); self.p=1.0; self.x=prices[0]; self.initialized=False
        for i,p in enumerate(prices): s[i]=self.update(p)
        return s

kalman_filter = KalmanPriceFilter()

# ============================================================
# SECTION 10: SIGNAL FILTERS (Super Smoother, MESA, Hurst, VPIN, EMD)
# ============================================================
def super_smoother(prices: np.ndarray, period: int = 10) -> np.ndarray:
    a1=math.exp(-1.414*math.pi/period); b1=2*a1*math.cos(1.414*math.pi/period)
    c2=b1; c3=-a1*a1; c1=1-c2-c3
    ss=np.zeros_like(prices)
    for i in range(2, len(prices)):
        ss[i]=c1*(prices[i]+prices[i-1])/2+c2*ss[i-1]+c3*ss[i-2]
    return ss

def mesa_adaptive_ma(prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mama=np.zeros_like(prices); fama=np.zeros_like(prices)
    smooth=np.zeros_like(prices); detrender=np.zeros_like(prices)
    period=np.zeros_like(prices); Q1=np.zeros_like(prices); I1=np.zeros_like(prices)
    jI=np.zeros_like(prices); jQ=np.zeros_like(prices)
    Re=np.zeros_like(prices); Im=np.zeros_like(prices)
    for i in range(6, len(prices)):
        smooth[i]=(4*prices[i]+3*prices[i-1]+2*prices[i-2]+prices[i-3])/10
        det=(0.0962*smooth[i]+0.5769*smooth[i-2]-0.5769*smooth[i-4]-0.0962*smooth[i-6])*(0.075*period[i-1]+0.54)
        detrender[i]=det
        q1=(0.0962*det+0.5769*(detrender[i-2] if i>7 else 0)-0.5769*(detrender[i-4] if i>9 else 0)-0.0962*(detrender[i-6] if i>11 else 0))*(0.075*period[i-1]+0.54)
        Q1[i]=q1; I1[i]=detrender[i-3] if i>8 else 0
        jI[i]=(0.0962*I1[i]+0.5769*(I1[i-2] if i>7 else 0)-0.5769*(I1[i-4] if i>9 else 0)-0.0962*(I1[i-6] if i>11 else 0))*(0.075*period[i-1]+0.54)
        jQ[i]=(0.0962*Q1[i]+0.5769*(Q1[i-2] if i>7 else 0)-0.5769*(Q1[i-4] if i>9 else 0)-0.0962*(Q1[i-6] if i>11 else 0))*(0.075*period[i-1]+0.54)
        I2=I1[i]-jQ[i]; Q2=Q1[i]+jI[i]
        Re[i]=0.2*(I2*I2)+0.8*Re[i-1]; Im[i]=0.2*(Q2*Q2)+0.8*Im[i-1]
        if Im[i]!=0 and Re[i]!=0:
            period[i]=2*math.pi/math.atan(Im[i]/Re[i])
        period[i]=max(1,min(period[i],50))
        period[i]=0.2*period[i]+0.8*period[i-1]
        alpha=max(0.05, min(0.5, 0.5/(max(1,period[i-1]-period[i])+1e-9)))
        mama[i]=alpha*prices[i]+(1-alpha)*mama[i-1]
        fama[i]=0.5*alpha*mama[i]+(1-0.5*alpha)*fama[i-1]
    return mama, fama

def hurst_exponent(ts: np.ndarray, max_lag: int = 20) -> float:
    if len(ts) < max_lag*2: return 0.5
    lags=range(2, max_lag); tau=[]
    for lag in lags:
        chunks=[ts[i:i+lag] for i in range(0, len(ts)-lag, lag)]
        rs=[]
        for c in chunks:
            if len(c)<2: continue
            d=np.cumsum(c-np.mean(c)); S=np.std(c)
            if S>0: rs.append((d.max()-d.min())/S)
        if rs: tau.append(np.mean(rs))
    if len(tau)<2: return 0.5
    try:
        return float(np.clip(np.polyfit(np.log(list(lags[:len(tau)])),np.log(tau),1)[0],0.01,0.99))
    except Exception:
        return 0.5

def compute_vpin(df: pd.DataFrame, bucket_size: int = 50) -> float:
    if len(df)<bucket_size*2: return 0.5
    try:
        bv=df['volume']*(df['close']>df['open']).astype(float)
        sv=df['volume']*(df['close']<=df['open']).astype(float)
        bb=[bv.iloc[i:i+bucket_size].sum() for i in range(0,len(df)-bucket_size,bucket_size)]
        bs=[sv.iloc[i:i+bucket_size].sum() for i in range(0,len(df)-bucket_size,bucket_size)]
        return float(np.clip(np.mean([abs(b-s)/(b+s+1e-10) for b,s in zip(bb,bs)]),0,1))
    except Exception:
        return 0.5

def empirical_mode_decomposition(signal: np.ndarray, num_imfs: int = 4) -> List[np.ndarray]:
    imfs=[]; residue=signal.copy()
    for _ in range(num_imfs):
        h=residue.copy()
        for _ in range(10):
            mx=[j for j in range(1,len(h)-1) if h[j]>h[j-1] and h[j]>h[j+1]]
            mn=[j for j in range(1,len(h)-1) if h[j]<h[j-1] and h[j]<h[j+1]]
            if len(mx)<2 or len(mn)<2: break
            from scipy.interpolate import CubicSpline
            x=np.arange(len(h))
            try:
                upper=CubicSpline(mx,h[mx])(x)
                lower=CubicSpline(mn,h[mn])(x)
            except Exception:
                break
            h=h-(upper+lower)/2
        imfs.append(h); residue=residue-h
        if np.std(residue)<1e-10: break
    return imfs

def granger_causality_test(y: np.ndarray, x: np.ndarray, max_lag: int = 5) -> float:
    if len(y)<max_lag*3 or len(x)<max_lag*3: return 0.5
    try:
        n=min(len(y),len(x))-max_lag
        Xr=np.column_stack([y[i:n+i] for i in range(max_lag)])
        yt=y[max_lag:n+max_lag]
        rss_r=np.sum((yt-LinearRegression().fit(Xr,yt).predict(Xr))**2)
        Xu=np.column_stack([y[i:n+i] for i in range(max_lag)]+[x[i:n+i] for i in range(max_lag)])
        rss_u=np.sum((yt-LinearRegression().fit(Xu,yt).predict(Xu))**2)
        if rss_u<1e-10: return 1.0
        f=((rss_r-rss_u)/max_lag)/(rss_u/(n-2*max_lag))
        return float(np.clip(f/(f+10),0,1))
    except Exception:
        return 0.5

def historic_match_score(current: np.ndarray, history: deque, threshold: float = 0.88,
                         weights: np.ndarray = None) -> float:
    """DTW/cosine/Wasserstein similarity of `current` to recent historic states.

    `weights` (optional, aligned to the feature vector, mean≈1) emphasises
    demonstrably-predictive features via √w component scaling (weighted cosine =
    plain cosine on the √w-scaled vectors). A dimension guard SKIPS any stored
    state whose length differs from `current` — this also closes a latent bug:
    historic states used to be stored as a 4-dim [returns,adx,hurst,vpin] vector
    while the query passes the full ~58-dim feature row, so every cosine_dist
    raised on the length mismatch and this whole path silently returned 0.0."""
    if not HAS_DTW or len(history) == 0: return 0.0
    cur = np.asarray(current, dtype=float)
    sw = None
    if weights is not None and len(weights) == len(cur):
        sw = np.sqrt(np.clip(np.asarray(weights, dtype=float), 0.0, None))
    scores = []
    for h in list(history)[-50:]:
        try:
            h = np.asarray(h, dtype=float)
            if h.shape != cur.shape:        # dimension guard (fixes silent-dead path)
                continue
            a, b = (cur, h) if sw is None else (cur * sw, h * sw)
            dtw_d, _  = fastdtw(a, b)
            cos_s     = 1 - cosine_dist(a, b)
            w_d       = wasserstein_distance(a, b)
            scores.append(cos_s*0.6 + 1/(1+dtw_d+w_d)*0.4)
        except Exception:
            continue
    return max(scores) if scores else 0.0

# ============================================================
# SECTION 11: (removed) DRIFT PROTOCOL CLIENT
# ------------------------------------------------------------
# The DriftTrader class was removed in v4.00. It was never instantiated after
# the migration to exchange_adapter.ExchangeRouter and pulled in the heavy
# anchorpy/solders/solana/driftpy stack solely as dead weight. All execution
# now flows through the router (Aster -> ApeX -> Paper); see SECTION 11b.

# ============================================================
# SECTION 11b: EXCHANGE ROUTER (v4.00 — replaces DriftTrader execution)
# ------------------------------------------------------------
# All execution flows through exchange_adapter.ExchangeRouter (Aster primary →
# ApeX failover → Paper). The router is a drop-in: it exposes the same method
# names the engine already calls (get_collateral / place_perp_order /
# close_position / get_open_positions / round_trip_fee_rate / is_paper) PLUS the
# richer venue-native interface (get_orderbook / get_open_interest /
# get_long_short_ratio / get_liquidations / cross_venue_basis / health).
#
# Engine boundary (unconditional): the router and its adapters are STANDARD
# HTTPS API clients only — no VPN / proxy / IP-masking / geo-evasion code. Had
# such a layer ever been coded it would have lived in the adapters' aiohttp
# ClientSession setup in exchange_adapter.py (AsterAdapter._session /
# ApexOmniAdapter._session). None exists in either; none is added.
from exchange_adapter import make_router

router = make_router(
    symbol_key=CONFIG.get('SYMBOL', 'SOL'),
    mode=CONFIG.get('ENGINE_MODE', 'paper'),
    starting_equity=CONFIG['STARTING_BALANCE'],
    promo=CONFIG.get('APEX_PROMO_MAKER', True),
)
CONFIG['ENGINE_MODE'] = router.mode   # router is authoritative (forced paper without DELPHI_LIVE_ACK)
# Order venue priority follows PRIMARY_VENUE (default aster → apex).
if CONFIG.get('PRIMARY_VENUE') == 'apex':
    router.venues = [router.apex, router.aster]
# Legacy alias: every existing `drift_trader.*` call site resolves to the
# router's drop-in API. New code should prefer `router` directly.
drift_trader = router

# ============================================================
# SECTION 12: DATA FETCHING
# ============================================================
async def _fetch_ohlcv_kraken(limit: int = 500) -> pd.DataFrame:
    t0 = time.time()
    try:
        url = "https://api.kraken.com/0/public/OHLC"
        params = {'pair': 'SOLUSD', 'interval': 1}
        async with aiohttp.ClientSession() as s:
            async with s.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as r:
                data = await r.json()
                if data.get('error'):
                    record_api('kraken', False, (time.time()-t0)*1000, str(data['error']))
                    return pd.DataFrame()
                rows = list(data['result'].values())[0]
                df = pd.DataFrame(rows, columns=['timestamp','open','high','low','close','vwap','volume','count'])
                df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='s', utc=True)
                df = df.set_index('timestamp').sort_index()
                for c in ['open','high','low','close','volume']:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
                record_api('kraken', True, (time.time()-t0)*1000)
                return df[['open','high','low','close','volume']].dropna().tail(limit)
    except Exception as e:
        record_api('kraken', False, (time.time()-t0)*1000, str(e))
        return pd.DataFrame()

_birdeye_cooldown_until = 0.0   # quota circuit-breaker: skip Birdeye after a usage-limit error

async def _fetch_ohlcv_aster(limit: int = 500) -> pd.DataFrame:
    """Venue-native 1m OHLCV from Aster klines (free, no quota) — first OHLCV fallback.
    Trading happens ON Aster, so these candles are the truest mark anyway."""
    try:
        kl = await router.klines('1m', int(limit))
        if not kl:
            return pd.DataFrame()
        rows = [[int(k[0])] + [float(k[i]) for i in range(1, 6)] for k in kl
                if isinstance(k, (list, tuple)) and len(k) >= 6]
        df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        return df.set_index('timestamp').sort_index().dropna()
    except Exception as e:
        logger.debug(f"aster ohlcv fallback: {e}")
        return pd.DataFrame()

async def fetch_ohlcv_birdeye(limit: int = 500, resolution: str = '1m') -> pd.DataFrame:
    # Chain: Birdeye → Aster klines (venue-native, free) → Kraken. Quota breaker: after a
    # "compute units / usage limit" error, skip Birdeye entirely for BIRDEYE_COOLDOWN_SEC so a
    # exhausted quota doesn't add a failing 10s call to EVERY cycle (it was failing each loop).
    global _birdeye_cooldown_until
    t0 = time.time()
    if time.time() < _birdeye_cooldown_until:
        df = await _fetch_ohlcv_aster(limit)
        return df if not df.empty else await _fetch_ohlcv_kraken(limit)
    try:
        end_ts=int(time.time())
        res_sec={'1m':60,'5m':300,'15m':900,'1h':3600}.get(resolution,60)
        url="https://public-api.birdeye.so/defi/ohlcv"
        params={'address':'So11111111111111111111111111111111111111112',
                'type':resolution,'time_from':end_ts-res_sec*limit,'time_to':end_ts}
        async with aiohttp.ClientSession() as s:
            async with s.get(url, headers={'X-API-KEY':CONFIG['BIRDEYE_API_KEY']},
                             params=params, timeout=aiohttp.ClientTimeout(total=10)) as r:
                data = await r.json()
                items = data.get('data',{}).get('items',[]) if data.get('success') else []
                if not items:
                    msg = str(data.get('message', 'no data'))
                    record_api('birdeye', False, (time.time()-t0)*1000, msg)
                    if 'limit' in msg.lower() or 'compute units' in msg.lower():
                        _birdeye_cooldown_until = time.time() + float(CONFIG.get('BIRDEYE_COOLDOWN_SEC', 3600))
                        logger.warning(f"  [ohlcv] Birdeye quota exhausted — cooling down "
                                       f"{CONFIG.get('BIRDEYE_COOLDOWN_SEC', 3600)}s; using Aster klines")
                    df = await _fetch_ohlcv_aster(limit)
                    return df if not df.empty else await _fetch_ohlcv_kraken(limit)
                df=pd.DataFrame(items).rename(columns={'o':'open','h':'high','l':'low',
                                                       'c':'close','v':'volume','unixTime':'timestamp'})
                df['timestamp']=pd.to_datetime(df['timestamp'],unit='s',utc=True)
                df=df.set_index('timestamp').sort_index()
                for c in ['open','high','low','close','volume']:
                    df[c]=pd.to_numeric(df[c],errors='coerce')
                record_api('birdeye', True, (time.time()-t0)*1000)
                return df.dropna()
    except Exception as e:
        record_api('birdeye', False, (time.time()-t0)*1000, str(e))
        df = await _fetch_ohlcv_aster(limit)
        return df if not df.empty else await _fetch_ohlcv_kraken(limit)

async def fetch_lunarcrush_sentiment() -> Dict:
    cached = ttl_get('lunarcrush', CONFIG.get('TTL_SENTIMENT_SEC', 60))
    if cached is not None:
        return cached
    t0 = time.time()
    try:
        url="https://lunarcrush.com/api4/public/coins/sol/v1"
        hdr={'Authorization':f"Bearer {CONFIG['LUNARCRUSH_API_KEY']}"}
        s = await get_http()
        async with s.get(url, headers=hdr, timeout=aiohttp.ClientTimeout(total=8)) as r:
            coin=(await r.json()).get('data',{})
            record_api('lunarcrush', True, (time.time()-t0)*1000)
            result = {'galaxy_score':float(coin.get('galaxy_score',50)),
                      'alt_rank':float(coin.get('alt_rank',500)),
                      'sentiment':float(coin.get('sentiment',3)),
                      'social_volume':float(coin.get('social_volume',0)),
                      'social_score':float(coin.get('social_score',0))}
            ttl_put('lunarcrush', result)
            return result
    except Exception as e:
        record_api('lunarcrush', False, (time.time()-t0)*1000, str(e))
        return {'galaxy_score':50,'alt_rank':500,'sentiment':3,'social_volume':0,'social_score':0}

async def fetch_x_sentiment() -> float:
    # RETIRED 2026-06-10: persistent HTTP 401 (token tier); replaced by news_rss+Ollama sentiment.
    # Disabled by default so it stops failing every cycle and polluting data-freshness.
    if not CONFIG.get('SOCIAL_SCRAPERS_ENABLE', False):
        return 0.0
    t0 = time.time()
    if not CONFIG['X_BEARER_TOKEN']:
        return 0.0
    try:
        hdr={'Authorization':f"Bearer {CONFIG['X_BEARER_TOKEN']}"}
        url="https://api.x.com/2/tweets/search/recent?query=SOL%20OR%20Solana&max_results=20"
        async with aiohttp.ClientSession() as s:
            async with s.get(url, headers=hdr, timeout=aiohttp.ClientTimeout(total=8)) as r:
                if r.status==200:
                    record_api('twitter', True, (time.time()-t0)*1000)
                    if HAS_RL:
                        analyzer=SentimentIntensityAnalyzer()
                        tweets=(await r.json()).get('data',[])
                        scores=[analyzer.polarity_scores(t.get('text',''))['compound']
                                for t in tweets if isinstance(t,dict)]
                        return float(np.mean(scores)) if scores else 0.0
                else:
                    record_api('twitter', False, (time.time()-t0)*1000, f'HTTP {r.status}')
        return 0.0
    except Exception as e:
        record_api('twitter', False, (time.time()-t0)*1000, str(e))
        return 0.0

async def fetch_reddit_sentiment() -> float:
    # RETIRED 2026-06-10: persistent HTTP 403 (cookieless blocked); replaced by news_rss+Ollama.
    if not CONFIG.get('SOCIAL_SCRAPERS_ENABLE', False):
        return 0.0
    """Cookieless Reddit JSON fetch — no OAuth/PRAW required.
    Uses Reddit's public .json endpoint with a browser-like User-Agent."""
    t0 = time.time()
    try:
        url = "https://www.reddit.com/r/solana/hot.json?limit=25"
        hdr = {
            'User-Agent': CONFIG['REDDIT_USER_AGENT'],
            'Accept': 'application/json',
        }
        async with aiohttp.ClientSession() as s:
            async with s.get(url, headers=hdr, timeout=aiohttp.ClientTimeout(total=10)) as r:
                if r.status != 200:
                    record_api('reddit', False, (time.time()-t0)*1000, f'HTTP {r.status}')
                    return 0.0
                data = await r.json()
                posts = data.get('data', {}).get('children', [])
                titles = [p.get('data', {}).get('title', '') for p in posts]
                titles = [t for t in titles if t]
                record_api('reddit', True, (time.time()-t0)*1000)
                if not titles or not HAS_RL:
                    return 0.0
                analyzer = SentimentIntensityAnalyzer()
                scores = [analyzer.polarity_scores(t)['compound'] for t in titles]
                return float(np.mean(scores)) if scores else 0.0
    except Exception as e:
        record_api('reddit', False, (time.time()-t0)*1000, str(e))
        return 0.0

async def fetch_google_trends() -> float:
    # RETIRED 2026-06-10: persistent HTTP 429 (rate-limited); replaced by news_rss+Ollama.
    if not CONFIG.get('SOCIAL_SCRAPERS_ENABLE', False):
        return 0.0
    if not HAS_PYTRENDS:
        return 0.5
    t0 = time.time()
    try:
        def _sync():
            pt=TrendReq(hl='en-US',tz=360)
            pt.build_payload(['Solana','SOL crypto'],timeframe='now 1-d')
            df=pt.interest_over_time()
            return float(df['Solana'].mean()/100) if not df.empty else 0.5
        loop=asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _sync)
        record_api('pytrends', True, (time.time()-t0)*1000)
        return result
    except Exception as e:
        record_api('pytrends', False, (time.time()-t0)*1000, str(e))
        return 0.5

async def fetch_combined_sentiment() -> float:
    reddit, x_s, google = await asyncio.gather(
        fetch_reddit_sentiment(), fetch_x_sentiment(), fetch_google_trends()
    )
    return float(np.clip(0.4*reddit + 0.3*x_s + 0.3*google, -1, 1))

async def fetch_polymarket_sol() -> Dict:
    t0 = time.time()
    try:
        url="https://clob.polymarket.com/markets"
        async with aiohttp.ClientSession() as s:
            async with s.get(url, params={'tag':'crypto','closed':'false'},
                             timeout=aiohttp.ClientTimeout(total=8)) as r:
                data=await r.json()
                sol=[m for m in data.get('data',[]) if 'SOL' in m.get('question','').upper()]
                record_api('polymarket', True, (time.time()-t0)*1000)
                if sol:
                    tokens=sol[0].get('tokens',[])
                    return {'yes_probability':float(tokens[0].get('price',0.5)) if tokens else 0.5,
                            'volume':float(sol[0].get('volume',0))}
        return {'yes_probability':0.5,'volume':0}
    except Exception as e:
        record_api('polymarket', False, (time.time()-t0)*1000, str(e))
        return {'yes_probability':0.5,'volume':0}

async def fetch_solana_on_chain() -> Dict:
    # Rate-limit guard (v6): the wallet SOL balance changes rarely, so cache it for
    # TTL_ONCHAIN_SEC (the CONFIG intent that was previously unwired). Without this the
    # public Solana RPC was hit on EVERY /api/status poll (~3s) AND every market cycle
    # (~1-2s) — wasteful + throttle-prone. Also use the shared HTTP session to avoid a
    # fresh TCP+TLS handshake per call. estimated_fee_usd is refreshed live below.
    _ck = 'onchain_sol'
    cached = ttl_get(_ck, CONFIG.get('TTL_ONCHAIN_SEC', 30))
    if cached is not None:
        return {**cached, 'estimated_fee_usd': cached.get('estimated_fee_sol', 0.000005) * state.sol_price}
    t0 = time.time()
    try:
        payload={"jsonrpc":"2.0","id":1,"method":"getBalance","params":[CONFIG['PUBLIC_KEY']]}
        s = await get_http()
        async with s.post(CONFIG['SOLANA_RPC_URL'], json=payload,
                          headers={'Content-Type':'application/json'},
                          timeout=aiohttp.ClientTimeout(total=5)) as r:
            sol_balance=float((await r.json()).get('result',{}).get('value',0))/1e9
        fee_sol=0.000005
        record_api('solana_rpc', True, (time.time()-t0)*1000)
        out={'sol_balance':sol_balance,'estimated_fee_sol':fee_sol,
             'estimated_fee_usd':fee_sol*state.sol_price}
        ttl_put(_ck, out)
        return out
    except Exception as e:
        record_api('solana_rpc', False, (time.time()-t0)*1000, str(e))
        return {'sol_balance':0,'estimated_fee_sol':0.000005,'estimated_fee_usd':0}

async def fetch_fred_series(series_id: str) -> float:
    key=CONFIG['FRED_API_KEY']
    if not key: return 0.0
    cached = ttl_get(f"fred:{series_id}", CONFIG.get('TTL_FRED_SEC', 3600))
    if cached is not None:
        return cached
    t0 = time.time()
    try:
        url=(f"https://api.stlouisfed.org/fred/series/observations"
             f"?series_id={series_id}&api_key={key}&file_type=json&limit=1&sort_order=desc")
        s = await get_http()
        async with s.get(url, timeout=aiohttp.ClientTimeout(total=8)) as r:
            obs=(await r.json()).get('observations',[{}])
            val = float(obs[0].get('value',0)) if obs else 0.0
            record_api('fred', True, (time.time()-t0)*1000)
            ttl_put(f"fred:{series_id}", val)
            return val
    except Exception as e:
        record_api('fred', False, (time.time()-t0)*1000, str(e))
        return 0.0

async def fetch_fred_macro() -> Dict:
    dxy, vix, t10y = await asyncio.gather(
        fetch_fred_series('DTWEXBGS'),
        fetch_fred_series('VIXCLS'),
        fetch_fred_series('DGS10'),
    )
    return {'dxy':dxy,'vix':vix,'treasury_10y':t10y}

async def fetch_dune_query(query_id: int) -> Dict:
    """Generic Dune query fetcher — uses cached query results."""
    key = CONFIG.get('DUNE_API_KEY','')
    if not key: return {}
    t0 = time.time()
    try:
        url = f"https://api.dune.com/api/v1/query/{query_id}/results"
        hdr = {'X-Dune-API-Key': key}
        async with aiohttp.ClientSession() as s:
            async with s.get(url, headers=hdr, timeout=aiohttp.ClientTimeout(total=10)) as r:
                data = await r.json()
                ok = r.status == 200
                record_api('dune', ok, (time.time()-t0)*1000, '' if ok else f'HTTP {r.status}')
                return data if ok else {}
    except Exception as e:
        record_api('dune', False, (time.time()-t0)*1000, str(e))
        return {}

async def fetch_btc_price_cex() -> float:
    """BTC spot from CEX index (Kraken → CoinGecko) for SOL/BTC cointegration.
    Replaces the retired Drift BTC oracle read. 30s TTL cached."""
    cached = ttl_get('btc_price_cex', 30)
    if cached is not None:
        return cached
    t0 = time.time()
    try:
        s = await get_http()
        async with s.get("https://api.kraken.com/0/public/Ticker",
                         params={'pair': 'XBTUSD'},
                         timeout=aiohttp.ClientTimeout(total=5)) as r:
            d = await r.json()
            res = d.get('result', {})
            if res:
                first = next(iter(res.values()))
                px = float(first['c'][0])
                if px > 0:
                    record_api('kraken', True, (time.time() - t0) * 1000)
                    ttl_put('btc_price_cex', px)
                    return px
    except Exception as e:
        record_api('kraken', False, (time.time() - t0) * 1000, str(e))
    try:
        s = await get_http()
        async with s.get("https://api.coingecko.com/api/v3/simple/price",
                         params={'ids': 'bitcoin', 'vs_currencies': 'usd',
                                 'x_cg_demo_api_key': CONFIG.get('COINGECKO_API_KEY', '')},
                         timeout=aiohttp.ClientTimeout(total=5)) as r:
            px = float((await r.json())['bitcoin']['usd'])
            ttl_put('btc_price_cex', px)
            return px
    except Exception:
        return 0.0

# Rolling venue-funding history so funding_slope / z-score have a time series.
_funding_hist: deque = deque(maxlen=12)

async def fetch_drift_funding_history() -> List[float]:
    """v4.00 — funding rate from the ACTIVE VENUE (Aster primary / ApeX) via the
    router, replacing the retired on-chain Drift source. Keeps a short rolling
    history so downstream funding_slope and z-score keep working, and appends
    the venue's *predicted* next funding so the slope is forward-aware.
    (Function name retained for call-site compatibility; source is venue-native.)"""
    t0 = time.time()
    try:
        f = await router.get_funding()
        rate = float(f.get('rate', 0.0) or 0.0)
        pred = float(f.get('predicted', rate) or rate)
        _funding_hist.append(rate)
        record_api(router.last_venue if router.last_venue in ('aster', 'apex') else 'aster',
                   True, (time.time() - t0) * 1000)
        hist = list(_funding_hist)
        hist.append(pred)                      # forward-aware tail
        if len(hist) < 2:                      # guarantee np.diff() has ≥2 points
            hist = [rate, pred]
        return hist
    except Exception as e:
        record_api('aster', False, (time.time() - t0) * 1000, str(e))
        return [0.0, 0.0]

async def fetch_liquidation_data() -> Dict:
    """v4.00 — venue liquidation feed (Aster /allForceOrders) via the router,
    plus open-interest and long/short ratio (microstructure signals that
    replace the retired Solana on-chain proxies). Cached briefly upstream."""
    t0 = time.time()
    try:
        liq, oi, lsr = await asyncio.gather(
            router.get_liquidations(), router.get_open_interest(),
            router.get_long_short_ratio(),
            return_exceptions=True)
        liq = liq if isinstance(liq, dict) else {}
        oi  = float(oi) if isinstance(oi, (int, float)) else 0.0
        lsr = float(lsr) if isinstance(lsr, (int, float)) else 1.0
        long_liq  = float(liq.get('long_liq', 0.0))
        short_liq = float(liq.get('short_liq', 0.0))
        total_usd = float(liq.get('total', long_liq + short_liq)) * max(state.sol_price, 1.0)
        # Surface microstructure to state for the dashboard + signal layer.
        state.open_interest    = oi
        state.long_short_ratio = lsr
        state.liq_long, state.liq_short = long_liq, short_liq
        record_api('aster', True, (time.time() - t0) * 1000)
        return {'total_liquidated_usd': total_usd,
                'liquidation_count': int((long_liq + short_liq) > 0),
                'long_liq': long_liq, 'short_liq': short_liq,
                'open_interest': oi, 'long_short_ratio': lsr}
    except Exception as e:
        record_api('aster', False, (time.time() - t0) * 1000, str(e))
        return {'total_liquidated_usd': 0, 'liquidation_count': 0}

async def fetch_helius_nft_activity() -> float:
    t0 = time.time()
    try:
        url="https://api.helius.xyz/v0/nft-events"
        params={'api-key':CONFIG['HELIUS_API_KEY'],'limit':20,'type':['NFT_SALE']}
        async with aiohttp.ClientSession() as s:
            async with s.get(url, params=params, timeout=aiohttp.ClientTimeout(total=8)) as r:
                data=await r.json()
                record_api('helius', True, (time.time()-t0)*1000)
                return float(len(data if isinstance(data,list) else []))/20.0
    except Exception as e:
        record_api('helius', False, (time.time()-t0)*1000, str(e))
        return 0.5

# ============================================================
# SECTION 13: TECHNICAL ANALYSIS
# ============================================================
FEATURE_COLS = [
    'rsi','rsi_fast','macd','macd_hist','bb_upper','bb_lower','bb_middle',
    'atr_pct','adx','cci','mfi','obv','vwap','ema9','ema21','ema50',
    'willr','stoch_k','stoch_d','aroon_up','aroon_down','ultosc',
    'plus_di','minus_di','trix','mom','roc','bop','natr',
    'vol_ratio','rel_vol','returns','returns_5','returns_10',
    'kalman_trend','mesa_signal','bk_pressure','fib_confluence',
    'inst_vol_trend','whale_cluster','ofi','ofi_ma','fvg_bull','fvg_bear',
    'supply_zone','demand_zone','poly_reg_diff','price_vs_poc',
    'vpin','hurst','squeeze','cdl_hammer','cdl_engulf','cdl_doji',
    'vol_ma20','ewma_vol','ht_trendmode','trend_strength',
]

# ---- Empirical historic-matching feature weights (#14) ---------------------
def _load_historic_feature_weights():
    """Load substrate-derived per-feature predictive weights (name->weight).
    Tries the configured path, then a local models/ copy, then the Projects copy.
    Returns {} if none readable (callers then fall back to uniform weighting)."""
    candidates = [
        CONFIG.get('HISTORIC_WEIGHTS_PATH', ''),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'feature_weights.json'),
        os.path.join(os.getenv('DELPHI_TRADING_ROOT', os.path.expanduser('~/Delphi_Trading')), 'models', 'feature_weights.json'),
    ]
    for p in candidates:
        try:
            if p and os.path.exists(p):
                d = json.load(open(p))
                wmap = d.get('feature_weights', {}) or {}
                if wmap:
                    logger.info(f"Historic-match weights loaded: {len(wmap)} features from {p}")
                    return {str(k): float(v) for k, v in wmap.items()}
        except Exception as _e:
            continue
    logger.info("Historic-match weights: none found — using uniform (legacy) weighting")
    return {}

FEATURE_WEIGHT_MAP = _load_historic_feature_weights()


def _norm_mean1(w) -> np.ndarray:
    """Clip negatives, then scale so the mean weight is 1.0 — this makes a uniform
    weight vector reproduce the UNWEIGHTED distance/cosine exactly, preserving the
    calibration of every downstream threshold (the 0.88 match gate, the ×0.02 nudge)."""
    w = np.clip(np.nan_to_num(np.asarray(w, dtype=float), nan=0.0), 0.0, None)
    m = float(w.mean()) if w.size else 0.0
    return (w / m) if m > 1e-9 else np.ones(len(w), dtype=float)


def _historic_weights(col_names, hist=None, hist_fwd=None) -> np.ndarray:
    """Empirical per-feature weights aligned to `col_names`, mean-normalized to 1.0.

    * Disabled or no weight file  -> uniform ones (EXACT legacy behavior).
    * Static  -> substrate prior (rank|Spearman| + rank|logit-coef|) per feature.
    * Dynamic -> blend the static prior with a LIVE |Spearman(feat, fwd_return)|
      computed over the aligned (hist, hist_fwd) rolling window, so the weighting
      adapts to the current regime's "demonstrated predictive value over a
      dynamically-determined time window" (the user's exact ask). Robust: any
      failure degrades to the static prior, never raises."""
    n = len(col_names)
    if not CONFIG.get('HISTORIC_WEIGHT_ENABLE', True) or not FEATURE_WEIGHT_MAP:
        return np.ones(n, dtype=float)
    static = np.array([float(FEATURE_WEIGHT_MAP.get(c, 0.5)) for c in col_names], dtype=float)
    if (CONFIG.get('HISTORIC_WEIGHT_DYNAMIC', True)
            and hist is not None and hist_fwd is not None):
        try:
            from scipy.stats import spearmanr
            H = np.asarray(hist, dtype=float); f = np.asarray(hist_fwd, dtype=float)
            if H.ndim == 2 and H.shape[1] == n and H.shape[0] >= 30 and len(f) == H.shape[0]:
                ic = np.zeros(n, dtype=float)
                for j in range(n):
                    cj = H[:, j]
                    if np.nanstd(cj) > 1e-12:
                        rho = spearmanr(cj, f, nan_policy='omit').statistic
                        ic[j] = abs(rho) if np.isfinite(rho) else 0.0
                rank = ic.argsort().argsort().astype(float)        # 0..n-1 by |IC|
                live = rank / max(rank.max(), 1.0)                  # -> [0,1]
                sp = static / max(static.max(), 1e-9)               # -> [0,1]
                blend = float(CONFIG.get('HISTORIC_WEIGHT_PRIOR', 0.5))
                static = blend * sp + (1.0 - blend) * live
        except Exception:
            pass
    return _norm_mean1(static)


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 60: return df
    c=df['close'].values.astype(float); h=df['high'].values.astype(float)
    l=df['low'].values.astype(float);  o=df['open'].values.astype(float)
    v=df['volume'].values.astype(float)

    df['rsi']=talib.RSI(c,14); df['rsi_fast']=talib.RSI(c,7)
    df['macd'],df['macd_signal'],df['macd_hist']=talib.MACD(c)
    df['bb_upper'],df['bb_middle'],df['bb_lower']=talib.BBANDS(c,20)
    df['atr']=talib.ATR(h,l,c,14); df['atr_pct']=df['atr']/c
    df['adx']=talib.ADX(h,l,c,14); df['cci']=talib.CCI(h,l,c,20)
    df['mfi']=talib.MFI(h,l,c,v,14); df['obv']=talib.OBV(c,v)
    df['vwap']=(df['close']*df['volume']).cumsum()/df['volume'].cumsum()
    df['ema9']=talib.EMA(c,9); df['ema21']=talib.EMA(c,21)
    df['ema50']=talib.EMA(c,50); df['ema200']=talib.EMA(c,200)
    df['sma20']=talib.SMA(c,20); df['willr']=talib.WILLR(h,l,c,14)
    df['stoch_k'],df['stoch_d']=talib.STOCH(h,l,c)
    df['sar']=talib.SAR(h,l)
    df['aroon_up'],df['aroon_down']=talib.AROON(h,l,25)
    df['dx']=talib.DX(h,l,c,14); df['ultosc']=talib.ULTOSC(h,l,c)
    df['plus_di']=talib.PLUS_DI(h,l,c,14); df['minus_di']=talib.MINUS_DI(h,l,c,14)
    df['trix']=talib.TRIX(c,18); df['t3']=talib.T3(c,5)
    df['tema']=talib.TEMA(c,20); df['dema']=talib.DEMA(c,20)
    df['kama']=talib.KAMA(c,30); df['mom']=talib.MOM(c,10)
    df['roc']=talib.ROC(c,10); df['rocp']=talib.ROCP(c,10)
    df['bop']=talib.BOP(o,h,l,c); df['ad']=talib.AD(h,l,c,v)
    df['natr']=talib.NATR(h,l,c,14)

    df['cdl_hammer']=talib.CDLHAMMER(o,h,l,c)
    df['cdl_engulf']=talib.CDLENGULFING(o,h,l,c)
    df['cdl_doji']=talib.CDLDOJI(o,h,l,c)
    df['cdl_morning_star']=talib.CDLMORNINGSTAR(o,h,l,c)
    df['cdl_shooting_star']=talib.CDLSHOOTINGSTAR(o,h,l,c)

    kc_m=talib.EMA(c,20); kc_b=2*talib.ATR(h,l,c,20)
    df['kc_upper']=kc_m+kc_b; df['kc_lower']=kc_m-kc_b
    df['squeeze']=((df['bb_upper']<df['kc_upper'])&(df['bb_lower']>df['kc_lower'])).astype(float)
    df['donchian_high']=df['high'].rolling(20).max()
    df['donchian_low']=df['low'].rolling(20).min()
    df['ewma_vol']=df['close'].pct_change().ewm(span=20).std()
    df['vol_ma20']=df['volume'].rolling(20).mean()
    df['vol_ratio']=df['volume']/df['vol_ma20']
    df['rel_vol']=(df['volume']-df['vol_ma20'])/df['vol_ma20'].replace(0,1)
    df['returns']=df['close'].pct_change()
    df['returns_5']=df['close'].pct_change(5)
    df['returns_10']=df['close'].pct_change(10)
    df['returns_20']=df['close'].pct_change(20)

    df['kalman_price']=kalman_filter.smooth_series(c)
    df['kalman_trend']=df['kalman_price'].diff()
    df['super_smoother']=super_smoother(c,10)
    mama,fama=mesa_adaptive_ma(c)
    df['mama']=mama; df['fama']=fama
    df['mesa_signal']=(mama>fama).astype(float)

    hurst_v=np.full(len(df),np.nan)
    for i in range(100,len(df)): hurst_v[i]=hurst_exponent(c[i-100:i])
    df['hurst']=hurst_v

    vpin_v=np.full(len(df),0.5)
    for i in range(100,len(df)): vpin_v[i]=compute_vpin(df.iloc[i-100:i])
    df['vpin']=vpin_v

    poc=np.full(len(df),np.nan)
    for i in range(50,len(df)):
        hist_,bins=np.histogram(c[i-50:i],bins=20,weights=v[i-50:i])
        poc[i]=bins[np.argmax(hist_)]
    df['vol_poc']=poc; df['price_vs_poc']=c-poc

    poly=np.full(len(df),np.nan)
    for i in range(50,len(df)):
        coeff=np.polyfit(np.arange(50),c[i-50:i],2)
        poly[i]=np.polyval(coeff,49)
    df['poly_reg']=poly; df['poly_reg_diff']=c-poly

    df['ofi']=((c-o)/(h-l+1e-10))*v
    df['ofi_ma']=df['ofi'].rolling(10).mean()
    df['fvg_bull']=(df['low'].shift(-1)>df['high'].shift(1)).astype(float)
    df['fvg_bear']=(df['high'].shift(-1)<df['low'].shift(1)).astype(float)
    df['supply_zone']=(c>df['high'].rolling(20).max()*0.98).astype(float)
    df['demand_zone']=(c<df['low'].rolling(20).min()*1.02).astype(float)
    bb_rng=df['bb_upper']-df['bb_lower']
    df['bk_pressure']=(c-df['bb_middle'])/(bb_rng.replace(0,1))
    rh=df['high'].rolling(50).max(); rl=df['low'].rolling(50).min()
    ret=(c-rl.values)/(rh.values-rl.values+1e-10)
    df['fib_618']=(np.abs(ret-0.618)<0.03).astype(float)
    df['fib_786']=(np.abs(ret-0.786)<0.03).astype(float)
    df['fib_50']=(np.abs(ret-0.500)<0.03).astype(float)
    df['fib_confluence']=((df['fib_618']+df['fib_786']+df['fib_50'])>0).astype(float)
    df['inst_vol_trend']=(talib.EMA(v,20)>talib.EMA(v,50)).astype(float)
    vmu=df['volume'].mean(); vstd=df['volume'].std()
    df['whale_cluster']=(v>vmu+2.5*vstd).astype(float)
    emd_f=np.full(len(df),np.nan)
    for i in range(100,len(df)):
        try:
            imfs=empirical_mode_decomposition(c[i-100:i],num_imfs=2)
            if imfs:
                ft=np.fft.fft(imfs[0])
                emd_f[i]=len(imfs[0])/(np.argmax(np.abs(ft)[:50])+1e-10)
        except Exception: pass
    df['emd_freq']=emd_f
    df['ht_sine'],df['ht_lead_sine']=talib.HT_SINE(c)
    df['ht_phase']=talib.HT_PHASOR(c)[0]
    df['ht_trendmode']=talib.HT_TRENDMODE(c)
    df['trend_strength']=df['adx']/100.0
    df['in_trend']=(df['adx']>25).astype(float)
    return df

def _triple_barrier_labels(df: pd.DataFrame) -> np.ndarray:
    """First-touch TP/SL/time labels sized off ATR (López de Prado): +1 if the TP
    barrier (entry + PT*ATR) is hit first, -1 if SL (entry - SL*ATR) is hit first,
    0 if neither before the vertical (time) barrier. Aligns labels to real outcomes."""
    close = df['close'].values.astype(float)
    if 'atr' in df.columns:
        atr = df['atr'].values.astype(float)
    else:
        atr = (df['close'].pct_change().rolling(14).std() * df['close']).values
    pt = float(CONFIG.get('TB_PT_ATR_MULT', 2.0))
    sl = float(CONFIG.get('TB_SL_ATR_MULT', 1.5))
    H = int(CONFIG.get('TB_MAX_HOLD_BARS', 15))
    n = len(close); lab = np.zeros(n, dtype=int)
    for i in range(n):
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        up = close[i] + pt * a
        dn = close[i] - sl * a
        end = min(i + H, n - 1)
        touched = 0
        for j in range(i + 1, end + 1):
            if close[j] >= up: touched = 1; break
            if close[j] <= dn: touched = -1; break
        if touched == 0 and end > i and close[i]:
            r = (close[end] - close[i]) / close[i]
            touched = 1 if r > 0 else (-1 if r < 0 else 0)
        lab[i] = touched
    return lab


def build_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    df=df.copy()
    # Quantile-based labels (Two Sigma / Jane Street style) — guarantees ~30/40/30
    # class balance regardless of volatility regime. Eliminates the "always-hold"
    # overfit that fixed ±0.3% labels produced in low-vol crypto markets.
    fwd = df['close'].pct_change(5).shift(-5)
    df['future_ret'] = fwd
    # Use *symmetric* quantiles around 0 to avoid biasing toward whichever side
    # of the distribution dominated this training window.
    valid = fwd.dropna()
    if len(valid) >= 50:
        hi = float(valid.quantile(0.70))
        lo = float(valid.quantile(0.30))
        # Enforce a minimum band so we still get hold-class examples in trends
        hi = max(hi, 0.0005); lo = min(lo, -0.0005)
    else:
        hi, lo = 0.003, -0.003
    if CONFIG.get('LABEL_MODE', 'quantile') == 'triple_barrier' and 'atr' in df.columns:
        df['label'] = _triple_barrier_labels(df)
    else:
        df['label'] = 0
        df.loc[fwd > hi, 'label'] = 1
        df.loc[fwd < lo, 'label'] = -1
    fc=[c for c in FEATURE_COLS if c in df.columns]
    df[fc]=df[fc].replace([np.inf,-np.inf],np.nan)
    df=df.dropna(subset=fc+['label'])
    y=df['label'].values
    return df[fc].values, np.where(y==-1,0,np.where(y==0,1,2))

# ============================================================
# SECTION 14: ML ENSEMBLE
# ============================================================
class MLEnsemble:
    def __init__(self):
        self.rf=RandomForestClassifier(n_estimators=200,max_depth=10,
                                       min_samples_leaf=5,n_jobs=1,random_state=42,
                                       class_weight='balanced')
        self.xgb=XGBClassifier(n_estimators=200,max_depth=6,learning_rate=0.05,
                                subsample=0.8,colsample_bytree=0.8,
                                eval_metric='mlogloss',verbosity=0,random_state=42,
                                n_jobs=1, tree_method='hist')
        self.gnb=GaussianNB()
        self.gb=GradientBoostingClassifier(n_estimators=100,max_depth=4,
                                            learning_rate=0.1,subsample=0.8,random_state=42)
        self.meta=LogisticRegression(max_iter=500,C=1.0,solver='lbfgs')
        self.scaler=RobustScaler(); self.fitted=False
        self.conformal_scores: Dict[int,np.ndarray]={}   # Mondrian split-conformal nonconformity
        self.feature_importance: Dict[str,float]={}
        self.retrain_counter=0

    def fit(self, X: np.ndarray, y: np.ndarray):
        if len(X)<50: return
        Xs=self.scaler.fit_transform(X)
        bases=(self.rf,self.xgb,self.gnb,self.gb)
        logger.info("    [ml] base fits (rf/xgb/gnb/gb)...")
        for m in bases:
            m.fit(Xs,y)            # production bases — fit on ALL rows (used at inference)
        use_oof=(CONFIG.get('META_OOF_ENABLE',True)
                 and len(X)>=CONFIG.get('META_OOF_MIN_ROWS',120))
        if use_oof:
            # De-leak: the meta sees OUT-OF-FOLD base proba (each row predicted by a
            # clone NOT trained on it). TimeSeriesSplit preserves time order (no shuffle)
            # and is NOT a partition (its first fold's rows are never predicted), so we
            # fill OOF proba manually per fold and trim the uncovered warm-up rows. The
            # production bases (fit on all rows above) are untouched.
            n_splits=int(CONFIG.get('META_OOF_SPLITS',3))
            cv=TimeSeriesSplit(n_splits=n_splits)
            classes=np.unique(y); n_cls=len(classes)
            col={int(c):i for i,c in enumerate(classes)}
            oof=np.full((len(Xs), n_cls*len(bases)), np.nan)
            covered=np.zeros(len(Xs), dtype=bool)
            for bi,m in enumerate(bases):
                for tr,te in cv.split(Xs):
                    b=clone(m).fit(Xs[tr], y[tr])
                    pr=b.predict_proba(Xs[te])
                    block=np.zeros((len(te), n_cls))
                    for j,c in enumerate(b.classes_):
                        if int(c) in col:
                            block[:, col[int(c)]]=pr[:, j]
                    oof[te, bi*n_cls:(bi+1)*n_cls]=block
                    covered[te]=True
            mX=oof[covered]; y_meta=y[covered]
            self.meta.fit(mX, y_meta)
            self._oof_first_test=int(np.argmax(covered)) if covered.any() else 0
            logger.info(f"    [ml] meta.fit OOF ({n_splits}-fold TSS) — "
                        f"meta rows={len(y_meta)}/{len(y)}")
            # per-model OOF telemetry (DISPLAY-ONLY) — reuse the OOF proba matrix
            try:
                _inv = {i: c for c, i in col.items()}
                _oc = oof[covered]
                _per = {}
                for _bi, _nm in enumerate(('rf', 'xgb', 'gnb', 'gb')):
                    _sub = _oc[:, _bi*n_cls:(_bi+1)*n_cls]
                    _pr = np.array([_inv[int(j)] for j in np.argmax(_sub, axis=1)])
                    _per[_nm] = {'accuracy': round(float(accuracy_score(y_meta, _pr)), 4),
                                 'f1': round(float(f1_score(y_meta, _pr, average='weighted', zero_division=0)), 4),
                                 'oof': True}
                _mp = self.meta.predict(mX)
                _per['meta_lr'] = {'accuracy': round(float(accuracy_score(y_meta, _mp)), 4),
                                   'f1': round(float(f1_score(y_meta, _mp, average='weighted', zero_division=0)), 4),
                                   'oof': False, 'note': 'in-sample'}
                _d = getattr(state, 'per_model_metrics', {}) or {}
                _d['ml'] = _per; _d['ml_updated'] = datetime.now(timezone.utc).isoformat()
                _d['n_train'] = int(len(y_meta))
                state.per_model_metrics = _d
            except Exception as _e:
                logger.debug(f"ml per-model metrics: {_e}")
        else:
            # Legacy in-sample path (KNOWN-LEAKY) — rollback / tiny-data only.
            mX=np.column_stack([m.predict_proba(Xs) for m in bases])
            self.meta.fit(mX,y)
            self._oof_first_test=0
            logger.info("    [ml] meta.fit LEGACY in-sample (OOF off or rows<MIN_ROWS)")
        fc=FEATURE_COLS[:X.shape[1]]
        self.feature_importance=dict(zip(fc,((self.rf.feature_importances_+
                                               self.xgb.feature_importances_)/2).tolist()))
        self.fitted=True
        if CONFIG.get('CONFORMAL_ENABLE', True):
            self._calibrate_conformal(Xs, y)
        state.benchmarks['ml_retrain_count']+=1
        logger.info(f"ML Ensemble trained — {len(X)} samples")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted or len(X)==0:
            return np.array([[0.33,0.34,0.33]])
        Xs=self.scaler.transform(X)
        mX=np.column_stack([m.predict_proba(Xs) for m in (self.rf,self.xgb,self.gnb,self.gb)])
        return self.meta.predict_proba(mX)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        if not self.fitted: return state.model_performance
        p=np.argmax(self.predict_proba(X),axis=1)
        return {'accuracy':float(accuracy_score(y,p)),
                'precision':float(precision_score(y,p,average='weighted',zero_division=0)),
                'recall':float(recall_score(y,p,average='weighted',zero_division=0)),
                'f1':float(f1_score(y,p,average='weighted',zero_division=0))}

    def evaluate_purged(self, X: np.ndarray, y: np.ndarray,
                        embargo: int = 5, n_splits: int = 3) -> Dict:
        """Honest walk-forward score with a purge+embargo gap = the label horizon,
        so train/test never share a forward-return window. Fits FRESH clones per
        fold; never touches the live production models (no leakage, no downgrade)."""
        if len(X) < (embargo + 60):
            return state.model_performance
        cv = TimeSeriesSplit(n_splits=n_splits)
        preds, trues = [], []
        for tr_idx, te_idx in cv.split(X):
            tr_idx = tr_idx[:-embargo] if len(tr_idx) > embargo else tr_idx
            te_idx = te_idx[embargo:]  if len(te_idx) > embargo else te_idx
            if len(tr_idx) < 40 or len(te_idx) < 5:
                continue
            ytr = y[tr_idx]
            if len(np.unique(ytr)) < 2:
                continue
            sc = RobustScaler().fit(X[tr_idx])
            Xtr, Xte = sc.transform(X[tr_idx]), sc.transform(X[te_idx])
            base_clones = [clone(m) for m in (self.rf, self.xgb, self.gnb, self.gb)]
            for m in base_clones:
                m.fit(Xtr, ytr)
            mXtr = np.column_stack([m.predict_proba(Xtr) for m in base_clones])
            mXte = np.column_stack([m.predict_proba(Xte) for m in base_clones])
            meta = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs').fit(mXtr, ytr)
            preds.append(np.argmax(meta.predict_proba(mXte), axis=1))
            trues.append(y[te_idx])
        if not preds:
            return state.model_performance
        p = np.concatenate(preds); t = np.concatenate(trues)
        return {'accuracy': float(accuracy_score(t, p)),
                'precision': float(precision_score(t, p, average='weighted', zero_division=0)),
                'recall':   float(recall_score(t, p, average='weighted', zero_division=0)),
                'f1':       float(f1_score(t, p, average='weighted', zero_division=0))}

    def _calibrate_conformal(self, Xs: np.ndarray, y: np.ndarray):
        """Split-conformal calibration: nonconformity = 1 - p(true class) on a held-out
        recent tail, bucketed per class (Mondrian). Uses the production bases+meta on
        the already-scaled Xs (no double-scaling). Stored for conformal_pvalue()."""
        try:
            n_cal = int(len(Xs) * float(CONFIG.get('CONFORMAL_CAL_FRAC', 0.20)))
            if n_cal < int(CONFIG.get('CONFORMAL_MIN_CAL', 40)):
                self.conformal_scores = {}; return
            Xc, yc = Xs[-n_cal:], y[-n_cal:]
            mX = np.column_stack([m.predict_proba(Xc) for m in (self.rf, self.xgb, self.gnb, self.gb)])
            proba = self.meta.predict_proba(mX)
            classes = list(self.meta.classes_)
            scores = {}
            for ci, cls in enumerate(classes):
                mask = (yc == cls)
                if int(mask.sum()) >= 5:
                    scores[int(cls)] = 1.0 - proba[mask, ci]
            self.conformal_scores = scores
        except Exception:
            self.conformal_scores = {}

    def conformal_pvalue(self, proba_row: np.ndarray, cls: int) -> float:
        """Mondrian split-conformal p-value for predicting `cls` (higher = more
        credible). Returns 1.0 (permissive) when the class is uncalibrated."""
        scores = self.conformal_scores.get(int(cls))
        if scores is None or len(scores) < 5:
            return 1.0
        s = 1.0 - float(proba_row[cls])
        return (float(np.sum(scores >= s)) + 1.0) / (len(scores) + 1.0)

ml_ensemble = MLEnsemble()

# ============================================================
# SECTION 15: DEEP LEARNING MODEL MANAGER
# ============================================================
class DLModelManager:
    def __init__(self, input_size: int = 64, seq_len: int = 30):
        self.seq_len=seq_len; self.input_size=input_size
        self.lstm=LSTMPriceModel(input_size); self.tf=TransformerSignalModel(input_size)
        self.bayes=BayesianUncertaintyHead(3)
        self.lstm_opt=optim.Adam(self.lstm.parameters(),lr=0.0003)
        self.tf_opt=optim.Adam(self.tf.parameters(),lr=0.0003)
        self.loss=nn.CrossEntropyLoss(); self.fitted=False
        self.alt_turn=True

    def _seqs(self, X, y):
        s,l=[],[]
        for i in range(self.seq_len,len(X)): s.append(X[i-self.seq_len:i]); l.append(y[i])
        return (np.array(s),np.array(l)) if s else (None,None)

    def _pad_to_input(self, arr: np.ndarray) -> np.ndarray:
        """Pad or truncate the last axis to match self.input_size.
        Avoids re-instantiating torch modules after solders has loaded
        (which triggers a Rust-extension/PyTorch memory segfault on macOS)."""
        n = arr.shape[-1]
        if n == self.input_size:
            return arr
        if n < self.input_size:
            pad_width = [(0,0)]*(arr.ndim-1) + [(0, self.input_size - n)]
            return np.pad(arr, pad_width, mode='constant')
        return arr[..., :self.input_size]

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 10):
        Xs=StandardScaler().fit_transform(X); seqs,labs=self._seqs(Xs,y)
        if seqs is None or len(seqs)<32: return
        seqs = self._pad_to_input(seqs)
        Xt=torch.tensor(seqs,dtype=torch.float32); yt=torch.tensor(labs,dtype=torch.long)
        loader=DataLoader(TensorDataset(Xt,yt),batch_size=32,shuffle=False)
        for model,opt in ((self.lstm,self.lstm_opt),(self.tf,self.tf_opt)):
            model.train()
            for _ in range(epochs):
                for xb,yb in loader:
                    opt.zero_grad(); out=model(xb); loss=self.loss(out,yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                    opt.step()
        self.fitted=True
        # per-model DL telemetry (DISPLAY-ONLY; train-set accuracy)
        try:
            _dlm = {}
            for _nm, _model in (('lstm', self.lstm), ('transformer', self.tf)):
                _model.eval()
                with torch.no_grad():
                    _pr = torch.argmax(_model(Xt), dim=1).cpu().numpy()
                _dlm[_nm] = {'accuracy': round(float(accuracy_score(labs, _pr)), 4),
                             'f1': round(float(f1_score(labs, _pr, average='weighted', zero_division=0)), 4),
                             'oof': False, 'note': 'train-set'}
            _d = getattr(state, 'per_model_metrics', {}) or {}
            _d['dl'] = _dlm; _d['dl_updated'] = datetime.now(timezone.utc).isoformat()
            state.per_model_metrics = _d
        except Exception as _e:
            logger.debug(f"dl per-model metrics: {_e}")
        logger.info("DL models trained (LSTM + Transformer)")

    def predict(self, X: np.ndarray, use_transformer: bool = False) -> Tuple[np.ndarray, float]:
        if not self.fitted or len(X)<self.seq_len:
            return np.array([0.33,0.34,0.33]), 1.0
        model=self.tf if use_transformer else self.lstm; model.eval()
        Xs=StandardScaler().fit_transform(X[-self.seq_len:])
        Xs = self._pad_to_input(Xs)
        xt=torch.tensor(Xs,dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            probs=model(xt).numpy()[0]
        pt=torch.tensor(probs,dtype=torch.float32).unsqueeze(0)
        _,unc=self.bayes(pt,training=True)
        return probs, float(unc.item())

dl_manager = DLModelManager()

# On-box AI analyst (Ollama delphi-trader-nox-dark) — lazy, guarded singleton. The
# engine runs unchanged if the module or Ollama are absent (local_ai stays None).
local_ai = None
try:
    if CONFIG.get('LOCAL_AI_ENABLE', True):
        from local_ai_analyst import (LocalAIAnalyst, build_ai_snapshot, apply_ai_to_signal,
                                       build_data_snapshot, augmentation_factor)
        local_ai = LocalAIAnalyst(CONFIG, logger)
        logger.info(f"Local-AI analyst ready (model={CONFIG.get('LOCAL_AI_MODEL')} "
                    f"fallback={CONFIG.get('LOCAL_AI_FALLBACK_MODEL')})")
except Exception as _ai_e:
    local_ai = None
    logger.warning(f"Local-AI analyst unavailable ({_ai_e}); continuing without it")

# Aster MLX analytics bridge — in-process access to the upgraded MCP tools (sizing +
# venue routing, 616-state vector DB, on-device MLX). Degrades to None → engine internals.
nox_bridge = None
try:
    if CONFIG.get('ASTER_MCP_ENABLE', True):
        import nox_analytics_bridge as _nbr
        if _nbr.available():
            nox_bridge = _nbr
            logger.info(f"Aster MLX bridge ready (root={_nbr.status()['root']})")
        else:
            logger.warning(f"Aster MLX bridge unavailable: {_nbr.status().get('import_error')}")
except Exception as _br_e:
    nox_bridge = None
    logger.warning(f"Aster MLX bridge unavailable ({_br_e}); using engine internals")

async def _fetch_one_feed(url: str) -> List[Tuple[float, str]]:
    """Fetch + parse one RSS feed → [(pub_epoch, title)]. Isolated so one bad feed can't sink
    the others (per-feed graceful degrade)."""
    import xml.etree.ElementTree as ET
    from email.utils import parsedate_to_datetime
    out: List[Tuple[float, str]] = []
    try:
        sess = await get_http()
        async with sess.get(url, headers={'User-Agent': 'Mozilla/5.0'},
                            timeout=aiohttp.ClientTimeout(total=6)) as r:
            raw = await r.read()
        root = ET.fromstring(raw)
        for it in root.findall('.//item'):
            title = (it.findtext('title') or '').strip()
            if not title:
                continue
            ts = 0.0
            pd = it.findtext('pubDate')
            if pd:
                try: ts = parsedate_to_datetime(pd).timestamp()
                except Exception: ts = 0.0
            out.append((ts, title))
    except Exception as e:
        logger.debug(f"news feed {url}: {e}")
    return out

async def fetch_crypto_news() -> List[str]:
    """Free crypto-news headlines via RSS (stdlib parse + shared HTTP session). Feeds are fetched
    in PARALLEL and merged RECENCY-FIRST (newest pubDate wins) so the model sees the freshest
    signal. TTL-cached (rate-limit guard). No API key — the reliable replacement for the dead
    X/Reddit/Google social scrapers."""
    cached = ttl_get('news_rss', CONFIG.get('TTL_NEWS_SEC', 540))
    if cached is not None:
        return cached
    t0 = time.time()
    feeds = CONFIG.get('NEWS_RSS_FEEDS', [])
    results = await asyncio.gather(*[_fetch_one_feed(u) for u in feeds], return_exceptions=True)
    items: List[Tuple[float, str]] = []
    ok_feeds = 0
    for r in results:
        if isinstance(r, list):
            if r: ok_feeds += 1
            items.extend(r)
    items.sort(key=lambda x: x[0], reverse=True)        # newest first
    seen = set(); heads: List[str] = []
    for _ts, title in items:
        k = title.lower()
        if k not in seen:
            seen.add(k); heads.append(title)
    heads = heads[:int(CONFIG.get('NEWS_MAX_HEADLINES', 18))]
    # top headlines for the dashboard news feed (dedup, newest-first, recency); DISPLAY-ONLY
    _now2 = time.time(); _seen2 = set(); _hl = []
    for _ts, _t in items:
        _k = _t.lower()
        if _k in _seen2:
            continue
        _seen2.add(_k)
        _hl.append({'title': _t[:140], 'age_min': (int((_now2 - _ts) / 60) if _ts else None)})
        if len(_hl) >= 6:
            break
    state.news_headlines = _hl
    record_api('news_rss', bool(heads), (time.time() - t0) * 1000,
               '' if heads else 'no items', )
    state.news_feeds_ok = ok_feeds                       # for the dashboard ("3/4 feeds")
    if heads:
        ttl_put('news_rss', heads)
    return heads


# ============================================================
# SECTION 16: MARKET DATA AGGREGATOR
# ============================================================
async def update_market_data() -> pd.DataFrame:
    df=await fetch_ohlcv_birdeye(limit=CONFIG['CANDLE_LIMIT'])
    if df.empty or len(df)<100:
        logger.warning("Insufficient candle data"); return pd.DataFrame()
    df=compute_all_indicators(df)

    sentiment, polymarket, onchain, funding_rates, liq_data, nft_score, macro = await asyncio.gather(
        fetch_lunarcrush_sentiment(), fetch_polymarket_sol(),
        fetch_solana_on_chain(), fetch_drift_funding_history(),
        fetch_liquidation_data(), fetch_helius_nft_activity(),
        fetch_fred_macro(),
    )

    df['social_volume']=sentiment.get('social_volume',0)
    df['galaxy_score']=sentiment.get('galaxy_score',50)
    df['sentiment_score']=sentiment.get('sentiment',3)
    df['polymarket_yes']=polymarket.get('yes_probability',0.5)
    df['funding_rate']=funding_rates[0] if funding_rates else 0.0
    df['funding_slope']=np.mean(np.diff(funding_rates)) if len(funding_rates)>1 else 0.0
    df['liq_usd']=liq_data.get('total_liquidated_usd',0)
    df['liq_count']=liq_data.get('liquidation_count',0)
    df['nft_activity']=nft_score
    df['dxy']=macro.get('dxy',0); df['vix']=macro.get('vix',0)

    if 'social_volume' in df.columns and len(df)>50:
        df['granger_social']=granger_causality_test(
            df['close'].values[-50:], df['social_volume'].fillna(0).values[-50:])
    else:
        df['granger_social']=0.5

    state.sol_price=float(df['close'].iloc[-1])
    state.funding_rate=float(df['funding_rate'].iloc[-1])
    state.macro_data=macro

    # Quant overlay updates — GARCH return, drawdown, cointegration,
    # sentiment trail, BOCPD probability, IC realized-side updates.
    try:
        ret_last = float(df['close'].pct_change().iloc[-1])
        quant_overlay.update_return(ret_last)
        quant_overlay.update_drawdown(state.balance)
        quant_overlay.update_sentiment(state.sentiment_score)
        quant_overlay.bocpd_proba()             # refresh changepoint cache
        quant_overlay.reflexivity_score()       # refresh reflexivity cache
        quant_overlay.vol_target_leverage()     # refresh vol-target lev
        quant_overlay.information_coefficient() # refresh IC cache
        # Crowded-trade score using funding + sentiment
        quant_overlay.crowded_trade_score(
            float(df['funding_rate'].iloc[-1]), state.sentiment_score)
        # SOL/BTC cointegration on CEX index prices (replaces Drift BTC oracle).
        try:
            btc_px = await fetch_btc_price_cex()
            if btc_px > 0 and state.sol_price > 0:
                quant_overlay.update_cointeg(state.sol_price, btc_px)
                quant_overlay.cointeg_zscore()
        except Exception:
            pass
        # IC: realized return vs prior forecast (1-bar-ahead)
        if len(quant_overlay.ic_forecasts) > len(quant_overlay.ic_realized):
            quant_overlay.ic_realized.append(ret_last)

        # Round-3: funding rate history + z-score, oracle-mark spread,
        # CF-CVaR, CUSUM event, pessimistic-Kelly refresh.
        quant_overlay.update_funding(state.funding_rate)
        quant_overlay.funding_zscore()
        quant_overlay.cornish_fisher_cvar()
        quant_overlay.cusum_event()
        quant_overlay.pessimistic_kelly()
        # Cross-venue basis (replaces Drift oracle-mark spread): ApeX mark vs
        # Aster mark divergence. Fed into the same oracle_mark_spread accumulator
        # so the downstream signal tilt keeps working — now a genuine two-venue
        # basis rather than an oracle-vs-close proxy.
        try:
            basis = await router.cross_venue_basis()
            state.cross_venue_basis = float(basis.get('basis', 0.0))
            # cross-venue lead-lag shadow feature (research; LOG-ONLY, NOT fingerprinted) — the
            # other venue often leads SOL; a widening/narrowing basis may predict Aster catching up.
            try:
                _xp = state.xv_features or {}
                _pb = _xp.get('xv_basis'); _pt = _xp.get('ts'); _nt = time.time()
                _chg = (state.cross_venue_basis - _pb) if _pb is not None else 0.0
                state.xv_features = {'xv_basis': round(state.cross_venue_basis, 6),
                                     'xv_basis_change': round(_chg, 6),
                                     'xv_basis_velocity': round(_chg / max(_nt - _pt, 1e-6), 6) if _pt else 0.0,
                                     'ts': _nt}
            except Exception:
                pass
            a_mark = float(basis.get('aster', 0.0))
            x_mark = float(basis.get('apex', 0.0))
            if a_mark > 0 and x_mark > 0:
                quant_overlay.oracle_mark_spread(x_mark, a_mark)  # (lead, ref)
            elif state.sol_price > 0:
                quant_overlay.oracle_mark_spread(state.sol_price, state.sol_price)
        except Exception:
            pass
        # OFI persistence from indicator column
        if 'ofi' in df.columns:
            quant_overlay.ofi_persistence(df['ofi'].values[-60:])
        # Multi-timeframe alignment
        try:
            mtf_score, _mtf_breakdown = QuantOverlay.multi_timeframe_alignment(df)
            quant_overlay.last_mtf_alignment = mtf_score
        except Exception:
            pass
    except Exception:
        pass
    state.ws_data.append({'ts':datetime.now(timezone.utc).isoformat(),
                           'price':state.sol_price,'volume':float(df['volume'].iloc[-1])})
    # Store the SAME feature vector the matcher queries with (all FEATURE_COLS
    # present in df, in order) so historic_match_score compares like-for-like.
    # (Previously stored a 4-dim [returns,adx,hurst,vpin] vector while the query
    # passes the ~58-dim X_seq[-1] → every cosine_dist raised on the length
    # mismatch and the DTW/Wasserstein historic-match silently returned 0.0.)
    _hfc = [c for c in FEATURE_COLS if c in df.columns]
    last_features=df[_hfc].iloc[-1].fillna(0).replace([np.inf,-np.inf],0).values.astype(float)
    state.historic_states.append(last_features)

    # HMM fit on 5-feature matrix: [returns, adx_norm, ewma_vol, vpin, vol_ratio]
    if not hmm_detector.fitted and len(df)>=100:
        feat_cols = ['returns','adx','ewma_vol','vpin','vol_ratio']
        available = [c for c in feat_cols if c in df.columns]
        if available:
            feat_mat = df[available].dropna().values
            if 'adx' in available:
                adx_idx = available.index('adx')
                feat_mat[:, adx_idx] /= 100.0
            hmm_detector.fit(feat_mat)

    return df

# ============================================================
# SECTION 17: REGIME DETECTION (HMM+GMM + RL + Hurst/ADX)
# ============================================================
def detect_regime(df: pd.DataFrame) -> str:
    if len(df)<50: return 'choppy'
    last=df.iloc[-1]
    hurst=float(last.get('hurst',0.5)) if not pd.isna(last.get('hurst',float('nan'))) else 0.5
    adx=float(last.get('adx',25))
    ewma_vol=float(last.get('ewma_vol',0.02))

    # HMM+GMM regime
    hmm_regime='choppy'
    if hmm_detector.fitted:
        feats=np.array([hurst,adx/100,ewma_vol,
                        float(last.get('vpin',0.5)),float(last.get('vol_ratio',1))])
        hmm_regime_raw=hmm_detector.predict(feats)
        if 'Bull' in hmm_regime_raw or 'LowVol' in hmm_regime_raw:
            hmm_regime='trending'
        elif 'Bear' in hmm_regime_raw:
            hmm_regime='trending'
        else:
            hmm_regime='mean_reverting' if hurst<0.4 else 'choppy'

    # RL regime classifier
    regime_feats=np.zeros(32)
    regime_feats[:8]=[hurst,adx/100,ewma_vol,float(last.get('vpin',0.5)),
                      float(last.get('vol_ratio',1)),float(last.get('macd_hist',0)),
                      float(last.get('rsi',50))/100,float(last.get('trend_strength',0.5))]
    rl_regime=rl_regime_classifier.predict(regime_feats)

    # Consensus
    if hurst>0.6 and adx>30:
        return 'trending'
    elif hurst<0.4:
        return 'mean_reverting'
    elif rl_regime!='choppy':
        return rl_regime
    return hmm_regime

# ============================================================
# SECTION 18: SIGNAL GENERATION (All 39 Ideas Integrated)
# ============================================================
def unidirectional_velocity(df: pd.DataFrame, n_fast: int = None,
                            n_slow: int = None, lb: int = None):
    """Efficiency-ratio-weighted price velocity for the latest bar.

    Returns (v_uni, v_norm, ER, accel):
      v_uni  = (net_window_return / n) × ER         — chop-discounted per-bar velocity
      ER     = |net| / Σ|per-bar move| ∈ [0,1]       — 1=clean ramp, ~0=chop (Kaufman)
      v_norm = v_uni / per-bar-vol(lb)               — regime-normalised (vol units)
      accel  = v_uni(n_fast) − v_uni(n_slow)         — momentum building (>0) or fading
    Empirically verified on the SOL 1m substrate (scripts/verify_velocity_scalp.py):
    ER-weighting lifts directional hit-rate to 0.567 on strong-|v_norm| bars and
    beats raw velocity. Used as a bounded timing overlay + momentum-death exit.
    """
    try:
        n_fast = int(n_fast or CONFIG.get('VELO_N_FAST', 13))
        n_slow = int(n_slow or CONFIG.get('VELO_N_SLOW', 26))
        lb = int(lb or CONFIG.get('VELO_LOOKBACK', 20))
        c = df['close'].to_numpy(dtype=float)
        if len(c) < n_slow + lb + 2:
            return 0.0, 0.0, 0.0, 0.0

        def _vel(P, n):
            seg = P[-(n + 1):]
            if seg[0] <= 0:
                return 0.0, 0.0
            net = (seg[-1] - seg[0]) / seg[0]
            r = np.diff(seg) / seg[:-1]
            path = float(np.sum(np.abs(r)))
            er = min(abs(net) / path, 1.0) if path > 1e-12 else 0.0
            return (net / n) * er, er

        vu_f, er_f = _vel(c, n_fast)
        vu_s, _ = _vel(c, n_slow)
        r_lb = np.diff(c[-(lb + 1):]) / c[-(lb + 1):-1]
        sig = float(np.std(r_lb))
        v_norm = (vu_f / sig) if sig > 1e-12 else 0.0
        accel = vu_f - vu_s
        if not all(np.isfinite(x) for x in (vu_f, v_norm, er_f, accel)):
            return 0.0, 0.0, 0.0, 0.0
        return float(vu_f), float(v_norm), float(er_f), float(accel)
    except Exception:
        return 0.0, 0.0, 0.0, 0.0


async def generate_trading_signal(df: pd.DataFrame) -> Tuple[bool, float, str, Dict]:
    if len(df)<100: return False, 0.5, 'hold', {}

    state.benchmarks['total_signal_checks']+=1
    last=df.iloc[-1]; prev=df.iloc[-2]
    regime=detect_regime(df); state.current_regime=regime

    hurst=float(last.get('hurst',0.5)) if not pd.isna(last.get('hurst',float('nan'))) else 0.5
    ewma_vol=float(last.get('ewma_vol',0.02))
    # Stash regime state for the blended dynamic-pessimistic sizing multiplier,
    # which runs later in execute_trade() where these aren't in scope.
    quant_overlay.last_hurst = hurst
    quant_overlay.last_ewma_vol = ewma_vol
    # Anchor the adaptive confidence gate on the ACTIVE profile's ceiling, then
    # let regime persistence (hurst) and volatility RELAX it toward the floor.
    # (Previously hardcoded 0.65 — a pre-aggressive-profile constant that, after
    # the small subtractions below, always exceeded the 0.58 cap and clamped,
    # making the restored aggressive band [0.52,0.58] dead. Anchoring on the
    # profile MAX restores genuine regime-adaptive thresholding within the band.)
    conf_base = float(CONFIG['CONFIDENCE_THRESHOLD_MAX'])
    dynamic_conf=conf_base-(0.05*(1-hurst))-(0.03*(ewma_vol/0.03))
    # Adaptive offset: tightens by +0.01 per consecutive loss (max +0.06),
    # loosens by -0.005 per 3 consecutive wins (max -0.02).
    dynamic_conf += getattr(state, 'adaptive_conf_offset', 0.0)
    # adaptive floor from the conf-floor optimizer (≥ profile MIN; only raises selectivity)
    _opt_floor = max(float(CONFIG['CONFIDENCE_THRESHOLD_MIN']), float(getattr(state, 'opt_conf_floor', 0.0) or 0.0))
    CONF=max(_opt_floor,min(CONFIG['CONFIDENCE_THRESHOLD_MAX'],dynamic_conf))

    # Macro circuit breaker
    dxy=float(last.get('dxy',0)); vix_val=float(last.get('vix',0))
    if (dxy>CONFIG['DXY_MAX'] or vix_val>CONFIG['VIX_MAX']) and dxy>0:
        state.benchmarks['circuit_breaker_count']+=1
        logger.info(f"Macro circuit breaker: DXY={dxy:.1f} VIX={vix_val:.1f}")
        return False, 0.5, 'hold', {'regime':regime,'circuit_breaker':True}

    details={'regime':regime,'hurst':hurst,'threshold':CONF,'filters_passed':[],'filters_failed':[]}

    def chk(name, cond):
        (details['filters_passed'] if cond else details['filters_failed']).append(name)
        return cond

    # Core filter chain (9 filters, require 6/9)
    # Thresholds relaxed from the original tuning (built for high-vol periods).
    # SOL in low-vol/low-volume regimes was clearing only 2/9 filters; we now
    # accept weaker confirmations and rely on confidence + EV gates for quality.
    fib_ok  = chk('fib_confluence', bool(last.get('fib_confluence',0)))
    vp_ok   = chk('volume_profile', abs(float(last.get('price_vs_poc',0)))<float(last.get('atr',1))*2)
    ofi_ok  = chk('order_flow', abs(float(last.get('ofi',0)))>abs(float(last.get('ofi_ma',0)))*0.3)
    iv_ok   = chk('inst_volume', bool(last.get('inst_vol_trend',0)))
    bk_ok   = chk('boll_keltner', abs(float(last.get('bk_pressure',0)))>0.15)
    adx_ok  = chk('adx_trend', float(last.get('adx',0))>15)
    vol_ok  = chk('volume_spike', float(last.get('vol_ratio',1))>1.05)
    vpin_ok = chk('vpin', float(last.get('vpin',0.5))>0.50)
    atr_ok  = chk('atr_range', 0.001<float(last.get('atr_pct',0.02))<0.10)
    filters_passed=sum([fib_ok,vp_ok,ofi_ok,iv_ok,bk_ok,adx_ok,vol_ok,vpin_ok,atr_ok])

    # Update filter pass-rate metric on EVERY check (running average).
    n = state.benchmarks['total_signal_checks']
    prev_rate = state.benchmarks['filters_pass_rate']
    state.benchmarks['filters_pass_rate'] = (prev_rate*(n-1) + filters_passed/9.0)/max(n,1)

    # === Quant overlay: Bayesian evidence aggregation ===
    # Replaces brittle "N-out-of-9" voting with a proper Naive-Bayes log-odds
    # update where each filter contributes its likelihood ratio. This unlocks
    # trades in low-volatility / low-confluence regimes where the binary count
    # would never reach 6/9 even when 2-3 strong filters firmly indicate edge.
    posterior, log_odds = quant_overlay.bayesian_posterior(
        details['filters_passed'], details['filters_failed']
    )
    gate = REGIME_POSTERIOR_GATE.get(regime, 0.60)
    details['posterior']    = float(posterior)
    details['log_odds']     = float(log_odds)
    details['posterior_gate'] = gate

    # Sanity floor — never trade on zero or one filter regardless of posterior
    if filters_passed < CONFIG.get('BAYES_MIN_FILTERS', 2) or posterior < gate:
        quant_overlay.bayes_blocks += 1
        if n % 30 == 0:
            logger.info(f"  bayes block | filters {filters_passed}/9 | "
                        f"post={posterior:.3f}<gate={gate:.2f} ({regime}) | "
                        f"pass:{details['filters_passed']}")
        return False, 0.5, 'hold', details
    quant_overlay.bayes_passes += 1
    # DEBUG-level to keep INFO log small (was ~20 MB/day at INFO).
    logger.debug(f"  bayes PASS | filters {filters_passed}/9 | "
                 f"post={posterior:.3f} | log_odds={log_odds:+.2f} | regime={regime}")

    # ML Ensemble
    fc=[c for c in FEATURE_COLS if c in df.columns]
    X_last=df[fc].fillna(0).replace([np.inf,-np.inf],0).values[-1:].astype(float)
    ml_probs=ml_ensemble.predict_proba(X_last)
    ml_short,ml_hold,ml_long=ml_probs[0]
    # live directional model confidence — source for the open-position adjusted-confidence
    # readout (DISPLAY+LOG only; computed here where the proba already exists → zero extra cost)
    state.live_model_proba={'long':float(ml_long),'short':float(ml_short),
                            'hold':float(ml_hold),'ts':time.time()}

    # Deep Learning (alternate LSTM/Transformer)
    use_tf=(state.trade_count%2==0)
    X_seq=df[fc].fillna(0).replace([np.inf,-np.inf],0).values.astype(float)
    dl_probs,unc=dl_manager.predict(X_seq,use_transformer=use_tf)
    dl_short,dl_hold,dl_long=dl_probs
    details['_ml_probs'] = {'ml_long': float(ml_long), 'ml_hold': float(ml_hold),
                            'ml_short': float(ml_short), 'dl_long': float(dl_long),
                            'dl_hold': float(dl_hold), 'dl_short': float(dl_short)}
    # Adaptive uncertainty gate — looser before first ML retrain has happened.
    unc_thr = (CONFIG['UNCERTAINTY_THRESHOLD_WARMUP']
               if state.benchmarks['ml_retrain_count'] < 3
               else CONFIG['UNCERTAINTY_THRESHOLD'])
    details['uncertainty']=float(unc); details['unc_threshold']=unc_thr
    if unc>unc_thr:
        if n % 30 == 0:
            logger.info(f"  [reject] uncertainty unc={unc:.3f} > thr={unc_thr:.3f}")
        return False, 0.5, 'hold', details

    # Composite score — when ML is overfit to hold (ml_hold > 0.90 and ML is
    # therefore uninformative), fall back to DL-only weights AND use dl_hold
    # for the hold comparator. Otherwise blend both.
    if ml_hold > 0.90:
        ens_long  = dl_long
        ens_short = dl_short
        ens_hold  = dl_hold
        details['ensemble_mode'] = 'dl_only'
    else:
        ens_long  = 0.6*ml_long  + 0.4*dl_long
        ens_short = 0.6*ml_short + 0.4*dl_short
        ens_hold  = 0.6*ml_hold  + 0.4*dl_hold
        details['ensemble_mode'] = 'blend'

    galaxy=float(last.get('galaxy_score',50))
    soc_vol=float(last.get('social_volume',0))
    soc_boost=0.02 if galaxy>60 and soc_vol>1000 else 0.0
    poly_yes=float(last.get('polymarket_yes',0.5))
    poly_boost=0.02 if poly_yes>0.65 else (-0.02 if poly_yes<0.35 else 0.0)
    funding=float(last.get('funding_rate',0))
    fund_adj_l=-0.03 if funding>0.001 else (0.02 if funding<-0.001 else 0)
    fund_adj_s=0.03 if funding>0.001 else (-0.02 if funding<-0.001 else 0)
    liq_cascade=float(last.get('liq_usd',0))>500000 or int(last.get('liq_count',0))>20
    if liq_cascade: ens_long*=0.8; ens_short*=0.8
    if bool(last.get('whale_cluster',0)):
        if ens_long>ens_short: ens_long+=0.03
        else: ens_short+=0.03
    if float(last.get('granger_social',0.5))>0.7: soc_boost+=0.01
    if float(last.get('nft_activity',0.5))>0.7: ens_long+=0.01

    # Sentiment boost (combined)
    sent_adj=0.015*state.sentiment_score

    # Historic matching boost (DTW/Wasserstein)
    hist_score=0.0
    if len(state.historic_states)>10 and len(X_seq)>4:
        _hist_cols = [c for c in FEATURE_COLS if c in df.columns]
        _hw = _historic_weights(_hist_cols)   # static substrate prior (no aligned fwd here)
        hist_score=historic_match_score(X_seq[-1],state.historic_states,weights=_hw)*0.02

    if ens_long>ens_short and ens_long>ens_hold:
        action='long'
        raw_conf=ens_long+soc_boost+poly_boost+fund_adj_l+sent_adj+hist_score
    elif ens_short>ens_long and ens_short>ens_hold:
        action='short'
        raw_conf=ens_short-soc_boost-poly_boost+fund_adj_s-sent_adj+hist_score
    else:
        # ML/DL both predict hold — overlay fallback. When the elite stack
        # (MTF + funding-z + crowded + oracle-mark + doppelganger + cointeg)
        # points the same direction with conviction, take a small position.
        # Confidence is capped tightly; pessimistic-Kelly sizes it small.
        mtf_v   = quant_overlay.last_mtf_alignment
        fz_v    = quant_overlay.last_funding_z
        crowd_v = quant_overlay.last_crowded_score
        oms_v   = quant_overlay.last_oracle_mark_spread
        doppel_v = quant_overlay.last_doppel_mean
        cointeg_v = quant_overlay.last_cointeg_z
        overlay_score = (
            CONFIG.get('OVERLAY_W_MTF', 0.30) * mtf_v
          + CONFIG.get('OVERLAY_W_FUNDZ', 0.20) * (1.0 if fz_v < -1.5 else (-1.0 if fz_v > 1.5 else 0.0))
          - CONFIG.get('OVERLAY_W_CROWD', 0.15) * np.sign(crowd_v)
          + CONFIG.get('OVERLAY_W_OMS', 0.10) * np.sign(oms_v)
          + CONFIG.get('OVERLAY_W_DOPPEL', 0.15) * np.tanh(doppel_v * 100)
          - CONFIG.get('OVERLAY_W_COINTEG', 0.10) * np.tanh(cointeg_v / 2.0)
        )
        # Dynamic override bar: demand a STRONGER overlay score when BOTH models
        # are near-unanimously flat (they have looked at the live substrate and
        # see no edge). When either model carries directional conviction (low
        # hold prob), keep the normal BASE bar so genuine signals still pass.
        fb_thresh = CONFIG.get('FALLBACK_BASE_SCORE', 0.25)
        if CONFIG.get('FALLBACK_MODEL_VETO_ENABLE', True):
            flat_conv = min(float(ml_hold), float(dl_hold))
            if flat_conv >= CONFIG.get('FALLBACK_FLAT_CONVICTION', 0.85):
                fb_thresh = CONFIG.get('FALLBACK_STRONG_SCORE', 0.45)
        details['fallback_threshold'] = float(fb_thresh)
        if abs(overlay_score) >= fb_thresh:
            action = 'long' if overlay_score > 0 else 'short'
            raw_conf = min(0.60, 0.52 + 0.10 * abs(overlay_score))
            details['fallback_overlay_score'] = float(overlay_score)
            details['fallback'] = True   # model-independent path → conformal gate skips it
            if n % 20 == 0:
                logger.info(f"  [fallback] overlays say {action} (score={overlay_score:+.2f} "
                            f"thr={fb_thresh:.2f}) ml_hold={ml_hold:.3f} dl_hold={dl_hold:.3f}")
        else:
            if n % 30 == 0:
                logger.info(f"  [reject] overlay<thr | score={overlay_score:+.2f} "
                            f"thr={fb_thresh:.2f} | "
                            f"ml(L/H/S)={ml_long:.3f}/{ml_hold:.3f}/{ml_short:.3f} "
                            f"dl(L/H/S)={dl_long:.3f}/{dl_hold:.3f}/{dl_short:.3f}")
            return False, float(ml_hold), 'hold', details

    # === Split-conformal credibility gate (Phase 2) ==============================
    # Act on a DIRECTIONAL class only when its calibrated p-value clears alpha. Soft
    # mode haircuts low-credibility calls; hard mode abstains. Skips the model-
    # v8 Step-0 calibration instrumentation: persist the raw meta-ensemble probability for the
    # CHOSEN direction (the number calibration will correct). Non-outcome (logging only) — pairs
    # with the closed trade's win/loss to build the ECE/reliability curve for Phase A.
    if action in ('long', 'short'):
        try:
            details['model_proba'] = float(ml_probs[0][2 if action == 'long' else 0])
        except Exception:
            pass
    # independent fallback path (no calibrated class proba there).
    if (CONFIG.get('CONFORMAL_ENABLE', True) and action in ('long', 'short')
            and not details.get('fallback')):
        _ccls = 2 if action == 'long' else 0
        if _ccls in CONFIG.get('CONFORMAL_CLASSES', [0, 2]):
            _pc = ml_ensemble.conformal_pvalue(ml_probs[0], _ccls)
            details['conformal_p'] = float(_pc)
            _alpha = float(CONFIG.get('CONFORMAL_ALPHA', 0.15))
            if _pc <= _alpha:
                if CONFIG.get('CONFORMAL_MODE', 'soft') == 'hard':
                    if n % 30 == 0:
                        logger.info(f"  [reject] conformal p={_pc:.3f}<=a={_alpha} "
                                    f"({action}) — class not credible")
                    return False, float(np.clip(raw_conf, 0, 1)), 'hold', details
                raw_conf -= float(CONFIG.get('CONFORMAL_SOFT_HAIRCUT', 0.06))
            else:
                raw_conf += min(0.02, 0.04 * (_pc - _alpha))

    # === Elite quant overlays applied to raw confidence ===
    # Snapshot the pre-overlay confidence so we can soft-cap total adjustment
    # at ±0.12 (mitigates risk of additive boosts stacking artificially).
    raw_conf_pre_overlay = raw_conf

    # 1. BOCPD changepoint: confirmed hard halt only on (a) p≥0.95 OR (b) 2-of-5
    # recent bars > 0.85. Below that, apply a graded soft haircut to confidence
    # so regime-break information isn't binary-thrown-away.
    bocpd_p = quant_overlay.last_bocpd_p
    details['bocpd_p'] = bocpd_p
    if quant_overlay.bocpd_confirmed_halt(
            hard_threshold=CONFIG.get('BOCPD_HARD_P',   0.95),
            spike_threshold=CONFIG.get('BOCPD_SPIKE_P', 0.85),
            required_spikes=CONFIG.get('BOCPD_SPIKES_REQ', 2)):
        quant_overlay.bocpd_halts += 1
        if n % 30 == 0:
            logger.info(f"  [reject] BOCPD confirmed halt | p={bocpd_p:.3f} "
                        f"history={[round(x,2) for x in quant_overlay.bocpd_history]}")
        return False, float(np.clip(raw_conf,0,1)), 'hold', details
    if bocpd_p > 0.65:
        # Soft haircut tier: scale confidence down proportionally
        raw_conf *= (1.0 - 0.15 * (bocpd_p - 0.65))

    # 2. Crowded-trade contrarian boost (Soros-style positioning edge)
    crowded = quant_overlay.last_crowded_score
    details['crowded'] = crowded
    if action == 'long' and crowded < -0.4:
        raw_conf += 0.025   # squeezed shorts = long edge
    elif action == 'short' and crowded > 0.4:
        raw_conf += 0.025   # crowded longs = short edge
    elif action == 'long' and crowded > 0.6:
        raw_conf -= 0.02    # already-crowded longs = avoid joining
    elif action == 'short' and crowded < -0.6:
        raw_conf -= 0.02

    # 3. Reflexivity dampener — high feedback-loop = late-stage move
    refx = quant_overlay.last_reflexivity
    details['reflexivity'] = refx
    if abs(refx) > 0.7:
        raw_conf *= 0.92   # 8% confidence haircut on reflexive trades

    # 4. Doppelganger forward-return alignment
    try:
        if 'returns' in df.columns and len(df) > 60:
            fc_cols = [c for c in FEATURE_COLS[:20] if c in df.columns]
            if fc_cols:
                hist = df[fc_cols].fillna(0).replace([np.inf,-np.inf],0).values[-200:-10]
                hist_fwd = df['close'].pct_change(5).shift(-5).fillna(0).values[-200:-10]
                cur = df[fc_cols].fillna(0).replace([np.inf,-np.inf],0).values[-1]
                # Dynamic empirical weights: blend the static substrate prior with a
                # live |Spearman(feature, fwd_return)| over THIS rolling window so the
                # k-NN match emphasises features that predict here-and-now.
                _hw_d = _historic_weights(fc_cols, hist=hist, hist_fwd=hist_fwd)
                d_mean, d_std = quant_overlay.doppelganger_forecast(cur, hist, hist_fwd, weights=_hw_d)
                details['doppel_mean'] = d_mean; details['doppel_std'] = d_std
                if action == 'long' and d_mean > 0.001:
                    raw_conf += 0.015
                elif action == 'short' and d_mean < -0.001:
                    raw_conf += 0.015
                elif (action == 'long' and d_mean < -0.001) or (action == 'short' and d_mean > 0.001):
                    raw_conf -= 0.02
    except Exception:
        pass

    # 5. Meta-label win-rate gate — historically losing (regime, action) combos
    mlwr = quant_overlay.metalabel_winrate(regime, action)
    details['metalabel_winrate'] = mlwr
    if mlwr < 0.35:
        if n % 30 == 0:
            logger.info(f"  [reject] meta-label WR={mlwr:.2%} for ({regime},{action})")
        return False, float(np.clip(raw_conf,0,1)), 'hold', details
    # Scale confidence by meta-label edge (centered at 0.5 ⇒ no change)
    raw_conf *= (0.8 + 0.4 * mlwr)

    # 6. Information Coefficient haircut — if recent signals are decaying
    ic = quant_overlay.last_ic
    details['info_coefficient'] = ic
    if ic < -0.1:
        raw_conf *= 0.90

    # === ROUND-3 ELITE OVERLAYS ===
    # 7. Multi-timeframe confluence — directional agreement boosts confidence
    mtf = quant_overlay.last_mtf_alignment
    details['mtf_alignment'] = mtf
    if action == 'long' and mtf >= 0.66:    raw_conf += 0.020
    elif action == 'short' and mtf <= -0.66: raw_conf += 0.020
    elif action == 'long' and mtf <= -0.33:  raw_conf -= 0.025
    elif action == 'short' and mtf >= 0.33:  raw_conf -= 0.025

    # 8. Funding-rate z-score reversal — crowded carry = contrarian edge
    fz = quant_overlay.last_funding_z
    details['funding_z'] = fz
    if action == 'short' and fz > 2.0:   raw_conf += 0.03    # funding pos, longs crowded
    elif action == 'long' and fz < -2.0: raw_conf += 0.03    # funding neg, shorts crowded
    elif action == 'long' and fz > 2.5:  raw_conf -= 0.025
    elif action == 'short' and fz < -2.5: raw_conf -= 0.025

    # 9. Oracle-Mark spread (Drift-native arbitrage edge).
    # Reduced magnitude: without a true L2 mark (dlob endpoint dead) we proxy
    # mark with Kraken's last close vs Drift's Pyth oracle. The directional
    # interpretation is venue-leader-dependent — small, conservative tilt only.
    oms = quant_overlay.last_oracle_mark_spread
    details['oracle_mark_spread'] = oms
    if action == 'short' and oms > 0.003:   raw_conf += 0.008
    elif action == 'long' and oms < -0.003: raw_conf += 0.008
    elif action == 'long' and oms > 0.005:  raw_conf -= 0.005
    elif action == 'short' and oms < -0.005: raw_conf -= 0.005

    # 10. OFI persistence — momentum vs. mean-reversion regime alignment
    ofi_p = quant_overlay.last_ofi_persistence
    details['ofi_persistence'] = ofi_p
    if regime == 'trending' and ofi_p > 0.3:        raw_conf += 0.015
    elif regime == 'mean_reverting' and ofi_p < -0.1: raw_conf += 0.015
    elif regime == 'trending' and ofi_p < -0.2:     raw_conf -= 0.015

    # 11. CUSUM event presence — sampling filter (López de Prado).
    # No meaningful cumulative move ⇒ likely noise; small haircut.
    cusum_on = quant_overlay.last_cusum_event
    details['cusum_event'] = cusum_on
    if cusum_on:
        raw_conf += 0.01
    else:
        raw_conf -= 0.015
        quant_overlay.cusum_blocks += 1

    # 12. Unidirectional-velocity scalp timing (verified: ER-weighted velocity →
    # +6.7pp directional edge at the strong-|v_norm| quantile; net-positive at
    # maker fees). Rewards entries aligned with CLEAN, efficient directional
    # momentum (scaled by magnitude × efficiency, + a small acceleration bonus);
    # haircuts entries fighting clean opposite velocity. Bounded by the soft cap.
    if CONFIG.get('VELO_NUDGE_ENABLE', True):
        vu, vn, er, acc = unidirectional_velocity(df)
        details['velocity'] = {'v_uni': round(vu, 6), 'v_norm': round(vn, 4),
                               'er': round(er, 3), 'accel': round(acc, 6)}
        if action in ('long', 'short') and np.isfinite(vn) and abs(vn) > 1e-9:
            aligned = (vu > 0) if action == 'long' else (vu < 0)
            # Hard falling-knife veto: refuse to OPEN directly into a strong, clean,
            # efficient ADVERSE momentum move (entry-side mirror of the #13 exit).
            if (CONFIG.get('VELO_ENTRY_VETO_ENABLE', True) and not aligned
                    and abs(vn) >= CONFIG.get('VELO_ENTRY_VETO_ZNORM', 1.00)
                    and er    >= CONFIG.get('VELO_ENTRY_VETO_ER',    0.45)):
                quant_overlay.velo_entry_vetoes = getattr(
                    quant_overlay, 'velo_entry_vetoes', 0) + 1
                if n % 30 == 0:
                    logger.info(f"  [reject] velocity-veto | {action} into "
                                f"v_norm={vn:+.2f} ER={er:.2f} (clean adverse momentum)")
                return False, float(np.clip(raw_conf, 0, 1)), 'hold', details
            mag = min(abs(vn), CONFIG['VELO_NORM_CAP']) / CONFIG['VELO_NORM_CAP']
            bump = CONFIG['VELO_CONF_MAX'] * mag * er
            accel_aligned = (acc >= 0) if action == 'long' else (acc <= 0)
            if aligned:
                raw_conf += bump + (CONFIG['VELO_ACCEL_BONUS'] if accel_aligned else 0.0)
            else:
                raw_conf -= bump

    # 13. Vector-DB regime prior (Aster MCP): cosine-match the live 58-dim feature
    # vector against 616 historical SOL states → similarity-weighted, direction-
    # aligned realized-outcome prior. Bounded nudge; needs ≥3 usable neighbours.
    if (nox_bridge is not None and CONFIG.get('VECTORDB_OVERLAY_ENABLE', True)
            and action in ('long', 'short')):
        try:
            _vres = nox_bridge.query_similar_states(
                X_last[0].tolist(), k=int(CONFIG.get('VECTORDB_K', 8)))
            _vp = nox_bridge.regime_prior_from_matches(_vres, action)
            details['vectordb'] = {'n': _vp['n'], 'agree': _vp['agree_frac'],
                                   'prior': _vp['prior']}
            if _vp['n'] >= 3:
                _vmax = float(CONFIG.get('VECTORDB_OVERLAY_MAX', 0.03))
                raw_conf += float(np.clip(_vp['prior'] * _vmax, -_vmax, _vmax))
        except Exception:
            pass

    # 14. MLX fast on-device inference (Aster MCP): a small bounded directional nudge
    # from a real mlx.core forward pass (complements the Ollama reasoning veto).
    if (nox_bridge is not None and CONFIG.get('MLX_OVERLAY_ENABLE', True)
            and action in ('long', 'short')):
        try:
            _ml = nox_bridge.mlx_infer(X_last[0].tolist())
            if _ml:
                details['mlx'] = {'backend': _ml.get('backend'),
                                  'argmax': _ml.get('argmax'), 'signal': _ml.get('signal')}
                _mm = float(CONFIG.get('MLX_OVERLAY_MAX', 0.015))
                _am = _ml.get('argmax')
                if _am == 2:                       # long
                    raw_conf += _mm if action == 'long' else -_mm
                elif _am == 0:                     # short
                    raw_conf += _mm if action == 'short' else -_mm
                elif _ml.get('signal') == action:  # heuristic-fallback agreement
                    raw_conf += _mm * 0.5
        except Exception:
            pass

    # Soft cap on total overlay adjustment (mitigates stacking risk).
    overlay_delta = raw_conf - raw_conf_pre_overlay
    overlay_delta = float(np.clip(overlay_delta, -0.12, 0.12))
    raw_conf = raw_conf_pre_overlay + overlay_delta
    details['overlay_delta'] = overlay_delta

    confidence=float(np.clip(raw_conf,0,1))

    # Record this forecast for IC tracking — realized side filled next bar
    quant_overlay.ic_forecasts.append(confidence - 0.5)

    # ---- HOOK A: local-AI second opinion (candidate trade only) ------------------
    # Independent confirmer/veto + bounded asymmetric fusion (≤+0.03 / ≤-0.08). Runs
    # AFTER the ±0.12 overlay cap is locked, in its own budget; BEFORE the EV gate so
    # a veto/haircut aborts cleanly through the existing machinery. Degrades silently
    # (ok=False ⇒ pass-through) when Ollama is down/slow. Never touches leverage/size.
    if local_ai is not None and action in ('long', 'short'):
        try:
            _snap = build_ai_snapshot(action, confidence, regime, last, details,
                                      state, quant_overlay, threshold=CONF)
            _av = await local_ai.evaluate(_snap)
            details['ai'] = _av.as_log_dict()
            _cont, confidence, action, _vetoed = apply_ai_to_signal(
                confidence, action, _av, CONFIG)
            confidence = float(np.clip(confidence, 0.0, 1.0))
            details['confidence_post_ai'] = confidence
            if _vetoed:
                if n % 20 == 0:
                    logger.info(f"  [reject] LOCAL-AI veto | engine={_snap['engine']['action']} "
                                f"ai={_av.direction} c={_av.confidence:.2f} "
                                f"why={_av.rationale[:70]}")
                return False, confidence, 'hold', details
        except Exception as _ai_err:
            details['ai_error'] = str(_ai_err)[:120]

    # ---- HOOK A2: AI data-augmentation haircut (down-only) ------------------------
    # Consumes the background loop's cached data-quality verdict (state.ai_augmentation) —
    # NO live model call here, so zero latency tax on the trade path. Shrinks confidence when
    # the AI judges the CURRENT market data low-quality/inconsistent; never raises it. Gated.
    if action in ('long', 'short') and CONFIG.get('LOCAL_AI_AUGMENT_APPLY_HAIRCUT', True):
        _aug = getattr(state, 'ai_augmentation', None) or {}
        if _aug.get('ok') and not _aug.get('stale'):
            try:
                _factor = augmentation_factor(_aug, CONFIG)
                if _factor < 1.0:
                    confidence = float(np.clip(confidence * _factor, 0.0, 1.0))
                    details['aug_factor'] = round(_factor, 4)
                    details['confidence_post_aug'] = round(confidence, 4)
            except Exception:
                pass

    # EV filter
    atr=float(last.get('atr',1.0)); atr_pct=float(last.get('atr_pct',0.02))
    sl_m,tp_m,rl_s,rl_a=rl_stop_optimizer.get_sl_tp_multipliers(
        hurst, ewma_vol, regime, atr_pct, float(last.get('mom',0)))
    ev_ratio=(confidence*atr*tp_m)/((1-confidence)*atr*sl_m+1e-10)
    if ev_ratio<CONFIG['MIN_EV_RATIO']:
        if n % 30 == 0:
            logger.info(f"  [reject] EV ratio {ev_ratio:.2f} < min {CONFIG['MIN_EV_RATIO']} "
                        f"| conf={confidence:.3f} | sl_m={sl_m:.2f} tp_m={tp_m:.2f}")
        return False, confidence, 'hold', details

    details.update({'action':action,'confidence':confidence,'ev_ratio':float(ev_ratio),
                    'ml_long':float(ml_long),'ml_short':float(ml_short),
                    'dl_long':float(dl_long),'dl_short':float(dl_short),
                    'uncertainty':float(unc),'sl_mult':sl_m,'tp_mult':tp_m,
                    'rl_state':rl_s,'rl_action':rl_a,'funding_rate':funding,
                    'filters_passed_count':filters_passed,'liq_cascade':liq_cascade})

    # Running avg confidence on every check that gets this far
    n = state.benchmarks['total_signal_checks']
    prev_conf = state.benchmarks['avg_confidence']
    state.benchmarks['avg_confidence'] = (prev_conf*(n-1) + confidence)/max(n,1)

    should_trade=confidence>=CONF
    if should_trade:
        state.benchmarks['signals_generated']+=1

    logger.info(f"Signal: {action} | Conf:{confidence:.3f} | EV:{ev_ratio:.2f} | "
                f"Regime:{regime} | Filters:{filters_passed}/9 | unc={unc:.2f}")
    return should_trade, confidence, action, details

# ============================================================
# SECTION 19: POSITION SIZING (Kelly + Risk Parity)
# ============================================================
def calculate_position_size(balance, confidence, atr, price, leverage, sl_mult, regime,
                            calibration: float = 1.0):
    risk_usd=balance*CONFIG['RISK_PER_TRADE']
    lp=1-confidence
    # Position-sizing policy (CONFIG['SIZING_MODE']; default 'pessimistic').
    # Downstream vol_scale, reg_mult, MAX_POSITION_PCT and the CVaR cap apply as
    # guards in EVERY mode — the selector only chooses the base Kelly fraction.
    sizing_mode = CONFIG.get('SIZING_MODE', 'pessimistic')
    if sizing_mode == 'dynamic_half_kelly':
        # Grok-style on empirical p/b. garch_vol×price ≈ a price-scale ATR
        # baseline; last realised return is the velocity proxy.
        atr_avg = quant_overlay.garch_vol() * max(price, 1e-6)
        vel = quant_overlay.returns[-1] if len(quant_overlay.returns) else 0.0
        kelly = float(min(quant_overlay.dynamic_half_kelly(
            confidence, atr=atr, atr_avg=atr_avg, velocity=vel), 0.25))
    elif sizing_mode == 'mcp_venue_routing' and nox_bridge is not None:
        # MANDATED FLOW (coding-standard 02): route the base fraction through the Aster
        # MCP pessimistic_dynamic_half_kelly_with_venue_routing tool. Engine guards
        # (vol_scale / reg_mult / streak-throttle / MAX_POSITION_PCT / CVaR) still apply.
        _tl = [t for t in (getattr(state, 'trade_log', None)
                           or getattr(state, 'trade_log_tail', []) or []) if isinstance(t, dict)][-50:]
        _pnls = [float(t.get('pnl_usd', t.get('pnl', 0)) or 0) for t in _tl]
        _wins = [p for p in _pnls if p > 0]; _losses = [-p for p in _pnls if p < 0]
        _wr = (len(_wins) / len(_pnls)) if _pnls else 0.5
        _aw = (sum(_wins) / len(_wins)) if _wins else 1.0
        _al = (sum(_losses) / len(_losses)) if _losses else 1.0
        _rec = nox_bridge.size_with_venue_routing(
            win_rate=_wr, avg_win=_aw, avg_loss=_al, balance=balance, confidence=confidence,
            hurst=float(getattr(state, 'last_hurst', 0.5)),
            ewma_vol=float(quant_overlay.garch_vol() or 0.02), regime=regime,
            n_trades=int(getattr(state, 'trade_count', 0)),
            max_leverage=int(leverage), default_leverage=int(CONFIG.get('LEVERAGE_CAP', 20)),
            atr_pct=(float(atr / price) if price else None),
            velocity=(quant_overlay.returns[-1] if len(quant_overlay.returns) else 0.0))
        if _rec and _rec.get('sizing'):
            kelly = float(np.clip(_rec['sizing']['final_fraction'], 0.02, 0.25))
            state.last_mcp_venue = _rec['recommendation'].get('venue')
        else:
            kelly = float(quant_overlay.pessimistic_kelly())   # graceful fallback
    elif sizing_mode == 'dynamic_pessimistic':
        # Pessimistic bootstrap base × bounded two-sided regime multiplier.
        kelly = float(min(quant_overlay.dynamic_pessimistic_kelly(
            confidence, regime=regime), 0.25))
    else:
        # Pessimistic-Kelly: bootstrap CI lower bound on win-rate, blended with
        # confidence-derived analytic Kelly. "Better to underbet" (Thorp).
        pess_k  = quant_overlay.pessimistic_kelly()
        analyt_k = float(np.clip(confidence-(lp/(confidence/(1-confidence+1e-10)+1e-10)),0.01,0.5))
        kelly   = min(pess_k, analyt_k)   # take the MORE pessimistic of the two
    vol_scale=float(np.clip(0.02/max(atr/price,0.001),0.5,3.0))
    reg_mult={'trending':1.2,'choppy':0.7,'mean_reverting':0.9}.get(regime,1.0)
    # --- Losing-streak / drawdown de-risk throttle (anti-martingale; ≤1.0) ------
    # kelly, vol_scale and reg_mult all scale UP into a low-vol "trending" chop;
    # nothing above sees the loss streak or the drawdown. Fold in a bounded factor
    # that ONLY reduces size as losses cluster, floored so trades keep flowing and
    # self-healing on the next win (consecutive_losses → 0). Reads module globals
    # (same pattern as dynamic_trade_limit/dynamic_slippage_bps).
    risk_throttle = 1.0
    if CONFIG.get('STREAK_DERISK_ENABLE', True):
        streak = int(getattr(state, 'consecutive_losses', 0) or 0)
        peak   = float(getattr(quant_overlay, 'peak_balance', 0.0) or 0.0)
        if peak <= 0:
            peak = float(getattr(state, 'peak_equity', balance) or balance)
        dd = max(0.0, (peak - balance) / peak) if peak > 0 else 0.0
        streak_cut = float(CONFIG['STREAK_DERISK_PER_LOSS']) * max(0, streak - 1)
        dd_cut     = float(CONFIG['STREAK_DERISK_DD_COEF']) * dd
        risk_throttle = float(np.clip(
            1.0 - streak_cut - dd_cut, float(CONFIG['STREAK_DERISK_FLOOR']), 1.0))
    state.last_risk_throttle = risk_throttle  # always reflect reality (incl. disabled=1.0)
    adj_risk=min(risk_usd*kelly*vol_scale*reg_mult*risk_throttle,
                 balance*CONFIG['MAX_POSITION_PCT'])
    sl_pct=(atr*sl_mult)/price
    # Respect the caller's leverage CAP (paper-mode caps at 10x; real at default).
    # Then derive the SL-implied leverage and take the MINIMUM of the two.
    lev_cap   = max(1, int(leverage))
    # Confidence-scaled leverage: low-conviction → small $ risk (floor_frac of cap at the
    # conf floor, full cap at CONF_LEV_FULL_AT). The 40x cap + SL-lev + CVaR guards remain.
    if CONFIG.get('CONF_SCALED_LEV_ENABLE', True):
        _ff   = float(CONFIG.get('CONF_LEV_FLOOR_FRAC', 0.35))
        # v9 Phase A — size leverage by the CALIBRATED win-probability when the calibrator is fit
        # (raw conf is compressed/under-confident → it under-sizes true winners). Falls back to the
        # raw conf band until fit. The 0.5% loss-cap bounds the now-larger leverage on true winners.
        if CONFIG.get('CALIB_ENABLE', True) and getattr(state, 'calib_a', None) is not None:
            _cc = calibrate_confidence(confidence)
            _lo = float(CONFIG.get('CALIB_LO', 0.50)); _hi = float(CONFIG.get('CALIB_HI', 0.85))
            _excess = float(np.clip((_cc - _lo) / max(_hi - _lo, 1e-6), 0.0, 1.0))
        else:
            _c_lo = float(CONFIG['CONFIDENCE_THRESHOLD_MIN'])
            _c_hi = float(CONFIG.get('CONF_LEV_FULL_AT', 0.72))
            _excess = float(np.clip((confidence - _c_lo) / max(_c_hi - _c_lo, 1e-6), 0.0, 1.0))
        # v7 1a — calibration: scale the ABOVE-FLOOR leverage by the conformal credibility, so an
        # overconfident-but-low-credibility trade is sized DOWN toward the floor (the mechanism
        # behind the high-conf left-tail losses). calibration=1.0 ⇒ byte-identical legacy sizing.
        if CONFIG.get('CONF_LEV_USE_CALIBRATION', True):
            _cmin = float(CONFIG.get('CONF_LEV_CALIB_MIN', 0.25))
            _excess *= float(np.clip(calibration, _cmin, 1.0))
        _frac = _ff + (1.0 - _ff) * _excess
        lev_cap = max(1, int(round(lev_cap * _frac)))
    # v7 Phase 2 (gated) — news event-risk de-risk: cut leverage when a market-moving event is
    # unfolding. Composes multiplicatively AFTER the calibrated conf-scaled leverage (no double-
    # count). Active ONLY when LOCAL_AI_NEWS_APPLY=True (default False => byte-identical legacy).
    if CONFIG.get('LOCAL_AI_NEWS_APPLY', False):
        _ns = getattr(state, 'news_sentiment', {}) or {}
        if (_ns.get('ok') and not _ns.get('stale')
                and float(_ns.get('event_risk', 0.0) or 0.0) >= float(CONFIG.get('NEWS_DERISK_RISK', 0.60))):
            lev_cap = max(1, int(round(lev_cap * float(CONFIG.get('NEWS_DERISK_LEV_FACTOR', 0.50)))))
    sl_lev    = int(np.clip(min(1/(sl_pct*10), lev_cap), 1, lev_cap))
    actual_lev = sl_lev
    size_sol=max(adj_risk*actual_lev/price, CONFIG['MIN_POSITION_SIZE'])

    # Quant overlay: cap by CVaR-95 (Expected Shortfall) budget — take the
    # MIN of Kelly-sized and CVaR-sized notional. Never exceed either constraint.
    if CONFIG.get('USE_CVAR_SIZING', True):
        cvar_notional_usd = quant_overlay.cvar_position_notional(
            balance, confidence, actual_lev)
        if cvar_notional_usd > 0 and price > 0:
            cvar_size_sol = cvar_notional_usd / price
            size_sol = min(size_sol, cvar_size_sol)
            size_sol = max(size_sol, CONFIG['MIN_POSITION_SIZE'])
    # v7 1b + v10 FIX — hard per-trade $-loss cap. BUG (pre-v10): the cap omitted the LEVERAGE
    # factor, but notional = size_sol*price*lev (execute_trade) and loss ≈ notional*sl_pct — so the
    # cap was ~lev× too loose and effectively INERT (the 2.5% CVaR was the real cap; losses ran to
    # ~1% e.g. the −$0.996 trade @ $213 notional). FIX: bound the LEVERAGED loss
    # size_sol*price*lev*sl_pct ≤ balance*MAX_TRADE_LOSS_PCT. (LOSS_CAP_LEVERAGED=False = legacy.)
    _mtl = float(CONFIG.get('MAX_TRADE_LOSS_PCT', 0.0) or 0.0)
    if _mtl > 0 and sl_pct > 0 and price > 0:
        if CONFIG.get('LOSS_CAP_LEVERAGED', True):
            _max_size = (balance * _mtl) / (sl_pct * price * max(int(actual_lev), 1))
        else:
            _max_size = (balance * _mtl) / sl_pct / price        # legacy (inert) formula
        size_sol = max(min(size_sol, _max_size), CONFIG['MIN_POSITION_SIZE'])
    return float(size_sol), actual_lev


# ---- v4.00 Grok-derived: dynamic trade-limit governor (#17) ----
def dynamic_trade_limit() -> int:
    """Regime/vol-aware CAP on opens-per-cycle. Trending+calm => more trades
    allowed (toward MAX); choppy/volatile/high-vol => fewer (toward MIN). This
    bounds OVERTRADING (fee drag + adverse selection) without ever blocking the
    early trades of a cycle — it only bites once cadence is already high."""
    if not CONFIG.get('DYNAMIC_TRADE_LIMIT', True):
        return 10**9
    base = int(CONFIG['TRADE_LIMIT_BASE'])
    lo, hi = int(CONFIG['TRADE_LIMIT_MIN']), int(CONFIG['TRADE_LIMIT_MAX'])
    reg_adj = {'trending': 8, 'mean_reverting': 2,
               'choppy': -8, 'volatile': -10}.get(state.current_regime, 0)
    vol = quant_overlay.garch_vol()
    # each +1% GARCH vol above the 2% target trims ~4 trades (bounded at -8)
    vol_adj = -int(min(8.0, max(0.0, (vol - 0.02) / 0.01 * 4.0)))
    lim = int(np.clip(base + reg_adj + vol_adj, lo, hi))
    state.last_trade_limit = lim
    return lim


# ---- v4.00 Grok-derived: dynamic slippage model (#7) ----
def dynamic_slippage_bps(notional_usd: float, regime: str, vol: float,
                         depth_usd: Optional[float] = None) -> float:
    """sqrt market-impact slippage (bps): grows with order size vs available
    depth, widens with volatility and choppy/volatile regimes. Used to mark
    PAPER fills realistically (live fills use the venue's actual fill price)."""
    if not CONFIG.get('DYNAMIC_SLIPPAGE', True):
        return 0.0
    base = float(CONFIG['SLIPPAGE_BASE_BPS'])
    ref_depth = depth_usd if (depth_usd and depth_usd > 0) else float(CONFIG['SLIPPAGE_REF_DEPTH_USD'])
    impact = base * math.sqrt(max(notional_usd, 1.0) / max(ref_depth, 1.0))
    vol_mult = 1.0 + min(3.0, max(0.0, (vol - 0.02) / 0.02))
    reg_mult = {'volatile': 1.5, 'choppy': 1.2,
                'trending': 1.0, 'mean_reverting': 1.1}.get(regime, 1.0)
    bps = float(np.clip(base + impact * vol_mult * reg_mult, 0.0, float(CONFIG['SLIPPAGE_MAX_BPS'])))
    state.last_slippage_bps = bps
    return bps


# ============================================================
# SECTION 20: TRADE EXECUTION
# ============================================================
async def execute_trade(action, confidence, signal_details, df):
    async with state.lock:
        if state.halt_trading: return None
        # 24-h cycle gate — DEFENSE-IN-DEPTH check at the lowest level.
        if not can_open_new_trade(None):
            return None
        # Data-confidence circuit breaker (#38): if cross-venue mark divergence
        # says the price feed is unreliable, skip opening (don't trade on bad
        # data). Generous threshold => only fires on genuine feed dislocations.
        if CONFIG.get('DATA_CONFIDENCE_BREAKER', True):
            feed_div = abs(getattr(quant_overlay, 'last_oracle_mark_spread', 0.0))
            if feed_div > CONFIG['MAX_FEED_DIVERGENCE']:
                state.data_breaker_skips += 1
                if state.data_breaker_skips % 20 == 1:
                    logger.warning(f"  [data-confidence] cross-venue divergence "
                                   f"{feed_div:.2%} > {CONFIG['MAX_FEED_DIVERGENCE']:.2%}; "
                                   f"skipping open (feed unreliable)")
                return None
        # NoxSSD data-drive guard — if the designated data drive is not mounted/accessible
        # (debounced in noxssd_guard_loop), skip NEW opens; open positions keep being managed.
        if CONFIG.get('NOXSSD_GUARD_ENABLE', True) and getattr(state, 'noxssd_block_opens', False):
            state.noxssd_skips += 1
            if state.noxssd_skips % 20 == 1:
                logger.warning(f"  [noxssd-guard] {CONFIG.get('NOXSSD_MOUNT_PATH','/Volumes/NoxSSD')} "
                               f"unavailable — skipping open (data drive offline)")
            return None
        # v7 Phase 2 (gated) — pause NEW entries on extreme news event-risk (open positions
        # untouched). Active ONLY when LOCAL_AI_NEWS_APPLY=True (default False => no effect).
        if CONFIG.get('LOCAL_AI_NEWS_APPLY', False):
            _ns = getattr(state, 'news_sentiment', {}) or {}
            if (_ns.get('ok') and not _ns.get('stale')
                    and float(_ns.get('event_risk', 0.0) or 0.0) >= float(CONFIG.get('NEWS_DERISK_PAUSE_RISK', 0.85))):
                if state.benchmarks.get('total_signal_checks', 0) % 20 == 1:
                    logger.info(f"  [news de-risk] event_risk {_ns.get('event_risk'):.2f} >= pause "
                                f"threshold — skipping new entry (crash/event protection)")
                return None
        daily_pnl_pct=(state.balance-state.start_balance_today)/state.start_balance_today
        if daily_pnl_pct<-CONFIG['MAX_DAILY_DRAWDOWN']:
            state.halt_trading=True
            logger.warning(f"Daily drawdown {daily_pnl_pct:.1%} — HALT")
            return None
        # Quant overlay: total peak-to-trough drawdown circuit breaker
        dd_pct, dd_halt = quant_overlay.update_drawdown(state.balance)
        if dd_halt:
            state.halt_trading=True
            logger.warning(f"Total drawdown {dd_pct:.1%} from peak ${quant_overlay.peak_balance:.2f} — HARD HALT")
            return None
        # Loss-cluster cooldown: after N consecutive losses, pause for cooldown_sec
        if (state.consecutive_losses >= CONFIG['LOSS_COOLDOWN_AFTER'] and
                state.last_loss_time is not None):
            since_loss = (datetime.now(timezone.utc) - state.last_loss_time).total_seconds()
            if since_loss < CONFIG['LOSS_COOLDOWN_SEC']:
                if state.trade_count % 5 == 0:
                    logger.info(f"  [cooldown] {since_loss:.0f}s/{CONFIG['LOSS_COOLDOWN_SEC']}s "
                                f"after {state.consecutive_losses} losses")
                return None

        price=state.sol_price
        if not (price > 0) or not np.isfinite(price):
            logger.warning(f"execute_trade: invalid price={price}"); return None
        last=df.iloc[-1]
        atr=float(last.get('atr', price*0.005))
        if not np.isfinite(atr) or atr <= price * 0.0001:
            atr = price * 0.005  # 0.5% floor on ATR

        sl_m=signal_details.get('sl_mult',1.5); tp_m=signal_details.get('tp_mult',3.0)
        # Dynamic INITIAL stop: widen the ATR multiplier in persistent/volatile
        # regimes (room to turn around), tighten in calm (cut losers fast). Bounded
        # so the liq-distance guard below stays valid. (DSL_ENABLE=False ⇒ legacy.)
        sl_m = _dynamic_initial_sl_mult(sl_m, last)

        # --- SL distance is known pre-sizing; use it for the liquidation guard.
        sl_dist     = max(atr * sl_m, price * CONFIG['MIN_SL_PCT_OF_PRICE'])
        tp_dist     = max(atr * tp_m, sl_dist * CONFIG['MIN_TP_RR_RATIO'])
        if CONFIG.get('DSL_RR_REDESIGN', True):
            # fee floor: TP must clear a round-trip-cost margin (no-op when TP already wide)
            tp_dist = max(tp_dist, price * CONFIG.get('DSL_TP_FEE_RT_RATE', 0.0007)
                          * CONFIG['DSL_TP_FEE_MULT'])
        sl_pct      = sl_dist / price

        # Mode + operating leverage cap (router is authoritative on paper vs live).
        paper_mode = router.is_paper()
        lev_cap = CONFIG['PAPER_LEVERAGE_CAP'] if paper_mode else CONFIG['LEVERAGE_CAP']

        # Volatility-targeted leverage, then the LIQUIDATION-DISTANCE GUARD: the
        # hard stop must sit within LIQ_STOP_FRACTION of the liquidation distance
        # (≈ 1/lev for isolated margin). Solving frac/lev ≥ sl_pct gives the max
        # liq-safe leverage. We take the min of vol-target, the cap, and this.
        vol_lev = quant_overlay.vol_target_leverage()
        liq_safe_lev = int(CONFIG['LIQ_STOP_FRACTION'] / max(sl_pct, 1e-6)) \
            if CONFIG.get('LIQ_GUARD_ENABLE', True) else CONFIG['MAX_LEVERAGE']
        chosen_lev = max(1, min(int(vol_lev), int(lev_cap), max(1, liq_safe_lev)))

        size_sol,lev=calculate_position_size(
            state.balance,confidence,atr,price,chosen_lev,sl_m,state.current_regime,
            calibration=float(signal_details.get('conformal_p', 1.0) or 1.0))
        # Hard sanity bounds on size
        if not np.isfinite(size_sol) or size_sol <= 0:
            logger.warning(f"execute_trade: invalid size_sol={size_sol}"); return None
        size_sol = float(np.clip(size_sol,
                                  CONFIG['MIN_POSITION_SIZE'],
                                  state.balance * CONFIG['MAX_POSITION_PCT'] / max(price, 1e-6)))

        # Final liq-distance sanity: at the chosen leverage the liquidation buffer
        # must exceed MIN_LIQ_DISTANCE_PCT (else any sane stop sits past liq).
        liq_dist_pct = 1.0 / max(lev, 1)
        if CONFIG.get('LIQ_GUARD_ENABLE', True) and liq_dist_pct < CONFIG['MIN_LIQ_DISTANCE_PCT']:
            if state.trade_count % 20 == 0:
                logger.info(f"  [liq-guard] lev {lev}x → liq dist {liq_dist_pct:.2%} "
                            f"< min {CONFIG['MIN_LIQ_DISTANCE_PCT']:.2%}; skip")
            return None

        if action == 'long':
            sl_price = price - sl_dist
            tp_price = price + tp_dist
            assert sl_price < price < tp_price, "SL/TP cross-side validation failed"
        else:  # short
            sl_price = price + sl_dist
            tp_price = price - tp_dist
            assert sl_price > price > tp_price, "SL/TP cross-side validation failed"

        # --- HARD LOSS-CAP BACKSTOP (B6 fix, fingerprinted) -----------------------
        # Downstream of ALL sizing/SL logic: clamp size so the worst-case loss at the
        # FINAL stop can't exceed MAX_TRADE_LOSS_PCT of balance. The in-sizing cap sized
        # against the candidate sl_pct, but sl_price is finalized HERE — so a few trades
        # opened 5-6x over the cap (worst −$2.88 ≈ 70% of the cohort loss). loss_at_sl =
        # size_sol * lev * |entry − sl| (notional = size*price*lev; loss = notional*sl_pct).
        if CONFIG.get('LOSS_CAP_BACKSTOP', True):
            _cap_usd = state.balance * float(CONFIG.get('MAX_TRADE_LOSS_PCT', 0.005) or 0.0)
            _stop_dist = abs(price - sl_price)
            _worst_loss = size_sol * lev * _stop_dist
            if _cap_usd > 0 and _stop_dist > 0 and _worst_loss > _cap_usd:
                _old_sz = size_sol
                size_sol = max(_cap_usd / (lev * _stop_dist), CONFIG['MIN_POSITION_SIZE'])
                if state.trade_count % 10 == 0:
                    logger.info(f"  [loss-cap backstop] size {_old_sz:.4f}->{size_sol:.4f} SOL "
                                f"(worst-loss ${_worst_loss:.3f} > cap ${_cap_usd:.3f} @ {lev}x)")

        # --- Honest fee math via the active venue (round-trip = open + close).
        notional = size_sol * price * lev
        maker = bool(CONFIG.get('MAKER_BIAS', True))
        rt_fee_rate = router.round_trip_fee_rate(maker=maker)
        fee_usd = notional * rt_fee_rate

        # --- MIN-EDGE-VS-FEE GATE: skip trades whose probability-weighted expected
        # price move can't clear k×RT_fee + |funding|. Leverage cancels (both the
        # move and the fee scale with notional), so we compare fractions of price.
        funding_cost = abs(float(signal_details.get('funding_rate', state.funding_rate) or 0.0))
        exp_move_pct = (confidence * (tp_dist / price)
                        - (1.0 - confidence) * (sl_dist / price))
        edge_hurdle = CONFIG['MIN_EDGE_FEE_K'] * rt_fee_rate + funding_cost
        if exp_move_pct < edge_hurdle:
            if state.trade_count % 20 == 0:
                logger.info(f"  [min-edge] exp_move {exp_move_pct:.4%} < hurdle "
                            f"{edge_hurdle:.4%} (k×fee {CONFIG['MIN_EDGE_FEE_K']}×{rt_fee_rate:.4%} "
                            f"+ funding {funding_cost:.4%}); skip")
            return None

        # --- FEE-DRAG GOVERNOR: bound cumulative fee spend per UTC day to a % of
        # equity. Soft zone (≥75% of budget) requires a stronger edge; hard zone
        # (would exceed budget) blocks the open entirely.
        if CONFIG.get('FEE_DRAG_GOVERNOR', True):
            today = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            if today != state.fee_day:           # lazy roll if daily_reset hasn't fired
                state.fee_day = today; state.fees_today = 0.0
            budget = CONFIG['DAILY_FEE_BUDGET_PCT'] * max(state.balance, 1e-9)
            projected = state.fees_today + fee_usd
            if projected > budget:
                state.governor_throttles += 1
                if state.governor_throttles % 10 == 1:
                    logger.info(f"  [governor] fee budget hit: spent ${state.fees_today:.4f} "
                                f"+ ${fee_usd:.4f} > ${budget:.4f}/day; throttling opens")
                return None
            if projected > CONFIG['GOVERNOR_SOFT_PCT'] * budget and confidence < 0.60:
                if state.governor_throttles % 10 == 1:
                    logger.info(f"  [governor] soft zone (${projected:.4f}/${budget:.4f}); "
                                f"require conf≥0.60, have {confidence:.3f}; skip")
                state.governor_throttles += 1
                return None

        # --- Place the order through the router (Aster→ApeX failover, or paper
        # fills on real prices). The router returns a "paper_*" tx in paper mode
        # and a venue order id when live.
        tx = await router.place_perp_order(action, size_sol, lev, sl_price, tp_price,
                                           platform="auto")
        if not tx:
            logger.error("ORDER FAILED — router returned no tx (all venues?); aborting")
            return None
        is_paper_tx = isinstance(tx, str) and tx.startswith('paper_')
        venue_tag = 'PAPER' if is_paper_tx else router.last_venue.upper()
        logger.info(f"{venue_tag} TRADE: {action.upper()} {size_sol:.4f} SOL @{price:.2f} "
                    f"SL={sl_price:.2f} TP={tp_price:.2f} ({lev}x) "
                    f"fee≈${fee_usd:.4f} TX={str(tx)[:14]}")
        # Book the fee against today's governor budget.
        state.fees_today += fee_usd
        # Dynamic slippage on PAPER fills only: model a volume/vol-aware
        # market-impact penalty so paper PnL doesn't overstate live edge.
        # The entry is worsened (long fills higher, short fills lower) while
        # SL/TP stay anchored to the signal price (conservative — the stop
        # is NOT loosened to compensate). Live fills use the venue's real
        # execution price, so this only adjusts the paper record.
        entry_fill = price
        if is_paper_tx and CONFIG.get('DYNAMIC_SLIPPAGE', True):
            slip_bps = dynamic_slippage_bps(notional, state.current_regime,
                                            quant_overlay.garch_vol())
            if action == 'long':
                entry_fill = price * (1.0 + slip_bps / 10000.0)
            else:
                entry_fill = price * (1.0 - slip_bps / 10000.0)
        rec={'id':f"trade_{state.trade_count}",
             'timestamp':datetime.now(timezone.utc).isoformat(),
             'direction':action,'entry_price':entry_fill,'size_sol':size_sol,
             'leverage':lev,'sl_price':sl_price,'tp_price':tp_price,
             'confidence':confidence,'regime':state.current_regime,'tx':tx,
             'status':'open','pnl_usd':0.0,'fee_usd':fee_usd,
             'market': CONFIG['ACTIVE_MARKET'],  # gates the main loop from over-opening
             'is_paper': is_paper_tx,
             'venue': 'paper' if is_paper_tx else router.last_venue,
             'notional_usd': notional, 'rt_fee_rate': rt_fee_rate,
             'rl_state':signal_details.get('rl_state'),
             'rl_action':signal_details.get('rl_action'),
             # v7 1d — per-trade overlay audit (enables clean before/after attribution of the
             # calibration / haircut / news chain; previously these effects were not in the ledger)
             'conformal_p': signal_details.get('conformal_p'),
             'model_proba': signal_details.get('model_proba'),   # v8: raw model prob (ECE/calibration)
             'calibrated_conf': calibrate_confidence(confidence), # v9 Phase A: Platt-calibrated win-prob
             'calib_n': getattr(state, 'calib_n', 0),
             'aug_factor': signal_details.get('aug_factor'),
             'aug_dq': (getattr(state, 'ai_augmentation', {}) or {}).get('data_quality'),
             'news_sentiment': (getattr(state, 'news_sentiment', {}) or {}).get('sentiment'),
             'news_event_risk': (getattr(state, 'news_sentiment', {}) or {}).get('event_risk'),
             # v10 Track 2b — order-book depth SHADOW features (log-only; validated before any wiring)
             'ob_microprice_dev': (getattr(state, 'ob_features', {}) or {}).get('ob_microprice_dev'),
             'ob_depth_imb': (getattr(state, 'ob_features', {}) or {}).get('ob_depth_imb'),
             'ob_spread_bps': (getattr(state, 'ob_features', {}) or {}).get('ob_spread_bps'),
             'ob_imb_velocity': (getattr(state, 'ob_features', {}) or {}).get('ob_imb_velocity'),
             'xv_basis': (getattr(state, 'xv_features', {}) or {}).get('xv_basis'),
             'xv_basis_change': (getattr(state, 'xv_features', {}) or {}).get('xv_basis_change'),
             'xv_basis_velocity': (getattr(state, 'xv_features', {}) or {}).get('xv_basis_velocity'),
             # TV-rating replica shadows (log-only; validator-gated)
             'tv_r_1m':  (getattr(state, 'tv_ratings', {}) or {}).get('1m', {}).get('rating'),
             'tv_r_5m':  (getattr(state, 'tv_ratings', {}) or {}).get('5m', {}).get('rating'),
             'tv_r_15m': (getattr(state, 'tv_ratings', {}) or {}).get('15m', {}).get('rating'),
             'tv_r_30m': (getattr(state, 'tv_ratings', {}) or {}).get('30m', {}).get('rating'),
             'tv_r_1h':  (getattr(state, 'tv_ratings', {}) or {}).get('1h', {}).get('rating'),
             'tv_r_4h':  (getattr(state, 'tv_ratings', {}) or {}).get('4h', {}).get('rating'),
             'tv_r_1d':  (getattr(state, 'tv_ratings', {}) or {}).get('1d', {}).get('rating'),
             'tv_comp_eq':     (getattr(state, 'tv_composites', {}) or {}).get('tv_comp_eq'),
             'tv_comp_hold':   (getattr(state, 'tv_composites', {}) or {}).get('tv_comp_hold'),
             'tv_align_count': (getattr(state, 'tv_composites', {}) or {}).get('tv_align_count'),
             'tv_strong_frac': (getattr(state, 'tv_composites', {}) or {}).get('tv_strong_frac'),
             'tv_persist_sec': (getattr(state, 'tv_composites', {}) or {}).get('tv_persist_sec'),
             # extra indicator shadows (SuperTrend/VWAP-dev/pivot-dist)
             'ti_supertrend': (getattr(state, 'ti_features', {}) or {}).get('ti_supertrend'),
             'ti_vwap_dev':   (getattr(state, 'ti_features', {}) or {}).get('ti_vwap_dev'),
             'ti_pivot_dist': (getattr(state, 'ti_features', {}) or {}).get('ti_pivot_dist'),
             'ti_ema_cross_age': (getattr(state, 'ti_features', {}) or {}).get('ti_ema_cross_age'),
             'ti_mr_zscore': (getattr(state, 'ti_features', {}) or {}).get('ti_mr_zscore'),
             # S/R shadow features (log-only; validator-gated)
             'sr_dist_supp':     (getattr(state, 'sr_feat', {}) or {}).get('sr_dist_supp'),
             'sr_dist_res':      (getattr(state, 'sr_feat', {}) or {}).get('sr_dist_res'),
             'sr_supp_strength': (getattr(state, 'sr_feat', {}) or {}).get('sr_supp_strength'),
             'sr_res_strength':  (getattr(state, 'sr_feat', {}) or {}).get('sr_res_strength'),
             'sr_room_ratio':    (getattr(state, 'sr_feat', {}) or {}).get('sr_room_ratio'),
             # v10 Track 2 — on-chain network-activity shadow features (different signal class; log-only)
             'oc_tps': (getattr(state, 'onchain_features', {}) or {}).get('oc_tps'),
             'oc_tps_trend': (getattr(state, 'onchain_features', {}) or {}).get('oc_tps_trend'),
             'cfg_fp': getattr(state, 'config_fingerprint', '')}  # rule-set this trade was made under
        state.trade_log.append(rec)
        state.open_positions.append(rec)
        state.cycle_trades.append(rec)   # shared reference; close updates flow through
        state.trade_count+=1
        state.last_trade_time=datetime.now(timezone.utc)
        logger.info(f"TRADE: {action.upper()} {size_sol:.4f} SOL @{price:.4f} "
                    f"| {lev}x | SL:{sl_price:.4f} TP:{tp_price:.4f} | TX:{tx}")
        return tx

# ============================================================
# SECTION 21: POSITION MONITORING
# ============================================================
def _dynamic_initial_sl_mult(base_mult: float, last) -> float:
    """Regime/vol-adaptive INITIAL stop multiplier (× ATR), bounded.

    Wider in persistent (high-Hurst) and/or volatile regimes so a trade that
    opens slightly negative has room to turn around; tighter in calm/mean-
    reverting regimes so losers are cut fast. The vol term only ever WIDENS
    (one-sided) so elevated volatility never shrinks the stop into the noise.
    Bounds keep the resulting sl_pct liq-safe; MIN_SL_PCT_OF_PRICE still floors
    the absolute distance downstream.
    """
    if not CONFIG.get('DSL_ENABLE', True):
        return float(base_mult)
    try:
        hurst = float(last.get('hurst', 0.5))
        if not np.isfinite(hurst):
            hurst = 0.5
        ewma_vol = float(last.get('ewma_vol', CONFIG['DSL_INIT_VOL_TARGET']))
        if not np.isfinite(ewma_vol) or ewma_vol <= 0:
            ewma_vol = CONFIG['DSL_INIT_VOL_TARGET']
        persist = hurst - 0.5                                   # >0 trending, <0 reverting
        vol_rel = (ewma_vol / CONFIG['DSL_INIT_VOL_TARGET']) - 1.0
        mult = base_mult * (1.0
                            + CONFIG['DSL_INIT_HURST_GAIN'] * persist
                            + CONFIG['DSL_INIT_VOL_GAIN'] * max(0.0, vol_rel))
        mult = float(np.clip(mult,
                             CONFIG['DSL_INIT_ATR_MULT_MIN'],
                             CONFIG['DSL_INIT_ATR_MULT_MAX']))
        # Hard $-risk cap: translate the price-pct ceiling into an ATR-mult ceiling so
        # the initial stop never exceeds DSL_MAX_INIT_SL_PCT of price (bounds loser $).
        if CONFIG.get('DSL_RR_REDESIGN', True):
            atr = float(last.get('atr', 0.0)); px = float(last.get('close', 0.0))
            if atr > 0 and px > 0:
                mult = min(mult, CONFIG['DSL_MAX_INIT_SL_PCT'] * px / atr)
        return float(mult)
    except Exception:
        return float(base_mult)


def _apply_dynamic_trail(pos: dict, price: float, atr_now: float,
                         entry: float, sl: float, d: str) -> None:
    """Peak-anchored, profit-progressive trailing stop (in-place on pos).

    Tracks max favourable excursion (peak for long / trough for short) and trails
    at an ATR distance that TIGHTENS as profit (in R = multiples of the initial
    stop) grows, so big winners give back little — while a volatility floor
    (DSL_TRAIL_FLOOR_PCT of price) keeps a short, unsustained reversal from
    closing us. Ratchets favourably only; never loosens the stop.
    """
    init_dist = abs(entry - sl)
    if init_dist <= 0:
        init_dist = price * CONFIG['MIN_SL_PCT_OF_PRICE']
    if d == 'long':
        peak = max(float(pos.get('peak_price', entry)), price)
        pos['peak_price'] = peak
        profit = peak - entry
    else:
        peak = min(float(pos.get('peak_price', entry)), price)
        pos['peak_price'] = peak
        profit = entry - peak
    if profit <= 0:
        return  # under water — initial stop stands (room to turn around)
    profit_R = profit / max(init_dist, 1e-9)
    # Move-to-breakeven: once modestly green (≥DSL_BE_TRIGGER_R), ratchet the stop to a
    # fee-covering offset beyond entry so a subsequent reversal SCRATCHES at ≥$0 instead
    # of paying the full initial stop. This is the highest-impact payoff lever (turns
    # most "losers" into ~$0 net-positive exits). Ratchet-only.
    if (CONFIG.get('DSL_RR_REDESIGN', True) and CONFIG.get('DSL_BREAKEVEN_ENABLE', True)
            and profit_R >= CONFIG['DSL_BE_TRIGGER_R']):
        rt_fee = float(pos.get('rt_fee_rate', 0.0) or 0.0)
        be_off = max(price * CONFIG['DSL_BE_MIN_OFFSET_PCT'],
                     price * rt_fee * CONFIG['DSL_BE_OFFSET_FEES_K'])
        if d == 'long':
            be_sl = entry + be_off
            if be_sl > sl:
                pos['sl_price'] = sl = be_sl; pos['be_armed'] = True
        else:
            be_sl = entry - be_off
            if be_sl < sl:
                pos['sl_price'] = sl = be_sl; pos['be_armed'] = True
    if profit_R < CONFIG['DSL_TRAIL_ACTIVATE_R']:
        return  # not enough profit yet to start trailing (breakeven, if armed, stands)
    extra_R = profit_R - CONFIG['DSL_TRAIL_ACTIVATE_R']
    atr_mult = CONFIG['DSL_TRAIL_ATR_MULT'] - CONFIG['DSL_TRAIL_TIGHTEN_PER_R'] * extra_R
    atr_mult = max(CONFIG['DSL_TRAIL_MIN_MULT'], atr_mult)
    trail_dist = max(atr_now * atr_mult, price * CONFIG['DSL_TRAIL_FLOOR_PCT'])
    if CONFIG.get('DSL_RR_REDESIGN', True):
        # anti-shakeout: the trail is never tighter than the ATR-noise floor, so a
        # normal <1×ATR noise dip can't stop us, but a genuine ≥1.4×ATR reversal does.
        trail_dist = max(trail_dist, atr_now * CONFIG.get('DSL_TRAIL_NOISE_ATR_MULT', 0.90))
    if d == 'long':
        new_sl = peak - trail_dist
        if new_sl > sl:
            pos['sl_price'] = new_sl
    else:
        new_sl = peak + trail_dist
        if new_sl < sl:
            pos['sl_price'] = new_sl


def _partial_take_profit(pos: dict, price: float, full_pnl_usd: float, frac: float,
                         entry: float, d: str) -> None:
    """Bank `frac` of an in-profit position (a realized partial WIN), shrink the runner,
    move its stop to breakeven, and stash the realized partial so the final close record
    reflects partial+runner. Paper-only book-keeping; balance credited here for the banked
    leg, the runner leg credited at full close. Idempotent via pos['tp1_done']."""
    size = float(pos.get('size_sol', 0.0)); lev = float(pos.get('leverage', 1))
    if size <= 0 or not (0.0 < frac < 1.0):
        return
    rt = float(pos.get('rt_fee_rate', 0.0) or 0.0)
    part_pnl = full_pnl_usd * frac                              # pnl on the banked fraction
    part_fee = rt * (size * frac * price * lev) * 0.5           # exit-leg fee on that fraction
    net = part_pnl - part_fee
    state.balance = max(state.balance + net, 0.01)
    pos['realized_partial_pnl'] = float(pos.get('realized_partial_pnl', 0.0)) + net
    pos['size_sol'] = size * (1.0 - frac)                       # runner = remaining fraction
    pos['notional_usd'] = float(pos.get('notional_usd', size * entry * lev)) * (1.0 - frac)
    pos['fee_usd'] = float(pos.get('fee_usd', 0.0)) * (1.0 - frac)
    pos['tp1_done'] = True
    be_off = max(price * CONFIG.get('DSL_BE_MIN_OFFSET_PCT', 0.0006),
                 price * rt * CONFIG.get('DSL_BE_OFFSET_FEES_K', 1.5))
    cur_sl = float(pos.get('sl_price', entry))
    pos['sl_price'] = max(cur_sl, entry + be_off) if d == 'long' else min(cur_sl, entry - be_off)
    pos['be_armed'] = True
    logger.info(f"  [TP-ladder] banked {frac:.0%} of {pos.get('id','?')} net={net:+.4f} "
                f"→ runner size {size:.4f}→{pos['size_sol']:.4f}, stop→BE")


async def monitor_positions(df: pd.DataFrame):
    if not state.open_positions: return
    price = state.sol_price
    if not (price > 0) or not np.isfinite(price): return
    # Velocity once per call (market-level — same for every open position) to power
    # the momentum-death "quick out": when an IN-PROFIT position's vol-normalised
    # velocity flips hard AGAINST its direction, snap the trailing stop right behind
    # price so a small further reversal closes us — capturing profit BEFORE the move
    # fully reverses. Ratchet-only, and gated on pnl>floor so it never locks a loss.
    velo_exit = bool(CONFIG.get('VELO_EXIT_ENABLE', True)) and bool(CONFIG.get('DSL_ENABLE', True))
    v_norm_now = 0.0
    if velo_exit:
        try:
            _, v_norm_now, _, _ = unidirectional_velocity(df)
        except Exception:
            v_norm_now = 0.0
    flip_z = float(CONFIG.get('VELO_EXIT_FLIP_Z', 0.50))
    floor_pct = float(CONFIG.get('DSL_TRAIL_FLOOR_PCT', 0.0035))
    to_close = []
    for pos in state.open_positions[:]:
        try:
            entry = float(pos.get('entry_price', 0))
            if entry <= 0: continue
            sl = float(pos.get('sl_price', 0)); tp = float(pos.get('tp_price', 0))
            d  = pos.get('direction', '')
            if d not in ('long','short'): continue
            pnl_pct = (price-entry)/entry if d=='long' else (entry-price)/entry
            pnl_usd = pnl_pct * float(pos.get('size_sol',0)) * entry * float(pos.get('leverage',1))
            pos['pnl_usd'] = pnl_usd
            # live adjusted confidence (DISPLAY+LOG only — NEVER affects exits/sizing).
            # Tracks how the model's current confidence in THIS direction evolves vs entry.
            try:
                _ac = (getattr(state, 'live_model_proba', None) or {}).get(d)
                if _ac is not None:
                    pos['adj_conf_last'] = round(float(_ac), 4)
                    _cp = pos.get('conf_path') or []
                    if (not _cp) or abs(_cp[-1] - float(_ac)) >= 0.005:
                        _cp.append(round(float(_ac), 4)); pos['conf_path'] = _cp[-40:]
            except Exception:
                pass
            hit_sl = (d=='long' and price<=sl) or (d=='short' and price>=sl)
            hit_tp = (d=='long' and price>=tp) or (d=='short' and price<=tp)
            if hit_sl or hit_tp:
                to_close.append((pos, 'tp' if hit_tp else 'sl', pnl_usd))
            else:
                atr_now = float(df.iloc[-1].get('atr', entry*0.02)) if not df.empty else entry*0.02
                if not np.isfinite(atr_now) or atr_now <= 0:
                    atr_now = entry * 0.005
                # Partial-TP ladder: bank half at +DSL_TP1_R, runner → breakeven + chandelier.
                if (CONFIG.get('DSL_RR_REDESIGN', True) and CONFIG.get('DSL_TP_LADDER_ENABLE', True)
                        and not pos.get('tp1_done')):
                    _init_dist = abs(entry - sl)
                    if _init_dist > 0:
                        _prof_R = ((price - entry) if d == 'long' else (entry - price)) / _init_dist
                        if _prof_R >= float(CONFIG.get('DSL_TP1_R', 1.5)):
                            _partial_take_profit(pos, price, pnl_usd,
                                                 float(CONFIG.get('DSL_TP1_FRACTION', 0.5)), entry, d)
                if CONFIG.get('DSL_ENABLE', True):
                    # Dynamic peak-anchored trail: tightens as profit (in R) grows
                    # to capture near-max on big winners, with a volatility floor so
                    # a short reversal doesn't close us. Ratchets favourably only.
                    _apply_dynamic_trail(pos, price, atr_now, entry, sl, d)
                else:
                    # Legacy: trail at fixed 2×ATR after ≥1.5×SL profit (from price).
                    if d=='long' and (price-entry) > 1.5 * abs(entry-sl):
                        trail = price - atr_now*2
                        if trail > sl: pos['sl_price'] = trail
                    elif d=='short' and (entry-price) > 1.5 * abs(sl-entry):
                        trail = price + atr_now*2
                        if trail < sl: pos['sl_price'] = trail
                # Momentum-death scalp exit (the "quick out"): in-profit position whose
                # velocity has flipped ≥flip_z AGAINST it ⇒ snap stop to floor-distance
                # behind price. Guard pnl_pct>floor_pct so the snapped stop stays at-or-
                # above breakeven (never converts a winner into a loss); ratchet-only.
                if velo_exit and np.isfinite(v_norm_now) and pnl_pct > floor_pct:
                    died = (d == 'long'  and v_norm_now <= -flip_z) or \
                           (d == 'short' and v_norm_now >=  flip_z)
                    if died:
                        cur_sl = float(pos.get('sl_price', sl))
                        if d == 'long':
                            snap = price * (1.0 - floor_pct)
                            if snap > cur_sl:
                                pos['sl_price'] = snap
                                pos['exit_reason_hint'] = 'velocity_fade'
                        else:
                            snap = price * (1.0 + floor_pct)
                            if snap < cur_sl:
                                pos['sl_price'] = snap
                                pos['exit_reason_hint'] = 'velocity_fade'
        except Exception as e:
            logger.warning(f"monitor_positions[pos={pos.get('id','?')}]: {e}")
            continue
    for pos, reason, pnl_usd in to_close:
        try:
            await _close_position(pos, reason, pnl_usd)
        except Exception as e:
            logger.error(f"close failed for {pos.get('id','?')}: {e}", exc_info=True)

async def _close_position(pos, reason, pnl_usd):
    # Paper-mode positions (tx starts with "paper_") never made a Drift call
    # to open — never call Drift to close. Just book-keep and credit PnL.
    is_paper = isinstance(pos.get('tx',''), str) and pos['tx'].startswith('paper_')
    if is_paper:
        closed = True
    else:
        closed = await drift_trader.close_position(CONFIG['ACTIVE_MARKET'])
    if not closed: return
    # Fold any partial-TP-ladder proceeds (already credited to balance at partial time)
    # into the trade RECORD + stats so won/PnL reflect partial + runner combined.
    realized_partial = float(pos.get('realized_partial_pnl', 0.0))
    pos.update({'status':'closed','close_reason':reason,'pnl_usd':pnl_usd + realized_partial,
                'close_time':datetime.now(timezone.utc).isoformat()})
    state.balance=max(state.balance+pnl_usd-pos['fee_usd'],0.01)   # runner leg (partial already booked)
    net_pnl = (pnl_usd - pos['fee_usd']) + realized_partial        # total trade net (partial + runner)
    won = net_pnl > 0
    # ---- HOOK B: post-trade AI rationale journal (fire-and-forget; never blocks) ----
    if local_ai is not None and CONFIG.get('LOCAL_AI_POSTTRADE_LOG', True):
        try:
            with open(CONFIG.get('LOCAL_AI_JOURNAL_PATH', 'local_ai_journal.jsonl'), 'a') as _jf:
                _jf.write(json.dumps({'ts': time.time(), 'dir': pos.get('direction'),
                                      'entry_ai': pos.get('entry_ai_verdict'),
                                      'net_pnl': round(float(net_pnl), 4), 'won': bool(won),
                                      'regime': getattr(state, 'current_regime', None),
                                      'exit_reason': reason}) + '\n')
        except Exception:
            pass
    # Loss-cluster + adaptive-threshold bookkeeping
    if won:
        state.consecutive_wins   += 1
        state.consecutive_losses  = 0
        # Loosen confidence threshold slightly as we accumulate wins (allow more trades)
        if state.consecutive_wins >= 3 and CONFIG.get('ADAPTIVE_CONF_ENABLE', True):
            state.adaptive_conf_offset = max(state.adaptive_conf_offset - 0.005, -0.02)
    else:
        state.consecutive_losses += 1
        state.consecutive_wins    = 0
        state.last_loss_time      = datetime.now(timezone.utc)
        # Tighten threshold proportional to streak (drives selectivity up after losses)
        if CONFIG.get('ADAPTIVE_CONF_ENABLE', True):
            state.adaptive_conf_offset = min(state.adaptive_conf_offset + 0.01, 0.06)
    # Quant overlay: record trade PnL → Deflated Sharpe, drawdown, meta-label
    try:
        quant_overlay.record_trade_pnl(net_pnl, state.balance)
        quant_overlay.update_metalabel(
            pos.get('regime','unknown'), pos.get('direction','hold'), won=won)
    except Exception: pass
    # Periodic state persistence (every N closed trades)
    if state.trade_count % CONFIG.get('STATE_PERSIST_EVERY', 5) == 0:
        persist_state()
    # Adaptive conf-floor recalibration (every N closes, current-fingerprint trades only)
    if (CONFIG.get('CONF_OPT_ENABLE', True)
            and state.trade_count % int(CONFIG.get('CONF_OPT_RECAL_EVERY', 20)) == 0):
        try: optimize_confidence_floor()
        except Exception: pass
    # v9 Phase A — refit the confidence calibrator every N closes as the cohort grows
    if (CONFIG.get('CALIB_ENABLE', True)
            and state.trade_count % int(CONFIG.get('CALIB_RECAL_EVERY', 20)) == 0):
        try: fit_confidence_calibration()
        except Exception: pass
    today=datetime.now(timezone.utc).strftime('%Y-%m-%d')
    state.daily_pnl[today]=state.daily_pnl.get(today,0)+pnl_usd
    if state.daily_pnl[today]>state.best_day['pnl']:
        pct=state.daily_pnl[today]/state.start_balance_today
        state.best_day={'date':today,'pnl':state.daily_pnl[today],'pct':pct}
    if pos in state.open_positions:
        state.open_positions.remove(pos)
    reward=float(np.clip(pnl_usd/(pos['size_sol']*pos['entry_price']+1e-10),-1,1))
    rl_stop_optimizer.update(pos.get('rl_state',(0,0,0,0,1)),pos.get('rl_action',0),
                              reward,(0,0,0,0,1),pos.get('regime','choppy'),done=True)
    rl_regime_classifier.online_update(np.zeros(32),reward)
    logger.info(f"CLOSED {reason.upper()} | PnL:{pnl_usd:+.4f} | Bal:{state.balance:.4f}")

# ============================================================
# SECTION 22: MODEL TRAINING
# ============================================================
async def train_models(df: pd.DataFrame):
    try:
        X,y=build_features(df)
        if len(X)<100:
            logger.info(f"Training skipped — only {len(X)} usable samples after feature build (need 100)")
            return
        logger.info(f"Training ML+DL on {len(X)} samples, {X.shape[1]} features...")
        logger.info("  → fitting RandomForest..."); state.benchmarks['_train_step']='rf'
        ml_ensemble.fit(X,y)
        logger.info("  → ML ensemble fitted, starting DL...")
        try:
            dl_manager.fit(X,y,epochs=5)
            logger.info("  → DL training complete")
        except Exception as dl_err:
            logger.error(f"  → DL training failed (continuing ML-only): {dl_err}")
        if CONFIG.get('EVAL_PURGED_ENABLE',True):
            state.model_performance=ml_ensemble.evaluate_purged(
                X,y,embargo=CONFIG.get('EVAL_EMBARGO_BARS',5),
                n_splits=CONFIG.get('META_OOF_SPLITS',3))
            logger.info(f"Model perf (purged): {state.model_performance}")
        else:
            split=int(len(X)*0.8)
            if split<len(X):
                state.model_performance=ml_ensemble.evaluate(X[split:],y[split:])
                logger.info(f"Model perf: {state.model_performance}")
        # Meta-label concept-drift decay: fade stale (regime,action) evidence so the
        # de-leaked model is not blocked by the prior leaky model's losing buckets.
        quant_overlay.decay_metalabels(CONFIG.get('METALABEL_DECAY', 0.98))
        # Regime classifier training
        feats=[]; labs=[]
        if 'hurst' in df.columns:
            for _,row in df.dropna(subset=['hurst','adx']).iterrows():
                f=np.zeros(32)
                f[0]=float(row.get('hurst',0.5)); f[1]=float(row.get('adx',25))/100
                f[2]=float(row.get('ewma_vol',0.02)); f[3]=float(row.get('vpin',0.5))
                feats.append(f)
                h=float(row.get('hurst',0.5)); adx=float(row.get('adx',25))
                labs.append(0 if h>0.6 and adx>30 else (2 if h<0.4 else 1))
            if len(feats)>20:
                rl_regime_classifier.fit(np.array(feats),np.array(labs))
        # PPO state update
        if HAS_RL:
            ppo_obs=np.array([state.sol_price/1000,state.balance/100,
                              float(state.current_regime=='trending'),
                              float(state.halt_trading),len(state.open_positions)/10,
                              state.model_performance.get('accuracy',0.5),
                              state.funding_rate*1000, state.sentiment_score,
                              state.macro_data.get('vix',15)/30,
                              state.macro_data.get('dxy',100)/110,
                              float(state.trade_count)/100,0.0],dtype=np.float32)
            ppo_agent.learn(total_timesteps=10)
    except Exception as e:
        logger.error(f"Training error: {e}", exc_info=True)

# ============================================================
# SECTION 23: FASTAPI DASHBOARD SERVER
# ============================================================
app=FastAPI(title=CONFIG['NAME'])

# ============================================================
# SECTION 23a: DASHBOARD AUTH GATE (session cookie; server-side content blocking)
# Pre-auth, the ONLY content served is the login page (decorative backdrop, zero real data) —
# stronger than any CSS blur: the dashboard/API payloads never reach the browser, so there is
# nothing to reverse/de-blur. All /api/* + /ws return 401 without a valid session.
# ============================================================
_AUTH_SESSIONS: Dict[str, float] = {}     # token -> expiry epoch
_AUTH_FAILS: Dict[str, list] = {}         # ip -> [fail_count, locked_until_epoch]
_AUTH_EXEMPT = {'/login', '/healthz'}

_LOGIN_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Delphi — Sign in</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>
body{margin:0;font-family:-apple-system,Segoe UI,sans-serif;min-height:100vh;display:flex;
align-items:center;justify-content:center;background:#0d1117;overflow:hidden}
.bg{position:fixed;inset:-60px;filter:blur(70px) saturate(1.4);opacity:.5;z-index:0;background:
radial-gradient(40% 50% at 20% 30%,#0a3b66 0%,transparent 60%),
radial-gradient(35% 45% at 75% 25%,#3b1d6e 0%,transparent 60%),
radial-gradient(45% 55% at 60% 75%,#0a5c46 0%,transparent 60%),
radial-gradient(30% 35% at 35% 80%,#5c0a2e 0%,transparent 60%)}
.card{position:relative;z-index:1;background:rgba(22,27,34,.92);border:1px solid #30363d;
border-radius:12px;padding:34px 38px;width:300px;box-shadow:0 20px 60px rgba(0,0,0,.5)}
h1{font-size:16px;color:#e6edf3;margin:0 0 4px;letter-spacing:.5px}
p{font-size:11px;color:#8b949e;margin:0 0 18px}
input{width:100%;box-sizing:border-box;margin:6px 0;padding:10px 12px;border-radius:7px;
border:1px solid #30363d;background:#0d1117;color:#e6edf3;font-size:13px;outline:none}
input:focus{border-color:#00d4ff}
button{width:100%;margin-top:14px;padding:10px;border-radius:7px;border:1px solid #00d4ff;
background:transparent;color:#00d4ff;font-weight:600;font-size:13px;cursor:pointer}
button:hover{background:rgba(0,212,255,.12)}
.msg{color:#ff4d6d;font-size:11px;min-height:14px;margin-top:10px}</style></head>
<body><div class="bg"></div><div class="card"><h1>DELPHI NOX DARK</h1>
<p>Authentication required</p>
<form method="post" action="/login" autocomplete="off">
<input name="username" placeholder="Username" autofocus required>
<input name="password" type="password" placeholder="Password" required>
<button type="submit">Sign in</button></form>
<div class="msg"><!--MSG--></div></div></body></html>"""

def _auth_enabled() -> bool:
    return (bool(CONFIG.get('DASH_AUTH_ENABLE', True))
            and bool(os.getenv('DELPHI_DASH_USER', ''))
            and bool(os.getenv('DELPHI_DASH_PWHASH', '')))

def _verify_dash_password(user: str, pw: str) -> bool:
    """Constant-time scrypt verification against .env (salt_hex$hash_hex). Never logs inputs."""
    u = os.getenv('DELPHI_DASH_USER', ''); ph = os.getenv('DELPHI_DASH_PWHASH', '')
    try:
        salt_hex, hash_hex = ph.split('$', 1)
        calc = hashlib.scrypt(pw.encode(), salt=bytes.fromhex(salt_hex),
                              n=16384, r=8, p=1, dklen=32).hex()
        return (hmac.compare_digest(user.encode(), u.encode())
                and hmac.compare_digest(calc, hash_hex))
    except Exception:
        return False

def _session_valid(token: str) -> bool:
    return bool(token) and _AUTH_SESSIONS.get(token, 0.0) > time.time()

@app.middleware("http")
async def _auth_gate(request: Request, call_next):
    if not _auth_enabled() or request.url.path in _AUTH_EXEMPT:
        resp = await call_next(request)
    else:
        tok = request.cookies.get('delphi_session', '')
        if _session_valid(tok):
            _AUTH_SESSIONS[tok] = time.time() + float(CONFIG.get('DASH_SESSION_TTL_SEC', 43200))
            resp = await call_next(request)
        elif request.url.path.startswith('/api') or request.url.path == '/ws':
            resp = JSONResponse({'error': 'unauthorized'}, status_code=401)
        else:
            resp = HTMLResponse(_LOGIN_HTML, status_code=401)   # NO dashboard content pre-auth
    # security headers on every response
    resp.headers['X-Frame-Options'] = 'DENY'
    resp.headers['X-Content-Type-Options'] = 'nosniff'
    resp.headers['Referrer-Policy'] = 'no-referrer'
    resp.headers['Strict-Transport-Security'] = 'max-age=31536000'
    return resp

@app.get("/login", include_in_schema=False)
async def login_page():
    return HTMLResponse(_LOGIN_HTML)

@app.post("/login", include_in_schema=False)
async def login(request: Request, username: str = Form(''), password: str = Form('')):
    ip = request.client.host if request.client else '?'
    rec = _AUTH_FAILS.get(ip, [0, 0.0])
    if rec[1] > time.time():
        logger.warning(f"[auth] locked-out attempt from {ip}")
        return HTMLResponse(_LOGIN_HTML.replace('<!--MSG-->',
                            'Too many attempts — try again later.'), status_code=429)
    if _verify_dash_password(username, password):
        _AUTH_FAILS.pop(ip, None)
        tok = secrets.token_urlsafe(32)
        _AUTH_SESSIONS[tok] = time.time() + float(CONFIG.get('DASH_SESSION_TTL_SEC', 43200))
        r = RedirectResponse('/', status_code=303)
        r.set_cookie('delphi_session', tok, httponly=True, samesite='strict',
                     secure=bool(CONFIG.get('DASH_COOKIE_SECURE', False)),
                     max_age=int(CONFIG.get('DASH_SESSION_TTL_SEC', 43200)))
        logger.info(f"[auth] login OK from {ip}")
        return r
    rec[0] += 1
    if rec[0] >= int(CONFIG.get('DASH_LOGIN_MAX_FAILS', 5)):
        rec = [0, time.time() + float(CONFIG.get('DASH_LOGIN_LOCK_SEC', 900))]
        logger.warning(f"[auth] lockout engaged for {ip}")
    _AUTH_FAILS[ip] = rec
    logger.info(f"[auth] login FAIL from {ip}")   # never log the submitted creds
    return HTMLResponse(_LOGIN_HTML.replace('<!--MSG-->', 'Invalid credentials.'), status_code=401)

@app.post("/logout", include_in_schema=False)
async def logout(request: Request):
    _AUTH_SESSIONS.pop(request.cookies.get('delphi_session', ''), None)
    r = RedirectResponse('/login', status_code=303)
    r.delete_cookie('delphi_session')
    return r

# Root + /dashboard both serve the HTML dashboard. Using a resolved absolute
# path so the route works regardless of the cwd the script is launched from.
_DASHBOARD_HTML = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'delphi_novalaunch_1_01.html'
)

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def serve_dashboard():
    if not os.path.exists(_DASHBOARD_HTML):
        return HTMLResponse(
            f"<h1>{CONFIG['NAME']}</h1><p>Dashboard HTML not found at "
            f"<code>{_DASHBOARD_HTML}</code>.</p>"
            f"<p>API is live at <a href='/api/status'>/api/status</a>.</p>",
            status_code=200,
        )
    return FileResponse(_DASHBOARD_HTML, media_type='text/html')

@app.get("/healthz", include_in_schema=False)
async def healthz():
    """Lightweight liveness probe — used by the macOS launcher startup loop."""
    return {"ok": True, "version": CONFIG['VERSION'],
            "uptime_s": int((datetime.now(timezone.utc) - state.start_time).total_seconds())}
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])
connected_ws: List[WebSocket]=[]

def compute_ece(n_bins: int = 10) -> Dict:
    """Expected Calibration Error over closed trades that recorded a model_proba (v8 Step-0
    instrumentation). Bins predicted prob vs realized win-frequency → ECE + reliability curve. This
    is the Phase-A calibration baseline; scoped to the current rule-set (cfg_fp) like the win-rate.
    Read-only — never affects trading."""
    cur = getattr(state, 'config_fingerprint', '')
    rows = [t for t in state.trade_log
            if isinstance(t, dict) and t.get('status') == 'closed'
            and t.get('cfg_fp') == cur and t.get('model_proba') is not None]
    n = len(rows)
    if n == 0:
        return {'n': 0, 'ece': None, 'bins': [], 'scope': 'current_rule_set'}
    bins = [{'lo': i / n_bins, 'hi': (i + 1) / n_bins, 'n': 0, 'p_sum': 0.0, 'wins': 0}
            for i in range(n_bins)]
    for t in rows:
        try:
            p = max(0.0, min(0.999999, float(t['model_proba'])))
        except Exception:
            continue
        win = 1 if (float(t.get('pnl_usd', 0) or 0) - float(t.get('fee_usd', 0) or 0)) > 0 else 0
        b = bins[min(n_bins - 1, int(p * n_bins))]
        b['n'] += 1; b['p_sum'] += p; b['wins'] += win
    ece = 0.0; out_bins = []
    for b in bins:
        if b['n'] == 0:
            continue
        conf = b['p_sum'] / b['n']        # avg predicted prob in the bin
        acc = b['wins'] / b['n']          # realized win-frequency
        ece += (b['n'] / n) * abs(conf - acc)
        out_bins.append({'lo': round(b['lo'], 2), 'hi': round(b['hi'], 2), 'n': b['n'],
                         'pred': round(conf, 3), 'actual': round(acc, 3)})
    return {'n': n, 'ece': round(ece, 4), 'bins': out_bins, 'scope': 'current_rule_set'}

def _data_freshness() -> Dict:
    """Verification summary of live data-feed health for the dashboard (v6).
    Read-only aggregation of state.api_health (maintained by record_api) + the data-confidence
    breaker — NEVER affects trading. A configured feed with no successful ping within
    DATA_STALE_SEC, or one whose last call failed, is surfaced so a frozen/broken upstream is
    caught immediately rather than silently feeding stale numbers into the dashboard."""
    now = datetime.now(timezone.utc)
    stale_sec = float(CONFIG.get('DATA_STALE_SEC', 180))
    # fresh-by-design ceilings: slow-moving sources poll on long TTLs ON PURPOSE — judge them
    # against their own cadence, not the hot-feed threshold (stops the chip false-warning on FRED).
    per_source_stale = {'fred': 7200.0, 'dune': 7200.0, 'polymarket': 1800.0, 'helius': 1800.0}
    total = ok = stale = failed = 0
    oldest = None
    for key, info in state.api_health.items():
        if not info.get('configured'):
            continue
        st = info.get('status', 'idle')
        if st == 'idle':           # never called this run → not counted as a live feed
            continue
        total += 1
        src_stale_sec = max(stale_sec, per_source_stale.get(key, 0.0))
        age = None
        lp = info.get('last_ping')
        if lp:
            try: age = (now - datetime.fromisoformat(lp)).total_seconds()
            except Exception: age = None
        if st == 'fail':
            failed += 1
        elif age is not None and age > src_stale_sec:
            stale += 1
        else:
            ok += 1
        if age is not None and (oldest is None or age > oldest[1]):
            oldest = (info.get('name', key), age)
    breaker = int(getattr(state, 'data_breaker_skips', 0))
    # Proportionate severity: a few flaky/optional feeds (e.g. social sentiment) ⇒ 'warn'
    # (yellow); only a broadly-unreliable feed set or a tripped data-confidence breaker ⇒
    # 'degraded' (red). Avoids desensitising the operator to a persistent red chip.
    overall = 'ok'
    if stale or failed:
        overall = 'warn'
    if breaker > 0 or (total and failed / total >= 0.5):
        overall = 'degraded'
    return {
        'overall': overall,
        'sources_active': total, 'ok': ok, 'stale': stale, 'failed': failed,
        'data_breaker_skips': breaker,
        'stale_threshold_sec': int(stale_sec),
        'oldest_name': (oldest[0] if oldest else None),
        'oldest_age_sec': (int(oldest[1]) if oldest else None),
        'checked_at': now.isoformat(),
    }

def _bench_one(rows):
    """Compute one benchmark row (DISPLAY-ONLY) over a set of closed trades."""
    n = len(rows)
    if not n:
        return {'n': 0, 'win_rate': 0.0, 'avg_conf': 0.0,
                'per_trade_mean': 0.0, 'per_trade_median': 0.0, 'net': 0.0}
    pns = np.array([float(t.get('pnl_usd', 0) or 0) - float(t.get('fee_usd', 0) or 0)
                    for t in rows], dtype=float)   # NET of fees (engine books balance += pnl − fee)
    cfs = [float(t.get('confidence', 0) or 0) for t in rows if t.get('confidence')]
    return {'n': n, 'win_rate': round(int((pns > 0).sum()) / n * 100, 1),
            'avg_conf': round(sum(cfs) / len(cfs), 4) if cfs else 0.0,
            'per_trade_mean': round(float(np.mean(pns)), 4),
            'per_trade_median': round(float(np.median(pns)), 4),
            'net': round(float(pns.sum()), 3)}

def _compute_benchmarks(now, cur_fp, today, epoch):
    """DISPLAY-ONLY consolidated 'script/model benchmarks' for the dashboard — scoped
    current-fp / today / all-time / since-last-reset, plus run-rate projections."""
    closed = [t for t in state.trade_log if isinstance(t, dict) and t.get('status') == 'closed']
    def _ct(t): return str(t.get('close_time') or t.get('timestamp') or '')
    cohort = _bench_one([t for t in closed if t.get('cfg_fp') == cur_fp])
    daily = _bench_one([t for t in closed if _ct(t).startswith(today)])
    all_time = _bench_one(closed)
    since = _bench_one([t for t in closed if epoch and _ct(t) >= epoch]) if epoch else None
    bal = state.balance; sod = state.start_balance_today or bal
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    frac = max((now - midnight).total_seconds() / 86400.0, 1e-6)
    dn = daily['n']; proj_trades = round(dn / frac) if dn else 0
    mp = state.model_performance or {}
    return {'cohort': cohort, 'daily': daily, 'all_time': all_time, 'since_reset': since,
            'projected_daily_pnl': round((bal - sod) / frac, 3),
            'projected_trades_per_day': proj_trades,
            'projected_per_trade_daily': round(daily['per_trade_mean'] * proj_trades, 3) if dn else 0.0,
            'accuracy': round(float(mp.get('accuracy', 0) or 0), 4),
            'precision': round(float(mp.get('precision', 0) or 0), 4),
            'day_fraction': round(frac, 3), 'display_epoch': epoch or ''}

def _build_status() -> Dict:
    now=datetime.now(timezone.utc)
    days=(now-state.start_time).days+1
    state.days_running=days
    state.benchmarks['uptime_seconds']=int((now-state.start_time).total_seconds())
    closed=[t for t in state.trade_log if t.get('status')=='closed']
    winning=[t for t in closed if (float(t.get('pnl_usd',0) or 0)-float(t.get('fee_usd',0) or 0))>0]
    win_rate_all=len(winning)/len(closed) if closed else 0.0
    # DISPLAY-ONLY accuracy reset (v6): scope the dashboard win-rate to the CURRENT
    # rule-set (config_fingerprint). Drops pre-change (e.g. pre-payoff-fix) trades from the
    # *displayed* accuracy WITHOUT mutating trade_log, the benchmark counters, or any
    # trade-outcome learner — so MCP-venue sizing (L3677), the uncertainty-gate warmup
    # (L3278) and the conf-floor optimizer (L715) are byte-for-byte unaffected. Auto-resets
    # whenever the strategy config changes (same fingerprint the data-hygiene quarantine uses).
    _cur_fp=getattr(state,'config_fingerprint','')
    _closed_fp=[t for t in closed if t.get('cfg_fp')==_cur_fp]
    _win_fp=[t for t in _closed_fp if (float(t.get('pnl_usd',0) or 0)-float(t.get('fee_usd',0) or 0))>0]
    win_rate=(len(_win_fp)/len(_closed_fp)) if _closed_fp else 0.0
    today=now.strftime('%Y-%m-%d')
    today_pnl=state.daily_pnl.get(today,0.0)
    total_profit=state.balance-state.initial_investment
    return {
        "system":{"status":"running" if not state.halt_trading else "halted",
                   "version":CONFIG['VERSION'],"name":CONFIG['NAME'],
                   "days_running":days,"start_time":state.start_time.isoformat(),
                   "halt_trading":state.halt_trading},
        "wallet":{"public_key":CONFIG['PUBLIC_KEY'],
                   "sol_balance":state.onchain_balance,
                   "usdc_collateral":state.balance,
                   "free_collateral":state.available_collateral,
                   "margin_used":max(0,state.balance-state.available_collateral),
                   "unrealized_pnl":state.unrealized_pnl,
                   "available_to_withdraw":state.available_collateral,
                   "estimated_gas_sol":0.000005,
                   "estimated_gas_usd":0.000005*state.sol_price},
        "market":{"sol_price":state.sol_price,"funding_rate":state.funding_rate,
                   "regime":state.current_regime,"market":"SOL-PERP",
                   "exchange":CONFIG['PRIMARY_VENUE'].capitalize(),
                   "max_leverage":CONFIG['LEVERAGE_CAP'],
                   "hard_max_leverage":CONFIG['MAX_LEVERAGE'],
                   "sentiment_score":state.sentiment_score,"macro":state.macro_data},
        # v4.00 venue / router + microstructure card
        "venue":{"mode":CONFIG['ENGINE_MODE'],"is_paper":router.is_paper(),
                  "profile":CONFIG['ACTIVE_PROFILE'],
                  "primary":CONFIG['PRIMARY_VENUE'],"secondary":CONFIG['SECONDARY_VENUE'],
                  "active_venue":state.active_venue,"last_venue":router.last_venue,
                  "health":state.venue_health,
                  "cross_venue_basis":state.cross_venue_basis,
                  "open_interest":state.open_interest,
                  "long_short_ratio":state.long_short_ratio,
                  "liq_long":state.liq_long,"liq_short":state.liq_short},
        # v4.00 fee-drag governor card
        "governor":{"enabled":CONFIG['FEE_DRAG_GOVERNOR'],
                     "fees_today_usd":round(state.fees_today,4),
                     "daily_budget_usd":round(CONFIG['DAILY_FEE_BUDGET_PCT']*max(state.balance,1e-9),4),
                     "budget_pct":CONFIG['DAILY_FEE_BUDGET_PCT']*100,
                     "throttles":state.governor_throttles,
                     "leverage_cap":CONFIG['LEVERAGE_CAP'],
                     "rt_fee_rate_maker":router.round_trip_fee_rate(maker=True),
                     "rt_fee_rate_taker":router.round_trip_fee_rate(maker=False),
                     # v4.00 Grok-derived governors (Phase C)
                     "trade_limit":state.last_trade_limit,
                     "trades_this_cycle":len(state.cycle_trades),
                     "trade_limit_skips":state.trade_limit_skips,
                     "slippage_bps":round(state.last_slippage_bps,2),
                     "data_breaker_skips":state.data_breaker_skips},
        # NoxSSD data-drive safety pause-switch (outcome-neutral; blocks opens when the drive is offline)
        "safety":{"noxssd_guard_enabled":CONFIG.get('NOXSSD_GUARD_ENABLE',True),
                   "noxssd_ok":getattr(state,'noxssd_ok',True),
                   "noxssd_block_opens":getattr(state,'noxssd_block_opens',False),
                   "noxssd_skips":getattr(state,'noxssd_skips',0),
                   "noxssd_path":CONFIG.get('NOXSSD_MOUNT_PATH',''),
                   "noxssd_last_check":getattr(state,'noxssd_last_check','')},
        # v4.00 position-sizing card (Phase A): which Kelly policy is live + last size.
        # All three policies clip the per-trade Kelly fraction to [2%, 25%].
        "sizing":{"mode":CONFIG.get('SIZING_MODE','pessimistic'),
                   "dyn_kelly_pct":round(quant_overlay.last_dyn_kelly*100,2),
                   "pess_kelly_pct":round(quant_overlay.last_pess_kelly*100,2),
                   "kelly_cap_pct":25.0,
                   "vol_target_lev":int(getattr(quant_overlay,'last_vol_target_lev',1)),
                   "realized_vol_annual_pct":round(quant_overlay.garch_vol()*725.0*100,1),
                   "risk_throttle":round(float(getattr(state,'last_risk_throttle',1.0)),3),
                   "consecutive_losses":int(state.consecutive_losses)},
        "performance":{"initial_investment":state.initial_investment,
                        "current_balance":state.balance,
                        "total_profit_usd":total_profit,
                        "total_profit_pct":(total_profit/state.initial_investment)*100,
                        "today_pnl_usd":today_pnl,
                        "today_pnl_pct":(today_pnl/state.start_balance_today)*100 if state.start_balance_today else 0,
                        "best_day_date":state.best_day['date'],
                        "best_day_pnl_usd":state.best_day['pnl'],
                        "best_day_pnl_pct":state.best_day['pct']*100,
                        "total_trades":state.trade_count,
                        "open_positions":len(state.open_positions),
                        "closed_trades":len(closed),
                        "closed_trades_fp":len(_closed_fp),     # closed under the CURRENT rule-set
                        "peak_equity":state.peak_equity,
                        "capital_tier":state.capital_tier,
                        "last_milestone":state.benchmarks.get('last_milestone'),
                        "win_rate":win_rate*100,                # current rule-set (display reset)
                        "win_rate_all":win_rate_all*100,        # cumulative (audit / unchanged math)
                        "stats_scope":"current_rule_set",
                        "stats_fp":(_cur_fp[:8] if _cur_fp else "")},
        "models":state.model_performance,
        "benchmarks":state.benchmarks,
        # Dashboard observability (DISPLAY-ONLY) — consolidated benchmarks, per-model metrics,
        # quarantine reflection, news headlines, and the manual Reset/Checkpoint epoch.
        "benchmarks_live":_compute_benchmarks(now,_cur_fp,today,getattr(state,'display_epoch','')),
        "per_model_metrics":getattr(state,'per_model_metrics',{}),
        "news_headlines":(getattr(state,'news_headlines',[]) or [])[:5],
        "quarantine":{"last":getattr(state,'last_quarantine',{}),
                       "fingerprint":(_cur_fp[:8] if _cur_fp else ""),
                       "learn_gate":CONFIG.get('LEARN_FINGERPRINT_GATE',True)},
        "display_epoch":getattr(state,'display_epoch',''),
        "display_epoch_depth":len(getattr(state,'display_epoch_history',[]) or []),
        "data_freshness":_data_freshness(),   # v6: live-feed verification summary
        "calibration":compute_ece(),          # v8 Step-0: ECE/reliability of confidence (Phase A baseline)
        "structural":{                         # v10 Track 1: the active structural levers
            "opt_conf_floor":round(float(getattr(state,'opt_conf_floor',0.0) or 0.0),4),
            "conf_floor_active":bool(getattr(state,'opt_conf_floor',0.0)),
            "maker_rt_fee_pct":round(router.round_trip_fee_rate(maker=True)*100,4),
            "fee_hurdle_pct":round(CONFIG.get('MIN_EDGE_FEE_K',1.5)*router.round_trip_fee_rate(maker=True)*100,4),
            "ob_shadow":{k:(round(v,6) if isinstance(v,(int,float)) else v)
                          for k,v in (getattr(state,'ob_features',{}) or {}).items() if k!='ts'}},
        "realtime":{"ws_price":round(state.ws_mark_price,4),   # v9 Phase D: Aster WS real-time feed
                     "ws_funding":state.ws_funding_rate,
                     "ws_age_sec":(round(time.time()-state.ws_ts,1) if state.ws_ts else None),
                     "ws_fresh":bool(state.ws_ts and (time.time()-state.ws_ts) < CONFIG.get('ASTER_WS_STALE_SEC',5.0)),
                     "fed_into_decisions":CONFIG.get('ASTER_WS_FEED',False)},
        "ai_augmentation":state.ai_augmentation,   # v6: latest AI data-augmentation verdict
        "news_sentiment":state.news_sentiment,     # Q2: latest AI news/event-sentiment verdict
        "open_positions":state.open_positions,
        "daily_pnl_history":state.daily_pnl,
        "feature_importance":ml_ensemble.feature_importance,
    }

@app.get("/api/status")
async def get_status():
    # Defensive: never let an upstream API failure 500 the dashboard.
    # Robustness (v6): on a transient timeout/error, KEEP the last-known values instead of
    # flickering free-collateral / unrealized-PnL to $0. Only overwrite state from a
    # SUCCESSFUL upstream read; the data-freshness summary surfaces a persistently dead feed.
    _to = float(CONFIG.get('STATUS_UPSTREAM_TIMEOUT', 2.0))
    try:
        coll = await asyncio.wait_for(drift_trader.get_collateral(), timeout=_to)
        state.available_collateral = coll.get('free', state.balance)
        state.unrealized_pnl       = coll.get('unrealized_pnl', 0.0)
    except Exception:
        pass
    try:
        onchain = await asyncio.wait_for(fetch_solana_on_chain(), timeout=_to)
        state.onchain_balance = onchain.get('sol_balance', state.onchain_balance)
    except Exception:
        pass
    try:
        return _build_status()
    except Exception as e:
        logger.error(f"status build error: {e}")
        return {'error': 'status build failed', 'detail': str(e),
                'system': {'status':'running','name':CONFIG['NAME']}}

@app.get("/api/technicals")
async def get_technicals():
    """TV-rating replica + composites + extra indicator shadows (display/research only)."""
    return {"enabled": CONFIG.get('TV_RATINGS_ENABLE', True),
            "tfs": CONFIG.get('TV_TFS', []),
            "ratings": getattr(state, 'tv_ratings', {}) or {},
            "composites": {k: v for k, v in (getattr(state, 'tv_composites', {}) or {}).items()
                            if not k.startswith('_')},
            "ti": getattr(state, 'ti_features', {}) or {},
            "sr": getattr(state, 'sr_levels', {}) or {},
            "sr_feat": getattr(state, 'sr_feat', {}) or {},
            "price": float(getattr(state, 'sol_price', 0.0) or 0.0),
            "weights": CONFIG.get('TV_HOLD_WEIGHTS', {})}

@app.get("/api/trades")
async def get_trades(limit: int=100):
    return {"trades":state.trade_log[-limit:]}

@app.post("/api/display/reset")
async def display_reset():
    """DISPLAY-ONLY: snapshot the current epoch to history and start a clean view from now.
    Does NOT touch trade_log, learners, benchmark counters, or any trading state."""
    state.display_epoch_history.append(getattr(state, 'display_epoch', '') or '')
    state.display_epoch_history = state.display_epoch_history[-20:]
    state.display_epoch = datetime.now(timezone.utc).isoformat()
    return {"ok": True, "display_epoch": state.display_epoch,
            "history_depth": len(state.display_epoch_history)}

@app.post("/api/display/checkpoint")
async def display_checkpoint():
    """DISPLAY-ONLY: restore the dashboard view to before the last Reset (pop the epoch stack)."""
    state.display_epoch = (state.display_epoch_history.pop() if state.display_epoch_history else '')
    return {"ok": True, "display_epoch": state.display_epoch,
            "history_depth": len(state.display_epoch_history)}

@app.get("/api/positions")
async def get_positions():
    chain=await drift_trader.get_open_positions()
    return {"positions":state.open_positions,"blockchain_verified":chain}

@app.get("/api/price_history")
async def price_history():
    return {"history":list(state.ws_data)[-200:]}

@app.get("/api/benchmarks")
async def get_benchmarks():
    return {"benchmarks":state.benchmarks,
            "model_performance":state.model_performance,
            "rl_epsilon":rl_stop_optimizer.epsilon,
            "rl_states_mapped":len(rl_stop_optimizer.q1),
            "adaptive": {
                "consecutive_wins":   state.consecutive_wins,
                "consecutive_losses": state.consecutive_losses,
                "conf_offset":        round(state.adaptive_conf_offset, 4),
                "last_loss_time":     state.last_loss_time.isoformat() if state.last_loss_time else None,
            }}

@app.get("/api/cycle")
async def get_cycle():
    """24-hour trading cycle status + historical summaries."""
    return {
        'cycle_index':           state.cycle_index,
        'phase':                 state.cycle_phase,
        'started_utc':           state.cycle_start_time.isoformat(),
        'elapsed_seconds':       int(cycle_elapsed_seconds()),
        'remaining_seconds':     int(cycle_remaining_seconds()),
        'duration_hours':        CONFIG['CYCLE_DURATION_HOURS'],
        'winddown_minutes':      CONFIG['CYCLE_WINDDOWN_MINUTES'],
        'in_winddown':           cycle_is_in_winddown(),
        'can_open_new':          can_open_new_trade(None),
        'trades_this_cycle':     len(state.cycle_trades),
        'open_at_end':           sum(1 for t in state.cycle_trades if t.get('status') == 'open'),
        'recent_summaries':      state.cycle_summaries[-5:],
    }

@app.get("/api/quant")
async def get_quant_overlay():
    """Elite-quant overlay metrics — Bayesian, GARCH, deflated Sharpe, etc."""
    snap = quant_overlay.snapshot()
    snap['regime'] = state.current_regime
    snap['regime_gate'] = REGIME_POSTERIOR_GATE.get(state.current_regime, 0.6)
    snap['filter_lr'] = {k: {'pass': v[0], 'fail': v[1]} for k,v in FILTER_LR.items()}
    return snap

@app.get("/api/health")
async def get_api_health():
    """API connection status grouped by category."""
    now = datetime.now(timezone.utc)
    grouped: Dict[str, List[Dict]] = {}
    for key, info in state.api_health.items():
        cat = info['category']
        grouped.setdefault(cat, [])
        last_ping = info['last_ping']
        age_sec = None
        if last_ping:
            try:
                age_sec = int((now - datetime.fromisoformat(last_ping)).total_seconds())
            except Exception:
                age_sec = None
        grouped[cat].append({
            'key': key,
            'name': info['name'],
            'status': info['status'],
            'configured': info['configured'],
            'last_ping': last_ping,
            'age_seconds': age_sec,
            'last_latency_ms': info['last_latency_ms'],
            'last_error': info['last_error'],
            'success_count': info['success_count'],
            'fail_count': info['fail_count'],
        })
    return {'categories': grouped, 'server_time': now.isoformat()}

@app.post("/api/control/start")
async def start_trading():
    state.halt_trading=False
    logger.info("Trading STARTED via dashboard")
    return {"status":"started","message":f"{CONFIG['NAME']} trading enabled"}

@app.post("/api/control/stop")
async def stop_trading():
    state.halt_trading=True
    logger.info("Trading STOPPED via dashboard")
    return {"status":"stopped","message":"Trading halted"}

@app.post("/api/control/restart")
async def restart_trading():
    state.halt_trading=True
    await asyncio.sleep(2)
    state.halt_trading=False
    logger.info("Trading RESTARTED via dashboard")
    return {"status":"restarted","message":f"{CONFIG['NAME']} restarted"}

@app.post("/api/control/close_all")
async def close_all_positions():
    await router.close_position()
    state.open_positions.clear()
    logger.info("All positions CLOSED via dashboard")
    return {"status":"all_closed","venue":getattr(router,'last_venue',None)}

# Venue-funding instructions. The engine is a plain API client: it NEVER
# auto-moves user funds between wallets and venues. Deposits/withdrawals are
# performed by the user on the venue UI; these endpoints return the steps.
_VENUE_FUNDING = {
    "aster": {
        "deposit": "Deposit USDT/USDC from your wallet at https://www.asterdex.com "
                   "(Assets → Deposit). API key/secret: https://www.asterdex.com/en/api-management",
        "withdraw": "Withdraw via the Aster UI (Assets → Withdraw) at https://www.asterdex.com",
    },
    "apex": {
        "deposit": "Bridge USDC to StarkEx L2 at https://omni.apex.exchange (Deposit). "
                   "API key/secret/passphrase: https://omni.apex.exchange/keyManagement",
        "withdraw": "Withdraw from StarkEx L2 via the ApeX Omni UI at https://omni.apex.exchange",
    },
    "paper": {
        "deposit": "Paper mode: no real funds. Set DELPHI_MODE=live and fund a venue to trade real money.",
        "withdraw": "Paper mode: no real funds to withdraw.",
    },
}

def _active_venue_key() -> str:
    v = (getattr(router, "last_venue", None)
         or (CONFIG.get('PRIMARY_VENUE') if not router.is_paper() else 'paper'))
    return (v or 'paper').lower()

@app.post("/api/wallet/withdraw")
async def withdraw(amount_usdc: float, destination_wallet: str = ""):
    if amount_usdc<=0:
        raise HTTPException(400,"Amount must be positive")
    if amount_usdc>state.available_collateral:
        raise HTTPException(400,f"Insufficient free collateral: ${state.available_collateral:.4f}")
    vk = _active_venue_key()
    logger.info(f"Withdrawal requested: ${amount_usdc:.4f} USDC ({vk}) → {destination_wallet or 'primary wallet'}")
    return {"status":"manual_required","amount_usdc":amount_usdc,
            "venue":vk,"destination":destination_wallet or CONFIG['PUBLIC_KEY'],
            "note":_VENUE_FUNDING.get(vk,_VENUE_FUNDING['paper'])["withdraw"]}

@app.post("/api/wallet/deposit")
async def deposit(amount_usdc: float, source_wallet: str = ""):
    """Funding instructions only. The engine never auto-moves funds — the user
    deposits on the venue UI. Returns per-venue steps for the active venue."""
    if amount_usdc<=0:
        raise HTTPException(400,"Amount must be positive")
    vk = _active_venue_key()
    logger.info(f"Deposit instructions requested: ${amount_usdc:.4f} USDC ({vk})")
    return {"status":"manual_required","amount_usdc":amount_usdc,"venue":vk,
            "note":_VENUE_FUNDING.get(vk,_VENUE_FUNDING['paper'])["deposit"]}

# Back-compat alias for the old Drift deposit route — now returns venue steps,
# never 500s, never moves funds.
@app.post("/api/wallet/drift_deposit")
async def drift_deposit(amount_usdc: float):
    return await deposit(amount_usdc)

@app.get("/api/venues")
async def venues_status():
    """Live venue/router health: which adapters are connected, fees, active venue."""
    try:
        health = await router.health()
    except Exception as e:
        health = {"error": str(e)}
    return {
        "mode": CONFIG['ENGINE_MODE'],
        "is_paper": router.is_paper(),
        "profile": CONFIG.get('ACTIVE_PROFILE'),
        "primary": CONFIG.get('PRIMARY_VENUE'),
        "secondary": CONFIG.get('SECONDARY_VENUE'),
        "active_venue": getattr(router, "last_venue", None),
        "venues": [getattr(v, "name", str(v)) for v in getattr(router, "venues", [])],
        "leverage_cap": CONFIG.get('LEVERAGE_CAP'),
        "rt_fee_taker": router.round_trip_fee_rate(maker=False),
        "rt_fee_maker": router.round_trip_fee_rate(maker=True),
        "health": health,
    }

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    # HTTP middleware does NOT cover websockets — enforce the same session gate here.
    if _auth_enabled() and not _session_valid(websocket.cookies.get('delphi_session', '')):
        await websocket.close(code=4401)
        return
    await websocket.accept()
    connected_ws.append(websocket)
    try:
        while True:
            await asyncio.sleep(1)
            data=_build_status()
            await websocket.send_json({"type":"status_update","data":data})
    except Exception:
        if websocket in connected_ws:
            connected_ws.remove(websocket)

# ============================================================
# SECTION 24: MAIN LOOP
# ============================================================
async def run_dashboard():
    cfg=uvicorn.Config(app, host=CONFIG['DASHBOARD_HOST'],
                       port=CONFIG['DASHBOARD_PORT'], log_level="warning")
    await uvicorn.Server(cfg).serve()

async def daily_reset():
    """Resets day-roll counters at UTC midnight. Polls every 5 min instead
    of one long sleep — survives clock changes, suspends, and process delays."""
    last_reset_date = datetime.now(timezone.utc).date()
    while True:
        await asyncio.sleep(300)   # 5-min poll
        try:
            now = datetime.now(timezone.utc)
            today = now.date()
            if today != last_reset_date:
                state.start_balance_today = state.balance
                state.days_running += 1
                last_reset_date = today
                # Reset the fee-drag governor's daily budget.
                state.fees_today = 0.0
                state.fee_day = today.strftime('%Y-%m-%d')
                logger.info(f"Day {state.days_running} reset | "
                            f"Balance: ${state.balance:.4f} | "
                            f"adaptive_offset={state.adaptive_conf_offset:+.3f} | "
                            f"fee budget reset")
                try: persist_state()
                except Exception: pass
        except Exception as e:
            logger.debug(f"daily_reset error: {e}")

async def sentiment_updater():
    while True:
        try:
            state.sentiment_score=await fetch_combined_sentiment()
            logger.debug(f"Sentiment: {state.sentiment_score:.3f}")
        except Exception as e:
            logger.debug(f"Sentiment update: {e}")
        await asyncio.sleep(120)

async def ai_augmentation_loop():
    """v6 — background AI data-augmentation (free/local Ollama). Every
    LOCAL_AI_AUGMENT_INTERVAL_SEC, ask the on-box model to judge the CURRENT market data's
    quality + directional lean, and stash it on state.ai_augmentation for the dashboard + a
    down-only confidence haircut. Fully OFF the trading path; on a transient model failure it
    keeps the last good verdict (flagged stale) so the haircut disengages but the panel persists."""
    interval = int(CONFIG.get('LOCAL_AI_AUGMENT_INTERVAL_SEC', 30))
    while True:
        try:
            # Gate on populated market data so the first verdict isn't a spurious "data unknown".
            if (local_ai is not None and CONFIG.get('LOCAL_AI_AUGMENT_ENABLE', True)
                    and getattr(state, 'sol_price', 0) > 0):
                snap = build_data_snapshot(state, quant_overlay)
                aug = await local_ai.augment(snap)
                d = aug.as_dict()
                if not d.get('ok') and state.ai_augmentation.get('ok'):
                    prev = dict(state.ai_augmentation)          # keep last good verdict, mark stale
                    prev['stale'] = True; prev['degraded'] = d.get('degraded')
                    state.ai_augmentation = prev
                else:
                    d['stale'] = False
                    state.ai_augmentation = d
                    if d.get('ok'):
                        logger.debug(f"AI augment: bias={d['bias']} dq={d['data_quality']:.2f} "
                                     f"flags={d['flags']}")
        except Exception as e:
            logger.debug(f"ai_augmentation_loop: {e}")
        # v8 Phase D (anti-staleness) — event-driven cache refresh: on a high news event_risk,
        # invalidate the TTL data caches so the next decision uses FRESH data (latched once per
        # event; resets when risk subsides). Non-outcome — only freshens data.
        try:
            _ns = getattr(state, 'news_sentiment', {}) or {}
            _er = float(_ns.get('event_risk', 0.0) or 0.0) if _ns.get('ok') else 0.0
            _thr = float(CONFIG.get('NEWS_EVENT_REFRESH_RISK', 0.60))
            if _er >= _thr and not state._event_refreshed:
                _ttl_cache.clear(); state._event_refreshed = True
                logger.info(f"  [freshness] news event_risk {_er:.2f} >= {_thr} — TTL data caches "
                            f"invalidated (fresh data forced before next decision)")
            elif _er < _thr:
                state._event_refreshed = False
        except Exception:
            pass
        await asyncio.sleep(interval)

async def news_sentiment_loop():
    """Q2 — background AI news/event-sentiment (free RSS → local Ollama). Every
    LOCAL_AI_NEWS_INTERVAL_SEC, pull recent crypto headlines and have the on-box model distil a
    structured SOL near-term {sentiment, event_risk, regime, catalysts} verdict onto
    state.news_sentiment. DISPLAY-FIRST: surfaced on the dashboard, NOT yet consumed by the
    signal. Off the trading path; degrades cleanly (keeps last good verdict, flagged stale).
    This is the reliable replacement for the dead X/Reddit/Google social feeds."""
    interval = int(CONFIG.get('LOCAL_AI_NEWS_INTERVAL_SEC', 600))
    await asyncio.sleep(20)        # let model warm-up + first market fetch settle
    while True:
        try:
            if local_ai is not None and CONFIG.get('LOCAL_AI_NEWS_ENABLE', True):
                heads = await fetch_crypto_news()
                if heads:
                    nv = await local_ai.analyze_news(heads)
                    d = nv.as_dict()
                    # v7 P3 — news-vs-price divergence: bearish news while price rises over the
                    # inter-fetch window (or vice versa) flags squeeze/divergence risk. Display-only.
                    _px = getattr(state, 'sol_price', 0.0) or 0.0
                    _prev = getattr(state, '_news_prev_price', 0.0) or 0.0
                    _chg = ((_px - _prev) / _prev) if _prev > 0 else 0.0
                    state._news_prev_price = _px
                    d['price_chg_pct'] = round(_chg * 100, 3)
                    d['feeds_ok'] = int(getattr(state, 'news_feeds_ok', 0))
                    _sent = float(d.get('sentiment', 0.0) or 0.0)
                    d['divergence'] = bool(_sent * _chg < 0 and abs(_sent) > 0.3 and abs(_chg) > 0.003)
                    if not d.get('ok') and state.news_sentiment.get('ok'):
                        prev = dict(state.news_sentiment)
                        prev['stale'] = True; prev['degraded'] = d.get('degraded')
                        state.news_sentiment = prev
                    else:
                        d['stale'] = False
                        state.news_sentiment = d
                        if d.get('ok'):
                            logger.info(f"AI news: sent={d['sentiment']:+.2f} "
                                        f"event_risk={d['event_risk']:.2f} regime={d['regime']} "
                                        f"catalysts={d['catalysts']}")
        except Exception as e:
            logger.debug(f"news_sentiment_loop: {e}")
        await asyncio.sleep(interval)

async def aster_ws_loop():
    """v9 Phase D — real-time Aster mark-price + funding via WebSocket. Pushes fresh price/funding
    to state.ws_* for the dashboard + (when ASTER_WS_FEED) the decision path. Auto-reconnects with
    backoff; the REST poll stays the hot backstop so a WS drop never stalls trading. Non-outcome
    while ASTER_WS_FEED=False (display-first)."""
    if not CONFIG.get('ASTER_WS_ENABLE', True):
        return
    sym = str(CONFIG.get('ASTER_WS_SYMBOL', 'solusdt')).lower()
    url = f"{CONFIG.get('ASTER_WS_URL', 'wss://fstream.asterdex.com')}/ws/{sym}@markPrice@1s"
    backoff = 1
    await asyncio.sleep(8)        # let the engine boot first
    while True:
        try:
            sess = await get_http()
            async with sess.ws_connect(url, timeout=10, heartbeat=15) as ws:
                backoff = 1
                logger.info(f"  [aster-ws] connected (real-time price+funding) {url}")
                record_api('aster', True, 0.0)
                async for m in ws:
                    if m.type == aiohttp.WSMsgType.TEXT:
                        try:
                            d = json.loads(m.data)
                            px = float(d.get('p', 0) or 0)
                            if px > 0:
                                state.ws_mark_price = px
                                state.ws_funding_rate = float(d.get('r', 0) or 0)
                                state.ws_ts = time.time()
                        except Exception:
                            pass
                    elif m.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break
        except Exception as e:
            logger.debug(f"aster_ws_loop reconnect ({type(e).__name__}): {str(e)[:60]}")
        await asyncio.sleep(min(backoff, 30))
        backoff = min(backoff * 2, 30)

async def compute_ob_features():
    """v10 Track 2b — order-book depth SHADOW features: microprice deviation, top-N depth imbalance,
    spread (bps), and imbalance velocity (the dynamic part the engine lacks — its OFI is a candle
    proxy). LOG-ONLY; validated by validate_winprob.py before any wiring. Non-outcome."""
    try:
        ob = await router.get_orderbook(depth=int(CONFIG.get('OB_DEPTH', 20)))
        bids = ob.get('bids', []); asks = ob.get('asks', [])
        if not bids or not asks:
            return
        bid_px, ask_px = float(bids[0][0]), float(asks[0][0])
        mid = (bid_px + ask_px) / 2.0
        if mid <= 0:
            return
        bq = sum(float(q) for _, q in bids); aq = sum(float(q) for _, q in asks)
        microprice = (bid_px * aq + ask_px * bq) / (aq + bq) if (aq + bq) > 0 else mid
        imb = (bq - aq) / (bq + aq) if (bq + aq) > 0 else 0.0
        prev_imb = (state.ob_features or {}).get('ob_depth_imb', 0.0)
        state.ob_features = {
            'ob_microprice_dev': (microprice - mid) / mid,
            'ob_depth_imb': imb,
            'ob_spread_bps': (ask_px - bid_px) / mid * 1e4,
            'ob_imb_velocity': imb - prev_imb,
            'ts': time.time(),
        }
    except Exception as e:
        logger.debug(f"compute_ob_features: {e}")

async def ob_features_loop():
    """Background poll for the order-book shadow features (Track 2b). Off the trade path."""
    if not CONFIG.get('OB_FEATURES_ENABLE', True):
        return
    interval = float(CONFIG.get('OB_POLL_SEC', 3.0))
    await asyncio.sleep(10)
    while True:
        await compute_ob_features()
        await asyncio.sleep(interval)

async def compute_onchain_features():
    """v10 Track 2 — on-chain network-activity SHADOW features (Solana TPS level + trend) via the
    public RPC getRecentPerformanceSamples. A DIFFERENT signal class (network interest, not price
    microstructure); LOG-ONLY, validated by validate_winprob.py before any wiring."""
    try:
        payload = {"jsonrpc": "2.0", "id": 1, "method": "getRecentPerformanceSamples", "params": [5]}
        s = await get_http()
        async with s.post(CONFIG['SOLANA_RPC_URL'], json=payload,
                          timeout=aiohttp.ClientTimeout(total=6)) as r:
            d = await r.json()
        samples = d.get('result', [])
        if not samples:
            return

        def _tps(x):
            p = x.get('samplePeriodSecs', 60) or 60
            return (x.get('numTransactions', 0) or 0) / p
        recent = _tps(samples[0])
        older = sum(_tps(x) for x in samples[1:]) / max(len(samples) - 1, 1)
        state.onchain_features = {
            'oc_tps': round(recent, 1),
            'oc_tps_trend': round((recent - older) / older, 4) if older > 0 else 0.0,
            'ts': time.time(),
        }
    except Exception as e:
        logger.debug(f"compute_onchain_features: {e}")

async def tv_ratings_loop():
    """SHADOW (log-only): TradingView Technical-Rating replica per timeframe + alignment composites.
    Native klines per TF via router.klines on tiered TTLs (trivial API load). Computes the
    continuous gauge value (the 'needle'), category, per-indicator votes, the hold-matched and
    equal-weight composites, alignment count, strong-fraction, persistence — plus the extra
    indicator shadows (SuperTrend 15m, VWAP-deviation 1m, daily-pivot distance) from the SAME
    candles. NOTHING here affects trading; the validator gates any future wiring."""
    if not CONFIG.get('TV_RATINGS_ENABLE', True):
        return
    try:
        import tv_ratings as _tv
    except Exception as e:
        logger.warning(f"tv_ratings module unavailable — TV shadow disabled: {e}")
        return
    ttls = CONFIG.get('TV_TTLS', {})
    weights = CONFIG.get('TV_HOLD_WEIGHTS', {})
    core = ('1m', '5m', '15m', '30m', '1h')
    last_fetch: Dict = {}
    dfs: Dict = {}
    await asyncio.sleep(10)
    while True:
        try:
            for tf in CONFIG.get('TV_TFS', []):
                if time.time() - last_fetch.get(tf, 0.0) < float(ttls.get(tf, 300)):
                    continue
                kl = await router.klines(tf, int(CONFIG.get('TV_KLINES_LIMIT', 300)))
                last_fetch[tf] = time.time()
                if not kl:
                    continue
                df = _tv.klines_to_df(kl)
                dfs[tf] = df
                r = _tv.compute_tv_rating(df)
                if r.get('rating') is None:
                    continue
                state.tv_ratings[tf] = {'rating': r['rating'], 'category': r['category'],
                                        'zone': _tv.zone(r['rating']), 'osc': r['osc_rating'],
                                        'ma': r['ma_rating'], 'votes': r['votes'],
                                        'n': r['n_votes'], 'ts': time.time()}
            # --- alignment composites over the core (hold-relevant) TFs ---
            avail = [tf for tf in core if tf in state.tv_ratings]
            if avail:
                rs = {tf: float(state.tv_ratings[tf]['rating']) for tf in avail}
                comp_eq = float(np.mean(list(rs.values())))
                wsum = sum(float(weights.get(tf, 0.2)) for tf in avail)
                comp_hold = (sum(float(weights.get(tf, 0.2)) * rs[tf] for tf in avail) / wsum) if wsum else comp_eq
                sgn = 1 if comp_hold >= 0.1 else (-1 if comp_hold <= -0.1 else 0)
                align = sum(1 for tf in avail if abs(rs[tf]) >= 0.1 and np.sign(rs[tf]) == sgn) if sgn else 0
                strong = sum(1 for tf in avail if abs(rs[tf]) >= 0.5) / len(avail)
                prev = state.tv_composites or {}
                since = float(prev.get('_since', time.time()))
                if sgn == 0 or sgn != int(prev.get('_sign', 0)):
                    since = time.time()
                state.tv_composites = {'tv_comp_eq': round(comp_eq, 4),
                                       'tv_comp_hold': round(comp_hold, 4),
                                       'tv_align_count': int(align),
                                       'tv_strong_frac': round(strong, 4),
                                       'tv_persist_sec': (round(time.time() - since, 1) if sgn else 0.0),
                                       '_sign': sgn, '_since': since, 'ts': time.time()}
            # --- extra indicator shadows from the same candles (no extra fetches) ---
            try:
                ti = {}
                if '15m' in dfs: ti['ti_supertrend'] = _tv.supertrend_state(dfs['15m'])
                if '1m' in dfs:  ti['ti_vwap_dev'] = _tv.vwap_deviation(dfs['1m'])
                if '1d' in dfs and state.sol_price:
                    ti['ti_pivot_dist'] = _tv.pivot_distance(dfs['1d'], float(state.sol_price))
                if '1m' in dfs:
                    ti['ti_ema_cross_age'] = _tv.ema_cross_age(dfs['1m'])   # signed fresh-flip age
                    ti['ti_mr_zscore'] = _tv.mr_zscore(dfs['1m'])           # MR stretch (AUC .531)
                if ti:
                    ti['ts'] = time.time(); state.ti_features = ti
            except Exception:
                pass
            # --- support/resistance ladder (same candles; display + shadow, log-only) ---
            try:
                if '1m' in dfs and state.sol_price:
                    import sr_levels as _sr
                    _lv = _sr.compute_sr_levels(dfs['1m'], dfs.get('1h'), dfs.get('1d'),
                                                float(state.sol_price))
                    if _lv.get('supports') or _lv.get('resistances'):
                        _lv['ts'] = time.time()
                        state.sr_levels = _lv
                        state.sr_feat = _sr.sr_features(_lv, float(state.sol_price))
            except Exception as _se:
                logger.debug(f"sr_levels: {_se}")
        except Exception as e:
            logger.debug(f"tv_ratings_loop: {e}")
        await asyncio.sleep(float(CONFIG.get('TV_POLL_SEC', 20.0)))

async def onchain_features_loop():
    """Background poll for the on-chain shadow features (Track 2). Off the trade path; slow (60s)."""
    if not CONFIG.get('ONCHAIN_FEATURES_ENABLE', True):
        return
    interval = float(CONFIG.get('ONCHAIN_POLL_SEC', 60.0))
    await asyncio.sleep(15)
    while True:
        await compute_onchain_features()
        await asyncio.sleep(interval)

async def noxssd_guard_loop():
    """SAFETY pause-switch — validate the NoxSSD data drive is mounted + R(/W)-accessible.
    On sustained loss of access (FAIL_DEBOUNCE consecutive) → block NEW opens (open positions
    keep being managed/exited); on sustained recovery (RECOVER_DEBOUNCE consecutive) → auto-resume.
    Outcome-NEUTRAL: not in _config_fingerprint, so it never resets the trade-outcome cohort."""
    if not CONFIG.get('NOXSSD_GUARD_ENABLE', True):
        return
    interval  = float(CONFIG.get('NOXSSD_GUARD_POLL_SEC', 20.0))
    fail_need = int(CONFIG.get('NOXSSD_FAIL_DEBOUNCE', 2))
    rec_need  = int(CONFIG.get('NOXSSD_RECOVER_DEBOUNCE', 2))
    fails = oks = 0
    await asyncio.sleep(5)
    while True:
        try:
            mount = CONFIG.get('NOXSSD_MOUNT_PATH', '/Volumes/NoxSSD')
            data_path = CONFIG.get('NOXSSD_DATA_PATH', '') or mount
            need = os.R_OK | (os.W_OK if CONFIG.get('NOXSSD_REQUIRE_WRITE', True) else 0)
            accessible = bool(os.path.ismount(mount) and os.access(data_path, need))
            state.noxssd_ok = accessible
            state.noxssd_last_check = datetime.now(timezone.utc).isoformat()
            if accessible:
                oks += 1; fails = 0
                if state.noxssd_block_opens and oks >= rec_need:
                    state.noxssd_block_opens = False
                    logger.warning(f"[noxssd-guard] {mount} accessible again ({oks}x) — RESUMING opens")
            else:
                fails += 1; oks = 0
                if not state.noxssd_block_opens and fails >= fail_need:
                    state.noxssd_block_opens = True
                    logger.warning(f"[noxssd-guard] {mount} NOT accessible ({fails}x) — BLOCKING new "
                                   f"opens (open positions still managed/exitable)")
        except Exception as e:
            logger.debug(f"noxssd_guard_loop: {e}")
        await asyncio.sleep(interval)

async def venue_health_check():
    """v4.00 — periodic venue health monitor (replaces the Drift WS monitor).
    Polls the router's per-venue health (price reachability + latency + trade
    capability), records it to state for the dashboard, and re-initializes the
    router if BOTH venues go dark. The router's own per-call circuit breakers
    handle transient failover; this is the slow background liveness sweep."""
    consecutive_fails = 0
    while True:
        await asyncio.sleep(30)
        try:
            h = await router.health()
            state.venue_health = h
            state.active_venue = h.get('active_data_venue', router.last_venue)
            venues_ok = [v for v in h.get('venues', []) if v.get('ok')]
            for v in h.get('venues', []):
                key = v.get('venue')
                if key in ('aster', 'apex'):
                    record_api(key, bool(v.get('ok')),
                               float(v.get('latency_ms', 0) or 0),
                               '' if v.get('ok') else 'health probe down')
            if venues_ok:
                consecutive_fails = 0
            else:
                consecutive_fails += 1
                logger.warning(f"venue_health: no venue healthy ({consecutive_fails}x)")
                if consecutive_fails >= 3:
                    logger.warning("venue_health: 3 consecutive sweeps with all venues "
                                   "down — re-initializing router")
                    try:
                        await router.connect()
                        consecutive_fails = 0
                    except Exception as ie:
                        logger.error(f"router re-init failed: {ie}")
        except Exception as e:
            consecutive_fails += 1
            logger.debug(f"venue_health check error: {e}")

async def state_persistence_loop():
    """Background loop — persists state every 60s as a belt-and-suspenders
    backup to the per-trade saves. Ensures latest state on crash."""
    while True:
        await asyncio.sleep(60)
        try: persist_state()
        except Exception as e: logger.debug(f"persistence loop: {e}")

def _macos_notify(title: str, message: str):
    """Best-effort macOS notification (no-op off macOS / if disabled)."""
    try:
        import subprocess, shlex
        safe_t = title.replace('"', "'"); safe_m = message.replace('"', "'")
        subprocess.run(
            ["osascript", "-e", f'display notification "{safe_m}" with title "{safe_t}"'],
            check=False, capture_output=True, timeout=5)
    except Exception:
        pass

async def capital_tier_monitor():
    """v4.00 — track equity growth and fire a milestone notification (dashboard
    banner + macOS notification + log) the first time each CAPITAL_TIERS level is
    crossed. Debounced via state.capital_tier (the highest tier already hit);
    persisted in delphi_state.json so restarts don't re-fire old milestones."""
    while True:
        await asyncio.sleep(20)
        try:
            eq = float(state.balance)
            if eq > state.peak_equity:
                state.peak_equity = eq
            # Find the highest tier index crossed.
            hit = state.capital_tier
            for i, lvl in enumerate(CAPITAL_TIERS):
                if eq >= lvl:
                    hit = i + 1
            if hit > state.capital_tier:
                for i in range(state.capital_tier, hit):
                    lvl = CAPITAL_TIERS[i]
                    growth = (eq / max(state.initial_investment, 1e-9) - 1.0) * 100.0
                    msg = (f"Equity ${eq:,.2f} crossed ${lvl:,} "
                           f"(+{growth:.0f}% from ${state.initial_investment:,.2f})")
                    logger.info(f"=== CAPITAL MILESTONE: {msg} ===")
                    state.benchmarks['last_milestone'] = {
                        'tier_usd': lvl, 'equity': round(eq, 2),
                        'growth_pct': round(growth, 1),
                        'at': datetime.now(timezone.utc).isoformat()}
                    _macos_notify(f"{CONFIG['NAME']} — Milestone", msg)
                state.capital_tier = hit
                try: persist_state()
                except Exception: pass
        except Exception as e:
            logger.debug(f"capital_tier_monitor: {e}")

# ============================================================
# 24-HOUR TRADING CYCLE COORDINATOR
# ============================================================
def cycle_elapsed_seconds() -> float:
    return (datetime.now(timezone.utc) - state.cycle_start_time).total_seconds()

def cycle_remaining_seconds() -> float:
    total = CONFIG['CYCLE_DURATION_HOURS'] * 3600
    return max(0.0, total - cycle_elapsed_seconds())

def cycle_is_in_winddown() -> bool:
    return cycle_remaining_seconds() <= CONFIG['CYCLE_WINDDOWN_MINUTES'] * 60

def can_open_new_trade(reason_holder: Optional[List[str]] = None) -> bool:
    """Single source of truth for whether a new position can be opened.
    Used by both the main loop's pre-signal gate AND execute_trade itself."""
    if state.cycle_phase != 'active':
        if reason_holder is not None: reason_holder.append(f"cycle_phase={state.cycle_phase}")
        return False
    if cycle_is_in_winddown():
        if reason_holder is not None: reason_holder.append("cycle_winding_down")
        return False
    # Dynamic trade-limit governor (#17): regime/vol-aware cap on opens-per-cycle.
    # Only bites at high cadence — never blocks the early trades of a cycle.
    if CONFIG.get('DYNAMIC_TRADE_LIMIT', True):
        lim = dynamic_trade_limit()
        opened = len(state.cycle_trades)
        if opened >= lim:
            state.trade_limit_skips += 1
            if reason_holder is not None:
                reason_holder.append(f"trade_limit {opened}/{lim}")
            return False
    return True

def _summarize_cycle() -> Dict:
    """Build the quant + qual cycle summary, emitted at cycle end."""
    started = state.cycle_start_time
    ended   = datetime.now(timezone.utc)
    trades  = state.cycle_trades
    closed  = [t for t in trades if t.get('status') == 'closed']
    wins    = [t for t in closed if (t.get('pnl_usd',0) - t.get('fee_usd',0)) > 0]
    losses  = [t for t in closed if (t.get('pnl_usd',0) - t.get('fee_usd',0)) <= 0]
    pnls    = [(t.get('pnl_usd',0) - t.get('fee_usd',0)) for t in closed]
    gross   = sum(t.get('pnl_usd',0) for t in closed)
    fees    = sum(t.get('fee_usd',0) for t in closed)
    net     = gross - fees
    best    = max(pnls) if pnls else 0.0
    worst   = min(pnls) if pnls else 0.0
    avg     = (sum(pnls) / len(pnls)) if pnls else 0.0
    wr      = len(wins) / len(closed) if closed else 0.0
    open_count = sum(1 for t in trades if t.get('status') == 'open')
    return {
        'cycle_index':       state.cycle_index,
        'started_utc':       started.isoformat(),
        'ended_utc':         ended.isoformat(),
        'duration_hours':    round((ended - started).total_seconds()/3600.0, 3),
        'trades_opened':     len(trades),
        'trades_closed':     len(closed),
        'trades_open_at_end': open_count,
        'wins':              len(wins),
        'losses':            len(losses),
        'win_rate':          round(wr, 4),
        'gross_pnl_usd':     round(gross, 4),
        'fees_usd':          round(fees, 4),
        'net_pnl_usd':       round(net, 4),
        'avg_pnl_per_trade': round(avg, 4),
        'best_trade_usd':    round(best, 4),
        'worst_trade_usd':   round(worst, 4),
        'end_balance':       round(state.balance, 4),
        'max_drawdown_pct':  round(getattr(quant_overlay, 'max_drawdown', 0.0) * 100, 3),
        'peak_balance':      round(getattr(quant_overlay, 'peak_balance', state.balance), 4),
        'bocpd_halts':       getattr(quant_overlay, 'bocpd_halts', 0),
        'oracle_arbs':       getattr(quant_overlay, 'oracle_arb_signals', 0),
        'bayes_passes':      getattr(quant_overlay, 'bayes_passes', 0),
        'bayes_blocks':      getattr(quant_overlay, 'bayes_blocks', 0),
        'deflated_sharpe':   round(getattr(quant_overlay, 'last_deflated_sr', 0.0), 4),
        'ml_retrains':       state.benchmarks.get('ml_retrain_count', 0),
        'rl_updates':        state.benchmarks.get('rl_update_count', 0),
        'circuit_breakers':  state.benchmarks.get('circuit_breaker_count', 0),
    }

async def intercycle_optimization(summary: Dict):
    """Comprehensive between-cycle optimization. Runs after wind-down (all
    positions naturally closed) and before the next cycle starts.

    Steps:
      1. Fresh OHLCV + sentiment + macro pull (data refresh)
      2. Full ML+DL retrain on the latest 500 bars (quantile labels)
      3. Walk-forward validation: train on first 80%, evaluate on last 20%,
         publish out-of-sample metrics to state.model_performance
      4. RL Stop/TP optimizer offline retrain + pickle save
      5. RL Regime classifier online fine-tune
      6. HMM regime detector re-fit on the latest features
      7. GARCH(1,1) auto-recalibrate from fresh returns
      8. Bayesian filter LR recalibration — adjust FILTER_LR based on
         which filters predicted winning trades in the just-finished cycle
      9. Reset per-cycle counters (BOCPD halts, oracle arbs, bayes pass/block)
     10. Adaptive confidence offset: decay back toward 0 by 25% per cycle
         (don't carry old loss bias forever)
     11. Information Coefficient buffer half-life: drop oldest 50% of samples
     12. State snapshot persisted to disk with cycle summary attached
    """
    cyc = summary['cycle_index']
    logger.info(f"=== Inter-cycle optimization #{cyc} STARTING ===")
    t0 = time.time()
    try:
        # 1. Data refresh — pull fresh bars + sentiment + macro
        logger.info("  [1/12] Refreshing market + macro + sentiment data...")
        df = await update_market_data()
        if df.empty or len(df) < 100:
            logger.warning(f"  insufficient data ({len(df)} bars) — using last good")
            df = pd.DataFrame()

        # 2. Full ML+DL retrain
        if not df.empty:
            logger.info(f"  [2/12] Training ML+DL on {len(df)} bars...")
            await train_models(df)

        # 3. Walk-forward validation (out-of-sample performance)
        if not df.empty and len(df) >= 200:
            logger.info("  [3/12] Walk-forward validation (80/20 split)...")
            try:
                X_all, y_all = build_features(df)
                if len(X_all) >= 100:
                    # BUG-1b fix: do NOT re-fit on 80% (that downgraded the live model
                    # to partial data with a leaky meta). Score honestly via purged
                    # walk-forward; the full-data fit from train_models(df) above stands.
                    perf = ml_ensemble.evaluate_purged(
                        X_all, y_all, embargo=CONFIG.get('EVAL_EMBARGO_BARS', 5),
                        n_splits=CONFIG.get('META_OOF_SPLITS', 3))
                    state.model_performance = perf
                    logger.info(f"      walk-forward OOS (purged): accuracy={perf['accuracy']:.3f} "
                                f"precision={perf['precision']:.3f} "
                                f"f1={perf['f1']:.3f}")
            except Exception as e:
                logger.warning(f"      walk-forward error: {e}")

        # 4. RL Stop/TP optimizer offline retrain + save
        logger.info("  [4/12] RL Stop/TP optimizer offline retrain + save...")
        try:
            rl_stop_optimizer._offline_retrain()
            rl_stop_optimizer.save()
        except Exception as e:
            logger.debug(f"      rl_stop offline: {e}")

        # 5. RL Regime classifier fine-tune
        if not df.empty:
            logger.info("  [5/12] RL Regime classifier fine-tune...")
            try:
                feats = []; labs = []
                for _, row in df.dropna(subset=['hurst', 'adx']).iterrows():
                    f = np.zeros(32)
                    f[0] = float(row.get('hurst', 0.5))
                    f[1] = float(row.get('adx', 25)) / 100.0
                    f[2] = float(row.get('ewma_vol', 0.02))
                    f[3] = float(row.get('vpin', 0.5))
                    feats.append(f)
                    h = float(row.get('hurst', 0.5)); adx = float(row.get('adx', 25))
                    labs.append(0 if h > 0.6 and adx > 30 else (2 if h < 0.4 else 1))
                if len(feats) > 20:
                    rl_regime_classifier.fit(np.array(feats), np.array(labs))
            except Exception as e:
                logger.debug(f"      regime classifier fine-tune: {e}")

        # 6. HMM regime detector re-fit
        if not df.empty and len(df) >= 100:
            logger.info("  [6/12] HMM regime detector re-fit...")
            try:
                feat_cols = ['returns','adx','ewma_vol','vpin','vol_ratio']
                avail = [c for c in feat_cols if c in df.columns]
                if avail:
                    mat = df[avail].dropna().values
                    if 'adx' in avail:
                        mat[:, avail.index('adx')] /= 100.0
                    hmm_detector.fitted = False  # force fresh fit
                    hmm_detector.fit(mat)
            except Exception as e:
                logger.debug(f"      HMM refit: {e}")

        # 7. GARCH auto-recalibrate
        logger.info("  [7/12] GARCH(1,1) auto-recalibrate...")
        try:
            if len(quant_overlay.returns) >= 50:
                r = np.array(list(quant_overlay.returns), dtype=float)
                quant_overlay.garch_sigma2 = float(np.var(r, ddof=1))
                quant_overlay.last_garch_vol = float(np.sqrt(quant_overlay.garch_sigma2))
                quant_overlay._cf_cvar_rev = -1   # invalidate memo
                quant_overlay._pess_kelly_rev = -1
        except Exception as e:
            logger.debug(f"      GARCH recal: {e}")

        # 8. Bayesian filter LR recalibration from cycle outcomes
        logger.info("  [8/12] Bayesian filter LR recalibration from cycle outcomes...")
        try:
            cycle_closed = [t for t in state.cycle_trades if t.get('status') == 'closed']
            if len(cycle_closed) >= 5:
                wins = [t for t in cycle_closed if (t.get('pnl_usd',0) - t.get('fee_usd',0)) > 0]
                win_rate = len(wins) / len(cycle_closed)
                # Gentle adjustment: scale LR_pass toward observed win rate.
                # If win_rate > 0.55, marginally increase LR_pass (trust filters more).
                # If win_rate < 0.45, marginally decrease LR_pass (trust less).
                if win_rate > 0.55:
                    factor = 1.0 + min((win_rate - 0.5) * 0.2, 0.05)  # max +5%
                elif win_rate < 0.45:
                    factor = 1.0 - min((0.5 - win_rate) * 0.2, 0.05)  # max -5%
                else:
                    factor = 1.0
                if abs(factor - 1.0) > 1e-6:
                    for name in list(FILTER_LR.keys()):
                        lr_p, lr_f = FILTER_LR[name]
                        new_lr_p = float(np.clip(lr_p * factor, 1.05, 5.0))
                        FILTER_LR[name] = (new_lr_p, lr_f)
                    logger.info(f"      LR_pass × {factor:.4f} (cycle WR {win_rate*100:.1f}%)")
        except Exception as e:
            logger.debug(f"      filter LR recal: {e}")

        # 9. Reset per-cycle counters
        logger.info("  [9/12] Resetting per-cycle counters...")
        try:
            quant_overlay.bocpd_halts = 0
            quant_overlay.oracle_arb_signals = 0
            quant_overlay.bayes_passes = 0
            quant_overlay.bayes_blocks = 0
            quant_overlay.cusum_blocks = 0
            state.benchmarks['circuit_breaker_count'] = 0
        except Exception as e:
            logger.debug(f"      counter reset: {e}")

        # 10. Adaptive confidence offset decay (don't carry old loss bias forever)
        logger.info("  [10/12] Adaptive confidence offset decay...")
        try:
            state.adaptive_conf_offset *= 0.75  # decay 25% per cycle
            state.consecutive_losses = 0
            state.consecutive_wins = 0
            state.last_loss_time = None
        except Exception as e:
            logger.debug(f"      adaptive decay: {e}")

        # 11. IC buffer half-life (drop oldest half)
        logger.info("  [11/12] IC buffer half-life...")
        try:
            n = len(quant_overlay.ic_forecasts)
            if n > 30:
                keep = n // 2
                quant_overlay.ic_forecasts = deque(
                    list(quant_overlay.ic_forecasts)[-keep:],
                    maxlen=quant_overlay.ic_forecasts.maxlen)
                quant_overlay.ic_realized = deque(
                    list(quant_overlay.ic_realized)[-keep:],
                    maxlen=quant_overlay.ic_realized.maxlen)
        except Exception as e:
            logger.debug(f"      IC half-life: {e}")

        # 12. Snapshot to disk
        logger.info("  [12/12] Persisting state + cycle summary...")
        _safe(persist_state)

        elapsed = time.time() - t0
        logger.info(f"=== Inter-cycle optimization #{cyc} COMPLETE in {elapsed:.1f}s ===")
    except Exception as e:
        logger.error(f"intercycle_optimization #{cyc} fatal: {e}", exc_info=True)

async def cycle_coordinator():
    """Background loop. Drives cycle_phase transitions every 30s:
       active -> winding_down (last 30 min): no new opens
       winding_down -> summarizing (when all positions closed OR timeout)
       summarizing -> active (next cycle begins, models retrained)"""
    while True:
        await asyncio.sleep(30)
        try:
            phase = state.cycle_phase
            remaining = cycle_remaining_seconds()
            if phase == 'active' and remaining <= CONFIG['CYCLE_WINDDOWN_MINUTES'] * 60:
                state.cycle_phase = 'winding_down'
                logger.info(f"=== Cycle {state.cycle_index} entering WIND-DOWN "
                            f"({CONFIG['CYCLE_WINDDOWN_MINUTES']} min remaining; "
                            f"no new opens). Open positions left to close: "
                            f"{len(state.open_positions)} ===")
            elif phase == 'winding_down':
                # Transition when all positions closed naturally, OR when
                # remaining time + winddown buffer fully elapsed.
                if len(state.open_positions) == 0 or remaining <= -CONFIG['CYCLE_WINDDOWN_MINUTES']*60:
                    state.cycle_phase = 'summarizing'
                    logger.info(f"=== Cycle {state.cycle_index} entering SUMMARY ===")
            elif phase == 'summarizing':
                # Build summary, retain it, retrain models, start next cycle
                summary = _summarize_cycle()
                state.cycle_summaries.append(summary)
                # Bound retention
                if len(state.cycle_summaries) > CONFIG['CYCLE_SUMMARY_RETAIN']:
                    state.cycle_summaries = state.cycle_summaries[-CONFIG['CYCLE_SUMMARY_RETAIN']:]
                logger.info(
                    f"=== CYCLE {summary['cycle_index']} SUMMARY ===\n"
                    f"  Duration:        {summary['duration_hours']:.2f} h\n"
                    f"  Trades opened:   {summary['trades_opened']}\n"
                    f"  Trades closed:   {summary['trades_closed']}\n"
                    f"  Wins / Losses:   {summary['wins']} / {summary['losses']}\n"
                    f"  Win rate:        {summary['win_rate']*100:.1f}%\n"
                    f"  Gross PnL:       ${summary['gross_pnl_usd']:+.4f}\n"
                    f"  Fees:            ${summary['fees_usd']:.4f}\n"
                    f"  Net PnL:         ${summary['net_pnl_usd']:+.4f}\n"
                    f"  Avg / trade:     ${summary['avg_pnl_per_trade']:+.4f}\n"
                    f"  Best / worst:    ${summary['best_trade_usd']:+.4f} / ${summary['worst_trade_usd']:+.4f}\n"
                    f"  End balance:     ${summary['end_balance']:.4f}\n"
                    f"  Deflated Sharpe: {summary['deflated_sharpe']:.4f}\n"
                    f"  Peak / max DD:   ${summary['peak_balance']:.4f} / {summary['max_drawdown_pct']:.2f}%\n"
                    f"  ML retrains:     {summary['ml_retrains']}\n"
                    f"  Oracle arbs:     {summary['oracle_arbs']}\n"
                    f"=== End Cycle {summary['cycle_index']} ==="
                )
                _safe(persist_state)
                # Comprehensive inter-cycle optimization (walk-forward + retrain + recalibrate)
                await intercycle_optimization(summary)
                # Start next cycle
                state.cycle_index += 1
                state.cycle_start_time = datetime.now(timezone.utc)
                state.cycle_trades = []
                state.cycle_phase = 'active'
                logger.info(f"=== CYCLE {state.cycle_index} STARTED (cleared trades, ready for new opens) ===")
        except Exception as e:
            logger.warning(f"cycle_coordinator error: {e}", exc_info=True)

async def main():
    logger.info("="*60)
    logger.info(f"  {CONFIG['NAME']} — STARTING")
    logger.info(f"  Profile : {CONFIG['ACTIVE_PROFILE']}  |  Mode: {CONFIG['ENGINE_MODE'].upper()}")
    logger.info(f"  Venues  : {CONFIG['PRIMARY_VENUE']} (primary) → {CONFIG['SECONDARY_VENUE']} (failover)")
    logger.info(f"  Leverage: cap {CONFIG['LEVERAGE_CAP']}x (hard max {CONFIG['MAX_LEVERAGE']}x)")
    logger.info(f"  Fee budget: {CONFIG['DAILY_FEE_BUDGET_PCT']*100:.0f}% equity/day")
    logger.info(f"  Balance : ${CONFIG['STARTING_BALANCE']:.2f}")
    logger.info(f"  Dashboard: http://localhost:{CONFIG['DASHBOARD_PORT']}")
    logger.info("="*60)

    restore_state()            # compute fingerprint + quarantine FIRST (may archive RL pkl)
    if not state.config_fingerprint:           # fresh start (no state file): still tag the live rule-set
        state.config_fingerprint = _config_fingerprint()
    rl_stop_optimizer.load()   # cold-starts if the pkl was just quarantined
    try: fit_confidence_calibration()   # v9 Phase A: fit the calibrator from accrued history at boot
    except Exception: pass
    try: optimize_confidence_floor()    # activate the conf-floor optimizer from accrued cohort at boot
    except Exception: pass              # (else it only fires on the next trade-count %N hook)
    router_ok = await router.connect()
    if not router_ok:
        logger.warning("Router connect returned False — paper fills only")
    # Surface live trade capability + the achievable-trade path up front.
    if router.is_paper():
        logger.info("ENGINE_MODE=paper → simulated fills on REAL venue prices "
                    "(no API keys needed). This is the first-trade path.")
    else:
        live = [v.name for v in router.venues if getattr(v, 'can_trade', False)]
        logger.info(f"ENGINE_MODE={CONFIG['ENGINE_MODE']} → venues with trade creds: "
                    f"{live or 'NONE (add keys to .env) — falling back to paper fills'}")

    # Background tasks — wrap each so a child crash logs and dies cleanly
    # without taking down the whole event loop.
    def _bg_done(t: asyncio.Task):
        if t.cancelled(): return
        exc = t.exception()
        if exc is not None:
            logger.error(f"background task {t.get_name()} died: {exc}", exc_info=exc)

    for coro, name in [
        (run_dashboard(),         'dashboard'),
        (daily_reset(),           'daily_reset'),
        (sentiment_updater(),     'sentiment'),
        (ai_augmentation_loop(),  'ai_augment'),
        (news_sentiment_loop(),   'news_sentiment'),
        (aster_ws_loop(),         'aster_ws'),
        (ob_features_loop(),      'ob_features'),
        (onchain_features_loop(), 'onchain_features'),
        (tv_ratings_loop(),       'tv_ratings'),
        (noxssd_guard_loop(),     'noxssd_guard'),
        (venue_health_check(),    'venue_health'),
        (capital_tier_monitor(),  'capital_monitor'),
        (state_persistence_loop(),'state_persist'),
        (cycle_coordinator(),     'cycle_coord'),
    ]:
        t = asyncio.create_task(coro, name=name)
        t.add_done_callback(_bg_done)

    candle_count=0
    while True:
        try:
            if state.halt_trading:
                await asyncio.sleep(10); continue

            df=await update_market_data()
            if df.empty:
                await asyncio.sleep(5); continue

            candle_count+=1
            # Warm-start: train on the very first valid batch so the ML/DL
            # uncertainty gate doesn't permanently block signals. After that,
            # retrain on the normal RETRAIN_INTERVAL cadence.
            should_train = (
                (candle_count == 1 and not ml_ensemble.fitted) or
                (candle_count % CONFIG['RETRAIN_INTERVAL'] == 0)
            )
            if should_train:
                logger.info(f"Retraining models (candle {candle_count}, rows={len(df)})...")
                await train_models(df)
                rl_stop_optimizer.save()

            await monitor_positions(df)

            if any(p.get('market')==CONFIG['ACTIVE_MARKET'] for p in state.open_positions):
                await asyncio.sleep(1); continue

            # 24-h cycle gate: no new opens during winddown / summarizing.
            # Existing positions are still monitored and close naturally.
            block_reasons: List[str] = []
            if not can_open_new_trade(block_reasons):
                if candle_count % 60 == 0:
                    logger.info(f"  [cycle gate] no new opens — {','.join(block_reasons)}")
                await asyncio.sleep(2); continue

            should,conf,action,details=await generate_trading_signal(df)
            if should and action!='hold':
                await execute_trade(action,conf,details,df)

            await asyncio.sleep(1)

        except KeyboardInterrupt:
            logger.info("Shutdown requested")
            state.halt_trading=True
            rl_stop_optimizer.save()
            break
        except Exception as e:
            logger.error(f"Loop error: {e}", exc_info=True)
            await asyncio.sleep(5)

def _shutdown_handler(signum, frame):
    """SIGTERM / SIGINT — flush state to disk before the process dies so a
    crash or kill doesn't lose the trade history or RL learnings."""
    try:
        logger.info(f"Signal {signum} received — persisting state before exit")
        persist_state()
        try: rl_stop_optimizer.save()
        except Exception: pass
    except Exception as e:
        logger.error(f"shutdown_handler error: {e}")
    # Re-raise default behavior so the process actually exits
    if signum == 2:    # SIGINT
        raise KeyboardInterrupt()

import signal
try:
    signal.signal(signal.SIGTERM, _shutdown_handler)
    signal.signal(signal.SIGINT,  _shutdown_handler)
except Exception:
    pass  # not on a platform that supports these

if __name__=="__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown via KeyboardInterrupt")
    finally:
        _safe(persist_state)
        _safe(rl_stop_optimizer.save)
        if _http_session is not None and not _http_session.closed:
            try: asyncio.run(_http_session.close())
            except Exception: pass
