# Delphi Nox Dark — paper-trading research engine

> **Paper trading / research only. Not financial advice.** This repository is published as an engineering portfolio sample. It runs in **paper mode** (simulated fills on real public market prices) and ships with **no API keys**. Leveraged perpetual futures can lose more than you put in; nothing here is a recommendation to trade.

Delphi Nox Dark is the "max-aggressive" profile of my Delphi Nova engine: a single-file async Python engine that looks for **high-momentum, high-velocity short-term swings** on SOL perpetuals and simulates leveraged entries/exits with realistic fee and slippage math, plus a FastAPI dashboard.

## What's in it
| File | What it does |
|---|---|
| `delphi_nova_1_0.py` | Engine (~6.5k lines): data fetch, technical analysis (TA-Lib), regime detection (HMM + GMM ensemble), ML/DL ensemble (scikit-learn, XGBoost, PyTorch), RL sizing optimizer, signal generation with a velocity/momentum timing overlay and "momentum-death" exits, execution via the router, FastAPI dashboard + session-auth gate, main loop |
| `exchange_adapter.py` | Venue abstraction (`ExchangeRouter`) with a paper adapter plus Aster / ApeX Omni adapters, circuit-breaker failover, fee estimation |
| `local_ai_analyst.py` | Optional local-LLM (Ollama) "veto"/analyst layer that can only shrink confidence |
| `tv_ratings.py`, `sr_levels.py` | TradingView-style technical ratings and support/resistance levels |
| `nox_analytics_bridge.py` | Optional bridge to a separate analytics project (`DELPHI_TRADING_ROOT`) |
| `dashboard.html`, `launch_nox_dark.command` | Dashboard UI and a macOS one-click launcher (paper mode) |

`nox_dark` profile (see `CONFIG['PROFILES']`): confidence band 0.51–0.55, min EV ratio 1.2, paper leverage cap 40x, 2.5% risk per trade, 20%/day fee budget, 1 s loop.

## Safety by default
- `DELPHI_MODE` defaults to `paper`. Paper mode needs no keys and signs nothing.
- **Live/testnet are hard-disabled**: `ExchangeRouter` forces paper mode unless `DELPHI_LIVE_ACK` is set to an exact acknowledgement phrase *and* venue keys are present in a local `.env`. The shipped launcher explicitly clears `DELPHI_LIVE_ACK`.
- The engine never moves funds: deposit/withdraw endpoints only return manual instructions.
- `.env`, state files, models and logs are git-ignored; `.env.example` has blank values only.

## Run (paper)
```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # TA-Lib needs the C library (brew install ta-lib)
DELPHI_MODE=paper python delphi_nova_1_0.py   # dashboard on http://127.0.0.1:8080
```

## Tests
```bash
pytest -q tests            # safety tests: paper default, live forced to paper without ack, no real venue orders
python test_tv_ratings.py  # 34 unit checks for the ratings module
```

## License
MIT — see `LICENSE`.
