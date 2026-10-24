"""
exchange_adapter.py — Delphi Nova v4.00 multi-venue perpetual execution layer.

Venues:  Aster (primary)  ·  ApeX Omni (secondary)  ·  Paper (default / validation).

DESIGN
------
* Market DATA is always pulled from REAL venues (Aster public REST needs no keys),
  so signals are real even in paper mode. Only ORDER PLACEMENT is simulated in paper.
* `ExchangeRouter` is a drop-in for the retired `DriftTrader`: it exposes the same
  method names/signatures the engine already calls (`initialize`, `get_collateral`,
  `place_order`, `close_position`, `get_open_positions`, `get_funding_rate`,
  `get_sol_price`) PLUS the richer venue-native interface (`get_orderbook`,
  `get_open_interest`, `get_long_short_ratio`, `get_liquidations`, `estimate_fees`,
  `health`, `place_perp_order`).
* Router failover: try primary (Aster); on API error / unhealthy venue, circuit-break
  to ApeX. Paper mode never touches a venue for fills.

SECURITY (non-negotiable)
-------------------------
* All secrets are read from environment variables ONLY and are NEVER logged.
* These adapters are *standard HTTPS API clients*. There is NO VPN / proxy /
  IP-masking / location-spoofing / geo-evasion code anywhere in this module, and
  none will be added. Lawful use depends on where the operator runs the process.
"""
from __future__ import annotations

import os
import time
import hmac
import hashlib
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Any, Tuple
from urllib.parse import urlencode

logger = logging.getLogger("delphi.adapter")

try:
    import aiohttp
except Exception:  # pragma: no cover
    aiohttp = None
    logger.warning("aiohttp unavailable — venue REST disabled, paper-only")

# --- sympy compatibility shim (apexomni <-> modern sympy required by torch) ---
# apexomni 3.3.1 pins sympy==1.6 and does `from sympy.core.numbers import igcdex`,
# but torch 2.12 requires sympy>=1.13.3, where `igcdex` was relocated to
# `sympy.core.intfunc`. Re-export it into its legacy location BEFORE importing
# apexomni so its StarkEx crypto (Pedersen/EC math in math_utils.py) loads.
# `is_quad_residue`/`sqrt_mod` stay top-level and are unaffected. This shim lives
# in our code (not site-packages), so it survives any apexomni reinstall.
try:  # pragma: no cover - environment-dependent
    import sympy.core.numbers as _sym_numbers  # type: ignore
    if not hasattr(_sym_numbers, "igcdex"):
        try:
            from sympy.core.intfunc import igcdex as _igcdex  # sympy>=1.13
        except Exception:
            from sympy import igcdex as _igcdex  # fallback for other layouts
        _sym_numbers.igcdex = _igcdex
except Exception as _e:  # never let the shim break import
    logger.debug("sympy igcdex shim skipped: %s", _e)

# Optional venue SDKs — import guarded so a missing/broken SDK never breaks startup.
try:
    import apexomni  # noqa: F401
    _HAS_APEX_SDK = True
except Exception:
    _HAS_APEX_SDK = False

try:
    import eth_account  # noqa: F401  (EIP-712 v3 signing for Aster, optional)
    _HAS_ETH = True
except Exception:
    _HAS_ETH = False


# ============================================================
# Fee schedules (round-trip = open + close). Verified May 2026.
# ============================================================
FEES = {
    # venue: (maker_per_side, taker_per_side)
    "aster": (0.0001, 0.00035),   # 0.01% / 0.035%
    "apex":  (0.0002, 0.0005),    # 0.02% / 0.05% (promo: 0% / 0.025%)
    "apex_promo": (0.0000, 0.00025),
    "paper": (0.0001, 0.00035),   # mirrors Aster (the paper venue of record)
}

# Symbol mapping per venue (engine speaks "SOL").
SYMBOLS = {
    "aster": {"SOL": "SOLUSDT", "BTC": "BTCUSDT", "ETH": "ETHUSDT"},
    "apex":  {"SOL": "SOL-USDT", "BTC": "BTC-USDT", "ETH": "ETH-USDT"},
}


def _env(*names: str, default: str = "") -> str:
    """First non-empty env var among `names`. Never logs the value."""
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return default


def _has_secret(*names: str) -> bool:
    return any(bool(os.getenv(n)) for n in names)


def _parse_epoch_ms(v) -> float:
    """Accept epoch (s/ms) or ISO-8601 ('2026-05-30T01:00:00Z') → epoch ms."""
    if not v:
        return 0.0
    try:
        return float(v)                       # already numeric epoch
    except (TypeError, ValueError):
        pass
    try:
        from datetime import datetime
        return datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp() * 1000.0
    except Exception:
        return 0.0


def _coin_from_symbol(symbol: str) -> str:
    """'SOLUSDT'/'SOL-USDT' → 'SOL' (base coin for coin-keyed CEX endpoints)."""
    s = (symbol or "").upper().replace("-", "").replace("_", "")
    for q in ("USDT", "USDC", "USD", "PERP"):
        if s.endswith(q) and len(s) > len(q):
            return s[: -len(q)]
    return s


async def _cex_long_short(symbol: str) -> float:
    """Market-wide long/short ACCOUNT ratio from public CEX futures-data, used as
    a sentiment proxy because Aster/ApeX don't expose one. Tries several venues
    because public-data availability varies by jurisdiction; returns neutral 1.0
    if none answer. Read-only PUBLIC data — no keys, no order flow, and NO
    geo-evasion: if a venue blocks this region we simply skip it."""
    if aiohttp is None:
        return 1.0
    coin = _coin_from_symbol(symbol)

    async def _binance(s):
        url = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
        async with s.get(url, params={"symbol": symbol, "period": "5m", "limit": 1}) as r:
            r.raise_for_status()
            d = await r.json()
            if isinstance(d, list) and d:
                return float(d[-1].get("longShortRatio", 0) or 0)
        return 0.0

    async def _okx(s):
        url = "https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio"
        async with s.get(url, params={"ccy": coin, "period": "5m"}) as r:
            r.raise_for_status()
            d = await r.json()
            rows = (d or {}).get("data", [])
            if rows:                              # newest first: [ts, ratio]
                return float(rows[0][1])
        return 0.0

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=6)) as s:
            for src in (_binance, _okx):
                try:
                    v = await src(s)
                    if v and v > 0:
                        return v
                except Exception:
                    continue
    except Exception:
        pass
    return 1.0


# ============================================================
# Abstract adapter
# ============================================================
class ExchangeAdapter(ABC):
    name: str = "base"

    def __init__(self, symbol_key: str = "SOL", testnet: bool = True):
        self.symbol_key = symbol_key
        self.testnet = testnet
        self.connected = False
        self.can_trade = False           # True only when valid creds are loaded
        self._fail_count = 0
        self._tripped_until = 0.0        # circuit-breaker cooldown timestamp

    # --- lifecycle ---
    @abstractmethod
    async def connect(self) -> bool: ...

    # --- market data (read-only; no keys required for public venues) ---
    @abstractmethod
    async def get_price(self) -> float: ...
    @abstractmethod
    async def get_funding(self) -> Dict[str, float]: ...
    @abstractmethod
    async def get_orderbook(self, depth: int = 20) -> Dict[str, list]: ...
    @abstractmethod
    async def get_open_interest(self) -> float: ...
    @abstractmethod
    async def get_long_short_ratio(self) -> float: ...
    @abstractmethod
    async def get_liquidations(self) -> Dict[str, float]: ...

    # --- account / trading ---
    @abstractmethod
    async def get_balance(self) -> Dict[str, float]: ...
    @abstractmethod
    async def get_positions(self) -> List[Dict]: ...
    @abstractmethod
    async def place_perp_order(self, side: str, size: float, leverage: int,
                               sl_price: float = 0.0, tp_price: float = 0.0) -> Optional[str]: ...
    @abstractmethod
    async def close_position(self) -> bool: ...

    # --- shared helpers ---
    def estimate_fees(self, notional: float, maker: bool = False) -> float:
        mk, tk = FEES.get(self.name, FEES["paper"])
        return float(notional) * (mk if maker else tk)

    def round_trip_fee_rate(self, maker: bool = False) -> float:
        mk, tk = FEES.get(self.name, FEES["paper"])
        return 2.0 * (mk if maker else tk)

    def _healthy(self) -> bool:
        return time.time() >= self._tripped_until

    def _trip(self, cooldown: float = 30.0):
        self._fail_count += 1
        self._tripped_until = time.time() + cooldown

    def _ok(self):
        self._fail_count = 0
        self._tripped_until = 0.0


# ============================================================
# Aster — Binance-style USDT-M perpetual REST (primary)
# ============================================================
class AsterAdapter(ExchangeAdapter):
    name = "aster"
    MAINNET = "https://fapi.asterdex.com"
    TESTNET = os.getenv("ASTER_TESTNET_URL", "https://testnet.asterdex.com")

    def __init__(self, symbol_key: str = "SOL", testnet: bool = True):
        super().__init__(symbol_key, testnet)
        self.base = self.TESTNET if testnet else self.MAINNET
        self.base = os.getenv("ASTER_BASE_URL", self.base)
        # Public market DATA always comes from MAINNET so signals are real even
        # in paper/testnet mode (Aster has no reachable public testnet host).
        # Only SIGNED trading uses self.base (mode-dependent).
        self.data_base = os.getenv("ASTER_DATA_URL", self.MAINNET)
        self.symbol = SYMBOLS["aster"].get(symbol_key, "SOLUSDT")
        self._key = _env("ASTER_API_KEY")
        self._secret = _env("ASTER_API_SECRET")
        self._sess: Optional["aiohttp.ClientSession"] = None

    async def _session(self) -> "aiohttp.ClientSession":
        if aiohttp is None:
            raise RuntimeError("aiohttp not installed")
        if self._sess is None or self._sess.closed:
            self._sess = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=8),
                headers={"X-MBX-APIKEY": self._key} if self._key else None,
            )
        return self._sess

    async def _public(self, path: str, params: dict = None) -> Any:
        s = await self._session()
        url = f"{self.data_base}{path}"      # public data → MAINNET always
        async with s.get(url, params=params or {}) as r:
            r.raise_for_status()
            return await r.json()

    def _sign(self, params: dict) -> dict:
        params = dict(params)
        params["timestamp"] = int(time.time() * 1000)
        params.setdefault("recvWindow", 5000)
        query = urlencode(params)
        sig = hmac.new(self._secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        params["signature"] = sig
        return params

    async def _signed(self, method: str, path: str, params: dict = None) -> Any:
        if not (self._key and self._secret):
            raise PermissionError("aster: API key/secret not configured")
        s = await self._session()
        params = self._sign(params or {})
        url = f"{self.base}{path}"
        async with s.request(method, url, params=params) as r:
            r.raise_for_status()
            return await r.json()

    async def connect(self) -> bool:
        try:
            await self.get_price()          # public reachability probe
            self.connected = True
            self.can_trade = bool(self._key and self._secret)
            self._ok()
            logger.info("Aster connected (base=%s, trade=%s, testnet=%s)",
                        self.base, self.can_trade, self.testnet)
            return True
        except Exception as e:
            logger.warning("Aster connect failed: %s", e)
            self._trip()
            return False

    async def get_price(self) -> float:
        try:
            d = await self._public("/fapi/v1/premiumIndex", {"symbol": self.symbol})
            mark = float(d.get("markPrice", 0) or 0)
            if mark > 0:
                self._ok(); return mark
        except Exception:
            pass
        try:
            d = await self._public("/fapi/v1/ticker/price", {"symbol": self.symbol})
            px = float(d.get("price", 0) or 0)
            if px > 0:
                self._ok(); return px
        except Exception as e:
            logger.debug("aster get_price: %s", e); self._trip()
        return 0.0

    async def get_funding(self) -> Dict[str, float]:
        try:
            d = await self._public("/fapi/v1/premiumIndex", {"symbol": self.symbol})
            rate = float(d.get("lastFundingRate", 0) or 0)
            return {"rate": rate, "predicted": rate,
                    "next_time": float(d.get("nextFundingTime", 0) or 0),
                    "mark": float(d.get("markPrice", 0) or 0),
                    "index": float(d.get("indexPrice", 0) or 0)}
        except Exception as e:
            logger.debug("aster funding: %s", e)
            return {"rate": 0.0, "predicted": 0.0, "next_time": 0.0, "mark": 0.0, "index": 0.0}

    async def get_orderbook(self, depth: int = 20) -> Dict[str, list]:
        try:
            d = await self._public("/fapi/v1/depth", {"symbol": self.symbol, "limit": depth})
            return {"bids": [[float(p), float(q)] for p, q in d.get("bids", [])],
                    "asks": [[float(p), float(q)] for p, q in d.get("asks", [])]}
        except Exception as e:
            logger.debug("aster depth: %s", e)
            return {"bids": [], "asks": []}

    async def get_open_interest(self) -> float:
        try:
            d = await self._public("/fapi/v1/openInterest", {"symbol": self.symbol})
            return float(d.get("openInterest", 0) or 0)
        except Exception:
            return 0.0

    async def get_long_short_ratio(self) -> float:
        try:
            d = await self._public("/futures/data/globalLongShortAccountRatio",
                                   {"symbol": self.symbol, "period": "5m", "limit": 1})
            if isinstance(d, list) and d:
                return float(d[-1].get("longShortRatio", 1.0) or 1.0)
        except Exception:
            pass
        # Aster's L/S endpoint is unavailable (serves HTML) → public CEX
        # futures-data (Binance→OKX) as the market-wide long/short proxy.
        return await _cex_long_short(self.symbol)

    async def get_liquidations(self) -> Dict[str, float]:
        # Best-effort; many Binance-style venues only stream forceOrders over WS.
        try:
            d = await self._public("/fapi/v1/allForceOrders",
                                   {"symbol": self.symbol, "limit": 50})
            longs = sum(float(o.get("origQty", 0)) for o in d if o.get("side") == "SELL")
            shorts = sum(float(o.get("origQty", 0)) for o in d if o.get("side") == "BUY")
            return {"long_liq": longs, "short_liq": shorts, "total": longs + shorts}
        except Exception:
            return {"long_liq": 0.0, "short_liq": 0.0, "total": 0.0}

    async def klines(self, interval: str = "1m", limit: int = 500) -> list:
        try:
            return await self._public("/fapi/v1/klines",
                                      {"symbol": self.symbol, "interval": interval, "limit": limit})
        except Exception as e:
            logger.debug("aster klines: %s", e)
            return []

    async def get_balance(self) -> Dict[str, float]:
        if not self.can_trade:
            return {"total": 0.0, "free": 0.0, "margin_used": 0.0, "unrealized_pnl": 0.0}
        try:
            d = await self._signed("GET", "/fapi/v2/account")
            total = float(d.get("totalWalletBalance", 0) or 0)
            free = float(d.get("availableBalance", 0) or 0)
            upnl = float(d.get("totalUnrealizedProfit", 0) or 0)
            return {"total": total, "free": free, "margin_used": max(total - free, 0.0),
                    "unrealized_pnl": upnl}
        except Exception as e:
            logger.debug("aster balance: %s", e)
            return {"total": 0.0, "free": 0.0, "margin_used": 0.0, "unrealized_pnl": 0.0}

    async def get_positions(self) -> List[Dict]:
        if not self.can_trade:
            return []
        try:
            d = await self._signed("GET", "/fapi/v2/positionRisk", {"symbol": self.symbol})
            out = []
            for p in (d if isinstance(d, list) else []):
                amt = float(p.get("positionAmt", 0) or 0)
                if amt != 0:
                    out.append({"market_index": 0,
                                "direction": "long" if amt > 0 else "short",
                                "size": abs(amt),
                                "entry_price": float(p.get("entryPrice", 0) or 0),
                                "pnl": float(p.get("unRealizedProfit", 0) or 0)})
            return out
        except Exception as e:
            logger.debug("aster positions: %s", e)
            return []

    async def set_leverage(self, leverage: int) -> bool:
        if not self.can_trade:
            return False
        try:
            await self._signed("POST", "/fapi/v1/leverage",
                               {"symbol": self.symbol, "leverage": int(leverage)})
            return True
        except Exception as e:
            logger.debug("aster set_leverage: %s", e)
            return False

    async def place_perp_order(self, side: str, size: float, leverage: int,
                               sl_price: float = 0.0, tp_price: float = 0.0) -> Optional[str]:
        if not self.can_trade:
            raise PermissionError("aster: cannot place live order without API key/secret")
        bside = "BUY" if side == "long" else "SELL"
        await self.set_leverage(leverage)
        try:
            o = await self._signed("POST", "/fapi/v1/order", {
                "symbol": self.symbol, "side": bside, "type": "MARKET",
                "quantity": round(size, 3)})
            oid = str(o.get("orderId", ""))
            # Best-effort reduce-only SL/TP brackets
            close_side = "SELL" if side == "long" else "BUY"
            for px, otype in ((sl_price, "STOP_MARKET"), (tp_price, "TAKE_PROFIT_MARKET")):
                if px and px > 0:
                    try:
                        await self._signed("POST", "/fapi/v1/order", {
                            "symbol": self.symbol, "side": close_side, "type": otype,
                            "stopPrice": round(px, 4), "closePosition": "true"})
                    except Exception as e:
                        logger.warning("aster bracket %s failed: %s", otype, e)
            self._ok()
            return oid or f"aster_{int(time.time())}"
        except Exception as e:
            logger.error("aster order failed: %s", e); self._trip()
            return None

    async def close_position(self) -> bool:
        if not self.can_trade:
            return True
        try:
            positions = await self.get_positions()
            for p in positions:
                cside = "SELL" if p["direction"] == "long" else "BUY"
                await self._signed("POST", "/fapi/v1/order", {
                    "symbol": self.symbol, "side": cside, "type": "MARKET",
                    "quantity": round(p["size"], 3), "reduceOnly": "true"})
            return True
        except Exception as e:
            logger.error("aster close failed: %s", e)
            return False

    async def health(self) -> Dict[str, Any]:
        t0 = time.time()
        px = await self.get_price()
        return {"venue": self.name, "ok": px > 0 and self._healthy(),
                "latency_ms": (time.time() - t0) * 1000, "price": px,
                "can_trade": self.can_trade}

    async def aclose(self):
        if self._sess and not self._sess.closed:
            await self._sess.close()


# ============================================================
# ApeX Omni — StarkEx L2 (secondary). SDK trading, best-effort public data.
# ============================================================
class ApexOmniAdapter(ExchangeAdapter):
    name = "apex"
    MAINNET = "https://omni.apex.exchange"
    TESTNET = os.getenv("APEX_TESTNET_URL", "https://testnet.omni.apex.exchange")

    def __init__(self, symbol_key: str = "SOL", testnet: bool = True, promo: bool = True):
        super().__init__(symbol_key, testnet)
        self.base = os.getenv("APEX_BASE_URL", self.TESTNET if testnet else self.MAINNET)
        # Public DATA from MAINNET always; the trading SDK endpoint is chosen
        # separately in connect() (APEX_OMNI_HTTP_TEST/MAIN).
        self.data_base = os.getenv("APEX_DATA_URL", self.MAINNET)
        self.symbol = SYMBOLS["apex"].get(symbol_key, "SOL-USDT")
        # ApeX public REST expects the hyphen-stripped symbol (SOLUSDT); the SDK
        # strips it internally, but our raw `_public` calls must do it ourselves.
        self.api_symbol = self.symbol.replace("-", "")
        self.promo = promo
        self._key = _env("APEX_API_KEY")
        self._secret = _env("APEX_API_SECRET")
        self._passphrase = _env("APEX_API_PASSPHRASE")
        self._l2key = _env("APEX_ZK_L2KEY", "APEX_L2KEY")
        self._seeds = _env("APEX_ZK_SEEDS", "APEX_SEEDS")
        self._client = None
        self._sess: Optional["aiohttp.ClientSession"] = None

    @property
    def name_fee(self) -> str:
        return "apex_promo" if self.promo else "apex"

    def estimate_fees(self, notional: float, maker: bool = False) -> float:
        mk, tk = FEES[self.name_fee]
        return float(notional) * (mk if maker else tk)

    async def _session(self) -> "aiohttp.ClientSession":
        if aiohttp is None:
            raise RuntimeError("aiohttp not installed")
        if self._sess is None or self._sess.closed:
            self._sess = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=8))
        return self._sess

    async def _public(self, path: str, params: dict = None) -> Any:
        s = await self._session()
        async with s.get(f"{self.data_base}{path}", params=params or {}) as r:  # MAINNET data
            r.raise_for_status()
            return await r.json()

    async def connect(self) -> bool:
        # Trading requires the full StarkEx credential set + SDK.
        self.can_trade = bool(_HAS_APEX_SDK and self._key and self._secret
                              and self._passphrase and self._l2key and self._seeds)
        if self.can_trade:
            try:
                from apexomni.http_private_v3 import HttpPrivate_v3  # type: ignore
                from apexomni.constants import APEX_OMNI_HTTP_MAIN, APEX_OMNI_HTTP_TEST  # type: ignore
                endpoint = APEX_OMNI_HTTP_TEST if self.testnet else APEX_OMNI_HTTP_MAIN
                self._client = HttpPrivate_v3(
                    endpoint, api_key_credentials={
                        "key": self._key, "secret": self._secret, "passphrase": self._passphrase},
                    zk_seeds=self._seeds, zk_l2Key=self._l2key)
                self._client.configs_v3()
                self._client.get_account_v3()
                logger.info("ApeX Omni connected (SDK, testnet=%s)", self.testnet)
            except Exception as e:
                logger.warning("ApeX SDK init failed (data-only): %s", e)
                self.can_trade = False
        # public reachability (data-only path is fine without creds)
        try:
            await self.get_price()
            self.connected = True
            self._ok()
            return True
        except Exception as e:
            logger.debug("ApeX public probe failed: %s", e)
            self.connected = self.can_trade
            return self.connected

    async def get_price(self) -> float:
        try:
            d = await self._public("/api/v3/ticker", {"symbol": self.api_symbol})
            data = d.get("data", d)
            if isinstance(data, list) and data:
                data = data[0]
            for key in ("lastPrice", "markPrice", "indexPrice", "close"):
                px = float(data.get(key, 0) or 0)
                if px > 0:
                    self._ok(); return px
        except Exception:
            self._trip()
        return 0.0

    async def get_funding(self) -> Dict[str, float]:
        try:
            d = await self._public("/api/v3/ticker", {"symbol": self.api_symbol})
            data = d.get("data", d)
            if isinstance(data, list) and data:
                data = data[0]
            rate = float(data.get("fundingRate", 0) or 0)
            # ApeX leaves oraclePrice empty; markPrice is the perp mark.
            mark = float(data.get("markPrice", 0) or data.get("oraclePrice", 0) or 0)
            return {"rate": rate,
                    "predicted": float(data.get("predictedFundingRate", rate) or rate),
                    "next_time": _parse_epoch_ms(data.get("nextFundingTime", 0)),
                    "mark": mark,
                    "index": float(data.get("indexPrice", 0) or 0)}
        except Exception:
            return {"rate": 0.0, "predicted": 0.0, "next_time": 0.0, "mark": 0.0, "index": 0.0}

    async def get_orderbook(self, depth: int = 20) -> Dict[str, list]:
        try:
            d = await self._public("/api/v3/depth", {"symbol": self.api_symbol, "limit": depth})
            data = d.get("data", d)
            return {"bids": [[float(p), float(q)] for p, q in data.get("b", data.get("bids", []))],
                    "asks": [[float(p), float(q)] for p, q in data.get("a", data.get("asks", []))]}
        except Exception:
            return {"bids": [], "asks": []}

    async def get_open_interest(self) -> float:
        try:
            d = await self._public("/api/v3/ticker", {"symbol": self.api_symbol})
            data = d.get("data", d)
            if isinstance(data, list) and data:
                data = data[0]
            return float(data.get("openInterest", 0) or 0)
        except Exception:
            return 0.0

    async def get_long_short_ratio(self) -> float:
        return 1.0  # ApeX does not expose a public global L/S ratio endpoint

    async def get_liquidations(self) -> Dict[str, float]:
        return {"long_liq": 0.0, "short_liq": 0.0, "total": 0.0}

    async def get_balance(self) -> Dict[str, float]:
        if not (self.can_trade and self._client):
            return {"total": 0.0, "free": 0.0, "margin_used": 0.0, "unrealized_pnl": 0.0}
        try:
            acct = self._client.get_account_balance_v3()
            data = acct.get("data", acct) if isinstance(acct, dict) else {}
            total = float(data.get("totalEquityValue", 0) or 0)
            free = float(data.get("availableBalance", total) or 0)
            return {"total": total, "free": free, "margin_used": max(total - free, 0.0),
                    "unrealized_pnl": float(data.get("unrealizedPnl", 0) or 0)}
        except Exception as e:
            logger.debug("apex balance: %s", e)
            return {"total": 0.0, "free": 0.0, "margin_used": 0.0, "unrealized_pnl": 0.0}

    async def get_positions(self) -> List[Dict]:
        if not (self.can_trade and self._client):
            return []
        try:
            acct = self._client.get_account_v3()
            positions = (acct or {}).get("positions", []) if isinstance(acct, dict) else []
            out = []
            for p in positions:
                sz = float(p.get("size", 0) or 0)
                if sz != 0 and p.get("symbol") == self.symbol:
                    out.append({"market_index": 0,
                                "direction": "long" if p.get("side") == "LONG" else "short",
                                "size": abs(sz), "entry_price": float(p.get("entryPrice", 0) or 0),
                                "pnl": float(p.get("unrealizedPnl", 0) or 0)})
            return out
        except Exception as e:
            logger.debug("apex positions: %s", e)
            return []

    async def place_perp_order(self, side: str, size: float, leverage: int,
                               sl_price: float = 0.0, tp_price: float = 0.0) -> Optional[str]:
        if not (self.can_trade and self._client):
            raise PermissionError("apex: cannot place live order without full StarkEx creds + SDK")
        try:
            px = await self.get_price()
            res = self._client.create_order_v3(
                symbol=self.symbol, side="BUY" if side == "long" else "SELL",
                type="MARKET", size=str(round(size, 3)), price=str(px),
                leverage=str(int(leverage)),
                slPrice=str(sl_price) if sl_price else None,
                tpPrice=str(tp_price) if tp_price else None)
            oid = str((res or {}).get("data", {}).get("orderId", "") or "")
            self._ok()
            return oid or f"apex_{int(time.time())}"
        except Exception as e:
            logger.error("apex order failed: %s", e); self._trip()
            return None

    async def close_position(self) -> bool:
        if not (self.can_trade and self._client):
            return True
        try:
            for p in await self.get_positions():
                self._client.create_order_v3(
                    symbol=self.symbol, side="SELL" if p["direction"] == "long" else "BUY",
                    type="MARKET", size=str(round(p["size"], 3)), reduceOnly=True)
            return True
        except Exception as e:
            logger.error("apex close failed: %s", e)
            return False

    async def health(self) -> Dict[str, Any]:
        t0 = time.time()
        px = await self.get_price()
        return {"venue": self.name, "ok": px > 0 and self._healthy(),
                "latency_ms": (time.time() - t0) * 1000, "price": px,
                "can_trade": self.can_trade}

    async def aclose(self):
        if self._sess and not self._sess.closed:
            await self._sess.close()


# ============================================================
# Paper — simulated fills on REAL prices + honest fee math (default)
# ============================================================
class PaperAdapter(ExchangeAdapter):
    name = "paper"

    def __init__(self, symbol_key: str = "SOL", data_venue: Optional[ExchangeAdapter] = None,
                 starting_equity: float = 100.0):
        super().__init__(symbol_key, testnet=True)
        self.data = data_venue          # a real adapter used for market data
        self.equity = starting_equity
        self.can_trade = True           # paper can always "trade"
        self._positions: List[Dict] = []

    def set_equity(self, eq: float):
        self.equity = float(eq)

    async def connect(self) -> bool:
        self.connected = True
        if self.data is not None:
            try:
                await self.data.connect()
            except Exception:
                pass
        logger.info("Paper adapter ready (data venue=%s)",
                    self.data.name if self.data else "none")
        return True

    # market data delegates to the real data venue (so signals are real)
    async def get_price(self) -> float:
        return await self.data.get_price() if self.data else 0.0

    async def get_funding(self) -> Dict[str, float]:
        return await self.data.get_funding() if self.data else \
            {"rate": 0.0, "predicted": 0.0, "next_time": 0.0, "mark": 0.0, "index": 0.0}

    async def get_orderbook(self, depth: int = 20) -> Dict[str, list]:
        return await self.data.get_orderbook(depth) if self.data else {"bids": [], "asks": []}

    async def get_open_interest(self) -> float:
        return await self.data.get_open_interest() if self.data else 0.0

    async def get_long_short_ratio(self) -> float:
        return await self.data.get_long_short_ratio() if self.data else 1.0

    async def get_liquidations(self) -> Dict[str, float]:
        return await self.data.get_liquidations() if self.data else \
            {"long_liq": 0.0, "short_liq": 0.0, "total": 0.0}

    async def get_balance(self) -> Dict[str, float]:
        return {"total": self.equity, "free": self.equity, "margin_used": 0.0,
                "unrealized_pnl": 0.0}

    async def get_positions(self) -> List[Dict]:
        return list(self._positions)

    async def place_perp_order(self, side: str, size: float, leverage: int,
                               sl_price: float = 0.0, tp_price: float = 0.0) -> Optional[str]:
        tx = f"paper_{int(time.time())}_{len(self._positions)}"
        return tx

    async def close_position(self) -> bool:
        self._positions.clear()
        return True

    async def health(self) -> Dict[str, Any]:
        h = {"venue": "paper", "ok": True, "latency_ms": 0.0, "can_trade": True}
        if self.data is not None:
            try:
                h["data_venue"] = await self.data.health()
            except Exception:
                pass
        return h

    async def aclose(self):
        if self.data is not None:
            await self.data.aclose()


# ============================================================
# Router — Aster primary → ApeX failover; drop-in for DriftTrader
# ============================================================
LIVE_ACK_PHRASE = "I-UNDERSTAND-THIS-RISKS-REAL-MONEY"


class ExchangeRouter:
    """
    mode: 'paper' (default, simulated fills) | 'testnet' | 'live'.
    Market data always comes from real venues. Live fills require valid creds
    AND mode != 'paper' AND an explicit go-live in the engine.
    """
    def __init__(self, symbol_key: str = "SOL", mode: str = "paper",
                 starting_equity: float = 100.0, promo: bool = True):
        self.symbol_key = symbol_key
        self.mode = mode if mode in ("paper", "testnet", "live") else "paper"
        # Portfolio/research release: non-paper modes are HARD-DISABLED unless the
        # operator deliberately sets DELPHI_LIVE_ACK to the exact phrase below.
        # Without it, everything is forced to paper (simulated fills, no keys used).
        if self.mode != "paper" and os.getenv("DELPHI_LIVE_ACK", "") != LIVE_ACK_PHRASE:
            logger.warning("mode=%s requested but DELPHI_LIVE_ACK not set -> forcing PAPER mode", self.mode)
            self.mode = "paper"
        testnet = (self.mode != "live")
        self.aster = AsterAdapter(symbol_key, testnet=testnet)
        self.apex = ApexOmniAdapter(symbol_key, testnet=testnet, promo=promo)
        self.paper = PaperAdapter(symbol_key, data_venue=self.aster,
                                  starting_equity=starting_equity)
        self.venues: List[ExchangeAdapter] = [self.aster, self.apex]   # priority order
        self.initialized = False
        self.last_venue = "paper"
        # legacy-compat: some old code paths poke `.client`; keep it None-safe.
        self.client = None

    # ---------- lifecycle ----------
    def is_paper(self) -> bool:
        return self.mode == "paper"

    async def connect(self) -> bool:
        await self.paper.connect()                      # also connects aster (data venue)
        try:
            await self.apex.connect()
        except Exception:
            pass
        self.initialized = True
        live_ok = any(v.can_trade for v in self.venues)
        logger.info("Router up — mode=%s, aster.trade=%s, apex.trade=%s",
                    self.mode, self.aster.can_trade, self.apex.can_trade)
        if self.mode != "paper" and not live_ok:
            logger.warning("mode=%s but NO venue has trading creds — falling back to paper fills",
                           self.mode)
        return True

    # legacy alias used by main(): drift_trader.initialize()
    async def initialize(self) -> bool:
        return await self.connect()

    def _data_venue(self) -> ExchangeAdapter:
        for v in self.venues:
            if v._healthy():
                return v
        return self.aster

    def _trade_venue(self) -> Optional[ExchangeAdapter]:
        for v in self.venues:                # Aster first, then ApeX
            if v.can_trade and v._healthy():
                return v
        return None

    # ---------- market data (always real) ----------
    async def get_price(self) -> float:
        for v in (self._data_venue(), *self.venues):
            try:
                px = await v.get_price()
                if px > 0:
                    self.last_venue = v.name
                    return px
            except Exception:
                v._trip()
        return await self._price_fallback()

    # legacy alias
    async def get_sol_price(self) -> float:
        px = await self.get_price()
        if px > 0:
            return px
        return await self._price_fallback()

    async def _price_fallback(self) -> float:
        """Birdeye → CoinGecko, mirroring the retired DriftTrader fallbacks."""
        if aiohttp is None:
            return 0.0
        try:
            url = ("https://public-api.birdeye.so/defi/price?address="
                   "So11111111111111111111111111111111111111112")
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as s:
                async with s.get(url, headers={"X-API-KEY": os.getenv("BIRDEYE_API_KEY", "")}) as r:
                    d = await r.json()
                    if d.get("success") and d.get("data", {}).get("value"):
                        return float(d["data"]["value"])
        except Exception:
            pass
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as s:
                async with s.get("https://api.coingecko.com/api/v3/simple/price",
                                 params={"ids": "solana", "vs_currencies": "usd",
                                         "x_cg_demo_api_key": os.getenv("COINGECKO_API_KEY", "")}) as r:
                    return float((await r.json())["solana"]["usd"])
        except Exception:
            return 0.0

    async def get_funding(self) -> Dict[str, float]:
        return await self._data_venue().get_funding()

    # legacy alias
    async def get_funding_rate(self) -> float:
        return (await self.get_funding()).get("rate", 0.0)

    async def get_orderbook(self, depth: int = 20) -> Dict[str, list]:
        return await self._data_venue().get_orderbook(depth)

    async def get_open_interest(self) -> float:
        return await self._data_venue().get_open_interest()

    async def get_long_short_ratio(self) -> float:
        return await self._data_venue().get_long_short_ratio()

    async def get_liquidations(self) -> Dict[str, float]:
        return await self._data_venue().get_liquidations()

    async def cross_venue_basis(self) -> Dict[str, float]:
        """ApeX mark vs Aster mark divergence (replacement for oracle-mark spread)."""
        out = {"aster": 0.0, "apex": 0.0, "basis": 0.0}
        try:
            a = await self.aster.get_price()
            x = await self.apex.get_price()
            out["aster"], out["apex"] = a, x
            if a > 0 and x > 0:
                out["basis"] = (x - a) / a
        except Exception:
            pass
        return out

    async def klines(self, interval: str = "1m", limit: int = 500) -> list:
        try:
            return await self.aster.klines(interval, limit)
        except Exception:
            return []

    # ---------- account ----------
    async def get_balance(self) -> Dict[str, float]:
        if self.is_paper():
            return await self.paper.get_balance()
        tv = self._trade_venue()
        return await tv.get_balance() if tv else await self.paper.get_balance()

    # legacy alias
    async def get_collateral(self) -> Dict[str, float]:
        return await self.get_balance()

    async def get_positions(self) -> List[Dict]:
        if self.is_paper():
            return await self.paper.get_positions()
        tv = self._trade_venue()
        return await tv.get_positions() if tv else []

    # legacy alias
    async def get_open_positions(self) -> List[Dict]:
        return await self.get_positions()

    def set_paper_equity(self, eq: float):
        self.paper.set_equity(eq)

    def estimate_fees(self, notional: float, maker: bool = False) -> float:
        v = self._trade_venue() or self.paper
        return v.estimate_fees(notional, maker)

    def round_trip_fee_rate(self, maker: bool = False) -> float:
        v = self._trade_venue() or self.paper
        return v.round_trip_fee_rate(maker)

    # ---------- trading (failover) ----------
    async def place_perp_order(self, side: str, size: float, leverage: int,
                               sl_price: float = 0.0, tp_price: float = 0.0,
                               platform: str = "auto") -> Optional[str]:
        if self.is_paper():
            self.last_venue = "paper"
            return await self.paper.place_perp_order(side, size, leverage, sl_price, tp_price)
        # live/testnet: try venues in priority order with circuit-breaker failover
        order_venues = self.venues
        if platform in ("aster", "apex"):
            order_venues = [getattr(self, platform)]
        last_err = None
        for v in order_venues:
            if not (v.can_trade and v._healthy()):
                continue
            try:
                tx = await v.place_perp_order(side, size, leverage, sl_price, tp_price)
                if tx:
                    self.last_venue = v.name
                    return tx
            except Exception as e:
                last_err = e
                v._trip()
                logger.warning("venue %s order failed, failing over: %s", v.name, e)
        if last_err:
            logger.error("all venues failed to place order: %s", last_err)
        return None

    # legacy alias: drift_trader.place_order(direction, size_sol, leverage, sl, tp)
    async def place_order(self, direction: str, size_sol: float, leverage: int,
                          sl_price: float, tp_price: float) -> Optional[str]:
        return await self.place_perp_order(direction, size_sol, leverage, sl_price, tp_price)

    async def close_position(self, market_index: int = None) -> bool:
        if self.is_paper():
            return await self.paper.close_position()
        tv = self._trade_venue()
        return await tv.close_position() if tv else True

    async def set_leverage(self, leverage: int) -> bool:
        tv = self._trade_venue()
        if tv and hasattr(tv, "set_leverage"):
            return await tv.set_leverage(leverage)
        return True

    # deprecated: venue deposits are user-side (bridge/transfer). Never auto-moves funds.
    async def deposit_usdc(self, amount_usdc: float) -> Optional[str]:
        logger.warning("deposit_usdc is user-side on Aster/ApeX (bridge/transfer). No-op.")
        return None

    async def health(self) -> Dict[str, Any]:
        out = {"mode": self.mode, "active_data_venue": self.last_venue, "venues": []}
        for v in (self.aster, self.apex):
            try:
                out["venues"].append(await v.health())
            except Exception as e:
                out["venues"].append({"venue": v.name, "ok": False, "error": str(e)[:80]})
        return out

    async def aclose(self):
        for v in (self.paper, self.aster, self.apex):
            try:
                await v.aclose()
            except Exception:
                pass


def make_router(symbol_key: str = "SOL", mode: str = "paper",
                starting_equity: float = 100.0, promo: bool = True) -> ExchangeRouter:
    return ExchangeRouter(symbol_key, mode, starting_equity, promo)
