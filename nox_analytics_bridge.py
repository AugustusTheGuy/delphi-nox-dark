"""nox_analytics_bridge.py — in-process bridge from the live Delphi Nox Dark engine
to the upgraded intelligence layer in ~/Delphi_Trading (set DELPHI_TRADING_ROOT).

The Aster MLX analytics MCP server exposes three pure-Python tool functions:
  * pessimistic_dynamic_half_kelly_with_venue_routing  (sizing + venue routing)
  * query_similar_historic_states                      (616-state 58-dim vector DB)
  * run_local_mlx_inference                            (real mlx.core forward pass)

MCP stdio is far too slow for the engine's 1s loop, so we import those functions and
call them DIRECTLY, in-process. Everything degrades gracefully: if the upgraded root
is absent or a dep is missing, `available()` is False and each wrapper returns None —
the engine then falls back to its internal logic unchanged. Nothing here ever raises
into the trading loop, and nothing silently fakes a result.

Impact analysis: read-only against the upgraded root (it only READS states.npy / model
weights). Reversion: set CONFIG['ASTER_MCP_ENABLE']=False (or remove this file) and the
engine reverts to its internal sizing/overlays with zero behavioural change.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

# Default upgraded-system root; override via env if it ever moves.
_DEFAULT_ROOT = os.getenv(
    "DELPHI_TRADING_ROOT", os.path.expanduser("~/Delphi_Trading")
)
_MCP_DIR = os.path.join(_DEFAULT_ROOT, "mcp_servers")

_sizing_fn = None
_query_fn = None
_mlx_fn = None
_IMPORT_ERROR: Optional[str] = None

try:
    if os.path.isdir(_MCP_DIR):
        if _MCP_DIR not in sys.path:
            sys.path.insert(0, _MCP_DIR)
        # Importing the server module is side-effect-free (run_server is __main__-guarded).
        import aster_mlx_analytics_server as _ams  # type: ignore

        _sizing_fn = _ams.tool_sizing_with_routing
        _query_fn = _ams.tool_query_similar_historic_states
        _mlx_fn = _ams.tool_run_local_mlx_inference
    else:
        _IMPORT_ERROR = f"upgraded root not found: {_MCP_DIR}"
except Exception as e:  # noqa: BLE001
    _IMPORT_ERROR = f"{type(e).__name__}: {e}"


def available() -> bool:
    """True when the in-process analytics functions imported cleanly."""
    return _sizing_fn is not None and _query_fn is not None and _mlx_fn is not None


def status() -> Dict[str, Any]:
    return {
        "available": available(),
        "root": _DEFAULT_ROOT,
        "mcp_dir": _MCP_DIR,
        "import_error": _IMPORT_ERROR,
    }


# ---------------------------------------------------------------------------
# Thin, never-raising wrappers (return None on any failure → engine falls back)
# ---------------------------------------------------------------------------
def size_with_venue_routing(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    balance: float,
    confidence: float = 0.55,
    hurst: float = 0.5,
    ewma_vol: float = 0.02,
    regime: str = "unknown",
    n_trades: int = 0,
    max_leverage: int = 40,
    default_leverage: int = 20,
    vol_ref_annual: float = 1.5,
    atr_pct: Optional[float] = None,
    velocity: Optional[float] = None,
    venues: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """pessimistic_dynamic_half_kelly_with_venue_routing — the authoritative sizing.
    Returns the full tool dict (sizing/leverage/venue_routing/recommendation) or None."""
    if _sizing_fn is None:
        return None
    try:
        args: Dict[str, Any] = {
            "win_rate": float(win_rate),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "balance": float(balance),
            "confidence": float(confidence),
            "hurst": float(hurst),
            "ewma_vol": float(ewma_vol),
            "regime": str(regime),
            "n_trades": int(n_trades),
            "max_leverage": int(max_leverage),
            "default_leverage": int(default_leverage),
            "vol_ref_annual": float(vol_ref_annual),
        }
        if atr_pct is not None:
            args["atr_pct"] = float(atr_pct)
        if velocity is not None:
            args["velocity"] = float(velocity)
        if venues is not None:
            args["venues"] = venues
        return _sizing_fn(args)
    except Exception:
        return None


def query_similar_states(features: List[float], k: int = 8) -> Optional[Dict[str, Any]]:
    """Cosine-similarity lookup vs the 616-state / 58-dim on-disk vector DB.
    Returns {matches:[{similarity, meta:{outcome,label,regime,...}}], indexed_states}."""
    if _query_fn is None:
        return None
    try:
        res = _query_fn({"features": list(features), "k": int(k)})
        if isinstance(res, dict) and res.get("error"):
            return None
        return res
    except Exception:
        return None


def mlx_infer(features: List[float], model_name: str = "delphi_linear") -> Optional[Dict[str, Any]]:
    """Fast on-device MLX forward pass (real when models/<name>.npz exists)."""
    if _mlx_fn is None:
        return None
    try:
        return _mlx_fn({"features": list(features), "model_name": str(model_name)})
    except Exception:
        return None


def regime_prior_from_matches(res: Optional[Dict[str, Any]], action: str) -> Dict[str, Any]:
    """Collapse k-NN matches into a bounded directional prior for the signal path.
    Returns {n, mean_outcome, agree_frac, prior} where `prior` ∈ roughly [-1,1] is the
    similarity-weighted, direction-aligned historical edge (0 if no usable matches)."""
    out = {"n": 0, "mean_outcome": 0.0, "agree_frac": 0.0, "prior": 0.0, "regimes": {}}
    if not res or not res.get("matches"):
        return out
    matches = res["matches"]
    sims, outs, agree, regimes = [], [], 0, {}
    want_up = (action == "long")
    for m in matches:
        meta = m.get("meta", {})
        o = meta.get("outcome")
        s = float(m.get("similarity", 0.0))
        if o is None:
            continue
        o = float(o)
        sims.append(max(s, 0.0))
        outs.append(o)
        if (o > 0) == want_up and o != 0:
            agree += 1
        rg = meta.get("regime", "?")
        regimes[rg] = regimes.get(rg, 0) + 1
    if not outs:
        return out
    n = len(outs)
    wsum = sum(sims) or 1.0
    mean_outcome = sum(outs) / n
    # similarity-weighted outcome, signed toward the intended direction
    w_outcome = sum(s * o for s, o in zip(sims, outs)) / wsum
    directional = w_outcome if want_up else -w_outcome
    prior = max(-1.0, min(1.0, directional * 200.0))   # ~0.5% move → full-scale
    out.update(
        n=n,
        mean_outcome=round(mean_outcome, 6),
        agree_frac=round(agree / n, 3),
        prior=round(prior, 4),
        regimes=regimes,
    )
    return out
