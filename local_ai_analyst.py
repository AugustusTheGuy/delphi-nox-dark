"""local_ai_analyst.py — on-box AI second-opinion for Delphi Nox Dark.

Wraps this Mac's local Ollama models (delphi-trader-nox-dark → qwen2.5:7b-instruct)
as an INDEPENDENT directional confirmer/veto + a tightly-bounded, downgrade-biased
confidence-fusion member on CANDIDATE trades only. Zero cloud/Claude tokens.
Stdlib + `requests` only.

Safety contract:
  * The AI can only REMOVE or SHRINK a trade (hard veto, or a negative confidence
    haircut up to LOCAL_AI_FUSE_MAX_DOWN) and can ADD at most LOCAL_AI_FUSE_MAX_UP.
    It never touches leverage, size, SL/TP, MIN_EV_RATIO, or the confidence band.
  * When the model is down / slow / returns garbage it yields a clearly-labelled
    DEGRADED verdict (ok=False) and the engine behaves EXACTLY as without the AI —
    no silent heuristic ever masquerades as the model.
  * Invoked on candidate trades only, with a hard timeout, throttle, quantized-state
    cache, and a circuit breaker, so the 1s trading loop stays healthy.
"""
import json, time, hashlib, asyncio
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

try:
    import requests
except Exception:                       # pragma: no cover - requests is a hard dep of the engine
    requests = None


SYSTEM_VERDICT_PROMPT = (
    "You are the risk-aware second opinion for an autonomous SOL perpetual-futures "
    "scalper. You receive a JSON snapshot of a CANDIDATE trade the engine already "
    "wants to take. Judge ONLY direction quality and obvious danger.\n"
    "Hard rules:\n"
    "- You never propose leverage, size, SL, or TP. Risk is controlled elsewhere.\n"
    "- 40x is a hard cap that already binds; never imply otherwise. This is PAPER.\n"
    "- Output STRICT JSON, no prose, no markdown, exactly these keys:\n"
    '  {"direction":"long"|"short"|"flat","confidence":0.0-1.0,'
    '"veto":true|false,"rationale":"<=140 chars"}\n'
    "- veto=true requires direction OPPOSITE the engine action OR direction \"flat\" "
    "with confidence>=0.66. Never veto a trade you actually agree with.\n"
    "- If the snapshot is ambiguous, return direction \"flat\", low confidence, "
    "veto false. Abstaining is correct and safe."
)


SYSTEM_AUGMENT_PROMPT = (
    "You are the data-quality + macro/microstructure synthesis layer for an autonomous SOL "
    "perpetual-futures scalper. You receive a JSON snapshot of the engine's CURRENT "
    "multi-source market data (price/vol, funding, order-flow/OI, regime, sentiment, macro, "
    "liquidations). Your job is to (a) judge how trustworthy and internally consistent that "
    "data is, and (b) summarize the net directional lean the DATA implies. You do NOT place "
    "a trade.\n"
    "Hard rules:\n"
    "- You never propose leverage, size, SL, or TP. This is PAPER; 40x already binds.\n"
    "- data_quality is LOW (→0) when sources conflict, look stale/missing, or are internally "
    "inconsistent (e.g. funding sign vs price action, sentiment vs flow); HIGH (→1) when they "
    "corroborate each other.\n"
    "- Output STRICT JSON, no prose, no markdown, exactly these keys:\n"
    '  {"bias":"long"|"short"|"neutral","strength":0.0-1.0,"data_quality":0.0-1.0,'
    '"flags":["<=24 chars", ...],"rationale":"<=140 chars"}\n'
    "- Prefer \"neutral\" with low strength when evidence is mixed. Abstaining is safe.\n"
    "- flags lists concrete data issues you detect (<=4 items); [] if the data looks clean."
)


SYSTEM_NEWS_PROMPT = (
    "You are the news/event analyst for an autonomous SOL perpetual-futures scalper. You "
    "receive a list of recent crypto news headlines. Extract the market-relevant signal for "
    "SOL over the next few hours. You do NOT place a trade.\n"
    "Hard rules:\n"
    "- You never propose leverage, size, SL, or TP. This is PAPER.\n"
    "- sentiment: -1.0 (strongly bearish) .. +1.0 (strongly bullish) for SOL/crypto near-term.\n"
    "- event_risk: 0.0 (calm/routine) .. 1.0 (a major market-moving event is unfolding: crash, "
    "exchange/DeFi hack, regulatory shock, ETF decision, stablecoin depeg, large liquidation cascade).\n"
    "- regime: \"risk_on\" | \"risk_off\" | \"neutral\".\n"
    "- Output STRICT JSON, no prose, no markdown, exactly these keys:\n"
    '  {"sentiment":-1.0..1.0,"event_risk":0.0..1.0,"regime":"risk_on"|"risk_off"|"neutral",'
    '"catalysts":["<=24 chars", ...],"rationale":"<=140 chars"}\n'
    "- catalysts lists the concrete market-moving items you see (<=4); [] if headlines are routine.\n"
    "- If the headlines are routine/irrelevant to SOL, return sentiment near 0, low event_risk, "
    "neutral. Abstaining is correct and safe."
)


def _opposite(a: str) -> str:
    return 'short' if a == 'long' else ('long' if a == 'short' else 'flat')


def _r(v, nd: int = 4):
    """Round-or-None: NaN/inf/non-numeric → None (JSON null)."""
    try:
        f = float(v)
        if f != f or f in (float('inf'), float('-inf')):
            return None
        return round(f, nd)
    except Exception:
        return None


@dataclass
class AIVerdict:
    direction: str = 'flat'
    confidence: float = 0.0
    veto: bool = False
    rationale: str = ''
    ok: bool = False                  # True ONLY when the model genuinely answered & parsed
    degraded: Optional[str] = None    # 'timeout'|'conn'|'parse'|'schema_*'|'breaker'|'disabled'|...
    tier: str = 'none'                # 'nox_dark'|'qwen_base'|'mlx'|'none'
    latency_ms: float = 0.0
    disagree_strength: float = 0.0

    def as_log_dict(self) -> Dict[str, Any]:
        return {'direction': self.direction, 'confidence': round(float(self.confidence), 3),
                'veto': bool(self.veto), 'ok': bool(self.ok), 'degraded': self.degraded,
                'tier': self.tier, 'latency_ms': round(float(self.latency_ms), 1),
                'rationale': str(self.rationale)[:140]}


def parse_verdict(content: str, action: str) -> AIVerdict:
    """Strict, dependency-light parse of the model's JSON reply against `action`.
    Any malformation degrades to a safe verdict that can never STRENGTHEN a trade."""
    raw = content
    obj = None
    try:
        obj = json.loads(raw)
    except Exception:
        # single salvage attempt: extract the outermost {...}
        try:
            i, j = raw.index('{'), raw.rindex('}')
            obj = json.loads(raw[i:j + 1])
        except Exception:
            return AIVerdict(ok=False, degraded='parse')
    if not isinstance(obj, dict):
        return AIVerdict(ok=False, degraded='parse')

    degraded = None
    d = str(obj.get('direction', 'flat')).strip().lower()
    if d not in ('long', 'short', 'flat'):
        d = 'flat'
    try:
        c_raw = float(obj.get('confidence', 0.5))
        if c_raw != c_raw:                       # NaN
            raise ValueError
        if c_raw < 0.0 or c_raw > 1.0:           # out of range → flag, do not silently clamp
            c, degraded = max(0.0, min(1.0, c_raw)), 'schema_conf'
        else:
            c = c_raw
    except Exception:
        c, degraded = 0.5, 'schema_conf'
    v = obj.get('veto', False)
    v = bool(v) if isinstance(v, bool) else str(v).strip().lower() in ('true', '1', 'yes')
    rationale = str(obj.get('rationale', ''))[:140]

    # self-consistency: cannot veto a trade you agree with
    if v and d == action:
        v, degraded = False, (degraded or 'inconsistent_veto')

    # disagreement strength vs the engine's committed action
    if d == _opposite(action) and d != 'flat':
        disagree = c
    elif d == 'flat':
        disagree = c * 0.5
    else:
        disagree = 0.0

    return AIVerdict(direction=d, confidence=c, veto=v, rationale=rationale,
                     ok=(degraded is None), degraded=degraded, tier='parsed',
                     disagree_strength=disagree)


def fuse_ai_confidence(c0: float, action: str, verdict: AIVerdict, config: Dict) -> float:
    """Bounded, asymmetric, downgrade-biased fusion. The AI gets its OWN budget,
    separate from the engine's ±0.12 overlay cap, applied after that cap is locked.
    Returns the adjusted confidence; never raises leverage/EV/size."""
    if verdict is None or not verdict.ok:
        return c0
    up = float(config.get('LOCAL_AI_FUSE_MAX_UP', 0.03))
    down = float(config.get('LOCAL_AI_FUSE_MAX_DOWN', 0.08))
    gain = float(config.get('LOCAL_AI_FUSE_GAIN', 0.10))
    d, q = verdict.direction, float(verdict.confidence)
    if d == action:
        s = +q
    elif d == _opposite(action):
        s = -q
    else:                                  # flat → mild evidence against
        s = -0.5 * q
    raw = gain * s
    delta = max(-down, min(up, raw))
    return max(0.0, min(1.0, c0 + delta))


def apply_ai_to_signal(confidence: float, action: str, verdict: AIVerdict, config: Dict):
    """Pure, testable mirror of engine Hook A. Returns
    (should_continue, new_confidence, new_action, veto_fired).
    should_continue=False ⇒ the engine returns (False, confidence, 'hold', ...)."""
    if verdict is None or not verdict.ok:
        return True, confidence, action, False
    if (config.get('LOCAL_AI_VETO_ENABLE', True) and verdict.veto
            and verdict.disagree_strength >= float(config.get('LOCAL_AI_VETO_MIN_DISAGREE', 0.62))):
        return False, confidence, 'hold', True
    return True, fuse_ai_confidence(confidence, action, verdict, config), action, False


# ───────────────────────── data-augmentation layer (v6) ─────────────────────────
@dataclass
class AIAugmentation:
    """The AI's read of the CURRENT market DATA (not a trade): a directional lean plus a
    data-quality score the engine uses for a down-only, risk-reducing confidence haircut."""
    bias: str = 'neutral'              # 'long'|'short'|'neutral'
    strength: float = 0.0              # 0..1 conviction in `bias`
    data_quality: float = 1.0          # 0..1 trust in the inputs (1.0 ⇒ no haircut)
    flags: Optional[List[str]] = None  # concrete data issues detected
    rationale: str = ''
    ok: bool = False
    degraded: Optional[str] = None
    tier: str = 'none'
    latency_ms: float = 0.0
    ts: float = 0.0                    # wall-clock (epoch s) when produced

    def __post_init__(self):
        if self.flags is None:
            self.flags = []

    def as_dict(self) -> Dict[str, Any]:
        return {'bias': self.bias, 'strength': round(float(self.strength), 3),
                'data_quality': round(float(self.data_quality), 3),
                'flags': list(self.flags or [])[:4], 'rationale': str(self.rationale)[:140],
                'ok': bool(self.ok), 'degraded': self.degraded, 'tier': self.tier,
                'latency_ms': round(float(self.latency_ms), 1), 'ts': self.ts}


def parse_augmentation(content: str) -> AIAugmentation:
    """Strict, dependency-light parse of the augmentation JSON. Any malformation degrades to a
    SAFE result (neutral bias, data_quality=1.0 ⇒ no haircut) so a bad parse can never by
    itself harm trading."""
    obj = None
    try:
        obj = json.loads(content)
    except Exception:
        try:
            i, j = content.index('{'), content.rindex('}')
            obj = json.loads(content[i:j + 1])
        except Exception:
            return AIAugmentation(ok=False, degraded='parse')
    if not isinstance(obj, dict):
        return AIAugmentation(ok=False, degraded='parse')
    degraded = None
    b = str(obj.get('bias', 'neutral')).strip().lower()
    if b not in ('long', 'short', 'neutral'):
        b = 'neutral'

    def _num(key, default):
        nonlocal degraded
        try:
            x = float(obj.get(key, default))
            if x != x:                                   # NaN
                raise ValueError
            if x < 0.0 or x > 1.0:                        # out of range → flag, clamp
                degraded = degraded or 'schema_range'
                return max(0.0, min(1.0, x))
            return x
        except Exception:
            degraded = degraded or 'schema_range'
            return default

    strength = _num('strength', 0.0)
    dq = _num('data_quality', 1.0)
    flags = obj.get('flags', [])
    if not isinstance(flags, list):
        flags = []
    flags = [str(x)[:24] for x in flags][:4]
    rationale = str(obj.get('rationale', ''))[:140]
    return AIAugmentation(bias=b, strength=strength, data_quality=dq, flags=flags,
                          rationale=rationale, ok=(degraded is None), degraded=degraded,
                          tier='parsed', ts=time.time())


def augmentation_factor(aug, config: Dict) -> float:
    """Down-only confidence multiplier from the AI data-quality verdict. Accepts either an
    AIAugmentation or its .as_dict() form (so the engine can pass the cached dict directly).
    Returns a value in [1-LOCAL_AI_AUGMENT_MAX_HAIRCUT, 1.0]; 1.0 = no change. Can ONLY shrink
    confidence when the AI judges the inputs low-quality/inconsistent — never raises it."""
    if aug is None:
        return 1.0
    if isinstance(aug, dict):
        ok, dq = aug.get('ok', False), aug.get('data_quality', 1.0)
    else:
        ok, dq = getattr(aug, 'ok', False), getattr(aug, 'data_quality', 1.0)
    if not ok:
        return 1.0
    max_cut = float(config.get('LOCAL_AI_AUGMENT_MAX_HAIRCUT', 0.06))
    dq = max(0.0, min(1.0, float(dq)))
    return max(1.0 - max_cut, 1.0 - max_cut * (1.0 - dq))


# ───────────────────────── news / event-sentiment layer (Q2) ─────────────────────────
@dataclass
class NewsVerdict:
    """The AI's read of recent crypto news for SOL near-term: a sentiment lean, an event-risk
    severity, a risk regime, and the concrete catalysts it found. Display-first; wired into the
    signal only after validation."""
    sentiment: float = 0.0        # -1..1 bearish..bullish (SOL/crypto, near-term)
    event_risk: float = 0.0       # 0..1 severity of a market-moving event unfolding
    regime: str = 'neutral'       # 'risk_on'|'risk_off'|'neutral'
    catalysts: Optional[List[str]] = None
    rationale: str = ''
    headlines_n: int = 0
    ok: bool = False
    degraded: Optional[str] = None
    tier: str = 'none'
    latency_ms: float = 0.0
    ts: float = 0.0

    def __post_init__(self):
        if self.catalysts is None:
            self.catalysts = []

    def as_dict(self) -> Dict[str, Any]:
        return {'sentiment': round(float(self.sentiment), 3),
                'event_risk': round(float(self.event_risk), 3), 'regime': self.regime,
                'catalysts': list(self.catalysts or [])[:4], 'rationale': str(self.rationale)[:140],
                'headlines_n': int(self.headlines_n), 'ok': bool(self.ok), 'degraded': self.degraded,
                'tier': self.tier, 'latency_ms': round(float(self.latency_ms), 1), 'ts': self.ts}


def parse_news(content: str) -> NewsVerdict:
    """Strict, dependency-light parse of the news JSON. Malformation ⇒ SAFE neutral result."""
    obj = None
    try:
        obj = json.loads(content)
    except Exception:
        try:
            i, j = content.index('{'), content.rindex('}')
            obj = json.loads(content[i:j + 1])
        except Exception:
            return NewsVerdict(ok=False, degraded='parse')
    if not isinstance(obj, dict):
        return NewsVerdict(ok=False, degraded='parse')
    degraded = None

    def _num(key, default, lo, hi):
        nonlocal degraded
        try:
            x = float(obj.get(key, default))
            if x != x:
                raise ValueError
            if x < lo or x > hi:
                degraded = degraded or 'schema_range'
                return max(lo, min(hi, x))
            return x
        except Exception:
            degraded = degraded or 'schema_range'
            return default

    sentiment = _num('sentiment', 0.0, -1.0, 1.0)
    event_risk = _num('event_risk', 0.0, 0.0, 1.0)
    reg = str(obj.get('regime', 'neutral')).strip().lower()
    if reg not in ('risk_on', 'risk_off', 'neutral'):
        reg = 'neutral'
    cats = obj.get('catalysts', [])
    if not isinstance(cats, list):
        cats = []
    cats = [str(x)[:24] for x in cats][:4]
    rationale = str(obj.get('rationale', ''))[:140]
    return NewsVerdict(sentiment=sentiment, event_risk=event_risk, regime=reg, catalysts=cats,
                       rationale=rationale, ok=(degraded is None), degraded=degraded,
                       tier='parsed', ts=time.time())


class LocalAIAnalyst:
    """Ollama-backed analyst with timeout, throttle, quantized-state cache, breaker,
    and tiered fallback. All network is localhost only."""

    def __init__(self, config: Dict, logger=None):
        self.cfg = config
        self.log = logger
        self.base = str(config.get('LOCAL_AI_BASE_URL', 'http://localhost:11434')).rstrip('/')
        self._cache: Dict[str, Any] = {}          # key -> (verdict, monotonic_ts)
        self._last_call_ts = 0.0
        self._fail_streak = 0
        self._breaker_until = 0.0
        self.calls = 0
        self.vetoes = 0
        self.degrades = 0
        self._lat = []                            # recent latencies (ms)
        self._aug_cache: Dict[str, Any] = {}      # data-augmentation cache: key -> (aug, ts)
        self.aug_calls = 0
        self._news_cache: Dict[str, Any] = {}     # news-analysis cache: key -> (verdict, ts)
        self.news_calls = 0

    # ----- cache key on a coarsely-quantized snapshot (determinism + dedup) -----
    @staticmethod
    def _quantize_key(snap: Dict) -> str:
        e = snap.get('engine', {})
        p = snap.get('price', {})
        f = snap.get('funding', {})
        ev = snap.get('evidence', {})
        coarse = {
            'a': e.get('action'),
            'r': e.get('regime'),
            'c': round(float(e.get('confidence', 0) or 0), 2),
            'atr': round(float(p.get('atr_pct', 0) or 0), 3),
            'rsi': round(float(p.get('rsi', 0) or 0) / 5.0),
            'fz': round(float(f.get('funding_z', 0) or 0) * 2) / 2.0,
            'mlh': round(float(snap.get('model_probs', {}).get('ml_hold', 0) or 0), 1),
            'post': round(float(ev.get('posterior', 0) or 0), 1),
        }
        blob = json.dumps(coarse, sort_keys=True)
        return hashlib.blake2b(blob.encode(), digest_size=12).hexdigest()

    def _cache_get(self, key: str):
        ttl = float(self.cfg.get('LOCAL_AI_CACHE_TTL_SEC', 8.0))
        hit = self._cache.get(key)
        if hit and (time.monotonic() - hit[1]) <= ttl:
            return hit[0]
        return None

    def _cache_put(self, key: str, verdict: AIVerdict):
        self._cache[key] = (verdict, time.monotonic())
        if len(self._cache) > 256:                # bound memory
            old = sorted(self._cache.items(), key=lambda kv: kv[1][1])[:64]
            for k, _ in old:
                self._cache.pop(k, None)

    # ----- one blocking HTTP call (run via executor) -----
    def _call_ollama_sync(self, model: str, snapshot: Dict,
                          system_prompt: str = None, max_tokens: int = None,
                          timeout_sec: float = None) -> str:
        if requests is None:
            raise RuntimeError('requests unavailable')
        sysp = system_prompt or SYSTEM_VERDICT_PROMPT
        ntok = int(max_tokens if max_tokens is not None
                   else self.cfg.get('LOCAL_AI_MAX_TOKENS', 96))
        to = float(timeout_sec if timeout_sec is not None
                   else self.cfg.get('LOCAL_AI_TIMEOUT_SEC', 1.2))
        payload = {
            'model': model,
            'stream': False,
            'format': 'json',
            'keep_alive': self.cfg.get('LOCAL_AI_KEEP_ALIVE', '30m'),
            'options': {
                'temperature': float(self.cfg.get('LOCAL_AI_TEMPERATURE', 0.2)),
                'top_p': 0.9,
                'num_predict': ntok,
                'num_ctx': int(self.cfg.get('LOCAL_AI_NUM_CTX', 4096)),
            },
            'messages': [
                {'role': 'system', 'content': sysp},
                {'role': 'user', 'content': json.dumps(snapshot, separators=(',', ':'))},
            ],
        }
        resp = requests.post(self.base + '/api/chat', json=payload,
                             timeout=(min(0.5, to), to))
        resp.raise_for_status()
        data = resp.json()
        return data.get('message', {}).get('content', '')

    async def _try_model(self, model: str, tier: str, snapshot: Dict) -> AIVerdict:
        action = snapshot.get('engine', {}).get('action', 'flat')
        to = float(self.cfg.get('LOCAL_AI_TIMEOUT_SEC', 1.2))
        loop = asyncio.get_event_loop()
        t0 = time.monotonic()
        try:
            content = await asyncio.wait_for(
                loop.run_in_executor(None, self._call_ollama_sync, model, snapshot), timeout=to)
        except asyncio.TimeoutError:
            return AIVerdict(ok=False, degraded='timeout', tier=tier)
        except Exception as e:
            nm = type(e).__name__
            if 'Timeout' in nm:
                kind = 'timeout'
            elif 'HTTPError' in nm or 'Status' in nm:
                kind = 'http'          # e.g. Ollama 500 (runner missing) — NOT a conn issue
            elif 'Connection' in nm:
                kind = 'conn'
            else:
                kind = 'error'
            return AIVerdict(ok=False, degraded=kind, tier=tier)
        v = parse_verdict(content, action)
        v.tier = tier
        v.latency_ms = (time.monotonic() - t0) * 1000.0
        return v

    async def evaluate(self, snapshot: Dict) -> AIVerdict:
        """Main entry: returns a verdict for the candidate trade. Never raises."""
        if not self.cfg.get('LOCAL_AI_ENABLE', True):
            return AIVerdict(ok=False, degraded='disabled')
        now = time.monotonic()
        # circuit breaker
        if now < self._breaker_until:
            return AIVerdict(ok=False, degraded='breaker')
        # cache
        key = self._quantize_key(snapshot)
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        # throttle (min interval between DISTINCT calls)
        min_int = float(self.cfg.get('LOCAL_AI_MIN_INTERVAL_SEC', 2.0))
        if (now - self._last_call_ts) < min_int:
            return AIVerdict(ok=False, degraded='throttled')
        self._last_call_ts = now
        self.calls += 1

        tiers: List = [(self.cfg.get('LOCAL_AI_MODEL', 'delphi-trader-nox-dark:latest'), 'nox_dark')]
        fb = self.cfg.get('LOCAL_AI_FALLBACK_MODEL')
        if fb:
            tiers.append((fb, 'qwen_base'))

        verdict = AIVerdict(ok=False, degraded='conn')
        for model, tier in tiers:
            verdict = await self._try_model(model, tier, snapshot)
            if verdict.ok:
                break

        # breaker / health bookkeeping
        if verdict.ok:
            self._fail_streak = 0
            self._lat.append(verdict.latency_ms)
            self._lat = self._lat[-50:]
        else:
            self.degrades += 1
            self._fail_streak += 1
            if self._fail_streak >= int(self.cfg.get('LOCAL_AI_FAIL_OPEN_N', 3)):
                self._breaker_until = now + float(self.cfg.get('LOCAL_AI_COOLDOWN_SEC', 60))
                self._fail_streak = 0
        if verdict.veto and verdict.ok:
            self.vetoes += 1
        self._cache_put(key, verdict)
        return verdict

    # ----- data-augmentation (background loop; off the trading path) -----
    async def _try_augment(self, model: str, tier: str, snapshot: Dict) -> AIAugmentation:
        to = float(self.cfg.get('LOCAL_AI_AUGMENT_TIMEOUT_SEC', 2.5))
        ntok = int(self.cfg.get('LOCAL_AI_AUGMENT_MAX_TOKENS', 110))
        loop = asyncio.get_event_loop()
        t0 = time.monotonic()
        try:
            content = await asyncio.wait_for(
                loop.run_in_executor(None, self._call_ollama_sync, model, snapshot,
                                     SYSTEM_AUGMENT_PROMPT, ntok, to), timeout=to + 0.5)
        except asyncio.TimeoutError:
            return AIAugmentation(ok=False, degraded='timeout', tier=tier)
        except Exception as e:
            nm = type(e).__name__
            kind = ('timeout' if 'Timeout' in nm
                    else 'http' if ('HTTPError' in nm or 'Status' in nm)
                    else 'conn' if 'Connection' in nm else 'error')
            return AIAugmentation(ok=False, degraded=kind, tier=tier)
        a = parse_augmentation(content)
        a.tier = tier
        a.latency_ms = (time.monotonic() - t0) * 1000.0
        return a

    async def augment(self, data_snapshot: Dict) -> AIAugmentation:
        """Judge data quality + net directional lean of the CURRENT market data. Designed to
        run on a background loop (NOT the trading path), so it may use a larger timeout. Shares
        the circuit breaker + fail-streak with evaluate(). Never raises; degrades cleanly."""
        if not self.cfg.get('LOCAL_AI_AUGMENT_ENABLE', True):
            return AIAugmentation(ok=False, degraded='disabled')
        now = time.monotonic()
        if now < self._breaker_until:
            return AIAugmentation(ok=False, degraded='breaker')
        try:
            key = 'aug:' + hashlib.blake2b(
                json.dumps(data_snapshot, sort_keys=True, default=str).encode(),
                digest_size=10).hexdigest()
        except Exception:
            key = 'aug:fallback'
        ttl = float(self.cfg.get('LOCAL_AI_AUGMENT_TTL_SEC', 45.0))
        hit = self._aug_cache.get(key)
        if hit and (time.monotonic() - hit[1]) <= ttl:
            return hit[0]
        self.aug_calls += 1
        tiers: List = [(self.cfg.get('LOCAL_AI_MODEL', 'delphi-trader-nox-dark:latest'), 'nox_dark')]
        fb = self.cfg.get('LOCAL_AI_FALLBACK_MODEL')
        if fb:
            tiers.append((fb, 'qwen_base'))
        aug = AIAugmentation(ok=False, degraded='conn')
        for model, tier in tiers:
            aug = await self._try_augment(model, tier, data_snapshot)
            if aug.ok:
                break
        if aug.ok:
            self._fail_streak = 0
            self._lat.append(aug.latency_ms); self._lat = self._lat[-50:]
        else:
            self.degrades += 1
            self._fail_streak += 1
            if self._fail_streak >= int(self.cfg.get('LOCAL_AI_FAIL_OPEN_N', 3)):
                self._breaker_until = now + float(self.cfg.get('LOCAL_AI_COOLDOWN_SEC', 60))
                self._fail_streak = 0
        self._aug_cache[key] = (aug, time.monotonic())
        if len(self._aug_cache) > 64:                 # bound memory
            for k, _ in sorted(self._aug_cache.items(), key=lambda kv: kv[1][1])[:32]:
                self._aug_cache.pop(k, None)
        return aug

    # ----- news / event-sentiment (background loop; off the trading path) -----
    async def _try_news(self, model: str, tier: str, headlines: List[str]) -> NewsVerdict:
        to = float(self.cfg.get('LOCAL_AI_NEWS_TIMEOUT_SEC', 8.0))
        ntok = int(self.cfg.get('LOCAL_AI_NEWS_MAX_TOKENS', 150))
        payload = {'schema': 'nox_dark.news.v1', 'headlines': list(headlines)[:20]}
        loop = asyncio.get_event_loop()
        t0 = time.monotonic()
        try:
            content = await asyncio.wait_for(
                loop.run_in_executor(None, self._call_ollama_sync, model, payload,
                                     SYSTEM_NEWS_PROMPT, ntok, to), timeout=to + 0.5)
        except asyncio.TimeoutError:
            return NewsVerdict(ok=False, degraded='timeout', tier=tier, headlines_n=len(headlines))
        except Exception as e:
            nm = type(e).__name__
            kind = ('timeout' if 'Timeout' in nm
                    else 'http' if ('HTTPError' in nm or 'Status' in nm)
                    else 'conn' if 'Connection' in nm else 'error')
            return NewsVerdict(ok=False, degraded=kind, tier=tier, headlines_n=len(headlines))
        v = parse_news(content)
        v.tier = tier
        v.headlines_n = len(headlines)
        v.latency_ms = (time.monotonic() - t0) * 1000.0
        return v

    async def analyze_news(self, headlines: List[str]) -> NewsVerdict:
        """Read recent crypto headlines into a structured SOL near-term sentiment / event-risk
        verdict. Background loop (off the trading path); shares the breaker. Never raises."""
        if not self.cfg.get('LOCAL_AI_NEWS_ENABLE', True):
            return NewsVerdict(ok=False, degraded='disabled')
        if not headlines:
            return NewsVerdict(ok=False, degraded='no_headlines')
        now = time.monotonic()
        if now < self._breaker_until:
            return NewsVerdict(ok=False, degraded='breaker')
        try:
            key = 'news:' + hashlib.blake2b('|'.join(headlines).encode(), digest_size=10).hexdigest()
        except Exception:
            key = 'news:fallback'
        ttl = float(self.cfg.get('LOCAL_AI_NEWS_TTL_SEC', 600.0))
        hit = self._news_cache.get(key)
        if hit and (time.monotonic() - hit[1]) <= ttl:
            return hit[0]
        self.news_calls += 1
        # Prefer a general-purpose model for news reading (better at it than the trade-specialised
        # fine-tune); fall back to the others if it is unavailable.
        news_model = (self.cfg.get('LOCAL_AI_NEWS_MODEL') or self.cfg.get('LOCAL_AI_FALLBACK_MODEL')
                      or self.cfg.get('LOCAL_AI_MODEL', 'delphi-trader-nox-dark:latest'))
        tiers: List = [(news_model, 'news')]
        for m, t in ((self.cfg.get('LOCAL_AI_MODEL'), 'nox_dark'),
                     (self.cfg.get('LOCAL_AI_FALLBACK_MODEL'), 'qwen_base')):
            if m and m != news_model:
                tiers.append((m, t))
        v = NewsVerdict(ok=False, degraded='conn')
        for model, tier in tiers:
            v = await self._try_news(model, tier, headlines)
            if v.ok:
                break
        if v.ok:
            self._fail_streak = 0
            self._lat.append(v.latency_ms); self._lat = self._lat[-50:]
        else:
            self.degrades += 1
            self._fail_streak += 1
            if self._fail_streak >= int(self.cfg.get('LOCAL_AI_FAIL_OPEN_N', 3)):
                self._breaker_until = now + float(self.cfg.get('LOCAL_AI_COOLDOWN_SEC', 60))
                self._fail_streak = 0
        self._news_cache[key] = (v, time.monotonic())
        if len(self._news_cache) > 32:
            for k, _ in sorted(self._news_cache.items(), key=lambda kv: kv[1][1])[:16]:
                self._news_cache.pop(k, None)
        return v

    def health(self) -> Dict[str, Any]:
        lat = sorted(self._lat)
        p50 = lat[len(lat) // 2] if lat else 0.0
        p95 = lat[int(len(lat) * 0.95)] if lat else 0.0
        return {'calls': self.calls, 'aug_calls': self.aug_calls, 'news_calls': self.news_calls,
                'vetoes': self.vetoes, 'degrades': self.degrades,
                'breaker_open': time.monotonic() < self._breaker_until,
                'p50_ms': round(p50, 1), 'p95_ms': round(p95, 1)}


def build_ai_snapshot(action: str, confidence: float, regime: str, last, details: Dict,
                      state, quant=None, threshold: float = 0.0) -> Dict:
    """Compact, decision-relevant, PII/secret-free snapshot built from objects already
    in scope at the signal hook. `last` is the latest bar (dict-like)."""
    mp = details.get('_ml_probs', {})
    def g(k, d=None):
        try:
            return last.get(k, d)
        except Exception:
            return d
    recent = []
    try:
        for t in (getattr(state, 'trade_log', []) or [])[-5:]:
            recent.append({'dir': t.get('direction'),
                           'pnl': _r((t.get('pnl_usd', 0) or 0) - (t.get('fee_usd', 0) or 0), 3),
                           'reg': t.get('regime')})
    except Exception:
        pass
    return {
        'schema': 'nox_dark.v1',
        'engine': {'action': action, 'confidence': _r(confidence, 3), 'regime': regime,
                   'threshold': _r(threshold, 3)},
        'model_probs': {'ml_long': _r(mp.get('ml_long')), 'ml_hold': _r(mp.get('ml_hold')),
                        'ml_short': _r(mp.get('ml_short')), 'dl_long': _r(mp.get('dl_long')),
                        'dl_hold': _r(mp.get('dl_hold')), 'dl_short': _r(mp.get('dl_short')),
                        'ensemble_mode': details.get('ensemble_mode')},
        'price': {'last': _r(g('close'), 4), 'atr_pct': _r(g('atr_pct'), 4),
                  'rsi': _r(g('rsi')), 'adx': _r(g('adx')),
                  'hurst': _r(details.get('hurst'), 3), 'ewma_vol': _r(g('ewma_vol'), 4)},
        'micro': {'vpin': _r(g('vpin'), 3), 'ofi': _r(g('ofi'), 3),
                  'mtf_alignment': _r(details.get('mtf_alignment'), 3),
                  'crowded': _r(details.get('crowded'), 3),
                  'velocity': details.get('velocity')},
        'funding': {'rate': _r(g('funding_rate'), 5),
                    'funding_z': _r(details.get('funding_z'), 2),
                    'oracle_mark_spread': _r(details.get('oracle_mark_spread'), 4)},
        'evidence': {'posterior': _r(details.get('posterior'), 3),
                     'metalabel_winrate': _r(details.get('metalabel_winrate'), 3),
                     'info_coefficient': _r(details.get('info_coefficient'), 3),
                     'bocpd_p': _r(details.get('bocpd_p'), 3),
                     'doppel_mean': _r(details.get('doppel_mean'), 4)},
        'account': {'consecutive_wins': getattr(state, 'consecutive_wins', 0),
                    'consecutive_losses': getattr(state, 'consecutive_losses', 0),
                    'recent': recent},
    }


def build_data_snapshot(state, quant=None) -> Dict:
    """Compact, PII/secret-free snapshot of the engine's CURRENT multi-source market DATA for
    the AI data-augmentation layer. Built purely from `state` fields the engine already
    maintains (no new fetches), so it never adds API load."""
    md = getattr(state, 'macro_data', {}) or {}
    vol = None
    try:
        vol = (quant.garch_vol() * 725.0 * 100) if quant is not None else None
    except Exception:
        vol = None
    return {
        'schema': 'nox_dark.aug.v1',
        'price': {'sol': _r(getattr(state, 'sol_price', 0), 4),
                  'regime': getattr(state, 'current_regime', 'unknown')},
        'funding': {'rate': _r(getattr(state, 'funding_rate', 0), 6),
                    'cross_venue_basis': _r(getattr(state, 'cross_venue_basis', 0), 5)},
        'flow': {'open_interest': _r(getattr(state, 'open_interest', 0), 0),
                 'long_short_ratio': _r(getattr(state, 'long_short_ratio', 0), 3),
                 'liq_long': _r(getattr(state, 'liq_long', 0), 0),
                 'liq_short': _r(getattr(state, 'liq_short', 0), 0)},
        'sentiment': {'score': _r(getattr(state, 'sentiment_score', 0), 3)},
        'macro': {'dxy': _r(md.get('dxy'), 2), 'vix': _r(md.get('vix'), 2),
                  'treasury_10y': _r(md.get('treasury_10y'), 2)},
        'vol': {'ewma_garch_annual_pct': _r(vol, 1)},
    }
