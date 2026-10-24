"""Safety smoke tests: this release must never place real orders by default."""
import asyncio, os, re, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import exchange_adapter as ea


def test_default_mode_is_paper(monkeypatch):
    monkeypatch.delenv("DELPHI_LIVE_ACK", raising=False)
    assert ea.make_router().is_paper()


def test_live_request_without_ack_is_forced_to_paper(monkeypatch):
    monkeypatch.delenv("DELPHI_LIVE_ACK", raising=False)
    for mode in ("live", "testnet"):
        assert ea.make_router(mode=mode).mode == "paper"


def test_wrong_ack_is_forced_to_paper(monkeypatch):
    monkeypatch.setenv("DELPHI_LIVE_ACK", "yes")
    assert ea.make_router(mode="live").mode == "paper"


def test_paper_orders_never_touch_real_venues(monkeypatch):
    monkeypatch.delenv("DELPHI_LIVE_ACK", raising=False)
    r = ea.make_router(mode="live")

    async def boom(*a, **k):
        raise AssertionError("real venue order attempted")

    async def paper_fill(*a, **k):
        return "paper-tx"

    r.aster.place_perp_order = boom
    r.apex.place_perp_order = boom
    r.paper.place_perp_order = paper_fill
    assert asyncio.run(r.place_perp_order("long", 1.0, 5)) == "paper-tx"


def test_no_hardcoded_credentials_or_personal_paths():
    bad = re.compile(r"/Users/[A-Za-z_]+|(?i:(api_?key|secret|passphrase|password)\s*=\s*['\"][^'\"$]{8,}['\"])")
    for p in ROOT.glob("*.py"):
        assert not bad.search(p.read_text(errors="ignore")), p.name
