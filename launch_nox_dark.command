#!/bin/bash
# ============================================================================
#  DELPHI NOX DARK 5.0 — combined launcher (engine + dashboard, paper mode)
#  Starts Ollama (AI veto) + the trading engine if down, then opens the
#  dashboard in Safari. Safe to double-click repeatedly (won't double-launch).
# ============================================================================
set -u
PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${DELPHI_PYTHON:-python3}"
URL="http://localhost:8080"
cd "$PROJ" || { echo "Project dir missing: $PROJ"; exit 1; }

echo "── DELPHI NOX DARK 5.0 ──────────────────────────────────────"

# 1) Ollama (local AI veto) — start if down, keep model resident
if ! pgrep -f "ollama serve" >/dev/null 2>&1; then
  echo "• starting Ollama…"
  OLLAMA_FLASH_ATTENTION=1 OLLAMA_KEEP_ALIVE=-1 nohup ollama serve >/tmp/ollama_cask.log 2>&1 &
else
  echo "• Ollama already running"
fi

# 2) Engine (paper, Nox Dark identity) — start if down
if ! pgrep -f "delphi_nova_1_0.py" >/dev/null 2>&1; then
  echo "• starting Delphi Nox Dark (paper)…"
  DELPHI_MODE=paper DELPHI_LIVE_ACK= DELPHI_PROFILE=nox_dark DELPHI_NAME="Delphi Nox Dark" DELPHI_SIZING_MODE=mcp_venue_routing \
    nohup "$PY" delphi_nova_1_0.py >/tmp/delphi_noxdark.log 2>&1 &
else
  echo "• Delphi Nox Dark already running (PID $(pgrep -f delphi_nova_1_0.py | head -1))"
fi

# 3) Wait for the API, then open the dashboard
echo -n "• waiting for dashboard API"
for i in $(seq 1 40); do
  curl -s --max-time 2 "$URL/api/status" >/dev/null 2>&1 && break
  echo -n "."; sleep 1
done
echo ""
open -a Safari "$URL"
echo "• dashboard → $URL"
echo "─────────────────────────────────────────────────────────────"
echo "  (paper mode — no real funds. Logs: /tmp/delphi_noxdark.log)"
