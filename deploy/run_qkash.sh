#!/usr/bin/env bash
#
# Serves QKash through Cloudflare Tunnel and cleans up both processes on exit.
#
#   deploy/run_qkash.sh named qkash.example.com   stable URL + Access gate
#   deploy/run_qkash.sh quick                     random URL, NO authentication
#
# Streamlit binds to loopback only; cloudflared is the sole public path in.
set -euo pipefail

MODE="${1:-named}"
HOSTNAME_ARG="${2:-}"
TUNNEL_NAME="${QKASH_TUNNEL_NAME:-qkash}"
PORT="${QKASH_PORT:-8503}"
CLOUDFLARED="${CLOUDFLARED:-$HOME/.local/bin/cloudflared}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${QKASH_PYTHON:-$ROOT/.venv/bin/python}"
LOGDIR="${TMPDIR:-/tmp}/qkash-tunnel"

cd "$ROOT"
mkdir -p "$LOGDIR"

if [ "$MODE" = "named" ] && [ -z "$HOSTNAME_ARG" ]; then
  echo "named mode needs a hostname: deploy/run_qkash.sh named qkash.example.com" >&2
  exit 2
fi

echo "== preflight =="
if ! ./deploy/preflight.sh; then
  echo
  echo "Preflight failed. Fix the items above before exposing the app." >&2
  exit 1
fi

APP_PID=""
TUNNEL_PID=""
cleanup() {
  echo
  echo "== stopping =="
  [ -n "$TUNNEL_PID" ] && kill "$TUNNEL_PID" 2>/dev/null || true
  [ -n "$APP_PID" ] && kill "$APP_PID" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# The project .streamlit/config.toml advertises host "qkash", which resolves
# nowhere but this workstation. In named mode the browser is pointed at the
# public hostname on 443; in quick mode the defaults are left to same-origin.
export STREAMLIT_SERVER_ADDRESS=127.0.0.1
export STREAMLIT_SERVER_PORT="$PORT"
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
if [ "$MODE" = "named" ]; then
  export STREAMLIT_BROWSER_SERVER_ADDRESS="$HOSTNAME_ARG"
  export STREAMLIT_BROWSER_SERVER_PORT=443
fi

echo
echo "== starting app on 127.0.0.1:${PORT} =="
"$PY" -m streamlit run app.py > "$LOGDIR/app.log" 2>&1 &
APP_PID=$!

for _ in $(seq 1 60); do
  if curl -sf -o /dev/null "http://127.0.0.1:${PORT}/_stcore/health" 2>/dev/null; then
    echo "app healthy (pid $APP_PID)"
    break
  fi
  if ! kill -0 "$APP_PID" 2>/dev/null; then
    echo "app died on startup:" >&2
    tail -20 "$LOGDIR/app.log" >&2
    exit 1
  fi
  sleep 1
done

echo
echo "== starting tunnel =="
if [ "$MODE" = "named" ]; then
  "$CLOUDFLARED" tunnel --no-autoupdate \
    --url "http://127.0.0.1:${PORT}" "$TUNNEL_NAME" > "$LOGDIR/tunnel.log" 2>&1 &
  TUNNEL_PID=$!
  PUBLIC_URL="https://${HOSTNAME_ARG}"
else
  "$CLOUDFLARED" tunnel --no-autoupdate \
    --url "http://127.0.0.1:${PORT}" > "$LOGDIR/tunnel.log" 2>&1 &
  TUNNEL_PID=$!
  PUBLIC_URL=""
  for _ in $(seq 1 40); do
    PUBLIC_URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' \
      "$LOGDIR/tunnel.log" 2>/dev/null | head -1)
    [ -n "$PUBLIC_URL" ] && break
    sleep 1
  done
fi

# A hostname is not a working tunnel: confirm the edge actually reaches back.
echo "verifying ${PUBLIC_URL:-<no url>} ..."
ok=0
for _ in $(seq 1 24); do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 \
    "${PUBLIC_URL}/_stcore/health" 2>/dev/null)
  if [ "$code" = "200" ]; then ok=1; break; fi
  sleep 5
done

echo
if [ "$ok" = "1" ]; then
  echo "=============================================================="
  echo "  QKash is live:  $PUBLIC_URL"
  [ "$MODE" = "quick" ] && echo "  WARNING: quick tunnels are PUBLIC and UNAUTHENTICATED."
  echo "  Ctrl-C stops the app and the tunnel."
  echo "=============================================================="
else
  echo "Tunnel did not come up (last HTTP code: ${code:-none})." >&2
  echo "HTTP 530 means the edge cannot reach back - check outbound 7844." >&2
  tail -15 "$LOGDIR/tunnel.log" >&2
  exit 1
fi

wait "$TUNNEL_PID"
