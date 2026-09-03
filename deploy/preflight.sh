#!/usr/bin/env bash
# Checks everything Cloudflare Tunnel needs before QKash is exposed.
# Exits non-zero on the first blocking failure.
set -uo pipefail

CLOUDFLARED="${CLOUDFLARED:-$HOME/.local/bin/cloudflared}"
PORT="${QKASH_PORT:-8503}"
fail=0

note() { printf '%-34s %s\n' "$1" "$2"; }

# 1. cloudflared present.
if [ -x "$CLOUDFLARED" ]; then
  note "cloudflared" "$("$CLOUDFLARED" --version 2>&1 | head -1)"
else
  note "cloudflared" "MISSING - see deploy/README.md"
  fail=1
fi

# 2. Outbound 7844. cloudflared reaches the edge only on this port, over UDP
#    (QUIC) or TCP (HTTP/2). Blocking it is the failure that returns HTTP 530
#    from a tunnel that otherwise looks healthy.
open=0
for host in region1.v2.argotunnel.com region2.v2.argotunnel.com; do
  if timeout 8 bash -c "cat < /dev/null > /dev/tcp/${host}/7844" 2>/dev/null; then
    note "edge 7844 (${host%%.*})" "OPEN"
    open=1
  else
    note "edge 7844 (${host%%.*})" "BLOCKED"
  fi
done
if [ "$open" -eq 0 ]; then
  note "" "-> This network blocks 7844. The tunnel will return HTTP 530."
  note "" "-> Switch networks or allow outbound 7844 (TCP+UDP)."
  fail=1
fi

# 3. Cloudflare login, required for a named tunnel only.
if [ -f "$HOME/.cloudflared/cert.pem" ]; then
  note "cloudflare login" "present"
else
  note "cloudflare login" "absent (run: cloudflared tunnel login)"
fi

# 4. Dataset. app.py raises FileNotFoundError at startup without it.
if [ -f data/processed/remittance_east_africa_clean.csv ] || [ -f data/RPW_dataset.csv ]; then
  note "dataset" "present"
else
  note "dataset" "MISSING - see docs/data_audit.md"
  fail=1
fi

# 5. App reachable, if already running.
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 \
  "http://127.0.0.1:${PORT}/_stcore/health" 2>/dev/null)
note "app on ${PORT}" "${code:-not running (run_qkash.sh will start it)}"

exit "$fail"
