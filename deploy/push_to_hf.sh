#!/usr/bin/env bash
#
# Publishes QKash to a private Hugging Face Space that builds the Dockerfile
# server-side. No local Docker daemon required.
#
#   HF_TOKEN=hf_xxx deploy/push_to_hf.sh <hf-username> [space-name]
#
# Get a WRITE token at https://huggingface.co/settings/tokens
#
# NOTE: creating this Space PRIVATE returns HTTP 402 on a free account -- HF
# requires PRO for private Docker Spaces. Set private:false below only if the
# dataset is cleared for publication; the Space repo is world-readable.
#
# The Space repo is separate from GitHub: the audited CSV is committed here so
# the app can read it at startup, and never to the GitHub repo. The Space is
# created PRIVATE. Flip it in Space settings only if that data is cleared for
# publication.
set -euo pipefail

USERNAME="${1:-}"
SPACE="${2:-qkash}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${TMPDIR:-/tmp}/qkash-hf-$$"

if [ -z "$USERNAME" ]; then
  echo "usage: HF_TOKEN=hf_xxx $0 <hf-username> [space-name]" >&2
  exit 2
fi
if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is not set. Create a WRITE token at" >&2
  echo "https://huggingface.co/settings/tokens" >&2
  exit 2
fi

CSV="$ROOT/data/processed/remittance_east_africa_clean.csv"
if [ ! -f "$CSV" ]; then
  echo "Missing $CSV - see docs/data_audit.md" >&2
  exit 1
fi

echo "== creating Space ${USERNAME}/${SPACE} (private, docker sdk) =="
curl -fsS -X POST https://huggingface.co/api/repos/create \
  -H "Authorization: Bearer ${HF_TOKEN}" \
  -H "Content-Type: application/json" \
  -d "{\"name\":\"${SPACE}\",\"type\":\"space\",\"sdk\":\"docker\",\"private\":true}" \
  > /dev/null && echo "created" || echo "already exists (continuing)"

echo "== assembling Space repo =="
rm -rf "$WORK"
# The token is supplied through a credential helper that reads it from the
# environment at call time, so it is never written into the remote URL or left
# behind in .git/config, and cannot leak through git's error output.
export HF_TOKEN
CRED_HELPER='!f(){ echo username=hf; echo "password=${HF_TOKEN}"; }; f'
git clone --quiet -c credential.helper="$CRED_HELPER" \
  "https://huggingface.co/spaces/${USERNAME}/${SPACE}" "$WORK"

# The Space card carries the sdk/app_port frontmatter HF reads at build time.
cp "$ROOT/deploy/hf-space/README.md" "$WORK/README.md"
cp "$ROOT/Dockerfile" "$ROOT/requirements.txt" "$ROOT/app.py" "$WORK/"
rm -rf "$WORK/qkash" && cp -r "$ROOT/qkash" "$WORK/qkash"
find "$WORK/qkash" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
mkdir -p "$WORK/data/processed"
cp "$ROOT/data/QKash.png" "$WORK/data/"
cp "$CSV" "$WORK/data/processed/"

# .dockerignore excludes the dataset directory patterns meant for GitHub; the
# Space needs the processed CSV inside the build context, so it is not copied.

cd "$WORK"
git config user.email "rogerzmukiibi@gmail.com"
git config user.name "rogerzmukiibi"
git add -A
if git diff --cached --quiet; then
  echo "no changes to push"
else
  git commit --quiet -m "Deploy QKash Streamlit app"
  git push --quiet origin main
  echo "pushed"
fi

cd "$ROOT" && rm -rf "$WORK"
echo
echo "=============================================================="
echo "  Building at: https://huggingface.co/spaces/${USERNAME}/${SPACE}"
echo "  Live URL:    https://${USERNAME}-${SPACE}.hf.space"
echo "  First build takes ~5-10 min (qiskit + scipy wheels)."
echo "=============================================================="
