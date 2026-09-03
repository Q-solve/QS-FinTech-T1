# Deployment

QKash is a Streamlit application: one stateful Python process holding a WebSocket
open per session, backed by ~360 MB of native scientific wheels (qiskit-aer,
scipy, scikit-learn, pandas). That shape rules out edge/serverless runtimes and
requires a container or VM.

## Why not Cloudflare Workers or Pages

- Workers run JavaScript/WASM in V8 isolates. Python Workers use Pyodide, which
  cannot load qiskit-aer's compiled C++ extension.
- There is no long-lived process, so Streamlit's WebSocket has nothing to attach to.

Cloudflare's useful role here is DNS, Tunnel, and Access **in front of** a
container running somewhere else — not as the runtime.

## Known blocker on the current workstation network

`cloudflared` reaches the Cloudflare edge only on **port 7844**, over UDP (QUIC)
or TCP (HTTP/2). On the network this workstation was on, 7844 is filtered in both
directions while 443 is open:

```
region1.v2.argotunnel.com:7844 TCP BLOCKED
region2.v2.argotunnel.com:7844 TCP BLOCKED
```

Both transports were tried. A quick tunnel is created (the API call goes over 443)
and returns a `*.trycloudflare.com` hostname, but no edge connection registers and
every request answers **HTTP 530**. `--protocol http2` does not help: it changes
the transport, not the port. The same probe from the Windows side hangs, so this
is a network/ISP filter rather than a WSL limitation.

**Cloudflare Tunnel from this machine requires either a network that permits
outbound 7844, or hosting the app elsewhere.**

## Preparing the config

`.streamlit/config.toml` advertises the app at host `qkash` on port 8503. That
alias resolves only on this workstation; left in place, a remote browser fails to
open the WebSocket. Every recipe below overrides it with `STREAMLIT_*`
environment variables, and `.dockerignore` keeps the file out of the image.

## Option B — Render from the private GitHub repo (the chosen path)

Free, and the dataset stays unexposed: Render builds the `Dockerfile` from
`Q-solve/QS-FinTech-T1`, which is **private**. A container filesystem is not
browsable, so the public app URL serves the app without offering the CSV for
download.

This requires the audited extract to be tracked, so `.gitignore` carries a
narrow exception for `data/processed/remittance_east_africa_clean.csv` only.
Raw sources under `data/*.csv` and `data/raw/` remain excluded, as does `.env`.

`render.yaml` is a Blueprint: on Render, choose **New → Blueprint**, connect the
repository, and it reads the file. Set `QBRAID_API_KEY` in the dashboard
(`sync: false` keeps it out of the repo).

The container takes its listen port from `$PORT`, which Render injects;
unset locally, it falls back to 8503.

### Free tier sizing, measured

Peak resident memory with every import plus the dataset loaded is **293 MB**:

```
+numpy/pandas       104.8 MB
+sklearn            196.6 MB
+qiskit/aer         252.0 MB
+streamlit          271.4 MB
+dataset            292.7 MB
```

That fits Render's 512 MB free tier with room for Aer's working buffers. The
real constraint is CPU: the free tier gives 0.1 CPU, and the QAOA optimizer loop
is CPU-bound, so solves will be slow. Free services also sleep when idle and
cold-start on the next request.

## Option A — named tunnel (blocked here; kept for a permitting network)

Gives a stable `https://qkash.<yourdomain>` and, unlike a quick tunnel, supports
an Access login gate. Requires a domain on Cloudflare.

```bash
cloudflared tunnel login
cloudflared tunnel create qkash
cloudflared tunnel route dns qkash qkash.<yourdomain>

# Terminal 1 — the app, bound to loopback only.
STREAMLIT_SERVER_ADDRESS=127.0.0.1 \
STREAMLIT_SERVER_PORT=8503 \
STREAMLIT_SERVER_HEADLESS=true \
STREAMLIT_BROWSER_SERVER_ADDRESS=qkash.<yourdomain> \
STREAMLIT_BROWSER_SERVER_PORT=443 \
  .venv/bin/python -m streamlit run app.py

# Terminal 2 — the tunnel.
cloudflared tunnel run --url http://127.0.0.1:8503 qkash
```

Then put an Access policy in front of the hostname (Zero Trust → Access →
Applications). A quick tunnel has **no authentication**: anyone with the URL
reaches the app. Prefer a named tunnel plus Access for anything non-public.

The URL is live only while both processes run.

## Option C — Hugging Face Space (needs PRO for a private Space)

HF builds the `Dockerfile` server-side, so no local Docker daemon is needed.
**The free tier will not host this privately**: creating a private Docker Space
returns HTTP 402 — *"hosting Gradio and Docker Spaces on free cpu-basic
requires a PRO subscription"*. A **public** Space is free, but its repo is
world-readable, which would publish the CSV. Kept here as a paid alternative.

The Space is its own git repo, separate from GitHub. That is what makes it a
good fit: the audited CSV is committed **there** so the app can read it at
startup, and never to the public GitHub repo. `deploy/push_to_hf.sh` creates the
Space **private** for that reason.

```bash
# WRITE token from https://huggingface.co/settings/tokens
HF_TOKEN=hf_xxx deploy/push_to_hf.sh <your-hf-username>
```

That creates the Space, assembles the repo, pushes, and prints the URL:

```
https://<your-hf-username>-qkash.hf.space
```

First build takes roughly 5-10 minutes while the qiskit, scipy, and
scikit-learn wheels install. Set `QBRAID_API_KEY` under Space settings →
Variables and secrets; never commit it.

### Putting Cloudflare in front

Once the Space is live, a Cloudflare DNS record (proxied CNAME) can point
`qkash.<yourdomain>` at it. This gets Cloudflare's TLS and caching without
needing outbound 7844 from your workstation at all — the connection is
Cloudflare-to-HF, not Cloudflare-to-here.

### Other container hosts

The same `Dockerfile` runs on Render or Fly.io. Both build from a **GitHub**
repo, though, where the CSV is untracked — so either commit the processed
extract or push a locally built image to a registry. HF avoids that fork
entirely, which is why it is the recommended target.

Sizing: QAOA is capped at `DEFAULT_MAX_QUBITS = 20` (`qkash/quantum.py`), a
2^20-amplitude statevector. Memory is modest, but the optimizer loop is
CPU-bound — free shared-CPU tiers will run it slowly.

## Notes

- The image pins Python 3.12. The workstation venv is 3.14; 3.12 has the widest
  wheel coverage for qiskit-aer and scipy.
- `deploy/preflight.sh` checks port 7844, login state, dataset, and app health.
- `deploy/run_qkash.sh named <host>` runs Option A once 7844 is reachable.
- `.env` is untracked and excluded from the build context. It has never been
  committed.
