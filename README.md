# Remit-Q

Remit-Q is an evidence-backed research prototype comparing exact classical
optimization, standard QAOA, and constraint-preserving XY-QAOA for direct
cross-border remittance service selection.

## Run the showcase

Use two terminals from the repository root:

```bash
.venv/bin/uvicorn backend.app.main:app --reload
```

```bash
cd frontend
npm run dev
```

Open `http://127.0.0.1:5173`. The interface reads the fixed validated showcase
from `GET /api/showcase`; opening the page does not rerun QAOA.

## Verify

```bash
.venv/bin/pytest -q
cd frontend && npm run lint && npm run build
```

See [docs/showcase_guide.md](docs/showcase_guide.md) for the presentation flow
and [docs/constraint_preserving_qaoa.md](docs/constraint_preserving_qaoa.md) for
the experiment methodology and limitations.
