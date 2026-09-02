# Remit-Q frontend

Minimal React, TypeScript, Vite, and Tailwind application shell for the Remit-Q research prototype.

## Local development

Start the FastAPI service from the repository root:

```bash
.venv/bin/uvicorn backend.app.main:app --reload
```

Then start Vite from this directory:

```bash
npm run dev
```

Vite proxies `/api` requests to `http://127.0.0.1:8000`. Set `VITE_API_BASE_URL` only when the API is hosted elsewhere.

## Validation

```bash
npm run lint
npm run build
```

This scaffold intentionally contains no recommendation or optimization UI.
