# API Skeleton

This is a FastAPI scaffold for the new stack. The `feature/api` branch wires the v1 endpoints described in `REBUILD_PLAN.md` (upload, file metadata, series, metrics, correlations, CTA, optimizers with SSE job streaming, and exports) to stub services so the OpenAPI contract is documented and explorable.

## Local dev
```sh
pip install -r requirements.txt
uvicorn api.app.main:app --reload --host 0.0.0.0 --port 8000
```

## Notes
- Uses Redis (see `.env.example`).
- Celery/RQ worker is not wired yet; worker container in docker-compose is a placeholder.
- Parquet/storage paths come from `.env`.
