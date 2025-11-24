# API Skeleton

This is a FastAPI scaffold for the new stack. It currently exposes a health check only. Agents should implement the full API surface described in `REBUILD_PLAN.md`.

## Local dev
```sh
pip install -r requirements.txt
uvicorn api.app.main:app --reload --host 0.0.0.0 --port 8000
```

## Notes
- Uses Redis (see `.env.example`).
- Celery/RQ worker is not wired yet; worker container in docker-compose is a placeholder.
- Parquet/storage paths come from `.env`.
