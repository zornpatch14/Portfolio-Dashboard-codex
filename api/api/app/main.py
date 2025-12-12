import os
import threading
from pathlib import Path
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .routes import correlations, cta, exports, files, metrics, optimizer, series, upload
from .utils.parquet_mirror import mirror_storage

app = FastAPI(title="Portfolio API", version="0.1.0", docs_url="/docs")

cors_origins = os.getenv("API_CORS_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[cors_origins] if cors_origins != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger(__name__)

_csv_thread: threading.Thread | None = None
_csv_stop = threading.Event()


def _csv_mirror_worker() -> None:
    storage_root = Path(os.getenv("CSV_MIRROR_STORAGE_ROOT", os.getenv("DATA_ROOT", "storage"))).resolve()
    interval = int(os.getenv("CSV_MIRROR_INTERVAL_SECONDS", "5"))
    force = os.getenv("CSV_MIRROR_FORCE", "0").lower() in {"1", "true", "yes"}

    while not _csv_stop.is_set():
        try:
            stats = mirror_storage(storage_root, force=force)
            logger.debug(
                "CSV mirror tick complete (converted=%s skipped=%s failed=%s)",
                stats.converted,
                stats.skipped,
                stats.failed,
            )
        except Exception as exc:  # pragma: no cover - defensive safeguard
            logger.warning("CSV mirror tick failed: %s", exc)
        _csv_stop.wait(interval)


@app.on_event("startup")
async def _start_csv_mirror() -> None:
    enabled = os.getenv("CSV_MIRROR_ENABLED", "1").lower() not in {"0", "false", "no"}
    if not enabled:
        return
    global _csv_thread
    if _csv_thread and _csv_thread.is_alive():
        return
    _csv_stop.clear()
    _csv_thread = threading.Thread(target=_csv_mirror_worker, name="csv-mirror", daemon=True)
    _csv_thread.start()


@app.on_event("shutdown")
async def _stop_csv_mirror() -> None:
    if not _csv_thread:
        return
    _csv_stop.set()
    _csv_thread.join(timeout=5)

@app.get("/health", tags=["health"])
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


# Register API routers (currently stubbed implementations; replace with real services).
app.include_router(upload.router)
app.include_router(files.router)
app.include_router(series.router)
app.include_router(metrics.router)
app.include_router(correlations.router)
app.include_router(cta.router)
app.include_router(optimizer.router)
app.include_router(exports.router)
