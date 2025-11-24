import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .routes import correlations, cta, exports, files, metrics, optimizer, series, upload

app = FastAPI(title="Portfolio API", version="0.1.0", docs_url="/docs")

cors_origins = os.getenv("API_CORS_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[cors_origins] if cors_origins != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(files.router)
app.include_router(series.router)
app.include_router(metrics.router)
app.include_router(correlations.router)
app.include_router(cta.router)
app.include_router(optimizer.router)
app.include_router(exports.router)


@app.get("/health", tags=["health"])
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})
