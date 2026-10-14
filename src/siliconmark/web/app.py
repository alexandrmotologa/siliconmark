"""FastAPI Application factory for SiliconMark Web Dashboard."""

from __future__ import annotations

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from siliconmark.web.routes import router as api_router

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="SiliconMark Web Dashboard",
    description="Standardized bare-metal hardware benchmark suite measuring LLM prefill TFLOPS, decode memory bandwidth, and attention efficiency.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

# Mount static files for dashboard frontend
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
