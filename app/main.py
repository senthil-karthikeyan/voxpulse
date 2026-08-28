"""Main FastAPI application entry point with lifespan management."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router as api_router
from app.api.websocket import router as ws_router
from app.core.config import settings
from app.core.logging import logger
from app.services.inference_service import inference_service


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for startup model initialization and graceful shutdown."""
    logger.info(
        f"Starting {settings.APP_NAME} v{settings.APP_VERSION}...",
        extra={"event": "application_started"},
    )
    # Warm up and load models into memory once at startup
    inference_service.initialize()

    yield

    logger.info(
        f"Shutting down {settings.APP_NAME}...",
        extra={"event": "application_stopped"},
    )


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Production-grade Voice Attribute Inference Service (Gender, Age Bracket, Audio Quality).",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Exception Handler to guarantee safe JSON error responses
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(f"Unhandled exception at {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error. Please try again later."},
    )


# Include API and WebSocket routers
app.include_router(api_router, tags=["Voice Analysis"])
app.include_router(ws_router, tags=["Streaming Analysis"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
