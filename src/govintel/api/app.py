"""FastAPI application factory."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from govintel.api.dependencies import dispose_async_engines
from govintel.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Release application-level resources on shutdown."""

    yield
    await dispose_async_engines()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="GovIntel",
        description="Federal Procurement Intelligence Engine",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app
