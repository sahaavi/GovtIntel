"""FastAPI application factory."""

from fastapi import FastAPI

from govintel.api.routes import router


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="GovIntel",
        description="Federal Procurement Intelligence Engine",
        version="0.1.0",
    )
    app.include_router(router)
    return app
