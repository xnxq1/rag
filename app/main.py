import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application started")
    yield
    logger.info("Application stopped")


def create_app() -> FastAPI:
    # logger.info("Initializing application", version=settings.app_version)

    app = FastAPI(
        title="MEGA RAG",
        description="Mini Ledger",
        version="1",
        lifespan=lifespan,
    )

    @app.get("/")
    async def root():
        return {
            "status": "ok",
        }

    return app
