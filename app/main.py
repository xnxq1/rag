from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.collections import router as collections_router
from app.api.documents import router as document_router
from app.api.rag import router as rag_router
from app.infra.config import settings
from app.infra.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application started")
    yield
    logger.info("Application stopped")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        description=settings.app_name,
        version=settings.app_version,
        lifespan=lifespan,
    )
    app.include_router(document_router)
    app.include_router(rag_router)
    app.include_router(collections_router)

    @app.get("/")
    async def root():
        return {
            "status": "ok",
        }

    return app
