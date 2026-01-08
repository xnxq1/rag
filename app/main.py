from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.documents import router as document_router
from app.infra.logging import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application started")
    yield
    logger.info("Application stopped")


def create_app() -> FastAPI:
    app = FastAPI(
        title="MEGA RAG",
        description="Mini Ledger",
        version="1",
        lifespan=lifespan,
    )
    app.include_router(document_router)

    @app.get("/")
    async def root():
        return {
            "status": "ok",
        }

    return app
