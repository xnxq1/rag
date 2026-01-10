from fastapi import APIRouter

from app.logic.factories import collection_service_factory

router = APIRouter(prefix="/collections", tags=["collections"])


@router.get("/")
async def get_collections() -> dict:
    service = collection_service_factory()
    return {"result": await service.get_all()}


@router.post("/")
async def create_collections(size: int, name: str) -> dict:
    service = collection_service_factory()
    await service.create(collection_name=name, size=size)
    return {"success": True}
