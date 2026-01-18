from fastapi import APIRouter
from yarl import URL

from app.logic.factories import ingest_pipeline_factory

router = APIRouter(prefix="/documents", tags=["Documents"])


# @router.post("/")
# async def load_document(collection_name: str, document: UploadFile = File(...)) -> dict:
#     pipeline = ingest_pipeline_factory()
#     await pipeline.execute(
#         ServiceFile(data=document.file, filename=document.filename),
#         collection_name,
#         content_type=document.filename.split('.')[:-1],
#     )
#     return {"success": True}


@router.post("/url_content")
async def load_url_content(collection_name: str, url: str) -> dict:
    pipeline = ingest_pipeline_factory()
    await pipeline.execute(
        collection_name=collection_name,
        url=URL(url),
        content_type="url",
    )
    return {"success": True}
