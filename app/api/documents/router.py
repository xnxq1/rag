from fastapi import APIRouter, File, UploadFile

from app.logic.factories import ingest_pipeline_factory
from app.logic.ingest import File as ServiceFile

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/")
async def load_document(document: UploadFile = File(...)) -> dict:
    pipeline = ingest_pipeline_factory()
    await pipeline.execute(ServiceFile(data=document.file, filename=document.filename))
    return {"success": True}
