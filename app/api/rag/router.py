from fastapi import APIRouter

from app.logic.factories import rag_pipeline_factory

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.post("/query")
async def query_to_llm(query: str, collection_name: str) -> dict:
    rag_pipeline = rag_pipeline_factory()
    result = await rag_pipeline.execute(query=query, collection_name=collection_name)
    return {"result": result["answer"]}
