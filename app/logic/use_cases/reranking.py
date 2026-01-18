import asyncio

from sentence_transformers import CrossEncoder

from app.infra.logging import get_logger
from app.logic.use_cases.base import UseCaseInterface

logger = get_logger(__name__)


class CrossEncoderRerankingUseCase(UseCaseInterface):
    def __init__(self, cross_encoder_model: CrossEncoder):
        self.cross_encoder_model = cross_encoder_model

    async def handle(self, query: str, docs: list[str], limit: int = 3) -> list[str]:
        input = [[query, d] for d in docs]
        scores = await asyncio.to_thread(self.cross_encoder_model.predict, sentences=input)
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:limit]]
