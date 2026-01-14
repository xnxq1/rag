import asyncio

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.logic.use_cases.base import UseCaseInterface


class ChunkingUseCase(UseCaseInterface):
    def __init__(self, splitter: RecursiveCharacterTextSplitter):
        self.splitter = splitter

    async def handle(self, text: str) -> list[str]:
        chunks = await asyncio.to_thread(self.splitter.split_text, text)
        return chunks
