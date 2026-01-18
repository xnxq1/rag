import asyncio
import dataclasses
import re
import unicodedata
from typing import BinaryIO

import fitz
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_community.document_transformers import Html2TextTransformer
from yarl import URL

from app.infra.logging import get_logger
from app.logic.use_cases.base import UseCaseInterface

logger = get_logger(__name__)


@dataclasses.dataclass
class PageMetaData:
    text: str
    metadata: dict


class PdfReaderUseCase(UseCaseInterface):
    @staticmethod
    def extract_pdf_text(document: BinaryIO) -> list[PageMetaData]:
        data = document.read()
        doc = fitz.open(stream=data, filetype="pdf")
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text()
            pages.append(PageMetaData(text=text, metadata={"position": {"page_number": i}}))
        return pages

    async def handle(self, document: BinaryIO):
        pages = await asyncio.to_thread(self.extract_pdf_text, document)
        for page_metadata in pages:
            # TODO: вынести в cleaner класс??
            # normalize unicode
            text = unicodedata.normalize("NFKC", page_metadata.text)

            # remove windows control symbols
            text = text.replace("\x0c", " ")

            # join broken lines (single \n)
            text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

            # squeeze multi spaces
            text = re.sub(r"\s{2,}", " ", text)

            # keep paragraphs (double \n)
            text = re.sub(r"\n{3,}", "\n\n", text)

            # remove page numbers (very naive)
            text = re.sub(r"\n\d+\n", "\n", text)
            page_metadata.text = text

        return pages


class LoadUrlContentUseCase(UseCaseInterface):
    async def handle(self, url: URL) -> list[PageMetaData]:
        recursive_loader = RecursiveUrlLoader(
            url=str(url),
            max_depth=2,
            prevent_outside=True,
        )
        data = await asyncio.to_thread(recursive_loader.load)
        transformer = Html2TextTransformer()
        docs = await asyncio.to_thread(transformer.transform_documents, data)
        filtered = []
        for d in docs:
            url = d.metadata["source"]
            if not url.endswith((".css", ".js", ".png", ".svg", ".jpg", ".woff")):
                filtered.append(
                    PageMetaData(
                        text=d.page_content,
                        metadata={
                            "position": {"url": d.metadata["source"]},
                            "title": d.metadata["title"],
                        },
                    )
                )
        return filtered
