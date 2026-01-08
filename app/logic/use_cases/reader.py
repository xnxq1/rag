import asyncio
import dataclasses
import re
import unicodedata
from typing import BinaryIO

import fitz
from docx import Document

from app.logic.use_cases.base import UseCaseInterface


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


class DocxReaderUseCase(UseCaseInterface):
    @staticmethod
    def extract_docx_text(document: BinaryIO) -> list[PageMetaData]:
        doc = Document(document)
        paragraphs = []
        for i, paragraph in enumerate(doc.paragraphs):
            text = paragraph.text
            paragraphs.append(
                PageMetaData(text=text, metadata={"position": {"paragraph_number": i}})
            )
        # TODO: получаются маленькие куски, сделать объединение по смыслу
        return paragraphs

    async def handle(self, document: BinaryIO) -> list[PageMetaData]:
        return await asyncio.to_thread(self.extract_docx_text, document)
