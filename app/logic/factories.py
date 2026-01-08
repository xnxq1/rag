from app.logic.ingest import IngestPipeline
from app.logic.use_cases.factories import (
    docx_reader_use_case_factory,
    embedding_use_case_factory,
    pdf_reader_use_case_factory,
    recurcive_text_splitter_use_case_factory,
)


def ingest_pipeline_factory() -> IngestPipeline:
    return IngestPipeline(
        embedding_use_case=embedding_use_case_factory(),
        chunking_use_case=recurcive_text_splitter_use_case_factory(),
        pdf_reader_use_case=pdf_reader_use_case_factory(),
        docx_reader_use_case=docx_reader_use_case_factory(),
    )
