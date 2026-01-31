from functools import partial

from app.infra.llm.client import LLMClient
from app.infra.llm.models import HopModel, hop_response_format
from app.infra.llm.prompts import CLASSIFIED_PROMPT, ANSWER_SYSTEM_PROMPT, ANSWER_USER_PROMPT
from app.infra.llm.tracing import tracer
from app.infra.logging import get_logger
from app.logic.rag.multi_hop import MultiHopContextSubPipeline
from app.logic.rag.single_hop import SingleHopContextSubPipeline
from app.logic.retrieval import RetrievalContextSubPipeline
from app.logic.use_cases.reranking import CrossEncoderRerankingUseCase

logger = get_logger(__name__)

class RAGPipeline:
    def __init__(
        self,
        multi_hop: MultiHopContextSubPipeline,
        single_hop: SingleHopContextSubPipeline,
        llm_client: LLMClient,
        reranking_use_case: CrossEncoderRerankingUseCase,
    ):
        self.multi_hop = multi_hop
        self.single_hop = single_hop
        self.llm_client = llm_client
        self.reranking_use_case = reranking_use_case

    @tracer.trace(name="RAG Pipeline", run_type="chain")
    async def execute(self, query, collection_name):
        client_prompt = """
            Запрос: {query}
        """
        hop: HopModel = await self.llm_client.completions_create(
            system_prompt=CLASSIFIED_PROMPT,
            user_query=client_prompt.format(query=query),
            response_format=hop_response_format,
            response_class=HopModel,
        )
        logger.info('hop, %s', hop)
        hop_map = {
            'multi-hop': partial(self.multi_hop.execute, hop.queries),
            'single-hop': partial(self.single_hop.execute, hop.queries[0]),
        }
        context = await hop_map[hop.reason](collection_name=collection_name)
        context = await self.reranking_use_case.handle(
            query=query, docs=context, limit=3
        )
        context = "\n-------\n".join([text for text in context])
        logger.info(f"Final context: {context}")
        answer = await self.llm_client.completions_create(
            system_prompt=ANSWER_SYSTEM_PROMPT,
            user_query=ANSWER_USER_PROMPT.format(
                context=context,
                query=query
            )
        )

        return {"answer": answer, "context": context}


