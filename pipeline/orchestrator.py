import time
from typing import Optional, List
from api.schemas import (
    AskRequest, AskResponse, Answer, Chunk,
    StageTrace, LatencyBreakdown,
    DebugInfo, RetrievedPassage,
)
from retrieval.search import Retriever
from retrieval.lang_detect import detect_query_language
from guardrails.input_guard import InputGuard
from guardrails.relevance_guard import RelevanceGuard
from guardrails.grounding_guard import GroundingGuard
from generation.extractive import ExtractiveGenerator
from generation.generative import GenerativeGenerator
from stt.sarvam import SarvamSTT


class PipelineOrchestrator:
    def __init__(
        self,
        retriever: Retriever,
        input_guard: InputGuard,
        relevance_guard: RelevanceGuard,
        grounding_guard: GroundingGuard,
        extractive_gen: ExtractiveGenerator,
        generative_gen: GenerativeGenerator,
        stt: SarvamSTT,
        llm_model: str = "",
    ):
        self.retriever = retriever
        self.input_guard = input_guard
        self.relevance_guard = relevance_guard
        self.grounding_guard = grounding_guard
        self.extractive_gen = extractive_gen
        self.generative_gen = generative_gen
        self.stt = stt
        self.llm_model = llm_model
        self._start_time = time.time()

    def process_text(self, request: AskRequest) -> AskResponse:
        e2e_start = time.perf_counter()
        stages: List[StageTrace] = []
        latency = LatencyBreakdown()
        debug_context: List[RetrievedPassage] = []

        # Stage 1: Input validation
        t0 = time.perf_counter()
        input_result = self.input_guard.check(request.query, request.lang)
        latency.guard_input_ms = (time.perf_counter() - t0) * 1000
        stages.append(StageTrace(
            stage="guard_input",
            latency_ms=latency.guard_input_ms,
            status="pass" if input_result.passed else "fail",
            details={"reason_code": input_result.reason_code},
        ))

        if not input_result.passed:
            latency.total_core_ms = (time.perf_counter() - e2e_start) * 1000
            return AskResponse(
                status="refused",
                query=request.query,
                refusal_reason=input_result.message,
                latency=latency,
                stages=stages,
            )

        # Stage 2: Embed query
        t0 = time.perf_counter()
        query_vector = self.retriever.embedder.embed_query(request.query)
        latency.embed_ms = (time.perf_counter() - t0) * 1000
        stages.append(StageTrace(
            stage="embed",
            latency_ms=latency.embed_ms,
            status="success",
        ))

        # Stage 3: Retrieve
        t0 = time.perf_counter()
        query_lang = request.lang or detect_query_language(request.query)
        retrieval = self.retriever.search(
            query=request.query,
            top_k=request.top_k,
            language_filter=request.lang,
            use_cache=request.use_cache,
            language_preference=None if request.lang else query_lang,
        )
        latency.retrieve_ms = (time.perf_counter() - t0) * 1000
        stages.append(StageTrace(
            stage="retrieve",
            latency_ms=latency.retrieve_ms,
            status="success",
            details={
                "chunks_found": len(retrieval.chunks),
                "cache_hit": retrieval.cache_hit,
                "strategy": retrieval.strategy_used,
                "query_language": query_lang,
            },
        ))

        # Stage 4: Relevance guard
        t0 = time.perf_counter()
        detected_lang = query_lang
        relevance_result = self.relevance_guard.check(retrieval.chunks, detected_lang)
        latency.guard_relevance_ms = (time.perf_counter() - t0) * 1000
        stages.append(StageTrace(
            stage="guard_relevance",
            latency_ms=latency.guard_relevance_ms,
            status="pass" if relevance_result.passed else "fail",
            details={"reason_code": relevance_result.reason_code},
        ))

        if not relevance_result.passed:
            latency.total_core_ms = (time.perf_counter() - e2e_start) * 1000
            return AskResponse(
                status="refused",
                query=request.query,
                detected_language=detected_lang,
                refusal_reason=relevance_result.message,
                latency=latency,
                stages=stages,
            )

        # Stage 5: Generate answer
        t0 = time.perf_counter()
        answer = self._generate_answer(request, retrieval.chunks)
        latency.answer_ms = (time.perf_counter() - t0) * 1000

        if answer is None:
            latency.total_core_ms = (time.perf_counter() - e2e_start) * 1000
            return AskResponse(
                status="refused",
                query=request.query,
                detected_language=detected_lang,
                refusal_reason="Could not generate answer from retrieved context",
                latency=latency,
                stages=stages,
            )

        stages.append(StageTrace(
            stage="generate",
            latency_ms=latency.answer_ms,
            status="success",
            details={"method": answer.method},
        ))

        # Stage 6: Grounding guard
        t0 = time.perf_counter()
        grounding_result = self.grounding_guard.check(answer.text, retrieval.chunks)
        latency.guard_grounding_ms = (time.perf_counter() - t0) * 1000
        stages.append(StageTrace(
            stage="guard_grounding",
            latency_ms=latency.guard_grounding_ms,
            status="pass" if grounding_result.passed else "fail",
            details={"reason_code": grounding_result.reason_code},
        ))

        grounding_status = "GROUNDED" if grounding_result.passed else "UNGROUNDED"

        if not grounding_result.passed:
            extractive_answer = self.extractive_gen.generate(request.query, retrieval.chunks)
            if extractive_answer:
                answer = extractive_answer
            else:
                latency.total_core_ms = (time.perf_counter() - e2e_start) * 1000
                return AskResponse(
                    status="refused",
                    query=request.query,
                    detected_language=detected_lang,
                    refusal_reason=f"Answer not grounded in context: {grounding_result.message}",
                    latency=latency,
                    stages=stages,
                )

        latency.total_core_ms = (time.perf_counter() - e2e_start) * 1000
        latency.total_e2e_ms = latency.total_core_ms

        # Build debug info
        debug_info = None
        if request.debug:
            debug_context = [
                RetrievedPassage(
                    rank=i + 1,
                    score=round(c.score, 4),
                    text=c.text[:500],
                    chunk_id=c.chunk_id,
                    metadata=c.metadata,
                )
                for i, c in enumerate(retrieval.chunks[:10])
            ]
            debug_info = DebugInfo(
                retrieved_context=debug_context,
                grounding_status=grounding_status,
                embedding_latency_ms=round(latency.embed_ms, 2),
                retrieval_latency_ms=round(latency.retrieve_ms, 2),
                generation_latency_ms=round(latency.answer_ms, 2),
                chunking_strategy=retrieval.strategy_used,
                index_size=self.retriever.vector_store.size(),
                llm_model=self.llm_model,
            )

        return AskResponse(
            status="success",
            query=request.query,
            detected_language=detected_lang,
            answer=answer,
            latency=latency,
            stages=stages,
            debug=debug_info,
        )

    def _generate_answer(self, request: AskRequest, chunks: List[Chunk]) -> Optional[Answer]:
        if request.allow_generative:
            generative_answer = self.generative_gen.generate(request.query, chunks)
            if generative_answer:
                return generative_answer

        extractive_answer = self.extractive_gen.generate(request.query, chunks)
        return extractive_answer

    async def process_voice(
        self,
        audio_bytes: bytes,
        top_k: int = 10,
        allow_generative: bool = True,
        debug: bool = False,
    ) -> AskResponse:
        e2e_start = time.perf_counter()
        stages: List[StageTrace] = []
        latency = LatencyBreakdown()

        # Stage 1: STT
        t0 = time.perf_counter()
        transcript, detected_lang, stt_latency = await self.stt.transcribe(audio_bytes)
        latency.stt_ms = stt_latency
        stages.append(StageTrace(
            stage="stt",
            latency_ms=latency.stt_ms,
            status="success" if transcript else "fail",
            details={"transcript": transcript[:100], "language": detected_lang},
        ))

        if not transcript:
            return AskResponse(
                status="error",
                query="[voice]",
                refusal_reason="Could not transcribe audio",
                latency=latency,
                stages=stages,
            )

        request = AskRequest(
            query=transcript,
            lang=detected_lang,
            top_k=top_k,
            allow_generative=allow_generative,
            debug=debug,
        )

        text_response = self.process_text(request)
        text_response.latency.stt_ms = latency.stt_ms
        text_response.latency.total_e2e_ms = (time.perf_counter() - e2e_start) * 1000
        text_response.stages = stages + text_response.stages
        return text_response

    def uptime(self) -> float:
        return time.time() - self._start_time
