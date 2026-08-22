import json
import os
import re
import time
import httpx
from typing import List, Optional, Tuple
from api.schemas import Chunk, Answer, Citation


class GenerativeGenerator:
    def __init__(
        self,
        provider: str = "openrouter",
        base_url: str = "https://openrouter.ai/api/v1",
        api_key: str = None,
        model: str = None,
        reasoning: bool = False,
    ):
        self.provider = provider
        self.base_url = base_url
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.model = model or os.environ.get("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free")
        self.reasoning = reasoning

    def generate(
        self,
        query: str,
        chunks: List[Chunk],
        max_tokens: int = 400,
        temperature: float = 0.1,
    ) -> Optional[Answer]:
        if not self.api_key:
            return None

        if not chunks:
            return None

        context_chunks = chunks[:5]
        context = "\n\n".join([
            f"[Source {i+1}] {chunk.text}" for i, chunk in enumerate(context_chunks)
        ])

        system_prompt = (
            "You are a grounded question-answering assistant. "
            "Answer ONLY using the supplied context. "
            "If the context does not contain sufficient information, set answer to: "
            "'I don't have enough information in the provided sources to answer that.' "
            "Never fabricate information. Keep answers concise (2-3 sentences max). "
            "Respond ONLY with a single JSON object with keys: "
            '"answer" (string), "citations" (array of source numbers you actually used, e.g. [1] or [2,3]), '
            '"confidence" (number between 0 and 1). No markdown, no code fences, no extra text.'
        )

        user_prompt = (
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {query}\n\n"
            f"JSON RESPONSE:"
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if self.reasoning:
            payload["reasoning"] = {"enabled": True}

        parsed = self._call_llm(payload)
        if parsed is None:
            return None

        answer_text, cited_numbers, confidence = parsed
        if not answer_text:
            return None

        valid_citations = []
        seen = set()
        for n in cited_numbers:
            if isinstance(n, int) and 1 <= n <= len(context_chunks) and n not in seen:
                seen.add(n)
                chunk = context_chunks[n - 1]
                valid_citations.append(Citation(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text[:200],
                    score=chunk.score,
                    source=chunk.metadata.get("source", "MSMARCO-XI") if isinstance(chunk.metadata, dict) else "MSMARCO-XI",
                ))

        if not valid_citations:
            return None

        return Answer(
            text=answer_text,
            method="generative",
            citations=valid_citations,
            confidence=max(0.0, min(1.0, confidence)),
        )

    def _call_llm(self, payload: dict) -> Optional[Tuple[str, list, float]]:
        start = time.perf_counter()
        strict_payload = dict(payload)
        strict_payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "grounded_answer",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string"},
                        "citations": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 1,
                        },
                        "confidence": {"type": "number"},
                    },
                    "required": ["answer", "citations", "confidence"],
                    "additionalProperties": False,
                },
            },
        }

        for attempt, use_schema in enumerate((True, False)):
            body = strict_payload if use_schema else payload
            try:
                response = httpx.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "HTTP-Referer": "https://hhgoa-rag.local",
                        "X-Title": "HH Goa Voice RAG",
                        "Content-Type": "application/json",
                    },
                    json=body,
                    timeout=30.0,
                )
                response.raise_for_status()
                result = response.json()

                message = result.get("choices", [{}])[0].get("message", {})
                content = message.get("content", "").strip()
                content = self._strip_thinking(content)

                parsed = self._parse_answer_json(content, expect_json=True)
                if parsed is not None:
                    latency_ms = (time.perf_counter() - start) * 1000
                    self.last_latency_ms = latency_ms
                    return parsed

                plain = self._parse_answer_json(content, expect_json=False)
                if plain is not None:
                    return plain
                if not content:
                    return None
                return (content, [], 0.5)
            except httpx.TimeoutException:
                return None
            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if status == 429:
                    return None
                if use_schema and 400 <= status < 500:
                    continue
                return None
            except Exception:
                return None
        return None

    @staticmethod
    def _parse_answer_json(text: str, expect_json: bool = True) -> Optional[Tuple[str, list, float]]:
        if not text:
            return None
        candidate = text.strip()
        fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", candidate, flags=re.DOTALL)
        if fence:
            candidate = fence.group(1).strip()
        try:
            data = json.loads(candidate)
        except Exception:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                return None
            try:
                data = json.loads(match.group(0))
            except Exception:
                return None
        if not isinstance(data, dict):
            return None
        answer_text = str(data.get("answer", "")).strip()
        citations = data.get("citations", [])
        if not isinstance(citations, list):
            citations = []
        confidence = data.get("confidence", 0.85)
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.85
        if expect_json and not answer_text:
            return None
        return (answer_text, citations, confidence)

    @staticmethod
    def _strip_thinking(text: str) -> str:
        text = re.sub(
            r"^(?:Here'?s?\s+(?:a\s+)?(?:my\s+)?(?:the\s+)?thinking(?:\s+process)?[:\s]*\n+).*",
            "", text, flags=re.IGNORECASE | re.DOTALL,
        )
        text = re.sub(
            r"(?:Let me (?:think|analyze|consider|break)[^.]*\.\s*)", "", text,
        )
        return text.strip()
