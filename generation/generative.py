import os
import time
import httpx
from typing import List, Optional
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
        max_tokens: int = 256,
        temperature: float = 0.1,
    ) -> Optional[Answer]:
        if not self.api_key:
            return None

        if not chunks:
            return None

        context = "\n\n".join([
            f"[Source {i+1}] {chunk.text}" for i, chunk in enumerate(chunks[:5])
        ])

        system_prompt = (
            "You are a grounded question-answering assistant. "
            "Answer ONLY using the supplied context. "
            "Provide your answer directly — do not show your thinking process, chain of thought, or step-by-step reasoning. "
            "If the context does not contain sufficient information, say: 'I don't have enough information in the provided sources to answer that.' "
            "Never fabricate information. Keep answers concise (2-3 sentences max)."
        )

        user_prompt = (
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {query}\n\n"
            f"ANSWER:"
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

        start = time.perf_counter()

        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": "https://hhgoa-rag.local",
                    "X-Title": "HH Goa Voice RAG",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=30.0,
            )
            response.raise_for_status()
            result = response.json()

            message = result.get("choices", [{}])[0].get("message", {})
            answer_text = message.get("content", "").strip()

            if not answer_text:
                return None

            answer_text = self._strip_thinking(answer_text)

            if not answer_text:
                return None

            latency_ms = (time.perf_counter() - start) * 1000

            citations = [
                Citation(
                    chunk_id=c.chunk_id,
                    text=c.text[:200],
                    score=c.score,
                    source=c.metadata.get("source", "MSMARCO-XI") if isinstance(c.metadata, dict) else "MSMARCO-XI",
                )
                for c in chunks[:3]
            ]

            return Answer(
                text=answer_text,
                method="generative",
                citations=citations,
                confidence=0.85,
            )

        except httpx.TimeoutException:
            return None
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                return None
            return None
        except Exception:
            return None

    @staticmethod
    def _strip_thinking(text: str) -> str:
        import re
        text = re.sub(
            r"^(?:Here'?s?\s+(?:a\s+)?(?:my\s+)?(?:the\s+)?thinking(?:\s+process)?[:\s]*\n+).*",
            "", text, flags=re.IGNORECASE | re.DOTALL,
        )
        text = re.sub(
            r"(?:Let me (?:think|analyze|consider|break)[^.]*\.\s*)", "", text,
        )
        return text.strip()
