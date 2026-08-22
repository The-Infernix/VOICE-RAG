import asyncio
import httpx
import logging
import os
from typing import Optional, Tuple

logger = logging.getLogger("voice-rag.stt")

_SUPPORTED_LANGS = {"en", "hi", "gu", "te"}


def _normalize_lang(code: Optional[str], fallback: str = "en") -> str:
    if not code:
        return fallback
    base = code.split("-")[0].lower()
    return base if base in _SUPPORTED_LANGS else fallback


class SarvamSTT:
    def __init__(self, api_key: str = None, model: str = "saaras:v3"):
        self.api_key = api_key or os.environ.get("SARVAM_API_KEY", "")
        self.base_url = "https://api.sarvam.ai"
        self.model = model
        self.last_error: str = ""

    async def transcribe(
        self,
        audio_bytes: bytes,
        language_code: Optional[str] = None,
        with_timestamps: bool = False,
    ) -> Tuple[str, str, float]:
        import time
        start = time.perf_counter()

        self.last_error = ""
        if not self.api_key:
            self.last_error = "missing SARVAM_API_KEY"
            logger.warning("Sarvam STT skipped: no API key configured")
            return ("", "en", 0.0)

        headers = {
            "API-Subscription-Key": self.api_key,
        }

        files = {
            "file": ("audio.wav", audio_bytes, "audio/wav"),
        }
        data = {
            "model": self.model,
            "with_timestamps": str(with_timestamps).lower(),
        }
        if language_code:
            data["language_configuration"] = language_code

        attempts = 2
        for attempt in range(1, attempts + 1):
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(
                        f"{self.base_url}/speech-to-text",
                        headers=headers,
                        files=files,
                        data=data,
                    )
                    if (
                        response.status_code in (429, 500, 502, 503, 504)
                        and attempt < attempts
                    ):
                        await asyncio.sleep(0.6 * attempt)
                        continue
                    response.raise_for_status()
                    result = response.json()

                    transcript = result.get("transcript", "")
                    lang = _normalize_lang(result.get("language_code"), language_code or "en")
                    latency = (time.perf_counter() - start) * 1000

                    if not transcript:
                        self.last_error = "STT service returned an empty transcript"
                    return (transcript, lang, latency)
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"[:200]
                logger.warning(
                    "Sarvam STT attempt %d/%d failed: %s", attempt, attempts, e
                )
                if attempt < attempts:
                    await asyncio.sleep(0.6 * attempt)
                else:
                    latency = (time.perf_counter() - start) * 1000
                    return ("", "en", latency)

        latency = (time.perf_counter() - start) * 1000
        return ("", "en", latency)

    def transcribe_sync(
        self,
        audio_bytes: bytes,
        language_code: Optional[str] = None,
    ) -> Tuple[str, str, float]:
        import time
        start = time.perf_counter()

        if not self.api_key:
            return ("", "en", 0.0)

        headers = {
            "API-Subscription-Key": self.api_key,
        }

        files = {
            "file": ("audio.wav", audio_bytes, "audio/wav"),
        }
        data = {
            "model": self.model,
        }
        if language_code:
            data["language_configuration"] = language_code

        try:
            response = httpx.post(
                f"{self.base_url}/speech-to-text",
                headers=headers,
                files=files,
                data=data,
                timeout=10.0,
            )
            response.raise_for_status()
            result = response.json()

            transcript = result.get("transcript", "")
            lang = _normalize_lang(result.get("language_code"), language_code or "en")
            latency = (time.perf_counter() - start) * 1000

            return (transcript, lang, latency)
        except Exception:
            latency = (time.perf_counter() - start) * 1000
            return ("", "en", latency)
