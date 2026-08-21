from typing import List
from api.schemas import Chunk, GuardResult


class RelevanceGuard:
    def __init__(self, config: dict = None):
        config = config or {}
        self.thresholds = config.get("per_language_thresholds", {
            "en": 0.867,
            "hi": 0.871,
            "gu": 0.864,
        })
        self.default_threshold = 0.85
        self.low_confidence_band = config.get("low_confidence_band", 0.03)

    def check(self, chunks: List[Chunk], lang: str = "en") -> GuardResult:
        if not chunks:
            return GuardResult(
                passed=False,
                reason_code="NO_RETRIEVED_CHUNKS",
                message="No chunks were retrieved",
            )

        top_score = chunks[0].score if chunks else 0.0
        threshold = self.thresholds.get(lang, self.default_threshold)

        if top_score < threshold:
            return GuardResult(
                passed=False,
                reason_code="OUT_OF_CORPUS",
                message=f"Top retrieval score {top_score:.3f} below relevance floor {threshold:.3f}",
                score=top_score,
            )

        if top_score < threshold + self.low_confidence_band:
            return GuardResult(
                passed=True,
                reason_code="LOW_CONFIDENCE",
                message=f"Retrieved but low confidence: {top_score:.3f}",
                score=top_score,
            )

        return GuardResult(passed=True, score=top_score)
