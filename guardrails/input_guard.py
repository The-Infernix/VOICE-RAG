import re
from api.schemas import GuardResult


class InputGuard:
    def __init__(self, config: dict = None):
        config = config or {}
        self.max_query_length = config.get("max_query_length", 500)
        self.blocked_patterns = config.get("blocked_patterns", [
            "ignore previous", "ignore all previous", "ignore above", "system prompt",
            "you are a", "pretend you", "act as",
            "forget instructions", "disregard instructions",
        ])
        self._injection_re = re.compile(
            r"ignore\s+.*?previous|ignore\s+above|system\s+prompt|"
            r"you\s+are\s+a|pretend\s+you|act\s+as|"
            r"disregard\s+.*?instructions|forget\s+.*?instructions|"
            r"override\s+.*?instructions|bypass\s+.*?instructions",
            re.IGNORECASE,
        )

    def check(self, query: str, lang: str = None) -> GuardResult:
        if not query or not query.strip():
            return GuardResult(passed=False, reason_code="EMPTY_INPUT", message="Query is empty")

        query = query.strip()

        if len(query) > self.max_query_length:
            return GuardResult(
                passed=False,
                reason_code="QUERY_TOO_LONG",
                message=f"Query exceeds {self.max_query_length} characters",
            )

        lower_query = query.lower()
        if self._injection_re.search(lower_query):
            return GuardResult(
                passed=False,
                reason_code="PROMPT_INJECTION",
                message="Query contains potential prompt injection",
            )
        for pattern in self.blocked_patterns:
            if pattern in lower_query:
                return GuardResult(
                    passed=False,
                    reason_code="PROMPT_INJECTION",
                    message="Query contains potential prompt injection",
                )

        if any(c in query for c in ['<script>', '{', '}', 'sudo', 'rm -']):
            return GuardResult(
                passed=False,
                reason_code="UNSAFE_INPUT",
                message="Query contains potentially unsafe content",
            )

        if lang and lang not in ("en", "hi", "gu", "te"):
            return GuardResult(
                passed=False,
                reason_code="UNSUPPORTED_LANGUAGE",
                message=f"Language '{lang}' is not supported",
            )

        return GuardResult(passed=True)
