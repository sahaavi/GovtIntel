"""Gemini Flash generation client."""

from __future__ import annotations

import logging
from typing import Any

from google import genai
from google.genai import types

from govintel.config import Settings, get_settings
from govintel.generation.errors import LLMGenerationError

DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_OUTPUT_TOKENS = 2048
logger = logging.getLogger(__name__)


class GeminiClient:
    """Async wrapper around the Gemini Flash SDK."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        model_name: str = DEFAULT_GEMINI_MODEL,
    ) -> None:
        self._settings = settings or get_settings()
        self._model_name = model_name

    async def generate(self, system: str, user_message: str) -> str:
        """Generate a grounded intelligence brief from rendered prompt text."""

        if not self._settings.gemini_api_key:
            raise LLMGenerationError("GEMINI_API_KEY is required for Gemini generation")

        client: Any | None = None
        try:
            client = genai.Client(api_key=self._settings.gemini_api_key)
            response = await client.aio.models.generate_content(
                model=self._model_name,
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=DEFAULT_TEMPERATURE,
                    max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
                ),
            )
        except Exception as exc:  # pragma: no cover - provider-specific exception tree
            logger.warning("Gemini generation failed", exc_info=exc)
            raise LLMGenerationError("Gemini generation failed") from exc
        finally:
            if client is not None:
                await client.aio.aclose()
                client.close()

        text = str(getattr(response, "text", "") or "").strip()
        if not text:
            raise LLMGenerationError("Gemini generation returned an empty response")
        return text


__all__ = [
    "DEFAULT_GEMINI_MODEL",
    "DEFAULT_MAX_OUTPUT_TOKENS",
    "DEFAULT_TEMPERATURE",
    "GeminiClient",
    "LLMGenerationError",
]
