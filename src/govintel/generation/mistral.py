"""HuggingFace Inference client for the fine-tuned Mistral model."""

from __future__ import annotations

import logging
from typing import Any, cast

from huggingface_hub import AsyncInferenceClient

from govintel.config import Settings, get_settings
from govintel.generation.errors import LLMGenerationError
from govintel.generation.gemini import DEFAULT_MAX_OUTPUT_TOKENS, DEFAULT_TEMPERATURE

DEFAULT_MISTRAL_MODEL = "avisheksaha/govintel-mistral-7b"
logger = logging.getLogger(__name__)


class MistralClient:
    """Async wrapper around HuggingFace chat completion inference."""

    def __init__(self, *, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    async def generate(self, system: str, user_message: str) -> str:
        """Generate text with the configured fine-tuned Mistral endpoint."""

        if not self._settings.hf_api_token:
            raise LLMGenerationError("HF_API_TOKEN is required for Mistral generation")

        model_id = self._settings.hf_model_id or DEFAULT_MISTRAL_MODEL
        client: AsyncInferenceClient | None = None
        try:
            client = AsyncInferenceClient(model=model_id, token=self._settings.hf_api_token)
            response = await client.chat_completion(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
                temperature=DEFAULT_TEMPERATURE,
            )
        except Exception as exc:  # pragma: no cover - provider-specific exception tree
            logger.warning("Mistral generation failed", exc_info=exc)
            raise LLMGenerationError("Mistral generation failed") from exc
        finally:
            if client is not None:
                await cast(Any, client).close()

        text = _extract_chat_content(response).strip()
        if not text:
            raise LLMGenerationError("Mistral generation returned an empty response")
        return text


def _extract_chat_content(response: Any) -> str:
    """Read OpenAI-compatible chat completion content from HF SDK responses."""

    choices = getattr(response, "choices", None)
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    return str(getattr(message, "content", "") or "")


__all__ = ["DEFAULT_MISTRAL_MODEL", "MistralClient"]
