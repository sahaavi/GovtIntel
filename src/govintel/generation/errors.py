"""Generation-layer domain exceptions."""

from __future__ import annotations


class LLMGenerationError(RuntimeError):
    """Raised when an LLM provider cannot generate a usable response."""


__all__ = ["LLMGenerationError"]
