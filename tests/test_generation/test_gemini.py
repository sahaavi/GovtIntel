"""Tests for the Gemini generation client."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from govintel.config import Settings
from govintel.generation import gemini
from govintel.generation.gemini import GeminiClient, LLMGenerationError


class FakeAsyncModels:
    """Deterministic stand-in for google-genai async models API."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def generate_content(self, *, model: str, contents: str, config: Any) -> Any:
        self.calls.append(
            {
                "model": model,
                "contents": contents,
                "system_instruction": config.system_instruction,
                "temperature": config.temperature,
                "max_output_tokens": config.max_output_tokens,
            }
        )
        return SimpleNamespace(text="generated procurement brief")


class FakeAsyncClient:
    def __init__(self) -> None:
        self.models = FakeAsyncModels()
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class FakeGenAIClient:
    """Deterministic stand-in for google.genai.Client."""

    instances: list[FakeGenAIClient] = []

    def __init__(self, *, api_key: str) -> None:
        self.api_key = api_key
        self.aio = FakeAsyncClient()
        self.closed = False
        FakeGenAIClient.instances.append(self)

    def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_generate_sends_system_instruction_temperature_and_token_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeGenAIClient.instances = []
    monkeypatch.setattr(gemini.genai, "Client", FakeGenAIClient)

    client = GeminiClient(settings=Settings(gemini_api_key="gemini-key"))
    text = await client.generate(
        system="You are a procurement analyst.",
        user_message="Generate a DHS brief.",
    )

    fake_client = FakeGenAIClient.instances[0]
    assert text == "generated procurement brief"
    assert fake_client.api_key == "gemini-key"
    assert fake_client.aio.models.calls == [
        {
            "model": "gemini-1.5-flash",
            "contents": "Generate a DHS brief.",
            "system_instruction": "You are a procurement analyst.",
            "temperature": 0.3,
            "max_output_tokens": 2048,
        }
    ]
    assert fake_client.aio.closed is True
    assert fake_client.closed is True


@pytest.mark.asyncio
async def test_generate_raises_configuration_error_when_api_key_missing() -> None:
    client = GeminiClient(settings=Settings(gemini_api_key=""))

    with pytest.raises(LLMGenerationError, match="GEMINI_API_KEY"):
        await client.generate(system="system", user_message="user")


@pytest.mark.asyncio
async def test_generate_wraps_provider_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingAsyncModels:
        async def generate_content(self, *, model: str, contents: str, config: Any) -> Any:
            raise RuntimeError("provider unavailable")

    class FailingAsyncClient:
        def __init__(self) -> None:
            self.models = FailingAsyncModels()
            self.closed = False

        async def aclose(self) -> None:
            self.closed = True

    class FailingGenAIClient:
        instances: list[FailingGenAIClient] = []

        def __init__(self, *, api_key: str) -> None:
            self.aio = FailingAsyncClient()
            self.closed = False
            FailingGenAIClient.instances.append(self)

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(gemini.genai, "Client", FailingGenAIClient)

    client = GeminiClient(settings=Settings(gemini_api_key="gemini-key"))

    with pytest.raises(LLMGenerationError, match="Gemini generation failed"):
        await client.generate(system="system", user_message="user")

    fake_client = FailingGenAIClient.instances[0]
    assert fake_client.aio.closed is True
    assert fake_client.closed is True
