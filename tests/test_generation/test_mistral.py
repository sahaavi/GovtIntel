"""Tests for the HuggingFace/Mistral generation client."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from govintel.config import Settings
from govintel.generation import mistral
from govintel.generation.gemini import LLMGenerationError
from govintel.generation.mistral import MistralClient


class FakeAsyncInferenceClient:
    """Deterministic stand-in for huggingface_hub.AsyncInferenceClient."""

    instances: list[FakeAsyncInferenceClient] = []

    def __init__(self, *, model: str, token: str) -> None:
        self.model = model
        self.token = token
        self.calls: list[dict[str, Any]] = []
        self.closed = False
        FakeAsyncInferenceClient.instances.append(self)

    async def chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
    ) -> Any:
        self.calls.append(
            {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="generated mistral brief"),
                )
            ]
        )

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_generate_sends_system_and_user_messages_to_huggingface_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeAsyncInferenceClient.instances = []
    monkeypatch.setattr(mistral, "AsyncInferenceClient", FakeAsyncInferenceClient)

    client = MistralClient(
        settings=Settings(
            hf_api_token="hf-token",
            hf_model_id="avisheksaha/govintel-mistral-7b",
        )
    )
    text = await client.generate(
        system="You are a procurement analyst.",
        user_message="Generate a DHS brief.",
    )

    assert text == "generated mistral brief"
    fake_client = FakeAsyncInferenceClient.instances[0]
    assert fake_client.model == "avisheksaha/govintel-mistral-7b"
    assert fake_client.token == "hf-token"
    assert fake_client.calls == [
        {
            "messages": [
                {"role": "system", "content": "You are a procurement analyst."},
                {"role": "user", "content": "Generate a DHS brief."},
            ],
            "max_tokens": 2048,
            "temperature": 0.3,
        }
    ]
    assert fake_client.closed is True


@pytest.mark.asyncio
async def test_generate_uses_default_fine_tuned_model_when_model_id_is_blank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeAsyncInferenceClient.instances = []
    monkeypatch.setattr(mistral, "AsyncInferenceClient", FakeAsyncInferenceClient)

    client = MistralClient(settings=Settings(hf_api_token="hf-token", hf_model_id=""))
    await client.generate(system="system", user_message="user")

    assert FakeAsyncInferenceClient.instances[0].model == "avisheksaha/govintel-mistral-7b"


@pytest.mark.asyncio
async def test_generate_raises_configuration_error_when_token_missing() -> None:
    client = MistralClient(settings=Settings(hf_api_token=""))

    with pytest.raises(LLMGenerationError, match="HF_API_TOKEN"):
        await client.generate(system="system", user_message="user")


@pytest.mark.asyncio
async def test_generate_wraps_provider_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingAsyncInferenceClient:
        instances: list[FailingAsyncInferenceClient] = []

        def __init__(self, *, model: str, token: str) -> None:
            self.closed = False
            FailingAsyncInferenceClient.instances.append(self)

        async def chat_completion(
            self,
            messages: list[dict[str, str]],
            *,
            max_tokens: int,
            temperature: float,
        ) -> Any:
            raise RuntimeError("provider unavailable")

        async def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(mistral, "AsyncInferenceClient", FailingAsyncInferenceClient)

    client = MistralClient(settings=Settings(hf_api_token="hf-token"))

    with pytest.raises(LLMGenerationError, match="Mistral generation failed"):
        await client.generate(system="system", user_message="user")

    assert FailingAsyncInferenceClient.instances[0].closed is True
