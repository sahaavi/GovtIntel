"""Tests for prompt loading and rendering."""

from __future__ import annotations

import pytest

from govintel.generation.prompts import list_prompts, load_prompt


class TestLoadPrompt:
    def test_load_zero_shot(self) -> None:
        prompt = load_prompt("zero_shot", version="v1")
        assert prompt.name == "zero_shot"
        assert prompt.version == "v1"

    def test_load_few_shot(self) -> None:
        prompt = load_prompt("few_shot", version="v1")
        assert prompt.name == "few_shot"

    def test_load_chain_of_thought(self) -> None:
        prompt = load_prompt("chain_of_thought", version="v1")
        assert prompt.name == "chain_of_thought"

    def test_load_nonexistent_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_prompt("nonexistent_prompt", version="v1")

    def test_render_zero_shot(self) -> None:
        prompt = load_prompt("zero_shot", version="v1")
        rendered = prompt.render(
            question="Who are the top DHS contractors?",
            agency_filter="DHS",
            date_range_years=3,
            context="Contract 1: Booz Allen, $5M",
            analytics="Top contractor: Booz Allen ($50M total)",
        )
        assert "Who are the top DHS contractors?" in rendered["user"]
        assert "DHS" in rendered["user"]
        assert "intelligence" in rendered["system"].lower()

    def test_render_without_agency_filter(self) -> None:
        prompt = load_prompt("zero_shot", version="v1")
        rendered = prompt.render(
            question="General spending trends",
            agency_filter=None,
            date_range_years=5,
            context="Sample context",
            analytics="Sample analytics",
        )
        assert "General spending trends" in rendered["user"]


class TestListPrompts:
    def test_list_v1_prompts(self) -> None:
        prompts = list_prompts("v1")
        assert "zero_shot" in prompts
        assert "few_shot" in prompts
        assert "chain_of_thought" in prompts

    def test_list_nonexistent_version(self) -> None:
        prompts = list_prompts("v999")
        assert prompts == []
