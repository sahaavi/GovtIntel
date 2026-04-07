"""Prompt template loader with YAML versioning and Jinja2 rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from jinja2 import Template

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "prompts"


class PromptTemplate:
    """A loaded prompt template with metadata and renderable content."""

    def __init__(self, name: str, version: str, system: str, user: str) -> None:
        self.name = name
        self.version = version
        self._system_template = Template(system)
        self._user_template = Template(user)

    def render(self, **variables: Any) -> dict[str, str]:
        """Render the prompt templates with the given variables.

        Returns:
            Dict with 'system' and 'user' keys containing rendered text.
        """
        return {
            "system": self._system_template.render(**variables),
            "user": self._user_template.render(**variables),
        }


def load_prompt(name: str, version: str = "v1") -> PromptTemplate:
    """Load a prompt template from YAML.

    Args:
        name: Template name (e.g., 'zero_shot', 'few_shot', 'chain_of_thought').
        version: Prompt version directory (default 'v1').

    Returns:
        A PromptTemplate ready to render with variables.

    Raises:
        FileNotFoundError: If the template file does not exist.
        KeyError: If required YAML fields are missing.
    """
    path = PROMPTS_DIR / version / f"{name}.yaml"

    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    return PromptTemplate(
        name=data["name"],
        version=version,
        system=data["system"],
        user=data["user"],
    )


def list_prompts(version: str = "v1") -> list[str]:
    """List available prompt template names for a version."""
    version_dir = PROMPTS_DIR / version
    if not version_dir.exists():
        return []
    return [p.stem for p in version_dir.glob("*.yaml")]
