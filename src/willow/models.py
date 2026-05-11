"""Model catalog and selection helpers for Willow."""

from __future__ import annotations

from dataclasses import dataclass

from willow.auth import get_credential


@dataclass(frozen=True)
class ModelChoice:
    """One selectable model, including the Willow provider that should run it."""

    model: str
    provider: str
    vendor: str
    description: str


MODEL_CATALOG: tuple[ModelChoice, ...] = (
    ModelChoice(
        model="gpt-5.5",
        provider="openai_codex",
        vendor="openai",
        description="Frontier model for complex coding, research, and real-world work.",
    ),
    ModelChoice(
        model="gpt-5.4",
        provider="openai_codex",
        vendor="openai",
        description="Strong model for everyday coding.",
    ),
    ModelChoice(
        model="gpt-5.4-mini",
        provider="openai_codex",
        vendor="openai",
        description="Small, fast, and cost-efficient model for simpler coding tasks.",
    ),
    ModelChoice(
        model="gpt-5.3-codex",
        provider="openai_codex",
        vendor="openai",
        description="Coding-optimized model.",
    ),
    ModelChoice(
        model="gpt-5.3-codex-spark",
        provider="openai_codex",
        vendor="openai",
        description="Ultra-fast coding model.",
    ),
    ModelChoice(
        model="gpt-5.2",
        provider="openai_codex",
        vendor="openai",
        description="Optimized for professional work and long-running agents.",
    ),
    ModelChoice(
        model="claude-sonnet-4-6",
        provider="anthropic",
        vendor="anthropic",
        description="Balanced Anthropic model for coding and agentic work.",
    ),
    ModelChoice(
        model="claude-opus-4-6",
        provider="anthropic",
        vendor="anthropic",
        description="Anthropic model for complex coding and deep reasoning.",
    ),
    ModelChoice(
        model="claude-haiku-4-6",
        provider="anthropic",
        vendor="anthropic",
        description="Fast Anthropic model for lightweight coding tasks.",
    ),
)


def has_vendor_auth(vendor: str) -> bool:
    """Return whether auth has a usable credential for ``vendor``."""
    try:
        get_credential(vendor)
    except (FileNotFoundError, KeyError, ValueError):
        return False
    return True


def available_model_choices() -> list[ModelChoice]:
    """Return catalog entries whose vendors are configured in auth."""
    available_vendors = {
        choice.vendor for choice in MODEL_CATALOG if has_vendor_auth(choice.vendor)
    }
    return [choice for choice in MODEL_CATALOG if choice.vendor in available_vendors]


def find_model_choice(selector: str, choices: list[ModelChoice]) -> ModelChoice | None:
    """Resolve a one-based numeric selector or model id against ``choices``."""
    if selector.isdecimal():
        index = int(selector) - 1
        if 0 <= index < len(choices):
            return choices[index]
        return None

    return next((choice for choice in choices if choice.model == selector), None)


def render_model_choices(
    choices: list[ModelChoice],
    *,
    current_model: str,
) -> str:
    """Render numbered model choices for slash-command output."""
    if not choices:
        return "No models available. Add openai or anthropic auth to ~/.willow/auth.json."

    width = max(
        len(choice.model) + (10 if choice.model == current_model else 0)
        for choice in choices
    )
    lines = ["Select Model", ""]
    for idx, choice in enumerate(choices, start=1):
        label = choice.model
        if choice.model == current_model:
            label += " (current)"
        lines.append(f"{idx:>2}. {label:<{width}}  {choice.description}")
    return "\n".join(lines)
