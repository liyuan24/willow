"""Helpers for preserving conversation-history semantics."""

from __future__ import annotations

from collections.abc import Sequence

from willow.providers.base import TextBlock


def queued_user_text_blocks(texts: Sequence[str]) -> list[TextBlock]:
    """Convert queued user inputs into history-safe text blocks.

    A single queued input is still the user's original text. When multiple
    inputs are batched into one Willow user message, each block gets a small
    model-visible label so providers that preserve content-block structure and
    providers that must flatten text both retain the message boundaries.
    """
    if len(texts) <= 1:
        return [TextBlock(text=text) for text in texts]

    total = len(texts)
    return [
        TextBlock(text=f"[queued user message {index} of {total}]\n{text}")
        for index, text in enumerate(texts, start=1)
    ]


def interrupted_user_text_blocks(
    interrupted: Sequence[str],
    interrupting: Sequence[str],
) -> list[TextBlock]:
    """Convert an interrupted turn plus new input into one explicit user turn."""
    total = len(interrupted) + len(interrupting)
    blocks: list[TextBlock] = []
    index = 1
    for text in interrupted:
        blocks.append(
            TextBlock(text=f"[interrupted user message {index} of {total}]\n{text}")
        )
        index += 1
    for text in interrupting:
        blocks.append(
            TextBlock(text=f"[interrupting user message {index} of {total}]\n{text}")
        )
        index += 1
    return blocks
