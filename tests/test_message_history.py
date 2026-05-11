"""Tests for conversation-history helper semantics."""

from __future__ import annotations

from willow.message_history import interrupted_user_text_blocks, queued_user_text_blocks
from willow.providers import TextBlock


def test_single_queued_user_text_is_unchanged() -> None:
    assert queued_user_text_blocks(["hello"]) == [TextBlock(text="hello")]


def test_multiple_queued_user_texts_are_labeled() -> None:
    assert queued_user_text_blocks(["first", "second"]) == [
        TextBlock(text="[queued user message 1 of 2]\nfirst"),
        TextBlock(text="[queued user message 2 of 2]\nsecond"),
    ]


def test_interrupted_user_texts_are_labeled_with_interrupt_context() -> None:
    assert interrupted_user_text_blocks(["old"], ["new"]) == [
        TextBlock(text="[interrupted user message 1 of 2]\nold"),
        TextBlock(text="[interrupting user message 2 of 2]\nnew"),
    ]
