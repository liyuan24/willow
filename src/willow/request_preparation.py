"""Shared provider-request preparation.

This module owns the runtime projection Willow sends to providers. Durable
session history can stay complete while request preparation estimates size,
compacts when needed, and resets stateful providers before sending a compacted
projection.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from willow.compaction import (
    RuntimeCompaction,
    estimate_request_context_tokens,
    maybe_compact_messages,
)
from willow.provider_errors import is_context_length_error
from willow.providers import CompletionRequest, CompletionResponse, Message, Provider, StreamEvent
from willow.session import message_to_dict

MODEL_CONTEXT_TOKENS: dict[str, int] = {
    "gpt-5.5": 1_050_000,
    "gpt-5.4": 1_050_000,
    "gpt-5.4-mini": 400_000,
    "gpt-5.3-codex": 400_000,
    "gpt-5.3-codex-spark": 400_000,
    "gpt-5.2": 400_000,
    "claude-sonnet-4-6": 200_000,
    "claude-opus-4-6": 200_000,
    "claude-haiku-4-6": 200_000,
}


@dataclass(frozen=True, slots=True)
class PreparedRequest:
    request: CompletionRequest
    context_tokens: int
    request_bytes: int
    compacted: bool


class RequestPreparer:
    """Build provider requests from complete session history."""

    def __init__(
        self,
        *,
        provider: Provider,
        model: str,
        system: str | None,
        tools: list[dict[str, object]],
        max_tokens: int,
        context_window: int | None = None,
        state: RuntimeCompaction | None = None,
        on_compaction_start: Callable[[], None] | None = None,
        on_compaction_end: Callable[[], None] | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.system = system
        self.tools = tools
        self.max_tokens = max_tokens
        self.context_window = (
            context_window_for_model(model) if context_window is None else context_window
        )
        self.state = state
        self.on_compaction_start = on_compaction_start
        self.on_compaction_end = on_compaction_end

    def prepare(
        self,
        messages: list[Message],
        *,
        force_compaction: bool = False,
    ) -> PreparedRequest:
        request_messages, state = maybe_compact_messages(
            provider=self.provider,
            model=self.model,
            system=self.system,
            messages=messages,
            tools=self.tools,
            max_tokens=self.max_tokens,
            context_window=self.context_window,
            state=self.state,
            on_start=self.on_compaction_start,
            on_end=self.on_compaction_end,
            force=force_compaction,
        )
        self.state = state
        compacted = state is not None
        if compacted:
            self.provider.reset_conversation()
        context_tokens = estimate_request_context_tokens(
            system=self.system,
            messages=request_messages,
            tools=self.tools,
        )
        return PreparedRequest(
            request=CompletionRequest(
                model=self.model,
                messages=request_messages,
                max_tokens=self.max_tokens,
                system=self.system,
                tools=self.tools,
            ),
            context_tokens=context_tokens,
            request_bytes=estimate_serialized_request_bytes(
                model=self.model,
                messages=request_messages,
                max_tokens=self.max_tokens,
                system=self.system,
                tools=self.tools,
            ),
            compacted=compacted,
        )


def complete_with_recovery(
    preparer: RequestPreparer,
    messages: list[Message],
) -> CompletionResponse:
    prepared = preparer.prepare(messages)
    try:
        return preparer.provider.complete(prepared.request)
    except Exception as exc:
        if not is_context_length_error(exc):
            raise
    prepared = preparer.prepare(messages, force_compaction=True)
    return preparer.provider.complete(prepared.request)


def stream_with_recovery(
    preparer: RequestPreparer,
    messages: list[Message],
) -> Iterator[StreamEvent]:
    prepared = preparer.prepare(messages)
    try:
        yield from preparer.provider.stream(prepared.request)
        return
    except Exception as exc:
        if not is_context_length_error(exc):
            raise
    prepared = preparer.prepare(messages, force_compaction=True)
    yield from preparer.provider.stream(prepared.request)


def context_window_for_model(model: str) -> int | None:
    if model in MODEL_CONTEXT_TOKENS:
        return MODEL_CONTEXT_TOKENS[model]
    if model.startswith("gpt-"):
        return 400_000
    if model.startswith("claude-"):
        return 200_000
    return None


def estimate_serialized_request_bytes(
    *,
    model: str,
    messages: list[Message],
    max_tokens: int,
    system: str | None,
    tools: list[dict[str, object]],
) -> int:
    payload = {
        "model": model,
        "messages": [message_to_dict(message) for message in messages],
        "max_tokens": max_tokens,
        "system": system,
        "tools": tools,
    }
    return len(json.dumps(payload, sort_keys=True, default=str).encode("utf-8"))
