"""The provider-agnostic agent loop.

Drives the standard tool-using conversation:

    user input
       -> assistant turn (text and/or tool_use blocks)
       -> if tool_use: dispatch all tools, append a single user message
          carrying the ToolResultBlocks, repeat
       -> else: stop

The loop never reaches into a provider SDK. It only ever speaks the types
defined in `willow.providers.base`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Literal

from .message_history import monitor_event_text_blocks
from .permissions import PermissionGate
from .providers.base import (
    CompletionResponse,
    ContentBlock,
    Message,
    Provider,
    StreamComplete,
    StreamEvent,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from .request_preparation import RequestPreparer, complete_with_recovery, stream_with_recovery
from .tools.base import Tool


def run(
    provider: Provider,
    tools_by_name: Mapping[str, Tool],
    system: str | None,
    user_input: str,
    model: str,
    max_tokens: int = 4096,
    max_iterations: int = 20,
    permission_gate: PermissionGate | None = None,
    context_window: int | None = None,
    thinking: bool = False,
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None,
) -> CompletionResponse:
    """Run the agent loop until the model stops or `max_iterations` is hit.

    Returns the final `CompletionResponse` from the provider. If the loop
    exits because of `max_iterations`, the most recent response is returned
    as-is (no exception); callers can inspect `stop_reason` to tell why.

    Multiple `ToolUseBlock`s in a single assistant turn are dispatched in
    emission order, and their results are bundled into one user message of
    `ToolResultBlock`s — this matches Anthropic's expected shape and is the
    natural interleaving for the OpenAI plugins to lift to their wire format.

    Tool exceptions never propagate. They are converted into
    `ToolResultBlock(is_error=True)` carrying the exception type and message,
    so the model can react to the failure on the next turn.
    """
    tool_specs = [t.spec() for t in tools_by_name.values()]
    runtime = _runtime_from_tools(tools_by_name)
    messages: list[Message] = [
        Message(role="user", content=[TextBlock(text=user_input)])
    ]
    preparer = RequestPreparer(
        provider=provider,
        model=model,
        system=system,
        tools=tool_specs,
        max_tokens=max_tokens,
        thinking=thinking,
        effort=effort,
        context_window=context_window,
    )

    response: CompletionResponse | None = None
    for _ in range(max_iterations):
        response = complete_with_recovery(preparer, messages)
        messages.append(Message(role="assistant", content=list(response.content)))

        if response.stop_reason != "tool_use":
            monitor_blocks = _drain_monitor_event_blocks(runtime)
            if not monitor_blocks:
                return response
            messages.append(Message(role="user", content=monitor_blocks))
            continue

        tool_results: list[ContentBlock] = []
        for block in response.content:
            if not isinstance(block, ToolUseBlock):
                continue
            tool_results.append(dispatch_tool(block, tools_by_name, permission_gate))
        monitor_blocks = _drain_monitor_event_blocks(runtime)

        # An assistant turn with stop_reason="tool_use" but no ToolUseBlocks
        # would be a provider bug; bail rather than spin.
        followup_blocks = [*tool_results, *monitor_blocks]
        if not followup_blocks:
            return response

        messages.append(Message(role="user", content=followup_blocks))

    # Hit max_iterations. `response` is non-None because the loop ran at least once.
    assert response is not None
    return response


def run_streaming(
    provider: Provider,
    tools_by_name: Mapping[str, Tool],
    system: str | None,
    user_input: str,
    model: str,
    max_tokens: int = 4096,
    max_iterations: int = 20,
    on_event: Callable[[StreamEvent], None] = lambda _e: None,
    permission_gate: PermissionGate | None = None,
    context_window: int | None = None,
    thinking: bool = False,
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None,
) -> CompletionResponse:
    """Run the agent loop using streaming for each model turn.

    Same semantics as `run()`, but each model turn is driven through
    `provider.stream()`. Every `StreamEvent` emitted by the provider is
    forwarded to `on_event` for incremental rendering. The terminal
    `StreamComplete.response` is used for tool dispatch and termination
    decisions, and is what this function ultimately returns.

    Returns the final `CompletionResponse` from the last streamed turn. If
    the loop exits because of `max_iterations`, the most recent response is
    returned as-is (no exception); callers can inspect `stop_reason` to tell
    why. `max_iterations` counts streamed turns identically to `run()`.

    `on_event` is called synchronously, in event-emission order, before the
    loop touches the assembled response. Exceptions from `on_event`
    propagate (the loop does not catch them).
    """
    tool_specs = [t.spec() for t in tools_by_name.values()]
    runtime = _runtime_from_tools(tools_by_name)
    messages: list[Message] = [
        Message(role="user", content=[TextBlock(text=user_input)])
    ]
    preparer = RequestPreparer(
        provider=provider,
        model=model,
        system=system,
        tools=tool_specs,
        max_tokens=max_tokens,
        thinking=thinking,
        effort=effort,
        context_window=context_window,
    )

    response: CompletionResponse | None = None
    for _ in range(max_iterations):
        response = None
        for event in stream_with_recovery(preparer, messages):
            on_event(event)
            if isinstance(event, StreamComplete):
                response = event.response
        if response is None:
            raise RuntimeError(
                "Provider stream ended without a StreamComplete event."
            )

        messages.append(Message(role="assistant", content=list(response.content)))

        if response.stop_reason != "tool_use":
            monitor_blocks = _drain_monitor_event_blocks(runtime)
            if not monitor_blocks:
                return response
            messages.append(Message(role="user", content=monitor_blocks))
            continue

        tool_results: list[ContentBlock] = []
        for block in response.content:
            if not isinstance(block, ToolUseBlock):
                continue
            tool_results.append(dispatch_tool(block, tools_by_name, permission_gate))
        monitor_blocks = _drain_monitor_event_blocks(runtime)

        followup_blocks = [*tool_results, *monitor_blocks]
        if not followup_blocks:
            return response

        messages.append(Message(role="user", content=followup_blocks))

    assert response is not None
    return response


def dispatch_tool(
    block: ToolUseBlock,
    tools_by_name: Mapping[str, Tool],
    permission_gate: PermissionGate | None = None,
) -> ToolResultBlock:
    """Run one tool call, packaging any failure as a ToolResultBlock."""
    tool = tools_by_name.get(block.name)
    if tool is None:
        return ToolResultBlock(
            tool_use_id=block.id,
            content=f"Unknown tool: {block.name!r}",
            is_error=True,
        )
    if permission_gate is not None:
        permission = permission_gate.check(block)
        if not permission.allowed:
            return ToolResultBlock(
                tool_use_id=block.id,
                content=permission.denial or "Tool execution denied.",
                is_error=True,
            )
    try:
        output = tool.run(**block.input)
    except Exception as exc:  # noqa: BLE001 — surface anything as a tool error
        return ToolResultBlock(
            tool_use_id=block.id,
            content=f"{type(exc).__name__}: {exc}",
            is_error=True,
        )
    return ToolResultBlock(tool_use_id=block.id, content=output)


def _runtime_from_tools(tools_by_name: Mapping[str, Tool]) -> object | None:
    for tool in tools_by_name.values():
        runtime = getattr(tool, "runtime", None)
        if runtime is not None and hasattr(runtime, "events"):
            return runtime
    return None


def _drain_monitor_event_blocks(runtime: object | None) -> list[ContentBlock]:
    if runtime is None:
        return []
    events = getattr(runtime, "events", None)
    if events is None or not hasattr(events, "drain"):
        return []
    drained = events.drain()
    if not drained:
        return []
    return list(monitor_event_text_blocks(drained))
