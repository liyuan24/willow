"""Tests for Willow's native terminal TUI."""

from __future__ import annotations

import argparse
import io
import os
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from willow import session, tui
from willow.models import ModelChoice
from willow.providers import (
    CompletionRequest,
    CompletionResponse,
    Message,
    Provider,
    StreamComplete,
    TextBlock,
    TextDelta,
    ThinkingDelta,
    ToolResultBlock,
    ToolUseBlock,
    ToolUseDelta,
)
from willow.tools import TOOLS_BY_NAME
from willow.tools.base import Tool


class _ScriptedStreamProvider(Provider):
    def __init__(self, scripts: list[list[Any]]) -> None:
        self._scripts: deque[list[Any]] = deque(scripts)
        self.requests: list[CompletionRequest] = []

    def complete(self, request: CompletionRequest) -> CompletionResponse:  # pragma: no cover
        raise NotImplementedError

    def stream(self, request: CompletionRequest) -> Iterator[Any]:
        self.requests.append(request)
        if not self._scripts:
            raise RuntimeError("provider exhausted")
        yield from self._scripts.popleft()


class _RecordingTool(Tool):
    name = "echo"
    description = "Return back the given message."
    input_schema = {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
    }

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def run(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return f"echoed: {kwargs.get('message', '')}"


class _TTYBuffer(io.StringIO):
    def isatty(self) -> bool:
        return True


def _make_args(**overrides: Any) -> argparse.Namespace:
    defaults = dict(
        provider="openai_responses",
        model="gpt-5.5",
        system=None,
        max_tokens=4096,
        max_iterations=20,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _drive(
    provider: Provider,
    inputs: list[str],
    args: argparse.Namespace | None = None,
) -> tuple[str, tui.WillowApp]:
    args = args or _make_args()
    inputs_iter = iter(inputs)

    def input_func(_prompt: str = "") -> str:
        try:
            return next(inputs_iter)
        except StopIteration as exc:
            raise EOFError from exc

    out = io.StringIO()
    app = tui.WillowApp(args, provider, input_func=input_func, out=out)
    assert app.run() == 0
    return out.getvalue(), app


def _session_record(
    session_id: str,
    *,
    provider: str = "openai_responses",
    model: str = "gpt-5.5",
    updated_at: str = "2026-05-09T10:00:00Z",
    text: str = "hello",
) -> session.SessionRecord:
    return session.SessionRecord(
        metadata=session.SessionMetadata(
            id=session_id,
            created_at="2026-05-09T09:00:00Z",
            updated_at=updated_at,
            title=None,
            cwd="/tmp/project",
        ),
        settings=session.SessionSettings(
            provider=provider,
            model=model,
            system="Be direct.",
            max_tokens=1234,
            max_iterations=6,
        ),
        messages=[Message(role="user", content=[TextBlock(text=text)])],
    )


def test_run_tui_builds_provider_and_runs_native_session(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = object()
    monkeypatch.setattr("willow.cli._build_provider", lambda _name: provider)

    instances: list[FakeWillowApp] = []

    class FakeWillowApp:
        def __init__(
            self,
            args: argparse.Namespace,
            built_provider: object,
            *,
            inline_mode: bool,
            state: dict[str, object] | None,
        ) -> None:
            self.args = args
            self.provider = built_provider
            self.inline_mode = inline_mode
            self.state = state
            instances.append(self)

        def run(self) -> int:
            return 42

    monkeypatch.setattr(tui, "WillowApp", FakeWillowApp)

    args = _make_args()
    assert tui.run_tui(args) == 42
    assert args.persist_session is True
    assert len(instances) == 1
    assert instances[0].provider is provider
    assert instances[0].inline_mode is True
    assert instances[0].state is None


def test_run_tui_resumes_session_id_and_restores_saved_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(session.SESSION_DIR_ENV, str(tmp_path / "sessions"))
    record = _session_record(
        "sess_123",
        provider="anthropic",
        model="claude-sonnet-4-6",
        text="continue this",
    )
    path = session.default_session_path(record.metadata.id)
    session.save_session(record, path)
    built: list[str] = []
    provider = object()

    def build_provider(name: str) -> object:
        built.append(name)
        return provider

    monkeypatch.setattr("willow.cli._build_provider", build_provider)

    instances: list[Any] = []

    class FakeWillowApp:
        def __init__(
            self,
            args: argparse.Namespace,
            built_provider: object,
            *,
            inline_mode: bool,
            state: dict[str, object] | None,
        ) -> None:
            self.args = args
            self.provider = built_provider
            self.inline_mode = inline_mode
            self.state = state
            instances.append(self)

        def run(self) -> int:
            return 0

    monkeypatch.setattr(tui, "WillowApp", FakeWillowApp)

    args = _make_args(resume="sess_123")
    assert tui.run_tui(args) == 0

    assert built == ["anthropic"]
    assert args.persist_session is True
    assert args.provider == "anthropic"
    assert args.model == "claude-sonnet-4-6"
    assert args.max_tokens == 1234
    assert args.max_iterations == 6
    assert args._resume_session_path == path
    assert instances[0].state is not None
    assert instances[0].state["messages"] == record.messages


def test_resumed_session_renders_saved_history_before_prompt() -> None:
    record = _session_record("sess_resume", text="old question")
    record = session.SessionRecord(
        metadata=record.metadata,
        settings=record.settings,
        messages=[
            *record.messages,
            Message(role="assistant", content=[TextBlock(text="old answer")]),
        ],
    )
    args = _make_args(persist_session=True)
    args.provider = record.settings.provider
    args.model = record.settings.model
    args.max_tokens = record.settings.max_tokens
    args.max_iterations = record.settings.max_iterations
    args._resume_session_record = record
    args._resume_session_path = Path("/tmp/sess_resume.json")

    inputs_iter = iter([])

    def input_func(_prompt: str = "") -> str:
        try:
            return next(inputs_iter)
        except StopIteration as exc:
            raise EOFError from exc

    out_buffer = io.StringIO()
    app = tui.WillowApp(
        args,
        _ScriptedStreamProvider([]),
        state=tui._state_from_session(record),
        input_func=input_func,
        out=out_buffer,
    )
    assert app.run() == 0
    out = out_buffer.getvalue()

    assert "[resumed] Loaded 2 saved message(s)" in out
    assert "old question" in out
    assert "old answer" in out
    assert out.index("old question") < out.index("Goodbye.")


def test_resumed_session_sends_saved_history_on_next_request() -> None:
    record = _session_record("sess_resume", text="old question")
    record = session.SessionRecord(
        metadata=record.metadata,
        settings=record.settings,
        messages=[
            *record.messages,
            Message(role="assistant", content=[TextBlock(text="old answer")]),
        ],
    )
    args = _make_args(persist_session=True)
    args.provider = record.settings.provider
    args.model = record.settings.model
    args.max_tokens = record.settings.max_tokens
    args.max_iterations = record.settings.max_iterations
    args._resume_session_record = record
    args._resume_session_path = Path("/tmp/sess_resume.json")
    response = CompletionResponse(content=[TextBlock(text="new answer")], stop_reason="end_turn")
    provider = _ScriptedStreamProvider([[StreamComplete(response=response)]])
    inputs_iter = iter(["new question", "/exit"])

    def input_func(_prompt: str = "") -> str:
        try:
            return next(inputs_iter)
        except StopIteration as exc:
            raise EOFError from exc

    app = tui.WillowApp(
        args,
        provider,
        state=tui._state_from_session(record),
        input_func=input_func,
        out=io.StringIO(),
    )
    assert app.run() == 0

    assert len(provider.requests) == 1
    request = provider.requests[0]
    assert [message.role for message in request.messages] == ["user", "assistant", "user"]
    assert request.messages == [
        *record.messages,
        Message(role="user", content=[TextBlock(text="new question")]),
    ]


def test_resumed_session_ending_with_tool_result_continues_turn(tmp_path: Path) -> None:
    record = _session_record("sess_tool_result", text="commit changes")
    record = session.SessionRecord(
        metadata=record.metadata,
        settings=record.settings,
        messages=[
            *record.messages,
            Message(
                role="assistant",
                content=[
                    ToolUseBlock(
                        id="call_1",
                        name="bash",
                        input={"command": "git commit"},
                    )
                ],
            ),
            Message(
                role="user",
                content=[
                    ToolResultBlock(
                        tool_use_id="call_1",
                        content="[stderr]\nfatal: no commits yet",
                    )
                ],
            ),
        ],
    )
    args = _make_args(persist_session=True)
    args.provider = record.settings.provider
    args.model = record.settings.model
    args.max_tokens = record.settings.max_tokens
    args.max_iterations = record.settings.max_iterations
    args._resume_session_record = record
    args._resume_session_path = tmp_path / "sess_tool_result.json"
    response = CompletionResponse(content=[TextBlock(text="continued")], stop_reason="end_turn")
    provider = _ScriptedStreamProvider(
        [[TextDelta(text="continued"), StreamComplete(response=response)]]
    )
    inputs_iter = iter(["/exit"])

    def input_func(_prompt: str = "") -> str:
        try:
            return next(inputs_iter)
        except StopIteration as exc:
            raise EOFError from exc

    out_buffer = io.StringIO()
    app = tui.WillowApp(
        args,
        provider,
        state=tui._state_from_session(record),
        input_func=input_func,
        out=out_buffer,
    )
    assert app.run() == 0

    assert len(provider.requests) == 1
    assert provider.requests[0].messages == record.messages
    out = out_buffer.getvalue()
    assert "[resumed] Continuing from saved tool result." in out
    saved = session.load_session(app._session_path)
    assert saved.messages[-1] == Message(role="assistant", content=[TextBlock(text="continued")])


def test_run_tui_resume_picker_uses_latest_session_when_not_interactive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    older = tui.SessionEntry(
        path=Path("/tmp/older.json"),
        record=_session_record("older", updated_at="2026-05-09T09:00:00Z"),
    )
    latest = tui.SessionEntry(
        path=Path("/tmp/latest.json"),
        record=_session_record(
            "latest",
            provider="anthropic",
            model="claude-sonnet-4-6",
            updated_at="2026-05-09T10:00:00Z",
        ),
    )
    monkeypatch.setattr(tui, "list_sessions", lambda: [latest, older])
    monkeypatch.setattr("willow.cli._build_provider", lambda _name: object())

    instances: list[Any] = []

    class FakeWillowApp:
        def __init__(
            self,
            args: argparse.Namespace,
            built_provider: object,
            *,
            inline_mode: bool,
            state: dict[str, object] | None,
        ) -> None:
            self.args = args
            self.state = state
            instances.append(self)

        def run(self) -> int:
            return 0

    monkeypatch.setattr(tui, "WillowApp", FakeWillowApp)

    args = _make_args(resume="")
    assert tui.run_tui(args) == 0

    assert args._resume_session_path == latest.path
    assert args.provider == "anthropic"
    assert instances[0].state is not None
    assert instances[0].state["current_model"] == "claude-sonnet-4-6"


def test_run_tui_resume_without_sessions_starts_new(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tui, "list_sessions", lambda: [])
    monkeypatch.setattr("willow.cli._build_provider", lambda _name: object())

    instances: list[Any] = []

    class FakeWillowApp:
        def __init__(
            self,
            args: argparse.Namespace,
            built_provider: object,
            *,
            inline_mode: bool,
            state: dict[str, object] | None,
        ) -> None:
            self.args = args
            self.state = state
            instances.append(self)

        def run(self) -> int:
            return 0

    monkeypatch.setattr(tui, "WillowApp", FakeWillowApp)

    args = _make_args(resume="")
    assert tui.run_tui(args) == 0

    assert not hasattr(args, "_resume_session_path")
    assert instances[0].state is None


def test_command_hints_match_slash_prefix() -> None:
    rendered = tui._render_command_hints("/stat")

    assert "/status  show session and status details" in rendered
    assert "/statusline  toggle status snapshots in the prompt" in rendered
    assert "/model" not in rendered
    assert "/clear" not in rendered


def test_skill_hints_match_skill_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    skill_path = tmp_path / "project" / ".willow" / "skills" / "reviewer" / "SKILL.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text("---\ndescription: Review code.\n---\nbody", encoding="utf-8")

    rendered = tui._render_input_hints("/rev", tmp_path / "project")

    assert "/reviewer  Review code." in rendered


def test_input_hints_show_commands_and_skills(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    skill_path = (
        tmp_path / "project" / ".willow" / "skills" / "hello_world" / "SKILL.md"
    )
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text(
        "---\ndescription: Say hello.\n---\nbody",
        encoding="utf-8",
    )

    rendered = tui._render_input_hints("/", tmp_path / "project")

    assert "/clear  reset conversation history" in rendered
    assert "/model  choose what model to use" in rendered
    assert "/hello_world  Say hello." in rendered


def test_input_hints_filter_commands_and_skills_by_same_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    skills_root = tmp_path / "project" / ".willow" / "skills"
    hello_path = skills_root / "hello_world" / "SKILL.md"
    review_path = skills_root / "reviewer" / "SKILL.md"
    hello_path.parent.mkdir(parents=True)
    review_path.parent.mkdir(parents=True)
    hello_path.write_text(
        "---\ndescription: Say hello.\n---\nbody",
        encoding="utf-8",
    )
    review_path.write_text(
        "---\ndescription: Review code.\n---\nbody",
        encoding="utf-8",
    )

    rendered = tui._render_input_hints("/he", tmp_path / "project")

    assert "/help  show commands and settings" in rendered
    assert "/hello_world  Say hello." in rendered
    assert "/reviewer" not in rendered
    assert "/model" not in rendered


def test_render_statusline_includes_context_usage_and_total_tokens() -> None:
    rendered = tui._render_statusline(
        model="gpt-5.5",
        context_tokens=10_500,
        context_window=1_050_000,
        input_tokens=40_000,
        cached_tokens=12_000,
        output_tokens=2_000,
        cwd="~/repos/willow",
    )

    assert (
        rendered
        == "gpt-5.5 | Context 1.0% used (10.5k tok / 1.1M tok) | "
        "input: 40.0k tok | cached total: 12.0k tok | output: 2.0k tok | cwd: ~/repos/willow"
    )


def test_render_statusline_shows_zero_before_provider_usage() -> None:
    rendered = tui._render_statusline(
        model="gpt-5.5",
        context_tokens=None,
        context_window=1_050_000,
        input_tokens=0,
        cached_tokens=0,
        output_tokens=0,
        cwd="~/repos/willow",
    )

    assert rendered == "gpt-5.5 | Context 0.0% used (0 tok / 1.1M tok) | cwd: ~/repos/willow"


def test_styled_statusline_colors_model_and_context_usage() -> None:
    rendered = tui._style_statusline_text(
        "gpt-5.5 | Context 0.0% used (3 tok / 1.1M tok) | "
        "input: 1 tok | cached total: 0 tok | output: 2 tok | cwd: ~/repos/willow"
    )

    assert f"{tui.STATUS_MODEL_STYLE}gpt-5.5{tui.STATUS_STYLE}" in rendered
    assert (
        f"{tui.STATUS_TOKEN_STYLE}Context 0.0% used (3 tok / 1.1M tok) | "
        f"input: 1 tok | cached total: 0 tok | output: 2 tok"
        f"{tui.STATUS_STYLE}"
    ) in rendered
    assert "cwd: ~/repos/willow" in rendered


def test_status_text_uses_last_provider_input_tokens_not_heuristic() -> None:
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=io.StringIO())
    app.system = None
    app.tool_specs = []
    app.messages = [Message(role="user", content=[TextBlock(text="x" * 42_000)])]
    app._last_context_tokens = 862
    app._total_input_tokens = 900_000
    app._total_cached_tokens = 300_000
    app._total_output_tokens = 100_000

    rendered = app._status_text()

    assert "Context 0.1% used (862 tok / 1.1M tok)" in rendered
    assert "input: 900.0k tok" in rendered
    assert "cached total: 300.0k tok" in rendered
    assert "output: 100.0k tok" in rendered


def test_terminal_appends_without_rewriting_scrollback() -> None:
    end_response = CompletionResponse(
        content=[TextBlock(text="hi there")],
        stop_reason="end_turn",
        usage={"input_tokens": 1, "cached_tokens": 1, "output_tokens": 2},
    )
    provider = _ScriptedStreamProvider(
        [[TextDelta(text="hi "), TextDelta(text="there"), StreamComplete(response=end_response)]]
    )

    out, _app = _drive(provider, ["hello", "/exit"])

    assert "Willow Agent" in out
    assert "~~ \\|/ ~~" in out
    assert "model:     gpt-5.5" in out
    assert "directory:" in out
    assert "Session:" not in out
    assert "hi there" in out
    assert "[status] gpt-5.5 | Context " in out
    assert "input: 1 tok | cached total: 1 tok | output: 2 tok" in out
    assert " | cwd: " in out
    assert "Goodbye." in out
    forbidden = ["\x1b[2J", "\x1b[H", "\x1b[1A", "\x1b[K", "\x1b[?1049h"]
    assert all(sequence not in out for sequence in forbidden)


def test_assistant_text_uses_terminal_background() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False

    app._write_block("hello", tui.USER_STYLE)
    app._write_block("hi", tui.ASSISTANT_STYLE)
    app._write_panel("tool result", "ok", tui.TOOL_STYLE)

    rendered = out.getvalue()
    assert tui.USER_STYLE in rendered
    assert tui.ASSISTANT_STYLE in rendered
    assert tui.TOOL_STYLE in rendered
    assert "48;5" not in tui.ASSISTANT_STYLE
    assert "48;5" not in tui.THINKING_STYLE
    assert "48;5" in tui.PROMPT_STYLE
    assert "you" not in rendered
    assert "assistant" not in rendered
    assert rendered.count(tui.RESET) >= 3


def test_styled_transcript_wraps_long_lines_without_dropping_text() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    app._terminal_width = lambda: 12  # type: ignore[method-assign]

    app._write_block("abcdefghijklmnopqrstuvwxyz", tui.ASSISTANT_STYLE)
    app._write_panel("status", "0123456789abcdefghijklmnopqrstuvwxyz", tui.STATUS_STYLE)

    rendered = out.getvalue()
    assert "mnopqrstuvw" in rendered
    assert "xyz" in rendered
    assert "z           " in rendered


def test_live_prompt_box_shows_working_when_streaming_without_input() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.streaming = True

    live._draw_prompt()

    rendered = out.getvalue()
    assert tui.STATUS_STYLE in rendered
    assert tui.PROMPT_STYLE in rendered
    assert "working..." in rendered
    assert " > " in rendered
    assert "\r\x1b[3C\x1b[?25h" in rendered
    assert rendered.index("working...") < rendered.index(tui.PROMPT_STYLE)
    assert "streaming" not in rendered
    assert "ready" not in rendered


def test_live_prompt_box_switches_to_type_box_when_streaming_with_input() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.streaming = True
    live.buffer = "queued text"
    live.cursor = len(live.buffer)

    live._draw_prompt()

    rendered = out.getvalue()
    assert " > queued text" in rendered
    assert "\r\x1b[14C\x1b[?25h" in rendered
    assert "working..." in rendered
    assert rendered.index("working...") < rendered.index(" > queued text")


def test_live_prompt_cursor_overlays_character_without_shifting_text() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.buffer = "abcdef"
    live.cursor = 2

    live._draw_prompt()

    rendered = out.getvalue()
    assert " > abcdef" in rendered
    assert tui.PROMPT_MARKER_STYLE not in rendered
    assert "\r\x1b[5C\x1b[?25h" in rendered
    assert " > ab▌cdef" not in rendered


def test_live_prompt_wraps_long_input_across_lines() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    app._statusline_enabled = False
    app._terminal_width = lambda: 10  # type: ignore[method-assign]
    live = tui._LiveTerminal(app)
    live.buffer = "abcdefghijkl"
    live.cursor = len(live.buffer)

    live._draw_prompt()

    rendered = out.getvalue()
    assert " > abcdefg" in rendered
    assert "\n" in rendered
    assert "   hijkl" in rendered
    assert " > hijkl" not in rendered
    assert live.prompt_lines == 2
    assert "\x1b[8C\x1b[?25h" in rendered


def test_live_prompt_shows_slash_command_hints_for_prefix() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.buffer = "/mo"
    live.cursor = len(live.buffer)

    live._draw_prompt()

    rendered = out.getvalue()
    assert "/model  choose what model to use" in rendered
    assert "/clear" not in rendered
    assert rendered.index(" > /mo") < rendered.index("/model")
    assert "tokens:" not in rendered


def test_live_input_hints_move_highlight_with_up_and_down(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    skill_path = project / ".willow" / "skills" / "hello_world" / "SKILL.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text(
        "---\ndescription: Say hello.\n---\nbody",
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.chdir(project)
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.buffer = "/he"
    live.cursor = len(live.buffer)

    live._draw_prompt()

    rendered = out.getvalue()
    assert f"{tui.SELECTED_ROW_STYLE} /help" in rendered

    out.seek(0)
    out.truncate(0)
    live._move_picker_selection(1)
    live._draw_prompt()

    rendered = out.getvalue()
    assert f"{tui.SELECTED_ROW_STYLE} /hello_world" in rendered


def test_live_input_hint_enter_executes_selected_skill(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    skill_path = project / ".willow" / "skills" / "hello_world" / "SKILL.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text("Say hello before doing the task.", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.chdir(project)
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.buffer = "/he write docs"
    live.cursor = len(live.buffer)
    started: list[bool] = []

    def fake_start_worker() -> None:
        started.append(True)

    live._start_worker = fake_start_worker  # type: ignore[method-assign]

    live._move_picker_selection(1)
    live._submit_buffer()

    assert started == [True]
    assert len(app.messages) == 1
    block = app.messages[0].content[0]
    assert isinstance(block, TextBlock)
    assert "Use the Willow skill 'hello_world'" in block.text
    assert "User task:\nwrite docs" in block.text


def test_live_tab_completes_selected_skill_without_submitting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    skill_path = project / ".willow" / "skills" / "hello_world" / "SKILL.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text("Say hello before doing the task.", encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    monkeypatch.chdir(project)
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.buffer = "/he write docs"
    live.cursor = len(live.buffer)

    live._move_picker_selection(1)
    completed = live._complete_selected_hint()

    assert completed is True
    assert live.buffer == "/hello_world write docs"
    assert live.cursor == len(live.buffer)
    assert app.messages == []


def test_live_tab_completes_selected_command_and_adds_space() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.buffer = "/he"
    live.cursor = len(live.buffer)

    completed = live._complete_selected_hint()

    assert completed is True
    assert live.buffer == "/help "
    assert live.cursor == len(live.buffer)
    assert app.messages == []


def test_live_input_hint_enter_executes_selected_command() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.buffer = "/he"
    live.cursor = len(live.buffer)

    live._submit_buffer()

    rendered = out.getvalue()
    assert "Commands:" in rendered
    assert "/model [name|#]" in rendered
    assert app.messages == []


def test_live_prompt_shows_model_picker_for_model_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    choices = [
        ModelChoice(
            model="gpt-5.5",
            provider="openai_responses",
            vendor="openai",
            description="Frontier model.",
        ),
        ModelChoice(
            model="claude-sonnet-4-6",
            provider="anthropic",
            vendor="anthropic",
            description="Balanced model.",
        ),
    ]
    monkeypatch.setattr(tui, "available_model_choices", lambda: choices)
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.buffer = "/model"
    live.cursor = len(live.buffer)

    live._draw_prompt()

    rendered = out.getvalue()
    assert " > /model" in rendered
    assert "gpt-5.5" in rendered
    assert "claude-sonnet-4-6" in rendered
    assert tui.SELECTED_ROW_STYLE in rendered
    assert rendered.index(" > /model") < rendered.index("gpt-5.5")
    assert "/model  choose what model to use" not in rendered
    assert "tokens:" not in rendered


def test_live_model_picker_moves_highlight_with_up_and_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    choices = [
        ModelChoice(
            model="gpt-5.5",
            provider="openai_responses",
            vendor="openai",
            description="Frontier model.",
        ),
        ModelChoice(
            model="claude-sonnet-4-6",
            provider="anthropic",
            vendor="anthropic",
            description="Balanced model.",
        ),
    ]
    monkeypatch.setattr(tui, "available_model_choices", lambda: choices)
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.buffer = "/model"
    live.cursor = len(live.buffer)

    live._move_model_picker_selection(1)
    live._draw_prompt()

    rendered = out.getvalue()
    assert f"{tui.SELECTED_ROW_STYLE} > claude-sonnet-4-6" in rendered

    out.seek(0)
    out.truncate(0)
    live._move_model_picker_selection(-1)
    live._draw_prompt()

    rendered = out.getvalue()
    assert f"{tui.SELECTED_ROW_STYLE} > gpt-5.5" in rendered


def test_live_model_picker_enter_applies_selected_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anthropic_provider = _ScriptedStreamProvider([])
    choices = [
        ModelChoice(
            model="gpt-5.5",
            provider="openai_responses",
            vendor="openai",
            description="Frontier model.",
        ),
        ModelChoice(
            model="claude-sonnet-4-6",
            provider="anthropic",
            vendor="anthropic",
            description="Balanced model.",
        ),
    ]

    monkeypatch.setattr(tui, "available_model_choices", lambda: choices)
    monkeypatch.setattr("willow.cli._build_provider", lambda _name: anthropic_provider)
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.buffer = "/model"
    live.cursor = len(live.buffer)

    live._move_model_picker_selection(1)
    live._submit_buffer()

    assert app.current_model == "claude-sonnet-4-6"
    assert app.current_provider_name == "anthropic"
    assert app.provider is anthropic_provider


def test_live_tab_completes_selected_model_without_applying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    choices = [
        ModelChoice(
            model="gpt-5.5",
            provider="openai_responses",
            vendor="openai",
            description="Frontier model.",
        ),
        ModelChoice(
            model="claude-sonnet-4-6",
            provider="anthropic",
            vendor="anthropic",
            description="Balanced model.",
        ),
    ]
    monkeypatch.setattr(tui, "available_model_choices", lambda: choices)
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.buffer = "/model"
    live.cursor = len(live.buffer)

    live._move_model_picker_selection(1)
    completed = live._complete_selected_hint()

    assert completed is True
    assert live.buffer == "/model claude-sonnet-4-6"
    assert live.cursor == len(live.buffer)
    assert app.current_model == "gpt-5.5"


def test_live_prompt_restores_statusline_when_command_prefix_is_removed() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.buffer = "/mo"
    live.cursor = len(live.buffer)

    live._draw_prompt()
    out.seek(0)
    out.truncate(0)
    live.buffer = ""
    live.cursor = 0
    live._draw_prompt()

    rendered = out.getvalue()
    assert "/model  choose what model to use" not in rendered
    assert "Context " in rendered


def test_live_stream_chunk_keeps_prompt_stable_until_completion() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.streaming = True
    live._draw_prompt()
    out.seek(0)
    out.truncate(0)

    live.events.put((live.active_turn_id, "stream", TextDelta(text="Hello")))
    live._drain_events()

    rendered = out.getvalue()
    assert rendered == ""
    assert live.stream_text == ["Hello"]
    assert "tokens:" not in rendered
    assert "streaming" not in rendered

    response = CompletionResponse(
        content=[TextBlock(text="Hello")],
        stop_reason="end_turn",
    )
    live.events.put((live.active_turn_id, "complete", response))
    live._drain_events()

    rendered = out.getvalue()
    assert "Hello" in rendered
    assert " > " in rendered


def test_live_prompt_shows_queued_messages_with_interrupt_hint() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.streaming = True
    live.pending_user_inputs = ["first", "second"]

    live._draw_prompt()

    rendered = out.getvalue()
    assert "Messages to be submitted after next tool call" in rendered
    assert "press esc to interrupt and send immediately" in rendered
    assert "↳ first" in rendered
    assert "↳ second" in rendered


def test_live_submit_while_streaming_writes_queued_status_panel() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.streaming = True
    live.buffer = "queued followup"
    live.cursor = len(live.buffer)

    live._submit_buffer()

    rendered = out.getvalue()
    assert live.pending_user_inputs == ["queued followup"]
    assert "Queued input while a turn is streaming" in rendered
    assert "press Esc to interrupt and send now" in rendered


def test_live_esc_interrupt_keeps_user_and_queued_messages_without_partial_assistant() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.streaming = True
    live.pending_user_inputs = ["queued followup"]
    live.stream_text = ["partial assistant text"]
    app.messages.append(Message(role="user", content=[TextBlock(text="refactor the TUI")]))
    started: list[int] = []

    def fake_reset_provider() -> None:
        return None

    def fake_start_worker() -> None:
        live.streaming = True
        live.active_turn_id += 1
        started.append(live.active_turn_id)

    live._reset_provider_for_interrupt = fake_reset_provider  # type: ignore[method-assign]
    live._start_worker = fake_start_worker  # type: ignore[method-assign]

    live._interrupt_and_send_queued()

    assert started
    assert [message.role for message in app.messages] == ["user"]
    assert app.messages[0].content == [
        TextBlock(text="[interrupted user message 1 of 2]\nrefactor the TUI"),
        TextBlock(text="[interrupting user message 2 of 2]\nqueued followup"),
    ]
    assert live.stream_text == []
    rendered = out.getvalue()
    assert "Interrupted current turn; sending queued input now." in rendered
    assert "partial assistant text" not in rendered


def test_live_esc_interrupt_sends_queued_message_to_provider_request() -> None:
    out = _TTYBuffer()
    response = CompletionResponse(content=[TextBlock(text="done")], stop_reason="end_turn")
    provider = _ScriptedStreamProvider([[StreamComplete(response=response)]])
    app = tui.WillowApp(_make_args(), provider, out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.streaming = True
    live.pending_user_inputs = ["surprise me"]
    app.messages.append(Message(role="user", content=[TextBlock(text="refactor the TUI")]))

    live._reset_provider_for_interrupt = lambda: None  # type: ignore[method-assign]

    live._interrupt_and_send_queued()
    assert live.worker is not None
    live.worker.join(timeout=1)

    assert len(provider.requests) == 1
    request = provider.requests[0]
    assert [message.role for message in request.messages] == ["user"]
    assert request.messages[0].content == [
        TextBlock(text="[interrupted user message 1 of 2]\nrefactor the TUI"),
        TextBlock(text="[interrupting user message 2 of 2]\nsurprise me"),
    ]


def test_live_esc_interrupt_replaces_provider_before_sending_queued_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = _TTYBuffer()
    old_provider = _ScriptedStreamProvider([])
    response = CompletionResponse(content=[TextBlock(text="done")], stop_reason="end_turn")
    new_provider = _ScriptedStreamProvider([[StreamComplete(response=response)]])
    app = tui.WillowApp(_make_args(), old_provider, out=out)
    app._force_plain = False
    app.messages.append(Message(role="user", content=[TextBlock(text="previous")]))
    app.messages.append(Message(role="assistant", content=[TextBlock(text="previous answer")]))
    app.messages.append(Message(role="user", content=[TextBlock(text="refactor the TUI")]))
    live = tui._LiveTerminal(app)
    live.streaming = True
    live.pending_user_inputs = ["steer now"]

    monkeypatch.setattr("willow.cli._build_provider", lambda _name: new_provider)

    live._interrupt_and_send_queued()
    assert live.worker is not None
    live.worker.join(timeout=1)

    assert app.provider is new_provider
    assert old_provider.requests == []
    assert len(new_provider.requests) == 1
    request = new_provider.requests[0]
    assert [message.role for message in request.messages] == ["user", "assistant", "user"]
    assert request.messages[0].content == [TextBlock(text="previous")]
    assert request.messages[1].content == [TextBlock(text="previous answer")]
    assert request.messages[2].content == [
        TextBlock(text="[interrupted user message 1 of 2]\nrefactor the TUI"),
        TextBlock(text="[interrupting user message 2 of 2]\nsteer now"),
    ]


def test_live_esc_interrupt_sends_current_buffer_to_provider_request() -> None:
    out = _TTYBuffer()
    response = CompletionResponse(content=[TextBlock(text="done")], stop_reason="end_turn")
    provider = _ScriptedStreamProvider([[StreamComplete(response=response)]])
    app = tui.WillowApp(_make_args(), provider, out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.streaming = True
    live.buffer = "surprise me"
    app.messages.append(Message(role="user", content=[TextBlock(text="refactor the TUI")]))

    live._reset_provider_for_interrupt = lambda: None  # type: ignore[method-assign]

    live._interrupt_with_buffer_if_possible()
    assert live.worker is not None
    live.worker.join(timeout=1)

    assert live.buffer == ""
    assert len(provider.requests) == 1
    request = provider.requests[0]
    assert [message.role for message in request.messages] == ["user"]
    assert request.messages[0].content == [
        TextBlock(text="[interrupted user message 1 of 2]\nrefactor the TUI"),
        TextBlock(text="[interrupting user message 2 of 2]\nsurprise me"),
    ]


def test_live_left_arrow_moves_cursor_for_mid_buffer_edit() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    read_fd, write_fd = os.pipe()
    try:
        live.fd = read_fd
        os.write(write_fd, b"abc\x1b[D\x1b[DX")
        live._read_available_input()
    finally:
        os.close(read_fd)
        os.close(write_fd)

    assert live.buffer == "aXbc"
    assert live.cursor == 2


def test_live_option_arrow_moves_cursor_by_word_for_mid_buffer_edit() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    read_fd, write_fd = os.pipe()
    try:
        live.fd = read_fd
        os.write(write_fd, b"alpha beta gamma\x1bb\x1bbX\x1b[1;3C!")
        live._read_available_input()
    finally:
        os.close(read_fd)
        os.close(write_fd)

    assert live.buffer == "alpha Xbeta! gamma"
    assert live.cursor == len("alpha Xbeta!")


def test_live_ignores_stale_stream_and_complete_events_after_interrupt() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    live = tui._LiveTerminal(app)
    live.active_turn_id = 2
    app.messages.append(Message(role="user", content=[TextBlock(text="first")]))
    stale_response = CompletionResponse(
        content=[TextBlock(text="stale final")],
        stop_reason="end_turn",
    )

    live.events.put((1, "stream", TextDelta(text="stale partial")))
    live.events.put((1, "complete", stale_response))
    live._drain_events()

    assert [message.role for message in app.messages] == ["user"]
    assert live.stream_text == []
    assert "stale partial" not in out.getvalue()
    assert "stale final" not in out.getvalue()


def test_terminal_streams_text_thinking_and_tool_markers_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _RecordingTool()
    monkeypatch.setitem(TOOLS_BY_NAME, tool.name, tool)
    tool_use_response = CompletionResponse(
        content=[ToolUseBlock(id="call_1", name="echo", input={"message": "hi"})],
        stop_reason="tool_use",
    )
    end_response = CompletionResponse(
        content=[TextBlock(text="done")],
        stop_reason="end_turn",
    )
    provider = _ScriptedStreamProvider(
        [
            [
                ThinkingDelta(thinking="reasoning"),
                TextDelta(text="answer"),
                ToolUseDelta(id="call_1", name="echo", partial_json=None),
                StreamComplete(response=tool_use_response),
            ],
            [TextDelta(text="done"), StreamComplete(response=end_response)],
        ]
    )

    out, _app = _drive(provider, ["use tool", "/exit"])

    assert tool.calls == [{"message": "hi"}]
    assert out.index("thinking") < out.index("reasoning")
    assert out.index("reasoning") < out.index("answer")
    assert out.index("answer") < out.index("● echo ok")
    assert out.index("● echo ok") < out.index("echoed: hi")
    assert out.index("echoed: hi") < out.index("done")
    assert "---" in out


def test_terminal_read_only_blocks_non_read_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = _RecordingTool()
    monkeypatch.setitem(TOOLS_BY_NAME, tool.name, tool)
    tool_use_response = CompletionResponse(
        content=[ToolUseBlock(id="call_1", name="echo", input={"message": "hi"})],
        stop_reason="tool_use",
    )
    end_response = CompletionResponse(
        content=[TextBlock(text="blocked")],
        stop_reason="end_turn",
    )
    provider = _ScriptedStreamProvider(
        [
            [
                ToolUseDelta(id="call_1", name="echo", partial_json=None),
                StreamComplete(response=tool_use_response),
            ],
            [TextDelta(text="blocked"), StreamComplete(response=end_response)],
        ]
    )

    out, _app = _drive(
        provider,
        ["use tool", "/exit"],
        args=_make_args(permission_mode=tui.PermissionMode.READ_ONLY),
    )

    assert tool.calls == []
    assert "● echo error" in out
    assert "read-only mode" in out


def test_terminal_ask_permission_denies_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = _RecordingTool()
    monkeypatch.setitem(TOOLS_BY_NAME, tool.name, tool)
    tool_use_response = CompletionResponse(
        content=[ToolUseBlock(id="call_1", name="echo", input={"message": "hi"})],
        stop_reason="tool_use",
    )
    end_response = CompletionResponse(content=[TextBlock(text="denied")], stop_reason="end_turn")
    provider = _ScriptedStreamProvider(
        [
            [
                ToolUseDelta(id="call_1", name="echo", partial_json=None),
                StreamComplete(response=tool_use_response),
            ],
            [TextDelta(text="denied"), StreamComplete(response=end_response)],
        ]
    )

    out, _app = _drive(
        provider,
        ["use tool", "n", "/exit"],
        args=_make_args(permission_mode=tui.PermissionMode.ASK),
    )

    assert tool.calls == []
    assert "[permission] Allow echo" in out
    assert "Permission denied by user" in out


def test_terminal_ask_permission_allows_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = _RecordingTool()
    monkeypatch.setitem(TOOLS_BY_NAME, tool.name, tool)
    tool_use_response = CompletionResponse(
        content=[ToolUseBlock(id="call_1", name="echo", input={"message": "hi"})],
        stop_reason="tool_use",
    )
    end_response = CompletionResponse(content=[TextBlock(text="done")], stop_reason="end_turn")
    provider = _ScriptedStreamProvider(
        [
            [
                ToolUseDelta(id="call_1", name="echo", partial_json=None),
                StreamComplete(response=tool_use_response),
            ],
            [TextDelta(text="done"), StreamComplete(response=end_response)],
        ]
    )

    out, _app = _drive(
        provider,
        ["use tool", "y", "/exit"],
        args=_make_args(permission_mode=tui.PermissionMode.ASK),
    )

    assert tool.calls == [{"message": "hi"}]
    assert "[permission] Allow echo" in out
    assert "● echo ok" in out


def test_terminal_bash_tool_renders_command_and_concise_output() -> None:
    tool_use_response = CompletionResponse(
        content=[
            ToolUseBlock(
                id="call_1",
                name="bash",
                input={"command": "seq 1 5"},
            )
        ],
        stop_reason="tool_use",
    )
    end_response = CompletionResponse(
        content=[TextBlock(text="done")],
        stop_reason="end_turn",
    )
    provider = _ScriptedStreamProvider(
        [
            [
                ToolUseDelta(id="call_1", name="bash", partial_json=None),
                StreamComplete(response=tool_use_response),
            ],
            [TextDelta(text="done"), StreamComplete(response=end_response)],
        ]
    )

    out, _app = _drive(provider, ["run command", "/exit"])

    assert "● bash ok - ran seq 1 5" in out
    assert "  1" in out
    assert "  4" in out
    assert "... +1 lines" in out
    assert "\n  5\n" not in out


def test_terminal_skill_command_expands_user_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    skill_path = tmp_path / ".willow" / "skills" / "writer" / "SKILL.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text("Write tersely.", encoding="utf-8")
    response = CompletionResponse(
        content=[TextBlock(text="done")],
        stop_reason="end_turn",
    )
    provider = _ScriptedStreamProvider([[StreamComplete(response=response)]])

    out, _app = _drive(provider, ["/writer draft notes", "/exit"])

    sent_text = provider.requests[0].messages[0].content[0].text
    assert "Write tersely." in sent_text
    assert "User task:\ndraft notes" in sent_text
    assert "/writer draft notes" in out


def test_terminal_project_hello_world_skill_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    monkeypatch.chdir(repo_root)
    response = CompletionResponse(
        content=[TextBlock(text="done")],
        stop_reason="end_turn",
    )
    provider = _ScriptedStreamProvider([[StreamComplete(response=response)]])

    out, _app = _drive(provider, ["/hello_world confirm skill wiring", "/exit"])

    sent_text = provider.requests[0].messages[0].content[0].text
    assert "Hello World" in sent_text
    assert "User task:\nconfirm skill wiring" in sent_text
    assert "/hello_world confirm skill wiring" in out


def test_terminal_tool_error_renders_tool_name_state_and_clean_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingTool(Tool):
        name = "explode"
        description = "Raise an error."
        input_schema = {"type": "object", "properties": {}}

        def run(self, **_kwargs: Any) -> str:
            raise ValueError("bad input")

    monkeypatch.setitem(TOOLS_BY_NAME, "explode", FailingTool())
    tool_use_response = CompletionResponse(
        content=[ToolUseBlock(id="call_1", name="explode", input={})],
        stop_reason="tool_use",
    )
    end_response = CompletionResponse(
        content=[TextBlock(text="done")],
        stop_reason="end_turn",
    )
    provider = _ScriptedStreamProvider(
        [
            [StreamComplete(response=tool_use_response)],
            [TextDelta(text="done"), StreamComplete(response=end_response)],
        ]
    )

    out, _app = _drive(provider, ["run tool", "/exit"])

    assert "● explode error" in out
    assert "ValueError: bad input" in out
    assert "ValueError('bad input')" not in out


def test_edit_tool_result_renders_diff_review_style() -> None:
    out = _TTYBuffer()
    app = tui.WillowApp(_make_args(), _ScriptedStreamProvider([]), out=out)
    app._force_plain = False
    block = ToolUseBlock(
        id="call_1",
        name="write",
        input={"path": "codex_tui_test_haha1/wobbly_pickle_dispatch.txt"},
    )
    result = ToolResultBlock(
        tool_use_id="call_1",
        content=(
            "Wrote 12 bytes to codex_tui_test_haha1/wobbly_pickle_dispatch.txt\n"
            "--- codex_tui_test_haha1/wobbly_pickle_dispatch.txt (before)\n"
            "+++ codex_tui_test_haha1/wobbly_pickle_dispatch.txt (after)\n"
            "@@ -0,0 +1,2 @@\n"
            "+hello\n"
            "+world"
        ),
    )

    app._write_tool_result(block, result)

    rendered = out.getvalue()
    assert (
        "write ok - added codex_tui_test_haha1/wobbly_pickle_dispatch.txt (+2 -0)"
        in rendered
    )
    assert "    1 +hello" in rendered
    assert "    2 +world" in rendered
    assert tui.DIFF_ADD_STYLE in rendered


def test_terminal_deduplicates_separators_between_consecutive_tool_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _RecordingTool()
    monkeypatch.setitem(TOOLS_BY_NAME, tool.name, tool)

    def tool_turn(idx: int) -> list[Any]:
        response = CompletionResponse(
            content=[ToolUseBlock(id=f"call_{idx}", name="echo", input={"message": str(idx)})],
            stop_reason="tool_use",
        )
        return [
            ToolUseDelta(id=f"call_{idx}", name="echo", partial_json=None),
            StreamComplete(response=response),
        ]

    end_response = CompletionResponse(
        content=[TextBlock(text="done")],
        stop_reason="end_turn",
    )
    provider = _ScriptedStreamProvider(
        [
            tool_turn(1),
            tool_turn(2),
            [TextDelta(text="done"), StreamComplete(response=end_response)],
        ]
    )

    out, _app = _drive(provider, ["use tools", "/exit"])

    assert out.count("---") == 3
    assert "---\n---" not in out
    assert out.index("● echo ok") < out.index("echoed: 1")
    assert out.rindex("● echo ok") < out.index("echoed: 2")


def test_tui_persists_session_file(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(session.SESSION_DIR_ENV, str(tmp_path / "sessions"))
    response = CompletionResponse(
        content=[TextBlock(text="assistant done")],
        stop_reason="end_turn",
    )
    provider = _ScriptedStreamProvider([[StreamComplete(response=response)]])

    out, app = _drive(
        provider,
        ["hello", "/exit"],
        args=_make_args(persist_session=True),
    )

    assert app._session_path is not None
    assert str(app._session_path) not in out
    saved = session.load_session(app._session_path)
    assert saved.settings.provider == "openai_responses"
    assert saved.settings.model == "gpt-5.5"
    assert [message.role for message in saved.messages] == ["user", "assistant"]
    assert saved.messages[0].content == [TextBlock(text="hello")]
    assert saved.messages[1].content == [TextBlock(text="assistant done")]


def test_terminal_session_command_shows_persistence_and_saved_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(session.SESSION_DIR_ENV, str(tmp_path / "sessions"))
    provider = _ScriptedStreamProvider([])

    out, app = _drive(
        provider,
        ["/session", "/exit"],
        args=_make_args(persist_session=True),
    )

    assert app._session_path is not None
    assert "[status] persistence: enabled" in out
    assert f"session: {app._session_path}" in out
    assert "statusline: on" in out


def test_terminal_status_command_shows_disabled_persistence() -> None:
    provider = _ScriptedStreamProvider([])

    out, _app = _drive(provider, ["/status", "/exit"])

    assert "[status] persistence: disabled" in out
    assert "session: (not saved)" in out


def test_terminal_clear_is_append_only_and_resets_history() -> None:
    end1 = CompletionResponse(content=[TextBlock(text="first done")], stop_reason="end_turn")
    end2 = CompletionResponse(content=[TextBlock(text="second done")], stop_reason="end_turn")
    provider = _ScriptedStreamProvider(
        [
            [TextDelta(text="first done"), StreamComplete(response=end1)],
            [TextDelta(text="second done"), StreamComplete(response=end2)],
        ]
    )

    out, _app = _drive(provider, ["first", "/clear", "second", "/exit"])

    assert "first done" in out
    assert "Conversation cleared." in out
    assert "second done" in out
    assert len(provider.requests) == 2
    assert len(provider.requests[1].messages) == 1
    block = provider.requests[1].messages[0].content[0]
    assert isinstance(block, TextBlock)
    assert block.text == "second"


def test_terminal_model_command_lists_models_inline(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _ScriptedStreamProvider([])
    monkeypatch.setattr(
        tui,
        "available_model_choices",
        lambda: [
            ModelChoice(
                model="gpt-5.5",
                provider="openai_responses",
                vendor="openai",
                description="Frontier model.",
            )
        ],
    )

    out, _app = _drive(provider, ["/model", "/exit"])

    assert "Select Model" in out
    assert "gpt-5.5" in out
    assert provider.requests == []


def test_terminal_model_selection_switches_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    anthropic_provider = _ScriptedStreamProvider([])
    openai_response = CompletionResponse(content=[TextBlock(text="ok")], stop_reason="end_turn")
    openai_provider = _ScriptedStreamProvider(
        [[TextDelta(text="ok"), StreamComplete(response=openai_response)]]
    )
    choices = [
        ModelChoice(
            model="gpt-5.5",
            provider="openai_responses",
            vendor="openai",
            description="Frontier model.",
        )
    ]

    def build_provider(name: str) -> Provider:
        return {
            "anthropic": anthropic_provider,
            "openai_responses": openai_provider,
        }[name]

    monkeypatch.setattr(tui, "available_model_choices", lambda: choices)
    monkeypatch.setattr("willow.cli._build_provider", build_provider)

    out, _app = _drive(
        anthropic_provider,
        ["/model 1", "go", "/exit"],
        args=_make_args(provider="anthropic"),
    )

    assert "using provider 'openai_responses'" in out
    assert anthropic_provider.requests == []
    assert len(openai_provider.requests) == 1
    assert openai_provider.requests[0].model == "gpt-5.5"


def test_terminal_statusline_can_be_disabled() -> None:
    end_response = CompletionResponse(
        content=[TextBlock(text="ok")],
        stop_reason="end_turn",
        usage={"input_tokens": 10, "output_tokens": 5},
    )
    provider = _ScriptedStreamProvider(
        [[TextDelta(text="ok"), StreamComplete(response=end_response)]]
    )

    out, app = _drive(provider, ["/statusline off", "go", "/exit"])

    assert "Status snapshots disabled." in out
    assert "[status]" not in out
    assert app._statusline_enabled is False


def test_max_iterations_caps_inner_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    tool = _RecordingTool()
    monkeypatch.setitem(TOOLS_BY_NAME, tool.name, tool)

    def make_tool_use_turn(idx: int) -> list[Any]:
        response = CompletionResponse(
            content=[ToolUseBlock(id=f"t{idx}", name="echo", input={"message": str(idx)})],
            stop_reason="tool_use",
        )
        return [
            ToolUseDelta(id=f"t{idx}", name="echo", partial_json=None),
            StreamComplete(response=response),
        ]

    provider = _ScriptedStreamProvider([make_tool_use_turn(i) for i in range(5)])

    out, _app = _drive(
        provider,
        ["loop forever", "/exit"],
        args=_make_args(max_iterations=3),
    )

    assert len(provider.requests) == 3
    assert "max_iterations=3" in out
