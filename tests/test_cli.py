"""Tests for the top-level `willow` command."""

from __future__ import annotations

import argparse
import io
import sys
from typing import Any

import pytest

from willow import cli
from willow.providers import CompletionResponse, TextBlock, ToolUseBlock


def test_build_parser_defaults_to_tui_settings() -> None:
    parser = cli._build_parser()
    args = parser.parse_args([])

    assert args.prompt is None
    assert args.provider == "openai_codex"
    assert args.model == "gpt-5.5"
    assert args.max_tokens == 4096
    assert args.max_iterations == 20
    assert args.resume is None
    assert args.permission_mode == cli.PermissionMode.YOLO


def test_build_parser_print_mode_accepts_prompt_and_shared_flags() -> None:
    parser = cli._build_parser()
    args = parser.parse_args(
        [
            "--provider",
            "anthropic",
            "--model",
            "claude-sonnet-4-6",
            "--max-tokens",
            "512",
            "--max-iterations",
            "3",
            "--read-only",
            "-p",
            "follow the prompt",
        ]
    )

    assert args.prompt == "follow the prompt"
    assert args.provider == "anthropic"
    assert args.model == "claude-sonnet-4-6"
    assert args.max_tokens == 512
    assert args.max_iterations == 3
    assert args.permission_mode == cli.PermissionMode.READ_ONLY


def test_build_parser_resume_accepts_optional_session() -> None:
    parser = cli._build_parser()

    choose_args = parser.parse_args(["--resume"])
    direct_args = parser.parse_args(["--resume", "sess_123"])

    assert choose_args.resume == ""
    assert direct_args.resume == "sess_123"


def test_build_parser_rejects_removed_subcommands() -> None:
    parser = cli._build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["chat"])
    with pytest.raises(SystemExit):
        parser.parse_args(["tui"])
    with pytest.raises(SystemExit):
        parser.parse_args(["headless", "prompt"])


def test_build_parser_permission_modes_are_mutually_exclusive() -> None:
    parser = cli._build_parser()

    assert parser.parse_args(["--read-only"]).permission_mode == cli.PermissionMode.READ_ONLY
    assert (
        parser.parse_args(["--ask-for-permission"]).permission_mode
        == cli.PermissionMode.ASK
    )
    assert parser.parse_args(["--yolo", "-p", "prompt"]).permission_mode == cli.PermissionMode.YOLO
    with pytest.raises(SystemExit):
        parser.parse_args(["--yolo", "--read-only"])


def test_main_without_prompt_runs_tui(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[argparse.Namespace] = []

    def fake_run_tui(args: argparse.Namespace) -> int:
        calls.append(args)
        return 42

    monkeypatch.setattr(cli, "_run_tui", fake_run_tui)
    monkeypatch.setattr(cli, "_run_headless", lambda _args: pytest.fail("headless called"))

    assert cli.main([]) == 42
    assert len(calls) == 1
    assert calls[0].prompt is None


def test_main_with_prompt_runs_headless(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[argparse.Namespace] = []

    def fake_run_headless(args: argparse.Namespace) -> int:
        calls.append(args)
        return 7

    monkeypatch.setattr(cli, "_run_tui", lambda _args: pytest.fail("tui called"))
    monkeypatch.setattr(cli, "_run_headless", fake_run_headless)

    assert cli.main(["-p", "follow the prompt"]) == 7
    assert len(calls) == 1
    assert calls[0].prompt == "follow the prompt"


def test_headless_calls_run_agent_and_prints_final_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = CompletionResponse(
        content=[TextBlock(text="hello"), TextBlock(text=" world")],
        stop_reason="end_turn",
    )
    calls: list[tuple[Any, ...]] = []

    def fake_run_agent(*args: Any, **kwargs: Any) -> CompletionResponse:
        calls.append((args, kwargs))
        return response

    stdout = io.StringIO()
    monkeypatch.setattr(cli, "run_agent", fake_run_agent)
    monkeypatch.setattr(sys, "stdout", stdout)

    rc = cli._run_headless(
        argparse.Namespace(
            provider="anthropic",
            model="claude-sonnet-4-6",
            max_tokens=512,
            max_iterations=3,
            permission_mode=cli.PermissionMode.READ_ONLY,
            prompt="follow the prompt",
        )
    )

    assert rc == 0
    assert stdout.getvalue() == "hello world\n"
    assert calls == [
        (
            ("anthropic", "claude-sonnet-4-6", "follow the prompt"),
            {
                "max_tokens": 512,
                "max_iterations": 3,
                "permission_mode": cli.PermissionMode.READ_ONLY,
            },
        )
    ]


def test_headless_returns_nonzero_if_tools_remain_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = CompletionResponse(
        content=[ToolUseBlock(id="call_1", name="echo", input={"message": "hi"})],
        stop_reason="tool_use",
    )
    stdout = io.StringIO()
    stderr = io.StringIO()

    monkeypatch.setattr(cli, "run_agent", lambda *args, **kwargs: response)
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)

    rc = cli._run_headless(
        argparse.Namespace(
            provider="openai_responses",
            model="gpt-5.5",
            max_tokens=4096,
            max_iterations=1,
            permission_mode=cli.PermissionMode.YOLO,
            prompt="use a tool",
        )
    )

    assert rc == 1
    assert stdout.getvalue() == ""
    assert "tools were still pending" in stderr.getvalue()


def test_headless_rejects_ask_for_permission(monkeypatch: pytest.MonkeyPatch) -> None:
    stderr = io.StringIO()
    monkeypatch.setattr(sys, "stderr", stderr)

    rc = cli._run_headless(
        argparse.Namespace(
            provider="openai_responses",
            model="gpt-5.5",
            max_tokens=4096,
            max_iterations=1,
            permission_mode=cli.PermissionMode.ASK,
            prompt="use a tool",
        )
    )

    assert rc == 2
    assert "--ask-for-permission" in stderr.getvalue()
