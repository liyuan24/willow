"""Willow's command-line entry point.

Default usage mirrors coding-agent CLIs:

    willow
    willow -p "follow this prompt"

Running without ``-p/--print`` opens the native terminal UI. Running with
``-p/--print`` executes one prompt headlessly and prints only the final
assistant text.
"""

from __future__ import annotations

import argparse
import sys

import anthropic
import openai

from willow.agent import _ProviderSpec, run_agent
from willow.auth import get_credential
from willow.permissions import PermissionMode
from willow.providers import (
    AnthropicProvider,
    OpenAICodexResponsesProvider,
    OpenAICompletionsProvider,
    OpenAIResponsesProvider,
    Provider,
    TextBlock,
)

# ---------------------------------------------------------------------------
# Provider construction
# ---------------------------------------------------------------------------


# `_ProviderSpec` is reused from `willow.agent` so the dispatch tables share
# the same typed surface while letting the TUI own a longer-lived provider.
_PROVIDER_DISPATCH: dict[
    str, _ProviderSpec[anthropic.Anthropic] | _ProviderSpec[openai.OpenAI] | _ProviderSpec[str]
] = {
    "anthropic": _ProviderSpec[anthropic.Anthropic](
        vendor="anthropic",
        client_factory=lambda key: anthropic.Anthropic(api_key=key),
        provider_factory=lambda client: AnthropicProvider(client=client),
    ),
    "openai_codex": _ProviderSpec[str](
        vendor="openai",
        client_factory=lambda key: key,
        provider_factory=lambda token: OpenAICodexResponsesProvider(bearer_token=token),
    ),
    "openai_completions": _ProviderSpec[openai.OpenAI](
        vendor="openai",
        client_factory=lambda key: openai.OpenAI(api_key=key),
        provider_factory=lambda client: OpenAICompletionsProvider(client=client),
    ),
    "openai_responses": _ProviderSpec[openai.OpenAI](
        vendor="openai",
        client_factory=lambda key: openai.OpenAI(api_key=key),
        provider_factory=lambda client: OpenAIResponsesProvider(client=client),
    ),
}


def _build_provider(provider_name: str) -> Provider:
    """Resolve a provider name to a fully-wired Provider instance."""
    spec = _PROVIDER_DISPATCH.get(provider_name)
    if spec is None:
        raise ValueError(
            f"Unknown provider: {provider_name!r}. "
            f"Choices: {sorted(_PROVIDER_DISPATCH)}"
        )
    credential = get_credential(spec.vendor)
    return spec.build(credential.bearer_token)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _run_headless(args: argparse.Namespace) -> int:
    """Run one prompt and print only the final assistant text."""
    permission_mode = getattr(args, "permission_mode", PermissionMode.YOLO)
    if permission_mode == PermissionMode.ASK:
        sys.stderr.write(
            "willow -p does not support --ask-for-permission; "
            "use --yolo or --read-only.\n"
        )
        sys.stderr.flush()
        return 2

    response = run_agent(
        args.provider,
        args.model,
        args.prompt,
        max_tokens=args.max_tokens,
        max_iterations=args.max_iterations,
        permission_mode=permission_mode,
    )

    if response.stop_reason == "tool_use":
        sys.stderr.write(
            f"willow -p stopped while tools were still pending; "
            f"increase --max-iterations above {args.max_iterations}.\n"
        )
        sys.stderr.flush()
        return 1

    text = "".join(
        block.text for block in response.content if isinstance(block, TextBlock)
    )
    if text:
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")
        sys.stdout.flush()
    return 0


def _run_tui(args: argparse.Namespace) -> int:
    """Lazy-import wrapper so headless mode doesn't pay the TUI import cost."""
    from willow.tui import run_tui

    return run_tui(args)


# ---------------------------------------------------------------------------
# argparse plumbing
# ---------------------------------------------------------------------------


def _add_permission_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--yolo",
        dest="permission_mode",
        action="store_const",
        const=PermissionMode.YOLO,
        default=PermissionMode.YOLO,
        help="Run requested tools without asking for confirmation (default).",
    )
    group.add_argument(
        "--read-only",
        dest="permission_mode",
        action="store_const",
        const=PermissionMode.READ_ONLY,
        help="Only allow read-only tools.",
    )
    group.add_argument(
        "--ask-for-permission",
        dest="permission_mode",
        action="store_const",
        const=PermissionMode.ASK,
        help="Ask before executing each tool. Not supported with -p.",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="willow",
        description="Willow — a pure-Python coding agent.",
    )
    parser.add_argument(
        "-p",
        "--print",
        dest="prompt",
        metavar="PROMPT",
        default=None,
        help="Run one prompt headlessly and print the final response.",
    )
    parser.add_argument(
        "--initial-prompt",
        default=None,
        help="Initial prompt to prefill when opening the interactive TUI.",
    )
    parser.add_argument(
        "--provider",
        choices=sorted(_PROVIDER_DISPATCH),
        default="openai_codex",
        help="Provider to talk to (default: openai_codex).",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.5",
        help="Model id forwarded to the provider (default: gpt-5.5).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Per-turn output token cap (default: 4096).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
        help="Hard cap on assistant turns for the prompt (default: 20).",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="",
        default=None,
        metavar="SESSION",
        help=(
            "Resume a saved TUI session. With SESSION, use that session id or JSON path; "
            "without SESSION, choose from recent sessions."
        ),
    )
    _add_permission_args(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.prompt is not None:
        return _run_headless(args)
    return _run_tui(args)


if __name__ == "__main__":
    raise SystemExit(main())
