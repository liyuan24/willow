"""Willow — a pure-Python agent framework.

Public top-level surface is exposed lazily so that `import willow` itself
has no side effects (in particular, no eager read of `~/.willow/auth.json`,
which the auth module's contract requires).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["PROVIDER_TO_VENDOR", "main", "run_agent"]


def main() -> int:
    """Console-script entry point — delegates to the CLI module.

    The `[project.scripts]` entry installed by uv points at `willow.cli:main`
    directly; this thin shim is kept so legacy `from willow import main`
    callers continue to work.
    """
    from willow.cli import main as cli_main

    return cli_main()


# Any: standard module-level `__getattr__` signature; the return type is
# the union of every attribute this hook can produce, which Python typing
# expresses as `Any` by convention.
def __getattr__(name: str) -> Any:
    if name in ("run_agent", "PROVIDER_TO_VENDOR"):
        from willow.agent import PROVIDER_TO_VENDOR, run_agent

        globals()["run_agent"] = run_agent
        globals()["PROVIDER_TO_VENDOR"] = PROVIDER_TO_VENDOR
        return globals()[name]
    raise AttributeError(f"module 'willow' has no attribute {name!r}")


if TYPE_CHECKING:
    from willow.agent import PROVIDER_TO_VENDOR, run_agent  # noqa: F401
