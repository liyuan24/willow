import shutil
import subprocess

from .base import Tool

_INSTALL_HINT = (
    "ripgrep (`rg`) is required but not found on PATH. Install it:\n"
    "  macOS:         brew install ripgrep\n"
    "  Debian/Ubuntu: sudo apt install ripgrep\n"
    "  Fedora:        sudo dnf install ripgrep\n"
    "  Arch:          sudo pacman -S ripgrep"
)


class GrepTool(Tool):
    name = "grep"
    description = (
        "Search for a regex pattern across files using ripgrep. "
        "Returns matches as `path:line:text`."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regex pattern to search for.",
            },
            "path": {
                "type": "string",
                "description": "Directory or file to search. Defaults to current directory.",
            },
            "glob": {
                "type": "string",
                "description": "Optional file glob filter, e.g. '*.py'.",
            },
        },
        "required": ["pattern"],
    }

    def run(self, pattern: str, path: str = ".", glob: str | None = None) -> str:
        if not shutil.which("rg"):
            raise RuntimeError(_INSTALL_HINT)

        cmd = ["rg", "--line-number", "--no-heading", "--color=never"]
        if glob:
            cmd += ["--glob", glob]
        cmd += [pattern, path]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 1 and not result.stdout:
            return "[no matches]"
        if result.returncode > 1:
            return f"[error]\n{result.stderr.rstrip()}"
        return result.stdout.rstrip("\n") or "[no matches]"
