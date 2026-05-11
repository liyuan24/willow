import subprocess

from .base import Tool


class BashTool(Tool):
    name = "bash"
    description = (
        "Execute a shell command in the user's default shell and return its combined "
        "stdout/stderr. Non-zero exit codes are reported in the output, not raised. "
        "Do not use bash to create or edit files; use write for new files and edit "
        "for existing file changes."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute.",
            },
            "timeout": {
                "type": "number",
                "description": "Timeout in seconds. Defaults to 120.",
            },
        },
        "required": ["command"],
    }

    def run(self, command: str, timeout: float = 120.0) -> str:
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return f"[timeout after {timeout}s]"

        parts = []
        if result.stdout:
            parts.append(result.stdout.rstrip("\n"))
        if result.stderr:
            parts.append(f"[stderr]\n{result.stderr.rstrip(chr(10))}")
        if result.returncode != 0:
            parts.append(f"[exit {result.returncode}]")
        return "\n".join(parts) if parts else "[no output]"
