from pathlib import Path

from .base import Tool


class ReadTool(Tool):
    name = "read"
    description = (
        "Read a text file from disk. Returns contents prefixed with 1-indexed line numbers. "
        "Use offset/limit for large files."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to the file.",
            },
            "offset": {
                "type": "integer",
                "description": "1-indexed line to start reading from. Defaults to 1.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of lines to return. Defaults to all.",
            },
        },
        "required": ["path"],
    }

    def run(self, path: str, offset: int = 1, limit: int | None = None) -> str:
        text = Path(path).read_text()
        lines = text.splitlines()
        start = max(0, offset - 1)
        end = start + limit if limit is not None else len(lines)
        selected = lines[start:end]
        if not selected:
            return "[empty]"
        return "\n".join(f"{start + i + 1:6d}\t{line}" for i, line in enumerate(selected))
