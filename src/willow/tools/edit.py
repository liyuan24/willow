from __future__ import annotations

import difflib
from pathlib import Path

from .base import Tool


def render_file_diff(
    before: str,
    after: str,
    path: Path,
    *,
    max_lines: int = 120,
) -> str:
    before_lines = before.splitlines()
    after_lines = after.splitlines()
    diff = list(
        difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile=f"{path} (before)",
            tofile=f"{path} (after)",
            lineterm="",
        )
    )
    if not diff:
        return "[no changes]"
    if len(diff) <= max_lines:
        return "\n".join(diff)
    shown = diff[:max_lines]
    return "\n".join([*shown, f"... +{len(diff) - max_lines} diff lines"])


class EditTool(Tool):
    name = "edit"
    description = (
        "Edit an existing text file by replacing old_text with new_text. Use this tool "
        "for file edits; do not use bash, sed, perl, or python scripts to modify files."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to the file.",
            },
            "old_text": {
                "type": "string",
                "description": "Exact text to replace. Must appear in the file.",
            },
            "new_text": {
                "type": "string",
                "description": "Replacement text.",
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace every occurrence instead of only the first.",
            },
        },
        "required": ["path", "old_text", "new_text"],
    }

    def run(
        self,
        path: str,
        old_text: str,
        new_text: str,
        replace_all: bool = False,
    ) -> str:
        p = Path(path)
        before = p.read_text()
        if old_text not in before:
            raise ValueError(f"old_text not found in {p}")
        count = -1 if replace_all else 1
        after = before.replace(old_text, new_text, count)
        p.write_text(after)
        diff = render_file_diff(before, after, p)
        occurrences = before.count(old_text) if replace_all else 1
        return f"Edited {p} ({occurrences} replacement{'s' if occurrences != 1 else ''})\n{diff}"
