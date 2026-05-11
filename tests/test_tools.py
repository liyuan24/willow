from __future__ import annotations

from willow.tools import TOOLS_BY_NAME
from willow.tools.edit import EditTool
from willow.tools.write import WriteTool


def test_edit_tool_is_registered() -> None:
    assert isinstance(TOOLS_BY_NAME["edit"], EditTool)


def test_write_tool_returns_unified_diff_for_new_file(tmp_path) -> None:
    path = tmp_path / "funny.txt"

    output = WriteTool().run(str(path), "alpha\nbeta\n")

    assert path.read_text() == "alpha\nbeta\n"
    assert "Wrote 11 bytes" in output
    assert "+++ " in output
    assert "+alpha" in output
    assert "+beta" in output


def test_edit_tool_replaces_text_and_returns_diff(tmp_path) -> None:
    path = tmp_path / "story.txt"
    path.write_text("hello old world\n")

    output = EditTool().run(str(path), "old", "new")

    assert path.read_text() == "hello new world\n"
    assert "Edited" in output
    assert "-hello old world" in output
    assert "+hello new world" in output
