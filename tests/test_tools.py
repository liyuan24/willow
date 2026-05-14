from __future__ import annotations

from pathlib import Path

from willow.runtime import WillowRuntime
from willow.tools import TOOLS_BY_NAME, build_tools
from willow.tools.bash import BashTool
from willow.tools.edit import EditTool
from willow.tools.monitor import MonitorTool
from willow.tools.read import ReadTool
from willow.tools.write import WriteTool


def test_edit_tool_is_registered() -> None:
    assert isinstance(TOOLS_BY_NAME["edit"], EditTool)


def test_grep_tool_is_not_registered() -> None:
    assert "grep" not in TOOLS_BY_NAME


def test_build_tools_shares_runtime_between_bash_and_monitor(tmp_path) -> None:
    runtime = WillowRuntime(root=tmp_path)
    tools = build_tools(runtime)

    bash = tools["bash"]
    monitor = tools["monitor"]

    assert isinstance(bash, BashTool)
    assert isinstance(monitor, MonitorTool)
    assert bash.runtime is runtime
    assert monitor.runtime is runtime


def test_default_tools_share_runtime_between_bash_and_monitor() -> None:
    bash = TOOLS_BY_NAME["bash"]
    monitor = TOOLS_BY_NAME["monitor"]

    assert isinstance(bash, BashTool)
    assert isinstance(monitor, MonitorTool)
    assert bash.runtime is monitor.runtime


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


def test_read_tool_large_output_is_externalized(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "large.txt"
    path.write_text("\n".join(f"line {i}" for i in range(5000)))

    output = ReadTool().run(str(path))
    fields = dict(line.split(": ", 1) for line in output.splitlines() if ": " in line)
    full_output_path = Path(fields["full_output_path"])
    full_output = full_output_path.read_text()

    assert output.startswith("[output truncated:")
    assert str(full_output_path).startswith(str(tmp_path / ".willow" / "artifacts"))
    assert "     1\tline 0" in full_output
    assert "  5000\tline 4999" in full_output
    assert len(output) < len(full_output)
