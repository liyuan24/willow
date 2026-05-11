from .base import Tool, ToolSpec
from .bash import BashTool
from .edit import EditTool
from .grep import GrepTool
from .read import ReadTool
from .write import WriteTool

ALL_TOOLS: list[Tool] = [BashTool(), ReadTool(), WriteTool(), EditTool(), GrepTool()]
TOOLS_BY_NAME: dict[str, Tool] = {t.name: t for t in ALL_TOOLS}

__all__ = [
    "ALL_TOOLS",
    "TOOLS_BY_NAME",
    "BashTool",
    "EditTool",
    "GrepTool",
    "ReadTool",
    "Tool",
    "ToolSpec",
    "WriteTool",
]
