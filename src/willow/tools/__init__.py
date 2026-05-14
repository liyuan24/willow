from willow.runtime import WillowRuntime

from .base import Tool, ToolSpec
from .bash import BashTool
from .edit import EditTool
from .monitor import MonitorTool
from .read import ReadTool
from .write import WriteTool

DEFAULT_RUNTIME = WillowRuntime()


def build_tools(runtime: WillowRuntime | None = None) -> dict[str, Tool]:
    shared_runtime = runtime if runtime is not None else DEFAULT_RUNTIME
    tools: list[Tool] = [
        BashTool(runtime=shared_runtime),
        MonitorTool(runtime=shared_runtime),
        ReadTool(),
        WriteTool(),
        EditTool(),
    ]
    return {tool.name: tool for tool in tools}


TOOLS_BY_NAME: dict[str, Tool] = build_tools()
ALL_TOOLS: list[Tool] = list(TOOLS_BY_NAME.values())

__all__ = [
    "ALL_TOOLS",
    "TOOLS_BY_NAME",
    "BashTool",
    "DEFAULT_RUNTIME",
    "EditTool",
    "MonitorTool",
    "ReadTool",
    "Tool",
    "ToolSpec",
    "WillowRuntime",
    "WriteTool",
    "build_tools",
]
