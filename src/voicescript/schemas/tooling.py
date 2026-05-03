from __future__ import annotations

from .base import JsonModel


class CommandProvenance(JsonModel):
    tool: str
    command: list[str]
    exit_code: int
    stdout_excerpt: str = ""
    stderr_excerpt: str = ""


class ToolResult(JsonModel):
    stdout: str
    stderr: str
    exit_code: int
    command: list[str]
