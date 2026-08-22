"""Theseus connection for the research agent."""

from collections.abc import Callable, Iterable
from pathlib import Path

from theseus import CodexAppServer, register_tools


def create_agent(
    project_root: Path, tools: Iterable[Callable[..., object]]
) -> tuple[CodexAppServer, str]:
    client = CodexAppServer(project_root)
    tool_names = register_tools(client, tools)
    thread_id = client.create_agent(tools=tool_names)
    return client, thread_id
