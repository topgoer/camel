"""
MCP module initialization for VerseMind-RAG.
"""

from app.mcp.mcp_server_manager import (
    start_mcp_server,
    stop_mcp_server,
    set_versemind_data,
    get_versemind_data,
)

__all__ = [
    "start_mcp_server",
    "stop_mcp_server",
    "set_versemind_data",
    "get_versemind_data",
]
