#!/bin/sh
# This script runs the MCP server in STDIO mode using uv

# Activate uv environment implicitly if needed, or ensure uv is in PATH
# Assuming uv run handles the environment correctly
echo "Starting MCP server in STDIO mode via wrapper script..." >&2
uv run python -m mcp_server.main 