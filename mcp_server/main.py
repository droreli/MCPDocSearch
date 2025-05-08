import traceback
import sys
import os

# Import the shared FastMCP instance 
# (It will be accessed via the CLI runner, not directly run here)
from mcp_server.app import mcp_server

# Import the data loading function and chunk access function
from mcp_server.data_loader import load_and_chunk_documents, get_all_chunks

# Import the tools module to ensure decorators run and register tools
import mcp_server.mcp_tools  # noqa: F401

# --- Main Execution --- 
# This block might only be needed for data loading now, 
# as the server execution is handled by `fastmcp run` command.
if __name__ == "__main__":
    # Load documents synchronously before starting the server
    print("Loading documents...", file=sys.stderr)
    load_and_chunk_documents()
    # Print status after loading
    num_chunks = len(get_all_chunks())
    print(f"Document loading complete. {num_chunks} chunks loaded.", file=sys.stderr)

    # The actual server run command is now external via `fastmcp run`
    print("Data loaded. Run the server using:", file=sys.stderr)
    print("fastmcp run mcp_server.app:mcp_server --transport sse --host 0.0.0.0 --port 8080", file=sys.stderr)

    # Keep the script alive if needed, or just exit after loading? 
    # Depending on how `fastmcp run` works, it might import and run,
    # or it might expect this script to exit after loading is done.
    # Let's assume it just needs the loading done.
    pass 

    # --- Old code removed --- 
    # host = "0.0.0.0"  
    # port = int(os.environ.get("PORT", 8080))
    # sse_endpoint = "/sse"
    # try:
    #     print(f"Starting MCP server on SSE {host}:{port}{sse_endpoint}...", file=sys.stderr)
    #     mcp_server.run(
    #         transport="sse",
    #         host=host,
    #         port=port,
    #         endpoint=sse_endpoint
    #     )
    # except KeyboardInterrupt:
    #     print("\nServer stopped by user.", file=sys.stderr)
    # except Exception as e:
    #     print(f"An error occurred: {e}", file=sys.stderr)
    #     traceback.print_exc()
