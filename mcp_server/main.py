import traceback
import sys
import os # Added for environment variables

# Import the shared FastMCP instance
from mcp_server.app import mcp_server as mcp_app_instance

# Import the data loading function and chunk access function
from mcp_server.data_loader import load_and_chunk_documents, get_all_chunks

# Import the tools module to ensure decorators run and register tools
import mcp_server.mcp_tools  # noqa: F401

# --- Main Execution ---
if __name__ == "__main__":
    # Load documents synchronously before starting the server
    print("Loading documents...", file=sys.stderr)
    load_and_chunk_documents()
    # Print status after loading
    num_chunks = len(get_all_chunks())
    print(f"Document loading complete. {num_chunks} chunks loaded.", file=sys.stderr)

    # Configuration for TCP server for deployment
    host = "0.0.0.0"  # Listen on all available network interfaces
    port = int(os.environ.get("PORT", 8080)) # Use $PORT from environment

    try:
        print(f"Starting MCP server on TCP {host}:{port}...", file=sys.stderr)
        # Replace with correct fastmcp TCP startup
        mcp_app_instance.run(transport="tcp", host=host, port=port)
    except KeyboardInterrupt:
        print("\nServer stopped by user.", file=sys.stderr)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        traceback.print_exc()
