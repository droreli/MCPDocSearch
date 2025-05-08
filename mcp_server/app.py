import torch  # Import torch to check for GPU
import mcp.types as types
from fastmcp import FastMCP
from sentence_transformers import SentenceTransformer
import os
import sys

# --- Configuration ---
# Determine the device
# Forcing CPU for now, remove or adjust logic if GPU/MPS is desired and available in deployment
device = "cpu"
# device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
# Specify the embedding model
EMBEDDING_MODEL_NAME = 'multi-qa-mpnet-base-dot-v1'

# --- Initialize Components ---
# Load the embedding model (this happens when the module is imported)
try:
    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...", file=sys.stderr)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
    print(f"Embedding model '{EMBEDDING_MODEL_NAME}' loaded on device: {embedding_model.device}", file=sys.stderr)
except Exception as e:
    print(f"FATAL: Failed to load embedding model '{EMBEDDING_MODEL_NAME}': {e}", file=sys.stderr)
    # Optionally, exit or raise a more specific error if the model is critical
    sys.exit(1) # Exit if model fails to load

# Create the FastMCP server instance (this also happens at import time)
app = FastMCP(
    title="MCP Document Query Server",
    description="Search documentation chunks using semantic embeddings.",
    version="0.1.0",
)

# --- Load Data (COMMENTED OUT - Load manually or via different mechanism) ---
# from mcp_server.data_loader import load_and_chunk_documents
# print("Triggering document loading from app.py...", file=sys.stderr)
# load_and_chunk_documents()
# print("Document loading triggered from app.py complete.", file=sys.stderr)

# --- Ensure Tools are Registered ---
import mcp_server.mcp_tools # Ensure tools are registered

# --- MCP Server Instance (Duplicate Removed) ---
# The instance named mcp_server created above is the one we want.
# Remove the duplicate definition below if it exists.
# mcp_server = FastMCP(
#     name="doc-query-server",
#     version="0.1.0",
#     capabilities=types.ServerCapabilities(
#         tools=types.ToolsCapability(listChanged=False)
#     ),
# )

# Make mcp_server an alias to app for any internal references if necessary,
# though ideally all internal references would also use 'app'.
# For now, this ensures that if any other part of your code specifically expects
# 'mcp_server', it will still find the application instance.
mcp_server = app
