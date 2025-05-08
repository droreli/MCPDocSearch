import re
import pickle  # For caching
import os  # For stat, getenv
import sys
from typing import List, Dict, Union, Any
import numpy as np
from pathlib import Path

from mcp_server.config import STORAGE_DIR as DEFAULT_STORAGE_DIR
from mcp_server.app import embedding_model

# --- Determine Storage Path ---
_volume_mount_path = os.getenv("MCP_STORAGE_PATH", "/data/storage") # Default for Fly.io
STORAGE_DIR = Path(_volume_mount_path)
CACHE_FILE_PATH = STORAGE_DIR / "document_chunks_cache.pkl"

print(f"Using storage directory: {STORAGE_DIR}", file=sys.stderr)
print(f"Using cache file path: {CACHE_FILE_PATH}", file=sys.stderr)

# Simple regex to find markdown headings (##, ###, etc.)
HEADING_RE = re.compile(r"^(#{2,4})\s+(.*)")
# Simple regex to find Source: URL lines
SOURCE_RE = re.compile(r"Source:\s*(https?://\S+)")

# In-memory storage for chunks - now includes embeddings
# In-memory storage for chunks, including embeddings
document_chunks: List[Dict[str, Union[str, np.ndarray]]] = []


def parse_markdown_to_chunks(filename: str, content: str) -> List[Dict[str, str]]:
    """
    Parses markdown content into chunks based on headings (## and deeper).

    Each chunk includes the heading, the content following it, and any
    detected Source: URL immediately after the heading.
    """
    chunks = []
    lines = content.splitlines()
    # Default for content before the first heading
    current_heading = "Introduction"
    current_content: List[str] = []
    # Use Union explicitly if Optional was removed from imports
    current_source_url: Union[str, None] = None
    heading_level = 1  # Default heading level

    for i, line in enumerate(lines):
        heading_match = HEADING_RE.match(line)
        source_match = SOURCE_RE.match(line)

        if heading_match:
            # If we have content for the previous heading, save it as a chunk
            if current_content:
                content_str = "\n".join(current_content).strip()
                if content_str:  # Only add if there's actual content
                    chunks.append(
                        {
                            "filename": filename,
                            "heading": current_heading,
                            "content": content_str,
                            "content_lower": content_str.lower(),
                            "heading_lower": current_heading.lower(),
                            "source_url": current_source_url or "",
                            "level": str(heading_level),
                        }
                    )

            # Start a new chunk
            # '#' count indicates level
            heading_level = len(heading_match.group(1))
            current_heading = heading_match.group(2).strip()
            current_content = []
            current_source_url = None  # Reset source URL for the new section

            # Check the *next* line for a potential Source: URL
            if i + 1 < len(lines):
                next_line_source_match = SOURCE_RE.match(lines[i + 1])
                if next_line_source_match:
                    current_source_url = next_line_source_match.group(1)

        elif source_match and current_heading == "Introduction":
            # Capture Source: URL if it appears before the first heading
            current_source_url = source_match.group(1)
        # Avoid adding the Source: line itself to content
        elif not source_match:
            current_content.append(line)

    # Add the last chunk
    if current_content:
        content_str = "\n".join(current_content).strip()
        if content_str:
            chunks.append(
                {
                    "filename": filename,
                    "heading": current_heading,
                    "content": content_str,
                    "content_lower": content_str.lower(),
                    "heading_lower": current_heading.lower(),
                    "source_url": current_source_url or "",
                    "level": str(heading_level),
                }
            )

    return chunks


def load_and_chunk_documents():
    """
    Scans the STORAGE_DIR, reads .md files, parses them into chunks,
    generates embeddings, and stores them in the global document_chunks list.
    Uses a cache file to speed up subsequent loads.
    """
    global document_chunks

    # --- Get current state of markdown files ---
    current_file_metadata = {}
    if STORAGE_DIR.exists() and STORAGE_DIR.is_dir():
        for file_path in STORAGE_DIR.glob("*.md"):
            try:
                # Get modification time
                mtime = os.path.getmtime(file_path)
                current_file_metadata[file_path.name] = mtime
            except OSError as e:
                print(
                    f"Warning: Could not get metadata for {file_path.name}: {e}",
                    file=sys.stderr,
                )
                # Decide how to handle - skip file? invalidate cache?
                # For now, assume cache should be invalidated if we can't check
                # all files.
                current_file_metadata = None  # Signal error
                break
    else:
        # Signal error / directory not found
        current_file_metadata = None

    # --- Try loading from cache and validate metadata ---
    cache_valid = False
    if current_file_metadata is not None and CACHE_FILE_PATH.exists():
        try:
            with open(CACHE_FILE_PATH, "rb") as f_cache:
                # Expecting a tuple: (metadata_dict, chunks_list)
                cached_metadata, cached_chunks = pickle.load(f_cache)

                # Validate cache structure and compare metadata
                if isinstance(cached_metadata, dict) and isinstance(
                    cached_chunks, list
                ):
                    if cached_metadata == current_file_metadata:
                        document_chunks = cached_chunks
                        cache_valid = True
                    else:
                        print(
                            "Cache metadata mismatch. Source files changed. "
                            "Regenerating cache.",
                            file=sys.stderr,
                        )
                else:
                    print(
                        "Warning: Cache file format is invalid. " "Regenerating cache.",
                        file=sys.stderr,
                    )
        except Exception as e:
            print(
                f"Warning: Failed to load or validate cache ({e}). "
                "Regenerating cache.",
                file=sys.stderr,
            )
            # Don't try deletion inside the exception handler for loading

        # --- Delete invalid/outdated cache file ---
        if not cache_valid and CACHE_FILE_PATH.exists():
            # Attempting to delete invalid/outdated cache file...
            try:
                CACHE_FILE_PATH.unlink(missing_ok=True)
            except OSError as unlink_e:
                print(
                    f"Warning: Could not delete cache file: {unlink_e}",
                    file=sys.stderr,
                )

    # --- If cache was invalid or didn't exist, process files ---
    if not cache_valid:
        print(
            "Processing documents and generating embeddings...",
            file=sys.stderr,
        )
        if not STORAGE_DIR.exists() or not STORAGE_DIR.is_dir():
            print(
                f"Error: Storage directory '{STORAGE_DIR}' not found or is "
                "not a directory.",
                file=sys.stderr,
            )
            document_chunks = []
            return

        loaded_chunks = []
        for file_path in STORAGE_DIR.glob("*.md"):
            try:
                content = file_path.read_text(encoding="utf-8")
                file_chunks = parse_markdown_to_chunks(file_path.name, content)
                loaded_chunks.extend(file_chunks)
            except Exception as e:
                # Rely on the global import of sys
                print(
                    f"Error processing file {file_path.name}: {e}",
                    file=sys.stderr,
                )

        # --- Generate Embeddings ---
        if loaded_chunks:
            # Prepare texts for embedding (e.g., combine heading and content)
            # Using just content for now. Consider heading+content?
            texts_to_embed = [chunk["content"] for chunk in loaded_chunks]
            try:
                # Encode all texts at once for efficiency
                # Set show_progress_bar=True for CLI progress feedback
                print(
                    f"Generating embeddings for {len(texts_to_embed)} chunks...",
                    file=sys.stderr,
                )
                embeddings = embedding_model.encode(
                    texts_to_embed,
                    show_progress_bar=True,  # Enable progress bar
                )
                # Add embeddings back to the chunk dictionaries
                for i, chunk in enumerate(loaded_chunks):
                    chunk["embedding"] = embeddings[i]
            except Exception as e:
                print(f"Error generating embeddings: {e}", file=sys.stderr)
                # Handle error: proceed without embeddings if encoding fails.
                for chunk in loaded_chunks:
                    # Indicate embedding failed/missing
                    chunk["embedding"] = None

        # Update the global list *after* potential embedding
        document_chunks = loaded_chunks

        # --- Save the processed data and metadata to cache ---
        # Only save if processing was successful and we have metadata
        if document_chunks and current_file_metadata is not None:
            print(
                f"Saving {len(document_chunks)} chunks and embeddings to "
                f"cache: {CACHE_FILE_PATH}",
                file=sys.stderr,
            )
            try:
                # Ensure parent directory exists
                STORAGE_DIR.mkdir(parents=True, exist_ok=True)
                # Data to cache: tuple of (metadata, chunks)
                # Using Any for chunk dict value type (mixed str, np.ndarray)
                data_to_cache: tuple[Dict[str, float], List[Dict[str, Any]]] = (
                    current_file_metadata,
                    document_chunks,
                )
                with open(CACHE_FILE_PATH, "wb") as f_cache:
                    pickle.dump(data_to_cache, f_cache)
            except Exception as e:
                print(
                    f"Warning: Failed to save cache to {CACHE_FILE_PATH}: {e}",
                    file=sys.stderr,
                )
        elif not document_chunks:
            print(
                "No document chunks loaded or generated, cache not saved.",
                file=sys.stderr,
            )
        else:  # current_file_metadata was None
            print(
                "Could not read current file metadata, cache not saved.",
                file=sys.stderr,
            )


def get_available_documents() -> List[str]:
    """Returns a list of unique filenames that have been loaded."""
    return sorted(list(set(chunk["filename"] for chunk in document_chunks)))


def get_document_headings(filename: str) -> List[Dict[str, Union[int, str]]]:
    """Returns the heading structure for a specific document."""
    headings = []
    seen_headings = set()
    for chunk in document_chunks:
        if chunk["filename"] == filename:
            # Use heading text + level to define uniqueness for this list
            heading_key = (chunk["heading"], chunk["level"])
            if heading_key not in seen_headings:
                headings.append(
                    {
                        # Convert back to int for output
                        "level": int(chunk["level"]),
                        "title": chunk["heading"],
                    }
                )
                seen_headings.add(heading_key)
    # Note: This simple approach doesn't guarantee perfect hierarchical order
    # if headings were out of order in the source, but gives a flat list.
    return headings


# Expose the chunks for the search module
def get_all_chunks() -> List[Dict[str, Union[str, np.ndarray]]]:
    return document_chunks
