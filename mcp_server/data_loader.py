import re
import pickle  # For caching
import os  # For stat, getenv
import sys
from typing import List, Dict, Union, Any
import numpy as np
from pathlib import Path

# Import config variables and the embedding model
from mcp_server.config import STORAGE_DIR as DEFAULT_STORAGE_DIR # This will be ./storage
from mcp_server.app import embedding_model

# --- Determine Storage Path ---
# Default to ./storage which will be /app/storage in Docker
# Allow override via MCP_STORAGE_PATH for flexibility if ever needed outside Docker
_storage_path_str = os.getenv("MCP_STORAGE_PATH", ".") 
STORAGE_DIR = Path(_storage_path_str) / "storage"
CACHE_FILE_PATH = STORAGE_DIR / "document_chunks_cache.pkl"

print(f"Using storage directory: {STORAGE_DIR.resolve()}", file=sys.stderr)
print(f"Using cache file path: {CACHE_FILE_PATH.resolve()}", file=sys.stderr)

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
    current_headings = ["Default Heading"] * 3  # H2, H3, H4
    current_content = []
    source_url = None

    lines = content.splitlines()
    for i, line in enumerate(lines):
        source_match = SOURCE_RE.match(line)
        if source_match:
            source_url = source_match.group(1)
            # Check if the next line is a heading to associate the source with it
            if i + 1 < len(lines):
                next_line_heading_match = HEADING_RE.match(lines[i+1])
                if next_line_heading_match:
                    continue # Source will be processed with its heading
            # If not followed by a heading, this source applies to prior content
            # Or it's a general source for the file (less ideal)

        heading_match = HEADING_RE.match(line)
        if heading_match:
            if current_content:  # Save previous chunk
                chunks.append({
                    "filename": filename,
                    "heading_h2": current_headings[0],
                    "heading_h3": current_headings[1],
                    "heading_h4": current_headings[2],
                    "content": "\n".join(current_content).strip(),
                    "source_url": source_url or "Not specified"
                })
                current_content = []
                # Reset source_url if it was tied to the content just saved
                # and not a heading-associated source

            level = len(heading_match.group(1)) - 2  # 0 for H2, 1 for H3, 2 for H4
            heading_text = heading_match.group(2).strip()
            if 0 <= level < 3:
                current_headings[level] = heading_text
                for l_idx in range(level + 1, 3):
                    current_headings[l_idx] = "Default Heading" # Reset deeper levels
            # If a source was found right before this heading, associate it
            if i > 0 and SOURCE_RE.match(lines[i-1]):
                 source_url = SOURCE_RE.match(lines[i-1]).group(1)
            else:
                 source_url = "Not specified" # Reset if no immediate preceding source
        else:
            current_content.append(line)

    if current_content:  # Save the last chunk
        chunks.append({
            "filename": filename,
            "heading_h2": current_headings[0],
            "heading_h3": current_headings[1],
            "heading_h4": current_headings[2],
            "content": "\n".join(current_content).strip(),
            "source_url": source_url or "Not specified"
        })
    return chunks


def load_and_chunk_documents():
    """
    Scans the STORAGE_DIR, reads .md files, parses them into chunks,
    generates embeddings, and stores them in the global document_chunks list.
    Uses a cache file to speed up subsequent loads.
    """
    global document_chunks
    document_chunks = []  # Clear existing chunks

    # Check if cache exists and if source files have changed
    cache_valid = False
    cached_metadata = {}
    source_file_metadata = {}

    if CACHE_FILE_PATH.exists():
        try:
            with open(CACHE_FILE_PATH, 'rb') as f:
                cached_data = pickle.load(f)
                # Verify structure of cached_data
                if isinstance(cached_data, dict) and \
                   'metadata' in cached_data and \
                   'chunks' in cached_data and \
                   isinstance(cached_data['metadata'], dict) and \
                   isinstance(cached_data['chunks'], list):
                    
                    cached_metadata = cached_data['metadata']
                    # Pre-load chunks from cache if metadata check passes later
                    # Ensure all expected keys are present in cached chunks
                    potential_cached_chunks = cached_data['chunks']
                    if all(isinstance(chunk, dict) and \
                             all(key in chunk for key in ['filename', 'heading_h2', 'heading_h3', 'heading_h4', 'content', 'embedding', 'source_url']) and \
                             isinstance(chunk['embedding'], np.ndarray) 
                             for chunk in potential_cached_chunks):
                        # Tentatively assume cache is valid, verify metadata next
                        cache_valid = True 
                    else:
                        print("Cache structure invalid (chunk content). Regenerating.", file=sys.stderr)
                        cache_valid = False                   
                else:
                    print("Cache structure invalid (metadata/chunks keys). Regenerating.", file=sys.stderr)
                    cache_valid = False
        except (pickle.UnpicklingError, EOFError, KeyError, TypeError) as e:
            print(f"Error reading cache file ({e}). Regenerating cache.", file=sys.stderr)
            cache_valid = False
            if CACHE_FILE_PATH.exists():
                try:
                    os.remove(CACHE_FILE_PATH)
                    print("Removed corrupted cache file.", file=sys.stderr)
                except OSError as ose:
                    print(f"Error removing corrupted cache file: {ose}", file=sys.stderr)

    # Get metadata for current source files
    if not STORAGE_DIR.exists() or not any(STORAGE_DIR.iterdir()):
        print(f"Error: Storage directory '{STORAGE_DIR.resolve()}' not found or is empty.", file=sys.stderr)
        print("Please ensure it exists and contains markdown files.", file=sys.stderr)
        # If storage dir is missing/empty and cache was invalid/missing, we have no data.
        if not cache_valid or not cached_metadata: # if cache was supposed to be valid, it means no files now is a change
            print("No source documents found and no valid cache. Server will have no data.", file=sys.stderr)
            document_chunks = [] # Ensure it's empty
            return # Exit early if no data can be loaded
        # If cache *was* valid, but dir is now empty, it's a change, so cache becomes invalid.
        elif cache_valid:
            print("Storage directory became empty. Invalidating cache.", file=sys.stderr)
            cache_valid = False 

    for md_file in STORAGE_DIR.glob("*.md"):
        source_file_metadata[md_file.name] = {
            'size': md_file.stat().st_size,
            'mtime': md_file.stat().st_mtime
        }
    
    # Compare metadata
    if cache_valid and cached_metadata == source_file_metadata and source_file_metadata: # also ensure source_file_metadata is not empty
        print("Cache metadata matches. Loading from cache.", file=sys.stderr)
        # Load the pre-validated chunks directly
        document_chunks = cached_data['chunks'] 
    else:
        if not source_file_metadata:
             # This case is hit if STORAGE_DIR exists but is empty of .md files, 
             # and cache was invalid or also empty. We might have loaded an empty cache if it existed.
            if not document_chunks: # If cache was also empty or invalid
                print("No markdown files found in storage and no valid cache. Server will have no data.", file=sys.stderr)
            else: # Cache was loaded but is now considered stale because current .md files are gone
                print("Markdown files previously in cache are no longer present. Re-evaluating (empty).", file=sys.stderr)
                document_chunks = [] # Clear stale cache if files disappeared
            # No need to proceed to embedding if there are no files
            if CACHE_FILE_PATH.exists(): # If an old cache exists, remove it as it's now invalid
                try:
                    os.remove(CACHE_FILE_PATH)
                    print(f"Removed outdated cache file: {CACHE_FILE_PATH}", file=sys.stderr)
                except OSError as e:
                    print(f"Error removing outdated cache file: {e}", file=sys.stderr)
            return

        print("Cache metadata mismatch or cache invalid/empty. Source files may have changed. Regenerating cache.", file=sys.stderr)
        document_chunks = [] # Ensure it's clear before reprocessing
        all_parsed_chunks = []
        for md_file in STORAGE_DIR.glob("*.md"):
            print(f"Processing source file: {md_file.name}", file=sys.stderr)
            with open(md_file, 'r', encoding='utf-8') as f_in:
                content = f_in.read()
            parsed_chunks = parse_markdown_to_chunks(md_file.name, content)
            all_parsed_chunks.extend(parsed_chunks)
        
        if not all_parsed_chunks:
            print("No content chunks parsed from markdown files. Server will have no data.", file=sys.stderr)
            # if an old cache exists, it should be removed as it's based on non-existent/empty new files
            if CACHE_FILE_PATH.exists():
                try:
                    os.remove(CACHE_FILE_PATH)
                    print(f"Removed cache file as no new chunks were parsed: {CACHE_FILE_PATH}", file=sys.stderr)
                except OSError as e:
                    print(f"Error removing cache file: {e}", file=sys.stderr)
            return

        print(f"Processing documents and generating embeddings for {len(all_parsed_chunks)} chunks...", file=sys.stderr)
        contents_to_embed = [chunk['content'] for chunk in all_parsed_chunks]
        embeddings = embedding_model.encode(contents_to_embed, batch_size=32, show_progress_bar=True)

        for i, chunk_data in enumerate(all_parsed_chunks):
            document_chunks.append({
                "filename": chunk_data["filename"],
                "heading_h2": chunk_data["heading_h2"],
                "heading_h3": chunk_data["heading_h3"],
                "heading_h4": chunk_data["heading_h4"],
                "content": chunk_data["content"],
                "source_url": chunk_data["source_url"],
                "embedding": embeddings[i]
            })
        
        # Save to cache
        if document_chunks: # Only save if there are chunks
            try:
                STORAGE_DIR.mkdir(parents=True, exist_ok=True) # Ensure storage dir exists for cache
                with open(CACHE_FILE_PATH, 'wb') as f:
                    pickle.dump({'metadata': source_file_metadata, 'chunks': document_chunks}, f)
                print(f"Saving {len(document_chunks)} chunks and embeddings to cache: {CACHE_FILE_PATH}", file=sys.stderr)
            except Exception as e:
                print(f"Error saving cache to {CACHE_FILE_PATH}: {e}", file=sys.stderr)
        elif CACHE_FILE_PATH.exists(): # If no chunks but old cache exists, remove it
             try:
                os.remove(CACHE_FILE_PATH)
                print(f"Removed cache file as no new chunks were generated: {CACHE_FILE_PATH}", file=sys.stderr)
             except OSError as e:
                 print(f"Error removing old cache file: {e}", file=sys.stderr)


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
            heading_key = (chunk["heading_h2"], chunk["heading_h3"], chunk["heading_h4"])
            if heading_key not in seen_headings:
                headings.append(
                    {
                        # Convert back to int for output
                        "level": int(chunk["level"]),
                        "title": chunk["heading_h2"],
                    }
                )
                seen_headings.add(heading_key)
    # Note: This simple approach doesn't guarantee perfect hierarchical order
    # if headings were out of order in the source, but gives a flat list.
    return headings


# Expose the chunks for the search module
def get_all_chunks() -> List[Dict[str, Union[str, np.ndarray]]]:
    return document_chunks
