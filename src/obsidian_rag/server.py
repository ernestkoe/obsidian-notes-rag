"""MCP server for mcp-obsidianRAG with semantic search tools."""

from __future__ import annotations

import os
from typing import Optional

from mcp.server.fastmcp import FastMCP

from .config import load_config, Config
from .indexer import create_embedder, Embedder, VaultIndexer
from .store import VectorStore

# Create MCP server
mcp = FastMCP("obsidian-rag")

# Global instances (lazy initialized)
_config: Optional[Config] = None
_embedder: Optional[Embedder] = None
_store: Optional[VectorStore] = None


def get_config() -> Config:
    """Get or create config instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def get_embedder() -> Embedder:
    """Get or create embedder instance."""
    global _embedder
    if _embedder is None:
        config = get_config()
        # Set API key in environment if configured
        if config.provider == "openai" and config.openai_api_key:
            os.environ["OPENAI_API_KEY"] = config.openai_api_key
        _embedder = create_embedder(
            provider=config.provider,
            model=config.openai_model if config.provider == "openai" else config.ollama_model,
            base_url=config.ollama_url,
        )
    return _embedder


def get_store() -> VectorStore:
    """Get or create store instance."""
    global _store
    if _store is None:
        config = get_config()
        _store = VectorStore(data_path=config.get_data_path())
    return _store


@mcp.tool()
def search_notes(
    query: str,
    limit: int = 10,
    note_type: Optional[str] = None
) -> list[dict]:
    """Search notes using semantic similarity.

    Args:
        query: Search query text
        limit: Maximum number of results (default: 10)
        note_type: Optional filter - "daily" or "note"

    Returns:
        List of matching notes with content, file path, and similarity score
    """
    embedder = get_embedder()
    store = get_store()

    # Generate query embedding
    query_embedding = embedder.embed(query)

    # Build filter
    where = {"type": note_type} if note_type else None

    # Search
    results = store.search(query_embedding, limit=limit, where=where)

    # Format results
    return [
        {
            "file_path": r["metadata"]["file_path"],
            "heading": r["metadata"].get("heading") or None,
            "content": r["content"][:500] if len(r["content"]) > 500 else r["content"],
            "similarity": round(1 - r["distance"], 3),
            "type": r["metadata"].get("type", "note")
        }
        for r in results
    ]


@mcp.tool()
def get_similar(note_path: str, limit: int = 5) -> list[dict]:
    """Find notes similar to the given note.

    Args:
        note_path: Path to the note (relative to vault root)
        limit: Number of similar notes to return (default: 5)

    Returns:
        List of similar notes with content preview and similarity score
    """
    embedder = get_embedder()
    store = get_store()

    # Get all chunks from this note by searching with file_path filter
    # First, we need to get the note content to embed
    results = store.search(
        query_embedding=embedder.embed(note_path),  # Dummy search to get note
        limit=50,
        where={"file_path": note_path}
    )

    if not results:
        return [{"error": f"Note not found: {note_path}"}]

    # Combine content from all chunks of this note
    note_content = "\n\n".join(r["content"] for r in results)

    # Generate embedding for the note content
    note_embedding = embedder.embed(note_content[:8000])  # Limit for embedding

    # Search for similar notes, excluding the source note
    all_results = store.search(note_embedding, limit=limit + 10)

    # Filter out chunks from the same file
    similar = [
        r for r in all_results
        if r["metadata"]["file_path"] != note_path
    ][:limit]

    return [
        {
            "file_path": r["metadata"]["file_path"],
            "heading": r["metadata"].get("heading") or None,
            "preview": r["content"][:200] if len(r["content"]) > 200 else r["content"],
            "similarity": round(1 - r["distance"], 3)
        }
        for r in similar
    ]


@mcp.tool()
def get_note_context(note_path: str, limit: int = 5) -> dict:
    """Get a note and its related context.

    Args:
        note_path: Path to the note (relative to vault root)
        limit: Number of similar notes to include (default: 5)

    Returns:
        Note content and list of similar notes for context
    """
    embedder = get_embedder()
    store = get_store()

    # Search for chunks from this specific file
    # Use a generic embedding and filter by file_path
    dummy_embedding = embedder.embed(note_path)
    results = store.search(
        query_embedding=dummy_embedding,
        limit=50,
        where={"file_path": note_path}
    )

    if not results:
        return {"error": f"Note not found: {note_path}"}

    # Combine chunks to get full note content
    note_content = "\n\n".join(r["content"] for r in results)

    # Get similar notes
    similar = get_similar(note_path, limit=limit)

    return {
        "file_path": note_path,
        "content": note_content,
        "similar_notes": similar if not (similar and "error" in similar[0]) else []
    }


@mcp.tool()
def get_stats() -> dict:
    """Get index statistics.

    Returns:
        Statistics about the indexed notes collection
    """
    store = get_store()
    return store.get_stats()


@mcp.tool()
def reindex(clear: bool = False, path_filter: Optional[str] = None) -> dict:
    """Re-index the Obsidian vault.

    Args:
        clear: If True, clear existing index before re-indexing (default: False)
        path_filter: Optional path prefix to limit indexing (e.g., "Daily Notes/")

    Returns:
        Statistics about the indexing operation
    """
    config = get_config()
    embedder = get_embedder()
    store = get_store()

    if not config.vault_path:
        return {"error": "No vault path configured. Run 'obsidian-rag setup' first."}

    indexer = VaultIndexer(vault_path=config.vault_path, embedder=embedder)

    if clear:
        store.clear()

    # Get files to index
    files = list(indexer.iter_markdown_files())

    # Apply path filter if specified
    if path_filter:
        files = [f for f in files if str(f.relative_to(indexer.vault_path)).startswith(path_filter)]

    # Index files
    chunk_count = 0
    file_count = 0
    errors = []
    batch_chunks = []
    batch_embeddings = []
    batch_size = 50

    for file_path in files:
        try:
            for chunk, embedding in indexer.index_file(file_path):
                batch_chunks.append(chunk)
                batch_embeddings.append(embedding)
                chunk_count += 1

                if len(batch_chunks) >= batch_size:
                    store.upsert_batch(batch_chunks, batch_embeddings)
                    batch_chunks = []
                    batch_embeddings = []

            file_count += 1
        except Exception as e:
            errors.append({"file": str(file_path), "error": str(e)})

    # Insert remaining
    if batch_chunks:
        store.upsert_batch(batch_chunks, batch_embeddings)

    return {
        "files_indexed": file_count,
        "chunks_created": chunk_count,
        "total_in_store": store.get_stats()["count"],
        "errors": errors if errors else None,
        "path_filter": path_filter,
        "cleared": clear
    }


def run_server():
    """Run the MCP server."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    run_server()
