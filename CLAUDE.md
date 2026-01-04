# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

obsidian-rag is an MCP (Model Context Protocol) server that provides semantic search over Obsidian notes. It uses OpenAI embeddings by default (or Ollama for local processing) with ChromaDB for vector storage.

## Commands

```bash
# Install dependencies
uv sync --dev

# Run tests
.venv/bin/python -m pytest -v

# Type checking
.venv/bin/python -m pyright

# Interactive setup wizard
uv run obsidian-notes-rag setup

# Index vault (manual refresh)
uv run obsidian-notes-rag index

# Run the MCP server (stdio transport)
uv run obsidian-notes-rag serve

# Watch vault for changes
uv run obsidian-notes-rag watch

# Search from CLI
uv run obsidian-notes-rag search "query"
```

## Architecture

### Data Flow

```
Obsidian Vault → VaultIndexer → Embedder (OpenAI/Ollama) → VectorStore (ChromaDB)
                                                                  ↓
MCP Client ← FastMCP Server ← search_notes/get_similar/etc.
```

### Key Components (src/obsidian_rag/)

- **config.py**: `Config` dataclass, `load_config()`/`save_config()` for TOML config file, cross-platform paths via `platformdirs`
- **indexer.py**: `VaultIndexer` scans markdown files, `chunk_by_heading()` splits content by `##`/`###` headings, `OpenAIEmbedder` and `OllamaEmbedder` generate embeddings, `create_embedder()` factory selects provider
- **store.py**: `VectorStore` wraps ChromaDB with cosine similarity search, handles upsert/delete by file path
- **server.py**: FastMCP server exposing 5 tools: `search_notes`, `get_similar`, `get_note_context`, `get_stats`, `reindex`
- **watcher.py**: `VaultWatcher` uses watchdog with debouncing (default 2s) to incrementally re-index on file changes
- **cli.py**: Click-based CLI with `setup` wizard, `--provider` option, commands for indexing, searching, watching, and service management

### Chunking Strategy

Files are split by `##` and `###` headings. Chunks smaller than `min_chunk_size` (default 100 chars) merge with the previous chunk. Files without headings become a single chunk.

### Metadata

Each chunk stores: `file_path`, `heading`, `heading_level`, `type` ("daily" if path starts with "Daily Notes/", else "note"), and tags from YAML frontmatter.

## Configuration

Config file location (created by `setup` command):
- macOS/Linux: `~/.config/obsidian-rag/config.toml`
- Windows: `%APPDATA%/obsidian-rag/config.toml`

Environment variables (override config file):
- `OPENAI_API_KEY` - OpenAI API key (required for default provider)
- `OBSIDIAN_RAG_PROVIDER` - `openai` (default) or `ollama`
- `OBSIDIAN_RAG_VAULT` - Path to Obsidian vault
- `OBSIDIAN_RAG_DATA` - ChromaDB storage path
- `OBSIDIAN_RAG_OLLAMA_URL` - Ollama API (default: `http://localhost:11434`)
- `OBSIDIAN_RAG_MODEL` - Override embedding model

## Testing

Tests are in `tests/`. Current coverage focuses on `test_indexer.py` for frontmatter parsing and chunking logic.

```bash
# Run a specific test
uv run pytest tests/test_indexer.py::TestChunkByHeading::test_multiple_headings -v
```
