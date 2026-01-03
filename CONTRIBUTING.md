# Contributing to mcp-obsidianRAG

## Development Setup

```bash
# Clone the repository
git clone https://github.com/ernestkoe/mcp-obsidianRAG.git
cd mcp-obsidianRAG

# Install with dev dependencies
uv sync --dev

# Pull the embedding model
ollama pull nomic-embed-text
```

## Project Structure

```
src/obsidian_rag/
├── __init__.py      # Package version
├── cli.py           # Click CLI commands
├── server.py        # MCP server (FastMCP)
├── indexer.py       # Markdown parsing, chunking, Ollama embeddings
├── store.py         # ChromaDB vector store wrapper
└── watcher.py       # File watcher daemon (watchdog)
```

## Architecture

### Indexing Pipeline

1. **VaultIndexer** scans markdown files, excludes patterns (`.obsidian/`, etc.)
2. **parse_frontmatter()** extracts YAML frontmatter
3. **chunk_by_heading()** splits content by `##` and `###` headings
4. **OllamaEmbedder** generates embeddings via local Ollama API
5. **VectorStore** persists chunks + embeddings to ChromaDB

### MCP Server

The server exposes 5 tools via FastMCP:
- `search_notes` - Semantic search
- `get_similar` - Find similar notes
- `get_note_context` - Note + related context
- `get_stats` - Index statistics
- `reindex` - Re-index vault

### File Watcher

Uses watchdog to monitor the vault directory:
- Debounces rapid file changes (default 2s)
- Handles create/modify/delete/move events
- Can run as macOS launchd service

## Running Tests

```bash
uv run pytest -v
```

## Type Checking

```bash
uv run pyright
```

## Code Style

- Python 3.11+
- Type hints required
- Docstrings for public functions

## Making Changes

1. Create a feature branch
2. Make changes with tests
3. Ensure `pytest` and `pyright` pass
4. Submit PR with conventional commit message

### Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new search filter
fix: handle empty vault gracefully
docs: update installation instructions
refactor: simplify chunking logic
```

## Release Process

Releases are automated via [release-please](https://github.com/googleapis/release-please). When PRs with conventional commits are merged to `main`, release-please creates a release PR that bumps the version and updates the changelog.

## Environment Variables

For development, you can set these in your shell or a `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `OBSIDIAN_RAG_VAULT` | (hardcoded) | Path to Obsidian vault |
| `OBSIDIAN_RAG_DATA` | (hardcoded) | Path to ChromaDB data |
| `OBSIDIAN_RAG_OLLAMA_URL` | `http://localhost:11434` | Ollama API URL |
| `OBSIDIAN_RAG_MODEL` | `nomic-embed-text` | Embedding model |
| `OBSIDIAN_RAG_DEBOUNCE` | `2.0` | Watcher debounce seconds |

## Key Dependencies

| Package | Purpose |
|---------|---------|
| chromadb | Vector store |
| watchdog | File system monitoring |
| mcp | MCP server framework |
| httpx | HTTP client for Ollama |
| click | CLI framework |
| pyyaml | Frontmatter parsing |
