# obsidian-rag

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MCP server for semantic search over your Obsidian vault. Uses OpenAI embeddings by default (or Ollama for local processing) with ChromaDB for vector storage.

## What it does

Ask natural language questions about your notes:
- "What did I write about project planning?"
- "Find notes similar to my meeting notes from last week"
- "What's in my daily notes about the API refactor?"

## Requirements

- Python 3.11+
- `OPENAI_API_KEY` environment variable (or [Ollama](https://ollama.ai/) for local embeddings)

## Quick Start

The easiest way to get started is with [uvx](https://docs.astral.sh/uv/guides/tools/) (no installation required):

```bash
# Run the setup wizard
uvx obsidian-notes-rag setup
```

### Add to Claude Code (CLI)

```bash
claude mcp add -s user obsidian-notes-rag -- uvx obsidian-notes-rag serve
```

### Add to Claude Desktop (JSON config)

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "obsidian-notes-rag": {
      "command": "uvx",
      "args": ["obsidian-notes-rag", "serve"]
    }
  }
}
```

### Alternative: Clone and install

```bash
git clone https://github.com/ernestkoe/obsidian-rag.git
cd obsidian-rag
uv sync

uv run obsidian-notes-rag setup
claude mcp add -s user obsidian-notes-rag -- uv run --directory /path/to/obsidian-rag obsidian-notes-rag serve
```

The setup wizard will:
1. Ask for your embedding provider (OpenAI or Ollama)
2. Configure your API key (for OpenAI)
3. Set your Obsidian vault path
4. Choose where to store the search index
5. Optionally run the initial indexing

### Manual Setup (alternative)

```bash
# Set your API key and index directly
export OPENAI_API_KEY=sk-...
uv run obsidian-notes-rag index --vault /path/to/your/vault
```

### Using Ollama (local, offline)

```bash
# Install Ollama and pull the embedding model
ollama pull nomic-embed-text

# Run setup with Ollama, or index directly:
uv run obsidian-notes-rag --provider ollama index --vault /path/to/your/vault
```

## MCP Tools

Once connected, these tools are available to Claude:

| Tool | What it does |
|------|--------------|
| `search_notes` | Find notes matching a query |
| `get_similar` | Find notes similar to a given note |
| `get_note_context` | Get a note with related context |
| `get_stats` | Show index statistics |
| `reindex` | Update the index |

## Keeping the Index Fresh

### Option 1: Manual reindex

```bash
uv run obsidian-notes-rag index
```

### Option 2: Watch for changes

```bash
uv run obsidian-notes-rag watch
```

### Option 3: Auto-start on login (macOS)

```bash
uv run obsidian-notes-rag install-service
```

## CLI Reference

```bash
obsidian-notes-rag setup                # Interactive setup wizard
obsidian-notes-rag serve                # Start MCP server (for Claude Code)
obsidian-notes-rag index [--clear]      # Index vault (--clear to rebuild)
obsidian-notes-rag search "query"       # Search from command line
obsidian-notes-rag watch                # Watch for file changes
obsidian-notes-rag stats                # Show index stats
obsidian-notes-rag install-service      # Install macOS launchd service
obsidian-notes-rag uninstall-service    # Remove service
obsidian-notes-rag service-status       # Check service status
```

## Configuration

Set your vault path and provider via CLI options or environment variables:

```bash
# CLI options
uv run obsidian-notes-rag --vault /path/to/vault index
uv run obsidian-notes-rag --provider ollama index

# Environment variables
export OBSIDIAN_RAG_VAULT=/path/to/vault
export OBSIDIAN_RAG_PROVIDER=ollama  # or "openai" (default)
```

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key (required for default provider) |
| `OBSIDIAN_RAG_PROVIDER` | Embedding provider: `openai` (default) or `ollama` |
| `OBSIDIAN_RAG_VAULT` | Path to Obsidian vault |
| `OBSIDIAN_RAG_DATA` | Where to store the index (default: `./data`) |
| `OBSIDIAN_RAG_OLLAMA_URL` | Ollama API URL (default: `http://localhost:11434`) |
| `OBSIDIAN_RAG_MODEL` | Override embedding model |

## How it works

1. Parses your markdown files and splits them by headings
2. Generates embeddings using OpenAI API (or Ollama for local processing)
3. Stores vectors in ChromaDB (local, persistent)
4. MCP server provides semantic search to Claude

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup.

## License

MIT
