# mcp-obsidianRAG

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

MCP server for semantic search over Obsidian notes using local RAG (Retrieval-Augmented Generation).

## Features

- Semantic search across your Obsidian vault
- ChromaDB-backed vector storage
- Local embeddings via Ollama (nomic-embed-text)
- MCP server for AI assistant integration
- File watcher daemon for auto-indexing

## Requirements

- Python 3.11+
- [Ollama](https://ollama.ai/) with `nomic-embed-text` model
- [uv](https://github.com/astral-sh/uv) (recommended)

## Installation

```bash
# Clone the repository
git clone https://github.com/ernestkoe/mcp-obsidianRAG.git
cd mcp-obsidianRAG

# Install with uv
uv sync

# Pull the embedding model
ollama pull nomic-embed-text
```

## Usage

### Index your vault

```bash
uv run obsidian-rag index
```

### Search notes

```bash
uv run obsidian-rag search "your query"
```

### Watch for changes (daemon)

```bash
uv run obsidian-rag watch
```

### Install as macOS service

```bash
uv run obsidian-rag install-service
```

### View statistics

```bash
uv run obsidian-rag stats
```

## MCP Server

Add to your Claude Code MCP config:

```json
{
  "mcpServers": {
    "obsidian-rag": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/mcp-obsidianRAG", "obsidian-rag-mcp"]
    }
  }
}
```

### Available Tools

| Tool | Description |
|------|-------------|
| `search_notes` | Semantic search with optional type filter |
| `get_similar` | Find notes similar to a given note |
| `get_note_context` | Get note content plus related context |
| `get_stats` | Collection statistics |
| `reindex` | Re-index vault (with optional clear and path filter) |

## Configuration

Default configuration is in `pyproject.toml`:

```toml
[tool.obsidian-rag]
vault_path = "/path/to/your/vault"
data_path = "./data"
ollama_url = "http://localhost:11434"
embedding_model = "nomic-embed-text"
exclude = ["attachments/**", ".obsidian/**", ".trash/**"]
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `OBSIDIAN_RAG_VAULT` | Path to Obsidian vault |
| `OBSIDIAN_RAG_DATA` | Path to vector store data |
| `OBSIDIAN_RAG_OLLAMA_URL` | Ollama API URL |
| `OBSIDIAN_RAG_MODEL` | Embedding model name |

## License

MIT
