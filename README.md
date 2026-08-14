[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/obsidian-notes-rag)](https://pypi.org/project/obsidian-notes-rag/)

# obsidian-notes-rag

MCP server and CLI for semantic search over your Obsidian vault — or any folder of linked markdown: an OKF knowledge bundle, a repo's docs tree, a wiki export. The CLI installs as `obsidian-rag` and as **`okf-search`** — same tool, use whichever name fits the corpus. Generates embeddings with OpenAI, Ollama, or LM Studio. Stores vectors locally in sqlite-vec (~200KB, no telemetry, no network calls). Nothing requires Obsidian itself: point `--root` at any markdown directory and both the semantic index and the link graph work the same.

## What it does

Search your notes by meaning, not just keywords:

```bash
obsidian-rag search "project architecture decisions" -n 5
obsidian-rag similar "Projects/Platform Hub.md"
obsidian-rag context "Daily Notes/2026-02-14.md"
```

As an MCP server, it gives any compatible AI assistant the same capabilities — searching your notes, finding related content, and pulling context during conversations.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (for running and installing)
- One of: `OPENAI_API_KEY`, [Ollama](https://ollama.ai/), or [LM Studio](https://lmstudio.ai/) for embeddings

## Setup

### 1. Run the setup wizard

```bash
uvx obsidian-notes-rag setup
```

This creates a config at `~/.config/obsidian-notes-rag/config.toml` with your vault path, embedding provider, and API key.

### 2. Build the index

```bash
uvx obsidian-notes-rag index
```

Parses your markdown files, chunks them by heading structure (using [Chonkie](https://github.com/chonkie-ai/chonkie) RecursiveChunker), generates embeddings, and stores everything in a local SQLite database.

### 3. Connect to an MCP client

Works with any MCP-compatible client. Examples:

**Claude Code:**

```bash
claude mcp add -s user obsidian-notes-rag -- uvx obsidian-notes-rag serve
```

**Claude Desktop, Cursor, Windsurf, etc. (JSON config):**

Add to your client's MCP config file (e.g. `~/Library/Application Support/Claude/claude_desktop_config.json` for Claude Desktop on macOS):

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

### 4. Install the CLI (optional)

If you want `obsidian-rag` available as a standalone command:

```bash
uv tool install obsidian-notes-rag
```

This installs both `obsidian-rag` and `obsidian-notes-rag` to `~/.local/bin/`.

### Using the CLI with AI coding assistants

Instead of running the MCP server, you can have your AI assistant call the CLI directly via shell commands. This avoids loading MCP tool definitions into the context window, freeing up tokens for your actual work.

To do this, create a rule or skill that tells your assistant when and how to use the CLI:

- **Claude Code**: Create a [skill](https://docs.anthropic.com/en/docs/claude-code/skills) with CLI usage instructions
- **Cursor**: Add a [rule](https://docs.cursor.com/context/rules) to `.cursor/rules/`
- **Windsurf**: Add a [rule](https://docs.windsurf.com/windsurf/memories#rules) to `.windsurfrules`

The rule should describe when to use each command (`search`, `similar`, `context`) and any project-specific conventions. This gives the assistant enough context to run the right CLI commands without the overhead of an MCP connection.

## CLI Reference

```bash
# Search
obsidian-rag search "query"                  # semantic search
obsidian-rag search "standup" --type daily   # filter by note type
obsidian-rag search "design" -n 10           # more results
obsidian-rag search "design" --expand 1      # + notes linked from the hits
obsidian-rag search "design" -e 2 --expand-limit 15  # deeper graph context

# Explore
obsidian-rag similar "Path/To/Note.md"       # find related notes (by meaning)
obsidian-rag context "Path/To/Note.md"       # note + links/backlinks + similar
obsidian-rag graph "Path/To/Note.md"         # link-graph neighborhood
obsidian-rag graph "Path/To/Note.md" -n 2    # traverse two hops

# Index
obsidian-rag index                            # re-index vault
obsidian-rag index --clear                    # rebuild from scratch
obsidian-rag index --path-filter "Daily Notes/"  # index subset

# Info
obsidian-rag stats                            # show index size

# Second collections (e.g. a project's docs bundle) — pass both flags on every command
obsidian-rag --vault ~/proj/internal --data ~/rag-data/proj index
obsidian-rag --vault ~/proj/internal --data ~/rag-data/proj search "query" -e 1

# Services
obsidian-rag serve                            # start MCP server
obsidian-rag watch                            # watch for changes, auto-reindex
obsidian-rag install-service                  # macOS launchd auto-start
obsidian-rag uninstall-service                # remove service
obsidian-rag service-status                   # check service status
```

## MCP Tools

Once connected, your AI assistant has access to:

| Tool | What it does |
|------|--------------|
| `search_notes` | Find notes matching a query; `expand` adds link-graph neighbors |
| `get_similar` | Find notes similar to a given note |
| `get_note_context` | Get a note with its links, backlinks, and similar notes |
| `get_note_graph` | Get a note's link-graph neighborhood |
| `get_stats` | Show index statistics |
| `reindex` | Rebuild the index (chunks and link graph) |

## Graph-Aware Retrieval

Your vault's links already form a knowledge graph. Indexing extracts every
wikilink and markdown link between notes into a local edge table (no LLM
involved), and `--expand` / `get_note_graph` traverse it at query time:
a vector search finds the notes that *sound like* your query, then expansion
follows real links outward for the connected context — the entity-anchored
retrieval that graph-RAG systems promise, at zero extra indexing cost.
Traversal is breadth-first (both links and backlinks), never revisits a
note, and reports which note bridged each hop.

## Keeping the Index Fresh

**Manual:** `obsidian-rag index`

**Auto-reindex on file changes:** `obsidian-rag watch` (run in a terminal or background)

**macOS background service:** `obsidian-rag install-service` (starts on login, appears in System Settings > Login Items)

## Using Ollama (local, no API key)

```bash
ollama pull nomic-embed-text
obsidian-rag --provider ollama index
```

## Using LM Studio (local, no API key)

Load an embedding model in LM Studio, then:

```bash
obsidian-rag --provider lmstudio index
```

## Configuration

The setup wizard writes to `~/.config/obsidian-notes-rag/config.toml`. You can also override with environment variables:

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `OBSIDIAN_RAG_PROVIDER` | `openai` (default), `ollama`, or `lmstudio` |
| `OBSIDIAN_RAG_VAULT` | Path to Obsidian vault |
| `OBSIDIAN_RAG_DATA` | Index storage path (default: platform-specific) |
| `OBSIDIAN_RAG_OLLAMA_URL` | Ollama URL (default: `http://localhost:11434`) |
| `OBSIDIAN_RAG_LMSTUDIO_URL` | LM Studio URL (default: `http://localhost:1234`) |
| `OBSIDIAN_RAG_MODEL` | Override embedding model |

## How it works

1. Parses markdown files, strips YAML frontmatter
2. Chunks content using Chonkie's RecursiveChunker (splits by headings > paragraphs > lines > sentences, max 1500 tokens per chunk)
3. Generates embeddings via your chosen provider
4. Stores metadata in SQLite, vectors in sqlite-vec (KNN search via vec0 virtual tables)
5. MCP server and CLI both query the same local database

## Upgrading

If you installed the CLI with `uv tool install`, upgrade with:

```bash
uv tool upgrade obsidian-notes-rag
```

If you use `uvx` to run commands or the MCP server, it automatically uses the latest version.

### Upgrading to v1.0.0

v1.0.0 replaces ChromaDB with sqlite-vec. After upgrading, rebuild your index:

```bash
obsidian-rag index --clear
```

The old ChromaDB data at `~/.local/share/obsidian-notes-rag/` (or your configured path) can be deleted.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup.

## Support

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black)](https://buymeacoffee.com/ernestkoe)

## License

MIT
