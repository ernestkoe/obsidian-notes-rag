"""Tests for the MCP server layer.

These tests exercise the real MCP protocol using the SDK's in-process client
(`mcp.client.Client` connected directly to the server object), so they cover
what unit tests on the tool functions alone cannot: that the server imports,
registers its tools, and serializes results the way MCP clients will see them.

Regression context: mcp 2.0.0 removed `mcp.server.fastmcp`, which broke this
server at import time. Nothing in the test suite touched server.py, so CI
stayed green while every fresh install crashed. These tests close that gap.
"""

import json

import pytest

from mcp.client import Client

import obsidian_rag.server as server
from obsidian_rag.config import Config
from obsidian_rag.indexer import IndexerConfig


EXPECTED_TOOLS = {"search_notes", "get_similar", "get_note_context", "get_stats", "reindex"}


class FakeEmbedder:
    """Deterministic embedder that records what it was asked to embed."""

    def __init__(self):
        self.embedded = []

    def embed(self, text, task_type=None):
        self.embedded.append((text, task_type))
        return [0.1, 0.2, 0.3, 0.4]


class FakeStore:
    """Canned-response store that records search calls."""

    def __init__(self, search_results=None, files=None, stats=None):
        self.search_results = search_results or []
        self.files = files or {}
        self.stats = stats or {"count": 0}
        self.search_calls = []

    def search(self, embedding, limit=5, where=None):
        self.search_calls.append({"limit": limit, "where": where})
        return self.search_results[:limit]

    def get_by_file(self, file_path):
        return self.files.get(file_path, [])

    def get_stats(self):
        return self.stats


def make_result(file_path, content, distance, heading="", type="note"):
    """Build a store search result in the shape VectorStore.search returns."""
    return {
        "metadata": {"file_path": file_path, "heading": heading, "type": type},
        "content": content,
        "distance": distance,
    }


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def fakes(monkeypatch):
    """Install fake config/embedder/store into the server's lazy singletons."""
    config = Config(provider="ollama", indexer=IndexerConfig(similarity_threshold=0.5))
    embedder = FakeEmbedder()
    store = FakeStore()
    monkeypatch.setattr(server, "_config", config)
    monkeypatch.setattr(server, "_embedder", embedder)
    monkeypatch.setattr(server, "_store", store)
    return config, embedder, store


def text_result(result):
    """Parse the JSON text content of a CallToolResult."""
    assert not result.is_error, result.content
    return json.loads(result.content[0].text)


def structured_list(result):
    """Unwrap a list-returning tool's structured content ({"result": [...]})."""
    assert not result.is_error, result.content
    return result.structured_content["result"]


def test_server_module_imports():
    """The regression that motivated this file: import must not raise."""
    assert server.mcp is not None
    assert callable(server.run_server)


@pytest.mark.anyio
async def test_lists_expected_tools(fakes):
    async with Client(server.mcp) as client:
        tools = await client.list_tools()
    assert {t.name for t in tools.tools} == EXPECTED_TOOLS


@pytest.mark.anyio
async def test_get_stats_over_protocol(fakes):
    _, _, store = fakes
    store.stats = {"count": 42, "files": 7}
    async with Client(server.mcp) as client:
        result = await client.call_tool("get_stats", {})
    assert text_result(result) == {"count": 42, "files": 7}


@pytest.mark.anyio
async def test_search_notes_result_shape(fakes):
    _, _, store = fakes
    store.search_results = [
        make_result("Notes/a.md", "short note", distance=0.25, heading="Intro"),
        make_result("Notes/b.md", "x" * 600, distance=0.30, type="daily"),
    ]
    async with Client(server.mcp) as client:
        result = await client.call_tool("search_notes", {"query": "hello"})

    rows = structured_list(result)
    assert rows[0] == {
        "file_path": "Notes/a.md",
        "heading": "Intro",
        "content": "short note",
        "similarity": 0.75,
        "type": "note",
    }
    # Long content is truncated to 500 chars for the client
    assert len(rows[1]["content"]) == 500
    assert rows[1]["type"] == "daily"


@pytest.mark.anyio
async def test_search_notes_applies_similarity_threshold(fakes):
    _, _, store = fakes
    # Fixture config sets threshold to 0.5: similarity 0.75 passes, 0.1 does not
    store.search_results = [
        make_result("Notes/keep.md", "relevant", distance=0.25),
        make_result("Notes/drop.md", "irrelevant", distance=0.90),
    ]
    async with Client(server.mcp) as client:
        result = await client.call_tool("search_notes", {"query": "hello"})

    rows = structured_list(result)
    assert [r["file_path"] for r in rows] == ["Notes/keep.md"]


@pytest.mark.anyio
async def test_search_notes_passes_type_filter_and_limit(fakes):
    _, _, store = fakes
    async with Client(server.mcp) as client:
        await client.call_tool("search_notes", {"query": "q", "limit": 3, "note_type": "daily"})

    assert store.search_calls == [{"limit": 3, "where": {"type": "daily"}}]


@pytest.mark.anyio
async def test_get_similar_excludes_source_note(fakes):
    _, _, store = fakes
    store.files = {"Notes/source.md": [make_result("Notes/source.md", "source text", 0.0)]}
    store.search_results = [
        make_result("Notes/source.md", "source text", distance=0.0),
        make_result("Notes/other.md", "related text", distance=0.2),
    ]
    async with Client(server.mcp) as client:
        result = await client.call_tool("get_similar", {"note_path": "Notes/source.md"})

    rows = structured_list(result)
    assert [r["file_path"] for r in rows] == ["Notes/other.md"]
    assert rows[0]["similarity"] == 0.8


@pytest.mark.anyio
async def test_get_similar_unknown_note_returns_error(fakes):
    async with Client(server.mcp) as client:
        result = await client.call_tool("get_similar", {"note_path": "Notes/missing.md"})

    rows = structured_list(result)
    assert "error" in rows[0]


@pytest.mark.anyio
async def test_get_note_context_combines_chunks_and_similar(fakes):
    _, _, store = fakes
    store.files = {
        "Notes/source.md": [
            make_result("Notes/source.md", "chunk one", 0.0),
            make_result("Notes/source.md", "chunk two", 0.0),
        ]
    }
    store.search_results = [make_result("Notes/other.md", "related", distance=0.2)]
    async with Client(server.mcp) as client:
        result = await client.call_tool("get_note_context", {"note_path": "Notes/source.md"})

    ctx = text_result(result)
    assert ctx["file_path"] == "Notes/source.md"
    assert ctx["content"] == "chunk one\n\nchunk two"
    assert [n["file_path"] for n in ctx["similar_notes"]] == ["Notes/other.md"]


@pytest.mark.anyio
async def test_reindex_without_vault_path_returns_error(fakes):
    config, _, _ = fakes
    assert config.vault_path is None
    async with Client(server.mcp) as client:
        result = await client.call_tool("reindex", {})

    assert "error" in text_result(result)
