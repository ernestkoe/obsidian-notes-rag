"""Tests for the link graph: extraction, resolution, storage, and traversal."""

import pytest

from obsidian_rag.indexer import Chunk
from obsidian_rag.links import LinkResolver, Neighbor, expand_neighbors, extract_links
from obsidian_rag.store import VectorStore

DIM = 4

VAULT_PATHS = [
    "notes/casa-manager-msa.md",
    "notes/casa-manager-sow.md",
    "notes/sunlight-group.md",
    "plans/tech-debt-tracker.md",
    "Daily Notes/2026-08-13.md",
    "index.md",
]


@pytest.fixture
def resolver():
    return LinkResolver(VAULT_PATHS)


@pytest.fixture
def store(tmp_path):
    return VectorStore(data_path=str(tmp_path))


def make_chunk(id: str, content: str, file_path: str) -> Chunk:
    return Chunk(id=id, content=content, file_path=file_path,
                 heading=None, heading_level=0, metadata={"type": "note"})


class TestLinkResolver:
    def test_exact_path(self, resolver):
        assert resolver.resolve_name("notes/sunlight-group.md") == "notes/sunlight-group.md"

    def test_path_without_extension(self, resolver):
        assert resolver.resolve_name("notes/sunlight-group") == "notes/sunlight-group.md"

    def test_bare_name(self, resolver):
        assert resolver.resolve_name("sunlight-group") == "notes/sunlight-group.md"

    def test_case_insensitive_fallback(self, resolver):
        assert resolver.resolve_name("Sunlight-Group") == "notes/sunlight-group.md"

    def test_unknown_name(self, resolver):
        assert resolver.resolve_name("does-not-exist") is None

    def test_relative_target(self, resolver):
        assert resolver.resolve_target(
            "../plans/tech-debt-tracker.md", "notes/casa-manager-sow.md"
        ) == "plans/tech-debt-tracker.md"

    def test_same_dir_target(self, resolver):
        assert resolver.resolve_target(
            "sunlight-group.md", "notes/casa-manager-sow.md"
        ) == "notes/sunlight-group.md"

    def test_root_relative_target(self, resolver):
        assert resolver.resolve_target(
            "/notes/sunlight-group.md", "index.md"
        ) == "notes/sunlight-group.md"

    def test_url_encoded_target(self, resolver):
        assert resolver.resolve_target(
            "Daily%20Notes/2026-08-13.md", "index.md"
        ) == "Daily Notes/2026-08-13.md"

    def test_external_urls_ignored(self, resolver):
        assert resolver.resolve_target("https://example.com/a.md", "index.md") is None
        assert resolver.resolve_target("mailto:x@y.com", "index.md") is None

    def test_anchor_only_ignored(self, resolver):
        assert resolver.resolve_target("#heading", "index.md") is None

    def test_non_markdown_ignored(self, resolver):
        assert resolver.resolve_target("image.png", "index.md") is None

    def test_anchor_stripped(self, resolver):
        assert resolver.resolve_target(
            "sunlight-group.md#people", "notes/casa-manager-sow.md"
        ) == "notes/sunlight-group.md"


class TestExtractLinks:
    def test_wikilink(self, resolver):
        body = "See [[sunlight-group]] for details."
        assert extract_links(body, "index.md", resolver) == {"notes/sunlight-group.md"}

    def test_wikilink_with_alias_and_heading(self, resolver):
        body = "Per [[sunlight-group#People|the client]] and [[casa-manager-msa|MSA]]."
        assert extract_links(body, "index.md", resolver) == {
            "notes/sunlight-group.md", "notes/casa-manager-msa.md",
        }

    def test_markdown_link(self, resolver):
        body = "Per the [MSA](casa-manager-msa.md) terms."
        assert extract_links(body, "notes/casa-manager-sow.md", resolver) == {
            "notes/casa-manager-msa.md",
        }

    def test_angle_bracket_target_with_spaces(self, resolver):
        body = "See [today](<Daily Notes/2026-08-13.md>)."
        assert extract_links(body, "index.md", resolver) == {"Daily Notes/2026-08-13.md"}

    def test_self_link_dropped(self, resolver):
        body = "See [self](casa-manager-sow.md)."
        assert extract_links(body, "notes/casa-manager-sow.md", resolver) == set()

    def test_links_in_code_ignored(self, resolver):
        body = (
            "```md\n[in code](casa-manager-msa.md)\n```\n"
            "and `[[sunlight-group]]` inline."
        )
        assert extract_links(body, "index.md", resolver) == set()

    def test_image_embeds_ignored(self, resolver):
        body = "![diagram](casa-manager-msa.md)"
        assert extract_links(body, "notes/casa-manager-sow.md", resolver) == set()

    def test_unresolvable_links_dropped(self, resolver):
        body = "[[future-concept]] and [gone](missing.md)"
        assert extract_links(body, "index.md", resolver) == set()


class TestStoreLinks:
    def test_replace_and_get(self, store):
        store.replace_links("a.md", ["b.md", "c.md"])
        assert sorted(store.get_links("a.md")) == ["b.md", "c.md"]
        assert store.get_backlinks("b.md") == ["a.md"]

    def test_replace_overwrites(self, store):
        store.replace_links("a.md", ["b.md"])
        store.replace_links("a.md", ["c.md"])
        assert store.get_links("a.md") == ["c.md"]
        assert store.get_backlinks("b.md") == []

    def test_delete_by_file_clears_outgoing(self, store):
        store.upsert(make_chunk("1", "text", "a.md"), [0.1] * DIM)
        store.replace_links("a.md", ["b.md"])
        store.delete_by_file("a.md")
        assert store.get_links("a.md") == []

    def test_clear_clears_links(self, store):
        store.replace_links("a.md", ["b.md"])
        store.clear()
        assert store.get_link_stats()["links"] == 0

    def test_link_stats(self, store):
        store.replace_links("a.md", ["b.md", "c.md"])
        store.replace_links("b.md", ["c.md"])
        assert store.get_link_stats()["links"] == 3


class TestExpandNeighbors:
    def _build_graph(self, store):
        # a -> b -> c -> d, plus e -> a (backlink into the seed)
        store.replace_links("a.md", ["b.md"])
        store.replace_links("b.md", ["c.md"])
        store.replace_links("c.md", ["d.md"])
        store.replace_links("e.md", ["a.md"])

    def test_one_hop(self, store):
        self._build_graph(store)
        neighbors = expand_neighbors(store, ["a.md"], hops=1)
        assert {(n.path, n.hop, n.direction) for n in neighbors} == {
            ("b.md", 1, "link"), ("e.md", 1, "backlink"),
        }

    def test_two_hops(self, store):
        self._build_graph(store)
        paths = {n.path for n in expand_neighbors(store, ["a.md"], hops=2)}
        assert paths == {"b.md", "c.md", "e.md"}

    def test_bfs_order_nearest_first(self, store):
        self._build_graph(store)
        hops = [n.hop for n in expand_neighbors(store, ["a.md"], hops=3)]
        assert hops == sorted(hops)

    def test_limit_cap(self, store):
        self._build_graph(store)
        assert len(expand_neighbors(store, ["a.md"], hops=3, limit=2)) == 2

    def test_seeds_never_returned(self, store):
        self._build_graph(store)
        paths = {n.path for n in expand_neighbors(store, ["a.md", "b.md"], hops=2)}
        assert "a.md" not in paths and "b.md" not in paths

    def test_cycle_terminates(self, store):
        store.replace_links("x.md", ["y.md"])
        store.replace_links("y.md", ["x.md"])
        neighbors = expand_neighbors(store, ["x.md"], hops=5)
        assert {n.path for n in neighbors} == {"y.md"}

    def test_zero_hops_empty(self, store):
        self._build_graph(store)
        assert expand_neighbors(store, ["a.md"], hops=0) == []

    def test_via_records_the_bridge(self, store):
        self._build_graph(store)
        by_path = {n.path: n for n in expand_neighbors(store, ["a.md"], hops=2)}
        assert by_path["c.md"].via == "b.md"
