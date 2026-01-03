"""Tests for the indexer module."""

import pytest

from obsidian_rag.indexer import chunk_by_heading, parse_frontmatter


class TestParseFrontmatter:
    def test_no_frontmatter(self):
        content = "# Hello\n\nThis is content."
        frontmatter, body = parse_frontmatter(content)
        assert frontmatter == {}
        assert body == content

    def test_with_frontmatter(self):
        content = """---
title: Test Note
tags:
  - test
  - example
---

# Hello

This is content."""
        frontmatter, body = parse_frontmatter(content)
        assert frontmatter["title"] == "Test Note"
        assert frontmatter["tags"] == ["test", "example"]
        assert body.startswith("# Hello")

    def test_invalid_yaml(self):
        content = """---
invalid: yaml: content
---

Content here."""
        frontmatter, body = parse_frontmatter(content)
        assert frontmatter == {}


class TestChunkByHeading:
    def test_single_chunk_no_headings(self):
        content = "This is a simple note without headings."
        chunks = chunk_by_heading(content, "test.md")
        assert len(chunks) == 1
        assert chunks[0].content == content
        assert chunks[0].heading is None

    def test_multiple_headings(self):
        content = """## First Section

Content for first section.

## Second Section

Content for second section."""
        chunks = chunk_by_heading(content, "test.md", min_chunk_size=10)
        assert len(chunks) == 2
        assert chunks[0].heading == "First Section"
        assert chunks[1].heading == "Second Section"

    def test_respects_min_chunk_size(self):
        content = """## Short

Hi

## Longer Section

This is a longer section with more content."""
        chunks = chunk_by_heading(content, "test.md", min_chunk_size=100)
        # Short section should be merged
        assert len(chunks) <= 2

    def test_preserves_file_path(self):
        content = "## Test\n\nContent here."
        chunks = chunk_by_heading(content, "notes/test.md", min_chunk_size=10)
        assert all(c.file_path == "notes/test.md" for c in chunks)
