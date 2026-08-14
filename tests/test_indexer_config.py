"""Tests for IndexerConfig, presets, and block preservation."""

import pytest

from obsidian_rag.indexer import (
    IndexerConfig,
    chunk_markdown,
    _protect_blocks,
    _restore_blocks,
    _NEWLINE_MASK,
)


class TestIndexerConfigDefaults:
    """Test IndexerConfig with no preset (original behavior)."""

    def test_default_values(self):
        cfg = IndexerConfig()
        assert cfg.preset == "default"
        assert cfg.chunk_size == 1500
        assert cfg.min_characters_per_chunk == 50
        assert cfg.heading_split_depth == 4
        assert cfg.preserve_latex_blocks is False
        assert cfg.preserve_code_blocks is True
        assert cfg.similarity_threshold == 0.10
        assert cfg.extra_exclude_patterns == []

    def test_explicit_override(self):
        cfg = IndexerConfig(chunk_size=800)
        assert cfg.chunk_size == 800
        assert cfg.preset == "default"


class TestIndexerConfigPresets:
    """Test preset application."""

    def test_math_preset(self):
        cfg = IndexerConfig(preset="math")
        assert cfg.chunk_size == 1024
        assert cfg.min_characters_per_chunk == 80
        assert cfg.heading_split_depth == 3
        assert cfg.preserve_latex_blocks is True
        assert cfg.preserve_code_blocks is True
        assert cfg.similarity_threshold == 0.10

    def test_prose_preset(self):
        cfg = IndexerConfig(preset="prose")
        assert cfg.chunk_size == 1200
        assert cfg.heading_split_depth == 4
        assert cfg.similarity_threshold == 0.10

    def test_unknown_preset_uses_defaults(self):
        cfg = IndexerConfig(preset="nonexistent")
        assert cfg.chunk_size == 1500
        assert cfg.heading_split_depth == 4

    def test_preset_with_explicit_override(self):
        cfg = IndexerConfig(preset="math", chunk_size=2048)
        assert cfg.chunk_size == 2048
        assert cfg.preserve_latex_blocks is True  # from preset


class TestIndexerConfigSerialization:
    """Test TOML round-trip via to_dict / from_dict."""

    def test_default_to_dict_minimal(self):
        cfg = IndexerConfig()
        d = cfg.to_dict()
        assert d == {"preset": "default"}

    def test_math_preset_to_dict_minimal(self):
        cfg = IndexerConfig(preset="math")
        d = cfg.to_dict()
        assert d == {"preset": "math"}

    def test_override_appears_in_dict(self):
        cfg = IndexerConfig(preset="math", chunk_size=2048)
        d = cfg.to_dict()
        assert d["preset"] == "math"
        assert d["chunk_size"] == 2048

    def test_from_dict_default(self):
        cfg = IndexerConfig.from_dict({})
        assert cfg.preset == "default"
        assert cfg.chunk_size == 1500

    def test_from_dict_math(self):
        cfg = IndexerConfig.from_dict({"preset": "math"})
        assert cfg.chunk_size == 1024
        assert cfg.preserve_latex_blocks is True

    def test_from_dict_with_overrides(self):
        cfg = IndexerConfig.from_dict({
            "preset": "math",
            "chunk_size": 512,
            "similarity_threshold": 0.30,
        })
        assert cfg.chunk_size == 512
        assert cfg.similarity_threshold == 0.30
        assert cfg.preserve_latex_blocks is True  # from preset

    def test_round_trip(self):
        original = IndexerConfig(preset="math", chunk_size=2048)
        d = original.to_dict()
        restored = IndexerConfig.from_dict(d)
        assert restored.preset == original.preset
        assert restored.chunk_size == original.chunk_size
        assert restored.preserve_latex_blocks == original.preserve_latex_blocks
        assert restored.similarity_threshold == original.similarity_threshold


class TestBlockProtection:
    """Test newline masking inside LaTeX and code blocks."""

    def test_no_protection_when_disabled(self):
        cfg = IndexerConfig(preserve_code_blocks=False)
        body = "Text\n$$\na + b\n$$\nMore"
        result = _protect_blocks(body, cfg)
        assert result == body

    def test_latex_display_math_masked(self):
        cfg = IndexerConfig(preserve_latex_blocks=True)
        body = "Text\n$$\na + b\n$$\nMore"
        result = _protect_blocks(body, cfg)
        assert "\n" not in result.split("$$")[1]  # inside $$ has no newlines
        assert _NEWLINE_MASK in result

    def test_latex_env_masked(self):
        cfg = IndexerConfig(preserve_latex_blocks=True)
        body = "Text\n\\begin{align}\nA &= B \\\\\nC &= D\n\\end{align}\nMore"
        result = _protect_blocks(body, cfg)
        # newlines inside the environment should be masked
        start = result.index("\\begin{align}")
        end = result.index("\\end{align}") + len("\\end{align}")
        inside = result[start:end]
        assert "\n" not in inside
        assert _NEWLINE_MASK in inside

    def test_code_blocks_protected_by_default(self):
        """Code blocks should be protected from splitting with default config."""
        cfg = IndexerConfig()
        body = "Text\n```python\ndef foo():\n    pass\n```\nMore"
        result = _protect_blocks(body, cfg)
        start = result.index("```python")
        end = result.rindex("```") + 3
        inside = result[start:end]
        assert "\n" not in inside

    def test_code_block_masked(self):
        cfg = IndexerConfig(preserve_code_blocks=True)
        body = "Text\n```python\ndef foo():\n    pass\n```\nMore"
        result = _protect_blocks(body, cfg)
        start = result.index("```python")
        end = result.rindex("```") + 3
        inside = result[start:end]
        assert "\n" not in inside

    def test_restore_reverses_mask(self):
        cfg = IndexerConfig(preserve_latex_blocks=True)
        body = "Text\n$$\na + b\n$$\nMore"
        protected = _protect_blocks(body, cfg)
        restored = _restore_blocks(protected)
        assert restored == body

    def test_prose_outside_blocks_unchanged(self):
        cfg = IndexerConfig(preserve_latex_blocks=True)
        body = "Paragraph one.\n\n## Heading\n\nParagraph two.\n$$\nx^2\n$$\nEnd."
        result = _protect_blocks(body, cfg)
        # heading newlines should still be real newlines
        assert "\n\n## Heading\n\n" in result


class TestChunkMarkdownWithConfig:
    """Test chunk_markdown respects IndexerConfig."""

    def test_default_config_backward_compatible(self):
        content = "# Title\n\nSome text here.\n\n## Section\n\nMore text."
        chunks = chunk_markdown(content, "test.md")
        assert len(chunks) >= 1
        assert chunks[0].file_path == "test.md"

    def test_config_passed_through(self):
        content = "# Title\n\n" + "word " * 500
        small = chunk_markdown(content, "test.md", config=IndexerConfig(chunk_size=100))
        large = chunk_markdown(content, "test.md", config=IndexerConfig(chunk_size=10000))
        assert len(small) > len(large)

    def test_heading_split_depth(self):
        # Each section needs enough content to exceed a small chunk_size
        section = "word " * 60  # ~300 chars per section
        content = f"# H1\n\n{section}\n\n## H2\n\n{section}\n\n### H3\n\n{section}\n\n#### H4\n\n{section}"
        # depth=2: split on h1 and h2, h3/h4 stay glued to h2's chunk
        chunks_shallow = chunk_markdown(
            content, "test.md",
            config=IndexerConfig(chunk_size=400, heading_split_depth=2),
        )
        # depth=4: split on all heading levels
        chunks_deep = chunk_markdown(
            content, "test.md",
            config=IndexerConfig(chunk_size=400, heading_split_depth=4),
        )
        assert len(chunks_deep) >= len(chunks_shallow)

    def test_latex_blocks_stay_intact(self):
        padding = "Discussion text. " * 30  # ~500 chars of prose
        content = (
            f"# Proof\n\n{padding}\n\n"
            "$$\n\\begin{aligned}\nA &= B \\\\\nC &= D\n\\end{aligned}\n$$\n\n"
            f"{padding}"
        )
        cfg = IndexerConfig(preserve_latex_blocks=True, chunk_size=600)
        chunks = chunk_markdown(content, "test.md", config=cfg)
        assert len(chunks) > 1, "Content should be split into multiple chunks"
        found = any("\\begin{aligned}" in c.content and "\\end{aligned}" in c.content
                     for c in chunks)
        assert found, "LaTeX aligned block was split across chunks"

    def test_empty_content(self):
        chunks = chunk_markdown("", "test.md", config=IndexerConfig(preset="math"))
        assert chunks == []

    def test_frontmatter_only(self):
        content = "---\ntags: [test]\n---\n"
        chunks = chunk_markdown(content, "test.md", config=IndexerConfig(preset="math"))
        assert chunks == []


class TestTypeRules:
    def test_default_daily_notes_rule(self):
        chunks = chunk_markdown("# Standup\n\nNotes here.", "Daily Notes/2026-08-13.md")
        assert chunks[0].metadata["type"] == "daily"

    def test_default_fallback_is_note(self):
        chunks = chunk_markdown("# Concept\n\nBody.", "notes/concept.md")
        assert chunks[0].metadata["type"] == "note"

    def test_custom_rules_override(self):
        cfg = IndexerConfig(type_rules={"meetings/": "meeting"})
        chunks = chunk_markdown("# Kickoff\n\nBody.", "meetings/kickoff.md", config=cfg)
        assert chunks[0].metadata["type"] == "meeting"
        # Default rule is replaced, not merged
        daily = chunk_markdown("# D\n\nBody.", "Daily Notes/x.md", config=cfg)
        assert daily[0].metadata["type"] == "note"

    def test_longest_prefix_wins(self):
        cfg = IndexerConfig(type_rules={"internal/": "doc", "internal/notes/": "concept"})
        concept = chunk_markdown("# C\n\nBody.", "internal/notes/a.md", config=cfg)
        doc = chunk_markdown("# P\n\nBody.", "internal/plans/b.md", config=cfg)
        assert concept[0].metadata["type"] == "concept"
        assert doc[0].metadata["type"] == "doc"

    def test_toml_roundtrip(self):
        cfg = IndexerConfig.from_dict({"type_rules": {"meetings/": "meeting"}})
        assert cfg.type_rules == {"meetings/": "meeting"}
        assert IndexerConfig().to_dict().get("type_rules") is None  # default omitted
        assert cfg.to_dict()["type_rules"] == {"meetings/": "meeting"}
