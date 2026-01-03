"""Markdown parsing, chunking, and embedding generation."""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, List, Dict, Tuple

import httpx
import yaml


@dataclass
class Chunk:
    """A chunk of text from a markdown file."""

    id: str
    content: str
    file_path: str
    heading: Optional[str]
    heading_level: int
    metadata: Dict


def parse_frontmatter(content: str) -> Tuple[Dict, str]:
    """Extract YAML frontmatter from markdown content."""
    if not content.startswith("---"):
        return {}, content

    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content

    try:
        frontmatter = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        frontmatter = {}

    return frontmatter, parts[2].strip()


def chunk_by_heading(content: str, file_path: str, min_chunk_size: int = 100) -> List[Chunk]:
    """Split markdown content by headings (## and ###).

    Args:
        content: Markdown content
        file_path: Path to the source file
        min_chunk_size: Minimum characters for a chunk (smaller chunks are merged up)

    Returns:
        List of Chunk objects
    """
    frontmatter, body = parse_frontmatter(content)

    # Split by headings (## or ###)
    heading_pattern = re.compile(r'^(#{2,3})\s+(.+)$', re.MULTILINE)

    chunks = []
    last_end = 0
    current_heading = None
    current_level = 0
    chunk_index = 0

    for match in heading_pattern.finditer(body):
        # Get content before this heading
        chunk_content = body[last_end:match.start()].strip()

        if chunk_content and len(chunk_content) >= min_chunk_size:
            chunk_id = _generate_chunk_id(file_path, current_heading, chunk_content, chunk_index)
            chunks.append(Chunk(
                id=chunk_id,
                content=chunk_content,
                file_path=file_path,
                heading=current_heading,
                heading_level=current_level,
                metadata={**frontmatter, "file_path": file_path}
            ))
            chunk_index += 1
        elif chunk_content and chunks:
            # Merge small chunk with previous
            chunks[-1].content += "\n\n" + chunk_content

        # Update for next iteration
        current_level = len(match.group(1))
        current_heading = match.group(2).strip()
        last_end = match.end()

    # Don't forget the last chunk
    chunk_content = body[last_end:].strip()
    if chunk_content:
        chunk_id = _generate_chunk_id(file_path, current_heading, chunk_content, chunk_index)
        chunks.append(Chunk(
            id=chunk_id,
            content=chunk_content,
            file_path=file_path,
            heading=current_heading,
            heading_level=current_level,
            metadata={**frontmatter, "file_path": file_path}
        ))
        chunk_index += 1

    # If no headings found, treat entire content as one chunk
    if not chunks and body.strip():
        chunk_id = _generate_chunk_id(file_path, None, body, 0)
        chunks.append(Chunk(
            id=chunk_id,
            content=body.strip(),
            file_path=file_path,
            heading=None,
            heading_level=0,
            metadata={**frontmatter, "file_path": file_path}
        ))

    return chunks


def _generate_chunk_id(file_path: str, heading: Optional[str], content: str, chunk_index: int = 0) -> str:
    """Generate a stable ID for a chunk."""
    # Include file path, heading, content hash, and index for uniqueness
    content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
    key = f"{file_path}:{heading or 'root'}:{content_hash}:{chunk_index}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


class OllamaEmbedder:
    """Generate embeddings using Ollama."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text"
    ):
        self.base_url = base_url
        self.model = model
        self.client = httpx.Client(timeout=60.0)

    def embed(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        response = self.client.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text}
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        # Ollama doesn't support batch, so we do sequential
        return [self.embed(text) for text in texts]

    def close(self):
        self.client.close()


class VaultIndexer:
    """Index an Obsidian vault."""

    def __init__(
        self,
        vault_path,
        embedder: OllamaEmbedder,
        exclude_patterns: Optional[List[str]] = None
    ):
        self.vault_path = Path(vault_path)
        self.embedder = embedder
        self.exclude_patterns = exclude_patterns or [
            "attachments/**",
            ".obsidian/**",
            ".trash/**"
        ]

    def iter_markdown_files(self) -> Iterator[Path]:
        """Iterate over all markdown files in the vault."""
        for md_file in self.vault_path.rglob("*.md"):
            rel_path = md_file.relative_to(self.vault_path)

            # Check exclusions
            skip = False
            for pattern in self.exclude_patterns:
                if rel_path.match(pattern):
                    skip = True
                    break

            if not skip:
                yield md_file

    def index_file(self, file_path: Path) -> List[Tuple[Chunk, List[float]]]:
        """Index a single file, returning chunks with embeddings."""
        content = file_path.read_text(encoding="utf-8")
        rel_path = str(file_path.relative_to(self.vault_path))

        chunks = chunk_by_heading(content, rel_path)

        results = []
        for chunk in chunks:
            # Add file type metadata
            if rel_path.startswith("Daily Notes/"):
                chunk.metadata["type"] = "daily"
            else:
                chunk.metadata["type"] = "note"

            embedding = self.embedder.embed(chunk.content)
            results.append((chunk, embedding))

        return results

    def index_all(self) -> Iterator[Tuple[Chunk, List[float]]]:
        """Index all files in the vault."""
        for file_path in self.iter_markdown_files():
            try:
                for chunk, embedding in self.index_file(file_path):
                    yield chunk, embedding
            except Exception as e:
                print(f"Error indexing {file_path}: {e}")
