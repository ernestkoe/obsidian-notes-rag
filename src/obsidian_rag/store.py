"""ChromaDB vector store wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import chromadb
from chromadb.config import Settings

from .indexer import Chunk


class VectorStore:
    """ChromaDB-backed vector store for Obsidian notes."""

    def __init__(self, data_path, collection_name: str = "obsidian_notes"):
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.data_path),
            settings=Settings(anonymized_telemetry=False)
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunk: Chunk, embedding: List[float]) -> None:
        """Add a single chunk to the store."""
        self.collection.add(
            ids=[chunk.id],
            embeddings=[embedding],
            documents=[chunk.content],
            metadatas=[self._prepare_metadata(chunk)]
        )

    def add_batch(
        self,
        chunks: List[Chunk],
        embeddings: Sequence[Sequence[float]]
    ) -> None:
        """Add multiple chunks to the store."""
        if not chunks:
            return

        self.collection.add(
            ids=[c.id for c in chunks],
            embeddings=list(embeddings),  # type: ignore[arg-type]
            documents=[c.content for c in chunks],
            metadatas=[self._prepare_metadata(c) for c in chunks]
        )

    def upsert(self, chunk: Chunk, embedding: List[float]) -> None:
        """Add or update a chunk."""
        self.collection.upsert(
            ids=[chunk.id],
            embeddings=[embedding],
            documents=[chunk.content],
            metadatas=[self._prepare_metadata(chunk)]
        )

    def upsert_batch(
        self,
        chunks: List[Chunk],
        embeddings: Sequence[Sequence[float]]
    ) -> None:
        """Add or update multiple chunks."""
        if not chunks:
            return

        self.collection.upsert(
            ids=[c.id for c in chunks],
            embeddings=list(embeddings),  # type: ignore[arg-type]
            documents=[c.content for c in chunks],
            metadatas=[self._prepare_metadata(c) for c in chunks]
        )

    def delete_by_file(self, file_path: str) -> None:
        """Delete all chunks from a specific file."""
        self.collection.delete(
            where={"file_path": file_path}
        )

    def search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        where: Optional[Dict] = None
    ) -> List[Dict]:
        """Search for similar chunks.

        Args:
            query_embedding: The embedding to search for
            limit: Maximum number of results
            where: Optional metadata filter (e.g., {"type": "daily"})

        Returns:
            List of results with id, content, metadata, and distance
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        output = []
        ids = results["ids"][0]
        documents = results["documents"] or [[]]
        metadatas = results["metadatas"] or [[]]
        distances = results["distances"] or [[]]

        for i in range(len(ids)):
            output.append({
                "id": ids[i],
                "content": documents[0][i] if documents[0] else "",
                "metadata": metadatas[0][i] if metadatas[0] else {},
                "distance": distances[0][i] if distances[0] else 0.0,
            })

        return output

    def get_stats(self) -> dict:
        """Get collection statistics."""
        return {
            "collection": self.collection.name,
            "count": self.collection.count(),
            "data_path": str(self.data_path)
        }

    def clear(self) -> None:
        """Clear all data from the collection."""
        # Delete and recreate collection
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )

    def _prepare_metadata(self, chunk: Chunk) -> Dict[str, Any]:
        """Prepare metadata for ChromaDB (must be flat dict with simple types)."""
        meta = {
            "file_path": chunk.file_path,
            "heading": chunk.heading or "",
            "heading_level": chunk.heading_level,
            "type": chunk.metadata.get("type", "note")
        }

        # Add any tags from frontmatter
        if "tags" in chunk.metadata:
            tags = chunk.metadata["tags"]
            if isinstance(tags, list):
                meta["tags"] = ",".join(str(t) for t in tags)
            else:
                meta["tags"] = str(tags)

        return meta
