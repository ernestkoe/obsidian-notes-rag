"""SQLite-vec vector store wrapper."""

from __future__ import annotations

import sqlite3
import struct
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import sqlite_vec

from .indexer import Chunk

# Default embedding dimension (nomic-embed-text = 768, OpenAI small = 1536)
# Detected automatically on first upsert.
DEFAULT_DIM = 768


def _serialize_f32(vec: Sequence[float]) -> bytes:
    """Serialize a list of floats to a compact bytes format for sqlite-vec."""
    return struct.pack(f"{len(vec)}f", *vec)


class VectorStore:
    """SQLite-vec backed vector store for Obsidian notes."""

    def __init__(self, data_path: str, collection_name: str = "obsidian_notes"):
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.db_path = self.data_path / f"{collection_name}.db"

        self._lock = threading.Lock()
        self.db = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)

        self._dim: Optional[int] = None
        self._ensure_metadata_table()
        self._try_load_vec_table()

    def _ensure_metadata_table(self) -> None:
        """Create the metadata table if it doesn't exist."""
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                heading TEXT,
                heading_level INTEGER,
                type TEXT,
                tags TEXT,
                content TEXT NOT NULL
            )
        """)
        self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_file_path
            ON chunks(file_path)
        """)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS links (
                source_path TEXT NOT NULL,
                target_path TEXT NOT NULL,
                PRIMARY KEY (source_path, target_path)
            )
        """)
        self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_links_target
            ON links(target_path)
        """)
        self.db.commit()

    def _try_load_vec_table(self) -> None:
        """Try to detect the dimension from an existing vec table."""
        try:
            row = self.db.execute(
                "SELECT embedding FROM chunks_vec LIMIT 1"
            ).fetchone()
            if row is not None:
                self._dim = len(row[0]) // 4  # 4 bytes per float32
        except sqlite3.OperationalError:
            pass  # Table doesn't exist yet

    def _ensure_vec_table(self, dim: int) -> None:
        """Create the vector table with the given dimension."""
        if self._dim == dim:
            return
        self._dim = dim
        self.db.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
                id TEXT PRIMARY KEY,
                embedding float[{dim}] distance_metric=cosine
            )
        """)
        self.db.commit()

    def upsert(self, chunk: Chunk, embedding: List[float]) -> None:
        """Add or update a chunk."""
        with self._lock:
            self._ensure_vec_table(len(embedding))
            meta = self._prepare_metadata(chunk)

            self.db.execute("""
                INSERT OR REPLACE INTO chunks (id, file_path, heading, heading_level, type, tags, content)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (chunk.id, meta["file_path"], meta["heading"], meta["heading_level"],
                  meta["type"], meta.get("tags", ""), chunk.content))

            # sqlite-vec: delete then insert (no native upsert on virtual tables)
            self.db.execute("DELETE FROM chunks_vec WHERE id = ?", (chunk.id,))
            self.db.execute(
                "INSERT INTO chunks_vec (id, embedding) VALUES (?, ?)",
                (chunk.id, _serialize_f32(embedding))
            )
            self.db.commit()

    def upsert_batch(self, chunks: List[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        """Add or update multiple chunks."""
        if not chunks:
            return
        with self._lock:
            self._ensure_vec_table(len(embeddings[0]))

            for chunk, embedding in zip(chunks, embeddings):
                meta = self._prepare_metadata(chunk)
                self.db.execute("""
                    INSERT OR REPLACE INTO chunks (id, file_path, heading, heading_level, type, tags, content)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (chunk.id, meta["file_path"], meta["heading"], meta["heading_level"],
                      meta["type"], meta.get("tags", ""), chunk.content))

                self.db.execute("DELETE FROM chunks_vec WHERE id = ?", (chunk.id,))
                self.db.execute(
                    "INSERT INTO chunks_vec (id, embedding) VALUES (?, ?)",
                    (chunk.id, _serialize_f32(embedding))
                )

            self.db.commit()

    def delete_by_file(self, file_path: str) -> None:
        """Delete all chunks (and outgoing links) from a specific file."""
        with self._lock:
            ids = [row[0] for row in
                   self.db.execute("SELECT id FROM chunks WHERE file_path = ?", (file_path,)).fetchall()]
            if ids:
                placeholders = ",".join("?" * len(ids))
                self.db.execute(f"DELETE FROM chunks_vec WHERE id IN ({placeholders})", ids)
                self.db.execute(f"DELETE FROM chunks WHERE id IN ({placeholders})", ids)
            self.db.execute("DELETE FROM links WHERE source_path = ?", (file_path,))
            self.db.commit()

    def replace_links(self, source_path: str, target_paths: Sequence[str]) -> None:
        """Replace all outgoing links for a source note."""
        with self._lock:
            self.db.execute("DELETE FROM links WHERE source_path = ?", (source_path,))
            self.db.executemany(
                "INSERT OR IGNORE INTO links (source_path, target_path) VALUES (?, ?)",
                [(source_path, t) for t in target_paths],
            )
            self.db.commit()

    def get_links(self, file_path: str) -> List[str]:
        """Notes this note links to."""
        with self._lock:
            rows = self.db.execute(
                "SELECT target_path FROM links WHERE source_path = ?", (file_path,)
            ).fetchall()
            return [row[0] for row in rows]

    def get_backlinks(self, file_path: str) -> List[str]:
        """Notes that link to this note."""
        with self._lock:
            rows = self.db.execute(
                "SELECT source_path FROM links WHERE target_path = ?", (file_path,)
            ).fetchall()
            return [row[0] for row in rows]

    def get_link_stats(self) -> dict:
        """Edge count for stats display."""
        with self._lock:
            count = self.db.execute("SELECT COUNT(*) FROM links").fetchone()[0]
            return {"links": count}

    def search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        where: Optional[Dict] = None
    ) -> List[Dict]:
        """Search for similar chunks."""
        with self._lock:
            if self._dim is None:
                return []

            query_bytes = _serialize_f32(query_embedding)

            if where:
                conditions = []
                params: list = []
                for key, value in where.items():
                    conditions.append(f"c.{key} = ?")
                    params.append(value)

                where_clause = " AND ".join(conditions)
                fetch_limit = limit * 5

                rows = self.db.execute(f"""
                    SELECT c.id, c.file_path, c.heading, c.heading_level, c.type, c.tags, c.content, v.distance
                    FROM chunks_vec v
                    JOIN chunks c ON c.id = v.id
                    WHERE v.embedding MATCH ? AND k = ?
                      AND {where_clause}
                    ORDER BY v.distance
                    LIMIT ?
                """, [query_bytes, fetch_limit] + params + [limit]).fetchall()
            else:
                rows = self.db.execute("""
                    SELECT c.id, c.file_path, c.heading, c.heading_level, c.type, c.tags, c.content, v.distance
                    FROM chunks_vec v
                    JOIN chunks c ON c.id = v.id
                    WHERE v.embedding MATCH ? AND k = ?
                    ORDER BY v.distance
                """, [query_bytes, limit]).fetchall()

            results = []
            for row in rows:
                results.append({
                    "id": row[0],
                    "metadata": {
                        "file_path": row[1],
                        "heading": row[2] or "",
                        "heading_level": row[3],
                        "type": row[4] or "note",
                        "tags": row[5] or "",
                    },
                    "content": row[6],
                    "distance": row[7],
                })
            return results

    def get_by_file(self, file_path: str) -> List[Dict]:
        """Get all chunks for a file path (direct lookup, no vector search)."""
        with self._lock:
            rows = self.db.execute("""
                SELECT id, file_path, heading, heading_level, type, tags, content
                FROM chunks WHERE file_path = ?
            """, (file_path,)).fetchall()
            return [{
                "id": row[0],
                "metadata": {
                    "file_path": row[1],
                    "heading": row[2] or "",
                    "heading_level": row[3],
                    "type": row[4] or "note",
                    "tags": row[5] or "",
                },
                "content": row[6],
            } for row in rows]

    def get_stats(self) -> dict:
        """Get collection statistics."""
        with self._lock:
            count = self.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            return {
                "collection": self.collection_name,
                "count": count,
                "data_path": str(self.data_path),
            }

    def clear(self) -> None:
        """Clear all data."""
        with self._lock:
            self.db.execute("DELETE FROM chunks")
            self.db.execute("DELETE FROM links")
            if self._dim is not None:
                self.db.execute("DROP TABLE IF EXISTS chunks_vec")
                self._dim = None
            self.db.commit()

    def _prepare_metadata(self, chunk: Chunk) -> Dict[str, Any]:
        """Prepare metadata for storage."""
        meta = {
            "file_path": chunk.file_path,
            "heading": chunk.heading or "",
            "heading_level": chunk.heading_level,
            "type": chunk.metadata.get("type", "note"),
        }
        if "tags" in chunk.metadata:
            tags = chunk.metadata["tags"]
            if isinstance(tags, list):
                meta["tags"] = ",".join(str(t) for t in tags)
            else:
                meta["tags"] = str(tags)
        return meta
