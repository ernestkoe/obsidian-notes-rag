"""File watcher daemon for auto-indexing Obsidian notes."""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver

from .indexer import OllamaEmbedder, VaultIndexer
from .store import VectorStore

# Configuration defaults (can be overridden by environment variables)
DEFAULT_VAULT_PATH = os.environ.get(
    "OBSIDIAN_RAG_VAULT", "/Users/ernestkoe/Documents/Brave Robot"
)
DEFAULT_DATA_PATH = os.environ.get(
    "OBSIDIAN_RAG_DATA", "/Users/ernestkoe/Projects/mcp-obsidianRAG/data"
)
DEFAULT_OLLAMA_URL = os.environ.get(
    "OBSIDIAN_RAG_OLLAMA_URL", "http://localhost:11434"
)
DEFAULT_MODEL = os.environ.get("OBSIDIAN_RAG_MODEL", "nomic-embed-text")
DEFAULT_DEBOUNCE = float(os.environ.get("OBSIDIAN_RAG_DEBOUNCE", "2.0"))

logger = logging.getLogger(__name__)


class DebouncedHandler:
    """Debounces file events to avoid processing rapid successive changes."""

    def __init__(self, delay: float = 2.0):
        self.delay = delay
        self._timers: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    def debounce(self, key: str, callback, *args):
        """Schedule a callback after delay, canceling any pending call for the same key."""
        with self._lock:
            if key in self._timers:
                self._timers[key].cancel()

            timer = threading.Timer(self.delay, self._execute, args=(key, callback, args))
            self._timers[key] = timer
            timer.start()

    def _execute(self, key: str, callback, args):
        """Execute the callback and clean up."""
        with self._lock:
            self._timers.pop(key, None)
        try:
            callback(*args)
        except Exception as e:
            logger.error(f"Error in debounced callback for {key}: {e}")

    def cancel_all(self):
        """Cancel all pending timers."""
        with self._lock:
            for timer in self._timers.values():
                timer.cancel()
            self._timers.clear()


class NoteEventHandler(FileSystemEventHandler):
    """Handles file system events for Obsidian notes."""

    def __init__(
        self,
        vault_path: Path,
        embedder: OllamaEmbedder,
        store: VectorStore,
        debounce_delay: float = 2.0,
        exclude_patterns: Optional[list[str]] = None,
    ):
        super().__init__()
        self.vault_path = vault_path
        self.embedder = embedder
        self.store = store
        self.debouncer = DebouncedHandler(delay=debounce_delay)
        self.exclude_patterns = exclude_patterns or [
            "attachments/**",
            ".obsidian/**",
            ".trash/**",
        ]
        self.indexer = VaultIndexer(
            vault_path=vault_path,
            embedder=embedder,
            exclude_patterns=self.exclude_patterns,
        )

    def _should_ignore(self, path: Path) -> bool:
        """Check if a path should be ignored based on exclude patterns."""
        if not path.suffix == ".md":
            return True

        try:
            rel_path = path.relative_to(self.vault_path)
        except ValueError:
            return True

        for pattern in self.exclude_patterns:
            if rel_path.match(pattern):
                return True

        return False

    def _get_relative_path(self, path: Path) -> str:
        """Get the path relative to the vault root."""
        return str(path.relative_to(self.vault_path))

    def _index_file(self, path: Path):
        """Index or re-index a single file."""
        if self._should_ignore(path):
            return

        rel_path = self._get_relative_path(path)
        logger.info(f"Indexing: {rel_path}")

        try:
            # Delete existing chunks for this file
            self.store.delete_by_file(rel_path)

            # Index the file
            results = self.indexer.index_file(path)
            if results:
                chunks, embeddings = zip(*results)
                self.store.upsert_batch(list(chunks), list(embeddings))
                logger.info(f"Indexed {len(chunks)} chunks from {rel_path}")
        except Exception as e:
            logger.error(f"Error indexing {rel_path}: {e}")

    def _delete_file(self, path: Path):
        """Remove a file from the index."""
        if not path.suffix == ".md":
            return

        try:
            rel_path = self._get_relative_path(path)
        except ValueError:
            return

        logger.info(f"Removing from index: {rel_path}")
        try:
            self.store.delete_by_file(rel_path)
        except Exception as e:
            logger.error(f"Error removing {rel_path}: {e}")

    def on_created(self, event: FileSystemEvent):
        """Handle file creation."""
        if event.is_directory:
            return
        src_path = event.src_path
        if isinstance(src_path, bytes):
            src_path = src_path.decode()
        path = Path(src_path)
        self.debouncer.debounce(str(path), self._index_file, path)

    def on_modified(self, event: FileSystemEvent):
        """Handle file modification."""
        if event.is_directory:
            return
        src_path = event.src_path
        if isinstance(src_path, bytes):
            src_path = src_path.decode()
        path = Path(src_path)
        self.debouncer.debounce(str(path), self._index_file, path)

    def on_deleted(self, event: FileSystemEvent):
        """Handle file deletion."""
        if event.is_directory:
            return
        src_path = event.src_path
        if isinstance(src_path, bytes):
            src_path = src_path.decode()
        path = Path(src_path)
        # No need to debounce deletes
        self._delete_file(path)

    def on_moved(self, event: FileSystemEvent):
        """Handle file move/rename."""
        if event.is_directory:
            return

        # Delete old location
        src_path = event.src_path
        if isinstance(src_path, bytes):
            src_path = src_path.decode()
        old_path = Path(src_path)
        self._delete_file(old_path)

        # Index new location
        dest_path = getattr(event, "dest_path", None)
        if dest_path:
            if isinstance(dest_path, bytes):
                dest_path = dest_path.decode()
            new_path = Path(dest_path)
            self.debouncer.debounce(str(new_path), self._index_file, new_path)

    def shutdown(self):
        """Clean up resources."""
        self.debouncer.cancel_all()


class VaultWatcher:
    """Watches an Obsidian vault for changes and auto-indexes."""

    def __init__(
        self,
        vault_path: str = DEFAULT_VAULT_PATH,
        data_path: str = DEFAULT_DATA_PATH,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        model: str = DEFAULT_MODEL,
        debounce_delay: float = DEFAULT_DEBOUNCE,
    ):
        self.vault_path = Path(vault_path)
        self.embedder = OllamaEmbedder(base_url=ollama_url, model=model)
        self.store = VectorStore(data_path=data_path)
        self.debounce_delay = debounce_delay

        self._observer: Optional[BaseObserver] = None
        self._handler: Optional[NoteEventHandler] = None
        self._running = False

    def start(self):
        """Start watching the vault."""
        if self._running:
            return

        logger.info(f"Starting watcher for vault: {self.vault_path}")
        logger.info(f"Debounce delay: {self.debounce_delay}s")

        self._handler = NoteEventHandler(
            vault_path=self.vault_path,
            embedder=self.embedder,
            store=self.store,
            debounce_delay=self.debounce_delay,
        )

        observer = Observer()
        observer.schedule(self._handler, str(self.vault_path), recursive=True)
        observer.start()
        self._observer = observer
        self._running = True

        logger.info("Watcher started. Press Ctrl+C to stop.")

    def stop(self):
        """Stop watching the vault."""
        if not self._running:
            return

        logger.info("Stopping watcher...")

        if self._handler:
            self._handler.shutdown()

        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)

        self.embedder.close()
        self._running = False

        logger.info("Watcher stopped.")

    def run_forever(self):
        """Run the watcher until interrupted."""
        self.start()

        # Set up signal handlers
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}")
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            while self._running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()


def run_watcher(
    vault_path: str = DEFAULT_VAULT_PATH,
    data_path: str = DEFAULT_DATA_PATH,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_MODEL,
    debounce: float = DEFAULT_DEBOUNCE,
):
    """Run the vault watcher (entry point for CLI)."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    watcher = VaultWatcher(
        vault_path=vault_path,
        data_path=data_path,
        ollama_url=ollama_url,
        model=model,
        debounce_delay=debounce,
    )
    watcher.run_forever()


if __name__ == "__main__":
    run_watcher()
