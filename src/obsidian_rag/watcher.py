"""File watcher daemon for auto-indexing Obsidian notes."""

from __future__ import annotations

import logging
import logging.handlers
import os
import signal
import subprocess
import sys
import threading
import time

import setproctitle
from collections import deque
from pathlib import Path
from typing import Optional

# Log rotation settings
MAX_LOG_BYTES = 10 * 1024 * 1024  # 10 MB per log file
LOG_BACKUP_COUNT = 3  # Keep 3 rotated files

import httpx
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer
from watchdog.observers.api import BaseObserver

from .config import load_config
from .indexer import create_embedder, Embedder, VaultIndexer, IndexerConfig
from .store import VectorStore

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 30  # seconds
HEALTH_CHECK_INTERVAL = 60  # seconds

# Load config for defaults
_config = load_config()

DEFAULT_VAULT_PATH = _config.vault_path or os.environ.get(
    "OBSIDIAN_RAG_VAULT", ""
)
DEFAULT_DATA_PATH = _config.get_data_path()
DEFAULT_PROVIDER = _config.provider
DEFAULT_OLLAMA_URL = _config.ollama_url
DEFAULT_LMSTUDIO_URL = _config.lmstudio_url
DEFAULT_OLLAMA_API_KEY: Optional[str] = _config.get_ollama_api_key()
DEFAULT_LMSTUDIO_API_KEY: Optional[str] = _config.get_lmstudio_api_key()
DEFAULT_MODEL: Optional[str] = None  # Use provider default
DEFAULT_DEBOUNCE = float(os.environ.get("OBSIDIAN_RAG_DEBOUNCE", "2.0"))

logger = logging.getLogger(__name__)


def check_ollama_health(ollama_url: str = "http://localhost:11434") -> bool:
    """Check if Ollama is running and accessible."""
    try:
        response = httpx.get(f"{ollama_url}/api/tags", timeout=5.0)
        return response.status_code == 200
    except Exception:
        return False


def send_notification(title: str, message: str):
    """Send a macOS notification."""
    try:
        subprocess.run(
            [
                "osascript",
                "-e",
                f'display notification "{message}" with title "{title}"',
            ],
            check=False,
            capture_output=True,
        )
    except Exception:
        pass  # Notifications are best-effort


class RetryQueue:
    """Queue for files that failed to index."""

    def __init__(self, max_retries: int = MAX_RETRIES):
        self.max_retries = max_retries
        self._queue: deque[tuple[Path, int]] = deque()
        self._lock = threading.Lock()

    def add(self, path: Path):
        """Add a file to the retry queue."""
        with self._lock:
            # Check if already in queue
            for queued_path, _ in self._queue:
                if queued_path == path:
                    return
            self._queue.append((path, 0))
            logger.info(f"Added to retry queue: {path}")

    def get_next(self) -> Optional[tuple[Path, int]]:
        """Get the next file to retry, if any."""
        with self._lock:
            if not self._queue:
                return None
            return self._queue.popleft()

    def requeue(self, path: Path, attempts: int):
        """Re-add a file with incremented attempt count."""
        with self._lock:
            if attempts < self.max_retries:
                self._queue.append((path, attempts + 1))
                logger.info(f"Re-queued {path} (attempt {attempts + 1}/{self.max_retries})")
            else:
                logger.error(f"Max retries exceeded for {path}")
                send_notification(
                    "Obsidian RAG Error",
                    f"Failed to index: {path.name}"
                )

    def is_empty(self) -> bool:
        """Check if the queue is empty."""
        with self._lock:
            return len(self._queue) == 0


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
        embedder: Embedder,
        store: VectorStore,
        debounce_delay: float = 2.0,
        exclude_patterns: Optional[list[str]] = None,
        retry_queue: Optional[RetryQueue] = None,
        indexer_config: Optional[IndexerConfig] = None,
    ):
        super().__init__()
        self.vault_path = vault_path
        self.embedder = embedder
        self.store = store
        self.debouncer = DebouncedHandler(delay=debounce_delay)
        self.retry_queue = retry_queue
        self.exclude_patterns = exclude_patterns or [
            "attachments/**",
            ".obsidian/**",
            ".trash/**",
            ".venv/**",
            "node_modules/**",
            "__pycache__/**",
            "*.egg-info/**",
            "build/**",
            "dist/**",
            ".git/**",
        ]
        self.indexer = VaultIndexer(
            vault_path=vault_path,
            embedder=embedder,
            exclude_patterns=self.exclude_patterns,
            config=indexer_config,
        )

    def _should_ignore(self, path: Path) -> bool:
        """Check if a path should be ignored based on exclude patterns."""
        if not path.suffix == ".md":
            return True

        # Ignore Obsidian temp/recovery files (pattern: .!NNNNN!filename.md)
        if path.name.startswith(".!") and "!" in path.name[2:]:
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

        # Check file exists before attempting to index (avoids retry loops for deleted files)
        if not path.exists():
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

            # Refresh the file's outgoing link edges
            self.store.replace_links(rel_path, sorted(self.indexer.file_links(path)))
        except Exception as e:
            logger.error(f"Error indexing {rel_path}: {e}")
            # Add to retry queue if available
            if self.retry_queue:
                self.retry_queue.add(path)

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
        provider: str = DEFAULT_PROVIDER,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        lmstudio_url: str = DEFAULT_LMSTUDIO_URL,
        ollama_api_key: Optional[str] = DEFAULT_OLLAMA_API_KEY,
        lmstudio_api_key: Optional[str] = DEFAULT_LMSTUDIO_API_KEY,
        model: Optional[str] = DEFAULT_MODEL,
        debounce_delay: float = DEFAULT_DEBOUNCE,
    ):
        self.vault_path = Path(vault_path)
        self.provider = provider
        self.ollama_url = ollama_url

        # Set OpenAI API key from config if needed
        if provider == "openai" and _config.openai_api_key:
            os.environ["OPENAI_API_KEY"] = _config.openai_api_key

        # Determine correct base_url and api_key based on provider
        if provider == "ollama":
            base_url = ollama_url
            api_key = ollama_api_key
            # Health check for Ollama before starting
            if not check_ollama_health(ollama_url):
                logger.warning("Ollama is not running! Waiting for it to start...")
                send_notification("Obsidian RAG", "Waiting for Ollama to start...")
                self._wait_for_ollama(ollama_url)
        elif provider == "lmstudio":
            base_url = lmstudio_url
            api_key = lmstudio_api_key
        else:
            base_url = None
            api_key = None

        self.embedder = create_embedder(provider=provider, model=model, base_url=base_url, api_key=api_key)
        self.store = VectorStore(data_path=data_path)
        self.debounce_delay = debounce_delay
        self.retry_queue = RetryQueue()

        self._observer: Optional[BaseObserver] = None
        self._handler: Optional[NoteEventHandler] = None
        self._running = False
        self._health_thread: Optional[threading.Thread] = None

    def _wait_for_ollama(self, ollama_url: str, timeout: int = 300):
        """Wait for Ollama to become available."""
        start = time.time()
        while time.time() - start < timeout:
            if check_ollama_health(ollama_url):
                logger.info("Ollama is now available!")
                return
            time.sleep(5)
        raise RuntimeError(f"Ollama did not start within {timeout} seconds")

    def _health_check_loop(self):
        """Periodically check embedding service health and process retry queue."""
        while self._running:
            time.sleep(HEALTH_CHECK_INTERVAL)

            if not self._running:
                break

            # Check health
            if self.provider == "ollama" and not check_ollama_health(self.ollama_url):
                logger.warning("Ollama health check failed!")
                send_notification("Obsidian RAG", "Ollama is not responding")
                continue

            # Process retry queue
            while not self.retry_queue.is_empty():
                item = self.retry_queue.get_next()
                if item is None:
                    break

                path, attempts = item
                try:
                    if self._handler:
                        self._handler._index_file(path)
                except Exception as e:
                    logger.error(f"Retry failed for {path}: {e}")
                    self.retry_queue.requeue(path, attempts)

    def start(self):
        """Start watching the vault."""
        if self._running:
            return

        logger.info(f"Starting watcher for vault: {self.vault_path}")
        logger.info(f"Debounce delay: {self.debounce_delay}s")
        logger.info(f"Provider: {self.provider}")

        self._handler = NoteEventHandler(
            vault_path=self.vault_path,
            embedder=self.embedder,
            store=self.store,
            debounce_delay=self.debounce_delay,
            retry_queue=self.retry_queue,
            indexer_config=_config.indexer,
        )

        observer = Observer()
        observer.schedule(self._handler, str(self.vault_path), recursive=True)
        observer.start()
        self._observer = observer
        self._running = True

        # Start health check thread
        self._health_thread = threading.Thread(target=self._health_check_loop, daemon=True)
        self._health_thread.start()

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


def _setup_logging():
    """Configure logging with rotation when running as a service."""
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Check if we're running as a service (stderr redirected to file)
    log_dir = Path.home() / "Library" / "Logs" / "obsidian-notes-rag"

    if not sys.stderr.isatty() and log_dir.exists():
        # Running as service - use rotating file handler
        log_file = log_dir / "watcher.log"
        handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=MAX_LOG_BYTES,
            backupCount=LOG_BACKUP_COUNT,
        )
        handler.setFormatter(logging.Formatter(log_format, date_format))

        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(handler)

        # Note: We intentionally don't add a StreamHandler when running as a service.
        # launchd captures stderr to watcher.err without rotation, which can grow unbounded.
    else:
        # Interactive mode - simple console logging
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            datefmt=date_format,
        )


def run_watcher(
    vault_path: str = DEFAULT_VAULT_PATH,
    data_path: str = DEFAULT_DATA_PATH,
    provider: str = DEFAULT_PROVIDER,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    lmstudio_url: str = DEFAULT_LMSTUDIO_URL,
    ollama_api_key: Optional[str] = DEFAULT_OLLAMA_API_KEY,
    lmstudio_api_key: Optional[str] = DEFAULT_LMSTUDIO_API_KEY,
    model: Optional[str] = DEFAULT_MODEL,
    debounce: float = DEFAULT_DEBOUNCE,
):
    """Run the vault watcher (entry point for CLI)."""
    # Set process title for Activity Monitor visibility
    setproctitle.setproctitle("obsidian-notes-rag")

    # Configure logging with rotation support
    _setup_logging()

    watcher = VaultWatcher(
        vault_path=vault_path,
        data_path=data_path,
        provider=provider,
        ollama_url=ollama_url,
        lmstudio_url=lmstudio_url,
        ollama_api_key=ollama_api_key,
        lmstudio_api_key=lmstudio_api_key,
        model=model,
        debounce_delay=debounce,
    )
    watcher.run_forever()


if __name__ == "__main__":
    run_watcher()
