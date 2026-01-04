"""Command-line interface for obsidian-rag."""

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import click

# Suppress noisy HTTP logs from httpx/openai during progress bars
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

from .config import Config, load_config, save_config, get_config_path, get_data_dir
from .indexer import create_embedder, VaultIndexer
from .server import run_server
from .store import VectorStore
from .watcher import VaultWatcher

# Default configuration
DEFAULT_VAULT = "/Users/ernestkoe/Documents/Brave Robot"
DEFAULT_DATA = "/Users/ernestkoe/Projects/obsidian-rag/data"
DEFAULT_PROVIDER = "openai"
DEFAULT_OLLAMA_URL = "http://localhost:11434"


@click.group()
@click.option("--vault", default=None, help="Path to Obsidian vault")
@click.option("--data", default=None, help="Path to vector store data")
@click.option("--provider", default=None,
              type=click.Choice(["openai", "ollama"]),
              help="Embedding provider (default: openai)")
@click.option("--ollama-url", default=None,
              help="Ollama API URL (only used with --provider ollama)")
@click.option("--model", default=None, help="Override embedding model name")
@click.pass_context
def main(ctx, vault, data, provider, ollama_url, model):
    """Obsidian RAG - Semantic search for your Obsidian vault."""
    ctx.ensure_object(dict)

    # Load config from file, then apply CLI overrides
    config = load_config()

    ctx.obj["vault"] = vault or config.vault_path or DEFAULT_VAULT
    ctx.obj["data"] = data or config.get_data_path()
    ctx.obj["provider"] = provider or config.provider
    ctx.obj["ollama_url"] = ollama_url or config.ollama_url
    ctx.obj["model"] = model  # None means use provider default
    ctx.obj["config"] = config


@main.command()
def setup():
    """Interactive setup wizard for obsidian-rag."""
    click.echo("\nWelcome to Obsidian RAG setup!\n")

    # Check for existing config
    config_path = get_config_path()
    if config_path.exists():
        if not click.confirm(f"Config already exists at {config_path}. Overwrite?"):
            click.echo("Setup cancelled.")
            return

    config = Config()

    # 1. Select provider
    click.echo("Select embedding provider:")
    click.echo("  1. OpenAI (recommended - requires API key)")
    click.echo("  2. Ollama (local, offline)")
    provider_choice = click.prompt("Choice", type=click.Choice(["1", "2"]), default="1")
    config.provider = "openai" if provider_choice == "1" else "ollama"

    # 2. Provider-specific setup
    if config.provider == "openai":
        # Check for existing API key
        existing_key = os.environ.get("OPENAI_API_KEY")
        if existing_key:
            click.echo(f"\n✓ Found OPENAI_API_KEY in environment")
            if not click.confirm("Save API key to config file?", default=False):
                config.openai_api_key = None
            else:
                config.openai_api_key = existing_key
        else:
            click.echo("\nNo OPENAI_API_KEY found in environment.")
            api_key = click.prompt("Enter your OpenAI API key", hide_input=True)
            config.openai_api_key = api_key
    else:
        # Ollama setup
        ollama_url = click.prompt(
            "\nOllama API URL",
            default="http://localhost:11434"
        )
        config.ollama_url = ollama_url

    # 3. Vault path
    while True:
        vault_path = click.prompt("\nPath to your Obsidian vault")
        vault_path = os.path.expanduser(vault_path)
        if Path(vault_path).exists():
            md_files = list(Path(vault_path).rglob("*.md"))
            click.echo(f"✓ Vault found ({len(md_files)} markdown files)")
            config.vault_path = vault_path
            break
        else:
            click.echo(f"✗ Directory not found: {vault_path}")
            if not click.confirm("Try again?", default=True):
                click.echo("Setup cancelled.")
                return

    # 4. Data directory
    default_data = str(get_data_dir())
    data_path = click.prompt(
        "\nWhere to store the search index?",
        default=default_data
    )
    data_path = os.path.expanduser(data_path)
    config.data_path = data_path

    # 5. Save config
    saved_path = save_config(config)
    click.echo(f"\n✓ Configuration saved to {saved_path}")

    # 6. Offer to run initial index
    if click.confirm("\nRun initial indexing now?", default=True):
        click.echo("\nIndexing vault...")
        try:
            # Create embedder based on provider
            if config.provider == "openai":
                # Set API key in environment for OpenAI client
                if config.openai_api_key:
                    os.environ["OPENAI_API_KEY"] = config.openai_api_key
                embedder = create_embedder(provider="openai")
            else:
                embedder = create_embedder(
                    provider="ollama",
                    base_url=config.ollama_url
                )

            store = VectorStore(data_path=config.data_path)
            indexer = VaultIndexer(vault_path=config.vault_path, embedder=embedder)

            files = list(indexer.iter_markdown_files())
            chunk_count = 0
            batch_chunks = []
            batch_embeddings = []
            batch_size = 50

            with click.progressbar(files, label="Indexing") as bar:
                for file_path in bar:
                    try:
                        for chunk, embedding in indexer.index_file(file_path):
                            batch_chunks.append(chunk)
                            batch_embeddings.append(embedding)
                            chunk_count += 1

                            if len(batch_chunks) >= batch_size:
                                store.upsert_batch(batch_chunks, batch_embeddings)
                                batch_chunks = []
                                batch_embeddings = []
                    except Exception as e:
                        click.echo(f"\n  Error: {file_path}: {e}", err=True)

            if batch_chunks:
                store.upsert_batch(batch_chunks, batch_embeddings)

            embedder.close()
            click.echo(f"\n✓ Indexed {chunk_count} chunks from {len(files)} files")

        except Exception as e:
            click.echo(f"\n✗ Indexing failed: {e}", err=True)
            click.echo("You can run indexing later with: obsidian-notes-rag index")

    click.echo("\nSetup complete! You can now:")
    click.echo("  - Search: obsidian-notes-rag search \"your query\"")
    click.echo("  - Add to Claude Code:")
    click.echo("      claude mcp add -s user obsidian-notes-rag -- uvx obsidian-notes-rag serve")


@main.command()
@click.option("--clear", is_flag=True, help="Clear existing index before indexing")
@click.pass_context
def index(ctx, clear):
    """Index all markdown files in the vault."""
    vault_path = ctx.obj["vault"]
    data_path = ctx.obj["data"]
    provider = ctx.obj["provider"]
    ollama_url = ctx.obj["ollama_url"]
    model = ctx.obj["model"]

    click.echo(f"Indexing vault: {vault_path}")
    click.echo(f"Data path: {data_path}")
    click.echo(f"Provider: {provider}")

    # Initialize components
    embedder = create_embedder(provider=provider, model=model, base_url=ollama_url)
    store = VectorStore(data_path=data_path)
    indexer = VaultIndexer(vault_path=vault_path, embedder=embedder)

    if clear:
        click.echo("Clearing existing index...")
        store.clear()

    # Count files first
    files = list(indexer.iter_markdown_files())
    click.echo(f"Found {len(files)} markdown files")

    # Index with progress
    chunk_count = 0
    batch_chunks = []
    batch_embeddings = []
    batch_size = 50

    with click.progressbar(files, label="Indexing") as bar:
        for file_path in bar:
            try:
                for chunk, embedding in indexer.index_file(file_path):
                    batch_chunks.append(chunk)
                    batch_embeddings.append(embedding)
                    chunk_count += 1

                    # Batch insert
                    if len(batch_chunks) >= batch_size:
                        store.upsert_batch(batch_chunks, batch_embeddings)
                        batch_chunks = []
                        batch_embeddings = []

            except Exception as e:
                click.echo(f"\nError indexing {file_path}: {e}", err=True)

    # Insert remaining
    if batch_chunks:
        store.upsert_batch(batch_chunks, batch_embeddings)

    embedder.close()

    click.echo(f"\nIndexed {chunk_count} chunks from {len(files)} files")
    click.echo(f"Total documents in store: {store.get_stats()['count']}")


@main.command()
@click.argument("query")
@click.option("--limit", "-n", default=5, help="Number of results")
@click.option("--type", "note_type", default=None, help="Filter by type (daily, note)")
@click.pass_context
def search(ctx, query, limit, note_type):
    """Search notes semantically."""
    data_path = ctx.obj["data"]
    provider = ctx.obj["provider"]
    ollama_url = ctx.obj["ollama_url"]
    model = ctx.obj["model"]

    # Initialize components
    embedder = create_embedder(provider=provider, model=model, base_url=ollama_url)
    store = VectorStore(data_path=data_path)

    # Generate query embedding
    click.echo(f"Searching for: {query}\n")
    query_embedding = embedder.embed(query)

    # Build filter
    where = None
    if note_type:
        where = {"type": note_type}

    # Search
    results = store.search(query_embedding, limit=limit, where=where)

    if not results:
        click.echo("No results found.")
        return

    # Display results
    for i, result in enumerate(results, 1):
        meta = result["metadata"]
        distance = result["distance"]
        similarity = 1 - distance  # Cosine distance to similarity

        click.echo(f"{'─' * 60}")
        click.echo(f"[{i}] {meta['file_path']}")
        if meta.get("heading"):
            click.echo(f"    Section: {meta['heading']}")
        click.echo(f"    Type: {meta.get('type', 'note')} | Similarity: {similarity:.2%}")
        click.echo()

        # Show truncated content
        content = result["content"]
        if len(content) > 300:
            content = content[:300] + "..."
        click.echo(f"    {content}")
        click.echo()

    embedder.close()


@main.command()
@click.pass_context
def stats(ctx):
    """Show index statistics."""
    data_path = ctx.obj["data"]
    store = VectorStore(data_path=data_path)

    stats = store.get_stats()
    click.echo(f"Collection: {stats['collection']}")
    click.echo(f"Documents: {stats['count']}")
    click.echo(f"Data path: {stats['data_path']}")


@main.command()
@click.option("--debounce", default=2.0, help="Seconds to wait before processing changes")
@click.pass_context
def watch(ctx, debounce):
    """Watch vault for changes and auto-reindex."""
    vault_path = ctx.obj["vault"]
    data_path = ctx.obj["data"]
    provider = ctx.obj["provider"]
    ollama_url = ctx.obj["ollama_url"]
    model = ctx.obj["model"]

    click.echo(f"Watching vault: {vault_path}")
    click.echo(f"Data path: {data_path}")
    click.echo(f"Provider: {provider}")
    click.echo(f"Debounce: {debounce}s")
    click.echo("Press Ctrl+C to stop.\n")

    watcher = VaultWatcher(
        vault_path=vault_path,
        data_path=data_path,
        provider=provider,
        ollama_url=ollama_url,
        model=model,
        debounce_delay=debounce,
    )
    watcher.run_forever()


@main.command()
def serve():
    """Start the MCP server (for Claude Code integration)."""
    run_server()


# Service management
# TODO: Add Linux systemd support (create .service file in ~/.config/systemd/user/)
# TODO: Add Windows Task Scheduler support (use schtasks or win32api)
PLIST_NAME = "com.obsidian-rag.watcher.plist"
LAUNCH_AGENTS_DIR = Path.home() / "Library" / "LaunchAgents"


def _get_plist_content(vault_path: str, data_path: str, provider: str, ollama_url: str, model: str | None) -> str:
    """Generate launchd plist content."""
    # Find the obsidian-memory-watch executable
    import sys
    python_path = sys.executable

    # Build environment variables section
    env_vars = f"""        <key>OBSIDIAN_RAG_VAULT</key>
        <string>{vault_path}</string>
        <key>OBSIDIAN_RAG_DATA</key>
        <string>{data_path}</string>
        <key>OBSIDIAN_RAG_PROVIDER</key>
        <string>{provider}</string>"""

    if provider == "ollama":
        env_vars += f"""
        <key>OBSIDIAN_RAG_OLLAMA_URL</key>
        <string>{ollama_url}</string>"""

    if model:
        env_vars += f"""
        <key>OBSIDIAN_RAG_MODEL</key>
        <string>{model}</string>"""

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.obsidian-rag.watcher</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python_path}</string>
        <string>-m</string>
        <string>obsidian_rag.watcher</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
{env_vars}
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/obsidian-rag.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/obsidian-rag.err</string>
    <key>WorkingDirectory</key>
    <string>{Path.cwd()}</string>
</dict>
</plist>
"""


@main.command("install-service")
@click.pass_context
def install_service(ctx):
    """Install launchd service for auto-start on macOS."""
    if sys.platform != "darwin":
        # TODO: Implement Linux systemd and Windows Task Scheduler support
        click.echo("Error: This command currently only supports macOS. Linux/Windows support planned.", err=True)
        sys.exit(1)

    vault_path = ctx.obj["vault"]
    data_path = ctx.obj["data"]
    provider = ctx.obj["provider"]
    ollama_url = ctx.obj["ollama_url"]
    model = ctx.obj["model"]

    plist_path = LAUNCH_AGENTS_DIR / PLIST_NAME

    # Create LaunchAgents directory if needed
    LAUNCH_AGENTS_DIR.mkdir(parents=True, exist_ok=True)

    # Unload existing service if present
    if plist_path.exists():
        click.echo("Unloading existing service...")
        subprocess.run(["launchctl", "unload", str(plist_path)], capture_output=True)

    # Write plist
    plist_content = _get_plist_content(vault_path, data_path, provider, ollama_url, model)
    plist_path.write_text(plist_content)
    click.echo(f"Created: {plist_path}")

    # Load service
    result = subprocess.run(["launchctl", "load", str(plist_path)], capture_output=True, text=True)
    if result.returncode != 0:
        click.echo(f"Error loading service: {result.stderr}", err=True)
        sys.exit(1)

    click.echo("Service installed and started.")
    click.echo("Logs: /tmp/obsidian-rag.log")
    click.echo("Errors: /tmp/obsidian-rag.err")


@main.command("uninstall-service")
def uninstall_service():
    """Uninstall launchd service on macOS."""
    if sys.platform != "darwin":
        # TODO: Implement Linux systemd and Windows Task Scheduler support
        click.echo("Error: This command currently only supports macOS. Linux/Windows support planned.", err=True)
        sys.exit(1)

    plist_path = LAUNCH_AGENTS_DIR / PLIST_NAME

    if not plist_path.exists():
        click.echo("Service not installed.")
        return

    # Unload service
    result = subprocess.run(["launchctl", "unload", str(plist_path)], capture_output=True, text=True)
    if result.returncode != 0:
        click.echo(f"Warning: Error unloading service: {result.stderr}", err=True)

    # Remove plist
    plist_path.unlink()
    click.echo("Service uninstalled.")


@main.command("service-status")
def service_status():
    """Check launchd service status on macOS."""
    if sys.platform != "darwin":
        # TODO: Implement Linux systemd and Windows Task Scheduler support
        click.echo("Error: This command currently only supports macOS. Linux/Windows support planned.", err=True)
        sys.exit(1)

    plist_path = LAUNCH_AGENTS_DIR / PLIST_NAME

    if not plist_path.exists():
        click.echo("Service not installed.")
        return

    result = subprocess.run(
        ["launchctl", "list", "com.obsidian-rag.watcher"],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        click.echo("Service is running.")
        click.echo(result.stdout)
    else:
        click.echo("Service is installed but not running.")


if __name__ == "__main__":
    main()
