"""Command-line interface for obsidian-notes-rag."""

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
from .indexer import create_embedder, VaultIndexer, is_ollama_running, get_ollama_models, is_lmstudio_running, get_lmstudio_models
from .links import expand_neighbors
from .server import run_server
from .store import VectorStore
from .watcher import VaultWatcher


def _store_link_graph(indexer: VaultIndexer, store: VectorStore) -> int:
    """Extract and persist the vault's link graph; returns edge count."""
    edge_count = 0
    for source, targets in indexer.link_graph().items():
        store.replace_links(source, sorted(targets))
        edge_count += len(targets)
    return edge_count


def _note_preview(store: VectorStore, path: str, max_len: int = 200) -> str:
    """First-chunk preview for a note, or a placeholder if unindexed."""
    chunks = store.get_by_file(path)
    if not chunks:
        return "(not indexed)"
    content = chunks[0]["content"]
    return content[:max_len] + "..." if len(content) > max_len else content

def _require_vault(ctx) -> str:
    """Return the configured vault path or exit with guidance."""
    vault = ctx.obj["vault"]
    if not vault:
        click.echo(
            "No vault configured. Run 'obsidian-rag setup' or pass --vault.",
            err=True,
        )
        sys.exit(1)
    return vault


@click.group()
@click.option("--vault", "--root", "vault", default=None,
              help="Path to the notes root — an Obsidian vault, an OKF bundle, or any folder of markdown")
@click.option("--data", default=None, help="Path to vector store data")
@click.option("--provider", default=None,
              type=click.Choice(["openai", "ollama", "lmstudio"]),
              help="Embedding provider (default: openai)")
@click.option("--ollama-url", default=None,
              help="Ollama API URL (only used with --provider ollama)")
@click.option("--lmstudio-url", default=None,
              help="LM Studio API URL (only used with --provider lmstudio)")
@click.option("--ollama-api-key", default=None,
              help="Bearer token for Ollama (only used with --provider ollama)")
@click.option("--lmstudio-api-key", default=None,
              help="Bearer token for LM Studio (only used with --provider lmstudio)")
@click.option("--model", default=None, help="Override embedding model name")
@click.pass_context
def main(ctx, vault, data, provider, ollama_url, lmstudio_url, ollama_api_key, lmstudio_api_key, model):
    """Obsidian RAG - Semantic search for your Obsidian vault."""
    ctx.ensure_object(dict)

    # Load config from file, then apply CLI overrides
    config = load_config()

    ctx.obj["vault"] = vault or config.vault_path
    ctx.obj["data"] = data or config.get_data_path()
    ctx.obj["provider"] = provider or config.provider
    ctx.obj["ollama_url"] = ollama_url or config.ollama_url
    ctx.obj["lmstudio_url"] = lmstudio_url or config.lmstudio_url
    ctx.obj["ollama_api_key"] = ollama_api_key or config.get_ollama_api_key()
    ctx.obj["lmstudio_api_key"] = lmstudio_api_key or config.get_lmstudio_api_key()
    ctx.obj["model"] = model  # None means use provider default
    ctx.obj["config"] = config


@main.command()
def setup():
    """Interactive setup wizard for obsidian-notes-rag."""
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
    click.echo("  3. LM Studio (local, offline)")
    provider_choice = click.prompt("Choice", type=click.Choice(["1", "2", "3"]), default="1")
    if provider_choice == "1":
        config.provider = "openai"
    elif provider_choice == "2":
        config.provider = "ollama"
    else:
        config.provider = "lmstudio"

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
    elif config.provider == "ollama":
        # Ollama setup - check connection first
        default_ollama_url = "http://localhost:11434"
        ollama_url = click.prompt(
            "\nOllama API URL",
            default=default_ollama_url
        )
        config.ollama_url = ollama_url
        
        # Optional Bearer token
        if click.confirm("\nDoes your Ollama instance require a Bearer token?", default=False):
            config.ollama_api_key = click.prompt("Enter Bearer token", hide_input=False)

        # Verify connection and get available models
        click.echo("Checking Ollama server...", nl=False)
        server_running = is_ollama_running(ollama_url, api_key=config.ollama_api_key)
        
        if server_running:
            click.echo(" ✓ connected")
            
            # Fetch available embedding models
            click.echo("Fetching available embedding models...", nl=False)
            available_models = get_ollama_models(ollama_url, api_key=config.ollama_api_key)
            
            if available_models:
                click.echo(f" found {len(available_models)}")
                click.echo("\nSelect embedding model:")
                for i, model in enumerate(available_models, 1):
                    click.echo(f"  {i}. {model}")
                click.echo(f"  {len(available_models) + 1}. Other (enter model name)")
                
                choices = [str(i) for i in range(1, len(available_models) + 2)]
                model_choice = click.prompt("Choice", type=click.Choice(choices), default="1")
                choice_idx = int(model_choice) - 1
                
                if choice_idx < len(available_models):
                    config.ollama_model = available_models[choice_idx]
                else:
                    config.ollama_model = click.prompt("Enter embedding model name")
            else:
                click.echo(" none found")
                click.echo("\nNo embedding models detected. Install nomic-embed-text:")
                click.echo("  ollama pull nomic-embed-text")
                config.ollama_model = click.prompt("\nEnter embedding model name", default="nomic-embed-text")
        else:
            click.echo(" not detected (server may still work)")
            click.echo("Could not auto-detect models.")
            config.ollama_model = click.prompt("\nEnter embedding model name", default="nomic-embed-text")
    else:
        # LM Studio setup - check connection after getting URL
        default_lmstudio_url = "http://localhost:1234"
        lmstudio_url = click.prompt(
            "\nLM Studio API URL",
            default=default_lmstudio_url
        )
        config.lmstudio_url = lmstudio_url
        
        # Optional Bearer token
        if click.confirm("\nDoes your LM Studio instance require a Bearer token?", default=False):
            config.lmstudio_api_key = click.prompt("Enter Bearer token", hide_input=False)

        # Verify connection and get available models
        click.echo("Checking LM Studio server...", nl=False)
        server_running = is_lmstudio_running(lmstudio_url, api_key=config.lmstudio_api_key)
        
        if server_running:
            click.echo(" ✓ connected")
            
            # Fetch available embedding models
            click.echo("Fetching available embedding models...", nl=False)
            available_models = get_lmstudio_models(lmstudio_url, api_key=config.lmstudio_api_key)
            
            if available_models:
                click.echo(f" found {len(available_models)}")
                click.echo("\nSelect embedding model:")
                for i, model in enumerate(available_models, 1):
                    click.echo(f"  {i}. {model}")
                click.echo(f"  {len(available_models) + 1}. Other (enter model identifier)")
                
                choices = [str(i) for i in range(1, len(available_models) + 2)]
                model_choice = click.prompt("Choice", type=click.Choice(choices), default="1")
                choice_idx = int(model_choice) - 1
                
                if choice_idx < len(available_models):
                    config.lmstudio_model = available_models[choice_idx]
                else:
                    config.lmstudio_model = click.prompt("Enter embedding model identifier")
            else:
                click.echo(" none found")
                click.echo("\nNo embedding models detected.")
                config.lmstudio_model = click.prompt("Enter embedding model identifier")
        else:
            click.echo(" not detected (server may still work)")
            click.echo("Could not auto-detect models.")
            config.lmstudio_model = click.prompt("Enter embedding model identifier")

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
                embedder = create_embedder(provider="openai", model=config.openai_model)
            elif config.provider == "ollama":
                embedder = create_embedder(
                    provider="ollama",
                    model=config.ollama_model,
                    base_url=config.ollama_url,
                    api_key=config.get_ollama_api_key(),
                )
            else:  # lmstudio
                embedder = create_embedder(
                    provider="lmstudio",
                    model=config.lmstudio_model,
                    base_url=config.lmstudio_url,
                    api_key=config.get_lmstudio_api_key(),
                )

            store = VectorStore(data_path=config.get_data_path())
            indexer = VaultIndexer(vault_path=config.vault_path, embedder=embedder, config=config.indexer)

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
            edge_count = _store_link_graph(indexer, store)
            click.echo(f"\n✓ Indexed {chunk_count} chunks from {len(files)} files ({edge_count} link edges)")

        except Exception as e:
            click.echo(f"\n✗ Indexing failed: {e}", err=True)
            click.echo("You can run indexing later with: obsidian-notes-rag index")

    # 7. Offer to install watcher service
    click.echo("\nThe watcher service auto-indexes notes when they change.")
    if sys.platform == "darwin":
        if click.confirm("Install watcher as a background service?", default=True):
            try:
                plist_path = LAUNCH_AGENTS_DIR / PLIST_NAME
                LAUNCH_AGENTS_DIR.mkdir(parents=True, exist_ok=True)

                # Unload existing service if present
                if plist_path.exists():
                    subprocess.run(["launchctl", "unload", str(plist_path)], capture_output=True)

                # Install wrapper script (shows descriptive name in System Settings)
                _install_wrapper_script()

                # Write plist with config values
                assert config.vault_path is not None  # Set in step 3
                plist_content = _get_plist_content(
                    config.vault_path,
                    config.data_path or str(get_data_dir()),
                    config.provider,
                    config.ollama_url,
                    None,  # model
                    config.get_ollama_api_key(),
                    config.get_lmstudio_api_key(),
                )
                plist_path.write_text(plist_content)

                # Load service
                result = subprocess.run(["launchctl", "load", str(plist_path)], capture_output=True, text=True)
                if result.returncode != 0:
                    click.echo(f"✗ Error starting service: {result.stderr}", err=True)
                else:
                    click.echo("✓ Watcher service installed and started")
                    click.echo("  Logs: /tmp/obsidian-notes-rag.log")
            except Exception as e:
                click.echo(f"✗ Service installation failed: {e}", err=True)
                click.echo("  You can install later with: obsidian-notes-rag install-service")
    else:
        # Linux/Windows: no background service support yet
        click.echo("  Background service not yet supported on this platform.")
        click.echo("  To auto-index on file changes, run: obsidian-notes-rag watch")

    click.echo("\nSetup complete! You can now:")
    click.echo("  - Search: obsidian-notes-rag search \"your query\"")
    click.echo("  - Add to Claude Code:")
    click.echo("      claude mcp add -s user obsidian-notes-rag -- uvx obsidian-notes-rag serve")


@main.command()
@click.option("--clear", is_flag=True, help="Clear existing index before indexing")
@click.option("--path-filter", default=None, help="Only index files under this path prefix (e.g. 'Daily Notes/')")
@click.pass_context
def index(ctx, clear, path_filter):
    """Index all markdown files in the vault."""
    vault_path = _require_vault(ctx)
    data_path = ctx.obj["data"]
    provider = ctx.obj["provider"]
    ollama_url = ctx.obj["ollama_url"]
    lmstudio_url = ctx.obj["lmstudio_url"]
    ollama_api_key = ctx.obj["ollama_api_key"]
    lmstudio_api_key = ctx.obj["lmstudio_api_key"]
    config = ctx.obj["config"]

    # Get model from CLI override or config file based on provider
    model = ctx.obj["model"]
    if model is None:
        if provider == "openai":
            model = config.openai_model
        elif provider == "ollama":
            model = config.ollama_model
        elif provider == "lmstudio":
            model = config.lmstudio_model

    click.echo(f"Indexing vault: {vault_path}")
    click.echo(f"Data path: {data_path}")
    click.echo(f"Provider: {provider}")
    click.echo(f"Model: {model}")

    # Determine the correct base_url and api_key based on provider
    if provider == "ollama":
        base_url = ollama_url
        api_key = ollama_api_key
    elif provider == "lmstudio":
        base_url = lmstudio_url
        api_key = lmstudio_api_key
    else:
        base_url = None
        api_key = None

    # Initialize components
    embedder = create_embedder(provider=provider, model=model, base_url=base_url, api_key=api_key)
    store = VectorStore(data_path=data_path)
    indexer = VaultIndexer(vault_path=vault_path, embedder=embedder, config=config.indexer)

    if clear:
        click.echo("Clearing existing index...")
        store.clear()

    # Count files first
    files = list(indexer.iter_markdown_files())
    click.echo(f"Found {len(files)} markdown files")

    if path_filter:
        files = [f for f in files if str(f.relative_to(indexer.vault_path)).startswith(path_filter)]
        click.echo(f"Filtered to {len(files)} files matching '{path_filter}'")

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

    click.echo("Extracting link graph...")
    edge_count = _store_link_graph(indexer, store)

    click.echo(f"\nIndexed {chunk_count} chunks from {len(files)} files")
    click.echo(f"Stored {edge_count} link edges")
    click.echo(f"Total documents in store: {store.get_stats()['count']}")


@main.command()
@click.argument("query")
@click.option("--limit", "-n", default=5, help="Number of results")
@click.option("--type", "note_type", default=None, help="Filter by type (daily, note)")
@click.option("--expand", "-e", default=0, help="Expand results N hops along the vault's link graph")
@click.option("--expand-limit", default=10, help="Max graph neighbors to include (default: 10)")
@click.pass_context
def search(ctx, query, limit, note_type, expand, expand_limit):
    """Search notes semantically."""
    data_path = ctx.obj["data"]
    provider = ctx.obj["provider"]
    ollama_url = ctx.obj["ollama_url"]
    lmstudio_url = ctx.obj["lmstudio_url"]
    ollama_api_key = ctx.obj["ollama_api_key"]
    lmstudio_api_key = ctx.obj["lmstudio_api_key"]
    config = ctx.obj["config"]

    # Get model from CLI override or config file based on provider
    model = ctx.obj["model"]
    if model is None:
        if provider == "openai":
            model = config.openai_model
        elif provider == "ollama":
            model = config.ollama_model
        elif provider == "lmstudio":
            model = config.lmstudio_model

    # Determine the correct base_url and api_key based on provider
    if provider == "ollama":
        base_url = ollama_url
        api_key = ollama_api_key
    elif provider == "lmstudio":
        base_url = lmstudio_url
        api_key = lmstudio_api_key
    else:
        base_url = None
        api_key = None

    # Initialize components
    embedder = create_embedder(provider=provider, model=model, base_url=base_url, api_key=api_key)
    store = VectorStore(data_path=data_path)

    # Generate query embedding
    click.echo(f"Searching for: {query}\n")
    query_embedding = embedder.embed(query, task_type="search_query")

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

    # Graph expansion: follow the vault's actual links out from the hits
    if expand > 0:
        seeds = list(dict.fromkeys(r["metadata"]["file_path"] for r in results))
        neighbors = expand_neighbors(store, seeds, hops=expand, limit=expand_limit)
        if neighbors:
            click.echo(f"{'═' * 60}")
            click.echo(f"Graph context ({len(neighbors)} notes within {expand} hop{'s' if expand > 1 else ''})")
            click.echo(f"{'═' * 60}")
            for i, nb in enumerate(neighbors, 1):
                arrow = "→" if nb.direction == "link" else "←"
                click.echo(f"[{i}] {nb.path}")
                click.echo(f"    hop {nb.hop} {arrow} via {nb.via} ({nb.direction})")
                click.echo(f"    {_note_preview(store, nb.path)}")
                click.echo()

    embedder.close()


@main.command()
@click.argument("note_path")
@click.option("--limit", "-n", default=5, help="Number of similar notes")
@click.pass_context
def similar(ctx, note_path, limit):
    """Find notes similar to a given note."""
    data_path = ctx.obj["data"]
    provider = ctx.obj["provider"]
    ollama_url = ctx.obj["ollama_url"]
    lmstudio_url = ctx.obj["lmstudio_url"]
    ollama_api_key = ctx.obj["ollama_api_key"]
    lmstudio_api_key = ctx.obj["lmstudio_api_key"]
    config = ctx.obj["config"]

    model = ctx.obj["model"]
    if model is None:
        if provider == "openai":
            model = config.openai_model
        elif provider == "ollama":
            model = config.ollama_model
        elif provider == "lmstudio":
            model = config.lmstudio_model

    if provider == "ollama":
        base_url = ollama_url
        api_key = ollama_api_key
    elif provider == "lmstudio":
        base_url = lmstudio_url
        api_key = lmstudio_api_key
    else:
        base_url = None
        api_key = None

    embedder = create_embedder(provider=provider, model=model, base_url=base_url, api_key=api_key)
    store = VectorStore(data_path=data_path)

    click.echo(f"Finding notes similar to: {note_path}\n")
    results = store.get_by_file(note_path)

    if not results:
        click.echo(f"Note not found in index: {note_path}")
        embedder.close()
        return

    note_content = "\n\n".join(r["content"] for r in results)
    note_embedding = embedder.embed(note_content[:8000])
    all_results = store.search(note_embedding, limit=limit + 10)

    similar_notes = [r for r in all_results if r["metadata"]["file_path"] != note_path][:limit]

    if not similar_notes:
        click.echo("No similar notes found.")
        embedder.close()
        return

    for i, result in enumerate(similar_notes, 1):
        meta = result["metadata"]
        similarity = 1 - result["distance"]
        click.echo(f"{'─' * 60}")
        click.echo(f"[{i}] {meta['file_path']}")
        if meta.get("heading"):
            click.echo(f"    Section: {meta['heading']}")
        click.echo(f"    Similarity: {similarity:.2%}")
        content = result["content"]
        if len(content) > 200:
            content = content[:200] + "..."
        click.echo(f"\n    {content}\n")

    embedder.close()


@main.command()
@click.argument("note_path")
@click.option("--limit", "-n", default=5, help="Number of similar notes to include")
@click.pass_context
def context(ctx, note_path, limit):
    """Get a note and its related context."""
    data_path = ctx.obj["data"]
    provider = ctx.obj["provider"]
    ollama_url = ctx.obj["ollama_url"]
    lmstudio_url = ctx.obj["lmstudio_url"]
    ollama_api_key = ctx.obj["ollama_api_key"]
    lmstudio_api_key = ctx.obj["lmstudio_api_key"]
    config = ctx.obj["config"]

    model = ctx.obj["model"]
    if model is None:
        if provider == "openai":
            model = config.openai_model
        elif provider == "ollama":
            model = config.ollama_model
        elif provider == "lmstudio":
            model = config.lmstudio_model

    if provider == "ollama":
        base_url = ollama_url
        api_key = ollama_api_key
    elif provider == "lmstudio":
        base_url = lmstudio_url
        api_key = lmstudio_api_key
    else:
        base_url = None
        api_key = None

    embedder = create_embedder(provider=provider, model=model, base_url=base_url, api_key=api_key)
    store = VectorStore(data_path=data_path)

    click.echo(f"Getting context for: {note_path}\n")
    results = store.get_by_file(note_path)

    if not results:
        click.echo(f"Note not found in index: {note_path}")
        embedder.close()
        return

    # Display note content
    note_content = "\n\n".join(r["content"] for r in results)
    click.echo(f"{'═' * 60}")
    click.echo(f"Note: {note_path}")
    click.echo(f"{'═' * 60}")
    click.echo(note_content)
    click.echo()

    # Graph neighbors: the note's actual links and backlinks
    links = store.get_links(note_path)
    backlinks = store.get_backlinks(note_path)
    if links or backlinks:
        click.echo(f"{'─' * 60}")
        click.echo(f"Linked Notes ({len(links)} links, {len(backlinks)} backlinks)")
        click.echo(f"{'─' * 60}")
        for target in sorted(links):
            click.echo(f"  → {target}")
        for source in sorted(backlinks):
            click.echo(f"  ← {source}")
        click.echo()

    # Find similar notes
    note_embedding = embedder.embed(note_content[:8000])
    all_results = store.search(note_embedding, limit=limit + 10)
    similar_notes = [r for r in all_results if r["metadata"]["file_path"] != note_path][:limit]

    if similar_notes:
        click.echo(f"{'─' * 60}")
        click.echo(f"Related Notes ({len(similar_notes)})")
        click.echo(f"{'─' * 60}")
        for i, result in enumerate(similar_notes, 1):
            meta = result["metadata"]
            similarity = 1 - result["distance"]
            click.echo(f"  [{i}] {meta['file_path']}")
            if meta.get("heading"):
                click.echo(f"      Section: {meta['heading']}")
            click.echo(f"      Similarity: {similarity:.2%}")
    else:
        click.echo("No related notes found.")

    embedder.close()


@main.command()
@click.argument("note_path")
@click.option("--hops", "-n", default=1, help="Traversal depth (default: 1)")
@click.option("--limit", default=20, help="Max neighbors to show (default: 20)")
@click.pass_context
def graph(ctx, note_path, hops, limit):
    """Show a note's link-graph neighborhood (links and backlinks)."""
    data_path = ctx.obj["data"]
    store = VectorStore(data_path=data_path)

    links = store.get_links(note_path)
    backlinks = store.get_backlinks(note_path)

    if not links and not backlinks and not store.get_by_file(note_path):
        click.echo(f"Note not found in index: {note_path}")
        return

    click.echo(f"{'═' * 60}")
    click.echo(f"Graph neighborhood: {note_path}")
    click.echo(f"{'═' * 60}")
    click.echo(f"\nLinks ({len(links)}):")
    for target in sorted(links):
        click.echo(f"  → {target}")
    click.echo(f"\nBacklinks ({len(backlinks)}):")
    for source in sorted(backlinks):
        click.echo(f"  ← {source}")

    if hops > 1:
        neighbors = expand_neighbors(store, [note_path], hops=hops, limit=limit)
        beyond = [nb for nb in neighbors if nb.hop > 1]
        if beyond:
            click.echo(f"\nBeyond 1 hop ({len(beyond)}):")
            for nb in beyond:
                arrow = "→" if nb.direction == "link" else "←"
                click.echo(f"  hop {nb.hop} {arrow} {nb.path} (via {nb.via})")
    click.echo()


@main.command()
@click.pass_context
def stats(ctx):
    """Show index statistics."""
    data_path = ctx.obj["data"]
    store = VectorStore(data_path=data_path)

    stats = store.get_stats()
    click.echo(f"Collection: {stats['collection']}")
    click.echo(f"Documents: {stats['count']}")
    click.echo(f"Link edges: {store.get_link_stats()['links']}")
    click.echo(f"Data path: {stats['data_path']}")


@main.command()
@click.option("--debounce", default=2.0, help="Seconds to wait before processing changes")
@click.pass_context
def watch(ctx, debounce):
    """Watch vault for changes and auto-reindex."""
    vault_path = _require_vault(ctx)
    data_path = ctx.obj["data"]
    provider = ctx.obj["provider"]
    ollama_url = ctx.obj["ollama_url"]
    lmstudio_url = ctx.obj["lmstudio_url"]
    ollama_api_key = ctx.obj["ollama_api_key"]
    lmstudio_api_key = ctx.obj["lmstudio_api_key"]
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
        lmstudio_url=lmstudio_url,
        ollama_api_key=ollama_api_key,
        lmstudio_api_key=lmstudio_api_key,
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
PLIST_NAME = "com.obsidian-notes-rag.watcher.plist"
LAUNCH_AGENTS_DIR = Path.home() / "Library" / "LaunchAgents"
WRAPPER_SCRIPT_DIR = Path.home() / ".local" / "bin"
WRAPPER_SCRIPT_NAME = "obsidian-notes-rag-watcher"
LOG_DIR = Path.home() / "Library" / "Logs" / "obsidian-notes-rag"


def _get_wrapper_script_content() -> str:
    """Generate wrapper script that calls the watcher module."""
    import sys
    python_path = sys.executable
    return f"""#!/bin/bash
# Wrapper script for obsidian-notes-rag watcher service
# This script exists so macOS System Settings shows a descriptive name
# instead of "python3" in Login Items & Extensions.
exec "{python_path}" -m obsidian_rag.watcher "$@"
"""


def _install_wrapper_script() -> Path:
    """Install the wrapper script and return its path."""
    WRAPPER_SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    wrapper_path = WRAPPER_SCRIPT_DIR / WRAPPER_SCRIPT_NAME
    wrapper_path.write_text(_get_wrapper_script_content())
    wrapper_path.chmod(0o755)
    return wrapper_path


def _uninstall_wrapper_script():
    """Remove the wrapper script if it exists."""
    wrapper_path = WRAPPER_SCRIPT_DIR / WRAPPER_SCRIPT_NAME
    if wrapper_path.exists():
        wrapper_path.unlink()


def _get_plist_content(vault_path: str, data_path: str, provider: str, ollama_url: str, model: str | None, ollama_api_key: str | None = None, lmstudio_api_key: str | None = None) -> str:
    """Generate launchd plist content."""
    # Use wrapper script for better System Settings appearance
    wrapper_path = WRAPPER_SCRIPT_DIR / WRAPPER_SCRIPT_NAME

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
        if ollama_api_key:
            env_vars += f"""
        <key>OBSIDIAN_RAG_OLLAMA_API_KEY</key>
        <string>{ollama_api_key}</string>"""
    elif provider == "lmstudio" and lmstudio_api_key:
        env_vars += f"""
        <key>OBSIDIAN_RAG_LMSTUDIO_API_KEY</key>
        <string>{lmstudio_api_key}</string>"""

    if model:
        env_vars += f"""
        <key>OBSIDIAN_RAG_MODEL</key>
        <string>{model}</string>"""

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.obsidian-notes-rag.watcher</string>
    <key>ProgramArguments</key>
    <array>
        <string>{wrapper_path}</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
{env_vars}
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>ThrottleInterval</key>
    <integer>30</integer>
    <key>StandardOutPath</key>
    <string>{LOG_DIR}/watcher.log</string>
    <key>StandardErrorPath</key>
    <string>{LOG_DIR}/watcher.err</string>
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

    vault_path = _require_vault(ctx)
    data_path = ctx.obj["data"]
    provider = ctx.obj["provider"]
    ollama_url = ctx.obj["ollama_url"]
    ollama_api_key = ctx.obj["ollama_api_key"]
    lmstudio_api_key = ctx.obj["lmstudio_api_key"]
    model = ctx.obj["model"]

    plist_path = LAUNCH_AGENTS_DIR / PLIST_NAME

    # Create directories if needed
    LAUNCH_AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Unload existing service if present
    if plist_path.exists():
        click.echo("Unloading existing service...")
        subprocess.run(["launchctl", "unload", str(plist_path)], capture_output=True)

    # Install wrapper script (shows descriptive name in System Settings)
    wrapper_path = _install_wrapper_script()
    click.echo(f"Created: {wrapper_path}")

    # Write plist
    plist_content = _get_plist_content(vault_path, data_path, provider, ollama_url, model, ollama_api_key, lmstudio_api_key)
    plist_path.write_text(plist_content)
    click.echo(f"Created: {plist_path}")

    # Load service
    result = subprocess.run(["launchctl", "load", str(plist_path)], capture_output=True, text=True)
    if result.returncode != 0:
        click.echo(f"Error loading service: {result.stderr}", err=True)
        sys.exit(1)

    click.echo("Service installed and started.")
    click.echo(f"Logs: {LOG_DIR}/watcher.log")
    click.echo(f"Errors: {LOG_DIR}/watcher.err")


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

    # Remove wrapper script
    _uninstall_wrapper_script()

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
        ["launchctl", "list", "com.obsidian-notes-rag.watcher"],
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
