"""Configuration management for obsidian-rag."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[import-not-found]

import tomli_w
from platformdirs import user_config_dir, user_data_dir

APP_NAME = "obsidian-rag"


def get_config_dir() -> Path:
    """Get the configuration directory (cross-platform)."""
    return Path(user_config_dir(APP_NAME))


def get_data_dir() -> Path:
    """Get the data directory (cross-platform)."""
    return Path(user_data_dir(APP_NAME))


def get_config_path() -> Path:
    """Get the path to the config file."""
    return get_config_dir() / "config.toml"


@dataclass
class Config:
    """Application configuration."""

    # Core settings
    provider: str = "openai"
    vault_path: Optional[str] = None
    data_path: Optional[str] = None

    # OpenAI settings
    openai_api_key: Optional[str] = None

    # Ollama settings
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "nomic-embed-text"

    # OpenAI model (optional override)
    openai_model: str = "text-embedding-3-small"

    def get_data_path(self) -> str:
        """Get the data path, using default if not set."""
        return self.data_path or str(get_data_dir())

    def get_openai_api_key(self) -> Optional[str]:
        """Get OpenAI API key from config or environment."""
        return self.openai_api_key or os.environ.get("OPENAI_API_KEY")


def load_config() -> Config:
    """Load configuration from file with environment variable overrides.

    Priority (highest to lowest):
    1. Environment variables
    2. Config file
    3. Defaults
    """
    config = Config()

    # Load from config file if it exists
    config_path = get_config_path()
    if config_path.exists():
        try:
            with open(config_path, "rb") as f:
                data = tomllib.load(f)

            config.provider = data.get("provider", config.provider)
            config.vault_path = data.get("vault_path", config.vault_path)
            config.data_path = data.get("data_path", config.data_path)

            # OpenAI settings
            if "openai" in data:
                config.openai_api_key = data["openai"].get("api_key")
                config.openai_model = data["openai"].get("model", config.openai_model)

            # Ollama settings
            if "ollama" in data:
                config.ollama_url = data["ollama"].get("url", config.ollama_url)
                config.ollama_model = data["ollama"].get("model", config.ollama_model)

        except Exception:
            pass  # Use defaults if config file is invalid

    # Environment variable overrides
    if os.environ.get("OBSIDIAN_RAG_PROVIDER"):
        config.provider = os.environ["OBSIDIAN_RAG_PROVIDER"]
    if os.environ.get("OBSIDIAN_RAG_VAULT"):
        config.vault_path = os.environ["OBSIDIAN_RAG_VAULT"]
    if os.environ.get("OBSIDIAN_RAG_DATA"):
        config.data_path = os.environ["OBSIDIAN_RAG_DATA"]
    if os.environ.get("OBSIDIAN_RAG_OLLAMA_URL"):
        config.ollama_url = os.environ["OBSIDIAN_RAG_OLLAMA_URL"]
    if os.environ.get("OBSIDIAN_RAG_MODEL"):
        if config.provider == "ollama":
            config.ollama_model = os.environ["OBSIDIAN_RAG_MODEL"]
        else:
            config.openai_model = os.environ["OBSIDIAN_RAG_MODEL"]

    return config


def save_config(config: Config) -> Path:
    """Save configuration to file.

    Returns:
        Path to the saved config file.
    """
    config_path = get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    data: dict = {
        "provider": config.provider,
    }

    if config.vault_path:
        data["vault_path"] = config.vault_path
    if config.data_path:
        data["data_path"] = config.data_path

    # OpenAI settings
    if config.provider == "openai" or config.openai_api_key:
        openai_section: dict = {}
        if config.openai_api_key:
            openai_section["api_key"] = config.openai_api_key
        if config.openai_model != "text-embedding-3-small":
            openai_section["model"] = config.openai_model
        if openai_section:
            data["openai"] = openai_section

    # Ollama settings
    if config.provider == "ollama":
        ollama_section: dict = {}
        if config.ollama_url != "http://localhost:11434":
            ollama_section["url"] = config.ollama_url
        if config.ollama_model != "nomic-embed-text":
            ollama_section["model"] = config.ollama_model
        if ollama_section:
            data["ollama"] = ollama_section

    with open(config_path, "wb") as f:
        tomli_w.dump(data, f)

    return config_path
