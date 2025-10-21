"""Tests for configuration module."""

import os
import pytest
from config import OpenAIConfig, EmbeddingConfig, AppConfig


class TestOpenAIConfig:
    """Tests for OpenAI configuration."""

    def test_from_env_missing_key(self, monkeypatch):
        """Test that missing API key raises ValueError."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(ValueError) as exc_info:
            OpenAIConfig.from_env()

        assert "OPENAI_API_KEY" in str(exc_info.value)

    def test_from_env_with_key(self, monkeypatch):
        """Test loading config from environment."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4")
        monkeypatch.setenv("OPENAI_MAX_TOKENS", "2000")

        config = OpenAIConfig.from_env()

        assert config.api_key == "test-key-123"
        assert config.model == "gpt-4"
        assert config.max_tokens == 2000

    def test_from_env_defaults(self, monkeypatch):
        """Test that defaults are used when env vars not set."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
        monkeypatch.delenv("OPENAI_MODEL", raising=False)
        monkeypatch.delenv("OPENAI_MAX_TOKENS", raising=False)

        config = OpenAIConfig.from_env()

        assert config.api_key == "test-key-123"
        assert config.model == "gpt-3.5-turbo"  # default
        assert config.max_tokens == 1000  # default


class TestEmbeddingConfig:
    """Tests for embedding configuration."""

    def test_defaults(self):
        """Test default configuration values."""
        config = EmbeddingConfig()

        assert config.trimming_tolerance == 1e-10
        assert config.max_passes == 100
        assert config.convergence_threshold == 0.01
        assert config.update_power == 6
        assert config.lower_bound_epsilon == 0.05


class TestAppConfig:
    """Tests for application configuration."""

    def test_load_with_api_key(self, monkeypatch):
        """Test loading full app config."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")

        config = AppConfig.load()

        assert config.openai.api_key == "test-key-123"
        assert isinstance(config.embedding, EmbeddingConfig)
        assert isinstance(config.processing, ProcessingConfig)
