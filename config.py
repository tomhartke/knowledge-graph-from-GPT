"""
Configuration module for Knowledge Graph from GPT.

Centralizes all configuration values including API settings, model parameters,
and algorithm hyperparameters.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OpenAIConfig:
    """Configuration for OpenAI API."""

    api_key: str
    model: str = "gpt-3.5-turbo"
    max_tokens: int = 1000
    temperature: float = 0.0
    request_timeout: int = 60

    @classmethod
    def from_env(cls) -> 'OpenAIConfig':
        """Load configuration from environment variables.

        Returns:
            OpenAIConfig instance with values from environment

        Raises:
            ValueError: If OPENAI_API_KEY is not set
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set.\n\n"
                "To fix this, set your OpenAI API key:\n"
                "  export OPENAI_API_KEY='your-key-here'\n\n"
                "Or create a .env file in the project root with:\n"
                "  OPENAI_API_KEY=your-key-here\n"
            )

        return cls(
            api_key=api_key,
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "1000")),
            temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.0")),
        )


@dataclass
class EmbeddingConfig:
    """Configuration for embedding calculations."""

    # Trimming threshold for embedding vectors
    trimming_tolerance: float = 1e-10

    # Maximum number of passes for iterative embedding convergence
    max_passes: int = 100

    # Convergence threshold (fraction of updates)
    convergence_threshold: float = 0.01

    # Update combination power (higher = slower convergence)
    update_power: int = 6

    # Lower bound epsilon for statistical significance
    lower_bound_epsilon: float = 0.05


@dataclass
class ProcessingConfig:
    """Configuration for card processing."""

    # Print progress every N cards
    progress_interval: int = 20

    # Maximum tokens for card processing requests
    card_processing_max_tokens: int = 400

    # Batch size for parallel processing
    batch_size: int = 10


@dataclass
class AppConfig:
    """Main application configuration."""

    openai: OpenAIConfig = field(default_factory=OpenAIConfig.from_env)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)

    @classmethod
    def load(cls) -> 'AppConfig':
        """Load application configuration from environment.

        Returns:
            AppConfig instance with all settings
        """
        return cls(
            openai=OpenAIConfig.from_env(),
            embedding=EmbeddingConfig(),
            processing=ProcessingConfig()
        )


# Global configuration instance
# This can be imported and used throughout the application
try:
    config = AppConfig.load()
except ValueError as e:
    # If API key is not set, provide helpful message but don't crash on import
    print(f"Warning: Configuration not fully loaded: {e}")
    config = None
