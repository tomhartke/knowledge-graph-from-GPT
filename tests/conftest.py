"""Pytest configuration and fixtures."""

import pytest
import os


@pytest.fixture(autouse=True)
def env_setup(monkeypatch):
    """Set up environment variables for tests."""
    # Ensure no real API key is used in tests
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-for-testing")


@pytest.fixture
def sample_flashcard_pair():
    """Sample flashcard question/answer pair."""
    return {
        "question": "What is a neural network?",
        "answer": "A neural network is a series of algorithms that endeavors to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates."
    }


@pytest.fixture
def sample_embedding_vector():
    """Sample embedding vector."""
    return {
        "Machine Learning": 0.5,
        "Neural Network": 0.3,
        "Deep Learning": 0.2
    }
