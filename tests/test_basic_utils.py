"""Tests for basic utility functions."""

import pytest
import numpy as np
from basic_utils import get_dict_items_sorted_by_decreasing_value


class TestDictSorting:
    """Tests for dictionary sorting utilities."""

    def test_get_dict_items_sorted_by_decreasing_value(self):
        """Test sorting dictionary by values in decreasing order."""
        test_dict = {
            "apple": 3.5,
            "banana": 1.2,
            "cherry": 5.0,
            "date": 2.1
        }

        sorted_keys, sorted_values = get_dict_items_sorted_by_decreasing_value(test_dict)

        # Check ordering
        assert list(sorted_keys) == ["cherry", "apple", "date", "banana"]
        assert list(sorted_values) == [5.0, 3.5, 2.1, 1.2]

    def test_empty_dict(self):
        """Test with empty dictionary."""
        test_dict = {}

        sorted_keys, sorted_values = get_dict_items_sorted_by_decreasing_value(test_dict)

        assert len(sorted_keys) == 0
        assert len(sorted_values) == 0

    def test_single_item(self):
        """Test with single item dictionary."""
        test_dict = {"only": 42.0}

        sorted_keys, sorted_values = get_dict_items_sorted_by_decreasing_value(test_dict)

        assert list(sorted_keys) == ["only"]
        assert list(sorted_values) == [42.0]

    def test_equal_values(self):
        """Test with equal values (order may vary)."""
        test_dict = {"a": 5.0, "b": 5.0, "c": 5.0}

        sorted_keys, sorted_values = get_dict_items_sorted_by_decreasing_value(test_dict)

        # All values should be 5.0
        assert all(v == 5.0 for v in sorted_values)
        assert len(sorted_keys) == 3
