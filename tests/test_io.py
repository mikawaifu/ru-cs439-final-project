"""Tests for I/O utilities."""

import pytest
import json
import tempfile
from pathlib import Path

from src.legaladapter.utils.io import (
    save_json,
    load_json,
    save_jsonl,
    load_jsonl,
)


def test_save_load_json():
    """Test JSON save and load."""
    data = {"key": "value", "number": 42, "list": [1, 2, 3]}
    
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "test.json"
        
        # Save
        save_json(data, file_path)
        assert file_path.exists()
        
        # Load
        loaded = load_json(file_path)
        assert loaded == data


def test_save_load_jsonl():
    """Test JSONL save and load."""
    data = [
        {"id": "1", "text": "First"},
        {"id": "2", "text": "Second"},
        {"id": "3", "text": "Third"},
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "test.jsonl"
        
        # Save
        save_jsonl(data, file_path)
        assert file_path.exists()
        
        # Load
        loaded = load_jsonl(file_path)
        assert loaded == data


def test_load_nonexistent_file():
    """Test loading nonexistent file raises error."""
    with pytest.raises(FileNotFoundError):
        load_json("nonexistent.json")
    
    with pytest.raises(FileNotFoundError):
        load_jsonl("nonexistent.jsonl")


