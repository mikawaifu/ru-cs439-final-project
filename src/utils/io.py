"""I/O utilities for LegalAdapter."""

import json
import yaml
from pathlib import Path
from typing import Any, Dict, List, Union


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load YAML configuration file.
    
    Args:
        path: Path to YAML file
        
    Returns:
        Dictionary containing configuration
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        path: Output file path
        indent: JSON indentation level
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(path: Union[str, Path]) -> Any:
    """
    Load data from JSON file.
    
    Args:
        path: Path to JSON file
        
    Returns:
        Loaded data
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data


def load_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load data from JSONL file.
    
    Args:
        path: Path to JSONL file
        
    Returns:
        List of dictionaries
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    
    return data


def save_jsonl(data: List[Dict[str, Any]], path: Union[str, Path]) -> None:
    """
    Save data to JSONL file.
    
    Args:
        data: List of dictionaries to save
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


