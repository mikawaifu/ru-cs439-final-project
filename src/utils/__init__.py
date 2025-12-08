"""Utility functions for LegalAdapter."""

from .logging import setup_logging
from .seed import set_seed
from .io import load_yaml, save_json, load_json, load_jsonl, save_jsonl
from .metrics import compute_accuracy, compute_f1, compute_auc
from .hallucination import detect_hallucination

__all__ = [
    "setup_logging",
    "set_seed",
    "load_yaml",
    "save_json",
    "load_json",
    "load_jsonl",
    "save_jsonl",
    "compute_accuracy",
    "compute_f1",
    "compute_auc",
    "detect_hallucination",
]


