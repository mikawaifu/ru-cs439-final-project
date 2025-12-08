"""Data processing modules for LegalAdapter."""

from .schemas import Question, Candidate, ScoredCandidate
from .loaders import load_dataset, DatasetLoader

__all__ = [
    "Question",
    "Candidate",
    "ScoredCandidate",
    "load_dataset",
    "DatasetLoader",
]


