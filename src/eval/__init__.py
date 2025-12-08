"""Evaluation modules for LegalAdapter."""

from .evaluate import evaluate_predictions
from .tables import generate_tables
from .plots import generate_plots

__all__ = [
    "evaluate_predictions",
    "generate_tables",
    "generate_plots",
]


