"""Tests for metrics utilities."""

import pytest
from src.legaladapter.utils.metrics import (
    compute_accuracy,
    compute_f1,
    compute_f1_batch,
    compute_auc,
)


def test_compute_accuracy_exact():
    """Test exact accuracy computation."""
    predictions = ["YES", "NO", "YES"]
    references = ["YES", "NO", "NO"]
    
    acc = compute_accuracy(predictions, references, strict=True, case_sensitive=False)
    assert acc == 2/3  # 2 correct out of 3


def test_compute_accuracy_with_abstain():
    """Test accuracy with abstention."""
    predictions = ["YES", "ABSTAIN", "NO"]
    references = ["YES", "NO", "NO"]
    
    acc = compute_accuracy(predictions, references, strict=True)
    assert acc == 2/3  # YES correct, ABSTAIN skipped, NO correct


def test_compute_f1_perfect():
    """Test F1 with perfect match."""
    pred = "The court ruled in favor of the defendant"
    ref = "The court ruled in favor of the defendant"
    
    result = compute_f1(pred, ref)
    assert result["f1"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0


def test_compute_f1_partial():
    """Test F1 with partial match."""
    pred = "The court ruled in favor"
    ref = "The court ruled against"
    
    result = compute_f1(pred, ref)
    assert result["f1"] > 0.0
    assert result["f1"] < 1.0


def test_compute_f1_no_match():
    """Test F1 with no match."""
    pred = "completely different text"
    ref = "totally unrelated content"
    
    result = compute_f1(pred, ref)
    assert result["f1"] == 0.0


def test_compute_f1_batch():
    """Test batch F1 computation."""
    predictions = [
        "The court ruled in favor",
        "The defendant was acquitted",
    ]
    references = [
        "The court ruled in favor of the defendant",
        "The defendant was convicted",
    ]
    
    result = compute_f1_batch(predictions, references)
    assert "f1" in result
    assert "precision" in result
    assert "recall" in result
    assert 0.0 <= result["f1"] <= 1.0


def test_compute_auc_perfect():
    """Test AUC with perfect predictions."""
    labels = [1, 1, 0, 0]
    scores = [0.9, 0.8, 0.3, 0.2]
    
    auc = compute_auc(labels, scores)
    assert auc == 1.0


def test_compute_auc_random():
    """Test AUC with random predictions."""
    labels = [1, 0, 1, 0]
    scores = [0.5, 0.5, 0.5, 0.5]
    
    auc = compute_auc(labels, scores)
    assert auc == 0.5


def test_compute_auc_with_bootstrap():
    """Test AUC with bootstrap confidence intervals."""
    labels = [1, 1, 1, 0, 0, 0]
    scores = [0.9, 0.8, 0.7, 0.4, 0.3, 0.2]
    
    auc, lower, upper = compute_auc(labels, scores, bootstrap=True, n_bootstrap=100)
    assert 0.0 <= lower <= auc <= upper <= 1.0


