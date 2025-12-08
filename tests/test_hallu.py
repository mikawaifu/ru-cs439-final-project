"""Tests for hallucination detection."""

import pytest
from src.legaladapter.utils.hallucination import (
    detect_hallucination,
    compute_hallucination_rate,
)


def test_no_hallucination():
    """Test answer with no hallucination."""
    answer = "YES. The statute clearly states this requirement."
    context = "The statute clearly states this requirement in Section 5."
    
    result = detect_hallucination(answer, context)
    assert result["is_hallucination"] == False
    assert result["confidence"] == 0.0


def test_fabricated_citation_with_context():
    """Test fabricated citation detection with context."""
    answer = "YES. According to Doe v. State (2020), this is required."
    context = "According to the statute, this is required."
    
    result = detect_hallucination(
        answer,
        context,
        check_fabricated_citations=True,
    )
    assert result["is_hallucination"] == True
    assert result["confidence"] > 0.7


def test_fabricated_citation_without_context():
    """Test fabricated citation detection without context."""
    answer = "YES. According to Smith v. Jones (2020), this is required."
    
    result = detect_hallucination(
        answer,
        context=None,
        check_fabricated_citations=True,
    )
    # Should still flag but with lower confidence
    assert result["is_hallucination"] == True
    assert result["confidence"] > 0.0


def test_uncertainty_markers():
    """Test detection of uncertainty markers."""
    answer = "I think this might be correct, but I'm not entirely sure."
    
    result = detect_hallucination(answer)
    assert result["is_hallucination"] == True
    assert len(result["reasons"]) > 0


def test_hallucination_rate():
    """Test hallucination rate computation."""
    answers = [
        "YES. The statute states this.",
        "NO. According to Fake v. Case (2020), this is not allowed.",
        "YES. This is correct.",
        "ABSTAIN",
    ]
    contexts = [
        "The statute states this.",
        "The statute does not mention this.",
        "This is correct in the law.",
        None,
    ]
    
    result = compute_hallucination_rate(
        answers,
        contexts,
        check_fabricated_citations=True,
        min_confidence=0.7,
    )
    
    assert "hallucination_rate" in result
    assert 0.0 <= result["hallucination_rate"] <= 1.0
    assert result["total_checked"] == 3  # ABSTAIN should be skipped


def test_valid_citation_in_context():
    """Test that valid citations in context are not flagged."""
    answer = "YES. According to Smith v. Jones, this is required."
    context = "In Smith v. Jones, the court held that this is required."
    
    result = detect_hallucination(
        answer,
        context,
        check_fabricated_citations=True,
        check_context_grounding=True,
    )
    # Should not flag as hallucination since citation is in context
    assert result["is_hallucination"] == False or result["confidence"] < 0.7


