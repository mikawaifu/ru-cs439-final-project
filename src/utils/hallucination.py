"""Hallucination detection utilities for LegalAdapter."""

import re
from typing import List, Dict, Any, Optional


def detect_hallucination(
    answer: str,
    context: Optional[str] = None,
    check_context_grounding: bool = True,
    check_fabricated_citations: bool = True,
    fabrication_patterns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Detect potential hallucinations in the answer.
    
    Args:
        answer: The generated answer to check
        context: Optional legal context provided to the model
        check_context_grounding: Whether to check if answer is grounded in context
        check_fabricated_citations: Whether to check for fake citations
        fabrication_patterns: List of regex patterns for fabricated citations
        
    Returns:
        Dictionary with:
            - is_hallucination: bool
            - confidence: float (0.0 to 1.0)
            - reasons: List of detected issues
    """
    reasons = []
    confidence_scores = []
    
    # Default fabrication patterns for legal citations
    if fabrication_patterns is None:
        fabrication_patterns = [
            r'\bv\.\s+[A-Z][a-z]+\s+\(\d{4}\)',  # e.g., "v. State (2020)"
            r'\b[A-Z][a-z]+\s+v\.\s+[A-Z][a-z]+',  # e.g., "Doe v. Smith"
            r'\b\d+\s+U\.S\.\s+\d+',  # U.S. Reports citation
            r'\b\d+\s+F\.\d+d\s+\d+',  # Federal Reporter citation
        ]
    
    # Check for fabricated citations
    if check_fabricated_citations:
        for pattern in fabrication_patterns:
            matches = re.findall(pattern, answer)
            if matches:
                # Check if these citations appear in context (if provided)
                if context:
                    for match in matches:
                        if match not in context:
                            reasons.append(f"Fabricated citation detected: {match}")
                            confidence_scores.append(0.9)
                else:
                    # Without context, we're less confident but still flag suspicious patterns
                    for match in matches:
                        reasons.append(f"Suspicious citation (no context to verify): {match}")
                        confidence_scores.append(0.6)
    
    # Check context grounding
    if check_context_grounding and context:
        # Extract potential factual claims (sentences with specific patterns)
        # This is a simplified heuristic - in production, you might want a more sophisticated approach
        
        # Look for definitive statements
        definitive_patterns = [
            r'The statute states',
            r'According to the law',
            r'The case of',
            r'In \d{4},',
            r'The court held',
            r'The regulation requires',
        ]
        
        for pattern in definitive_patterns:
            if re.search(pattern, answer, re.IGNORECASE):
                # Check if similar content exists in context
                # This is a rough check - more sophisticated semantic similarity would be better
                pattern_words = pattern.lower().split()
                context_lower = context.lower()
                
                if not any(word in context_lower for word in pattern_words if len(word) > 3):
                    reasons.append(f"Definitive statement without context support: {pattern}")
                    confidence_scores.append(0.7)
    
    # Check for common hallucination markers
    hallucination_markers = [
        (r'\bI think\b', 0.3),
        (r'\bI believe\b', 0.3),
        (r'\bprobably\b', 0.2),
        (r'\bmight be\b', 0.2),
        (r'\bcould be\b', 0.2),
    ]
    
    for pattern, conf in hallucination_markers:
        if re.search(pattern, answer, re.IGNORECASE):
            reasons.append(f"Uncertainty marker detected: {pattern}")
            confidence_scores.append(conf)
    
    # Determine if it's a hallucination
    is_hallucination = len(reasons) > 0
    
    # Average confidence across all detected issues
    if confidence_scores:
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
    else:
        avg_confidence = 0.0
    
    return {
        "is_hallucination": is_hallucination,
        "confidence": avg_confidence,
        "reasons": reasons,
    }


def compute_hallucination_rate(
    answers: List[str],
    contexts: Optional[List[Optional[str]]] = None,
    check_context_grounding: bool = True,
    check_fabricated_citations: bool = True,
    min_confidence: float = 0.7,
) -> Dict[str, Any]:
    """
    Compute hallucination rate over a batch of answers.
    
    Args:
        answers: List of generated answers
        contexts: Optional list of contexts (can be None for some answers)
        check_context_grounding: Whether to check context grounding
        check_fabricated_citations: Whether to check for fake citations
        min_confidence: Minimum confidence threshold to count as hallucination
        
    Returns:
        Dictionary with hallucination statistics
    """
    if contexts is None:
        contexts = [None] * len(answers)
    
    if len(answers) != len(contexts):
        raise ValueError("answers and contexts must have the same length")
    
    hallucinations = []
    total_checked = 0
    
    for answer, context in zip(answers, contexts):
        if answer == "ABSTAIN":
            continue
        
        result = detect_hallucination(
            answer=answer,
            context=context,
            check_context_grounding=check_context_grounding,
            check_fabricated_citations=check_fabricated_citations,
        )
        
        if result["is_hallucination"] and result["confidence"] >= min_confidence:
            hallucinations.append(result)
        
        total_checked += 1
    
    hallucination_rate = len(hallucinations) / total_checked if total_checked > 0 else 0.0
    
    return {
        "hallucination_rate": hallucination_rate,
        "total_hallucinations": len(hallucinations),
        "total_checked": total_checked,
        "details": hallucinations,
    }


