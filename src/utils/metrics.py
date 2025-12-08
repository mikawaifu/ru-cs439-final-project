"""Evaluation metrics for LegalAdapter."""

import re
import string
from typing import List, Dict, Any, Union, Tuple
from collections import Counter
import numpy as np
from sklearn.metrics import roc_auc_score


def normalize_text(text: str, lowercase: bool = True, remove_punct: bool = True) -> str:
    """
    Normalize text for comparison.
    
    Args:
        text: Input text
        lowercase: Whether to lowercase
        remove_punct: Whether to remove punctuation
        
    Returns:
        Normalized text
    """
    if lowercase:
        text = text.lower()
    
    if remove_punct:
        text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Normalize whitespace
    text = " ".join(text.split())
    
    return text


def compute_accuracy(
    predictions: List[str],
    references: List[str],
    strict: bool = True,
    case_sensitive: bool = False,
) -> float:
    """
    Compute accuracy metric.
    
    Args:
        predictions: List of predicted answers
        references: List of ground truth answers
        strict: If True, require exact match; if False, check substring match
        case_sensitive: Whether to perform case-sensitive comparison
        
    Returns:
        Accuracy score (0.0 to 1.0)
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    
    if len(predictions) == 0:
        return 0.0
    
    correct = 0
    for pred, ref in zip(predictions, references):
        # Handle abstention
        if pred == "ABSTAIN":
            continue
        
        if not case_sensitive:
            pred = pred.lower()
            ref = ref.lower()
        
        if strict:
            if pred.strip() == ref.strip():
                correct += 1
        else:
            # Substring match or vice versa
            if pred.strip() in ref.strip() or ref.strip() in pred.strip():
                correct += 1
    
    return correct / len(predictions)


def compute_f1(
    prediction: str,
    reference: str,
    lowercase: bool = True,
    remove_punctuation: bool = True,
) -> Dict[str, float]:
    """
    Compute token-level F1 score for a single prediction-reference pair.
    
    Args:
        prediction: Predicted answer
        reference: Ground truth answer
        lowercase: Whether to lowercase text
        remove_punctuation: Whether to remove punctuation
        
    Returns:
        Dictionary with precision, recall, and f1 scores
    """
    # Normalize texts
    pred_norm = normalize_text(prediction, lowercase, remove_punctuation)
    ref_norm = normalize_text(reference, lowercase, remove_punctuation)
    
    # Tokenize
    pred_tokens = pred_norm.split()
    ref_tokens = ref_norm.split()
    
    if len(pred_tokens) == 0 and len(ref_tokens) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    # Count common tokens
    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)
    
    common = sum((pred_counter & ref_counter).values())
    
    precision = common / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
    recall = common / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
    
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_f1_batch(
    predictions: List[str],
    references: List[str],
    lowercase: bool = True,
    remove_punctuation: bool = True,
) -> Dict[str, float]:
    """
    Compute average F1 scores over a batch of predictions.
    
    Args:
        predictions: List of predicted answers
        references: List of ground truth answers
        lowercase: Whether to lowercase text
        remove_punctuation: Whether to remove punctuation
        
    Returns:
        Dictionary with average precision, recall, and f1 scores
    """
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    
    if len(predictions) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    scores = [
        compute_f1(pred, ref, lowercase, remove_punctuation)
        for pred, ref in zip(predictions, references)
    ]
    
    avg_precision = np.mean([s["precision"] for s in scores])
    avg_recall = np.mean([s["recall"] for s in scores])
    avg_f1 = np.mean([s["f1"] for s in scores])
    
    return {
        "precision": float(avg_precision),
        "recall": float(avg_recall),
        "f1": float(avg_f1),
    }


def compute_auc(
    labels: List[int],
    scores: List[float],
    bootstrap: bool = False,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
) -> Union[float, Tuple[float, float, float]]:
    """
    Compute ROC-AUC score for verifier evaluation.
    
    Args:
        labels: Binary labels (0 or 1)
        scores: Prediction scores (0.0 to 1.0)
        bootstrap: Whether to compute bootstrap confidence intervals
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for intervals
        
    Returns:
        If bootstrap=False: AUC score
        If bootstrap=True: Tuple of (AUC, lower_bound, upper_bound)
    """
    labels = np.array(labels)
    scores = np.array(scores)
    
    if len(np.unique(labels)) < 2:
        # Cannot compute AUC with only one class
        if bootstrap:
            return 0.0, 0.0, 0.0
        return 0.0
    
    auc = roc_auc_score(labels, scores)
    
    if not bootstrap:
        return float(auc)
    
    # Bootstrap confidence intervals
    rng = np.random.RandomState(42)
    auc_scores = []
    
    for _ in range(n_bootstrap):
        indices = rng.choice(len(labels), size=len(labels), replace=True)
        if len(np.unique(labels[indices])) < 2:
            continue
        auc_scores.append(roc_auc_score(labels[indices], scores[indices]))
    
    auc_scores = np.array(auc_scores)
    alpha = 1 - confidence_level
    lower = np.percentile(auc_scores, 100 * alpha / 2)
    upper = np.percentile(auc_scores, 100 * (1 - alpha / 2))
    
    return float(auc), float(lower), float(upper)


