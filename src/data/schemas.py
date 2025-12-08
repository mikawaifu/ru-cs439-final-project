"""Data schemas for LegalAdapter."""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List


@dataclass
class Question:
    """Schema for a legal question."""
    
    id: str
    question: str
    context: Optional[str]
    answer: str  # Can be string or dict with "label" key
    split: str  # "train", "dev", or "test"
    dataset: str
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Question":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class Candidate:
    """Schema for a generated candidate answer."""
    
    question_id: str
    candidate_id: int  # 0 to K-1
    generator: str  # e.g., "openai_gpt4o", "vllm_llama3"
    text: str
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Candidate":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ScoredCandidate:
    """Schema for a scored candidate answer."""
    
    question_id: str
    candidate_id: int
    generator: str
    text: str
    score: float  # Verifier score (0.0 to 1.0)
    rank: int  # After re-ranking (0 = best)
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ScoredCandidate":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class VerifierTrainingExample:
    """Schema for verifier training data."""
    
    question: str
    context: Optional[str]
    candidate: str
    label: int  # 0 (incorrect) or 1 (correct)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerifierTrainingExample":
        """Create from dictionary."""
        return cls(**data)


