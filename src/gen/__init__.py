"""Candidate generation modules for LegalAdapter."""

from .prompts import LegalQAPrompt, create_prompt
from .llm_backend import LLMBackend, create_backend
from .generate import CandidateGenerator

__all__ = [
    "LegalQAPrompt",
    "create_prompt",
    "LLMBackend",
    "create_backend",
    "CandidateGenerator",
]


