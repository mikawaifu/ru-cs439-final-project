"""Prompt templates for legal QA."""

from typing import Optional, Dict, Any


class LegalQAPrompt:
    """Prompt template for legal question answering."""
    
    SYSTEM_PROMPT = (
        "You are a careful legal assistant. Answer concisely and only within "
        "provided legal context if any. If unsure, say you are unsure."
    )
    
    USER_TEMPLATE_WITH_CONTEXT = """Question:
{question}

Legal Context:
{context}

Instructions:
- Think step by step briefly, then give the final answer.
- If YES/NO, start with "YES" or "NO", followed by one concise rationale.
- If open QA, provide a focused 2-4 sentence answer.
- Do NOT fabricate cases, statutes, or citations. If citing, use only items present in the provided context."""
    
    USER_TEMPLATE_WITHOUT_CONTEXT = """Question:
{question}

Instructions:
- Think step by step briefly, then give the final answer.
- If YES/NO, start with "YES" or "NO", followed by one concise rationale.
- If open QA, provide a focused 2-4 sentence answer.
- Do NOT fabricate cases, statutes, or citations. Only cite if you are certain."""
    
    @classmethod
    def create(cls, question: str, context: Optional[str] = None) -> Dict[str, str]:
        """
        Create a prompt for legal QA.
        
        Args:
            question: The legal question
            context: Optional legal context (statutes, case excerpts, etc.)
            
        Returns:
            Dictionary with "system" and "user" keys
        """
        if context:
            user_content = cls.USER_TEMPLATE_WITH_CONTEXT.format(
                question=question,
                context=context,
            )
        else:
            user_content = cls.USER_TEMPLATE_WITHOUT_CONTEXT.format(
                question=question,
            )
        
        return {
            "system": cls.SYSTEM_PROMPT,
            "user": user_content,
        }


def create_prompt(question: str, context: Optional[str] = None) -> Dict[str, str]:
    """
    Convenience function to create a legal QA prompt.
    
    Args:
        question: The legal question
        context: Optional legal context
        
    Returns:
        Dictionary with "system" and "user" keys
    """
    return LegalQAPrompt.create(question, context)


