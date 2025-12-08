"""Verifier model implementation."""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Dict, Any, Optional


class VerifierModel(nn.Module):
    """BERT-size verifier model for scoring candidate answers."""
    
    def __init__(
        self,
        backbone: str = "microsoft/deberta-v3-base",
        pooling: str = "cls",
        dropout: float = 0.1,
    ):
        """
        Initialize verifier model.
        
        Args:
            backbone: Pretrained model name
            pooling: Pooling strategy ("cls" or "mean")
            dropout: Dropout rate
        """
        super().__init__()
        
        self.backbone_name = backbone
        self.pooling = pooling
        
        # Load pretrained model
        self.config = AutoConfig.from_pretrained(backbone)
        self.backbone = AutoModel.from_pretrained(backbone)
        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
        
        # Classification head
        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len]
            
        Returns:
            Logits [batch_size, 1]
        """
        # Get backbone outputs
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        # Pool
        if self.pooling == "cls":
            pooled = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        elif self.pooling == "mean":
            # Mean pooling over non-padded tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size())
            sum_embeddings = torch.sum(outputs.last_hidden_state * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        # Classify
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        return logits
    
    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict probability scores.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs
            
        Returns:
            Probabilities [batch_size]
        """
        logits = self.forward(input_ids, attention_mask, token_type_ids)
        probs = torch.sigmoid(logits).squeeze(-1)
        return probs
    
    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            "model_state_dict": self.state_dict(),
            "backbone_name": self.backbone_name,
            "pooling": self.pooling,
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "VerifierModel":
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=device)
        
        model = cls(
            backbone=checkpoint["backbone_name"],
            pooling=checkpoint.get("pooling", "cls"),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        
        return model


