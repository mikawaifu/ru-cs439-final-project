"""Verifier dataset implementation."""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional
from pathlib import Path

from ..data.schemas import Question, VerifierTrainingExample
from ..data.loaders import DatasetLoader
from ..utils.io import load_json
from ..utils.logging import get_logger
from ..utils.metrics import compute_f1


class VerifierDataset(Dataset):
    """Dataset for training verifier model."""
    
    def __init__(
        self,
        examples: List[VerifierTrainingExample],
        tokenizer: Any,
        max_length: int = 512,
    ):
        """
        Initialize verifier dataset.
        
        Args:
            examples: List of training examples
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        # Build input text: [CLS] Q [SEP] CTX? [SEP] CAND [SEP]
        if example.context:
            text = f"{example.question} [SEP] {example.context} [SEP] {example.candidate}"
        else:
            text = f"{example.question} [SEP] {example.candidate}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding.get("token_type_ids", torch.zeros_like(encoding["input_ids"])).squeeze(0),
            "labels": torch.tensor(example.label, dtype=torch.float),
        }


def create_verifier_dataset(
    dataset_name: str,
    split: str,
    candidates_dir: Path,
    tokenizer: Any,
    max_length: int = 512,
    datasets_config: str = "configs/datasets.yaml",
) -> VerifierDataset:
    """
    Create verifier dataset from questions and candidates.
    
    Args:
        dataset_name: Name of the dataset
        split: Dataset split
        candidates_dir: Directory containing generated candidates
        tokenizer: Tokenizer instance
        max_length: Maximum sequence length
        datasets_config: Path to datasets configuration
        
    Returns:
        VerifierDataset instance
    """
    logger = get_logger("verifier_dataset")
    logger.info(f"Creating verifier dataset for {dataset_name} ({split})")
    
    # Load questions
    loader = DatasetLoader(datasets_config)
    questions = loader.load(dataset_name, split)
    question_map = {q.id: q for q in questions}
    
    # Load candidates and create training examples
    examples = []
    candidates_path = candidates_dir / dataset_name / split
    
    if not candidates_path.exists():
        logger.warning(f"Candidates not found: {candidates_path}")
        return VerifierDataset([], tokenizer, max_length)
    
    for candidate_file in candidates_path.glob("*.json"):
        try:
            candidate_data = load_json(candidate_file)
            question_id = candidate_data["question_id"]
            
            if question_id not in question_map:
                continue
            
            question_obj = question_map[question_id]
            ground_truth = question_obj.answer
            
            # Process each candidate
            for candidate in candidate_data["candidates"]:
                candidate_text = candidate["text"]
                
                # Determine label (1 if correct, 0 if incorrect)
                # For binary QA (YES/NO)
                if isinstance(ground_truth, dict) and "label" in ground_truth:
                    gt_label = ground_truth["label"].upper()
                    cand_upper = candidate_text.upper()
                    
                    # Check if candidate starts with the correct label
                    if cand_upper.startswith(gt_label):
                        label = 1
                    else:
                        label = 0
                else:
                    # For open QA, use F1 as proxy (threshold at 0.5)
                    f1_scores = compute_f1(candidate_text, str(ground_truth))
                    label = 1 if f1_scores["f1"] >= 0.5 else 0
                
                example = VerifierTrainingExample(
                    question=question_obj.question,
                    context=question_obj.context,
                    candidate=candidate_text,
                    label=label,
                )
                examples.append(example)
        
        except Exception as e:
            logger.error(f"Error processing {candidate_file}: {e}")
            continue
    
    logger.info(f"Created {len(examples)} training examples")
    logger.info(f"Positive examples: {sum(e.label for e in examples)}")
    logger.info(f"Negative examples: {len(examples) - sum(e.label for e in examples)}")
    
    return VerifierDataset(examples, tokenizer, max_length)


