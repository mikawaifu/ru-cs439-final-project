"""Verifier inference script."""

import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from ..utils.io import load_yaml, load_json, save_json
from ..utils.logging import setup_logging, get_logger
from ..utils.seed import set_seed
from ..data.loaders import DatasetLoader
from ..data.schemas import ScoredCandidate
from .model import VerifierModel


def score_candidates(
    models_config: dict,
    datasets_config_path: str,
    base_config: dict,
    split: str = "test",
) -> None:
    """
    Score candidates using trained verifier.
    
    Args:
        models_config: Models configuration
        datasets_config_path: Path to datasets configuration
        base_config: Base configuration
        split: Dataset split to score
    """
    logger = get_logger("score_candidates")
    
    # Get config
    verifier_config = models_config.get("verifier", {})
    device = base_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(base_config.get("output_dir", "artifacts"))
    candidates_dir = output_dir / "candidates"
    scored_dir = output_dir / "candidates_scored"
    scored_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = Path(verifier_config.get("save_dir", output_dir / "verifier")) / "best_model.pt"
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        logger.error("Please train the verifier first: make train")
        return
    
    logger.info(f"Loading model from {model_path}")
    model = VerifierModel.load(str(model_path), device=device)
    model.eval()
    
    # Load datasets
    datasets_config = load_yaml(datasets_config_path)
    
    for dataset_name in datasets_config.get("datasets", {}).keys():
        if not datasets_config["datasets"][dataset_name].get("enabled", False):
            continue
        
        logger.info(f"Scoring candidates for {dataset_name} ({split})")
        
        # Get candidates directory
        dataset_candidates_dir = candidates_dir / dataset_name / split
        if not dataset_candidates_dir.exists():
            logger.warning(f"Candidates not found: {dataset_candidates_dir}")
            continue
        
        # Output directory
        dataset_scored_dir = scored_dir / dataset_name / split
        dataset_scored_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each question's candidates
        for candidate_file in tqdm(list(dataset_candidates_dir.glob("*.json")), desc=dataset_name):
            try:
                candidate_data = load_json(candidate_file)
                question_id = candidate_data["question_id"]
                question_text = candidate_data["question"]
                context = candidate_data.get("context", "")
                
                scored_candidates = []
                
                for candidate in candidate_data["candidates"]:
                    # Build input text
                    if context:
                        text = f"{question_text} [SEP] {context} [SEP] {candidate['text']}"
                    else:
                        text = f"{question_text} [SEP] {candidate['text']}"
                    
                    # Tokenize
                    encoding = model.tokenizer(
                        text,
                        max_length=verifier_config.get("max_length", 512),
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    )
                    
                    # Score
                    with torch.no_grad():
                        input_ids = encoding["input_ids"].to(device)
                        attention_mask = encoding["attention_mask"].to(device)
                        token_type_ids = encoding.get("token_type_ids", torch.zeros_like(input_ids)).to(device)
                        
                        score = model.predict_proba(input_ids, attention_mask, token_type_ids).item()
                    
                    scored_candidate = {
                        "question_id": question_id,
                        "candidate_id": candidate["candidate_id"],
                        "generator": candidate["generator"],
                        "text": candidate["text"],
                        "score": score,
                    }
                    scored_candidates.append(scored_candidate)
                
                # Sort by score (descending) and assign ranks
                scored_candidates.sort(key=lambda x: x["score"], reverse=True)
                for rank, candidate in enumerate(scored_candidates):
                    candidate["rank"] = rank
                
                # Save scored candidates
                output_data = {
                    "question_id": question_id,
                    "question": question_text,
                    "context": context,
                    "scored_candidates": scored_candidates,
                }
                output_file = dataset_scored_dir / f"{question_id}.json"
                save_json(output_data, output_file)
            
            except Exception as e:
                logger.error(f"Error scoring {candidate_file}: {e}")
                continue
        
        logger.info(f"Scored candidates saved to {dataset_scored_dir}")


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Score candidates with verifier")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/models.yaml",
        help="Path to models configuration file",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="configs/datasets.yaml",
        help="Path to datasets configuration file",
    )
    parser.add_argument(
        "--base",
        type=str,
        default="configs/base.yaml",
        help="Path to base configuration file",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to score",
    )
    args = parser.parse_args()
    
    # Load configurations
    models_config = load_yaml(args.config)
    base_config = load_yaml(args.base)
    
    # Setup
    setup_logging(
        log_level=base_config.get("log_level", "INFO"),
        log_file=f"{base_config.get('log_dir', 'artifacts/logs')}/score_candidates.log",
    )
    set_seed(base_config.get("seed", 42))
    
    # Score
    score_candidates(models_config, args.datasets, base_config, args.split)


if __name__ == "__main__":
    main()


