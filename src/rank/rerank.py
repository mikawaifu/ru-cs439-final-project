"""Candidate re-ranking script."""

import argparse
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm

from ..utils.io import load_yaml, load_json, save_json
from ..utils.logging import setup_logging, get_logger
from ..data.loaders import DatasetLoader


def rerank_candidates(
    base_config: Dict[str, Any],
    datasets_config_path: str,
    split: str = "test",
) -> None:
    """
    Re-rank candidates based on verifier scores and apply abstain threshold.
    
    Args:
        base_config: Base configuration
        datasets_config_path: Path to datasets configuration
        split: Dataset split to rerank
    """
    logger = get_logger("rerank")
    
    # Get config
    output_dir = Path(base_config.get("output_dir", "artifacts"))
    scored_dir = output_dir / "candidates_scored"
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    abstain_threshold = base_config.get("abstain_threshold", 0.5)
    logger.info(f"Using abstain threshold: {abstain_threshold}")
    
    # Load datasets
    datasets_config = load_yaml(datasets_config_path)
    loader = DatasetLoader(datasets_config_path)
    
    for dataset_name in datasets_config.get("datasets", {}).keys():
        if not datasets_config["datasets"][dataset_name].get("enabled", False):
            continue
        
        logger.info(f"Re-ranking candidates for {dataset_name} ({split})")
        
        # Load questions
        try:
            questions = loader.load(dataset_name, split)
            question_map = {q.id: q for q in questions}
        except FileNotFoundError:
            logger.warning(f"Questions not found for {dataset_name} ({split})")
            continue
        
        # Get scored candidates directory
        dataset_scored_dir = scored_dir / dataset_name / split
        if not dataset_scored_dir.exists():
            logger.warning(f"Scored candidates not found: {dataset_scored_dir}")
            continue
        
        # Process each question
        predictions = []
        
        for scored_file in tqdm(list(dataset_scored_dir.glob("*.json")), desc=dataset_name):
            try:
                scored_data = load_json(scored_file)
                question_id = scored_data["question_id"]
                
                if question_id not in question_map:
                    continue
                
                question_obj = question_map[question_id]
                scored_candidates = scored_data["scored_candidates"]
                
                # Already sorted by rank, get top candidate
                if not scored_candidates:
                    final_answer = "ABSTAIN"
                    final_score = 0.0
                else:
                    top_candidate = scored_candidates[0]
                    final_score = top_candidate["score"]
                    
                    # Apply abstain threshold
                    if final_score < abstain_threshold:
                        final_answer = "ABSTAIN"
                    else:
                        final_answer = top_candidate["text"]
                
                prediction = {
                    "question_id": question_id,
                    "question": question_obj.question,
                    "context": question_obj.context,
                    "ground_truth": question_obj.answer,
                    "prediction": final_answer,
                    "verifier_score": final_score,
                    "all_candidates": scored_candidates,
                }
                predictions.append(prediction)
            
            except Exception as e:
                logger.error(f"Error processing {scored_file}: {e}")
                continue
        
        # Save predictions
        output_file = results_dir / f"{dataset_name}_{split}_predictions.json"
        save_json(predictions, output_file)
        logger.info(f"Saved {len(predictions)} predictions to {output_file}")
        
        # Count abstentions
        num_abstain = sum(1 for p in predictions if p["prediction"] == "ABSTAIN")
        logger.info(f"Abstentions: {num_abstain}/{len(predictions)} ({100*num_abstain/len(predictions):.1f}%)")


def main():
    """Main re-ranking function."""
    parser = argparse.ArgumentParser(description="Re-rank candidates")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to base configuration file",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="configs/datasets.yaml",
        help="Path to datasets configuration file",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to rerank",
    )
    args = parser.parse_args()
    
    # Load configuration
    base_config = load_yaml(args.config)
    
    # Setup
    setup_logging(
        log_level=base_config.get("log_level", "INFO"),
        log_file=f"{base_config.get('log_dir', 'artifacts/logs')}/rerank.log",
    )
    
    # Rerank
    rerank_candidates(base_config, args.datasets, args.split)


if __name__ == "__main__":
    main()


