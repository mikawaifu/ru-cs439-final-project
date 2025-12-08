"""Dataset preprocessing for LegalAdapter."""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List
from ..utils.io import load_yaml, save_jsonl, load_json
from ..utils.logging import setup_logging
from .schemas import Question


def preprocess_coliee(raw_dir: Path, dataset_name: str) -> List[Dict[str, Any]]:
    """
    Preprocess COLIEE Task 4 dataset.
    
    Args:
        raw_dir: Directory containing raw data
        dataset_name: Name of the dataset
        
    Returns:
        List of processed questions
    """
    logger = setup_logging()
    logger.info(f"Preprocessing COLIEE dataset from {raw_dir}")
    
    # TODO: Implement actual COLIEE preprocessing based on the real format
    # This is a placeholder that expects JSON files with specific structure
    
    processed = []
    dataset_dir = raw_dir / dataset_name
    
    for split in ["train", "dev", "test"]:
        json_file = dataset_dir / f"{split}.json"
        
        if not json_file.exists():
            logger.warning(f"File not found: {json_file}, skipping {split} split")
            continue
        
        logger.info(f"Processing {split} split...")
        data = load_json(json_file)
        
        # Adapt to actual COLIEE format
        # Expected format: list of items with question, context, answer (YES/NO)
        for idx, item in enumerate(data):
            question = Question(
                id=item.get("id", f"{dataset_name}_{split}_{idx}"),
                question=item.get("question", item.get("text", "")),
                context=item.get("context", item.get("articles", None)),
                answer=item.get("answer", item.get("label", "")),
                split=split,
                dataset=dataset_name,
                metadata=item.get("metadata", {}),
            )
            processed.append(question.to_dict())
    
    return processed


def preprocess_generic(raw_dir: Path, dataset_name: str) -> List[Dict[str, Any]]:
    """
    Preprocess generic legal QA dataset.
    
    Args:
        raw_dir: Directory containing raw data
        dataset_name: Name of the dataset
        
    Returns:
        List of processed questions
    """
    logger = setup_logging()
    logger.info(f"Preprocessing {dataset_name} dataset from {raw_dir}")
    
    processed = []
    dataset_dir = raw_dir / dataset_name
    
    for split in ["train", "dev", "test"]:
        json_file = dataset_dir / f"{dataset_name}_{split}.json"
        
        if not json_file.exists():
            logger.warning(f"File not found: {json_file}, skipping {split} split")
            continue
        
        logger.info(f"Processing {split} split...")
        data = load_json(json_file)
        
        # Adapt to actual format
        for idx, item in enumerate(data):
            question = Question(
                id=item.get("id", f"{dataset_name}_{split}_{idx}"),
                question=item.get("question", ""),
                context=item.get("context", None),
                answer=item.get("answer", ""),
                split=split,
                dataset=dataset_name,
                metadata=item.get("metadata", {}),
            )
            processed.append(question.to_dict())
    
    return processed


def main():
    """Main preprocessing function."""
    parser = argparse.ArgumentParser(description="Preprocess legal QA datasets")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/datasets.yaml",
        help="Path to datasets configuration file",
    )
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting dataset preprocessing")
    
    # Load configuration
    config = load_yaml(args.config)
    root = Path(config.get("root", "data"))
    raw_dir = Path(config.get("raw_dir", root / "raw"))
    processed_dir = Path(config.get("processed_dir", root / "processed"))
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = config.get("datasets", {})
    
    if not datasets:
        logger.error("No datasets configured in datasets.yaml")
        sys.exit(1)
    
    # Process each enabled dataset
    for dataset_name, dataset_config in datasets.items():
        if not dataset_config.get("enabled", False):
            logger.info(f"Skipping disabled dataset: {dataset_name}")
            continue
        
        logger.info(f"\nPreprocessing dataset: {dataset_name}")
        
        # Check if raw data exists
        dataset_dir = raw_dir / dataset_name
        if not dataset_dir.exists():
            logger.error(f"Raw data directory not found: {dataset_dir}")
            logger.error(f"Please run 'make download' first")
            continue
        
        # Preprocess based on dataset type
        try:
            if dataset_name == "coliee_task4":
                processed = preprocess_coliee(raw_dir, dataset_name)
            else:
                processed = preprocess_generic(raw_dir, dataset_name)
            
            if not processed:
                logger.warning(f"No data processed for {dataset_name}")
                continue
            
            # Group by split and save
            splits = {}
            for item in processed:
                split = item["split"]
                if split not in splits:
                    splits[split] = []
                splits[split].append(item)
            
            for split, items in splits.items():
                output_file = processed_dir / f"{dataset_name}_{split}.jsonl"
                save_jsonl(items, output_file)
                logger.info(f"Saved {len(items)} items to {output_file}")
        
        except Exception as e:
            logger.error(f"Error preprocessing {dataset_name}: {e}")
            continue
    
    logger.info("\nPreprocessing complete!")


if __name__ == "__main__":
    main()


