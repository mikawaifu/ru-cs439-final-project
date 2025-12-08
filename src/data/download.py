"""Dataset download utilities for LegalAdapter."""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any
from ..utils.io import load_yaml
from ..utils.logging import setup_logging


def download_dataset(dataset_name: str, dataset_config: Dict[str, Any], raw_dir: Path) -> bool:
    """
    Download a single dataset.
    
    Args:
        dataset_name: Name of the dataset
        dataset_config: Configuration for the dataset
        raw_dir: Directory to store raw data
        
    Returns:
        True if download successful or manual download required, False otherwise
    """
    logger = setup_logging()
    
    source = dataset_config.get("source", "MANUAL_OR_URL")
    
    if source == "MANUAL_OR_URL":
        # Manual download required
        logger.warning(f"\n{'='*70}")
        logger.warning(f"Manual Download Required: {dataset_name}")
        logger.warning(f"{'='*70}")
        logger.warning(f"Dataset: {dataset_config.get('name', dataset_name)}")
        logger.warning(f"Description: {dataset_config.get('description', 'N/A')}")
        logger.warning(f"Official Info: {dataset_config.get('official_info', 'N/A')}")
        logger.warning(f"\nExpected files:")
        for file in dataset_config.get("expected_files", []):
            logger.warning(f"  - {file}")
        logger.warning(f"\nPlease download these files and place them in:")
        logger.warning(f"  {raw_dir / dataset_name}/")
        logger.warning(f"{'='*70}\n")
        return True
    else:
        # Automatic download (URL provided)
        logger.info(f"Attempting to download {dataset_name} from {source}")
        # TODO: Implement automatic download logic here
        # This would require verifying the URL is accessible and legitimate
        logger.error(f"Automatic download not yet implemented. Please download manually.")
        return False


def main():
    """Main download function."""
    parser = argparse.ArgumentParser(description="Download legal QA datasets")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/datasets.yaml",
        help="Path to datasets configuration file",
    )
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting dataset download process")
    
    # Load configuration
    config = load_yaml(args.config)
    root = Path(config.get("root", "data"))
    raw_dir = Path(config.get("raw_dir", root / "raw"))
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    datasets = config.get("datasets", {})
    
    if not datasets:
        logger.error("No datasets configured in datasets.yaml")
        sys.exit(1)
    
    # Process each enabled dataset
    manual_downloads_needed = []
    
    for dataset_name, dataset_config in datasets.items():
        if not dataset_config.get("enabled", False):
            logger.info(f"Skipping disabled dataset: {dataset_name}")
            continue
        
        logger.info(f"\nProcessing dataset: {dataset_name}")
        
        # Create dataset directory
        dataset_dir = raw_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Attempt download
        result = download_dataset(dataset_name, dataset_config, raw_dir)
        
        if dataset_config.get("source") == "MANUAL_OR_URL":
            manual_downloads_needed.append(dataset_name)
    
    # Summary
    if manual_downloads_needed:
        logger.warning(f"\n{'='*70}")
        logger.warning("SUMMARY: Manual downloads required for the following datasets:")
        for ds in manual_downloads_needed:
            logger.warning(f"  - {ds}")
        logger.warning(f"\nAfter downloading, run: make preprocess")
        logger.warning(f"{'='*70}")
        sys.exit(2)  # Exit code 2 indicates manual action needed
    else:
        logger.info("\nAll datasets downloaded successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()


