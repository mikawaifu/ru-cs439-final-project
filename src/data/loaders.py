"""Dataset loaders for LegalAdapter."""

from pathlib import Path
from typing import List, Dict, Any, Optional
from ..utils.io import load_jsonl, load_yaml
from .schemas import Question


class DatasetLoader:
    """Loader for processed legal QA datasets."""
    
    def __init__(self, config_path: str = "configs/datasets.yaml"):
        """
        Initialize dataset loader.
        
        Args:
            config_path: Path to datasets configuration file
        """
        self.config = load_yaml(config_path)
        self.root = Path(self.config.get("root", "data"))
        self.processed_dir = Path(self.config.get("processed_dir", self.root / "processed"))
    
    def load(
        self,
        dataset_name: str,
        split: str = "test",
    ) -> List[Question]:
        """
        Load a dataset split.
        
        Args:
            dataset_name: Name of the dataset
            split: Split to load ("train", "dev", or "test")
            
        Returns:
            List of Question objects
        """
        file_path = self.processed_dir / f"{dataset_name}_{split}.jsonl"
        
        if not file_path.exists():
            raise FileNotFoundError(
                f"Processed data not found: {file_path}\n"
                f"Please run 'make preprocess' first"
            )
        
        data = load_jsonl(file_path)
        questions = [Question.from_dict(item) for item in data]
        
        return questions
    
    def load_all_enabled(self, split: str = "test") -> Dict[str, List[Question]]:
        """
        Load all enabled datasets for a given split.
        
        Args:
            split: Split to load
            
        Returns:
            Dictionary mapping dataset names to lists of questions
        """
        datasets = {}
        
        for dataset_name, dataset_config in self.config.get("datasets", {}).items():
            if not dataset_config.get("enabled", False):
                continue
            
            try:
                datasets[dataset_name] = self.load(dataset_name, split)
            except FileNotFoundError:
                continue
        
        return datasets


def load_dataset(
    dataset_name: str,
    split: str = "test",
    config_path: str = "configs/datasets.yaml",
) -> List[Question]:
    """
    Convenience function to load a dataset.
    
    Args:
        dataset_name: Name of the dataset
        split: Split to load
        config_path: Path to configuration file
        
    Returns:
        List of Question objects
    """
    loader = DatasetLoader(config_path)
    return loader.load(dataset_name, split)


