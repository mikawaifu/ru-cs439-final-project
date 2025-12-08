"""Candidate generation orchestration."""

import argparse
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

from ..utils.io import load_yaml, save_json
from ..utils.logging import setup_logging, get_logger
from ..utils.seed import set_seed
from ..data.loaders import DatasetLoader
from ..data.schemas import Question, Candidate
from .prompts import create_prompt
from .llm_backend import create_backend


class CandidateGenerator:
    """Orchestrates K-candidate generation using multiple LLM backends."""
    
    def __init__(
        self,
        base_config: Dict[str, Any],
        models_config: Dict[str, Any],
    ):
        """
        Initialize candidate generator.
        
        Args:
            base_config: Base configuration
            models_config: Models configuration
        """
        self.base_config = base_config
        self.models_config = models_config
        self.logger = get_logger("generator")
        
        # Initialize backends
        self.backends = {}
        for name, config in models_config.get("generators", {}).items():
            if not config.get("enabled", False):
                self.logger.info(f"Skipping disabled generator: {name}")
                continue
            
            try:
                backend = create_backend(name, config)
                self.backends[name] = backend
                self.logger.info(f"Initialized backend: {name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize {name}: {e}")
        
        if not self.backends:
            raise ValueError("No backends initialized. Please check your configuration.")
        
        self.k = base_config.get("k_candidates", 5)
        self.output_dir = Path(base_config.get("output_dir", "artifacts"))
    
    def generate_for_question(
        self,
        question: Question,
    ) -> List[Candidate]:
        """
        Generate K candidates for a single question.
        
        Args:
            question: Question object
            
        Returns:
            List of candidate answers
        """
        # Create prompt
        prompt = create_prompt(question.question, question.context)
        
        candidates = []
        candidate_id = 0
        
        # Generate one candidate per backend
        for backend_name, backend in self.backends.items():
            try:
                response = backend.generate(
                    system_prompt=prompt["system"],
                    user_prompt=prompt["user"],
                )
                
                candidate = Candidate(
                    question_id=question.id,
                    candidate_id=candidate_id,
                    generator=backend_name,
                    text=response,
                )
                candidates.append(candidate)
                candidate_id += 1
                
            except Exception as e:
                self.logger.error(f"Failed to generate with {backend_name}: {e}")
        
        # If we have fewer than K candidates, sample again from the best backend
        # (for simplicity, we'll just use the first backend)
        if len(candidates) < self.k and len(self.backends) > 0:
            first_backend_name = list(self.backends.keys())[0]
            first_backend = self.backends[first_backend_name]
            
            while len(candidates) < self.k:
                try:
                    response = first_backend.generate(
                        system_prompt=prompt["system"],
                        user_prompt=prompt["user"],
                    )
                    
                    candidate = Candidate(
                        question_id=question.id,
                        candidate_id=candidate_id,
                        generator=f"{first_backend_name}_extra",
                        text=response,
                    )
                    candidates.append(candidate)
                    candidate_id += 1
                    
                except Exception as e:
                    self.logger.error(f"Failed to generate extra candidate: {e}")
                    break
        
        return candidates
    
    def generate_for_dataset(
        self,
        dataset_name: str,
        questions: List[Question],
    ) -> None:
        """
        Generate candidates for all questions in a dataset.
        
        Args:
            dataset_name: Name of the dataset
            questions: List of questions
        """
        self.logger.info(f"Generating candidates for {dataset_name} ({len(questions)} questions)")
        
        output_dir = self.output_dir / "candidates" / dataset_name / questions[0].split
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for question in tqdm(questions, desc=f"Generating {dataset_name}"):
            # Check if already generated
            output_file = output_dir / f"{question.id}.json"
            if output_file.exists():
                self.logger.debug(f"Skipping {question.id} (already exists)")
                continue
            
            # Generate candidates
            candidates = self.generate_for_question(question)
            
            # Save candidates
            candidates_data = {
                "question_id": question.id,
                "question": question.question,
                "context": question.context,
                "candidates": [c.to_dict() for c in candidates],
            }
            save_json(candidates_data, output_file)


def main():
    """Main generation function."""
    parser = argparse.ArgumentParser(description="Generate K candidates for legal QA")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to base configuration file",
    )
    parser.add_argument(
        "--models",
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
        "--split",
        type=str,
        default="test",
        help="Dataset split to process",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Quick test mode (only process first 3 questions)",
    )
    args = parser.parse_args()
    
    # Load configurations
    base_config = load_yaml(args.config)
    models_config = load_yaml(args.models)
    
    # Setup
    setup_logging(
        log_level=base_config.get("log_level", "INFO"),
        log_file=f"{base_config.get('log_dir', 'artifacts/logs')}/generate.log",
    )
    set_seed(base_config.get("seed", 42))
    
    logger = get_logger("generate")
    logger.info("Starting candidate generation")
    
    # Initialize generator
    generator = CandidateGenerator(base_config, models_config)
    
    # Load datasets
    loader = DatasetLoader(args.datasets)
    datasets = loader.load_all_enabled(args.split)
    
    if not datasets:
        logger.error(f"No datasets loaded for split '{args.split}'")
        return
    
    # Generate candidates for each dataset
    for dataset_name, questions in datasets.items():
        if args.quick_test:
            questions = questions[:3]
        
        generator.generate_for_dataset(dataset_name, questions)
    
    logger.info("Candidate generation complete!")


if __name__ == "__main__":
    main()


