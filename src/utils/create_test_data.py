"""Create synthetic test data for quick testing."""

import argparse
from pathlib import Path
from ..utils.io import save_jsonl, save_json
from ..utils.logging import setup_logging, get_logger
from ..data.schemas import Question


def create_test_data(output_dir: str = "data/processed"):
    """Create minimal synthetic test data."""
    logger = get_logger("create_test_data")
    logger.info("Creating synthetic test data")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create test questions
    test_questions = [
        Question(
            id="test_coliee_1",
            question="Is a written contract required for the sale of real property?",
            context="Section 123: A contract for the sale of real property must be in writing to be enforceable.",
            answer={"label": "YES"},
            split="test",
            dataset="coliee_task4",
        ),
        Question(
            id="test_coliee_2",
            question="Can oral contracts be enforced for services?",
            context="Section 456: Oral contracts for services may be enforceable if the consideration is under $500.",
            answer={"label": "YES"},
            split="test",
            dataset="coliee_task4",
        ),
        Question(
            id="test_coliee_3",
            question="Is a minor legally bound by a contract?",
            context="Section 789: Contracts entered into by minors are generally voidable at the minor's option.",
            answer={"label": "NO"},
            split="test",
            dataset="coliee_task4",
        ),
    ]
    
    # Save test data
    test_file = output_path / "coliee_task4_test.jsonl"
    save_jsonl([q.to_dict() for q in test_questions], test_file)
    logger.info(f"Created test data: {test_file}")
    
    # Also create train/dev with same data for testing training
    train_file = output_path / "coliee_task4_train.jsonl"
    save_jsonl([q.to_dict() for q in test_questions], train_file)
    logger.info(f"Created train data: {train_file}")
    
    dev_file = output_path / "coliee_task4_dev.jsonl"
    save_jsonl([q.to_dict() for q in test_questions], dev_file)
    logger.info(f"Created dev data: {dev_file}")
    
    logger.info("Synthetic test data created successfully!")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Create synthetic test data")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory for test data",
    )
    args = parser.parse_args()
    
    setup_logging(log_level="INFO")
    create_test_data(args.output_dir)


if __name__ == "__main__":
    main()


