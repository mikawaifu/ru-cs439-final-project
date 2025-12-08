"""Generate evaluation tables."""

import argparse
from pathlib import Path
import pandas as pd

from ..utils.io import load_yaml, load_json
from ..utils.logging import setup_logging, get_logger


def generate_tables(
    eval_config: dict,
    datasets_config_path: str,
    split: str = "test",
) -> None:
    """
    Generate evaluation tables in various formats.
    
    Args:
        eval_config: Evaluation configuration
        datasets_config_path: Path to datasets configuration
        split: Dataset split
    """
    logger = get_logger("tables")
    
    report_config = eval_config.get("report", {})
    results_dir = Path(report_config.get("output_dir", "artifacts/results"))
    
    # Load results
    results_file = results_dir / f"evaluation_results_{split}.json"
    if not results_file.exists():
        logger.error(f"Results not found: {results_file}")
        return
    
    results = load_json(results_file)
    
    # Build table data
    table_data = []
    for dataset_name, metrics in results.items():
        if dataset_name.startswith("_"):
            continue
        
        row = {"Dataset": dataset_name}
        row["Accuracy"] = f"{metrics.get('accuracy', 0):.4f}"
        row["F1"] = f"{metrics.get('f1', 0):.4f}"
        row["AUC"] = f"{metrics.get('auc', 0):.4f}"
        row["Hallu Rate"] = f"{metrics.get('hallucination_rate', 0):.4f}"
        row["Abstain Rate"] = f"{metrics.get('abstain_rate', 0):.4f}"
        row["Examples"] = metrics.get("total_examples", 0)
        
        table_data.append(row)
    
    if not table_data:
        logger.warning("No data to generate tables")
        return
    
    df = pd.DataFrame(table_data)
    
    # Save in different formats
    formats = report_config.get("table_formats", ["csv", "markdown", "latex"])
    
    if "csv" in formats:
        csv_file = results_dir / f"results_table_{split}.csv"
        df.to_csv(csv_file, index=False)
        logger.info(f"Saved CSV table to {csv_file}")
    
    if "markdown" in formats:
        md_file = results_dir / f"results_table_{split}.md"
        with open(md_file, "w") as f:
            f.write(df.to_markdown(index=False))
        logger.info(f"Saved Markdown table to {md_file}")
    
    if "latex" in formats:
        latex_file = results_dir / f"results_table_{split}.tex"
        with open(latex_file, "w") as f:
            f.write(df.to_latex(index=False))
        logger.info(f"Saved LaTeX table to {latex_file}")
    
    # Print to console
    logger.info("\n" + "="*80)
    logger.info(f"Results Summary ({split})")
    logger.info("="*80)
    logger.info("\n" + df.to_string(index=False))
    logger.info("="*80 + "\n")


def main():
    """Main table generation function."""
    parser = argparse.ArgumentParser(description="Generate evaluation tables")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/eval.yaml",
        help="Path to evaluation configuration file",
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
        help="Dataset split",
    )
    args = parser.parse_args()
    
    # Load configuration
    eval_config = load_yaml(args.config)
    
    # Setup logging
    setup_logging(log_level="INFO")
    
    # Generate tables
    generate_tables(eval_config, args.datasets, args.split)


if __name__ == "__main__":
    main()


