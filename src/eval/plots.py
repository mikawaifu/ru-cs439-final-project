"""Generate evaluation plots."""

import argparse
from pathlib import Path
from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from ..utils.io import load_yaml, load_json
from ..utils.logging import setup_logging, get_logger


def generate_plots(
    eval_config: Dict[str, Any],
    datasets_config_path: str,
    split: str = "test",
) -> None:
    """
    Generate evaluation plots.
    
    Args:
        eval_config: Evaluation configuration
        datasets_config_path: Path to datasets configuration
        split: Dataset split
    """
    logger = get_logger("plots")
    
    report_config = eval_config.get("report", {})
    results_dir = Path(report_config.get("output_dir", "artifacts/results"))
    figures_dir = Path(report_config.get("figures_dir", "artifacts/figures"))
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    plots_list = report_config.get("plots_list", [
        "score_distribution",
        "roc_curve",
        "accuracy_by_dataset",
    ])
    
    # Load results
    results_file = results_dir / f"evaluation_results_{split}.json"
    if not results_file.exists():
        logger.warning(f"Results not found: {results_file}")
        return
    
    results = load_json(results_file)
    
    # Load datasets config
    datasets_config = load_yaml(datasets_config_path)
    
    # Plot 1: Score distribution
    if "score_distribution" in plots_list:
        logger.info("Generating score distribution plot")
        
        fig, axes = plt.subplots(1, len([k for k in results.keys() if not k.startswith("_")]), 
                                 figsize=(6*len(results), 4))
        if len(results) == 1:
            axes = [axes]
        
        for idx, (dataset_name, _) in enumerate([(k, v) for k, v in results.items() if not k.startswith("_")]):
            # Load predictions to get scores
            predictions_file = results_dir / f"{dataset_name}_{split}_predictions.json"
            if not predictions_file.exists():
                continue
            
            predictions = load_json(predictions_file)
            scores = [p["verifier_score"] for p in predictions]
            
            axes[idx].hist(scores, bins=20, alpha=0.7, edgecolor="black")
            axes[idx].set_title(f"{dataset_name}")
            axes[idx].set_xlabel("Verifier Score")
            axes[idx].set_ylabel("Count")
            axes[idx].grid(alpha=0.3)
        
        plt.tight_layout()
        plot_file = figures_dir / f"score_distribution_{split}.png"
        plt.savefig(plot_file, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved score distribution to {plot_file}")
    
    # Plot 2: ROC curve
    if "roc_curve" in plots_list:
        logger.info("Generating ROC curve")
        
        plt.figure(figsize=(8, 6))
        
        for dataset_name in [k for k in results.keys() if not k.startswith("_")]:
            predictions_file = results_dir / f"{dataset_name}_{split}_predictions.json"
            if not predictions_file.exists():
                continue
            
            predictions = load_json(predictions_file)
            scores = [p["verifier_score"] for p in predictions]
            
            # Build labels
            labels = []
            for p in predictions:
                pred = p["prediction"]
                gt = str(p["ground_truth"]) if not isinstance(p["ground_truth"], dict) else p["ground_truth"].get("label", "")
                
                if pred == "ABSTAIN":
                    labels.append(0)
                elif pred.upper().startswith(gt.upper()):
                    labels.append(1)
                else:
                    labels.append(0)
            
            if len(np.unique(labels)) > 1:
                fpr, tpr, _ = roc_curve(labels, scores)
                auc = results[dataset_name].get("auc", 0)
                plt.plot(fpr, tpr, label=f"{dataset_name} (AUC={auc:.3f})")
        
        plt.plot([0, 1], [0, 1], 'k--', label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve - Verifier Performance")
        plt.legend()
        plt.grid(alpha=0.3)
        
        plot_file = figures_dir / f"roc_curve_{split}.png"
        plt.savefig(plot_file, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved ROC curve to {plot_file}")
    
    # Plot 3: Accuracy by dataset
    if "accuracy_by_dataset" in plots_list:
        logger.info("Generating accuracy by dataset plot")
        
        datasets = [k for k in results.keys() if not k.startswith("_")]
        accuracies = [results[k].get("accuracy", 0) for k in datasets]
        
        plt.figure(figsize=(max(8, len(datasets)*1.5), 6))
        bars = plt.bar(datasets, accuracies, alpha=0.7, edgecolor="black")
        
        # Color bars
        for bar, acc in zip(bars, accuracies):
            if acc >= 0.8:
                bar.set_color("green")
            elif acc >= 0.6:
                bar.set_color("orange")
            else:
                bar.set_color("red")
        
        plt.xlabel("Dataset")
        plt.ylabel("Accuracy")
        plt.title("Accuracy by Dataset")
        plt.xticks(rotation=45, ha="right")
        plt.ylim(0, 1.0)
        plt.grid(alpha=0.3, axis="y")
        
        plot_file = figures_dir / f"accuracy_by_dataset_{split}.png"
        plt.savefig(plot_file, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved accuracy plot to {plot_file}")


def main():
    """Main plot generation function."""
    parser = argparse.ArgumentParser(description="Generate evaluation plots")
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
    
    # Generate plots
    generate_plots(eval_config, args.datasets, args.split)


if __name__ == "__main__":
    main()


