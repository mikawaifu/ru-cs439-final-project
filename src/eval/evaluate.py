"""Evaluation script for LegalAdapter."""

import argparse
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

from ..utils.io import load_yaml, load_json, save_json
from ..utils.logging import setup_logging, get_logger
from ..utils.metrics import compute_accuracy, compute_f1_batch, compute_auc
from ..utils.hallucination import compute_hallucination_rate


def evaluate_predictions(
    eval_config: Dict[str, Any],
    datasets_config_path: str,
    split: str = "test",
) -> Dict[str, Any]:
    """
    Evaluate predictions and compute metrics.
    
    Args:
        eval_config: Evaluation configuration
        datasets_config_path: Path to datasets configuration
        split: Dataset split to evaluate
        
    Returns:
        Dictionary of evaluation results
    """
    logger = get_logger("evaluate")
    
    # Get config
    metrics = eval_config.get("metrics", ["accuracy", "f1", "auc_verifier", "hallucination_rate"])
    accuracy_config = eval_config.get("accuracy", {})
    f1_config = eval_config.get("f1", {})
    auc_config = eval_config.get("auc_verifier", {})
    hallu_config = eval_config.get("hallucination", {})
    report_config = eval_config.get("report", {})
    
    results_dir = Path(report_config.get("output_dir", "artifacts/results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load datasets config
    datasets_config = load_yaml(datasets_config_path)
    
    all_results = {}
    
    for dataset_name in datasets_config.get("datasets", {}).keys():
        if not datasets_config["datasets"][dataset_name].get("enabled", False):
            continue
        
        logger.info(f"\nEvaluating {dataset_name} ({split})")
        
        # Load predictions
        predictions_file = results_dir / f"{dataset_name}_{split}_predictions.json"
        if not predictions_file.exists():
            logger.warning(f"Predictions not found: {predictions_file}")
            continue
        
        predictions = load_json(predictions_file)
        
        # Extract data
        pred_texts = [p["prediction"] for p in predictions]
        gt_texts = [str(p["ground_truth"]) if not isinstance(p["ground_truth"], dict) 
                   else p["ground_truth"].get("label", "") for p in predictions]
        contexts = [p.get("context") for p in predictions]
        scores = [p["verifier_score"] for p in predictions]
        
        # Build labels for AUC (1 if prediction matches ground truth, 0 otherwise)
        labels = []
        for pred, gt in zip(pred_texts, gt_texts):
            if pred == "ABSTAIN":
                labels.append(0)
            elif pred.upper().startswith(gt.upper()):
                labels.append(1)
            else:
                labels.append(0)
        
        # Compute metrics
        dataset_results = {}
        
        # Accuracy
        if "accuracy" in metrics:
            acc = compute_accuracy(
                pred_texts,
                gt_texts,
                strict=accuracy_config.get("strict", True),
                case_sensitive=accuracy_config.get("case_sensitive", False),
            )
            dataset_results["accuracy"] = acc
            logger.info(f"Accuracy: {acc:.4f}")
        
        # F1
        if "f1" in metrics:
            f1_results = compute_f1_batch(
                pred_texts,
                gt_texts,
                lowercase=f1_config.get("lowercase", True),
                remove_punctuation=f1_config.get("remove_punctuation", True),
            )
            dataset_results["f1"] = f1_results["f1"]
            dataset_results["precision"] = f1_results["precision"]
            dataset_results["recall"] = f1_results["recall"]
            logger.info(f"F1: {f1_results['f1']:.4f}")
        
        # Verifier AUC
        if "auc_verifier" in metrics:
            if auc_config.get("bootstrap", False):
                auc, lower, upper = compute_auc(
                    labels,
                    scores,
                    bootstrap=True,
                    n_bootstrap=auc_config.get("n_bootstrap", 1000),
                    confidence_level=auc_config.get("confidence_level", 0.95),
                )
                dataset_results["auc"] = auc
                dataset_results["auc_ci_lower"] = lower
                dataset_results["auc_ci_upper"] = upper
                logger.info(f"AUC: {auc:.4f} [{lower:.4f}, {upper:.4f}]")
            else:
                auc = compute_auc(labels, scores, bootstrap=False)
                dataset_results["auc"] = auc
                logger.info(f"AUC: {auc:.4f}")
        
        # Hallucination rate
        if "hallucination_rate" in metrics and hallu_config.get("enabled", True):
            hallu_results = compute_hallucination_rate(
                pred_texts,
                contexts,
                check_context_grounding=hallu_config.get("check_context_grounding", True),
                check_fabricated_citations=hallu_config.get("check_fabricated_citations", True),
                min_confidence=hallu_config.get("min_confidence", 0.7),
            )
            dataset_results["hallucination_rate"] = hallu_results["hallucination_rate"]
            dataset_results["total_hallucinations"] = hallu_results["total_hallucinations"]
            logger.info(f"Hallucination Rate: {hallu_results['hallucination_rate']:.4f}")
        
        # Additional stats
        dataset_results["total_examples"] = len(predictions)
        dataset_results["abstain_count"] = sum(1 for p in pred_texts if p == "ABSTAIN")
        dataset_results["abstain_rate"] = dataset_results["abstain_count"] / len(predictions)
        
        all_results[dataset_name] = dataset_results
    
    # Aggregate results
    if len(all_results) > 1:
        aggregate = {}
        for metric in ["accuracy", "f1", "auc", "hallucination_rate", "abstain_rate"]:
            values = [r[metric] for r in all_results.values() if metric in r]
            if values:
                aggregate[f"{metric}_mean"] = float(np.mean(values))
                aggregate[f"{metric}_std"] = float(np.std(values))
        
        all_results["_aggregate"] = aggregate
    
    # Save results
    results_file = results_dir / f"evaluation_results_{split}.json"
    save_json(all_results, results_file, indent=2)
    logger.info(f"\nSaved evaluation results to {results_file}")
    
    return all_results


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate predictions")
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
        help="Dataset split to evaluate",
    )
    args = parser.parse_args()
    
    # Load configuration
    eval_config = load_yaml(args.config)
    
    # Setup logging
    setup_logging(log_level="INFO")
    
    # Evaluate
    evaluate_predictions(eval_config, args.datasets, args.split)


if __name__ == "__main__":
    main()


