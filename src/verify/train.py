"""Verifier training script."""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np

from ..utils.io import load_yaml
from ..utils.logging import setup_logging, get_logger
from ..utils.seed import set_seed
from ..data.loaders import DatasetLoader
from .model import VerifierModel
from .dataset import create_verifier_dataset


def train_epoch(
    model: VerifierModel,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    
    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        
        logits = model(input_ids, attention_mask, token_type_ids).squeeze(-1)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate(
    model: VerifierModel,
    eval_loader: DataLoader,
    device: str,
) -> dict:
    """Evaluate model."""
    model.eval()
    all_probs = []
    all_labels = []
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)
            
            logits = model(input_ids, attention_mask, token_type_ids).squeeze(-1)
            loss = criterion(logits, labels)
            
            probs = torch.sigmoid(logits)
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_loss += loss.item()
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Compute AUC if we have both classes
    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_probs)
    else:
        auc = 0.0
    
    return {
        "loss": total_loss / len(eval_loader),
        "auc": auc,
    }


def train_verifier(
    models_config: dict,
    datasets_config_path: str,
    base_config: dict,
) -> None:
    """
    Train verifier model.
    
    Args:
        models_config: Models configuration
        datasets_config_path: Path to datasets configuration
        base_config: Base configuration
    """
    logger = get_logger("train_verifier")
    
    # Get verifier config
    verifier_config = models_config.get("verifier", {})
    backbone = verifier_config.get("backbone", "microsoft/deberta-v3-base")
    max_length = verifier_config.get("max_length", 512)
    lr = verifier_config.get("learning_rate", 3e-5)
    batch_size = verifier_config.get("batch_size", 32)
    eval_batch_size = verifier_config.get("eval_batch_size", 64)
    num_epochs = verifier_config.get("num_epochs", 4)
    early_stopping_patience = verifier_config.get("early_stopping_patience", 2)
    
    device = base_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(base_config.get("output_dir", "artifacts"))
    candidates_dir = output_dir / "candidates"
    save_dir = Path(verifier_config.get("save_dir", output_dir / "verifier"))
    save_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Training verifier with backbone: {backbone}")
    logger.info(f"Device: {device}")
    
    # Initialize model
    model = VerifierModel(backbone=backbone)
    model.to(device)
    
    # Load datasets
    datasets_config = load_yaml(datasets_config_path)
    loader = DatasetLoader(datasets_config_path)
    
    all_train_examples = []
    all_dev_examples = []
    
    for dataset_name in datasets_config.get("datasets", {}).keys():
        if not datasets_config["datasets"][dataset_name].get("enabled", False):
            continue
        
        logger.info(f"Loading data for {dataset_name}")
        
        # Training data
        train_dataset = create_verifier_dataset(
            dataset_name=dataset_name,
            split="train",
            candidates_dir=candidates_dir,
            tokenizer=model.tokenizer,
            max_length=max_length,
            datasets_config=datasets_config_path,
        )
        all_train_examples.extend(train_dataset.examples)
        
        # Dev data
        dev_dataset = create_verifier_dataset(
            dataset_name=dataset_name,
            split="dev",
            candidates_dir=candidates_dir,
            tokenizer=model.tokenizer,
            max_length=max_length,
            datasets_config=datasets_config_path,
        )
        all_dev_examples.extend(dev_dataset.examples)
    
    if not all_train_examples:
        logger.error("No training examples found. Please generate candidates first.")
        return
    
    # Create dataloaders
    from .dataset import VerifierDataset
    train_dataset = VerifierDataset(all_train_examples, model.tokenizer, max_length)
    dev_dataset = VerifierDataset(all_dev_examples, model.tokenizer, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=eval_batch_size, shuffle=False)
    
    logger.info(f"Training examples: {len(train_dataset)}")
    logger.info(f"Dev examples: {len(dev_dataset)}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Training loop
    best_auc = 0.0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        logger.info(f"Train loss: {train_loss:.4f}")
        
        # Evaluate
        eval_metrics = evaluate(model, dev_loader, device)
        logger.info(f"Dev loss: {eval_metrics['loss']:.4f}, AUC: {eval_metrics['auc']:.4f}")
        
        # Early stopping
        if eval_metrics["auc"] > best_auc:
            best_auc = eval_metrics["auc"]
            patience_counter = 0
            
            # Save best model
            model_path = save_dir / "best_model.pt"
            model.save(str(model_path))
            logger.info(f"Saved best model to {model_path}")
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping_patience:
            logger.info(f"Early stopping after {epoch + 1} epochs")
            break
    
    logger.info(f"\nTraining complete! Best AUC: {best_auc:.4f}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train verifier model")
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
    args = parser.parse_args()
    
    # Load configurations
    models_config = load_yaml(args.config)
    base_config = load_yaml(args.base)
    
    # Setup
    setup_logging(
        log_level=base_config.get("log_level", "INFO"),
        log_file=f"{base_config.get('log_dir', 'artifacts/logs')}/train_verifier.log",
    )
    set_seed(base_config.get("seed", 42))
    
    # Train
    train_verifier(models_config, args.datasets, base_config)


if __name__ == "__main__":
    main()


