#!/usr/bin/env python3
"""
Training script for molecular generator models
Supports LSTM, Transformer, and GPT-2 architectures with GTX 1650 optimization
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / 'src'))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_training_data(data_path: Path, tokenizer):
    """Load and prepare training data."""
    logger.info(f"Loading training data from {data_path}")
    
    molecules = []
    with open(data_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                molecules.append(line)
    
    logger.info(f"Loaded {len(molecules)} molecules")
    return molecules

def train_generator(
    model_type: str,
    profile: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    data_dir: Path,
    output_dir: Path,
    device: str = 'auto'
):
    """Train a molecular generator model."""
    
    logger.info("="*60)
    logger.info("GENERATOR TRAINING")
    logger.info("="*60)
    logger.info(f"Model type: {model_type}")
    logger.info(f"Profile: {profile}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    
    # Auto-detect device
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f} GB)")
            
            # Verify profile matches GPU
            if profile == 'light' and gpu_memory > 6:
                logger.warning(f"GPU has {gpu_memory:.1f}GB but using 'light' profile. Consider 'medium' or 'full'.")
            elif profile == 'full' and gpu_memory < 10:
                logger.warning(f"GPU has only {gpu_memory:.1f}GB but using 'full' profile. May encounter OOM errors.")
        else:
            device = 'cpu'
            logger.info("Using CPU (no GPU detected)")
    
    # Load tokenizer
    tokenizer_path = project_root / "data" / "models" / "tokenizer"
    if not tokenizer_path.exists():
        logger.error(f"Tokenizer not found at {tokenizer_path}")
        logger.info("Run: python scripts/download_models.py --profile " + profile)
        return False
    
    logger.info(f"Loading tokenizer from {tokenizer_path}")
    
    try:
        from tokenizers.selfies_tokenizer import SELFIESTokenizer
        tokenizer = SELFIESTokenizer.load(tokenizer_path)
        logger.info(f"Tokenizer loaded with vocab size: {len(tokenizer)}")
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        return False
    
    # Load training data
    train_data_path = data_dir / "training_molecules.txt"
    if not train_data_path.exists():
        logger.error(f"Training data not found at {train_data_path}")
        logger.info("Run: python scripts/download_models.py --data-only")
        return False
    
    train_molecules = load_training_data(train_data_path, tokenizer)
    
    # Load validation data
    val_data_path = data_dir / "validation_molecules.txt"
    val_molecules = load_training_data(val_data_path, tokenizer) if val_data_path.exists() else None
    
    # Create model
    logger.info(f"Creating {model_type} model...")
    
    try:
        if model_type == 'lstm':
            from models.generator.lightweight_generator import (
                create_lightweight_generator,
                LightweightTrainer,
                MolecularDataset
            )
            
            model = create_lightweight_generator(
                vocab_size=len(tokenizer),
                profile=profile
            )
            
            logger.info(f"Model created with {model.get_num_parameters():,} parameters")
            
            # Create dataset and dataloader
            train_dataset = MolecularDataset(
                train_molecules,
                tokenizer,
                max_length=128 if profile == 'light' else 256
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,  # Avoid multiprocessing issues
                pin_memory=(device == 'cuda')
            )
            
            # Create trainer
            gradient_accumulation = 4 if profile == 'light' else 2
            trainer = LightweightTrainer(
                model=model,
                tokenizer=tokenizer,
                device=device,
                mixed_precision=(device == 'cuda' and profile != 'full'),
                gradient_accumulation_steps=gradient_accumulation
            )
            
            # Training loop
            logger.info("\nStarting training...")
            best_loss = float('inf')
            
            for epoch in range(epochs):
                metrics = trainer.train_epoch(train_loader, epoch=epoch)
                
                logger.info(f"\nEpoch {epoch+1}/{epochs}")
                logger.info(f"  Loss: {metrics['loss']:.4f}")
                logger.info(f"  Learning Rate: {metrics['learning_rate']:.2e}")
                
                if device == 'cuda':
                    logger.info(f"  Peak Memory: {metrics['memory_peak']:.1f} GB")
                
                # Save best model
                if metrics['loss'] < best_loss:
                    best_loss = metrics['loss']
                    save_path = output_dir / f"generator_{model_type}_best.pt"
                    trainer.save_model(save_path, epoch=epoch, metrics=metrics)
                    logger.info(f"  ✓ Saved best model (loss: {best_loss:.4f})")
                
                # Save checkpoint every 5 epochs
                if (epoch + 1) % 5 == 0:
                    checkpoint_path = output_dir / f"generator_{model_type}_epoch{epoch+1}.pt"
                    trainer.save_model(checkpoint_path, epoch=epoch, metrics=metrics)
            
            # Final save
            final_path = output_dir / f"generator_{model_type}_final.pt"
            trainer.save_model(final_path, epoch=epochs, metrics=metrics)
            
            logger.info("\n" + "="*60)
            logger.info("TRAINING COMPLETE")
            logger.info("="*60)
            logger.info(f"Best loss: {best_loss:.4f}")
            logger.info(f"Final model: {final_path}")
            logger.info(f"Best model: {output_dir / f'generator_{model_type}_best.pt'}")
            
            return True
            
        else:
            logger.error(f"Model type '{model_type}' not yet implemented")
            logger.info("Available: lstm")
            logger.info("Coming soon: transformer, gpt2")
            return False
            
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train molecular generator models"
    )
    
    parser.add_argument(
        '--model_type',
        choices=['lstm', 'transformer', 'gpt2'],
        default='lstm',
        help='Model architecture to train'
    )
    
    parser.add_argument(
        '--profile',
        choices=['light', 'medium', 'full'],
        default='light',
        help='Memory profile (default: light for GTX 1650)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Batch size (auto-selected based on profile if not specified)'
    )
    
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-4,
        help='Initial learning rate'
    )
    
    parser.add_argument(
        '--data_dir',
        type=Path,
        default=None,
        help='Directory containing training data'
    )
    
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=None,
        help='Directory to save trained models'
    )
    
    parser.add_argument(
        '--device',
        choices=['auto', 'cuda', 'cpu'],
        default='auto',
        help='Device to use for training'
    )
    
    args = parser.parse_args()
    
    # Set default paths
    if args.data_dir is None:
        args.data_dir = project_root / "data" / "raw"
    
    if args.output_dir is None:
        args.output_dir = project_root / "data" / "models" / "generator"
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Auto-select batch size based on profile
    if args.batch_size is None:
        batch_sizes = {
            'light': 1,
            'medium': 8,
            'full': 16
        }
        args.batch_size = batch_sizes[args.profile]
        logger.info(f"Auto-selected batch size: {args.batch_size} for profile '{args.profile}'")
    
    # Train model
    success = train_generator(
        model_type=args.model_type,
        profile=args.profile,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        device=args.device
    )
    
    if not success:
        sys.exit(1)
    
    logger.info("\nNext steps:")
    logger.info("1. Test generation: python scripts/test_generator.py")
    logger.info("2. Run the app: streamlit run app/main.py")

if __name__ == "__main__":
    main()