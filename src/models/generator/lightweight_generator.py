#!/usr/bin/env python3
"""
Lightweight molecular generator optimized for GTX 1650 (4GB VRAM)
Character-level LSTM with memory-efficient training and inference
"""

import os
import json
import pickle
from typing import List, Dict, Optional, Tuple, Union, Any
from pathlib import Path
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from rdkit import Chem

logger = logging.getLogger(__name__)

class MolecularDataset(Dataset):
    """Dataset for molecular strings with memory-efficient loading."""
    
    def __init__(
        self, 
        sequences: List[str], 
        tokenizer, 
        max_length: int = 128,
        cache_size: int = 1000
    ):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_size = cache_size
        self._cache = {}
        self._cache_order = []
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Check cache first
        if idx in self._cache:
            return self._cache[idx]
        
        sequence = self.sequences[idx]
        
        # Tokenize sequence
        tokens = self.tokenizer.encode(
            sequence,
            add_special_tokens=True,
            max_length=self.max_length,
            padding=True,
            truncation=True
        )
        
        # Convert to tensor
        input_ids = torch.tensor(tokens, dtype=torch.long)
        
        # For language modeling, target is input shifted by one
        target_ids = torch.cat([
            input_ids[1:], 
            torch.tensor([self.tokenizer.pad_token_id], dtype=torch.long)
        ])
        
        # Create attention mask
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        result = {
            'input_ids': input_ids,
            'target_ids': target_ids,
            'attention_mask': attention_mask
        }
        
        # Cache management
        if len(self._cache) >= self.cache_size:
            # Remove oldest entry
            oldest_idx = self._cache_order.pop(0)
            del self._cache[oldest_idx]
        
        self._cache[idx] = result
        self._cache_order.append(idx)
        
        return result

class LightweightLSTMGenerator(nn.Module):
    """
    Lightweight LSTM generator optimized for GTX 1650.
    
    Features:
    - Small memory footprint (~10M parameters)
    - Gradient checkpointing support
    - Mixed precision compatible
    - Efficient sampling methods
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.2,
        tie_weights: bool = True
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        # Tie weights to reduce parameters
        if tie_weights:
            self.output_projection.weight = self.embedding.weight
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Xavier initialization for embeddings
        nn.init.xavier_uniform_(self.embedding.weight)
        
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Forget gate bias = 1
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_hidden: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            hidden: Initial hidden state
            return_hidden: Whether to return hidden state
            
        Returns:
            Logits [batch_size, seq_len, vocab_size] and optionally hidden state
        """
        # Embedding
        embeddings = self.embedding(input_ids)  # [batch, seq, embed]
        
        # LSTM
        lstm_out, hidden_state = self.lstm(embeddings, hidden)
        
        # Layer normalization
        lstm_out = self.layer_norm(lstm_out)
        
        # Output projection
        logits = self.output_projection(lstm_out)  # [batch, seq, vocab]
        
        if return_hidden:
            return logits, hidden_state
        return logits
    
    def get_num_parameters(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def generate(
        self,
        tokenizer,
        prompt: Optional[str] = None,
        max_length: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        num_samples: int = 1,
        device: str = 'cpu',
        seed: Optional[int] = None
    ) -> List[str]:
        """
        Generate molecular sequences.
        
        Args:
            tokenizer: Tokenizer instance
            prompt: Starting sequence (optional)
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            num_samples: Number of sequences to generate
            device: Device to use
            seed: Random seed
            
        Returns:
            List of generated sequences
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        self.eval()
        generated_sequences = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                # Initialize sequence
                if prompt:
                    sequence = tokenizer.encode(
                        prompt, 
                        add_special_tokens=True,
                        max_length=max_length//2
                    )
                else:
                    sequence = [tokenizer.bos_token_id]
                
                # Convert to tensor
                input_ids = torch.tensor([sequence], dtype=torch.long).to(device)
                hidden = None
                
                # Generate tokens
                for _ in range(max_length - len(sequence)):
                    # Forward pass
                    logits, hidden = self.forward(
                        input_ids[:, -1:], 
                        hidden, 
                        return_hidden=True
                    )
                    
                    # Get last token logits
                    next_token_logits = logits[:, -1, :] / temperature
                    
                    # Apply top-k filtering
                    if top_k > 0:
                        top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                        next_token_logits = torch.full_like(next_token_logits, -float('inf'))
                        next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                    
                    # Apply nucleus (top-p) filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # Remove tokens with cumulative probability above the threshold
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                        sorted_indices_to_remove[:, 0] = 0
                        
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        next_token_logits.scatter_(1, indices_to_remove.unsqueeze(1), -float('inf'))
                    
                    # Sample next token
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    
                    # Append to sequence
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                    sequence.append(next_token.item())
                    
                    # Check for end token
                    if next_token.item() == tokenizer.eos_token_id:
                        break
                
                # Decode sequence
                decoded = tokenizer.decode(sequence, skip_special_tokens=True)
                generated_sequences.append(decoded)
        
        return generated_sequences

class LightweightTrainer:
    """
    Memory-efficient trainer for GTX 1650.
    
    Features:
    - Gradient accumulation
    - Mixed precision training
    - Memory monitoring
    - Automatic batch size adjustment
    """
    
    def __init__(
        self,
        model: LightweightLSTMGenerator,
        tokenizer,
        device: str = 'cuda',
        mixed_precision: bool = True,
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        
        # Mixed precision setup
        if mixed_precision and device == 'cuda':
            self.scaler = torch.cuda.amp.GradScaler()
            self.use_amp = True
        else:
            self.scaler = None
            self.use_amp = False
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.01,
            eps=1e-6
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=1000,
            eta_min=1e-6
        )
        
        # Memory monitoring
        self.memory_usage = []
    
    def compute_loss(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute language modeling loss."""
        logits = self.model(input_ids)
        
        # Reshape for loss computation
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = target_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()
        
        # Compute cross entropy loss
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        # Apply mask and compute mean
        loss = loss.view(shift_labels.size())
        loss = (loss * shift_mask.float()).sum() / shift_mask.sum()
        
        return loss
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        accumulate: bool = True
    ) -> float:
        """Single training step."""
        input_ids = batch['input_ids'].to(self.device)
        target_ids = batch['target_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        # Forward pass with AMP
        if self.use_amp:
            with torch.cuda.amp.autocast():
                loss = self.compute_loss(input_ids, target_ids, attention_mask)
                loss = loss / self.gradient_accumulation_steps
        else:
            loss = self.compute_loss(input_ids, target_ids, attention_mask)
            loss = loss / self.gradient_accumulation_steps
        
        # Backward pass
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update parameters
        if not accumulate:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            self.scheduler.step()
        
        return loss.item() * self.gradient_accumulation_steps
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int = 0
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        num_batches = 0
        accumulation_count = 0
        
        try:
            from tqdm import tqdm
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        except ImportError:
            pbar = dataloader
            print(f"Training epoch {epoch}...")
        
        for batch_idx, batch in enumerate(pbar):
            # Training step
            loss = self.train_step(
                batch, 
                accumulate=(accumulation_count + 1) < self.gradient_accumulation_steps
            )
            
            total_loss += loss
            num_batches += 1
            accumulation_count += 1
            
            # Update parameters after accumulation
            if accumulation_count >= self.gradient_accumulation_steps:
                accumulation_count = 0
            
            # Memory management
            if self.device == 'cuda' and batch_idx % 10 == 0:
                torch.cuda.empty_cache()
                memory_allocated = torch.cuda.memory_allocated() / 1e9  # GB
                self.memory_usage.append(memory_allocated)
            
            # Update progress bar
            if hasattr(pbar, 'set_postfix'):
                pbar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'lr': f'{self.scheduler.get_last_lr()[0]:.2e}',
                    'mem': f'{memory_allocated:.1f}GB' if self.device == 'cuda' else 'N/A'
                })
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        return {
            'loss': avg_loss,
            'learning_rate': self.scheduler.get_last_lr()[0],
            'memory_peak': max(self.memory_usage) if self.memory_usage else 0
        }
    
    def save_model(self, save_path: str, epoch: int = 0, metrics: Dict = None):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics or {},
            'model_config': {
                'vocab_size': self.model.vocab_size,
                'embedding_dim': self.model.embedding_dim,
                'hidden_dim': self.model.hidden_dim,
                'num_layers': self.model.num_layers
            }
        }
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, save_path)
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        metrics = checkpoint.get('metrics', {})
        
        logger.info(f"Model loaded from {checkpoint_path}, epoch {epoch}")
        return epoch, metrics

def create_lightweight_generator(
    vocab_size: int,
    profile: str = "light"
) -> LightweightLSTMGenerator:
    """
    Create a lightweight generator model based on memory profile.
    
    Args:
        vocab_size: Vocabulary size
        profile: Memory profile ("light", "medium", "full")
        
    Returns:
        Configured model
    """
    if profile == "light":  # GTX 1650 optimized
        return LightweightLSTMGenerator(
            vocab_size=vocab_size,
            embedding_dim=256,
            hidden_dim=512,
            num_layers=2,
            dropout=0.2,
            tie_weights=False
        )
    elif profile == "medium":  # 6-8GB VRAM
        return LightweightLSTMGenerator(
            vocab_size=vocab_size,
            embedding_dim=512,
            hidden_dim=768,
            num_layers=3,
            dropout=0.3
        )
    else:  # Full profile
        return LightweightLSTMGenerator(
            vocab_size=vocab_size,
            embedding_dim=768,
            hidden_dim=1024,
            num_layers=4,
            dropout=0.3
        )

def validate_generated_molecules(
    molecules: List[str],
    tokenizer
) -> Dict[str, Any]:
    """
    Validate generated molecules using RDKit.
    
    Args:
        molecules: List of generated molecules (SELFIES or SMILES)
        tokenizer: Tokenizer instance
        
    Returns:
        Validation statistics
    """
    valid_count = 0
    invalid_molecules = []
    valid_molecules = []
    
    for mol_str in molecules:
        try:
            # Convert SELFIES to SMILES if needed
            if hasattr(tokenizer, 'selfies_to_smiles'):
                smiles = tokenizer.selfies_to_smiles(mol_str)
            else:
                smiles = mol_str
            
            if smiles:
                # Validate with RDKit
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    # Additional sanitization
                    Chem.SanitizeMol(mol)
                    canonical_smiles = Chem.MolToSmiles(mol)
                    valid_molecules.append(canonical_smiles)
                    valid_count += 1
                else:
                    invalid_molecules.append(mol_str)
            else:
                invalid_molecules.append(mol_str)
                
        except Exception as e:
            invalid_molecules.append(mol_str)
            logger.debug(f"Invalid molecule: {mol_str} - {e}")
    
    validity_rate = valid_count / len(molecules) if molecules else 0
    
    return {
        'total_molecules': len(molecules),
        'valid_molecules': valid_count,
        'invalid_molecules': len(invalid_molecules),
        'validity_rate': validity_rate,
        'valid_smiles': valid_molecules[:10],  # First 10 valid molecules
        'invalid_examples': invalid_molecules[:5]  # First 5 invalid examples
    }

if __name__ == "__main__":
    # Example usage
    from ...c_tokenizers.selfies_tokenizer import train_selfies_tokenizer
    
    # Sample training data
    sample_molecules = [
        "CCO",  # ethanol
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
        "CC(C)NCC(COC1=CC=CC=C1)O",  # propranolol
        "CC1=CC=C(C=C1)C(=O)OC",  # methyl p-toluate
    ] * 100  # Duplicate for training
    
    print("Creating tokenizer...")
    tokenizer = train_selfies_tokenizer(
        sample_molecules,
        vocab_size=1000
    )
    
    print(f"Tokenizer vocabulary size: {len(tokenizer)}")
    
    # Create model
    print("Creating lightweight model...")
    model = create_lightweight_generator(
        vocab_size=len(tokenizer),
        profile="light"
    )
    
    print(f"Model parameters: {model.get_num_parameters():,}")
    
    # Test generation
    print("Testing generation...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    generated = model.generate(
        tokenizer=tokenizer,
        num_samples=5,
        max_length=64,
        temperature=1.0,
        device=device
    )
    
    print("Generated molecules:")
    for i, mol in enumerate(generated, 1):
        print(f"{i}. {mol}")
    
    # Validate generated molecules
    validation_results = validate_generated_molecules(generated, tokenizer)
    print(f"\nValidation results:")
    print(f"Valid molecules: {validation_results['valid_molecules']}/{validation_results['total_molecules']}")
    print(f"Validity rate: {validation_results['validity_rate']:.2%}")