#!/usr/bin/env python3
"""
Production-ready SELFIES tokenizer implementation
Optimized for molecular representation with robustness guarantees
"""

import os
import json
import pickle
from typing import List, Dict, Optional, Tuple, Union, Any
from pathlib import Path
import logging
from dataclasses import dataclass
from collections import Counter

import selfies as sf
from transformers import PreTrainedTokenizer
import torch
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class TokenizerConfig:
    """Configuration for SELFIES tokenizer."""
    vocab_size: int = 10000
    max_length: int = 256
    pad_token: str = "[PAD]"
    bos_token: str = "[BOS]"
    eos_token: str = "[EOS]"
    unk_token: str = "[UNK]"
    mask_token: str = "[MASK]"
    special_tokens: List[str] = None
    min_frequency: int = 2
    
    def __post_init__(self):
        if self.special_tokens is None:
            self.special_tokens = [
                self.pad_token, self.bos_token, self.eos_token, 
                self.unk_token, self.mask_token
            ]

class SELFIESTokenizer:
    """
    SELFIES-aware tokenizer for molecular representation.
    
    Features:
    - Robust molecular string representation
    - Guaranteed valid molecule generation
    - Memory-efficient vocabulary building
    - Compatible with transformers library
    """
    
    def __init__(self, config: Optional[TokenizerConfig] = None):
        self.config = config or TokenizerConfig()
        self.vocab: Dict[str, int] = {}
        self.reverse_vocab: Dict[int, str] = {}
        self.token_frequencies: Dict[str, int] = {}
        self._is_trained = False
        
        # Initialize special tokens
        self._initialize_special_tokens()
    
    def _initialize_special_tokens(self):
        """Initialize special tokens vocabulary."""
        for i, token in enumerate(self.config.special_tokens):
            self.vocab[token] = i
            self.reverse_vocab[i] = token
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)
    
    @property
    def pad_token_id(self) -> int:
        """Get padding token ID."""
        return self.vocab[self.config.pad_token]
    
    @property
    def bos_token_id(self) -> int:
        """Get beginning of sequence token ID."""
        return self.vocab[self.config.bos_token]
    
    @property
    def eos_token_id(self) -> int:
        """Get end of sequence token ID."""
        return self.vocab[self.config.eos_token]
    
    @property
    def unk_token_id(self) -> int:
        """Get unknown token ID."""
        return self.vocab[self.config.unk_token]
    
    @property
    def mask_token_id(self) -> int:
        """Get mask token ID."""
        return self.vocab[self.config.mask_token]
    
    def smiles_to_selfies(self, smiles: str) -> Optional[str]:
        """
        Convert SMILES to SELFIES with error handling.
        
        Args:
            smiles: SMILES string
            
        Returns:
            SELFIES string or None if conversion fails
        """
        try:
            return sf.encoder(smiles)
        except Exception as e:
            logger.warning(f"Failed to convert SMILES to SELFIES: {smiles} - {e}")
            return None
    
    def selfies_to_smiles(self, selfies: str) -> Optional[str]:
        """
        Convert SELFIES to SMILES with error handling.
        
        Args:
            selfies: SELFIES string
            
        Returns:
            SMILES string or None if conversion fails
        """
        try:
            return sf.decoder(selfies)
        except Exception as e:
            logger.warning(f"Failed to convert SELFIES to SMILES: {selfies} - {e}")
            return None
    
    def tokenize_selfies(self, selfies: str) -> List[str]:
        """
        Tokenize SELFIES string into individual tokens.
        
        Args:
            selfies: SELFIES string
            
        Returns:
            List of SELFIES tokens
        """
        if not selfies:
            return []
        
        # SELFIES tokens are bracketed, split by brackets
        tokens = []
        current_token = ""
        in_bracket = False
        
        for char in selfies:
            if char == '[':
                if current_token and in_bracket:
                    tokens.append(current_token + ']')
                current_token = '['
                in_bracket = True
            elif char == ']':
                if in_bracket:
                    current_token += ']'
                    tokens.append(current_token)
                    current_token = ""
                    in_bracket = False
            else:
                current_token += char
        
        # Handle any remaining token
        if current_token and in_bracket:
            tokens.append(current_token + ']')
        
        return tokens
    
    def train_from_smiles(self, smiles_list: List[str], show_progress: bool = True) -> None:
        """
        Train tokenizer from SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            show_progress: Show training progress
        """
        if show_progress:
            try:
                from tqdm import tqdm
                smiles_iter = tqdm(smiles_list, desc="Converting SMILES to SELFIES")
            except ImportError:
                smiles_iter = smiles_list
                logger.info(f"Training tokenizer on {len(smiles_list)} molecules")
        else:
            smiles_iter = smiles_list
        
        # Convert SMILES to SELFIES and collect tokens
        all_tokens = []
        successful_conversions = 0
        
        for smiles in smiles_iter:
            selfies = self.smiles_to_selfies(smiles)
            if selfies:
                tokens = self.tokenize_selfies(selfies)
                all_tokens.extend(tokens)
                successful_conversions += 1
        
        logger.info(f"Successfully converted {successful_conversions}/{len(smiles_list)} molecules")
        
        # Count token frequencies
        self.token_frequencies = Counter(all_tokens)
        
        # Build vocabulary
        self._build_vocabulary()
        self._is_trained = True
        
        logger.info(f"Tokenizer trained with vocabulary size: {self.vocab_size}")
    
    def train_from_selfies(self, selfies_list: List[str], show_progress: bool = True) -> None:
        """
        Train tokenizer from SELFIES strings.
        
        Args:
            selfies_list: List of SELFIES strings
            show_progress: Show training progress
        """
        if show_progress:
            try:
                from tqdm import tqdm
                selfies_iter = tqdm(selfies_list, desc="Processing SELFIES")
            except ImportError:
                selfies_iter = selfies_list
                logger.info(f"Training tokenizer on {len(selfies_list)} molecules")
        else:
            selfies_iter = selfies_list
        
        # Collect all tokens
        all_tokens = []
        for selfies in selfies_iter:
            tokens = self.tokenize_selfies(selfies)
            all_tokens.extend(tokens)
        
        # Count token frequencies
        self.token_frequencies = Counter(all_tokens)
        
        # Build vocabulary
        self._build_vocabulary()
        self._is_trained = True
        
        logger.info(f"Tokenizer trained with vocabulary size: {self.vocab_size}")
    
    def _build_vocabulary(self) -> None:
        """Build vocabulary from token frequencies."""
        # Start with special tokens already in vocab
        vocab_size_without_special = len(self.config.special_tokens)
        
        # Sort tokens by frequency (descending)
        sorted_tokens = sorted(
            self.token_frequencies.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Add tokens to vocabulary until we reach max vocab size
        for token, freq in sorted_tokens:
            if len(self.vocab) >= self.config.vocab_size:
                break
            
            if (freq >= self.config.min_frequency and 
                token not in self.vocab):
                
                token_id = len(self.vocab)
                self.vocab[token] = token_id
                self.reverse_vocab[token_id] = token
        
        logger.info(f"Built vocabulary with {len(self.vocab)} tokens")
        logger.info(f"Most common tokens: {sorted_tokens[:10]}")
    
    def encode(
        self, 
        text: str, 
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False
    ) -> List[int]:
        """
        Encode text (SELFIES or SMILES) to token IDs.
        
        Args:
            text: Input text (SELFIES or SMILES)
            add_special_tokens: Add BOS/EOS tokens
            max_length: Maximum sequence length
            padding: Pad to max_length
            truncation: Truncate to max_length
            
        Returns:
            List of token IDs
        """
        if not self._is_trained:
            raise RuntimeError("Tokenizer must be trained before encoding")
        
        # Convert SMILES to SELFIES if needed
        if text and not text.startswith('['):
            # Likely SMILES, convert to SELFIES
            selfies = self.smiles_to_selfies(text)
            if selfies is None:
                logger.warning(f"Failed to convert to SELFIES: {text}")
                return [self.unk_token_id]
        else:
            selfies = text
        
        # Tokenize
        tokens = self.tokenize_selfies(selfies)
        
        # Convert to IDs
        token_ids = []
        
        if add_special_tokens:
            token_ids.append(self.bos_token_id)
        
        for token in tokens:
            token_id = self.vocab.get(token, self.unk_token_id)
            token_ids.append(token_id)
        
        if add_special_tokens:
            token_ids.append(self.eos_token_id)
        
        # Apply max_length constraints
        if max_length is None:
            max_length = self.config.max_length
        
        # Truncation
        if truncation and len(token_ids) > max_length:
            if add_special_tokens:
                # Keep BOS, truncate middle, keep EOS
                token_ids = (
                    token_ids[:max_length-1] + [self.eos_token_id]
                )
            else:
                token_ids = token_ids[:max_length]
        
        # Padding
        if padding and len(token_ids) < max_length:
            token_ids.extend([self.pad_token_id] * (max_length - len(token_ids)))
        
        return token_ids
    
    def decode(
        self, 
        token_ids: List[int], 
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decode token IDs back to SELFIES string.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Skip special tokens in output
            
        Returns:
            SELFIES string
        """
        tokens = []
        
        for token_id in token_ids:
            if token_id in self.reverse_vocab:
                token = self.reverse_vocab[token_id]
                
                if skip_special_tokens and token in self.config.special_tokens:
                    continue
                
                tokens.append(token)
            else:
                logger.warning(f"Unknown token ID: {token_id}")
                if not skip_special_tokens:
                    tokens.append(self.config.unk_token)
        
        return ''.join(tokens)
    
    def batch_encode(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
        return_tensors: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Batch encode multiple texts.
        
        Args:
            texts: List of input texts
            add_special_tokens: Add BOS/EOS tokens
            max_length: Maximum sequence length
            padding: Pad sequences
            truncation: Truncate sequences
            return_tensors: Return format ('pt' for PyTorch)
            
        Returns:
            Dictionary with input_ids and attention_mask
        """
        if max_length is None:
            max_length = self.config.max_length
        
        # Encode all texts
        all_token_ids = []
        for text in texts:
            token_ids = self.encode(
                text, 
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                padding=padding,
                truncation=truncation
            )
            all_token_ids.append(token_ids)
        
        # Create attention masks
        attention_masks = []
        for token_ids in all_token_ids:
            mask = [1 if token_id != self.pad_token_id else 0 for token_id in token_ids]
            attention_masks.append(mask)
        
        result = {
            'input_ids': all_token_ids,
            'attention_mask': attention_masks
        }
        
        # Convert to tensors if requested
        if return_tensors == 'pt':
            result['input_ids'] = torch.tensor(result['input_ids'], dtype=torch.long)
            result['attention_mask'] = torch.tensor(result['attention_mask'], dtype=torch.long)
        elif return_tensors == 'np':
            result['input_ids'] = np.array(result['input_ids'])
            result['attention_mask'] = np.array(result['attention_mask'])
        
        return result
    
    def save(self, directory: Union[str, Path]) -> None:
        """
        Save tokenizer to directory.
        
        Args:
            directory: Directory path to save tokenizer
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_dict = {
            'vocab_size': self.config.vocab_size,
            'max_length': self.config.max_length,
            'pad_token': self.config.pad_token,
            'bos_token': self.config.bos_token,
            'eos_token': self.config.eos_token,
            'unk_token': self.config.unk_token,
            'mask_token': self.config.mask_token,
            'special_tokens': self.config.special_tokens,
            'min_frequency': self.config.min_frequency
        }
        
        with open(directory / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Save vocabulary
        with open(directory / 'vocab.json', 'w') as f:
            json.dump(self.vocab, f, indent=2)
        
        # Save token frequencies
        with open(directory / 'frequencies.json', 'w') as f:
            json.dump(self.token_frequencies, f, indent=2)
        
        # Save training status
        with open(directory / 'training_info.json', 'w') as f:
            json.dump({
                'is_trained': self._is_trained,
                'vocab_size': len(self.vocab),
                'num_special_tokens': len(self.config.special_tokens)
            }, f, indent=2)
        
        logger.info(f"Tokenizer saved to {directory}")
    
    @classmethod
    def load(cls, directory: Union[str, Path]) -> 'SELFIESTokenizer':
        """
        Load tokenizer from directory.
        
        Args:
            directory: Directory path to load tokenizer from
            
        Returns:
            Loaded SELFIESTokenizer instance
        """
        directory = Path(directory)
        
        # Load configuration
        with open(directory / 'config.json', 'r') as f:
            config_dict = json.load(f)
        
        # Get all valid field names from the TokenizerConfig dataclass
        valid_keys = TokenizerConfig.__annotations__.keys()

        # Filter the loaded config dict to only include valid keys
        filtered_config_dict = {
        k: v for k, v in config_dict.items() if k in valid_keys
        }

        config = TokenizerConfig(**config_dict)
        
        # Create tokenizer instance
        tokenizer = cls(config)
        
        # Load vocabulary
        with open(directory / 'vocab.json', 'r') as f:
            tokenizer.vocab = json.load(f)
        
        # Rebuild reverse vocabulary
        tokenizer.reverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
        
        # Load token frequencies
        try:
            with open(directory / 'frequencies.json', 'r') as f:
                tokenizer.token_frequencies = json.load(f)
        except FileNotFoundError:
            logger.warning("Frequencies file not found, initializing empty")
            tokenizer.token_frequencies = {}
        
        # Load training info
        try:
            with open(directory / 'training_info.json', 'r') as f:
                training_info = json.load(f)
                tokenizer._is_trained = training_info.get('is_trained', True)
        except FileNotFoundError:
            logger.warning("Training info not found, assuming trained")
            tokenizer._is_trained = True
        
        logger.info(f"Tokenizer loaded from {directory}")
        logger.info(f"Vocabulary size: {len(tokenizer.vocab)}")
        
        return tokenizer
    
    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary dictionary."""
        return self.vocab.copy()
    
    def __len__(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)
    
    def __call__(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
        return_tensors: Optional[str] = None
    ) -> Union[List[int], Dict[str, Any]]:
        """
        Tokenize text (callable interface).
        
        Args:
            text: Input text or list of texts
            add_special_tokens: Add BOS/EOS tokens
            max_length: Maximum sequence length
            padding: Pad sequences
            truncation: Truncate sequences
            return_tensors: Return format
            
        Returns:
            Token IDs or batch encoding dictionary
        """
        if isinstance(text, str):
            return self.encode(
                text,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                padding=padding,
                truncation=truncation
            )
        else:
            return self.batch_encode(
                text,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                padding=padding,
                truncation=truncation,
                return_tensors=return_tensors
            )

def train_selfies_tokenizer(
    molecules: List[str],
    vocab_size: int = 10000,
    max_length: int = 256,
    min_frequency: int = 2,
    save_path: Optional[str] = None,
    format_type: str = "smiles"
) -> SELFIESTokenizer:
    """
    Train a SELFIES tokenizer from molecular data.
    
    Args:
        molecules: List of molecular strings (SMILES or SELFIES)
        vocab_size: Maximum vocabulary size
        max_length: Maximum sequence length
        min_frequency: Minimum token frequency to include in vocabulary
        save_path: Path to save trained tokenizer
        format_type: Input format ("smiles" or "selfies")
        
    Returns:
        Trained SELFIESTokenizer
    """
    # Create configuration
    config = TokenizerConfig(
        vocab_size=vocab_size,
        max_length=max_length,
        min_frequency=min_frequency
    )
    
    # Create tokenizer
    tokenizer = SELFIESTokenizer(config)
    
    # Train tokenizer
    if format_type.lower() == "smiles":
        tokenizer.train_from_smiles(molecules, show_progress=True)
    else:
        tokenizer.train_from_selfies(molecules, show_progress=True)
    
    # Save if path provided
    if save_path:
        tokenizer.save(save_path)
    
    return tokenizer

if __name__ == "__main__":
    # Example usage
    sample_smiles = [
        "CCO",  # ethanol
        "CC(=O)OC1=CC=CC=C1C(=O)O",  # aspirin
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # caffeine
        "CC(C)NCC(COC1=CC=CC=C1)O",  # propranolol
    ]
    
    print("Training SELFIES tokenizer...")
    tokenizer = train_selfies_tokenizer(
        sample_smiles,
        vocab_size=1000,
        save_path="./selfies_tokenizer"
    )
    
    print("\nTesting tokenizer...")
    test_smiles = "CCO"
    encoded = tokenizer.encode(test_smiles)
    decoded = tokenizer.decode(encoded)
    
    print(f"Original SMILES: {test_smiles}")
    print(f"Encoded: {encoded}")
    print(f"Decoded SELFIES: {decoded}")
    
    # Convert back to SMILES
    decoded_smiles = tokenizer.selfies_to_smiles(decoded)
    print(f"Back to SMILES: {decoded_smiles}")