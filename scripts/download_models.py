#!/usr/bin/env python3
"""
Model Download and Setup Script for Drug Discovery Assistant
Downloads and prepares pre-trained models optimized for different memory profiles
"""

import os
import sys
import json
import argparse
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import urllib.request
import shutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelDownloader:
    """Handle model downloads and setup for Drug Discovery Assistant."""
    
    # Model configurations for different profiles
    MODEL_CONFIGS = {
        "light": {
            "description": "Optimized for GTX 1650 (4GB VRAM)",
            "models": {
                "tokenizer": {
                    "name": "SELFIES Tokenizer (Light)",
                    "size": "5MB",
                    "type": "selfies",
                    "required": True
                },
                "generator": {
                    "name": "Lightweight LSTM Generator",
                    "size": "40MB",
                    "params": "10M",
                    "type": "lstm",
                    "required": True
                },
                "predictor_solubility": {
                    "name": "Solubility Predictor (Fingerprint+XGBoost)",
                    "size": "15MB",
                    "type": "xgboost",
                    "required": False
                },
                "predictor_toxicity": {
                    "name": "Toxicity Classifier (Lightweight)",
                    "size": "25MB",
                    "type": "classifier",
                    "required": False
                }
            }
        },
        "medium": {
            "description": "Optimized for 6-8GB VRAM",
            "models": {
                "tokenizer": {
                    "name": "SELFIES Tokenizer (Medium)",
                    "size": "10MB",
                    "type": "selfies",
                    "required": True
                },
                "generator": {
                    "name": "Small Transformer Generator",
                    "size": "200MB",
                    "params": "50M",
                    "type": "transformer",
                    "required": True
                },
                "predictor_gnn": {
                    "name": "Graph Neural Network Predictor",
                    "size": "100MB",
                    "type": "gnn",
                    "required": False
                }
            }
        },
        "full": {
            "description": "Full performance for 12GB+ VRAM",
            "models": {
                "tokenizer": {
                    "name": "SELFIES Tokenizer (Full)",
                    "size": "20MB",
                    "type": "selfies",
                    "required": True
                },
                "generator": {
                    "name": "GPT-2 Fine-tuned Generator",
                    "size": "500MB",
                    "params": "124M",
                    "type": "gpt2",
                    "required": True
                },
                "predictor_deepchem": {
                    "name": "DeepChem Multi-task Predictor",
                    "size": "300MB",
                    "type": "deepchem",
                    "required": False
                }
            }
        },
        "colab": {
            "description": "Optimized for Google Colab (GPU runtime)",
            "models": {
                "tokenizer": {
                    "name": "SELFIES Tokenizer (Colab)",
                    "size": "10MB",
                    "type": "selfies",
                    "required": True
                },
                "generator": {
                    "name": "Small Transformer Generator",
                    "size": "200MB",
                    "params": "50M",
                    "type": "transformer",
                    "required": True
                },
                "predictor_hybrid": {
                    "name": "Hybrid Predictor",
                    "size": "80MB",
                    "type": "hybrid",
                    "required": False
                }
            }
        }
    }
    
    # Sample training data URLs (replace with actual model hosting)
    SAMPLE_DATA_URLS = {
        "chembl_subset": "https://raw.githubusercontent.com/rdkit/rdkit/master/Data/NCI/first_200.props.sdf",
        "zinc_subset": None  # Will generate synthetic data
    }
    
    def __init__(self, profile: str = "light", force: bool = False):
        self.profile = profile
        self.force = force
        self.project_root = Path(__file__).parent.parent
        self.models_dir = self.project_root / "data" / "models"
        self.data_dir = self.project_root / "data" / "raw"
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Model downloader initialized for profile: {profile}")
        logger.info(f"Models directory: {self.models_dir}")
    
    def get_profile_info(self) -> Dict:
        """Get information about the selected profile."""
        if self.profile not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown profile: {self.profile}. Available: {list(self.MODEL_CONFIGS.keys())}")
        return self.MODEL_CONFIGS[self.profile]
    
    def list_models(self):
        """List all models for the current profile."""
        profile_info = self.get_profile_info()
        
        logger.info("="*60)
        logger.info(f"Profile: {self.profile}")
        logger.info(f"Description: {profile_info['description']}")
        logger.info("="*60)
        
        total_size = 0
        required_size = 0
        
        for model_id, model_info in profile_info['models'].items():
            status = "REQUIRED" if model_info['required'] else "OPTIONAL"
            size_mb = int(model_info['size'].replace('MB', ''))
            
            logger.info(f"\n{model_info['name']}")
            logger.info(f"  Type: {model_info['type']}")
            logger.info(f"  Size: {model_info['size']}")
            logger.info(f"  Status: {status}")
            
            if 'params' in model_info:
                logger.info(f"  Parameters: {model_info['params']}")
            
            total_size += size_mb
            if model_info['required']:
                required_size += size_mb
        
        logger.info("\n" + "="*60)
        logger.info(f"Total size: ~{total_size}MB")
        logger.info(f"Required size: ~{required_size}MB")
        logger.info("="*60)
    
    def create_tokenizer(self, model_id: str) -> bool:
        """Create and save a pre-trained tokenizer."""
        logger.info(f"Creating tokenizer: {model_id}")
        
        try:
            # Import tokenizer class
            sys.path.append(str(self.project_root / 'src' / 'tokenizers'))
            
            # Create sample molecules for training
            sample_molecules = self._get_sample_molecules()
            
            # Create tokenizer directory
            tokenizer_dir = self.models_dir / model_id
            tokenizer_dir.mkdir(parents=True, exist_ok=True)
            
            # Save tokenizer configuration
            config = {
                "vocab_size": 1000 if self.profile == "light" else 5000,
                "max_length": 128 if self.profile == "light" else 256,
                "pad_token": "[PAD]",
                "bos_token": "[BOS]",
                "eos_token": "[EOS]",
                "unk_token": "[UNK]",
                "mask_token": "[MASK]",
                "special_tokens": ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "[MASK]"],
                "min_frequency": 2,
                "model_type": "selfies",
                "profile": self.profile
            }
            
            with open(tokenizer_dir / 'config.json', 'w') as f:
                json.dump(config, f, indent=2)
            
            # Create vocabulary from sample molecules
            vocab = self._build_vocab_from_molecules(sample_molecules)
            
            with open(tokenizer_dir / 'vocab.json', 'w') as f:
                json.dump(vocab, f, indent=2)
            
            # Save training info
            training_info = {
                "is_trained": True,
                "vocab_size": len(vocab),
                "num_special_tokens": len(config['special_tokens']),
                "training_molecules": len(sample_molecules),
                "profile": self.profile
            }
            
            with open(tokenizer_dir / 'training_info.json', 'w') as f:
                json.dump(training_info, f, indent=2)
            
            logger.info(f"✓ Tokenizer created with {len(vocab)} tokens")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create tokenizer: {e}")
            return False
    
    def create_generator_model(self, model_id: str) -> bool:
        """Create a placeholder generator model configuration."""
        logger.info(f"Creating generator model: {model_id}")
        
        try:
            model_dir = self.models_dir / model_id
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Model configuration based on profile
            if self.profile == "light":
                config = {
                    "model_type": "lstm",
                    "vocab_size": 1000,
                    "embedding_dim": 256,
                    "hidden_dim": 512,
                    "num_layers": 2,
                    "dropout": 0.2,
                    "tie_weights": True,
                    "profile": "light"
                }
            elif self.profile == "medium":
                config = {
                    "model_type": "transformer",
                    "vocab_size": 5000,
                    "embedding_dim": 512,
                    "hidden_dim": 768,
                    "num_layers": 6,
                    "num_heads": 8,
                    "dropout": 0.3,
                    "profile": "medium"
                }
            else:  # full
                config = {
                    "model_type": "gpt2",
                    "vocab_size": 10000,
                    "embedding_dim": 768,
                    "hidden_dim": 1024,
                    "num_layers": 12,
                    "num_heads": 12,
                    "dropout": 0.3,
                    "profile": "full"
                }
            
            # Save model config
            with open(model_dir / 'config.json', 'w') as f:
                json.dump(config, f, indent=2)
            
            # Create README
            readme_content = f"""# {model_id.replace('_', ' ').title()}

Profile: {self.profile}
Model Type: {config['model_type']}

## Configuration
- Vocabulary Size: {config['vocab_size']}
- Embedding Dimension: {config['embedding_dim']}
- Hidden Dimension: {config['hidden_dim']}
- Layers: {config['num_layers']}

## Training Instructions
To train this model, run:
```bash
python scripts/train_generator.py --model_type {config['model_type']} --profile {self.profile}
```

## Usage
```python
from src.models.generator import create_lightweight_generator
from src.inference import GeneratorInference

# Load model
model = create_lightweight_generator(vocab_size={config['vocab_size']}, profile="{self.profile}")
inference = GeneratorInference(model_path="data/models/{model_id}")

# Generate molecules
molecules = inference.generate(num_samples=10)
```
"""
            
            with open(model_dir / 'README.md', 'w') as f:
                f.write(readme_content)
            
            logger.info(f"✓ Generator model configuration created")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create generator model: {e}")
            return False
    
    def create_predictor_model(self, model_id: str, model_type: str) -> bool:
        """Create a placeholder predictor model configuration."""
        logger.info(f"Creating predictor model: {model_id}")
        
        try:
            model_dir = self.models_dir / model_id
            model_dir.mkdir(parents=True, exist_ok=True)
            
            config = {
                "model_type": model_type,
                "profile": self.profile,
                "input_features": "molecular_fingerprints" if model_type == "xgboost" else "graph",
                "properties": ["solubility", "logp", "molecular_weight"]
            }
            
            # Save model config
            with open(model_dir / 'config.json', 'w') as f:
                json.dump(config, f, indent=2)
            
            # Create README
            readme_content = f"""# {model_id.replace('_', ' ').title()}

Profile: {self.profile}
Model Type: {model_type}

## Training Instructions
To train this model, run:
```bash
python scripts/train_predictor.py --model_type {model_type} --profile {self.profile}
```

## Usage
```python
from src.inference import PredictorInference

# Load predictor
predictor = PredictorInference(model_path="data/models/{model_id}")

# Predict properties
results = predictor.predict(molecules=["CCO", "CC(=O)O"])
```
"""
            
            with open(model_dir / 'README.md', 'w') as f:
                f.write(readme_content)
            
            logger.info(f"✓ Predictor model configuration created")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create predictor model: {e}")
            return False
    
    def download_sample_data(self):
        """Download or generate sample molecular data."""
        logger.info("Downloading sample data...")
        
        try:
            # Create sample SMILES dataset
            sample_molecules = self._get_sample_molecules(extended=True)
            
            # Save to files
            data_files = {
                "training_molecules.txt": sample_molecules[:80],
                "validation_molecules.txt": sample_molecules[80:90],
                "test_molecules.txt": sample_molecules[90:100]
            }
            
            for filename, molecules in data_files.items():
                file_path = self.data_dir / filename
                with open(file_path, 'w') as f:
                    for mol in molecules:
                        f.write(mol + '\n')
                logger.info(f"✓ Created {filename} with {len(molecules)} molecules")
            
            # Create dataset info file
            dataset_info = {
                "total_molecules": len(sample_molecules),
                "train_size": 80,
                "val_size": 10,
                "test_size": 10,
                "source": "Generated sample data",
                "profile": self.profile
            }
            
            with open(self.data_dir / 'dataset_info.json', 'w') as f:
                json.dump(dataset_info, f, indent=2)
            
            logger.info("✓ Sample data downloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download sample data: {e}")
            return False
    
    def _get_sample_molecules(self, extended: bool = False) -> List[str]:
        """Get sample molecules for training/testing."""
        # Drug-like molecules for training
        base_molecules = [
            "CCO",  # Ethanol
            "CC(=O)O",  # Acetic acid
            "C1=CC=CC=C1",  # Benzene
            "CC(=O)OC1=CC=CC=C1C(=O)O",  # Aspirin
            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
            "CC(C)NCC(COC1=CC=CC=C1)O",  # Propranolol
            "CC1=CC=C(C=C1)C(=O)O",  # p-Toluic acid
            "CCOC(=O)C1=CN=CC=C1",  # Ethyl nicotinate
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
            "CN(C)CCN1C2=CC=CC=C2SC3=C1C=C(C=C3)Cl",  # Chlorpromazine
        ]
        
        if extended:
            # Add more diverse molecules
            extended_molecules = [
                "COC1=CC=CC=C1",  # Anisole
                "CC(C)C1=CC=C(C=C1)C(C)C",  # p-Cymene
                "C1CCCCC1",  # Cyclohexane
                "CC1=CC=CC=C1C",  # o-Xylene
                "C1=CC=C2C(=C1)C=CC=C2",  # Naphthalene
                "CC(C)O",  # Isopropanol
                "CC(C)(C)O",  # tert-Butanol
                "CCCC(=O)O",  # Butyric acid
                "CC(C)C(=O)O",  # Isobutyric acid
                "C1=CC=C(C=C1)CO",  # Benzyl alcohol
                "CC1=CC=C(C=C1)N",  # p-Toluidine
                "CC(C)N",  # Isopropylamine
                "CCCCN",  # Butylamine
                "C1=CC=C(C=C1)O",  # Phenol
                "CC(C)C1=CC(=C(C=C1)O)C(C)C",  # Thymol
                "COC1=CC=C(C=C1)O",  # Mequinol
                "CC(=O)OCC",  # Ethyl acetate
                "CCOC(=O)C",  # Ethyl propanoate
                "CC1=CC=C(C=C1)C",  # p-Xylene
                "C1=CC(=CC=C1O)O",  # Hydroquinone
                # Additional diverse structures
                "C1=CC=C(C=C1)C(=O)C2=CC=CC=C2",  # Benzophenone
                "CC(C)C1=CC=C(C=C1)OC",  # p-Methoxycumene
                "CC1=C(C=CC=C1)N",  # o-Toluidine
                "CCCCCC",  # Hexane
                "CC(C)CC(C)C",  # 2,4-Dimethylpentane
                "C1=CC=C(C=C1)Br",  # Bromobenzene
                "CC(C)C1=CC=C(C=C1)Cl",  # p-Chlorocumene
                "C1=CC=C(C=C1)N(C)C",  # N,N-Dimethylaniline
                "CC1=CC=C(C=C1)S",  # p-Thiocresol
                "CCOC1=CC=CC=C1",  # Phenetole
                # More complex drug-like molecules
                "CC1=C(C=C(C=C1)NC(=O)C2=CC=CC=C2)C",  # Toluamide derivative
                "COC1=CC=C(C=C1)C=O",  # p-Anisaldehyde
                "CC(=O)NC1=CC=C(C=C1)O",  # Paracetamol
                "CC1=CC(=NO1)C2=CC=CC=C2",  # Isoxazole derivative
                "C1=CC=C(C=C1)CC(=O)O",  # Phenylacetic acid
                "CC(C)(C)C1=CC=C(C=C1)O",  # 4-tert-Butylphenol
                "COC1=CC=C(C=C1)CCN",  # p-Methoxyphenethylamine
                "CC1=CC=C(C=C1)C(C)(C)C",  # p-tert-Butyltoluene
                "C1=CC=C(C=C1)C(C)(C)N",  # Cumylamine
                "CC(C)OC(=O)C1=CC=CC=C1",  # Isopropyl benzoate
                # Additional diverse structures for better training
                "C1CCNCC1",  # Piperidine
                "C1COCCN1",  # Morpholine
                "CC(C)C(C)(C)C",  # 2,2,3-Trimethylbutane
                "C1=CC=C2C(=C1)C(=O)C=C(O2)C",  # Coumarin derivative
                "CC1=CC=C(C=C1)C=C",  # p-Methylstyrene
                "CCCCOC(=O)C",  # Butyl acetate
                "CC(C)C(=O)OC(C)C",  # Isopropyl isobutyrate
                "C1=CC=C(C=C1)C#N",  # Benzonitrile
                "CC1=CC=C(C=C1)C#C",  # p-Methylphenylacetylene
                "C1=CC=C(C=C1)C(=O)N",  # Benzamide
                # More diverse functional groups
                "CC(C)C1=CC=CC=C1O",  # o-Isopropylphenol
                "CCOC(C)=O",  # Ethyl acetate
                "C1=CC=C(C=C1)S(=O)(=O)O",  # Benzenesulfonic acid
                "CC1=CC=CC=C1N(C)C",  # N,N-Dimethyl-o-toluidine
                "C1=CC=C2C(=C1)NC=N2",  # Benzimidazole
                "CC(C)C1=NC=NC=C1",  # Isopropylpyrimidine
                "C1=CC=C(C=C1)OCC=C",  # Allyl phenyl ether
                "CC1=CC=C(C=C1)C(C)O",  # 1-(p-Tolyl)ethanol
                "CCOC1=CC=C(C=C1)C=O",  # p-Ethoxybenzaldehyde
                "C1=CC=C(C=C1)C(C)C",  # Cumene
                "CC(C)(C)C1=CC=CC=C1",  # tert-Butylbenzene
                "COC1=CC=CC=C1C=O",  # o-Anisaldehyde
                "CC1=CC=C(C=C1)OC(C)C",  # p-Cresyl isopropyl ether
                "C1=CC=C(C=C1)C(C)(C)Cl",  # Cumyl chloride
                "CC(C)C1=CC=C(C=C1)C(C)C",  # p-Diisopropylbenzene
                "CCOC(=O)C1=CC=CC=C1",  # Ethyl benzoate
            ]
            return base_molecules + extended_molecules
        
        return base_molecules
    
    def _build_vocab_from_molecules(self, molecules: List[str]) -> Dict[str, int]:
        """Build vocabulary from SMILES molecules using SELFIES."""
        try:
            import selfies as sf
        except ImportError:
            logger.warning("SELFIES not available, using character-level vocabulary")
            # Fallback to character-level
            vocab = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3, "[MASK]": 4}
            chars = set()
            for mol in molecules:
                chars.update(mol)
            for i, char in enumerate(sorted(chars), start=5):
                vocab[char] = i
            return vocab
        
        # Build SELFIES-based vocabulary
        vocab = {"[PAD]": 0, "[BOS]": 1, "[EOS]": 2, "[UNK]": 3, "[MASK]": 4}
        selfies_tokens = set()
        
        for smiles in molecules:
            try:
                selfies = sf.encoder(smiles)
                # Extract tokens (bracketed)
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
                
                selfies_tokens.update(tokens)
            except Exception as e:
                logger.debug(f"Failed to process {smiles}: {e}")
        
        # Add tokens to vocabulary
        for i, token in enumerate(sorted(selfies_tokens), start=5):
            vocab[token] = i
        
        return vocab
    
    def setup_profile(self, required_only: bool = False):
        """Setup all models for the selected profile."""
        logger.info("\n" + "="*60)
        logger.info(f"Setting up profile: {self.profile}")
        logger.info("="*60 + "\n")
        
        profile_info = self.get_profile_info()
        success_count = 0
        total_count = 0
        
        # Download sample data first
        logger.info("Step 1: Downloading sample data...")
        if self.download_sample_data():
            success_count += 1
        total_count += 1
        
        # Setup each model
        logger.info("\nStep 2: Setting up models...")
        for model_id, model_info in profile_info['models'].items():
            # Skip optional models if required_only is True
            if required_only and not model_info['required']:
                logger.info(f"Skipping optional model: {model_id}")
                continue
            
            total_count += 1
            logger.info(f"\nSetting up: {model_info['name']}")
            
            # Check if model already exists
            model_path = self.models_dir / model_id
            if model_path.exists() and not self.force:
                logger.info(f"Model already exists: {model_id} (use --force to overwrite)")
                success_count += 1
                continue
            
            # Create model based on type
            if model_info['type'] == 'selfies':
                success = self.create_tokenizer(model_id)
            elif model_info['type'] in ['lstm', 'transformer', 'gpt2']:
                success = self.create_generator_model(model_id)
            else:
                success = self.create_predictor_model(model_id, model_info['type'])
            
            if success:
                success_count += 1
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("SETUP COMPLETE")
        logger.info("="*60)
        logger.info(f"Profile: {self.profile}")
        logger.info(f"Success: {success_count}/{total_count}")
        logger.info(f"Models directory: {self.models_dir}")
        
        if success_count == total_count:
            logger.info("\n✓ All models successfully configured!")
            logger.info("\nNext steps:")
            logger.info("1. Train models: python scripts/train_generator.py --profile " + self.profile)
            logger.info("2. Run the app: streamlit run app/main.py")
        else:
            logger.warning(f"\n⚠ Some models failed to setup ({total_count - success_count} failures)")
        
        return success_count == total_count

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Download and setup models for Drug Discovery Assistant"
    )
    
    parser.add_argument(
        '--profile',
        choices=['light', 'medium', 'full', 'colab'],
        default='light',
        help='Memory profile (default: light for GTX 1650)'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available models for the profile'
    )
    
    parser.add_argument(
        '--required-only',
        action='store_true',
        help='Download only required models'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download even if models exist'
    )
    
    parser.add_argument(
        '--data-only',
        action='store_true',
        help='Download only sample data, skip models'
    )
    
    args = parser.parse_args()
    
    # Create downloader
    downloader = ModelDownloader(profile=args.profile, force=args.force)
    
    # List models
    if args.list:
        downloader.list_models()
        return
    
    # Download data only
    if args.data_only:
        logger.info("Downloading sample data only...")
        downloader.download_sample_data()
        return
    
    # Setup full profile
    success = downloader.setup_profile(required_only=args.required_only)
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()