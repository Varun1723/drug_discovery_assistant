#!/usr/bin/env python3
"""
Training script for Toxicity Classifier (Binary Classification)
"""

import os
import sys
import argparse
import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_fingerprint(smiles: str):
    """Convert SMILES to Morgan Fingerprint."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
    except:
        return None
    return None

def train_toxicity_model(data_path: Path, output_dir: Path):
    logger.info(f"Loading training data from {data_path}")
    molecules = []
    with open(data_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                molecules.append(line)
    
    logger.info(f"Loaded {len(molecules)} molecules")
    
    # --- CREATE DUMMY TOXICITY DATA (0=Safe, 1=Toxic) ---
    # In a real scenario, you would load the Tox21 dataset.
    # Here we randomly assign toxicity for demonstration.
    data = {
        "smiles": molecules,
        "is_toxic": np.random.randint(0, 2, size=len(molecules)) 
    }
    df = pd.DataFrame(data)
    
    # Get fingerprints
    logger.info("Generating fingerprints...")
    df['fp'] = df['smiles'].apply(get_fingerprint)
    df = df.dropna() 
    
    # Prepare data
    X = np.array(df['fp'].tolist())
    y = df['is_toxic'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    logger.info(f"Training XGBoost Classifier on {len(X_train)} samples...")
    
    # XGBoost Classifier
    model = XGBClassifier(
        n_estimators=100, 
        learning_rate=0.1, 
        max_depth=6,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    # Test model
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    logger.info(f"Model Accuracy on test set: {acc:.2%}")
    
    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "toxicity_xgb_model.pkl"
    joblib.dump(model, save_path)
    
    logger.info(f"✓ Toxicity Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=Path, default=project_root / "data" / "raw" / "training_molecules.txt")
    parser.add_argument('--output_dir', type=Path, default=project_root / "data" / "models" / "predictor_toxicity")
    args = parser.parse_args()
    
    train_toxicity_model(args.data_file, args.output_dir)