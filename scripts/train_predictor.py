#!/usr/bin/env python3
"""
Training script for property predictor models
"""

import os
import sys
import argparse
import logging
import joblib # For saving the model
import numpy as np
import pandas as pd
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_fingerprint(smiles: str):
    """Convert SMILES to Morgan Fingerprint."""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    return None

def train_predictor(data_path: Path, output_dir: Path):
    logger.info(f"Loading training data from {data_path}")
    molecules = []
    with open(data_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                molecules.append(line)

    logger.info(f"Loaded {len(molecules)} molecules")

    # Create a dummy dataset (SMILES, fake Solubility)
    # In a real project, you'd load a real dataset
    data = {
        "smiles": molecules,
        "solubility": [np.random.uniform(-5, 0) for _ in range(len(molecules))]
    }
    df = pd.DataFrame(data)

    # Get fingerprints
    df['fp'] = df['smiles'].apply(get_fingerprint)
    df = df.dropna() # Remove any invalid SMILES

    # Prepare data for XGBoost
    X = np.array(df['fp'].tolist())
    y = df['solubility'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logger.info(f"Training XGBoost model on {len(X_train)} samples...")

    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    # Test model
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    logger.info(f"Model RMSE on test set: {rmse:.4f}")

    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "solubility_xgb_model.pkl"
    joblib.dump(model, save_path)

    logger.info(f"✓ Model trained and saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train property predictor models")
    parser.add_argument('--data_file', type=Path, default=project_root / "data" / "raw" / "training_molecules.txt")
    parser.add_argument('--output_dir', type=Path, default=project_root / "data" / "models" / "predictor_solubility")
    args = parser.parse_args()

    train_predictor(args.data_file, args.output_dir)