#!/usr/bin/env python3
"""
Solubility Predictor Statistical Rigor Test
Addresses Reviewer 1's concerns:
1. Missing solubility results (Calculates R^2, RMSE, MAE)
2. Lack of statistical rigor (Runs 3 distinct random seeds and averages them)
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

project_root = Path(__file__).parent.parent
data_file = project_root / "data" / "raw" / "delaney-processed.csv"

# If the ESOL dataset isn't downloaded yet, download it automatically
if not data_file.exists():
    print("Downloading Delaney (ESOL) dataset...")
    import urllib.request
    url = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"
    urllib.request.urlretrieve(url, data_file)
    print("Download complete.")

def get_fingerprint(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
    except:
        return None
    return None

def run_experiment():
    print("Loading ESOL dataset for solubility prediction...")
    df = pd.read_csv(data_file)
    
    print("Calculating Morgan Fingerprints... (This will take a few seconds)")
    df['fp'] = df['smiles'].apply(get_fingerprint)
    df = df.dropna(subset=['fp'])
    
    X = np.stack(df['fp'].values)
    y = df['measured log solubility in mols per litre'].values
    
    # ---------------------------------------------------------
    # 3-RUN STATISTICAL RIGOR TEST (As requested by Reviewer)
    # ---------------------------------------------------------
    seeds = [42, 100, 2026]
    r2_scores, rmse_scores, mae_scores = [], [], []
    
    print("\nStarting 3-Fold Statistical Rigor Testing...")
    
    for idx, seed in enumerate(seeds, 1):
        # Split data differently each time
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        
        # Train model
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=seed, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Predict and evaluate
        preds = model.predict(X_test)
        
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        
        r2_scores.append(r2)
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        
        print(f"Run {idx} (Seed {seed}): R2 = {r2:.3f}, RMSE = {rmse:.3f}, MAE = {mae:.3f}")
        
    # ---------------------------------------------------------
    # CALCULATE AVERAGES AND STANDARD DEVIATIONS
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("📊 FINAL SOLUBILITY RESULTS FOR YOUR PAPER")
    print("="*50)
    print(f"R² Score: {np.mean(r2_scores):.3f} ± {np.std(r2_scores):.3f}")
    print(f"RMSE:     {np.mean(rmse_scores):.3f} ± {np.std(rmse_scores):.3f}")
    print(f"MAE:      {np.mean(mae_scores):.3f} ± {np.std(mae_scores):.3f}")
    print("="*50)

if __name__ == "__main__":
    run_experiment()