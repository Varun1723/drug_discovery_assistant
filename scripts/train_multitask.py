#!/usr/bin/env python3
"""
Training script for a Multi-Output Toxicity Classifier (Full Tox21)
Updated for Reviewers: Includes Random Forest Baseline, custom threshold (0.3) for high recall,
and extended metrics (AUPRC, F1, Precision, Recall).
"""
import json
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
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score,
    average_precision_score, f1_score, recall_score, precision_score
)
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

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
    except Exception as e:
        logger.warning(f"Failed to parse SMILES: {smiles} - {e}")
        return None
    return None

def train_multitask_model(data_path: Path, output_dir: Path):
    logger.info(f"Loading training data from {data_path}")

    try:
        df = pd.read_csv(data_path, compression='gzip')
        # Extract tasks (all columns except mol_id and smiles)
        tasks = [c for c in df.columns if c not in ['mol_id', 'smiles']]
        logger.info(f"Found {len(tasks)} tasks: {tasks}")

        # Calculate fingerprints
        logger.info("Calculating fingerprints...")
        df['fp'] = df['smiles'].apply(get_fingerprint)

        # Drop only rows with invalid SMILES. Fill missing toxicity labels with 0
        df = df.dropna(subset=['fp'])
        df[tasks] = df[tasks].fillna(0)

        X = np.stack(df['fp'].values)
        y = df[tasks].values

        logger.info(f"Dataset shape after preprocessing: X={X.shape}, y={y.shape}")

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ---------------------------------------------------------
    # 1. Train XGBoost Model (Multi-Output Ensemble)
    # ---------------------------------------------------------
    logger.info("Training Multi-Output XGBoost model...")
    base_xgb = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        n_jobs=-1,
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=15 # <--- Forces the AI to prioritize catching toxins!
    )
    xgb_model = MultiOutputClassifier(base_xgb)
    xgb_model.fit(X_train, y_train)

    # ---------------------------------------------------------
    # 2. Evaluate XGBoost with Custom Threshold
    # ---------------------------------------------------------
    logger.info("Evaluating XGBoost model...")
    xgb_preds_proba_raw = xgb_model.predict_proba(X_test)
    # Re-format proba for multi-output: list of arrays -> single array shape (n_samples, n_tasks)
    xgb_probs = np.array([p[:, 1] for p in xgb_preds_proba_raw]).T

    # Set custom threshold for HIGH RECALL (0.3 instead of 0.5)
    CUSTOM_THRESHOLD = 0.30
    xgb_preds = (xgb_probs >= CUSTOM_THRESHOLD).astype(int)

    try:
        xgb_auc = roc_auc_score(y_test, xgb_probs, average='weighted')
        xgb_auprc = average_precision_score(y_test, xgb_probs, average='weighted')
        xgb_f1 = f1_score(y_test, xgb_preds, average='weighted', zero_division=0)
        xgb_recall = recall_score(y_test, xgb_preds, average='weighted', zero_division=0)
        xgb_precision = precision_score(y_test, xgb_preds, average='weighted', zero_division=0)
    except Exception as e:
        logger.warning(f"Could not calculate some XGBoost metrics: {e}")
        xgb_auc, xgb_auprc, xgb_f1, xgb_recall, xgb_precision = 0, 0, 0, 0, 0

    logger.info(f"--- XGBoost Results (Threshold={CUSTOM_THRESHOLD}) ---")
    logger.info(f"XGB AUROC:     {xgb_auc:.4f}")
    logger.info(f"XGB AUPRC:     {xgb_auprc:.4f}")
    logger.info(f"XGB F1-Score:  {xgb_f1:.4f}")
    logger.info(f"XGB Recall:    {xgb_recall:.4f}")
    logger.info(f"XGB Precision: {xgb_precision:.4f}")

    # ---------------------------------------------------------
    # 3. Train & Evaluate Baseline Random Forest (Reviewer requested)
    # ---------------------------------------------------------
    logger.info("Training Baseline Random Forest model...")
    base_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model = MultiOutputClassifier(base_rf)
    rf_model.fit(X_train, y_train)

    rf_preds_proba_raw = rf_model.predict_proba(X_test)
    rf_probs = np.array([p[:, 1] for p in rf_preds_proba_raw]).T
    rf_preds = (rf_probs >= CUSTOM_THRESHOLD).astype(int)

    try:
        rf_auc = roc_auc_score(y_test, rf_probs, average='weighted')
        rf_auprc = average_precision_score(y_test, rf_probs, average='weighted')
        rf_f1 = f1_score(y_test, rf_preds, average='weighted', zero_division=0)
        rf_recall = recall_score(y_test, rf_preds, average='weighted', zero_division=0)
        rf_precision = precision_score(y_test, rf_preds, average='weighted', zero_division=0)
    except Exception as e:
        logger.warning(f"Could not calculate some RF metrics: {e}")
        rf_auc, rf_auprc, rf_f1, rf_recall, rf_precision = 0, 0, 0, 0, 0

    logger.info(f"--- Random Forest Results (Threshold={CUSTOM_THRESHOLD}) ---")
    logger.info(f"RF AUROC:     {rf_auc:.4f}")
    logger.info(f"RF AUPRC:     {rf_auprc:.4f}")
    logger.info(f"RF F1-Score:  {rf_f1:.4f}")
    logger.info(f"RF Recall:    {rf_recall:.4f}")
    logger.info(f"RF Precision: {rf_precision:.4f}")

    # --- Save best model (XGBoost) ---
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "toxicity_multitask_model.pkl"
    joblib.dump(xgb_model, save_path)
    logger.info(f"✓ Multi-Output XGBoost Model saved to {save_path}")

    # --- Save metrics to JSON ---
    metrics = {
        "xgboost": {
            "weighted_auc": xgb_auc,
            "weighted_auprc": xgb_auprc,
            "weighted_f1": xgb_f1,
            "weighted_recall": xgb_recall,
            "weighted_precision": xgb_precision,
            "custom_threshold": CUSTOM_THRESHOLD
        },
        "random_forest_baseline": {
            "weighted_auc": rf_auc,
            "weighted_auprc": rf_auprc,
            "weighted_f1": rf_f1,
            "weighted_recall": rf_recall,
            "weighted_precision": rf_precision,
            "custom_threshold": CUSTOM_THRESHOLD
        },
        "tasks": tasks,
        "n_samples_train": len(X_train),
        "n_samples_test": len(X_test)
    }
    metrics_path = output_dir / "toxicity_multitask_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"✓ Multi-Task metrics saved to {metrics_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multi-Output Toxicity Predictor")
    parser.add_argument('--data_file', type=Path, default=project_root / "data" / "raw" / "tox21.csv.gz")
    parser.add_argument('--output_dir', type=Path, default=project_root / "data" / "models" / "predictor_multitask_toxicity")
    args = parser.parse_args()

    train_multitask_model(args.data_file, args.output_dir)