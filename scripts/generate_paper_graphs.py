#!/usr/bin/env python3
"""
Generates publication-ready plots for the revised manuscript.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from sklearn.manifold import TSNE

project_root = Path(__file__).parent.parent
OUTPUT_DIR = project_root / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def get_data(filepath, label):
    with open(filepath, 'r') as f:
        smiles_list = [line.strip() for line in f if line.strip()]
    
    data = []
    fps = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol:
            data.append({
                'Source': label,
                'MW': Descriptors.MolWt(mol),
                'LogP': Descriptors.MolLogP(mol)
            })
            fps.append(np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)))
    return pd.DataFrame(data), np.array(fps)

print("Loading molecules and calculating properties...")
df_train, fp_train = get_data(project_root / "data" / "raw" / "training_molecules.txt", "Training Data (Delaney)")
df_gen, fp_gen = get_data(project_root / "data" / "raw" / "newly_generated_molecules.txt", "Generated (Length-Aware)")

df_all = pd.concat([df_train, df_gen], ignore_index=True)

# 1. Plot Property Distributions
print("Generating Property Distribution Plot...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
sns.kdeplot(data=df_all, x='MW', hue='Source', fill=True, ax=ax1, common_norm=False)
ax1.set_title('Molecular Weight Distribution (Bias Fixed)')
ax1.set_xlabel('Molecular Weight (Da)')

sns.kdeplot(data=df_all, x='LogP', hue='Source', fill=True, ax=ax2, common_norm=False)
ax2.set_title('LogP (Lipophilicity) Distribution')
ax2.set_xlabel('LogP')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "new_property_distribution.png", dpi=300)
plt.close()

# 2. Plot t-SNE
print("Calculating t-SNE (This takes a minute)...")
fp_all = np.concatenate([fp_train, fp_gen])
tsne = TSNE(n_components=2, random_state=42).fit_transform(fp_all)

df_tsne = pd.DataFrame({'t-SNE 1': tsne[:, 0], 't-SNE 2': tsne[:, 1], 'Source': df_all['Source']})

print("Generating Chemical Space Plot...")
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_tsne, x='t-SNE 1', y='t-SNE 2', hue='Source', alpha=0.6, s=30)
plt.title('t-SNE Projection of Chemical Space')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "new_tsne_chemical_space.png", dpi=300)
plt.close()

print(f"✅ Success! Graphs saved to {OUTPUT_DIR}")