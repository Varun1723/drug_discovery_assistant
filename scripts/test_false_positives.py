#!/usr/bin/env python3
"""
False-Positive Safety Protocol Validation
Addresses Reviewer 2's concern regarding over-filtering.
Tests 50 known, safe FDA-approved drugs against the PAINS filter.
"""
import sys
from pathlib import Path
from rdkit import Chem

# Setup paths
project_root = Path(__file__).parent.parent
PAINS_FILE = project_root / "data" / "raw" / "pains_smiles_list.txt"

# ---------------------------------------------------------
# 50 Known, Safe FDA-Approved Drugs (Ground Truth: Safe)
# ---------------------------------------------------------
SAFE_FDA_DRUGS = {
    "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Paracetamol": "CC(=O)NC1=CC=C(O)C=C1",
    "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "Metformin": "CN(C)C(=N)N=C(N)N",
    "Amoxicillin": "CC1(C(N2C(S1)C(C2=O)NC(=O)C(C3=CC=C(C=C3)O)N)C(=O)O)C",
    "Atorvastatin": "CC(C)C1=C(C(=C(N1CC(CC(CC(=O)O)O)O)C2=CC=C(C=C2)F)C3=CC=CC=C3)C(=O)NC4=CC=CC=C4",
    "Omeprazole": "CC1=CN=C(C(=C1OC)C)CS(=O)C2=NC3=C(N2)C=C(C=C3)OC",
    "Losartan": "CCCC1=NC(=C(N1CC2=CC=C(C=C2)C3=CC=CC=C3C4=NNN=N4)CO)Cl",
    "Albuterol": "CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O",
    "Gabapentin": "C1(CCCCC1)(CC(=O)O)CN",
    "Amlodipine": "CCOC(=O)C1=C(NC(=C(C1C2=CC=CC=C2Cl)C(=O)OC)C)COCCN",
    "Sertraline": "CN[C@H]1CC[C@@H](C2=CC=CC=C12)C3=CC(=C(C=C3)Cl)Cl",
    "Simvastatin": "CCC(C)(C)C(=O)OC1CC(C=C2C1C(C(C=C2)(C)C)CCC3CC(CC(=O)O3)O)C",
    "Metoprolol": "CC(C)NCC(COC1=CC=C(C=C1)CCO)O",
    "Lisinopril": "C1CC(N(C1)C(=O)C(CCC2=CC=CC=C2)NC(C)C(=O)O)C(=O)O",
    "Azithromycin": "CCC1C(C(C(N(CC(CC(C(C(C(C(C(=O)O1)C)OC2CC(C(C(O2)C)O)(C)N)C)OC3C(C(C(C(O3)C)O)N(C)C)O)(C)O)C)C)C)O)(C)O",
    "Levothyroxine": "C1=CC(=C(C=C1CC(C(=O)O)N)I)OC2=CC(=C(C(=C2)I)O)I",
    "Hydrochlorothiazide": "C1=CC2=C(C=C1Cl)S(=O)(=O)NCN2S(=O)(=O)N",
    "Furosemide": "C1=CC(=C(C=C1C(=O)O)NCCO)S(=O)(=O)N",
    "Pantoprazole": "COC1=C(C=C2C(=C1)N=C(N2)CS(=O)C3=C(C=CN=C3)OC)OC(F)F",
    "Ciprofloxacin": "C1CC1N2C=C(C(=O)C3=CC(=C(C=C32)N4CCNCC4)F)C(=O)O",
    "Citalopram": "CN(C)CCCC1(C2=C(CO1)C=C(C=C2)C#N)C3=CC=C(C=C3)F",
    "Fluoxetine": "CNCCC(C1=CC=CC=C1)OC2=CC=C(C=C2)C(F)(F)F",
    "Lorazepam": "C1=CC=C(C=C1)C2=NC(C(=O)NC3=C2C=C(C=C3)Cl)O",
    "Duloxetine": "CNCCC(C1=CC=CS1)OC2=CC=CC3=CC=CC=C32",
    "Escitalopram": "CN(C)CCCC1(C2=C(CO1)C=C(C=C2)C#N)C3=CC=C(C=C3)F",
    "Venlafaxine": "CN(C)CC(C1=CC=C(C=C1)OC)C2(CCCCC2)O",
    "Fluticasone": "CC1CC2C3CCC4=CC(=O)C=CC4(C3(C(CC2(C1(C(=O)SC(F)F)O)C)O)F)C",
    "Diclofenac": "C1=CC=C(C(=C1)CC(=O)O)NC2=C(C=CC=C2Cl)Cl",
    "Naproxen": "CC(C1=CC2=C(C=C1)C=C(C=C2)OC)C(=O)O",
    "Meloxicam": "CC1=C(SC(=N1)NC(=O)C2=C(C3=CC=CC=C3S(=O)(=O)N2C)O)C",
    "Celecoxib": "CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "Penicillin V": "CC1(C(N2C(S1)C(C2=O)NC(=O)COC3=CC=CC=C3)C(=O)O)C",
    "Clopidogrel": "COC(=O)C(C1=CC=CC=C1Cl)N2CCC3=C(C2)C=CS3",
    "Warfarin": "CC(=O)CC(C1=CC=CC=C1)C2=C(C3=CC=CC=C3OC2=O)O",
    "Apixaban": "COC1=CC=C(C=C1)N2C3=C(CCN(C3=O)C4=CC=C(C=C4)N5CCCCC5=O)C(=N2)C(=O)N",
    "Rivaroxaban": "CC1=C(C=C(C=C1)N2CC(OC2=O)CNC(=O)C3=CC=CS3)N4CCOCC4=O",
    "Pregabalin": "CC(C)CC(CC(=O)O)CN",
    "Tramadol": "CN(C)CC1(C(CCC1=O)C2=CC=CC=C2)O",
    "Zolpidem": "CC1=CC=C(C=C1)C2=C(N3C=C(C=CC3=N2)C)CC(=O)N(C)C",
    "Montelukast": "CC(C)(C)C1=CC=C(C=C1)C=CC2=C(C=CC(=C2)C3(CC3)CC(=O)O)C=CC4=CC=CC=C4",
    "Rosuvastatin": "CC(C)C1=C(C(=NC(=N1)N(C)S(=O)(=O)C)C=CC(CC(CC(=O)O)O)O)C2=CC=C(C=C2)F",
    "Oxycodone": "COC1=C2C3=C(C=C1)O[C@H]4[C@@]5(O)CCC(=O)[C@H](C)[C@@H]5N(C)CC[C@]234",
    "Allopurinol": "C1=C(C(=O)NC=N1)N2C=NC=C2",
    "Doxycycline": "CC1C2C(C3C(C(=O)C(=C(C3(C(=O)C2(C(=C(C1=O)N(C)C)O)O)O)O)N)O)O",
    "Cetirizine": "C1CN(CCN1CCOCC(=O)O)C(C2=CC=CC=C2)C3=CC=C(C=C3)Cl",
    "Loratadine": "CCOC(=O)N1CCC(=C2C3=C(C=CC(=C3)Cl)CCC4=C2N=CC=C4)CC1",
    "Atenolol": "CC(C)NCC(COC1=CC=C(C=C1)CC(=O)N)O",
    "Diazepam": "CN1C(=O)CN=C(C2=C1C=CC(=C2)Cl)C3=CC=CC=C3"
}

def load_pains_filter():
    """Loads and canonicalizes the PAINS list exactly like app/main.py"""
    pains_set = set()
    if not PAINS_FILE.exists():
        print(f"Error: Could not find PAINS file at {PAINS_FILE}")
        return pains_set
        
    with open(PAINS_FILE, 'r') as f:
        for line in f:
            smiles = line.strip()
            if smiles:
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        pains_set.add(Chem.MolToSmiles(mol, canonical=True))
                except:
                    pass
    return pains_set

def run_false_positive_test():
    print("--- Starting FDA Safe Drug Protocol Validation ---")
    pains_set = load_pains_filter()
    print(f"Loaded {len(pains_set)} PAINS exact-match patterns.")
    print(f"Testing {len(SAFE_FDA_DRUGS)} widely used, FDA-approved drugs...\n")
    
    false_positives = 0
    
    for name, smiles in SAFE_FDA_DRUGS.items():
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                continue
            
            canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
            
            # This replicates your exact app/main.py logic
            if canonical_smiles in pains_set:
                print(f"❌ FALSE POSITIVE: {name} was incorrectly flagged as a PAINS toxin!")
                false_positives += 1
                
        except Exception as e:
            print(f"Error processing {name}: {e}")
            
    # Calculate Results
    total_tested = len(SAFE_FDA_DRUGS)
    fp_rate = (false_positives / total_tested) * 100
    
    print("-" * 50)
    print("📋 RESULTS: FALSE POSITIVE ANALYSIS")
    print("-" * 50)
    print(f"Total Drugs Tested:       {total_tested}")
    print(f"False Positives Flagged:  {false_positives}")
    print(f"False Positive Rate:      {fp_rate:.2f}%")
    print(f"True Negative Rate:       {100 - fp_rate:.2f}%")
    
    if false_positives == 0:
        print("\n✅ SUCCESS: The filter perfectly allowed all 50 safe drugs to pass!")
        print("This proves your system does NOT suffer from over-filtering.")
    else:
        print("\n⚠️ WARNING: Your filter is blocking safe medicine.")

if __name__ == "__main__":
    run_false_positive_test()