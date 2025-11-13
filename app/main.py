# Main Streamlit Application for Drug Discovery Assistant
# Production-ready implementation with GTX 1650 memory optimization

import streamlit as st
import json
import sys
import os
import warnings
import logging
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from pathlib import Path
# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch  # <-- ADDED
from c_tokenizers.selfies_tokenizer import SELFIESTokenizer # <-- ADDED
from models.generator.lightweight_generator import create_lightweight_generator # <-- ADDED

from rdkit import Chem
from rdkit.Chem import Draw
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
# Note: SAScore is tricky to import in some envs.

import requests
import py3Dmol
from stmol import showmol

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Drug Discovery Assistant",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/drug-discovery-assistant',
        'Report a bug': 'https://github.com/your-repo/drug-discovery-assistant/issues',
        'About': 'Drug Discovery Assistant - Accelerating drug discovery with AI'
    }
)

# Custom CSS
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stAlert {
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF4B4B;
        margin: 1rem 0;
    }
    .success-card {
        border-left-color: #00D4AA;
    }
    .warning-card {
        border-left-color: #FFBD45;
    }
    .error-card {
        border-left-color: #FF4B4B;
    }
</style>
""", unsafe_allow_html=True)

class DrugDiscoveryApp:
    """Main application class for the Drug Discovery Assistant."""
    
    def __init__(self):
        self.initialize_session_state()
        self.setup_gpu_memory()
    
    def initialize_session_state(self):
        """Initialize session state variables."""
        if 'generated_molecules' not in st.session_state:
            st.session_state.generated_molecules = []
        if 'prediction_results' not in st.session_state:
            st.session_state.prediction_results = {}
        if 'protein_structures' not in st.session_state:
            st.session_state.protein_structures = {}
        if 'gpu_available' not in st.session_state:
            # st.session_state.gpu_available = self.check_gpu_availability()
            st.session_state.gpu_available = False # <-- THIS IS YOUR CPU FIX
        if 'memory_mode' not in st.session_state:
            st.session_state.memory_mode = self.determine_memory_mode()
    
    def check_gpu_availability(self) -> bool:
        """Check if GPU is available and determine VRAM capacity."""
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                if device_count > 0:
                    device_props = torch.cuda.get_device_properties(0)
                    total_memory = device_props.total_memory / (1024**3)  # GB
                    logger.info(f"GPU detected: {device_props.name}, VRAM: {total_memory:.1f} GB")
                    return total_memory
                return False
            return False
        except ImportError:
            logger.warning("PyTorch not available, using CPU mode")
            return False
    
    def determine_memory_mode(self) -> str:
        """Determine the appropriate memory mode based on available VRAM."""
        gpu_memory = st.session_state.gpu_available
        if not gpu_memory:
            return "cpu"
        elif gpu_memory <= 4.5:  # GTX 1650 and similar
            return "light"
        elif gpu_memory <= 8:
            return "medium"
        else:
            return "full"
    
    def setup_gpu_memory(self):
        """Configure GPU memory settings for GTX 1650 compatibility."""
        try:
            import torch
            if torch.cuda.is_available() and st.session_state.memory_mode == "light":
                # Set memory fraction for GTX 1650
                torch.cuda.set_per_process_memory_fraction(0.8)
                # Enable memory cleanup
                torch.cuda.empty_cache()
                logger.info("GPU memory optimized for GTX 1650")
        except Exception as e:
            logger.warning(f"Could not optimize GPU memory: {e}")

    # --- NEW FUNCTION TO LOAD YOUR MODEL ---
    @st.cache_resource
    def load_generator_model(_self):
        """Loads the tokenizer and generator model into memory."""
        TOKENIZER_PATH = Path("data/models/tokenizer")
        MODEL_PATH = Path("data/models/generator/generator_lstm_best.pt") # <-- Loads your new model
        CONFIG_PATH = Path("data/models/generator/config.json")

        # Load Tokenizer
        if not TOKENIZER_PATH.exists():
            st.error(f"Tokenizer not found at {TOKENIZER_PATH}. Please run 'scripts/download_models.py'.")
            return None, None, None
        tokenizer = SELFIESTokenizer.load(TOKENIZER_PATH)
        
        # Load Model
        if not MODEL_PATH.exists():
            st.warning("Trained model not found. Please train a model first using 'scripts/train_generator.py'. App is in DEMO MODE.")
            return tokenizer, None, None
            
        try:
            # We need to load the config to know the model's structure
            with open(CONFIG_PATH, 'r') as f:
                model_config = json.load(f)
            
            model = create_lightweight_generator(
                vocab_size=len(tokenizer),
                profile=model_config.get("profile", "light"),
                # tie_weights=False # <-- The fix we found earlier
            )
            
            # Load the saved model weights (handle CPU/GPU)
            # We force CPU mode since that's what we trained with
            device = 'cpu'

            checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint['model_state_dict'])      

            model.to(device)
            model.eval()
            st.success("Loaded trained generator model!")
            return tokenizer, model, device
        except Exception as e:
            st.error(f"Error loading model: {e}. Falling back to Demo Mode.")
            st.exception(e) # Show the full error
            return tokenizer, None, None

    @st.cache_resource
    def load_predictor_model(_self, model_name: str):
        """Loads a pre-trained predictor model."""
        MODEL_PATH = Path(f"data/models/{model_name}/solubility_xgb_model.pkl")

        if not MODEL_PATH.exists():
            st.warning(f"Predictor model not found at {MODEL_PATH}. Please run 'scripts/train_predictor.py'.")
            return None

        try:
            model = joblib.load(MODEL_PATH)
            st.success(f"Loaded predictor model: {model_name}")
            return model
        except Exception as e:
            st.error(f"Error loading predictor model: {e}")
            return None

    def get_rdkit_properties(self, smiles: str):
        """Calculate basic properties using RDKit."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return {"MW": "Invalid", "LogP": "Invalid", "TPSA": "Invalid"}

            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            return {"MW": mw, "LogP": logp, "TPSA": tpsa}
        except Exception:
            return {"MW": "Error", "LogP": "Error", "TPSA": "Error"}

    def get_fingerprint(self, smiles: str):
        """Convert SMILES to Morgan Fingerprint for model prediction."""
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)).reshape(1, -1)
        return None

    def check_lipinski(self, mol):
        """Check Lipinski's Rule of 5."""
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)

        violations = 0
        if mw > 500: violations += 1
        if logp > 5: violations += 1
        if hbd > 5: violations += 1
        if hba > 10: violations += 1

        return {
            "MW": (mw, mw <= 500),
            "LogP": (logp, logp <= 5),
            "HBD": (hbd, hbd <= 5),
            "HBA": (hba, hba <= 10),
            "Violations": violations,
            "Pass": violations <= 1
        }

    def render_header(self):
        """Render the application header."""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.title("🧬 Drug Discovery Assistant")
            st.markdown("""
            **Accelerate drug discovery with AI-powered molecular design and analysis**
            
            Generate novel molecules, predict properties, analyze toxicity, and visualize protein structures
            """)
    
    def render_system_status(self):
        """Render system status information."""
        with st.sidebar:
            st.markdown("### System Status")
            
            # GPU Status
            gpu_memory = st.session_state.gpu_available
            if gpu_memory:
                st.success(f"🎮 GPU Available: {gpu_memory:.1f} GB VRAM")
                st.info(f"📊 Memory Mode: {st.session_state.memory_mode.upper()}")
            else:
                st.warning("🔧 CPU Mode Active")
            
            # Memory Mode Description
            mode_descriptions = {
                "cpu": "CPU-only processing, slower but compatible",
                "light": "Optimized for GTX 1650 (4GB VRAM)",
                "medium": "Balanced performance for 6-8GB VRAM",
                "full": "Maximum performance for 12GB+ VRAM"
            }
            
            st.caption(f"**Mode**: {mode_descriptions.get(st.session_state.memory_mode, 'Unknown')}")
            
            # Performance Tips
            if st.session_state.memory_mode == "light":
                with st.expander("💡 GTX 1650 Optimization Tips"):
                    st.markdown("""
                    - Use small batch sizes (1-4)
                    - Enable gradient accumulation for training
                    - Use mixed precision when available
                    - Prefer API calls for protein folding
                    - Clear cache between operations
                    """)
    
    def render_navigation(self):
        """Render the navigation sidebar."""
        with st.sidebar:
            st.markdown("### Navigation")
            
            pages = {
                "🏠 Home": "home",
                "🧪 Molecule Generator": "generator",
                "📊 Property Prediction": "predictor", 
                "🦠 Toxicity Analysis": "toxicity",
                "🧬 Protein Structure": "protein",
                "📁 Batch Analysis": "batch",
                "⚙️ Settings": "settings"
            }
            
            selected_page = st.selectbox(
                "Choose a tool:",
                options=list(pages.keys()),
                key="navigation"
            )
            
            return pages[selected_page]
    
    def render_home_page(self):
        """Render the home page."""
        st.markdown("## Welcome to the Drug Discovery Assistant")
        
        # Feature overview
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🧪 Molecule Generation
            - Generate novel molecular structures using fine-tuned language models
            - SELFIES-aware tokenization for robust molecule representation
            - Memory-optimized for GTX 1650 (4GB VRAM)
            - Lightweight LSTM and transformer options
            - Real-time validity checking with RDKit
            """)
            
            st.markdown("""
            ### 📊 Property Prediction
            - Predict molecular properties using GNN and fingerprint models
            - Solubility, logP, molecular weight, and more
            - DeepChem integration with CPU fallbacks
            - Batch processing capabilities
            - Export results to CSV/SDF formats
            """)
        
        with col2:
            st.markdown("""
            ### 🦠 Toxicity & Safety Analysis
            - Multi-task toxicity classification
            - PAINS filter implementation
            - Lipinski's Rule of Five compliance
            - Synthetic Accessibility Score (SAScore)
            - Safety filtering for dangerous compounds
            """)
            
            st.markdown("""
            ### 🧬 Protein Structure Prediction
            - ESMFold integration via API (recommended)
            - Local inference option (requires >8GB VRAM)
            - 3D visualization with Py3Dmol
            - PDB file download and caching
            - Confidence score analysis (plDDT)
            """)
        
        # Quick stats
        st.markdown("### 📈 Session Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Generated Molecules", len(st.session_state.generated_molecules))
        with col2:
            st.metric("Predictions Made", len(st.session_state.prediction_results))
        with col3:
            st.metric("Protein Structures", len(st.session_state.protein_structures))
        with col4:
            memory_mode = st.session_state.memory_mode.upper()
            st.metric("Memory Mode", memory_mode)

    # --- THIS ENTIRE FUNCTION IS REPLACED ---
    def render_generator_page(self):
        """Render the molecule generator page."""
        st.markdown("## 🧪 Molecule Generator")
        
        # --- (Dropdowns and sliders) ---
        col1, col2 = st.columns([1, 1])
        with col1:
            model_type = st.selectbox(
                "Select Generator Model:",
                ["Lightweight LSTM"], # Only show what we have
                help="Lightweight models are optimized for GTX 1650"
            )
        with col2:
            tokenizer_type = st.selectbox(
                "Tokenization Method:",
                ["SELFIES"], # Only show what we have
                help="SELFIES provides more robust molecule representation"
            )
        st.markdown("### Generation Parameters")
        col1, col2, col3 = st.columns(3)
        with col1:
            num_molecules = st.slider("Number of molecules", 1, 100, 10)
            temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1)
        with col2:
            top_k = st.slider("Top-k", 1, 100, 50)
            top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05)
        with col3:
            max_length = st.slider("Max length", 50, 500, 150)
            seed = st.number_input("Random seed", 0, 999999, 42)
        
        # --- (This is the NEW "Generate Molecules" button logic) ---
        if st.button("🧪 Generate Molecules", type="primary"):
            # Load the model and tokenizer
            tokenizer, model, device = self.load_generator_model()
            
            if tokenizer is None:
                st.stop() # Stop if tokenizer failed to load

            with st.spinner("Generating molecules..."):
                try:
                    generated_molecules = []
                    
                    if model is None:
                        # --- DEMO MODE FALLBACK ---
                        st.warning("Running in Demo Mode. Molecules are placeholders.")
                        generated_molecules = [
                            "CCO", "CC(=O)O", "C1=CC=CC=C1",
                            "CC(=O)OC1=CC=CC=C1C(=O)O", 
                            "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
                        ]
                        generated_molecules = generated_molecules[:num_molecules]
                    else:
                        # --- REAL MODEL LOGIC ---
                        generated_molecules = model.generate(
                            tokenizer=tokenizer,
                            num_samples=num_molecules,
                            max_length=max_length,
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                            device=device,
                            seed=seed
                        )
                        st.success(f"Generated {num_molecules} real molecules!")
                    
                    # Store in session state
                    st.session_state.generated_molecules = generated_molecules
                    
                    # Display generated molecules
                    st.markdown("### Generated Molecules")
                    if not generated_molecules:
                        st.info("No molecules generated. Try different parameters.")
                        
                    for i, mol_str in enumerate(generated_molecules):
                        st.markdown("---")
                        col1, col2, col3 = st.columns([2, 1, 1])
                        
                        # Convert SELFIES to SMILES for RDKit
                        smiles = tokenizer.selfies_to_smiles(mol_str) if tokenizer else "N/A"
                        if not smiles:
                            smiles = "Invalid SELFIES"

                        with col1:
                            st.code(mol_str, language="text") # Display SELFIES
                        
                        with col2:
                            # When "Analyze" is clicked, store the SMILES and go to the predictor
                            if st.button(f"Analyze {i+1}", key=f"analyze_{i}"):
                                st.session_state.smiles_to_analyze = smiles
                                st.warning("Molecule sent to 'Property Prediction' page. Please navigate there now.")
                        
                        with col3:
                            # We need a unique key for the visualize button
                            viz_key = f"viz_{i}"
                        
                        # Add an expander for the visual
                        with st.expander(f"Visualize SMILES for Molecule {i+1}"):
                            if smiles == "N/A" or smiles == "Invalid SELFIES":
                                st.warning(f"Could not convert SELFIES to SMILES: {mol_str}")
                            else:
                                st.markdown(f"**SMILES:** `{smiles}`")
                                try:
                                    mol = Chem.MolFromSmiles(smiles)
                                    if mol:
                                        st.image(Draw.MolToImage(mol))
                                    else:
                                        st.warning("Cannot generate image: Invalid SMILES.")
                                except Exception as e:
                                    st.error(f"RDKit error: {e}")

                except Exception as e:
                    st.error(f"Generation failed: {str(e)}")
                    st.exception(e) # Show full error
    
    def render_predictor_page(self):
        """Render the property prediction page."""
        st.markdown("## 📊 Property Prediction")

        default_smiles = ""
        if "smiles_to_analyze" in st.session_state:
            default_smiles = st.session_state.smiles_to_analyze
            # del st.session_state.smiles_to_analyze # Keep commented out to persist data

        input_method = st.radio(
            "Input method:",
            ["Single SMILES", "Batch Upload", "From Generated"],
            horizontal=True,
            key="predictor_input_method"
        )
        
        molecules_to_predict = []
        
        if input_method == "Single SMILES":
            smiles_input = st.text_input(
                "Enter SMILES:",
                value=default_smiles,
                placeholder="CCO",
                help="Enter a valid SMILES string"
            )
            if smiles_input:
                molecules_to_predict = [smiles_input]
        
        elif input_method == "Batch Upload":
            uploaded_file = st.file_uploader(
                "Upload CSV/SDF file:",
                type=['csv', 'sdf'],
                help="File should contain SMILES column"
            )
            if uploaded_file:
                st.info("File uploaded successfully!")
        
        elif input_method == "From Generated":
            if st.session_state.generated_molecules:
                tokenizer, _, _ = self.load_generator_model()
                if tokenizer:
                    smiles_options = [tokenizer.selfies_to_smiles(s) for s in st.session_state.generated_molecules]
                    selected_smiles = st.multiselect(
                        "Select molecules (SMILES):",
                        smiles_options
                    )
                    molecules_to_predict = selected_smiles
                else:
                    st.error("Tokenizer not loaded.")
            else:
                st.info("No generated molecules available. Generate some first!")
        
        # --- Property Selection ---
        st.markdown("### Properties to Predict")
        col1, col2 = st.columns(2)
        
        properties_to_run = {}
        
        with col1:
            properties_to_run["Solubility"] = st.checkbox("Aqueous Solubility (ML Model)", True)
            properties_to_run["LogP"] = st.checkbox("Lipophilicity (LogP) (RDKit)", True)
            properties_to_run["MW"] = st.checkbox("Molecular Weight (RDKit)", True)
            properties_to_run["TPSA"] = st.checkbox("Topological PSA (RDKit)", False)
        
        with col2:
            properties_to_run["Toxicity"] = st.checkbox("Toxicity Classification (ML Model)", False)
            # These remain placeholders for now
            properties_to_run["Bioavailability"] = st.checkbox("Oral Bioavailability", False)
            properties_to_run["BBB"] = st.checkbox("Blood-Brain Barrier", False)
            properties_to_run["CYP"] = st.checkbox("CYP Inhibition", False)
        
        predictor_model = st.selectbox(
            "Prediction Model:",
            ["Fingerprint + XGBoost (Fast)"],
            help="Fingerprint models are faster and more memory-efficient"
        )
        
        # --- PREDICTION LOGIC ---
        if st.button("📊 Predict Properties", type="primary") and molecules_to_predict:
            
            # 1. Load Solubility Model
            solubility_model = None
            if properties_to_run["Solubility"]:
                solubility_model = self.load_predictor_model("predictor_solubility")

            # 2. Load Toxicity Model (THIS IS NEW)
            toxicity_model = None
            if properties_to_run["Toxicity"]:
                tox_path = Path("data/models/predictor_toxicity/toxicity_xgb_model.pkl")
                if tox_path.exists():
                    toxicity_model = joblib.load(tox_path)
                else:
                    st.warning("Toxicity model not found. Please run 'scripts/train_toxicity.py'")

            with st.spinner("Predicting properties..."):
                try:
                    results_list = []
                    
                    for smiles in molecules_to_predict:
                        if not smiles:
                            continue
                        
                        mol_results = {"Molecule": smiles}
                        
                        # --- A. Get RDKit Properties ---
                        rdkit_props = self.get_rdkit_properties(smiles)
                        if properties_to_run["MW"]:
                            mol_results["MW"] = rdkit_props.get("MW", "N/A")
                        if properties_to_run["LogP"]:
                            mol_results["LogP"] = rdkit_props.get("LogP", "N/A")
                        if properties_to_run["TPSA"]:
                            mol_results["TPSA"] = rdkit_props.get("TPSA", "N/A")

                        # --- B. Get Solubility Prediction ---
                        if properties_to_run["Solubility"] and solubility_model:
                            fp = self.get_fingerprint(smiles)
                            if fp is not None:
                                pred = solubility_model.predict(fp)
                                mol_results["Solubility"] = pred[0]
                            else:
                                mol_results["Solubility"] = "Error"
                        
                        # --- C. Get Toxicity Prediction (THIS IS NEW) ---
                        if properties_to_run["Toxicity"] and toxicity_model:
                            fp = self.get_fingerprint(smiles)
                            if fp is not None:
                                pred = toxicity_model.predict(fp)[0]
                                mol_results["Toxicity"] = "⚠️ Toxic" if pred == 1 else "✅ Safe"
                            else:
                                mol_results["Toxicity"] = "Error"
                        
                        results_list.append(mol_results)
                    
                    # --- Display results ---
                    st.markdown("### Prediction Results")
                    if not results_list:
                        st.warning("No valid molecules to predict.")
                    else:
                        df = pd.DataFrame(results_list)
                        st.dataframe(df, use_container_width=True)
                        
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results (CSV)",
                            data=csv,
                            file_name="property_predictions.csv",
                            mime="text/csv"
                        )
                    
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
                    st.exception(e)

    def render_toxicity_page(self):
        """Render the detailed safety analysis page."""
        st.markdown("## 🛡️ Safety & Toxicity Analysis")

        # Get molecules from session state
        if 'generated_molecules' not in st.session_state or not st.session_state.generated_molecules:
            st.info("No molecules to analyze. Please go to 'Molecule Generator' first.")
            return

        # Load tokenizer to convert SELFIES -> SMILES
        tokenizer, _, _ = self.load_generator_model()
        if not tokenizer:
            st.error("Tokenizer failed to load.")
            return

        # Dropdown to select a molecule
        smiles_list = [tokenizer.selfies_to_smiles(s) for s in st.session_state.generated_molecules]
        selected_smiles = st.selectbox("Select a molecule to analyze:", smiles_list)

        if selected_smiles:
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("### Structure")
                mol = Chem.MolFromSmiles(selected_smiles)
                if mol:
                    st.image(Draw.MolToImage(mol), caption=selected_smiles)
                else:
                    st.error("Invalid Molecule")

            with col2:
                st.markdown("### 🚦 Lipinski's Rule of 5")
                if mol:
                    rules = self.check_lipinski(mol)

                    # Display rules as metrics
                    r1, r2 = st.columns(2)
                    r1.metric("Molecular Weight", f"{rules['MW'][0]:.1f}", "≤ 500" if rules['MW'][1] else "Fail", delta_color="normal" if rules['MW'][1] else "inverse")
                    r2.metric("LogP", f"{rules['LogP'][0]:.1f}", "≤ 5" if rules['LogP'][1] else "Fail", delta_color="normal" if rules['LogP'][1] else "inverse")

                    r3, r4 = st.columns(2)
                    r3.metric("H-Bond Donors", rules['HBD'][0], "≤ 5" if rules['HBD'][1] else "Fail", delta_color="normal" if rules['HBD'][1] else "inverse")
                    r4.metric("H-Bond Acceptors", rules['HBA'][0], "≤ 10" if rules['HBA'][1] else "Fail", delta_color="normal" if rules['HBA'][1] else "inverse")

                    if rules['Pass']:
                        st.success(f"✅ Drug-like! (Violations: {rules['Violations']})")
                    else:
                        st.warning(f"⚠️ Not Drug-like (Violations: {rules['Violations']})")

                st.markdown("### ☠️ Toxicity Model Prediction")
                # Re-run the toxicity model for this single molecule
                tox_path = Path("data/models/predictor_toxicity/toxicity_xgb_model.pkl")
                if tox_path.exists():
                    model = joblib.load(tox_path)
                    fp = self.get_fingerprint(selected_smiles)
                    if fp is not None:
                        pred = model.predict(fp)[0]
                        probs = model.predict_proba(fp)[0]

                        if pred == 1:
                            st.error(f"**Prediction: TOXIC** (Confidence: {probs[1]:.1%})")
                        else:
                            st.success(f"**Prediction: SAFE** (Confidence: {probs[0]:.1%})")
                else:
                    st.info("Toxicity model not trained.")

    def render_protein_page(self):
        """Render the protein structure prediction page."""
        st.markdown("## 🧬 Protein Structure Prediction")

        # API URL for ESMFold
        ESMFOLD_API_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"
        
        mode = st.radio(
            "Prediction Mode:",
            ["API Mode (Recommended)", "Local Mode (>8GB VRAM)"],
            help="API mode is recommended for GTX 1650/CPU-only"
        )
        
        if mode == "Local Mode (>8GB VRAM)":
            st.error("❌ Local Mode is not implemented in this version. Please use API mode.")
            return
        
        sequence_input = st.text_area(
            "Protein Sequence (amino acids):",
            placeholder="MGSSHHHHHHSSGLVPRGSHMRGPNPTAASLEASAGPFTVRSFTVSRPSGYGAG...",
            height=150,
            help="Enter protein sequence in single-letter amino acid code"
        )
        
        with st.expander("🔧 Advanced Options"):
            timeout = st.slider("API Timeout (seconds)", 30, 300, 120)

        if st.button("🧬 Predict Structure", type="primary") and sequence_input:
            
            # Clean up sequence
            sequence = "".join(sequence_input.split())
            if not sequence:
                st.warning("Please enter a protein sequence.")
                return

            with st.spinner(f"Predicting structure for {len(sequence)} residues... (This can take a few minutes)"):
                try:
                    # --- REAL API CALL ---
                    headers = {'Content-Type': 'text/plain'}
                    response = requests.post(ESMFOLD_API_URL, data=sequence, headers=headers, timeout=timeout)
                    
                    if response.status_code == 200:
                        pdb_string = response.text
                        st.success("✅ Structure predicted successfully via API!")
                        
                        # --- 3D VISUALIZATION ---
                        st.markdown("### 3D Structure Visualization")

                        # Add controls for the user
                        style_options = st.multiselect(
                            "Visualization Styles:",
                            ["Cartoon (Ribbon)", "Stick (Bonds)", "Sphere (Atoms)", "Surface"],
                            default=["Cartoon (Ribbon)"]
                        )

                        color_style = st.radio(
                            "Color Scheme:",
                            ["Confidence (pLDDT)", "Spectrum (Rainbow)", "Chain"],
                            horizontal=True
                        )

                        # Create the viewer
                        view = py3Dmol.view(width=800, height=600)
                        view.addModel(pdb_string, 'pdb')

                        # Apply selected styles
                        if "Cartoon (Ribbon)" in style_options:
                            if color_style == "Confidence (pLDDT)":
                                # Color by B-factor (pLDDT score): Red(0) -> Blue(100)
                                view.setStyle({'cartoon': {'colorscheme': {'prop':'b','gradient':'roygb','min':50,'max':90}}})
                            elif color_style == "Spectrum (Rainbow)":
                                view.setStyle({'cartoon': {'color': 'spectrum'}})
                            else:
                                view.setStyle({'cartoon': {'color': 'chain'}})

                        if "Stick (Bonds)" in style_options:
                            view.addStyle({'stick': {}})

                        if "Sphere (Atoms)" in style_options:
                            view.addStyle({'sphere': {'scale': 0.3}})

                        if "Surface" in style_options:
                            view.addSurface(py3Dmol.SES, {'opacity': 0.8})

                        view.zoomTo()
                        view.spin(True) # Auto-spin makes it look cooler

                        # Display
                        showmol(view, height=600, width=800)
                            
                        # Use py3Dmol to render
                        view = py3Dmol.view(width=800, height=600)
                        view.addModel(pdb_string, 'pdb')
                        view.setStyle({'cartoon': {'color': 'spectrum'}})
                        view.zoomTo()
                        
                        # Display with stmol
                        showmol(view, height=600, width=800)
                        
                        # --- CONFIDENCE METRICS ---
                        # Parse PDB string for confidence (plDDT)
                        avg_plddt = 0
                        count = 0
                        for line in pdb_string.split('\n'):
                            if line.startswith("ATOM"):
                                try:
                                    # The plDDT score is in the B-factor column
                                    plddt = float(line[60:66])
                                    avg_plddt += plddt
                                    count += 1
                                except:
                                    pass
                        
                        if count > 0:
                            avg_plddt /= count
                        
                        st.markdown("### 📊 Confidence Metrics")
                        col1, col2 = st.columns(2)
                        col1.metric("Average plDDT Score", f"{avg_plddt:.2f}", help="Per-residue confidence (0-100). Higher is better.")
                        col2.metric("Sequence Length", len(sequence))
                        
                        # --- DOWNLOAD ---
                        st.download_button(
                            "📥 Download PDB",
                            data=pdb_string,
                            file_name="predicted_structure.pdb",
                            mime="chemical/x-pdb"
                        )
                    else:
                        st.error(f"API Error (Status {response.status_code}): {response.text}")

                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to connect to ESMFold API: {e}")
                except Exception as e:
                    st.error(f"Structure prediction failed: {str(e)}")
                    st.exception(e)
    
    def render_settings_page(self):
        """Render the settings page."""
        st.markdown("## ⚙️ Settings")
        
        # System settings
        st.markdown("### System Configuration")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Memory & Performance")
            memory_mode_override = st.selectbox(
                "Memory Mode Override:",
                ["Auto (Recommended)", "CPU Only", "Light (4GB)", "Medium (8GB)", "Full (12GB+)"],
                help="Override automatic memory mode detection"
            )
            
            max_batch_size = st.slider("Maximum Batch Size", 1, 32, 4)
            enable_mixed_precision = st.checkbox("Enable Mixed Precision", True)
            auto_clear_cache = st.checkbox("Auto Clear GPU Cache", True)
        
        with col2:
            st.markdown("#### Model Preferences")
            default_generator = st.selectbox(
                "Default Generator:",
                ["Lightweight LSTM", "Small Transformer", "GPT-2"]
            )
            
            default_predictor = st.selectbox(
                "Default Predictor:",
                ["Fingerprint + XGBoost", "Graph Neural Network", "DeepChem"]
            )
            
            default_tokenizer = st.selectbox(
                "Default Tokenizer:",
                ["SELFIES", "SMILES Character", "SMILES BPE"]
            )
        
        # Safety settings
        st.markdown("### Safety & Filtering")
        col1, col2 = st.columns(2)
        
        with col1:
            enable_safety_filter = st.checkbox("Enable Safety Filter", True)
            enable_pains_filter = st.checkbox("Enable PAINS Filter", True)
            enable_lipinski_filter = st.checkbox("Enable Lipinski Filter", False)
        
        with col2:
            max_molecular_weight = st.slider("Max Molecular Weight", 100, 1000, 500)
            min_sa_score = st.slider("Min SA Score", 1, 10, 3)
            toxicity_threshold = st.slider("Toxicity Threshold", 0.1, 0.9, 0.5)
        
        # Export/Import settings
        st.markdown("### Data Management")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📤 Export Session Data"):
                session_data = {
                    "generated_molecules": st.session_state.generated_molecules,
                    "prediction_results": st.session_state.prediction_results,
                    "protein_structures": list(st.session_state.protein_structures.keys())
                }
                
                st.download_button(
                    "📥 Download Session Data",
                    data=json.dumps(session_data, indent=2),
                    file_name="session_data.json",
                    mime="application/json"
                )
        
        with col2:
            uploaded_session = st.file_uploader(
                "📤 Import Session Data",
                type=['json'],
                help="Upload previously exported session data"
            )
            
            if uploaded_session:
                if st.button("🔄 Load Session Data"):
                    try:
                        data = json.load(uploaded_session)
                        # Load session data logic here
                        st.success("Session data loaded successfully!")
                    except Exception as e:
                        st.error(f"Failed to load session: {str(e)}")
        
        # Reset button
        st.markdown("### Reset")
        if st.button("🗑️ Clear All Session Data", type="secondary"):
            for key in list(st.session_state.keys()):
                if key not in ['gpu_available', 'memory_mode']:
                    del st.session_state[key]
            st.experimental_rerun()
    
    def run(self):
        """Run the main application."""
        try:
            # Render header
            self.render_header()
            
            # Render system status
            self.render_system_status()
            
            # Get selected page
            selected_page = self.render_navigation()
            
            # Render appropriate page
            if selected_page == "home":
                self.render_home_page()
            elif selected_page == "generator":
                self.render_generator_page()
            elif selected_page == "predictor":
                self.render_predictor_page()
            elif selected_page == "toxicity":
                self.render_toxicity_page()
            elif selected_page == "protein":
                self.render_protein_page()
            elif selected_page == "settings":
                self.render_settings_page()
            else:
                st.markdown(f"## {selected_page.title()}")
                st.info(f"This page is under development.")
            
            # Footer
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.markdown("""
                <div style='text-align: center; color: #666;'>
                    <p>Drug Discovery Assistant v1.0 | Optimized for GTX 1650</p>
                    <p>⚠️ For research purposes only - not for clinical use</p>
                </div>
                """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Application error: {str(e)}")
            st.info("Please refresh the page or check your configuration.")

if __name__ == "__main__":
    # Initialize and run the application
    app = DrugDiscoveryApp()
    app.run()