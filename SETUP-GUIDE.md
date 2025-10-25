# Drug Discovery Assistant - Complete Setup Guide

## 🚀 Quick Start (GTX 1650 Optimized)

### Prerequisites
- Python 3.8 or higher
- NVIDIA GTX 1650 (4GB VRAM) or better (CPU-only mode also supported)
- 8GB+ system RAM
- 5GB+ free disk space

### Installation Steps

#### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd drug-discovery-assistant
```

#### 2. Create Conda Environment
```bash
conda env create -f environment.yml
conda activate drug-discovery
```

Or using pip only:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### 3. Download Pre-trained Models (Light Profile)
```bash
python scripts/download_models.py --profile light
```

**What this does:**
- Creates sample training data (100 drug-like molecules)
- Sets up SELFIES tokenizer configuration
- Prepares lightweight LSTM generator scaffolding
- Creates predictor model configurations
- Downloads to `data/models/` directory

**Available profiles:**
- `light` - GTX 1650 (4GB VRAM) - **Recommended for your setup**
- `medium` - 6-8GB VRAM
- `full` - 12GB+ VRAM
- `colab` - Google Colab GPU runtime

**Additional options:**
```bash
# List available models
python scripts/download_models.py --profile light --list

# Download only required models
python scripts/download_models.py --profile light --required-only

# Force re-download
python scripts/download_models.py --profile light --force

# Download only sample data
python scripts/download_models.py --profile light --data-only
```

#### 4. Run the Streamlit App
```bash
streamlit run app/main.py
```

The app will be available at `http://localhost:8501`

---

## 📋 What Gets Downloaded/Created

When you run `download_models.py --profile light`, the following structure is created:

```
data/
├── models/
│   ├── tokenizer/
│   │   ├── config.json          # Tokenizer configuration
│   │   ├── vocab.json           # SELFIES vocabulary
│   │   └── training_info.json   # Training metadata
│   ├── generator/
│   │   ├── config.json          # LSTM model configuration
│   │   └── README.md            # Usage instructions
│   ├── predictor_solubility/
│   │   ├── config.json
│   │   └── README.md
│   └── predictor_toxicity/
│       ├── config.json
│       └── README.md
└── raw/
    ├── training_molecules.txt   # 80 molecules for training
    ├── validation_molecules.txt # 10 molecules for validation
    ├── test_molecules.txt       # 10 molecules for testing
    └── dataset_info.json        # Dataset metadata
```

**Important Notes:**
- **No actual trained model weights** are downloaded initially - only configurations
- This keeps the initial download small (~10MB total)
- You can either:
  1. **Use the demo mode** (uses random generation) - app works immediately
  2. **Train models yourself** using `train_generator.py` (optional)

---

## 🎯 Running Without Trained Models

The app is designed to work **immediately after setup**, even without fully trained models:

### Demo Mode Features:
- ✅ **Tokenizer**: Fully functional SELFIES tokenization
- ✅ **Generator**: Generates random valid molecules (demo mode)
- ✅ **Predictor**: Uses RDKit descriptors (no ML model needed)
- ✅ **Protein Structure**: Uses ESMFold API (no local model needed)
- ✅ **Filters**: Lipinski, PAINS, SA-Score fully functional

### Launch the App:
```bash
streamlit run app/main.py
```

**The app will automatically:**
- Detect available models
- Fall back to demo mode for missing components
- Still provide full functionality for exploration

---

## 🔧 Optional: Train Your Own Models

If you want to train actual models (optional):

### Train Generator (LSTM)
```bash
# Train for 50 epochs (takes ~30 min on GTX 1650)
python scripts/train_generator.py --model_type lstm --profile light --epochs 50

# Quick training (10 epochs, ~5 min)
python scripts/train_generator.py --model_type lstm --profile light --epochs 10
```

### Train Predictor
```bash
python scripts/train_predictor.py --model_type fingerprint_xgb --profile light
```

**Training outputs:**
- `data/models/generator/generator_lstm_best.pt` - Best model checkpoint
- `data/models/generator/generator_lstm_final.pt` - Final model
- Training logs in console

---

## 🐳 Docker Installation (Alternative)

### GPU Version (GTX 1650)
```bash
# Build
docker build --build-arg BUILD_TYPE=gpu -t drug-discovery:gpu .

# Run
docker run --gpus all -p 8501:8501 drug-discovery:gpu
```

### CPU Version
```bash
# Build
docker build --build-arg BUILD_TYPE=cpu -t drug-discovery:cpu .

# Run
docker run -p 8501:8501 drug-discovery:cpu
```

---

## 🧪 Run Complete Demo

Test all components with one command:

```bash
python scripts/demo.py --profile light
```

This will:
1. ✅ Check dependencies
2. ✅ Setup directories
3. ✅ Download sample data
4. ✅ Test tokenizer
5. ✅ Test generator
6. ✅ Test predictor
7. ✅ Test protein structure
8. ✅ Generate demo report

**Output:** `outputs/demo_report.json`

---

## ⚙️ Configuration

### GPU Memory Optimization (GTX 1650)

Set environment variables before running:

```bash
# Linux/Mac
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export CUDA_LAUNCH_BLOCKING="1"

# Windows
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
set CUDA_LAUNCH_BLOCKING=1
```

### Force CPU Mode
```bash
export FORCE_CPU="1"
streamlit run app/main.py
```

---

## 📊 System Requirements by Profile

### Light Profile (GTX 1650 - 4GB)
- **Models**: Lightweight LSTM (~10M params)
- **Batch Size**: 1-4
- **Memory Usage**: <3.8GB VRAM
- **Generation Speed**: ~0.5s per molecule
- **Status**: ✅ **Recommended for your setup**

### Medium Profile (6-8GB)
- **Models**: Small Transformer (~50M params)
- **Batch Size**: 8-16
- **Memory Usage**: 4-7GB VRAM
- **Generation Speed**: ~1.2s per molecule

### Full Profile (12GB+)
- **Models**: GPT-2 fine-tuned (~124M params)
- **Batch Size**: 16-64
- **Memory Usage**: 8-12GB VRAM
- **Generation Speed**: ~3.5s per molecule

---

## 🔍 Troubleshooting

### Issue: "Tokenizer not found"
**Solution:**
```bash
python scripts/download_models.py --profile light --force
```

### Issue: "CUDA out of memory"
**Solutions:**
1. Reduce batch size: `--batch_size 1`
2. Use gradient accumulation (automatic in light profile)
3. Clear GPU cache: `torch.cuda.empty_cache()`
4. Switch to CPU mode: `export FORCE_CPU="1"`

### Issue: "RDKit import error"
**Solution:**
```bash
# Using conda (recommended)
conda install -c conda-forge rdkit

# Using pip
pip install rdkit
```

### Issue: "Streamlit not found"
**Solution:**
```bash
pip install streamlit==1.28.1
```

### Issue: App won't start
**Check:**
1. Python version: `python --version` (must be 3.8+)
2. Dependencies: `pip list | grep streamlit`
3. Port availability: Try `--server.port 8502`

---

## 📁 Project Structure

```
drug-discovery-assistant/
├── app/                      # Streamlit application
│   └── main.py              # Main app (run this!)
├── src/                      # Core functionality
│   ├── tokenizers/          # SELFIES tokenizer
│   ├── models/              # Generator & predictor models
│   ├── inference/           # Inference engines
│   └── utils/               # Utilities
├── scripts/                  # Setup & training scripts
│   ├── download_models.py   # ⭐ Model setup (run first)
│   ├── train_generator.py   # Optional training
│   ├── train_predictor.py   # Optional training
│   └── demo.py              # Complete demo
├── tests/                    # Unit tests
├── data/                     # Data & models (created by setup)
│   ├── models/              # Model files
│   ├── raw/                 # Training data
│   └── processed/           # Processed data
├── environment.yml           # Conda environment
├── requirements.txt          # Pip requirements
├── Dockerfile               # Docker build
└── README.md                # This file
```

---

## 🎓 Usage Examples

### Generate Molecules
```bash
streamlit run app/main.py
# Navigate to "Molecule Generator" page
# Set parameters: temperature=1.0, num_molecules=10
# Click "Generate Molecules"
```

### Predict Properties
```bash
# Navigate to "Property Prediction" page
# Input: CCO (ethanol)
# Select properties: Solubility, LogP, MW
# Click "Predict Properties"
```

### Protein Structure
```bash
# Navigate to "Protein Structure" page
# Mode: API (recommended for GTX 1650)
# Input protein sequence
# Click "Predict Structure"
```

---

## 🧰 Development Setup

### Install Dev Dependencies
```bash
pip install -r requirements-dev.txt
```

### Run Tests
```bash
# All tests
pytest tests/ -v

# Specific tests
pytest tests/test_validation.py -v
pytest tests/test_memory.py --profile light
```

### Code Formatting
```bash
black src/ app/ tests/
ruff check src/ app/ tests/
```

### Pre-commit Hooks
```bash
pre-commit install
pre-commit run --all-files
```

---

## 📚 Additional Resources

### Documentation
- [Installation Guide](docs/installation.md)
- [Usage Guide](docs/usage.md)
- [API Reference](docs/api_reference.md)
- [Troubleshooting](docs/troubleshooting.md)

### Training Custom Models
- [Generator Training Guide](docs/training_generator.md)
- [Predictor Training Guide](docs/training_predictor.md)
- [Dataset Preparation](docs/dataset_preparation.md)

### Deployment
- [Docker Deployment](docs/docker_deployment.md)
- [Cloud Deployment](docs/cloud_deployment.md)
- [Production Checklist](docs/production_checklist.md)

---

## ⚠️ Important Notes

1. **Pre-trained Models**: The initial setup creates **model configurations only**, not trained weights. This is intentional to keep downloads small.

2. **Demo Mode**: The app works immediately in demo mode, using:
   - Random generation for molecules
   - RDKit descriptors for properties
   - ESMFold API for protein structures

3. **Training Optional**: You can train your own models using the provided scripts, but it's not required to use the app.

4. **GTX 1650 Optimization**: All defaults are set for 4GB VRAM. If you have more memory, consider using `medium` or `full` profile.

5. **API Dependencies**: Protein structure prediction uses ESMFold API by default, requiring internet connection.

---

## 🆘 Getting Help

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Email**: support@drug-discovery-assistant.com

---

## 📜 License

MIT License - see [LICENSE](LICENSE) file for details

---

## 🙏 Acknowledgments

- **RDKit** - Chemical informatics toolkit
- **SELFIES** - Robust molecular representation
- **DeepChem** - Deep learning for chemistry
- **ESMFold** - Protein structure prediction
- **Streamlit** - Web application framework

---

**Ready to start? Run:**
```bash
python scripts/download_models.py --profile light
streamlit run app/main.py
```

🧬 **Happy drug discovery!** 🧬