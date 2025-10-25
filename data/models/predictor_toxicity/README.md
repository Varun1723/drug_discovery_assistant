# Predictor Toxicity

Profile: light
Model Type: classifier

## Training Instructions
To train this model, run:
```bash
python scripts/train_predictor.py --model_type classifier --profile light
```

## Usage
```python
from src.inference import PredictorInference

# Load predictor
predictor = PredictorInference(model_path="data/models/predictor_toxicity")

# Predict properties
results = predictor.predict(molecules=["CCO", "CC(=O)O"])
```
