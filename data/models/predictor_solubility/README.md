# Predictor Solubility

Profile: light
Model Type: xgboost

## Training Instructions
To train this model, run:
```bash
python scripts/train_predictor.py --model_type xgboost --profile light
```

## Usage
```python
from src.inference import PredictorInference

# Load predictor
predictor = PredictorInference(model_path="data/models/predictor_solubility")

# Predict properties
results = predictor.predict(molecules=["CCO", "CC(=O)O"])
```
