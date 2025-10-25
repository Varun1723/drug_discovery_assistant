# Generator

Profile: light
Model Type: lstm

## Configuration
- Vocabulary Size: 1000
- Embedding Dimension: 256
- Hidden Dimension: 512
- Layers: 2

## Training Instructions
To train this model, run:
```bash
python scripts/train_generator.py --model_type lstm --profile light
```

## Usage
```python
from src.models.generator import create_lightweight_generator
from src.inference import GeneratorInference

# Load model
model = create_lightweight_generator(vocab_size=1000, profile="light")
inference = GeneratorInference(model_path="data/models/generator")

# Generate molecules
molecules = inference.generate(num_samples=10)
```
