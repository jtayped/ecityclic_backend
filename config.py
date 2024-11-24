from pathlib import Path
import torch

# Model Parameters
MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "tramit_model_big.pth"
MAPPING_PATH = MODEL_DIR / "tramit_mapping.json"

NUM_TRAMITS = 502
CONTEXT_SIZE = 3
EMBEDDING_DIM = 32
HIDDEN_DIM = 64

# Device Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data Configuration
DATA_FILE_PATH = Path("data") / "tramits.csv"
