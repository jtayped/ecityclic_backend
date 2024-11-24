import torch
import json
from config import (
    MODEL_PATH,
    MAPPING_PATH,
    DEVICE,
    NUM_TRAMITS,
    CONTEXT_SIZE,
    EMBEDDING_DIM,
    HIDDEN_DIM,
)
from model.model import TramitPredictor


def load_mapping():
    try:
        with open(MAPPING_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading mapping: {e}")
        return None


def load_model():
    try:
        model = TramitPredictor(
            num_tramits=NUM_TRAMITS,
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=HIDDEN_DIM,
            context_size=CONTEXT_SIZE,
        )
        model.load_state_dict(
            torch.load(MODEL_PATH, weights_only=True, map_location=DEVICE)
        )
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
