from flask import Blueprint, request, jsonify
import torch
from model.loader import load_model, load_mapping
from model.utils import get_tramit_by_id
from config import DEVICE, CONTEXT_SIZE

prediction_bp = Blueprint("prediction", __name__)
model = load_model()
tramit_to_index = load_mapping()
index_to_tramit = {v: k for k, v in tramit_to_index.items()} if tramit_to_index else {}


@prediction_bp.route("/predict", methods=["POST"])
def predict_next_tramit():
    data = request.get_json(force=True)

    if "sequence" not in data or not data["sequence"]:
        return {"error": "Invalid input", "message": "Sequence required"}, 400

    try:
        numerical_sequence = [
            tramit_to_index[tramit_id] for tramit_id in data["sequence"]
        ]
    except KeyError as e:
        return {"error": "Unknown trámite ID", "message": str(e)}, 400

    # Pad or truncate the sequence
    if len(numerical_sequence) > CONTEXT_SIZE:
        numerical_sequence = numerical_sequence[-CONTEXT_SIZE:]
    else:
        numerical_sequence = [0] * (
            CONTEXT_SIZE - len(numerical_sequence)
        ) + numerical_sequence

    input_tensor = torch.tensor(numerical_sequence).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=-1).squeeze(0)

    # Mask probabilities of already seen items
    mask = torch.zeros_like(probabilities)
    for tramit_id in data["sequence"]:
        index = tramit_to_index.get(tramit_id, None)
        if index is not None:
            mask[index] = 1

    masked_probabilities = probabilities * (1 - mask)
    masked_probabilities /= masked_probabilities.sum()  # Renormalize probabilities

    predicted_index = torch.argmax(masked_probabilities).item()
    predicted_tramit_id = index_to_tramit.get(predicted_index, "Unknown")
    predicted_tramit = get_tramit_by_id(predicted_tramit_id)

    return jsonify(
        {
            "predicted_tramit": predicted_tramit,
            "probability": float(masked_probabilities[predicted_index]),
        }
    )
