from __future__ import annotations

from pathlib import Path

from app.ml.model_io import load_model
from app.ml.preprocessing import preprocess_ticket


CATEGORY_MODEL_PATH = Path("models/category_model.pkl")
PRIORITY_MODEL_PATH = Path("models/priority_model.pkl")


def predict_category(subject: str, description: str, model_path: Path = CATEGORY_MODEL_PATH) -> str:
    model = load_model(str(model_path))
    clean_text = preprocess_ticket(subject, description)
    prediction = model.predict([clean_text])
    return str(prediction[0])


def predict_priority(subject: str, description: str, model_path: Path = PRIORITY_MODEL_PATH) -> str:
    model = load_model(str(model_path))
    clean_text = preprocess_ticket(subject, description)
    prediction = model.predict([clean_text])
    return str(prediction[0])
