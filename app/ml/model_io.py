from __future__ import annotations

import joblib


def save_model(model: object, path: str) -> None:
    joblib.dump(model, path)


def load_model(path: str) -> object:
    return joblib.load(path)
