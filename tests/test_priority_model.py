from pathlib import Path

from app.ml.predict import predict_priority


def test_priority_model_artifact_exists() -> None:
    assert Path("models/priority_model.pkl").exists()


def test_predict_priority_returns_high_for_urgent_ticket() -> None:
    priority = predict_priority(
        subject="Account locked after many attempts",
        description="My business account is locked. This is blocking my team and needs immediate attention.",
    )

    assert priority == "High"
