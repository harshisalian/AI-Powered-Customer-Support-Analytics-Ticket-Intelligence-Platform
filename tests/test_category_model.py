from pathlib import Path

from app.ml.predict import predict_category


def test_category_model_artifact_exists() -> None:
    assert Path("models/category_model.pkl").exists()


def test_predict_category_returns_expected_label_for_login_ticket() -> None:
    category = predict_category(
        subject="Cannot login to dashboard",
        description="The password reset link expired and I cannot access my account.",
    )

    assert category == "Login Problems"
