from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from app.ml.model_io import save_model


DATASET_PATH = Path("data/processed/support_tickets_processed.csv")
MODEL_OUTPUT_PATH = Path("models/priority_model.pkl")
REPORT_OUTPUT_PATH = Path("reports/priority_model_report.txt")
RANDOM_STATE = 42


def load_training_data(dataset_path: Path = DATASET_PATH) -> tuple[pd.Series, pd.Series]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {dataset_path}")

    dataset = pd.read_csv(dataset_path)
    required_columns = {"clean_text", "priority"}
    missing_columns = required_columns.difference(dataset.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    return dataset["clean_text"], dataset["priority"]


def build_models() -> dict[str, Pipeline]:
    return {
        "naive_bayes": Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.95,
                    ),
                ),
                ("classifier", MultinomialNB()),
            ]
        ),
        "logistic_regression": Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.95,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1_000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def evaluate_model(model: Pipeline, x_test: pd.Series, y_test: pd.Series) -> dict[str, float | str]:
    predictions = model.predict(x_test)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "classification_report": classification_report(y_test, predictions, zero_division=0),
    }


def train_and_compare_models() -> tuple[str, Pipeline, dict[str, dict[str, float | str]]]:
    features, labels = load_training_data()
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    models = build_models()
    metrics: dict[str, dict[str, float | str]] = {}

    for model_name, model in models.items():
        model.fit(x_train, y_train)
        metrics[model_name] = evaluate_model(model, x_test, y_test)

    best_model_name = max(metrics, key=lambda name: float(metrics[name]["f1_score"]))
    return best_model_name, models[best_model_name], metrics


def write_report(best_model_name: str, metrics: dict[str, dict[str, float | str]]) -> None:
    REPORT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "Priority Classification Model Report",
        "====================================",
        "",
        f"Best model: {best_model_name}",
        "",
    ]

    for model_name, model_metrics in metrics.items():
        lines.extend(
            [
                f"Model: {model_name}",
                f"Accuracy: {float(model_metrics['accuracy']):.4f}",
                f"Precision: {float(model_metrics['precision']):.4f}",
                f"Recall: {float(model_metrics['recall']):.4f}",
                f"F1 Score: {float(model_metrics['f1_score']):.4f}",
                "",
                str(model_metrics["classification_report"]),
                "",
            ]
        )

    REPORT_OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    best_model_name, best_model, metrics = train_and_compare_models()

    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_model(best_model, str(MODEL_OUTPUT_PATH))
    write_report(best_model_name, metrics)

    print(f"Best priority model: {best_model_name}")
    print(f"Saved model to {MODEL_OUTPUT_PATH}")
    print(f"Saved report to {REPORT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
