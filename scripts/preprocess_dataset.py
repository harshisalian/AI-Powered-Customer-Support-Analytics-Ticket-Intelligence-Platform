from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from app.ml.preprocessing import preprocess_ticket


INPUT_PATH = Path("data/raw/support_tickets.csv")
OUTPUT_PATH = Path("data/processed/support_tickets_processed.csv")


REQUIRED_COLUMNS = {
    "ticket_id",
    "customer_id",
    "subject",
    "description",
    "category",
    "priority",
    "channel",
    "status",
    "created_at",
}


def load_dataset(input_path: Path = INPUT_PATH) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    dataset = pd.read_csv(input_path)
    missing_columns = REQUIRED_COLUMNS.difference(dataset.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")

    return dataset


def build_processed_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    processed_dataset = dataset.copy()
    processed_dataset["combined_text"] = (
        processed_dataset["subject"].fillna("") + " " + processed_dataset["description"].fillna("")
    ).str.strip()
    processed_dataset["clean_text"] = processed_dataset.apply(
        lambda row: preprocess_ticket(row["subject"], row["description"]),
        axis=1,
    )
    return processed_dataset


def main() -> None:
    dataset = load_dataset()
    processed_dataset = build_processed_dataset(dataset)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    processed_dataset.to_csv(OUTPUT_PATH, index=False)

    print(f"Processed {len(processed_dataset)} tickets")
    print(f"Saved processed dataset to {OUTPUT_PATH}")
    print(processed_dataset[["combined_text", "clean_text", "category", "priority"]].head(3).to_string(index=False))


if __name__ == "__main__":
    main()
