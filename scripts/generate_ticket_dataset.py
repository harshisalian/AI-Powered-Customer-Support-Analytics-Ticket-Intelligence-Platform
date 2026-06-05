from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


RANDOM_SEED = 42
TOTAL_TICKETS = 1_200
OUTPUT_PATH = Path("data/raw/support_tickets.csv")

CATEGORIES = {
    "Payment Issues": {
        "subjects": [
            "Payment failed during checkout",
            "Card was charged twice",
            "Invoice amount looks incorrect",
            "Unable to complete payment",
            "Bank transaction succeeded but order failed",
        ],
        "descriptions": [
            "I tried to pay for my order, but the payment failed even though my card details are correct.",
            "My credit card was charged two times for the same subscription renewal.",
            "The invoice shows a higher amount than expected for my monthly plan.",
            "The transaction was deducted from my bank account, but the payment status still says pending.",
            "I cannot add a new payment method because the checkout page keeps rejecting my card.",
        ],
        "priority_weights": {"Low": 0.15, "Medium": 0.60, "High": 0.25},
    },
    "Login Problems": {
        "subjects": [
            "Cannot login to my account",
            "Password reset link not working",
            "Account locked after many attempts",
            "Two factor authentication problem",
            "Unable to access dashboard",
        ],
        "descriptions": [
            "I cannot login even after entering the correct email address and password.",
            "The password reset email link has expired and I cannot generate a new one.",
            "My account is locked because I entered the wrong password several times.",
            "The two factor authentication code is not arriving on my phone.",
            "I can login on mobile but the web dashboard keeps showing an authentication error.",
        ],
        "priority_weights": {"Low": 0.10, "Medium": 0.55, "High": 0.35},
    },
    "Account Management": {
        "subjects": [
            "Update account email address",
            "Change company profile details",
            "Add another team member",
            "Remove old phone number",
            "Update billing contact",
        ],
        "descriptions": [
            "I want to update the email address linked to my customer account.",
            "Please help me change the company name and profile information in my account.",
            "I need to invite a new teammate and assign account access permissions.",
            "My old phone number is still attached to the account and I want to remove it.",
            "The billing contact has changed and I need the account details updated.",
        ],
        "priority_weights": {"Low": 0.55, "Medium": 0.35, "High": 0.10},
    },
    "Refund Requests": {
        "subjects": [
            "Request refund for duplicate payment",
            "Refund needed for cancelled order",
            "Money back for failed service",
            "Cancel purchase and refund",
            "Refund status not updated",
        ],
        "descriptions": [
            "I was charged for an order that I cancelled, so I need a refund.",
            "Please refund the duplicate payment made from my card yesterday.",
            "The service did not work as promised and I would like my money back.",
            "I cancelled the subscription within the trial period but have still been charged.",
            "My refund request was approved last week, but the amount has not reached my bank account.",
        ],
        "priority_weights": {"Low": 0.10, "Medium": 0.50, "High": 0.40},
    },
    "Technical Support": {
        "subjects": [
            "Application keeps crashing",
            "Dashboard page is loading slowly",
            "Error while uploading file",
            "API integration is failing",
            "Feature not working after update",
        ],
        "descriptions": [
            "The application crashes every time I open the reports page.",
            "The dashboard takes a very long time to load and sometimes shows a server error.",
            "I am getting an error while uploading a CSV file to the platform.",
            "Our API integration started failing with timeout errors after the latest update.",
            "The export feature is not working and the downloaded file is empty.",
        ],
        "priority_weights": {"Low": 0.15, "Medium": 0.45, "High": 0.40},
    },
    "Subscription Issues": {
        "subjects": [
            "Cancel my subscription",
            "Upgrade monthly plan",
            "Subscription renewal problem",
            "Plan changed without confirmation",
            "Trial period expired early",
        ],
        "descriptions": [
            "I want to cancel my monthly subscription before the next billing cycle.",
            "Please help me upgrade from the basic plan to the premium plan.",
            "My subscription renewal failed even though my payment method is active.",
            "My plan seems to have changed without my confirmation.",
            "The free trial expired earlier than expected and I cannot access premium features.",
        ],
        "priority_weights": {"Low": 0.25, "Medium": 0.55, "High": 0.20},
    },
}

CHANNELS = ["email", "chat", "phone", "web_portal"]
STATUSES = ["open", "in_progress", "resolved"]

PRIORITY_CONTEXT = {
    "Low": [
        "This is not urgent, but I would like it updated when possible.",
        "There is no immediate impact on my work.",
        "I can continue using the service for now.",
    ],
    "Medium": [
        "This is affecting my work and I need help soon.",
        "Please resolve this as soon as possible.",
        "I have already contacted support once and need an update.",
    ],
    "High": [
        "This is blocking my team and needs immediate attention.",
        "Our business workflow is stopped because of this issue.",
        "This is urgent because customers are affected right now.",
    ],
}


def weighted_priority(priority_weights: dict[str, float]) -> str:
    priorities = list(priority_weights.keys())
    weights = list(priority_weights.values())
    return random.choices(priorities, weights=weights, k=1)[0]


def random_created_at() -> datetime:
    today = datetime.now(timezone.utc)
    days_ago = random.randint(0, 180)
    minutes_ago = random.randint(0, 1_440)
    return today - timedelta(days=days_ago, minutes=minutes_ago)


def build_ticket(ticket_id: int) -> dict[str, object]:
    category = random.choice(list(CATEGORIES.keys()))
    category_data = CATEGORIES[category]

    subject = random.choice(category_data["subjects"])
    priority = weighted_priority(category_data["priority_weights"])
    base_description = random.choice(category_data["descriptions"])
    priority_context = random.choice(PRIORITY_CONTEXT[priority])
    description = f"{base_description} {priority_context}".strip()
    created_at = random_created_at()

    return {
        "ticket_id": ticket_id,
        "customer_id": random.randint(1000, 9999),
        "subject": subject,
        "description": description,
        "category": category,
        "priority": priority,
        "channel": random.choice(CHANNELS),
        "status": random.choice(STATUSES),
        "created_at": created_at.isoformat(),
    }


def generate_dataset(total_tickets: int = TOTAL_TICKETS) -> pd.DataFrame:
    random.seed(RANDOM_SEED)
    tickets = [build_ticket(ticket_id) for ticket_id in range(1, total_tickets + 1)]
    return pd.DataFrame(tickets)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset = generate_dataset()
    dataset.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated {len(dataset)} tickets at {OUTPUT_PATH}")
    print(dataset["category"].value_counts().sort_index())
    print(dataset["priority"].value_counts().sort_index())


if __name__ == "__main__":
    main()
