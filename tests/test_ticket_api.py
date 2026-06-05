from fastapi.testclient import TestClient

from app.main import app
from app.services.ticket_service import ticket_service


client = TestClient(app)


def setup_function() -> None:
    ticket_service.reset()


def test_create_ticket_returns_predictions() -> None:
    response = client.post(
        "/ticket",
        json={
            "customer_id": 101,
            "subject": "Payment failed for subscription",
            "description": "The transaction failed and this is blocking my team and needs immediate attention.",
        },
    )

    body = response.json()

    assert response.status_code == 201
    assert body["ticket_id"] == 1
    assert body["predicted_category"] == "Payment Issues"
    assert body["predicted_priority"] == "High"
    assert body["status"] == "created"


def test_get_ticket_returns_created_ticket() -> None:
    create_response = client.post(
        "/ticket",
        json={
            "customer_id": 202,
            "subject": "Password reset link not working",
            "description": "I cannot access my account because the login code expired.",
        },
    )
    ticket_id = create_response.json()["ticket_id"]

    response = client.get(f"/ticket/{ticket_id}")

    assert response.status_code == 200
    assert response.json()["predicted_category"] == "Login Problems"


def test_get_ticket_returns_404_for_unknown_ticket() -> None:
    response = client.get("/ticket/999")

    assert response.status_code == 404
    assert response.json()["detail"] == "Ticket 999 not found"


def test_list_tickets_returns_total_and_items() -> None:
    client.post(
        "/ticket",
        json={
            "customer_id": 303,
            "subject": "Cancel my subscription",
            "description": "I want to cancel my monthly subscription before the next billing cycle.",
        },
    )

    response = client.get("/tickets")

    assert response.status_code == 200
    assert response.json()["total"] == 1
    assert len(response.json()["tickets"]) == 1


def test_analytics_returns_ticket_distributions() -> None:
    client.post(
        "/ticket",
        json={
            "customer_id": 404,
            "subject": "Application keeps crashing",
            "description": "The application crashes every time I open the reports page. This is urgent because customers are affected right now.",
        },
    )

    response = client.get("/analytics")
    body = response.json()

    assert response.status_code == 200
    assert body["total_tickets"] == 1
    assert body["category_distribution"]["Technical Support"] == 1
    assert body["priority_distribution"]["High"] == 1
    assert body["status_distribution"]["created"] == 1


def test_create_ticket_validates_short_description() -> None:
    response = client.post(
        "/ticket",
        json={
            "customer_id": 505,
            "subject": "Hi",
            "description": "Too short",
        },
    )

    assert response.status_code == 422
