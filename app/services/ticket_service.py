from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

from app.ml.predict import predict_category, predict_priority
from app.schemas.ticket_schema import AnalyticsResponse, TicketCreate, TicketListResponse, TicketResponse


class TicketService:
    def __init__(self) -> None:
        self._tickets: dict[int, TicketResponse] = {}
        self._next_ticket_id = 1

    def create_ticket(self, ticket_data: TicketCreate) -> TicketResponse:
        ticket_id = self._next_ticket_id
        self._next_ticket_id += 1

        predicted_category = predict_category(ticket_data.subject, ticket_data.description)
        predicted_priority = predict_priority(ticket_data.subject, ticket_data.description)

        ticket = TicketResponse(
            ticket_id=ticket_id,
            customer_id=ticket_data.customer_id,
            subject=ticket_data.subject,
            description=ticket_data.description,
            predicted_category=predicted_category,
            predicted_priority=predicted_priority,
            status="created",
            created_at=datetime.now(timezone.utc),
        )

        self._tickets[ticket_id] = ticket
        return ticket

    def get_ticket(self, ticket_id: int) -> TicketResponse | None:
        return self._tickets.get(ticket_id)

    def list_tickets(self) -> TicketListResponse:
        tickets = list(self._tickets.values())
        return TicketListResponse(total=len(tickets), tickets=tickets)

    def get_analytics(self) -> AnalyticsResponse:
        tickets = list(self._tickets.values())

        return AnalyticsResponse(
            total_tickets=len(tickets),
            category_distribution=dict(Counter(ticket.predicted_category for ticket in tickets)),
            priority_distribution=dict(Counter(ticket.predicted_priority for ticket in tickets)),
            status_distribution=dict(Counter(ticket.status for ticket in tickets)),
        )

    def reset(self) -> None:
        self._tickets.clear()
        self._next_ticket_id = 1


ticket_service = TicketService()
