from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class TicketCreate(BaseModel):
    customer_id: int = Field(..., examples=[101])
    subject: str = Field(..., min_length=3, max_length=200, examples=["Unable to login"])
    description: str = Field(
        ...,
        min_length=10,
        max_length=2_000,
        examples=["I forgot my password and cannot access my account."],
    )


class TicketResponse(BaseModel):
    ticket_id: int
    customer_id: int
    subject: str
    description: str
    predicted_category: str
    predicted_priority: str
    status: str
    created_at: datetime


class TicketListResponse(BaseModel):
    total: int
    tickets: list[TicketResponse]


class AnalyticsResponse(BaseModel):
    total_tickets: int
    category_distribution: dict[str, int]
    priority_distribution: dict[str, int]
    status_distribution: dict[str, int]
