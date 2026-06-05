from fastapi import APIRouter, HTTPException, status

from app.schemas.ticket_schema import AnalyticsResponse, TicketCreate, TicketListResponse, TicketResponse
from app.services.ticket_service import ticket_service

router = APIRouter()


@router.get("/health", tags=["System"])
def health_check() -> dict[str, str]:
    return {
        "status": "healthy",
        "service": "smart-ticket-classification",
    }


@router.post(
    "/ticket",
    response_model=TicketResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Tickets"],
)
def create_ticket(ticket_data: TicketCreate) -> TicketResponse:
    return ticket_service.create_ticket(ticket_data)


@router.get("/ticket/{ticket_id}", response_model=TicketResponse, tags=["Tickets"])
def get_ticket(ticket_id: int) -> TicketResponse:
    ticket = ticket_service.get_ticket(ticket_id)

    if ticket is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ticket {ticket_id} not found",
        )

    return ticket


@router.get("/tickets", response_model=TicketListResponse, tags=["Tickets"])
def list_tickets() -> TicketListResponse:
    return ticket_service.list_tickets()


@router.get("/analytics", response_model=AnalyticsResponse, tags=["Analytics"])
def get_analytics() -> AnalyticsResponse:
    return ticket_service.get_analytics()
