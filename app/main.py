from fastapi import FastAPI

from app.api.routes import router
from app.core.config import settings


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    description="ML-powered customer support ticket classification system.",
)

app.include_router(router)
