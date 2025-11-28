import logging
from fastapi import FastAPI

from app.core.logger import setup_logging
setup_logging()

logger = logging.getLogger(__name__)

logger.info("Application is starting...")

app = FastAPI(
    title="My FastAPI Application",
    description="This is a sample FastAPI application with custom metadata.",
    version="1.0.0",
    root_path="/api/v1",
    docs_url="/docs",
    redoc_url="/redoc"
)

logger.info("Application has started successfully.")

@app.get("/")
async def read_root():
    return {"message": "Welcome to My FastAPI Application!"}

from app.routes.v1.UFC.schedule import router as schedule_router
app.include_router(schedule_router)