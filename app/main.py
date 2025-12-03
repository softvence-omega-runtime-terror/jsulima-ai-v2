import logging
from fastapi import FastAPI

from app.core.logger import setup_logging
from app.routes import prediction

setup_logging()

logger = logging.getLogger(__name__)

logger.info("Application is starting...")

app = FastAPI(
    title="UFC Prediction API",
    description="API for predicting UFC fight winners based on historical data.",
    version="1.0.0",
    root_path="/",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Include the prediction router
app.include_router(prediction.router)

logger.info("Application has started successfully.")

@app.get("/")
async def read_root():
    return {"message": "Welcome to the UFC Prediction API!"}