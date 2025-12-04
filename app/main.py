import logging
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

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

from app.routes.v1.UFC.schedule import router as schedule_router
app.include_router(schedule_router)

from app.routes.v1.UFC.ufc_predict import router as ufc_router
app.include_router(ufc_router, prefix="/api/v1/ufc", tags=["UFC"])

