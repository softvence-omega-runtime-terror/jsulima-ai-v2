import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import PlainTextResponse

from app.core.logger import setup_logging

setup_logging()

logger = logging.getLogger(__name__)

logger.info("Application is starting...")

app = FastAPI(
    title="NBA Win Prediction API",
    description="API for predicting NBA fight winners based on historical data.",
    version="1.0.0",
    root_path="/api/v1",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.mount("/static", StaticFiles(directory="app/data/ufc"), name="static")

@app.on_event("startup")
async def startup_event():
    for route in app.routes:
        logger.info(f"Route: {route.path} {route.name}")
    logger.info("Application has started successfully.")

@app.get("/")
async def read_root():
    return {"message": "Welcome to the NBA Win Prediction API!"}

from app.routes.v1.UFC.prediction import router as prediction_router
from app.routes.v1.UFC.schedule import router as schedule_router
from app.routes.v1.UFC.ufc_predict import router as ufc_router
from app.routes.Basketball.basketball_schedule import router as basketball_schedule_router
from app.routes.Basketball.basketball_prediction import router as basketball_prediction_router
from app.routes.v1.UFC.head_to_head import router as h2h_router
from app.routes.v1.UFC.basketball_schedule import router as basketball_router


app.include_router(schedule_router, prefix="/api/v1/ufc/schedule", tags=["UFC"])
app.include_router(ufc_router, prefix="/api/v1/ufc/stats", tags=["UFC"])
app.include_router(prediction_router, prefix="/api/v1/ufc/predict", tags=["UFC"])
app.include_router(basketball_schedule_router, prefix="/api/v1/basketball/schedule", tags=["Basketball"])
app.include_router(basketball_prediction_router, prefix="/api/v1/basketball", tags=["Basketball"])
app.include_router(h2h_router, prefix="/api/v1/ufc/head_to_head", tags=["UFC"])
app.include_router(basketball_router, prefix="/api/v1/basketball/schedule", tags=["Basketball"])
