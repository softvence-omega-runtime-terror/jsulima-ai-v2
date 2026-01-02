from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from app.routes import lineup
from app.models.model_manager import ModelManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model manager
model_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage model loading on startup and cleanup on shutdown
    """
    global model_manager
    logger.info("Starting up - Loading model...")
    model_manager = ModelManager()
    model_manager.load_model()
    yield
    logger.info("Shutting down...")

# Initialize FastAPI app
app = FastAPI(
    title="Basketball Lineup Predictor",
    description="API to predict basketball player lineups for upcoming matches",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(lineup.router)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Basketball Lineup Predictor API",
        "status": "running",
        "model_loaded": model_manager is not None and model_manager.model is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
