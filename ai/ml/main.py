from fastapi import FastAPI
from routers.prediction import router as prediction_router

app = FastAPI()

@app.get("/api/v1/health")
def health_check():
    return {
        "status": "OK",
        "name": "Mood and energy prediction service"
    }

app.include_router(
    prediction_router,
    prefix="/api/v1/predict"
)
