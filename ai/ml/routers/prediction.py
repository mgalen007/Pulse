
from fastapi import APIRouter
import joblib
from pydantic import BaseModel
from typing import Literal
import pandas as pd


model = joblib.load("C:/Users/HP/OneDrive/Documents/Codes/Projects/Pulse/ai/ml/models/mood_predictor.pkl")

class MoodInput(BaseModel):
    sleep_hours: list[float]
    screen_time: list[float]
    stress_level: list[int]
    exercise_minutes: list[float]
    diet_quality: list[Literal[0, 1, 2]]


router = APIRouter()

@router.post("/mood")
def predict_mood(body: MoodInput):
    sample = pd.DataFrame(body.model_dump())
    predicted_mood = model.predict(sample)[0]
    return {
        "success": True,
        "predicted_mood": int(predicted_mood)
    }
