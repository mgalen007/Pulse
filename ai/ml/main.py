import joblib
import pandas as pd

model = joblib.load("models/mood_predictor.pkl")

sample = pd.DataFrame({
    "sleep_hours": [6.2],
    "screen_time": [24.6], 
    "stress_level": [1.331], 
    "exercise_minutes": [25.0], 
    "diet_quality": [1]
})

if __name__ == "__main__":
    prediction = model.predict(sample)
    print(f"Predicted mood score: {prediction[0]}")
