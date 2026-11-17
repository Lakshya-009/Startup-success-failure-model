from fastapi import FastAPI
from pathlib import Path
import joblib

app = FastAPI()

# Get project directory
BASE_DIR = Path(__file__).resolve().parent

# Actual model location
MODEL_PATH = BASE_DIR / "startup_model" / "artifacts" / "model.joblib"

# Load trained model
model = joblib.load(MODEL_PATH)

@app.get("/")
def home():
    return {"message": "ML Model is running successfully!"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }