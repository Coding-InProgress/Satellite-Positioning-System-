# backend.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn

# Example: load a trained model (RandomForest, etc.)
# For demo, we simulate
class SatelliteModel:
    def predict(self, X):
        # Fake: if stability < 0.5 or signal < 0.5 → unhealthy
        return [1 if (x[0] > 0.5 and x[1] > 0.5) else 0 for x in X]

model = SatelliteModel()

app = FastAPI()

class SatData(BaseModel):
    orbit_stability: float
    signal_quality: float

@app.post("/predict")
def predict(data: SatData):
    features = np.array([[data.orbit_stability, data.signal_quality]])
    pred = model.predict(features)
    return {"health_status": int(pred[0])}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
