from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn

class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float
    feature5: float

scaler = joblib.load('scaler.pkl')
svc_model = joblib.load('svc_model.pkl')    

app = FastAPI()

@app.post("/predict")
def predict(data: InputData):
    features = np.array([[
        data.feature1,
        data.feature2,
        data.feature3,
        data.feature4,
        data.feature5
        ]])
    
    features_scaled = scaler.transform(features)

    prediction = svc_model.predict(features_scaled)

    return {"prediction": int(prediction[0])}

if __name__ == "__main__":
    
    uvicorn.run(app, host="127.0.0.1", port=8000)