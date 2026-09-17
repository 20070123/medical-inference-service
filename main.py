from fastapi import FastAPI
from pydantic import Field,BaseModel

MODEL_VERSION= "v1"

app=FastAPI()

@app.get("/health")
def health():
	return {"status" :"ok"}

class PredictRequest(BaseModel):
	age:int = Field(...,ge=0,le=120)
	heart_rate:int =Field(...,ge=20,le=250)
	spo2:int =Field(...,ge=0,le=100)

import joblib
import numpy as np

model=joblib.load("models/model.joblib")

@app.post("/predict")
def predict (req : PredictRequest):
	new_X=np.array([[req.age , req.heart_rate, req.spo2]])

	prediction=model.predict(new_X)

	probability=model.predict_proba(new_X)[0][1]

	return{
		"prediction" : int(prediction[0]),
		"probability" : float(probability),
		"model_version" : MODEL_VERSION,
		"status" : "success"
	}