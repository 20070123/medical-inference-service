from fastapi import FastAPI
from pydantic import Field,BaseModel
import logging
import time

MODEL_VERSION= "v1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

app=FastAPI()

@app.get("/health")
def health():
	return {"status": "ok"}

class PredictRequest(BaseModel):
	age:int = Field(...,ge=0,le=120)
	heart_rate:int =Field(...,ge=20,le=250)
	spo2:int =Field(...,ge=0,le=100)

import joblib
import numpy as np

model=joblib.load("models/model.joblib")

@app.post("/predict")
def predict (req : PredictRequest):
	start_time = time.time()
	new_X=np.array([[req.age , req.heart_rate, req.spo2]])

	prediction=model.predict(new_X)

	probability=model.predict_proba(new_X)[0][1]

	latency = time.time() - start_time
	logger.info(f"/predict | latency={latency:.4f}s | model_version={MODEL_VERSION} | prediction={int(prediction[0])}")

	return{
		"prediction" : int(prediction[0]),
		"probability" : float(probability),
		"model_version" : MODEL_VERSION,
		"status" : "success"
	}