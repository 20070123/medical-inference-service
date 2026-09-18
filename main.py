from fastapi import FastAPI,Request
from fastapi.responses import JSONResponse
from pydantic import Field,BaseModel
import logging
import time
import os
from dotenv import load_dotenv

load_dotenv()
MODEL_VERSION= os.getenv("MODEL_VERSION", "v1")
MODEL_PATH= os.getenv("MODEL_PATH", "models/model.joblib")
API_KEY= os.getenv("API_KEY", "default_secret")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

app=FastAPI()

from fastapi import HTTPException, Security, Depends
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "服务器内部错误，请稍后重试", "status": "error"},
    )

@app.get("/health")
def health():
	return {"status": "ok"}

class PredictRequest(BaseModel):
	age:int = Field(...,ge=0,le=120)
	heart_rate:int =Field(...,ge=20,le=250)
	spo2:int =Field(...,ge=0,le=100)

import joblib
import numpy as np

model=joblib.load(MODEL_PATH)

@app.post("/predict")
def predict(req: PredictRequest, api_key: str = Depends(verify_api_key)):
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