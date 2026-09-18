from fastapi.testclient import TestClient
from main import app

client =TestClient(app)

def test_health():
	response = client.get("/health")
	assert response.status_code==200
	assert response.json()=={"status": "ok"}

def test_predict():
	payload={
	 "age": 60,
	"heart_rate": 85,
	"spo2": 95
	}
	
	api_key = os.getenv("API_KEY", "default_secret")
	headers = {"X-API-Key": "hospital_secret_123"}
	response= client.post("/predict",json=payload, headers=headers)

	assert response.status_code == 200

	response_data = response.json()
	assert "prediction" in response_data
	assert "probability" in response_data
	assert "model_version" in response_data
	assert response_data["status"] == "success" 