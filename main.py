# ==========================================
# 阶段一：导入“工具箱”
# ==========================================
import os
import time
import logging
import joblib 
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI,Request,HTTPException,Security,Depends
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel,Field

# ==========================================
# 阶段二：读取配置（环境变量）
# ==========================================
load_dotenv()
MODEL_VERSION=os.getenv("MODEL_VERSION","v1")
MODEL_PATH=os.getenv("MODEL_PATH","models/model.joblib")
API_KEY=os.getenv("API_KEY","default_secret")

# ==========================================
# 阶段三：日志配置
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger=logging.getLogger(__name__)

# ==========================================
# 阶段四：创建 FastAPI 应用 + 全局异常处理
# ==========================================
app=FastAPI()

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(F"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "服务器内部错误，请稍后重试", "status": "error"}
    )

# ==========================================
# 阶段五：API 鉴权“门卫”逻辑
# ==========================================
api_key_header=APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key

# ==========================================
# 阶段六：定义请求数据结构（Pydantic 校验）
# ==========================================
class PredictRequest(BaseModel):
    age: int = Field(ge=0, le=120),
    heart_rate: int = Field(ge=20, le=250),
    spo2: int = Field(ge=0, le=100)

# ==========================================
# 阶段七：加载模型（启动时只加载一次）
# ==========================================
model = joblib.load(MODEL_PATH)

# ==========================================
# 阶段八：预测接口（核心业务逻辑）
# ==========================================
@app.post("/predict")
def predict(req: PredictRequest, api_key: str = Depends(verify_api_key)):
    # 1. 记录开始时间
    start_time = time.time()

    # 2. 拼装 numpy 二维数组 (1, 3)
    new_X = np.array([[req.age, req.heart_rate, req.spo2]])

    # 3. 调用模型预测类别和概率
    prediction = model.predict(new_X)
    probability = model.predict_proba(new_X)[0][1]

    # 4. 计算耗时并记录日志
    latency = time.time() - start_time
    logger.info(f"/predict | latency={latency:.4f}s | model_version={MODEL_VERSION} | prediction={int(prediction[0])}")

    # 5. 组装返回结果
    return{
        "prediction": int(prediction[0]),
        "probability": float(probability),
        "model_version": MODEL_VERSION,
        "status": "success"
    }
# ==========================================
# 阶段九：健康检查接口
# ==========================================
@app.get("/health")
def health():
    return {"status": "ok"}