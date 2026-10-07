# 阶段一：导入“工具箱”
import json
import logging
import os
import time

import joblib
import numpy as np
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Security
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import Field, create_model

# 阶段二：读取配置（环境变量）
load_dotenv()
MODEL_VERSION=os.getenv("MODEL_VERSION","v1")
MODEL_PATH=os.getenv("MODEL_PATH","models/model.joblib")
APP_ENV = os.getenv("APP_ENV", "dev")
API_KEY=os.getenv("DOCTOR_API_KEY")
if not API_KEY:
    raise RuntimeError("必须设置 DOCTOR_API_KEY 环境变量")

# 阶段三：日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger=logging.getLogger(__name__)

# 阶段四：创建 FastAPI 应用 + 全局异常处理
app=FastAPI(title="模型推理服务", 
    docs_url="/docs" if APP_ENV == "dev" else None,
    redoc_url="/redoc" if APP_ENV == "dev" else None
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "服务器内部错误，请稍后重试", "status": "error"}
    )

# 阶段五：API 鉴权“门卫”逻辑
api_key_header=APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key


# 阶段六：定义请求数据结构（Pydantic 校验）
# 从模型旁边的 schema.json 读取特征定义
with open("models/schema.json", encoding="utf-8") as f:
    FEATURES = json.load(f)["features"]

# 把字符串 "int" 转成 Python 的 int 类型
TYPE_MAP = {"int": int, "float": float, "str": str}
for f in FEATURES:
    f["type"] = TYPE_MAP[f["type"]]
    
def build_dynamic_request(features):
    fields={}
    for f in features:
        fields[f["name"]]=(f["type"],Field(ge=f.get("ge"),le=f.get("le")))
    return create_model("DynamicRequest",**fields)

DynamicRequest=build_dynamic_request(FEATURES)

# 阶段七：加载模型（启动时只加载一次）
try:
    model = joblib.load(MODEL_PATH)
    logger.info(f"模型加载成功：{MODEL_PATH}")
except Exception as e:
    logger.error(f"模型加载失败：{MODEL_PATH}, 错误：{e}")
    raise RuntimeError(f"模型加载失败, 路径：{MODEL_PATH}")

# 阶段八：预测接口（核心业务逻辑）
@app.post("/v1/predict")
def predict(req: DynamicRequest, api_key: str = Depends(verify_api_key), X_client_id: str = Header(default="unknown")):
    # 1. 记录开始时间
    start_time = time.time()

    # 2. 按 FEATURES 里的顺序提取特征值,拼装 numpy 二维数组 (1, 3)
    values = [getattr(req, f["name"]) for f in FEATURES]
    new_X = np.array([values])

    # 3. 调用模型预测类别和概率
    prediction = model.predict(new_X)
    probability = model.predict_proba(new_X)[0][1]

    # 4. 计算耗时并记录日志
    latency = time.time() - start_time
    logger.info(f"/predict | client={X_client_id} | latency={latency:.4f}s | model_version={MODEL_VERSION} | prediction={int(prediction[0])}")

    # 5. 组装返回结果
    return{
        "prediction": int(prediction[0]),
        "probability": float(probability),
        "model_version": MODEL_VERSION,
        "status": "success"
    }
# 阶段九：健康检查接口
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ready")
def  ready():
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载完成")
    return {"status": "ready", "model_version": MODEL_VERSION}