from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import os
import uuid
from PIL import Image
import uvicorn

app = FastAPI()

# 挂载静态文件
os.makedirs("static", exist_ok=True)
os.makedirs("static/results", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# 加载 YOLO 模型（替换为你的模型路径）
model = YOLO("yolo11n.pt")  # 或你的自定义模型权重

# 允许 Vue 前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vue 默认端口
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 测试连接接口（已验证可用）
@app.get("/api/test/connect")
async def test_connect():
    return {"code": 200, "message": "前后端连通成功！"}

# 核心推理接口
@app.post("/api/inference/single")
async def inference_single(file: UploadFile = File(...)):
    try:
        # 1. 保存临时文件
        temp_filename = f"temp_{uuid.uuid4().hex}.jpg"
        temp_path = os.path.join("static", temp_filename)
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # 2. YOLO 推理，固定保存目录，避免 exp1/exp2
        results = model(
            temp_path,
            save=True,
            project="static/results",
            name="latest",  # 固定目录名，每次覆盖
            exist_ok=True
        )

        # 3. 获取标注图片路径（更可靠的方式）
        result_img_path = os.path.join(
            "static/results/latest",
            os.path.basename(temp_path)
        )

        # 验证文件是否存在
        if not os.path.exists(result_img_path):
            raise Exception(f"结果图片未生成：{result_img_path}")

        # 4. 生成可访问的 URL
        result_img_url = f"http://localhost:8000/{result_img_path}"

        # 5. 解析检测结果
        detections = []
        for box in results[0].boxes:
            detections.append({
                "class": model.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "bbox": box.xyxy[0].tolist()
            })

        # 6. 清理临时文件
        os.remove(temp_path)

        return {
            "code": 200,
            "message": "推理成功",
            "data": {
                "detections": detections,
                "image_url": result_img_url
            }
        }

    except Exception as e:
        # 打印完整错误栈，方便调试
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"code": 500, "message": f"推理失败: {str(e)}"}
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
