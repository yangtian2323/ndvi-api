
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import matplotlib.pyplot as plt
import cv2
import io
import base64

app = FastAPI()

# CORS 跨域支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze_ndvi/")
async def analyze_ndvi(
    red_band: UploadFile = File(...),
    nir_band: UploadFile = File(...)
):
    # 读取上传图像
    red_bytes = await red_band.read()
    nir_bytes = await nir_band.read()

    red_array = cv2.imdecode(np.frombuffer(red_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    nir_array = cv2.imdecode(np.frombuffer(nir_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)

    if red_array is None or nir_array is None or red_array.shape != nir_array.shape:
        return JSONResponse(content={"error": "图像读取失败或尺寸不一致"}, status_code=400)

    # 转换为 float32 并归一化
    red = red_array.astype(np.float32) / 255.0
    nir = nir_array.astype(np.float32) / 255.0

    # 计算 NDVI
    ndvi = (nir - red) / (nir + red + 1e-6)
    ndvi = np.clip(ndvi, -1, 1)

    mean_ndvi = float(np.mean(ndvi))

    # 绘制 NDVI 热力图
    fig, ax = plt.subplots()
    heatmap = ax.imshow(ndvi, cmap="RdYlGn")
    fig.colorbar(heatmap, ax=ax, label="NDVI")
    ax.set_title("NDVI Heatmap")
    ax.axis("off")

    # 图像转 base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    ndvi_img_base64 = base64.b64encode(buf.read()).decode("utf-8")

    # 输出 JSON 响应
    return {
        "mean_ndvi": round(mean_ndvi, 3),
        "ndvi_summary": f"该区域 NDVI 平均值为 {round(mean_ndvi,3)}，植物覆盖度较{'高' if mean_ndvi > 0.5 else '一般' if mean_ndvi > 0.2 else '较低'}。",
        "ndvi_image_base64": ndvi_img_base64,
    }
