
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from typing import Tuple
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image

app = FastAPI(title="NDVI 图像分析 API", description="上传遥感图像（含红光和近红外波段），返回NDVI分析结果。")

def calculate_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    # 避免除0错误
    bottom = (nir + red).astype(float)
    bottom[bottom == 0] = 0.01
    ndvi = (nir - red) / bottom
    return ndvi

def encode_image(image: np.ndarray) -> str:
    pil_image = Image.fromarray(image)
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@app.post("/analyze_ndvi/")
async def analyze_ndvi(red_band: UploadFile = File(...), nir_band: UploadFile = File(...)):
    # 读取上传的红光和近红外波段图像
    red = np.array(Image.open(red_band.file).convert("L"))
    nir = np.array(Image.open(nir_band.file).convert("L"))
    
    ndvi = calculate_ndvi(red, nir)
    ndvi_normalized = ((ndvi + 1) / 2 * 255).astype(np.uint8)  # 归一化显示

    # 构造伪彩色图（红色低、绿色高）
    ndvi_colored = cv2.applyColorMap(ndvi_normalized, cv2.COLORMAP_JET)
    ndvi_base64 = encode_image(ndvi_colored)
    
    mean_ndvi = float(np.mean(ndvi))
    result_text = f"该区域NDVI平均值为 {mean_ndvi:.3f}，植被覆盖度处于 {'较低' if mean_ndvi < 0.3 else '中等' if mean_ndvi < 0.6 else '较高'} 水平。"

    return JSONResponse({
        "mean_ndvi": mean_ndvi,
        "ndvi_summary": result_text,
        "ndvi_image_base64": ndvi_base64
    })
