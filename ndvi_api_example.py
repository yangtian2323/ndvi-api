from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from fastapi.middleware.cors import CORSMiddleware   # 🔧 新增

app = FastAPI()

# 🔧 新增：允许跨域访问（建议改成你的前端站点地址）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产建议写为 ['https://你的Netlify网址']
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def read_image(upload_file):
    contents = upload_file.file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
    return image

@app.post("/analyze_ndvi/")
async def analyze_ndvi(red_band: UploadFile = File(...), nir_band: UploadFile = File(...)):
    red = read_image(red_band)
    nir = read_image(nir_band)

    # Compute NDVI
    red = red.astype(float)
    nir = nir.astype(float)
    denominator = (nir + red)
    denominator[denominator == 0] = 1e-5
    ndvi = (nir - red) / denominator

    # Mean NDVI value
    mean_ndvi = np.mean(ndvi)

    # Create NDVI color map image
    plt.figure(figsize=(5, 5))
    plt.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.colorbar()
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    ndvi_image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()

    return {
        "mean_ndvi": float(np.round(mean_ndvi, 3)),
        "ndvi_summary": f"该区域NDVI平均值为 {np.round(mean_ndvi, 3)}，植被覆盖度处于{'较高' if mean_ndvi > 0.4 else '一般' if mean_ndvi > 0.2 else '较低'}。",
        "ndvi_image_base64": ndvi_image_base64
    }
