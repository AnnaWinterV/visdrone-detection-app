from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import cv2
import numpy as np
import io
from starlette.responses import StreamingResponse
import json

app = FastAPI()

# Загружаем модели 
model_speed = YOLO("yolo11s_ov_640", task="detect") 
model_accuracy = YOLO("yolo11m_visdrone_best.pt", task="detect")

# Функция детекции для Streamlit
def detect_objects(img, model):
    results = model.predict(img, conf=0.25)
    res_plotted = results[0].plot()
    total_objects = len(results[0].boxes)
    # Конвертируем из BGR (OpenCV) в RGB (Streamlit)
    return cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB), {"total": int(total_objects)}

# Логика для FastAPI 
@app.post("/detect")
async def detect(mode: str = "speed", file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    current_model = model_accuracy if mode == "accuracy" else model_speed
    res_img, stats = detect_objects(img, current_model)
    
    # Конвертируем обратно в BGR для энкодинга в JPG
    res_bgr = cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR)
    _, im_jpg = cv2.imencode(".jpg", res_bgr)
    
    return StreamingResponse(
        io.BytesIO(im_jpg.tobytes()), 
        media_type="image/jpeg",
        headers={"X-Detection-Stats": json.dumps(stats)}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
