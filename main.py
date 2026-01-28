import os
os.environ['YOLO_CONFIG_DIR'] = '/tmp'

from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import cv2
import numpy as np
import io
from starlette.responses import StreamingResponse
import json
import time  
from collections import Counter 

app = FastAPI()

# Загружаем модели 
model_speed = YOLO("yolo11s_ov_640/yolo11s_ov_640_openvino_model.xml", task="detect")
model_accuracy = YOLO("yolo11m_visdrone_best.pt", task="detect")

def detect_objects(img, model):
    start_time = time.time()  
    
    results = model.predict(img, conf=0.25)
    
    end_time = time.time()    
    inference_duration = round(end_time - start_time, 3) 
    
    res_plotted = results[0].plot()
    
    # Сбор статистики по классам
    names = results[0].names
    classes = results[0].boxes.cls.cpu().numpy()
    class_counts = Counter([names[int(c)] for c in classes])
    
    stats = {
        "total": int(len(results[0].boxes)),
        "inference_time_sec": inference_duration,
        "class_counts": dict(class_counts) # Передаем словарь с подсчетом
    }
    
    return cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB), stats

@app.post("/detect")
async def detect(mode: str = "speed", file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    current_model = model_accuracy if mode == "accuracy" else model_speed
    res_img, stats = detect_objects(img, current_model)
    
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
