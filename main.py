from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import cv2
import numpy as np
import io
from starlette.responses import StreamingResponse
import json

app = FastAPI()

# Загружаем модели (убедись, что файлы .pt лежат в этой же папке)
# Если файлы называются иначе, просто подправь названия здесь
# Загружаем модели
models = {
    "speed": YOLO("yolo11s_ov_640/yolo11s_ov_640_openvino_model.xml"),
    
    "accuracy": YOLO("yolo11m_visdrone_best.pt")
}

@app.post("/detect")
async def detect(mode: str = "speed", file: UploadFile = File(...)):
    # Читаем картинку, которую прислал сайт
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Выбираем модель (Speed или Accuracy)
    model = models.get(mode, models["speed"])
    
    # Запускаем детекцию
    results = model.predict(img, conf=0.25)

    # Рисуем рамки на фото
    res_plotted = results[0].plot()
    
    # Считаем количество найденных объектов
    total_objects = len(results[0].boxes)
    
    # Кодируем обработанное фото обратно в JPEG
    _, im_jpg = cv2.imencode(".jpg", res_plotted)
    
    # Отправляем результат обратно на сайт
    return StreamingResponse(
        io.BytesIO(im_jpg.tobytes()), 
        media_type="image/jpeg",
        headers={"X-Detection-Stats": json.dumps({"total": int(total_objects)})}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)