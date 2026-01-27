import streamlit as st
from PIL import Image
import numpy as np

from main import model_accuracy, model_speed, detect_objects

st.set_page_config(page_title="VisDrone Detector", layout="wide")

st.title("VisDrone Monitoring Cloud")

# Боковая панель
st.sidebar.header("Настройки")
model_choice = st.sidebar.radio("Выберите режим:", ("Speed (OpenVINO)", "Accuracy (YOLO11m)"))
uploaded_file = st.sidebar.file_uploader("Загрузить фото", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    # Загружаем фото и принудительно переводим в RGB 
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    
    if st.sidebar.button("Запустить анализ"):
        with st.spinner('Нейросеть работает...'):
            # Выбираем модель на основе выбора пользователя
            model = model_accuracy if "Accuracy" in model_choice else model_speed
            
            # Запускаем детекцию напрямую через функцию в main.py
            res_img, stats = detect_objects(img_array, model)
            
            # Показываем результат
            col1, col2 = st.columns(2)
            col1.image(image, caption="Оригинал", use_container_width=True)
            col2.image(res_img, caption=f"Результат ({model_choice})", use_container_width=True)
            
            # Вывод статистики из JSON
            st.success(f"Анализ завершен! Найдено объектов: {stats['total']}")
            st.json(stats)
else:
    st.info("Загрузите фото в боковой панели")
