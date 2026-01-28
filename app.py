import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd 
from main import model_accuracy, model_speed, detect_objects

st.set_page_config(page_title="VisDrone Detector", layout="wide")
st.title("VisDrone Monitoring Cloud")

st.sidebar.header("Настройки")
model_choice = st.sidebar.radio("Выберите режим:", ("Speed (OpenVINO)", "Accuracy (YOLO11m)"))
uploaded_file = st.sidebar.file_uploader("Загрузить фото", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    
    if st.sidebar.button("Запустить анализ"):
        with st.spinner('Нейросеть работает...'):
            model = model_accuracy if "Accuracy" in model_choice else model_speed
            res_img, stats = detect_objects(img_array, model)
            
            # Вывод метрик (верхний ряд)
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.metric("Найдено объектов", stats['total'])
            with col_res2:
                st.metric("Время обработки", f"{stats['inference_time_sec']} сек")
            
            # Вывод изображений
            col1, col2 = st.columns(2)
            col1.image(image, caption="Оригинал", use_container_width=True)
            col2.image(res_img, caption=f"Результат ({model_choice})", use_container_width=True)
            
            # Вывод таблицы статистики по классам
            if 'class_counts' in stats and stats['class_counts']:
                st.write("Распределение объектов по типам")
                df_counts = pd.DataFrame(
                    stats['class_counts'].items(), 
                    columns=['Тип объекта', 'Количество']
                ).sort_values(by='Количество', ascending=False)
                st.table(df_counts)
            
            with st.expander("Технические детали (JSON)"):
                st.json(stats)
else:
    st.info("Загрузите фото в боковой панели")
