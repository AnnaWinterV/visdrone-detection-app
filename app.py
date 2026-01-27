import streamlit as st
from PIL import Image
import io

from main import model_accuracy, model_speed, detect_objects

# Настройка страницы
st.set_page_config(page_title="VisDrone Monitoring MVP", layout="wide")

st.title("🛸 Система мониторинга БПЛА (VisDrone)")
st.write("Интерфейс для анализа транспортного потока. Работает напрямую в облаке.")

# Настройки в боковой панели
st.sidebar.header("Настройки")

model_choice = st.sidebar.radio(
    "Выберите модель:",
    ("Speed (YOLO11s + OpenVINO)", "Accuracy (YOLO11m)")
)

# Загрузка файла
uploaded_file = st.sidebar.file_uploader("Загрузить снимок с дрона", type=['jpg', 'jpeg', 'png'])

# Основная часть экрана
col1, col2 = st.columns(2)

if uploaded_file is not None:
    # Показываем оригинал
    image = Image.open(uploaded_file)
    with col1:
        st.subheader("Оригинал")
        st.image(image, use_container_width=True)

    #  Кнопка запуска
    if st.sidebar.button("Запустить детекцию"):
        with st.spinner('Нейросеть обрабатывает изображение внутри сервера...'):
            try:
                # Прямой вызов функции 
                if "Accuracy" in model_choice:
                    result_img, stats = detect_objects(image, model_accuracy)
                    mode_label = "Accuracy"
                else:
                    result_img, stats = detect_objects(image, model_speed)
                    mode_label = "Speed"

                # Вывод результата
                with col2:
                    st.subheader(f"Результат ({mode_label})")
                    st.image(result_img, use_container_width=True)
                
                st.success("Детекция завершена!")
                st.subheader("Аналитика объектов")
                st.json(stats) 

            except Exception as e:
                st.error(f"Ошибка при обработке: {e}")
                st.warning("Убедитесь, что функции загрузки моделей в main.py доступны.")
else:
    st.info("Загрузите изображение в боковой панели, чтобы начать.")
