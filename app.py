import streamlit as st
import requests
from PIL import Image
import io

# Настройка внешнего вида страницы
st.set_page_config(page_title="VisDrone Monitoring MVP", layout="wide")

st.title("🛸 Система мониторинга БПЛА (VisDrone)")
st.write("Интерфейс для анализа транспортного потока по фото.")

# Настройки в боковой панели
st.sidebar.header("Настройки")
# По умолчанию ставим адрес локального сервера, который ты запускаешь через main.py
api_url = st.sidebar.text_input("Адрес API сервера:", "http://127.0.0.1:8000")

model_mode = st.sidebar.radio(
    "Выберите модель:",
    ("Speed (YOLO11s + OpenVINO)", "Accuracy (YOLO11m)")
)

# Переводим выбор пользователя в понятный для сервера формат
api_mode = "speed" if "Speed" in model_mode else "accuracy"

# Загрузка файла
uploaded_file = st.sidebar.file_uploader("Загрузить снимок с дрона", type=['jpg', 'jpeg', 'png'])

# Основная часть экрана
col1, col2 = st.columns(2)

if uploaded_file is not None:
    # 1. Показываем оригинал
    image = Image.open(uploaded_file)
    with col1:
        st.subheader("Оригинал")
        st.image(image, use_container_width=True)

    # 2. Кнопка запуска
    if st.sidebar.button("🚀 Запустить детекцию"):
        with st.spinner('Нейросеть обрабатывает изображение...'):
            try:
                # Подготовка фото для отправки
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='JPEG')
                files = {'file': ('image.jpg', img_byte_arr.getvalue(), 'image/jpeg')}
                
                # Отправка запроса на твой запущенный main.py
                response = requests.post(f"{api_url}/detect?mode={api_mode}", files=files)
                
                if response.status_code == 200:
                    # Показываем результат
                    result_img = Image.open(io.BytesIO(response.content))
                    stats = response.headers.get("X-Detection-Stats", "{}")
                    
                    with col2:
                        st.subheader("Результат")
                        st.image(result_img, use_container_width=True)
                    
                    st.success(f"Готово! Аналитика: {stats}")
                else:
                    st.error(f"Ошибка сервера: {response.status_code}")
            except Exception as e:
                st.error(f"Не удалось связаться с бэкендом: {e}")
else:
    st.info("👈 Загрузите изображение в боковой панели, чтобы начать.")