import streamlit as st
import numpy as np
from PIL import Image
from tensorflow import keras
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "model_cuaca.keras"

model = keras.models.load_model(MODEL_PATH)
kelas = ['Cloudy', 'Rain', 'Shine', 'Sunrise']
st.title('Klasifikasi Kondisi Cuaca')
st.write('Prediksi kondisi cuaca menggunakan CNN')

file = st.file_uploader(
    'Upload gambar cuaca',
    type=['jpg', 'jpeg', 'png']
)

if file is not None:
    gambar = Image.open(file).convert('RGB')
    st.image(gambar, caption='Gambar Input')
    gambar = gambar.resize((128, 128))
    x = np.array(gambar)
    x = np.expand_dims(x, axis=0)
    hasil = model.predict(x, verbose=0)[0]
    index = np.argmax(hasil)
    confidence = np.max(hasil)

    st.subheader('Hasil Prediksi')
    st.write('Kondisi:', kelas[index])
    st.write('Confidence:', f'{confidence:.2%}')

    if confidence < 0.75:
        st.warning('Prediksi perlu ditinjau')
    else:
        st.success('Prediksi diterima')
    st.subheader('Probabilitas Kelas')

    for nama, nilai in zip(kelas, hasil):
        st.write(f'{nama}: {nilai:.2%}')
