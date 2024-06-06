import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np
from streamlit_option_menu import option_menu
from proses import classify, set_background, text_to_speech
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import base64

st.set_page_config(
    page_title="TunaMate",
    page_icon="🤖"
)

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = load_model('./model/best_model.h5')
        with open('./model/labels.txt', 'r') as f:
            self.class_names = [a[:-1].split(' ')[1] for a in f.readlines()]

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.image = img
        return img

    def get_image(self):
        return self.image

# Navigasi
with st.sidebar:
    selected = option_menu(
        menu_title="Pilih",
        options=["Deteksi Uang", "About"],
        icons=['house', "list-task"],
        menu_icon="cast",
        default_index=0,
    )

if selected == "Deteksi Uang":
    set_background('./doc/bg2.jpg')
    st.title('Deteksi Nominal Uang Rupiah')
    webrtc_ctx = webrtc_streamer(
        key="example",
        video_transformer_factory=VideoTransformer,
        desired_playing_state=True,
    )
    
    if webrtc_ctx.video_transformer:
        if st.button("Prediksi"):
            video_transformer = webrtc_ctx.video_transformer
            image = video_transformer.get_image()
            if image is not None:
                image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                st.image(image_pil, caption="Gambar dari Webcam", use_column_width=True)
                class_name, conf_score = classify(image_pil, video_transformer.model, video_transformer.class_names)
                st.write("## Rp{}".format(class_name))
                st.write("### Skor Prediksi: {}%".format(int(conf_score * 1000) / 10))

                # Convert text to speech
                text = f"{class_name}"
                audio_file = text_to_speech(text)

                # Play the audio file using st.empty to force refresh
                audio_placeholder = st.empty()
                audio_bytes = open(audio_file, "rb").read()
                audio_html = """
                <audio autoplay>
                <source src="data:audio/mp3;base64,{}" type="audio/mp3">
                </audio>
                """.format(base64.b64encode(audio_bytes).decode())
                audio_placeholder.markdown(audio_html, unsafe_allow_html=True)

if selected == "About":
    st.markdown("""
        # 🤖 RepoMedUNM
        ### Hallo Sahabat 👋

        Ini penjelasan aplikasi.
    """, True)
    st.write("---")

    st.markdown("""
        ### 📑 Informasi Tim
    """, True)
    
    # Info anggota tim
    team_info = [
        {"name": "Anggota 1", "role": "Role 1", "image": "./doc/murni.png"},
        {"name": "Anggota 2", "role": "Role 2", "image": "./doc/murni.png"},
        {"name": "Anggota 3", "role": "Role 3", "image": "./doc/murni.png"},
        {"name": "Anggota 4", "role": "Role 4", "image": "./doc/murni.png"},
        {"name": "Anggota 5", "role": "Role 5", "image": "./doc/murni.png"},
    ]
    
    # Menampilkan anggota tim dalam dua baris
    col1, col2, col3 = st.columns(3)
    for i, member in enumerate(team_info[:3]):
        with [col1, col2, col3][i]:
            st.image(member["image"], width=150)
            st.markdown(f"<div style='text-align: center; margin-top: -10px;'><strong>{member['name']}</strong></div>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align: center; margin-top: -5px;'>{member['role']}</div>", unsafe_allow_html=True)

    col_empty, col4, col5, col_empty2 = st.columns([1, 2, 2, 1])
    with col_empty:
        st.empty()
    with col4:
        st.image(team_info[3]["image"], width=150)
        st.markdown(f"<div style='text-align: center; margin-top: -10px;'><strong>{team_info[3]['name']}</strong></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align: center; margin-top: -5px;'>{team_info[3]['role']}</div>", unsafe_allow_html=True)

    with col5:
        st.image(team_info[4]["image"], width=150)
        st.markdown(f"<div style='text-align: center; margin-top: -10px;'><strong>{team_info[4]['name']}</strong></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align: center; margin-top: -5px;'>{team_info[4]['role']}</div>", unsafe_allow_html=True)

    with col_empty2:
        st.empty()
