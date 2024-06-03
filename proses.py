import base64
import streamlit as st
import numpy as np
from gtts import gTTS

def set_background(image_file):
    with open(image_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)

def classify(image, model, class_names):
    image = image.resize((200, 200))
    image_array = np.asarray(image)
    prediction = model.predict(np.expand_dims(image_array, axis=0))
    class_index = np.argmax(prediction)
    class_name = class_names[class_index]
    confidence_score = prediction[0][class_index]
    return class_name, confidence_score

def text_to_speech(text, lang='id'):
    tts = gTTS(text=text, lang=lang)
    audio_file = "output.mp3"
    tts.save(audio_file)
    return audio_file