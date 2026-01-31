import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt


nltk.download("stopwords", quiet=True)


# Load model & tokenizer
model = load_model("fake_news_bilstm_model.h5")

class_names = ["FAKE", "REAL"]

def predict_proba(texts):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=MAX_LEN)
    preds = model.predict(padded)

    # LIME expects probability for each class
    return np.hstack([(1 - preds), preds])

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

MAX_LEN = 300

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", layout="centered")

st.title("📰 Fake News Detection System")
st.write("Enter a news article to check whether it is **Fake or Real**.")

user_input = st.text_area("Paste News Article Here")

explainer = LimeTextExplainer(class_names=class_names)

if st.button("Predict", key="predict_button"):
    # from lime.lime_text import LimeTextExplainer
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)

        seq = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(seq, maxlen=MAX_LEN)

        prob = model.predict(padded)[0][0]

        if prob > 0.5:
            label = "REAL"
            confidence = prob * 100
            st.success(f"✅ Prediction: {label}")
        else:
            label = "FAKE"
            confidence = (1 - prob) * 100
            st.error(f"❌ Prediction: {label}")

        st.write(f"Confidence: **{confidence:.2f}%**")
        st.progress(int(confidence))

        # st.subheader("🔍 Explanation (LIME)")
        # exp = explainer.explain_instance(
        #       cleaned,
        #       predict_proba,
        #       num_features=10
        # )

        # fig = exp.as_pyplot_figure()
        # st.pyplot(fig)




