"""
Cat vs Dog Web Predictor
Run with: streamlit run app.py
"""
import streamlit as st
import torch
import numpy as np
import cv2 as cv
import pickle

# ---------- Page setup ----------
st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐱", layout="centered")
st.title("🐱 Cat vs Dog Classifier 🐶")
st.write("Upload a photo and the neural network will predict whether it's a cat or a dog.")
st.caption("Mini Project 4 — 3-layer ANN trained from scratch")

# ---------- Model definition (same as the notebook) ----------
def my_ANN(X, W1, b1, W2, b2, W3, b3):
    Z1 = torch.matmul(X, W1) + b1
    A1 = torch.relu(Z1)
    Z2 = torch.matmul(A1, W2) + b2
    A2 = torch.relu(Z2)
    Z3 = torch.matmul(A2, W3) + b3
    return Z3

# ---------- Load trained weights + scaler ----------
@st.cache_resource
def load_model():
    weights = torch.load("model_weights.pt", map_location="cpu")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return weights, scaler

try:
    weights, scaler = load_model()
    W1 = weights["W1"]; b1 = weights["b1"]
    W2 = weights["W2"]; b2 = weights["b2"]
    W3 = weights["W3"]; b3 = weights["b3"]
    model_ok = True
except FileNotFoundError:
    st.error("⚠️ Model files not found. Run the **Save Model** cell in your notebook "
             "to create `model_weights.pt` and `scaler.pkl`, then put them next to this app.py.")
    model_ok = False

# ---------- File uploader ----------
uploaded = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png"])

if uploaded is not None and model_ok:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img_color = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
    img_gray  = cv.imdecode(file_bytes, cv.IMREAD_GRAYSCALE)

    if img_color is None or img_gray is None:
        st.error("Could not read the image. Try another file.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Your photo")
            st.image(cv.cvtColor(img_color, cv.COLOR_BGR2RGB), use_container_width=True)
            st.caption(f"Original size: {img_color.shape[1]}×{img_color.shape[0]}")

        # ----- Preprocess EXACTLY like training -----
        img_resized = cv.resize(img_gray, (64, 64))
        x = img_resized.flatten() / 255.0
        x = scaler.transform(x.reshape(1, -1))
        x_tensor = torch.tensor(x, dtype=torch.float32)

        with torch.no_grad():
            Z = my_ANN(x_tensor, W1, b1, W2, b2, W3, b3)
            probs = torch.softmax(Z, dim=1)
            pred = torch.argmax(Z, dim=1).item()

        class_names = ["🐱 Cat", "🐶 Dog"]
        confidence = probs[0, pred].item() * 100
        cat_p = probs[0, 0].item() * 100
        dog_p = probs[0, 1].item() * 100

        with col2:
            st.subheader("Prediction")
            if confidence >= 75:
                st.success(f"## {class_names[pred]}")
                st.write(f"Confident ({confidence:.1f}%)")
            elif confidence >= 60:
                st.warning(f"## {class_names[pred]}")
                st.write(f"Fairly sure ({confidence:.1f}%)")
            else:
                st.info(f"## {class_names[pred]}")
                st.write(f"Not very sure ({confidence:.1f}%) — model is guessing")

            st.write("**Probabilities**")
            st.write(f"🐱 Cat: {cat_p:.1f}%")
            st.progress(cat_p / 100)
            st.write(f"🐶 Dog: {dog_p:.1f}%")
            st.progress(dog_p / 100)

        with st.expander("See what the model sees (64×64 grayscale)"):
            st.image(img_resized, width=200, clamp=True)
            st.caption("This is the tiny grayscale version the network actually processes.")

elif not model_ok:
    st.stop()
else:
    st.info("👆 Upload a photo above to get a prediction.")
