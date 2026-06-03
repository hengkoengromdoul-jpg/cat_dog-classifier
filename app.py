"""
Cat vs Dog Classifier — Streamlit web app
Mini Project 4 · Intro to Machine Learning
Run with: streamlit run app.py
"""
import streamlit as st
import torch
import numpy as np
import cv2 as cv
import pickle
import time

# ============================================================
#  PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐾",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ============================================================
#  CLEAN, MINIMAL CSS
# ============================================================
st.markdown("""
<style>
    /* Hide Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Soft, neutral background */
    .stApp {
        background: #f8f9fa;
    }

    /* Typography */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        color: #2c3e50;
    }

    /* Hero header */
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1a1a1a;
        margin: 0.5rem 0 0.25rem 0;
        letter-spacing: -0.5px;
    }
    .hero-sub {
        color: #6c757d;
        font-size: 1.05rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    /* Clean cards */
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #e9ecef;
        margin-bottom: 1rem;
    }
    .card h3 {
        margin-top: 0;
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a1a1a;
    }

    /* Result cards — subtle, not loud */
    .result {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #e9ecef;
        margin-bottom: 1rem;
    }
    .result-cat { border-top: 4px solid #e67e22; }
    .result-dog { border-top: 4px solid #3498db; }
    .result-unknown { border-top: 4px solid #95a5a6; }

    .result-label {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.3rem 0;
        color: #1a1a1a;
    }
    .result-emoji { font-size: 3rem; line-height: 1; }
    .result-conf {
        color: #6c757d;
        font-size: 1rem;
        margin-top: 0.3rem;
    }

    /* Probability bars */
    .bar-row {
        display: flex;
        align-items: center;
        margin: 0.75rem 0;
        gap: 0.75rem;
    }
    .bar-label {
        min-width: 60px;
        font-size: 0.9rem;
        font-weight: 500;
        color: #495057;
    }
    .bar-track {
        flex: 1;
        background: #f1f3f5;
        height: 8px;
        border-radius: 4px;
        overflow: hidden;
    }
    .bar-fill-cat { background: #e67e22; height: 100%; border-radius: 4px; }
    .bar-fill-dog { background: #3498db; height: 100%; border-radius: 4px; }
    .bar-value {
        min-width: 50px;
        text-align: right;
        font-size: 0.9rem;
        font-weight: 600;
        color: #1a1a1a;
    }

    /* Info note */
    .note {
        background: #f8f9fa;
        border-left: 3px solid #adb5bd;
        padding: 0.75rem 1rem;
        border-radius: 4px;
        color: #495057;
        font-size: 0.9rem;
        margin-top: 1rem;
        line-height: 1.5;
    }

    /* Sidebar — clean dark */
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e9ecef;
    }
    [data-testid="stSidebar"] * {
        color: #2c3e50 !important;
    }
    [data-testid="stSidebar"] h1 { font-size: 1.05rem !important; font-weight: 600 !important; }
    [data-testid="stSidebar"] h2 { font-size: 0.95rem !important; font-weight: 600 !important; color: #495057 !important; margin-top: 1.5rem !important; }

    /* Stat tags in sidebar */
    .tag {
        display: inline-block;
        background: #f1f3f5;
        color: #495057;
        padding: 0.25rem 0.7rem;
        border-radius: 4px;
        font-size: 0.8rem;
        margin: 0.15rem 0.15rem 0.15rem 0;
        font-weight: 500;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: white;
        border-radius: 8px;
    }

    /* Pipeline */
    .pipeline {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 1rem 0 0.5rem 0;
        flex-wrap: wrap;
        gap: 0.4rem;
    }
    .step {
        background: #f1f3f5;
        color: #495057;
        border-radius: 8px;
        padding: 0.5rem 0.8rem;
        font-size: 0.85rem;
        font-weight: 500;
        flex-shrink: 0;
    }
    .arrow {
        color: #adb5bd;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
#  MODEL
# ============================================================
def my_ANN(X, W1, b1, W2, b2, W3, b3):
    Z1 = torch.matmul(X, W1) + b1
    A1 = torch.relu(Z1)
    Z2 = torch.matmul(A1, W2) + b2
    A2 = torch.relu(Z2)
    Z3 = torch.matmul(A2, W3) + b3
    return Z3

@st.cache_resource
def load_model():
    weights = torch.load("model_weights.pt", map_location="cpu")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return weights, scaler

try:
    weights, scaler = load_model()
    W1, b1 = weights["W1"], weights["b1"]
    W2, b2 = weights["W2"], weights["b2"]
    W3, b3 = weights["W3"], weights["b3"]
    model_ok = True
except FileNotFoundError:
    model_ok = False

# ============================================================
#  SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("# Cat vs Dog Classifier")
    st.markdown("A simple neural network for image classification.")

    st.markdown("## Model")
    st.markdown("""
3-layer fully-connected network, trained from scratch.

```
Input  (4096)
  ↓ ReLU
Hidden (16)
  ↓ ReLU
Hidden (8)
  ↓
Output (2)
```
    """)

    st.markdown("## Training")
    st.markdown("""
<div>
  <span class="tag">100 cats</span>
  <span class="tag">100 dogs</span>
  <span class="tag">64×64</span>
  <span class="tag">Grayscale</span>
  <span class="tag">80/20 split</span>
  <span class="tag">SGD</span>
</div>
    """, unsafe_allow_html=True)

    st.markdown("## About")
    st.markdown("""
This is a simple model trained on a small dataset. Predictions are not always correct — the model can be wrong even when it seems sure.

Mini Project 4
Intro to Machine Learning
    """)

# ============================================================
#  MAIN PAGE
# ============================================================
st.markdown('<h1 class="hero-title">Cat vs Dog Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Upload a photo and see what the model predicts.</p>',
            unsafe_allow_html=True)

if not model_ok:
    st.error("Model files not found. Please make sure `model_weights.pt` and `scaler.pkl` are in the same folder as this app.")
    st.stop()

# ----- Pipeline -----
st.markdown("""
<div class="card">
  <h3>How it works</h3>
  <div class="pipeline">
    <div class="step">Photo</div>
    <span class="arrow">→</span>
    <div class="step">Grayscale</div>
    <span class="arrow">→</span>
    <div class="step">Resize 64×64</div>
    <span class="arrow">→</span>
    <div class="step">Flatten</div>
    <span class="arrow">→</span>
    <div class="step">Neural network</div>
    <span class="arrow">→</span>
    <div class="step">Prediction</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ----- Upload -----
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### Upload a photo")
uploaded = st.file_uploader("upload", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

if uploaded is not None:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img_color = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
    img_gray  = cv.imdecode(file_bytes, cv.IMREAD_GRAYSCALE)

    if img_color is None or img_gray is None:
        st.error("Couldn't read that file. Please try another image.")
    else:
        with st.spinner("Analyzing..."):
            time.sleep(0.4)
            img_resized = cv.resize(img_gray, (64, 64))
            x = img_resized.flatten() / 255.0
            x = scaler.transform(x.reshape(1, -1))
            x_tensor = torch.tensor(x, dtype=torch.float32)

            with torch.no_grad():
                Z = my_ANN(x_tensor, W1, b1, W2, b2, W3, b3)
                probs = torch.softmax(Z, dim=1)
                pred = torch.argmax(Z, dim=1).item()

            cat_p = probs[0, 0].item() * 100
            dog_p = probs[0, 1].item() * 100
            confidence = max(cat_p, dog_p)

        UNKNOWN_THRESHOLD = 60.0
        is_unknown = confidence < UNKNOWN_THRESHOLD

        col1, col2 = st.columns([1, 1], gap="medium")

        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Your photo")
            st.image(cv.cvtColor(img_color, cv.COLOR_BGR2RGB), use_container_width=True)
            st.caption(f"{img_color.shape[1]} × {img_color.shape[0]} pixels")

            with st.expander("What the model actually sees"):
                st.image(img_resized, width=160, clamp=True)
                st.caption("The photo is reduced to a 64×64 grayscale image before being passed to the network.")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            if is_unknown:
                st.markdown(f"""
                <div class="result result-unknown">
                    <div class="result-emoji">🤔</div>
                    <div class="result-label">Not Sure</div>
                    <div class="result-conf">This doesn't appear to be a cat or a dog</div>
                </div>
                """, unsafe_allow_html=True)
            elif pred == 0:
                st.markdown(f"""
                <div class="result result-cat">
                    <div class="result-emoji">🐱</div>
                    <div class="result-label">Cat</div>
                    <div class="result-conf">{confidence:.1f}% confidence</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result result-dog">
                    <div class="result-emoji">🐶</div>
                    <div class="result-label">Dog</div>
                    <div class="result-conf">{confidence:.1f}% confidence</div>
                </div>
                """, unsafe_allow_html=True)

            # Probability bars
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("**Probabilities**")
            st.markdown(f"""
            <div class="bar-row">
                <div class="bar-label">Cat</div>
                <div class="bar-track"><div class="bar-fill-cat" style="width:{cat_p}%;"></div></div>
                <div class="bar-value">{cat_p:.1f}%</div>
            </div>
            <div class="bar-row">
                <div class="bar-label">Dog</div>
                <div class="bar-track"><div class="bar-fill-dog" style="width:{dog_p}%;"></div></div>
                <div class="bar-value">{dog_p:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            if is_unknown:
                st.markdown("""
                <div class="note">
                    The model only knows two categories — cat and dog. When the probabilities are close to 50/50, it usually means the photo is something else.
                </div>
                """, unsafe_allow_html=True)
            elif confidence >= 80:
                st.markdown("""
                <div class="note">
                    High confidence does not always mean correct. This model can occasionally be confidently wrong.
                </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)
