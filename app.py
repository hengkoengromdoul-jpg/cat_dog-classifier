"""
Pawsense — A neural network that sees cats and dogs
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
    page_title="Pawsense · AI Vision",
    page_icon="◐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
#  CREATIVE CSS — premium "AI startup" aesthetic
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Deep space background with subtle mesh gradient */
    .stApp {
        background:
            radial-gradient(at 20% 20%, rgba(139, 92, 246, 0.25) 0px, transparent 50%),
            radial-gradient(at 80% 30%, rgba(236, 72, 153, 0.18) 0px, transparent 50%),
            radial-gradient(at 50% 90%, rgba(59, 130, 246, 0.18) 0px, transparent 50%),
            #0a0b1e;
        background-attachment: fixed;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
        color: #e2e8f0;
    }

    /* === HERO === */
    .hero {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .badge {
        display: inline-block;
        background: rgba(139, 92, 246, 0.15);
        border: 1px solid rgba(139, 92, 246, 0.4);
        color: #c4b5fd;
        padding: 0.35rem 0.9rem;
        border-radius: 100px;
        font-size: 0.78rem;
        font-weight: 500;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #fff 0%, #c4b5fd 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0.3rem 0;
        letter-spacing: -2px;
        line-height: 1.1;
    }
    .hero-sub {
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: 400;
        margin: 0.5rem 0 2rem 0;
        max-width: 560px;
        margin-left: auto;
        margin-right: auto;
        line-height: 1.6;
    }

    /* === GLASS CARDS === */
    .card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 1.75rem;
        margin-bottom: 1rem;
        transition: border-color 0.3s ease;
    }
    .card:hover {
        border-color: rgba(139, 92, 246, 0.3);
    }
    .card h3, .card h4 {
        color: #f1f5f9;
        margin-top: 0;
        font-weight: 600;
        letter-spacing: -0.3px;
    }
    .card-label {
        font-size: 0.72rem;
        color: #8b5cf6;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    /* === RESULT CARD === */
    .result {
        position: relative;
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 2.5rem 1.5rem;
        text-align: center;
        overflow: hidden;
        animation: rise 0.6s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .result::before {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; height: 200px;
        opacity: 0.18;
        pointer-events: none;
        z-index: 0;
    }
    .result-cat::before {
        background: radial-gradient(ellipse at top, #f97316 0%, transparent 70%);
    }
    .result-dog::before {
        background: radial-gradient(ellipse at top, #3b82f6 0%, transparent 70%);
    }
    .result-unknown::before {
        background: radial-gradient(ellipse at top, #64748b 0%, transparent 70%);
    }
    .result > * { position: relative; z-index: 1; }
    .result-emoji {
        font-size: 4rem;
        line-height: 1;
        margin-bottom: 0.5rem;
        filter: drop-shadow(0 4px 20px rgba(0,0,0,0.3));
    }
    .result-label {
        font-size: 3rem;
        font-weight: 800;
        margin: 0.3rem 0 0.2rem 0;
        letter-spacing: -1.5px;
        color: #fff;
    }
    .result-conf {
        color: #cbd5e1;
        font-size: 1rem;
        font-weight: 500;
    }
    .result-conf .num {
        color: #fff;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }

    @keyframes rise {
        from { opacity: 0; transform: translateY(20px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* === PROBABILITY BARS === */
    .prob-row {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin: 0.85rem 0;
    }
    .prob-name {
        min-width: 50px;
        font-size: 0.9rem;
        font-weight: 500;
        color: #e2e8f0;
    }
    .prob-track {
        flex: 1;
        background: rgba(255, 255, 255, 0.06);
        height: 6px;
        border-radius: 999px;
        overflow: hidden;
    }
    .prob-fill {
        height: 100%;
        border-radius: 999px;
        transition: width 1.2s cubic-bezier(0.16, 1, 0.3, 1);
        box-shadow: 0 0 12px currentColor;
    }
    .prob-fill-cat {
        background: linear-gradient(90deg, #fb923c, #f97316);
        color: #f97316;
    }
    .prob-fill-dog {
        background: linear-gradient(90deg, #60a5fa, #3b82f6);
        color: #3b82f6;
    }
    .prob-value {
        min-width: 60px;
        text-align: right;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        font-size: 0.9rem;
        color: #fff;
    }

    /* === PIPELINE === */
    .pipeline {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.4rem;
        flex-wrap: wrap;
        margin-top: 0.5rem;
    }
    .step {
        flex: 1;
        min-width: 90px;
        background: rgba(139, 92, 246, 0.08);
        border: 1px solid rgba(139, 92, 246, 0.2);
        color: #c4b5fd;
        padding: 0.65rem 0.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 0.78rem;
        font-weight: 500;
    }
    .step-icon {
        font-size: 1.1rem;
        margin-bottom: 0.2rem;
        display: block;
    }
    .arrow {
        color: rgba(139, 92, 246, 0.4);
        font-weight: 700;
        font-size: 1rem;
    }

    /* === NOTE === */
    .note {
        background: rgba(139, 92, 246, 0.08);
        border-left: 2px solid #8b5cf6;
        padding: 0.85rem 1rem;
        border-radius: 6px;
        color: #cbd5e1;
        font-size: 0.88rem;
        margin-top: 1rem;
        line-height: 1.55;
    }

    /* === SIDEBAR === */
    [data-testid="stSidebar"] {
        background: rgba(10, 11, 30, 0.92);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
        backdrop-filter: blur(20px);
    }
    [data-testid="stSidebar"] * { color: #cbd5e1 !important; }
    [data-testid="stSidebar"] h1 {
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        color: #fff !important;
    }
    [data-testid="stSidebar"] h2 {
        font-size: 0.7rem !important;
        font-weight: 600 !important;
        color: #8b5cf6 !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
        margin-top: 1.5rem !important;
    }
    [data-testid="stSidebar"] code, [data-testid="stSidebar"] pre {
        background: rgba(139, 92, 246, 0.08) !important;
        color: #c4b5fd !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.78rem !important;
        border-radius: 6px !important;
    }
    .tag {
        display: inline-block;
        background: rgba(139, 92, 246, 0.12);
        border: 1px solid rgba(139, 92, 246, 0.25);
        color: #c4b5fd !important;
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        font-size: 0.72rem;
        margin: 0.15rem 0.15rem 0.15rem 0;
        font-weight: 500;
        font-family: 'JetBrains Mono', monospace;
    }

    /* === UPLOADER === */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.04);
        border-radius: 16px;
        border: 1px dashed rgba(139, 92, 246, 0.4);
    }
    [data-testid="stFileUploader"] section {
        background: transparent !important;
        border: none !important;
    }
    [data-testid="stFileUploader"] button {
        background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%) !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
    }

    /* Streamlit captions */
    [data-testid="stCaptionContainer"] { color: #64748b !important; }

    /* Expander */
    [data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
    }
    [data-testid="stExpander"] summary { color: #c4b5fd !important; }
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
    st.markdown("# ◐ Pawsense")
    st.markdown("<p style='color:#64748b; font-size:0.85rem; margin-top:-0.5rem;'>Neural network · v1.0</p>",
                unsafe_allow_html=True)

    st.markdown("## Architecture")
    st.markdown("""
```
Input    4096
  ↓ ReLU
Hidden     16
  ↓ ReLU
Hidden      8
  ↓
Output      2
```
    """)

    st.markdown("## Dataset")
    st.markdown("""
<div>
  <span class="tag">100 cats</span>
  <span class="tag">100 dogs</span>
  <span class="tag">64×64</span>
  <span class="tag">grayscale</span>
  <span class="tag">80/20 split</span>
  <span class="tag">SGD</span>
</div>
    """, unsafe_allow_html=True)

    st.markdown("## Notes")
    st.markdown("""
<p style='font-size:0.85rem; color:#94a3b8; line-height:1.6;'>
Trained from scratch on a small dataset. Predictions can be wrong even when the model is confident — this is a known limitation of simple networks trained on limited data.
</p>
    """, unsafe_allow_html=True)

    st.markdown("## Project")
    st.markdown("""
<p style='font-size:0.78rem; color:#64748b;'>
Mini Project 4<br/>
Intro to Machine Learning<br/>
Year 3 ITC · I3 AMS S2
</p>
    """, unsafe_allow_html=True)

# ============================================================
#  HERO
# ============================================================
st.markdown("""
<div class="hero">
    <div class="badge">◐ AI VISION DEMO</div>
    <h1 class="hero-title">See like a network.</h1>
    <p class="hero-sub">A neural network trained from scratch to distinguish cats from dogs. Upload a photo and watch the model reason — pixel by pixel, layer by layer.</p>
</div>
""", unsafe_allow_html=True)

if not model_ok:
    st.error("Model files not found. Make sure `model_weights.pt` and `scaler.pkl` are next to this app.")
    st.stop()

# ============================================================
#  PIPELINE
# ============================================================
st.markdown("""
<div class="card">
  <div class="card-label">Pipeline</div>
  <h3 style="margin-bottom:1rem;">How the model sees your photo</h3>
  <div class="pipeline">
    <div class="step"><span class="step-icon">📷</span>Input</div>
    <span class="arrow">→</span>
    <div class="step"><span class="step-icon">◐</span>Grayscale</div>
    <span class="arrow">→</span>
    <div class="step"><span class="step-icon">▦</span>64×64</div>
    <span class="arrow">→</span>
    <div class="step"><span class="step-icon">∥</span>Flatten</div>
    <span class="arrow">→</span>
    <div class="step"><span class="step-icon">⌘</span>Network</div>
    <span class="arrow">→</span>
    <div class="step"><span class="step-icon">✦</span>Prediction</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
#  UPLOAD
# ============================================================
st.markdown("""
<div class="card" style="border-left: 3px solid #8b5cf6;">
  <div style="display:flex; gap:0.8rem; align-items:flex-start;">
    <div style="font-size:1.3rem;">💡</div>
    <div style="font-size:0.92rem; color:#cbd5e1; line-height:1.55;">
      <b style="color:#fff;">For best results, upload a photo of a cat or a dog.</b><br/>
      This model was trained on only those two animals — it will always pick one of them, even if the photo is of something else. That's a known limitation of simple classifiers.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="card"><div class="card-label">Upload</div><h3>Choose a photo</h3>', unsafe_allow_html=True)
uploaded = st.file_uploader("upload", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

if uploaded is not None:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img_color = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
    img_gray  = cv.imdecode(file_bytes, cv.IMREAD_GRAYSCALE)

    if img_color is None or img_gray is None:
        st.error("Couldn't read that file. Please try another image.")
    else:
        with st.spinner("Running inference..."):
            time.sleep(0.5)
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

        col1, col2 = st.columns([1, 1], gap="medium")

        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-label">Input</div>', unsafe_allow_html=True)
            st.markdown('<h3>Your photo</h3>', unsafe_allow_html=True)
            st.image(cv.cvtColor(img_color, cv.COLOR_BGR2RGB), use_container_width=True)
            st.caption(f"Original · {img_color.shape[1]} × {img_color.shape[0]} px")

            with st.expander("◐ What the network actually sees"):
                st.image(img_resized, width=180, clamp=True)
                st.caption("Reduced to 64×64 grayscale — 4,096 pixel values.")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            # Result card — Cat or Dog only
            if pred == 0:
                emoji, label, cls = "🐱", "Cat", "result-cat"
            else:
                emoji, label, cls = "🐶", "Dog", "result-dog"

            st.markdown(f"""
            <div class="result {cls}">
                <div class="card-label">Prediction</div>
                <div class="result-emoji">{emoji}</div>
                <div class="result-label">{label}</div>
                <div class="result-conf">Confidence · <span class="num">{confidence:.1f}%</span></div>
            </div>
            """, unsafe_allow_html=True)

            # Probabilities
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-label">Distribution</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="prob-row">
                <div class="prob-name">Cat</div>
                <div class="prob-track">
                    <div class="prob-fill prob-fill-cat" style="width:{cat_p}%;"></div>
                </div>
                <div class="prob-value">{cat_p:.1f}%</div>
            </div>
            <div class="prob-row">
                <div class="prob-name">Dog</div>
                <div class="prob-track">
                    <div class="prob-fill prob-fill-dog" style="width:{dog_p}%;"></div>
                </div>
                <div class="prob-value">{dog_p:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="note">
                This model was trained only on cats and dogs. It will always pick one — even for unrelated photos. High confidence does not always mean correct.
            </div>
            """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)
