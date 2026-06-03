"""
🐾 Pawsome Predictor — Cat vs Dog AI Classifier
A creative Streamlit web app for Mini Project 4
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
    page_title="Pawsome Predictor 🐾",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
#  CUSTOM CSS — the creative magic
# ============================================================
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Animated gradient background */
    .stApp {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .hero-title {
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #FFD93D, #FF6B6B, #6BCB77, #4D96FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0.5rem 0;
        letter-spacing: -1px;
    }
    .hero-sub {
        text-align: center;
        color: white;
        font-size: 1.1rem;
        opacity: 0.95;
        margin-bottom: 2rem;
        font-weight: 500;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.92);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 24px;
        padding: 1.8rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.18);
        border: 1px solid rgba(255,255,255,0.4);
        margin-bottom: 1.2rem;
    }

    .pred-cat {
        background: linear-gradient(135deg, #FFB88C 0%, #DE6262 100%);
        color: white;
        border-radius: 20px;
        padding: 2rem 1.5rem;
        text-align: center;
        box-shadow: 0 8px 24px rgba(222, 98, 98, 0.4);
        animation: popIn 0.5s cubic-bezier(0.18, 0.89, 0.32, 1.28);
    }
    .pred-dog {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border-radius: 20px;
        padding: 2rem 1.5rem;
        text-align: center;
        box-shadow: 0 8px 24px rgba(0, 196, 254, 0.4);
        animation: popIn 0.5s cubic-bezier(0.18, 0.89, 0.32, 1.28);
    }
    .pred-label {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0.3rem 0;
        letter-spacing: -1px;
    }
    .pred-emoji { font-size: 5rem; line-height: 1; }
    .pred-confidence {
        font-size: 1.4rem;
        font-weight: 600;
        opacity: 0.95;
    }

    @keyframes popIn {
        0%   { opacity: 0; transform: scale(0.7); }
        100% { opacity: 1; transform: scale(1); }
    }

    .gauge-wrap {
        background: rgba(0,0,0,0.08);
        border-radius: 12px;
        height: 28px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    .gauge-fill-cat {
        background: linear-gradient(90deg, #FFB88C, #DE6262);
        height: 100%;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 10px;
        color: white;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .gauge-fill-dog {
        background: linear-gradient(90deg, #4facfe, #00f2fe);
        height: 100%;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 10px;
        color: white;
        font-weight: 700;
        font-size: 0.9rem;
    }

    .stat-pill {
        background: rgba(255,255,255,0.15);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 50px;
        display: inline-block;
        margin: 0.2rem;
        font-size: 0.85rem;
        font-weight: 600;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] * { color: white !important; }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #FFD93D !important;
    }

    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.95);
        border-radius: 16px;
        padding: 1rem;
    }

    .pipeline {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 1rem 0;
        flex-wrap: wrap;
        gap: 0.5rem;
    }
    .step {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 50%;
        width: 60px; height: 60px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        box-shadow: 0 4px 12px rgba(102,126,234,0.4);
        flex-shrink: 0;
    }
    .arrow { color: #764ba2; font-size: 1.5rem; font-weight: 800; }

    .honesty {
        background: rgba(255,236,179,0.95);
        border-left: 4px solid #F39C12;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        color: #5a4500;
        font-size: 0.92rem;
        margin-top: 1rem;
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
    st.markdown("# 🧠 About the Model")
    st.markdown("""
A 3-layer fully-connected neural network, built from scratch with manual weight matrices.

```
Input (4096)
    ↓ ReLU
Hidden (16)
    ↓ ReLU
Hidden (8)
    ↓
Output (2)
```
    """)

    st.markdown("---")
    st.markdown("# 📊 Training Stats")
    st.markdown("""
<div>
  <span class="stat-pill">🐱 100 cats</span>
  <span class="stat-pill">🐶 100 dogs</span>
  <span class="stat-pill">📐 64×64</span>
  <span class="stat-pill">⚫ Grayscale</span>
  <span class="stat-pill">📚 80/20 split</span>
  <span class="stat-pill">🧪 SGD</span>
</div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("# 🎯 Model Limitations")
    st.markdown("""
This is a **simple ANN** trained on only 160 photos. It will sometimes be wrong — especially when:

- The animal is small in the photo
- Photo is from an unusual angle
- The background is busy
- The animal is lying down / partly hidden

For higher accuracy, a CNN with thousands of photos would be needed.
    """)

    st.markdown("---")
    st.markdown("# 📚 Project Info")
    st.markdown("""
**Mini Project 4** — Intro to Machine Learning
Year 3 ITC · I3 AMS S2
    """)

# ============================================================
#  HEADER
# ============================================================
st.markdown('<h1 class="hero-title">🐾 Pawsome Predictor 🐾</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">A neural network that tries its best to tell cats from dogs ✨</p>',
            unsafe_allow_html=True)

if not model_ok:
    st.error("⚠️ **Model files not found.** Make sure `model_weights.pt` and `scaler.pkl` are next to this app.")
    st.stop()

# ============================================================
#  PIPELINE VISUALIZATION
# ============================================================
st.markdown("""
<div class="glass-card">
  <h3 style="color:#333; margin-top:0;">⚙️ How the model thinks</h3>
  <div class="pipeline">
    <div style="text-align:center;">
      <div class="step">📷</div>
      <small style="color:#555;">Your photo</small>
    </div>
    <div class="arrow">→</div>
    <div style="text-align:center;">
      <div class="step">⚫</div>
      <small style="color:#555;">Grayscale</small>
    </div>
    <div class="arrow">→</div>
    <div style="text-align:center;">
      <div class="step">📐</div>
      <small style="color:#555;">Resize 64×64</small>
    </div>
    <div class="arrow">→</div>
    <div style="text-align:center;">
      <div class="step">🔢</div>
      <small style="color:#555;">4096 numbers</small>
    </div>
    <div class="arrow">→</div>
    <div style="text-align:center;">
      <div class="step">🧠</div>
      <small style="color:#555;">Neural net</small>
    </div>
    <div class="arrow">→</div>
    <div style="text-align:center;">
      <div class="step">🎯</div>
      <small style="color:#555;">Cat or Dog</small>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
#  UPLOAD
# ============================================================
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("### 📤 Upload a photo")
uploaded = st.file_uploader("upload", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

if uploaded is not None:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img_color = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
    img_gray  = cv.imdecode(file_bytes, cv.IMREAD_GRAYSCALE)

    if img_color is None or img_gray is None:
        st.error("Couldn't read that file. Try a different photo.")
    else:
        with st.spinner("🧠 Neural network is thinking..."):
            time.sleep(0.6)

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

        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 📸 Your photo")
            st.image(cv.cvtColor(img_color, cv.COLOR_BGR2RGB), use_container_width=True)
            st.caption(f"Size: {img_color.shape[1]}×{img_color.shape[0]} pixels")
            st.markdown("**👁️ What the model actually sees:**")
            st.image(img_resized, width=160, clamp=True)
            st.caption("Tiny 64×64 grayscale — a lot of detail is lost!")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            # ----- Detect "probably not an animal" -----
            # When the model is very unsure (close to 50/50), the photo
            # likely isn't a cat or a dog at all.
            UNKNOWN_THRESHOLD = 60.0   # confidence below this = "unknown"
            is_unknown = confidence < UNKNOWN_THRESHOLD

            if is_unknown:
                # Show "Not an animal" card
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #757F9A 0%, #D7DDE8 100%);
                            color: white; border-radius: 20px; padding: 2rem 1.5rem;
                            text-align: center;
                            box-shadow: 0 8px 24px rgba(117, 127, 154, 0.4);
                            animation: popIn 0.5s cubic-bezier(0.18, 0.89, 0.32, 1.28);">
                    <div class="pred-emoji">🤷</div>
                    <div class="pred-label" style="font-size:2.5rem;">Not Sure</div>
                    <div class="pred-confidence">This doesn't look like a cat or a dog</div>
                    <div style="margin-top:0.5rem; opacity:0.9;">
                        The model is too unsure ({confidence:.1f}%) — try an actual cat or dog photo
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                is_cat = pred == 0
                card_class = "pred-cat" if is_cat else "pred-dog"
                emoji = "🐱" if is_cat else "🐶"
                label = "CAT" if is_cat else "DOG"

                if confidence >= 75:
                    vibe = "✨ Pretty confident!"
                else:
                    vibe = "🤔 Fairly sure..."

                st.markdown(f"""
                <div class="{card_class}">
                    <div class="pred-emoji">{emoji}</div>
                    <div class="pred-label">{label}</div>
                    <div class="pred-confidence">{confidence:.1f}% confident</div>
                    <div style="margin-top:0.5rem; opacity:0.9;">{vibe}</div>
                </div>
                """, unsafe_allow_html=True)

            # ----- Probability breakdown (always show) -----
            st.markdown('<div class="glass-card" style="margin-top:1rem;">', unsafe_allow_html=True)
            st.markdown("#### 📊 Probability Breakdown")
            st.markdown(f"""
            <div style="margin-bottom: 0.3rem; color:#333;">🐱 Cat</div>
            <div class="gauge-wrap">
                <div class="gauge-fill-cat" style="width: {cat_p}%;">{cat_p:.1f}%</div>
            </div>
            <div style="margin: 0.8rem 0 0.3rem 0; color:#333;">🐶 Dog</div>
            <div class="gauge-wrap">
                <div class="gauge-fill-dog" style="width: {dog_p}%;">{dog_p:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            if is_unknown:
                st.markdown("""
                <div class="honesty">
                    🔍 <b>Why "Not Sure"?</b> The model was trained only on cats and dogs.
                    When it sees something else (a person, object, or other animal), the probabilities
                    end up close to 50/50 — that's the signal we use to say "this doesn't look like a cat or a dog."
                </div>
                """, unsafe_allow_html=True)
            elif confidence >= 80:
                st.markdown("""
                <div class="honesty" style="background:rgba(200,230,201,0.95); border-left-color:#27AE60; color:#1a4d1f;">
                    💡 <b>Heads up:</b> high confidence doesn't always mean correct!
                    This simple model can be confidently wrong sometimes.
                </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("🔬 See the technical details"):
            colA, colB, colC = st.columns(3)
            with colA:
                st.metric("Cat Probability", f"{cat_p:.2f}%")
            with colB:
                st.metric("Dog Probability", f"{dog_p:.2f}%")
            with colC:
                st.metric("Confidence Margin", f"{abs(cat_p - dog_p):.2f}%")
            st.markdown(f"""
            **Logits (raw model output):**
            - Cat: `{Z[0, 0].item():.4f}`
            - Dog: `{Z[0, 1].item():.4f}`

            **Preprocessing pipeline:** original photo → grayscale → resized to 64×64 → flattened to 4096-D vector → divided by 255 → standardized → fed to the network.
            """)

# ============================================================
#  FOOTER
# ============================================================
st.markdown("""
<div style="text-align:center; margin-top:2rem; color:white; opacity:0.85; font-size:0.9rem;">
    Made with 💖 using PyTorch & Streamlit · Mini Project 4
</div>
""", unsafe_allow_html=True)
