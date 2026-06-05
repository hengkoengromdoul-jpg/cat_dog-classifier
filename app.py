"""
Cat vs Dog Web Predictor — "Neural Showdown" edition
Mini Project 4 — Intro to Machine Learning
Run with: streamlit run app.py
"""
import streamlit as st
import torch
import numpy as np
import cv2 as cv
import pickle

# ====================================================================
# PAGE SETUP
# ====================================================================
st.set_page_config(
    page_title="Cat vs Dog · Neural Showdown",
    page_icon="🐾",
    layout="centered",
)

# ====================================================================
# GLOBAL STYLING  (one big CSS block — not an f-string, so braces are safe)
# ====================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@400;500;600;700&family=Nunito:wght@400;600;700;800;900&display=swap');

:root{
  --cat-1:#ff7a45;  --cat-2:#ff9e64;  --cat-glow:rgba(255,122,69,.45);
  --dog-1:#5b7cfa;  --dog-2:#8aa2ff;  --dog-glow:rgba(91,124,250,.45);
  --ink:#23263a;    --muted:#6b7088;  --card:#ffffff;
  --line:#ececf4;
}

/* ---- page background: warm (cat) sweeping into cool (dog) ---- */
.stApp{
  background:
    radial-gradient(900px 500px at 12% -5%, rgba(255,122,69,.10), transparent 60%),
    radial-gradient(900px 500px at 100% 0%, rgba(91,124,250,.12), transparent 55%),
    linear-gradient(160deg,#fff6ef 0%, #f4f5ff 100%);
  background-attachment: fixed;
}
html, body, [class*="css"]{ font-family:'Nunito',sans-serif; color:var(--ink); }

/* hide default chrome for a cleaner canvas */
[data-testid="stHeader"]{ background:transparent; }
#MainMenu, footer{ visibility:hidden; }
.block-container{ padding-top:2rem; padding-bottom:4rem; max-width:760px; }

/* ============ HERO ============ */
.hero{
  position:relative; text-align:center; padding:38px 26px 32px;
  border-radius:30px; overflow:hidden;
  background:linear-gradient(135deg,#171a2b 0%, #232845 60%, #2b2150 100%);
  box-shadow:0 24px 60px -22px rgba(43,33,80,.55), inset 0 1px 0 rgba(255,255,255,.06);
}
.hero::before{ /* soft colored glow blobs */
  content:""; position:absolute; inset:0;
  background:
    radial-gradient(280px 180px at 18% 20%, var(--cat-glow), transparent 70%),
    radial-gradient(280px 180px at 82% 80%, var(--dog-glow), transparent 70%);
  pointer-events:none;
}
.hero-emojis{ position:relative; font-size:46px; letter-spacing:6px; }
.hero-emojis .cat{ display:inline-block; animation:bobL 3s ease-in-out infinite; }
.hero-emojis .dog{ display:inline-block; animation:bobR 3s ease-in-out infinite; }
.hero-emojis .vs{
  font-family:'Fredoka'; font-size:20px; font-weight:700; letter-spacing:1px;
  color:#fff; background:rgba(255,255,255,.12); padding:5px 12px; border-radius:999px;
  margin:0 14px; vertical-align:middle; border:1px solid rgba(255,255,255,.18);
}
.hero h1{
  position:relative; font-family:'Fredoka',sans-serif; font-weight:700;
  font-size:54px; line-height:1.05; margin:14px 0 6px;
  background:linear-gradient(90deg,#ff8a5b 0%, #ff6a8e 48%, #7d93ff 100%);
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
}
.hero p.sub{ position:relative; color:#c4c8e0; font-size:17px; margin:0 auto; max-width:430px; }
.hero .badge{
  position:relative; display:inline-block; margin-top:18px; font-size:12.5px; font-weight:700;
  letter-spacing:.4px; color:#dfe3ff; background:rgba(255,255,255,.08);
  border:1px solid rgba(255,255,255,.16); padding:7px 15px; border-radius:999px;
}

/* ============ NOTICE BANNER ============ */
.notice{
  display:flex; gap:14px; align-items:flex-start; margin:22px 0 8px;
  background:linear-gradient(135deg,#fff8f2,#f3f6ff);
  border:1px solid var(--line); border-left:5px solid #ffb37a;
  border-radius:18px; padding:16px 18px;
  box-shadow:0 10px 26px -18px rgba(43,33,80,.4);
}
.notice .ico{ font-size:22px; line-height:1.2; }
.notice .txt{ font-size:14.5px; color:#4b4f66; line-height:1.5; }
.notice .txt b{ color:var(--ink); }

/* ============ SECTION LABEL ============ */
.seclabel{
  font-family:'Fredoka'; font-weight:600; font-size:13px; letter-spacing:1.5px;
  text-transform:uppercase; color:var(--muted); margin:26px 0 10px;
}

/* ============ FILE UPLOADER ============ */
[data-testid="stFileUploader"]{
  background:var(--card); border:2px dashed #d8dcf0; border-radius:20px;
  padding:8px 14px; transition:border-color .2s, box-shadow .2s;
}
[data-testid="stFileUploader"]:hover{
  border-color:var(--dog-1); box-shadow:0 12px 30px -20px var(--dog-glow);
}
[data-testid="stFileUploaderDropzone"]{ background:transparent; }

/* ============ CARDS ============ */
.card{
  background:var(--card); border:1px solid var(--line); border-radius:22px;
  padding:18px; box-shadow:0 18px 40px -28px rgba(43,33,80,.45);
  height:100%;
}
.card h3{ font-family:'Fredoka'; font-weight:600; font-size:17px; margin:0 0 12px; color:var(--ink); }
.card .meta{ font-size:12.5px; color:var(--muted); margin-top:8px; }
.stImage img{ border-radius:14px; }

/* ============ WINNER REVEAL BANNER ============ */
.reveal{
  position:relative; text-align:center; border-radius:24px; padding:26px 22px;
  margin-bottom:8px; overflow:hidden; color:#fff;
  animation:popIn .55s cubic-bezier(.18,.89,.32,1.28) both;
}
.reveal.cat{ background:linear-gradient(135deg,var(--cat-1),var(--cat-2)); box-shadow:0 22px 46px -22px var(--cat-glow); }
.reveal.dog{ background:linear-gradient(135deg,var(--dog-1),var(--dog-2)); box-shadow:0 22px 46px -22px var(--dog-glow); }
.reveal .crown{ font-size:18px; letter-spacing:2px; font-weight:800; opacity:.9; text-transform:uppercase; }
.reveal .big{ font-size:62px; line-height:1; margin:6px 0; }
.reveal .name{ font-family:'Fredoka'; font-weight:700; font-size:34px; }
.reveal .conf{
  display:inline-block; margin-top:12px; font-weight:800; font-size:13.5px; letter-spacing:.4px;
  background:rgba(255,255,255,.22); border:1px solid rgba(255,255,255,.35);
  padding:7px 16px; border-radius:999px; backdrop-filter:blur(4px);
}

/* ============ THE SIGNATURE BATTLE BAR ============ */
.battle-head{ display:flex; justify-content:space-between; font-weight:800; font-size:14px; margin:4px 2px 10px; }
.battle-head .l{ color:var(--cat-1); } .battle-head .r{ color:var(--dog-1); }
.battle-track{
  position:relative; height:60px; border-radius:999px; overflow:hidden;
  background:#eef0fa; box-shadow:inset 0 2px 6px rgba(43,33,80,.12);
  border:1px solid var(--line);
}
.battle-track .fill{
  position:absolute; top:0; height:100%; display:flex; align-items:center;
  color:#fff; font-weight:900; font-size:16px; white-space:nowrap; overflow:hidden;
}
.battle-track .cat-fill{
  left:0; justify-content:flex-start; padding-left:18px;
  background:linear-gradient(90deg,var(--cat-1),var(--cat-2));
  animation:growCat 1s cubic-bezier(.22,.61,.36,1) forwards;
}
.battle-track .dog-fill{
  right:0; justify-content:flex-end; padding-right:18px;
  background:linear-gradient(270deg,var(--dog-1),var(--dog-2));
  animation:growDog 1s cubic-bezier(.22,.61,.36,1) forwards;
}
.battle-track .seam{
  position:absolute; top:-4px; height:68px; width:3px; left:50%; transform:translateX(-50%);
  background:rgba(255,255,255,.85); box-shadow:0 0 8px rgba(255,255,255,.7); z-index:3;
}
.tug{ text-align:center; font-size:13px; color:var(--muted); margin-top:12px; font-weight:700; }

/* ============ EXPANDER ============ */
[data-testid="stExpander"]{ border:1px solid var(--line); border-radius:16px; background:var(--card); }
[data-testid="stExpander"] summary{ font-weight:700; color:var(--ink); }

/* ============ THINKING DOTS (idle) ============ */
.empty{
  text-align:center; padding:42px 20px; color:var(--muted);
  border:2px dashed #dde1f2; border-radius:22px; background:rgba(255,255,255,.5);
}
.empty .pulse{ font-size:40px; animation:bob 2.4s ease-in-out infinite; }

/* ============ KEYFRAMES ============ */
@keyframes growCat{ from{ width:0; } to{ width:var(--w); } }
@keyframes growDog{ from{ width:0; } to{ width:var(--w); } }
@keyframes bob{ 0%,100%{ transform:translateY(0); } 50%{ transform:translateY(-8px); } }
@keyframes bobL{ 0%,100%{ transform:translateY(0) rotate(-6deg);} 50%{ transform:translateY(-7px) rotate(-2deg);} }
@keyframes bobR{ 0%,100%{ transform:translateY(0) rotate(6deg);} 50%{ transform:translateY(-7px) rotate(2deg);} }
@keyframes popIn{ 0%{ opacity:0; transform:scale(.86) translateY(14px);} 100%{ opacity:1; transform:scale(1) translateY(0);} }
@keyframes fadeUp{ 0%{ opacity:0; transform:translateY(16px);} 100%{ opacity:1; transform:translateY(0);} }
.fade-up{ animation:fadeUp .5s ease-out both; }
</style>
""", unsafe_allow_html=True)

# ====================================================================
# HERO
# ====================================================================
st.markdown("""
<div class="hero">
  <div class="hero-emojis"><span class="cat">🐱</span><span class="vs">VS</span><span class="dog">🐶</span></div>
  <h1>Cat vs Dog</h1>
  <p class="sub">Upload a photo and watch the neural network duke it out to decide the winner.</p>
  <div class="badge">⚡ MINI PROJECT 4 · 3-LAYER ANN BUILT FROM SCRATCH</div>
</div>
""", unsafe_allow_html=True)

# ====================================================================
# IMPORTANT NOTICE (kept — it's pedagogically important)
# ====================================================================
st.markdown("""
<div class="notice">
  <div class="ico">📌</div>
  <div class="txt">
    <b>Two contestants only.</b> This model was trained on just <b>cats</b> and <b>dogs</b>.
    Feed it anything else — a person, an object, a screenshot — and it will <i>still</i> declare a
    Cat or a Dog, because those are the only two answers it knows. For real results, upload a clear
    photo of a cat or a dog.
  </div>
</div>
""", unsafe_allow_html=True)

# ====================================================================
# MODEL  (logic identical to the notebook — do not change behavior)
# ====================================================================
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
    W1 = weights["W1"]; b1 = weights["b1"]
    W2 = weights["W2"]; b2 = weights["b2"]
    W3 = weights["W3"]; b3 = weights["b3"]
    model_ok = True
except FileNotFoundError:
    st.error("⚠️ Model files not found. Run the **Save Model** cell in your notebook "
             "to create `model_weights.pt` and `scaler.pkl`, then put them next to this app.py.")
    model_ok = False

# ====================================================================
# UPLOADER
# ====================================================================
st.markdown('<div class="seclabel">① Enter a contestant</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Drop a photo of a cat or a dog", type=["jpg", "jpeg", "png"],
                            label_visibility="collapsed")

if uploaded is not None and model_ok:
    file_bytes = np.frombuffer(uploaded.read(), np.uint8)
    img_color = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
    img_gray  = cv.imdecode(file_bytes, cv.IMREAD_GRAYSCALE)

    if img_color is None or img_gray is None:
        st.error("Could not read the image. Try another file.")
    else:
        # ----- Preprocess EXACTLY like training -----
        img_resized = cv.resize(img_gray, (64, 64))
        x = img_resized.flatten() / 255.0
        x = scaler.transform(x.reshape(1, -1))
        x_tensor = torch.tensor(x, dtype=torch.float32)

        with st.spinner("🤔 The network is sizing up the contestant..."):
            with torch.no_grad():
                Z = my_ANN(x_tensor, W1, b1, W2, b2, W3, b3)
                probs = torch.softmax(Z, dim=1)
                pred = torch.argmax(Z, dim=1).item()

        names      = ["Cat", "Dog"]
        emojis     = ["🐱", "🐶"]
        confidence = probs[0, pred].item() * 100
        cat_p      = probs[0, 0].item() * 100
        dog_p      = probs[0, 1].item() * 100
        winner_cls = "cat" if pred == 0 else "dog"

        # confidence wording
        if confidence >= 75:
            conf_txt = f"💪 Confident · {confidence:.1f}%"
            st.balloons()
        elif confidence >= 60:
            conf_txt = f"🙂 Fairly sure · {confidence:.1f}%"
        else:
            conf_txt = f"🤷 Just guessing · {confidence:.1f}%"

        # ---------------- WINNER REVEAL ----------------
        st.markdown('<div class="seclabel">② And the network says…</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="reveal {winner_cls}">
          <div class="crown">🏆 Winner</div>
          <div class="big">{emojis[pred]}</div>
          <div class="name">It's a {names[pred]}!</div>
          <div class="conf">{conf_txt}</div>
        </div>
        """, unsafe_allow_html=True)

        # ---------------- THE BATTLE BAR ----------------
        st.markdown(f"""
        <div class="card fade-up" style="margin-top:14px;">
          <h3>⚔️ The Showdown</h3>
          <div class="battle-head"><span class="l">🐱 Cat {cat_p:.1f}%</span><span class="r">{dog_p:.1f}% Dog 🐶</span></div>
          <div class="battle-track">
            <div class="fill cat-fill" style="--w:{cat_p:.2f}%;">{('🐱 ' + format(cat_p, '.0f') + '%') if cat_p >= 18 else ''}</div>
            <div class="fill dog-fill" style="--w:{dog_p:.2f}%;">{(format(dog_p, '.0f') + '% 🐶') if dog_p >= 18 else ''}</div>
            <div class="seam"></div>
          </div>
          <div class="tug">{'🐱 Team Cat pulled ahead!' if pred == 0 else '🐶 Team Dog pulled ahead!'}</div>
        </div>
        """, unsafe_allow_html=True)

        # ---------------- PHOTO + WHAT MODEL SEES ----------------
        st.markdown('<div class="seclabel">③ The evidence</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="card"><h3>📷 Your photo</h3>', unsafe_allow_html=True)
            st.image(cv.cvtColor(img_color, cv.COLOR_BGR2RGB), use_container_width=True)
            st.markdown(f'<div class="meta">Original size: {img_color.shape[1]}×{img_color.shape[0]} px</div></div>',
                        unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card"><h3>🤖 What the network sees</h3>', unsafe_allow_html=True)
            st.image(img_resized, use_container_width=True, clamp=True)
            st.markdown('<div class="meta">64×64 grayscale — the tiny version the ANN actually reads. '
                        'It judges fur, ears, and snout from <i>this</i>, not your full photo.</div></div>',
                        unsafe_allow_html=True)

        # ---------------- REMINDER ----------------
        st.markdown("""
        <p style="text-align:center; color:#8a8fa6; font-size:13px; margin-top:18px;">
          💡 High confidence isn't the same as correct — for non-animal photos the network still
          has to crown a Cat or a Dog.
        </p>
        """, unsafe_allow_html=True)

elif not model_ok:
    st.stop()
else:
    st.markdown("""
    <div class="empty">
      <div class="pulse">🐾</div>
      <p style="margin:10px 0 0; font-weight:700;">Upload a photo above to start the showdown.</p>
    </div>
    """, unsafe_allow_html=True)
