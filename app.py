import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import time

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="K-Pop Scandal Impact Predictor",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def load_model():
    return joblib.load("model_base.joblib")

@st.cache_data
def load_config():
    with open("pipeline_config.json") as f:
        return json.load(f)

model = load_model()
config = load_config()

# ─────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────────────────────

ARTISTS = [
    "BTS", "BLACKPINK", "EXO", "TWICE", "Stray Kids", "aespa",
    "Red Velvet", "NCT", "SEVENTEEN", "GOT7", "BIGBANG", "2NE1",
    "SHINee", "MAMAMOO", "(G)I-DLE", "IVE", "LE SSERAFIM", "ENHYPEN",
    "TXT", "ITZY", "MONSTA X", "ATEEZ", "iKON", "WINNER",
    "f(x)", "T-ara", "Super Junior", "TVXQ", "SNSD", "B.A.P",
    "VIXX", "Block B", "Pentagon", "BTOB", "LOONA", "FIFTY FIFTY",
    "NewJeans", "NMIXX", "Kep1er", "TREASURE", "OMEGA X", "4MINUTE"
]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Space+Mono:wght@400;700&display=swap');

/* ── GLOBAL RESET ── */
.stApp {
    background: #0a0a0f;
    color: #e0e0e0;
    font-family: 'Outfit', sans-serif;
}

.block-container {
    padding-top: 1rem !important;
    max-width: 1200px;
}

header[data-testid="stHeader"] { background: transparent; }
div[data-testid="stToolbar"] { display: none; }

/* ── HIDE STREAMLIT BRANDING ── */
#MainMenu, footer, .stDeployButton { display: none !important; }

/* ── LANDING PAGE ── */
.landing-container {
    text-align: center;
    padding: 2rem 1rem 0 1rem;
    position: relative;
    overflow: hidden;
}

.main-title {
    font-family: 'Outfit', sans-serif;
    font-weight: 900;
    font-size: clamp(2.5rem, 7vw, 5.5rem);
    letter-spacing: -0.02em;
    line-height: 1.05;
    background: linear-gradient(135deg, #ff2d7b, #ff6b9d, #c850ff, #7b61ff, #ff2d7b);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gradientShift 4s ease infinite;
    margin-bottom: 0.3rem;
    text-transform: uppercase;
}

.sub-title {
    font-family: 'Space Mono', monospace;
    font-size: clamp(0.75rem, 2vw, 1rem);
    color: #888;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

@keyframes gradientShift {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

/* ── ARTIST GRID ── */
.artist-cloud {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 0.5rem;
    max-width: 900px;
    margin: 0 auto 2.5rem auto;
    padding: 0 1rem;
}

.artist-tag {
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    font-size: 0.8rem;
    padding: 0.35rem 0.85rem;
    border-radius: 100px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    background: rgba(255, 255, 255, 0.03);
    color: #666;
    transition: all 0.4s ease;
    animation: tagFadeIn 0.6s ease backwards;
}

.artist-tag:nth-child(3n+1) { color: #ff2d7b55; border-color: #ff2d7b22; }
.artist-tag:nth-child(3n+2) { color: #c850ff55; border-color: #c850ff22; }
.artist-tag:nth-child(3n+3) { color: #7b61ff55; border-color: #7b61ff22; }

@keyframes tagFadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ── START BUTTON ── */
.start-btn-wrap {
    display: flex;
    justify-content: center;
    margin: 1rem 0 2rem 0;
}

/* ── STREAMLIT BUTTON OVERRIDES ── */
div.stButton > button {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 100px !important;
    transition: all 0.3s ease !important;
}

div.stButton > button[kind="primary"],
div.stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #ff2d7b, #c850ff) !important;
    color: #fff !important;
    font-size: 1.1rem !important;
    padding: 0.7rem 3rem !important;
    box-shadow: 0 4px 25px rgba(255, 45, 123, 0.3) !important;
}

div.stButton > button[kind="primary"]:hover,
div.stButton > button[data-testid="stBaseButton-primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 35px rgba(255, 45, 123, 0.5) !important;
}

div.stButton > button[kind="secondary"],
div.stButton > button[data-testid="stBaseButton-secondary"] {
    background: rgba(255,255,255,0.06) !important;
    color: #ccc !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 2rem !important;
}

div.stButton > button[kind="secondary"]:hover,
div.stButton > button[data-testid="stBaseButton-secondary"]:hover {
    background: rgba(255,255,255,0.1) !important;
    color: #fff !important;
}

/* ── FORM PAGE ── */
.form-header {
    font-family: 'Outfit', sans-serif;
    font-weight: 800;
    font-size: 2rem;
    text-align: center;
    margin-bottom: 0.3rem;
    background: linear-gradient(135deg, #ff6b9d, #c850ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.form-subheader {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: #666;
    text-align: center;
    letter-spacing: 0.08em;
    margin-bottom: 2rem;
}

.section-label {
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #ff6b9d;
    margin: 1.5rem 0 0.8rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid rgba(255, 107, 157, 0.15);
}

/* Style select boxes and sliders */
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    color: #bbb !important;
}

div[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.04) !important;
    border-color: rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
}

/* Toggle styling */
div[data-testid="stCheckbox"] label span {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 500 !important;
    color: #bbb !important;
}

/* ── RESULT PAGE ── */
.result-container {
    text-align: center;
    padding: 2rem 1rem;
}

.result-verdict {
    font-family: 'Outfit', sans-serif;
    font-weight: 900;
    font-size: clamp(2rem, 5vw, 3.5rem);
    letter-spacing: -0.02em;
    margin-bottom: 0.3rem;
}

.verdict-high {
    color: #ff3b5c;
    text-shadow: 0 0 40px rgba(255, 59, 92, 0.4);
}

.verdict-manageable {
    color: #00e676;
    text-shadow: 0 0 40px rgba(0, 230, 118, 0.4);
}

.result-subtitle {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: #888;
    margin-bottom: 2rem;
}

/* Probability meter */
.meter-container {
    max-width: 500px;
    margin: 0 auto 2rem auto;
}

.meter-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 0.4rem;
}

.meter-bar-bg {
    width: 100%;
    height: 12px;
    background: rgba(255,255,255,0.06);
    border-radius: 100px;
    overflow: hidden;
    position: relative;
}

.meter-bar-fill {
    height: 100%;
    border-radius: 100px;
    transition: width 1.5s cubic-bezier(0.25, 0.46, 0.45, 0.94);
}

.meter-bar-fill-high {
    background: linear-gradient(90deg, #ff9800, #ff3b5c);
    box-shadow: 0 0 15px rgba(255, 59, 92, 0.4);
}

.meter-bar-fill-low {
    background: linear-gradient(90deg, #00e676, #69f0ae);
    box-shadow: 0 0 15px rgba(0, 230, 118, 0.3);
}

.meter-value {
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    font-size: 1.8rem;
    margin-top: 0.5rem;
}

/* Factor cards */
.factors-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 0.8rem;
    max-width: 700px;
    margin: 1.5rem auto;
}

.factor-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1rem;
    text-align: left;
}

.factor-name {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 0.3rem;
}

.factor-value {
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    font-size: 1rem;
}

.factor-raising { color: #ff6b6b; }
.factor-lowering { color: #69f0ae; }
.factor-neutral { color: #888; }

/* ── ANIMATIONS ── */
@keyframes pulseRed {
    0%, 100% { box-shadow: 0 0 20px rgba(255, 59, 92, 0.2); }
    50% { box-shadow: 0 0 40px rgba(255, 59, 92, 0.4); }
}

@keyframes pulseGreen {
    0%, 100% { box-shadow: 0 0 20px rgba(0, 230, 118, 0.15); }
    50% { box-shadow: 0 0 40px rgba(0, 230, 118, 0.3); }
}

.result-icon-circle {
    width: 120px;
    height: 120px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 1.5rem auto;
    font-size: 3rem;
    animation-duration: 2s;
    animation-iteration-count: infinite;
}

.icon-high {
    background: rgba(255, 59, 92, 0.1);
    border: 2px solid rgba(255, 59, 92, 0.3);
    animation-name: pulseRed;
}

.icon-manageable {
    background: rgba(0, 230, 118, 0.08);
    border: 2px solid rgba(0, 230, 118, 0.25);
    animation-name: pulseGreen;
}

/* Confetti for manageable outcome */
.confetti-container {
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    pointer-events: none;
    z-index: 9999;
    overflow: hidden;
}

.confetti-piece {
    position: absolute;
    width: 8px;
    height: 8px;
    top: -10px;
    border-radius: 2px;
    animation: confettiFall linear forwards;
}

@keyframes confettiFall {
    0% { transform: translateY(0) rotate(0deg); opacity: 1; }
    100% { transform: translateY(100vh) rotate(720deg); opacity: 0; }
}

/* Warning flash for high crisis */
@keyframes warningFlash {
    0% { opacity: 0; }
    10% { opacity: 0.08; }
    20% { opacity: 0; }
    30% { opacity: 0.05; }
    40% { opacity: 0; }
    100% { opacity: 0; }
}

.warning-overlay {
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background: #ff3b5c;
    pointer-events: none;
    z-index: 9999;
    animation: warningFlash 2s ease-out forwards;
}

/* ── MODEL INFO ── */
.model-info-bar {
    max-width: 700px;
    margin: 2rem auto 0 auto;
    padding: 1rem 1.5rem;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #555;
    text-align: center;
    line-height: 1.7;
}

/* ── DIVIDER ── */
.styled-divider {
    width: 60px;
    height: 3px;
    background: linear-gradient(90deg, #ff2d7b, #c850ff);
    border-radius: 100px;
    margin: 0 auto 2rem auto;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "landing"
if "prediction" not in st.session_state:
    st.session_state.prediction = None


# ─────────────────────────────────────────────────────────────
# PREDICTION LOGIC
# ─────────────────────────────────────────────────────────────
def predict_crisis(inputs: dict) -> dict:
    feature_names = config["feature_columns"]
    row = pd.DataFrame([inputs])
    row_encoded = pd.get_dummies(row, columns=["scandal_type"], prefix="type")
    for col in feature_names:
        if col not in row_encoded.columns:
            row_encoded[col] = 0
    row_encoded = row_encoded[feature_names]

    prob = model.predict_proba(row_encoded)[0][1]
    pred = "high" if prob > 0.5 else "manageable"

    # Compute factor contributions (simplified: compare to baseline averages)
    importances = config["feature_importances"]
    top_features = sorted(importances.items(), key=lambda x: -x[1])[:5]
    factors = []
    for feat, imp in top_features:
        val = row_encoded[feat].values[0]
        if feat == "reaction_spike":
            if val > 3:
                factors.append(("Public Reaction Spike", f"+{val:.1f} above baseline", "raising"))
            elif val < 0:
                factors.append(("Public Reaction Spike", f"{val:.1f} (below baseline)", "lowering"))
            else:
                factors.append(("Public Reaction Spike", f"{val:.1f}", "neutral"))
        elif feat == "fandom_size_num":
            labels = {2: "Medium", 3: "Large", 4: "Mega"}
            if val >= 4:
                factors.append(("Fandom Size", f"{labels.get(int(val), val)} — protective", "lowering"))
            elif val <= 2:
                factors.append(("Fandom Size", f"{labels.get(int(val), val)} — vulnerable", "raising"))
            else:
                factors.append(("Fandom Size", labels.get(int(val), str(val)), "neutral"))
        elif feat == "response_delay_days":
            val_days = inputs.get("response_delay_days", val)
            if val_days >= 5:
                factors.append(("Response Delay", f"{int(val_days)} days — slow", "raising"))
            elif val_days <= 1:
                factors.append(("Response Delay", f"{int(val_days)} day(s) — fast", "lowering"))
            else:
                factors.append(("Response Delay", f"{int(val_days)} days", "neutral"))
        elif feat == "type_criminal" and val == 1:
            factors.append(("Scandal Type", "Criminal — highest risk", "raising"))
        elif feat == "company_response":
            resp_labels = {0: "Silence — risky", 1: "Partial", 2: "Strong defense"}
            direction = "raising" if val == 0 else ("lowering" if val == 2 else "neutral")
            factors.append(("Company Response", resp_labels.get(int(val), str(val)), direction))

    return {"prob": prob, "pred": pred, "factors": factors}


# ─────────────────────────────────────────────────────────────
# PAGES
# ─────────────────────────────────────────────────────────────

def render_landing():
    st.markdown('<div class="landing-container">', unsafe_allow_html=True)

    st.markdown('<div class="main-title">Scandal Impact<br>Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Machine learning meets K-Pop crisis management</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    # Artist cloud
    tags = "".join(
        f'<span class="artist-tag" style="animation-delay: {i * 0.03:.2f}s">{a}</span>'
        for i, a in enumerate(ARTISTS)
    )
    st.markdown(f'<div class="artist-cloud">{tags}</div>', unsafe_allow_html=True)

    # Start button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Start Prediction", type="primary", use_container_width=True):
            st.session_state.page = "form"
            st.rerun()

    # Model info
    m = config["model_metrics"]
    st.markdown(
        f"""<div class="model-info-bar">
        Random Forest · {m['n_samples']} scandals analyzed · {m['accuracy']*100:.1f}% accuracy · AUC {m['auc']:.2f}<br>
        Trained on real K-pop scandal data from 2006–2024
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)


def render_form():
    st.markdown('<div class="form-header">Configure the Scandal</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="form-subheader">Set the parameters and let the model predict the outcome</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    col_left, col_spacer, col_right = st.columns([5, 1, 5])

    with col_left:
        st.markdown('<div class="section-label">🎤 The Scandal</div>', unsafe_allow_html=True)

        scandal_type = st.selectbox(
            "Type of Scandal",
            options=config["scandal_types"],
            format_func=lambda x: x.title(),
        )

        reaction_spike = st.slider(
            "Public Reaction Spike (Google Trends)",
            min_value=-6.0, max_value=14.0, value=1.0, step=0.5,
            help="Normalized search spike relative to the artist's baseline. "
                 "Negative = barely registered. Above 5 = massive."
        )

        international = st.selectbox(
            "International Attention?",
            options=[1, 0],
            format_func=lambda x: "Yes — global media pickup" if x == 1 else "No — Korea-only",
        )

        is_solo = st.selectbox(
            "Artist Type",
            options=[0, 1],
            format_func=lambda x: "Solo artist" if x == 1 else "Group member",
        )

        prior_scandal = st.selectbox(
            "Prior Scandal on Record?",
            options=[0, 1],
            format_func=lambda x: "Yes — repeat offender" if x == 1 else "No — first incident",
        )

    with col_right:
        st.markdown('<div class="section-label">🏢 The Response</div>', unsafe_allow_html=True)

        fandom_size = st.selectbox(
            "Fandom Size",
            options=[2, 3, 4],
            format_func=lambda x: {2: "Medium", 3: "Large", 4: "Mega (BTS/BP level)"}.get(x),
        )

        agency_tier = st.selectbox(
            "Agency Tier",
            options=[1, 2, 3],
            format_func=lambda x: {
                1: "Small / Indie", 2: "Mid-tier", 3: "Big3/Big4 (SM, YG, JYP, HYBE)"
            }.get(x),
        )

        company_response = st.selectbox(
            "Company Response Strength",
            options=[0, 1, 2],
            format_func=lambda x: {0: "Silence", 1: "Partial / vague statement", 2: "Strong defense"}.get(x),
        )

        response_delay = st.slider(
            "Response Delay (days)",
            min_value=0, max_value=30, value=1,
            help="0 = same-day response. Higher = slower reaction."
        )

        apology = st.selectbox(
            "Public Apology Issued?",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    col_b1, col_b2, col_b3 = st.columns([2, 1, 2])
    with col_b1:
        if st.button("← Back", type="secondary"):
            st.session_state.page = "landing"
            st.rerun()
    with col_b3:
        if st.button("Predict Outcome →", type="primary", use_container_width=True):
            inputs = {
                "scandal_type": scandal_type,
                "fandom_size_num": fandom_size,
                "agency_tier": agency_tier,
                "company_response": company_response,
                "response_delay_days": response_delay,
                "apology": apology,
                "international": international,
                "is_solo": is_solo,
                "prior_scandal": prior_scandal,
                "reaction_spike": reaction_spike,
            }
            st.session_state.prediction = predict_crisis(inputs)
            st.session_state.page = "result"
            st.rerun()


def render_result():
    r = st.session_state.prediction
    if r is None:
        st.session_state.page = "landing"
        st.rerun()
        return

    prob = r["prob"]
    is_high = r["pred"] == "high"
    prob_display = prob if is_high else 1 - prob

    # Animation overlays
    if is_high:
        st.markdown('<div class="warning-overlay"></div>', unsafe_allow_html=True)
    else:
        confetti_colors = ["#00e676", "#69f0ae", "#b9f6ca", "#ff6b9d", "#c850ff",
                           "#7b61ff", "#fff176", "#ffd54f"]
        pieces = ""
        for i in range(60):
            left = np.random.randint(0, 100)
            delay = np.random.uniform(0, 2)
            dur = np.random.uniform(2, 4.5)
            color = confetti_colors[i % len(confetti_colors)]
            size = np.random.randint(6, 12)
            pieces += (
                f'<div class="confetti-piece" style="'
                f"left:{left}%; "
                f"width:{size}px; height:{size}px; "
                f"background:{color}; "
                f"animation-delay:{delay:.1f}s; "
                f"animation-duration:{dur:.1f}s;"
                f'"></div>'
            )
        st.markdown(f'<div class="confetti-container">{pieces}</div>', unsafe_allow_html=True)

    st.markdown('<div class="result-container">', unsafe_allow_html=True)

    # Icon
    if is_high:
        st.markdown(
            '<div class="result-icon-circle icon-high">⚠️</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="result-verdict verdict-high">HIGH CRISIS</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="result-subtitle">Career-altering damage likely — group departure, termination, or criminal charges</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="result-icon-circle icon-manageable">✓</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="result-verdict verdict-manageable">MANAGEABLE</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="result-subtitle">Recovery expected within weeks to months — career survives</div>',
            unsafe_allow_html=True,
        )

    # Probability meter
    fill_class = "meter-bar-fill-high" if is_high else "meter-bar-fill-low"
    prob_pct = prob * 100
    meter_color = "#ff3b5c" if is_high else "#00e676"

    st.markdown(
        f"""
        <div class="meter-container">
            <div class="meter-label">Crisis Probability</div>
            <div class="meter-bar-bg">
                <div class="meter-bar-fill {fill_class}" style="width: {prob_pct:.0f}%"></div>
            </div>
            <div class="meter-value" style="color: {meter_color}">{prob_pct:.0f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Factor cards
    if r["factors"]:
        cards = ""
        for name, value, direction in r["factors"]:
            cards += f"""
            <div class="factor-card">
                <div class="factor-name">{name}</div>
                <div class="factor-value factor-{direction}">{value}</div>
            </div>
            """
        st.markdown(
            f"""
            <div style="text-align:center; margin-top: 1rem;">
                <div class="meter-label">Key Factors</div>
            </div>
            <div class="factors-grid">{cards}</div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # Buttons
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Try Another Scenario", type="primary", use_container_width=True):
            st.session_state.prediction = None
            st.session_state.page = "form"
            st.rerun()

    # Disclaimer
    m = config["model_metrics"]
    st.markdown(
        f"""<div class="model-info-bar">
        ⚠️ This is a research prototype, not a professional crisis management tool.<br>
        Model: Random Forest · {m['n_samples']} training cases · F1={m['f1_high']:.2f} on minority class · AUC={m['auc']:.2f}<br>
        40% of reaction_spike values were imputed. See project documentation for full limitations.
        </div>""",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────
page = st.session_state.page

if page == "landing":
    render_landing()
elif page == "form":
    render_form()
elif page == "result":
    render_result()
