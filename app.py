import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

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
def load_models():
    return joblib.load("model_base.joblib"), joblib.load("model_full.joblib")

@st.cache_data
def load_config():
    with open("pipeline_config.json") as f:
        return json.load(f)

model_base, model_full = load_models()
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

.stApp {
    background: #0a0a0f;
    color: #e0e0e0;
    font-family: 'Outfit', sans-serif;
}
.block-container { padding-top: 1rem !important; max-width: 1200px; }
header[data-testid="stHeader"] { background: transparent; }
div[data-testid="stToolbar"] { display: none; }
#MainMenu, footer, .stDeployButton { display: none !important; }

/* ── LANDING ── */
.landing-container { text-align: center; padding: 2rem 1rem 0 1rem; }
.main-title {
    font-family: 'Outfit', sans-serif; font-weight: 900;
    font-size: clamp(2.5rem, 7vw, 5.5rem); letter-spacing: -0.02em; line-height: 1.05;
    background: linear-gradient(135deg, #ff2d7b, #ff6b9d, #c850ff, #7b61ff, #ff2d7b);
    background-size: 300% 300%;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    animation: gradientShift 4s ease infinite; margin-bottom: 0.3rem; text-transform: uppercase;
}
.sub-title {
    font-family: 'Space Mono', monospace; font-size: clamp(0.75rem, 2vw, 1rem);
    color: #888; letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 2rem;
}
@keyframes gradientShift { 0%,100%{background-position:0% 50%} 50%{background-position:100% 50%} }

.artist-cloud {
    display: flex; flex-wrap: wrap; justify-content: center; gap: 0.5rem;
    max-width: 900px; margin: 0 auto 2.5rem auto; padding: 0 1rem;
}
.artist-tag {
    font-family: 'Outfit', sans-serif; font-weight: 600; font-size: 0.8rem;
    padding: 0.35rem 0.85rem; border-radius: 100px;
    border: 1px solid rgba(255,255,255,0.08); background: rgba(255,255,255,0.03);
    color: #666; animation: tagFadeIn 0.6s ease backwards;
}
.artist-tag:nth-child(3n+1) { color: #ff2d7b55; border-color: #ff2d7b22; }
.artist-tag:nth-child(3n+2) { color: #c850ff55; border-color: #c850ff22; }
.artist-tag:nth-child(3n+3) { color: #7b61ff55; border-color: #7b61ff22; }
@keyframes tagFadeIn { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }

/* ── BUTTONS ── */
div.stButton > button {
    font-family: 'Outfit', sans-serif !important; font-weight: 700 !important;
    border: none !important; border-radius: 100px !important; transition: all 0.3s ease !important;
}
div.stButton > button[kind="primary"],
div.stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, #ff2d7b, #c850ff) !important; color: #fff !important;
    font-size: 1.1rem !important; padding: 0.7rem 3rem !important;
    box-shadow: 0 4px 25px rgba(255,45,123,0.3) !important;
}
div.stButton > button[kind="primary"]:hover,
div.stButton > button[data-testid="stBaseButton-primary"]:hover {
    transform: translateY(-2px) !important; box-shadow: 0 8px 35px rgba(255,45,123,0.5) !important;
}
div.stButton > button[kind="secondary"],
div.stButton > button[data-testid="stBaseButton-secondary"] {
    background: rgba(255,255,255,0.06) !important; color: #ccc !important;
    border: 1px solid rgba(255,255,255,0.12) !important; font-size: 0.95rem !important;
    padding: 0.6rem 2rem !important;
}
div.stButton > button[kind="secondary"]:hover,
div.stButton > button[data-testid="stBaseButton-secondary"]:hover {
    background: rgba(255,255,255,0.1) !important; color: #fff !important;
}

/* ── FORM ── */
.form-header {
    font-family: 'Outfit', sans-serif; font-weight: 800; font-size: 2rem; text-align: center;
    margin-bottom: 0.3rem; background: linear-gradient(135deg, #ff6b9d, #c850ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.form-subheader {
    font-family: 'Space Mono', monospace; font-size: 0.8rem; color: #666;
    text-align: center; letter-spacing: 0.08em; margin-bottom: 2rem;
}
.section-label {
    font-family: 'Outfit', sans-serif; font-weight: 700; font-size: 0.7rem;
    letter-spacing: 0.2em; text-transform: uppercase; color: #ff6b9d;
    margin: 1.5rem 0 0.8rem 0; padding-bottom: 0.4rem;
    border-bottom: 1px solid rgba(255,107,157,0.15);
}
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label {
    font-family: 'Outfit', sans-serif !important; font-weight: 500 !important;
    font-size: 0.85rem !important; color: #bbb !important;
}
div[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.04) !important; border-color: rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
}

/* ── RESULT ── */
.result-container { text-align: center; padding: 2rem 1rem; }
.result-verdict {
    font-family: 'Outfit', sans-serif; font-weight: 900;
    font-size: clamp(2rem, 5vw, 3.5rem); letter-spacing: -0.02em; margin-bottom: 0.3rem;
}
.verdict-high { color: #ff3b5c; text-shadow: 0 0 40px rgba(255,59,92,0.4); }
.verdict-manageable { color: #00e676; text-shadow: 0 0 40px rgba(0,230,118,0.4); }
.result-subtitle {
    font-family: 'Space Mono', monospace; font-size: 0.85rem; color: #888; margin-bottom: 2rem;
}

/* Probability meter */
.meter-container { max-width: 500px; margin: 0 auto 2rem auto; }
.meter-label {
    font-family: 'Space Mono', monospace; font-size: 0.7rem; letter-spacing: 0.15em;
    text-transform: uppercase; color: #888; margin-bottom: 0.4rem;
}
.meter-bar-bg {
    width: 100%; height: 12px; background: rgba(255,255,255,0.06);
    border-radius: 100px; overflow: hidden;
}
.meter-bar-fill { height: 100%; border-radius: 100px; }
.meter-bar-fill-high { background: linear-gradient(90deg, #ff9800, #ff3b5c); box-shadow: 0 0 15px rgba(255,59,92,0.4); }
.meter-bar-fill-low { background: linear-gradient(90deg, #00e676, #69f0ae); box-shadow: 0 0 15px rgba(0,230,118,0.3); }
.meter-value { font-family: 'Space Mono', monospace; font-weight: 700; font-size: 1.8rem; margin-top: 0.5rem; }

/* Factor cards */
.factors-grid {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 0.8rem; max-width: 800px; margin: 1.5rem auto;
}
.factor-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px; padding: 1rem 0.8rem; text-align: center;
    position: relative; overflow: hidden;
}
.factor-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
}
.factor-card-raising::before { background: linear-gradient(90deg, #ff9800, #ff3b5c); }
.factor-card-lowering::before { background: linear-gradient(90deg, #00e676, #69f0ae); }
.factor-card-neutral::before { background: rgba(255,255,255,0.1); }
.factor-icon {
    font-size: 1.6rem; margin-bottom: 0.5rem; display: block;
}
.factor-name {
    font-family: 'Space Mono', monospace; font-size: 0.6rem; letter-spacing: 0.1em;
    text-transform: uppercase; color: #aaa; margin-bottom: 0.4rem;
}
.factor-value { font-family: 'Outfit', sans-serif; font-weight: 700; font-size: 0.9rem; }
.factor-raising { color: #ff6b6b; }
.factor-lowering { color: #69f0ae; }
.factor-neutral { color: #999; }
.factor-direction {
    font-family: 'Space Mono', monospace; font-size: 0.6rem; margin-top: 0.35rem;
    letter-spacing: 0.05em;
}
.factor-dir-raising { color: #ff6b6b99; }
.factor-dir-lowering { color: #69f0ae99; }
.factor-dir-neutral { color: #66666699; }

/* Animations */
@keyframes pulseRed { 0%,100%{box-shadow:0 0 20px rgba(255,59,92,0.2)} 50%{box-shadow:0 0 40px rgba(255,59,92,0.4)} }
@keyframes pulseGreen { 0%,100%{box-shadow:0 0 20px rgba(0,230,118,0.15)} 50%{box-shadow:0 0 40px rgba(0,230,118,0.3)} }
.result-icon-circle {
    width: 120px; height: 120px; border-radius: 50%; display: flex;
    align-items: center; justify-content: center; margin: 0 auto 1.5rem auto;
    font-size: 3rem; animation-duration: 2s; animation-iteration-count: infinite;
}
.icon-high { background: rgba(255,59,92,0.1); border: 2px solid rgba(255,59,92,0.3); animation-name: pulseRed; }
.icon-manageable { background: rgba(0,230,118,0.08); border: 2px solid rgba(0,230,118,0.25); animation-name: pulseGreen; }

.confetti-container {
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    pointer-events: none; z-index: 9999; overflow: hidden;
}
.confetti-piece {
    position: absolute; width: 8px; height: 8px; top: -10px;
    border-radius: 2px; animation: confettiFall linear forwards;
}
@keyframes confettiFall { 0%{transform:translateY(0) rotate(0deg);opacity:1} 100%{transform:translateY(100vh) rotate(720deg);opacity:0} }
@keyframes warningFlash { 0%{opacity:0} 10%{opacity:0.08} 20%{opacity:0} 30%{opacity:0.05} 40%,100%{opacity:0} }
@keyframes screenShake {
    0%,100%{transform:translate(0,0)}
    15%{transform:translate(-5px,-3px)}
    30%{transform:translate(5px,3px)}
    45%{transform:translate(-4px,4px)}
    60%{transform:translate(4px,-4px)}
    75%{transform:translate(-3px,3px)}
    90%{transform:translate(3px,-3px)}
}
@keyframes glitch1 {
    0%,100%{clip-path:inset(0 0 92% 0);transform:translate(-3px,0)}
    50%{clip-path:inset(0 0 82% 0);transform:translate(3px,0)}
}
@keyframes glitch2 {
    0%,100%{clip-path:inset(55% 0 25% 0);transform:translate(3px,0)}
    50%{clip-path:inset(65% 0 15% 0);transform:translate(-3px,0)}
}
.glitch-text { position: relative; display: inline-block; }
.glitch-text::before, .glitch-text::after {
    content: attr(data-text); position: absolute;
    top: 0; left: 0; width: 100%; background: #0a0a0f;
    font-weight: 900;
}
.glitch-text::before { color: #ff0055; animation: glitch1 0.45s infinite; }
.glitch-text::after  { color: #00ffff; animation: glitch2 0.45s infinite; }
.shake-wrapper { animation: screenShake 0.6s ease-out; }
.danger-particle {
    position: fixed; top: -30px;
    pointer-events: none; z-index: 9998; font-size: 1.4rem;
    animation: dangerFall linear forwards;
}
@keyframes dangerFall {
    0%  { transform: translateY(0)     rotate(0deg)   scale(1);   opacity: 1; }
    80% { opacity: 0.7; }
    100%{ transform: translateY(105vh) rotate(400deg) scale(0.4); opacity: 0; }
}
.warning-overlay {
    position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: #ff3b5c;
    pointer-events: none; z-index: 9999; animation: warningFlash 2s ease-out forwards;
}

/* ── SENSITIVITY SECTION ── */
.sensitivity-header {
    font-family: 'Outfit', sans-serif; font-weight: 800; font-size: 1.6rem;
    text-align: center; margin: 2.5rem 0 0.5rem 0; color: #fff;
}
.sensitivity-sub {
    font-family: 'Outfit', sans-serif; font-size: 0.95rem; color: #bbb;
    text-align: center; margin-bottom: 1.5rem; max-width: 650px;
    margin-left: auto; margin-right: auto; line-height: 1.6;
}
.threshold-card {
    max-width: 550px; margin: 1.5rem auto; padding: 1.5rem 2rem;
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px; text-align: center;
}
.threshold-label {
    font-family: 'Space Mono', monospace; font-size: 0.7rem; letter-spacing: 0.15em;
    text-transform: uppercase; color: #bbb; margin-bottom: 0.5rem;
}
.threshold-value {
    font-family: 'Outfit', sans-serif; font-weight: 800; font-size: 2.2rem;
}
.threshold-explainer {
    font-family: 'Outfit', sans-serif; font-size: 0.95rem; color: #e0e0e0;
    margin-top: 0.6rem; line-height: 1.5;
}
.threshold-context {
    font-family: 'Space Mono', monospace; font-size: 0.7rem; color: #888;
    margin-top: 0.5rem; font-style: italic;
}

/* ── COMMON ── */
.model-info-bar {
    max-width: 700px; margin: 2rem auto 0 auto; padding: 1rem 1.5rem;
    background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px; font-family: 'Space Mono', monospace; font-size: 0.7rem;
    color: #555; text-align: center; line-height: 1.7;
}
.styled-divider {
    width: 60px; height: 3px; background: linear-gradient(90deg, #ff2d7b, #c850ff);
    border-radius: 100px; margin: 0 auto 2rem auto;
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
def build_base_vector(inputs: dict) -> pd.DataFrame:
    """Build feature vector for the 9-feature base model."""
    feature_names = config["base_feature_columns"]
    row = pd.DataFrame([inputs])
    row_encoded = pd.get_dummies(row, columns=["scandal_type"], prefix="type")
    for col in feature_names:
        if col not in row_encoded.columns:
            row_encoded[col] = 0
    return row_encoded[feature_names]


def build_full_vector(inputs: dict, spike: float) -> pd.DataFrame:
    """Build feature vector for the 10-feature full model."""
    feature_names = config["full_feature_columns"]
    full_inputs = {**inputs, "reaction_spike": spike}
    row = pd.DataFrame([full_inputs])
    row_encoded = pd.get_dummies(row, columns=["scandal_type"], prefix="type")
    for col in feature_names:
        if col not in row_encoded.columns:
            row_encoded[col] = 0
    return row_encoded[feature_names]


def predict_crisis(inputs: dict) -> dict:
    # ── Base prediction (9 features) ──
    X_base = build_base_vector(inputs)
    base_prob = model_base.predict_proba(X_base)[0][1]
    base_pred = "high" if base_prob > 0.5 else "manageable"

    # ── Sensitivity sweep (full model, varying reaction_spike) ──
    spike_range = np.linspace(-6, 14, 81)  # 81 points for smooth curve
    spike_probs = []
    for s in spike_range:
        X_full = build_full_vector(inputs, s)
        p = model_full.predict_proba(X_full)[0][1]
        spike_probs.append(p)
    spike_probs = np.array(spike_probs)

    # ── Find critical threshold (where P(high) crosses 0.5) ──
    threshold = None
    for i in range(len(spike_probs) - 1):
        if spike_probs[i] < 0.5 <= spike_probs[i + 1]:
            # Linear interpolation
            frac = (0.5 - spike_probs[i]) / (spike_probs[i + 1] - spike_probs[i])
            threshold = spike_range[i] + frac * (spike_range[i + 1] - spike_range[i])
            break

    # If always above or below 0.5
    if threshold is None:
        if spike_probs[0] >= 0.5:
            threshold = "always_high"
        else:
            threshold = "always_manageable"

    # ── Key factors for the base prediction ──
    importances = config["base_feature_importances"]
    top_features = sorted(importances.items(), key=lambda x: -x[1])[:5]
    factors = []
    for feat, imp in top_features:
        val = X_base[feat].values[0]
        if feat == "fandom_size_num":
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
        elif feat == "agency_tier":
            tier_labels = {1: "Small/Indie — vulnerable", 2: "Mid-tier", 3: "Big3/Big4 — protective"}
            direction = "raising" if val == 1 else ("lowering" if val == 3 else "neutral")
            factors.append(("Agency Tier", tier_labels.get(int(val), str(val)), direction))

    return {
        "base_prob": base_prob,
        "base_pred": base_pred,
        "factors": factors,
        "spike_range": spike_range.tolist(),
        "spike_probs": spike_probs.tolist(),
        "threshold": threshold,
    }


# ─────────────────────────────────────────────────────────────
# PAGES
# ─────────────────────────────────────────────────────────────

def render_landing():
    st.markdown('<div class="landing-container">', unsafe_allow_html=True)
    st.markdown('<div class="main-title">Scandal Impact<br>Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Machine learning meets K-Pop crisis management</div>', unsafe_allow_html=True)
    st.markdown('<div class="styled-divider"></div>', unsafe_allow_html=True)

    tags = "".join(
        f'<span class="artist-tag" style="animation-delay: {i * 0.03:.2f}s">{a}</span>'
        for i, a in enumerate(ARTISTS)
    )
    st.markdown(f'<div class="artist-cloud">{tags}</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        def go_to_form():
            st.session_state.page = "form"
        st.button("Start Prediction", type="primary", use_container_width=True, on_click=go_to_form)

    m = config["base_model_metrics"]
    st.markdown(
        f"""<div class="model-info-bar">
        Random Forest · {m['n_samples']} scandals analyzed · {m['accuracy']*100:.1f}% accuracy · AUC {m['auc']:.2f}<br>
        Trained on real K-pop scandal data from 2006–2024 · Includes reaction spike sensitivity analysis
        </div>""",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_form():
    st.markdown('<div class="form-header">Configure the Scandal</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="form-subheader">Set the parameters — the model predicts using only what you\'d know when the scandal breaks</div>',
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
            key="form_scandal_type",
        )

        international = st.selectbox(
            "International Attention?",
            options=[1, 0],
            format_func=lambda x: "Yes — global media pickup" if x == 1 else "No — Korea-only",
            key="form_international",
        )

        is_solo = st.selectbox(
            "Artist Type",
            options=[0, 1],
            format_func=lambda x: "Solo artist" if x == 1 else "Group member",
            key="form_is_solo",
        )

        prior_scandal = st.selectbox(
            "Prior Scandal on Record?",
            options=[0, 1],
            format_func=lambda x: "Yes — repeat offender" if x == 1 else "No — first incident",
            key="form_prior_scandal",
        )

    with col_right:
        st.markdown('<div class="section-label">🏢 The Response</div>', unsafe_allow_html=True)

        fandom_size = st.selectbox(
            "Fandom Size",
            options=[2, 3, 4],
            format_func=lambda x: {2: "Medium", 3: "Large", 4: "Mega (BTS/BP level)"}.get(x),
            key="form_fandom_size",
        )

        agency_tier = st.selectbox(
            "Agency Tier",
            options=[1, 2, 3],
            format_func=lambda x: {
                1: "Small / Indie", 2: "Mid-tier", 3: "Big3/Big4 (SM, YG, JYP, HYBE)"
            }.get(x),
            key="form_agency_tier",
        )

        company_response = st.selectbox(
            "Company Response Strength",
            options=[0, 1, 2],
            format_func=lambda x: {0: "Silence", 1: "Partial / vague statement", 2: "Strong defense"}.get(x),
            key="form_company_response",
        )

        response_delay = st.slider(
            "Response Delay (days)",
            min_value=0, max_value=30, value=1,
            help="0 = same-day response. Higher = slower reaction.",
            key="form_response_delay",
        )

        apology = st.selectbox(
            "Public Apology Issued?",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            key="form_apology",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    def go_back():
        st.session_state.page = "landing"

    def run_prediction():
        inputs = {
            "scandal_type": st.session_state.form_scandal_type,
            "fandom_size_num": st.session_state.form_fandom_size,
            "agency_tier": st.session_state.form_agency_tier,
            "company_response": st.session_state.form_company_response,
            "response_delay_days": st.session_state.form_response_delay,
            "apology": st.session_state.form_apology,
            "international": st.session_state.form_international,
            "is_solo": st.session_state.form_is_solo,
            "prior_scandal": st.session_state.form_prior_scandal,
        }
        st.session_state.prediction = predict_crisis(inputs)
        st.session_state.page = "result"

    col_b1, col_b2, col_b3 = st.columns([2, 1, 2])
    with col_b1:
        st.button("← Back", type="secondary", on_click=go_back)
    with col_b3:
        st.button("Predict Outcome →", type="primary", use_container_width=True, on_click=run_prediction)


def render_result():
    r = st.session_state.prediction
    if r is None:
        st.session_state.page = "landing"
        st.rerun()
        return

    base_prob = r["base_prob"]
    is_high = r["base_pred"] == "high"

# ── Animation overlays ──
    if is_high:
        st.markdown('<div class="warning-overlay"></div>', unsafe_allow_html=True)
        symbols = ["⚠️", "💀", "🚨", "❌", "⛔"]
        particles = "".join(
            f'<div class="danger-particle" style="'
            f'left:{np.random.randint(0,100)}%;'
            f'animation-delay:{np.random.uniform(0,2.5):.2f}s;'
            f'animation-duration:{np.random.uniform(3,5.5):.2f}s;">'
            f'{symbols[i % len(symbols)]}</div>'
            for i in range(45)
        )
        st.markdown(f'<div class="confetti-container">{particles}</div>', unsafe_allow_html=True)
    else:
        colors = ["#00e676", "#69f0ae", "#b9f6ca", "#ff6b9d", "#c850ff", "#7b61ff", "#fff176", "#ffd54f"]
        shapes = ["border-radius:50%", "border-radius:2px", "border-radius:0", "transform:rotate(45deg)"]
        pieces = "".join(
            f'<div class="confetti-piece" style="left:{np.random.randint(0,100)}%;'
            f'width:{np.random.randint(7,14)}px;height:{np.random.randint(7,14)}px;'
            f'background:{colors[i%len(colors)]};{shapes[i%len(shapes)]};'
            f'animation-delay:{np.random.uniform(0,2):.1f}s;'
            f'animation-duration:{np.random.uniform(2,4.5):.1f}s;"></div>'
            for i in range(80)
        )
        st.markdown(f'<div class="confetti-container">{pieces}</div>', unsafe_allow_html=True)

    # ── Verdict ──
    st.markdown('<div class="result-container">', unsafe_allow_html=True)

    if is_high:
        st.markdown('<div class="result-icon-circle icon-high">⚠️</div>', unsafe_allow_html=True)
        st.markdown('<div class="result-verdict verdict-high">HIGH CRISIS</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="result-subtitle">Career-altering damage likely — group departure, termination, or criminal charges</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="result-icon-circle icon-manageable">✓</div>', unsafe_allow_html=True)
        st.markdown('<div class="result-verdict verdict-manageable">MANAGEABLE</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="result-subtitle">Recovery expected within weeks to months — career survives</div>',
            unsafe_allow_html=True,
        )

    # ── Probability meter ──
    fill_class = "meter-bar-fill-high" if is_high else "meter-bar-fill-low"
    meter_color = "#ff3b5c" if is_high else "#00e676"
    prob_pct = base_prob * 100

    st.markdown(
        f"""<div class="meter-container">
            <div class="meter-label">Crisis Probability (before public reaction)</div>
            <div class="meter-bar-bg">
                <div class="meter-bar-fill {fill_class}" style="width: {prob_pct:.0f}%"></div>
            </div>
            <div class="meter-value" style="color: {meter_color}">{prob_pct:.0f}%</div>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Key factors ──
    if r["factors"]:
        icon_map = {
            "Fandom Size": ("👥", ),
            "Response Delay": ("⏱️", ),
            "Scandal Type": ("⚡", ),
            "Company Response": ("🏢", ),
            "Agency Tier": ("🏛️", ),
        }
        dir_labels = {
            "raising": "↑ Increases risk",
            "lowering": "↓ Lowers risk",
            "neutral": "— Neutral",
        }
        cards = ""
        for name, value, direction in r["factors"]:
            icon = icon_map.get(name, ("📊",))[0]
            dir_label = dir_labels.get(direction, "")
            cards += (
                f'<div class="factor-card factor-card-{direction}">'
                f'<span class="factor-icon">{icon}</span>'
                f'<div class="factor-name">{name}</div>'
                f'<div class="factor-value factor-{direction}">{value}</div>'
                f'<div class="factor-direction factor-dir-{direction}">{dir_label}</div>'
                f'</div>'
            )
        st.markdown(
            f'<div style="text-align:center;margin-top:1rem;"><div class="meter-label">Key Factors</div></div>'
            f'<div class="factors-grid">{cards}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ── SENSITIVITY ANALYSIS ──
    st.markdown('<div class="sensitivity-header">What If It Goes Viral?</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sensitivity-sub">'
        'The prediction above is based on 9 structural features — things you know when the scandal breaks. '
        'But how much the public actually reacts (measured as a Google Trends spike) can shift the outcome. '
        'The curve below sweeps that reaction from minimal to maximum and shows where the crisis probability lands.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Build SVG chart
    spike_range = r["spike_range"]
    spike_probs = r["spike_probs"]
    threshold = r["threshold"]

    chart_w, chart_h = 600, 260
    pad_l, pad_r, pad_t, pad_b = 55, 25, 20, 40

    plot_w = chart_w - pad_l - pad_r
    plot_h = chart_h - pad_t - pad_b

    def to_x(val):
        return pad_l + (val - spike_range[0]) / (spike_range[-1] - spike_range[0]) * plot_w

    def to_y(val):
        return pad_t + (1.0 - val) * plot_h

    # Build path
    points = " ".join(f"{to_x(spike_range[i]):.1f},{to_y(spike_probs[i]):.1f}" for i in range(len(spike_range)))

    # Gradient fill under curve
    fill_points = (
        f"{to_x(spike_range[0]):.1f},{to_y(0):.1f} "
        + points
        + f" {to_x(spike_range[-1]):.1f},{to_y(0):.1f}"
    )

    # 50% threshold line y position
    y50 = to_y(0.5)

    # Threshold marker
    threshold_marker = ""
    threshold_text_svg = ""
    if isinstance(threshold, (int, float)):
        tx = to_x(threshold)
        threshold_marker = (
            f'<line x1="{tx:.1f}" y1="{pad_t}" x2="{tx:.1f}" y2="{pad_t + plot_h}" '
            f'stroke="#ff6b9d" stroke-width="1.5" stroke-dasharray="6,4" opacity="0.8"/>'
            f'<circle cx="{tx:.1f}" cy="{y50:.1f}" r="5" fill="#ff6b9d" opacity="0.9"/>'
        )
        threshold_text_svg = (
            f'<text x="{tx:.1f}" y="{pad_t - 6}" text-anchor="middle" '
            f'fill="#ff6b9d" font-family="Space Mono, monospace" font-size="10" font-weight="700">'
            f'Spike = {threshold:.1f}</text>'
        )

    # Y-axis labels
    y_labels = ""
    for pct in [0, 25, 50, 75, 100]:
        yy = to_y(pct / 100)
        y_labels += (
            f'<text x="{pad_l - 8}" y="{yy + 4}" text-anchor="end" fill="#666" '
            f'font-family="Space Mono, monospace" font-size="10">{pct}%</text>'
            f'<line x1="{pad_l}" y1="{yy}" x2="{pad_l + plot_w}" y2="{yy}" '
            f'stroke="rgba(255,255,255,0.06)" stroke-width="1"/>'
        )

    # X-axis labels
    x_labels = ""
    for xv in [-4, -2, 0, 2, 4, 6, 8, 10, 12]:
        xx = to_x(xv)
        x_labels += (
            f'<text x="{xx}" y="{pad_t + plot_h + 16}" text-anchor="middle" fill="#666" '
            f'font-family="Space Mono, monospace" font-size="10">{xv:+d}</text>'
        )

    # X-axis title
    x_title = (
        f'<text x="{pad_l + plot_w / 2}" y="{chart_h - 2}" text-anchor="middle" fill="#888" '
        f'font-family="Space Mono, monospace" font-size="10">Reaction Spike (Google Trends)</text>'
    )

    svg = f"""<svg viewBox="0 0 {chart_w} {chart_h}" xmlns="http://www.w3.org/2000/svg"
        style="width:100%;max-width:650px;margin:0 auto;display:block;">
        <defs>
            <linearGradient id="curveGrad" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0%" stop-color="#69f0ae"/>
                <stop offset="40%" stop-color="#ffd54f"/>
                <stop offset="70%" stop-color="#ff9800"/>
                <stop offset="100%" stop-color="#ff3b5c"/>
            </linearGradient>
            <linearGradient id="fillGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stop-color="url(#curveGrad)" stop-opacity="0.15"/>
                <stop offset="100%" stop-color="url(#curveGrad)" stop-opacity="0.02"/>
            </linearGradient>
        </defs>
        {y_labels}
        {x_labels}
        {x_title}
        <!-- 50% line -->
        <line x1="{pad_l}" y1="{y50:.1f}" x2="{pad_l + plot_w}" y2="{y50:.1f}"
            stroke="#ff6b9d" stroke-width="1" stroke-dasharray="3,5" opacity="0.4"/>
        <text x="{pad_l + plot_w + 4}" y="{y50 + 3}" fill="#ff6b9d" opacity="0.6"
            font-family="Space Mono, monospace" font-size="9">50%</text>
        <!-- Fill under curve -->
        <polygon points="{fill_points}" fill="url(#fillGrad)"/>
        <!-- Curve -->
        <polyline points="{points}" fill="none" stroke="url(#curveGrad)" stroke-width="2.5"
            stroke-linecap="round" stroke-linejoin="round"/>
        <!-- Threshold marker -->
        {threshold_marker}
        {threshold_text_svg}
    </svg>"""

    st.markdown(f'<div>{svg}</div>', unsafe_allow_html=True)

    # ── Threshold card ──
    if isinstance(threshold, (int, float)):
        color = "#ff6b9d"
        spike_stats = config["reaction_spike_stats"]
        if threshold < spike_stats["median"]:
            context = f"The median spike in the dataset is {spike_stats['median']:.1f} — this threshold is below it, meaning even a modest public reaction could escalate things."
        elif threshold > spike_stats["mean"] + spike_stats["std"]:
            context = f"The average spike is {spike_stats['mean']:.1f}. This threshold is well above that — only an unusually viral moment would push this into crisis territory."
        else:
            context = f"The average spike is {spike_stats['mean']:.1f}. This threshold is in a realistic range — public reaction could realistically tip this either way."

        st.markdown(
            f"""<div class="threshold-card">
                <div class="threshold-label">Critical Tipping Point</div>
                <div class="threshold-value" style="color: {color}">Spike ≥ {threshold:.1f}</div>
                <div class="threshold-explainer">
                    If the Google Trends reaction spike reaches <strong>{threshold:.1f}</strong> or higher,
                    the model flips from <span style="color:#69f0ae">Manageable</span>
                    to <span style="color:#ff3b5c">High Crisis</span>.
                </div>
                <div class="threshold-context">{context}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    elif threshold == "always_high":
        st.markdown(
            """<div class="threshold-card">
                <div class="threshold-label">Critical Tipping Point</div>
                <div class="threshold-value" style="color: #ff3b5c">No Tipping Point</div>
                <div class="threshold-explainer">
                    The structural factors alone (scandal type, fandom size, agency response) already
                    put this above 50% crisis probability. Public reaction doesn't change the outcome —
                    this is a <span style="color:#ff3b5c">High Crisis</span> regardless of virality.
                </div>
            </div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """<div class="threshold-card">
                <div class="threshold-label">Critical Tipping Point</div>
                <div class="threshold-value" style="color: #69f0ae">Resilient</div>
                <div class="threshold-explainer">
                    Even at maximum virality, this scenario stays below 50% crisis probability.
                    The protective factors (large fandom, strong agency, scandal type) are strong enough
                    that public reaction alone can't push this into <span style="color:#ff3b5c">High Crisis</span>.
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

    # ── Buttons ──
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        def try_again():
            st.session_state.prediction = None
            st.session_state.page = "form"
        st.button("Try Another Scenario", type="primary", use_container_width=True, on_click=try_again)

    # ── Disclaimer ──
    m_base = config["base_model_metrics"]
    m_full = config["full_model_metrics"]
    st.markdown(
        f"""<div class="model-info-bar">
        ⚠️ Research prototype — not a crisis management tool.<br>
        Base model (9 features): {m_base['n_samples']} cases · F1={m_base['f1_high']:.2f} · AUC={m_base['auc']:.2f}<br>
        Full model (+ reaction spike): F1={m_full['f1_high']:.2f} · AUC={m_full['auc']:.2f}<br>
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
