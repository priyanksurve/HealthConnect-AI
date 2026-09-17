import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import urllib.parse
import streamlit.components.v1 as components

import db
import utils
import symptom_checker


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="HealthConnect-AI | Hospital Management",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# DB INIT
# =========================================================

@st.cache_resource
def _init_db_once():
    db.init_db()
    return True


try:
    _init_db_once()
    DB_READY = True
    DB_ERROR = ""
except Exception as e:
    DB_READY = False
    DB_ERROR = str(e)


# =========================================================
# HTML RENDER HELPER
# =========================================================

def render_html(html: str):
    cleaned = "\n".join(line.strip() for line in html.strip().split("\n"))
    st.markdown(cleaned, unsafe_allow_html=True)


# =========================================================
# CUSTOM CSS — Light Hospital Management Theme (matches reference)
# =========================================================

render_html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Poppins:wght@600;700;800&display=swap');

:root {
    --bg: #f4f8fc;
    --bg-soft: #eef5fb;
    --white: #ffffff;
    --text-dark: #1e2a3a;
    --text-muted: #5a6a7e;
    --text-light: #7a8a9e;
    --primary: #2563eb;
    --primary-dark: #1d4ed8;
    --primary-light: #3b82f6;
    --accent: #0ea5e9;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
    --border: #e2eaf3;
    --radius: 16px;
    --radius-sm: 10px;
    --shadow: 0 10px 30px rgba(37, 99, 235, 0.08);
    --shadow-lg: 0 20px 50px rgba(37, 99, 235, 0.12);
}

html, body, .stApp {
    font-family: 'Inter', sans-serif !important;
    background: var(--bg) !important;
    color: var(--text-dark) !important;
}

.stApp {
    background:
        radial-gradient(circle at 90% 10%, rgba(59, 130, 246, 0.12), transparent 40%),
        radial-gradient(circle at 10% 80%, rgba(14, 165, 233, 0.08), transparent 35%),
        linear-gradient(180deg, #f0f7ff 0%, #f8fafc 40%, #ffffff 100%) !important;
}

/* Force light text colors */
.stApp, .block-container, .stMarkdown, .stText, .streamlit-expanderHeader,
.stFormLabel, .stRadio, .stCheckbox, label, p, span, div {
    color: var(--text-dark) !important;
}

h1, h2, h3, h4, h5, h6 {
    color: var(--text-dark) !important;
    font-family: 'Poppins', 'Inter', sans-serif !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* ========== FULL-WIDTH real website layout ========== */
.block-container {
    padding-top: 0 !important;
    padding-bottom: 3rem !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
    max-width: 100% !important;
}
section.main > div {
    max-width: 100% !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
}
[data-testid="stAppViewContainer"] > .main {
    padding: 0 !important;
}
.inner {
    max-width: 1280px;
    margin: 0 auto;
    padding: 0 40px;
    width: 100%;
    box-sizing: border-box;
}
.inner-wide {
    max-width: 1400px;
    margin: 0 auto;
    padding: 0 5vw;
    width: 100%;
    box-sizing: border-box;
}
.content-pad {
    max-width: 960px;
    margin: 0 auto;
    padding: 12px 5vw 40px;
    box-sizing: border-box;
}

/* -------------------- Top bar (full bleed) -------------------- */
.top-bar {
    background: linear-gradient(90deg, #2563eb, #3b82f6);
    color: white !important;
    padding: 11px 5vw;
    display: flex;
    justify-content: space-between;
    align-items: center;
    width: 100%;
    box-sizing: border-box;
    font-size: 13.5px;
}
.top-bar * { color: white !important; }
.top-bar a { text-decoration: none; margin-left: 16px; opacity: 0.95; }
.top-bar .left { display: flex; gap: 16px; align-items: center; }
.top-bar .right { display: flex; gap: 22px; align-items: center; }

/* -------------------- Navbar (full width) -------------------- */
.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 18px 5vw;
    margin: 0;
    background: rgba(255,255,255,0.9);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border);
    width: 100%;
    box-sizing: border-box;
}
.navbar .brand {
    font-family: 'Poppins', sans-serif;
    font-size: 26px;
    font-weight: 800;
    color: var(--text-dark) !important;
    letter-spacing: -0.4px;
}
.navbar .brand span { color: var(--primary) !important; }
.navbar .nav-links {
    display: flex;
    gap: 32px;
    font-weight: 600;
    font-size: 15px;
    color: var(--text-muted) !important;
}
.navbar .nav-links span {
    cursor: default;
    transition: color 0.15s;
}
.navbar .nav-links span:hover { color: var(--primary) !important; }

/* -------------------- Inputs -------------------- */
.stTextInput > div > div > input,
.stTextInput > div > div > textarea,
.stSelectbox > div > div > div,
.stNumberInput > div > div > input,
.stTextArea > div > div > textarea {
    background: var(--white) !important;
    color: var(--text-dark) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    padding: 12px 14px !important;
    font-size: 15px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}

.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus,
.stNumberInput > div > div > input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15) !important;
}

.stTextInput > div > div > input::placeholder,
.stTextArea > div > div > textarea::placeholder {
    color: var(--text-light) !important;
}

.stTextInput > div > label,
.stSelectbox > label,
.stNumberInput > label,
.stTextArea > label,
.stFormLabel {
    color: var(--text-muted) !important;
    font-weight: 600 !important;
    font-size: 13.5px !important;
}

/* -------------------- Forms -------------------- */
.stForm {
    background: var(--white) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 32px 28px !important;
    box-shadow: var(--shadow) !important;
}

/* -------------------- Buttons -------------------- */
.stButton > button {
    width: 100%;
    border: none !important;
    border-radius: 50px !important;
    padding: 13px 22px !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    letter-spacing: 0.2px !important;
    color: white !important;
    background: linear-gradient(105deg, var(--primary), var(--primary-light)) !important;
    box-shadow: 0 8px 20px rgba(37, 99, 235, 0.25) !important;
    transition: transform 0.18s, box-shadow 0.18s !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 28px rgba(37, 99, 235, 0.35) !important;
}

.stButton > button:disabled {
    opacity: 0.45 !important;
    cursor: not-allowed !important;
    box-shadow: none !important;
}

/* Secondary buttons in columns */
div[data-testid="stHorizontalBlock"] .stButton > button {
    background: var(--white) !important;
    color: var(--primary) !important;
    border: 2px solid var(--primary) !important;
    box-shadow: none !important;
}
div[data-testid="stHorizontalBlock"] .stButton > button:hover {
    background: #eff6ff !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.12) !important;
}

/* -------------------- Sidebar (minimal) -------------------- */
[data-testid="stSidebar"] {
    background: var(--white) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * {
    color: var(--text-dark) !important;
}

/* -------------------- Hero (full-width section) -------------------- */
.hero-section {
    width: 100%;
    padding: 48px 0 56px;
    background: linear-gradient(180deg, rgba(239,246,255,0.6) 0%, transparent 100%);
}
.hero {
    display: grid;
    grid-template-columns: 1.15fr 0.85fr;
    gap: 48px;
    align-items: center;
    max-width: 1280px;
    margin: 0 auto;
    padding: 0 5vw;
    box-sizing: border-box;
}
.hero-left h1 {
    font-family: 'Poppins', sans-serif !important;
    font-size: 48px !important;
    font-weight: 800 !important;
    line-height: 1.15 !important;
    color: var(--text-dark) !important;
    margin: 0 0 20px !important;
    letter-spacing: -0.7px !important;
}
.hero-left p {
    color: var(--text-muted) !important;
    font-size: 16.5px !important;
    line-height: 1.75 !important;
    max-width: 520px !important;
    margin: 0 0 28px !important;
}
.hero-cta {
    display: inline-block;
    background: linear-gradient(105deg, var(--primary), var(--primary-light));
    color: white !important;
    font-weight: 700;
    font-size: 14.5px;
    padding: 14px 28px;
    border-radius: 50px;
    box-shadow: 0 8px 22px rgba(37, 99, 235, 0.3);
    text-decoration: none;
    letter-spacing: 0.3px;
}
.hero-right {
    position: relative;
    display: flex;
    justify-content: center;
    align-items: center;
}
.hero-card {
    width: 100%;
    max-width: 380px;
    aspect-ratio: 1;
    border-radius: 50%;
    background: linear-gradient(145deg, #dbeafe, #bfdbfe);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    position: relative;
    box-shadow: var(--shadow-lg);
}
.hero-card::before {
    content: '';
    position: absolute;
    width: 85%;
    height: 85%;
    border-radius: 50%;
    border: 2px dashed rgba(37, 99, 235, 0.25);
}
.hero-emoji {
    font-size: 90px;
    z-index: 1;
    filter: drop-shadow(0 8px 16px rgba(37, 99, 235, 0.2));
}
.hero-badge-float {
    position: absolute;
    background: white;
    border-radius: 12px;
    padding: 10px 14px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.08);
    font-size: 13px;
    font-weight: 700;
    color: var(--text-dark) !important;
    display: flex;
    align-items: center;
    gap: 8px;
}
.hero-badge-float.top-right { top: 18%; right: 0; }
.hero-badge-float.bottom-left { bottom: 18%; left: 0; }
.plus-deco {
    position: absolute;
    color: var(--primary);
    font-size: 22px;
    font-weight: 300;
    opacity: 0.5;
}

/* -------------------- Section titles -------------------- */
.section-head {
    text-align: center;
    margin: 56px auto 32px;
    max-width: 1280px;
    padding: 0 5vw;
    box-sizing: border-box;
}
.section-head h2 {
    font-family: 'Poppins', sans-serif !important;
    font-size: 30px !important;
    font-weight: 800 !important;
    color: var(--text-dark) !important;
    margin: 0 0 12px !important;
}
.section-head p {
    color: var(--text-muted) !important;
    font-size: 15px !important;
    max-width: 560px;
    margin: 0 auto !important;
    line-height: 1.65 !important;
}

/* -------------------- Service cards (exact style from image) -------------------- */
.services-wrap {
    max-width: 1280px;
    margin: 0 auto;
    padding: 0 5vw 20px;
    box-sizing: border-box;
}
.service-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 22px;
    margin-bottom: 18px;
}
.service-grid-2 {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 22px;
    max-width: 720px;
    margin: 0 auto 48px;
}
.svc-card {
    background: linear-gradient(145deg, #2563eb, #3b82f6);
    border-radius: 18px;
    padding: 28px 22px;
    color: white !important;
    box-shadow: 0 12px 28px rgba(37, 99, 235, 0.28);
    transition: transform 0.2s, box-shadow 0.2s;
    text-align: left;
}
.svc-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 18px 36px rgba(37, 99, 235, 0.35);
}
.svc-card .icon-circle {
    width: 52px;
    height: 52px;
    border-radius: 50%;
    background: rgba(255,255,255,0.2);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    margin-bottom: 16px;
}
.svc-card h4 {
    margin: 0 0 8px !important;
    font-size: 17px !important;
    font-weight: 700 !important;
    color: white !important;
}
.svc-card p {
    margin: 0 !important;
    font-size: 13px !important;
    line-height: 1.55 !important;
    color: rgba(255,255,255,0.85) !important;
}

/* -------------------- Role / portal cards -------------------- */
.portal-wrap {
    max-width: 900px;
    margin: 0 auto;
    padding: 0 5vw 40px;
    box-sizing: border-box;
}
.portal-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 28px;
    margin: 20px 0 10px;
}
.portal-card {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 40px 28px;
    text-align: center;
    box-shadow: var(--shadow);
    transition: transform 0.2s, box-shadow 0.2s, border-color 0.2s;
}
.portal-card:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-lg);
    border-color: #bfdbfe;
}
.portal-card .p-icon {
    font-size: 48px;
    margin-bottom: 14px;
}
.portal-card h3 {
    font-size: 20px !important;
    font-weight: 700 !important;
    margin: 0 0 10px !important;
    color: var(--text-dark) !important;
}
.portal-card p {
    font-size: 14px !important;
    color: var(--text-muted) !important;
    line-height: 1.6 !important;
    margin: 0 !important;
}

/* -------------------- Auth pages -------------------- */
.auth-wrap {
    max-width: 480px;
    margin: 32px auto 10px;
    padding: 0 24px;
}
.auth-head {
    text-align: center;
    margin-bottom: 28px;
}
.auth-icon-box {
    width: 70px;
    height: 70px;
    border-radius: 20px;
    background: linear-gradient(145deg, #dbeafe, #bfdbfe);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 32px;
    margin: 0 auto 16px;
    box-shadow: 0 8px 20px rgba(37, 99, 235, 0.15);
}
.auth-head h2 {
    font-family: 'Poppins', sans-serif !important;
    font-size: 26px !important;
    font-weight: 800 !important;
    margin: 0 0 8px !important;
    color: var(--text-dark) !important;
}
.auth-head p {
    color: var(--text-muted) !important;
    font-size: 14.5px !important;
    margin: 0 !important;
    line-height: 1.55 !important;
}

/* -------------------- Metric / feature cards -------------------- */
.feat-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 20px;
    max-width: 1280px;
    margin: 36px auto;
    padding: 0 5vw;
    box-sizing: border-box;
}
.feat-card {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px 18px;
    box-shadow: var(--shadow);
    transition: transform 0.2s;
}
.feat-card:hover { transform: translateY(-3px); }
.feat-card .label {
    font-size: 11.5px;
    font-weight: 700;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: var(--primary) !important;
    margin-bottom: 10px;
}
.feat-card .value {
    font-size: 26px;
    font-weight: 800;
    color: var(--text-dark) !important;
    margin-bottom: 6px;
}
.feat-card .note {
    font-size: 12.5px;
    color: var(--text-muted) !important;
    line-height: 1.5;
    margin: 0;
}

/* -------------------- Result / status boxes -------------------- */
.success-box {
    padding: 16px 18px;
    border-radius: 12px;
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
    color: #047857 !important;
    text-align: center;
    font-weight: 700;
}
.warning-box {
    padding: 16px 18px;
    border-radius: 12px;
    background: #fef2f2;
    border: 1px solid #fecaca;
    color: #b91c1c !important;
    text-align: center;
    font-weight: 700;
}
.result-card {
    padding: 22px;
    border-radius: 16px;
    text-align: center;
    background: var(--white);
    border: 1px solid var(--border);
    box-shadow: var(--shadow);
}
.result-title {
    color: var(--text-light) !important;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.8px;
    text-transform: uppercase;
}
.result-value {
    font-size: 24px;
    font-weight: 800;
    margin-top: 8px;
    color: var(--text-dark) !important;
}

/* -------------------- Hospital / patient cards -------------------- */
.hospital-card, .patient-row, .doctor-card, .token-card {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px 22px;
    box-shadow: var(--shadow);
    margin: 12px 0;
}
.hospital-name {
    font-size: 17px;
    font-weight: 700;
    color: var(--text-dark) !important;
    margin-bottom: 8px;
}
.hospital-info, .patient-row {
    color: var(--text-muted) !important;
    line-height: 1.8;
    font-size: 14px;
}
.patient-row b, .hospital-info b {
    color: var(--text-dark) !important;
}
.doctor-card {
    background: linear-gradient(145deg, #eff6ff, #dbeafe);
    border-color: #bfdbfe;
}
.doctor-card h3 { color: var(--text-dark) !important; margin-bottom: 8px; }
.doctor-card p { color: var(--text-muted) !important; margin: 0; line-height: 1.65; }
.token-card {
    text-align: center;
    background: linear-gradient(145deg, #eff6ff, #e0f2fe);
    border-color: #93c5fd;
}
.token-number {
    font-size: 28px;
    font-weight: 800;
    color: var(--primary) !important;
    letter-spacing: 2px;
}

.bed-pill {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 12px;
    margin-top: 6px;
}
.bed-good { background: #d1fae5; color: #047857 !important; }
.bed-low { background: #fef3c7; color: #b45309 !important; }
.bed-full { background: #fee2e2; color: #b91c1c !important; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: var(--white) !important;
    border: 2px dashed #93c5fd !important;
    border-radius: 14px !important;
    padding: 12px !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    padding: 10px 18px !important;
    font-weight: 600 !important;
}

/* Alerts */
.stAlert { border-radius: 12px !important; }

/* Streamlit columns / content full-width padding */
div[data-testid="stHorizontalBlock"],
div[data-testid="stVerticalBlock"] > div {
    max-width: 100%;
}
/* Give interactive Streamlit widgets comfortable side padding on full-width pages */
.stButton, .stTextInput, .stSelectbox, .stSlider, .stFileUploader,
.stForm, .stTabs, .stAlert, .stProgress, .stImage, .stCaption,
[data-testid="stMarkdownContainer"] {
    /* handled by parent .inner where needed */
}
/* Dashboard content area */
.dash-inner {
    max-width: 1280px;
    margin: 0 auto;
    padding: 12px 48px 40px;
    box-sizing: border-box;
}

/* -------------------- Symptom Checker Styling -------------------- */
.symptom-hero-card {
    background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 55%, #0284c7 100%);
    border-radius: 20px;
    padding: 30px 28px;
    color: white !important;
    box-shadow: 0 16px 36px rgba(37, 99, 235, 0.22);
    margin-bottom: 24px;
}
.symptom-hero-card * { color: white !important; }
.symptom-hero-card h2 { font-size: 24px !important; margin: 0 0 8px !important; font-weight: 800 !important; }
.symptom-hero-card p { font-size: 14.5px !important; opacity: 0.93; line-height: 1.6 !important; margin: 0 !important; }

.emergency-card {
    background: linear-gradient(135deg, #7f1d1d, #b91c1c);
    border: 2px solid #ef4444;
    border-radius: 16px;
    padding: 22px 24px;
    color: white !important;
    box-shadow: 0 12px 30px rgba(185, 28, 28, 0.35);
    margin: 18px 0;
}
.emergency-card * { color: white !important; }
.emergency-card h3 { font-size: 20px !important; font-weight: 800 !important; margin: 0 0 8px !important; color: #fef2f2 !important; }
.emergency-card ul { margin: 8px 0 0 20px; padding: 0; }
.emergency-card li { font-size: 14px; line-height: 1.6; }

.diagnosis-hero {
    background: var(--white);
    border: 2px solid #93c5fd;
    border-radius: 18px;
    padding: 24px 26px;
    box-shadow: var(--shadow);
    margin: 18px 0;
}
.diagnosis-hero-title {
    font-size: 22px;
    font-weight: 800;
    color: var(--text-dark) !important;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 6px;
}
.diagnosis-cat {
    display: inline-block;
    padding: 4px 12px;
    background: #eff6ff;
    color: #2563eb !important;
    border-radius: 999px;
    font-weight: 700;
    font-size: 12px;
    margin-bottom: 12px;
}
.symptom-chip {
    display: inline-block;
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    color: #334155 !important;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 12.5px;
    font-weight: 600;
    margin: 3px 4px 3px 0;
}
.specialist-box {
    background: linear-gradient(145deg, #f0fdf4, #dcfce7);
    border: 1.5px solid #86efac;
    border-radius: 16px;
    padding: 22px 24px;
    margin: 16px 0;
}
.specialist-box h3 {
    color: #166534 !important;
    font-size: 20px !important;
    margin: 0 0 8px !important;
    font-weight: 800 !important;
}
.specialist-box p {
    color: #14532d !important;
    font-size: 14px !important;
    line-height: 1.65 !important;
    margin: 0 !important;
}
.urgency-badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 13px;
    margin-top: 10px;
}
.urgency-emergency { background: #fee2e2; color: #991b1b !important; }
.urgency-high { background: #ffedd5; color: #9a3412 !important; }
.urgency-moderate { background: #fef9c3; color: #854d0e !important; }
.urgency-routine { background: #dcfce7; color: #166534 !important; }

.section-box {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px 22px;
    box-shadow: var(--shadow);
    margin: 14px 0;
}
.section-box h4 {
    margin: 0 0 12px !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    color: var(--text-dark) !important;
}
.section-box ul {
    margin: 0 0 0 20px;
    padding: 0;
}
.section-box li {
    font-size: 14px;
    color: var(--text-muted) !important;
    line-height: 1.7;
    margin-bottom: 4px;
}
.diff-item {
    background: #f8fafc;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 10px;
}

/* Responsive */
@media (max-width: 900px) {
    .hero { grid-template-columns: 1fr; gap: 24px; text-align: center; padding: 0 24px; }
    .hero-left p { margin-left: auto; margin-right: auto; }
    .hero-left h1 { font-size: 32px !important; }
    .service-grid { grid-template-columns: 1fr; }
    .service-grid-2 { grid-template-columns: 1fr; }
    .portal-grid { grid-template-columns: 1fr; }
    .feat-grid { grid-template-columns: 1fr 1fr; padding: 0 24px; }
    .navbar { padding: 14px 24px; }
    .navbar .nav-links { display: none; }
    .top-bar { padding: 10px 24px; }
    .section-head, .services-wrap, .portal-wrap { padding-left: 24px; padding-right: 24px; }
}
</style>
""")


# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "page": "landing",
    "user": None,
    "doctor": None,
    "last_diagnosis_id": None,
    "last_result": None,
    "last_confidence": None,
    "hospitals": None,
    "symptom_result": None,
    "landing_symptom_result": None,
    "symptom_saved_id": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def go_to(page_name):
    st.session_state.page = page_name
    st.rerun()


def logout():
    st.session_state.user = None
    st.session_state.doctor = None
    st.session_state.last_diagnosis_id = None
    st.session_state.last_result = None
    st.session_state.last_confidence = None
    st.session_state.hospitals = None
    st.session_state.symptom_result = None
    st.session_state.landing_symptom_result = None
    st.session_state.symptom_saved_id = None
    go_to("landing")


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():
    segmentation_model = tf.keras.models.load_model("unet_lung_segmentation.keras")
    classifier_model = tf.keras.models.load_model("pneumonia_classifier.keras")
    return segmentation_model, classifier_model


# =========================================================
# TOP BAR + NAV
# =========================================================

def show_top_bar():
    render_html("""
    <div class="top-bar">
        <div class="left">
            <span>f</span><span>𝕏</span><span>📷</span><span>📌</span>
        </div>
        <div class="right">
            <span>📞 +1-123-556-5523</span>
            <span>✉️ support@healthconnect.ai</span>
        </div>
    </div>
    """)


def show_navbar():
    render_html("""
    <div class="navbar">
        <div class="brand">Health<span>Connect</span>-AI</div>
        <div class="nav-links">
            <span>Home</span>
            <span>About</span>
            <span>Doctors</span>
            <span>Services</span>
            <span>AI Screening</span>
        </div>
    </div>
    """)


# =========================================================
# PAGE: LANDING
# =========================================================

def page_landing():
    show_top_bar()
    show_navbar()

    # Hero — full-width section
    render_html("""
    <div class="hero-section">
        <div class="hero">
            <div class="hero-left">
                <h1>Your Partner In Health<br>and Wellness</h1>
                <p>
                    AI-powered chest X-ray analysis with lung segmentation and pneumonia detection.
                    Secure patient access, specialist recommendations, and real hospital token booking
                    — all in one trusted portal.
                </p>
            </div>
            <div class="hero-right">
                <div class="hero-card">
                    <div class="hero-emoji">🩺</div>
                    <div class="hero-badge-float top-right">🫁 AI Lung Scan</div>
                    <div class="hero-badge-float bottom-left">✅ 98% Accuracy</div>
                </div>
            </div>
        </div>
    </div>
    """)

    # Primary CTA — full width centered
    render_html('<div class="inner" style="text-align:center;margin:8px auto 28px;">')
    c1, c2, c3 = st.columns([1.2, 1.4, 1.2])
    with c2:
        if st.button("BOOK AN APPOINTMENT / START SCREENING", key="hero_cta"):
            go_to("patient_login")
    render_html('</div>')

    # Our Healthcare Service
    render_html("""
    <div class="section-head">
        <h2>Our Healthcare Services</h2>
        <p>End-to-end respiratory care and symptom triage powered by AI diagnostics and real hospital networks.</p>
    </div>
    <div class="services-wrap">
        <div class="service-grid">
            <div class="svc-card">
                <div class="icon-circle">🩺</div>
                <h4>AI Symptom Checker</h4>
                <p>Input symptoms like fever, chest pain, or cough to get instant disease predictions and specialist doctor recommendations.</p>
            </div>
            <div class="svc-card">
                <div class="icon-circle">🫁</div>
                <h4>Lung Segmentation</h4>
                <p>U-Net based precise lung mask prediction from chest X-rays for accurate pulmonary analysis.</p>
            </div>
            <div class="svc-card">
                <div class="icon-circle">🧠</div>
                <h4>Pneumonia Detection</h4>
                <p>CNN classifier for rapid NORMAL vs PNEUMONIA screening with confidence scores.</p>
            </div>
        </div>
        <div class="service-grid">
            <div class="svc-card">
                <div class="icon-circle">👨‍⚕️</div>
                <h4>Doctor Referral & Triage</h4>
                <p>Matched specialist recommendations (Pulmonologist, Cardiologist, ENT) with clinical test checklists.</p>
            </div>
            <div class="svc-card">
                <div class="icon-circle">🏥</div>
                <h4>Hospital Network</h4>
                <p>Find real nearby hospitals with live ratings, directions and simulated bed availability insights.</p>
            </div>
            <div class="svc-card">
                <div class="icon-circle">🎫</div>
                <h4>Token Booking & Records</h4>
                <p>Secure token booking linked directly to your AI diagnosis and symptom assessment records.</p>
            </div>
        </div>
    </div>
    """)

    # Interactive Quick Symptom Checker Preview on Landing Page
    render_html("""
    <div class="section-head" style="margin-top:36px;">
        <h2>🔍 Quick AI Symptom Checker & Doctor Recommendation</h2>
        <p>Try it right now: enter symptoms such as fever, chest pain, or cough to see the AI specialist recommendation.</p>
    </div>
    """)

    render_html('<div class="inner" style="max-width:960px;margin:0 auto 36px;">')
    with st.container():
        render_html("""
        <div class="diagnosis-hero" style="border-color:#3b82f6;">
            <h3 style="margin-top:0;color:var(--primary);">🩺 Interactive Symptom Analysis Preview</h3>
            <p style="color:var(--text-muted);font-size:14px;">
                Type symptoms in plain words (e.g. <i>"I have had high fever for 3 days and sharp chest pain with cough"</i> or <i>"fever, chest pain"</i>).
            </p>
        </div>
        """)
        quick_sym_text = st.text_input(
            "📝 Enter your symptoms:",
            placeholder="e.g. fever, chest pain, shortness of breath, productive cough",
            key="landing_quick_symptoms"
        )
        col_q1, col_q2, col_q3 = st.columns([1, 1.5, 1])
        with col_q2:
            if st.button("🔍 GET DOCTOR RECOMMENDATION", key="landing_sym_btn"):
                if not quick_sym_text.strip():
                    st.warning("⚠️ Please enter at least one symptom (e.g. fever, chest pain).")
                else:
                    with st.spinner("🧠 Analyzing symptoms and evaluating medical conditions..."):
                        extracted = symptom_checker.extract_symptoms_from_text(quick_sym_text)
                        if not extracted:
                            # Try direct tokenization
                            tokens = [t.strip().lower() for t in quick_sym_text.replace(",", " ").split() if t.strip()]
                            for tok in tokens:
                                if tok in symptom_checker.SYMPTOM_ALIASES:
                                    extracted.append(symptom_checker.SYMPTOM_ALIASES[tok])

                        if not extracted:
                            st.warning("Could not clearly identify recognized symptoms. Please try keywords like 'fever', 'chest pain', 'cough', 'wheezing', etc.")
                        else:
                            res = symptom_checker.analyze_symptoms(extracted, duration_days=2, severity_rating=6)
                            st.session_state.landing_symptom_result = res

        if st.session_state.landing_symptom_result:
            render_symptom_result_view(st.session_state.landing_symptom_result, show_save_button=False, key_prefix="landing_res")
            st.info("💡 To save your symptom check to your medical records and book appointments, please log in as a patient.")
    render_html('</div>')

    # Feature metrics
    render_html("""
    <div class="feat-grid">
        <div class="feat-card">
            <div class="label">AI Diagnostics</div>
            <div class="value">98%</div>
            <p class="note">Accuracy on lung segmentation & pneumonia prediction</p>
        </div>
        <div class="feat-card">
            <div class="label">Hospital Network</div>
            <div class="value">20+</div>
            <p class="note">Real hospitals searchable with live ratings</p>
        </div>
        <div class="feat-card">
            <div class="label">Secure Access</div>
            <div class="value">HIPAA</div>
            <p class="note">Strong authentication & patient data protection</p>
        </div>
        <div class="feat-card">
            <div class="label">Care Support</div>
            <div class="value">24/7</div>
            <p class="note">Always-ready AI screening & coordination</p>
        </div>
    </div>
    """)

    # Choose portal
    render_html("""
    <div class="section-head">
        <h2>Choose Your Access Portal</h2>
        <p>Sign in as a patient for AI screening or as a doctor to monitor cases.</p>
    </div>
    """)

    render_html('<div class="portal-wrap">')
    col1, col2 = st.columns(2)
    with col1:
        render_html("""
        <div class="portal-card">
            <div class="p-icon">🧑‍🤝‍🧑</div>
            <h3>Patient Access</h3>
            <p>Upload your chest X-ray, get AI analysis, find nearby hospitals and book care tokens.</p>
        </div>
        """)
        if st.button("👤 Continue as Patient", key="landing_patient"):
            go_to("patient_login")

    with col2:
        render_html("""
        <div class="portal-card">
            <div class="p-icon">🩺</div>
            <h3>Doctor Access</h3>
            <p>Review pneumonia diagnoses, monitor patients and validate hospital token bookings.</p>
        </div>
        """)
        if st.button("🩺 Continue as Doctor", key="landing_doctor"):
            go_to("doctor_login")
    render_html('</div>')

    # Footer note
    render_html("""
    <div class="inner" style="text-align:center;padding:24px 0 40px;color:#7a8a9e;font-size:13px;">
        ⚠️ Educational and research purpose only. This application is not a medical diagnosis system.
    </div>
    """)
    if not DB_READY:
        st.error(f"Database not connected: {DB_ERROR}")


# =========================================================
# AUTH PAGES
# =========================================================

def page_patient_register():
    show_top_bar()
    show_navbar()

    render_html("""
    <div class="auth-wrap">
        <div class="auth-head">
            <div class="auth-icon-box">🧑‍🤝‍🧑</div>
            <h2>Patient Registration</h2>
            <p>Create your secure account to access AI lung screening and hospital coordination.</p>
        </div>
    </div>
    """)

    with st.form("patient_register_form"):
        full_name = st.text_input("Full Name", placeholder="Enter your full name")
        email = st.text_input("Email", placeholder="you@example.com")
        password = st.text_input("Password", type="password", placeholder="Minimum 6 characters")
        confirm_password = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")
        submitted = st.form_submit_button("✅ Create Account")

    if submitted:
        if not full_name.strip() or not email.strip() or not password:
            st.warning("⚠️ Please fill in all fields.")
        elif password != confirm_password:
            st.warning("⚠️ Passwords do not match.")
        elif len(password) < 6:
            st.warning("⚠️ Password should be at least 6 characters.")
        else:
            success, message = db.register_user(full_name.strip(), email.strip().lower(), password)
            if success:
                st.success(f"✅ {message}")
                st.session_state.page = "patient_login"
                st.rerun()
            else:
                st.error(f"❌ {message}")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("⬅️ Back to Home", key="preg_back"):
            go_to("landing")
    with c2:
        if st.button("Already have an account? Log in", key="preg_to_login"):
            go_to("patient_login")


def page_patient_login():
    show_top_bar()
    show_navbar()

    render_html("""
    <div class="auth-wrap">
        <div class="auth-head">
            <div class="auth-icon-box">🔐</div>
            <h2>Patient Sign-In</h2>
            <p>Access your dashboard to upload X-rays, view AI results, and manage hospital tokens.</p>
        </div>
    </div>
    """)

    with st.form("patient_login_form"):
        email = st.text_input("Email / User ID", placeholder="Enter your registered email")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        submitted = st.form_submit_button("🔓 Log In")

    if submitted:
        if not email.strip() or not password:
            st.warning("⚠️ Please enter both email and password.")
        else:
            user, message = db.login_user(email.strip().lower(), password)
            if user:
                st.session_state.user = user
                st.success(f"✅ Welcome back, {user['full_name']}!")
                go_to("patient_dashboard")
            else:
                st.error(f"❌ {message}")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("⬅️ Back to Home", key="plog_back"):
            go_to("landing")
    with c2:
        if st.button("New here? Create an account", key="plog_to_reg"):
            go_to("patient_register")


def page_doctor_register():
    show_top_bar()
    show_navbar()

    render_html("""
    <div class="auth-wrap">
        <div class="auth-head">
            <div class="auth-icon-box">🩺</div>
            <h2>Doctor Registration</h2>
            <p>Join the care network to monitor pneumonia cases and support patient coordination.</p>
        </div>
    </div>
    """)

    with st.form("doctor_register_form"):
        full_name = st.text_input("Full Name", placeholder="Dr. Full Name")
        email = st.text_input("Email", placeholder="doctor@example.com")
        specialization = st.selectbox(
            "Specialization",
            ["Pulmonologist", "General Physician", "Internal Medicine", "Chest Physician"]
        )
        password = st.text_input("Password", type="password", placeholder="Minimum 6 characters")
        confirm_password = st.text_input("Confirm Password", type="password", placeholder="Re-enter password")
        submitted = st.form_submit_button("✅ Create Doctor Account")

    if submitted:
        if not full_name.strip() or not email.strip() or not password:
            st.warning("⚠️ Please fill in all fields.")
        elif password != confirm_password:
            st.warning("⚠️ Passwords do not match.")
        elif len(password) < 6:
            st.warning("⚠️ Password should be at least 6 characters.")
        else:
            success, message = db.register_doctor(
                full_name.strip(), email.strip().lower(), password, specialization
            )
            if success:
                st.success(f"✅ {message}")
                st.session_state.page = "doctor_login"
                st.rerun()
            else:
                st.error(f"❌ {message}")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("⬅️ Back to Home", key="dreg_back"):
            go_to("landing")
    with c2:
        if st.button("Already have a doctor account? Log in", key="dreg_to_login"):
            go_to("doctor_login")


def page_doctor_login():
    show_top_bar()
    show_navbar()

    render_html("""
    <div class="auth-wrap">
        <div class="auth-head">
            <div class="auth-icon-box">🔒</div>
            <h2>Doctor Portal</h2>
            <p>Sign in to review patient diagnoses, pneumonia cases, and token bookings.</p>
        </div>
    </div>
    """)

    with st.form("doctor_login_form"):
        email = st.text_input("Email / User ID", placeholder="Enter your doctor email")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        submitted = st.form_submit_button("🔓 Log In")

    if submitted:
        if not email.strip() or not password:
            st.warning("⚠️ Please enter both email and password.")
        else:
            doctor, message = db.login_doctor(email.strip().lower(), password)
            if doctor:
                st.session_state.doctor = doctor
                st.success(f"✅ Welcome back, Dr. {doctor['full_name']}!")
                go_to("doctor_dashboard")
            else:
                st.error(f"❌ {message}")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("⬅️ Back to Home", key="dlog_back"):
            go_to("landing")
    with c2:
        if st.button("New doctor? Register here", key="dlog_to_reg"):
            go_to("doctor_register")


# =========================================================
# HELPERS
# =========================================================

def bed_pill(available_beds, total_beds):
    ratio = available_beds / total_beds if total_beds else 0
    if ratio > 0.3:
        css_class, label = "bed-good", "🟢 Good Availability"
    elif ratio > 0.1:
        css_class, label = "bed-low", "🟡 Limited Availability"
    else:
        css_class, label = "bed-full", "🔴 Nearly Full"
    return f'<span class="bed-pill {css_class}">{label}</span>'


DOCTOR_SUGGESTIONS = {
    "PNEUMONIA": {
        "icon": "🫁",
        "title": "Pulmonologist / Chest Physician",
        "description": (
            "A pulmonologist specializes in respiratory and lung-related "
            "conditions, including pneumonia. Please consult a qualified "
            "healthcare professional for proper clinical evaluation, "
            "diagnosis and treatment."
        ),
    },
    "NORMAL": {
        "icon": "🩺",
        "title": "General Physician",
        "description": (
            "No signs of pneumonia were detected by the AI model. If you "
            "still have symptoms like cough, fever, or breathing difficulty, "
            "a general physician can evaluate you and refer you to a "
            "specialist if needed."
        ),
    },
}


def show_doctor_suggestion(result):
    info = DOCTOR_SUGGESTIONS.get(result, DOCTOR_SUGGESTIONS["NORMAL"])
    st.markdown("## 👨‍⚕️ Recommended Doctor to Consult")
    render_html(f"""
    <div class="doctor-card">
        <h3>{info['icon']} {info['title']}</h3>
        <p>{info['description']}</p>
    </div>
    """)


def render_symptom_result_view(result, show_save_button=True, user=None, key_prefix="symptom_res"):
    """
    Renders rich clinical symptom analysis, disease prediction,
    and doctor recommendation card.
    """
    if not result or not result.get("success"):
        st.warning("⚠️ No symptom analysis data to display.")
        return

    top = result["top_prediction"]
    red_flags = result.get("red_flags_detected", [])
    has_emergency = result.get("has_emergency", False)

    # 1. EMERGENCY ALERT BANNER (If red flags exist)
    if has_emergency or red_flags:
        flags_html = "".join([f"<li>🚨 <b>{rf}</b></li>" for rf in red_flags]) if red_flags else "<li>🚨 Severe emergency indicator detected.</li>"
        render_html(f"""
        <div class="emergency-card">
            <h3>🚨 EMERGENCY MEDICAL ALERT — IMMEDIATE ATTENTION NEEDED</h3>
            <p>One or more critical high-risk symptoms were detected that require immediate medical evaluation:</p>
            <ul>{flags_html}</ul>
            <p style="margin-top:10px;font-weight:700;">
                📞 Please call emergency services (911 / 112 / 108) or go to the nearest hospital Emergency Room right away.
            </p>
        </div>
        """)

    # 2. PRIMARY DIAGNOSIS CARD
    conf = top["confidence"]
    sev = top["severity"]
    sev_class = "urgency-emergency" if ("Critical" in sev or "Emergency" in sev) else ("urgency-high" if "High" in sev else ("urgency-moderate" if "Moderate" in sev else "urgency-routine"))

    render_html(f"""
    <div class="diagnosis-hero">
        <div class="diagnosis-hero-title">
            <span>{top['icon']}</span>
            <span>Primary Suspected Condition: <b>{top['name']}</b></span>
        </div>
        <span class="diagnosis-cat">📂 {top['category']}</span>
        <span class="urgency-badge {sev_class}">⚠️ Severity: {sev}</span>
        <p style="color:var(--text-muted);font-size:14.5px;line-height:1.7;margin:12px 0 16px;">
            {top['summary']}
        </p>
        <div style="margin-bottom:8px;"><b>🎯 AI Clinical Match Confidence:</b> {conf:.1f}%</div>
    </div>
    """)
    st.progress(int(min(max(conf, 0), 100)))

    # Matched Symptoms Chips
    matched_labels = [symptom_checker.SYMPTOM_LABELS_DICT.get(k, k.replace("_", " ").title()) for k in top.get("matched_symptoms", [])]
    if matched_labels:
        chips_html = "".join([f'<span class="symptom-chip">✓ {label}</span>' for label in matched_labels])
        render_html(f"""
        <div style="margin:12px 0 16px;">
            <div style="font-size:13px;font-weight:700;color:var(--text-muted);margin-bottom:6px;">MATCHED PATIENT SYMPTOMS:</div>
            <div>{chips_html}</div>
        </div>
        """)

    # 3. RECOMMENDED DOCTOR / SPECIALIST CARD (Primary Feature)
    urgency_badge_class = "urgency-emergency" if "EMERGENCY" in top['urgency_level'].upper() else ("urgency-high" if "High" in top['urgency_level'] else ("urgency-moderate" if "Moderate" in top['urgency_level'] else "urgency-routine"))

    render_html(f"""
    <div class="specialist-box">
        <h3>{top['specialist_icon']} Recommended Doctor: {top['specialist']}</h3>
        <p><b>Clinical Rationale:</b> {top['specialist_reason']}</p>
        <div class="urgency-badge {urgency_badge_class}">
            ⏱️ Consultation Urgency: {top['urgency_level']}
        </div>
    </div>
    """)

    # 4. TWO COLUMNS: DIAGNOSTIC TESTS & PRECAUTIONS
    col_tests, col_prec = st.columns(2)
    with col_tests:
        tests_li = "".join([f"<li>🔬 {t}</li>" for t in top["tests"]])
        render_html(f"""
        <div class="section-box">
            <h4>📋 Recommended Diagnostic Tests</h4>
            <ul>{tests_li}</ul>
        </div>
        """)

    with col_prec:
        prec_li = "".join([f"<li>💊 {p}</li>" for p in top["precautions"]])
        render_html(f"""
        <div class="section-box">
            <h4>🛡️ Home Care & Precautions</h4>
            <ul>{prec_li}</ul>
        </div>
        """)

    # 5. PULMONARY CALLOUT (If Chest / Lung condition)
    if top.get("is_pulmonary"):
        st.info(
            f"🫁 **AI Chest X-ray Screening Recommended:** Since **{top['name']}** involves the lungs or respiratory system, "
            "a Chest X-ray is strongly indicated. You can upload your Chest X-ray in the **'🩻 Chest X-ray AI Screening'** tab "
            "for instant U-Net lung segmentation and pneumonia detection!"
        )

    # 6. DIFFERENTIAL DIAGNOSES (Alternative possibilities)
    differentials = result.get("differential_diagnoses", [])
    if differentials:
        with st.expander("📊 View Differential Diagnoses & Alternative Possibilities", expanded=False):
            for diff in differentials:
                render_html(f"""
                <div class="diff-item">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <b>{diff['icon']} {diff['name']}</b>
                        <span style="font-weight:700;color:var(--primary);">{diff['confidence']:.1f}% match</span>
                    </div>
                    <div style="font-size:13px;color:var(--text-muted);margin:4px 0;">
                        👨‍⚕️ Specialist: <b>{diff['specialist']}</b> | Severity: <b>{diff['severity']}</b>
                    </div>
                    <div style="font-size:13px;color:var(--text-muted);">{diff['summary']}</div>
                </div>
                """)

    # 7. SAVE RECORD TO DB (If user logged in)
    if show_save_button and user:
        st.markdown("---")
        col_s1, col_s2 = st.columns([1.5, 1])
        with col_s1:
            if st.button("💾 Save Symptom Assessment to My Medical Records", key=f"{key_prefix}_save"):
                try:
                    symptoms_str = ", ".join(result.get("input_symptom_labels", []))
                    check_id = db.save_symptom_check(
                        user_id=user["id"],
                        symptoms=symptoms_str,
                        predicted_disease=top["name"],
                        severity=top["severity"],
                        confidence=top["confidence"],
                        recommended_doctor=top["specialist"],
                        doctor_advice=top["specialist_reason"]
                    )
                    st.session_state.symptom_saved_id = check_id
                    st.success("✅ Symptom check saved successfully to your Medical Records!")
                except Exception as ex:
                    st.error(f"❌ Could not save to database: {str(ex)}")

    st.caption("⚠️ Medical Disclaimer: This AI tool is for preliminary educational triage and doctor recommendation guidance only. It is not a formal medical diagnosis. Always consult a licensed physician for clinical examination.")


# =========================================================
# PATIENT DASHBOARD
# =========================================================

def page_patient_dashboard():
    user = st.session_state.user
    show_top_bar()
    show_navbar()

    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.markdown(f"### 👋 Welcome, {user['full_name']}")
    with col_b:
        if st.button("🚪 Logout", key="pat_logout"):
            logout()

    tab_symptoms, tab_analyze, tab_tokens, tab_history = st.tabs([
        "� AI Symptom Checker & Doctor Recommendation",
        "� Chest X-ray AI Screening",
        "🎫 My Booked Tokens",
        "📋 My Health Records & History"
    ])

    with tab_symptoms:
        render_html("""
        <div class="symptom-hero-card">
            <h2>🩺 AI Symptom Assessment & Specialist Recommendation</h2>
            <p>
                Describe your symptoms (e.g., fever, sharp chest pain, chronic cough, shortness of breath)
                or select them from the clinical list below. The AI evaluates possible conditions, calculates risk levels,
                recommends the appropriate medical specialist (doctor), and suggests diagnostic tests & home precautions.
            </p>
        </div>
        """)

        col_in1, col_in2 = st.columns([1.3, 1])
        with col_in1:
            st.markdown("#### ✍️ 1. Describe Your Symptoms in Plain Words")
            user_free_text = st.text_area(
                "What symptoms are you experiencing?",
                placeholder="Example: I have had high fever for 3 days with sharp chest pain when breathing, chills, and coughing up yellow phlegm...",
                height=130,
                key="pat_free_symptom_text"
            )
        with col_in2:
            st.markdown("#### 🔍 2. Or Select Common Symptoms")
            selected_multi = st.multiselect(
                "Choose symptoms from clinical checklist:",
                options=list(symptom_checker.SYMPTOM_LABELS_DICT.keys()),
                format_func=lambda x: symptom_checker.SYMPTOM_LABELS_DICT.get(x, x),
                key="pat_multi_symptoms"
            )

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            duration_days = st.slider("⏳ How many days have you had these symptoms?", min_value=1, max_value=30, value=3, key="pat_duration")
        with col_d2:
            severity_rating = st.slider("�️ Discomfort / Pain Severity Scale (1 = Mild, 10 = Severe/Unbearable)", min_value=1, max_value=10, value=5, key="pat_severity")

        col_btn1, col_btn2, col_btn3 = st.columns([1, 1.8, 1])
        with col_btn2:
            analyze_sym_clicked = st.button("🚀 ANALYZE SYMPTOMS & GET DOCTOR RECOMMENDATION", key="pat_analyze_sym_btn")

        if analyze_sym_clicked:
            combined_symptoms = set(selected_multi)
            if user_free_text.strip():
                extracted = symptom_checker.extract_symptoms_from_text(user_free_text)
                combined_symptoms.update(extracted)
                if not extracted and not selected_multi:
                    tokens = [t.strip().lower() for t in user_free_text.replace(",", " ").split() if t.strip()]
                    for tok in tokens:
                        if tok in symptom_checker.SYMPTOM_ALIASES:
                            combined_symptoms.add(symptom_checker.SYMPTOM_ALIASES[tok])

            if not combined_symptoms:
                st.warning("⚠️ Please enter or select at least one symptom (e.g. fever, chest pain, cough).")
            else:
                with st.spinner("🧠 Analyzing clinical symptom profile and matching specialist..."):
                    sym_result = symptom_checker.analyze_symptoms(
                        list(combined_symptoms),
                        duration_days=duration_days,
                        severity_rating=severity_rating
                    )
                    st.session_state.symptom_result = sym_result
                    try:
                        top_pred = sym_result["top_prediction"]
                        symptoms_str = ", ".join(sym_result.get("input_symptom_labels", []))
                        check_id = db.save_symptom_check(
                            user_id=user["id"],
                            symptoms=symptoms_str,
                            predicted_disease=top_pred["name"],
                            severity=top_pred["severity"],
                            confidence=top_pred["confidence"],
                            recommended_doctor=top_pred["specialist"],
                            doctor_advice=top_pred["specialist_reason"]
                        )
                        st.session_state.symptom_saved_id = check_id
                    except Exception:
                        pass

        if st.session_state.symptom_result:
            st.markdown("---")
            render_symptom_result_view(st.session_state.symptom_result, show_save_button=False, user=user, key_prefix="pat_sym_view")
            st.success("✅ This assessment has been securely saved to your '📋 My Health Records & History' tab.")

    with tab_analyze:
        uploaded_file = st.file_uploader(
            "📤 Upload Chest X-ray Image", type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.markdown("#### 🖼️ Selected X-ray")
            st.image(image, caption="Uploaded Chest X-ray", use_container_width=True)

            if st.button("🚀 START AI ANALYSIS"):
                try:
                    with st.spinner("🔄 Loading AI models..."):
                        segmentation_model, classifier_model = load_models()

                    st.info("🫁 Performing lung segmentation...")
                    gray_image = image.convert("L")
                    seg_image = gray_image.resize((256, 256))
                    seg_array = np.array(seg_image) / 255.0
                    seg_input = np.expand_dims(seg_array, axis=(0, -1))
                    prediction = segmentation_model.predict(seg_input, verbose=0)
                    predicted_mask = prediction[0]
                    if predicted_mask.shape[-1] == 1:
                        predicted_mask = predicted_mask[:, :, 0]
                    predicted_mask = (predicted_mask > 0.5).astype(np.uint8)

                    st.info("🧠 Detecting pneumonia...")
                    classifier_image = image.resize((224, 224))
                    classifier_array = np.array(classifier_image) / 255.0
                    classifier_input = np.expand_dims(classifier_array, axis=0)
                    classifier_prediction = classifier_model.predict(classifier_input, verbose=0)[0][0]
                    confidence = float(classifier_prediction)

                    if confidence >= 0.5:
                        result = "PNEUMONIA"
                        confidence_percentage = confidence * 100
                    else:
                        result = "NORMAL"
                        confidence_percentage = (1 - confidence) * 100

                    diagnosis_id = db.save_diagnosis(user["id"], result, confidence_percentage)
                    st.session_state.last_diagnosis_id = diagnosis_id
                    st.session_state.last_result = result
                    st.session_state.last_confidence = confidence_percentage
                    st.session_state.hospitals = None

                    render_html('<div class="success-box">✅ AI ANALYSIS COMPLETED SUCCESSFULLY</div>')

                    st.markdown("## 📊 AI ANALYSIS RESULTS")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown("**🩻 Original X-ray**")
                        st.image(image, use_container_width=True)
                    with col2:
                        st.markdown("**🫁 Predicted Lung Mask**")
                        fig, ax = plt.subplots()
                        ax.imshow(predicted_mask, cmap="gray")
                        ax.axis("off")
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                    with col3:
                        st.markdown("**🔬 Analyzed X-ray**")
                        st.image(image, use_container_width=True)

                    if result == "PNEUMONIA":
                        render_html('<div class="warning-box">🦠 PNEUMONIA DETECTED</div>')
                    else:
                        render_html('<div class="success-box">✅ NO PNEUMONIA DETECTED</div>')

                    col1, col2 = st.columns(2)
                    with col1:
                        render_html(f"""
                        <div class="result-card">
                            <div class="result-title">🧠 AI PREDICTION</div>
                            <div class="result-value">{result}</div>
                        </div>
                        """)
                    with col2:
                        render_html(f"""
                        <div class="result-card">
                            <div class="result-title">🎯 CONFIDENCE</div>
                            <div class="result-value">{confidence_percentage:.2f}%</div>
                        </div>
                        """)

                    st.progress(int(min(max(confidence_percentage, 0), 100)))
                    st.warning(
                        "⚠️ IMPORTANT MEDICAL DISCLAIMER: This is an AI-based educational and "
                        "research project, not a substitute for professional medical diagnosis. "
                        "Always consult a qualified healthcare professional."
                    )

                except Exception as e:
                    st.error(f"❌ Error occurred: {str(e)}")

        if st.session_state.last_result and st.session_state.last_diagnosis_id:
            st.markdown("---")
            show_doctor_suggestion(st.session_state.last_result)

        if st.session_state.last_result == "PNEUMONIA" and st.session_state.last_diagnosis_id:
            st.markdown("---")
            st.markdown("## 📍 Find Real Nearby Hospitals")
            st.info("Enter your city or area. Google Places API will search real nearby hospitals.")

            location = st.text_input("📌 Enter your location", placeholder="Example: Chandigarh, India", key="hospital_location")
            radius_km = st.slider("🔎 Search radius (km)", min_value=1, max_value=25, value=5, key="hospital_radius")

            if st.button("🔍 SEARCH NEARBY HOSPITALS"):
                if not location.strip():
                    st.warning("⚠️ Please enter your location.")
                else:
                    with st.spinner("📍 Finding location..."):
                        user_lat, user_lon = utils.get_coordinates(location)
                    if user_lat is None:
                        st.error("❌ Location not found.")
                    else:
                        st.success(f"📍 Location found: {user_lat:.5f}, {user_lon:.5f}")
                        with st.spinner("🏥 Searching nearby hospitals..."):
                            hospitals = utils.find_nearby_hospitals(user_lat, user_lon, radius_km * 1000)
                        st.session_state.hospitals = hospitals

            hospitals = st.session_state.hospitals
            if hospitals is not None:
                if not hospitals:
                    st.warning("No nearby hospitals found.")
                else:
                    st.success(f"🏥 Found {len(hospitals)} nearby hospitals.")
                    st.markdown("### 🏥 Nearby Hospitals")

                    for index, hospital in enumerate(hospitals[:20], start=1):
                        if hospital["open_now"] is True:
                            availability = "🟢 Open Now"
                        elif hospital["open_now"] is False:
                            availability = "🔴 Closed Now"
                        else:
                            availability = "ℹ️ Status unavailable"

                        render_html(f"""
                        <div class="hospital-card">
                            <div class="hospital-name">{index}. 🏥 {hospital["name"]}</div>
                            <div class="hospital-info">
                                📏 Distance: <b>{hospital["distance"]:.2f} km</b><br>
                                📍 Address: {hospital["address"]}<br>
                                ⭐ Rating: {hospital["rating"]} ({hospital["reviews"]} reviews)<br>
                                {availability}<br>
                                🛏️ Beds: <b>{hospital["available_beds"]} available</b> / {hospital["total_beds"]} total
                                {bed_pill(hospital["available_beds"], hospital["total_beds"])}
                            </div>
                        </div>
                        """)

                        col1, col2, col3 = st.columns(3)
                        maps_url = (
                            "https://www.google.com/maps/dir/?api=1"
                            f"&destination={hospital['latitude']},{hospital['longitude']}"
                        )
                        appointment_url = (
                            "https://www.google.com/search?q="
                            + urllib.parse.quote(hospital["name"] + " appointment booking")
                        )
                        with col1:
                            st.link_button("🗺️ GET DIRECTIONS", maps_url)
                        with col2:
                            st.link_button("🎫 SEARCH APPOINTMENT INFO", appointment_url)
                        with col3:
                            if hospital["available_beds"] <= 0:
                                st.button("🚫 NO BEDS", key=f"nobook_{index}", disabled=True)
                            else:
                                if st.button("✅ BOOK TOKEN", key=f"book_{index}"):
                                    token_number = db.book_token(
                                        st.session_state.last_diagnosis_id,
                                        user["id"],
                                        hospital["name"],
                                        hospital["address"],
                                        hospital["place_id"],
                                    )
                                    render_html(f"""
                                    <div class="token-card">
                                        <div>🎫 Token Booked at {hospital["name"]}</div>
                                        <div class="token-number">{token_number}</div>
                                    </div>
                                    """)
                                    st.success("✅ Your token has been booked! Check the 'My Booked Tokens' tab.")

                    st.info(
                        "ℹ️ Hospital list, distance, rating and open/closed status come from Google Places API. "
                        "Bed numbers are SIMULATED for demo purposes."
                    )

    with tab_tokens:
        st.markdown("### 🎫 My Booked Tokens")
        tokens = db.get_user_tokens(user["id"])
        if not tokens:
            st.info("You haven't booked any hospital tokens yet.")
        else:
            for token in tokens:
                render_html(f"""
                <div class="patient-row">
                    <b>Token {token['token_number']}</b> — {token['hospital_name']}<br>
                    📍 {token['hospital_address'] or 'N/A'}<br>
                    🧠 Diagnosis: {token['result']} ({token['confidence']:.2f}%)<br>
                    📅 Booked: {token['created_at']}<br>
                    Status: <b>{token['status']}</b>
                </div>
                """)

    with tab_history:
        st.markdown("### 📋 My Health Records & Assessments")
        hist_col1, hist_col2 = st.columns(2)

        with hist_col1:
            st.markdown("#### 🩺 Saved Symptom & Doctor Assessments")
            try:
                symptom_records = db.get_user_symptom_checks(user["id"])
                if not symptom_records:
                    st.info("No saved symptom assessments yet. Use the 'AI Symptom Checker' tab to evaluate symptoms.")
                else:
                    for s_rec in symptom_records:
                        urgency_class = "urgency-emergency" if ("Critical" in s_rec['severity'] or "Emergency" in s_rec['severity']) else ("urgency-high" if "High" in s_rec['severity'] else "urgency-routine")
                        render_html(f"""
                        <div class="patient-row">
                            <b>🏥 Suspected: {s_rec['predicted_disease']}</b> ({s_rec['confidence']:.1f}% Match)<br>
                            👨‍⚕️ Recommended Doctor: <b>{s_rec['recommended_doctor']}</b><br>
                            <span class="urgency-badge {urgency_class}">Severity: {s_rec['severity']}</span><br>
                            📝 Reported Symptoms: <i>{s_rec['symptoms']}</i><br>
                            📅 Date: {s_rec['created_at']}
                        </div>
                        """)
            except Exception as e:
                st.error(f"Error fetching symptom records: {str(e)}")

        with hist_col2:
            st.markdown("#### 🩻 Chest X-ray Diagnoses")
            try:
                user_tokens = db.get_user_tokens(user["id"])
                if not user_tokens:
                    st.info("No X-ray scan records yet.")
                else:
                    for t_rec in user_tokens:
                        render_html(f"""
                        <div class="patient-row">
                            <b>🩻 AI Diagnosis: {t_rec['result']}</b> ({t_rec['confidence']:.2f}%)<br>
                            🎫 Hospital Token: <b>{t_rec['token_number']}</b> at {t_rec['hospital_name']}<br>
                            📍 {t_rec['hospital_address'] or 'N/A'}<br>
                            📅 Date: {t_rec['created_at']}<br>
                            Status: <b>{t_rec['status']}</b>
                        </div>
                        """)
            except Exception as e:
                st.error(f"Error fetching X-ray records: {str(e)}")


# =========================================================
# DOCTOR DASHBOARD
# =========================================================

def page_doctor_dashboard():
    doctor = st.session_state.doctor
    show_top_bar()
    show_navbar()

    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.markdown(f"### 🩺 Welcome, Dr. {doctor['full_name']} ({doctor['specialization']})")
    with col_b:
        if st.button("🚪 Logout", key="doc_logout"):
            logout()

    doc_tab1, doc_tab2 = st.tabs([
        "🫁 Confirmed Pneumonia Patients & Tokens",
        "🩺 Patient Symptom Assessments & Triage"
    ])

    with doc_tab1:
        st.markdown("## 🧾 Patients Diagnosed with Pneumonia")
        rows = db.get_pneumonia_patients_with_tokens()

        if not rows:
            st.info("No pneumonia cases have been recorded yet.")
        else:
            for row in rows:
                token_info = (
                    f"🎫 Token <b>{row['token_number']}</b> at {row['hospital_name']} "
                    f"({row['hospital_address'] or 'N/A'}) — Status: <b>{row['status']}</b>"
                    if row["token_number"]
                    else "🎫 No token booked yet"
                )
                render_html(f"""
                <div class="patient-row">
                    <b>👤 {row['full_name']}</b> ({row['email']})<br>
                    🧠 Result: <b>PNEUMONIA</b> — Confidence: {row['confidence']:.2f}%<br>
                    📅 Diagnosed: {row['diagnosed_at']}<br>
                    {token_info}
                </div>
                """)

    with doc_tab2:
        st.markdown("## 🩺 Patient Symptom Triage & Specialist Referrals")
        try:
            symptom_rows = db.get_all_symptom_checks()
            if not symptom_rows:
                st.info("No patient symptom assessments recorded yet.")
            else:
                for srow in symptom_rows:
                    urgency_class = "urgency-emergency" if ("Critical" in srow['severity'] or "Emergency" in srow['severity']) else ("urgency-high" if "High" in srow['severity'] else "urgency-routine")
                    render_html(f"""
                    <div class="patient-row">
                        <b>👤 Patient: {srow['full_name']}</b> ({srow['email']})<br>
                        🏥 Suspected Condition: <b>{srow['predicted_disease']}</b> (Confidence: {srow['confidence']:.1f}%)<br>
                        👨‍⚕️ Recommended Doctor: <b>{srow['recommended_doctor']}</b><br>
                        <span class="urgency-badge {urgency_class}">⚠️ Severity: {srow['severity']}</span><br>
                        📝 Symptoms Reported: <i>{srow['symptoms']}</i><br>
                        📅 Submitted: {srow['created_at']}
                    </div>
                    """)
        except Exception as e:
            st.error(f"Error loading symptom assessments: {str(e)}")


# =========================================================
# ROUTER
# =========================================================

page = st.session_state.page

if not DB_READY and page not in ("landing",):
    st.error(
        "❌ Cannot reach the MySQL database. Please check your .streamlit/secrets.toml "
        "[mysql] settings and that your MySQL server is running."
    )

if page == "landing":
    page_landing()
elif page == "patient_register":
    page_patient_register()
elif page == "patient_login":
    page_patient_login()
elif page == "doctor_register":
    page_doctor_register()
elif page == "doctor_login":
    page_doctor_login()
elif page == "patient_dashboard":
    if st.session_state.user:
        page_patient_dashboard()
    else:
        go_to("patient_login")
elif page == "doctor_dashboard":
    if st.session_state.doctor:
        page_doctor_dashboard()
    else:
        go_to("doctor_login")
else:
    go_to("landing")
