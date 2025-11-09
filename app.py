# app.py
"""
CrowdPitch Pro — KYC Edition (Enhanced)
- Landing page (login/signup)
- Startup onboarding (more docs + target funding)
- Checker with inline PDF/image preview
- Investor marketplace with search/filters + card grid
- Investor dashboard (my investments) and correct wallet handling
- ARIMA-based ROI prediction (6-month average)
"""

import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from PIL import Image
import hashlib
import re
from urllib.parse import urlparse
from datetime import datetime
import base64
import html
import textwrap

# ---------- Page config ----------
st.set_page_config(page_title="CrowdPitch Pro — KYC Edition", page_icon="🚀", layout="wide")
sns.set_style("darkgrid")
plt.rcParams["figure.dpi"] = 100

# ---------- CSS for cards/grid ----------
st.markdown(
    """
    <style>
    body { background: linear-gradient(180deg,#071A2A,#071622); color: #e6eef6; }
    .auth-card { background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01)); padding:18px; border-radius:10px; box-shadow:0 6px 18px rgba(0,0,0,0.45); }
    .card { background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01)); border-radius:12px; padding:12px; box-shadow:0 6px 18px rgba(0,0,0,0.45); margin-bottom:14px;}
    .grid { display:grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap:16px; }
    .badge { padding:4px 8px; border-radius:8px; font-weight:600; font-size:0.85rem; }
    .badge-approved { background-color:#16a34a; color:white; }
    .badge-pending { background-color:#f59e0b; color:#012; }
    .badge-rejected { background-color:#ef4444; color:white; }
    .progress { height:10px; background: rgba(255,255,255,0.05); border-radius:8px; overflow:hidden; margin-top:8px; }
    .progress > div { background: linear-gradient(90deg,#06b6d4,#0891b2); height:100%; }
    .small-muted { color:#9fb4c9; font-size:0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Utilities ----------
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def image_bytes(img_file) -> bytes:
    img = Image.open(img_file).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def extract_domain(url: str) -> str:
    try:
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path
        domain = domain.lower().replace("www.", "")
        domain = domain.split(':')[0]
        return domain
    except:
        return ""

def email_domain(email: str) -> str:
    parts = email.split("@")
    return parts[1].lower() if len(parts) == 2 else ""

def name_in_filename(name: str, fname: str) -> bool:
    if not name or not fname: return False
    n = re.sub(r'[^a-z0-9]', '', name.lower())
    f = re.sub(r'[^a-z0-9]', '', fname.lower())
    return n in f

def status_badge_html(status: str) -> str:
    s = (status or "").lower()
    if s == "approved":
        return '<span class="badge badge-approved">🟢 Approved</span>'
    if s == "rejected":
        return '<span class="badge badge-rejected">🔴 Rejected</span>'
    return '<span class="badge badge-pending">🟡 Pending</span>'

def safe_read_csv(file) -> pd.DataFrame:
    # read and canonicalize
    df = pd.read_csv(file)
    if df.shape[1] >= 2:
        df = df.iloc[:, :2]
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df

def predict_forecast_and_roi(df: pd.DataFrame, steps: int = 6):
    # try ARIMA, fallback to linear projection
    try:
        model = sm.tsa.ARIMA(df["value"], order=(1,1,1))
        res = model.fit()
        forecast = res.forecast(steps=steps)
        conf = res.get_forecast(steps=steps).conf_int()
        future_dates = pd.date_range(df["date"].iloc[-1], periods=steps+1, freq="M")[1:]
        latest = df["value"].iloc[-1]
        mean_forecast = np.mean(forecast)
        roi = (mean_forecast / latest - 1) * 100.0
        return future_dates, np.array(forecast), conf, roi
    except Exception:
        # fallback simple projection
        try:
            if len(df) < 3:
                return None, None, None, 0.0
            pct_changes = df["value"].pct_change().dropna()
            avg = pct_changes[-3:].mean() if len(pct_changes) >= 3 else pct_changes.mean()
            latest = df["value"].iloc[-1]
            vals = []
            dates = pd.date_range(df["date"].iloc[-1], periods=steps+1, freq="M")[1:]
            v = latest
            for _ in range(steps):
                v = v * (1 + avg)
                vals.append(v)
            mean_forecast = np.mean(vals)
            roi = (mean_forecast / latest - 1) * 100.0
            return dates, np.array(vals), None, roi
        except Exception:
            return None, None, None, 0.0

def embed_pdf_bytes(pdf_bytes: bytes, width="100%", height="600px"):
    # embed pdf using base64 data URI (works for moderate sizes)
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    src = f"data:application/pdf;base64,{b64}"
    html_tag = f'<iframe src="{src}" width="{width}" height="{height}"></iframe>'
    st.components.v1.html(html_tag, height=int(height.replace("px",""))+20)

def embed_image_bytes(img_bytes: bytes, width=300):
    st.image(img_bytes, width=width)

# ---------- session init ----------
if "users" not in st.session_state:
    st.session_state.users = {}
if "current_user" not in st.session_state:
    st.session_state.current_user = None
if "pitches" not in st.session_state:
    st.session_state.pitches = []
if "investments" not in st.session_state:
    st.session_state.investments = []
if "complaints" not in st.session_state:
    st.session_state.complaints = []
if "page" not in st.session_state:
    st.session_state.page = "home"

# demo & checker
if "investor_demo" not in st.session_state.users:
    st.session_state.users["investor_demo"] = {"password": hash_password("pass123"), "role": "Investor", "wallet": 10000.0}
if "startup_demo" not in st.session_state.users:
    st.session_state.users["startup_demo"] = {"password": hash_password("pass123"), "role": "Startup"}
if "checker_agent" not in st.session_state.users:
    st.session_state.users["checker_agent"] = {"password": hash_password("Check@2025!"), "role": "Checker"}

# helper ui keys
if "_view_graph" not in st.session_state: st.session_state._view_graph = None
if "_complaint_for" not in st.session_state: st.session_state._complaint_for = None
if "_checker_view" not in st.session_state: st.session_state._checker_view = None

# ---------- auth ----------
def signup(username, password, role):
    if not username or not password:
        st.warning("Enter username & password.")
        return False
    if username in st.session_state.users:
        st.warning("Username taken.")
        return False
    if role == "Checker":
        st.error("Cannot signup as Checker.")
        return False
    st.session_state.users[username] = {"password": hash_password(password), "role": role}
    if role == "Investor":
        st.session_state.users[username]["wallet"] = 10000.0
    st.success("Account created — please login.")
    return True

def login(username, password):
    u = st.session_state.users.get(username)
    if not u:
        st.error("No such user.")
        return False
    if u["password"] != hash_password(password):
        st.error("Incorrect password.")
        return False
    st.session_state.current_user = {"username": username, "role": u["role"]}
    st.session_state.page = "app"
    st.success(f"Welcome {username} ({u['role']})")
    st.rerun()
    return True

def logout():
    st.session_state.current_user = None
    st.session_state.page = "home"
    st.rerun()

# ---------- landing / cover ----------
def landing_page():
    st.markdown("<h1 style='text-align:center;color:#06b6d4;'>🚀 CrowdPitch Pro</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#9fb4c9;'>Verified Crowdfunding Platform — KYC & Manual Verification Demo</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        mode = st.radio("Action", ["Login", "Signup"], horizontal=True)
        if mode == "Login":
            st.subheader("Sign in")
            li_user = st.text_input("Username", key="li_user")
            li_pass = st.text_input("Password", type="password", key="li_pass")
            if st.button("Login"):
                if li_user and li_pass:
                    login(li_user.strip(), li_pass)
                else:
                    st.warning("Enter username and password.")
            st.markdown("<div class='small-muted'>Demo accounts: <b>investor_demo</b>/<i>pass123</i> or <b>startup_demo</b>/<i>pass123</i></div>", unsafe_allow_html=True)
        else:
            st.subheader("Create account")
            su_user = st.text_input("Choose username", key="su_user")
            su_pass = st.text_input("Choose password", type="password", key="su_pass")
            role = st.selectbox("Role", ["Startup", "Investor"], key="su_role")
            if st.button("Create account"):
                if su_user and su_pass:
                    signup(su_user.strip(), su_pass, role)
                else:
                    st.warning("Enter username & password.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    left, right = st.columns([2,1])
    with left:
        st.subheader("About CrowdPitch Pro")
        st.write(
            "CrowdPitch Pro is a demo platform to showcase startup onboarding with multi-document KYC, "
            "manual checker review, investor marketplace with predictions, and complaint workflows."
        )
        st.markdown("**Contact / Support**")
        st.write("Email: support@crowdpitch.example")
        st.write("Twitter: @crowdpitch")
    with right:
        st.subheader("FAQs")
        st.write("Is this real funding? — No, demo only.")
        st.write("Checker (hidden): `checker_agent` / `Check@2025!`")
    st.markdown("---")

# ---------- startup page (with target funding + store file bytes) ----------
def startup_page(user):
    # ... (unchanged content)
    pass

# ---------- checker page ----------
def checker_page(user):
    # ... (unchanged content)
    pass

# ---------- investor marketplace + dashboard ----------
def investor_page(user):
    # ... (unchanged content)
    pass

# ---------- main app routing ----------
def main_app():
    user = st.session_state.current_user
    st.sidebar.markdown(f"**Logged in as:** {user['username']} ({user['role']})")
    if user["role"] == "Investor":
        st.sidebar.metric("Wallet (₹)", f"{st.session_state.users[user['username']].get('wallet',0.0):,.2f}")
    if st.sidebar.button("Logout"):
        logout()

    if user["role"] == "Startup":
        startup_page(user)
    elif user["role"] == "Checker":
        checker_page(user)
    elif user["role"] == "Investor":
        investor_page(user)
    else:
        st.info("Unknown role.")

# ---------- router ----------
if st.session_state.page == "home" or st.session_state.current_user is None:
    landing_page()
else:
    main_app()

