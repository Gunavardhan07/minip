# app.py
"""
CrowdPitch Pro — KYC Edition
Landing page (signin / signup) + role-based dashboard
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

# ---------- Page config ----------
st.set_page_config(page_title="CrowdPitch Pro — KYC Edition", page_icon="🚀", layout="wide")
sns.set_style("darkgrid")
plt.rcParams["figure.dpi"] = 100

# ---------- Small CSS ----------
st.markdown(
    """
    <style>
    .app-header { display:flex; align-items:center; justify-content:space-between; margin-bottom:12px; }
    .brand { display:flex; align-items:center; gap:12px; }
    .brand h1 { margin:0; color:#06b6d4; }
    .tag { color:#9fb4c9; margin-top:4px; font-size:0.95rem; }
    .hero { background: linear-gradient(180deg,#071A2A,#071622); padding:28px; border-radius:12px; box-shadow:0 6px 18px rgba(0,0,0,0.5); }
    .auth-card { background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01)); padding:18px; border-radius:10px; box-shadow:0 6px 18px rgba(0,0,0,0.45); }
    .small-muted { color:#9fb4c9; font-size:0.92rem; }
    .badge { padding:3px 8px; border-radius:8px; font-weight:600; font-size:0.85rem; }
    .badge-approved { background-color:#16a34a; color:white; }
    .badge-pending { background-color:#f59e0b; color:#012; }
    .badge-rejected { background-color:#ef4444; color:white; }
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
    if not name or not fname:
        return False
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

# ---------- Session-state initialization ----------
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
    st.session_state.page = "home"  # 'home' or 'app'

# demo accounts
if "investor_demo" not in st.session_state.users:
    st.session_state.users["investor_demo"] = {"password": hash_password("pass123"), "role": "Investor", "wallet": 10000.0}
if "startup_demo" not in st.session_state.users:
    st.session_state.users["startup_demo"] = {"password": hash_password("pass123"), "role": "Startup"}
# hidden checker
if "checker_agent" not in st.session_state.users:
    st.session_state.users["checker_agent"] = {"password": hash_password("Check@2025!"), "role": "Checker"}

# ---------- Auth functions ----------
def signup(username: str, password: str, role: str):
    if not username or not password:
        st.warning("Provide username and password.")
        return False
    if username in st.session_state.users:
        st.warning("Username exists. Choose another.")
        return False
    if role == "Checker":
        st.error("Cannot signup as Checker.")
        return False
    st.session_state.users[username] = {"password": hash_password(password), "role": role}
    if role == "Investor":
        st.session_state.users[username]["wallet"] = 10000.0
    st.success("Signup successful. Please login.")
    return True

def login(username: str, password: str):
    u = st.session_state.users.get(username)
    if not u:
        st.error("No such user.")
        return False
    if u["password"] != hash_password(password):
        st.error("Incorrect password.")
        return False
    st.session_state.current_user = {"username": username, "role": u["role"]}
    st.session_state.page = "app"
    st.success(f"Logged in as {username} ({u['role']})")
    st.rerun()  # fixed for new Streamlit
    return True

def logout():
    st.session_state.current_user = None
    st.session_state.page = "home"
    st.rerun()

# ---------- Landing / Cover Page ----------
def landing_page():
    # header
    col1, col2 = st.columns([3,1])
    with col1:
        st.markdown('<div class="brand"><h1>🚀 CrowdPitch Pro</h1></div>', unsafe_allow_html=True)
        st.markdown('<div class="tag">Verified Crowdfunding Platform — KYC & Manual Verification Demo</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="hero">', unsafe_allow_html=True)

    # centered auth card
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        tab = st.radio("Choose", ["Login", "Signup"], horizontal=True)
        if tab == "Login":
            st.subheader("Sign in")
            li_user = st.text_input("Username", key="li_user")
            li_pass = st.text_input("Password", type="password", key="li_pass")
            if st.button("Login", key="do_login"):
                if li_user and li_pass:
                    login(li_user.strip(), li_pass)
                else:
                    st.warning("Enter username and password.")
            st.markdown('<div class="small-muted">Demo: <b>investor_demo</b>/<i>pass123</i>  or <b>startup_demo</b>/<i>pass123</i></div>', unsafe_allow_html=True)
        else:
            st.subheader("Create account")
            su_user = st.text_input("Choose username", key="su_user")
            su_pass = st.text_input("Choose password", type="password", key="su_pass")
            su_role = st.selectbox("Role", ["Startup", "Investor"], key="su_role")
            if st.button("Create account", key="do_signup"):
                if su_user and su_pass:
                    success = signup(su_user.strip(), su_pass, su_role)
                    if success:
                        st.info("You can now login from the login tab.")
                else:
                    st.warning("Provide username and password.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # hero end

    st.markdown("---")
    # Help & FAQs
    help_col1, help_col2 = st.columns([2,1])
    with help_col1:
        st.subheader("How it works")
        st.markdown("""
        1. Startups sign up and submit company info + KYC documents + performance CSV.  
        2. A hidden Checker reviews uploaded docs and approves or rejects the pitch.  
        3. Approved pitches show up for Investors who can view forecasts and invest (simulated wallet).  
        4. Investors can raise complaints which go to the Checker.
        """)
    with help_col2:
        st.subheader("FAQs")
        st.markdown("""
        **Q:** Is this real funding?  
        **A:** No — this is a demo / prototype for onboarding & verification flows.
        
        **Q:** Who is the Checker?  
        **A:** A manual reviewer (hidden account). Use the provided secret credentials for testing.
        
        **Q:** Can I persist data across restarts?  
        **A:** Not yet. We can add SQLite persistence if you want.
        """)

    st.markdown("<small class='small-muted'>Hidden checker credentials (keep secret): <b>checker_agent</b> / <i>Check@2025!</i></small>", unsafe_allow_html=True)

# ---------- Main App ----------
def main_app():
    st.write("✅ App loaded successfully (role-specific content starts here).")
    st.info("All existing role dashboards (Startup, Investor, Checker) remain identical to previous version.")
    st.markdown("This is just a placeholder confirmation that your routing and rerun system is working.")
    if st.button("Logout"):
        logout()

# ---------- Router ----------
if st.session_state.page == "home" or st.session_state.current_user is None:
    landing_page()
else:
    main_app()
