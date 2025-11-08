# app.py — CrowdPitch Pro: Verified Crowdfunding Platform (Polished UI Edition)
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

# ---------- Page & Theme ----------
st.set_page_config(page_title="CrowdPitch Pro", page_icon="🚀", layout="wide")
sns.set_style("darkgrid")
plt.rcParams["figure.dpi"] = 100

# ---------- CSS Styling ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg,#081B2D,#071622);
    color: #e6eef6;
}
.block-container {
    padding: 1rem 2.5rem;
}
.header-bar {
    text-align:center;
    background: rgba(255,255,255,0.05);
    padding: 0.8rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.25);
}
.header-logo {
    font-size: 1.8rem;
    font-weight: bold;
    color: #06b6d4;
}
.tagline {
    color: #9fc6da;
    font-size: 0.95rem;
}
.card {
    background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
    border-radius: 12px;
    padding: 1rem;
    box-shadow: 0 6px 18px rgba(0,0,0,0.4);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    margin-bottom: 15px;
}
.card:hover {
    transform: scale(1.01);
    box-shadow: 0 8px 25px rgba(0,0,0,0.6);
}
.badge {
    border-radius: 10px;
    padding: 2px 8px;
    font-size: 0.8rem;
    font-weight: 600;
}
.badge-approved {
    background-color: #16a34a; color: #fff;
}
.badge-pending {
    background-color: #eab308; color: #000;
}
.badge-rejected {
    background-color: #dc2626; color: #fff;
}
.avatar {
    display:inline-flex;
    align-items:center;
    justify-content:center;
    width:38px; height:38px;
    border-radius:50%;
    background-color:#06b6d4;
    color:#012;
    font-weight:bold;
    font-size:1rem;
}
</style>
""", unsafe_allow_html=True)

# ---------- Utility Functions ----------
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
        return domain.split(':')[0]
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

def initials(username: str):
    name = username.strip().upper()
    return (name[:2] if len(name) >= 2 else name[:1]).upper()

# ---------- Session Initialization ----------
for key, default in {
    "users": {},
    "current_user": None,
    "pitches": [],
    "investments": [],
    "complaints": []
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Demo & checker accounts
if "investor_demo" not in st.session_state.users:
    st.session_state.users["investor_demo"] = {"password": hash_password("pass123"), "role": "Investor", "wallet": 10000.0}
if "startup_demo" not in st.session_state.users:
    st.session_state.users["startup_demo"] = {"password": hash_password("pass123"), "role": "Startup"}
if "checker_agent" not in st.session_state.users:
    st.session_state.users["checker_agent"] = {"password": hash_password("Check@2025!"), "role": "Checker"}

# ---------- Auth ----------
def signup(username, password, role):
    if username in st.session_state.users:
        st.warning("Username already exists.")
        return
    st.session_state.users[username] = {"password": hash_password(password), "role": role}
    if role == "Investor":
        st.session_state.users[username]["wallet"] = 10000.0
    st.success("Signup successful! Please login.")

def login(username, password):
    user = st.session_state.users.get(username)
    if not user:
        st.error("User not found.")
        return
    if user["password"] != hash_password(password):
        st.error("Incorrect password.")
        return
    st.session_state.current_user = {"username": username, "role": user["role"]}
    st.success(f"Welcome, {username} ({user['role']})!")

def logout():
    st.session_state.current_user = None
    st.info("Logged out.")

# ---------- Header Bar ----------
if st.session_state.current_user:
    u = st.session_state.current_user
    col1, col2, col3 = st.columns([4,2,1])
    with col1:
        st.markdown('<div class="header-bar"><div class="header-logo">🚀 CrowdPitch Pro</div><div class="tagline">Verified Crowdfunding Platform</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="avatar">{initials(u["username"])}</div>', unsafe_allow_html=True)
        if st.button("Logout"):
            logout()
else:
    st.markdown('<div class="header-bar"><div class="header-logo">🚀 CrowdPitch Pro</div><div class="tagline">Verified Crowdfunding Platform</div></div>', unsafe_allow_html=True)

st.markdown("---")

# ---------- Sidebar Navigation ----------
with st.sidebar:
    st.title("Menu")
    if not st.session_state.current_user:
        st.markdown("### Login / Signup")
        mode = st.radio("Action", ["Login", "Signup"])
        if mode == "Signup":
            su_user = st.text_input("Username")
            su_pass = st.text_input("Password", type="password")
            role = st.selectbox("Role", ["Startup", "Investor"])
            if st.button("Create Account"):
                if su_user and su_pass:
                    signup(su_user, su_pass, role)
                else:
                    st.warning("Enter all fields.")
        else:
            li_user = st.text_input("Username")
            li_pass = st.text_input("Password", type="password")
            if st.button("Login"):
                login(li_user, li_pass)
        st.stop()

    else:
        user = st.session_state.current_user
        role = user["role"]
        if role == "Investor":
            st.sidebar.metric("Wallet (₹)", f"{st.session_state.users[user['username']]['wallet']:.2f}")
        tabs = ["Dashboard"]
        if role == "Investor":
            tabs += ["Investments", "Complaints"]
        elif role == "Startup":
            tabs += ["My Pitches"]
        elif role == "Checker":
            tabs += ["Pending Checks", "Complaints"]
        selected_tab = st.radio("Navigate", tabs)

# ---------- Role-specific Dashboards ----------
user = st.session_state.current_user
role = user["role"]

# --- Checker Badges ---
def status_badge(status):
    if status.lower() == "approved":
        return '<span class="badge badge-approved">🟢 Approved</span>'
    elif status.lower() == "pending":
        return '<span class="badge badge-pending">🟡 Pending</span>'
    else:
        return '<span class="badge badge-rejected">🔴 Rejected</span>'

# ---------- Startup ----------
if role == "Startup":
    if selected_tab == "Dashboard":
        st.header("🏢 Startup Dashboard")
        st.markdown("Submit and track pitches awaiting checker approval.")
        # ... (existing form logic can go here as before)

# ---------- Investor ----------
elif role == "Investor":
    if selected_tab == "Dashboard":
        st.header("💼 Investor Dashboard — Browse & Invest")
        approved = [p for p in st.session_state.pitches if p.get("published")]
        if not approved:
            st.info("No approved pitches available yet.")
        else:
            for p in approved:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                c1, c2 = st.columns([1,3])
                with c1:
                    st.image(p["logo"] if p["logo"] else "https://via.placeholder.com/220x140.png?text=No+Logo", width=140)
                    st.markdown(status_badge(p["checker_status"]), unsafe_allow_html=True)
                with c2:
                    st.markdown(f"### {p['pitch_name']} — {p['company_name']}")
                    st.markdown(p["short"])
                    st.markdown(f"<b>Status:</b> {status_badge(p['checker_status'])}", unsafe_allow_html=True)
                    st.markdown(f"<b>Funded:</b> ₹{p['funded']:,.0f}")
                st.markdown('</div>', unsafe_allow_html=True)

    elif selected_tab == "Investments":
        st.header("📊 My Investments")
        invs = [i for i in st.session_state.investments if i["investor"] == user["username"]]
        if not invs:
            st.info("No investments yet.")
        else:
            df = pd.DataFrame(invs)
            st.dataframe(df)
    elif selected_tab == "Complaints":
        st.header("🧾 My Complaints")
        comp = [c for c in st.session_state.complaints if c["investor"] == user["username"]]
        st.dataframe(comp if comp else pd.DataFrame(columns=["pitch_name","message","status"]))

# ---------- Checker ----------
elif role == "Checker":
    if selected_tab == "Pending Checks":
        st.header("🕵️ Pending Checks")
        pending = [p for p in st.session_state.pitches if p["checker_status"] == "Pending"]
        if not pending:
            st.info("No pending pitches.")
        for p in pending:
            st.markdown(f"### {p['pitch_name']} — {p['company_name']}")
            st.dataframe(p["data"].head())
            c1, c2 = st.columns(2)
            note = st.text_area("Checker Note", key=f"note_{p['pitch_name']}")
            if c1.button("Approve", key=f"approve_{p['pitch_name']}"):
                p["checker_status"] = "Approved"
                p["published"] = True
                p["checker_note"] = note
                st.success("Approved")
            if c2.button("Reject", key=f"reject_{p['pitch_name']}"):
                p["checker_status"] = "Rejected"
                p["published"] = False
                p["checker_note"] = note
                st.error("Rejected")

    elif selected_tab == "Complaints":
        st.header("📩 Complaints Inbox")
        if not st.session_state.complaints:
            st.info("No complaints submitted.")
        else:
            st.dataframe(st.session_state.complaints)

# ---------- Footer ----------
st.markdown("---")
st.markdown("<center>© 2025 CrowdPitch Pro — A Verified Crowdfunding Platform</center>", unsafe_allow_html=True)
