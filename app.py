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

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="CrowdPitch Pro — KYC Edition", page_icon="🚀", layout="wide")
sns.set_style("darkgrid")
plt.rcParams["figure.dpi"] = 100

# ---------- CSS ----------
st.markdown("""
<style>
body { background: linear-gradient(180deg,#071A2A,#071622); color: #e6eef6; }
.auth-card { background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02)); padding: 2rem; border-radius: 12px; box-shadow: 0 8px 24px rgba(0,0,0,0.5); text-align: center; }
.tagline { color:#9fb4c9; font-size:1rem; margin-bottom:1rem; }
.badge { padding:4px 8px; border-radius:8px; font-weight:600; font-size:0.85rem; }
.badge-approved { background-color:#16a34a; color:white; }
.badge-pending { background-color:#facc15; color:#012; }
.badge-rejected { background-color:#ef4444; color:white; }
</style>
""", unsafe_allow_html=True)

# ---------- UTILITIES ----------
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

# ---------- SESSION INIT ----------
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

# Demo accounts
if "investor_demo" not in st.session_state.users:
    st.session_state.users["investor_demo"] = {"password": hash_password("pass123"), "role": "Investor", "wallet": 10000.0}
if "startup_demo" not in st.session_state.users:
    st.session_state.users["startup_demo"] = {"password": hash_password("pass123"), "role": "Startup"}
if "checker_agent" not in st.session_state.users:
    st.session_state.users["checker_agent"] = {"password": hash_password("Check@2025!"), "role": "Checker"}

# ---------- AUTH ----------
def signup(username: str, password: str, role: str):
    if username in st.session_state.users:
        st.warning("Username exists.")
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
    st.success(f"Welcome {username} ({u['role']})!")
    st.rerun()
    return True

def logout():
    st.session_state.current_user = None
    st.session_state.page = "home"
    st.rerun()

# ---------- LANDING PAGE ----------
def landing_page():
    st.markdown("<h1 style='text-align:center;color:#06b6d4;'>🚀 CrowdPitch Pro</h1>", unsafe_allow_html=True)
    st.markdown("<p class='tagline' style='text-align:center;'>Verified Crowdfunding Platform — KYC Edition</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        mode = st.radio("Select", ["Login", "Signup"], horizontal=True)
        if mode == "Login":
            li_user = st.text_input("Username")
            li_pass = st.text_input("Password", type="password")
            if st.button("Login"):
                if li_user and li_pass:
                    login(li_user.strip(), li_pass)
                else:
                    st.warning("Enter credentials.")
            st.markdown("<small>Demo: investor_demo/pass123 or startup_demo/pass123</small>", unsafe_allow_html=True)
        else:
            su_user = st.text_input("Choose username")
            su_pass = st.text_input("Choose password", type="password")
            su_role = st.selectbox("Role", ["Startup", "Investor"])
            if st.button("Create Account"):
                if su_user and su_pass:
                    signup(su_user.strip(), su_pass, su_role)
                else:
                    st.warning("Fill all fields.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.subheader("Help & FAQs")
    st.markdown("""
    **Q:** What is CrowdPitch Pro?  
    **A:** A demo platform for verified startup onboarding and investor simulation.

    **Q:** How does verification work?  
    **A:** Startups upload documents → Checker manually approves → Investors see verified ones.

    **Q:** Hidden Checker Login:**  
    **A:** `checker_agent` / `Check@2025!`
    """)

# ---------- DASHBOARDS ----------
def startup_dashboard(user):
    st.header("🏢 Startup Onboarding")
    st.info("Submit your company details and upload required documents. Checker must approve before investors see your pitch.")
    with st.form("onboard_form"):
        company = st.text_input("Company Name")
        pitch = st.text_input("Pitch Name")
        csv_file = st.file_uploader("Performance CSV (date,value)", type="csv")
        kyc = st.file_uploader("KYC Document")
        addr = st.file_uploader("Address Proof")
        bank = st.file_uploader("Bank Proof")
        desc = st.text_area("Pitch Description")
        submitted = st.form_submit_button("Submit Pitch")
    if submitted:
        if not (company and pitch and csv_file and kyc and addr and bank):
            st.error("All fields required.")
        else:
            df = pd.read_csv(csv_file)
            df.columns = ["date","value"]
            df["date"] = pd.to_datetime(df["date"])
            pitch_data = {
                "company_name": company,
                "pitch_name": pitch,
                "data": df,
                "desc": desc,
                "owner": user["username"],
                "funded": 0,
                "published": False,
                "checker_status": "Pending",
                "verification": {"KYC": "Uploaded", "Address": "Uploaded", "Bank": "Uploaded"}
            }
            st.session_state.pitches.append(pitch_data)
            st.success("Pitch submitted for checker review.")

    st.subheader("My Pitches")
    mine = [p for p in st.session_state.pitches if p["owner"] == user["username"]]
    if not mine:
        st.info("No pitches yet.")
    for p in mine:
        st.markdown(f"### {p['pitch_name']} ({status_badge_html(p['checker_status'])})", unsafe_allow_html=True)
        st.write(p["desc"])

def checker_dashboard():
    st.header("🕵️ Checker Dashboard")
    pending = [p for p in st.session_state.pitches if p["checker_status"] == "Pending"]
    if not pending:
        st.info("No pending pitches.")
    for p in pending:
        st.markdown(f"### {p['pitch_name']} — {p['company_name']}")
        st.dataframe(p["data"].head())
        c1, c2 = st.columns(2)
        if c1.button("Approve", key=f"a_{p['pitch_name']}"):
            p["checker_status"] = "Approved"
            p["published"] = True
            st.success("Approved.")
        if c2.button("Reject", key=f"r_{p['pitch_name']}"):
            p["checker_status"] = "Rejected"
            st.error("Rejected.")

    st.subheader("Complaints")
    for c in st.session_state.complaints:
        st.write(f"Pitch: {c['pitch_name']} — {c['message']} ({c['status']})")

def investor_dashboard(user):
    st.header("💼 Investor Dashboard")
    published = [p for p in st.session_state.pitches if p["published"]]
    if not published:
        st.info("No verified pitches yet.")
    for p in published:
        st.markdown(f"### {p['pitch_name']} — {p['company_name']}")
        st.write(p["desc"])
        df = p["data"]
        fig, ax = plt.subplots()
        sns.lineplot(x="date", y="value", data=df, ax=ax)
        st.pyplot(fig)
        if st.button(f"Invest in {p['pitch_name']}", key=f"i_{p['pitch_name']}"):
            amt = st.number_input("Amount", 100.0, 100000.0, 500.0, 100.0)
            wallet = st.session_state.users[user["username"]]["wallet"]
            if amt > wallet:
                st.error("Insufficient balance.")
            else:
                p["funded"] += amt
                st.session_state.users[user["username"]]["wallet"] -= amt
                st.session_state.investments.append({"investor": user["username"], "pitch": p["pitch_name"], "amount": amt})
                st.success("Investment successful.")
        if st.button(f"Complain on {p['pitch_name']}", key=f"c_{p['pitch_name']}"):
            msg = st.text_area("Complaint Message", key=f"m_{p['pitch_name']}")
            if st.button("Submit", key=f"s_{p['pitch_name']}"):
                st.session_state.complaints.append({"pitch_name": p["pitch_name"], "investor": user["username"], "message": msg, "status": "Open"})
                st.success("Complaint submitted.")

# ---------- MAIN APP ----------
def main_app():
    user = st.session_state.current_user
    st.sidebar.success(f"Logged in as {user['username']} ({user['role']})")
    if st.sidebar.button("Logout"):
        logout()

    if user["role"] == "Startup":
        startup_dashboard(user)
    elif user["role"] == "Checker":
        checker_dashboard()
    elif user["role"] == "Investor":
        investor_dashboard(user)

# ---------- ROUTER ----------
if st.session_state.page == "home" or st.session_state.current_user is None:
    landing_page()
else:
    main_app()
