# app.py (CrowdPitch Pro — KYC Edition, with Checker + Complaints)
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

# ---------------- Page config & theme -----------------
st.set_page_config(page_title="CrowdPitch Pro — KYC Edition", page_icon="🚀", layout="wide")
sns.set_style("darkgrid")
plt.rcParams["figure.dpi"] = 100

# ----------------- CSS -----------------
st.markdown("""
<style>
.stApp { background: linear-gradient(180deg,#071A2A,#071622); color: #e6eef6; }
.block-container { padding: 1.25rem 2rem; }
.card { background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01)); 
       border-radius: 12px; padding: 0.9rem; box-shadow: 0 6px 18px rgba(1,8,16,0.6); margin-bottom: 12px;}
.sidebar .sidebar-content { background: linear-gradient(180deg,#071622,#03121A); }
.stButton>button { background-color: #06b6d4; color: #012; border-radius: 8px; padding: 0.45rem 0.75rem; }
.metric-label { color:#9fb4c9; }
.small-muted { color:#9fb4c9; font-size:0.9rem; }
</style>
""", unsafe_allow_html=True)

# ----------------- Utilities -----------------
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

# ----------------- Initialize session state -----------------
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

# Demo & hidden accounts
if "investor_demo" not in st.session_state.users:
    st.session_state.users["investor_demo"] = {"password": hash_password("pass123"), "role": "Investor", "wallet": 10000.0}
if "startup_demo" not in st.session_state.users:
    st.session_state.users["startup_demo"] = {"password": hash_password("pass123"), "role": "Startup"}
if "checker_agent" not in st.session_state.users:
    st.session_state.users["checker_agent"] = {"password": hash_password("Check@2025!"), "role": "Checker"}

# ----------------- Auth -----------------
def signup(username, password, role):
    if username in st.session_state.users:
        st.warning("Username exists. Choose another.")
        return False
    if role == "Checker":
        st.error("Cannot signup as Checker.")
        return False
    st.session_state.users[username] = {"password": hash_password(password), "role": role}
    if role == "Investor":
        st.session_state.users[username]["wallet"] = 10000.0
    st.success("Signup successful! Please login.")
    return True

def login(username, password):
    user = st.session_state.users.get(username)
    if not user:
        st.error("No such user.")
        return False
    if user["password"] != hash_password(password):
        st.error("Incorrect password.")
        return False
    st.session_state.current_user = {"username": username, "role": user["role"]}
    st.success(f"Logged in as {username} ({user['role']})")
    return True

def logout():
    st.session_state.current_user = None
    st.info("Logged out.")

# ----------------- Header -----------------
c1, c2 = st.columns([3,1])
with c1:
    st.title("🚀 CrowdPitch Pro — KYC & Verification Demo")
    st.markdown("**Startups upload documents → Checker verifies → Investors browse & invest.**")
with c2:
    if st.session_state.current_user:
        u = st.session_state.current_user
        st.markdown(f"**{u['username']}**")
        st.markdown(f"_{u['role']}_")
        if st.button("Logout"):
            logout()
    else:
        st.markdown("Not signed in")

st.markdown("---")

# ----------------- Sidebar auth -----------------
with st.sidebar:
    st.markdown("## Account")
    if not st.session_state.current_user:
        action = st.radio("Action", ["Login", "Signup"], index=0)
        if action == "Signup":
            su_user = st.text_input("Username (signup)")
            su_pass = st.text_input("Password", type="password")
            su_role = st.selectbox("Role", ["Startup", "Investor"])
            if st.button("Create account"):
                if su_user and su_pass:
                    signup(su_user, su_pass, su_role)
                else:
                    st.warning("Enter all fields.")
        else:
            li_user = st.text_input("Username (login)")
            li_pass = st.text_input("Password", type="password")
            if st.button("Login"):
                if li_user and li_pass:
                    login(li_user, li_pass)
                else:
                    st.warning("Enter username and password.")
        st.markdown("---")
        st.markdown("**Demo accounts:**")
        st.markdown("- investor_demo / pass123")
        st.markdown("- startup_demo / pass123")
        st.markdown("- Secret checker credentials (see chat).")
    else:
        u = st.session_state.current_user
        if u["role"] == "Investor":
            wallet = st.session_state.users[u["username"]].get("wallet", 0.0)
            st.markdown(f"### Wallet ₹{wallet:,.2f}")
            if st.button("Add ₹1000"):
                st.session_state.users[u["username"]]["wallet"] += 1000.0
                st.success("₹1000 added")

# ----------------- Stop if not logged -----------------
if not st.session_state.current_user:
    st.info("Please login or signup.")
    st.stop()

user = st.session_state.current_user
role = user["role"]
username = user["username"]

# ----------------- STARTUP -----------------
if role == "Startup":
    st.header("🏢 Startup Onboarding & Pitch Creation")
    st.markdown("Submit company details and documents. A checker must approve before investors can view your pitch.")

    with st.form("onboard_form"):
        st.subheader("Company Info")
        company_name = st.text_input("Company Legal Name")
        reg_number = st.text_input("Registration / Incorporation Number")
        country = st.text_input("Country", value="India")
        founders = st.text_area("Founders (comma separated)")
        official_email = st.text_input("Official Email")
        website = st.text_input("Company Website")

        st.subheader("Documents")
        logo_file = st.file_uploader("Logo (PNG/JPG)", type=["png", "jpg", "jpeg"])
        kyc_file = st.file_uploader("KYC Document", type=["png", "jpg", "jpeg", "pdf"])
        address_file = st.file_uploader("Proof of Address", type=["png", "jpg", "jpeg", "pdf"])
        bank_file = st.file_uploader("Bank Verification Document", type=["png", "jpg", "jpeg", "pdf"])

        st.subheader("Pitch & Data")
        pitch_name = st.text_input("Pitch / Product Name")
        short_desc = st.text_input("Short Description")
        long_desc = st.text_area("Long Description")
        csv_file = st.file_uploader("Performance CSV (date,value)", type=["csv"])
        video_link = st.text_input("Video pitch link (optional)")

        submitted = st.form_submit_button("Submit Pitch")

    if submitted:
        missing = []
        for f, name in [(company_name, "Company Name"), (pitch_name, "Pitch Name"),
                        (csv_file, "CSV Data"), (kyc_file, "KYC"), 
                        (address_file, "Address"), (bank_file, "Bank Doc")]:
            if not f: missing.append(name)
        if missing:
            st.error("Missing: " + ", ".join(missing))
        else:
            df = pd.read_csv(csv_file)
            df.columns = ["date", "value"]
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")
            logo_bytes = image_bytes(logo_file) if logo_file else None
            website_domain = extract_domain(website)
            email_dom = email_domain(official_email)
            email_ok = website_domain and email_dom and (website_domain in email_dom or email_dom in website_domain)
            bank_match = name_in_filename(company_name, getattr(bank_file, "name", ""))

            verification = {
                "email_domain": "Verified" if email_ok else "Pending",
                "bank_doc": "Verified" if bank_match else "Pending",
                "kyc_uploaded": True,
                "address_uploaded": True
            }

            pitch = {
                "company_name": company_name, "reg_number": reg_number, "country": country,
                "founders": founders, "official_email": official_email, "website": website,
                "logo": logo_bytes, "pitch_name": pitch_name, "short": short_desc, "desc": long_desc,
                "owner": username, "data": df, "video": video_link,
                "funded": 0.0, "investors": [], "verification": verification,
                "published": False, "checker_status": "Pending", "checker_note": ""
            }
            st.session_state.pitches.append(pitch)
            st.success(f"Pitch '{pitch_name}' submitted for checker review.")
            st.balloons()

    st.markdown("### Your Pitches")
    mine = [p for p in st.session_state.pitches if p["owner"] == username]
    for p in mine:
        st.markdown(f"- **{p['pitch_name']}** — {p['checker_status']}")

# ----------------- CHECKER -----------------
elif role == "Checker":
    st.header("🕵️ Checker Dashboard")
    st.markdown("Approve or reject startup submissions and manage complaints.")
    pending = [p for p in st.session_state.pitches if p["checker_status"] == "Pending"]

    st.subheader("Pending Pitches")
    if not pending:
        st.info("No pending pitches.")
    for p in pending:
        st.markdown(f"### {p['pitch_name']} — {p['company_name']}")
        st.dataframe(p["data"].head())
        note = st.text_area("Decision note", key=f"note_{p['pitch_name']}")
        c1, c2 = st.columns(2)
        if c1.button("Approve", key=f"approve_{p['pitch_name']}"):
            p["checker_status"] = "Approved"
            p["published"] = True
            p["checker_note"] = note
            st.success(f"Approved {p['pitch_name']}")
        if c2.button("Reject", key=f"reject_{p['pitch_name']}"):
            p["checker_status"] = "Rejected"
            p["checker_note"] = note
            st.error(f"Rejected {p['pitch_name']}")

    st.subheader("Investor Complaints")
    complaints = st.session_state.complaints
    if not complaints:
        st.info("No complaints yet.")
    for i, c in enumerate(complaints):
        st.markdown(f"**#{i+1} — Pitch:** {c['pitch_name']} by {c['investor']}")
        st.markdown(f"Message: {c['message']}")
        st.markdown(f"Status: {c.get('status','Open')}")
        note = st.text_input("Resolution note", key=f"rnote_{i}")
        if st.button("Mark Resolved", key=f"resolve_{i}"):
            c["status"] = "Resolved"
            c["resolution_note"] = note
            st.success("Complaint marked resolved.")

# ----------------- INVESTOR -----------------
elif role == "Investor":
    st.header("💼 Investor Dashboard")
    published = [p for p in st.session_state.pitches if p.get("published")]
    if not published:
        st.info("No published pitches yet.")
        st.stop()

    for p in published:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        cols = st.columns([1, 3])
        with cols[0]:
            if p["logo"]:
                st.image(p["logo"], width=160)
            else:
                st.image("https://via.placeholder.com/220x140.png?text=No+Logo", width=160)
            st.markdown(f"**Funded:** ₹{p['funded']:,.0f}")
        with cols[1]:
            st.markdown(f"### {p['pitch_name']} — {p['company_name']}")
            st.markdown(p["short"])
            st.markdown(p["desc"][:400] + "...")
            df = p["data"]
            fig, ax = plt.subplots(figsize=(8,3))
            sns.lineplot(x="date", y="value", data=df, ax=ax, marker="o", label="Historical")
            try:
                model = sm.tsa.ARIMA(df["value"], order=(1,1,1))
                res = model.fit()
                fc = res.forecast(steps=6)
                fut = pd.date_range(df["date"].iloc[-1], periods=7, freq="M")[1:]
                ax.plot(fut, fc, linestyle="--", marker="x", label="Forecast")
            except:
                pass
            ax.legend(); ax.set_title("Trend & Forecast")
            st.pyplot(fig)
            if st.button(f"Invest in {p['pitch_name']}", key=f"inv_{p['pitch_name']}"):
                amt = st.number_input("Amount", 100.0, 100000.0, 500.0, 100.0, key=f"amt_{p['pitch_name']}")
                if st.button("Confirm", key=f"conf_{p['pitch_name']}"):
                    wallet = st.session_state.users[username]["wallet"]
                    if amt > wallet:
                        st.error("Insufficient funds.")
                    else:
                        p["funded"] += amt
                        st.session_state.users[username]["wallet"] -= amt
                        st.success("Investment recorded.")
            if st.button(f"Complain about {p['pitch_name']}", key=f"complain_{p['pitch_name']}"):
                msg = st.text_area("Describe issue", key=f"msg_{p['pitch_name']}")
                if st.button("Submit Complaint", key=f"subc_{p['pitch_name']}"):
                    st.session_state.complaints.append({
                        "pitch_name": p["pitch_name"], "investor": username,
                        "message": msg, "status": "Open"
                    })
                    st.success("Complaint sent to checker.")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("---")

# ----------------- Footer -----------------
st.markdown("---")
st.metric("Total Pitches", len(st.session_state.pitches))
st.metric("Total Funded", f"₹{sum(p['funded'] for p in st.session_state.pitches):,.0f}")
st.metric("Complaints", len(st.session_state.complaints))
st.markdown("<center>Built with ❤️ Streamlit</center>", unsafe_allow_html=True)
