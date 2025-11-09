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

st.set_page_config(page_title="SeedConnect — KYC Edition", page_icon="🚀", layout="wide")
sns.set_style("darkgrid")
plt.rcParams["figure.dpi"] = 100

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

def safe_read_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    if df.shape[1] >= 2:
        df = df.iloc[:, :2]
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df

def predict_forecast_and_roi(df: pd.DataFrame, steps: int = 6):
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
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    src = f"data:application/pdf;base64,{b64}"
    html_tag = f'<iframe src="{src}" width="{width}" height="{height}"></iframe>'
    st.components.v1.html(html_tag, height=int(height.replace("px",""))+20)

def embed_image_bytes(img_bytes: bytes, width=300):
    st.image(img_bytes, width=width)

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

if "investor_seedconnect" not in st.session_state.users:
    st.session_state.users["investor_seedconnect"] = {"password": hash_password("pass123"), "role": "Investor", "wallet": 10000.0}
if "startup_seedconnect" not in st.session_state.users:
    st.session_state.users["startup_seedconnect"] = {"password": hash_password("pass123"), "role": "Startup"}
if "compliance_officer" not in st.session_state.users:
    st.session_state.users["compliance_officer"] = {"password": hash_password("Check@2025!"), "role": "Checker"}

if "_view_graph" not in st.session_state:
    st.session_state._view_graph = None
if "_complaint_for" not in st.session_state:
    st.session_state._complaint_for = None
if "_checker_view" not in st.session_state:
    st.session_state._checker_view = None

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

def landing_page():
    st.markdown("<h1 style='text-align:center;color:#06b6d4;'>🚀 SeedConnect</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#9fb4c9;'>Verified Crowdfunding Platform — KYC & Manual Verification</p>", unsafe_allow_html=True)
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
        st.subheader("About SeedConnect")
        st.write(
            "SeedConnect is a platform to showcase startup onboarding with multi-document KYC, "
            "manual compliance review, investor marketplace with predictions, and complaint workflows."
        )
        st.markdown("**Contact / Support**")
        st.write("Email: support@seedconnect.com")
        st.write("Twitter: @seedconnect")
    with right:
        st.subheader("FAQs")
        st.write("Is this real funding? — No, demo only.")
    st.markdown("---")

def startup_page(user):
    st.header("Startup Onboarding")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Company details")
    name = st.text_input("Company name", key="su_name")
    website = st.text_input("Website", key="su_website")
    contact_email = st.text_input("Contact email", key="su_email")
    target = st.number_input("Target funding (₹)", min_value=0.0, value=500000.0, step=10000.0)
    files = st.file_uploader("Upload KYC / Pitch documents (PDFs, images)", accept_multiple_files=True)
    if st.button("Submit application"):
        if not name or not contact_email:
            st.warning("Provide company name and contact email.")
        else:
            pitch = {
                "id": len(st.session_state.pitches) + 1,
                "name": name,
                "website": website,
                "email": contact_email,
                "target": float(target),
                "files": [],
                "submitted_by": user["username"],
                "status": "Pending",
                "created_at": datetime.utcnow().isoformat()
            }
            for f in files:
                content = f.read()
                pitch["files"].append({"name": f.name, "content": content, "type": f.type})
            st.session_state.pitches.append(pitch)
            st.success("Application submitted.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("My applications")
    mine = [p for p in st.session_state.pitches if p.get("submitted_by") == user["username"]]
    if not mine:
        st.info("No applications yet.")
    else:
        for p in mine:
            st.markdown(f'<div class="card"><h3>{html.escape(p["name"])} {status_badge_html(p.get("status"))}</h3>', unsafe_allow_html=True)
            st.write("Website:", p.get("website") or "-")
            st.write("Contact:", p.get("email"))
            st.write("Target:", f"₹{p.get('target',0):,.2f}")
            if p.get("files"):
                for idx, f in enumerate(p["files"]):
                    st.write(f"File {idx+1}: {f['name']}")
                    if f['name'].lower().endswith(".pdf"):
                        embed_pdf_bytes(f['content'], height="480px")
                    else:
                        try:
                            embed_image_bytes(f['content'], width=320)
                        except Exception:
                            st.write("Preview not available.")
            st.markdown("</div>", unsafe_allow_html=True)

def checker_page(user):
    st.header("Compliance Review")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    view = st.selectbox("View applications", ["Pending", "All"], key="checker_view")
    to_review = st.session_state.pitches if view == "All" else [p for p in st.session_state.pitches if p.get("status") != "Approved"]
    if not to_review:
        st.info("No applications to review.")
    else:
        for p in to_review:
            st.markdown(f'<div class="card"><h4>{html.escape(p["name"])} — {p.get("submitted_by")}</h4>', unsafe_allow_html=True)
            st.write("Email:", p.get("email"))
            st.write("Website:", p.get("website"))
            st.write("Target:", f"₹{p.get('target',0):,.2f}")
            cols = st.columns([1,1,1,1])
            if cols[0].button(f"Approve-{p['id']}"):
                p["status"] = "Approved"
                st.success(f"Application {p['id']} approved.")
            if cols[1].button(f"Reject-{p['id']}"):
                p["status"] = "Rejected"
                st.error(f"Application {p['id']} rejected.")
            if cols[2].button(f"Request Info-{p['id']}"):
                p["status"] = "Pending"
                st.info(f"Requested more information for {p['id']}.")
            if cols[3].button(f"View-{p['id']}"):
                if p.get("files"):
                    for f in p["files"]:
                        st.write(f["name"])
                        if f['name'].lower().endswith(".pdf"):
                            embed_pdf_bytes(f['content'], height="480px")
                        else:
                            try:
                                embed_image_bytes(f['content'], width=320)
                            except Exception:
                                st.write("Preview not available.")
            st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def investor_page(user):
    st.header("Investor Marketplace")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    q = st.text_input("Search startups", key="inv_search")
    status_filter = st.selectbox("Status", ["Any", "Approved", "Pending", "Rejected"], key="inv_status")
    results = st.session_state.pitches
    if q:
        results = [p for p in results if q.lower() in (p.get("name") or "").lower() or q.lower() in (p.get("website") or "").lower()]
    if status_filter != "Any":
        results = [p for p in results if p.get("status","").lower() == status_filter.lower()]
    if not results:
        st.info("No startups match your search.")
    else:
        st.markdown('<div class="grid">', unsafe_allow_html=True)
        for p in results:
            st.markdown(f'<div class="card"><h4>{html.escape(p["name"])}</h4>', unsafe_allow_html=True)
            st.markdown(f'<div class="small-muted">By {html.escape(p.get("submitted_by","-"))}</div>', unsafe_allow_html=True)
            st.write("Website:", p.get("website") or "-")
            st.write("Target:", f"₹{p.get('target',0):,.2f}")
            st.markdown(status_badge_html(p.get("status")))
            if st.button(f"Invest-{p['id']}"):
                investor = st.session_state.users.get(user["username"])
                if investor and investor.get("wallet", 0) >= 1000:
                    amount = 1000.0
                    investor["wallet"] -= amount
                    st.session_state.investments.append({"investor": user["username"], "pitch_id": p["id"], "amount": amount, "date": datetime.utcnow().isoformat()})
                    st.success(f"Invested ₹{amount:,.2f} in {p['name']}")
                else:
                    st.error("Insufficient wallet balance.")
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.subheader("My investments")
    mine = [inv for inv in st.session_state.investments if inv["investor"] == user["username"]]
    if not mine:
        st.info("No investments yet.")
    else:
        for inv in mine:
            p = next((x for x in st.session_state.pitches if x["id"] == inv["pitch_id"]), None)
            st.markdown(f'<div class="card"><b>{p["name"] if p else "Unknown"}</b> — ₹{inv["amount"]:,.2f} on {inv["date"]}</div>', unsafe_allow_html=True)

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

if st.session_state.page == "home" or st.session_state.current_user is None:
    landing_page()
else:
    main_app()

