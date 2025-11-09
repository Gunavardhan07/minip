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
    .product-card { padding:12px; border-radius:8px; background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); box-shadow:0 6px 18px rgba(0,0,0,0.45); }
    .cart { position: sticky; top: 20px; }
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

# ---------------- SESSION STATE INIT ---------------- #
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
if "cart" not in st.session_state:
    st.session_state.cart = []
if "orders" not in st.session_state:
    st.session_state.orders = []

if "investor_seedconnect" not in st.session_state.users:
    st.session_state.users["investor_seedconnect"] = {"password": hash_password("pass123"), "role": "Investor", "wallet": 10000.0}
if "startup_seedconnect" not in st.session_state.users:
    st.session_state.users["startup_seedconnect"] = {"password": hash_password("pass123"), "role": "Startup"}
if "compliance_officer" not in st.session_state.users:
    st.session_state.users["compliance_officer"] = {"password": hash_password("Check@2025!"), "role": "Checker"}

# ---------------- AUTH ---------------- #
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

# ---------------- LANDING ---------------- #
def landing_page():
    st.markdown("<h1 style='text-align:center;color:#06b6d4;'>🚀 SeedConnect</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#9fb4c9;'>Verified Crowdfunding Platform — KYC & Compliance</p>", unsafe_allow_html=True)
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
            "SeedConnect is a platform for verified startup onboarding with multi-document KYC, "
            "manual compliance review, investor marketplace with predictions, and complaint workflows."
        )
        st.markdown("**Contact / Support**")
        st.write("Email: support@seedconnect.com")
        st.write("Twitter: @seedconnect")
    with right:
        st.subheader("Quick access")
        st.write("Checker credentials:")
        st.markdown("<div class='small-muted'><b>Username:</b> compliance_officer<br><b>Password:</b> Check@2025!</div>", unsafe_allow_html=True)
        st.subheader("FAQs")
        st.write("Is this real funding? — No, this environment is a controlled demo for verification and testing.")
    st.markdown("---")

# ---------------- STARTUP PAGE ---------------- #
# (unchanged - omitted for brevity)

def startup_page(user):
    pass  # placeholder; unchanged from previous code

# ---------------- CHECKER PAGE ---------------- #
# (unchanged - omitted for brevity)

def checker_page(user):
    pass  # placeholder; unchanged from previous code

# ---------------- INVESTOR PAGE ---------------- #
def investor_page(user):
    st.header("Investor Marketplace — Shop-like Experience")
    left, right = st.columns([3,1])
    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        q = st.text_input("Search startups", key="inv_search")
        status_filter = st.selectbox("Status", ["Any", "Approved", "Pending", "Rejected"], key="inv_status")
        sort_by = st.selectbox("Sort by", ["Relevance", "Target: Low→High", "Target: High→Low"], key="inv_sort")
        results = st.session_state.pitches
        if q:
            results = [p for p in results if q.lower() in (p.get("name") or "").lower() or q.lower() in (p.get("website") or "").lower()]
        if status_filter != "Any":
            results = [p for p in results if p.get("status","").lower() == status_filter.lower()]
        if sort_by == "Target: Low→High":
            results = sorted(results, key=lambda x: x.get("target", 0))
        elif sort_by == "Target: High→Low":
            results = sorted(results, key=lambda x: x.get("target", 0), reverse=True)
        if not results:
            st.info("No startups match your search.")
        else:
            st.markdown('<div class="grid">', unsafe_allow_html=True)
            for p in results:
                st.markdown(f'<div class="product-card"><h4>{html.escape(p["name"])}</h4>', unsafe_allow_html=True)
                st.markdown(f'<div class="small-muted">By {html.escape(p.get("submitted_by","-"))}</div>', unsafe_allow_html=True)
                st.write("Target:", f"₹{p.get('target',0):,.2f}")
                st.write("Equity:", f"{p.get('equity',0):.2f}%")
                st.markdown(status_badge_html(p.get("status")))
                if st.button(f"View details-{p['id']}"):
                    with st.expander(f"Details — {p['name']}"):
                        st.write("Website:", p.get("website") or "-")
                        st.write("Contact:", p.get("email"))
                        st.write("Target:", f"₹{p.get('target',0):,.2f}")
                        st.write("Equity:", f"{p.get('equity',0):.2f}%")
                        st.markdown("### ROI Prediction (ARIMA Forecast)")
                        sample_data = pd.DataFrame({
                            "date": pd.date_range("2024-01-01", periods=12, freq="M"),
                            "value": np.linspace(100, 160, 12) + np.random.randn(12) * 3
                        })
                        dates, forecast, conf, roi = predict_forecast_and_roi(sample_data)
                        st.write(f"**Predicted ROI (next 6 months):** {roi:.2f}%")
                        if dates is not None:
                            plt.figure()
                            plt.plot(sample_data["date"], sample_data["value"], label="Historical")
                            plt.plot(dates, forecast, linestyle="--", label="Forecast")
                            plt.xlabel("Date")
                            plt.ylabel("Value")
                            plt.title(f"{p['name']} ROI Forecast (6 months)")
                            plt.legend()
                            st.pyplot(plt)
                        if p.get("files"):
                            st.write("Documents:")
                            for f in p["files"]:
                                st.write(f["name"])
                                if f["name"].lower().endswith(".pdf"):
                                    embed_pdf_bytes(f["content"], height="240px")
                                else:
                                    try:
                                        embed_image_bytes(f["content"], width=220)
                                    except Exception:
                                        st.write("Preview not available.")
                st.markdown("</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="card cart">', unsafe_allow_html=True)
        st.subheader("Cart")
        if not st.session_state.cart:
            st.info("Cart is empty.")
        else:
            total = sum(item["amount"] for item in st.session_state.cart)
            for i, item in enumerate(st.session_state.cart, start=1):
                st.write(f"{i}. {item['name']} — ₹{item['amount']:,.2f}")
            st.markdown(f"**Total:** ₹{total:,.2f}")
        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- ROUTING ---------------- #
def main_app():
    user = st.session_state.current_user
    st.sidebar.markdown(f"**Logged in as:** {user['username']} ({user['role']})")
    if user["role"] == "Investor":
        st.sidebar.metric("Wallet (₹)", f"{st.session_state.users[user['username']].get('wallet',0.0):,.2f}")
    if st.sidebar.button("Logout"):
        logout()
    if user["role"] == "Investor":
        investor_page(user)
    elif user["role"] == "Checker":
        checker_page(user)
    elif user["role"] == "Startup":
        startup_page(user)

if st.session_state.page == "home" or st.session_state.current_user is None:
    landing_page()
else:
    main_app()

