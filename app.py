# app.py
"""
CrowdPitch Pro — KYC Edition (final)
Landing page + Startup onboarding + Checker verification + Investor cards + ARIMA ROI prediction
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

# ---------- CSS ----------
st.markdown(
    """
    <style>
    body { background: linear-gradient(180deg,#071A2A,#071622); color: #e6eef6; }
    .hero { padding:20px; border-radius:10px; margin-bottom:18px; }
    .auth-card { background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01)); padding:18px; border-radius:10px; box-shadow: 0 6px 18px rgba(0,0,0,0.5); }
    .small-muted { color:#9fb4c9; font-size:0.9rem; }
    .card { background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01)); border-radius:12px; padding:12px; box-shadow: 0 6px 18px rgba(0,0,0,0.45); margin-bottom:14px; }
    .card-grid { display:flex; gap:12px; flex-wrap:wrap; }
    .badge { padding:4px 8px; border-radius:8px; font-weight:600; font-size:0.85rem; }
    .badge-approved { background-color:#16a34a; color:white; }
    .badge-pending { background-color:#f59e0b; color:#012; }
    .badge-rejected { background-color:#ef4444; color:white; }
    .progress { height: 12px; background: rgba(255,255,255,0.06); border-radius: 8px; overflow:hidden; margin-top:6px; }
    .progress > div { background: linear-gradient(90deg,#06b6d4,#0891b2); height:100%; }
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

def img_from_bytes(b: bytes):
    return Image.open(BytesIO(b))

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

def safe_read_csv(file) -> pd.DataFrame:
    try:
        df = pd.read_csv(file)
        if len(df.columns) >= 2:
            df = df.iloc[:, :2]
            df.columns = ["date", "value"]
        else:
            df.columns = ["date", "value"]
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        return df
    except Exception as e:
        raise

def predict_forecast_and_roi(df: pd.DataFrame, steps: int = 6):
    """
    Given df with columns date,value returns:
    - forecast_dates (DatetimeIndex)
    - forecast_values (np.array)
    - conf_int (DataFrame with lower, upper) or None
    - roi_percent (float) computed as (mean_forecast / latest - 1)*100
    """
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
        # fallback simple linear projection: compute monthly growth rate from last few points
        try:
            if len(df) < 3:
                return None, None, None, 0.0
            pct_changes = df["value"].pct_change().dropna()
            avg_monthly = pct_changes[-3:].mean() if len(pct_changes) >=3 else pct_changes.mean()
            latest = df["value"].iloc[-1]
            forecast = []
            dates = pd.date_range(df["date"].iloc[-1], periods=steps+1, freq="M")[1:]
            val = latest
            for _ in range(steps):
                val = val * (1 + avg_monthly)
                forecast.append(val)
            mean_forecast = np.mean(forecast)
            roi = (mean_forecast / latest - 1) * 100.0
            return dates, np.array(forecast), None, roi
        except Exception:
            return None, None, None, 0.0

# ---------- Session-state init ----------
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

# Create demo users and hidden checker
if "investor_demo" not in st.session_state.users:
    st.session_state.users["investor_demo"] = {"password": hash_password("pass123"), "role": "Investor", "wallet": 10000.0}
if "startup_demo" not in st.session_state.users:
    st.session_state.users["startup_demo"] = {"password": hash_password("pass123"), "role": "Startup"}
if "checker_agent" not in st.session_state.users:
    st.session_state.users["checker_agent"] = {"password": hash_password("Check@2025!"), "role": "Checker"}

# Helper keys in session_state
if "_view_pitch" not in st.session_state:
    st.session_state._view_pitch = None
if "_view_graph" not in st.session_state:
    st.session_state._view_graph = None
if "_invest_in" not in st.session_state:
    st.session_state._invest_in = None
if "_complaint_for" not in st.session_state:
    st.session_state._complaint_for = None
if "_checker_view" not in st.session_state:
    st.session_state._checker_view = None

# ---------- Auth functions ----------
def signup(username, password, role):
    if not username or not password:
        st.warning("Enter username & password.")
        return False
    if username in st.session_state.users:
        st.warning("Username already exists.")
        return False
    if role == "Checker":
        st.error("Cannot signup as Checker.")
        return False
    st.session_state.users[username] = {"password": hash_password(password), "role": role}
    if role == "Investor":
        st.session_state.users[username]["wallet"] = 10000.0
    st.success("Signup successful. Now login.")
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
    st.session_state.page = "app"
    st.success(f"Logged in as {username} ({user['role']})")
    st.rerun()
    return True

def logout():
    st.session_state.current_user = None
    st.session_state.page = "home"
    st.rerun()

# ---------- Landing page ----------
def landing_page():
    st.markdown("<h1 style='color:#06b6d4; text-align:center;'>🚀 CrowdPitch Pro</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#9fb4c9;'>Verified Crowdfunding Platform — KYC & Manual Verification Demo</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # centered auth card
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        choice = st.radio("Action", ["Login", "Signup"], horizontal=True)
        if choice == "Login":
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
            su_role = st.selectbox("Role", ["Startup", "Investor"], key="su_role")
            if st.button("Create account"):
                if su_user and su_pass:
                    signup(su_user.strip(), su_pass, su_role)
                else:
                    st.warning("Enter username & password.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    # About / Contact / FAQs
    left, right = st.columns([2,1])
    with left:
        st.subheader("About CrowdPitch Pro")
        st.write(
            "CrowdPitch Pro is a demo platform to showcase startup onboarding with KYC/document verification, "
            "manual checker review, investor browsing and investing, and basic forecasting for pitch metrics."
        )
        st.subheader("Why choose us")
        st.write("- Manual KYC checker to reduce fraud\n- Simple forecasting to guide investor decisions\n- Complaint workflow for transparency")
    with right:
        st.subheader("Contact")
        st.markdown("Email: support@crowdpitch.example")
        st.markdown("Twitter: @crowdpitch")
        st.subheader("FAQs")
        st.markdown("**Is this real funding?** No — demo only.")
        st.markdown("**Checker account (hidden)**: `checker_agent` / `Check@2025!`")
    st.markdown("---")

# ---------- Startup onboarding ----------
def startup_page(user):
    st.header("🏢 Startup Onboarding & Pitch Creation")
    st.markdown("Submit company details, upload documents (KYC, address, bank, tax, incorporation), upload logo & company image, include video pitch link, and upload CSV for forecasting.")

    with st.form("onboard_form", clear_on_submit=False):
        st.subheader("Company Identity")
        company_name = st.text_input("Company Legal Name")
        reg_number = st.text_input("Business Registration / Incorporation Number")
        country = st.text_input("Country of Registration", value="India")
        founders = st.text_area("Founders' Full Names & Roles (comma separated)")
        official_email = st.text_input("Official Email (company domain preferred)")
        website = st.text_input("Company Website (https://...)")

        st.subheader("Media & Documents")
        logo_file = st.file_uploader("Logo (PNG/JPG)", type=["png","jpg","jpeg"])
        company_img_file = st.file_uploader("Company Image (team/office)", type=["png","jpg","jpeg"])
        video_link = st.text_input("Video pitch link (YouTube/Vimeo - optional)")

        st.markdown("**Proof documents (required)**")
        kyc_file = st.file_uploader("KYC Document (ID) (PNG/PDF)", type=["png","jpg","jpeg","pdf"])
        address_file = st.file_uploader("Proof of Address (PNG/PDF)", type=["png","jpg","jpeg","pdf"])
        bank_file = st.file_uploader("Bank Account Verification (PNG/PDF)", type=["png","jpg","jpeg","pdf"])
        tax_file = st.file_uploader("Tax Document (optional)", type=["png","jpg","jpeg","pdf"])
        inc_file = st.file_uploader("Incorporation Certificate (optional)", type=["png","jpg","jpeg","pdf"])

        st.subheader("Pitch & Data")
        pitch_name = st.text_input("Pitch / Product Name")
        short_desc = st.text_input("Short Description")
        long_desc = st.text_area("Long Description")
        csv_file = st.file_uploader("Performance CSV (date,value) for forecasting", type=["csv"])

        submitted = st.form_submit_button("Submit Pitch & Onboard")

    if submitted:
        missing = []
        if not company_name: missing.append("Company Name")
        if not pitch_name: missing.append("Pitch Name")
        if not csv_file: missing.append("CSV data")
        if not kyc_file: missing.append("KYC")
        if not address_file: missing.append("Address proof")
        if not bank_file: missing.append("Bank doc")
        if missing:
            st.error("Missing required: " + ", ".join(missing))
            return

        try:
            df = safe_read_csv(csv_file)
        except Exception as e:
            st.error(f"CSV parsing error: {e}")
            return

        # save bytes for images
        logo_bytes = image_bytes(logo_file) if logo_file else None
        company_img_bytes = image_bytes(company_img_file) if company_img_file else None

        # filenames
        kyc_name = getattr(kyc_file, "name", "")
        addr_name = getattr(address_file, "name", "")
        bank_name = getattr(bank_file, "name", "")
        tax_name = getattr(tax_file, "name", "") if tax_file else ""
        inc_name = getattr(inc_file, "name", "") if inc_file else ""

        # heuristics
        website_domain = extract_domain(website) if website else ""
        email_dom = email_domain(official_email) if official_email else ""
        email_ok = False
        if official_email and website_domain:
            email_ok = (website_domain in email_dom) or (email_dom in website_domain)
        else:
            if official_email and not any(d in official_email for d in ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com"]):
                email_ok = True

        bank_match = name_in_filename(company_name, bank_name)

        verification = {
            "email_domain": "Verified" if email_ok else "Pending",
            "bank_doc": "Verified" if bank_match else "Pending",
            "kyc_uploaded": True,
            "address_uploaded": True
        }

        pitch = {
            "company_name": company_name,
            "reg_number": reg_number,
            "country": country,
            "founders": founders,
            "official_email": official_email,
            "website": website,
            "logo": logo_bytes,
            "company_image": company_img_bytes,
            "video": video_link,
            "kyc_file_name": kyc_name,
            "address_file_name": addr_name,
            "bank_file_name": bank_name,
            "tax_file_name": tax_name,
            "inc_file_name": inc_name,
            "pitch_name": pitch_name,
            "short": short_desc,
            "desc": long_desc,
            "owner": user["username"],
            "data": df,
            "funded": 0.0,
            "investors": [],
            "verification": verification,
            "published": False,
            "checker_status": "Pending",
            "checker_note": ""
        }

        st.session_state.pitches.append(pitch)
        st.success(f"Pitch '{pitch_name}' submitted for checker review.")
        st.balloons()

    # Show created pitches
    st.markdown("---")
    st.subheader("Your Created Pitches")
    mine = [p for p in st.session_state.pitches if p["owner"] == user["username"]]
    if not mine:
        st.info("No created pitches yet.")
    else:
        for p in mine:
            st.markdown(f"### {p['pitch_name']}")
            cols = st.columns([1,3])
            with cols[0]:
                if p["logo"]:
                    st.image(p["logo"], width=180)
                else:
                    st.image("https://via.placeholder.com/220x140.png?text=No+Logo", width=180)
                st.markdown(f"**Funded:** ₹{p['funded']:,.2f}")
                st.markdown(f"Status: {p['checker_status']}")
            with cols[1]:
                st.markdown(p["short"])
                if st.button(f"View & Forecast — {p['pitch_name']}", key=f"view_{p['pitch_name']}"):
                    st.session_state._view_pitch = p["pitch_name"]

# ---------- Checker page ----------
def checker_page(user):
    st.header("🕵️ Checker — Pending Reviews & Complaints")
    pending = [p for p in st.session_state.pitches if p["checker_status"] == "Pending"]
    if not pending:
        st.info("No pending pitches to review.")
    for p in pending:
        st.markdown(f"<div class='card'><h3>{p['pitch_name']} — {p['company_name']}</h3>", unsafe_allow_html=True)
        cols = st.columns([1,3])
        with cols[0]:
            if p.get("logo"):
                st.image(p["logo"], width=160)
            else:
                st.image("https://via.placeholder.com/220x140.png?text=No+Logo", width=160)
            st.markdown(f"Owner: {p['owner']}")
            st.markdown(f"Email verif: {p['verification'].get('email_domain')}")
        with cols[1]:
            st.markdown(p["short"])
            if st.button("View Details", key=f"checker_view_{p['pitch_name']}"):
                st.session_state._checker_view = p["pitch_name"]
        st.markdown("</div>", unsafe_allow_html=True)

    # Checker view details
    if st.session_state._checker_view:
        target = st.session_state._checker_view
        sel = next((x for x in st.session_state.pitches if x["pitch_name"] == target), None)
        if sel:
            st.markdown("---")
            st.subheader(f"Review — {sel['pitch_name']} ({sel['company_name']})")
            st.markdown("**Company info**")
            st.write({
                "Company": sel.get("company_name"),
                "Registration": sel.get("reg_number"),
                "Country": sel.get("country"),
                "Founders": sel.get("founders"),
                "Email": sel.get("official_email"),
                "Website": sel.get("website")
            })
            st.markdown("**Uploaded media & documents**")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown("Logo")
                if sel.get("logo"):
                    st.image(sel["logo"], width=220)
                else:
                    st.write("—")
            with c2:
                st.markdown("Company image")
                if sel.get("company_image"):
                    st.image(sel["company_image"], width=220)
                else:
                    st.write("—")
            with c3:
                st.markdown("Video Pitch")
                if sel.get("video"):
                    try:
                        st.video(sel["video"])
                    except:
                        st.write(sel.get("video"))
                else:
                    st.write("—")
            st.markdown("**Documents (filenames)**")
            st.write({
                "KYC": sel.get("kyc_file_name"),
                "Address": sel.get("address_file_name"),
                "Bank": sel.get("bank_file_name"),
                "Tax": sel.get("tax_file_name"),
                "Incorporation": sel.get("inc_file_name")
            })
            st.markdown("**CSV preview**")
            st.dataframe(sel["data"].head())

            note = st.text_area("Decision note (optional)", key=f"ck_note_{sel['pitch_name']}")
            c1, c2, c3 = st.columns([1,1,1])
            if c1.button("Approve", key=f"ck_approve_{sel['pitch_name']}"):
                sel["checker_status"] = "Approved"
                sel["published"] = True
                sel["checker_note"] = note or "Approved by checker"
                st.success("Approved — now visible to Investors.")
                st.session_state._checker_view = None
            if c2.button("Reject", key=f"ck_reject_{sel['pitch_name']}"):
                sel["checker_status"] = "Rejected"
                sel["published"] = False
                sel["checker_note"] = note or "Rejected by checker"
                st.error("Rejected.")
                st.session_state._checker_view = None
            if c3.button("Request Re-check", key=f"ck_recheck_{sel['pitch_name']}"):
                sel["checker_status"] = "Pending"
                sel["published"] = False
                sel["checker_note"] = note or "Marked for re-check"
                st.info("Marked for re-check.")
                st.session_state._checker_view = None

# ---------- Investor page ----------
def investor_page(user):
    st.header("💼 Investor Marketplace")
    # Wallet top
    wallet = st.session_state.users[user["username"]].get("wallet", 0.0)
    st.markdown(f"**Wallet:** ₹{wallet:,.2f}")
    st.markdown("---")

    published = [p for p in st.session_state.pitches if p.get("published")]
    if not published:
        st.info("No published pitches yet.")
        return

    # Layout cards
    for p in published:
        roi_text = "—"
        try:
            _, _, _, roi = predict_forecast_and_roi(p["data"], steps=6)
            if roi is None:
                roi = 0.0
            roi_text = f"{roi:.1f}%"
        except Exception:
            roi_text = "—"

        # funding progress (if target exists else show relative)
        # We don't have a target; show funded as-is and investors count
        funded = p.get("funded", 0.0)
        investors_count = len(p.get("investors", []))
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        cols = st.columns([1,2])
        with cols[0]:
            if p.get("logo"):
                st.image(p["logo"], width=180)
            else:
                st.image("https://via.placeholder.com/220x140.png?text=No+Logo", width=180)
            if p.get("company_image"):
                st.image(p["company_image"], width=220)
        with cols[1]:
            st.markdown(f"### {p['pitch_name']} — {p['company_name']}")
            st.markdown(p.get("short",""))
            st.markdown(f"**Predicted ROI (6 mo avg):** {roi_text}")
            st.markdown(f"**Funded:** ₹{funded:,.0f} • **Investors:** {investors_count}")
            # action buttons
            b1, b2, b3 = st.columns([1,1,1])
            if b1.button("View Graph", key=f"vg_{p['pitch_name']}"):
                st.session_state._view_graph = p["pitch_name"]
            # Invest form in-line to avoid re-render issues
            with b2:
                form_key = f"invest_form_{p['pitch_name']}"
                with st.form(form_key):
                    amount = st.number_input("Amount (INR)", min_value=100.0, value=500.0, step=100.0, key=f"amt_{p['pitch_name']}")
                    sub = st.form_submit_button("Invest")
                    if sub:
                        wallet = st.session_state.users[user["username"]]["wallet"]
                        if amount <= 0:
                            st.error("Enter positive amount.")
                        elif amount > wallet:
                            st.error("Insufficient balance.")
                        else:
                            # perform transaction
                            st.session_state.users[user["username"]]["wallet"] -= amount
                            p["funded"] += amount
                            p["investors"].append({"investor": user["username"], "amount": amount})
                            st.session_state.investments.append({"investor": user["username"], "pitch": p["pitch_name"], "amount": amount})
                            st.success(f"Invested ₹{amount:,.0f} in {p['pitch_name']}")
            if b3.button("Complain", key=f"comp_{p['pitch_name']}"):
                st.session_state._complaint_for = p["pitch_name"]
        st.markdown("</div>", unsafe_allow_html=True)

    # Graph modal / area
    if st.session_state._view_graph:
        target = st.session_state._view_graph
        sel = next((x for x in st.session_state.pitches if x["pitch_name"] == target), None)
        if sel:
            st.markdown("---")
            st.subheader(f"Historical + Forecast — {sel['pitch_name']}")
            df = sel["data"]
            fig, ax = plt.subplots(figsize=(9,4))
            sns.lineplot(x="date", y="value", data=df, marker="o", ax=ax, label="Historical")
            try:
                future_dates, forecast_vals, conf, roi = predict_forecast_and_roi(df, steps=6)
                if future_dates is not None:
                    ax.plot(future_dates, forecast_vals, marker="X", linestyle="--", label="Forecast")
                    if conf is not None:
                        ax.fill_between(future_dates, conf.iloc[:,0], conf.iloc[:,1], alpha=0.2)
                ax.set_title("Metric (historical + forecast)")
            except Exception as e:
                st.warning(f"Forecast error: {e}")
            st.pyplot(fig)
            if st.button("Close Graph"):
                st.session_state._view_graph = None

    # Complaint flow
    if st.session_state._complaint_for:
        pitch_name = st.session_state._complaint_for
        st.markdown("---")
        st.subheader(f"Raise Complaint for {pitch_name}")
        msg = st.text_area("Describe the issue (min 10 chars)")
        if st.button("Submit Complaint"):
            if not msg or len(msg.strip()) < 10:
                st.warning("Please write a clearer complaint (min 10 chars).")
            else:
                st.session_state.complaints.append({
                    "pitch_name": pitch_name,
                    "investor": user["username"],
                    "message": msg.strip(),
                    "status": "Open",
                    "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "resolution_note": ""
                })
                st.success("Complaint submitted to checker.")
                st.session_state._complaint_for = None

# ---------- Main routing ----------
def main_app():
    user = st.session_state.current_user
    st.sidebar.markdown(f"**Logged in as:** {user['username']} ({user['role']})")
    if user["role"] == "Investor":
        wallet = st.session_state.users[user["username"]].get("wallet", 0.0)
        st.sidebar.markdown(f"**Wallet:** ₹{wallet:,.2f}")
    if st.sidebar.button("Logout"):
        logout()

    # role pages
    if user["role"] == "Startup":
        startup_page(user)
    elif user["role"] == "Checker":
        checker_page(user)
    elif user["role"] == "Investor":
        investor_page(user)
    else:
        st.info("Unknown role.")

# ---------- Router ----------
if st.session_state.page == "home" or st.session_state.current_user is None:
    landing_page()
else:
    main_app()
